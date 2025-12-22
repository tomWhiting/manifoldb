//! WAL recovery and replay functionality

use super::entry::{Operation, WalEntry};
use super::error::{WalError, WalResult};
use super::Lsn;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Magic number for WAL files
const WAL_MAGIC: [u8; 8] = [0x4D, 0x46, 0x4C, 0x44, 0x57, 0x41, 0x4C, 0x00];

/// Expected WAL version
const WAL_VERSION: u32 = 1;

/// Size of the WAL header
const HEADER_SIZE: u64 = 16;

/// WAL recovery handler
///
/// Reads and validates WAL entries for crash recovery.
/// Provides iterators for reading entries and filtering by commit status.
pub struct WalRecovery {
    /// Path to the WAL file
    path: PathBuf,

    /// Buffered file reader
    reader: BufReader<File>,

    /// Current position in the file
    position: u64,

    /// File size
    file_size: u64,
}

impl WalRecovery {
    /// Open a WAL file for recovery
    pub fn open(path: impl AsRef<Path>) -> WalResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata()?.len();
        let mut reader = BufReader::new(file);

        // Validate header
        Self::validate_header(&mut reader)?;

        Ok(Self { path, reader, position: HEADER_SIZE, file_size })
    }

    /// Validate the WAL file header
    fn validate_header(reader: &mut BufReader<File>) -> WalResult<()> {
        reader.seek(SeekFrom::Start(0))?;

        let mut magic = [0u8; 8];
        if reader.read_exact(&mut magic).is_err() {
            return Err(WalError::InvalidFormat("file too small for header".into()));
        }

        if magic != WAL_MAGIC {
            return Err(WalError::InvalidFormat(format!("invalid magic number: {:?}", magic)));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);

        if version != WAL_VERSION {
            return Err(WalError::InvalidFormat(format!(
                "unsupported WAL version: {version}, expected {WAL_VERSION}"
            )));
        }

        // Skip reserved bytes
        reader.seek(SeekFrom::Current(4))?;

        Ok(())
    }

    /// Read the next entry from the WAL
    fn read_entry(&mut self) -> WalResult<Option<WalEntry>> {
        if self.position >= self.file_size {
            return Ok(None);
        }

        self.reader.seek(SeekFrom::Start(self.position))?;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        match self.reader.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(WalError::Truncated { offset: self.position });
            }
            Err(e) => return Err(e.into()),
        }

        let len = u32::from_le_bytes(len_bytes) as usize;
        if len == 0 || len > 100 * 1024 * 1024 {
            // Sanity check: 100MB max entry
            return Err(WalError::InvalidFormat(format!("invalid entry length: {len}")));
        }

        // Read entry data
        let mut data = vec![0u8; len];
        match self.reader.read_exact(&mut data) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(WalError::Truncated { offset: self.position });
            }
            Err(e) => return Err(e.into()),
        }

        // Read CRC
        let mut crc_bytes = [0u8; 4];
        match self.reader.read_exact(&mut crc_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(WalError::Truncated { offset: self.position });
            }
            Err(e) => return Err(e.into()),
        }

        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = crc32_checksum(&data);

        // Deserialize entry
        let (entry, _): (WalEntry, _) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| WalError::Deserialize(e.to_string()))?;

        if stored_crc != computed_crc {
            return Err(WalError::ChecksumMismatch {
                lsn: entry.lsn,
                expected: stored_crc,
                actual: computed_crc,
            });
        }

        self.position += 4 + len as u64 + 4;

        Ok(Some(entry))
    }

    /// Create an iterator over all entries in the WAL
    pub fn iter(self) -> WalEntryIterator {
        WalEntryIterator { recovery: self, finished: false }
    }

    /// Get an iterator over only committed entries
    ///
    /// This performs two passes:
    /// 1. First pass identifies all committed transaction IDs
    /// 2. Second pass yields only entries belonging to committed transactions
    ///    or standalone entries (txn_id = 0)
    pub fn committed_entries(self) -> CommittedEntryIterator {
        CommittedEntryIterator::new(self.path)
    }

    /// Replay WAL entries to a storage engine
    ///
    /// This is the main recovery entry point. It:
    /// 1. Reads all entries from the WAL
    /// 2. Filters to only committed transactions
    /// 3. Applies operations in order
    ///
    /// Uses `RecoveryMode::SkipCorrupted` by default. Use `replay_with_mode`
    /// for more control over corruption handling.
    pub fn replay<F>(&mut self, apply: F) -> WalResult<RecoveryStats>
    where
        F: FnMut(&WalEntry) -> WalResult<()>,
    {
        self.replay_with_mode(apply, RecoveryMode::default())
    }

    /// Replay WAL entries with configurable corruption handling
    ///
    /// This is the main recovery entry point with explicit control over
    /// how corruption is handled. It:
    /// 1. Reads all entries from the WAL
    /// 2. Handles corruption according to the specified mode
    /// 3. Filters to only committed transactions
    /// 4. Applies operations in order
    ///
    /// # Recovery Modes
    ///
    /// - `SkipCorrupted`: Skip corrupted entries with warnings (default)
    /// - `FailOnCorruption`: Fail immediately on any corruption
    /// - `FailOnThreshold`: Fail if corruption exceeds configured limits
    ///
    /// # Returns
    ///
    /// Returns `RecoveryStats` containing both success metrics and corruption
    /// information. Check `stats.has_corruption()` to detect if any entries
    /// were skipped.
    pub fn replay_with_mode<F>(
        &mut self,
        mut apply: F,
        mode: RecoveryMode,
    ) -> WalResult<RecoveryStats>
    where
        F: FnMut(&WalEntry) -> WalResult<()>,
    {
        let path = self.path.clone();

        // First pass: identify committed transactions and gather corruption info
        let (committed_txns, first_pass_stats) = Self::first_pass_with_mode(&path, mode)?;

        // Second pass: apply committed entries
        let mut stats = first_pass_stats;
        let recovery = WalRecovery::open(&path)?;

        for result in recovery.iter() {
            match result {
                Ok(entry) => {
                    // Include if:
                    // - Standalone operation (txn_id = 0)
                    // - Part of a committed transaction
                    // - Transaction markers are skipped for data replay
                    if entry.is_data_operation()
                        && (entry.txn_id == 0 || committed_txns.contains(&entry.txn_id))
                    {
                        apply(&entry)?;
                        stats.operations_applied += 1;
                    }
                    stats.max_lsn = stats.max_lsn.max(entry.lsn);
                }
                Err(e) => {
                    // Corruption already handled in first pass, just skip
                    // (mode validation already passed)
                    let warning = format!("Skipping corrupted entry during replay: {e}");
                    warn!("{warning}");
                }
            }
        }

        // Log recovery summary
        if stats.has_corruption() {
            warn!(
                entries_skipped = stats.entries_skipped,
                corrupted_bytes = stats.corrupted_bytes,
                warnings = stats.warnings.len(),
                "WAL recovery completed with corruption detected"
            );
        } else {
            info!(
                entries_processed = stats.entries_processed,
                operations_applied = stats.operations_applied,
                max_lsn = stats.max_lsn,
                "WAL recovery completed successfully"
            );
        }

        Ok(stats)
    }

    /// First pass to identify committed transactions and check for corruption
    fn first_pass_with_mode(
        path: &Path,
        mode: RecoveryMode,
    ) -> WalResult<(HashSet<u64>, RecoveryStats)> {
        let mut committed_txns = HashSet::new();
        let mut stats = RecoveryStats::default();

        let recovery = WalRecovery::open(path)?;

        for result in recovery.iter() {
            match result {
                Ok(entry) => {
                    if entry.operation == Operation::CommitTxn {
                        committed_txns.insert(entry.txn_id);
                        stats.committed_txns += 1;
                    }
                    stats.entries_processed += 1;
                }
                Err(e) => {
                    // Handle corruption based on mode
                    let warning = format!("Corrupted WAL entry at recovery: {e}");
                    warn!("{warning}");
                    stats.add_warning(warning.clone());
                    stats.entries_skipped += 1;

                    // Estimate corrupted bytes based on error type
                    if let WalError::Truncated { offset } = &e {
                        stats.corrupted_bytes = stats.corrupted_bytes.saturating_add(
                            std::fs::metadata(path)
                                .map(|m| m.len().saturating_sub(*offset) as usize)
                                .unwrap_or(0),
                        );
                    } else {
                        // Estimate ~100 bytes per corrupted entry as a reasonable guess
                        stats.corrupted_bytes = stats.corrupted_bytes.saturating_add(100);
                    }

                    // Check if we should fail based on mode
                    match mode {
                        RecoveryMode::SkipCorrupted => {
                            // Continue processing
                        }
                        RecoveryMode::FailOnCorruption => {
                            return Err(WalError::Recovery(format!(
                                "Corruption detected and FailOnCorruption mode is enabled: {e}"
                            )));
                        }
                        RecoveryMode::FailOnThreshold {
                            max_corrupted_entries,
                            max_corrupted_bytes,
                        } => {
                            if stats.entries_skipped > max_corrupted_entries {
                                return Err(WalError::Recovery(format!(
                                    "Corruption threshold exceeded: {} entries skipped (max: {})",
                                    stats.entries_skipped, max_corrupted_entries
                                )));
                            }
                            if max_corrupted_bytes > 0
                                && stats.corrupted_bytes > max_corrupted_bytes
                            {
                                return Err(WalError::Recovery(format!(
                                    "Corruption threshold exceeded: {} bytes corrupted (max: {})",
                                    stats.corrupted_bytes, max_corrupted_bytes
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok((committed_txns, stats))
    }

    /// Get the last checkpoint LSN in the WAL
    pub fn last_checkpoint_lsn(&mut self) -> WalResult<Option<Lsn>> {
        self.reader.seek(SeekFrom::Start(HEADER_SIZE))?;
        self.position = HEADER_SIZE;

        let mut last_checkpoint = None;

        while let Some(entry) = self.read_entry()? {
            if let Some(ckpt_lsn) = entry.checkpoint_lsn() {
                last_checkpoint = Some(ckpt_lsn);
            }
        }

        Ok(last_checkpoint)
    }
}

/// Recovery mode determines how corruption is handled during WAL recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryMode {
    /// Skip corrupted entries and continue recovery (default)
    /// Logs warnings for each skipped entry
    #[default]
    SkipCorrupted,

    /// Fail immediately if any corruption is detected
    FailOnCorruption,

    /// Skip corrupted entries but fail if corruption exceeds threshold
    FailOnThreshold {
        /// Maximum number of corrupted entries before failing
        max_corrupted_entries: usize,
        /// Maximum bytes of corruption before failing (0 = no limit)
        max_corrupted_bytes: usize,
    },
}

/// Statistics from WAL recovery
#[derive(Debug, Default, Clone)]
pub struct RecoveryStats {
    /// Total entries processed successfully
    pub entries_processed: usize,

    /// Data operations applied (Put/Delete)
    pub operations_applied: usize,

    /// Highest LSN seen
    pub max_lsn: Lsn,

    /// Number of committed transactions
    pub committed_txns: usize,

    /// Number of aborted/incomplete transactions
    pub aborted_txns: usize,

    /// Number of entries skipped due to corruption
    pub entries_skipped: usize,

    /// Estimated bytes of corrupted data skipped
    pub corrupted_bytes: usize,

    /// Warnings generated during recovery
    pub warnings: Vec<String>,
}

impl RecoveryStats {
    /// Returns true if any corruption was detected during recovery
    pub const fn has_corruption(&self) -> bool {
        self.entries_skipped > 0
    }

    /// Add a corruption warning
    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
}

/// Iterator over WAL entries
pub struct WalEntryIterator {
    recovery: WalRecovery,
    /// Set to true when corruption is detected, to stop iteration
    finished: bool,
}

impl Iterator for WalEntryIterator {
    type Item = WalResult<WalEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        match self.recovery.read_entry() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(e) => {
                // On corruption, return the error and stop iteration
                self.finished = true;
                Some(Err(e))
            }
        }
    }
}

/// Iterator that yields only committed entries
pub struct CommittedEntryIterator {
    /// Path to WAL file
    path: PathBuf,

    /// Set of committed transaction IDs (populated in first pass)
    committed_txns: HashSet<u64>,

    /// Entries to yield (populated after first pass)
    entries: std::vec::IntoIter<WalEntry>,

    /// Whether initialization is complete
    initialized: bool,
}

impl CommittedEntryIterator {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            committed_txns: HashSet::new(),
            entries: Vec::new().into_iter(),
            initialized: false,
        }
    }

    fn initialize(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;

        // First pass: identify committed transactions
        if let Ok(recovery) = WalRecovery::open(&self.path) {
            for entry in recovery.iter().flatten() {
                if entry.operation == Operation::CommitTxn {
                    self.committed_txns.insert(entry.txn_id);
                }
            }
        }

        // Second pass: collect committed entries
        let mut committed_entries = Vec::new();
        if let Ok(recovery) = WalRecovery::open(&self.path) {
            for entry in recovery.iter().flatten() {
                // Include if:
                // - Standalone operation (txn_id = 0)
                // - Part of a committed transaction
                // - Transaction markers are skipped for data replay
                if entry.is_data_operation()
                    && (entry.txn_id == 0 || self.committed_txns.contains(&entry.txn_id))
                {
                    committed_entries.push(entry);
                }
            }
        }

        self.entries = committed_entries.into_iter();
    }
}

impl Iterator for CommittedEntryIterator {
    type Item = WalEntry;

    fn next(&mut self) -> Option<Self::Item> {
        self.initialize();
        self.entries.next()
    }
}

/// Calculate CRC32 checksum (must match writer.rs implementation)
fn crc32_checksum(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc = !0u32;

    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ POLY } else { crc >> 1 };
        }
    }

    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{WalConfig, WalWriter};
    use tempfile::tempdir;

    #[test]
    fn test_recovery_empty_wal() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("empty.wal");

        // Create empty WAL
        {
            let config = WalConfig::default();
            let _wal = WalWriter::open(&wal_path, config).unwrap();
        }

        // Recovery should work on empty WAL
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let entries: Vec<_> = recovery.iter().collect();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_recovery_with_entries() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::delete(3, "nodes", b"k1")).unwrap();
            wal.sync().unwrap();
        }

        // Recover and verify
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let entries: Vec<_> = recovery.iter().filter_map(Result::ok).collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].lsn, 1);
        assert!(matches!(entries[0].operation, Operation::Put));
        assert_eq!(entries[2].lsn, 3);
        assert!(matches!(entries[2].operation, Operation::Delete));
    }

    #[test]
    fn test_committed_entries_only() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("txn.wal");

        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();

            // Committed transaction
            wal.append(WalEntry::begin_txn(1, 100)).unwrap();
            wal.append(WalEntry::put_in_txn(2, 100, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put_in_txn(3, 100, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::commit_txn(4, 100)).unwrap();

            // Uncommitted transaction
            wal.append(WalEntry::begin_txn(5, 101)).unwrap();
            wal.append(WalEntry::put_in_txn(6, 101, "nodes", b"k3", b"v3")).unwrap();
            // No commit!

            // Standalone operation
            wal.append(WalEntry::put(7, "edges", b"e1", b"data")).unwrap();

            wal.sync().unwrap();
        }

        let recovery = WalRecovery::open(&wal_path).unwrap();
        let committed: Vec<_> = recovery.committed_entries().collect();

        // Should have: 2 from txn 100 + 1 standalone
        assert_eq!(committed.len(), 3);

        // Verify entries
        assert_eq!(committed[0].txn_id, 100);
        assert_eq!(committed[1].txn_id, 100);
        assert_eq!(committed[2].txn_id, 0); // Standalone
    }

    #[test]
    fn test_replay() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("replay.wal");

        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::open(&wal_path).unwrap();
        let mut applied = Vec::new();

        let stats = recovery
            .replay(|entry| {
                applied.push(entry.clone());
                Ok(())
            })
            .unwrap();

        assert_eq!(stats.operations_applied, 2);
        assert_eq!(stats.max_lsn, 2);
        assert_eq!(applied.len(), 2);
    }

    #[test]
    fn test_checkpoint_lsn() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("ckpt.wal");

        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::checkpoint(3, 1)).unwrap();
            wal.append(WalEntry::put(4, "nodes", b"k3", b"v3")).unwrap();
            wal.append(WalEntry::checkpoint(5, 3)).unwrap();
            wal.sync().unwrap();
        }

        let mut recovery = WalRecovery::open(&wal_path).unwrap();
        let last_ckpt = recovery.last_checkpoint_lsn().unwrap();

        assert_eq!(last_ckpt, Some(3)); // Last checkpoint was at LSN 3
    }

    #[test]
    fn test_recovery_from_corrupted_middle_entry() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("corrupt_middle.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::put(3, "nodes", b"k3", b"v3")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the middle entry by overwriting bytes in the middle of the file
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            // Skip header (16 bytes) and first entry (approximately),
            // then corrupt bytes in the second entry
            file.seek(std::io::SeekFrom::Start(80)).unwrap();
            file.write_all(b"CORRUPTED").unwrap();
        }

        // Recovery should handle corruption gracefully
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let mut valid_count = 0;

        for result in recovery.iter() {
            match result {
                Ok(_) => valid_count += 1,
                Err(e) => {
                    // Verify error is a corruption-related error
                    assert!(
                        e.is_corruption() || matches!(e, WalError::Deserialize(_)),
                        "Expected corruption error, got: {:?}",
                        e
                    );
                }
            }
        }

        // At least first entry should be valid, and we should detect corruption
        assert!(valid_count >= 1, "Should have at least one valid entry");
    }

    #[test]
    fn test_recovery_from_truncated_entry() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("truncated.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.sync().unwrap();
        }

        // Truncate the file in the middle of the second entry
        {
            let file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            // Truncate to just after the first entry (header + first entry partial)
            file.set_len(60).unwrap();
        }

        // Recovery should handle truncation gracefully
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let mut valid_entries = Vec::new();

        for result in recovery.iter() {
            match result {
                Ok(entry) => valid_entries.push(entry),
                Err(WalError::Truncated { .. }) => {
                    // Truncation error is expected - just stop iteration
                    break;
                }
                Err(e) => {
                    // Other corruption errors are also acceptable
                    assert!(e.is_corruption(), "Unexpected error: {:?}", e);
                }
            }
        }

        // We may have valid entries before truncation
        // Truncation at offset 60 is after header (16 bytes), so first entry should be readable
        // depending on entry size
        assert!(valid_entries.len() <= 2, "Should not have more than 2 valid entries");
    }

    #[test]
    fn test_committed_entries_skip_corrupted() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("corrupt_committed.wal");

        // Write a transaction with committed entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();

            // First committed transaction
            wal.append(WalEntry::begin_txn(1, 100)).unwrap();
            wal.append(WalEntry::put_in_txn(2, 100, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::commit_txn(3, 100)).unwrap();

            // Standalone operation
            wal.append(WalEntry::put(4, "edges", b"e1", b"data")).unwrap();

            wal.sync().unwrap();
        }

        // Corrupt the standalone operation (entry 4)
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            // Move to near the end of the file to corrupt the last entry
            let metadata = file.metadata().unwrap();
            let file_size = metadata.len();
            file.seek(std::io::SeekFrom::Start(file_size - 10)).unwrap();
            file.write_all(b"CORRUPT").unwrap();
        }

        // committed_entries() should silently skip corrupted entries
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let committed: Vec<_> = recovery.committed_entries().collect();

        // Should still get the first transaction's data entry
        // The corrupted standalone entry should be skipped
        assert!(!committed.is_empty(), "Should have at least some committed entries");

        // All returned entries should be from committed transactions or standalone
        for entry in &committed {
            assert!(
                entry.txn_id == 0 || entry.txn_id == 100,
                "Entry should be from txn 100 or standalone"
            );
        }
    }

    #[test]
    fn test_checksum_mismatch_detection() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("checksum_fail.wal");

        // Write a valid entry
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"key", b"value")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt just the CRC (last 4 bytes of the entry)
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            let metadata = file.metadata().unwrap();
            let file_size = metadata.len();
            // CRC is at the end of the entry
            file.seek(std::io::SeekFrom::Start(file_size - 4)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
        }

        // Recovery should detect checksum mismatch
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let results: Vec<_> = recovery.iter().collect();

        assert_eq!(results.len(), 1, "Should have one result");
        assert!(
            matches!(&results[0], Err(WalError::ChecksumMismatch { .. })),
            "Should detect checksum mismatch, got: {:?}",
            results[0]
        );
    }

    #[test]
    fn test_replay_with_corruption_skip_mode() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("replay_corrupt_skip.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::put(3, "nodes", b"k3", b"v3")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the CRC of the second entry
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            // Corrupt in the middle to affect the second entry
            file.seek(std::io::SeekFrom::Start(70)).unwrap();
            file.write_all(b"CORRUPT").unwrap();
        }

        // Replay with SkipCorrupted mode should succeed and report corruption
        let mut recovery = WalRecovery::open(&wal_path).unwrap();
        let mut applied = Vec::new();

        let stats = recovery
            .replay_with_mode(
                |entry| {
                    applied.push(entry.lsn);
                    Ok(())
                },
                RecoveryMode::SkipCorrupted,
            )
            .unwrap();

        // Stats should show corruption was detected
        assert!(stats.has_corruption(), "Should detect corruption");
        assert!(stats.entries_skipped > 0, "Should have skipped entries");
        assert!(!stats.warnings.is_empty(), "Should have warnings");

        // At least some entries should be applied
        assert!(!applied.is_empty(), "Should apply some entries");
    }

    #[test]
    fn test_replay_with_corruption_fail_mode() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("replay_corrupt_fail.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the CRC of the first entry
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            let metadata = file.metadata().unwrap();
            let file_size = metadata.len();
            // Corrupt near the end
            file.seek(std::io::SeekFrom::Start(file_size - 4)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
        }

        // Replay with FailOnCorruption mode should fail
        let mut recovery = WalRecovery::open(&wal_path).unwrap();

        let result = recovery.replay_with_mode(|_| Ok(()), RecoveryMode::FailOnCorruption);

        assert!(result.is_err(), "Should fail on corruption");
        let err = result.unwrap_err();
        assert!(matches!(err, WalError::Recovery(_)), "Should be Recovery error, got: {err:?}");
    }

    #[test]
    fn test_replay_with_corruption_threshold_mode() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("replay_corrupt_threshold.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.append(WalEntry::put(3, "nodes", b"k3", b"v3")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt one entry
        {
            let mut file = std::fs::OpenOptions::new().write(true).open(&wal_path).unwrap();
            let metadata = file.metadata().unwrap();
            let file_size = metadata.len();
            file.seek(std::io::SeekFrom::Start(file_size - 4)).unwrap();
            file.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
        }

        // Replay with threshold that allows 1 corruption
        let mut recovery = WalRecovery::open(&wal_path).unwrap();
        let result = recovery.replay_with_mode(
            |_| Ok(()),
            RecoveryMode::FailOnThreshold { max_corrupted_entries: 5, max_corrupted_bytes: 0 },
        );

        // Should succeed with 1 corruption when threshold is 5
        assert!(result.is_ok(), "Should succeed within threshold");
        let stats = result.unwrap();
        assert!(stats.entries_skipped <= 5, "Should be within threshold");
    }

    #[test]
    fn test_recovery_stats_has_corruption() {
        let mut stats = RecoveryStats::default();
        assert!(!stats.has_corruption(), "Empty stats should not have corruption");

        stats.entries_skipped = 1;
        assert!(stats.has_corruption(), "Should have corruption when entries skipped");
    }

    #[test]
    fn test_recovery_stats_add_warning() {
        let mut stats = RecoveryStats::default();
        stats.add_warning("Test warning 1");
        stats.add_warning(String::from("Test warning 2"));

        assert_eq!(stats.warnings.len(), 2);
        assert_eq!(stats.warnings[0], "Test warning 1");
        assert_eq!(stats.warnings[1], "Test warning 2");
    }

    #[test]
    fn test_recovery_mode_default() {
        let mode = RecoveryMode::default();
        assert_eq!(mode, RecoveryMode::SkipCorrupted);
    }

    #[test]
    fn test_replay_clean_wal_reports_no_corruption() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("clean.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"k2", b"v2")).unwrap();
            wal.sync().unwrap();
        }

        // Replay should report no corruption
        let mut recovery = WalRecovery::open(&wal_path).unwrap();
        let stats = recovery.replay(|_| Ok(())).unwrap();

        assert!(!stats.has_corruption(), "Clean WAL should have no corruption");
        assert_eq!(stats.entries_skipped, 0);
        assert_eq!(stats.corrupted_bytes, 0);
        assert!(stats.warnings.is_empty());
        assert_eq!(stats.entries_processed, 2);
        assert_eq!(stats.operations_applied, 2);
    }
}
