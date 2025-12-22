//! WAL writer implementation

use super::entry::WalEntry;
use super::error::{WalError, WalResult};
use super::Lsn;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Magic number at the start of WAL files: "MFLDWAL\0"
const WAL_MAGIC: [u8; 8] = [0x4D, 0x46, 0x4C, 0x44, 0x57, 0x41, 0x4C, 0x00];

/// Current WAL format version
const WAL_VERSION: u32 = 1;

/// Size of the WAL file header
const HEADER_SIZE: u64 = 16; // 8 bytes magic + 4 bytes version + 4 bytes reserved

/// Configuration for the WAL writer
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Buffer size for writes (default: 64KB)
    pub buffer_size: usize,

    /// Sync mode for durability
    pub sync_mode: SyncMode,

    /// Maximum WAL file size before rotation (default: 64MB)
    pub max_file_size: u64,
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024,          // 64KB
            sync_mode: SyncMode::Immediate,  // Safest default
            max_file_size: 64 * 1024 * 1024, // 64MB
        }
    }
}

/// Sync mode determines when writes are flushed to disk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Sync after every write (safest, slowest)
    Immediate,
    /// Sync after each transaction commit
    OnCommit,
    /// Batch syncs - sync after a certain number of operations
    Batched(usize),
    /// Never explicitly sync (fastest, relies on OS)
    None,
}

/// Write-ahead log writer
///
/// Handles appending entries to the WAL file with checksums and
/// supports checkpoint operations to truncate the log.
pub struct WalWriter {
    /// Path to the WAL file
    path: PathBuf,

    /// Buffered file writer
    writer: BufWriter<File>,

    /// Current file position
    position: u64,

    /// Current LSN (highest written)
    current_lsn: AtomicU64,

    /// Last checkpointed LSN
    checkpoint_lsn: AtomicU64,

    /// Configuration
    config: WalConfig,

    /// Number of operations since last sync (for batched mode)
    ops_since_sync: usize,
}

impl WalWriter {
    /// Open or create a WAL file
    ///
    /// If the file exists, it will be opened for appending and the
    /// current LSN will be recovered from the last entry.
    pub fn open(path: impl AsRef<Path>, config: WalConfig) -> WalResult<Self> {
        let path = path.as_ref().to_path_buf();
        let exists = path.exists();

        let file =
            OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;

        let mut writer = BufWriter::with_capacity(config.buffer_size, file);
        let (position, current_lsn, checkpoint_lsn) = if exists {
            Self::recover_state(&path)?
        } else {
            // Write header for new file
            Self::write_header(&mut writer)?;
            (HEADER_SIZE, 0, 0)
        };

        // Seek to end for appending
        writer.seek(SeekFrom::Start(position))?;

        Ok(Self {
            path,
            writer,
            position,
            current_lsn: AtomicU64::new(current_lsn),
            checkpoint_lsn: AtomicU64::new(checkpoint_lsn),
            config,
            ops_since_sync: 0,
        })
    }

    /// Write the WAL file header
    fn write_header(writer: &mut BufWriter<File>) -> WalResult<()> {
        writer.seek(SeekFrom::Start(0))?;
        writer.write_all(&WAL_MAGIC)?;
        writer.write_all(&WAL_VERSION.to_le_bytes())?;
        writer.write_all(&[0u8; 4])?; // Reserved
        writer.flush()?;
        Ok(())
    }

    /// Recover state from an existing WAL file
    fn recover_state(path: &Path) -> WalResult<(u64, Lsn, Lsn)> {
        use super::recovery::WalRecovery;

        let recovery = WalRecovery::open(path)?;
        let mut max_lsn = 0u64;
        let mut checkpoint_lsn = 0u64;

        for result in recovery.iter() {
            match result {
                Ok(entry) => {
                    max_lsn = max_lsn.max(entry.lsn);
                    if let Some(ckpt_lsn) = entry.checkpoint_lsn() {
                        checkpoint_lsn = checkpoint_lsn.max(ckpt_lsn);
                    }
                }
                Err(WalError::Truncated { offset }) => {
                    // File was truncated, use this as our append position
                    return Ok((offset, max_lsn, checkpoint_lsn));
                }
                Err(e) if e.is_corruption() => {
                    // Stop at corruption, we'll append after last valid entry
                    break;
                }
                Err(e) => return Err(e),
            }
        }

        // Get file size for append position
        let metadata = std::fs::metadata(path)?;
        Ok((metadata.len(), max_lsn, checkpoint_lsn))
    }

    /// Append an entry to the WAL
    ///
    /// The entry is serialized with a length prefix and CRC32 checksum.
    /// Format: [4-byte length][entry data][4-byte CRC32]
    pub fn append(&mut self, entry: WalEntry) -> WalResult<Lsn> {
        let current = self.current_lsn.load(Ordering::SeqCst);
        if entry.lsn <= current && current > 0 {
            return Err(WalError::InvalidLsn { attempted: entry.lsn, current });
        }

        // Serialize the entry
        let data = bincode::serde::encode_to_vec(&entry, bincode::config::standard())
            .map_err(|e| WalError::Serialize(e.to_string()))?;

        // Calculate CRC32
        let crc = crc32_checksum(&data);

        // Write: [length: u32][data: bytes][crc: u32]
        let len = data.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&data)?;
        self.writer.write_all(&crc.to_le_bytes())?;

        self.position += 4 + data.len() as u64 + 4;
        self.current_lsn.store(entry.lsn, Ordering::SeqCst);
        self.ops_since_sync += 1;

        // Handle sync based on mode
        match self.config.sync_mode {
            SyncMode::Immediate => self.sync()?,
            SyncMode::OnCommit
                if matches!(
                    entry.operation,
                    super::entry::Operation::CommitTxn | super::entry::Operation::AbortTxn
                ) =>
            {
                self.sync()?
            }
            SyncMode::Batched(n) if self.ops_since_sync >= n => self.sync()?,
            _ => {}
        }

        Ok(entry.lsn)
    }

    /// Sync buffered writes to disk
    pub fn sync(&mut self) -> WalResult<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        self.ops_since_sync = 0;
        Ok(())
    }

    /// Create a checkpoint
    ///
    /// This marks that all entries up to `checkpoint_lsn` have been
    /// safely persisted to the main storage and can be discarded.
    ///
    /// The checkpoint is implemented by:
    /// 1. Writing a checkpoint marker to the current WAL
    /// 2. Truncating entries before the checkpoint (optional, deferred)
    pub fn checkpoint(&mut self, checkpoint_lsn: Lsn) -> WalResult<()> {
        let next_lsn = self.current_lsn.load(Ordering::SeqCst) + 1;

        // Write checkpoint marker
        let entry = WalEntry::checkpoint(next_lsn, checkpoint_lsn);
        self.append(entry)?;
        self.sync()?;

        self.checkpoint_lsn.store(checkpoint_lsn, Ordering::SeqCst);

        // Optionally truncate the WAL file
        self.truncate_before_checkpoint()?;

        Ok(())
    }

    /// Truncate entries before the last checkpoint
    ///
    /// This creates a new WAL file with only entries after the checkpoint,
    /// then atomically replaces the old file.
    fn truncate_before_checkpoint(&mut self) -> WalResult<()> {
        let checkpoint_lsn = self.checkpoint_lsn.load(Ordering::SeqCst);
        if checkpoint_lsn == 0 {
            return Ok(()); // No checkpoint yet
        }

        // Create temp file for new WAL
        let temp_path = self.path.with_extension("wal.tmp");
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        let mut temp_writer = BufWriter::with_capacity(self.config.buffer_size, temp_file);

        // Write header
        Self::write_header(&mut temp_writer)?;

        // Re-read original WAL and copy entries after checkpoint
        let recovery = super::recovery::WalRecovery::open(&self.path)?;
        let mut new_position = HEADER_SIZE;

        for result in recovery.iter() {
            match result {
                Ok(entry) if entry.lsn > checkpoint_lsn => {
                    // Copy this entry to the new file
                    let data = bincode::serde::encode_to_vec(&entry, bincode::config::standard())
                        .map_err(|e| WalError::Serialize(e.to_string()))?;
                    let crc = crc32_checksum(&data);
                    let len = data.len() as u32;

                    temp_writer.write_all(&len.to_le_bytes())?;
                    temp_writer.write_all(&data)?;
                    temp_writer.write_all(&crc.to_le_bytes())?;

                    new_position += 4 + data.len() as u64 + 4;
                }
                Ok(_) => {} // Skip entries before checkpoint
                Err(e) if e.is_corruption() => break,
                Err(e) => return Err(e),
            }
        }

        temp_writer.flush()?;
        temp_writer.get_ref().sync_all()?;
        drop(temp_writer);

        // Atomic rename
        std::fs::rename(&temp_path, &self.path)?;

        // Reopen the file
        let file = OpenOptions::new().read(true).write(true).open(&self.path)?;

        self.writer = BufWriter::with_capacity(self.config.buffer_size, file);
        self.writer.seek(SeekFrom::Start(new_position))?;
        self.position = new_position;

        Ok(())
    }

    /// Get the current LSN
    pub fn current_lsn(&self) -> Lsn {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Get the last checkpoint LSN
    pub fn checkpoint_lsn(&self) -> Lsn {
        self.checkpoint_lsn.load(Ordering::SeqCst)
    }

    /// Get the next LSN to use
    pub fn next_lsn(&self) -> Lsn {
        self.current_lsn.load(Ordering::SeqCst) + 1
    }

    /// Get the path to the WAL file
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Calculate CRC32 checksum using the IEEE polynomial
fn crc32_checksum(data: &[u8]) -> u32 {
    // Simple CRC32 implementation (IEEE polynomial)
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
    use tempfile::tempdir;

    #[test]
    fn test_wal_writer_create() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let config = WalConfig::default();
        let wal = WalWriter::open(&wal_path, config).unwrap();

        assert_eq!(wal.current_lsn(), 0);
        assert_eq!(wal.checkpoint_lsn(), 0);
        assert!(wal_path.exists());
    }

    #[test]
    fn test_wal_writer_append() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let config = WalConfig::default();
        let mut wal = WalWriter::open(&wal_path, config).unwrap();

        let lsn1 = wal.append(WalEntry::put(1, "table", b"k1", b"v1")).unwrap();
        let lsn2 = wal.append(WalEntry::put(2, "table", b"k2", b"v2")).unwrap();

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(wal.current_lsn(), 2);
    }

    #[test]
    fn test_wal_writer_invalid_lsn() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let config = WalConfig::default();
        let mut wal = WalWriter::open(&wal_path, config).unwrap();

        wal.append(WalEntry::put(5, "table", b"k", b"v")).unwrap();

        // LSN must be monotonically increasing
        let result = wal.append(WalEntry::put(3, "table", b"k", b"v"));
        assert!(matches!(result, Err(WalError::InvalidLsn { .. })));
    }

    #[test]
    fn test_wal_writer_reopen() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Write some entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "table", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::put(2, "table", b"k2", b"v2")).unwrap();
            wal.sync().unwrap();
        }

        // Reopen and verify state
        {
            let config = WalConfig::default();
            let wal = WalWriter::open(&wal_path, config).unwrap();
            assert_eq!(wal.current_lsn(), 2);
        }
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, World!";
        let crc = crc32_checksum(data);
        // Verify consistency
        assert_eq!(crc, crc32_checksum(data));

        // Different data should have different checksum
        let crc2 = crc32_checksum(b"Different data");
        assert_ne!(crc, crc2);
    }

    #[test]
    fn test_sync_modes() {
        let dir = tempdir().unwrap();

        // Test immediate sync
        {
            let wal_path = dir.path().join("immediate.wal");
            let config = WalConfig { sync_mode: SyncMode::Immediate, ..Default::default() };
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "t", b"k", b"v")).unwrap();
            // File should be synced immediately
        }

        // Test batched sync
        {
            let wal_path = dir.path().join("batched.wal");
            let config = WalConfig { sync_mode: SyncMode::Batched(3), ..Default::default() };
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "t", b"k1", b"v")).unwrap();
            wal.append(WalEntry::put(2, "t", b"k2", b"v")).unwrap();
            assert_eq!(wal.ops_since_sync, 2);
            wal.append(WalEntry::put(3, "t", b"k3", b"v")).unwrap();
            assert_eq!(wal.ops_since_sync, 0); // Should have synced
        }
    }
}
