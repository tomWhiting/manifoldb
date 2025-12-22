//! Write-Ahead Logging (WAL) for ManifoldDB
//!
//! This module provides durability guarantees through write-ahead logging.
//! All modifications are first written to an append-only log file before
//! being applied to the main storage engine. This enables:
//!
//! - **Crash recovery**: Replay uncommitted operations after a crash
//! - **Durability**: Committed transactions are persistent even if the main
//!   storage hasn't been flushed
//! - **Replication**: The WAL can be streamed to replicas for synchronization
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │   Transaction   │────▶│   WAL Writer    │────▶│   WAL File      │
//! │     Commit      │     │   (append-only) │     │   (on disk)     │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//!                                                         │
//!                                                         ▼
//! ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
//! │  Main Storage   │◀────│   Checkpoint    │◀────│   Recovery      │
//! │     (redb)      │     │    Manager      │     │    (replay)     │
//! └─────────────────┘     └─────────────────┘     └─────────────────┘
//! ```
//!
//! # Log Format
//!
//! Each WAL entry contains:
//! - **LSN (Log Sequence Number)**: Monotonically increasing identifier
//! - **Operation Type**: Put, Delete, or transaction markers
//! - **Table**: Logical table name
//! - **Key/Value**: The actual data being modified
//! - **CRC32**: Checksum for integrity validation
//!
//! # Usage
//!
//! ```ignore
//! use manifoldb_storage::wal::{WalWriter, WalConfig};
//!
//! // Create a WAL writer
//! let config = WalConfig::default();
//! let wal = WalWriter::open("database.wal", config)?;
//!
//! // Append operations (typically done during transaction commit)
//! wal.append(&WalEntry::put(1, "nodes", b"key", b"value"))?;
//! wal.sync()?; // Ensure durability
//!
//! // Checkpoint periodically to truncate the WAL
//! wal.checkpoint()?;
//! ```

mod entry;
mod error;
mod recovery;
mod writer;

pub use entry::{Operation, WalEntry};
pub use error::{WalError, WalResult};
pub use recovery::{RecoveryMode, RecoveryStats, WalRecovery};
pub use writer::{WalConfig, WalWriter};

/// Log Sequence Number - monotonically increasing identifier for WAL entries
pub type Lsn = u64;

/// Transaction ID for grouping operations
pub type TxnId = u64;

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_wal_basic_operations() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Create writer and append entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();

            // Write some operations
            wal.append(WalEntry::put(1, "nodes", b"key1", b"value1")).unwrap();
            wal.append(WalEntry::put(2, "nodes", b"key2", b"value2")).unwrap();
            wal.append(WalEntry::delete(3, "nodes", b"key1")).unwrap();
            wal.sync().unwrap();
        }

        // Recover and verify
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let entries: Vec<_> = recovery.iter().collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].as_ref().unwrap().lsn, 1);
        assert_eq!(entries[1].as_ref().unwrap().lsn, 2);
        assert_eq!(entries[2].as_ref().unwrap().lsn, 3);
    }

    #[test]
    fn test_wal_transaction_boundaries() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_txn.wal");

        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();

            // Transaction 1
            wal.append(WalEntry::begin_txn(1, 100)).unwrap();
            wal.append(WalEntry::put_in_txn(2, 100, "nodes", b"k1", b"v1")).unwrap();
            wal.append(WalEntry::commit_txn(3, 100)).unwrap();

            // Transaction 2 (uncommitted - simulates crash)
            wal.append(WalEntry::begin_txn(4, 101)).unwrap();
            wal.append(WalEntry::put_in_txn(5, 101, "nodes", b"k2", b"v2")).unwrap();
            // No commit!

            wal.sync().unwrap();
        }

        // Recovery should only return committed transaction entries
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let committed: Vec<_> = recovery.committed_entries().collect();

        // Only txn 100's operations should be recovered
        assert_eq!(committed.len(), 1);
        assert_eq!(committed[0].table.as_deref(), Some("nodes"));
        assert_eq!(committed[0].key.as_deref(), Some(b"k1".as_slice()));
    }

    #[test]
    fn test_wal_checkpoint() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_ckpt.wal");

        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();

            // Write some entries
            for i in 1..=10 {
                wal.append(WalEntry::put(i, "nodes", &[i as u8], &[i as u8])).unwrap();
            }
            wal.sync().unwrap();

            // Checkpoint at LSN 5
            wal.checkpoint(5).unwrap();
        }

        // After checkpoint, the WAL is truncated.
        // A checkpoint marker was written at LSN 11, then entries 6-10 were kept
        // plus the checkpoint entry itself.
        // The truncated WAL contains: entries 6-10 (5 entries) plus checkpoint marker (1 entry)
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let entries: Vec<_> = recovery.iter().filter_map(|e| e.ok()).collect();

        // Entries 6-10 should remain after truncation, plus the checkpoint entry at LSN 11
        assert!(entries.len() >= 5);
        // First entry after truncation should be LSN 6
        assert_eq!(entries[0].lsn, 6);
    }

    #[test]
    fn test_wal_corruption_detection() {
        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test_corrupt.wal");

        // Write valid entries
        {
            let config = WalConfig::default();
            let mut wal = WalWriter::open(&wal_path, config).unwrap();
            wal.append(WalEntry::put(1, "nodes", b"key", b"value")).unwrap();
            wal.sync().unwrap();
        }

        // Corrupt the file by appending garbage that looks like a valid length
        // followed by data that will fail checksum
        {
            use std::io::Write;
            let mut file = fs::OpenOptions::new().append(true).open(&wal_path).unwrap();
            // Write a small valid-looking length (8 bytes) followed by garbage data
            let fake_len: u32 = 8;
            file.write_all(&fake_len.to_le_bytes()).unwrap();
            file.write_all(b"garbage!").unwrap(); // 8 bytes of garbage
            file.write_all(&[0u8; 4]).unwrap(); // Fake CRC that won't match
        }

        // Recovery should detect corruption
        let recovery = WalRecovery::open(&wal_path).unwrap();
        let mut found_valid = false;
        let mut found_error = false;

        for entry in recovery.iter() {
            match entry {
                Ok(e) if e.lsn == 1 => found_valid = true,
                Err(_) => found_error = true,
                _ => {}
            }
        }

        // First entry should be valid
        assert!(found_valid, "Expected to find valid first entry");
        // Should detect corruption (either checksum mismatch or deserialization error)
        assert!(found_error, "Expected to detect corruption");
    }
}
