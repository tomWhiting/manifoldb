//! WAL-enabled storage engine wrapper
//!
//! This module provides a storage engine that wraps any [`StorageEngine`]
//! implementation and adds write-ahead logging for enhanced durability
//! and crash recovery.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::engine::{StorageEngine, StorageError, Transaction};
use crate::wal::{Lsn, Operation, TxnId, WalConfig, WalEntry, WalRecovery, WalWriter};

/// Configuration for the WAL-enabled storage engine
#[derive(Debug, Clone)]
pub struct WalEngineConfig {
    /// WAL configuration
    pub wal: WalConfig,

    /// Auto-checkpoint after this many operations (0 = disabled)
    pub auto_checkpoint_ops: usize,

    /// Recover from WAL on startup
    pub recover_on_open: bool,
}

impl Default for WalEngineConfig {
    fn default() -> Self {
        Self { wal: WalConfig::default(), auto_checkpoint_ops: 10000, recover_on_open: true }
    }
}

/// A storage engine wrapper that provides write-ahead logging
///
/// This wraps any [`StorageEngine`] implementation and logs all write
/// operations to a WAL file before applying them to the underlying store.
/// This provides:
///
/// - **Crash recovery**: Uncommitted operations can be replayed after a crash
/// - **Durability**: Even if the main store hasn't flushed, the WAL ensures
///   committed data is persistent
/// - **Replication foundation**: The WAL can be streamed to replicas
///
/// # Example
///
/// ```ignore
/// use manifoldb_storage::backends::{RedbEngine, WalEngine, WalEngineConfig};
///
/// // Create the underlying storage
/// let inner = RedbEngine::open("database.redb")?;
///
/// // Wrap with WAL
/// let config = WalEngineConfig::default();
/// let engine = WalEngine::open(inner, "database.wal", config)?;
///
/// // Use normally - WAL is transparent
/// let mut tx = engine.begin_write()?;
/// tx.put("nodes", b"key", b"value")?;
/// tx.commit()?; // Operations logged to WAL before commit
/// ```
pub struct WalEngine<E: StorageEngine> {
    /// The underlying storage engine
    inner: E,

    /// Path to the WAL file
    wal_path: PathBuf,

    /// WAL writer (protected by mutex for concurrent access)
    wal: Mutex<WalWriter>,

    /// Next transaction ID
    next_txn_id: AtomicU64,

    /// Operations since last checkpoint
    ops_since_checkpoint: AtomicU64,

    /// Configuration
    config: WalEngineConfig,
}

impl<E: StorageEngine> WalEngine<E> {
    /// Open or create a WAL-enabled storage engine
    ///
    /// If the WAL file exists and `recover_on_open` is true, any uncommitted
    /// transactions will be replayed to the underlying storage.
    pub fn open(
        inner: E,
        wal_path: impl AsRef<Path>,
        config: WalEngineConfig,
    ) -> Result<Self, StorageError> {
        let wal_path = wal_path.as_ref().to_path_buf();

        // Recover from existing WAL if present
        if config.recover_on_open && wal_path.exists() {
            Self::recover(&inner, &wal_path)?;
        }

        // Open WAL writer
        let wal = WalWriter::open(&wal_path, config.wal.clone())
            .map_err(|e| StorageError::Open(format!("failed to open WAL: {e}")))?;

        // Determine next transaction ID from WAL state
        let next_txn_id = wal.current_lsn() + 1;

        Ok(Self {
            inner,
            wal_path,
            wal: Mutex::new(wal),
            next_txn_id: AtomicU64::new(next_txn_id),
            ops_since_checkpoint: AtomicU64::new(0),
            config,
        })
    }

    /// Recover by replaying the WAL to the storage engine
    fn recover(inner: &E, wal_path: &Path) -> Result<(), StorageError> {
        let recovery = WalRecovery::open(wal_path)
            .map_err(|e| StorageError::Open(format!("failed to open WAL for recovery: {e}")))?;

        // Iterate through committed entries and apply them
        for entry in recovery.committed_entries() {
            let mut tx = inner.begin_write()?;

            match entry.operation {
                Operation::Put => {
                    if let (Some(table), Some(key), Some(value)) =
                        (&entry.table, &entry.key, &entry.value)
                    {
                        tx.put(table, key, value)?;
                    }
                }
                Operation::Delete => {
                    if let (Some(table), Some(key)) = (&entry.table, &entry.key) {
                        tx.delete(table, key)?;
                    }
                }
                _ => {}
            }

            tx.commit()?;
        }

        Ok(())
    }

    /// Create a checkpoint
    ///
    /// This marks the current LSN as checkpointed, meaning all operations
    /// up to this point have been safely persisted to the main storage.
    /// The WAL can then be truncated.
    pub fn checkpoint(&self) -> Result<(), StorageError> {
        let mut wal =
            self.wal.lock().map_err(|_| StorageError::Internal("WAL lock poisoned".into()))?;

        let current_lsn = wal.current_lsn();
        wal.checkpoint(current_lsn)
            .map_err(|e| StorageError::Transaction(format!("checkpoint failed: {e}")))?;

        self.ops_since_checkpoint.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Get the current LSN
    pub fn current_lsn(&self) -> Lsn {
        self.wal.lock().map(|w| w.current_lsn()).unwrap_or(0)
    }

    /// Get the checkpoint LSN
    pub fn checkpoint_lsn(&self) -> Lsn {
        self.wal.lock().map(|w| w.checkpoint_lsn()).unwrap_or(0)
    }

    /// Get the path to the WAL file
    pub fn wal_path(&self) -> &Path {
        &self.wal_path
    }

    /// Get a reference to the inner storage engine
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    /// Allocate the next transaction ID
    fn allocate_txn_id(&self) -> TxnId {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Increment operation count and auto-checkpoint if needed
    fn increment_ops(&self) {
        let ops = self.ops_since_checkpoint.fetch_add(1, Ordering::SeqCst);
        if self.config.auto_checkpoint_ops > 0 && ops >= self.config.auto_checkpoint_ops as u64 {
            // Best effort auto-checkpoint
            let _ = self.checkpoint();
        }
    }
}

impl<E: StorageEngine> StorageEngine for WalEngine<E> {
    type Transaction<'a>
        = WalTransaction<'a, E>
    where
        E: 'a;

    fn begin_read(&self) -> Result<Self::Transaction<'_>, StorageError> {
        let inner_tx = self.inner.begin_read()?;
        Ok(WalTransaction {
            inner: inner_tx,
            engine: self,
            txn_id: 0, // Read-only, no txn_id needed
            pending_ops: Vec::new(),
            is_read_only: true,
        })
    }

    fn begin_write(&self) -> Result<Self::Transaction<'_>, StorageError> {
        let inner_tx = self.inner.begin_write()?;
        let txn_id = self.allocate_txn_id();

        // Log transaction begin
        {
            let mut wal =
                self.wal.lock().map_err(|_| StorageError::Internal("WAL lock poisoned".into()))?;
            let lsn = wal.next_lsn();
            wal.append(WalEntry::begin_txn(lsn, txn_id))
                .map_err(|e| StorageError::Transaction(format!("failed to log begin_txn: {e}")))?;
        }

        Ok(WalTransaction {
            inner: inner_tx,
            engine: self,
            txn_id,
            pending_ops: Vec::new(),
            is_read_only: false,
        })
    }

    fn flush(&self) -> Result<(), StorageError> {
        // Sync the WAL
        {
            let mut wal =
                self.wal.lock().map_err(|_| StorageError::Internal("WAL lock poisoned".into()))?;
            wal.sync().map_err(|e| StorageError::Io(std::io::Error::other(e.to_string())))?;
        }

        // Flush the underlying storage
        self.inner.flush()
    }
}

/// A pending operation to be logged on commit
struct PendingOp {
    table: String,
    key: Vec<u8>,
    value: Option<Vec<u8>>, // None for delete
}

/// A transaction that logs operations to the WAL
pub struct WalTransaction<'a, E: StorageEngine + 'a> {
    /// The underlying transaction
    inner: E::Transaction<'a>,

    /// Reference to the engine for WAL access
    engine: &'a WalEngine<E>,

    /// Transaction ID (0 for read-only)
    txn_id: TxnId,

    /// Operations pending WAL logging (logged on commit)
    pending_ops: Vec<PendingOp>,

    /// Whether this is a read-only transaction
    is_read_only: bool,
}

impl<'a, E: StorageEngine + 'a> Transaction for WalTransaction<'a, E> {
    type Cursor<'c>
        = <E::Transaction<'a> as Transaction>::Cursor<'c>
    where
        Self: 'c;

    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError> {
        self.inner.get(table, key)
    }

    fn put(&mut self, table: &str, key: &[u8], value: &[u8]) -> Result<(), StorageError> {
        if self.is_read_only {
            return Err(StorageError::ReadOnly);
        }

        // Record for WAL logging
        self.pending_ops.push(PendingOp {
            table: table.to_string(),
            key: key.to_vec(),
            value: Some(value.to_vec()),
        });

        // Apply to underlying transaction
        self.inner.put(table, key, value)
    }

    fn delete(&mut self, table: &str, key: &[u8]) -> Result<bool, StorageError> {
        if self.is_read_only {
            return Err(StorageError::ReadOnly);
        }

        // Record for WAL logging
        self.pending_ops.push(PendingOp {
            table: table.to_string(),
            key: key.to_vec(),
            value: None,
        });

        // Apply to underlying transaction
        self.inner.delete(table, key)
    }

    fn cursor(&self, table: &str) -> Result<Self::Cursor<'_>, StorageError> {
        self.inner.cursor(table)
    }

    fn range(
        &self,
        table: &str,
        start: std::ops::Bound<&[u8]>,
        end: std::ops::Bound<&[u8]>,
    ) -> Result<Self::Cursor<'_>, StorageError> {
        self.inner.range(table, start, end)
    }

    fn commit(self) -> Result<(), StorageError> {
        if self.is_read_only {
            return Ok(());
        }

        // Log all operations to WAL before committing to storage
        {
            let mut wal = self
                .engine
                .wal
                .lock()
                .map_err(|_| StorageError::Internal("WAL lock poisoned".into()))?;

            for op in &self.pending_ops {
                let lsn = wal.next_lsn();
                let entry = if let Some(value) = &op.value {
                    WalEntry::put_in_txn(lsn, self.txn_id, &op.table, &op.key, value)
                } else {
                    WalEntry::delete_in_txn(lsn, self.txn_id, &op.table, &op.key)
                };

                wal.append(entry).map_err(|e| {
                    StorageError::Transaction(format!("failed to log operation: {e}"))
                })?;
            }

            // Log commit marker
            let commit_lsn = wal.next_lsn();
            wal.append(WalEntry::commit_txn(commit_lsn, self.txn_id))
                .map_err(|e| StorageError::Transaction(format!("failed to log commit: {e}")))?;

            // Sync WAL to ensure durability before storage commit
            wal.sync()
                .map_err(|e| StorageError::Transaction(format!("failed to sync WAL: {e}")))?;
        }

        // Now commit to underlying storage
        self.inner.commit()?;

        // Track operations for auto-checkpoint
        for _ in 0..self.pending_ops.len() {
            self.engine.increment_ops();
        }

        Ok(())
    }

    fn rollback(self) -> Result<(), StorageError> {
        if self.is_read_only {
            return Ok(());
        }

        // Log abort to WAL
        {
            let mut wal = self
                .engine
                .wal
                .lock()
                .map_err(|_| StorageError::Internal("WAL lock poisoned".into()))?;

            let abort_lsn = wal.next_lsn();
            wal.append(WalEntry::abort_txn(abort_lsn, self.txn_id))
                .map_err(|e| StorageError::Transaction(format!("failed to log abort: {e}")))?;
        }

        // Rollback underlying transaction
        self.inner.rollback()
    }

    fn is_read_only(&self) -> bool {
        self.is_read_only
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::RedbEngine;
    use tempfile::tempdir;

    #[test]
    fn test_wal_engine_basic() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let wal_path = dir.path().join("test.wal");

        let inner = RedbEngine::open(&db_path).unwrap();
        let config = WalEngineConfig::default();
        let engine = WalEngine::open(inner, &wal_path, config).unwrap();

        // Write some data
        {
            let mut tx = engine.begin_write().unwrap();
            tx.put("nodes", b"key1", b"value1").unwrap();
            tx.put("nodes", b"key2", b"value2").unwrap();
            tx.commit().unwrap();
        }

        // Read it back
        {
            let tx = engine.begin_read().unwrap();
            assert_eq!(tx.get("nodes", b"key1").unwrap(), Some(b"value1".to_vec()));
            assert_eq!(tx.get("nodes", b"key2").unwrap(), Some(b"value2".to_vec()));
        }

        // WAL file should exist
        assert!(wal_path.exists());
    }

    #[test]
    fn test_wal_engine_recovery() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let wal_path = dir.path().join("test.wal");

        // Write data with WAL
        {
            let inner = RedbEngine::open(&db_path).unwrap();
            let config = WalEngineConfig::default();
            let engine = WalEngine::open(inner, &wal_path, config).unwrap();

            let mut tx = engine.begin_write().unwrap();
            tx.put("nodes", b"key", b"value").unwrap();
            tx.commit().unwrap();
        }

        // Simulate crash: create new storage (empty) but with existing WAL
        {
            // Delete the redb file to simulate losing it
            std::fs::remove_file(&db_path).unwrap();

            // Create fresh storage
            let inner = RedbEngine::open(&db_path).unwrap();
            let config = WalEngineConfig { recover_on_open: true, ..Default::default() };
            let engine = WalEngine::open(inner, &wal_path, config).unwrap();

            // Data should be recovered from WAL
            let tx = engine.begin_read().unwrap();
            assert_eq!(tx.get("nodes", b"key").unwrap(), Some(b"value".to_vec()));
        }
    }

    #[test]
    fn test_wal_engine_rollback() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let wal_path = dir.path().join("test.wal");

        let inner = RedbEngine::open(&db_path).unwrap();
        let config = WalEngineConfig::default();
        let engine = WalEngine::open(inner, &wal_path, config).unwrap();

        // Start a transaction but rollback
        {
            let mut tx = engine.begin_write().unwrap();
            tx.put("nodes", b"key", b"value").unwrap();
            tx.rollback().unwrap();
        }

        // Data should not be present
        {
            let tx = engine.begin_read().unwrap();
            assert_eq!(tx.get("nodes", b"key").unwrap(), None);
        }
    }

    #[test]
    fn test_wal_engine_checkpoint() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let wal_path = dir.path().join("test.wal");

        let inner = RedbEngine::open(&db_path).unwrap();
        let config = WalEngineConfig::default();
        let engine = WalEngine::open(inner, &wal_path, config).unwrap();

        // Write data
        {
            let mut tx = engine.begin_write().unwrap();
            for i in 0..10 {
                tx.put("nodes", &[i], &[i]).unwrap();
            }
            tx.commit().unwrap();
        }

        // Checkpoint
        engine.checkpoint().unwrap();

        // WAL should still function after checkpoint
        {
            let mut tx = engine.begin_write().unwrap();
            tx.put("nodes", b"after_ckpt", b"value").unwrap();
            tx.commit().unwrap();
        }

        let tx = engine.begin_read().unwrap();
        assert_eq!(tx.get("nodes", b"after_ckpt").unwrap(), Some(b"value".to_vec()));
    }

    #[test]
    fn test_wal_engine_read_only() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test.redb");
        let wal_path = dir.path().join("test.wal");

        let inner = RedbEngine::open(&db_path).unwrap();
        let config = WalEngineConfig::default();
        let engine = WalEngine::open(inner, &wal_path, config).unwrap();

        // Read transaction should not allow writes
        let mut tx = engine.begin_read().unwrap();
        assert!(tx.is_read_only());

        let result = tx.put("nodes", b"key", b"value");
        assert!(matches!(result, Err(StorageError::ReadOnly)));
    }
}
