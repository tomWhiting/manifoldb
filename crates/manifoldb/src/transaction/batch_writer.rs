//! Write batching for concurrent transaction optimization.
//!
//! This module provides a [`BatchWriter`] that groups multiple transactions
//! together for a single commit, improving throughput under concurrent load.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         BatchWriter                                  │
//! │  - Accepts writes from multiple concurrent transactions             │
//! │  - Buffers operations per-transaction for isolation                 │
//! │  - Groups commits for efficiency                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       WriteQueue                                     │
//! │  - Thread-safe queue of pending write batches                        │
//! │  - Configurable batch size and flush interval                        │
//! │  - Notifies waiters when commit completes                            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Group Commit Strategy
//!
//! When a transaction commits, instead of immediately flushing to storage:
//!
//! 1. The transaction's writes are added to a pending batch
//! 2. If batch size threshold is reached, trigger a group commit
//! 3. If flush interval elapses, trigger a group commit
//! 4. All transactions in the batch are committed together
//! 5. Waiters are notified of success or failure
//!
//! This amortizes the cost of commit (fsync) across multiple transactions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use manifoldb_core::TransactionError;
use manifoldb_storage::{StorageEngine, Transaction};

/// Configuration for batch writer behavior.
#[derive(Debug, Clone)]
pub struct BatchWriterConfig {
    /// Maximum number of transactions to batch before forcing a commit.
    /// Default: 100 transactions.
    pub max_batch_size: usize,

    /// Maximum time to wait for more transactions before committing.
    /// Default: 10 milliseconds.
    pub flush_interval: Duration,

    /// Whether batching is enabled. If false, commits happen immediately.
    /// Default: true.
    pub enabled: bool,
}

impl Default for BatchWriterConfig {
    fn default() -> Self {
        Self { max_batch_size: 100, flush_interval: Duration::from_millis(10), enabled: true }
    }
}

impl BatchWriterConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum batch size.
    #[must_use]
    pub const fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set the flush interval.
    #[must_use]
    pub const fn flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Enable or disable batching.
    #[must_use]
    pub const fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Create a configuration that disables batching (immediate commits).
    #[must_use]
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }
}

/// A single write operation in a batch.
#[derive(Debug, Clone)]
pub enum WriteOp {
    /// Put a key-value pair.
    Put {
        /// The table name.
        table: String,
        /// The key.
        key: Vec<u8>,
        /// The value.
        value: Vec<u8>,
    },
    /// Delete a key.
    Delete {
        /// The table name.
        table: String,
        /// The key.
        key: Vec<u8>,
    },
}

/// A buffer of writes for a single logical transaction.
///
/// This provides isolation: each transaction sees its own uncommitted writes.
#[derive(Debug, Default)]
pub struct WriteBuffer {
    /// Operations in order they were applied.
    ops: Vec<WriteOp>,
    /// Index by (table, key) for read-your-own-writes.
    /// Value is index into `ops` or None if deleted.
    index: HashMap<(String, Vec<u8>), Option<usize>>,
}

impl WriteBuffer {
    /// Create a new empty write buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a put operation.
    pub fn put(&mut self, table: String, key: Vec<u8>, value: Vec<u8>) {
        let idx = self.ops.len();
        self.ops.push(WriteOp::Put { table: table.clone(), key: key.clone(), value });
        self.index.insert((table, key), Some(idx));
    }

    /// Record a delete operation.
    pub fn delete(&mut self, table: String, key: Vec<u8>) {
        self.ops.push(WriteOp::Delete { table: table.clone(), key: key.clone() });
        self.index.insert((table, key), None);
    }

    /// Get a value from the buffer, if written.
    ///
    /// Returns:
    /// - `Some(Some(value))` if the key was written
    /// - `Some(None)` if the key was deleted
    /// - `None` if the key was not modified in this buffer
    #[must_use]
    pub fn get(&self, table: &str, key: &[u8]) -> Option<Option<&[u8]>> {
        self.index.get(&(table.to_string(), key.to_vec())).map(|idx| {
            idx.map(|i| {
                if let WriteOp::Put { value, .. } = &self.ops[i] {
                    value.as_slice()
                } else {
                    // This shouldn't happen due to how we maintain the index
                    &[][..]
                }
            })
        })
    }

    /// Check if a key was deleted in this buffer.
    #[must_use]
    pub fn is_deleted(&self, table: &str, key: &[u8]) -> bool {
        matches!(self.index.get(&(table.to_string(), key.to_vec())), Some(None))
    }

    /// Get the number of operations in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Take ownership of all operations.
    #[must_use]
    pub fn into_ops(self) -> Vec<WriteOp> {
        self.ops
    }

    /// Get all operations as a slice.
    #[must_use]
    pub fn ops(&self) -> &[WriteOp] {
        &self.ops
    }
}

/// Status of a pending batch entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchEntryStatus {
    /// Waiting to be committed.
    Pending,
    /// Commit succeeded.
    Committed,
    /// Commit failed.
    Failed,
}

/// A pending batch entry waiting for group commit.
struct PendingEntry {
    /// Transaction ID for tracking.
    tx_id: u64,
    /// The write operations to apply.
    ops: Vec<WriteOp>,
    /// Current status.
    status: BatchEntryStatus,
    /// Error message if failed.
    error: Option<String>,
}

/// Internal state for the write queue.
struct WriteQueueState {
    /// Pending entries waiting for commit.
    pending: Vec<PendingEntry>,
    /// When the current batch started.
    batch_start: Instant,
}

/// A thread-safe queue for batching writes across transactions.
pub struct WriteQueue<E: StorageEngine> {
    /// The underlying storage engine.
    engine: Arc<E>,
    /// Configuration.
    config: BatchWriterConfig,
    /// Internal state protected by mutex.
    state: Mutex<WriteQueueState>,
    /// Condition variable for waiters.
    commit_complete: Condvar,
    /// Whether a flush is in progress.
    flushing: AtomicBool,
    /// Counter for generating unique transaction IDs within batches.
    tx_counter: AtomicU64,
}

impl<E: StorageEngine> WriteQueue<E> {
    /// Create a new write queue with the given engine and config.
    pub fn new(engine: Arc<E>, config: BatchWriterConfig) -> Self {
        Self {
            engine,
            config,
            state: Mutex::new(WriteQueueState { pending: Vec::new(), batch_start: Instant::now() }),
            commit_complete: Condvar::new(),
            flushing: AtomicBool::new(false),
            tx_counter: AtomicU64::new(0),
        }
    }

    /// Generate a unique transaction ID for this batch writer.
    #[must_use]
    pub fn next_tx_id(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::Relaxed)
    }

    /// Submit a batch of writes for group commit.
    ///
    /// This will block until the batch containing these writes is committed
    /// (or fails).
    pub fn submit(&self, tx_id: u64, ops: Vec<WriteOp>) -> Result<(), TransactionError> {
        if !self.config.enabled || ops.is_empty() {
            // Batching disabled or no writes - commit immediately
            return self.commit_immediately(ops);
        }

        // Add to pending batch
        let should_flush = {
            let mut state = self.state.lock().map_err(|e| {
                TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
            })?;

            // If this is the first entry, record batch start time
            if state.pending.is_empty() {
                state.batch_start = Instant::now();
            }

            state.pending.push(PendingEntry {
                tx_id,
                ops,
                status: BatchEntryStatus::Pending,
                error: None,
            });

            // Check if we should flush
            state.pending.len() >= self.config.max_batch_size
        };

        if should_flush {
            self.flush()?;
        } else {
            // Check if flush interval elapsed
            self.maybe_flush_on_timeout()?;
        }

        // Wait for our entry to be committed
        self.wait_for_commit(tx_id)
    }

    /// Commit operations immediately without batching.
    fn commit_immediately(&self, ops: Vec<WriteOp>) -> Result<(), TransactionError> {
        if ops.is_empty() {
            return Ok(());
        }

        let mut tx = self.engine.begin_write().map_err(|e| {
            TransactionError::Storage(format!("failed to begin write transaction: {e}"))
        })?;

        for op in ops {
            match op {
                WriteOp::Put { table, key, value } => {
                    tx.put(&table, &key, &value)
                        .map_err(|e| TransactionError::Storage(format!("put failed: {e}")))?;
                }
                WriteOp::Delete { table, key } => {
                    tx.delete(&table, &key)
                        .map_err(|e| TransactionError::Storage(format!("delete failed: {e}")))?;
                }
            }
        }

        tx.commit().map_err(|e| TransactionError::Storage(format!("commit failed: {e}")))
    }

    /// Check if flush interval elapsed and trigger flush if needed.
    fn maybe_flush_on_timeout(&self) -> Result<(), TransactionError> {
        let should_flush = {
            let state = self.state.lock().map_err(|e| {
                TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
            })?;

            !state.pending.is_empty() && state.batch_start.elapsed() >= self.config.flush_interval
        };

        if should_flush {
            self.flush()?;
        }

        Ok(())
    }

    /// Flush all pending writes to storage.
    pub fn flush(&self) -> Result<(), TransactionError> {
        // Only one thread should flush at a time
        if self.flushing.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err()
        {
            // Another thread is already flushing
            return Ok(());
        }

        let result = self.do_flush();

        self.flushing.store(false, Ordering::SeqCst);
        self.commit_complete.notify_all();

        result
    }

    /// Internal flush implementation.
    fn do_flush(&self) -> Result<(), TransactionError> {
        // Collect all pending entries
        let entries: Vec<PendingEntry> = {
            let mut state = self.state.lock().map_err(|e| {
                TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
            })?;

            std::mem::take(&mut state.pending)
        };

        if entries.is_empty() {
            return Ok(());
        }

        // Begin a write transaction and apply all operations
        let commit_result = self.apply_batch(&entries);

        // Update entry statuses
        {
            let mut state = self.state.lock().map_err(|e| {
                TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
            })?;

            // Re-add entries with updated status
            for mut entry in entries {
                match &commit_result {
                    Ok(()) => {
                        entry.status = BatchEntryStatus::Committed;
                    }
                    Err(e) => {
                        entry.status = BatchEntryStatus::Failed;
                        entry.error = Some(e.to_string());
                    }
                }
                state.pending.push(entry);
            }
        }

        commit_result
    }

    /// Apply a batch of entries to storage.
    fn apply_batch(&self, entries: &[PendingEntry]) -> Result<(), TransactionError> {
        let mut tx = self.engine.begin_write().map_err(|e| {
            TransactionError::Storage(format!("failed to begin write transaction: {e}"))
        })?;

        for entry in entries {
            for op in &entry.ops {
                match op {
                    WriteOp::Put { table, key, value } => {
                        tx.put(table, key, value)
                            .map_err(|e| TransactionError::Storage(format!("put failed: {e}")))?;
                    }
                    WriteOp::Delete { table, key } => {
                        tx.delete(table, key).map_err(|e| {
                            TransactionError::Storage(format!("delete failed: {e}"))
                        })?;
                    }
                }
            }
        }

        tx.commit().map_err(|e| TransactionError::Storage(format!("commit failed: {e}")))
    }

    /// Wait for a transaction to be committed.
    fn wait_for_commit(&self, tx_id: u64) -> Result<(), TransactionError> {
        loop {
            // First check our entry status
            {
                let mut state = self.state.lock().map_err(|e| {
                    TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
                })?;

                // Look for our entry
                let mut found_idx = None;
                for (i, entry) in state.pending.iter().enumerate() {
                    if entry.tx_id == tx_id {
                        match entry.status {
                            BatchEntryStatus::Pending => {
                                // Still waiting - will check if flush needed below
                            }
                            BatchEntryStatus::Committed => {
                                found_idx = Some(i);
                                break;
                            }
                            BatchEntryStatus::Failed => {
                                let error =
                                    entry.error.clone().unwrap_or_else(|| "unknown".to_string());
                                // Remove the entry
                                state.pending.remove(i);
                                return Err(TransactionError::Storage(format!(
                                    "batch commit failed: {error}"
                                )));
                            }
                        }
                    }
                }

                if let Some(idx) = found_idx {
                    // Remove the committed entry
                    state.pending.remove(idx);
                    return Ok(());
                }
            }

            // Check if we should flush (outside of the lock)
            self.maybe_flush_on_timeout()?;

            // Wait for notification with a timeout (to check flush interval periodically)
            {
                let state = self.state.lock().map_err(|e| {
                    TransactionError::Internal(format!("failed to acquire write queue lock: {e}"))
                })?;

                // Wait with timeout so we can check flush interval periodically
                let _result = self
                    .commit_complete
                    .wait_timeout(state, Duration::from_millis(1))
                    .map_err(|e| {
                        TransactionError::Internal(format!("condition variable wait failed: {e}"))
                    })?;
            }
        }
    }

    /// Get the number of pending entries.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.state.lock().map(|s| s.pending.len()).unwrap_or(0)
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &BatchWriterConfig {
        &self.config
    }
}

/// A batch writer that coordinates write batching across transactions.
///
/// This is the main entry point for write batching. It wraps a storage engine
/// and provides batched transaction operations.
pub struct BatchWriter<E: StorageEngine> {
    /// The write queue for batching.
    queue: Arc<WriteQueue<E>>,
}

impl<E: StorageEngine> BatchWriter<E> {
    /// Create a new batch writer with the given engine.
    pub fn new(engine: Arc<E>, config: BatchWriterConfig) -> Self {
        Self { queue: Arc::new(WriteQueue::new(engine, config)) }
    }

    /// Create a new batch writer with default configuration.
    pub fn with_defaults(engine: Arc<E>) -> Self {
        Self::new(engine, BatchWriterConfig::default())
    }

    /// Get the write queue.
    #[must_use]
    pub fn queue(&self) -> &Arc<WriteQueue<E>> {
        &self.queue
    }

    /// Create a new batched transaction.
    ///
    /// Returns a `BatchedTransaction` that buffers writes and commits them
    /// through the batch writer for group commit.
    #[must_use]
    pub fn begin(&self) -> BatchedTransaction<E> {
        let tx_id = self.queue.next_tx_id();
        BatchedTransaction::new(tx_id, Arc::clone(&self.queue))
    }

    /// Flush any pending writes immediately.
    pub fn flush(&self) -> Result<(), TransactionError> {
        self.queue.flush()
    }

    /// Get the number of pending entries.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.pending_count()
    }
}

impl<E: StorageEngine> Clone for BatchWriter<E> {
    fn clone(&self) -> Self {
        Self { queue: Arc::clone(&self.queue) }
    }
}

/// A transaction that buffers writes for batch commit.
///
/// This transaction maintains its own write buffer for isolation - each
/// transaction sees its own uncommitted writes. On commit, all writes are
/// submitted to the batch writer for group commit.
pub struct BatchedTransaction<E: StorageEngine> {
    /// Unique transaction ID.
    tx_id: u64,
    /// The write queue for commit.
    queue: Arc<WriteQueue<E>>,
    /// Buffered writes for this transaction.
    buffer: WriteBuffer,
    /// Whether the transaction has been completed.
    completed: bool,
}

impl<E: StorageEngine> BatchedTransaction<E> {
    /// Create a new batched transaction.
    fn new(tx_id: u64, queue: Arc<WriteQueue<E>>) -> Self {
        Self { tx_id, queue, buffer: WriteBuffer::new(), completed: false }
    }

    /// Get the transaction ID.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.tx_id
    }

    /// Read a value, checking the local buffer first.
    ///
    /// This provides read-your-own-writes semantics.
    pub fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        if self.completed {
            return Err(TransactionError::AlreadyCompleted);
        }

        // Check local buffer first
        if let Some(buffered) = self.buffer.get(table, key) {
            return Ok(buffered.map(|v| v.to_vec()));
        }

        // Read from storage
        let tx = self.queue.engine.begin_read().map_err(|e| {
            TransactionError::Storage(format!("failed to begin read transaction: {e}"))
        })?;

        tx.get(table, key).map_err(|e| TransactionError::Storage(format!("get failed: {e}")))
    }

    /// Buffer a put operation.
    pub fn put(&mut self, table: &str, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        if self.completed {
            return Err(TransactionError::AlreadyCompleted);
        }

        self.buffer.put(table.to_string(), key.to_vec(), value.to_vec());
        Ok(())
    }

    /// Buffer a delete operation.
    pub fn delete(&mut self, table: &str, key: &[u8]) -> Result<bool, TransactionError> {
        if self.completed {
            return Err(TransactionError::AlreadyCompleted);
        }

        // Check if key exists (either in buffer or storage)
        let exists = self.get(table, key)?.is_some();

        if exists {
            self.buffer.delete(table.to_string(), key.to_vec());
        }

        Ok(exists)
    }

    /// Commit the transaction through the batch writer.
    ///
    /// This submits all buffered writes to the batch writer for group commit.
    /// The method blocks until the batch containing this transaction is committed.
    pub fn commit(mut self) -> Result<(), TransactionError> {
        if self.completed {
            return Err(TransactionError::AlreadyCompleted);
        }

        self.completed = true;
        let ops = std::mem::take(&mut self.buffer).into_ops();
        self.queue.submit(self.tx_id, ops)
    }

    /// Rollback the transaction, discarding all buffered writes.
    pub fn rollback(mut self) -> Result<(), TransactionError> {
        if self.completed {
            return Err(TransactionError::AlreadyCompleted);
        }

        self.completed = true;
        // Just drop the buffer - nothing was written to storage
        Ok(())
    }

    /// Get the number of buffered operations.
    #[must_use]
    pub fn buffered_ops(&self) -> usize {
        self.buffer.len()
    }
}

impl<E: StorageEngine> Drop for BatchedTransaction<E> {
    fn drop(&mut self) {
        // If not completed, this is an implicit rollback
        // Nothing to do since writes are only in the buffer
        if !self.completed {
            self.completed = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    fn create_test_engine() -> RedbEngine {
        RedbEngine::in_memory().expect("failed to create in-memory engine")
    }

    #[test]
    fn test_write_buffer_basic() {
        let mut buffer = WriteBuffer::new();

        buffer.put("table".to_string(), b"key1".to_vec(), b"value1".to_vec());
        buffer.put("table".to_string(), b"key2".to_vec(), b"value2".to_vec());

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.get("table", b"key1"), Some(Some(b"value1".as_slice())));
        assert_eq!(buffer.get("table", b"key2"), Some(Some(b"value2".as_slice())));
        assert_eq!(buffer.get("table", b"key3"), None);
    }

    #[test]
    fn test_write_buffer_overwrite() {
        let mut buffer = WriteBuffer::new();

        buffer.put("table".to_string(), b"key".to_vec(), b"value1".to_vec());
        buffer.put("table".to_string(), b"key".to_vec(), b"value2".to_vec());

        // Should return the latest value
        assert_eq!(buffer.get("table", b"key"), Some(Some(b"value2".as_slice())));
    }

    #[test]
    fn test_write_buffer_delete() {
        let mut buffer = WriteBuffer::new();

        buffer.put("table".to_string(), b"key".to_vec(), b"value".to_vec());
        buffer.delete("table".to_string(), b"key".to_vec());

        // Should return None (deleted)
        assert_eq!(buffer.get("table", b"key"), Some(None));
        assert!(buffer.is_deleted("table", b"key"));
    }

    #[test]
    fn test_batch_writer_immediate_commit() {
        let engine = Arc::new(create_test_engine());
        let writer = BatchWriter::new(engine.clone(), BatchWriterConfig::disabled());

        let mut tx = writer.begin();
        tx.put("test", b"key", b"value").expect("put failed");
        tx.commit().expect("commit failed");

        // Verify the data was written
        let read_tx = engine.begin_read().expect("begin_read failed");
        let value = read_tx.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"value".to_vec()));
    }

    #[test]
    fn test_batch_writer_read_your_writes() {
        let engine = Arc::new(create_test_engine());
        let writer = BatchWriter::new(engine, BatchWriterConfig::default());

        let mut tx = writer.begin();
        tx.put("test", b"key", b"value").expect("put failed");

        // Should be able to read our own write
        let value = tx.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"value".to_vec()));

        tx.commit().expect("commit failed");
    }

    #[test]
    fn test_batch_writer_isolation() {
        let engine = Arc::new(create_test_engine());
        let writer = BatchWriter::new(engine.clone(), BatchWriterConfig::disabled());

        // Write initial value
        {
            let mut tx = writer.begin();
            tx.put("test", b"key", b"initial").expect("put failed");
            tx.commit().expect("commit failed");
        }

        // Start two transactions
        let mut tx1 = writer.begin();
        let mut tx2 = writer.begin();

        // tx1 writes
        tx1.put("test", b"key", b"tx1_value").expect("put failed");

        // tx2 should not see tx1's write (tx1 hasn't committed)
        let value = tx2.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"initial".to_vec()));

        // tx2 writes something different
        tx2.put("test", b"key", b"tx2_value").expect("put failed");

        // tx2 should see its own write
        let value = tx2.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"tx2_value".to_vec()));

        tx1.commit().expect("commit failed");
        tx2.commit().expect("commit failed");

        // Final value depends on commit order (tx2 committed last)
        let read_tx = engine.begin_read().expect("begin_read failed");
        let value = read_tx.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"tx2_value".to_vec()));
    }

    #[test]
    fn test_batch_writer_rollback() {
        let engine = Arc::new(create_test_engine());
        let writer = BatchWriter::new(engine.clone(), BatchWriterConfig::disabled());

        let mut tx = writer.begin();
        tx.put("test", b"key", b"value").expect("put failed");
        tx.rollback().expect("rollback failed");

        // Verify nothing was written
        let read_tx = engine.begin_read().expect("begin_read failed");
        let value = read_tx.get("test", b"key").expect("get failed");
        assert_eq!(value, None);
    }

    #[test]
    fn test_batch_writer_concurrent() {
        let engine = Arc::new(create_test_engine());
        let writer = BatchWriter::new(
            engine.clone(),
            BatchWriterConfig::default()
                .max_batch_size(10)
                .flush_interval(Duration::from_millis(5)),
        );

        let num_threads = 4;
        let writes_per_thread = 25;
        let counter = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let writer = writer.clone();
                let counter = Arc::clone(&counter);

                thread::spawn(move || {
                    for i in 0..writes_per_thread {
                        let key = format!("thread{thread_id}_key{i}");
                        let value = format!("value{i}");

                        let mut tx = writer.begin();
                        tx.put("test", key.as_bytes(), value.as_bytes()).expect("put failed");
                        tx.commit().expect("commit failed");

                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        assert_eq!(counter.load(Ordering::Relaxed), num_threads * writes_per_thread);

        // Verify all writes are persisted
        let read_tx = engine.begin_read().expect("begin_read failed");
        for thread_id in 0..num_threads {
            for i in 0..writes_per_thread {
                let key = format!("thread{thread_id}_key{i}");
                let expected_value = format!("value{i}");
                let value = read_tx.get("test", key.as_bytes()).expect("get failed");
                assert_eq!(
                    value,
                    Some(expected_value.into_bytes()),
                    "missing or wrong value for {key}"
                );
            }
        }
    }

    #[test]
    fn test_batch_writer_config() {
        let config = BatchWriterConfig::new()
            .max_batch_size(50)
            .flush_interval(Duration::from_millis(20))
            .enabled(true);

        assert_eq!(config.max_batch_size, 50);
        assert_eq!(config.flush_interval, Duration::from_millis(20));
        assert!(config.enabled);

        let disabled = BatchWriterConfig::disabled();
        assert!(!disabled.enabled);
    }
}
