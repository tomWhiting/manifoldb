//! Read transaction pool for efficient concurrent reads.
//!
//! This module provides a [`ReadPool`] that manages read transactions with
//! configurable staleness policies based on age and write activity.
//!
//! # Design
//!
//! Due to Rust's lifetime constraints, this implementation doesn't actually
//! pool transactions (which would require unsafe code). Instead, it provides:
//!
//! 1. Centralized transaction creation with consistent configuration
//! 2. Write activity tracking for staleness detection
//! 3. A foundation for future pooling if/when Rust's GAT support improves
//!
//! # Staleness Policy
//!
//! The pool tracks write activity via `notify_write()`. Users of pooled
//! transactions can check staleness based on:
//! - Age (time since creation)
//! - Write count (writes since creation)
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::transaction::{ReadPool, ReadPoolConfig};
//! use manifoldb_storage::backends::RedbEngine;
//! use std::sync::Arc;
//!
//! let engine = Arc::new(RedbEngine::open("db.redb")?);
//! let pool = ReadPool::new(engine, ReadPoolConfig::default())?;
//!
//! // Acquire a read transaction
//! let tx = pool.acquire()?;
//! let value = tx.get("table", b"key")?;
//!
//! // After writes, notify the pool
//! pool.notify_write();
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use manifoldb_core::TransactionError;
use manifoldb_storage::StorageEngine;

/// Configuration for the read transaction pool.
#[derive(Debug, Clone)]
pub struct ReadPoolConfig {
    /// Maximum number of transactions to keep in the pool (future use).
    /// Default: 16
    pub max_size: usize,

    /// Maximum age of a transaction before it's considered stale.
    /// Default: 100ms
    pub max_age: Duration,

    /// Refresh transactions after this many writes to the database.
    /// Default: 100
    pub refresh_after_writes: u64,

    /// Whether to pre-populate the pool on creation (future use).
    /// Default: false
    pub prefill: bool,
}

impl Default for ReadPoolConfig {
    fn default() -> Self {
        Self {
            max_size: 16,
            max_age: Duration::from_millis(100),
            refresh_after_writes: 100,
            prefill: false,
        }
    }
}

impl ReadPoolConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum pool size.
    #[must_use]
    pub const fn max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    /// Set the maximum age for transactions.
    #[must_use]
    pub const fn max_age(mut self, age: Duration) -> Self {
        self.max_age = age;
        self
    }

    /// Set the write threshold for refreshing transactions.
    #[must_use]
    pub const fn refresh_after_writes(mut self, count: u64) -> Self {
        self.refresh_after_writes = count;
        self
    }

    /// Enable or disable pool prefilling.
    #[must_use]
    pub const fn prefill(mut self, prefill: bool) -> Self {
        self.prefill = prefill;
        self
    }

    /// Create a configuration that disables pooling.
    #[must_use]
    pub fn disabled() -> Self {
        Self { max_size: 0, ..Default::default() }
    }
}

/// A pool for managing read transactions with staleness tracking.
///
/// This pool provides:
/// - Centralized transaction creation
/// - Write activity tracking for staleness detection
/// - Configuration for refresh policies
///
/// # Thread Safety
///
/// `ReadPool` is `Send + Sync` and can be shared across threads.
pub struct ReadPool<E: StorageEngine> {
    /// The underlying storage engine.
    engine: Arc<E>,

    /// Configuration for the pool.
    config: ReadPoolConfig,

    /// Counter of writes to the database.
    write_counter: AtomicU64,
}

impl<E: StorageEngine> ReadPool<E> {
    /// Create a new read pool with the given engine and configuration.
    pub fn new(engine: Arc<E>, config: ReadPoolConfig) -> Result<Self, TransactionError> {
        Ok(Self { engine, config, write_counter: AtomicU64::new(0) })
    }

    /// Create a new read pool with default configuration.
    pub fn with_defaults(engine: Arc<E>) -> Result<Self, TransactionError> {
        Self::new(engine, ReadPoolConfig::default())
    }

    /// Acquire a read transaction from the pool.
    ///
    /// Returns a [`PooledReadTx`] that wraps a fresh read transaction.
    /// The transaction includes metadata for staleness checking.
    ///
    /// # Errors
    ///
    /// Returns an error if transaction creation fails.
    pub fn acquire(&self) -> Result<PooledReadTx<'_, E>, TransactionError> {
        let write_count = self.write_counter.load(Ordering::Relaxed);
        let created_at = Instant::now();

        let tx = self
            .engine
            .begin_read()
            .map_err(|e| TransactionError::Storage(format!("failed to begin read: {e}")))?;

        Ok(PooledReadTx { pool: self, tx, created_at, created_at_write: write_count })
    }

    /// Notify the pool that a write has occurred.
    ///
    /// This increments the write counter, which is used to detect stale
    /// transactions. Call this after committing a write transaction.
    pub fn notify_write(&self) {
        self.write_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Notify the pool that multiple writes have occurred.
    ///
    /// This is useful when batch committing multiple writes at once.
    pub fn notify_writes(&self, count: u64) {
        self.write_counter.fetch_add(count, Ordering::Relaxed);
    }

    /// Get the current write counter value.
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_counter.load(Ordering::Relaxed)
    }

    /// Get the number of transactions currently in the pool.
    ///
    /// Note: The current implementation doesn't pool transactions,
    /// so this always returns 0.
    #[must_use]
    pub fn available_count(&self) -> usize {
        0 // No actual pooling yet
    }

    /// Get the pool configuration.
    #[must_use]
    pub const fn config(&self) -> &ReadPoolConfig {
        &self.config
    }

    /// Clear all transactions from the pool.
    ///
    /// Note: The current implementation doesn't pool transactions,
    /// so this is a no-op.
    pub fn clear(&self) {
        // No-op: no actual pooling
    }

    /// Refresh all pooled transactions.
    ///
    /// Note: The current implementation doesn't pool transactions,
    /// so this is a no-op.
    pub fn refresh(&self) {
        self.clear();
    }
}

/// A read transaction from the pool.
///
/// This wrapper provides access to the underlying transaction and includes
/// metadata for staleness checking.
pub struct PooledReadTx<'pool, E: StorageEngine> {
    /// Reference to the pool.
    pool: &'pool ReadPool<E>,
    /// The underlying transaction.
    tx: E::Transaction<'pool>,
    /// When the transaction was created.
    created_at: Instant,
    /// The write counter value when the transaction was created.
    created_at_write: u64,
}

impl<'pool, E: StorageEngine> PooledReadTx<'pool, E> {
    /// Get a reference to the underlying transaction.
    ///
    /// Note: Due to Rust's lifetime constraints with associated types,
    /// direct Deref is not supported. Use this method or the forwarded
    /// methods like `get()` and `is_read_only()`.
    #[must_use]
    pub fn transaction(&self) -> &E::Transaction<'pool> {
        &self.tx
    }

    /// Get the age of this transaction.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Check if this transaction is stale based on age.
    #[must_use]
    pub fn is_stale_by_age(&self) -> bool {
        self.age() > self.pool.config.max_age
    }

    /// Check if this transaction is stale based on write count.
    #[must_use]
    pub fn is_stale_by_writes(&self) -> bool {
        let current_writes = self.pool.write_counter.load(Ordering::Relaxed);
        let writes_since = current_writes.saturating_sub(self.created_at_write);
        writes_since > self.pool.config.refresh_after_writes
    }

    /// Check if this transaction is stale (by either age or write count).
    #[must_use]
    pub fn is_stale(&self) -> bool {
        self.is_stale_by_age() || self.is_stale_by_writes()
    }

    /// Get the write count at the time this transaction was created.
    #[must_use]
    pub fn created_at_write(&self) -> u64 {
        self.created_at_write
    }

    /// Discard this transaction without returning it to the pool.
    ///
    /// Note: Since the current implementation doesn't pool transactions,
    /// this just consumes self.
    pub fn discard(self) {
        // Just drop
    }

    // ========================================================================
    // Forwarded Transaction trait methods
    // ========================================================================

    /// Get a value by key from a table.
    pub fn get(
        &self,
        table: &str,
        key: &[u8],
    ) -> Result<Option<Vec<u8>>, manifoldb_storage::StorageError> {
        use manifoldb_storage::Transaction;
        self.tx.get(table, key)
    }

    /// Check if this is a read-only transaction.
    #[must_use]
    pub fn is_read_only(&self) -> bool {
        use manifoldb_storage::Transaction;
        self.tx.is_read_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;
    use std::thread;

    fn create_test_engine() -> Arc<RedbEngine> {
        Arc::new(RedbEngine::in_memory().expect("failed to create in-memory engine"))
    }

    #[test]
    fn test_config_default() {
        let config = ReadPoolConfig::default();
        assert_eq!(config.max_size, 16);
        assert_eq!(config.max_age, Duration::from_millis(100));
        assert_eq!(config.refresh_after_writes, 100);
        assert!(!config.prefill);
    }

    #[test]
    fn test_config_builder() {
        let config = ReadPoolConfig::new()
            .max_size(8)
            .max_age(Duration::from_millis(50))
            .refresh_after_writes(50)
            .prefill(true);

        assert_eq!(config.max_size, 8);
        assert_eq!(config.max_age, Duration::from_millis(50));
        assert_eq!(config.refresh_after_writes, 50);
        assert!(config.prefill);
    }

    #[test]
    fn test_config_disabled() {
        let config = ReadPoolConfig::disabled();
        assert_eq!(config.max_size, 0);
    }

    #[test]
    fn test_pool_creation() {
        let engine = create_test_engine();
        let pool = ReadPool::new(engine, ReadPoolConfig::default()).expect("pool creation failed");
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_acquire() {
        let engine = create_test_engine();
        let pool = ReadPool::new(engine, ReadPoolConfig::default()).expect("pool creation failed");

        let tx = pool.acquire().expect("acquire failed");
        assert!(tx.is_read_only());
    }

    #[test]
    fn test_staleness_by_age() {
        let engine = create_test_engine();
        let config = ReadPoolConfig::new().max_age(Duration::from_millis(1));
        let pool = ReadPool::new(engine, config).expect("pool creation failed");

        let tx = pool.acquire().expect("acquire failed");
        assert!(!tx.is_stale_by_age());

        // Wait for transaction to become stale
        thread::sleep(Duration::from_millis(5));

        assert!(tx.is_stale_by_age());
        assert!(tx.is_stale());
    }

    #[test]
    fn test_staleness_by_writes() {
        let engine = create_test_engine();
        let config = ReadPoolConfig::new().refresh_after_writes(5);
        let pool = ReadPool::new(engine, config).expect("pool creation failed");

        let tx = pool.acquire().expect("acquire failed");
        assert!(!tx.is_stale_by_writes());

        // Simulate writes
        pool.notify_writes(10);

        assert!(tx.is_stale_by_writes());
        assert!(tx.is_stale());
    }

    #[test]
    fn test_write_counter() {
        let engine = create_test_engine();
        let pool = ReadPool::new(engine, ReadPoolConfig::default()).expect("pool creation failed");

        assert_eq!(pool.write_count(), 0);

        pool.notify_write();
        assert_eq!(pool.write_count(), 1);

        pool.notify_writes(5);
        assert_eq!(pool.write_count(), 6);
    }

    #[test]
    fn test_transaction_operations() {
        let engine = create_test_engine();

        // Write some data
        {
            use manifoldb_storage::Transaction;
            let mut tx = engine.begin_write().expect("begin_write failed");
            tx.put("test", b"key", b"value").expect("put failed");
            tx.commit().expect("commit failed");
        }

        let pool = ReadPool::new(Arc::clone(&engine), ReadPoolConfig::default())
            .expect("pool creation failed");

        // Read through pooled transaction
        let tx = pool.acquire().expect("acquire failed");
        let value = tx.get("test", b"key").expect("get failed");
        assert_eq!(value, Some(b"value".to_vec()));
    }

    #[test]
    fn test_concurrent_access() {
        let engine = create_test_engine();
        let pool = Arc::new(
            ReadPool::new(engine, ReadPoolConfig::default()).expect("pool creation failed"),
        );

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for _ in 0..10 {
                        let tx = pool.acquire().expect("acquire failed");
                        assert!(tx.is_read_only());
                        // Small work
                        thread::sleep(Duration::from_micros(100));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread panicked");
        }
    }
}
