//! Transaction manager implementation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use manifoldb_core::TransactionError;
use manifoldb_storage::{StorageEngine, StorageError};

use super::batch_writer::{BatchWriter, BatchWriterConfig, BatchedTransaction};
use super::handle::DatabaseTransaction;

/// Strategy for synchronizing vector index updates with transactions.
///
/// This enum controls how vector index updates are handled relative to
/// the storage transaction lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorSyncStrategy {
    /// Update vector indexes synchronously within the same transaction.
    ///
    /// This provides strong consistency at the cost of slower writes.
    /// All vector index updates are committed atomically with data changes.
    ///
    /// This is the default and recommended for correctness.
    #[default]
    Synchronous,

    /// Queue vector index updates for asynchronous processing.
    ///
    /// This provides faster writes but eventual consistency for vector searches.
    /// Vector search results may be slightly stale until updates are processed.
    ///
    /// Use this for high-throughput scenarios where temporary staleness is acceptable.
    Async,

    /// Use synchronous updates for small batches, async for large bulk operations.
    ///
    /// This provides a balance between write performance and consistency.
    /// The threshold for switching modes is configurable.
    Hybrid {
        /// Number of vector updates before switching to async mode.
        async_threshold: usize,
    },
}

/// Configuration for the transaction manager.
#[derive(Debug, Clone)]
pub struct TransactionManagerConfig {
    /// Strategy for vector index synchronization.
    pub vector_sync_strategy: VectorSyncStrategy,

    /// Configuration for write batching.
    pub batch_writer_config: BatchWriterConfig,
}

impl Default for TransactionManagerConfig {
    fn default() -> Self {
        Self {
            vector_sync_strategy: VectorSyncStrategy::Synchronous,
            batch_writer_config: BatchWriterConfig::default(),
        }
    }
}

/// Coordinates transactions across storage, graph indexes, and vector indexes.
///
/// The `TransactionManager` is the central coordinator for all database transactions.
/// It manages the lifecycle of transactions and ensures ACID guarantees across
/// all subsystems.
///
/// # Transaction Semantics
///
/// - **Read transactions**: Provide snapshot isolation with multiple concurrent readers
/// - **Write transactions**: Currently serialized (single writer at a time)
/// - All index updates occur within the same transaction as data changes
///
/// # Thread Safety
///
/// `TransactionManager` is `Send + Sync` and can be safely shared across threads
/// using `Arc<TransactionManager>`.
///
/// # Example
///
/// ```ignore
/// use manifoldb::transaction::{TransactionManager, VectorSyncStrategy};
/// use manifoldb_storage::backends::RedbEngine;
///
/// let engine = RedbEngine::open("db.redb")?;
/// let manager = TransactionManager::new(engine);
///
/// // Concurrent reads are allowed
/// let tx1 = manager.begin_read()?;
/// let tx2 = manager.begin_read()?;
///
/// // Write transactions
/// let mut tx = manager.begin_write()?;
/// tx.put_entity(&entity)?;
/// tx.commit()?;
/// ```
pub struct TransactionManager<E: StorageEngine> {
    /// The underlying storage engine.
    engine: Arc<E>,

    /// Configuration for the transaction manager.
    config: TransactionManagerConfig,

    /// Counter for generating unique transaction IDs.
    next_tx_id: AtomicU64,

    /// Batch writer for concurrent write optimization.
    batch_writer: BatchWriter<E>,
}

impl<E: StorageEngine> TransactionManager<E> {
    /// Create a new transaction manager with the given storage engine.
    ///
    /// Uses the default configuration with synchronous vector index updates.
    pub fn new(engine: E) -> Self {
        Self::with_config(engine, TransactionManagerConfig::default())
    }

    /// Create a new transaction manager with custom configuration.
    pub fn with_config(engine: E, config: TransactionManagerConfig) -> Self {
        let engine = Arc::new(engine);
        let batch_writer =
            BatchWriter::new(Arc::clone(&engine), config.batch_writer_config.clone());
        Self { engine, config, next_tx_id: AtomicU64::new(1), batch_writer }
    }

    /// Get the vector synchronization strategy.
    #[must_use]
    pub const fn vector_sync_strategy(&self) -> VectorSyncStrategy {
        self.config.vector_sync_strategy
    }

    /// Begin a read-only transaction.
    ///
    /// Read transactions provide a consistent snapshot of the database.
    /// Multiple read transactions can run concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started.
    pub fn begin_read(&self) -> Result<DatabaseTransaction<E::Transaction<'_>>, TransactionError> {
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
        let storage_tx =
            self.engine.begin_read().map_err(|e| storage_error_to_transaction_error(&e))?;

        Ok(DatabaseTransaction::new_read(tx_id, storage_tx))
    }

    /// Begin a read-write transaction.
    ///
    /// Write transactions allow modifying the database. Currently, write
    /// transactions are serialized (only one at a time).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started.
    pub fn begin_write(&self) -> Result<DatabaseTransaction<E::Transaction<'_>>, TransactionError> {
        let tx_id = self.next_tx_id.fetch_add(1, Ordering::Relaxed);
        let storage_tx =
            self.engine.begin_write().map_err(|e| storage_error_to_transaction_error(&e))?;

        Ok(DatabaseTransaction::new_write(tx_id, storage_tx, self.config.vector_sync_strategy))
    }

    /// Flush any buffered data to durable storage.
    ///
    /// This is typically called after committing important transactions.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<(), TransactionError> {
        self.engine.flush().map_err(|e| storage_error_to_transaction_error(&e))
    }

    /// Get a reference to the underlying storage engine.
    ///
    /// This is primarily useful for testing or advanced use cases.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Get an Arc to the underlying storage engine.
    ///
    /// Returns a cloned Arc reference to the engine, useful for creating
    /// components that need shared ownership of the engine.
    #[must_use]
    pub fn engine_arc(&self) -> Arc<E> {
        Arc::clone(&self.engine)
    }

    /// Begin a batched write transaction.
    ///
    /// Batched transactions buffer writes locally and commit them through
    /// the batch writer for group commit optimization. This can significantly
    /// improve throughput under concurrent load.
    ///
    /// Unlike regular write transactions, batched transactions:
    /// - Buffer all writes in memory until commit
    /// - Are committed together with other concurrent transactions
    /// - Provide read-your-own-writes semantics within the transaction
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut tx = manager.begin_batched_write();
    /// tx.put("table", b"key", b"value")?;
    /// tx.commit()?;  // Batched with other concurrent commits
    /// ```
    #[must_use]
    pub fn begin_batched_write(&self) -> BatchedTransaction<E> {
        self.batch_writer.begin()
    }

    /// Get a reference to the batch writer.
    ///
    /// This provides access to the batch writer for manual control over
    /// batched transactions, including flushing pending writes.
    #[must_use]
    pub fn batch_writer(&self) -> &BatchWriter<E> {
        &self.batch_writer
    }

    /// Flush any pending batched writes immediately.
    ///
    /// This forces all pending batched transactions to be committed,
    /// even if the batch size or flush interval hasn't been reached.
    pub fn flush_batched(&self) -> Result<(), TransactionError> {
        self.batch_writer.flush()
    }

    /// Get the number of pending batched transactions.
    #[must_use]
    pub fn pending_batched_count(&self) -> usize {
        self.batch_writer.pending_count()
    }

    /// Get the batch writer configuration.
    #[must_use]
    pub fn batch_writer_config(&self) -> &BatchWriterConfig {
        &self.config.batch_writer_config
    }
}

/// Convert a storage error to a transaction error.
fn storage_error_to_transaction_error(err: &StorageError) -> TransactionError {
    match err {
        StorageError::ReadOnly => TransactionError::ReadOnly,
        StorageError::Conflict(msg) => TransactionError::Conflict(msg.clone()),
        StorageError::Serialization(msg) => TransactionError::Serialization(msg.clone()),
        StorageError::NotFound(msg) => TransactionError::EntityNotFound(msg.clone()),
        StorageError::KeyNotFound => TransactionError::EntityNotFound("key not found".to_string()),
        _ => TransactionError::Storage(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_sync_strategy_default() {
        let strategy = VectorSyncStrategy::default();
        assert_eq!(strategy, VectorSyncStrategy::Synchronous);
    }

    #[test]
    fn test_config_default() {
        let config = TransactionManagerConfig::default();
        assert_eq!(config.vector_sync_strategy, VectorSyncStrategy::Synchronous);
    }
}
