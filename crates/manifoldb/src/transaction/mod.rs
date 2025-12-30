//! Transaction management for `ManifoldDB`.
//!
//! This module provides the [`TransactionManager`] and [`DatabaseTransaction`] types
//! that coordinate transactions across storage, graph indexes, and vector indexes.
//!
//! # Write Batching
//!
//! For high-throughput concurrent writes, the module provides [`BatchWriter`] which
//! groups multiple transactions into a single commit for improved performance.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::transaction::{TransactionManager, VectorSyncStrategy};
//! use manifoldb_storage::backends::RedbEngine;
//!
//! // Create the engine and manager
//! let engine = RedbEngine::open("db.redb")?;
//! let manager = TransactionManager::new(engine);
//!
//! // Write transaction
//! let mut tx = manager.begin_write()?;
//! tx.put_entity(&entity)?;
//! tx.commit()?;
//!
//! // Read transaction
//! let tx = manager.begin_read()?;
//! let entity = tx.get_entity(entity_id)?;
//! ```
//!
//! # Batched Writes
//!
//! ```ignore
//! use manifoldb::transaction::{BatchWriter, BatchWriterConfig};
//! use manifoldb_storage::backends::RedbEngine;
//! use std::sync::Arc;
//!
//! let engine = Arc::new(RedbEngine::open("db.redb")?);
//! let writer = BatchWriter::new(engine, BatchWriterConfig::default());
//!
//! // Multiple threads can use the batch writer concurrently
//! let mut tx = writer.begin();
//! tx.put("table", b"key", b"value")?;
//! tx.commit()?;  // Batched with other concurrent commits
//! ```

mod batch_writer;
mod handle;
mod manager;
mod read_pool;

pub use batch_writer::{
    BatchWriter, BatchWriterConfig, BatchedTransaction, WriteBuffer, WriteOp, WriteQueue,
};
pub use handle::DatabaseTransaction;
pub use manager::{TransactionManager, TransactionManagerConfig, VectorSyncStrategy};
pub use read_pool::{PooledReadTx, ReadPool, ReadPoolConfig};
