//! Transaction management for `ManifoldDB`.
//!
//! This module provides the [`TransactionManager`] and [`DatabaseTransaction`] types
//! that coordinate transactions across storage, graph indexes, and vector indexes.
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

mod handle;
mod manager;

pub use handle::DatabaseTransaction;
pub use manager::{TransactionManager, TransactionManagerConfig, VectorSyncStrategy};
