//! `ManifoldDB`
//!
//! A multi-paradigm embedded database that unifies graph, vector, and relational
//! data operations in a single system.
//!
//! # Features
//!
//! - **Graph Database**: Store and traverse nodes and relationships
//! - **Vector Database**: Store embeddings and perform similarity search
//! - **SQL Support**: Query using familiar SQL syntax with extensions
//! - **ACID Transactions**: Full transactional support across all operations
//!
//! # Transaction Example
//!
//! ```ignore
//! use manifoldb::transaction::TransactionManager;
//! use manifoldb_storage::backends::RedbEngine;
//!
//! // Create the transaction manager
//! let engine = RedbEngine::open("mydb.redb")?;
//! let manager = TransactionManager::new(engine);
//!
//! // Write transaction
//! let mut tx = manager.begin_write()?;
//! let entity = tx.create_entity()?.with_label("Person");
//! tx.put_entity(&entity)?;
//! tx.commit()?;
//!
//! // Read transaction
//! let tx = manager.begin_read()?;
//! let entity = tx.get_entity(entity_id)?;
//! ```
//!
//! # SQL Example
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::open("mydb.manifold")?;
//!
//! // Create entities
//! db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")?;
//!
//! // Query with SQL
//! let results = db.query("SELECT * FROM users WHERE age > 25")?;
//!
//! // Graph traversal
//! let friends = db.query("
//!     SELECT * FROM users
//!     MATCH (u)-[:FRIENDS]->(f)
//!     WHERE u.name = 'Alice'
//! ")?;
//!
//! // Vector similarity search
//! let similar = db.query("
//!     SELECT * FROM documents
//!     ORDER BY embedding <-> $query_vector
//!     LIMIT 10
//! ")?;
//! ```

// Re-export core types
pub use manifoldb_core::{
    Edge, EdgeId, EdgeType, Entity, EntityId, Label, Property, TransactionError, TransactionResult,
    Value,
};

// Re-export storage types
pub use manifoldb_storage::{StorageEngine, Transaction};

pub mod config;
pub mod database;
pub mod error;
pub mod transaction;

pub use database::Database;
pub use error::Error;
pub use transaction::{DatabaseTransaction, TransactionManager, VectorSyncStrategy};
