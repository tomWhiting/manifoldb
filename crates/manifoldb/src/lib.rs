//! `ManifoldDB` - A Multi-Paradigm Embedded Database
//!
//! ManifoldDB is an embedded database that unifies graph, vector, and relational
//! data operations in a single system.
//!
//! # Features
//!
//! - **Graph Database**: Store and traverse nodes and relationships
//! - **Vector Database**: Store embeddings and perform similarity search
//! - **SQL Support**: Query using familiar SQL syntax with extensions
//! - **ACID Transactions**: Full transactional support across all operations
//!
//! # Quick Start
//!
//! ## Opening a Database
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! // Open or create a database file
//! let db = Database::open("mydb.manifold")?;
//!
//! // Or create an in-memory database for testing
//! let db = Database::in_memory()?;
//! ```
//!
//! ## Using Transactions
//!
//! ManifoldDB provides ACID transactions for all operations:
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::in_memory()?;
//!
//! // Write transaction
//! let mut tx = db.begin()?;
//! let entity = tx.create_entity()?.with_label("Person").with_property("name", "Alice");
//! tx.put_entity(&entity)?;
//! tx.commit()?;
//!
//! // Read transaction
//! let tx = db.begin_read()?;
//! if let Some(entity) = tx.get_entity(entity_id)? {
//!     println!("Found: {:?}", entity);
//! }
//! ```
//!
//! ## Executing SQL Queries
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::open("mydb.manifold")?;
//!
//! // Execute statements
//! db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")?;
//!
//! // Query data
//! let results = db.query("SELECT * FROM users WHERE age > 25")?;
//! for row in results {
//!     println!("{:?}", row);
//! }
//! ```
//!
//! ## Graph Operations
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::in_memory()?;
//! let mut tx = db.begin()?;
//!
//! // Create nodes
//! let alice = tx.create_entity()?.with_label("Person").with_property("name", "Alice");
//! let bob = tx.create_entity()?.with_label("Person").with_property("name", "Bob");
//! tx.put_entity(&alice)?;
//! tx.put_entity(&bob)?;
//!
//! // Create edges
//! let follows = tx.create_edge(alice.id, bob.id, "FOLLOWS")?;
//! tx.put_edge(&follows)?;
//!
//! tx.commit()?;
//!
//! // Query with graph patterns
//! let friends = db.query("
//!     SELECT * FROM users
//!     MATCH (u)-[:FOLLOWS]->(f)
//!     WHERE u.name = 'Alice'
//! ")?;
//! ```
//!
//! ## Vector Search
//!
//! ```ignore
//! use manifoldb::{Database, Value};
//!
//! let db = Database::open("mydb.manifold")?;
//!
//! // Store vectors as entity properties
//! let mut tx = db.begin()?;
//! let doc = tx.create_entity()?
//!     .with_label("Document")
//!     .with_property("embedding", vec![0.1f32, 0.2, 0.3]);
//! tx.put_entity(&doc)?;
//! tx.commit()?;
//!
//! // Vector similarity search
//! let query_vector = vec![0.1f32, 0.2, 0.3];
//! let similar = db.query_with_params("
//!     SELECT * FROM documents
//!     ORDER BY embedding <-> $1
//!     LIMIT 10
//! ", &[Value::Vector(query_vector)])?;
//! ```
//!
//! # Configuration
//!
//! Use [`DatabaseBuilder`] for advanced configuration:
//!
//! ```ignore
//! use manifoldb::{DatabaseBuilder, VectorSyncStrategy};
//!
//! let db = DatabaseBuilder::new()
//!     .path("mydb.manifold")
//!     .cache_size(128 * 1024 * 1024)  // 128MB cache
//!     .vector_sync_strategy(VectorSyncStrategy::Async)
//!     .open()?;
//! ```
//!
//! # Modules
//!
//! - [`config`] - Database configuration and builder
//! - [`database`] - Main database interface
//! - [`error`] - Error types
//! - [`transaction`] - Transaction management

// Deny unwrap in library code to ensure proper error handling
#![deny(clippy::unwrap_used)]

// Re-export core types
pub use manifoldb_core::{
    CollectionId, DeleteResult, Edge, EdgeId, EdgeType, Entity, EntityId, Label, Property,
    ScoredEntity, ScoredId, TransactionError, TransactionResult, Value, VectorData,
};

// Re-export storage types
pub use manifoldb_storage::{StorageEngine, Transaction};

// Modules
pub mod backup;
pub mod cache;
pub mod config;
pub mod database;
pub mod error;
pub mod execution;
mod filter;
pub mod index;
pub mod metrics;
pub mod prepared;
pub mod schema;
mod search;
pub mod transaction;
pub mod vector;

// Collection module - internal implementation details
// Note: This is being deprecated in favor of the unified Entity API.
// Users should use Entity.with_vector() and db.search() instead.
#[doc(hidden)]
pub mod collection;

// Public API re-exports
pub use config::{Config, DatabaseBuilder};
pub use database::{Database, FromValue, QueryParams, QueryResult, QueryRow};
pub use error::{Error, Result};
pub use filter::Filter;
pub use metrics::{
    CacheMetricsSnapshot, DatabaseMetrics, MetricsSnapshot, QueryMetrics, QueryMetricsSnapshot,
    StorageMetrics, StorageMetricsSnapshot, TransactionMetrics, TransactionMetricsSnapshot,
    VectorMetrics, VectorMetricsSnapshot,
};
pub use prepared::{PreparedStatement, PreparedStatementCache};
pub use search::EntitySearchBuilder;
pub use transaction::{
    BatchWriter, BatchWriterConfig, BatchedTransaction, DatabaseTransaction, TransactionManager,
    TransactionManagerConfig, VectorSyncStrategy, WriteBuffer, WriteOp, WriteQueue,
};

// Re-export distance metrics for collection configuration
pub use manifoldb_vector::distance::sparse::SparseDistanceMetric;
pub use manifoldb_vector::distance::DistanceMetric;

// Re-export index types
pub use index::{IndexInfo, IndexMetadata, IndexStats, IndexType};
