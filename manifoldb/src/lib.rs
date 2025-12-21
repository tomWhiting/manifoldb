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
//!
//! # Example
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
pub use manifoldb_core::{Edge, EdgeId, EdgeType, Entity, EntityId, Label, Property, Value};

// Re-export storage types
pub use manifoldb_storage::{StorageEngine, Transaction};

pub mod config;
pub mod database;
pub mod error;

pub use database::Database;
pub use error::Error;
