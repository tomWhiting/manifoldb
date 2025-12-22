//! Redb storage backend.
//!
//! This module provides a storage backend implementation using Redb,
//! a pure-Rust embedded database. Redb provides ACID transactions,
//! excellent performance, and works well on all platforms including macOS.
//!
//! # Features
//!
//! - **Pure Rust**: No C dependencies, works everywhere Rust does
//! - **ACID Transactions**: Full transactional support with isolation
//! - **Cross-platform**: Works on macOS, Linux, Windows, and more
//! - **Embedded**: No external database server required
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_storage::backends::RedbEngine;
//! use manifoldb_storage::{StorageEngine, Transaction};
//!
//! // Open a database (creates if it doesn't exist)
//! let engine = RedbEngine::open("my_database.redb")?;
//!
//! // Write some data
//! let mut tx = engine.begin_write()?;
//! tx.put("users", b"user:1", b"Alice")?;
//! tx.put("users", b"user:2", b"Bob")?;
//! tx.commit()?;
//!
//! // Read it back
//! let tx = engine.begin_read()?;
//! let alice = tx.get("users", b"user:1")?;
//! assert_eq!(alice, Some(b"Alice".to_vec()));
//! ```
//!
//! # In-Memory Databases
//!
//! For testing, you can create an in-memory database that doesn't persist:
//!
//! ```ignore
//! let engine = RedbEngine::in_memory()?;
//! ```
//!
//! # Configuration
//!
//! Use `RedbConfig` to customize the database behavior:
//!
//! ```ignore
//! use manifoldb_storage::backends::redb::{RedbEngine, RedbConfig};
//!
//! let config = RedbConfig::new()
//!     .cache_size(100 * 1024 * 1024); // 100 MB cache
//!
//! let engine = RedbEngine::open_with_config("my_database.redb", config)?;
//! ```

mod engine;
pub mod tables;
mod transaction;

pub use engine::{RedbConfig, RedbEngine};
pub use transaction::{RedbCursor, RedbTransaction};
