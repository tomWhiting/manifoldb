//! `ManifoldDB` Storage
//!
//! This crate provides the storage engine abstraction and backend implementations
//! for `ManifoldDB`.
//!
//! # Overview
//!
//! The storage layer provides a transactional key-value interface that backends
//! implement. This allows `ManifoldDB` to support multiple storage engines while
//! providing consistent ACID semantics.
//!
//! # Core Traits
//!
//! - [`StorageEngine`] - The main entry point for storage operations
//! - [`Transaction`] - ACID transaction support with read/write operations
//! - [`Cursor`] - Ordered iteration over key-value pairs
//!
//! # Error Handling
//!
//! All storage operations return [`StorageResult<T>`], which is an alias for
//! `Result<T, StorageError>`. The [`StorageError`] enum covers all possible
//! failure modes from database-level issues to I/O errors.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_storage::{StorageEngine, Transaction};
//! use manifoldb_storage::backends::RedbEngine;
//!
//! // Open or create a database
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
//! # Modules
//!
//! - [`engine`] - Storage engine traits and abstractions
//! - [`backends`] - Concrete storage backend implementations
//! - [`wal`] - Write-ahead logging for durability and recovery

pub mod backends;
pub mod engine;
pub mod wal;

pub use engine::{
    Cursor, CursorResult, ErrorContext, KeyValue, StorageEngine, StorageError, StorageResult,
    Transaction,
};

pub use wal::{
    Lsn, Operation, RecoveryMode, RecoveryStats, TxnId, WalConfig, WalEntry, WalError, WalRecovery,
    WalResult, WalWriter,
};
