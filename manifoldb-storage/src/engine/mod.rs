//! Storage engine traits and abstractions.
//!
//! This module defines the core traits that storage backends must implement:
//!
//! - [`StorageEngine`] - Main entry point for creating transactions
//! - [`Transaction`] - ACID transaction with get/put/delete/range operations
//! - [`Cursor`] - Ordered iteration over key-value pairs
//!
//! # Error Handling
//!
//! All operations return [`StorageResult<T>`] which is an alias for
//! `Result<T, StorageError>`. See [`StorageError`] for the possible error variants.

mod error;
mod traits;

pub use error::{ErrorContext, StorageError, StorageResult};
pub use traits::{Cursor, CursorResult, KeyValue, StorageEngine, Transaction};
