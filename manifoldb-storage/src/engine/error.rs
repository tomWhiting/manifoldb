//! Storage error types.
//!
//! This module defines the error types for storage operations. All errors
//! implement [`std::error::Error`] and provide descriptive messages.

use std::fmt;

use thiserror::Error;

/// Errors that can occur in storage operations.
///
/// This enum covers all possible failure modes for storage backends,
/// from database-level issues to transaction and I/O errors.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The database could not be opened or created.
    #[error("failed to open database: {0}")]
    Open(String),

    /// The database file or directory does not exist.
    #[error("database not found: {0}")]
    NotFound(String),

    /// A table does not exist.
    #[error("table not found: {0}")]
    TableNotFound(String),

    /// A key was not found in the table.
    #[error("key not found")]
    KeyNotFound,

    /// A transaction error occurred (failed to begin, commit, or rollback).
    #[error("transaction error: {0}")]
    Transaction(String),

    /// Attempted a write operation on a read-only transaction.
    #[error("cannot write in read-only transaction")]
    ReadOnly,

    /// A conflict occurred due to concurrent modification.
    #[error("write conflict: {0}")]
    Conflict(String),

    /// The database is corrupted.
    #[error("database corruption detected: {0}")]
    Corruption(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The storage is full or a size limit was exceeded.
    #[error("storage full: {0}")]
    StorageFull(String),

    /// An invalid argument was provided.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The operation is not supported by this backend.
    #[error("operation not supported: {0}")]
    Unsupported(String),

    /// An internal error occurred in the storage backend.
    #[error("internal error: {0}")]
    Internal(String),
}

impl StorageError {
    /// Returns `true` if this error is recoverable.
    ///
    /// Recoverable errors include transient conditions like conflicts
    /// that may succeed on retry. Non-recoverable errors include
    /// corruption and configuration issues.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Conflict(_) | Self::Transaction(_) | Self::Io(_)
        )
    }

    /// Returns `true` if this is a "not found" type error.
    #[must_use]
    pub const fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::NotFound(_) | Self::TableNotFound(_) | Self::KeyNotFound
        )
    }
}

/// Result type alias for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;

/// Additional context for storage errors.
#[derive(Debug)]
pub struct ErrorContext {
    /// The table involved in the error, if any.
    pub table: Option<String>,
    /// The key involved in the error, if any (as hex string).
    pub key: Option<String>,
    /// Additional context message.
    pub message: Option<String>,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(ref table) = self.table {
            parts.push(format!("table={table}"));
        }
        if let Some(ref key) = self.key {
            parts.push(format!("key={key}"));
        }
        if let Some(ref msg) = self.message {
            parts.push(msg.clone());
        }
        write!(f, "{}", parts.join(", "))
    }
}
