//! Storage error types.

use thiserror::Error;

/// Errors that can occur in storage operations.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The database could not be opened.
    #[error("failed to open database: {0}")]
    Open(String),

    /// A table does not exist.
    #[error("table not found: {0}")]
    TableNotFound(String),

    /// A transaction error occurred.
    #[error("transaction error: {0}")]
    Transaction(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A serialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The operation is not supported.
    #[error("operation not supported: {0}")]
    Unsupported(String),
}
