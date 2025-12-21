//! Error types for the main database crate.

use thiserror::Error;

/// Errors that can occur when using `ManifoldDB`.
#[derive(Debug, Error)]
pub enum Error {
    /// A storage error occurred.
    #[error("storage error: {0}")]
    Storage(#[from] manifoldb_storage::engine::StorageError),

    /// A query parsing error occurred.
    #[error("parse error: {0}")]
    Parse(String),

    /// A query execution error occurred.
    #[error("execution error: {0}")]
    Execution(String),

    /// The database could not be opened.
    #[error("failed to open database: {0}")]
    Open(String),
}
