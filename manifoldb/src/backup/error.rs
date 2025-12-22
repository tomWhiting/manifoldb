//! Error types for backup and restore operations.

use std::io;

use thiserror::Error;

/// Errors that can occur during backup and restore operations.
#[derive(Debug, Error)]
pub enum BackupError {
    /// An I/O error occurred while reading or writing backup data.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// A serialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A deserialization error occurred.
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// The backup format is invalid or corrupted.
    #[error("invalid backup format: {0}")]
    InvalidFormat(String),

    /// The backup version is not supported.
    #[error("unsupported backup version: {0}")]
    UnsupportedVersion(u32),

    /// A database transaction error occurred.
    #[error("transaction error: {0}")]
    Transaction(#[from] manifoldb_core::TransactionError),

    /// A storage error occurred.
    #[error("storage error: {0}")]
    Storage(#[from] manifoldb_storage::engine::StorageError),

    /// Integrity verification failed.
    #[error("integrity check failed: {0}")]
    IntegrityError(String),

    /// The backup is empty or incomplete.
    #[error("incomplete backup: {0}")]
    Incomplete(String),

    /// A record type mismatch occurred during restore.
    #[error("record type mismatch: expected {expected}, got {actual}")]
    RecordTypeMismatch {
        /// The expected record type.
        expected: String,
        /// The actual record type.
        actual: String,
    },

    /// A duplicate record was found during restore.
    #[error("duplicate record: {0}")]
    DuplicateRecord(String),

    /// Referenced entity not found during restore.
    #[error("missing reference: {0}")]
    MissingReference(String),

    /// A malformed record was encountered during restore.
    #[error("malformed record at line {line}: {message}")]
    MalformedRecord {
        /// The line number where the error occurred.
        line: u64,
        /// A description of the malformation.
        message: String,
    },
}

impl BackupError {
    /// Create a serialization error from a serde_json error.
    pub fn serialization(err: serde_json::Error) -> Self {
        Self::Serialization(err.to_string())
    }

    /// Create a deserialization error from a serde_json error.
    pub fn deserialization(err: serde_json::Error) -> Self {
        Self::Deserialization(err.to_string())
    }

    /// Create an invalid format error.
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidFormat(msg.into())
    }

    /// Create an integrity error.
    pub fn integrity(msg: impl Into<String>) -> Self {
        Self::IntegrityError(msg.into())
    }

    /// Create an incomplete backup error.
    pub fn incomplete(msg: impl Into<String>) -> Self {
        Self::Incomplete(msg.into())
    }

    /// Create a malformed record error with line number context.
    pub fn malformed_record(line: u64, msg: impl Into<String>) -> Self {
        Self::MalformedRecord { line, message: msg.into() }
    }
}

/// A specialized `Result` type for backup operations.
pub type BackupResult<T> = Result<T, BackupError>;

impl From<crate::Error> for BackupError {
    fn from(err: crate::Error) -> Self {
        match err {
            crate::Error::Transaction(e) => Self::Transaction(e),
            crate::Error::Storage(e) => Self::Storage(e),
            other => Self::Serialization(other.to_string()),
        }
    }
}
