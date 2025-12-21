//! Error types for `ManifoldDB`.
//!
//! This module provides the [`enum@Error`] type that represents all possible errors
//! when using `ManifoldDB`.

use thiserror::Error;

/// Errors that can occur when using `ManifoldDB`.
///
/// This enum covers all error conditions from configuration to query execution.
#[derive(Debug, Error)]
pub enum Error {
    /// A configuration error occurred.
    #[error("configuration error: {0}")]
    Config(String),

    /// A storage error occurred.
    #[error("storage error: {0}")]
    Storage(#[from] manifoldb_storage::engine::StorageError),

    /// A transaction error occurred.
    #[error("transaction error: {0}")]
    Transaction(#[from] manifoldb_core::TransactionError),

    /// A query parsing error occurred.
    #[error("parse error: {0}")]
    Parse(String),

    /// A query execution error occurred.
    #[error("execution error: {0}")]
    Execution(String),

    /// The database could not be opened.
    #[error("failed to open database: {0}")]
    Open(String),

    /// An invalid query parameter was provided.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A type conversion error occurred.
    #[error("type error: {0}")]
    Type(String),

    /// The database is closed.
    #[error("database is closed")]
    Closed,
}

impl Error {
    /// Returns `true` if this error is recoverable.
    ///
    /// Recoverable errors can typically be retried or handled gracefully.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Parse(_) | Self::InvalidParameter(_) | Self::Type(_) | Self::Execution(_)
        )
    }

    /// Returns `true` if this is a transaction error.
    #[must_use]
    pub const fn is_transaction_error(&self) -> bool {
        matches!(self, Self::Transaction(_))
    }

    /// Returns `true` if this is a storage error.
    #[must_use]
    pub const fn is_storage_error(&self) -> bool {
        matches!(self, Self::Storage(_))
    }

    /// Create a parse error.
    #[must_use]
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }

    /// Create an execution error.
    #[must_use]
    pub fn execution(msg: impl Into<String>) -> Self {
        Self::Execution(msg.into())
    }

    /// Create a config error.
    #[must_use]
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
}

impl From<manifoldb_query::ParseError> for Error {
    fn from(err: manifoldb_query::ParseError) -> Self {
        Self::Parse(err.to_string())
    }
}

/// A specialized `Result` type for `ManifoldDB` operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_is_recoverable() {
        assert!(Error::Parse("test".to_string()).is_recoverable());
        assert!(Error::InvalidParameter("test".to_string()).is_recoverable());
        assert!(Error::Type("test".to_string()).is_recoverable());
        assert!(Error::Execution("test".to_string()).is_recoverable());

        assert!(!Error::Open("test".to_string()).is_recoverable());
        assert!(!Error::Config("test".to_string()).is_recoverable());
        assert!(!Error::Closed.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = Error::parse("unexpected token");
        assert_eq!(err.to_string(), "parse error: unexpected token");

        let err = Error::execution("query timeout");
        assert_eq!(err.to_string(), "execution error: query timeout");
    }
}
