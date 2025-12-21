//! Error types for the core crate.

use thiserror::Error;

/// Errors that can occur in the core crate.
#[derive(Debug, Error)]
pub enum CoreError {
    /// An encoding or decoding error occurred.
    #[error("encoding error: {0}")]
    Encoding(String),

    /// A value type mismatch occurred.
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// The expected type.
        expected: String,
        /// The actual type.
        actual: String,
    },

    /// A validation error occurred.
    #[error("validation error: {0}")]
    Validation(String),
}
