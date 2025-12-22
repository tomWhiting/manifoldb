//! Error types for the core crate.

use thiserror::Error;

/// Maximum length for value display in error messages.
const MAX_VALUE_DISPLAY_LEN: usize = 100;

/// Errors that can occur in the core crate.
#[derive(Debug, Error)]
pub enum CoreError {
    /// An encoding or decoding error occurred.
    #[error("encoding error: {0}")]
    Encoding(String),

    /// A value type mismatch occurred.
    #[error("type mismatch: expected {expected}, got {actual}{}", value.as_ref().map(|v| format!(" (value: {})", v)).unwrap_or_default())]
    TypeMismatch {
        /// The expected type.
        expected: String,
        /// The actual type.
        actual: String,
        /// The value that caused the mismatch (truncated for display).
        value: Option<String>,
    },

    /// A validation error occurred.
    #[error("validation error: {0}")]
    Validation(String),
}

impl CoreError {
    /// Creates a type mismatch error without a value.
    #[must_use]
    pub fn type_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::TypeMismatch { expected: expected.into(), actual: actual.into(), value: None }
    }

    /// Creates a type mismatch error with a value for debugging.
    ///
    /// The value is truncated to 100 characters for display.
    #[must_use]
    pub fn type_mismatch_with_value(
        expected: impl Into<String>,
        actual: impl Into<String>,
        value: impl std::fmt::Display,
    ) -> Self {
        let value_str = value.to_string();
        let truncated = if value_str.len() > MAX_VALUE_DISPLAY_LEN {
            format!("{}...", &value_str[..MAX_VALUE_DISPLAY_LEN])
        } else {
            value_str
        };
        Self::TypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
            value: Some(truncated),
        }
    }
}
