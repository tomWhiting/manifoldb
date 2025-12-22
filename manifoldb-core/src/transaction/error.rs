//! Transaction error types.

use std::fmt;

use thiserror::Error;

/// Errors that can occur during transaction operations.
#[derive(Debug, Error)]
pub enum TransactionError {
    /// The storage layer returned an error.
    #[error("storage error: {0}")]
    Storage(String),

    /// Attempted a write operation on a read-only transaction.
    #[error("cannot write in read-only transaction")]
    ReadOnly,

    /// The transaction has already been committed or rolled back.
    #[error("transaction already completed")]
    AlreadyCompleted,

    /// A conflict occurred due to concurrent modification.
    #[error("transaction conflict: {0}")]
    Conflict(String),

    /// An entity was not found.
    #[error("entity not found: {0}")]
    EntityNotFound(String),

    /// An edge was not found.
    #[error("edge not found: {0}")]
    EdgeNotFound(String),

    /// A serialization or deserialization error occurred.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A constraint violation occurred (e.g., duplicate key).
    #[error("constraint violation: {0}")]
    ConstraintViolation(String),

    /// A referential integrity violation occurred (e.g., deleting entity with connected edges).
    #[error("referential integrity violation: {0}")]
    ReferentialIntegrity(String),

    /// The transaction manager is not properly initialized.
    #[error("transaction manager not initialized: {0}")]
    NotInitialized(String),

    /// An internal error occurred.
    #[error("internal error: {0}")]
    Internal(String),
}

impl TransactionError {
    /// Returns `true` if this error is recoverable (e.g., may succeed on retry).
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(self, Self::Conflict(_) | Self::Storage(_))
    }

    /// Returns `true` if this is a "not found" type error.
    #[must_use]
    pub const fn is_not_found(&self) -> bool {
        matches!(self, Self::EntityNotFound(_) | Self::EdgeNotFound(_))
    }
}

/// Result type alias for transaction operations.
pub type TransactionResult<T> = Result<T, TransactionError>;

/// Additional context for transaction errors.
#[derive(Debug, Default)]
pub struct TransactionErrorContext {
    /// The operation that failed.
    pub operation: Option<String>,
    /// The entity ID involved, if any.
    pub entity_id: Option<u64>,
    /// The edge ID involved, if any.
    pub edge_id: Option<u64>,
    /// Additional context message.
    pub message: Option<String>,
}

impl fmt::Display for TransactionErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(ref op) = self.operation {
            parts.push(format!("operation={op}"));
        }
        if let Some(id) = self.entity_id {
            parts.push(format!("entity_id={id}"));
        }
        if let Some(id) = self.edge_id {
            parts.push(format!("edge_id={id}"));
        }
        if let Some(ref msg) = self.message {
            parts.push(msg.clone());
        }
        write!(f, "{}", parts.join(", "))
    }
}
