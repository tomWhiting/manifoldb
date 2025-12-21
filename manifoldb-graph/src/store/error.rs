//! Error types for graph storage operations.

use manifoldb_core::{CoreError, EdgeId, EntityId};
use manifoldb_storage::StorageError;
use thiserror::Error;

/// Errors that can occur in graph storage operations.
#[derive(Debug, Error)]
pub enum GraphError {
    /// An entity (node) was not found.
    #[error("entity not found: {0}")]
    EntityNotFound(EntityId),

    /// An edge was not found.
    #[error("edge not found: {0}")]
    EdgeNotFound(EdgeId),

    /// Referenced entity does not exist when creating an edge.
    #[error("referenced entity does not exist: {0}")]
    InvalidEntityReference(EntityId),

    /// An entity with the given ID already exists.
    #[error("entity already exists: {0}")]
    EntityAlreadyExists(EntityId),

    /// An edge with the given ID already exists.
    #[error("edge already exists: {0}")]
    EdgeAlreadyExists(EdgeId),

    /// An encoding or decoding error occurred.
    #[error("encoding error: {0}")]
    Encoding(String),

    /// A storage backend error occurred.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// An internal error occurred.
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<CoreError> for GraphError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::Encoding(msg) => Self::Encoding(msg),
            CoreError::TypeMismatch { expected, actual } => {
                Self::Encoding(format!("type mismatch: expected {expected}, got {actual}"))
            }
            CoreError::Validation(msg) => Self::Internal(msg),
        }
    }
}

/// Result type for graph operations.
pub type GraphResult<T> = Result<T, GraphError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = GraphError::EntityNotFound(EntityId::new(42));
        assert!(err.to_string().contains("42"));

        let err = GraphError::EdgeNotFound(EdgeId::new(123));
        assert!(err.to_string().contains("123"));
    }

    #[test]
    fn from_core_error() {
        let core_err = CoreError::Encoding("test error".to_owned());
        let graph_err: GraphError = core_err.into();
        assert!(matches!(graph_err, GraphError::Encoding(_)));
    }
}
