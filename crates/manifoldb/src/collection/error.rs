//! Collection API errors.
//!
//! This module provides error types specific to the programmatic collection API.

use manifoldb_core::PointId;
use thiserror::Error;

use super::manager::CollectionError;

/// Errors that can occur during collection API operations.
#[derive(Debug, Error)]
pub enum ApiError {
    /// Collection operation failed.
    #[error(transparent)]
    Collection(#[from] CollectionError),

    /// Point not found in collection.
    #[error("point {point_id} not found in collection '{collection}'")]
    PointNotFound {
        /// The ID of the point that was not found.
        point_id: PointId,
        /// The name of the collection.
        collection: String,
    },

    /// Point already exists in collection.
    #[error("point {point_id} already exists in collection '{collection}'")]
    PointAlreadyExists {
        /// The ID of the point that already exists.
        point_id: PointId,
        /// The name of the collection.
        collection: String,
    },

    /// Vector dimension mismatch.
    #[error("vector '{vector_name}' dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The name of the vector.
        vector_name: String,
        /// The expected dimension.
        expected: usize,
        /// The actual dimension.
        actual: usize,
    },

    /// Vector type mismatch.
    #[error("vector '{vector_name}' type mismatch: expected {expected}, got {actual}")]
    VectorTypeMismatch {
        /// The name of the vector.
        vector_name: String,
        /// The expected type.
        expected: String,
        /// The actual type.
        actual: String,
    },

    /// Named vector not found in collection schema.
    #[error("vector '{vector_name}' not found in collection '{collection}'")]
    VectorNotInSchema {
        /// The name of the vector.
        vector_name: String,
        /// The name of the collection.
        collection: String,
    },

    /// Search query vector is empty.
    #[error("search query vector cannot be empty")]
    EmptyQueryVector,

    /// Invalid filter expression.
    #[error("invalid filter: {0}")]
    InvalidFilter(String),

    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),

    /// Serialization/deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Collection handle is invalid (collection was deleted).
    #[error("collection '{0}' no longer exists")]
    InvalidHandle(String),

    /// Search limit must be greater than zero.
    #[error("search limit must be greater than zero")]
    InvalidSearchLimit,

    /// Hybrid search requires multiple vectors.
    #[error("hybrid search requires at least two vector names")]
    InsufficientVectorsForHybrid,
}

impl From<manifoldb_vector::error::VectorError> for ApiError {
    fn from(err: manifoldb_vector::error::VectorError) -> Self {
        Self::Storage(err.to_string())
    }
}

/// A specialized `Result` type for collection API operations.
pub type ApiResult<T> = std::result::Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ApiError::PointNotFound {
            point_id: PointId::new(42),
            collection: "documents".to_string(),
        };
        assert!(err.to_string().contains("42"));
        assert!(err.to_string().contains("documents"));

        let err = ApiError::DimensionMismatch {
            vector_name: "text".to_string(),
            expected: 768,
            actual: 384,
        };
        assert!(err.to_string().contains("768"));
        assert!(err.to_string().contains("384"));
    }
}
