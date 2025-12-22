//! Error types for the vector crate.

use thiserror::Error;

/// Errors that can occur in vector operations.
#[derive(Debug, Error)]
pub enum VectorError {
    /// Dimension mismatch between vectors or embedding spaces.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension.
        actual: usize,
    },

    /// Invalid dimension (e.g., zero).
    #[error("invalid dimension: expected at least {expected}, got {actual}")]
    InvalidDimension {
        /// The minimum expected dimension.
        expected: usize,
        /// The actual dimension.
        actual: usize,
    },

    /// Invalid value in a vector (NaN, Infinity).
    #[error("invalid value at index {index}: {value} - {reason}")]
    InvalidValue {
        /// The index of the invalid value.
        index: usize,
        /// The invalid value.
        value: f32,
        /// The reason the value is invalid.
        reason: &'static str,
    },

    /// Invalid embedding name.
    #[error("invalid embedding name: {0}")]
    InvalidName(String),

    /// Embedding space not found.
    #[error("embedding space not found: {0}")]
    SpaceNotFound(String),

    /// Embedding not found for entity.
    #[error("embedding not found for entity {entity_id} in space '{space}'")]
    EmbeddingNotFound {
        /// The entity ID.
        entity_id: u64,
        /// The embedding space name.
        space: String,
    },

    /// Encoding/decoding error.
    #[error("encoding error: {0}")]
    Encoding(String),

    /// Storage backend error.
    #[error("storage error: {0}")]
    Storage(#[from] manifoldb_storage::StorageError),

    /// Lock poisoned - indicates concurrent panic corrupted the data structure.
    ///
    /// This error is unrecoverable - the index must be dropped and recreated.
    #[error("index corrupted: lock poisoned due to prior panic in another thread")]
    LockPoisoned,
}
