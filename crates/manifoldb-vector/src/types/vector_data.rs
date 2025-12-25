//! Core types for separated vector storage.
//!
//! This module provides types for storing vectors separately from entities,
//! enabling efficient storage and retrieval of vector embeddings.

use serde::{Deserialize, Serialize};

use crate::error::VectorError;
use crate::types::Embedding;

/// Vector data supporting different types.
///
/// This enum represents the actual vector data that can be stored for an entity.
/// It supports dense, sparse, multi-vector, and binary formats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorData {
    /// Dense floating-point vector.
    Dense(Vec<f32>),
    /// Sparse vector with (index, value) pairs.
    Sparse(Vec<(u32, f32)>),
    /// Multi-vector (ColBERT-style token embeddings).
    Multi(Vec<Vec<f32>>),
    /// Binary vector (bit-packed).
    Binary(Vec<u8>),
}

impl VectorData {
    /// Get the dimension of the vector.
    ///
    /// For dense vectors, this is the length of the vector.
    /// For sparse vectors, this is the maximum index + 1.
    /// For multi-vectors, this is the dimension of each inner vector.
    /// For binary vectors, this is the number of bits.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Sparse(v) => v.iter().map(|(i, _)| *i as usize + 1).max().unwrap_or(0),
            Self::Multi(v) => v.first().map(|inner| inner.len()).unwrap_or(0),
            Self::Binary(v) => v.len() * 8,
        }
    }

    /// Check if this is a dense vector.
    #[must_use]
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Check if this is a sparse vector.
    #[must_use]
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Check if this is a multi-vector.
    #[must_use]
    pub fn is_multi(&self) -> bool {
        matches!(self, Self::Multi(_))
    }

    /// Check if this is a binary vector.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary(_))
    }

    /// Get as dense vector slice.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Get as sparse vector slice.
    #[must_use]
    pub fn as_sparse(&self) -> Option<&[(u32, f32)]> {
        match self {
            Self::Sparse(v) => Some(v),
            _ => None,
        }
    }

    /// Get as multi-vector slice.
    #[must_use]
    pub fn as_multi(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::Multi(v) => Some(v),
            _ => None,
        }
    }

    /// Get as binary vector slice.
    #[must_use]
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Self::Binary(v) => Some(v),
            _ => None,
        }
    }

    /// Convert to Embedding for HNSW (dense vectors only).
    ///
    /// # Errors
    ///
    /// Returns an error if the vector is not dense or if the embedding is invalid.
    pub fn to_embedding(&self) -> Result<Embedding, VectorError> {
        match self {
            Self::Dense(v) => {
                Embedding::new(v.clone()).map_err(|e| VectorError::Encoding(e.to_string()))
            }
            _ => Err(VectorError::Encoding(
                "Cannot convert non-dense vector to Embedding".to_string(),
            )),
        }
    }

    /// Get the type discriminant for encoding.
    #[must_use]
    pub(crate) fn type_discriminant(&self) -> u8 {
        match self {
            Self::Dense(_) => 0,
            Self::Sparse(_) => 1,
            Self::Multi(_) => 2,
            Self::Binary(_) => 3,
        }
    }
}

impl From<Vec<f32>> for VectorData {
    fn from(v: Vec<f32>) -> Self {
        Self::Dense(v)
    }
}

impl From<&[f32]> for VectorData {
    fn from(v: &[f32]) -> Self {
        Self::Dense(v.to_vec())
    }
}

impl From<Vec<(u32, f32)>> for VectorData {
    fn from(v: Vec<(u32, f32)>) -> Self {
        Self::Sparse(v)
    }
}

impl From<Vec<Vec<f32>>> for VectorData {
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::Multi(v)
    }
}

impl From<Vec<u8>> for VectorData {
    fn from(v: Vec<u8>) -> Self {
        Self::Binary(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_vector() {
        let data = VectorData::Dense(vec![1.0, 2.0, 3.0]);
        assert!(data.is_dense());
        assert!(!data.is_sparse());
        assert_eq!(data.dimension(), 3);
        assert_eq!(data.as_dense(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn test_sparse_vector() {
        let data = VectorData::Sparse(vec![(0, 1.0), (5, 2.0), (10, 3.0)]);
        assert!(data.is_sparse());
        assert!(!data.is_dense());
        assert_eq!(data.dimension(), 11); // max index + 1
        assert_eq!(data.as_sparse(), Some([(0, 1.0), (5, 2.0), (10, 3.0)].as_slice()));
    }

    #[test]
    fn test_multi_vector() {
        let data = VectorData::Multi(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(data.is_multi());
        assert_eq!(data.dimension(), 2);
    }

    #[test]
    fn test_binary_vector() {
        let data = VectorData::Binary(vec![0xFF, 0x00]);
        assert!(data.is_binary());
        assert_eq!(data.dimension(), 16);
    }

    #[test]
    fn test_from_vec_f32() {
        let v = vec![1.0, 2.0, 3.0];
        let data: VectorData = v.into();
        assert!(data.is_dense());
    }

    #[test]
    fn test_to_embedding_dense() {
        let data = VectorData::Dense(vec![1.0, 2.0, 3.0]);
        let embedding = data.to_embedding().unwrap();
        assert_eq!(embedding.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_to_embedding_sparse_fails() {
        let data = VectorData::Sparse(vec![(0, 1.0)]);
        assert!(data.to_embedding().is_err());
    }
}
