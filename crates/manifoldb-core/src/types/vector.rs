//! Vector data types for entity embeddings.
//!
//! This module provides the [`VectorData`] enum for storing different types
//! of vector embeddings on entities.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::VectorData;
//!
//! // Dense vector (most common)
//! let dense = VectorData::Dense(vec![0.1, 0.2, 0.3, 0.4]);
//!
//! // Sparse vector for keyword matching
//! let sparse = VectorData::Sparse(vec![(10, 0.5), (42, 0.8), (100, 0.3)]);
//!
//! // Multi-vector for ColBERT-style token embeddings
//! let multi = VectorData::Multi(vec![
//!     vec![0.1, 0.2],
//!     vec![0.3, 0.4],
//! ]);
//! ```

use serde::{Deserialize, Serialize};

/// Vector data supporting different embedding types.
///
/// ManifoldDB supports three types of vector embeddings:
///
/// - **Dense**: Standard floating-point vectors (e.g., BERT, BGE, Jina)
/// - **Sparse**: Index-value pairs for keyword/BM25-style matching (e.g., SPLADE)
/// - **Multi**: Multiple vectors per entity for late interaction (e.g., ColBERT)
///
/// # Example
///
/// ```
/// use manifoldb_core::VectorData;
///
/// let embedding = VectorData::Dense(vec![0.1; 768]);
/// assert_eq!(embedding.dimension(), Some(768));
///
/// let sparse = VectorData::Sparse(vec![(5, 0.9), (100, 0.5)]);
/// assert_eq!(sparse.dimension(), None); // Sparse vectors have no fixed dimension
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorData {
    /// Dense floating-point vector.
    ///
    /// Used for standard embedding models like BERT, BGE, GTE, Jina, etc.
    /// The dimension is fixed and determined by the model.
    Dense(Vec<f32>),

    /// Sparse vector with (index, value) pairs.
    ///
    /// Used for sparse embedding models like SPLADE, or traditional
    /// keyword/BM25-style representations. Only non-zero dimensions are stored.
    Sparse(Vec<(u32, f32)>),

    /// Multi-vector (array of dense vectors).
    ///
    /// Used for late-interaction models like ColBERT where each token
    /// gets its own embedding vector.
    Multi(Vec<Vec<f32>>),
}

impl VectorData {
    /// Returns the dimension of the vector, if applicable.
    ///
    /// - For `Dense`: returns the length of the vector
    /// - For `Sparse`: returns `None` (sparse vectors have no fixed dimension)
    /// - For `Multi`: returns the dimension of the first sub-vector, or `None` if empty
    #[must_use]
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::Dense(v) => Some(v.len()),
            Self::Sparse(_) => None,
            Self::Multi(vecs) => vecs.first().map(Vec::len),
        }
    }

    /// Returns `true` if this is a dense vector.
    #[inline]
    #[must_use]
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Returns `true` if this is a sparse vector.
    #[inline]
    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Returns `true` if this is a multi-vector.
    #[inline]
    #[must_use]
    pub const fn is_multi(&self) -> bool {
        matches!(self, Self::Multi(_))
    }

    /// Returns the dense vector data, if this is a dense vector.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the sparse vector data, if this is a sparse vector.
    #[must_use]
    pub fn as_sparse(&self) -> Option<&[(u32, f32)]> {
        match self {
            Self::Sparse(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the multi-vector data, if this is a multi-vector.
    #[must_use]
    pub fn as_multi(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::Multi(v) => Some(v),
            _ => None,
        }
    }
}

impl From<Vec<f32>> for VectorData {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        Self::Dense(v)
    }
}

impl From<Vec<(u32, f32)>> for VectorData {
    #[inline]
    fn from(v: Vec<(u32, f32)>) -> Self {
        Self::Sparse(v)
    }
}

impl From<Vec<Vec<f32>>> for VectorData {
    #[inline]
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::Multi(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_vector() {
        let v = VectorData::Dense(vec![0.1, 0.2, 0.3]);
        assert!(v.is_dense());
        assert!(!v.is_sparse());
        assert!(!v.is_multi());
        assert_eq!(v.dimension(), Some(3));
        assert_eq!(v.as_dense(), Some(&[0.1, 0.2, 0.3][..]));
    }

    #[test]
    fn sparse_vector() {
        let v = VectorData::Sparse(vec![(10, 0.5), (42, 0.8)]);
        assert!(v.is_sparse());
        assert_eq!(v.dimension(), None);
        assert_eq!(v.as_sparse(), Some(&[(10, 0.5), (42, 0.8)][..]));
    }

    #[test]
    fn multi_vector() {
        let v = VectorData::Multi(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert!(v.is_multi());
        assert_eq!(v.dimension(), Some(2));
    }

    #[test]
    fn from_impls() {
        let dense: VectorData = vec![0.1, 0.2].into();
        assert!(dense.is_dense());

        let sparse: VectorData = vec![(1, 0.5)].into();
        assert!(sparse.is_sparse());

        let multi: VectorData = vec![vec![0.1]].into();
        assert!(multi.is_multi());
    }
}
