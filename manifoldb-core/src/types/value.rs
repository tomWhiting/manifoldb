//! Property values that can be stored on entities and edges.
//!
//! This module provides the [`Value`] enum, which represents all possible value
//! types that can be stored as properties in `ManifoldDB`.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::Value;
//!
//! // Create values via From trait
//! let name: Value = "Alice".into();
//! let age: Value = 30i64.into();
//! let score: Value = 95.5f64.into();
//! let active: Value = true.into();
//!
//! // Access typed values
//! assert_eq!(name.as_str(), Some("Alice"));
//! assert_eq!(age.as_int(), Some(30));
//! assert_eq!(score.as_float(), Some(95.5));
//! assert_eq!(active.as_bool(), Some(true));
//!
//! // Vector embeddings for similarity search
//! let embedding: Value = vec![0.1f32, 0.2, 0.3].into();
//! assert_eq!(embedding.as_vector().map(|v| v.len()), Some(3));
//! ```

use serde::{Deserialize, Serialize};

/// A value that can be stored as a property on an entity or edge.
///
/// This enum represents all possible value types in `ManifoldDB`, including
/// vector embeddings for similarity search.
///
/// # Supported Types
///
/// | Variant | Rust Type | Use Case |
/// |---------|-----------|----------|
/// | `Null` | - | Missing/optional values |
/// | `Bool` | `bool` | Boolean flags |
/// | `Int` | `i64` | Integers, counters, timestamps |
/// | `Float` | `f64` | Numeric measurements |
/// | `String` | `String` | Text data |
/// | `Bytes` | `Vec<u8>` | Binary data |
/// | `Vector` | `Vec<f32>` | Dense embeddings (BERT, OpenAI, etc.) |
/// | `SparseVector` | `Vec<(u32, f32)>` | Sparse embeddings (SPLADE, BM25) |
/// | `MultiVector` | `Vec<Vec<f32>>` | Per-token embeddings (ColBERT) |
/// | `Array` | `Vec<Value>` | Lists of values |
///
/// # Example
///
/// ```
/// use manifoldb_core::Value;
///
/// // Standard types
/// let v1 = Value::from("hello");
/// let v2 = Value::from(42i64);
/// let v3 = Value::from(3.14f64);
///
/// // Dense embedding (e.g., from BERT, OpenAI)
/// let embedding = Value::from(vec![0.1f32, 0.2, 0.3, 0.4]);
///
/// // Sparse embedding (only non-zero indices stored)
/// let sparse = Value::from(vec![(10u32, 0.5f32), (50, 0.3), (100, 0.2)]);
///
/// // Multi-vector (per-token embeddings for ColBERT)
/// let multi = Value::from(vec![
///     vec![0.1f32, 0.2],  // Token 1
///     vec![0.3, 0.4],     // Token 2
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Null/missing value
    Null,
    /// Boolean value
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point number
    Float(f64),
    /// UTF-8 string
    String(String),
    /// Raw bytes
    Bytes(Vec<u8>),
    /// Dense vector embedding (for similarity search)
    Vector(Vec<f32>),
    /// Sparse vector embedding (for SPLADE, sparse retrievers, etc.)
    ///
    /// Stored as a list of (index, value) pairs, sorted by index.
    /// Only non-zero values are stored for efficiency.
    SparseVector(Vec<(u32, f32)>),
    /// Multi-vector embedding (for ColBERT-style late interaction models)
    ///
    /// Stores per-token embeddings as a list of dense vectors.
    /// Each inner vector represents a token's embedding.
    /// Used with MaxSim scoring: `max(dot(q_i, d_j))` for all query/doc token pairs.
    MultiVector(Vec<Vec<f32>>),
    /// Array of values
    Array(Vec<Value>),
}

impl Value {
    /// Returns `true` if the value is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Returns the value as a boolean if it is one.
    #[inline]
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the value as an integer if it is one.
    #[inline]
    #[must_use]
    pub const fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the value as a float if it is one.
    #[inline]
    #[must_use]
    pub const fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the value as a string slice if it is one.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the value as a dense vector slice if it is one.
    #[inline]
    #[must_use]
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as a sparse vector slice if it is one.
    ///
    /// The sparse vector is represented as `(index, value)` pairs.
    #[inline]
    #[must_use]
    pub fn as_sparse_vector(&self) -> Option<&[(u32, f32)]> {
        match self {
            Self::SparseVector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the value as a multi-vector slice if it is one.
    ///
    /// Multi-vectors are used for ColBERT-style late interaction models,
    /// where each token has its own embedding vector.
    #[inline]
    #[must_use]
    pub fn as_multi_vector(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::MultiVector(v) => Some(v),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    #[inline]
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<i64> for Value {
    #[inline]
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}

impl From<f64> for Value {
    #[inline]
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}

impl From<String> for Value {
    #[inline]
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<&str> for Value {
    #[inline]
    fn from(s: &str) -> Self {
        Self::String(s.to_owned())
    }
}

impl From<Vec<f32>> for Value {
    #[inline]
    fn from(v: Vec<f32>) -> Self {
        Self::Vector(v)
    }
}

impl From<Vec<(u32, f32)>> for Value {
    #[inline]
    fn from(v: Vec<(u32, f32)>) -> Self {
        Self::SparseVector(v)
    }
}

impl From<Vec<Vec<f32>>> for Value {
    #[inline]
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::MultiVector(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_type_checks() {
        assert!(Value::Null.is_null());
        assert!(!Value::Bool(true).is_null());
    }

    #[test]
    fn value_conversions() {
        assert_eq!(Value::from(true).as_bool(), Some(true));
        assert_eq!(Value::from(42i64).as_int(), Some(42));
        assert_eq!(Value::from(2.5f64).as_float(), Some(2.5));
        assert_eq!(Value::from("hello").as_str(), Some("hello"));
    }

    #[test]
    fn vector_value() {
        let embedding = vec![0.1, 0.2, 0.3];
        let value = Value::from(embedding.clone());
        assert_eq!(value.as_vector(), Some(embedding.as_slice()));
    }

    #[test]
    fn sparse_vector_value() {
        let sparse = vec![(0, 0.5), (10, 0.3), (100, 0.2)];
        let value = Value::from(sparse.clone());
        assert_eq!(value.as_sparse_vector(), Some(sparse.as_slice()));
    }

    #[test]
    fn sparse_vector_vs_dense_vector() {
        let dense = Value::from(vec![0.1, 0.2, 0.3]);
        let sparse = Value::from(vec![(0u32, 0.5f32)]);

        // Dense vector should not be accessible as sparse
        assert!(dense.as_sparse_vector().is_none());
        // Sparse vector should not be accessible as dense
        assert!(sparse.as_vector().is_none());
    }

    #[test]
    fn multi_vector_value() {
        let multi = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let value = Value::from(multi.clone());
        assert_eq!(value.as_multi_vector(), Some(multi.as_slice()));
    }

    #[test]
    fn multi_vector_vs_other_vectors() {
        let dense = Value::from(vec![0.1, 0.2, 0.3]);
        let sparse = Value::from(vec![(0u32, 0.5f32)]);
        let multi = Value::from(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);

        // Dense and sparse should not be accessible as multi-vector
        assert!(dense.as_multi_vector().is_none());
        assert!(sparse.as_multi_vector().is_none());

        // Multi-vector should not be accessible as dense or sparse
        assert!(multi.as_vector().is_none());
        assert!(multi.as_sparse_vector().is_none());
    }
}
