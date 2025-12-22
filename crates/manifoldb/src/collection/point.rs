//! Point types for collection operations.
//!
//! This module provides types for representing points in a collection,
//! including [`PointStruct`] for point data and [`Vector`] for vector values.

use std::collections::HashMap;

use manifoldb_core::PointId;
use serde_json::Value as JsonValue;

/// A vector value that can be dense, sparse, or multi-vector.
///
/// This enum represents the different types of vectors that can be stored
/// in a collection. Each variant corresponds to a different vector type
/// in the collection schema.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::Vector;
///
/// // Dense vector for semantic embeddings
/// let dense = Vector::Dense(vec![0.1, 0.2, 0.3, 0.4]);
///
/// // Sparse vector for keyword features
/// let sparse = Vector::Sparse(vec![(100, 0.5), (200, 0.3)]);
///
/// // Multi-vector for ColBERT-style embeddings
/// let multi = Vector::Multi(vec![
///     vec![0.1, 0.2],
///     vec![0.3, 0.4],
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Vector {
    /// Dense vector (fixed-dimension f32 array).
    ///
    /// Used for traditional embeddings like BERT, OpenAI, etc.
    Dense(Vec<f32>),

    /// Sparse vector (sorted index-value pairs).
    ///
    /// Used for sparse representations like SPLADE, BM25, etc.
    /// Indices must be sorted in ascending order.
    Sparse(Vec<(u32, f32)>),

    /// Multi-vector (array of f32 arrays).
    ///
    /// Used for ColBERT-style per-token embeddings.
    Multi(Vec<Vec<f32>>),
}

impl Vector {
    /// Get the dimension of this vector.
    ///
    /// For sparse vectors, returns the number of non-zero elements.
    /// For multi-vectors, returns the dimension of the inner vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Sparse(v) => v.len(),
            Self::Multi(v) => v.first().map_or(0, Vec::len),
        }
    }

    /// Check if this is a dense vector.
    #[must_use]
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Check if this is a sparse vector.
    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Check if this is a multi-vector.
    #[must_use]
    pub const fn is_multi(&self) -> bool {
        matches!(self, Self::Multi(_))
    }

    /// Get as a dense vector.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Get as a sparse vector.
    #[must_use]
    pub fn as_sparse(&self) -> Option<&[(u32, f32)]> {
        match self {
            Self::Sparse(v) => Some(v),
            _ => None,
        }
    }

    /// Get as a multi-vector.
    #[must_use]
    pub fn as_multi(&self) -> Option<&[Vec<f32>]> {
        match self {
            Self::Multi(v) => Some(v),
            _ => None,
        }
    }
}

impl From<Vec<f32>> for Vector {
    fn from(v: Vec<f32>) -> Self {
        Self::Dense(v)
    }
}

impl From<Vec<(u32, f32)>> for Vector {
    fn from(v: Vec<(u32, f32)>) -> Self {
        Self::Sparse(v)
    }
}

impl From<Vec<Vec<f32>>> for Vector {
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::Multi(v)
    }
}

/// A point in a collection with its ID, payload, and vectors.
///
/// `PointStruct` represents a single point to be inserted or updated
/// in a collection. It combines an ID, optional JSON payload, and
/// named vectors.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::{PointStruct, Vector};
/// use serde_json::json;
/// use std::collections::HashMap;
///
/// let point = PointStruct {
///     id: 1.into(),
///     payload: Some(json!({
///         "title": "Rust Book",
///         "category": "programming"
///     })),
///     vectors: HashMap::from([
///         ("text".to_string(), Vector::Dense(vec![0.1; 768])),
///     ]),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct PointStruct {
    /// The unique identifier for this point.
    pub id: PointId,

    /// Optional JSON payload associated with the point.
    ///
    /// This can contain arbitrary metadata like titles, categories,
    /// timestamps, etc.
    pub payload: Option<JsonValue>,

    /// Named vectors for this point.
    ///
    /// The keys must match the vector names defined in the collection schema.
    pub vectors: HashMap<String, Vector>,
}

impl PointStruct {
    /// Create a new point with the given ID.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::collection::PointStruct;
    ///
    /// let point = PointStruct::new(1);
    /// ```
    #[must_use]
    pub fn new(id: impl Into<PointId>) -> Self {
        Self { id: id.into(), payload: None, vectors: HashMap::new() }
    }

    /// Set the payload for this point.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::collection::PointStruct;
    /// use serde_json::json;
    ///
    /// let point = PointStruct::new(1)
    ///     .with_payload(json!({"title": "Hello World"}));
    /// ```
    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<JsonValue>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Add a vector to this point.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::collection::{PointStruct, Vector};
    ///
    /// let point = PointStruct::new(1)
    ///     .with_vector("text", vec![0.1; 768])
    ///     .with_vector("sparse", Vector::Sparse(vec![(100, 0.5)]));
    /// ```
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, vector: impl Into<Vector>) -> Self {
        self.vectors.insert(name.into(), vector.into());
        self
    }

    /// Add multiple vectors to this point.
    #[must_use]
    pub fn with_vectors(mut self, vectors: impl IntoIterator<Item = (String, Vector)>) -> Self {
        self.vectors.extend(vectors);
        self
    }
}

/// A search result containing the matched point and its score.
///
/// Search results are ordered by score in descending order
/// (higher score = better match).
#[derive(Debug, Clone)]
pub struct ScoredPoint {
    /// The ID of the matched point.
    pub id: PointId,

    /// The similarity score.
    ///
    /// The meaning of the score depends on the distance metric:
    /// - Cosine: 0.0 (perpendicular) to 1.0 (identical)
    /// - DotProduct: unbounded, higher is better
    /// - Euclidean: 0.0 (identical) to infinity (inverted for scoring)
    pub score: f32,

    /// The payload of the matched point, if requested.
    pub payload: Option<JsonValue>,

    /// The vectors of the matched point, if requested.
    pub vectors: Option<HashMap<String, Vector>>,
}

impl ScoredPoint {
    /// Create a new scored point.
    #[must_use]
    pub fn new(id: PointId, score: f32) -> Self {
        Self { id, score, payload: None, vectors: None }
    }

    /// Set the payload for this scored point.
    #[must_use]
    pub fn with_payload(mut self, payload: JsonValue) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Set the vectors for this scored point.
    #[must_use]
    pub fn with_vectors(mut self, vectors: HashMap<String, Vector>) -> Self {
        self.vectors = Some(vectors);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_vector_types() {
        let dense = Vector::Dense(vec![0.1, 0.2, 0.3]);
        assert!(dense.is_dense());
        assert!(!dense.is_sparse());
        assert!(!dense.is_multi());
        assert_eq!(dense.dimension(), 3);
        assert_eq!(dense.as_dense(), Some(&[0.1, 0.2, 0.3][..]));

        let sparse = Vector::Sparse(vec![(0, 0.5), (10, 0.3)]);
        assert!(sparse.is_sparse());
        assert_eq!(sparse.dimension(), 2);

        let multi = Vector::Multi(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert!(multi.is_multi());
        assert_eq!(multi.dimension(), 2);
    }

    #[test]
    fn test_vector_from() {
        let dense: Vector = vec![0.1, 0.2, 0.3].into();
        assert!(dense.is_dense());

        let sparse: Vector = vec![(0u32, 0.5f32), (10, 0.3)].into();
        assert!(sparse.is_sparse());

        let multi: Vector = vec![vec![0.1, 0.2], vec![0.3, 0.4]].into();
        assert!(multi.is_multi());
    }

    #[test]
    fn test_point_struct_builder() {
        let point = PointStruct::new(42u64)
            .with_payload(json!({"title": "Test"}))
            .with_vector("text", vec![0.1, 0.2, 0.3])
            .with_vector("sparse", Vector::Sparse(vec![(100, 0.5)]));

        assert_eq!(point.id.as_u64(), 42);
        assert!(point.payload.is_some());
        assert_eq!(point.vectors.len(), 2);
        assert!(point.vectors.contains_key("text"));
        assert!(point.vectors.contains_key("sparse"));
    }

    #[test]
    fn test_scored_point() {
        let scored =
            ScoredPoint::new(PointId::new(1), 0.95).with_payload(json!({"title": "Best match"}));

        assert_eq!(scored.id.as_u64(), 1);
        assert!((scored.score - 0.95).abs() < 0.001);
        assert!(scored.payload.is_some());
    }
}
