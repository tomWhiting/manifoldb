//! Collection builder for fluent collection creation.
//!
//! This module provides the [`CollectionBuilder`] for creating collections
//! with a fluent API.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::collection::DistanceMetric;
//!
//! let collection = db.create_collection("documents")
//!     .with_dense_vector("text", 768, DistanceMetric::Cosine)
//!     .with_sparse_vector("keywords")
//!     .with_multi_vector("colbert", 128)
//!     .build()?;
//! ```

use std::sync::Arc;

use manifoldb_storage::StorageEngine;
use manifoldb_vector::distance::sparse::SparseDistanceMetric;
use manifoldb_vector::distance::DistanceMetric;

use super::config::{
    AggregationMethod, HnswParams, IndexConfig, InvertedIndexParams, VectorConfig, VectorType,
};
use super::error::{ApiError, ApiResult};
use super::handle::CollectionHandle;
use super::metadata::CollectionName;
use crate::collection::config::DistanceType;

/// A builder for creating collections with a fluent API.
///
/// Use this to configure and create a new collection with one or more
/// named vectors.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::DistanceMetric;
///
/// // Create a collection with dense and sparse vectors
/// let collection = db.create_collection("documents")
///     .with_dense_vector("text", 768, DistanceMetric::Cosine)
///     .with_sparse_vector("keywords")
///     .build()?;
///
/// // Create a hybrid collection for semantic + keyword search
/// let collection = db.create_collection("articles")
///     .with_dense_vector("semantic", 384, DistanceMetric::DotProduct)
///     .with_sparse_vector_config("bm25", 30522, SparseDistanceMetric::BM25)
///     .build()?;
/// ```
pub struct CollectionBuilder<E: StorageEngine> {
    /// The storage engine.
    pub(crate) engine: Arc<E>,
    /// The collection name.
    pub(crate) name: CollectionName,
    /// Named vectors to create.
    pub(crate) vectors: Vec<(String, VectorConfig)>,
}

impl<E: StorageEngine> CollectionBuilder<E> {
    /// Create a new collection builder.
    ///
    /// This is typically called via `db.create_collection(name)`.
    #[allow(dead_code)]
    pub(crate) fn new(engine: Arc<E>, name: CollectionName) -> Self {
        Self { engine, name, vectors: Vec::new() }
    }

    /// Add a dense vector with default HNSW index.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector (e.g., "text", "image")
    /// * `dimension` - The vector dimension (e.g., 768 for BERT)
    /// * `distance` - The distance metric for similarity search
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::collection::DistanceMetric;
    ///
    /// let builder = db.create_collection("docs")
    ///     .with_dense_vector("embedding", 768, DistanceMetric::Cosine);
    /// ```
    #[must_use]
    pub fn with_dense_vector(
        mut self,
        name: impl Into<String>,
        dimension: usize,
        distance: DistanceMetric,
    ) -> Self {
        self.vectors.push((name.into(), VectorConfig::dense(dimension, distance)));
        self
    }

    /// Add a dense vector with custom HNSW parameters.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector
    /// * `dimension` - The vector dimension
    /// * `distance` - The distance metric
    /// * `hnsw_params` - Custom HNSW index parameters
    #[must_use]
    pub fn with_dense_vector_hnsw(
        mut self,
        name: impl Into<String>,
        dimension: usize,
        distance: DistanceMetric,
        hnsw_params: HnswParams,
    ) -> Self {
        let config = VectorConfig {
            vector_type: VectorType::Dense { dimension },
            distance: DistanceType::Dense(distance),
            index: IndexConfig::hnsw(hnsw_params),
        };
        self.vectors.push((name.into(), config));
        self
    }

    /// Add a sparse vector with default inverted index.
    ///
    /// Uses DotProduct distance by default.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector (e.g., "keywords", "bm25")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = db.create_collection("docs")
    ///     .with_sparse_vector("keywords");
    /// ```
    #[must_use]
    pub fn with_sparse_vector(self, name: impl Into<String>) -> Self {
        self.with_sparse_vector_config(name, 30522, SparseDistanceMetric::DotProduct)
    }

    /// Add a sparse vector with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector
    /// * `max_dimension` - Maximum vocabulary size
    /// * `distance` - The sparse distance metric
    #[must_use]
    pub fn with_sparse_vector_config(
        mut self,
        name: impl Into<String>,
        max_dimension: u32,
        distance: SparseDistanceMetric,
    ) -> Self {
        let config = VectorConfig {
            vector_type: VectorType::Sparse { max_dimension },
            distance: DistanceType::Sparse(distance),
            index: IndexConfig::inverted_default(),
        };
        self.vectors.push((name.into(), config));
        self
    }

    /// Add a sparse vector with custom inverted index parameters.
    #[must_use]
    pub fn with_sparse_vector_inverted(
        mut self,
        name: impl Into<String>,
        max_dimension: u32,
        distance: SparseDistanceMetric,
        inverted_params: InvertedIndexParams,
    ) -> Self {
        let config = VectorConfig {
            vector_type: VectorType::Sparse { max_dimension },
            distance: DistanceType::Sparse(distance),
            index: IndexConfig::inverted(inverted_params),
        };
        self.vectors.push((name.into(), config));
        self
    }

    /// Add a multi-vector for ColBERT-style embeddings.
    ///
    /// Uses DotProduct distance and MaxSim aggregation by default.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector (e.g., "colbert")
    /// * `token_dim` - The dimension of each token embedding (e.g., 128)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let builder = db.create_collection("docs")
    ///     .with_multi_vector("colbert", 128);
    /// ```
    #[must_use]
    pub fn with_multi_vector(mut self, name: impl Into<String>, token_dim: usize) -> Self {
        self.vectors.push((name.into(), VectorConfig::multi_vector(token_dim)));
        self
    }

    /// Add a multi-vector with custom aggregation method.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector
    /// * `token_dim` - The dimension of each token embedding
    /// * `aggregation` - The aggregation method for scoring
    #[must_use]
    pub fn with_multi_vector_aggregation(
        mut self,
        name: impl Into<String>,
        token_dim: usize,
        aggregation: AggregationMethod,
    ) -> Self {
        let config = VectorConfig {
            vector_type: VectorType::Multi { token_dim },
            distance: DistanceType::Dense(DistanceMetric::DotProduct),
            index: IndexConfig::hnsw_with_aggregation(aggregation),
        };
        self.vectors.push((name.into(), config));
        self
    }

    /// Add a binary vector for LSH or SimHash.
    ///
    /// Uses Hamming distance by default.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for this vector
    /// * `bits` - The number of bits in the binary vector
    #[must_use]
    pub fn with_binary_vector(mut self, name: impl Into<String>, bits: usize) -> Self {
        self.vectors.push((name.into(), VectorConfig::binary(bits)));
        self
    }

    /// Add a custom vector configuration.
    ///
    /// Use this for advanced configurations not covered by the convenience methods.
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, config: VectorConfig) -> Self {
        self.vectors.push((name.into(), config));
        self
    }

    /// Build and create the collection.
    ///
    /// # Returns
    ///
    /// A handle to the newly created collection that can be used for
    /// point operations and search.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A collection with this name already exists
    /// - No vectors were configured
    /// - Storage operations fail
    ///
    /// # Example
    ///
    /// ```ignore
    /// let collection = db.create_collection("documents")
    ///     .with_dense_vector("text", 768, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// // Now you can use the collection handle
    /// collection.upsert_point(point)?;
    /// ```
    pub fn build(self) -> ApiResult<CollectionHandle<E>> {
        if self.vectors.is_empty() {
            return Err(ApiError::InvalidFilter(
                "collection must have at least one vector".to_string(),
            ));
        }

        CollectionHandle::create(self.engine, self.name, self.vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a storage engine.
    // These tests verify the builder API and configuration.

    #[test]
    fn test_vector_config_dense() {
        let config = VectorConfig::dense(768, DistanceMetric::Cosine);
        assert!(config.vector_type.is_dense());
        assert_eq!(config.dimension(), Some(768));
    }

    #[test]
    fn test_vector_config_sparse() {
        let config = VectorConfig::sparse(30522);
        assert!(config.vector_type.is_sparse());
        assert_eq!(config.dimension(), None);
    }

    #[test]
    fn test_vector_config_multi() {
        let config = VectorConfig::multi_vector(128);
        assert!(config.vector_type.is_multi());
        assert_eq!(config.dimension(), Some(128));
    }

    #[test]
    fn test_hnsw_params() {
        let params = HnswParams::new(32).with_ef_construction(400).with_ef_search(100);

        assert_eq!(params.m, 32);
        assert_eq!(params.m_max0, 64);
        assert_eq!(params.ef_construction, 400);
        assert_eq!(params.ef_search, 100);
    }
}
