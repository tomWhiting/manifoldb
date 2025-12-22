//! Vector configuration types for collections.
//!
//! This module defines the configuration for named vectors within a collection,
//! including vector types, dimensions, distance metrics, and index settings.

use manifoldb_vector::distance::sparse::SparseDistanceMetric;
use manifoldb_vector::distance::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Configuration for a named vector within a collection.
///
/// Each named vector in a collection has its own type, dimension, distance metric,
/// and index configuration. This allows storing heterogeneous embeddings together.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::{VectorConfig, VectorType, IndexConfig};
/// use manifoldb_vector::distance::DistanceMetric;
///
/// // Dense vector with HNSW index
/// let dense = VectorConfig::dense(768, DistanceMetric::Cosine);
///
/// // Sparse vector with inverted index
/// let sparse = VectorConfig::sparse(30522);
///
/// // Multi-vector for ColBERT with MaxSim aggregation
/// let colbert = VectorConfig::multi_vector(128);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorConfig {
    /// The type of vector (dense, sparse, multi, binary).
    pub vector_type: VectorType,
    /// The distance metric for similarity search.
    pub distance: DistanceType,
    /// The index configuration.
    pub index: IndexConfig,
}

impl VectorConfig {
    /// Create a new dense vector configuration.
    ///
    /// Dense vectors have a fixed dimension and support all distance metrics.
    /// By default, uses an HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The fixed dimension of the vector
    /// * `distance` - The distance metric for similarity search
    #[must_use]
    pub fn dense(dimension: usize, distance: DistanceMetric) -> Self {
        Self {
            vector_type: VectorType::Dense { dimension },
            distance: DistanceType::Dense(distance),
            index: IndexConfig::hnsw_default(),
        }
    }

    /// Create a new sparse vector configuration.
    ///
    /// Sparse vectors have a variable dimension (up to `max_dimension`) and
    /// store only non-zero elements. Uses an inverted index by default.
    ///
    /// # Arguments
    ///
    /// * `max_dimension` - The maximum vocabulary size (e.g., 30522 for BERT)
    #[must_use]
    pub fn sparse(max_dimension: u32) -> Self {
        Self {
            vector_type: VectorType::Sparse { max_dimension },
            distance: DistanceType::Sparse(SparseDistanceMetric::DotProduct),
            index: IndexConfig::inverted_default(),
        }
    }

    /// Create a new multi-vector configuration for ColBERT-style embeddings.
    ///
    /// Multi-vectors store per-token embeddings and use MaxSim aggregation
    /// for scoring. Uses an HNSW index with DotProduct distance by default.
    ///
    /// # Arguments
    ///
    /// * `token_dim` - The dimension of each token embedding (e.g., 128)
    #[must_use]
    pub fn multi_vector(token_dim: usize) -> Self {
        Self {
            vector_type: VectorType::Multi { token_dim },
            distance: DistanceType::Dense(DistanceMetric::DotProduct),
            index: IndexConfig::hnsw_with_aggregation(AggregationMethod::MaxSim),
        }
    }

    /// Create a new binary vector configuration.
    ///
    /// Binary vectors are bit-packed and support Hamming distance.
    /// Uses an HNSW index by default.
    ///
    /// # Arguments
    ///
    /// * `bits` - The number of bits in the binary vector
    #[must_use]
    pub fn binary(bits: usize) -> Self {
        Self {
            vector_type: VectorType::Binary { bits },
            distance: DistanceType::Binary(BinaryDistanceType::Hamming),
            index: IndexConfig::hnsw_default(),
        }
    }

    /// Set a custom distance metric.
    #[must_use]
    pub fn with_distance(mut self, distance: DistanceType) -> Self {
        self.distance = distance;
        self
    }

    /// Set a custom index configuration.
    #[must_use]
    pub fn with_index(mut self, index: IndexConfig) -> Self {
        self.index = index;
        self
    }

    /// Get the dimension of the vector.
    ///
    /// Returns `None` for sparse vectors (which have variable dimension).
    #[must_use]
    pub fn dimension(&self) -> Option<usize> {
        match &self.vector_type {
            VectorType::Dense { dimension } => Some(*dimension),
            VectorType::Sparse { .. } => None,
            VectorType::Multi { token_dim } => Some(*token_dim),
            VectorType::Binary { bits } => Some(*bits),
        }
    }
}

/// The type of vector stored in a named vector space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorType {
    /// Dense fixed-dimension vector (e.g., BERT, OpenAI embeddings).
    Dense {
        /// The fixed dimension of vectors in this space.
        dimension: usize,
    },
    /// Sparse vector with variable non-zero elements (e.g., SPLADE, BM25).
    Sparse {
        /// Maximum vocabulary index (exclusive upper bound).
        max_dimension: u32,
    },
    /// Multi-vector for per-token embeddings (e.g., ColBERT).
    Multi {
        /// The dimension of each token embedding.
        token_dim: usize,
    },
    /// Binary bit-packed vector (e.g., LSH, SimHash).
    Binary {
        /// The number of bits.
        bits: usize,
    },
}

impl VectorType {
    /// Check if this is a dense vector type.
    #[must_use]
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }

    /// Check if this is a sparse vector type.
    #[must_use]
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse { .. })
    }

    /// Check if this is a multi-vector type.
    #[must_use]
    pub fn is_multi(&self) -> bool {
        matches!(self, Self::Multi { .. })
    }

    /// Check if this is a binary vector type.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary { .. })
    }
}

/// Distance metric configuration.
///
/// Different vector types support different distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceType {
    /// Distance metrics for dense and multi-vectors.
    Dense(DistanceMetric),
    /// Distance metrics for sparse vectors.
    Sparse(SparseDistanceMetric),
    /// Distance metrics for binary vectors.
    Binary(BinaryDistanceType),
}

/// Binary distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryDistanceType {
    /// Hamming distance (raw bit difference count).
    Hamming,
    /// Normalized Hamming distance (bit differences / dimension).
    HammingNormalized,
    /// Jaccard distance (1 - intersection/union).
    Jaccard,
}

/// Index configuration for a named vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    /// The indexing method.
    pub method: IndexMethod,
    /// Aggregation method for multi-vector queries.
    pub aggregation: Option<AggregationMethod>,
}

impl IndexConfig {
    /// Create a default HNSW index configuration.
    #[must_use]
    pub fn hnsw_default() -> Self {
        Self { method: IndexMethod::Hnsw(HnswParams::default()), aggregation: None }
    }

    /// Create an HNSW index with custom parameters.
    #[must_use]
    pub fn hnsw(params: HnswParams) -> Self {
        Self { method: IndexMethod::Hnsw(params), aggregation: None }
    }

    /// Create an HNSW index with aggregation (for multi-vectors).
    #[must_use]
    pub fn hnsw_with_aggregation(aggregation: AggregationMethod) -> Self {
        Self { method: IndexMethod::Hnsw(HnswParams::default()), aggregation: Some(aggregation) }
    }

    /// Create a default inverted index configuration.
    #[must_use]
    pub fn inverted_default() -> Self {
        Self { method: IndexMethod::Inverted(InvertedIndexParams::default()), aggregation: None }
    }

    /// Create an inverted index with custom parameters.
    #[must_use]
    pub fn inverted(params: InvertedIndexParams) -> Self {
        Self { method: IndexMethod::Inverted(params), aggregation: None }
    }

    /// Create a flat (brute-force) index configuration.
    #[must_use]
    pub fn flat() -> Self {
        Self { method: IndexMethod::Flat, aggregation: None }
    }

    /// Set the aggregation method for multi-vector queries.
    #[must_use]
    pub fn with_aggregation(mut self, aggregation: AggregationMethod) -> Self {
        self.aggregation = Some(aggregation);
        self
    }
}

/// Indexing methods for vector search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IndexMethod {
    /// Hierarchical Navigable Small World graph.
    Hnsw(HnswParams),
    /// Inverted index for sparse vectors.
    Inverted(InvertedIndexParams),
    /// Flat (brute-force) search.
    Flat,
}

/// HNSW index parameters.
///
/// These parameters control the construction and search behavior of HNSW indexes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswParams {
    /// Maximum number of connections per node (M parameter).
    /// Typical values: 16-64. Higher = better recall, more memory.
    pub m: usize,
    /// Maximum connections in layer 0 (typically 2 * M).
    pub m_max0: usize,
    /// Beam width during construction. Higher = better quality, slower build.
    /// Typical values: 100-500.
    pub ef_construction: usize,
    /// Default beam width during search. Higher = better recall, slower search.
    /// Typical values: 10-500. Can be overridden per-query.
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, m_max0: 32, ef_construction: 200, ef_search: 50 }
    }
}

impl HnswParams {
    /// Create new HNSW parameters with the specified M value.
    #[must_use]
    pub fn new(m: usize) -> Self {
        let m = m.max(2);
        Self { m, m_max0: m * 2, ef_construction: 200, ef_search: 50 }
    }

    /// Set the construction beam width.
    #[must_use]
    pub const fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the search beam width.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }
}

/// Inverted index parameters for sparse vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InvertedIndexParams {
    /// Whether to apply IDF weighting during search.
    pub use_idf: bool,
    /// Minimum term frequency to include in the index.
    pub min_term_freq: u32,
}

impl Default for InvertedIndexParams {
    fn default() -> Self {
        Self { use_idf: true, min_term_freq: 1 }
    }
}

impl InvertedIndexParams {
    /// Create new inverted index parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self { use_idf: true, min_term_freq: 1 }
    }

    /// Enable or disable IDF weighting.
    #[must_use]
    pub const fn with_idf(mut self, use_idf: bool) -> Self {
        self.use_idf = use_idf;
        self
    }

    /// Set minimum term frequency.
    #[must_use]
    pub const fn with_min_term_freq(mut self, min_freq: u32) -> Self {
        self.min_term_freq = min_freq;
        self
    }
}

/// Aggregation method for multi-vector queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Maximum similarity (ColBERT-style late interaction).
    /// For each query token, finds the max similarity across document tokens,
    /// then sums these max similarities.
    MaxSim,
    /// Average of all pairwise similarities.
    Average,
    /// Sum of all pairwise similarities.
    Sum,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_config() {
        let config = VectorConfig::dense(768, DistanceMetric::Cosine);
        assert!(config.vector_type.is_dense());
        assert_eq!(config.dimension(), Some(768));
        assert!(matches!(config.distance, DistanceType::Dense(DistanceMetric::Cosine)));
        assert!(matches!(config.index.method, IndexMethod::Hnsw(_)));
    }

    #[test]
    fn test_sparse_config() {
        let config = VectorConfig::sparse(30522);
        assert!(config.vector_type.is_sparse());
        assert_eq!(config.dimension(), None);
        assert!(matches!(config.distance, DistanceType::Sparse(SparseDistanceMetric::DotProduct)));
        assert!(matches!(config.index.method, IndexMethod::Inverted(_)));
    }

    #[test]
    fn test_multi_vector_config() {
        let config = VectorConfig::multi_vector(128);
        assert!(config.vector_type.is_multi());
        assert_eq!(config.dimension(), Some(128));
        assert!(matches!(config.distance, DistanceType::Dense(DistanceMetric::DotProduct)));
        assert_eq!(config.index.aggregation, Some(AggregationMethod::MaxSim));
    }

    #[test]
    fn test_binary_config() {
        let config = VectorConfig::binary(1024);
        assert!(config.vector_type.is_binary());
        assert_eq!(config.dimension(), Some(1024));
        assert!(matches!(config.distance, DistanceType::Binary(BinaryDistanceType::Hamming)));
    }

    #[test]
    fn test_hnsw_params_builder() {
        let params = HnswParams::new(32).with_ef_construction(400).with_ef_search(100);

        assert_eq!(params.m, 32);
        assert_eq!(params.m_max0, 64);
        assert_eq!(params.ef_construction, 400);
        assert_eq!(params.ef_search, 100);
    }

    #[test]
    fn test_custom_index_config() {
        let config = VectorConfig::dense(768, DistanceMetric::Euclidean).with_index(IndexConfig {
            method: IndexMethod::Hnsw(HnswParams::new(32).with_ef_construction(500)),
            aggregation: None,
        });

        if let IndexMethod::Hnsw(params) = &config.index.method {
            assert_eq!(params.m, 32);
            assert_eq!(params.ef_construction, 500);
        } else {
            panic!("Expected HNSW index");
        }
    }
}
