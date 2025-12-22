//! Search builders for vector similarity search.
//!
//! This module provides builder types for constructing search queries
//! against collections.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::collection::{Filter, Vector};
//!
//! // Simple search
//! let results = collection.search("text")
//!     .query(query_vector)
//!     .limit(10)
//!     .execute()?;
//!
//! // Search with filter
//! let results = collection.search("text")
//!     .query(query_vector)
//!     .limit(10)
//!     .filter(Filter::eq("category", "programming"))
//!     .with_payload(true)
//!     .execute()?;
//! ```

use manifoldb_storage::StorageEngine;

use super::error::{ApiError, ApiResult};
use super::filter::Filter;
use super::handle::CollectionHandle;
use super::point::{ScoredPoint, Vector};

/// A builder for constructing a single-vector search query.
///
/// Use this to search a collection by a single named vector.
/// The search returns points ordered by similarity score.
pub struct SearchBuilder<'a, E: StorageEngine> {
    /// Reference to the collection handle.
    pub(crate) handle: &'a CollectionHandle<E>,
    /// Name of the vector to search.
    pub(crate) vector_name: String,
    /// The query vector.
    pub(crate) query: Option<Vector>,
    /// Maximum number of results.
    pub(crate) limit: usize,
    /// Offset for pagination.
    pub(crate) offset: usize,
    /// Optional filter to apply.
    pub(crate) filter: Option<Filter>,
    /// Whether to include payloads in results.
    pub(crate) with_payload: bool,
    /// Whether to include vectors in results.
    pub(crate) with_vectors: bool,
    /// Score threshold (only return results above this score).
    pub(crate) score_threshold: Option<f32>,
    /// HNSW ef parameter override for this search.
    pub(crate) ef: Option<usize>,
}

impl<'a, E: StorageEngine> SearchBuilder<'a, E> {
    /// Create a new search builder for the specified vector name.
    pub(crate) fn new(handle: &'a CollectionHandle<E>, vector_name: impl Into<String>) -> Self {
        Self {
            handle,
            vector_name: vector_name.into(),
            query: None,
            limit: 10,
            offset: 0,
            filter: None,
            with_payload: false,
            with_vectors: false,
            score_threshold: None,
            ef: None,
        }
    }

    /// Set the query vector.
    ///
    /// The vector type must match the vector configuration in the collection schema.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = collection.search("text")
    ///     .query(vec![0.1; 768])
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn query(mut self, vector: impl Into<Vector>) -> Self {
        self.query = Some(vector.into());
        self
    }

    /// Set the maximum number of results to return.
    ///
    /// Default is 10.
    #[must_use]
    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the offset for pagination.
    ///
    /// Use with `limit` to paginate through results.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get second page of 10 results
    /// let results = collection.search("text")
    ///     .query(query_vector)
    ///     .limit(10)
    ///     .offset(10)
    ///     .execute()?;
    /// ```
    #[must_use]
    pub const fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Add a filter to narrow down results.
    ///
    /// Filters are applied to the payload before vector similarity is computed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::collection::Filter;
    ///
    /// let results = collection.search("text")
    ///     .query(query_vector)
    ///     .filter(Filter::eq("category", "programming"))
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Include point payloads in search results.
    ///
    /// Default is `false`.
    #[must_use]
    pub const fn with_payload(mut self, include: bool) -> Self {
        self.with_payload = include;
        self
    }

    /// Include point vectors in search results.
    ///
    /// Default is `false`.
    #[must_use]
    pub const fn with_vectors(mut self, include: bool) -> Self {
        self.with_vectors = include;
        self
    }

    /// Set a minimum score threshold.
    ///
    /// Only results with scores above this threshold will be returned.
    #[must_use]
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Override the HNSW ef (beam width) parameter for this search.
    ///
    /// Higher values give more accurate results but slower search.
    /// Default is the collection's configured ef_search value.
    #[must_use]
    pub const fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Execute the search and return results.
    ///
    /// Results are ordered by similarity score in descending order.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No query vector was provided
    /// - The query vector type doesn't match the schema
    /// - The limit is zero
    /// - Storage operations fail
    pub fn execute(self) -> ApiResult<Vec<ScoredPoint>> {
        let query = self.query.ok_or(ApiError::EmptyQueryVector)?;

        if self.limit == 0 {
            return Err(ApiError::InvalidSearchLimit);
        }

        self.handle.execute_search(
            &self.vector_name,
            query,
            self.limit,
            self.offset,
            self.filter,
            self.with_payload,
            self.with_vectors,
            self.score_threshold,
            self.ef,
        )
    }
}

/// A builder for constructing hybrid (multi-vector) search queries.
///
/// Hybrid search combines results from multiple vectors using a fusion
/// strategy like Reciprocal Rank Fusion (RRF).
pub struct HybridSearchBuilder<'a, E: StorageEngine> {
    /// Reference to the collection handle.
    pub(crate) handle: &'a CollectionHandle<E>,
    /// Query vectors by name.
    pub(crate) queries: Vec<(String, Vector, f32)>, // (name, vector, weight)
    /// Maximum number of results.
    pub(crate) limit: usize,
    /// Offset for pagination.
    pub(crate) offset: usize,
    /// Optional filter to apply.
    pub(crate) filter: Option<Filter>,
    /// Whether to include payloads in results.
    pub(crate) with_payload: bool,
    /// Whether to include vectors in results.
    pub(crate) with_vectors: bool,
    /// Fusion strategy.
    pub(crate) fusion: FusionStrategy,
}

impl<'a, E: StorageEngine> HybridSearchBuilder<'a, E> {
    /// Create a new hybrid search builder.
    pub(crate) fn new(handle: &'a CollectionHandle<E>) -> Self {
        Self {
            handle,
            queries: Vec::new(),
            limit: 10,
            offset: 0,
            filter: None,
            with_payload: false,
            with_vectors: false,
            fusion: FusionStrategy::Rrf { k: 60.0 },
        }
    }

    /// Add a query vector with a weight.
    ///
    /// Higher weights give more importance to the vector in the fusion.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = collection.hybrid_search()
    ///     .query("dense", dense_vector, 0.7)
    ///     .query("sparse", sparse_vector, 0.3)
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn query(
        mut self,
        vector_name: impl Into<String>,
        vector: impl Into<Vector>,
        weight: f32,
    ) -> Self {
        self.queries.push((vector_name.into(), vector.into(), weight));
        self
    }

    /// Set the maximum number of results to return.
    #[must_use]
    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the offset for pagination.
    #[must_use]
    pub const fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Add a filter to narrow down results.
    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Include point payloads in search results.
    #[must_use]
    pub const fn with_payload(mut self, include: bool) -> Self {
        self.with_payload = include;
        self
    }

    /// Include point vectors in search results.
    #[must_use]
    pub const fn with_vectors(mut self, include: bool) -> Self {
        self.with_vectors = include;
        self
    }

    /// Set the fusion strategy for combining results.
    ///
    /// Default is Reciprocal Rank Fusion with k=60.
    #[must_use]
    pub const fn fusion(mut self, strategy: FusionStrategy) -> Self {
        self.fusion = strategy;
        self
    }

    /// Execute the hybrid search and return fused results.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Less than two query vectors were provided
    /// - The limit is zero
    /// - Storage operations fail
    pub fn execute(self) -> ApiResult<Vec<ScoredPoint>> {
        if self.queries.len() < 2 {
            return Err(ApiError::InsufficientVectorsForHybrid);
        }

        if self.limit == 0 {
            return Err(ApiError::InvalidSearchLimit);
        }

        self.handle.execute_hybrid_search(
            self.queries,
            self.limit,
            self.offset,
            self.filter,
            self.with_payload,
            self.with_vectors,
            self.fusion,
        )
    }
}

/// Strategy for fusing results from multiple vector searches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion.
    ///
    /// Combines rankings using: score = sum(1 / (k + rank))
    /// where k is a constant (default 60).
    Rrf {
        /// The k constant for RRF. Higher values give more weight to top results.
        k: f32,
    },

    /// Weighted average of normalized scores.
    ///
    /// Scores are normalized to [0, 1] before averaging with weights.
    WeightedAverage,

    /// Sum of raw scores multiplied by weights.
    WeightedSum,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Rrf { k: 60.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_strategy_default() {
        let strategy = FusionStrategy::default();
        assert!(matches!(strategy, FusionStrategy::Rrf { k } if (k - 60.0).abs() < 0.001));
    }
}
