//! Traits for vector indexes.

use manifoldb_core::EntityId;

use crate::error::VectorError;
use crate::types::Embedding;

/// Result of a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The entity ID of the matching embedding.
    pub entity_id: EntityId,
    /// The distance to the query vector (lower is more similar for most metrics).
    pub distance: f32,
}

impl SearchResult {
    /// Create a new search result.
    #[must_use]
    pub const fn new(entity_id: EntityId, distance: f32) -> Self {
        Self { entity_id, distance }
    }
}

/// Configuration for filtered search operations.
#[derive(Debug, Clone)]
pub struct FilteredSearchConfig {
    /// The minimum ef_search to use when filtering.
    /// When filters are selective, ef_search may need to be increased
    /// to maintain recall.
    pub min_ef_search: usize,

    /// The maximum ef_search to use when filtering.
    pub max_ef_search: usize,

    /// The ef_search multiplier to apply when filtering.
    /// When a filter is applied, ef_search is multiplied by this factor
    /// to account for nodes that may be filtered out during traversal.
    pub ef_multiplier: f32,
}

impl Default for FilteredSearchConfig {
    fn default() -> Self {
        Self { min_ef_search: 40, max_ef_search: 500, ef_multiplier: 2.0 }
    }
}

impl FilteredSearchConfig {
    /// Create a new filtered search configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self { min_ef_search: 40, max_ef_search: 500, ef_multiplier: 2.0 }
    }

    /// Set the minimum ef_search.
    #[must_use]
    pub const fn with_min_ef_search(mut self, min_ef: usize) -> Self {
        self.min_ef_search = min_ef;
        self
    }

    /// Set the maximum ef_search.
    #[must_use]
    pub const fn with_max_ef_search(mut self, max_ef: usize) -> Self {
        self.max_ef_search = max_ef;
        self
    }

    /// Set the ef_search multiplier.
    #[must_use]
    pub const fn with_ef_multiplier(mut self, multiplier: f32) -> Self {
        self.ef_multiplier = multiplier;
        self
    }

    /// Calculate the adjusted ef_search based on filter selectivity.
    ///
    /// # Arguments
    ///
    /// * `base_ef` - The base ef_search value
    /// * `selectivity` - Optional estimated filter selectivity (0.0 to 1.0).
    ///   1.0 means all nodes pass, 0.0 means no nodes pass.
    ///   If None, the default multiplier is applied.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn adjusted_ef(&self, base_ef: usize, selectivity: Option<f32>) -> usize {
        let multiplier = match selectivity {
            Some(s) if s > 0.0 && s < 1.0 => {
                // More aggressive expansion for more selective filters
                // For selectivity 0.1 (10% pass), expand by 10x
                // For selectivity 0.5 (50% pass), expand by 2x
                (1.0 / s).min(self.ef_multiplier * 5.0)
            }
            Some(_) => 1.0, // No multiplier needed if all/none pass
            None => self.ef_multiplier,
        };

        let adjusted = ((base_ef as f32) * multiplier) as usize;
        adjusted.clamp(self.min_ef_search, self.max_ef_search)
    }
}

/// Trait for vector similarity search indexes.
///
/// This trait defines the interface for approximate nearest neighbor (ANN)
/// indexes that support insert, delete, and k-NN search operations.
pub trait VectorIndex {
    /// Insert an embedding into the index.
    ///
    /// If an embedding already exists for the given entity, it will be updated.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding dimension doesn't match the index dimension,
    /// or if there's a storage error.
    fn insert(&mut self, entity_id: EntityId, embedding: &Embedding) -> Result<(), VectorError>;

    /// Insert multiple embeddings into the index in a batch operation.
    ///
    /// This is significantly more efficient than calling `insert` multiple times
    /// as it minimizes transaction overhead and optimizes HNSW graph construction.
    /// All embeddings are inserted within a single operation.
    ///
    /// If an embedding already exists for any entity, it will be updated.
    ///
    /// # Performance
    ///
    /// For bulk loading, this can provide up to 10x throughput improvement over
    /// individual `insert` calls.
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Slice of (EntityId, Embedding reference) pairs to insert
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding dimension doesn't match the index dimension,
    /// or if there's a storage error.
    fn insert_batch(&mut self, embeddings: &[(EntityId, &Embedding)]) -> Result<(), VectorError> {
        // Default implementation: insert one at a time
        // Implementations can override for better performance
        for (entity_id, embedding) in embeddings {
            self.insert(*entity_id, embedding)?;
        }
        Ok(())
    }

    /// Remove an embedding from the index.
    ///
    /// # Returns
    ///
    /// Returns `true` if the embedding was removed, `false` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if there's a storage error.
    fn delete(&mut self, entity_id: EntityId) -> Result<bool, VectorError>;

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// # Arguments
    ///
    /// * `query` - The query embedding
    /// * `k` - The number of nearest neighbors to return
    /// * `ef_search` - Optional beam width for search (uses default if None)
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by distance (closest first).
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index dimension,
    /// or if there's a storage error.
    fn search(
        &self,
        query: &Embedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError>;

    /// Search for the k nearest neighbors with a filter predicate.
    ///
    /// This performs filtered search where the predicate is applied during
    /// graph traversal, not as a post-filter. This is more efficient when
    /// the filter is selective, as it avoids exploring many non-matching paths.
    ///
    /// # Arguments
    ///
    /// * `query` - The query embedding
    /// * `k` - The number of nearest neighbors to return
    /// * `predicate` - A function that returns true for entity IDs that should be included
    /// * `ef_search` - Optional beam width for search (uses adjusted default if None)
    /// * `config` - Optional configuration for filtered search (uses defaults if None)
    ///
    /// # Returns
    ///
    /// A vector of search results that pass the predicate, sorted by distance (closest first).
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index dimension,
    /// or if there's a storage error.
    fn search_with_filter<F>(
        &self,
        query: &Embedding,
        k: usize,
        predicate: F,
        ef_search: Option<usize>,
        config: Option<FilteredSearchConfig>,
    ) -> Result<Vec<SearchResult>, VectorError>
    where
        F: Fn(EntityId) -> bool;

    /// Check if an embedding exists in the index.
    fn contains(&self, entity_id: EntityId) -> Result<bool, VectorError>;

    /// Get the number of embeddings in the index.
    fn len(&self) -> Result<usize, VectorError>;

    /// Check if the index is empty.
    fn is_empty(&self) -> Result<bool, VectorError> {
        self.len().map(|n| n == 0)
    }

    /// Get the dimension of embeddings in this index.
    ///
    /// # Errors
    ///
    /// Returns an error if there's a concurrency error (lock poisoning).
    fn dimension(&self) -> Result<usize, VectorError>;
}
