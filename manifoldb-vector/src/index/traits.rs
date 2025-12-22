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
