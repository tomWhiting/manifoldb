//! Coordinator for HNSW indexes and separated vector storage.
//!
//! This module provides [`VectorIndexCoordinator`] which bridges the HNSW index
//! system with the [`CollectionVectorStore`]. It handles:
//!
//! - Automatic HNSW updates when vectors are stored
//! - Index rebuild from vector storage
//! - Cascade operations (delete entity → delete from index)
//! - Unified search across collections with named vectors
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                     VectorIndexCoordinator                           │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │                                                                      │
//! │  ┌────────────────────┐          ┌──────────────────────┐           │
//! │  │ CollectionVector   │          │   HnswIndexManager   │           │
//! │  │     Store          │          │                      │           │
//! │  │                    │          │  ┌────────────────┐  │           │
//! │  │ (collection_id,    │   ─────▶ │  │ documents_text │  │           │
//! │  │  entity_id,        │          │  │   _hnsw        │  │           │
//! │  │  vector_name)      │          │  ├────────────────┤  │           │
//! │  │      ↓             │          │  │ documents_image│  │           │
//! │  │   VectorData       │          │  │   _hnsw        │  │           │
//! │  └────────────────────┘          │  └────────────────┘  │           │
//! │                                   └──────────────────────┘           │
//! │                                                                      │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::index::{VectorIndexCoordinator, HnswConfig};
//! use manifoldb_vector::distance::DistanceMetric;
//! use manifoldb_core::{CollectionId, EntityId};
//!
//! let coordinator = VectorIndexCoordinator::new(engine);
//!
//! // Create an HNSW index for a collection's named vector
//! coordinator.create_index(
//!     "documents",
//!     "text_embedding",
//!     1536,
//!     DistanceMetric::Cosine,
//!     &HnswConfig::default(),
//! )?;
//!
//! // Store a vector and automatically update HNSW
//! let collection_id = CollectionId::new(1);
//! let entity_id = EntityId::new(42);
//! coordinator.upsert_vector(
//!     collection_id,
//!     entity_id,
//!     "documents",
//!     "text_embedding",
//!     &VectorData::Dense(vec![0.1; 1536]),
//! )?;
//!
//! // Search returns entity IDs
//! let query = Embedding::new(vec![0.1; 1536])?;
//! let results = coordinator.search("documents", "text_embedding", &query, 10)?;
//! ```

use std::sync::Arc;

use manifoldb_core::{CollectionId, EntityId};
use manifoldb_storage::StorageEngine;

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::store::CollectionVectorStore;
use crate::types::{CollectionName, Embedding, VectorData};

use super::config::HnswConfig;
use super::manager::HnswIndexManager;
use super::traits::{SearchResult, VectorIndex};

/// Coordinator for HNSW indexes with separated vector storage.
///
/// This struct provides a unified interface for:
/// - Storing vectors in [`CollectionVectorStore`]
/// - Maintaining HNSW indexes via [`HnswIndexManager`]
/// - Coordinating insert/update/delete operations
/// - Rebuilding indexes from stored vectors
///
/// The coordinator ensures that:
/// 1. Vectors are stored in the vector store for persistence
/// 2. HNSW indexes are updated automatically for fast similarity search
/// 3. Delete operations cascade properly to both stores
pub struct VectorIndexCoordinator<E: StorageEngine> {
    /// The vector store for persisted vector data.
    vector_store: CollectionVectorStore<E>,
    /// The HNSW index manager for similarity search.
    index_manager: Arc<HnswIndexManager<E>>,
}

impl<E: StorageEngine> VectorIndexCoordinator<E> {
    /// Create a new coordinator.
    ///
    /// # Arguments
    ///
    /// * `engine` - The storage engine
    #[must_use]
    pub fn new(engine: E) -> Self {
        Self {
            vector_store: CollectionVectorStore::new(engine),
            index_manager: Arc::new(HnswIndexManager::new()),
        }
    }

    /// Create a coordinator with an existing index manager.
    ///
    /// Use this when you need to share an index manager across multiple coordinators.
    #[must_use]
    pub fn with_manager(engine: E, index_manager: Arc<HnswIndexManager<E>>) -> Self {
        Self { vector_store: CollectionVectorStore::new(engine), index_manager }
    }

    /// Get a reference to the vector store.
    #[must_use]
    pub fn vector_store(&self) -> &CollectionVectorStore<E> {
        &self.vector_store
    }

    /// Get a reference to the index manager.
    #[must_use]
    pub fn index_manager(&self) -> &Arc<HnswIndexManager<E>> {
        &self.index_manager
    }

    // ========================================================================
    // Vector Operations (with automatic HNSW updates)
    // ========================================================================

    /// Upsert a vector, updating both storage and HNSW index.
    ///
    /// This method:
    /// 1. Stores the vector in [`CollectionVectorStore`]
    /// 2. Updates the HNSW index (if one exists for this named vector)
    ///
    /// # Arguments
    ///
    /// * `collection_id` - The collection ID
    /// * `entity_id` - The entity ID
    /// * `collection_name` - Collection name for index lookup
    /// * `vector_name` - The vector name within the collection
    /// * `data` - The vector data to store
    ///
    /// # Errors
    ///
    /// Returns an error if storage or index update fails.
    pub fn upsert_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError> {
        // 1. Store in vector store
        self.vector_store.put_vector(collection_id, entity_id, vector_name, data)?;

        // 2. Update HNSW index (if one exists and this is a dense vector)
        if let Some(dense) = data.as_dense() {
            // Get the index from the manager
            if let Ok(Some(index)) = self.index_manager.get_index(collection_name, vector_name) {
                let embedding = Embedding::new(dense.to_vec())?;
                let mut guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                guard.insert(entity_id, &embedding)?;
            }
        }

        Ok(())
    }

    /// Upsert multiple vectors in a batch.
    ///
    /// More efficient than calling `upsert_vector` multiple times as it
    /// batches storage operations and HNSW insertions.
    ///
    /// # Arguments
    ///
    /// * `collection_id` - The collection ID
    /// * `collection_name` - Collection name for index lookup
    /// * `vectors` - List of (entity_id, vector_name, vector_data) tuples
    pub fn upsert_vectors_batch(
        &self,
        collection_id: CollectionId,
        collection_name: &str,
        vectors: &[(EntityId, &str, &VectorData)],
    ) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Ok(());
        }

        // 1. Batch store in vector store
        self.vector_store.put_vectors_batch(collection_id, vectors)?;

        // 2. Group vectors by name and update each HNSW index
        use std::collections::HashMap;
        let mut by_name: HashMap<&str, Vec<(EntityId, &VectorData)>> = HashMap::new();

        for (entity_id, name, data) in vectors {
            by_name.entry(*name).or_default().push((*entity_id, *data));
        }

        for (vector_name, entity_vectors) in by_name {
            if let Ok(Some(index)) = self.index_manager.get_index(collection_name, vector_name) {
                // Filter to dense vectors only and convert to embeddings
                let embeddings: Vec<(EntityId, Embedding)> = entity_vectors
                    .into_iter()
                    .filter_map(|(id, data)| {
                        data.as_dense().map(|d| Embedding::new(d.to_vec()).map(|e| (id, e)))
                    })
                    .filter_map(Result::ok)
                    .collect();

                if !embeddings.is_empty() {
                    let refs: Vec<(EntityId, &Embedding)> =
                        embeddings.iter().map(|(id, e)| (*id, e)).collect();

                    let mut guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                    guard.insert_batch(&refs)?;
                }
            }
        }

        Ok(())
    }

    /// Delete a vector from both storage and HNSW index.
    ///
    /// # Returns
    ///
    /// Returns `true` if the vector was deleted from storage.
    pub fn delete_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        // 1. Delete from vector store
        let deleted = self.vector_store.delete_vector(collection_id, entity_id, vector_name)?;

        // 2. Delete from HNSW index
        if let Ok(Some(index)) = self.index_manager.get_index(collection_name, vector_name) {
            let mut guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
            let _ = guard.delete(entity_id);
        }

        Ok(deleted)
    }

    /// Delete all vectors for an entity.
    ///
    /// This cascades the delete to all HNSW indexes for the collection.
    ///
    /// # Returns
    ///
    /// Returns the number of vectors deleted from storage.
    pub fn delete_entity_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
    ) -> Result<usize, VectorError> {
        // 1. Get all vector names before deleting (for HNSW cleanup)
        let vectors = self.vector_store.get_all_vectors(collection_id, entity_id)?;

        // 2. Delete from vector store
        let count = self.vector_store.delete_all_vectors(collection_id, entity_id)?;

        // 3. Delete from all relevant HNSW indexes
        for vector_name in vectors.keys() {
            if let Ok(Some(index)) = self.index_manager.get_index(collection_name, vector_name) {
                let mut guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                let _ = guard.delete(entity_id);
            }
        }

        Ok(count)
    }

    // ========================================================================
    // Vector Retrieval
    // ========================================================================

    /// Get a vector from storage.
    pub fn get_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<Option<VectorData>, VectorError> {
        self.vector_store.get_vector(collection_id, entity_id, vector_name)
    }

    /// Get all vectors for an entity.
    pub fn get_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<std::collections::HashMap<String, VectorData>, VectorError> {
        self.vector_store.get_all_vectors(collection_id, entity_id)
    }

    // ========================================================================
    // Similarity Search
    // ========================================================================

    /// Search for similar vectors using HNSW.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The collection to search
    /// * `vector_name` - The named vector to search
    /// * `query` - The query embedding
    /// * `k` - Number of results to return
    /// * `ef_search` - Optional beam width (uses default if None)
    ///
    /// # Returns
    ///
    /// A list of search results with entity IDs and distances.
    pub fn search(
        &self,
        collection_name: &str,
        vector_name: &str,
        query: &Embedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        let index =
            self.index_manager.get_index(collection_name, vector_name)?.ok_or_else(|| {
                VectorError::SpaceNotFound(format!(
                    "no HNSW index for {}.{}",
                    collection_name, vector_name
                ))
            })?;

        let guard = index.read().map_err(|_| VectorError::LockPoisoned)?;
        guard.search(query, k, ef_search)
    }

    /// Search with a filter predicate.
    ///
    /// The predicate is applied during graph traversal, not as a post-filter.
    pub fn search_with_filter<F>(
        &self,
        collection_name: &str,
        vector_name: &str,
        query: &Embedding,
        k: usize,
        predicate: F,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError>
    where
        F: Fn(EntityId) -> bool,
    {
        let index =
            self.index_manager.get_index(collection_name, vector_name)?.ok_or_else(|| {
                VectorError::SpaceNotFound(format!(
                    "no HNSW index for {}.{}",
                    collection_name, vector_name
                ))
            })?;

        let guard = index.read().map_err(|_| VectorError::LockPoisoned)?;
        guard.search_with_filter(query, k, predicate, ef_search, None)
    }

    // ========================================================================
    // Index Rebuild
    // ========================================================================

    /// Rebuild an HNSW index from the vector store.
    ///
    /// This is useful for:
    /// - Recovering from index corruption
    /// - Building an index for existing vectors
    /// - Re-indexing after configuration changes
    ///
    /// # Arguments
    ///
    /// * `collection_id` - The collection ID
    /// * `collection_name` - Collection name for index lookup
    /// * `vector_name` - The named vector to rebuild
    ///
    /// # Returns
    ///
    /// The number of vectors indexed.
    pub fn rebuild_index_from_store(
        &self,
        collection_id: CollectionId,
        collection_name: &str,
        vector_name: &str,
    ) -> Result<usize, VectorError> {
        // Get all entities with this vector
        let entity_ids = self.vector_store.list_entities_with_vector(collection_id, vector_name)?;

        // Collect vectors
        let mut vectors = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            if let Some(data) =
                self.vector_store.get_vector(collection_id, entity_id, vector_name)?
            {
                if let Some(dense) = data.as_dense() {
                    vectors.push((entity_id, dense.to_vec()));
                }
            }
        }

        // Rebuild using the index manager
        let points =
            vectors.into_iter().map(|(id, v)| (manifoldb_core::PointId::new(id.as_u64()), v));

        self.index_manager.rebuild_index(collection_name, vector_name, points)
    }

    // ========================================================================
    // Index Status
    // ========================================================================

    /// Check if an index is loaded in memory.
    pub fn is_index_loaded(&self, collection: &str, vector_name: &str) -> bool {
        self.index_manager.is_index_loaded(collection, vector_name).unwrap_or(false)
    }
}

/// Extension methods for the coordinator that require ownership of a storage engine.
impl<E: StorageEngine> VectorIndexCoordinator<E> {
    /// Create an HNSW index for a collection's named vector.
    ///
    /// Note: This method takes a separate engine instance for index persistence.
    /// You may want to call this with a shared engine reference or a cloned engine.
    ///
    /// # Arguments
    ///
    /// * `engine` - Storage engine for the index persistence
    /// * `collection` - Collection name (e.g., "documents")
    /// * `vector_name` - Vector name within the collection (e.g., "text_embedding")
    /// * `dimension` - Vector dimension
    /// * `distance_metric` - Distance metric for similarity
    /// * `config` - HNSW configuration
    ///
    /// # Returns
    ///
    /// The name of the created index (e.g., "documents_text_embedding_hnsw")
    pub fn create_index(
        &self,
        engine: E,
        collection: &str,
        vector_name: &str,
        dimension: usize,
        distance_metric: DistanceMetric,
        config: &HnswConfig,
    ) -> Result<String, VectorError> {
        let collection_name = CollectionName::new(collection)?;
        self.index_manager.create_index_for_vector(
            engine,
            &collection_name,
            vector_name,
            dimension,
            distance_metric,
            config,
        )
    }

    /// Drop an HNSW index.
    pub fn drop_index(&self, engine: &E, index_name: &str) -> Result<bool, VectorError> {
        self.index_manager.drop_index(engine, index_name)
    }

    /// Drop all HNSW indexes for a collection.
    pub fn drop_collection_indexes(
        &self,
        engine: &E,
        collection: &str,
    ) -> Result<Vec<String>, VectorError> {
        let collection_name = CollectionName::new(collection)?;
        self.index_manager.drop_indexes_for_collection(engine, &collection_name)
    }

    /// Check if an HNSW index exists for a collection's named vector.
    pub fn has_index(&self, engine: &E, collection: &str, vector_name: &str) -> bool {
        self.index_manager.has_index(engine, collection, vector_name).unwrap_or(false)
    }

    /// Load an existing index into memory.
    pub fn load_index(&self, engine: E, index_name: &str) -> Result<(), VectorError> {
        self.index_manager.load_index(engine, index_name)
    }

    /// Rebuild an index from scratch with new configuration.
    ///
    /// This drops the existing index data and creates a fresh index.
    ///
    /// # Arguments
    ///
    /// * `engine` - Storage engine for the new index
    /// * `collection_id` - The collection ID
    /// * `collection_name` - Collection name
    /// * `vector_name` - The named vector to rebuild
    ///
    /// # Returns
    ///
    /// The number of vectors indexed.
    pub fn rebuild_index_from_scratch(
        &self,
        engine: E,
        collection_id: CollectionId,
        collection_name: &str,
        vector_name: &str,
    ) -> Result<usize, VectorError> {
        // Get all entities with this vector
        let entity_ids = self.vector_store.list_entities_with_vector(collection_id, vector_name)?;

        // Collect vectors
        let mut vectors = Vec::with_capacity(entity_ids.len());
        for entity_id in entity_ids {
            if let Some(data) =
                self.vector_store.get_vector(collection_id, entity_id, vector_name)?
            {
                if let Some(dense) = data.as_dense() {
                    vectors.push((entity_id, dense.to_vec()));
                }
            }
        }

        // Rebuild from scratch using the index manager
        let points =
            vectors.into_iter().map(|(id, v)| (manifoldb_core::PointId::new(id.as_u64()), v));

        self.index_manager.rebuild_index_from_scratch(engine, collection_name, vector_name, points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_engines() -> (RedbEngine, RedbEngine) {
        // Create two separate in-memory engines for coordinator and index creation
        // This is needed because we can't clone RedbEngine
        (RedbEngine::in_memory().unwrap(), RedbEngine::in_memory().unwrap())
    }

    #[test]
    fn test_create_coordinator() {
        let (coord_engine, _) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);
        // Basic sanity check
        assert!(!coordinator.is_index_loaded("test", "vec"));
    }

    #[test]
    fn test_create_index() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        let index_name = coordinator
            .create_index(
                index_engine,
                "documents",
                "text_embedding",
                384,
                DistanceMetric::Cosine,
                &HnswConfig::default(),
            )
            .unwrap();

        assert_eq!(index_name, "documents_text_embedding_hnsw");
        assert!(coordinator.is_index_loaded("documents", "text_embedding"));
    }

    #[test]
    fn test_upsert_and_search() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        // Create an index
        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);

        // Upsert vectors
        for i in 1..=5 {
            let data = VectorData::Dense(vec![i as f32; 4]);
            coordinator
                .upsert_vector(collection_id, EntityId::new(i), "docs", "vec", &data)
                .unwrap();
        }

        // Search
        let query = Embedding::new(vec![3.0; 4]).unwrap();
        let results = coordinator.search("docs", "vec", &query, 3, None).unwrap();

        assert_eq!(results.len(), 3);
        // Entity 3 should be closest (exact match)
        assert_eq!(results[0].entity_id, EntityId::new(3));
    }

    #[test]
    fn test_delete_vector() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(1);

        // Upsert
        coordinator
            .upsert_vector(
                collection_id,
                entity_id,
                "docs",
                "vec",
                &VectorData::Dense(vec![1.0; 4]),
            )
            .unwrap();

        // Verify it exists
        assert!(coordinator.get_vector(collection_id, entity_id, "vec").unwrap().is_some());

        // Delete
        let deleted = coordinator.delete_vector(collection_id, entity_id, "docs", "vec").unwrap();
        assert!(deleted);

        // Verify it's gone
        assert!(coordinator.get_vector(collection_id, entity_id, "vec").unwrap().is_none());

        // Search should return no results
        let query = Embedding::new(vec![1.0; 4]).unwrap();
        let results = coordinator.search("docs", "vec", &query, 1, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_upsert() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);

        // Create batch data
        let data1 = VectorData::Dense(vec![1.0; 4]);
        let data2 = VectorData::Dense(vec![2.0; 4]);
        let data3 = VectorData::Dense(vec![3.0; 4]);

        let vectors: Vec<(EntityId, &str, &VectorData)> = vec![
            (EntityId::new(1), "vec", &data1),
            (EntityId::new(2), "vec", &data2),
            (EntityId::new(3), "vec", &data3),
        ];

        coordinator.upsert_vectors_batch(collection_id, "docs", &vectors).unwrap();

        // Verify all vectors are stored
        for i in 1..=3 {
            assert!(coordinator
                .get_vector(collection_id, EntityId::new(i), "vec")
                .unwrap()
                .is_some());
        }

        // Search should work
        let query = Embedding::new(vec![2.0; 4]).unwrap();
        let results = coordinator.search("docs", "vec", &query, 3, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_rebuild_from_store() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        let collection_id = CollectionId::new(1);

        // Store vectors directly (without HNSW index)
        for i in 1..=5 {
            coordinator
                .vector_store()
                .put_vector(
                    collection_id,
                    EntityId::new(i),
                    "vec",
                    &VectorData::Dense(vec![i as f32; 4]),
                )
                .unwrap();
        }

        // Create index (empty initially)
        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        // Rebuild from store
        let count = coordinator.rebuild_index_from_store(collection_id, "docs", "vec").unwrap();
        assert_eq!(count, 5);

        // Search should now work
        let query = Embedding::new(vec![3.0; 4]).unwrap();
        let results = coordinator.search("docs", "vec", &query, 3, None).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_with_filter() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);

        // Upsert vectors
        for i in 1..=10 {
            let data = VectorData::Dense(vec![i as f32; 4]);
            coordinator
                .upsert_vector(collection_id, EntityId::new(i), "docs", "vec", &data)
                .unwrap();
        }

        // Search with filter: only even IDs
        let query = Embedding::new(vec![5.0; 4]).unwrap();
        let predicate = |id: EntityId| id.as_u64() % 2 == 0;

        let results =
            coordinator.search_with_filter("docs", "vec", &query, 3, predicate, None).unwrap();

        // All results should be even
        for result in &results {
            assert_eq!(result.entity_id.as_u64() % 2, 0);
        }
    }

    #[test]
    fn test_sparse_vector_ignored_for_hnsw() {
        let (coord_engine, index_engine) = create_test_engines();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        coordinator
            .create_index(
                index_engine,
                "docs",
                "vec",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);

        // Upsert a sparse vector (should be stored but not indexed)
        let sparse = VectorData::Sparse(vec![(0, 1.0), (2, 0.5)]);
        coordinator.upsert_vector(collection_id, EntityId::new(1), "docs", "vec", &sparse).unwrap();

        // Vector should be stored
        let retrieved = coordinator.get_vector(collection_id, EntityId::new(1), "vec").unwrap();
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_sparse());

        // HNSW search should return nothing (sparse vectors not indexed)
        let query = Embedding::new(vec![1.0; 4]).unwrap();
        let results = coordinator.search("docs", "vec", &query, 1, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_multiple_named_vectors() {
        let (coord_engine, index_engine1) = create_test_engines();
        let index_engine2 = RedbEngine::in_memory().unwrap();
        let coordinator = VectorIndexCoordinator::new(coord_engine);

        // Create indexes for different named vectors
        coordinator
            .create_index(
                index_engine1,
                "docs",
                "text",
                4,
                DistanceMetric::Cosine,
                &HnswConfig::default(),
            )
            .unwrap();

        coordinator
            .create_index(
                index_engine2,
                "docs",
                "image",
                8,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(1);

        // Upsert different vectors for the same entity
        coordinator
            .upsert_vector(
                collection_id,
                entity_id,
                "docs",
                "text",
                &VectorData::Dense(vec![0.5; 4]),
            )
            .unwrap();

        coordinator
            .upsert_vector(
                collection_id,
                entity_id,
                "docs",
                "image",
                &VectorData::Dense(vec![0.25; 8]),
            )
            .unwrap();

        // Both vectors should be stored
        let text_vec = coordinator.get_vector(collection_id, entity_id, "text").unwrap();
        let image_vec = coordinator.get_vector(collection_id, entity_id, "image").unwrap();

        assert!(text_vec.is_some());
        assert!(image_vec.is_some());
        assert_eq!(text_vec.unwrap().dimension(), 4);
        assert_eq!(image_vec.unwrap().dimension(), 8);

        // Search should work on each index
        let text_query = Embedding::new(vec![0.5; 4]).unwrap();
        let text_results = coordinator.search("docs", "text", &text_query, 1, None).unwrap();
        assert_eq!(text_results.len(), 1);

        let image_query = Embedding::new(vec![0.25; 8]).unwrap();
        let image_results = coordinator.search("docs", "image", &image_query, 1, None).unwrap();
        assert_eq!(image_results.len(), 1);
    }
}
