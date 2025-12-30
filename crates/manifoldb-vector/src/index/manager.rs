//! HNSW index manager for named vector system.
//!
//! This module provides the `HnswIndexManager` which coordinates HNSW index
//! lifecycle with the collection system. It handles:
//!
//! - Creating and dropping indexes for named vectors
//! - Updating indexes on point insert/update/delete
//! - Index recovery and rebuild

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use manifoldb_core::PointId;
use manifoldb_storage::{StorageEngine, Transaction};

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::types::{CollectionName, Embedding, NamedVector};

use super::config::HnswConfig;
use super::hnsw::HnswIndex;
use super::registry::{HnswIndexEntry, HnswRegistry};
use super::traits::VectorIndex;

/// Manages HNSW indexes for the named vector system.
///
/// The `HnswIndexManager` provides a unified interface for creating, updating,
/// and querying HNSW indexes that are associated with named vectors in collections.
///
/// # Type Parameters
///
/// * `E` - The storage engine type. The engine must be clonable so it can be
///   shared across multiple HNSW indexes.
pub struct HnswIndexManager<E: StorageEngine> {
    /// In-memory cache of loaded HNSW indexes.
    /// Key is the index name (e.g., "documents_embedding_hnsw").
    indexes: RwLock<HashMap<String, Arc<RwLock<HnswIndex<E>>>>>,
    /// Marker for the storage engine type.
    _phantom: PhantomData<E>,
}

impl<E: StorageEngine> HnswIndexManager<E> {
    /// Create a new empty index manager.
    ///
    /// The manager doesn't own the storage engine. Instead, each operation
    /// that requires storage access takes the engine as a parameter.
    #[must_use]
    pub fn new() -> Self {
        Self { indexes: RwLock::new(HashMap::new()), _phantom: PhantomData }
    }

    // ========================================================================
    // Index Lifecycle
    // ========================================================================

    /// Create an HNSW index for a specific named vector.
    ///
    /// # Arguments
    ///
    /// * `engine` - The storage engine to use
    /// * `collection` - The collection name
    /// * `vector_name` - The vector name
    /// * `dimension` - Vector dimension
    /// * `distance_metric` - Distance metric to use
    /// * `config` - HNSW configuration
    ///
    /// # Returns
    ///
    /// The name of the created index.
    pub fn create_index_for_vector(
        &self,
        engine: E,
        collection: &CollectionName,
        vector_name: &str,
        dimension: usize,
        distance_metric: DistanceMetric,
        config: &HnswConfig,
    ) -> Result<String, VectorError> {
        let collection_str = collection.as_str();
        let index_name = HnswRegistry::index_name_for_vector(collection_str, vector_name);

        // Check if index already exists
        {
            let tx = engine.begin_read()?;
            if HnswRegistry::exists(&tx, &index_name)? {
                return Err(VectorError::InvalidName(format!(
                    "index '{}' already exists",
                    index_name
                )));
            }
        }

        // Register in the registry first
        let entry = HnswIndexEntry::for_named_vector(
            collection_str,
            vector_name,
            dimension,
            distance_metric,
            config,
        );

        {
            let mut tx = engine.begin_write()?;
            HnswRegistry::register(&mut tx, &entry)?;
            tx.commit()?;
        }

        // Create the HNSW index
        let hnsw = HnswIndex::new(engine, &index_name, dimension, distance_metric, config.clone())?;

        // Cache the index
        {
            let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
            indexes.insert(index_name.clone(), Arc::new(RwLock::new(hnsw)));
        }

        Ok(index_name)
    }

    /// Drop all HNSW indexes for a collection.
    ///
    /// This is called when a collection is deleted.
    ///
    /// Note: This method takes the engine as a parameter for registry operations
    /// but cannot actually delete the index data since the indexes own their engines.
    pub fn drop_indexes_for_collection(
        &self,
        engine: &E,
        collection: &CollectionName,
    ) -> Result<Vec<String>, VectorError> {
        let collection_str = collection.as_str();
        let mut dropped = Vec::new();

        // Get all indexes for this collection from the registry
        let entries = {
            let tx = engine.begin_read()?;
            HnswRegistry::list_for_collection(&tx, collection_str)?
        };

        // Drop each index from registry and cache
        for entry in entries {
            // Remove from cache
            {
                let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
                indexes.remove(&entry.name);
            }

            // Remove from registry
            {
                let mut tx = engine.begin_write()?;
                HnswRegistry::drop(&mut tx, &entry.name)?;
                // Note: Index data cleanup should be done by the index itself when dropped
                tx.commit()?;
            }

            dropped.push(entry.name);
        }

        Ok(dropped)
    }

    /// Drop a specific HNSW index by name.
    pub fn drop_index(&self, engine: &E, index_name: &str) -> Result<bool, VectorError> {
        // Remove from cache
        {
            let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
            indexes.remove(index_name);
        }

        // Remove from registry
        let mut tx = engine.begin_write()?;
        let existed = HnswRegistry::drop(&mut tx, index_name)?;
        if existed {
            // Clear the index data from storage
            super::persistence::clear_index_tx(
                &mut tx,
                &super::persistence::table_name(index_name),
            )?;
        }
        tx.commit()?;

        Ok(existed)
    }

    // ========================================================================
    // Index Updates
    // ========================================================================

    /// Update HNSW indexes when a point is inserted or updated.
    ///
    /// This inserts/updates the point's vectors in all relevant HNSW indexes.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection name
    /// * `point_id` - The point ID
    /// * `vectors` - Map of vector names to their values
    pub fn on_point_upsert(
        &self,
        collection: &CollectionName,
        point_id: PointId,
        vectors: &HashMap<String, NamedVector>,
    ) -> Result<(), VectorError> {
        let collection_str = collection.as_str();

        for (vector_name, vector) in vectors {
            // Only handle dense vectors for HNSW
            if let NamedVector::Dense(data) = vector {
                // Check if there's an index for this vector in the cache
                let index_name = HnswRegistry::index_name_for_vector(collection_str, vector_name);

                let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
                if let Some(index) = indexes.get(&index_name) {
                    // Convert to embedding and insert
                    let embedding = Embedding::new(data.clone())?;
                    // Use point_id as entity_id (they're both u64 wrappers)
                    let entity_id = manifoldb_core::EntityId::new(point_id.as_u64());

                    let mut index_guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                    index_guard.insert(entity_id, &embedding)?;
                }
            }
        }

        Ok(())
    }

    /// Update HNSW indexes when a specific vector is updated.
    pub fn on_vector_update(
        &self,
        collection: &CollectionName,
        point_id: PointId,
        vector_name: &str,
        vector: &NamedVector,
    ) -> Result<(), VectorError> {
        // Only handle dense vectors for HNSW
        if let NamedVector::Dense(data) = vector {
            let collection_str = collection.as_str();
            let index_name = HnswRegistry::index_name_for_vector(collection_str, vector_name);

            let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
            if let Some(index) = indexes.get(&index_name) {
                let embedding = Embedding::new(data.clone())?;
                let entity_id = manifoldb_core::EntityId::new(point_id.as_u64());

                let mut index_guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                index_guard.insert(entity_id, &embedding)?;
            }
        }

        Ok(())
    }

    /// Update HNSW indexes when a point is deleted.
    ///
    /// This removes the point from all HNSW indexes in the collection that are cached.
    pub fn on_point_delete(
        &self,
        collection: &CollectionName,
        point_id: PointId,
    ) -> Result<(), VectorError> {
        let collection_str = collection.as_str();
        let entity_id = manifoldb_core::EntityId::new(point_id.as_u64());

        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;

        // Delete from each cached index for this collection
        for (name, index) in indexes.iter() {
            if name.starts_with(&format!("{}_", collection_str)) && name.ends_with("_hnsw") {
                let mut index_guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
                let _ = index_guard.delete(entity_id)?;
            }
        }

        Ok(())
    }

    /// Update HNSW indexes when a specific vector is deleted from a point.
    pub fn on_vector_delete(
        &self,
        collection: &CollectionName,
        point_id: PointId,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let collection_str = collection.as_str();
        let index_name = HnswRegistry::index_name_for_vector(collection_str, vector_name);
        let entity_id = manifoldb_core::EntityId::new(point_id.as_u64());

        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        if let Some(index) = indexes.get(&index_name) {
            let mut index_guard = index.write().map_err(|_| VectorError::LockPoisoned)?;
            return index_guard.delete(entity_id);
        }

        Ok(false)
    }

    // ========================================================================
    // Index Access
    // ========================================================================

    /// Get an HNSW index for a specific collection and vector name from the cache.
    ///
    /// Returns `None` if no index is loaded for this vector.
    pub fn get_index(
        &self,
        collection: &str,
        vector_name: &str,
    ) -> Result<Option<Arc<RwLock<HnswIndex<E>>>>, VectorError> {
        let index_name = HnswRegistry::index_name_for_vector(collection, vector_name);
        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(indexes.get(&index_name).cloned())
    }

    /// Get an HNSW index by name from the cache.
    pub fn get_index_by_name(
        &self,
        index_name: &str,
    ) -> Result<Option<Arc<RwLock<HnswIndex<E>>>>, VectorError> {
        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(indexes.get(index_name).cloned())
    }

    /// Load an existing HNSW index from storage into the cache.
    ///
    /// This is used after a restart to reload indexes that were registered.
    pub fn load_index(&self, engine: E, index_name: &str) -> Result<(), VectorError> {
        // Check if already loaded
        {
            let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
            if indexes.contains_key(index_name) {
                return Ok(());
            }
        }

        // Load the index
        let hnsw = HnswIndex::open(engine, index_name)?;

        // Cache it
        {
            let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
            indexes.insert(index_name.to_string(), Arc::new(RwLock::new(hnsw)));
        }

        Ok(())
    }

    /// Check if an index exists in the registry.
    pub fn has_index(
        &self,
        engine: &E,
        collection: &str,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let tx = engine.begin_read()?;
        HnswRegistry::exists_for_named_vector(&tx, collection, vector_name)
    }

    /// Check if an index is loaded in the cache.
    pub fn is_index_loaded(
        &self,
        collection: &str,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let index_name = HnswRegistry::index_name_for_vector(collection, vector_name);
        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(indexes.contains_key(&index_name))
    }

    /// List all index entries for a collection from the registry.
    pub fn list_indexes(
        &self,
        engine: &E,
        collection: &str,
    ) -> Result<Vec<HnswIndexEntry>, VectorError> {
        let tx = engine.begin_read()?;
        HnswRegistry::list_for_collection(&tx, collection)
    }

    // ========================================================================
    // Index Recovery
    // ========================================================================

    /// Rebuild an HNSW index from the point store data.
    ///
    /// This is used for crash recovery or when an index needs to be rebuilt.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection name
    /// * `vector_name` - The vector name
    /// * `points` - Iterator of (point_id, embedding) pairs
    pub fn rebuild_index<I>(
        &self,
        collection: &str,
        vector_name: &str,
        points: I,
    ) -> Result<usize, VectorError>
    where
        I: IntoIterator<Item = (PointId, Vec<f32>)>,
    {
        let index_name = HnswRegistry::index_name_for_vector(collection, vector_name);

        // Get the index from cache
        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        let index = indexes.get(&index_name).ok_or_else(|| {
            VectorError::SpaceNotFound(format!("index '{}' not found in cache", index_name))
        })?;

        let mut index_guard = index.write().map_err(|_| VectorError::LockPoisoned)?;

        // Collect points for batch insert
        let embeddings: Vec<(manifoldb_core::EntityId, Embedding)> = points
            .into_iter()
            .map(|(pid, data)| {
                let entity_id = manifoldb_core::EntityId::new(pid.as_u64());
                let embedding = Embedding::new(data)?;
                Ok((entity_id, embedding))
            })
            .collect::<Result<Vec<_>, VectorError>>()?;

        let count = embeddings.len();

        // Batch insert all embeddings
        let refs: Vec<(manifoldb_core::EntityId, &Embedding)> =
            embeddings.iter().map(|(id, emb)| (*id, emb)).collect();

        index_guard.insert_batch(&refs)?;
        index_guard.flush()?;

        Ok(count)
    }

    /// Load all registered indexes for a collection on startup.
    ///
    /// This is called during database recovery to restore HNSW indexes
    /// from persisted storage. For each registered index, it attempts to
    /// open the index from storage and load it into the cache.
    ///
    /// # Arguments
    ///
    /// * `engine_factory` - A closure that creates a new engine instance for each index
    /// * `collection` - The collection name to load indexes for
    ///
    /// # Returns
    ///
    /// A vector of (index_name, result) pairs indicating success or failure for each index.
    pub fn load_indexes_for_collection<F>(
        &self,
        engine: &E,
        engine_factory: F,
        collection: &str,
    ) -> Result<Vec<(String, Result<(), VectorError>)>, VectorError>
    where
        F: Fn() -> Result<E, VectorError>,
    {
        // Get all registered indexes for this collection
        let entries = {
            let tx = engine.begin_read()?;
            HnswRegistry::list_for_collection(&tx, collection)?
        };

        let mut results = Vec::with_capacity(entries.len());

        for entry in entries {
            let index_name = entry.name.clone();

            // Try to load each index
            let load_result = (|| -> Result<(), VectorError> {
                // Check if already loaded
                {
                    let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
                    if indexes.contains_key(&index_name) {
                        return Ok(());
                    }
                }

                // Create a new engine instance for this index
                let new_engine = engine_factory()?;

                // Try to open the index
                let hnsw = HnswIndex::open(new_engine, &index_name)?;

                // Cache it
                {
                    let mut indexes =
                        self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
                    indexes.insert(index_name.clone(), Arc::new(RwLock::new(hnsw)));
                }

                Ok(())
            })();

            results.push((index_name, load_result));
        }

        Ok(results)
    }

    /// Verify and repair an index if needed.
    ///
    /// This checks if the index data is consistent with the stored point data.
    /// If inconsistencies are found, the index can be rebuilt.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection name
    /// * `vector_name` - The vector name
    /// * `expected_point_count` - The expected number of points in the index
    ///
    /// # Returns
    ///
    /// A `RecoveryStatus` indicating whether the index is valid or needs repair.
    pub fn verify_index(
        &self,
        collection: &str,
        vector_name: &str,
        expected_point_count: usize,
    ) -> Result<RecoveryStatus, VectorError> {
        let index_name = HnswRegistry::index_name_for_vector(collection, vector_name);

        let indexes = self.indexes.read().map_err(|_| VectorError::LockPoisoned)?;
        let index = match indexes.get(&index_name) {
            Some(idx) => idx,
            None => return Ok(RecoveryStatus::NotLoaded),
        };

        let guard = index.read().map_err(|_| VectorError::LockPoisoned)?;
        let actual_count = guard.len()?;

        if actual_count == expected_point_count {
            Ok(RecoveryStatus::Valid)
        } else {
            Ok(RecoveryStatus::NeedsRebuild {
                expected: expected_point_count,
                actual: actual_count,
            })
        }
    }

    /// Clear and rebuild an index from scratch.
    ///
    /// This creates a fresh index with the same configuration and inserts
    /// all provided points. Used when verification detects inconsistencies.
    ///
    /// # Arguments
    ///
    /// * `engine` - The storage engine
    /// * `collection` - The collection name
    /// * `vector_name` - The vector name
    /// * `points` - Iterator of (point_id, embedding) pairs
    ///
    /// # Returns
    ///
    /// The number of points inserted into the rebuilt index.
    pub fn rebuild_index_from_scratch<I>(
        &self,
        engine: E,
        collection: &str,
        vector_name: &str,
        points: I,
    ) -> Result<usize, VectorError>
    where
        I: IntoIterator<Item = (PointId, Vec<f32>)>,
    {
        let index_name = HnswRegistry::index_name_for_vector(collection, vector_name);

        // Get the entry from registry to get configuration
        let entry = {
            let tx = engine.begin_read()?;
            HnswRegistry::get(&tx, &index_name)?.ok_or_else(|| {
                VectorError::SpaceNotFound(format!("index '{}' not in registry", index_name))
            })?
        };

        // Remove old index from cache if present
        {
            let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
            indexes.remove(&index_name);
        }

        // Clear old index data
        {
            let mut tx = engine.begin_write()?;
            super::persistence::clear_index_tx(
                &mut tx,
                &super::persistence::table_name(&index_name),
            )?;
            tx.commit()?;
        }

        // Create a fresh index with the same config
        let config = entry.config();
        let distance_metric = entry.distance_metric.into();
        let hnsw = HnswIndex::new(engine, &index_name, entry.dimension, distance_metric, config)?;

        // Collect and insert points
        let embeddings: Vec<(manifoldb_core::EntityId, Embedding)> = points
            .into_iter()
            .map(|(pid, data)| {
                let entity_id = manifoldb_core::EntityId::new(pid.as_u64());
                let embedding = Embedding::new(data)?;
                Ok((entity_id, embedding))
            })
            .collect::<Result<Vec<_>, VectorError>>()?;

        let count = embeddings.len();

        if embeddings.is_empty() {
            // Cache empty index
            {
                let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
                indexes.insert(index_name, Arc::new(RwLock::new(hnsw)));
            }
        } else {
            let refs: Vec<(manifoldb_core::EntityId, &Embedding)> =
                embeddings.iter().map(|(id, emb)| (*id, emb)).collect();
            let mut hnsw_guard = hnsw;
            hnsw_guard.insert_batch(&refs)?;
            hnsw_guard.flush()?;

            // Cache the rebuilt index
            {
                let mut indexes = self.indexes.write().map_err(|_| VectorError::LockPoisoned)?;
                indexes.insert(index_name, Arc::new(RwLock::new(hnsw_guard)));
            }
        }

        Ok(count)
    }
}

/// Status of index recovery/verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStatus {
    /// Index is valid and consistent.
    Valid,
    /// Index is not loaded in the cache.
    NotLoaded,
    /// Index needs to be rebuilt due to inconsistencies.
    NeedsRebuild {
        /// Expected number of points.
        expected: usize,
        /// Actual number of points in the index.
        actual: usize,
    },
}

impl<E: StorageEngine> Default for HnswIndexManager<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_manager() -> HnswIndexManager<RedbEngine> {
        HnswIndexManager::new()
    }

    #[test]
    fn test_create_index_for_vector() {
        let manager = create_test_manager();
        let engine = RedbEngine::in_memory().unwrap();
        let collection = CollectionName::new("documents").unwrap();

        let index_name = manager
            .create_index_for_vector(
                engine,
                &collection,
                "embedding",
                384,
                DistanceMetric::Cosine,
                &HnswConfig::default(),
            )
            .unwrap();

        assert_eq!(index_name, "documents_embedding_hnsw");

        // Check it's in the cache
        assert!(manager.is_index_loaded("documents", "embedding").unwrap());
    }

    #[test]
    fn test_point_upsert_and_delete() {
        let manager = create_test_manager();
        let engine = RedbEngine::in_memory().unwrap();
        let collection = CollectionName::new("documents").unwrap();

        // Create an index
        manager
            .create_index_for_vector(
                engine,
                &collection,
                "embedding",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        // Insert a point
        let point_id = PointId::new(1);
        let mut vectors = HashMap::new();
        vectors.insert("embedding".to_string(), NamedVector::Dense(vec![1.0, 2.0, 3.0, 4.0]));

        manager.on_point_upsert(&collection, point_id, &vectors).unwrap();

        // Verify it's in the index
        let index = manager.get_index("documents", "embedding").unwrap().unwrap();
        let guard = index.read().unwrap();
        assert!(guard.contains(manifoldb_core::EntityId::new(1)).unwrap());
        drop(guard);

        // Delete the point
        manager.on_point_delete(&collection, point_id).unwrap();

        // Verify it's gone
        let guard = index.read().unwrap();
        assert!(!guard.contains(manifoldb_core::EntityId::new(1)).unwrap());
    }

    #[test]
    fn test_vector_update() {
        let manager = create_test_manager();
        let engine = RedbEngine::in_memory().unwrap();
        let collection = CollectionName::new("documents").unwrap();

        manager
            .create_index_for_vector(
                engine,
                &collection,
                "embedding",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        let point_id = PointId::new(1);

        // Insert initial vector
        let vector = NamedVector::Dense(vec![1.0, 2.0, 3.0, 4.0]);
        manager.on_vector_update(&collection, point_id, "embedding", &vector).unwrap();

        // Update to new vector
        let new_vector = NamedVector::Dense(vec![5.0, 6.0, 7.0, 8.0]);
        manager.on_vector_update(&collection, point_id, "embedding", &new_vector).unwrap();

        // Point should still be in index (same entity, updated embedding)
        let index = manager.get_index("documents", "embedding").unwrap().unwrap();
        let guard = index.read().unwrap();
        assert!(guard.contains(manifoldb_core::EntityId::new(1)).unwrap());
        assert_eq!(guard.len().unwrap(), 1);
    }

    #[test]
    fn test_sparse_vector_ignored() {
        let manager = create_test_manager();
        let engine = RedbEngine::in_memory().unwrap();
        let collection = CollectionName::new("documents").unwrap();

        // Create a dense index
        manager
            .create_index_for_vector(
                engine,
                &collection,
                "embedding",
                4,
                DistanceMetric::Euclidean,
                &HnswConfig::default(),
            )
            .unwrap();

        // Try to upsert with sparse vector - should be silently ignored
        let point_id = PointId::new(1);
        let mut vectors = HashMap::new();
        vectors.insert("sparse_vec".to_string(), NamedVector::Sparse(vec![(0, 1.0), (5, 0.5)]));

        // This should succeed (sparse vectors are ignored for HNSW)
        manager.on_point_upsert(&collection, point_id, &vectors).unwrap();
    }

    #[test]
    fn test_get_index() {
        let manager = create_test_manager();
        let engine = RedbEngine::in_memory().unwrap();
        let collection = CollectionName::new("test").unwrap();

        // No index yet
        assert!(manager.get_index("test", "v1").unwrap().is_none());

        // Create index
        manager
            .create_index_for_vector(
                engine,
                &collection,
                "v1",
                64,
                DistanceMetric::Cosine,
                &HnswConfig::default(),
            )
            .unwrap();

        // Now it should exist
        let index = manager.get_index("test", "v1").unwrap();
        assert!(index.is_some());
    }
}
