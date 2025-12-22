//! Sparse vector store implementation.

use std::ops::Bound;

use manifoldb_core::EntityId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{
    decode_sparse_embedding_entity_id, encode_sparse_embedding_key, encode_sparse_embedding_prefix,
    encode_sparse_embedding_space_key, PREFIX_SPARSE_EMBEDDING_SPACE,
};
use crate::error::VectorError;
use crate::types::{EmbeddingName, SparseEmbedding, SparseEmbeddingSpace};

/// Table name for sparse embedding space metadata.
const TABLE_SPARSE_EMBEDDING_SPACES: &str = "sparse_vector_spaces";

/// Table name for sparse entity embeddings.
const TABLE_SPARSE_EMBEDDINGS: &str = "sparse_vector_embeddings";

/// A store for sparse vector embeddings.
///
/// `SparseVectorStore` provides CRUD operations for sparse embeddings organized into
/// named embedding spaces. Sparse embeddings only store non-zero values, making them
/// efficient for high-dimensional vectors with few active elements.
pub struct SparseVectorStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> SparseVectorStore<E> {
    /// Create a new sparse vector store with the given storage engine.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Create a new sparse embedding space.
    ///
    /// # Errors
    ///
    /// Returns an error if the space already exists or if the storage operation fails.
    pub fn create_space(&self, space: &SparseEmbeddingSpace) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let key = encode_sparse_embedding_space_key(space.name());

        // Check if space already exists
        if tx.get(TABLE_SPARSE_EMBEDDING_SPACES, &key)?.is_some() {
            return Err(VectorError::InvalidName(format!(
                "sparse embedding space '{}' already exists",
                space.name()
            )));
        }

        // Store the space metadata
        tx.put(TABLE_SPARSE_EMBEDDING_SPACES, &key, &space.to_bytes()?)?;
        tx.commit()?;

        Ok(())
    }

    /// Get a sparse embedding space by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn get_space(&self, name: &EmbeddingName) -> Result<SparseEmbeddingSpace, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_sparse_embedding_space_key(name);

        let bytes = tx
            .get(TABLE_SPARSE_EMBEDDING_SPACES, &key)?
            .ok_or_else(|| VectorError::SpaceNotFound(name.to_string()))?;

        SparseEmbeddingSpace::from_bytes(&bytes)
    }

    /// Delete a sparse embedding space and all its embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn delete_space(&self, name: &EmbeddingName) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let space_key = encode_sparse_embedding_space_key(name);

        // Check if space exists
        if tx.get(TABLE_SPARSE_EMBEDDING_SPACES, &space_key)?.is_none() {
            return Err(VectorError::SpaceNotFound(name.to_string()));
        }

        // Delete all embeddings in this space
        let prefix = encode_sparse_embedding_prefix(name);
        let prefix_end = next_prefix(&prefix);

        let mut keys_to_delete = Vec::new();
        {
            let cursor = tx.range(
                TABLE_SPARSE_EMBEDDINGS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )?;

            let mut cursor = cursor;
            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key);
            }
        }

        for key in keys_to_delete {
            tx.delete(TABLE_SPARSE_EMBEDDINGS, &key)?;
        }

        // Delete the space metadata
        tx.delete(TABLE_SPARSE_EMBEDDING_SPACES, &space_key)?;

        tx.commit()?;
        Ok(())
    }

    /// List all sparse embedding spaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_spaces(&self) -> Result<Vec<SparseEmbeddingSpace>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = vec![PREFIX_SPARSE_EMBEDDING_SPACE];
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_SPARSE_EMBEDDING_SPACES,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut spaces = Vec::new();
        while let Some((_, value)) = cursor.next()? {
            spaces.push(SparseEmbeddingSpace::from_bytes(&value)?);
        }

        Ok(spaces)
    }

    /// Store a sparse embedding for an entity in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - Any index exceeds the space's max dimension
    /// - The storage operation fails
    pub fn put(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
        embedding: &SparseEmbedding,
    ) -> Result<(), VectorError> {
        // Get space to validate max dimension
        let space = self.get_space(space_name)?;

        // Validate indices are within bounds
        for &(idx, _) in embedding.as_pairs() {
            if idx >= space.max_dimension() {
                return Err(VectorError::Encoding(format!(
                    "sparse vector index {} exceeds max dimension {}",
                    idx,
                    space.max_dimension()
                )));
            }
        }

        let mut tx = self.engine.begin_write()?;

        let key = encode_sparse_embedding_key(space_name, entity_id);
        tx.put(TABLE_SPARSE_EMBEDDINGS, &key, &embedding.to_bytes())?;

        tx.commit()?;
        Ok(())
    }

    /// Get a sparse embedding for an entity from a space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - The embedding doesn't exist for this entity
    /// - The storage operation fails
    pub fn get(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
    ) -> Result<SparseEmbedding, VectorError> {
        // Check space exists
        let _ = self.get_space(space_name)?;

        let tx = self.engine.begin_read()?;

        let key = encode_sparse_embedding_key(space_name, entity_id);
        let bytes = tx.get(TABLE_SPARSE_EMBEDDINGS, &key)?.ok_or_else(|| {
            VectorError::EmbeddingNotFound {
                entity_id: entity_id.as_u64(),
                space: space_name.to_string(),
            }
        })?;

        SparseEmbedding::from_bytes(&bytes)
    }

    /// Delete a sparse embedding for an entity from a space.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the embedding was deleted, `Ok(false)` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;

        let key = encode_sparse_embedding_key(space_name, entity_id);
        let existed = tx.delete(TABLE_SPARSE_EMBEDDINGS, &key)?;

        tx.commit()?;
        Ok(existed)
    }

    /// Check if a sparse embedding exists for an entity in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn exists(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
    ) -> Result<bool, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_sparse_embedding_key(space_name, entity_id);
        Ok(tx.get(TABLE_SPARSE_EMBEDDINGS, &key)?.is_some())
    }

    /// List all entity IDs that have sparse embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_entities(&self, space_name: &EmbeddingName) -> Result<Vec<EntityId>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_sparse_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_SPARSE_EMBEDDINGS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut entities = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(entity_id) = decode_sparse_embedding_entity_id(&key) {
                entities.push(entity_id);
            }
        }

        Ok(entities)
    }

    /// Count the number of sparse embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn count(&self, space_name: &EmbeddingName) -> Result<usize, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_sparse_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_SPARSE_EMBEDDINGS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get multiple sparse embeddings at once.
    ///
    /// Returns a vector of `(EntityId, Option<SparseEmbedding>)` tuples.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_many(
        &self,
        entity_ids: &[EntityId],
        space_name: &EmbeddingName,
    ) -> Result<Vec<(EntityId, Option<SparseEmbedding>)>, VectorError> {
        let tx = self.engine.begin_read()?;

        let mut results = Vec::with_capacity(entity_ids.len());

        for &entity_id in entity_ids {
            let key = encode_sparse_embedding_key(space_name, entity_id);
            let embedding = tx
                .get(TABLE_SPARSE_EMBEDDINGS, &key)?
                .map(|bytes| SparseEmbedding::from_bytes(&bytes))
                .transpose()?;

            results.push((entity_id, embedding));
        }

        Ok(results)
    }

    /// Store multiple sparse embeddings at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding space doesn't exist, any index is out of bounds,
    /// or the storage operation fails.
    pub fn put_many(
        &self,
        embeddings: &[(EntityId, SparseEmbedding)],
        space_name: &EmbeddingName,
    ) -> Result<(), VectorError> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Get space to validate dimensions
        let space = self.get_space(space_name)?;

        // Validate all indices first
        for (entity_id, embedding) in embeddings {
            for &(idx, _) in embedding.as_pairs() {
                if idx >= space.max_dimension() {
                    return Err(VectorError::Encoding(format!(
                        "sparse vector index {} exceeds max dimension {} for entity {}",
                        idx,
                        space.max_dimension(),
                        entity_id.as_u64()
                    )));
                }
            }
        }

        let mut tx = self.engine.begin_write()?;

        for (entity_id, embedding) in embeddings {
            let key = encode_sparse_embedding_key(space_name, *entity_id);
            tx.put(TABLE_SPARSE_EMBEDDINGS, &key, &embedding.to_bytes())?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Delete all sparse embeddings for an entity across all spaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_entity(&self, entity_id: EntityId) -> Result<usize, VectorError> {
        let spaces = self.list_spaces()?;

        let mut deleted = 0;
        for space in spaces {
            if self.delete(entity_id, space.name())? {
                deleted += 1;
            }
        }

        Ok(deleted)
    }
}

/// Calculate the next prefix for range scanning.
fn next_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut result = prefix.to_vec();

    for byte in result.iter_mut().rev() {
        if *byte < 0xFF {
            *byte += 1;
            return result;
        }
    }

    result.push(0xFF);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::sparse::SparseDistanceMetric;
    use manifoldb_storage::backends::RedbEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_store() -> SparseVectorStore<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        SparseVectorStore::new(engine)
    }

    fn unique_space_name() -> EmbeddingName {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        EmbeddingName::new(format!("sparse_test_space_{}", count)).unwrap()
    }

    #[test]
    fn create_and_get_space() {
        let store = create_test_store();
        let name = unique_space_name();
        let space =
            SparseEmbeddingSpace::new(name.clone(), 30522, SparseDistanceMetric::DotProduct);

        store.create_space(&space).unwrap();

        let retrieved = store.get_space(&name).unwrap();
        assert_eq!(retrieved.max_dimension(), 30522);
        assert_eq!(retrieved.distance_metric(), SparseDistanceMetric::DotProduct);
    }

    #[test]
    fn create_duplicate_space_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 10000, SparseDistanceMetric::Cosine);

        store.create_space(&space).unwrap();
        let result = store.create_space(&space);

        assert!(result.is_err());
    }

    #[test]
    fn get_nonexistent_space_fails() {
        let store = create_test_store();
        let name = EmbeddingName::new("nonexistent").unwrap();

        let result = store.get_space(&name);
        assert!(result.is_err());
    }

    #[test]
    fn put_and_get_embedding() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = SparseEmbedding::new(vec![(10, 0.5), (50, 0.3), (100, 0.2)]).unwrap();
        store.put(EntityId::new(42), &name, &embedding).unwrap();

        let retrieved = store.get(EntityId::new(42), &name).unwrap();
        assert_eq!(retrieved.nnz(), 3);
        assert!((retrieved.get(10) - 0.5).abs() < 1e-6);
        assert!((retrieved.get(50) - 0.3).abs() < 1e-6);
        assert!((retrieved.get(100) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn put_index_exceeds_dimension_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 100, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = SparseEmbedding::new(vec![(0, 0.5), (100, 0.3)]).unwrap(); // 100 >= max_dim 100

        let result = store.put(EntityId::new(1), &name, &embedding);
        assert!(result.is_err());
    }

    #[test]
    fn get_nonexistent_embedding_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let result = store.get(EntityId::new(999), &name);
        assert!(result.is_err());
    }

    #[test]
    fn delete_embedding() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = SparseEmbedding::new(vec![(10, 0.5)]).unwrap();
        store.put(EntityId::new(1), &name, &embedding).unwrap();

        assert!(store.exists(EntityId::new(1), &name).unwrap());
        assert!(store.delete(EntityId::new(1), &name).unwrap());
        assert!(!store.exists(EntityId::new(1), &name).unwrap());

        // Deleting again returns false
        assert!(!store.delete(EntityId::new(1), &name).unwrap());
    }

    #[test]
    fn list_entities() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        for i in 1..=5 {
            let embedding = SparseEmbedding::new(vec![(i as u32, i as f32)]).unwrap();
            store.put(EntityId::new(i), &name, &embedding).unwrap();
        }

        let entities = store.list_entities(&name).unwrap();
        assert_eq!(entities.len(), 5);

        let ids: Vec<u64> = entities.iter().map(|e| e.as_u64()).collect();
        for i in 1..=5 {
            assert!(ids.contains(&i));
        }
    }

    #[test]
    fn count_embeddings() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        assert_eq!(store.count(&name).unwrap(), 0);

        for i in 1..=10 {
            let embedding = SparseEmbedding::new(vec![(i as u32, 1.0)]).unwrap();
            store.put(EntityId::new(i), &name, &embedding).unwrap();
        }

        assert_eq!(store.count(&name).unwrap(), 10);
    }

    #[test]
    fn delete_space_removes_embeddings() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = SparseEmbedding::new(vec![(10, 0.5)]).unwrap();
        store.put(EntityId::new(1), &name, &embedding).unwrap();

        store.delete_space(&name).unwrap();

        assert!(store.get_space(&name).is_err());
    }

    #[test]
    fn put_many_and_get_many() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = SparseEmbeddingSpace::new(name.clone(), 1000, SparseDistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embeddings: Vec<_> = (1..=5)
            .map(|i| {
                (EntityId::new(i), SparseEmbedding::new(vec![(i as u32 * 10, i as f32)]).unwrap())
            })
            .collect();

        store.put_many(&embeddings, &name).unwrap();

        assert_eq!(store.count(&name).unwrap(), 5);

        let results = store
            .get_many(&[EntityId::new(1), EntityId::new(2), EntityId::new(999)], &name)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some());
        assert!(results[1].1.is_some());
        assert!(results[2].1.is_none());
    }
}
