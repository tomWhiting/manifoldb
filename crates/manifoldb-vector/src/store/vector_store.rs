//! Vector store implementation.

use std::ops::Bound;

use manifoldb_core::EntityId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{
    decode_embedding_entity_id, encode_embedding_key, encode_embedding_prefix,
    encode_embedding_space_key,
};
use crate::error::VectorError;
use crate::types::{Embedding, EmbeddingName, EmbeddingSpace};

/// Table name for embedding space metadata.
const TABLE_EMBEDDING_SPACES: &str = "vector_spaces";

/// Table name for entity embeddings.
const TABLE_EMBEDDINGS: &str = "vector_embeddings";

/// A store for vector embeddings.
///
/// `VectorStore` provides CRUD operations for embeddings organized into named
/// embedding spaces. Each space defines a dimension and distance metric, and
/// the store validates that all embeddings match their space's dimension.
pub struct VectorStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> VectorStore<E> {
    /// Create a new vector store with the given storage engine.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Create a new embedding space.
    ///
    /// # Errors
    ///
    /// Returns an error if the space already exists or if the storage operation fails.
    pub fn create_space(&self, space: &EmbeddingSpace) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let key = encode_embedding_space_key(space.name());

        // Check if space already exists
        if tx.get(TABLE_EMBEDDING_SPACES, &key)?.is_some() {
            return Err(VectorError::InvalidName(format!(
                "embedding space '{}' already exists",
                space.name()
            )));
        }

        // Store the space metadata
        tx.put(TABLE_EMBEDDING_SPACES, &key, &space.to_bytes()?)?;
        tx.commit()?;

        Ok(())
    }

    /// Get an embedding space by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn get_space(&self, name: &EmbeddingName) -> Result<EmbeddingSpace, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_embedding_space_key(name);

        let bytes = tx
            .get(TABLE_EMBEDDING_SPACES, &key)?
            .ok_or_else(|| VectorError::SpaceNotFound(name.to_string()))?;

        EmbeddingSpace::from_bytes(&bytes)
    }

    /// Delete an embedding space and all its embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn delete_space(&self, name: &EmbeddingName) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let space_key = encode_embedding_space_key(name);

        // Check if space exists
        if tx.get(TABLE_EMBEDDING_SPACES, &space_key)?.is_none() {
            return Err(VectorError::SpaceNotFound(name.to_string()));
        }

        // Delete all embeddings in this space
        let prefix = encode_embedding_prefix(name);
        let prefix_end = next_prefix(&prefix);

        let mut keys_to_delete = Vec::new();
        {
            let cursor = tx.range(
                TABLE_EMBEDDINGS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )?;

            // Collect keys to delete
            let mut cursor = cursor;
            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key);
            }
        }

        for key in keys_to_delete {
            tx.delete(TABLE_EMBEDDINGS, &key)?;
        }

        // Delete the space metadata
        tx.delete(TABLE_EMBEDDING_SPACES, &space_key)?;

        tx.commit()?;
        Ok(())
    }

    /// List all embedding spaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_spaces(&self) -> Result<Vec<EmbeddingSpace>, VectorError> {
        let tx = self.engine.begin_read()?;

        // Scan all space metadata
        let prefix = vec![crate::encoding::PREFIX_EMBEDDING_SPACE];
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_EMBEDDING_SPACES,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut spaces = Vec::new();
        while let Some((_, value)) = cursor.next()? {
            spaces.push(EmbeddingSpace::from_bytes(&value)?);
        }

        Ok(spaces)
    }

    /// Store an embedding for an entity in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - The embedding dimension doesn't match the space dimension
    /// - The storage operation fails
    pub fn put(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
        embedding: &Embedding,
    ) -> Result<(), VectorError> {
        // Get space to validate dimension
        let space = self.get_space(space_name)?;

        if embedding.dimension() != space.dimension() {
            return Err(VectorError::DimensionMismatch {
                expected: space.dimension(),
                actual: embedding.dimension(),
            });
        }

        let mut tx = self.engine.begin_write()?;

        let key = encode_embedding_key(space_name, entity_id);
        tx.put(TABLE_EMBEDDINGS, &key, &embedding.to_bytes())?;

        tx.commit()?;
        Ok(())
    }

    /// Get an embedding for an entity from a space.
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
    ) -> Result<Embedding, VectorError> {
        // Check space exists
        let _ = self.get_space(space_name)?;

        let tx = self.engine.begin_read()?;

        let key = encode_embedding_key(space_name, entity_id);
        let bytes =
            tx.get(TABLE_EMBEDDINGS, &key)?.ok_or_else(|| VectorError::EmbeddingNotFound {
                entity_id: entity_id.as_u64(),
                space: space_name.to_string(),
            })?;

        Embedding::from_bytes(&bytes)
    }

    /// Delete an embedding for an entity from a space.
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

        let key = encode_embedding_key(space_name, entity_id);
        let existed = tx.delete(TABLE_EMBEDDINGS, &key)?;

        tx.commit()?;
        Ok(existed)
    }

    /// Check if an embedding exists for an entity in a space.
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
        let key = encode_embedding_key(space_name, entity_id);
        Ok(tx.get(TABLE_EMBEDDINGS, &key)?.is_some())
    }

    /// List all entity IDs that have embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_entities(&self, space_name: &EmbeddingName) -> Result<Vec<EntityId>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_EMBEDDINGS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut entities = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(entity_id) = decode_embedding_entity_id(&key) {
                entities.push(entity_id);
            }
        }

        Ok(entities)
    }

    /// Count the number of embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn count(&self, space_name: &EmbeddingName) -> Result<usize, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_EMBEDDINGS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get multiple embeddings at once.
    ///
    /// Returns a vector of `(EntityId, Option<Embedding>)` tuples.
    /// If an embedding doesn't exist for an entity, the option is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_many(
        &self,
        entity_ids: &[EntityId],
        space_name: &EmbeddingName,
    ) -> Result<Vec<(EntityId, Option<Embedding>)>, VectorError> {
        let tx = self.engine.begin_read()?;

        let mut results = Vec::with_capacity(entity_ids.len());

        for &entity_id in entity_ids {
            let key = encode_embedding_key(space_name, entity_id);
            let embedding = tx
                .get(TABLE_EMBEDDINGS, &key)?
                .map(|bytes| Embedding::from_bytes(&bytes))
                .transpose()?;

            results.push((entity_id, embedding));
        }

        Ok(results)
    }

    /// Store multiple embeddings at once.
    ///
    /// All embeddings must match the space's dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - Any embedding dimension doesn't match the space dimension
    /// - The storage operation fails
    pub fn put_many(
        &self,
        embeddings: &[(EntityId, Embedding)],
        space_name: &EmbeddingName,
    ) -> Result<(), VectorError> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Get space to validate dimension
        let space = self.get_space(space_name)?;

        // Validate all dimensions first
        for (entity_id, embedding) in embeddings {
            if embedding.dimension() != space.dimension() {
                return Err(VectorError::DimensionMismatch {
                    expected: space.dimension(),
                    actual: embedding.dimension(),
                });
            }
            let _ = entity_id; // Silence warning
        }

        let mut tx = self.engine.begin_write()?;

        for (entity_id, embedding) in embeddings {
            let key = encode_embedding_key(space_name, *entity_id);
            tx.put(TABLE_EMBEDDINGS, &key, &embedding.to_bytes())?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Delete all embeddings for an entity across all spaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_entity(&self, entity_id: EntityId) -> Result<usize, VectorError> {
        // Get all spaces
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
///
/// Increments the last byte of the prefix, or extends with 0xFF if at max.
fn next_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut result = prefix.to_vec();

    // Find the last byte that can be incremented
    for byte in result.iter_mut().rev() {
        if *byte < 0xFF {
            *byte += 1;
            return result;
        }
    }

    // All bytes are 0xFF, extend with 0xFF
    result.push(0xFF);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use manifoldb_storage::backends::RedbEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_store() -> VectorStore<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        VectorStore::new(engine)
    }

    fn unique_space_name() -> EmbeddingName {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        EmbeddingName::new(format!("test_space_{}", count)).unwrap()
    }

    #[test]
    fn create_and_get_space() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);

        store.create_space(&space).unwrap();

        let retrieved = store.get_space(&name).unwrap();
        assert_eq!(retrieved.dimension(), 128);
        assert_eq!(retrieved.distance_metric(), DistanceMetric::Cosine);
    }

    #[test]
    fn create_duplicate_space_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);

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
        match result.unwrap_err() {
            VectorError::SpaceNotFound(n) => assert_eq!(n, "nonexistent"),
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn list_spaces() {
        let store = create_test_store();

        let name1 = unique_space_name();
        let name2 = unique_space_name();

        let space1 = EmbeddingSpace::new(name1.clone(), 128, DistanceMetric::Cosine);
        let space2 = EmbeddingSpace::new(name2.clone(), 256, DistanceMetric::Euclidean);

        store.create_space(&space1).unwrap();
        store.create_space(&space2).unwrap();

        let spaces = store.list_spaces().unwrap();
        assert!(spaces.len() >= 2);

        let names: Vec<_> = spaces.iter().map(|s| s.name().as_str()).collect();
        assert!(names.contains(&name1.as_str()));
        assert!(names.contains(&name2.as_str()));
    }

    #[test]
    fn delete_space() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);

        store.create_space(&space).unwrap();

        // Add an embedding
        let embedding = Embedding::new(vec![1.0; 128]).unwrap();
        store.put(EntityId::new(1), &name, &embedding).unwrap();

        // Delete the space
        store.delete_space(&name).unwrap();

        // Space should not exist
        assert!(store.get_space(&name).is_err());

        // Embedding should be gone too
        assert!(!store.exists(EntityId::new(1), &name).unwrap_or(true));
    }

    #[test]
    fn put_and_get_embedding() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        store.put(EntityId::new(42), &name, &embedding).unwrap();

        let retrieved = store.get(EntityId::new(42), &name).unwrap();
        assert_eq!(retrieved.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn put_wrong_dimension_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap(); // Wrong dimension

        let result = store.put(EntityId::new(1), &name, &embedding);
        assert!(result.is_err());
        match result.unwrap_err() {
            VectorError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 3);
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn get_nonexistent_embedding_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let result = store.get(EntityId::new(999), &name);
        assert!(result.is_err());
        match result.unwrap_err() {
            VectorError::EmbeddingNotFound { entity_id, space } => {
                assert_eq!(entity_id, 999);
                assert_eq!(space, name.as_str());
            }
            _ => panic!("unexpected error type"),
        }
    }

    #[test]
    fn delete_embedding() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
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
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        for i in 1..=5 {
            let embedding = Embedding::new(vec![i as f32; 3]).unwrap();
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
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        assert_eq!(store.count(&name).unwrap(), 0);

        for i in 1..=10 {
            let embedding = Embedding::new(vec![i as f32; 3]).unwrap();
            store.put(EntityId::new(i), &name, &embedding).unwrap();
        }

        assert_eq!(store.count(&name).unwrap(), 10);
    }

    #[test]
    fn get_many() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embedding = Embedding::new(vec![1.0, 2.0, 3.0]).unwrap();
        store.put(EntityId::new(1), &name, &embedding).unwrap();
        store.put(EntityId::new(3), &name, &embedding).unwrap();

        let results =
            store.get_many(&[EntityId::new(1), EntityId::new(2), EntityId::new(3)], &name).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some()); // Entity 1 exists
        assert!(results[1].1.is_none()); // Entity 2 doesn't exist
        assert!(results[2].1.is_some()); // Entity 3 exists
    }

    #[test]
    fn put_many() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 3, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embeddings: Vec<_> = (1..=5)
            .map(|i| (EntityId::new(i), Embedding::new(vec![i as f32; 3]).unwrap()))
            .collect();

        store.put_many(&embeddings, &name).unwrap();

        assert_eq!(store.count(&name).unwrap(), 5);

        for i in 1..=5 {
            let retrieved = store.get(EntityId::new(i), &name).unwrap();
            assert_eq!(retrieved.as_slice(), &[i as f32; 3]);
        }
    }

    #[test]
    fn put_many_wrong_dimension_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = EmbeddingSpace::new(name.clone(), 128, DistanceMetric::Cosine);
        store.create_space(&space).unwrap();

        let embeddings = vec![
            (EntityId::new(1), Embedding::new(vec![1.0; 128]).unwrap()),
            (EntityId::new(2), Embedding::new(vec![1.0, 2.0, 3.0]).unwrap()), // Wrong dimension
        ];

        let result = store.put_many(&embeddings, &name);
        assert!(result.is_err());
    }

    #[test]
    fn delete_entity_across_spaces() {
        let store = create_test_store();

        let name1 = unique_space_name();
        let name2 = unique_space_name();

        let space1 = EmbeddingSpace::new(name1.clone(), 3, DistanceMetric::Cosine);
        let space2 = EmbeddingSpace::new(name2.clone(), 5, DistanceMetric::Euclidean);

        store.create_space(&space1).unwrap();
        store.create_space(&space2).unwrap();

        let entity_id = EntityId::new(42);

        store.put(entity_id, &name1, &Embedding::new(vec![1.0; 3]).unwrap()).unwrap();
        store.put(entity_id, &name2, &Embedding::new(vec![2.0; 5]).unwrap()).unwrap();

        assert!(store.exists(entity_id, &name1).unwrap());
        assert!(store.exists(entity_id, &name2).unwrap());

        let deleted = store.delete_entity(entity_id).unwrap();
        assert_eq!(deleted, 2);

        assert!(!store.exists(entity_id, &name1).unwrap());
        assert!(!store.exists(entity_id, &name2).unwrap());
    }

    #[test]
    fn multiple_embeddings_per_entity() {
        let store = create_test_store();

        let text_space = unique_space_name();
        let image_space = unique_space_name();

        store
            .create_space(&EmbeddingSpace::new(text_space.clone(), 384, DistanceMetric::Cosine))
            .unwrap();
        store
            .create_space(&EmbeddingSpace::new(
                image_space.clone(),
                512,
                DistanceMetric::DotProduct,
            ))
            .unwrap();

        let entity_id = EntityId::new(1);

        let text_embedding = Embedding::new(vec![0.1; 384]).unwrap();
        let image_embedding = Embedding::new(vec![0.2; 512]).unwrap();

        store.put(entity_id, &text_space, &text_embedding).unwrap();
        store.put(entity_id, &image_space, &image_embedding).unwrap();

        let retrieved_text = store.get(entity_id, &text_space).unwrap();
        let retrieved_image = store.get(entity_id, &image_space).unwrap();

        assert_eq!(retrieved_text.dimension(), 384);
        assert_eq!(retrieved_image.dimension(), 512);
    }

    #[test]
    fn next_prefix_increments_correctly() {
        assert_eq!(next_prefix(&[0x00]), vec![0x01]);
        assert_eq!(next_prefix(&[0x10, 0x00]), vec![0x10, 0x01]);
        assert_eq!(next_prefix(&[0x10, 0xFF]), vec![0x11, 0xFF]);
        assert_eq!(next_prefix(&[0xFF]), vec![0xFF, 0xFF]);
        assert_eq!(next_prefix(&[0xFF, 0xFF]), vec![0xFF, 0xFF, 0xFF]);
    }
}
