//! Multi-vector store implementation for ColBERT-style embeddings.

use std::ops::Bound;

use manifoldb_core::EntityId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{
    decode_embedding_entity_id, encode_embedding_key, encode_embedding_prefix,
    PREFIX_MULTI_VECTOR_SPACE,
};
use crate::error::VectorError;
use crate::types::{EmbeddingName, MultiVectorEmbedding, MultiVectorEmbeddingSpace};

/// Table name for multi-vector embedding space metadata.
const TABLE_MULTI_VECTOR_SPACES: &str = "multi_vector_spaces";

/// Table name for entity multi-vector embeddings.
const TABLE_MULTI_VECTOR_EMBEDDINGS: &str = "multi_vector_embeddings";

/// A store for multi-vector embeddings (ColBERT-style).
///
/// `MultiVectorStore` provides CRUD operations for multi-vector embeddings
/// organized into named embedding spaces. Each multi-vector stores per-token
/// embeddings for late interaction models like ColBERT.
pub struct MultiVectorStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> MultiVectorStore<E> {
    /// Create a new multi-vector store with the given storage engine.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Create a new multi-vector embedding space.
    ///
    /// # Errors
    ///
    /// Returns an error if the space already exists or if the storage operation fails.
    pub fn create_space(&self, space: &MultiVectorEmbeddingSpace) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let key = encode_multi_vector_space_key(space.name());

        // Check if space already exists
        if tx.get(TABLE_MULTI_VECTOR_SPACES, &key)?.is_some() {
            return Err(VectorError::InvalidName(format!(
                "multi-vector embedding space '{}' already exists",
                space.name()
            )));
        }

        // Store the space metadata
        tx.put(TABLE_MULTI_VECTOR_SPACES, &key, &space.to_bytes()?)?;
        tx.commit()?;

        Ok(())
    }

    /// Get a multi-vector embedding space by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn get_space(
        &self,
        name: &EmbeddingName,
    ) -> Result<MultiVectorEmbeddingSpace, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_multi_vector_space_key(name);

        let bytes = tx
            .get(TABLE_MULTI_VECTOR_SPACES, &key)?
            .ok_or_else(|| VectorError::SpaceNotFound(name.to_string()))?;

        MultiVectorEmbeddingSpace::from_bytes(&bytes)
    }

    /// Delete a multi-vector embedding space and all its embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the space doesn't exist or if the storage operation fails.
    pub fn delete_space(&self, name: &EmbeddingName) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let space_key = encode_multi_vector_space_key(name);

        // Check if space exists
        if tx.get(TABLE_MULTI_VECTOR_SPACES, &space_key)?.is_none() {
            return Err(VectorError::SpaceNotFound(name.to_string()));
        }

        // Delete all embeddings in this space
        let prefix = encode_embedding_prefix(name);
        let prefix_end = next_prefix(&prefix);

        let mut keys_to_delete = Vec::new();
        {
            let cursor = tx.range(
                TABLE_MULTI_VECTOR_EMBEDDINGS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )?;

            let mut cursor = cursor;
            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key);
            }
        }

        for key in keys_to_delete {
            tx.delete(TABLE_MULTI_VECTOR_EMBEDDINGS, &key)?;
        }

        // Delete the space metadata
        tx.delete(TABLE_MULTI_VECTOR_SPACES, &space_key)?;

        tx.commit()?;
        Ok(())
    }

    /// List all multi-vector embedding spaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_spaces(&self) -> Result<Vec<MultiVectorEmbeddingSpace>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = vec![PREFIX_MULTI_VECTOR_SPACE];
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_MULTI_VECTOR_SPACES,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut spaces = Vec::new();
        while let Some((_, value)) = cursor.next()? {
            spaces.push(MultiVectorEmbeddingSpace::from_bytes(&value)?);
        }

        Ok(spaces)
    }

    /// Store a multi-vector embedding for an entity in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - The token embedding dimension doesn't match the space dimension
    /// - The storage operation fails
    pub fn put(
        &self,
        entity_id: EntityId,
        space_name: &EmbeddingName,
        embedding: &MultiVectorEmbedding,
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

        // Encode multi-vector: dimension (u32) + data bytes
        let mut bytes = Vec::new();
        let dim = embedding.dimension() as u32;
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.extend_from_slice(&embedding.to_bytes());

        tx.put(TABLE_MULTI_VECTOR_EMBEDDINGS, &key, &bytes)?;

        tx.commit()?;
        Ok(())
    }

    /// Get a multi-vector embedding for an entity from a space.
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
    ) -> Result<MultiVectorEmbedding, VectorError> {
        // Check space exists
        let _ = self.get_space(space_name)?;

        let tx = self.engine.begin_read()?;

        let key = encode_embedding_key(space_name, entity_id);
        let bytes = tx.get(TABLE_MULTI_VECTOR_EMBEDDINGS, &key)?.ok_or_else(|| {
            VectorError::EmbeddingNotFound {
                entity_id: entity_id.as_u64(),
                space: space_name.to_string(),
            }
        })?;

        // Decode: dimension (u32) + data bytes
        if bytes.len() < 4 {
            return Err(VectorError::Encoding("multi-vector data too short".to_string()));
        }

        let dim_bytes: [u8; 4] = bytes[..4]
            .try_into()
            .map_err(|_| VectorError::Encoding("failed to read dimension".to_string()))?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        MultiVectorEmbedding::from_bytes(&bytes[4..], dimension)
    }

    /// Delete a multi-vector embedding for an entity from a space.
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
        let existed = tx.delete(TABLE_MULTI_VECTOR_EMBEDDINGS, &key)?;

        tx.commit()?;
        Ok(existed)
    }

    /// Check if a multi-vector embedding exists for an entity in a space.
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
        Ok(tx.get(TABLE_MULTI_VECTOR_EMBEDDINGS, &key)?.is_some())
    }

    /// List all entity IDs that have multi-vector embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_entities(&self, space_name: &EmbeddingName) -> Result<Vec<EntityId>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_MULTI_VECTOR_EMBEDDINGS,
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

    /// Count the number of multi-vector embeddings in a space.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn count(&self, space_name: &EmbeddingName) -> Result<usize, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_embedding_prefix(space_name);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_MULTI_VECTOR_EMBEDDINGS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get multiple multi-vector embeddings at once.
    ///
    /// Returns a vector of `(EntityId, Option<MultiVectorEmbedding>)` tuples.
    /// If an embedding doesn't exist for an entity, the option is `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_many(
        &self,
        entity_ids: &[EntityId],
        space_name: &EmbeddingName,
    ) -> Result<Vec<(EntityId, Option<MultiVectorEmbedding>)>, VectorError> {
        // Get space to know dimension
        let space = self.get_space(space_name)?;
        let dimension = space.dimension();

        let tx = self.engine.begin_read()?;

        let mut results = Vec::with_capacity(entity_ids.len());

        for &entity_id in entity_ids {
            let key = encode_embedding_key(space_name, entity_id);
            let embedding = tx
                .get(TABLE_MULTI_VECTOR_EMBEDDINGS, &key)?
                .map(|bytes| {
                    if bytes.len() < 4 {
                        return Err(VectorError::Encoding(
                            "multi-vector data too short".to_string(),
                        ));
                    }
                    MultiVectorEmbedding::from_bytes(&bytes[4..], dimension)
                })
                .transpose()?;

            results.push((entity_id, embedding));
        }

        Ok(results)
    }

    /// Store multiple multi-vector embeddings at once.
    ///
    /// All embeddings must have token dimensions matching the space's dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding space doesn't exist
    /// - Any embedding dimension doesn't match the space dimension
    /// - The storage operation fails
    pub fn put_many(
        &self,
        embeddings: &[(EntityId, MultiVectorEmbedding)],
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
            let _ = entity_id;
        }

        let mut tx = self.engine.begin_write()?;

        for (entity_id, embedding) in embeddings {
            let key = encode_embedding_key(space_name, *entity_id);

            let mut bytes = Vec::new();
            let dim = embedding.dimension() as u32;
            bytes.extend_from_slice(&dim.to_le_bytes());
            bytes.extend_from_slice(&embedding.to_bytes());

            tx.put(TABLE_MULTI_VECTOR_EMBEDDINGS, &key, &bytes)?;
        }

        tx.commit()?;
        Ok(())
    }
}

/// Encode a multi-vector space key.
fn encode_multi_vector_space_key(name: &EmbeddingName) -> Vec<u8> {
    let name_bytes = name.as_str().as_bytes();
    let mut key = Vec::with_capacity(1 + name_bytes.len());
    key.push(PREFIX_MULTI_VECTOR_SPACE);
    key.extend_from_slice(name_bytes);
    key
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
    use crate::distance::DistanceMetric;
    use manifoldb_storage::backends::RedbEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_store() -> MultiVectorStore<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        MultiVectorStore::new(engine)
    }

    fn unique_space_name() -> EmbeddingName {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        EmbeddingName::new(format!("multi_test_space_{}", count)).unwrap()
    }

    #[test]
    fn create_and_get_space() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 128, DistanceMetric::DotProduct);

        store.create_space(&space).unwrap();

        let retrieved = store.get_space(&name).unwrap();
        assert_eq!(retrieved.dimension(), 128);
        assert_eq!(retrieved.distance_metric(), DistanceMetric::DotProduct);
    }

    #[test]
    fn create_duplicate_space_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 128, DistanceMetric::DotProduct);

        store.create_space(&space).unwrap();
        let result = store.create_space(&space);

        assert!(result.is_err());
    }

    #[test]
    fn put_and_get_multi_vector() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 3, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = MultiVectorEmbedding::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        store.put(EntityId::new(42), &name, &embedding).unwrap();

        let retrieved = store.get(EntityId::new(42), &name).unwrap();
        assert_eq!(retrieved.num_vectors(), 3);
        assert_eq!(retrieved.dimension(), 3);
        assert_eq!(retrieved.get_vector(0), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(retrieved.get_vector(1), Some([4.0, 5.0, 6.0].as_slice()));
        assert_eq!(retrieved.get_vector(2), Some([7.0, 8.0, 9.0].as_slice()));
    }

    #[test]
    fn put_wrong_dimension_fails() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 128, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = MultiVectorEmbedding::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();

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
    fn delete_multi_vector() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 3, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        let embedding = MultiVectorEmbedding::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        store.put(EntityId::new(1), &name, &embedding).unwrap();

        assert!(store.exists(EntityId::new(1), &name).unwrap());
        assert!(store.delete(EntityId::new(1), &name).unwrap());
        assert!(!store.exists(EntityId::new(1), &name).unwrap());
    }

    #[test]
    fn list_entities() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 3, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        for i in 1..=5 {
            let embedding =
                MultiVectorEmbedding::new(vec![vec![i as f32, i as f32, i as f32]]).unwrap();
            store.put(EntityId::new(i), &name, &embedding).unwrap();
        }

        let entities = store.list_entities(&name).unwrap();
        assert_eq!(entities.len(), 5);
    }

    #[test]
    fn count_embeddings() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 3, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        assert_eq!(store.count(&name).unwrap(), 0);

        for i in 1..=10 {
            let embedding =
                MultiVectorEmbedding::new(vec![vec![i as f32, i as f32, i as f32]]).unwrap();
            store.put(EntityId::new(i), &name, &embedding).unwrap();
        }

        assert_eq!(store.count(&name).unwrap(), 10);
    }

    #[test]
    fn variable_token_count() {
        let store = create_test_store();
        let name = unique_space_name();
        let space = MultiVectorEmbeddingSpace::new(name.clone(), 4, DistanceMetric::DotProduct);
        store.create_space(&space).unwrap();

        // Documents with different numbers of tokens
        let doc1 = MultiVectorEmbedding::new(vec![vec![1.0, 0.0, 0.0, 0.0]]).unwrap();

        let doc2 =
            MultiVectorEmbedding::new(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]])
                .unwrap();

        let doc3 = MultiVectorEmbedding::new(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();

        store.put(EntityId::new(1), &name, &doc1).unwrap();
        store.put(EntityId::new(2), &name, &doc2).unwrap();
        store.put(EntityId::new(3), &name, &doc3).unwrap();

        let retrieved1 = store.get(EntityId::new(1), &name).unwrap();
        let retrieved2 = store.get(EntityId::new(2), &name).unwrap();
        let retrieved3 = store.get(EntityId::new(3), &name).unwrap();

        assert_eq!(retrieved1.num_vectors(), 1);
        assert_eq!(retrieved2.num_vectors(), 2);
        assert_eq!(retrieved3.num_vectors(), 4);
    }
}
