//! Collection vector storage.
//!
//! This module provides dedicated vector storage separate from entity properties.
//! Vectors are stored in a dedicated table with keys that enable:
//!
//! - Fast lookup of a specific vector by (collection, entity, name)
//! - Efficient retrieval of all vectors for an entity
//! - Cascade deletion when an entity is removed
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::store::CollectionVectorStore;
//! use manifoldb_vector::types::VectorData;
//! use manifoldb_core::{CollectionId, EntityId};
//!
//! let store = CollectionVectorStore::new(engine);
//!
//! // Store a vector
//! let collection_id = CollectionId::new(1);
//! let entity_id = EntityId::new(42);
//! let data = VectorData::Dense(vec![0.1, 0.2, 0.3]);
//!
//! store.put_vector(collection_id, entity_id, "text_embedding", &data)?;
//!
//! // Retrieve it
//! let vector = store.get_vector(collection_id, entity_id, "text_embedding")?;
//! ```

use std::collections::HashMap;
use std::ops::Bound;

use manifoldb_core::{CollectionId, EntityId};
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{encode_collection_vector_key, encode_entity_vector_prefix, hash_name};
use crate::error::VectorError;
use crate::types::VectorData;

/// Table name for collection vectors.
const TABLE_COLLECTION_VECTORS: &str = "collection_vectors";

/// Version byte for vector storage format.
const VECTOR_FORMAT_VERSION: u8 = 1;

/// Vector type discriminants.
const VECTOR_TYPE_DENSE: u8 = 0;
const VECTOR_TYPE_SPARSE: u8 = 1;
const VECTOR_TYPE_MULTI: u8 = 2;
const VECTOR_TYPE_BINARY: u8 = 3;

/// Vector storage for collections.
///
/// Provides CRUD operations for vectors stored separately from entities.
/// This enables efficient vector access without loading entity data, and
/// supports multiple named vectors per entity.
pub struct CollectionVectorStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> CollectionVectorStore<E> {
    /// Create a new collection vector store.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Store a vector for an entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn put_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;
        self.put_vector_tx(&mut tx, collection_id, entity_id, vector_name, data)?;
        tx.commit()?;
        Ok(())
    }

    /// Store a vector within a transaction.
    ///
    /// Use this method when you need to store vectors as part of a larger
    /// atomic operation (e.g., upserting an entity with vectors).
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn put_vector_tx<T: Transaction>(
        &self,
        tx: &mut T,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError> {
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);
        let value = encode_vector_value(data, vector_name);
        tx.put(TABLE_COLLECTION_VECTORS, &key, &value)?;
        Ok(())
    }

    /// Get a vector for an entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<Option<VectorData>, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);

        match tx.get(TABLE_COLLECTION_VECTORS, &key)? {
            Some(bytes) => Ok(Some(decode_vector_value(&bytes)?.0)),
            None => Ok(None),
        }
    }

    /// Get all vectors for an entity.
    ///
    /// Returns a map of vector names to their data.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<HashMap<String, VectorData>, VectorError> {
        let tx = self.engine.begin_read()?;
        let prefix = encode_entity_vector_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_COLLECTION_VECTORS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut vectors = HashMap::new();
        while let Some((_, value)) = cursor.next()? {
            let (data, vector_name) = decode_vector_value(&value)?;
            vectors.insert(vector_name, data);
        }

        Ok(vectors)
    }

    /// Delete a specific vector.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);
        let deleted = tx.delete(TABLE_COLLECTION_VECTORS, &key)?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Delete all vectors for an entity.
    ///
    /// This is called when an entity is deleted to cascade-delete its vectors.
    ///
    /// # Returns
    ///
    /// Returns the number of vectors deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let count = self.delete_all_vectors_tx(&mut tx, collection_id, entity_id)?;
        tx.commit()?;
        Ok(count)
    }

    /// Delete all vectors for an entity within a transaction.
    ///
    /// Use this method when you need to delete vectors as part of a larger
    /// atomic operation (e.g., deleting an entity with its vectors).
    ///
    /// # Returns
    ///
    /// Returns the number of vectors deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_all_vectors_tx<T: Transaction>(
        &self,
        tx: &mut T,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError> {
        let prefix = encode_entity_vector_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        // Collect keys to delete
        let mut keys_to_delete = Vec::new();
        {
            let mut cursor = tx.range(
                TABLE_COLLECTION_VECTORS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )?;

            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key);
            }
        }

        // Delete collected keys
        let count = keys_to_delete.len();
        for key in keys_to_delete {
            tx.delete(TABLE_COLLECTION_VECTORS, &key)?;
        }

        Ok(count)
    }

    /// Store multiple vectors at once.
    ///
    /// This is more efficient than calling `put_vector` multiple times
    /// as it uses a single transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn put_vectors_batch(
        &self,
        collection_id: CollectionId,
        vectors: &[(EntityId, &str, &VectorData)],
    ) -> Result<(), VectorError> {
        if vectors.is_empty() {
            return Ok(());
        }

        let mut tx = self.engine.begin_write()?;
        for (entity_id, vector_name, data) in vectors {
            self.put_vector_tx(&mut tx, collection_id, *entity_id, vector_name, data)?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Check if a vector exists for an entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn exists(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);
        Ok(tx.get(TABLE_COLLECTION_VECTORS, &key)?.is_some())
    }

    /// Count the number of vectors for an entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn count_entity_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError> {
        let tx = self.engine.begin_read()?;
        let prefix = encode_entity_vector_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_COLLECTION_VECTORS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// List all entity IDs that have vectors with a specific name in a collection.
    ///
    /// Note: This is an expensive operation as it requires scanning all vectors
    /// in the collection. Use with caution on large collections.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_entities_with_vector(
        &self,
        collection_id: CollectionId,
        vector_name: &str,
    ) -> Result<Vec<EntityId>, VectorError> {
        use crate::encoding::{decode_collection_vector_key, encode_collection_vector_prefix};

        let tx = self.engine.begin_read()?;
        let prefix = encode_collection_vector_prefix(collection_id);
        let prefix_end = next_prefix(&prefix);

        let target_hash = hash_name(vector_name);

        let mut cursor = tx.range(
            TABLE_COLLECTION_VECTORS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut entities = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(decoded) = decode_collection_vector_key(&key) {
                if decoded.vector_name_hash == target_hash {
                    entities.push(decoded.entity_id);
                }
            }
        }

        Ok(entities)
    }
}

// Value encoding functions

/// Encode a vector value for storage.
///
/// Format:
/// - `[version: 1 byte]`
/// - `[type: 1 byte]` (0=dense, 1=sparse, 2=multi, 3=binary)
/// - `[timestamp: 8 bytes]` (seconds since Unix epoch)
/// - `[vector_name_len: 2 bytes]` (length of vector name)
/// - `[vector_name: variable]` (UTF-8 encoded name for reverse lookup)
/// - `[data_len: 4 bytes]`
/// - `[data: variable]`
fn encode_vector_value(data: &VectorData, vector_name: &str) -> Vec<u8> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let name_bytes = vector_name.as_bytes();
    let name_len = name_bytes.len().min(u16::MAX as usize);

    let mut bytes = Vec::new();
    bytes.push(VECTOR_FORMAT_VERSION);
    bytes.push(data.type_discriminant());
    bytes.extend_from_slice(&timestamp.to_be_bytes());
    bytes.extend_from_slice(&(name_len as u16).to_be_bytes());
    bytes.extend_from_slice(&name_bytes[..name_len]);

    match data {
        VectorData::Dense(v) => {
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            for &val in v {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        VectorData::Sparse(v) => {
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            for &(idx, val) in v {
                bytes.extend_from_slice(&idx.to_be_bytes());
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        VectorData::Multi(v) => {
            let num_vectors = v.len() as u32;
            let dim = v.first().map(|inner| inner.len() as u32).unwrap_or(0);
            bytes.extend_from_slice(&num_vectors.to_be_bytes());
            bytes.extend_from_slice(&dim.to_be_bytes());
            for inner in v {
                for &val in inner {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        VectorData::Binary(v) => {
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            bytes.extend_from_slice(v);
        }
    }

    bytes
}

/// Decode a vector value from storage.
///
/// Returns the vector data and the vector name.
fn decode_vector_value(bytes: &[u8]) -> Result<(VectorData, String), VectorError> {
    if bytes.len() < 12 {
        return Err(VectorError::Encoding("truncated vector value".to_string()));
    }

    let version = bytes[0];
    if version != VECTOR_FORMAT_VERSION {
        return Err(VectorError::Encoding(format!(
            "unsupported vector format version: {}",
            version
        )));
    }

    let vec_type = bytes[1];
    // Skip timestamp (bytes 2-9)
    let name_len = u16::from_be_bytes([bytes[10], bytes[11]]) as usize;

    if bytes.len() < 12 + name_len + 4 {
        return Err(VectorError::Encoding("truncated vector value (name)".to_string()));
    }

    let vector_name = String::from_utf8(bytes[12..12 + name_len].to_vec())
        .map_err(|e| VectorError::Encoding(format!("invalid vector name: {}", e)))?;

    let data_offset = 12 + name_len;
    let data_len = u32::from_be_bytes([
        bytes[data_offset],
        bytes[data_offset + 1],
        bytes[data_offset + 2],
        bytes[data_offset + 3],
    ]) as usize;

    let payload_offset = data_offset + 4;

    let data = match vec_type {
        VECTOR_TYPE_DENSE => {
            let expected_len = payload_offset + data_len * 4;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("dense vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(data_len);
            for i in 0..data_len {
                let offset = payload_offset + i * 4;
                let val = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                v.push(val);
            }
            VectorData::Dense(v)
        }
        VECTOR_TYPE_SPARSE => {
            let expected_len = payload_offset + data_len * 8;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("sparse vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(data_len);
            for i in 0..data_len {
                let offset = payload_offset + i * 8;
                let idx = u32::from_be_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                let val = f32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
                v.push((idx, val));
            }
            VectorData::Sparse(v)
        }
        VECTOR_TYPE_MULTI => {
            if bytes.len() < payload_offset + 4 {
                return Err(VectorError::Encoding("truncated multi-vector".to_string()));
            }
            let num_vectors = data_len;
            let dim = u32::from_be_bytes([
                bytes[payload_offset],
                bytes[payload_offset + 1],
                bytes[payload_offset + 2],
                bytes[payload_offset + 3],
            ]) as usize;
            let expected_len = payload_offset + 4 + num_vectors * dim * 4;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("multi-vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(num_vectors);
            for i in 0..num_vectors {
                let mut inner = Vec::with_capacity(dim);
                for j in 0..dim {
                    let offset = payload_offset + 4 + (i * dim + j) * 4;
                    let val = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    inner.push(val);
                }
                v.push(inner);
            }
            VectorData::Multi(v)
        }
        VECTOR_TYPE_BINARY => {
            let expected_len = payload_offset + data_len;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("binary vector length mismatch".to_string()));
            }
            VectorData::Binary(bytes[payload_offset..].to_vec())
        }
        _ => {
            return Err(VectorError::Encoding(format!("unknown vector type: {}", vec_type)));
        }
    };

    Ok((data, vector_name))
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
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_store() -> CollectionVectorStore<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        CollectionVectorStore::new(engine)
    }

    #[test]
    fn test_put_and_get_dense_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let data = VectorData::Dense(vec![1.0, 2.0, 3.0]);

        store.put_vector(collection_id, entity_id, "text", &data).unwrap();

        let retrieved = store.get_vector(collection_id, entity_id, "text").unwrap().unwrap();

        assert_eq!(retrieved.as_dense(), Some([1.0, 2.0, 3.0].as_slice()));
    }

    #[test]
    fn test_put_and_get_sparse_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let data = VectorData::Sparse(vec![(0, 1.0), (5, 2.0), (10, 3.0)]);

        store.put_vector(collection_id, entity_id, "sparse", &data).unwrap();

        let retrieved = store.get_vector(collection_id, entity_id, "sparse").unwrap().unwrap();

        assert_eq!(retrieved.as_sparse(), Some([(0, 1.0), (5, 2.0), (10, 3.0)].as_slice()));
    }

    #[test]
    fn test_put_and_get_multi_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let data = VectorData::Multi(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        store.put_vector(collection_id, entity_id, "multi", &data).unwrap();

        let retrieved = store.get_vector(collection_id, entity_id, "multi").unwrap().unwrap();

        assert!(retrieved.is_multi());
    }

    #[test]
    fn test_put_and_get_binary_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let data = VectorData::Binary(vec![0xFF, 0x00, 0xAB]);

        store.put_vector(collection_id, entity_id, "binary", &data).unwrap();

        let retrieved = store.get_vector(collection_id, entity_id, "binary").unwrap().unwrap();

        assert_eq!(retrieved.as_binary(), Some([0xFF, 0x00, 0xAB].as_slice()));
    }

    #[test]
    fn test_get_nonexistent_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        let result = store.get_vector(collection_id, entity_id, "nonexistent").unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_get_all_vectors() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        store
            .put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![1.0, 2.0]))
            .unwrap();
        store
            .put_vector(collection_id, entity_id, "image", &VectorData::Dense(vec![3.0, 4.0]))
            .unwrap();

        let vectors = store.get_all_vectors(collection_id, entity_id).unwrap();

        assert_eq!(vectors.len(), 2);
        assert!(vectors.contains_key("text"));
        assert!(vectors.contains_key("image"));
    }

    #[test]
    fn test_delete_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        store.put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![1.0])).unwrap();

        assert!(store.exists(collection_id, entity_id, "text").unwrap());

        let deleted = store.delete_vector(collection_id, entity_id, "text").unwrap();
        assert!(deleted);

        assert!(!store.exists(collection_id, entity_id, "text").unwrap());

        // Deleting again returns false
        let deleted = store.delete_vector(collection_id, entity_id, "text").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_delete_all_vectors() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        store.put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![1.0])).unwrap();
        store.put_vector(collection_id, entity_id, "image", &VectorData::Dense(vec![2.0])).unwrap();
        store
            .put_vector(collection_id, entity_id, "summary", &VectorData::Dense(vec![3.0]))
            .unwrap();

        let count = store.delete_all_vectors(collection_id, entity_id).unwrap();
        assert_eq!(count, 3);

        let vectors = store.get_all_vectors(collection_id, entity_id).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_put_vectors_batch() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);

        let text_data = VectorData::Dense(vec![1.0, 2.0]);
        let image_data = VectorData::Dense(vec![3.0, 4.0]);

        let vectors: Vec<(EntityId, &str, &VectorData)> = vec![
            (EntityId::new(1), "text", &text_data),
            (EntityId::new(1), "image", &image_data),
            (EntityId::new(2), "text", &text_data),
        ];

        store.put_vectors_batch(collection_id, &vectors).unwrap();

        assert!(store.exists(collection_id, EntityId::new(1), "text").unwrap());
        assert!(store.exists(collection_id, EntityId::new(1), "image").unwrap());
        assert!(store.exists(collection_id, EntityId::new(2), "text").unwrap());
    }

    #[test]
    fn test_count_entity_vectors() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        assert_eq!(store.count_entity_vectors(collection_id, entity_id).unwrap(), 0);

        store.put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![1.0])).unwrap();
        store.put_vector(collection_id, entity_id, "image", &VectorData::Dense(vec![2.0])).unwrap();

        assert_eq!(store.count_entity_vectors(collection_id, entity_id).unwrap(), 2);
    }

    #[test]
    fn test_list_entities_with_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);

        // Create entities with different vectors
        store
            .put_vector(collection_id, EntityId::new(1), "text", &VectorData::Dense(vec![1.0]))
            .unwrap();
        store
            .put_vector(collection_id, EntityId::new(2), "text", &VectorData::Dense(vec![2.0]))
            .unwrap();
        store
            .put_vector(collection_id, EntityId::new(2), "image", &VectorData::Dense(vec![3.0]))
            .unwrap();
        store
            .put_vector(collection_id, EntityId::new(3), "image", &VectorData::Dense(vec![4.0]))
            .unwrap();

        let text_entities = store.list_entities_with_vector(collection_id, "text").unwrap();
        assert_eq!(text_entities.len(), 2);
        assert!(text_entities.contains(&EntityId::new(1)));
        assert!(text_entities.contains(&EntityId::new(2)));

        let image_entities = store.list_entities_with_vector(collection_id, "image").unwrap();
        assert_eq!(image_entities.len(), 2);
        assert!(image_entities.contains(&EntityId::new(2)));
        assert!(image_entities.contains(&EntityId::new(3)));
    }

    #[test]
    fn test_update_vector() {
        let store = create_test_store();
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        // Store initial vector
        store
            .put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![1.0, 2.0]))
            .unwrap();

        // Update vector
        store
            .put_vector(collection_id, entity_id, "text", &VectorData::Dense(vec![3.0, 4.0, 5.0]))
            .unwrap();

        let retrieved = store.get_vector(collection_id, entity_id, "text").unwrap().unwrap();

        assert_eq!(retrieved.as_dense(), Some([3.0, 4.0, 5.0].as_slice()));
    }

    #[test]
    fn test_isolation_between_collections() {
        let store = create_test_store();
        let collection1 = CollectionId::new(1);
        let collection2 = CollectionId::new(2);
        let entity_id = EntityId::new(42);

        store.put_vector(collection1, entity_id, "text", &VectorData::Dense(vec![1.0])).unwrap();
        store.put_vector(collection2, entity_id, "text", &VectorData::Dense(vec![2.0])).unwrap();

        let v1 = store.get_vector(collection1, entity_id, "text").unwrap().unwrap();
        let v2 = store.get_vector(collection2, entity_id, "text").unwrap().unwrap();

        assert_eq!(v1.as_dense(), Some([1.0].as_slice()));
        assert_eq!(v2.as_dense(), Some([2.0].as_slice()));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = VectorData::Dense(vec![1.0, 2.0, 3.0]);
        let encoded = encode_vector_value(&data, "test_vector");
        let (decoded, name) = decode_vector_value(&encoded).unwrap();

        assert_eq!(decoded, data);
        assert_eq!(name, "test_vector");
    }
}
