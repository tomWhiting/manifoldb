//! Point store implementation for Qdrant-style vector collections.
//!
//! This module provides storage for points with multiple named vectors (dense,
//! sparse, or multi-vector) and JSON payloads.
//!
//! # Storage Tables
//!
//! - `point_collections`: Collection metadata (schema)
//! - `point_payloads`: Point payloads (JSON data)
//! - `point_dense_vectors`: Dense vectors
//! - `point_sparse_vectors`: Sparse vectors
//! - `point_multi_vectors`: Multi-vectors
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::store::PointStore;
//! use manifoldb_vector::types::{CollectionName, CollectionSchema, VectorConfig, Payload, NamedVector};
//! use manifoldb_core::PointId;
//!
//! let store = PointStore::new(engine);
//!
//! // Create a collection
//! let name = CollectionName::new("documents")?;
//! let schema = CollectionSchema::new()
//!     .with_vector("dense", VectorConfig::dense(384))
//!     .with_vector("sparse", VectorConfig::sparse(30522));
//! store.create_collection(&name, schema)?;
//!
//! // Insert a point
//! let mut payload = Payload::new();
//! payload.insert("title", "Hello World".into());
//!
//! let mut vectors = HashMap::new();
//! vectors.insert("dense".to_string(), NamedVector::Dense(vec![0.1; 384]));
//! vectors.insert("sparse".to_string(), NamedVector::Sparse(vec![(100, 0.5)]));
//!
//! store.upsert_point(&name, PointId::new(1), payload, vectors)?;
//! ```

use std::collections::HashMap;
use std::ops::Bound;

use manifoldb_core::PointId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::encoding::{
    decode_point_payload_point_id, encode_collection_key, encode_collection_prefix,
    encode_dense_vector_collection_prefix, encode_dense_vector_key,
    encode_dense_vector_point_prefix, encode_multi_vector_collection_prefix,
    encode_multi_vector_key, encode_multi_vector_point_prefix, encode_point_payload_key,
    encode_point_payload_prefix, encode_sparse_vector_collection_prefix, encode_sparse_vector_key,
    encode_sparse_vector_point_prefix,
};
use crate::error::VectorError;
use crate::types::{
    Collection, CollectionName, CollectionSchema, NamedVector, Payload, VectorConfig, VectorType,
};

/// Table name for point collection metadata.
pub const TABLE_POINT_COLLECTIONS: &str = "point_collections";

/// Private alias for internal use.
const TABLE_COLLECTIONS: &str = TABLE_POINT_COLLECTIONS;

/// Table name for point payloads.
const TABLE_PAYLOADS: &str = "point_payloads";

/// Table name for dense vectors.
const TABLE_DENSE_VECTORS: &str = "point_dense_vectors";

/// Table name for sparse vectors.
const TABLE_SPARSE_VECTORS: &str = "point_sparse_vectors";

/// Table name for multi-vectors.
const TABLE_MULTI_VECTORS: &str = "point_multi_vectors";

/// A store for points with multiple named vectors.
///
/// `PointStore` provides CRUD operations for points organized into named
/// collections. Each point can have multiple named vectors (dense, sparse,
/// or multi-vector) and a JSON payload.
pub struct PointStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> PointStore<E> {
    /// Create a new point store with the given storage engine.
    #[must_use]
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Get a reference to the storage engine.
    #[must_use]
    pub fn engine(&self) -> &E {
        &self.engine
    }

    // ========================================================================
    // Collection operations
    // ========================================================================

    /// Create a new collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection already exists or if the storage
    /// operation fails.
    pub fn create_collection(
        &self,
        name: &CollectionName,
        schema: CollectionSchema,
    ) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let key = encode_collection_key(name.as_str());

        // Check if collection already exists
        if tx.get(TABLE_COLLECTIONS, &key)?.is_some() {
            return Err(VectorError::InvalidName(format!("collection '{}' already exists", name)));
        }

        // Store the collection metadata
        let collection = Collection::new(name.clone(), schema);
        tx.put(TABLE_COLLECTIONS, &key, &collection.to_bytes()?)?;
        tx.commit()?;

        Ok(())
    }

    /// Get a collection by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist or if the storage
    /// operation fails.
    pub fn get_collection(&self, name: &CollectionName) -> Result<Collection, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_collection_key(name.as_str());

        let bytes = tx
            .get(TABLE_COLLECTIONS, &key)?
            .ok_or_else(|| VectorError::SpaceNotFound(format!("collection '{}'", name)))?;

        Collection::from_bytes(&bytes)
    }

    /// Delete a collection and all its points.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist or if the storage
    /// operation fails.
    pub fn delete_collection(&self, name: &CollectionName) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;

        let collection_key = encode_collection_key(name.as_str());

        // Check if collection exists
        if tx.get(TABLE_COLLECTIONS, &collection_key)?.is_none() {
            return Err(VectorError::SpaceNotFound(format!("collection '{}'", name)));
        }

        // Delete all payloads
        delete_by_prefix(&mut tx, TABLE_PAYLOADS, &encode_point_payload_prefix(name.as_str()))?;

        // Delete all dense vectors
        delete_by_prefix(
            &mut tx,
            TABLE_DENSE_VECTORS,
            &encode_dense_vector_collection_prefix(name.as_str()),
        )?;

        // Delete all sparse vectors
        delete_by_prefix(
            &mut tx,
            TABLE_SPARSE_VECTORS,
            &encode_sparse_vector_collection_prefix(name.as_str()),
        )?;

        // Delete all multi-vectors
        delete_by_prefix(
            &mut tx,
            TABLE_MULTI_VECTORS,
            &encode_multi_vector_collection_prefix(name.as_str()),
        )?;

        // Delete the collection metadata
        tx.delete(TABLE_COLLECTIONS, &collection_key)?;

        tx.commit()?;
        Ok(())
    }

    /// List all collections.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_collections(&self) -> Result<Vec<Collection>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_collection_prefix();
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_COLLECTIONS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut collections = Vec::new();
        while let Some((_, value)) = cursor.next()? {
            collections.push(Collection::from_bytes(&value)?);
        }

        Ok(collections)
    }

    /// Check if a collection exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn collection_exists(&self, name: &CollectionName) -> Result<bool, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_collection_key(name.as_str());
        Ok(tx.get(TABLE_COLLECTIONS, &key)?.is_some())
    }

    // ========================================================================
    // Point operations
    // ========================================================================

    /// Upsert a point (insert or update).
    ///
    /// This operation will:
    /// - Insert the point if it doesn't exist
    /// - Update the point if it exists, replacing payload and specified vectors
    /// - Vectors not specified in this call are left unchanged
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection doesn't exist
    /// - A vector type doesn't match the schema
    /// - A vector dimension doesn't match the schema
    /// - The storage operation fails
    pub fn upsert_point(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
        payload: Payload,
        vectors: HashMap<String, NamedVector>,
    ) -> Result<(), VectorError> {
        // Get collection to validate schema
        let collection = self.get_collection(collection_name)?;
        let schema = collection.schema();

        // Validate vectors against schema
        for (vector_name, vector) in &vectors {
            if let Some(config) = schema.get_vector(vector_name) {
                validate_vector(vector, config)?;
            }
            // Allow vectors not in schema (flexible schema)
        }

        let mut tx = self.engine.begin_write()?;
        let collection_str = collection_name.as_str();

        // Store payload
        let payload_key = encode_point_payload_key(collection_str, point_id);
        tx.put(TABLE_PAYLOADS, &payload_key, &payload.to_bytes()?)?;

        // Store each vector
        for (vector_name, vector) in vectors {
            match vector {
                NamedVector::Dense(data) => {
                    let key = encode_dense_vector_key(collection_str, point_id, &vector_name);
                    tx.put(TABLE_DENSE_VECTORS, &key, &encode_dense_vector(&data))?;
                }
                NamedVector::Sparse(data) => {
                    let key = encode_sparse_vector_key(collection_str, point_id, &vector_name);
                    tx.put(TABLE_SPARSE_VECTORS, &key, &encode_sparse_vector(&data))?;
                }
                NamedVector::Multi(data) => {
                    let key = encode_multi_vector_key(collection_str, point_id, &vector_name);
                    tx.put(TABLE_MULTI_VECTORS, &key, &encode_multi_vector(&data))?;
                }
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Insert a point. Fails if the point already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the point already exists, the collection doesn't exist,
    /// or the storage operation fails.
    pub fn insert_point(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
        payload: Payload,
        vectors: HashMap<String, NamedVector>,
    ) -> Result<(), VectorError> {
        // Check if point exists
        if self.point_exists(collection_name, point_id)? {
            return Err(VectorError::Encoding(format!(
                "point {} already exists in collection '{}'",
                point_id, collection_name
            )));
        }

        self.upsert_point(collection_name, point_id, payload, vectors)
    }

    /// Get a point's payload.
    ///
    /// # Errors
    ///
    /// Returns an error if the point doesn't exist or the storage operation fails.
    pub fn get_payload(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
    ) -> Result<Payload, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_point_payload_key(collection_name.as_str(), point_id);

        let bytes =
            tx.get(TABLE_PAYLOADS, &key)?.ok_or_else(|| VectorError::EmbeddingNotFound {
                entity_id: point_id.as_u64(),
                space: format!("collection '{}'", collection_name),
            })?;

        Payload::from_bytes(&bytes)
    }

    /// Get a specific vector from a point.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector doesn't exist or the storage operation fails.
    pub fn get_vector(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
        vector_name: &str,
    ) -> Result<NamedVector, VectorError> {
        let tx = self.engine.begin_read()?;
        let collection_str = collection_name.as_str();

        // Try dense first
        let dense_key = encode_dense_vector_key(collection_str, point_id, vector_name);
        if let Some(bytes) = tx.get(TABLE_DENSE_VECTORS, &dense_key)? {
            return Ok(NamedVector::Dense(decode_dense_vector(&bytes)?));
        }

        // Try sparse
        let sparse_key = encode_sparse_vector_key(collection_str, point_id, vector_name);
        if let Some(bytes) = tx.get(TABLE_SPARSE_VECTORS, &sparse_key)? {
            return Ok(NamedVector::Sparse(decode_sparse_vector(&bytes)?));
        }

        // Try multi
        let multi_key = encode_multi_vector_key(collection_str, point_id, vector_name);
        if let Some(bytes) = tx.get(TABLE_MULTI_VECTORS, &multi_key)? {
            return Ok(NamedVector::Multi(decode_multi_vector(&bytes)?));
        }

        Err(VectorError::EmbeddingNotFound {
            entity_id: point_id.as_u64(),
            space: format!("vector '{}' in collection '{}'", vector_name, collection_name),
        })
    }

    /// Get all vectors for a point.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_all_vectors(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
    ) -> Result<HashMap<String, NamedVector>, VectorError> {
        let tx = self.engine.begin_read()?;
        let collection_str = collection_name.as_str();
        let mut vectors = HashMap::new();

        // Get all dense vectors for this point
        let dense_prefix = encode_dense_vector_point_prefix(collection_str, point_id);
        let dense_prefix_end = next_prefix(&dense_prefix);
        let mut cursor = tx.range(
            TABLE_DENSE_VECTORS,
            Bound::Included(dense_prefix.as_slice()),
            Bound::Excluded(dense_prefix_end.as_slice()),
        )?;
        while let Some((key, value)) = cursor.next()? {
            if let Some(name) = extract_vector_name_from_key(&key, collection_str, point_id) {
                vectors.insert(name, NamedVector::Dense(decode_dense_vector(&value)?));
            }
        }
        drop(cursor);

        // Get all sparse vectors for this point
        let sparse_prefix = encode_sparse_vector_point_prefix(collection_str, point_id);
        let sparse_prefix_end = next_prefix(&sparse_prefix);
        let mut cursor = tx.range(
            TABLE_SPARSE_VECTORS,
            Bound::Included(sparse_prefix.as_slice()),
            Bound::Excluded(sparse_prefix_end.as_slice()),
        )?;
        while let Some((key, value)) = cursor.next()? {
            if let Some(name) = extract_vector_name_from_key(&key, collection_str, point_id) {
                vectors.insert(name, NamedVector::Sparse(decode_sparse_vector(&value)?));
            }
        }
        drop(cursor);

        // Get all multi-vectors for this point
        let multi_prefix = encode_multi_vector_point_prefix(collection_str, point_id);
        let multi_prefix_end = next_prefix(&multi_prefix);
        let mut cursor = tx.range(
            TABLE_MULTI_VECTORS,
            Bound::Included(multi_prefix.as_slice()),
            Bound::Excluded(multi_prefix_end.as_slice()),
        )?;
        while let Some((key, value)) = cursor.next()? {
            if let Some(name) = extract_vector_name_from_key(&key, collection_str, point_id) {
                vectors.insert(name, NamedVector::Multi(decode_multi_vector(&value)?));
            }
        }

        Ok(vectors)
    }

    /// Update a point's payload without touching vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the point doesn't exist or the storage operation fails.
    pub fn update_payload(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
        payload: Payload,
    ) -> Result<(), VectorError> {
        // Check point exists
        if !self.point_exists(collection_name, point_id)? {
            return Err(VectorError::EmbeddingNotFound {
                entity_id: point_id.as_u64(),
                space: format!("collection '{}'", collection_name),
            });
        }

        let mut tx = self.engine.begin_write()?;
        let key = encode_point_payload_key(collection_name.as_str(), point_id);
        tx.put(TABLE_PAYLOADS, &key, &payload.to_bytes()?)?;
        tx.commit()?;

        Ok(())
    }

    /// Update a specific vector without touching the payload or other vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist or the storage operation fails.
    pub fn update_vector(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
        vector_name: &str,
        vector: NamedVector,
    ) -> Result<(), VectorError> {
        // Validate against schema if defined
        let collection = self.get_collection(collection_name)?;
        if let Some(config) = collection.schema().get_vector(vector_name) {
            validate_vector(&vector, config)?;
        }

        let mut tx = self.engine.begin_write()?;
        let collection_str = collection_name.as_str();

        match vector {
            NamedVector::Dense(data) => {
                let key = encode_dense_vector_key(collection_str, point_id, vector_name);
                tx.put(TABLE_DENSE_VECTORS, &key, &encode_dense_vector(&data))?;
            }
            NamedVector::Sparse(data) => {
                let key = encode_sparse_vector_key(collection_str, point_id, vector_name);
                tx.put(TABLE_SPARSE_VECTORS, &key, &encode_sparse_vector(&data))?;
            }
            NamedVector::Multi(data) => {
                let key = encode_multi_vector_key(collection_str, point_id, vector_name);
                tx.put(TABLE_MULTI_VECTORS, &key, &encode_multi_vector(&data))?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Delete a point and all its vectors.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the point was deleted, `Ok(false)` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn delete_point(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let collection_str = collection_name.as_str();

        // Delete payload
        let payload_key = encode_point_payload_key(collection_str, point_id);
        let existed = tx.delete(TABLE_PAYLOADS, &payload_key)?;

        // Delete all dense vectors for this point
        delete_by_prefix(
            &mut tx,
            TABLE_DENSE_VECTORS,
            &encode_dense_vector_point_prefix(collection_str, point_id),
        )?;

        // Delete all sparse vectors for this point
        delete_by_prefix(
            &mut tx,
            TABLE_SPARSE_VECTORS,
            &encode_sparse_vector_point_prefix(collection_str, point_id),
        )?;

        // Delete all multi-vectors for this point
        delete_by_prefix(
            &mut tx,
            TABLE_MULTI_VECTORS,
            &encode_multi_vector_point_prefix(collection_str, point_id),
        )?;

        tx.commit()?;
        Ok(existed)
    }

    /// Delete a specific vector from a point.
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
        collection_name: &CollectionName,
        point_id: PointId,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let collection_str = collection_name.as_str();

        // Try to delete from each vector table
        let dense_key = encode_dense_vector_key(collection_str, point_id, vector_name);
        if tx.delete(TABLE_DENSE_VECTORS, &dense_key)? {
            tx.commit()?;
            return Ok(true);
        }

        let sparse_key = encode_sparse_vector_key(collection_str, point_id, vector_name);
        if tx.delete(TABLE_SPARSE_VECTORS, &sparse_key)? {
            tx.commit()?;
            return Ok(true);
        }

        let multi_key = encode_multi_vector_key(collection_str, point_id, vector_name);
        if tx.delete(TABLE_MULTI_VECTORS, &multi_key)? {
            tx.commit()?;
            return Ok(true);
        }

        tx.commit()?;
        Ok(false)
    }

    /// Check if a point exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn point_exists(
        &self,
        collection_name: &CollectionName,
        point_id: PointId,
    ) -> Result<bool, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_point_payload_key(collection_name.as_str(), point_id);
        Ok(tx.get(TABLE_PAYLOADS, &key)?.is_some())
    }

    /// List all point IDs in a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn list_points(
        &self,
        collection_name: &CollectionName,
    ) -> Result<Vec<PointId>, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_point_payload_prefix(collection_name.as_str());
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_PAYLOADS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut points = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(point_id) = decode_point_payload_point_id(&key) {
                points.push(point_id);
            }
        }

        Ok(points)
    }

    /// Count the number of points in a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn count_points(&self, collection_name: &CollectionName) -> Result<usize, VectorError> {
        let tx = self.engine.begin_read()?;

        let prefix = encode_point_payload_prefix(collection_name.as_str());
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_PAYLOADS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Get multiple points at once.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage operation fails.
    pub fn get_points(
        &self,
        collection_name: &CollectionName,
        point_ids: &[PointId],
    ) -> Result<Vec<(PointId, Option<Payload>)>, VectorError> {
        let tx = self.engine.begin_read()?;

        let mut results = Vec::with_capacity(point_ids.len());

        for &point_id in point_ids {
            let key = encode_point_payload_key(collection_name.as_str(), point_id);
            let payload = tx
                .get(TABLE_PAYLOADS, &key)?
                .map(|bytes| Payload::from_bytes(&bytes))
                .transpose()?;

            results.push((point_id, payload));
        }

        Ok(results)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

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

/// Delete all keys matching a prefix.
fn delete_by_prefix<T: Transaction>(
    tx: &mut T,
    table: &str,
    prefix: &[u8],
) -> Result<(), VectorError> {
    let prefix_end = next_prefix(prefix);

    let mut keys_to_delete = Vec::new();
    {
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(prefix_end.as_slice()))?;

        while let Some((key, _)) = cursor.next()? {
            keys_to_delete.push(key);
        }
    }

    for key in keys_to_delete {
        tx.delete(table, &key)?;
    }

    Ok(())
}

/// Validate a vector against its configuration.
fn validate_vector(vector: &NamedVector, config: &VectorConfig) -> Result<(), VectorError> {
    match (vector, config.vector_type) {
        (NamedVector::Dense(data), VectorType::Dense) => {
            if data.len() != config.dimension as usize {
                return Err(VectorError::DimensionMismatch {
                    expected: config.dimension as usize,
                    actual: data.len(),
                });
            }
        }
        (NamedVector::Sparse(data), VectorType::Sparse) => {
            // Check all indices are within bounds
            for &(idx, _) in data {
                if idx >= config.dimension {
                    return Err(VectorError::Encoding(format!(
                        "sparse vector index {} exceeds max dimension {}",
                        idx, config.dimension
                    )));
                }
            }
        }
        (NamedVector::Multi(data), VectorType::Multi) => {
            // Check all inner vectors have the correct dimension
            for (i, inner) in data.iter().enumerate() {
                if inner.len() != config.dimension as usize {
                    return Err(VectorError::Encoding(format!(
                        "multi-vector inner vector {} has dimension {} but expected {}",
                        i,
                        inner.len(),
                        config.dimension
                    )));
                }
            }
        }
        (actual, expected) => {
            return Err(VectorError::Encoding(format!(
                "vector type mismatch: expected {:?}, got {:?}",
                expected,
                actual.vector_type()
            )));
        }
    }

    Ok(())
}

/// Extract the vector name from a vector key.
///
/// We need to reverse the hash to get the name. Since we can't do that,
/// we store the name in the collection schema and look it up by hash.
/// For now, we return None since we need the schema to do the reverse lookup.
fn extract_vector_name_from_key(
    _key: &[u8],
    _collection: &str,
    _point_id: PointId,
) -> Option<String> {
    // Vector keys are: [prefix][collection_hash][point_id][vector_name_hash]
    // We can't reverse a hash, so we need a different approach.
    // For now, this function isn't used in production - vectors are looked up by name.
    None
}

// ============================================================================
// Vector encoding/decoding
// ============================================================================

/// Encode a dense vector to bytes.
fn encode_dense_vector(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + data.len() * 4);
    bytes.extend_from_slice(&(data.len() as u32).to_be_bytes());
    for &value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decode a dense vector from bytes.
fn decode_dense_vector(bytes: &[u8]) -> Result<Vec<f32>, VectorError> {
    if bytes.len() < 4 {
        return Err(VectorError::Encoding("truncated dense vector".to_string()));
    }

    let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected_len = 4 + count * 4;

    if bytes.len() != expected_len {
        return Err(VectorError::Encoding(format!(
            "dense vector length mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        )));
    }

    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 4 + i * 4;
        let value = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        data.push(value);
    }

    Ok(data)
}

/// Encode a sparse vector to bytes.
fn encode_sparse_vector(data: &[(u32, f32)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + data.len() * 8);
    bytes.extend_from_slice(&(data.len() as u32).to_be_bytes());
    for &(idx, value) in data {
        bytes.extend_from_slice(&idx.to_be_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decode a sparse vector from bytes.
fn decode_sparse_vector(bytes: &[u8]) -> Result<Vec<(u32, f32)>, VectorError> {
    if bytes.len() < 4 {
        return Err(VectorError::Encoding("truncated sparse vector".to_string()));
    }

    let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let expected_len = 4 + count * 8;

    if bytes.len() != expected_len {
        return Err(VectorError::Encoding(format!(
            "sparse vector length mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        )));
    }

    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 4 + i * 8;
        let idx = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let value = f32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]);
        data.push((idx, value));
    }

    Ok(data)
}

/// Encode a multi-vector to bytes.
fn encode_multi_vector(data: &[Vec<f32>]) -> Vec<u8> {
    // Format: count (u32) + dimension (u32) + flat f32 data
    if data.is_empty() {
        return vec![0, 0, 0, 0, 0, 0, 0, 0];
    }

    let count = data.len();
    let dimension = data[0].len();
    let mut bytes = Vec::with_capacity(8 + count * dimension * 4);

    bytes.extend_from_slice(&(count as u32).to_be_bytes());
    bytes.extend_from_slice(&(dimension as u32).to_be_bytes());

    for inner in data {
        for &value in inner {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    bytes
}

/// Decode a multi-vector from bytes.
fn decode_multi_vector(bytes: &[u8]) -> Result<Vec<Vec<f32>>, VectorError> {
    if bytes.len() < 8 {
        return Err(VectorError::Encoding("truncated multi-vector".to_string()));
    }

    let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let dimension = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

    if count == 0 {
        return Ok(Vec::new());
    }

    let expected_len = 8 + count * dimension * 4;
    if bytes.len() != expected_len {
        return Err(VectorError::Encoding(format!(
            "multi-vector length mismatch: expected {}, got {}",
            expected_len,
            bytes.len()
        )));
    }

    let mut data = Vec::with_capacity(count);
    for i in 0..count {
        let mut inner = Vec::with_capacity(dimension);
        for j in 0..dimension {
            let offset = 8 + (i * dimension + j) * 4;
            let value = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            inner.push(value);
        }
        data.push(inner);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn create_test_store() -> PointStore<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        PointStore::new(engine)
    }

    fn unique_collection_name() -> CollectionName {
        let count = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        CollectionName::new(format!("test_collection_{}", count)).unwrap()
    }

    #[test]
    fn create_and_get_collection() {
        let store = create_test_store();
        let name = unique_collection_name();
        let schema = CollectionSchema::new()
            .with_vector("dense", VectorConfig::dense(384))
            .with_vector("sparse", VectorConfig::sparse(30522));

        store.create_collection(&name, schema.clone()).unwrap();

        let collection = store.get_collection(&name).unwrap();
        assert_eq!(collection.name().as_str(), name.as_str());
        assert_eq!(collection.schema().len(), 2);
    }

    #[test]
    fn create_duplicate_collection_fails() {
        let store = create_test_store();
        let name = unique_collection_name();
        let schema = CollectionSchema::new();

        store.create_collection(&name, schema.clone()).unwrap();
        let result = store.create_collection(&name, schema);

        assert!(result.is_err());
    }

    #[test]
    fn list_collections() {
        let store = create_test_store();

        let name1 = unique_collection_name();
        let name2 = unique_collection_name();

        store.create_collection(&name1, CollectionSchema::new()).unwrap();
        store.create_collection(&name2, CollectionSchema::new()).unwrap();

        let collections = store.list_collections().unwrap();
        assert!(collections.len() >= 2);
    }

    #[test]
    fn delete_collection() {
        let store = create_test_store();
        let name = unique_collection_name();

        store.create_collection(&name, CollectionSchema::new()).unwrap();

        // Add a point
        let mut vectors = HashMap::new();
        vectors.insert("v".to_string(), NamedVector::Dense(vec![0.1, 0.2]));
        store.upsert_point(&name, PointId::new(1), Payload::new(), vectors).unwrap();

        // Delete collection
        store.delete_collection(&name).unwrap();

        // Collection should not exist
        assert!(!store.collection_exists(&name).unwrap());
    }

    #[test]
    fn upsert_and_get_point() {
        let store = create_test_store();
        let name = unique_collection_name();
        let schema = CollectionSchema::new().with_vector("dense", VectorConfig::dense(3));

        store.create_collection(&name, schema).unwrap();

        // Create payload
        let mut payload = Payload::new();
        payload.insert("title", json!("Test Document"));
        payload.insert("count", json!(42));

        // Create vectors
        let mut vectors = HashMap::new();
        vectors.insert("dense".to_string(), NamedVector::Dense(vec![0.1, 0.2, 0.3]));

        // Upsert
        store.upsert_point(&name, PointId::new(1), payload, vectors).unwrap();

        // Get payload
        let retrieved_payload = store.get_payload(&name, PointId::new(1)).unwrap();
        assert_eq!(retrieved_payload.get("title"), Some(&json!("Test Document")));

        // Get vector
        let retrieved_vector = store.get_vector(&name, PointId::new(1), "dense").unwrap();
        assert_eq!(retrieved_vector.as_dense(), Some(&[0.1, 0.2, 0.3][..]));
    }

    #[test]
    fn upsert_updates_existing_point() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        // First upsert
        let mut payload1 = Payload::new();
        payload1.insert("version", json!(1));

        let mut vectors1 = HashMap::new();
        vectors1.insert("v".to_string(), NamedVector::Dense(vec![1.0]));

        store.upsert_point(&name, PointId::new(1), payload1, vectors1).unwrap();

        // Second upsert (update)
        let mut payload2 = Payload::new();
        payload2.insert("version", json!(2));

        let mut vectors2 = HashMap::new();
        vectors2.insert("v".to_string(), NamedVector::Dense(vec![2.0]));

        store.upsert_point(&name, PointId::new(1), payload2, vectors2).unwrap();

        // Check updated values
        let payload = store.get_payload(&name, PointId::new(1)).unwrap();
        assert_eq!(payload.get("version"), Some(&json!(2)));

        let vector = store.get_vector(&name, PointId::new(1), "v").unwrap();
        assert_eq!(vector.as_dense(), Some(&[2.0][..]));
    }

    #[test]
    fn insert_duplicate_fails() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        store.insert_point(&name, PointId::new(1), Payload::new(), HashMap::new()).unwrap();

        let result = store.insert_point(&name, PointId::new(1), Payload::new(), HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn multi_vector_point() {
        let store = create_test_store();
        let name = unique_collection_name();
        let schema = CollectionSchema::new()
            .with_vector("dense", VectorConfig::dense(3))
            .with_vector("sparse", VectorConfig::sparse(1000))
            .with_vector("multi", VectorConfig::multi(2));

        store.create_collection(&name, schema).unwrap();

        let mut vectors = HashMap::new();
        vectors.insert("dense".to_string(), NamedVector::Dense(vec![0.1, 0.2, 0.3]));
        vectors.insert("sparse".to_string(), NamedVector::Sparse(vec![(10, 0.5), (50, 0.3)]));
        vectors
            .insert("multi".to_string(), NamedVector::Multi(vec![vec![0.1, 0.2], vec![0.3, 0.4]]));

        store.upsert_point(&name, PointId::new(1), Payload::new(), vectors).unwrap();

        // Retrieve each vector
        let dense = store.get_vector(&name, PointId::new(1), "dense").unwrap();
        assert!(dense.as_dense().is_some());

        let sparse = store.get_vector(&name, PointId::new(1), "sparse").unwrap();
        assert!(sparse.as_sparse().is_some());

        let multi = store.get_vector(&name, PointId::new(1), "multi").unwrap();
        assert!(multi.as_multi().is_some());
    }

    #[test]
    fn update_individual_vector() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        // Insert with vector v1
        let mut vectors = HashMap::new();
        vectors.insert("v1".to_string(), NamedVector::Dense(vec![1.0, 2.0]));
        store.upsert_point(&name, PointId::new(1), Payload::new(), vectors).unwrap();

        // Update v1
        store
            .update_vector(&name, PointId::new(1), "v1", NamedVector::Dense(vec![3.0, 4.0]))
            .unwrap();

        let v1 = store.get_vector(&name, PointId::new(1), "v1").unwrap();
        assert_eq!(v1.as_dense(), Some(&[3.0, 4.0][..]));
    }

    #[test]
    fn delete_point() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        let mut vectors = HashMap::new();
        vectors.insert("v".to_string(), NamedVector::Dense(vec![0.1]));
        store.upsert_point(&name, PointId::new(1), Payload::new(), vectors).unwrap();

        assert!(store.point_exists(&name, PointId::new(1)).unwrap());
        assert!(store.delete_point(&name, PointId::new(1)).unwrap());
        assert!(!store.point_exists(&name, PointId::new(1)).unwrap());

        // Delete again returns false
        assert!(!store.delete_point(&name, PointId::new(1)).unwrap());
    }

    #[test]
    fn delete_vector() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        let mut vectors = HashMap::new();
        vectors.insert("v1".to_string(), NamedVector::Dense(vec![1.0]));
        vectors.insert("v2".to_string(), NamedVector::Dense(vec![2.0]));
        store.upsert_point(&name, PointId::new(1), Payload::new(), vectors).unwrap();

        assert!(store.delete_vector(&name, PointId::new(1), "v1").unwrap());

        // v1 should be gone
        assert!(store.get_vector(&name, PointId::new(1), "v1").is_err());

        // v2 should still exist
        assert!(store.get_vector(&name, PointId::new(1), "v2").is_ok());
    }

    #[test]
    fn list_and_count_points() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        for i in 1..=5 {
            store.insert_point(&name, PointId::new(i), Payload::new(), HashMap::new()).unwrap();
        }

        let points = store.list_points(&name).unwrap();
        assert_eq!(points.len(), 5);

        let count = store.count_points(&name).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn get_multiple_points() {
        let store = create_test_store();
        let name = unique_collection_name();
        store.create_collection(&name, CollectionSchema::new()).unwrap();

        store.insert_point(&name, PointId::new(1), Payload::new(), HashMap::new()).unwrap();
        store.insert_point(&name, PointId::new(3), Payload::new(), HashMap::new()).unwrap();

        let results =
            store.get_points(&name, &[PointId::new(1), PointId::new(2), PointId::new(3)]).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].1.is_some()); // Point 1 exists
        assert!(results[1].1.is_none()); // Point 2 doesn't exist
        assert!(results[2].1.is_some()); // Point 3 exists
    }

    #[test]
    fn dimension_mismatch_fails() {
        let store = create_test_store();
        let name = unique_collection_name();
        let schema = CollectionSchema::new().with_vector("dense", VectorConfig::dense(3));

        store.create_collection(&name, schema).unwrap();

        let mut vectors = HashMap::new();
        vectors.insert("dense".to_string(), NamedVector::Dense(vec![0.1, 0.2])); // Wrong dimension

        let result = store.upsert_point(&name, PointId::new(1), Payload::new(), vectors);
        assert!(result.is_err());
    }

    #[test]
    fn vector_encoding_roundtrip() {
        // Dense
        let dense = vec![0.1, 0.2, 0.3, 0.4];
        let encoded = encode_dense_vector(&dense);
        let decoded = decode_dense_vector(&encoded).unwrap();
        assert_eq!(dense, decoded);

        // Sparse
        let sparse = vec![(10, 0.5), (50, 0.3), (100, 0.2)];
        let encoded = encode_sparse_vector(&sparse);
        let decoded = decode_sparse_vector(&encoded).unwrap();
        assert_eq!(sparse, decoded);

        // Multi
        let multi = vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]];
        let encoded = encode_multi_vector(&multi);
        let decoded = decode_multi_vector(&encoded).unwrap();
        assert_eq!(multi, decoded);
    }
}
