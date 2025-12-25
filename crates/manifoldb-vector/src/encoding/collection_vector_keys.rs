//! Key encoding for collection vector storage.
//!
//! This module provides key encoding for vectors stored separately from entities.
//! The key format enables efficient access patterns:
//!
//! - Get a specific named vector for an entity in a collection
//! - Get all vectors for an entity in a collection
//! - Delete all vectors for an entity (cascade delete)
//!
//! # Key Format
//!
//! ## Collection vectors
//! - `0x40` - Collection vector: `[0x40][collection_id][entity_id][vector_name_hash]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.

use manifoldb_core::{CollectionId, EntityId};

use super::hash_name;

/// Key prefix for collection vectors.
pub const PREFIX_COLLECTION_VECTOR: u8 = 0x40;

/// Encode a key for a collection vector.
///
/// Key format: `[PREFIX_COLLECTION_VECTOR][collection_id][entity_id][vector_name_hash]`
///
/// This format enables:
/// - Fast lookup of a specific vector by (collection, entity, name)
/// - Prefix scan for all vectors of an entity within a collection
/// - Ordered iteration within each collection
#[must_use]
pub fn encode_collection_vector_key(
    collection_id: CollectionId,
    entity_id: EntityId,
    vector_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25); // 1 + 8 + 8 + 8
    key.push(PREFIX_COLLECTION_VECTOR);
    key.extend_from_slice(&collection_id.as_u64().to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for all vectors of an entity within a collection.
///
/// This prefix can be used for range scans to get all vectors for an entity.
#[must_use]
pub fn encode_entity_vector_prefix(collection_id: CollectionId, entity_id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17); // 1 + 8 + 8
    key.push(PREFIX_COLLECTION_VECTOR);
    key.extend_from_slice(&collection_id.as_u64().to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for all vectors in a collection.
///
/// This prefix can be used for range scans to iterate all vectors in a collection.
#[must_use]
pub fn encode_collection_vector_prefix(collection_id: CollectionId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9); // 1 + 8
    key.push(PREFIX_COLLECTION_VECTOR);
    key.extend_from_slice(&collection_id.as_u64().to_be_bytes());
    key
}

/// A decoded collection vector key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionVectorKey {
    /// The collection ID.
    pub collection_id: CollectionId,
    /// The entity ID.
    pub entity_id: EntityId,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
}

/// Decode a collection vector key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_collection_vector_key(key: &[u8]) -> Option<CollectionVectorKey> {
    if key.len() != 25 || key[0] != PREFIX_COLLECTION_VECTOR {
        return None;
    }

    let collection_id_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let entity_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let name_hash_bytes: [u8; 8] = key[17..25].try_into().ok()?;

    Some(CollectionVectorKey {
        collection_id: CollectionId::new(u64::from_be_bytes(collection_id_bytes)),
        entity_id: EntityId::new(u64::from_be_bytes(entity_id_bytes)),
        vector_name_hash: u64::from_be_bytes(name_hash_bytes),
    })
}

/// Decode the entity ID from a collection vector key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_collection_vector_entity_id(key: &[u8]) -> Option<EntityId> {
    decode_collection_vector_key(key).map(|k| k.entity_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_collection_vector_key() {
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let key = encode_collection_vector_key(collection_id, entity_id, "text_embedding");

        assert_eq!(key.len(), 25);
        assert_eq!(key[0], PREFIX_COLLECTION_VECTOR);
    }

    #[test]
    fn test_decode_collection_vector_key() {
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);
        let vector_name = "text_embedding";

        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);
        let decoded = decode_collection_vector_key(&key).unwrap();

        assert_eq!(decoded.collection_id, collection_id);
        assert_eq!(decoded.entity_id, entity_id);
        assert_eq!(decoded.vector_name_hash, hash_name(vector_name));
    }

    #[test]
    fn test_entity_vector_prefix() {
        let collection_id = CollectionId::new(1);
        let entity_id = EntityId::new(42);

        let prefix = encode_entity_vector_prefix(collection_id, entity_id);
        let key1 = encode_collection_vector_key(collection_id, entity_id, "text");
        let key2 = encode_collection_vector_key(collection_id, entity_id, "image");
        let key3 = encode_collection_vector_key(collection_id, EntityId::new(43), "text");

        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));
        assert!(!key3.starts_with(&prefix));
    }

    #[test]
    fn test_collection_vector_prefix() {
        let collection_id = CollectionId::new(1);
        let collection_id_2 = CollectionId::new(2);

        let prefix = encode_collection_vector_prefix(collection_id);
        let key1 = encode_collection_vector_key(collection_id, EntityId::new(1), "text");
        let key2 = encode_collection_vector_key(collection_id, EntityId::new(2), "image");
        let key3 = encode_collection_vector_key(collection_id_2, EntityId::new(1), "text");

        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));
        assert!(!key3.starts_with(&prefix));
    }

    #[test]
    fn test_key_ordering() {
        let coll = CollectionId::new(1);

        // Keys should be ordered by entity ID, then by vector name hash
        let key1 = encode_collection_vector_key(coll, EntityId::new(1), "a");
        let key2 = encode_collection_vector_key(coll, EntityId::new(2), "a");

        // Different entities should be ordered
        assert!(key1 < key2);

        // Same entity, different vector names are grouped together under entity prefix
        let key3 = encode_collection_vector_key(coll, EntityId::new(1), "b");
        let prefix = encode_entity_vector_prefix(coll, EntityId::new(1));
        assert!(key1.starts_with(&prefix));
        assert!(key3.starts_with(&prefix));
    }

    #[test]
    fn test_decode_invalid_key() {
        // Wrong length
        assert!(decode_collection_vector_key(&[PREFIX_COLLECTION_VECTOR; 10]).is_none());

        // Wrong prefix
        assert!(decode_collection_vector_key(&[0xFF; 25]).is_none());

        // Empty
        assert!(decode_collection_vector_key(&[]).is_none());
    }
}
