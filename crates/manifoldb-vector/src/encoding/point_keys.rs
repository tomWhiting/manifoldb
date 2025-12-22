//! Key encoding for point storage.
//!
//! This module provides key encoding functions for point storage, supporting
//! efficient lookups and range scans by collection and point ID.
//!
//! # Key Format
//!
//! All keys use big-endian encoding for proper sort order in range scans.
//!
//! ## Collection metadata
//! `[PREFIX_COLLECTION][collection_name_hash]`
//!
//! ## Point payload
//! `[PREFIX_POINT_PAYLOAD][collection_name_hash][point_id]`
//!
//! ## Dense vectors
//! `[PREFIX_POINT_DENSE_VECTOR][collection_name_hash][point_id][vector_name_hash]`
//!
//! ## Sparse vectors
//! `[PREFIX_POINT_SPARSE_VECTOR][collection_name_hash][point_id][vector_name_hash]`
//!
//! ## Multi-vectors
//! `[PREFIX_POINT_MULTI_VECTOR][collection_name_hash][point_id][vector_name_hash]`

use manifoldb_core::PointId;

/// Key prefix for collection metadata.
pub const PREFIX_COLLECTION: u8 = 0x20;

/// Key prefix for point payloads.
pub const PREFIX_POINT_PAYLOAD: u8 = 0x21;

/// Key prefix for dense vectors.
pub const PREFIX_POINT_DENSE_VECTOR: u8 = 0x22;

/// Key prefix for sparse vectors.
pub const PREFIX_POINT_SPARSE_VECTOR: u8 = 0x23;

/// Key prefix for multi-vectors.
pub const PREFIX_POINT_MULTI_VECTOR: u8 = 0x24;

/// Compute a hash for a name using FNV-1a.
///
/// This is the same hash function used for embedding names.
#[must_use]
pub fn hash_name(name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ============================================================================
// Collection keys
// ============================================================================

/// Encode a key for collection metadata.
///
/// Key format: `[PREFIX_COLLECTION][collection_name_hash]`
#[must_use]
pub fn encode_collection_key(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_COLLECTION);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all collections.
#[must_use]
pub fn encode_collection_prefix() -> Vec<u8> {
    vec![PREFIX_COLLECTION]
}

/// A decoded collection key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectionKey {
    /// The hash of the collection name.
    pub name_hash: u64,
}

/// Decode a collection key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_collection_key(key: &[u8]) -> Option<CollectionKey> {
    if key.len() != 9 || key[0] != PREFIX_COLLECTION {
        return None;
    }
    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(CollectionKey { name_hash: u64::from_be_bytes(name_hash_bytes) })
}

// ============================================================================
// Point payload keys
// ============================================================================

/// Encode a key for a point's payload.
///
/// Key format: `[PREFIX_POINT_PAYLOAD][collection_name_hash][point_id]`
#[must_use]
pub fn encode_point_payload_key(collection_name: &str, point_id: PointId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POINT_PAYLOAD);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all points in a collection.
#[must_use]
pub fn encode_point_payload_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POINT_PAYLOAD);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded point payload key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointPayloadKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The point ID.
    pub point_id: PointId,
}

/// Decode a point payload key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_point_payload_key(key: &[u8]) -> Option<PointPayloadKey> {
    if key.len() != 17 || key[0] != PREFIX_POINT_PAYLOAD {
        return None;
    }
    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let point_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    Some(PointPayloadKey {
        collection_name_hash: u64::from_be_bytes(name_hash_bytes),
        point_id: PointId::new(u64::from_be_bytes(point_id_bytes)),
    })
}

/// Decode just the point ID from a point payload key.
#[must_use]
pub fn decode_point_payload_point_id(key: &[u8]) -> Option<PointId> {
    decode_point_payload_key(key).map(|k| k.point_id)
}

// ============================================================================
// Dense vector keys
// ============================================================================

/// Encode a key for a point's dense vector.
///
/// Key format: `[PREFIX_POINT_DENSE_VECTOR][collection_name_hash][point_id][vector_name_hash]`
#[must_use]
pub fn encode_dense_vector_key(
    collection_name: &str,
    point_id: PointId,
    vector_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_POINT_DENSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all dense vectors for a point.
#[must_use]
pub fn encode_dense_vector_point_prefix(collection_name: &str, point_id: PointId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POINT_DENSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all dense vectors in a collection.
#[must_use]
pub fn encode_dense_vector_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POINT_DENSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded dense vector key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseVectorKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The point ID.
    pub point_id: PointId,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
}

/// Decode a dense vector key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_dense_vector_key(key: &[u8]) -> Option<DenseVectorKey> {
    if key.len() != 25 || key[0] != PREFIX_POINT_DENSE_VECTOR {
        return None;
    }
    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let point_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let vector_name_hash_bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(DenseVectorKey {
        collection_name_hash: u64::from_be_bytes(name_hash_bytes),
        point_id: PointId::new(u64::from_be_bytes(point_id_bytes)),
        vector_name_hash: u64::from_be_bytes(vector_name_hash_bytes),
    })
}

// ============================================================================
// Sparse vector keys
// ============================================================================

/// Encode a key for a point's sparse vector.
///
/// Key format: `[PREFIX_POINT_SPARSE_VECTOR][collection_name_hash][point_id][vector_name_hash]`
#[must_use]
pub fn encode_sparse_vector_key(
    collection_name: &str,
    point_id: PointId,
    vector_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_POINT_SPARSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all sparse vectors for a point.
#[must_use]
pub fn encode_sparse_vector_point_prefix(collection_name: &str, point_id: PointId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POINT_SPARSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all sparse vectors in a collection.
#[must_use]
pub fn encode_sparse_vector_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POINT_SPARSE_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded sparse vector key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseVectorKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The point ID.
    pub point_id: PointId,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
}

/// Decode a sparse vector key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_sparse_vector_key(key: &[u8]) -> Option<SparseVectorKey> {
    if key.len() != 25 || key[0] != PREFIX_POINT_SPARSE_VECTOR {
        return None;
    }
    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let point_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let vector_name_hash_bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(SparseVectorKey {
        collection_name_hash: u64::from_be_bytes(name_hash_bytes),
        point_id: PointId::new(u64::from_be_bytes(point_id_bytes)),
        vector_name_hash: u64::from_be_bytes(vector_name_hash_bytes),
    })
}

// ============================================================================
// Multi-vector keys
// ============================================================================

/// Encode a key for a point's multi-vector.
///
/// Key format: `[PREFIX_POINT_MULTI_VECTOR][collection_name_hash][point_id][vector_name_hash]`
#[must_use]
pub fn encode_multi_vector_key(
    collection_name: &str,
    point_id: PointId,
    vector_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_POINT_MULTI_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all multi-vectors for a point.
#[must_use]
pub fn encode_multi_vector_point_prefix(collection_name: &str, point_id: PointId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POINT_MULTI_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all multi-vectors in a collection.
#[must_use]
pub fn encode_multi_vector_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POINT_MULTI_VECTOR);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded multi-vector key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiVectorKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The point ID.
    pub point_id: PointId,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
}

/// Decode a multi-vector key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_multi_vector_key(key: &[u8]) -> Option<MultiVectorKey> {
    if key.len() != 25 || key[0] != PREFIX_POINT_MULTI_VECTOR {
        return None;
    }
    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let point_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let vector_name_hash_bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(MultiVectorKey {
        collection_name_hash: u64::from_be_bytes(name_hash_bytes),
        point_id: PointId::new(u64::from_be_bytes(point_id_bytes)),
        vector_name_hash: u64::from_be_bytes(vector_name_hash_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_key_roundtrip() {
        let key = encode_collection_key("documents");
        assert_eq!(key.len(), 9);
        assert_eq!(key[0], PREFIX_COLLECTION);

        let decoded = decode_collection_key(&key).unwrap();
        assert_eq!(decoded.name_hash, hash_name("documents"));
    }

    #[test]
    fn point_payload_key_roundtrip() {
        let collection = "documents";
        let point_id = PointId::new(42);
        let key = encode_point_payload_key(collection, point_id);

        assert_eq!(key.len(), 17);
        assert_eq!(key[0], PREFIX_POINT_PAYLOAD);

        let decoded = decode_point_payload_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.point_id, point_id);
    }

    #[test]
    fn dense_vector_key_roundtrip() {
        let collection = "documents";
        let point_id = PointId::new(123);
        let vector_name = "text_embedding";
        let key = encode_dense_vector_key(collection, point_id, vector_name);

        assert_eq!(key.len(), 25);
        assert_eq!(key[0], PREFIX_POINT_DENSE_VECTOR);

        let decoded = decode_dense_vector_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.point_id, point_id);
        assert_eq!(decoded.vector_name_hash, hash_name(vector_name));
    }

    #[test]
    fn sparse_vector_key_roundtrip() {
        let collection = "documents";
        let point_id = PointId::new(456);
        let vector_name = "sparse_embedding";
        let key = encode_sparse_vector_key(collection, point_id, vector_name);

        assert_eq!(key.len(), 25);
        assert_eq!(key[0], PREFIX_POINT_SPARSE_VECTOR);

        let decoded = decode_sparse_vector_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.point_id, point_id);
        assert_eq!(decoded.vector_name_hash, hash_name(vector_name));
    }

    #[test]
    fn multi_vector_key_roundtrip() {
        let collection = "documents";
        let point_id = PointId::new(789);
        let vector_name = "colbert";
        let key = encode_multi_vector_key(collection, point_id, vector_name);

        assert_eq!(key.len(), 25);
        assert_eq!(key[0], PREFIX_POINT_MULTI_VECTOR);

        let decoded = decode_multi_vector_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.point_id, point_id);
        assert_eq!(decoded.vector_name_hash, hash_name(vector_name));
    }

    #[test]
    fn keys_ordered_within_collection() {
        let collection = "test";

        let key1 = encode_point_payload_key(collection, PointId::new(1));
        let key2 = encode_point_payload_key(collection, PointId::new(2));
        let key3 = encode_point_payload_key(collection, PointId::new(100));

        // Points should be ordered by ID within a collection
        assert!(key1 < key2);
        assert!(key2 < key3);
    }

    #[test]
    fn prefix_matches_keys() {
        let collection = "documents";
        let prefix = encode_point_payload_prefix(collection);

        let key1 = encode_point_payload_key(collection, PointId::new(1));
        let key2 = encode_point_payload_key(collection, PointId::new(1000));

        // All keys in collection should start with the prefix
        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));

        // Different collection should not match
        let key_other = encode_point_payload_key("other_collection", PointId::new(1));
        assert!(!key_other.starts_with(&prefix));
    }

    #[test]
    fn vector_key_prefix_matches() {
        let collection = "docs";
        let point_id = PointId::new(42);

        let prefix = encode_dense_vector_point_prefix(collection, point_id);
        let key1 = encode_dense_vector_key(collection, point_id, "embedding1");
        let key2 = encode_dense_vector_key(collection, point_id, "embedding2");

        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));

        // Different point should not match
        let key_other = encode_dense_vector_key(collection, PointId::new(99), "embedding1");
        assert!(!key_other.starts_with(&prefix));
    }

    #[test]
    fn decode_invalid_keys() {
        // Wrong prefix
        assert!(decode_collection_key(&[PREFIX_POINT_PAYLOAD; 9]).is_none());
        assert!(decode_point_payload_key(&[PREFIX_COLLECTION; 17]).is_none());
        assert!(decode_dense_vector_key(&[PREFIX_POINT_SPARSE_VECTOR; 25]).is_none());

        // Wrong length
        assert!(decode_collection_key(&[PREFIX_COLLECTION; 5]).is_none());
        assert!(decode_point_payload_key(&[PREFIX_POINT_PAYLOAD; 10]).is_none());
        assert!(decode_dense_vector_key(&[PREFIX_POINT_DENSE_VECTOR; 20]).is_none());

        // Empty
        assert!(decode_collection_key(&[]).is_none());
        assert!(decode_point_payload_key(&[]).is_none());
        assert!(decode_dense_vector_key(&[]).is_none());
    }
}
