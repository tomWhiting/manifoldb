//! Key encoding for inverted index storage.
//!
//! This module provides key encoding functions for sparse vector inverted indexes,
//! supporting efficient posting list lookups and range scans.
//!
//! # Key Format
//!
//! All keys use big-endian encoding for proper sort order in range scans.
//!
//! ## Posting list entry
//! `[PREFIX_POSTING][collection_hash][vector_name_hash][token_id]` → `[(point_id, weight), ...]`
//!
//! ## Index metadata
//! `[PREFIX_INVERTED_META][collection_hash][vector_name_hash]` → `InvertedIndexMeta`
//!
//! ## Point tokens (reverse mapping for deletion)
//! `[PREFIX_POINT_TOKENS][collection_hash][vector_name_hash][point_id]` → `[token_ids]`

use manifoldb_core::PointId;

use super::point_keys::hash_name;

/// Key prefix for posting lists.
pub const PREFIX_POSTING: u8 = 0x30;

/// Key prefix for inverted index metadata.
pub const PREFIX_INVERTED_META: u8 = 0x31;

/// Key prefix for point-to-tokens reverse mapping.
pub const PREFIX_POINT_TOKENS: u8 = 0x32;

// ============================================================================
// Posting list keys
// ============================================================================

/// Encode a key for a posting list entry.
///
/// Key format: `[PREFIX_POSTING][collection_hash][vector_name_hash][token_id]`
#[must_use]
pub fn encode_posting_key(collection_name: &str, vector_name: &str, token_id: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(21);
    key.push(PREFIX_POSTING);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key.extend_from_slice(&token_id.to_be_bytes());
    key
}

/// Encode a prefix for scanning all posting lists for a vector.
#[must_use]
pub fn encode_posting_prefix(collection_name: &str, vector_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POSTING);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all posting lists in a collection.
#[must_use]
pub fn encode_posting_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POSTING);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded posting key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
    /// The token ID.
    pub token_id: u32,
}

/// Decode a posting key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_posting_key(key: &[u8]) -> Option<PostingKey> {
    if key.len() != 21 || key[0] != PREFIX_POSTING {
        return None;
    }
    let collection_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let vector_hash_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let token_id_bytes: [u8; 4] = key[17..21].try_into().ok()?;
    Some(PostingKey {
        collection_name_hash: u64::from_be_bytes(collection_hash_bytes),
        vector_name_hash: u64::from_be_bytes(vector_hash_bytes),
        token_id: u32::from_be_bytes(token_id_bytes),
    })
}

// ============================================================================
// Index metadata keys
// ============================================================================

/// Encode a key for inverted index metadata.
///
/// Key format: `[PREFIX_INVERTED_META][collection_hash][vector_name_hash]`
#[must_use]
pub fn encode_inverted_meta_key(collection_name: &str, vector_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_INVERTED_META);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all inverted indexes in a collection.
#[must_use]
pub fn encode_inverted_meta_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_INVERTED_META);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded inverted meta key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvertedMetaKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
}

/// Decode an inverted meta key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_inverted_meta_key(key: &[u8]) -> Option<InvertedMetaKey> {
    if key.len() != 17 || key[0] != PREFIX_INVERTED_META {
        return None;
    }
    let collection_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let vector_hash_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    Some(InvertedMetaKey {
        collection_name_hash: u64::from_be_bytes(collection_hash_bytes),
        vector_name_hash: u64::from_be_bytes(vector_hash_bytes),
    })
}

// ============================================================================
// Point tokens keys (reverse mapping)
// ============================================================================

/// Encode a key for point-to-tokens reverse mapping.
///
/// Key format: `[PREFIX_POINT_TOKENS][collection_hash][vector_name_hash][point_id]`
#[must_use]
pub fn encode_point_tokens_key(
    collection_name: &str,
    vector_name: &str,
    point_id: PointId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_POINT_TOKENS);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key.extend_from_slice(&point_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all point tokens for a vector.
#[must_use]
pub fn encode_point_tokens_prefix(collection_name: &str, vector_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_POINT_TOKENS);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key.extend_from_slice(&hash_name(vector_name).to_be_bytes());
    key
}

/// Encode a prefix for scanning all point tokens in a collection.
#[must_use]
pub fn encode_point_tokens_collection_prefix(collection_name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_POINT_TOKENS);
    key.extend_from_slice(&hash_name(collection_name).to_be_bytes());
    key
}

/// A decoded point tokens key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointTokensKey {
    /// The hash of the collection name.
    pub collection_name_hash: u64,
    /// The hash of the vector name.
    pub vector_name_hash: u64,
    /// The point ID.
    pub point_id: PointId,
}

/// Decode a point tokens key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_point_tokens_key(key: &[u8]) -> Option<PointTokensKey> {
    if key.len() != 25 || key[0] != PREFIX_POINT_TOKENS {
        return None;
    }
    let collection_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let vector_hash_bytes: [u8; 8] = key[9..17].try_into().ok()?;
    let point_id_bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(PointTokensKey {
        collection_name_hash: u64::from_be_bytes(collection_hash_bytes),
        vector_name_hash: u64::from_be_bytes(vector_hash_bytes),
        point_id: PointId::new(u64::from_be_bytes(point_id_bytes)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posting_key_roundtrip() {
        let collection = "documents";
        let vector = "keywords";
        let token_id = 12345u32;

        let key = encode_posting_key(collection, vector, token_id);
        assert_eq!(key.len(), 21);
        assert_eq!(key[0], PREFIX_POSTING);

        let decoded = decode_posting_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.vector_name_hash, hash_name(vector));
        assert_eq!(decoded.token_id, token_id);
    }

    #[test]
    fn posting_keys_sorted_by_token_id() {
        let key1 = encode_posting_key("col", "vec", 1);
        let key2 = encode_posting_key("col", "vec", 2);
        let key3 = encode_posting_key("col", "vec", 100);

        assert!(key1 < key2);
        assert!(key2 < key3);
    }

    #[test]
    fn posting_prefix_matches() {
        let collection = "documents";
        let vector = "keywords";
        let prefix = encode_posting_prefix(collection, vector);

        let key1 = encode_posting_key(collection, vector, 1);
        let key2 = encode_posting_key(collection, vector, 1000);

        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));

        // Different vector should not match
        let key_other = encode_posting_key(collection, "other_vector", 1);
        assert!(!key_other.starts_with(&prefix));
    }

    #[test]
    fn inverted_meta_key_roundtrip() {
        let collection = "documents";
        let vector = "keywords";

        let key = encode_inverted_meta_key(collection, vector);
        assert_eq!(key.len(), 17);
        assert_eq!(key[0], PREFIX_INVERTED_META);

        let decoded = decode_inverted_meta_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.vector_name_hash, hash_name(vector));
    }

    #[test]
    fn point_tokens_key_roundtrip() {
        let collection = "documents";
        let vector = "keywords";
        let point_id = PointId::new(42);

        let key = encode_point_tokens_key(collection, vector, point_id);
        assert_eq!(key.len(), 25);
        assert_eq!(key[0], PREFIX_POINT_TOKENS);

        let decoded = decode_point_tokens_key(&key).unwrap();
        assert_eq!(decoded.collection_name_hash, hash_name(collection));
        assert_eq!(decoded.vector_name_hash, hash_name(vector));
        assert_eq!(decoded.point_id, point_id);
    }

    #[test]
    fn decode_invalid_keys() {
        // Wrong prefix
        assert!(decode_posting_key(&[PREFIX_INVERTED_META; 21]).is_none());
        assert!(decode_inverted_meta_key(&[PREFIX_POSTING; 17]).is_none());
        assert!(decode_point_tokens_key(&[PREFIX_POSTING; 25]).is_none());

        // Wrong length
        assert!(decode_posting_key(&[PREFIX_POSTING; 10]).is_none());
        assert!(decode_inverted_meta_key(&[PREFIX_INVERTED_META; 10]).is_none());
        assert!(decode_point_tokens_key(&[PREFIX_POINT_TOKENS; 10]).is_none());

        // Empty
        assert!(decode_posting_key(&[]).is_none());
        assert!(decode_inverted_meta_key(&[]).is_none());
        assert!(decode_point_tokens_key(&[]).is_none());
    }
}
