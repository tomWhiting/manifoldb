//! Key encoding for vector storage.

use manifoldb_core::EntityId;

use crate::types::EmbeddingName;

/// Key prefix for embedding space metadata.
pub const PREFIX_EMBEDDING_SPACE: u8 = 0x10;

/// Key prefix for entity embeddings.
pub const PREFIX_EMBEDDING: u8 = 0x11;

/// Key prefix for sparse embedding space metadata.
pub const PREFIX_SPARSE_EMBEDDING_SPACE: u8 = 0x12;

/// Key prefix for sparse entity embeddings.
pub const PREFIX_SPARSE_EMBEDDING: u8 = 0x13;

/// Key prefix for multi-vector embedding space metadata.
pub const PREFIX_MULTI_VECTOR_SPACE: u8 = 0x14;

/// Key prefix for multi-vector entity embeddings.
pub const PREFIX_MULTI_VECTOR: u8 = 0x15;

/// Compute a hash for an embedding name.
///
/// Uses FNV-1a hash for efficient computation.
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

/// Encode a key for embedding space metadata.
///
/// Key format: `[PREFIX_EMBEDDING_SPACE][name_hash]`
#[must_use]
pub fn encode_embedding_space_key(name: &EmbeddingName) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EMBEDDING_SPACE);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key
}

/// Decode an embedding space name hash from a space key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_embedding_space_key(key: &[u8]) -> Option<u64> {
    if key.len() != 9 || key[0] != PREFIX_EMBEDDING_SPACE {
        return None;
    }
    let bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

/// Encode a key for an entity's embedding in a space.
///
/// Key format: `[PREFIX_EMBEDDING][name_hash][entity_id]`
///
/// This enables efficient range scans for "all embeddings in space X" or
/// "embedding for entity Y in space X".
#[must_use]
pub fn encode_embedding_key(name: &EmbeddingName, entity_id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_EMBEDDING);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all embeddings in a space.
///
/// Returns a key that can be used as the start of a range scan
/// for all embeddings in the given space.
#[must_use]
pub fn encode_embedding_prefix(name: &EmbeddingName) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EMBEDDING);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key
}

/// A decoded embedding key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingKey {
    /// The hash of the embedding space name.
    pub name_hash: u64,
    /// The entity ID.
    pub entity_id: EntityId,
}

/// Decode an embedding key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_embedding_key(key: &[u8]) -> Option<EmbeddingKey> {
    if key.len() != 17 || key[0] != PREFIX_EMBEDDING {
        return None;
    }

    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let entity_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;

    Some(EmbeddingKey {
        name_hash: u64::from_be_bytes(name_hash_bytes),
        entity_id: EntityId::new(u64::from_be_bytes(entity_id_bytes)),
    })
}

/// Decode an entity ID from an embedding key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_embedding_entity_id(key: &[u8]) -> Option<EntityId> {
    decode_embedding_key(key).map(|k| k.entity_id)
}

// --- Sparse embedding keys ---

/// Encode a key for sparse embedding space metadata.
///
/// Key format: `[PREFIX_SPARSE_EMBEDDING_SPACE][name_hash]`
#[must_use]
pub fn encode_sparse_embedding_space_key(name: &EmbeddingName) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_SPARSE_EMBEDDING_SPACE);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key
}

/// Decode a sparse embedding space name hash from a space key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_sparse_embedding_space_key(key: &[u8]) -> Option<u64> {
    if key.len() != 9 || key[0] != PREFIX_SPARSE_EMBEDDING_SPACE {
        return None;
    }
    let bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

/// Encode a key for an entity's sparse embedding in a space.
///
/// Key format: `[PREFIX_SPARSE_EMBEDDING][name_hash][entity_id]`
#[must_use]
pub fn encode_sparse_embedding_key(name: &EmbeddingName, entity_id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_SPARSE_EMBEDDING);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning all sparse embeddings in a space.
#[must_use]
pub fn encode_sparse_embedding_prefix(name: &EmbeddingName) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_SPARSE_EMBEDDING);
    key.extend_from_slice(&hash_name(name.as_str()).to_be_bytes());
    key
}

/// A decoded sparse embedding key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseEmbeddingKey {
    /// The hash of the embedding space name.
    pub name_hash: u64,
    /// The entity ID.
    pub entity_id: EntityId,
}

/// Decode a sparse embedding key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_sparse_embedding_key(key: &[u8]) -> Option<SparseEmbeddingKey> {
    if key.len() != 17 || key[0] != PREFIX_SPARSE_EMBEDDING {
        return None;
    }

    let name_hash_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    let entity_id_bytes: [u8; 8] = key[9..17].try_into().ok()?;

    Some(SparseEmbeddingKey {
        name_hash: u64::from_be_bytes(name_hash_bytes),
        entity_id: EntityId::new(u64::from_be_bytes(entity_id_bytes)),
    })
}

/// Decode an entity ID from a sparse embedding key.
///
/// Returns `None` if the key doesn't have the correct format.
#[must_use]
pub fn decode_sparse_embedding_entity_id(key: &[u8]) -> Option<EntityId> {
    decode_sparse_embedding_key(key).map(|k| k.entity_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_space_key_roundtrip() {
        let name = EmbeddingName::new("test_space").unwrap();
        let key = encode_embedding_space_key(&name);

        assert_eq!(key.len(), 9);
        assert_eq!(key[0], PREFIX_EMBEDDING_SPACE);

        let decoded = decode_embedding_space_key(&key);
        assert_eq!(decoded, Some(hash_name("test_space")));
    }

    #[test]
    fn embedding_key_roundtrip() {
        let name = EmbeddingName::new("text_embedding").unwrap();
        let entity_id = EntityId::new(12345);

        let key = encode_embedding_key(&name, entity_id);
        assert_eq!(key.len(), 17);
        assert_eq!(key[0], PREFIX_EMBEDDING);

        let decoded = decode_embedding_key(&key).unwrap();
        assert_eq!(decoded.name_hash, hash_name("text_embedding"));
        assert_eq!(decoded.entity_id, entity_id);
    }

    #[test]
    fn embedding_keys_ordered_by_space_then_entity() {
        let space1 = EmbeddingName::new("aaa").unwrap();
        let space2 = EmbeddingName::new("zzz").unwrap();

        let key1 = encode_embedding_key(&space1, EntityId::new(1));
        let key2 = encode_embedding_key(&space1, EntityId::new(2));
        let key3 = encode_embedding_key(&space2, EntityId::new(1));

        // Same space: ordered by entity ID
        assert!(key1 < key2);

        // Note: Different spaces may not be lexicographically ordered by name
        // due to hashing, but embeddings within a space ARE ordered by entity ID
        let prefix1 = encode_embedding_prefix(&space1);
        assert!(key1.starts_with(&prefix1));
        assert!(key2.starts_with(&prefix1));
        assert!(!key3.starts_with(&prefix1));
    }

    #[test]
    fn embedding_prefix_groups_embeddings_by_space() {
        let name = EmbeddingName::new("my_space").unwrap();
        let prefix = encode_embedding_prefix(&name);

        for id in [0u64, 1, 100, u64::MAX] {
            let key = encode_embedding_key(&name, EntityId::new(id));
            assert!(key.starts_with(&prefix));
        }
    }

    #[test]
    fn decode_invalid_embedding_key() {
        // Wrong prefix
        assert!(decode_embedding_key(&[PREFIX_EMBEDDING_SPACE; 17]).is_none());

        // Wrong length
        assert!(decode_embedding_key(&[PREFIX_EMBEDDING; 10]).is_none());

        // Empty
        assert!(decode_embedding_key(&[]).is_none());
    }

    #[test]
    fn decode_invalid_space_key() {
        // Wrong prefix
        assert!(decode_embedding_space_key(&[PREFIX_EMBEDDING; 9]).is_none());

        // Wrong length
        assert!(decode_embedding_space_key(&[PREFIX_EMBEDDING_SPACE; 5]).is_none());

        // Empty
        assert!(decode_embedding_space_key(&[]).is_none());
    }

    #[test]
    fn key_prefixes_partition_keyspace() {
        let name = EmbeddingName::new("test").unwrap();
        let space_key = encode_embedding_space_key(&name);
        let embedding_key = encode_embedding_key(&name, EntityId::new(1));

        assert!(space_key[0] != embedding_key[0]);
    }

    #[test]
    fn hash_consistency() {
        // Same name should produce same hash
        assert_eq!(hash_name("test"), hash_name("test"));

        // Different names should (almost certainly) produce different hashes
        assert_ne!(hash_name("test1"), hash_name("test2"));
    }
}
