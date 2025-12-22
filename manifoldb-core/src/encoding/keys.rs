//! Key encoding for ordered storage.
//!
//! This module provides key encoding that preserves sort order for range queries
//! in key-value storage backends. Keys are designed to support efficient
//! prefix-based range scans.
//!
//! # Key Prefixes
//!
//! Different data types use different key prefixes to partition the keyspace:
//!
//! - `0x01` - Entity keys: `[0x01][entity_id]`
//! - `0x02` - Edge keys: `[0x02][edge_id]`
//! - `0x03` - Edge by source: `[0x03][source_id][edge_type_hash][edge_id]`
//! - `0x04` - Edge by target: `[0x04][target_id][edge_type_hash][edge_id]`
//! - `0x05` - Label index: `[0x05][label_hash][entity_id]`
//! - `0x06` - Edge type index: `[0x06][edge_type_hash][edge_id]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.

use crate::types::{EdgeId, EdgeType, EntityId, Label};

/// Key prefix for entity data.
pub const PREFIX_ENTITY: u8 = 0x01;
/// Key prefix for edge data.
pub const PREFIX_EDGE: u8 = 0x02;
/// Key prefix for edges indexed by source entity.
pub const PREFIX_EDGE_BY_SOURCE: u8 = 0x03;
/// Key prefix for edges indexed by target entity.
pub const PREFIX_EDGE_BY_TARGET: u8 = 0x04;
/// Key prefix for label index.
pub const PREFIX_LABEL_INDEX: u8 = 0x05;
/// Key prefix for edge type index.
pub const PREFIX_EDGE_TYPE_INDEX: u8 = 0x06;

/// Encode an entity ID as a storage key.
///
/// The key format is: `[PREFIX_ENTITY][entity_id as big-endian u64]`
#[inline]
#[must_use]
pub fn encode_entity_key(id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_ENTITY);
    key.extend_from_slice(&id.as_u64().to_be_bytes());
    key
}

/// Encode an edge ID as a storage key.
///
/// The key format is: `[PREFIX_EDGE][edge_id as big-endian u64]`
#[inline]
#[must_use]
pub fn encode_edge_key(id: EdgeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EDGE);
    key.extend_from_slice(&id.as_u64().to_be_bytes());
    key
}

/// Compute a hash for a string that preserves some ordering properties.
///
/// This uses a simple FNV-1a hash for efficient computation.
#[inline]
#[must_use]
fn hash_string(s: &str) -> u64 {
    // FNV-1a hash
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in s.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Encode a key for looking up edges by source entity.
///
/// The key format is: `[PREFIX_EDGE_BY_SOURCE][source_id][edge_type_hash][edge_id]`
///
/// This enables efficient range scans for "all edges from entity X" or
/// "all edges of type Y from entity X".
#[must_use]
pub fn encode_edge_by_source_key(
    source: EntityId,
    edge_type: &EdgeType,
    edge_id: EdgeId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_EDGE_BY_SOURCE);
    key.extend_from_slice(&source.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key.extend_from_slice(&edge_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning edges by source entity.
///
/// Returns a key that can be used as the start of a range scan
/// for all edges from the given source entity.
#[inline]
#[must_use]
pub fn encode_edge_by_source_prefix(source: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EDGE_BY_SOURCE);
    key.extend_from_slice(&source.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning edges by source entity and type.
///
/// Returns a key that can be used as the start of a range scan
/// for all edges of a specific type from the given source entity.
#[must_use]
pub fn encode_edge_by_source_type_prefix(source: EntityId, edge_type: &EdgeType) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_EDGE_BY_SOURCE);
    key.extend_from_slice(&source.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key
}

/// Encode a key for looking up edges by target entity.
///
/// The key format is: `[PREFIX_EDGE_BY_TARGET][target_id][edge_type_hash][edge_id]`
#[must_use]
pub fn encode_edge_by_target_key(
    target: EntityId,
    edge_type: &EdgeType,
    edge_id: EdgeId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(25);
    key.push(PREFIX_EDGE_BY_TARGET);
    key.extend_from_slice(&target.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key.extend_from_slice(&edge_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning edges by target entity.
#[inline]
#[must_use]
pub fn encode_edge_by_target_prefix(target: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EDGE_BY_TARGET);
    key.extend_from_slice(&target.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning edges by target entity and type.
#[must_use]
pub fn encode_edge_by_target_type_prefix(target: EntityId, edge_type: &EdgeType) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_EDGE_BY_TARGET);
    key.extend_from_slice(&target.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key
}

/// Encode a key for the label index.
///
/// The key format is: `[PREFIX_LABEL_INDEX][label_hash][entity_id]`
#[must_use]
pub fn encode_label_index_key(label: &Label, entity_id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_LABEL_INDEX);
    key.extend_from_slice(&hash_string(label.as_str()).to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning entities by label.
#[must_use]
pub fn encode_label_index_prefix(label: &Label) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_LABEL_INDEX);
    key.extend_from_slice(&hash_string(label.as_str()).to_be_bytes());
    key
}

/// Encode a key for the edge type index.
///
/// The key format is: `[PREFIX_EDGE_TYPE_INDEX][edge_type_hash][edge_id]`
#[must_use]
pub fn encode_edge_type_index_key(edge_type: &EdgeType, edge_id: EdgeId) -> Vec<u8> {
    let mut key = Vec::with_capacity(17);
    key.push(PREFIX_EDGE_TYPE_INDEX);
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key.extend_from_slice(&edge_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning edges by type.
#[must_use]
pub fn encode_edge_type_index_prefix(edge_type: &EdgeType) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_EDGE_TYPE_INDEX);
    key.extend_from_slice(&hash_string(edge_type.as_str()).to_be_bytes());
    key
}

/// Decode an entity ID from an entity key.
///
/// Returns `None` if the key doesn't have the correct format.
#[inline]
#[must_use]
pub fn decode_entity_key(key: &[u8]) -> Option<EntityId> {
    if key.len() != 9 || key[0] != PREFIX_ENTITY {
        return None;
    }
    let bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(EntityId::new(u64::from_be_bytes(bytes)))
}

/// Decode an edge ID from an edge key.
///
/// Returns `None` if the key doesn't have the correct format.
#[inline]
#[must_use]
pub fn decode_edge_key(key: &[u8]) -> Option<EdgeId> {
    if key.len() != 9 || key[0] != PREFIX_EDGE {
        return None;
    }
    let bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(EdgeId::new(u64::from_be_bytes(bytes)))
}

/// Decode an entity ID from a label index key.
///
/// Returns `None` if the key doesn't have the correct format.
#[inline]
#[must_use]
pub fn decode_label_index_entity_id(key: &[u8]) -> Option<EntityId> {
    if key.len() != 17 || key[0] != PREFIX_LABEL_INDEX {
        return None;
    }
    let bytes: [u8; 8] = key[9..17].try_into().ok()?;
    Some(EntityId::new(u64::from_be_bytes(bytes)))
}

/// Decode an edge ID from an edge-by-source key.
///
/// Returns `None` if the key doesn't have the correct format.
#[inline]
#[must_use]
pub fn decode_edge_by_source_edge_id(key: &[u8]) -> Option<EdgeId> {
    if key.len() != 25 || key[0] != PREFIX_EDGE_BY_SOURCE {
        return None;
    }
    let bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(EdgeId::new(u64::from_be_bytes(bytes)))
}

/// Decode an edge ID from an edge-by-target key.
///
/// Returns `None` if the key doesn't have the correct format.
#[inline]
#[must_use]
pub fn decode_edge_by_target_edge_id(key: &[u8]) -> Option<EdgeId> {
    if key.len() != 25 || key[0] != PREFIX_EDGE_BY_TARGET {
        return None;
    }
    let bytes: [u8; 8] = key[17..25].try_into().ok()?;
    Some(EdgeId::new(u64::from_be_bytes(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_key_roundtrip() {
        for id in [0u64, 1, 42, u64::MAX] {
            let entity_id = EntityId::new(id);
            let key = encode_entity_key(entity_id);
            let decoded = decode_entity_key(&key);
            assert_eq!(decoded, Some(entity_id));
        }
    }

    #[test]
    fn edge_key_roundtrip() {
        for id in [0u64, 1, 42, u64::MAX] {
            let edge_id = EdgeId::new(id);
            let key = encode_edge_key(edge_id);
            let decoded = decode_edge_key(&key);
            assert_eq!(decoded, Some(edge_id));
        }
    }

    #[test]
    fn entity_keys_are_ordered() {
        let key1 = encode_entity_key(EntityId::new(1));
        let key2 = encode_entity_key(EntityId::new(2));
        let key3 = encode_entity_key(EntityId::new(100));
        assert!(key1 < key2);
        assert!(key2 < key3);
    }

    #[test]
    fn edge_keys_are_ordered() {
        let key1 = encode_edge_key(EdgeId::new(1));
        let key2 = encode_edge_key(EdgeId::new(2));
        let key3 = encode_edge_key(EdgeId::new(100));
        assert!(key1 < key2);
        assert!(key2 < key3);
    }

    #[test]
    fn edge_by_source_keys_group_by_source() {
        let edge_type = EdgeType::new("FOLLOWS");
        let key1 = encode_edge_by_source_key(EntityId::new(1), &edge_type, EdgeId::new(100));
        let key2 = encode_edge_by_source_key(EntityId::new(1), &edge_type, EdgeId::new(200));
        let key3 = encode_edge_by_source_key(EntityId::new(2), &edge_type, EdgeId::new(50));

        // Keys from the same source should be grouped together
        let prefix1 = encode_edge_by_source_prefix(EntityId::new(1));
        assert!(key1.starts_with(&prefix1));
        assert!(key2.starts_with(&prefix1));
        assert!(!key3.starts_with(&prefix1));

        // Keys are ordered: source 1 edges come before source 2 edges
        assert!(key1 < key3);
        assert!(key2 < key3);
    }

    #[test]
    fn label_index_keys_group_by_label() {
        let label = Label::new("Person");
        let key1 = encode_label_index_key(&label, EntityId::new(10));
        let key2 = encode_label_index_key(&label, EntityId::new(20));

        let prefix = encode_label_index_prefix(&label);
        assert!(key1.starts_with(&prefix));
        assert!(key2.starts_with(&prefix));
    }

    #[test]
    fn decode_invalid_entity_key() {
        // Wrong prefix
        assert_eq!(decode_entity_key(&[PREFIX_EDGE, 0, 0, 0, 0, 0, 0, 0, 1]), None);
        // Wrong length
        assert_eq!(decode_entity_key(&[PREFIX_ENTITY, 0, 0, 0]), None);
        // Empty
        assert_eq!(decode_entity_key(&[]), None);
    }

    #[test]
    fn key_prefixes_partition_keyspace() {
        let entity_key = encode_entity_key(EntityId::new(1));
        let edge_key = encode_edge_key(EdgeId::new(1));
        let edge_by_source =
            encode_edge_by_source_key(EntityId::new(1), &EdgeType::new("X"), EdgeId::new(1));

        // Different prefixes ensure keys don't collide
        assert!(entity_key[0] != edge_key[0]);
        assert!(entity_key[0] != edge_by_source[0]);
        assert!(edge_key[0] != edge_by_source[0]);
    }
}
