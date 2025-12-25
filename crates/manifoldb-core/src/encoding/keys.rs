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
//! - `0x07` - Property index: `[0x07][index_id][sortable_value][entity_id]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.
//!
//! # Property Index Keys
//!
//! Property indexes use a composite key that enables efficient range queries:
//!
//! ```text
//! [PREFIX_PROPERTY_INDEX][index_id: u64][sortable_value: bytes][entity_id: u64]
//! ```
//!
//! - `index_id`: Unique identifier for the index (combines label + property name)
//! - `sortable_value`: Value encoded using sort-order preserving encoding
//! - `entity_id`: The entity that has this property value
//!
//! This layout enables:
//! - Point lookups: `WHERE property = value`
//! - Range scans: `WHERE property > value`, `WHERE property BETWEEN a AND b`
//! - Prefix scans: `WHERE property LIKE 'prefix%'`

use crate::types::{EdgeId, EdgeType, EntityId, Label, Value};

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
/// Key prefix for property index.
pub const PREFIX_PROPERTY_INDEX: u8 = 0x07;

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

// ============================================================================
// Property Index Keys
// ============================================================================

/// An index identifier combining label and property name.
///
/// This is used to partition the property index keyspace so that
/// different indexes don't interfere with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexId(u64);

impl IndexId {
    /// Create a new index ID from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Create an index ID from a label and property name.
    ///
    /// This combines the hash of both the label and property name
    /// to create a unique identifier for the index.
    #[must_use]
    pub fn from_label_property(label: &str, property: &str) -> Self {
        // Combine hashes with a separator to avoid collisions
        // e.g., ("ab", "c") vs ("a", "bc")
        let label_hash = hash_string(label);
        let prop_hash = hash_string(property);
        // XOR with rotation to mix the hashes
        let combined = label_hash ^ prop_hash.rotate_left(32);
        Self(combined)
    }

    /// Get the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Encode a property index key.
///
/// The key format is: `[PREFIX_PROPERTY_INDEX][index_id][sortable_value][entity_id]`
///
/// This enables efficient range scans on property values within an index.
///
/// # Arguments
///
/// * `index_id` - The index identifier (from label + property name)
/// * `value` - The property value (must be sortable-encodable)
/// * `entity_id` - The entity ID that has this property value
///
/// # Errors
///
/// Returns `None` if the value cannot be sortable-encoded (e.g., vectors, arrays).
///
/// # Example
///
/// ```
/// use manifoldb_core::encoding::keys::{encode_property_index_key, IndexId};
/// use manifoldb_core::types::{EntityId, Value};
///
/// let index_id = IndexId::from_label_property("Person", "age");
/// let value = Value::Int(30);
/// let entity_id = EntityId::new(42);
///
/// let key = encode_property_index_key(index_id, &value, entity_id).unwrap();
/// ```
#[must_use]
pub fn encode_property_index_key(
    index_id: IndexId,
    value: &Value,
    entity_id: EntityId,
) -> Option<Vec<u8>> {
    use super::sortable::encode_sortable;

    let sortable_value = encode_sortable(value).ok()?;
    let mut key = Vec::with_capacity(1 + 8 + sortable_value.len() + 8);
    key.push(PREFIX_PROPERTY_INDEX);
    key.extend_from_slice(&index_id.as_u64().to_be_bytes());
    key.extend_from_slice(&sortable_value);
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    Some(key)
}

/// Encode a prefix for scanning all entries in a property index.
///
/// Returns a key that can be used as the start of a range scan
/// for all entries in the given index.
#[inline]
#[must_use]
pub fn encode_property_index_prefix(index_id: IndexId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_PROPERTY_INDEX);
    key.extend_from_slice(&index_id.as_u64().to_be_bytes());
    key
}

/// Encode a prefix for scanning property index entries with a specific value.
///
/// Returns a key that can be used as the start of a range scan
/// for all entities with the given property value.
///
/// # Arguments
///
/// * `index_id` - The index identifier
/// * `value` - The property value to scan for
///
/// # Returns
///
/// `None` if the value cannot be sortable-encoded.
#[must_use]
pub fn encode_property_index_value_prefix(index_id: IndexId, value: &Value) -> Option<Vec<u8>> {
    use super::sortable::encode_sortable;

    let sortable_value = encode_sortable(value).ok()?;
    let mut key = Vec::with_capacity(1 + 8 + sortable_value.len());
    key.push(PREFIX_PROPERTY_INDEX);
    key.extend_from_slice(&index_id.as_u64().to_be_bytes());
    key.extend_from_slice(&sortable_value);
    Some(key)
}

/// Encode the exclusive upper bound for scanning a property index value.
///
/// This is used with `encode_property_index_value_prefix` to create a range
/// that includes all entities with exactly the given value.
///
/// # Arguments
///
/// * `index_id` - The index identifier
/// * `value` - The property value to create the bound for
///
/// # Returns
///
/// `None` if the value cannot be sortable-encoded.
#[must_use]
pub fn encode_property_index_value_bound(index_id: IndexId, value: &Value) -> Option<Vec<u8>> {
    use super::sortable::encode_sortable;

    let sortable_value = encode_sortable(value).ok()?;
    let mut key = Vec::with_capacity(1 + 8 + sortable_value.len() + 9);
    key.push(PREFIX_PROPERTY_INDEX);
    key.extend_from_slice(&index_id.as_u64().to_be_bytes());
    key.extend_from_slice(&sortable_value);
    // Append 9 bytes of 0xFF to create an exclusive upper bound
    // Valid keys have exactly 8 bytes for entity_id, so 9 bytes of 0xFF
    // is always greater than any valid key with this value prefix
    key.extend_from_slice(&[0xFF; 9]);
    Some(key)
}

/// Decode an entity ID from a property index key.
///
/// The entity ID is the last 8 bytes of the key.
///
/// # Arguments
///
/// * `key` - The property index key
///
/// # Returns
///
/// The entity ID, or `None` if the key is malformed.
#[must_use]
pub fn decode_property_index_entity_id(key: &[u8]) -> Option<EntityId> {
    // Minimum key size: prefix(1) + index_id(8) + min_value(1) + entity_id(8) = 18
    if key.len() < 18 || key[0] != PREFIX_PROPERTY_INDEX {
        return None;
    }
    let entity_bytes: [u8; 8] = key[key.len() - 8..].try_into().ok()?;
    Some(EntityId::new(u64::from_be_bytes(entity_bytes)))
}

/// Decode the index ID from a property index key.
///
/// # Arguments
///
/// * `key` - The property index key
///
/// # Returns
///
/// The index ID, or `None` if the key is malformed.
#[must_use]
pub fn decode_property_index_index_id(key: &[u8]) -> Option<IndexId> {
    // Minimum key size: prefix(1) + index_id(8) = 9
    if key.len() < 9 || key[0] != PREFIX_PROPERTY_INDEX {
        return None;
    }
    let id_bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(IndexId::new(u64::from_be_bytes(id_bytes)))
}

/// Decode the value portion from a property index key.
///
/// # Arguments
///
/// * `key` - The property index key
///
/// # Returns
///
/// The decoded value, or `None` if the key is malformed.
#[must_use]
pub fn decode_property_index_value(key: &[u8]) -> Option<Value> {
    use super::sortable::decode_sortable;

    // Minimum key size: prefix(1) + index_id(8) + min_value(1) + entity_id(8) = 18
    if key.len() < 18 || key[0] != PREFIX_PROPERTY_INDEX {
        return None;
    }
    // Value is between index_id and entity_id
    let value_bytes = &key[9..key.len() - 8];
    decode_sortable(value_bytes).ok()
}

/// Create an exclusive upper bound key by incrementing a prefix.
///
/// This is useful for creating range scan bounds. The returned key
/// is the smallest key that is greater than all keys with the given prefix.
///
/// # Arguments
///
/// * `prefix` - The prefix to increment
///
/// # Returns
///
/// The incremented prefix, or a key with an appended byte if all bytes are 0xFF.
#[must_use]
pub fn increment_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut end_prefix = prefix.to_vec();
    let mut i = end_prefix.len();
    while i > 0 {
        i -= 1;
        if end_prefix[i] < 255 {
            end_prefix[i] += 1;
            return end_prefix;
        }
        end_prefix[i] = 0;
    }
    // All bytes were 255, append a byte
    end_prefix.push(0);
    end_prefix
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

    // ========================================================================
    // Property Index Tests
    // ========================================================================

    #[test]
    fn index_id_from_label_property() {
        let id1 = IndexId::from_label_property("Person", "age");
        let id2 = IndexId::from_label_property("Person", "age");
        let id3 = IndexId::from_label_property("Person", "name");
        let id4 = IndexId::from_label_property("Company", "age");

        // Same label+property should produce same ID
        assert_eq!(id1, id2);
        // Different property should produce different ID
        assert_ne!(id1, id3);
        // Different label should produce different ID
        assert_ne!(id1, id4);
    }

    #[test]
    fn property_index_key_roundtrip() {
        let index_id = IndexId::from_label_property("Person", "age");
        let value = Value::Int(30);
        let entity_id = EntityId::new(42);

        let key = encode_property_index_key(index_id, &value, entity_id).unwrap();

        // Decode components
        let decoded_index = decode_property_index_index_id(&key).unwrap();
        let decoded_value = decode_property_index_value(&key).unwrap();
        let decoded_entity = decode_property_index_entity_id(&key).unwrap();

        assert_eq!(decoded_index, index_id);
        assert_eq!(decoded_value, value);
        assert_eq!(decoded_entity, entity_id);
    }

    #[test]
    fn property_index_key_prefix_matching() {
        let index_id = IndexId::from_label_property("Person", "age");
        let value = Value::Int(30);

        let key1 = encode_property_index_key(index_id, &value, EntityId::new(1)).unwrap();
        let key2 = encode_property_index_key(index_id, &value, EntityId::new(2)).unwrap();
        let key3 = encode_property_index_key(index_id, &Value::Int(31), EntityId::new(1)).unwrap();

        // Keys with same index_id should share the index prefix
        let index_prefix = encode_property_index_prefix(index_id);
        assert!(key1.starts_with(&index_prefix));
        assert!(key2.starts_with(&index_prefix));
        assert!(key3.starts_with(&index_prefix));

        // Keys with same value should share the value prefix
        let value_prefix = encode_property_index_value_prefix(index_id, &value).unwrap();
        assert!(key1.starts_with(&value_prefix));
        assert!(key2.starts_with(&value_prefix));
        assert!(!key3.starts_with(&value_prefix)); // Different value
    }

    #[test]
    fn property_index_values_are_ordered() {
        let index_id = IndexId::from_label_property("Person", "age");
        let entity = EntityId::new(1);

        // Test integer ordering
        let key_neg = encode_property_index_key(index_id, &Value::Int(-10), entity).unwrap();
        let key_zero = encode_property_index_key(index_id, &Value::Int(0), entity).unwrap();
        let key_pos = encode_property_index_key(index_id, &Value::Int(10), entity).unwrap();

        assert!(key_neg < key_zero, "negative should sort before zero");
        assert!(key_zero < key_pos, "zero should sort before positive");
    }

    #[test]
    fn property_index_string_values_ordered() {
        let index_id = IndexId::from_label_property("Person", "name");
        let entity = EntityId::new(1);

        let key_a =
            encode_property_index_key(index_id, &Value::String("alice".into()), entity).unwrap();
        let key_b =
            encode_property_index_key(index_id, &Value::String("bob".into()), entity).unwrap();
        let key_c =
            encode_property_index_key(index_id, &Value::String("charlie".into()), entity).unwrap();

        assert!(key_a < key_b);
        assert!(key_b < key_c);
    }

    #[test]
    fn property_index_value_bound_creates_range() {
        let index_id = IndexId::from_label_property("Person", "age");
        let value = Value::Int(30);

        let prefix = encode_property_index_value_prefix(index_id, &value).unwrap();
        let bound = encode_property_index_value_bound(index_id, &value).unwrap();

        // Keys for entities with this value should be in the range [prefix, bound)
        let key1 = encode_property_index_key(index_id, &value, EntityId::new(0)).unwrap();
        let key2 = encode_property_index_key(index_id, &value, EntityId::new(u64::MAX)).unwrap();

        assert!(key1.as_slice() >= prefix.as_slice());
        assert!(key1.as_slice() < bound.as_slice());
        assert!(key2.as_slice() >= prefix.as_slice());
        assert!(key2.as_slice() < bound.as_slice());

        // Key for a different value should be outside the range
        let key_other =
            encode_property_index_key(index_id, &Value::Int(31), EntityId::new(0)).unwrap();
        assert!(key_other.as_slice() >= bound.as_slice());
    }

    #[test]
    fn property_index_vector_not_supported() {
        let index_id = IndexId::from_label_property("Person", "embedding");
        let value = Value::Vector(vec![1.0, 2.0, 3.0]);
        let entity_id = EntityId::new(1);

        assert!(encode_property_index_key(index_id, &value, entity_id).is_none());
    }

    #[test]
    fn property_index_array_not_supported() {
        let index_id = IndexId::from_label_property("Person", "tags");
        let value = Value::Array(vec![Value::String("foo".into())]);
        let entity_id = EntityId::new(1);

        assert!(encode_property_index_key(index_id, &value, entity_id).is_none());
    }

    #[test]
    fn increment_prefix_basic() {
        assert_eq!(increment_prefix(&[0x00]), vec![0x01]);
        assert_eq!(increment_prefix(&[0x01, 0x02]), vec![0x01, 0x03]);
        assert_eq!(increment_prefix(&[0x01, 0xFF]), vec![0x02, 0x00]);
    }

    #[test]
    fn increment_prefix_overflow() {
        assert_eq!(increment_prefix(&[0xFF]), vec![0x00, 0x00]);
        assert_eq!(increment_prefix(&[0xFF, 0xFF]), vec![0x00, 0x00, 0x00]);
    }

    #[test]
    fn decode_property_index_invalid_key() {
        // Too short
        assert!(decode_property_index_entity_id(&[0x07, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
        // Wrong prefix
        assert!(decode_property_index_entity_id(&[0x01; 20]).is_none());
        // Empty
        assert!(decode_property_index_entity_id(&[]).is_none());
    }
}
