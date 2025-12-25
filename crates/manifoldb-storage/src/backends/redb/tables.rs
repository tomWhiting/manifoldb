//! Redb table definitions and key encoding utilities.
//!
//! This module provides utilities for working with tables in the Redb backend.
//! Since Redb requires static table names, we use a key prefixing strategy
//! to support dynamic "logical" table names within a single physical table.

use redb::TableDefinition;

/// The physical table that stores all key-value pairs.
/// Logical table names are prefixed to keys.
pub const DATA_TABLE: TableDefinition<'static, &[u8], &[u8]> =
    TableDefinition::new("manifold_data");

/// Separator byte between table name and key in the encoded key.
pub const KEY_SEPARATOR: u8 = 0x00;

/// Encode a logical table name and key into a physical key.
///
/// The format is: `<table_name><separator><key>`
/// This allows us to store multiple logical tables in one physical table.
pub fn encode_key(table: &str, key: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(table.len() + 1 + key.len());
    encoded.extend_from_slice(table.as_bytes());
    encoded.push(KEY_SEPARATOR);
    encoded.extend_from_slice(key);
    encoded
}

/// Decode a physical key into its logical table name and original key.
///
/// Returns `None` if the key is malformed (missing separator).
pub fn decode_key(encoded: &[u8]) -> Option<(&str, &[u8])> {
    let sep_pos = encoded.iter().position(|&b| b == KEY_SEPARATOR)?;
    let table = std::str::from_utf8(&encoded[..sep_pos]).ok()?;
    let key = &encoded[sep_pos + 1..];
    Some((table, key))
}

/// Create the start key for range scans on a logical table.
pub fn table_start_key(table: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + 1);
    key.extend_from_slice(table.as_bytes());
    key.push(KEY_SEPARATOR);
    key
}

/// Create the end key for range scans on a logical table.
/// This is the first key that would NOT belong to the table.
pub fn table_end_key(table: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(table.len() + 1);
    key.extend_from_slice(table.as_bytes());
    key.push(KEY_SEPARATOR + 1);
    key
}

/// Well-known table names for graph storage.
pub mod names {
    /// Table for storing node/entity data.
    pub const NODES: &str = "nodes";

    /// Table for storing edge/relationship data.
    pub const EDGES: &str = "edges";

    /// Table for storing outgoing edge indexes (source -> edges).
    pub const EDGES_OUT: &str = "edges_out";

    /// Table for storing incoming edge indexes (target -> edges).
    pub const EDGES_IN: &str = "edges_in";

    /// Table for storing label indexes (label -> node ids).
    pub const LABEL_INDEX: &str = "label_index";

    /// Table for storing edge type indexes (type -> edge ids).
    pub const EDGE_TYPE_INDEX: &str = "edge_type_index";

    /// Table for storing property indexes.
    pub const PROPERTY_INDEX: &str = "property_index";

    /// Table for storing metadata (counters, schema info, etc.).
    pub const METADATA: &str = "metadata";

    /// Table for storing HNSW index registry (maps index name to configuration).
    pub const HNSW_REGISTRY: &str = "hnsw_registry";

    /// Table for storing index catalog (index definitions and metadata).
    pub const INDEX_CATALOG: &str = "index_catalog";
}

/// Generate an HNSW index table name from the index name.
///
/// HNSW indexes use a naming convention: `hnsw_{index_name}`
/// This allows multiple HNSW indexes to coexist in the same database.
#[must_use]
pub fn hnsw_table_name(index_name: &str) -> String {
    format!("hnsw_{index_name}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_key() {
        let table = "users";
        let key = b"user:123";

        let encoded = encode_key(table, key);

        let (decoded_table, decoded_key) = decode_key(&encoded).unwrap();
        assert_eq!(decoded_table, table);
        assert_eq!(decoded_key, key);
    }

    #[test]
    fn test_encode_decode_empty_key() {
        let table = "config";
        let key = b"";

        let encoded = encode_key(table, key);

        let (decoded_table, decoded_key) = decode_key(&encoded).unwrap();
        assert_eq!(decoded_table, table);
        assert_eq!(decoded_key, key);
    }

    #[test]
    fn test_key_ordering() {
        // Keys from the same table should be adjacent
        let key_a = encode_key("users", b"a");
        let key_b = encode_key("users", b"b");
        let key_other = encode_key("zother", b"a");

        assert!(key_a < key_b);
        assert!(key_b < key_other);
    }

    #[test]
    fn test_table_range_keys() {
        let start = table_start_key("users");
        let end = table_end_key("users");

        // Any key in the "users" table should be >= start and < end
        let user_key = encode_key("users", b"test");
        assert!(user_key.as_slice() >= start.as_slice());
        assert!(user_key.as_slice() < end.as_slice());

        // A key from another table should be outside the range
        let other_key = encode_key("zother", b"test");
        assert!(other_key.as_slice() >= end.as_slice());
    }

    #[test]
    fn test_table_names() {
        // Verify table names are non-empty strings
        assert!(!names::NODES.is_empty());
        assert!(!names::EDGES.is_empty());
        assert!(!names::EDGES_OUT.is_empty());
        assert!(!names::EDGES_IN.is_empty());
        assert!(!names::LABEL_INDEX.is_empty());
        assert!(!names::EDGE_TYPE_INDEX.is_empty());
        assert!(!names::PROPERTY_INDEX.is_empty());
        assert!(!names::METADATA.is_empty());
    }
}
