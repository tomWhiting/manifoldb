//! Property index entry type and operations.

use crate::encoding::keys::{
    decode_property_index_entity_id, decode_property_index_index_id, decode_property_index_value,
    encode_property_index_key, encode_property_index_prefix, encode_property_index_value_bound,
    encode_property_index_value_prefix, increment_prefix, IndexId,
};
use crate::types::{EntityId, Value};

/// An entry in a property index.
///
/// Each entry represents a single (value, entity_id) pair for a specific index.
/// The index is identified by an [`IndexId`] which is derived from the label
/// and property name.
///
/// # Example
///
/// ```
/// use manifoldb_core::index::{PropertyIndexEntry, IndexId};
/// use manifoldb_core::types::{EntityId, Value};
///
/// let index_id = IndexId::from_label_property("Person", "age");
/// let entry = PropertyIndexEntry::new(index_id, Value::Int(25), EntityId::new(1));
///
/// // Encode for storage
/// let key = entry.encode_key().unwrap();
///
/// // Decode from storage
/// let decoded = PropertyIndexEntry::decode_key(&key).unwrap();
/// assert_eq!(decoded.value, Value::Int(25));
/// assert_eq!(decoded.entity_id, EntityId::new(1));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyIndexEntry {
    /// The index this entry belongs to.
    pub index_id: IndexId,
    /// The property value.
    pub value: Value,
    /// The entity ID that has this value.
    pub entity_id: EntityId,
}

impl PropertyIndexEntry {
    /// Create a new property index entry.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index identifier
    /// * `value` - The property value
    /// * `entity_id` - The entity ID
    #[inline]
    #[must_use]
    pub const fn new(index_id: IndexId, value: Value, entity_id: EntityId) -> Self {
        Self { index_id, value, entity_id }
    }

    /// Encode this entry as a storage key.
    ///
    /// Returns `None` if the value cannot be encoded (e.g., vectors, arrays).
    #[must_use]
    pub fn encode_key(&self) -> Option<Vec<u8>> {
        encode_property_index_key(self.index_id, &self.value, self.entity_id)
    }

    /// Decode an entry from a storage key.
    ///
    /// Returns `None` if the key is malformed.
    #[must_use]
    pub fn decode_key(key: &[u8]) -> Option<Self> {
        let index_id = decode_property_index_index_id(key)?;
        let value = decode_property_index_value(key)?;
        let entity_id = decode_property_index_entity_id(key)?;
        Some(Self { index_id, value, entity_id })
    }

    /// Check if a value type is supported for indexing.
    ///
    /// Vectors, sparse vectors, multi-vectors, and arrays are not supported
    /// because they don't have a natural total ordering.
    #[must_use]
    pub fn is_indexable(value: &Value) -> bool {
        matches!(
            value,
            Value::Null
                | Value::Bool(_)
                | Value::Int(_)
                | Value::Float(_)
                | Value::String(_)
                | Value::Bytes(_)
        )
    }
}

/// Utilities for scanning property indexes.
///
/// This struct provides methods for generating the start and end keys
/// for various types of index scans.
pub struct PropertyIndexScan;

impl PropertyIndexScan {
    /// Create keys for scanning all entries in an index.
    ///
    /// Returns (start_key, end_key) for a range scan.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::index::{PropertyIndexScan, IndexId};
    ///
    /// let index_id = IndexId::from_label_property("Person", "age");
    /// let (start, end) = PropertyIndexScan::full_index_range(index_id);
    ///
    /// // Use start and end for a range scan on the storage layer
    /// ```
    #[must_use]
    pub fn full_index_range(index_id: IndexId) -> (Vec<u8>, Vec<u8>) {
        let start = encode_property_index_prefix(index_id);
        let end = increment_prefix(&start);
        (start, end)
    }

    /// Create keys for scanning entries with a specific value.
    ///
    /// Returns (start_key, end_key) for a range scan, or `None` if the
    /// value cannot be encoded.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    /// * `value` - The exact value to match
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::index::{PropertyIndexScan, IndexId};
    /// use manifoldb_core::types::Value;
    ///
    /// let index_id = IndexId::from_label_property("Person", "age");
    /// if let Some((start, end)) = PropertyIndexScan::exact_value_range(index_id, &Value::Int(30)) {
    ///     // Scan for all entities where Person.age = 30
    /// }
    /// ```
    #[must_use]
    pub fn exact_value_range(index_id: IndexId, value: &Value) -> Option<(Vec<u8>, Vec<u8>)> {
        let start = encode_property_index_value_prefix(index_id, value)?;
        let end = encode_property_index_value_bound(index_id, value)?;
        Some((start, end))
    }

    /// Create keys for scanning entries >= a value.
    ///
    /// Returns (start_key, end_key) for a range scan.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    /// * `min_value` - The minimum value (inclusive)
    #[must_use]
    pub fn range_from(index_id: IndexId, min_value: &Value) -> Option<(Vec<u8>, Vec<u8>)> {
        let start = encode_property_index_value_prefix(index_id, min_value)?;
        let index_prefix = encode_property_index_prefix(index_id);
        let end = increment_prefix(&index_prefix);
        Some((start, end))
    }

    /// Create keys for scanning entries < a value.
    ///
    /// Returns (start_key, end_key) for a range scan.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    /// * `max_value` - The maximum value (exclusive)
    #[must_use]
    pub fn range_to(index_id: IndexId, max_value: &Value) -> Option<(Vec<u8>, Vec<u8>)> {
        let start = encode_property_index_prefix(index_id);
        let end = encode_property_index_value_prefix(index_id, max_value)?;
        Some((start, end))
    }

    /// Create keys for scanning entries in a range [min, max).
    ///
    /// Returns (start_key, end_key) for a range scan.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    /// * `min_value` - The minimum value (inclusive)
    /// * `max_value` - The maximum value (exclusive)
    #[must_use]
    pub fn range_between(
        index_id: IndexId,
        min_value: &Value,
        max_value: &Value,
    ) -> Option<(Vec<u8>, Vec<u8>)> {
        let start = encode_property_index_value_prefix(index_id, min_value)?;
        let end = encode_property_index_value_prefix(index_id, max_value)?;
        Some((start, end))
    }

    /// Create keys for prefix matching (e.g., LIKE 'prefix%').
    ///
    /// Returns (start_key, end_key) for a range scan that matches all
    /// strings starting with the given prefix.
    ///
    /// # Arguments
    ///
    /// * `index_id` - The index to scan
    /// * `prefix` - The string prefix to match
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::index::{PropertyIndexScan, IndexId};
    ///
    /// let index_id = IndexId::from_label_property("Person", "name");
    /// if let Some((start, end)) = PropertyIndexScan::string_prefix_range(index_id, "Alice") {
    ///     // Scan for all entities where Person.name LIKE 'Alice%'
    /// }
    /// ```
    #[must_use]
    pub fn string_prefix_range(index_id: IndexId, prefix: &str) -> Option<(Vec<u8>, Vec<u8>)> {
        // For prefix matching, we need to find the range of strings that start with the prefix.
        // We create the start as the prefix itself, and the end as the next string lexicographically
        // after all strings that start with the prefix.
        //
        // For example, for prefix "Alice":
        // - Start: "Alice" (encoded)
        // - End: "Alicf" (increment the last character)

        let prefix_bytes = prefix.as_bytes();

        // Find the end string by incrementing the prefix
        let mut end_bytes = prefix_bytes.to_vec();
        while !end_bytes.is_empty() {
            let last = end_bytes.len() - 1;
            if end_bytes[last] < 0xFF {
                end_bytes[last] += 1;
                break;
            }
            // Last byte is 0xFF, remove it and try incrementing the previous byte
            end_bytes.pop();
        }

        let start_value = Value::String(prefix.to_owned());
        let start = encode_property_index_value_prefix(index_id, &start_value)?;

        if end_bytes.is_empty() {
            // All bytes were 0xFF, scan to end of index
            let index_prefix = encode_property_index_prefix(index_id);
            let end = increment_prefix(&index_prefix);
            Some((start, end))
        } else {
            // Use the incremented string as the end bound
            let end_value = Value::String(String::from_utf8_lossy(&end_bytes).into_owned());
            let end = encode_property_index_value_prefix(index_id, &end_value)?;
            Some((start, end))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn entry_roundtrip() {
        let index_id = IndexId::from_label_property("Person", "age");
        let entry = PropertyIndexEntry::new(index_id, Value::Int(30), EntityId::new(42));

        let key = entry.encode_key().unwrap();
        let decoded = PropertyIndexEntry::decode_key(&key).unwrap();

        assert_eq!(decoded.index_id, entry.index_id);
        assert_eq!(decoded.value, entry.value);
        assert_eq!(decoded.entity_id, entry.entity_id);
    }

    #[test]
    fn entry_roundtrip_string() {
        let index_id = IndexId::from_label_property("Person", "name");
        let entry =
            PropertyIndexEntry::new(index_id, Value::String("Alice".into()), EntityId::new(123));

        let key = entry.encode_key().unwrap();
        let decoded = PropertyIndexEntry::decode_key(&key).unwrap();

        assert_eq!(decoded.value, Value::String("Alice".into()));
        assert_eq!(decoded.entity_id, EntityId::new(123));
    }

    #[test]
    fn entry_roundtrip_null() {
        let index_id = IndexId::from_label_property("Person", "nickname");
        let entry = PropertyIndexEntry::new(index_id, Value::Null, EntityId::new(1));

        let key = entry.encode_key().unwrap();
        let decoded = PropertyIndexEntry::decode_key(&key).unwrap();

        assert_eq!(decoded.value, Value::Null);
    }

    #[test]
    fn entry_vector_not_supported() {
        let index_id = IndexId::from_label_property("Document", "embedding");
        let entry =
            PropertyIndexEntry::new(index_id, Value::Vector(vec![1.0, 2.0, 3.0]), EntityId::new(1));

        assert!(entry.encode_key().is_none());
    }

    #[test]
    fn is_indexable() {
        assert!(PropertyIndexEntry::is_indexable(&Value::Null));
        assert!(PropertyIndexEntry::is_indexable(&Value::Bool(true)));
        assert!(PropertyIndexEntry::is_indexable(&Value::Int(42)));
        assert!(PropertyIndexEntry::is_indexable(&Value::Float(3.14)));
        assert!(PropertyIndexEntry::is_indexable(&Value::String("test".into())));
        assert!(PropertyIndexEntry::is_indexable(&Value::Bytes(vec![1, 2, 3])));

        assert!(!PropertyIndexEntry::is_indexable(&Value::Vector(vec![1.0])));
        assert!(!PropertyIndexEntry::is_indexable(&Value::SparseVector(vec![(0, 1.0)])));
        assert!(!PropertyIndexEntry::is_indexable(&Value::MultiVector(vec![vec![1.0]])));
        assert!(!PropertyIndexEntry::is_indexable(&Value::Array(vec![Value::Int(1)])));
    }

    #[test]
    fn scan_full_index_range() {
        let index_id = IndexId::from_label_property("Person", "age");
        let (start, end) = PropertyIndexScan::full_index_range(index_id);

        // All entries in this index should fall within [start, end)
        let entry = PropertyIndexEntry::new(index_id, Value::Int(30), EntityId::new(1));
        let key = entry.encode_key().unwrap();

        assert!(key.as_slice() >= start.as_slice());
        assert!(key.as_slice() < end.as_slice());
    }

    #[test]
    fn scan_exact_value_range() {
        let index_id = IndexId::from_label_property("Person", "age");
        let value = Value::Int(30);
        let (start, end) = PropertyIndexScan::exact_value_range(index_id, &value).unwrap();

        // Entry with matching value should be in range
        let entry1 = PropertyIndexEntry::new(index_id, Value::Int(30), EntityId::new(1));
        let key1 = entry1.encode_key().unwrap();
        assert!(key1.as_slice() >= start.as_slice());
        assert!(key1.as_slice() < end.as_slice());

        // Entry with different value should be outside range
        let entry2 = PropertyIndexEntry::new(index_id, Value::Int(31), EntityId::new(1));
        let key2 = entry2.encode_key().unwrap();
        assert!(key2.as_slice() >= end.as_slice());
    }

    #[test]
    fn scan_range_from() {
        let index_id = IndexId::from_label_property("Person", "age");
        let (start, end) = PropertyIndexScan::range_from(index_id, &Value::Int(18)).unwrap();

        // Entry with age >= 18 should be in range
        let entry1 = PropertyIndexEntry::new(index_id, Value::Int(25), EntityId::new(1));
        let key1 = entry1.encode_key().unwrap();
        assert!(key1.as_slice() >= start.as_slice());
        assert!(key1.as_slice() < end.as_slice());

        // Entry with age < 18 should be outside range
        let entry2 = PropertyIndexEntry::new(index_id, Value::Int(10), EntityId::new(1));
        let key2 = entry2.encode_key().unwrap();
        assert!(key2.as_slice() < start.as_slice());
    }

    #[test]
    fn scan_range_to() {
        let index_id = IndexId::from_label_property("Person", "age");
        let (start, end) = PropertyIndexScan::range_to(index_id, &Value::Int(18)).unwrap();

        // Entry with age < 18 should be in range
        let entry1 = PropertyIndexEntry::new(index_id, Value::Int(10), EntityId::new(1));
        let key1 = entry1.encode_key().unwrap();
        assert!(key1.as_slice() >= start.as_slice());
        assert!(key1.as_slice() < end.as_slice());

        // Entry with age >= 18 should be outside range
        let entry2 = PropertyIndexEntry::new(index_id, Value::Int(25), EntityId::new(1));
        let key2 = entry2.encode_key().unwrap();
        assert!(key2.as_slice() >= end.as_slice());
    }

    #[test]
    fn scan_range_between() {
        let index_id = IndexId::from_label_property("Person", "age");
        let (start, end) =
            PropertyIndexScan::range_between(index_id, &Value::Int(18), &Value::Int(65)).unwrap();

        // Entry with 18 <= age < 65 should be in range
        let entry1 = PropertyIndexEntry::new(index_id, Value::Int(30), EntityId::new(1));
        let key1 = entry1.encode_key().unwrap();
        assert!(key1.as_slice() >= start.as_slice());
        assert!(key1.as_slice() < end.as_slice());

        // Entry with age < 18 should be outside range
        let entry2 = PropertyIndexEntry::new(index_id, Value::Int(10), EntityId::new(1));
        let key2 = entry2.encode_key().unwrap();
        assert!(key2.as_slice() < start.as_slice());

        // Entry with age >= 65 should be outside range
        let entry3 = PropertyIndexEntry::new(index_id, Value::Int(70), EntityId::new(1));
        let key3 = entry3.encode_key().unwrap();
        assert!(key3.as_slice() >= end.as_slice());
    }

    #[test]
    fn scan_string_prefix() {
        let index_id = IndexId::from_label_property("Person", "name");
        let (start, end) = PropertyIndexScan::string_prefix_range(index_id, "Alice").unwrap();

        // Names starting with "Alice" should be in range
        let entry1 =
            PropertyIndexEntry::new(index_id, Value::String("Alice".into()), EntityId::new(1));
        let key1 = entry1.encode_key().unwrap();
        assert!(key1.as_slice() >= start.as_slice());
        assert!(key1.as_slice() < end.as_slice());

        let entry2 = PropertyIndexEntry::new(
            index_id,
            Value::String("Alice Smith".into()),
            EntityId::new(2),
        );
        let key2 = entry2.encode_key().unwrap();
        assert!(key2.as_slice() >= start.as_slice());
        assert!(key2.as_slice() < end.as_slice());

        // Names not starting with "Alice" should be outside range
        let entry3 =
            PropertyIndexEntry::new(index_id, Value::String("Bob".into()), EntityId::new(3));
        let key3 = entry3.encode_key().unwrap();
        assert!(key3.as_slice() >= end.as_slice());
    }

    #[test]
    fn entries_sorted_by_value() {
        let index_id = IndexId::from_label_property("Person", "age");
        let entity = EntityId::new(1);

        let key_neg =
            PropertyIndexEntry::new(index_id, Value::Int(-10), entity).encode_key().unwrap();
        let key_zero =
            PropertyIndexEntry::new(index_id, Value::Int(0), entity).encode_key().unwrap();
        let key_pos =
            PropertyIndexEntry::new(index_id, Value::Int(10), entity).encode_key().unwrap();

        assert!(key_neg < key_zero);
        assert!(key_zero < key_pos);
    }

    #[test]
    fn entries_with_same_value_sorted_by_entity_id() {
        let index_id = IndexId::from_label_property("Person", "age");
        let value = Value::Int(30);

        let key1 = PropertyIndexEntry::new(index_id, value.clone(), EntityId::new(1))
            .encode_key()
            .unwrap();
        let key2 = PropertyIndexEntry::new(index_id, value.clone(), EntityId::new(2))
            .encode_key()
            .unwrap();
        let key3 =
            PropertyIndexEntry::new(index_id, value, EntityId::new(100)).encode_key().unwrap();

        assert!(key1 < key2);
        assert!(key2 < key3);
    }
}
