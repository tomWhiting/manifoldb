//! Payload indexing for efficient filtered vector search.
//!
//! This module provides B-tree indexes on entity properties to speed up
//! filtered queries. Instead of scanning all entities during HNSW traversal,
//! we can use indexes to narrow down candidates first.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::in_memory()?;
//!
//! // Create an index on the "language" property for entities with "Symbol" label
//! db.create_index("Symbol", "language")?;
//!
//! // Searches using Filter::eq("language", "rust") will now use the index
//! let results = db.search("symbols", "dense")?
//!     .query(query_vector)
//!     .filter(Filter::eq("language", "rust"))  // Uses index!
//!     .limit(10)
//!     .execute()?;
//! ```
//!
//! # Index Types
//!
//! - **Equality**: Supports `eq`, `ne`, `in` operators. Best for enum-like fields.
//! - **Range**: Supports `gt`, `gte`, `lt`, `lte`, `range`. Best for numeric fields.
//! - **Prefix**: Supports `starts_with`. Best for paths and names.

use std::ops::Bound;
use std::sync::Arc;

use manifoldb_core::encoding::keys::encode_entity_key;
use manifoldb_core::encoding::sortable::encode_sortable;
use manifoldb_core::encoding::Decoder;
use manifoldb_core::{Entity, EntityId, Value};
use manifoldb_storage::backends::redb::tables;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Type of index, determining which filter operations it supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IndexType {
    /// Equality index: supports `eq`, `ne`, `in` operators.
    /// Best for enum-like fields with a small number of distinct values.
    #[default]
    Equality,

    /// Range index: supports `gt`, `gte`, `lt`, `lte`, `range` operators.
    /// Best for numeric fields and dates.
    Range,

    /// Prefix index: supports `starts_with` operator.
    /// Best for file paths and names.
    Prefix,
}

/// Metadata about an index stored in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// The label this index applies to.
    pub label: String,

    /// The property being indexed.
    pub property: String,

    /// Type of index (equality, range, prefix).
    pub index_type: IndexType,

    /// Unix timestamp when index was created.
    pub created_at: u64,

    /// Number of entries in the index.
    pub entry_count: u64,

    /// Number of distinct values (for selectivity estimation).
    pub distinct_values: u64,
}

impl IndexMetadata {
    /// Create new index metadata.
    pub fn new(
        label: impl Into<String>,
        property: impl Into<String>,
        index_type: IndexType,
    ) -> Self {
        Self {
            label: label.into(),
            property: property.into(),
            index_type,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            entry_count: 0,
            distinct_values: 0,
        }
    }

    /// Generate the catalog key for this index.
    pub fn catalog_key(&self) -> Vec<u8> {
        make_catalog_key(&self.label, &self.property)
    }
}

/// Information about an index returned by list_indexes.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// The label this index applies to.
    pub label: String,

    /// The property being indexed.
    pub property: String,

    /// Type of index.
    pub index_type: IndexType,

    /// Number of entries in the index.
    pub entry_count: u64,
}

impl From<IndexMetadata> for IndexInfo {
    fn from(meta: IndexMetadata) -> Self {
        Self {
            label: meta.label,
            property: meta.property,
            index_type: meta.index_type,
            entry_count: meta.entry_count,
        }
    }
}

/// Statistics about an index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of entries in the index.
    pub entry_count: u64,

    /// Number of distinct values.
    pub distinct_values: u64,

    /// Estimated selectivity (1 / distinct_values).
    pub selectivity: f64,

    /// Unix timestamp when index was created.
    pub created_at: u64,
}

impl From<IndexMetadata> for IndexStats {
    fn from(meta: IndexMetadata) -> Self {
        let selectivity =
            if meta.distinct_values > 0 { 1.0 / meta.distinct_values as f64 } else { 1.0 };

        Self {
            entry_count: meta.entry_count,
            distinct_values: meta.distinct_values,
            selectivity,
            created_at: meta.created_at,
        }
    }
}

// ============================================================================
// Key Encoding
// ============================================================================

/// Table name for the payload index.
const PAYLOAD_INDEX_TABLE: &str = "payload_index";

/// Table name for the index catalog (metadata).
const INDEX_CATALOG_TABLE: &str = "index_catalog";

/// Create a catalog key for an index definition.
///
/// Format: `<label>\0<property>`
fn make_catalog_key(label: &str, property: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(label.len() + 1 + property.len());
    key.extend_from_slice(label.as_bytes());
    key.push(0x00);
    key.extend_from_slice(property.as_bytes());
    key
}

/// Parse a catalog key back to (label, property).
#[allow(dead_code)]
fn parse_catalog_key(key: &[u8]) -> Option<(String, String)> {
    let sep_pos = key.iter().position(|&b| b == 0x00)?;
    let label = std::str::from_utf8(&key[..sep_pos]).ok()?;
    let property = std::str::from_utf8(&key[sep_pos + 1..]).ok()?;
    Some((label.to_string(), property.to_string()))
}

/// Create an index entry key.
///
/// Format: `<label>\0<property>\0<encoded_value><entity_id>`
///
/// The entity_id is appended as 8 bytes big-endian to ensure uniqueness
/// and allow multiple entities to have the same property value.
fn make_index_key(
    label: &str,
    property: &str,
    value: &Value,
    entity_id: EntityId,
) -> Result<Vec<u8>> {
    let encoded_value = encode_sortable(value)
        .map_err(|e| Error::InvalidInput(format!("Cannot index value: {e}")))?;

    let mut key =
        Vec::with_capacity(label.len() + 1 + property.len() + 1 + encoded_value.len() + 8);
    key.extend_from_slice(label.as_bytes());
    key.push(0x00);
    key.extend_from_slice(property.as_bytes());
    key.push(0x00);
    key.extend_from_slice(&encoded_value);
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    Ok(key)
}

/// Create the prefix for scanning all index entries for a (label, property) pair.
fn make_index_prefix(label: &str, property: &str) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(label.len() + 1 + property.len() + 1);
    prefix.extend_from_slice(label.as_bytes());
    prefix.push(0x00);
    prefix.extend_from_slice(property.as_bytes());
    prefix.push(0x00);
    prefix
}

/// Create the prefix for scanning index entries with a specific value.
fn make_index_value_prefix(label: &str, property: &str, value: &Value) -> Result<Vec<u8>> {
    let encoded_value = encode_sortable(value)
        .map_err(|e| Error::InvalidInput(format!("Cannot encode value for index lookup: {e}")))?;

    let mut prefix = Vec::with_capacity(label.len() + 1 + property.len() + 1 + encoded_value.len());
    prefix.extend_from_slice(label.as_bytes());
    prefix.push(0x00);
    prefix.extend_from_slice(property.as_bytes());
    prefix.push(0x00);
    prefix.extend_from_slice(&encoded_value);
    Ok(prefix)
}

/// Extract entity ID from an index key.
///
/// The entity ID is the last 8 bytes of the key.
fn extract_entity_id_from_key(key: &[u8]) -> Option<EntityId> {
    if key.len() < 8 {
        return None;
    }
    let id_bytes: [u8; 8] = key[key.len() - 8..].try_into().ok()?;
    Some(EntityId::new(u64::from_be_bytes(id_bytes)))
}

// ============================================================================
// Index Manager
// ============================================================================

/// Manages payload indexes for a database.
pub struct IndexManager {
    engine: Arc<RedbEngine>,
}

impl IndexManager {
    /// Create a new index manager.
    pub fn new(engine: Arc<RedbEngine>) -> Self {
        Self { engine }
    }

    /// Create an index on a property for entities with the given label.
    ///
    /// This scans all existing entities with the label and builds the index.
    pub fn create_index(&self, label: &str, property: &str, index_type: IndexType) -> Result<()> {
        // Check if index already exists
        if self.get_index_metadata(label, property)?.is_some() {
            return Err(Error::InvalidInput(format!("Index already exists on {label}.{property}")));
        }

        // Create metadata
        let mut metadata = IndexMetadata::new(label, property, index_type);

        // Start a write transaction
        let mut tx = self.engine.begin_write()?;

        // Scan all entities with this label and index them
        let mut entry_count = 0u64;
        let mut distinct_values = std::collections::HashSet::new();

        // Get all entity IDs with this label by scanning the label index
        // Key format in label index: <length:2 bytes><label:N bytes><entity_id:8 bytes>
        let label_bytes = label.as_bytes();
        let label_len = label_bytes.len() as u16;

        // Start key: length prefix + label (no entity_id suffix)
        let label_start = {
            let mut key = Vec::with_capacity(2 + label_bytes.len());
            key.extend_from_slice(&label_len.to_be_bytes());
            key.extend_from_slice(label_bytes);
            key
        };

        // End key: length prefix + label + max entity_id + extra byte
        let label_end = {
            let mut key = Vec::with_capacity(2 + label_bytes.len() + 8 + 1);
            key.extend_from_slice(&label_len.to_be_bytes());
            key.extend_from_slice(label_bytes);
            key.extend_from_slice(&u64::MAX.to_be_bytes());
            key.push(0);
            key
        };

        // Collect entity IDs first to avoid borrow issues
        let entity_ids: Vec<EntityId> = {
            let mut cursor = tx.range(
                tables::names::LABEL_INDEX,
                Bound::Included(label_start.as_slice()),
                Bound::Excluded(label_end.as_slice()),
            )?;

            let mut ids = Vec::new();
            while let Some((key, _)) = cursor.next()? {
                // Key format: <length:2 bytes><label:N bytes><entity_id:8 bytes>
                let expected_prefix_len = 2 + label_bytes.len();
                if key.len() >= expected_prefix_len + 8 {
                    // Entity ID is at the end after length prefix + label
                    let id_start = expected_prefix_len;
                    if let Ok(id_bytes) = key[id_start..id_start + 8].try_into() {
                        let id_bytes: [u8; 8] = id_bytes;
                        ids.push(EntityId::new(u64::from_be_bytes(id_bytes)));
                    }
                }
            }
            ids
        };

        // Now index each entity
        for entity_id in entity_ids {
            // Get entity data
            let entity_key = encode_entity_key(entity_id);

            if let Some(data) = tx.get(tables::names::NODES, &entity_key)? {
                // Decode entity using the standard encoder format
                if let Ok(entity) = Entity::decode(&data) {
                    // Get property value
                    if let Some(value) = entity.get_property(property) {
                        // Create index entry
                        let index_key = make_index_key(label, property, value, entity_id)?;

                        tx.put(PAYLOAD_INDEX_TABLE, &index_key, &[])?;

                        entry_count += 1;

                        // Track distinct values (use encoded form for hashing)
                        if let Ok(encoded) = encode_sortable(value) {
                            distinct_values.insert(encoded);
                        }
                    }
                }
            }
        }

        // Update metadata
        metadata.entry_count = entry_count;
        metadata.distinct_values = distinct_values.len() as u64;

        // Store metadata in catalog
        let catalog_key = metadata.catalog_key();
        let catalog_value = bincode::serde::encode_to_vec(&metadata, bincode::config::standard())
            .map_err(|e| Error::Serialization(e.to_string()))?;

        tx.put(INDEX_CATALOG_TABLE, &catalog_key, &catalog_value)?;

        // Commit
        tx.commit()?;

        Ok(())
    }

    /// Drop an index.
    pub fn drop_index(&self, label: &str, property: &str) -> Result<()> {
        // Check if index exists
        if self.get_index_metadata(label, property)?.is_none() {
            return Err(Error::InvalidInput(format!("No index exists on {label}.{property}")));
        }

        let mut tx = self.engine.begin_write()?;

        // Delete all index entries
        let prefix = make_index_prefix(label, property);
        let end_prefix = {
            let mut end = prefix.clone();
            end.push(0xFF);
            end
        };

        // Collect keys to delete
        let keys_to_delete: Vec<Vec<u8>> = {
            let mut cursor = tx.range(
                PAYLOAD_INDEX_TABLE,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(end_prefix.as_slice()),
            )?;

            let mut keys = Vec::new();
            while let Some((key, _)) = cursor.next()? {
                keys.push(key.clone());
            }
            keys
        };

        // Delete index entries
        for key in keys_to_delete {
            tx.delete(PAYLOAD_INDEX_TABLE, &key)?;
        }

        // Delete catalog entry
        let catalog_key = make_catalog_key(label, property);
        tx.delete(INDEX_CATALOG_TABLE, &catalog_key)?;

        tx.commit()?;

        Ok(())
    }

    /// Get metadata for an index.
    pub fn get_index_metadata(&self, label: &str, property: &str) -> Result<Option<IndexMetadata>> {
        let tx = self.engine.begin_read()?;

        let catalog_key = make_catalog_key(label, property);

        if let Some(data) = tx.get(INDEX_CATALOG_TABLE, &catalog_key)? {
            let (metadata, _): (IndexMetadata, _) =
                bincode::serde::decode_from_slice(&data, bincode::config::standard())
                    .map_err(|e| Error::Serialization(e.to_string()))?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// List all indexes.
    pub fn list_indexes(&self) -> Result<Vec<IndexInfo>> {
        let tx = self.engine.begin_read()?;

        let mut cursor = tx.cursor(INDEX_CATALOG_TABLE)?;

        let mut indexes = Vec::new();
        while let Some((_key, value)) = cursor.next()? {
            if let Ok((metadata, _)) = bincode::serde::decode_from_slice::<IndexMetadata, _>(
                &value,
                bincode::config::standard(),
            ) {
                indexes.push(IndexInfo::from(metadata));
            }
        }

        Ok(indexes)
    }

    /// Get statistics for an index.
    pub fn index_stats(&self, label: &str, property: &str) -> Result<IndexStats> {
        let metadata = self
            .get_index_metadata(label, property)?
            .ok_or_else(|| Error::InvalidInput(format!("No index exists on {label}.{property}")))?;

        Ok(IndexStats::from(metadata))
    }

    /// Lookup entity IDs matching a filter value using the index.
    ///
    /// Returns None if no index exists for this label/property combination.
    pub fn lookup_eq(
        &self,
        label: &str,
        property: &str,
        value: &Value,
    ) -> Result<Option<Vec<EntityId>>> {
        // Check if index exists
        if self.get_index_metadata(label, property)?.is_none() {
            return Ok(None);
        }

        let tx = self.engine.begin_read()?;

        let prefix = make_index_value_prefix(label, property, value)?;
        let end_prefix = {
            let mut end = prefix.clone();
            end.push(0xFF);
            end
        };

        let mut cursor = tx.range(
            PAYLOAD_INDEX_TABLE,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(end_prefix.as_slice()),
        )?;

        let mut entity_ids = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(entity_id) = extract_entity_id_from_key(&key) {
                entity_ids.push(entity_id);
            }
        }

        Ok(Some(entity_ids))
    }

    /// Lookup entity IDs matching an "in" filter using the index.
    pub fn lookup_in(
        &self,
        label: &str,
        property: &str,
        values: &[Value],
    ) -> Result<Option<Vec<EntityId>>> {
        // Check if index exists
        if self.get_index_metadata(label, property)?.is_none() {
            return Ok(None);
        }

        let mut all_ids = Vec::new();
        for value in values {
            if let Some(ids) = self.lookup_eq(label, property, value)? {
                all_ids.extend(ids);
            }
        }

        // Deduplicate
        all_ids.sort_by_key(|id| id.as_u64());
        all_ids.dedup_by_key(|id| id.as_u64());

        Ok(Some(all_ids))
    }

    /// Maintain index on entity upsert.
    ///
    /// Call this when an entity is inserted or updated.
    /// This version works within an existing transaction.
    pub fn on_entity_upsert_tx<T: Transaction>(
        &self,
        tx: &mut T,
        entity: &Entity,
        old_entity: Option<&Entity>,
    ) -> Result<()> {
        // Get labels from entity
        for label in &entity.labels {
            let label_str = label.as_str();

            // Check what indexes exist for this label
            let catalog_prefix = {
                let mut p = label_str.as_bytes().to_vec();
                p.push(0x00);
                p
            };
            let catalog_end = {
                let mut end = catalog_prefix.clone();
                end.push(0xFF);
                end
            };

            // Get index definitions for this label
            let indexes: Vec<IndexMetadata> = {
                let mut cursor = tx.range(
                    INDEX_CATALOG_TABLE,
                    Bound::Included(catalog_prefix.as_slice()),
                    Bound::Excluded(catalog_end.as_slice()),
                )?;

                let mut metas = Vec::new();
                while let Some((_, value)) = cursor.next()? {
                    if let Ok((m, _)) = bincode::serde::decode_from_slice::<IndexMetadata, _>(
                        &value,
                        bincode::config::standard(),
                    ) {
                        metas.push(m);
                    }
                }
                metas
            };

            // Update each index
            for index_meta in indexes {
                let property = &index_meta.property;

                let new_value = entity.get_property(property);
                let old_value = old_entity.and_then(|e| e.get_property(property));

                // Only update if value changed
                if new_value != old_value {
                    // Remove old index entry if exists
                    if let Some(old) = old_value {
                        let old_key = make_index_key(label_str, property, old, entity.id)?;
                        tx.delete(PAYLOAD_INDEX_TABLE, &old_key)?;
                    }

                    // Add new index entry if value exists
                    if let Some(new) = new_value {
                        let new_key = make_index_key(label_str, property, new, entity.id)?;
                        tx.put(PAYLOAD_INDEX_TABLE, &new_key, &[])?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Maintain index on entity delete.
    /// This version works within an existing transaction.
    pub fn on_entity_delete_tx<T: Transaction>(&self, tx: &mut T, entity: &Entity) -> Result<()> {
        for label in &entity.labels {
            let label_str = label.as_str();

            // Get index definitions for this label
            let catalog_prefix = {
                let mut p = label_str.as_bytes().to_vec();
                p.push(0x00);
                p
            };
            let catalog_end = {
                let mut end = catalog_prefix.clone();
                end.push(0xFF);
                end
            };

            let indexes: Vec<IndexMetadata> = {
                let mut cursor = tx.range(
                    INDEX_CATALOG_TABLE,
                    Bound::Included(catalog_prefix.as_slice()),
                    Bound::Excluded(catalog_end.as_slice()),
                )?;

                let mut metas = Vec::new();
                while let Some((_, value)) = cursor.next()? {
                    if let Ok((m, _)) = bincode::serde::decode_from_slice::<IndexMetadata, _>(
                        &value,
                        bincode::config::standard(),
                    ) {
                        metas.push(m);
                    }
                }
                metas
            };

            for index_meta in indexes {
                let property = &index_meta.property;

                if let Some(value) = entity.get_property(property) {
                    let key = make_index_key(label_str, property, value, entity.id)?;
                    tx.delete(PAYLOAD_INDEX_TABLE, &key)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_catalog_key() {
        let key = make_catalog_key("Symbol", "language");
        assert_eq!(key, b"Symbol\0language");
    }

    #[test]
    fn test_parse_catalog_key() {
        let key = b"Symbol\0language".to_vec();
        let (label, property) = parse_catalog_key(&key).unwrap();
        assert_eq!(label, "Symbol");
        assert_eq!(property, "language");
    }

    #[test]
    fn test_make_index_key() {
        let entity_id = EntityId::new(42);
        let value = Value::String("rust".to_string());
        let key = make_index_key("Symbol", "language", &value, entity_id).unwrap();

        // Key should end with entity ID
        let id_bytes = &key[key.len() - 8..];
        assert_eq!(id_bytes, &42u64.to_be_bytes());

        // Key should start with label\0property\0
        assert!(key.starts_with(b"Symbol\0language\0"));
    }

    #[test]
    fn test_extract_entity_id() {
        let entity_id = EntityId::new(12345);
        let value = Value::String("test".to_string());
        let key = make_index_key("Label", "prop", &value, entity_id).unwrap();

        let extracted = extract_entity_id_from_key(&key).unwrap();
        assert_eq!(extracted.as_u64(), 12345);
    }

    #[test]
    fn test_index_metadata_new() {
        let meta = IndexMetadata::new("Symbol", "language", IndexType::Equality);
        assert_eq!(meta.label, "Symbol");
        assert_eq!(meta.property, "language");
        assert_eq!(meta.index_type, IndexType::Equality);
        assert_eq!(meta.entry_count, 0);
    }

    #[test]
    fn test_index_type_default() {
        let index_type: IndexType = IndexType::default();
        assert_eq!(index_type, IndexType::Equality);
    }

    #[test]
    fn test_index_key_ordering() {
        // Keys with same label/property should sort by value then entity ID
        let id1 = EntityId::new(1);
        let id2 = EntityId::new(2);

        let key_a1 = make_index_key("L", "p", &Value::String("a".into()), id1).unwrap();
        let key_a2 = make_index_key("L", "p", &Value::String("a".into()), id2).unwrap();
        let key_b1 = make_index_key("L", "p", &Value::String("b".into()), id1).unwrap();

        assert!(key_a1 < key_a2); // Same value, id1 < id2
        assert!(key_a2 < key_b1); // "a" < "b"
    }
}
