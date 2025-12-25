//! Property index types, encoding, and catalog.
//!
//! This module provides types for secondary property indexes that enable
//! efficient queries like `WHERE property = value` and `WHERE property > value`.
//!
//! # Overview
//!
//! Property indexes are stored in the storage layer using composite keys that
//! preserve sort order. This enables efficient:
//!
//! - **Point lookups**: `WHERE age = 30`
//! - **Range scans**: `WHERE age > 18 AND age < 65`
//! - **Prefix matching**: `WHERE name LIKE 'Alice%'`
//!
//! # Index Catalog
//!
//! The [`IndexCatalog`] tracks all user-created indexes and their metadata.
//! It provides fast lookups by name, table, and column for query planning.
//!
//! # Key Format
//!
//! Property index keys have the following structure:
//!
//! ```text
//! [PREFIX_PROPERTY_INDEX][index_id: 8 bytes][sortable_value: variable][entity_id: 8 bytes]
//! ```
//!
//! Where:
//! - `PREFIX_PROPERTY_INDEX` - Distinguishes property indexes from other data
//! - `index_id` - Unique identifier for the index (hash of label + property name)
//! - `sortable_value` - Value encoded to preserve sort order
//! - `entity_id` - The entity that has this property value
//!
//! # Example
//!
//! ```
//! use manifoldb_core::index::{PropertyIndexEntry, IndexId};
//! use manifoldb_core::types::{EntityId, Value};
//!
//! // Create an index entry
//! let index_id = IndexId::from_label_property("Person", "age");
//! let entry = PropertyIndexEntry::new(index_id, Value::Int(30), EntityId::new(42));
//!
//! // Encode to storage key
//! let key = entry.encode_key().unwrap();
//!
//! // Later, decode from storage
//! let decoded = PropertyIndexEntry::decode_key(&key).unwrap();
//! assert_eq!(decoded.entity_id, EntityId::new(42));
//! ```

mod catalog;
mod property;

pub use catalog::{CatalogError, CatalogIndexId, IndexCatalog, IndexDef, IndexDefBuilder, IndexType};
pub use crate::encoding::keys::IndexId;
pub use property::{PropertyIndexEntry, PropertyIndexScan};
