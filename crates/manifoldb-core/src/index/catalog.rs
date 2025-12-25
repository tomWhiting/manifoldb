//! Index metadata catalog for tracking index definitions.
//!
//! This module provides the [`IndexDef`] struct for representing index metadata
//! and the [`IndexCatalog`] for managing index definitions in memory with
//! persistence support.
//!
//! # Overview
//!
//! The index catalog is a system table that tracks all user-created indexes.
//! It stores metadata including:
//!
//! - Index name and unique identifier
//! - Target table/collection name
//! - Indexed columns (ordered list)
//! - Index type (btree, hash, fulltext, etc.)
//! - Uniqueness constraint
//! - Creation timestamp
//!
//! # Example
//!
//! ```
//! use manifoldb_core::index::{IndexDef, IndexType, IndexCatalog};
//!
//! let mut catalog = IndexCatalog::new();
//!
//! // Create an index definition
//! let def = IndexDef::builder("idx_users_email", "users")
//!     .column("email")
//!     .unique(true)
//!     .build();
//!
//! // Register the index
//! let id = catalog.create_index(def).unwrap();
//!
//! // Look up the index
//! let found = catalog.get_index("idx_users_email");
//! assert!(found.is_some());
//!
//! // Find indexes for a column
//! let indexes = catalog.find_indexes_for_column("users", "email");
//! assert_eq!(indexes.len(), 1);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique identifier for an index definition.
///
/// This is distinct from [`super::IndexId`] which is used for property index
/// key encoding. `CatalogIndexId` is a simple auto-incrementing counter for
/// the catalog system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CatalogIndexId(u64);

impl CatalogIndexId {
    /// Create a new catalog index ID.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

/// Type of index.
///
/// Designed for extensibility - currently only btree is implemented,
/// but the enum allows for future index types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum IndexType {
    /// B-tree index, optimized for range queries and equality lookups.
    #[default]
    BTree,
    /// Hash index, optimized for equality lookups only.
    Hash,
    /// Full-text search index.
    FullText,
    /// Vector similarity index (for embedding search).
    Vector,
}

impl IndexType {
    /// Get a string representation of the index type.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            IndexType::BTree => "btree",
            IndexType::Hash => "hash",
            IndexType::FullText => "fulltext",
            IndexType::Vector => "vector",
        }
    }

    /// Parse an index type from a string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "btree" => Some(IndexType::BTree),
            "hash" => Some(IndexType::Hash),
            "fulltext" => Some(IndexType::FullText),
            "vector" => Some(IndexType::Vector),
            _ => None,
        }
    }
}

/// Index definition containing all metadata about an index.
///
/// This struct represents the complete metadata for a secondary index,
/// including its name, target table, indexed columns, and configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexDef {
    /// Unique identifier assigned by the catalog.
    pub id: CatalogIndexId,
    /// User-provided name for the index (e.g., "idx_users_email").
    pub name: String,
    /// Target table/collection/label this index covers.
    pub table_name: String,
    /// Ordered list of property names being indexed.
    pub columns: Vec<String>,
    /// Type of index (btree, hash, fulltext, etc.).
    pub index_type: IndexType,
    /// Whether this index enforces uniqueness.
    pub is_unique: bool,
    /// Unix timestamp when the index was created.
    pub created_at: u64,
}

impl IndexDef {
    /// Create a new index definition builder.
    ///
    /// # Arguments
    ///
    /// * `name` - User-provided name for the index
    /// * `table_name` - Target table/collection this index covers
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::index::{IndexDef, IndexType};
    ///
    /// let def = IndexDef::builder("idx_users_email", "users")
    ///     .column("email")
    ///     .index_type(IndexType::BTree)
    ///     .unique(true)
    ///     .build();
    ///
    /// assert_eq!(def.name, "idx_users_email");
    /// assert_eq!(def.table_name, "users");
    /// assert_eq!(def.columns, vec!["email"]);
    /// assert!(def.is_unique);
    /// ```
    #[must_use]
    pub fn builder(name: impl Into<String>, table_name: impl Into<String>) -> IndexDefBuilder {
        IndexDefBuilder::new(name, table_name)
    }

    /// Check if this index covers a specific column.
    #[must_use]
    pub fn covers_column(&self, column: &str) -> bool {
        self.columns.iter().any(|c| c == column)
    }

    /// Check if this is a single-column index.
    #[must_use]
    pub fn is_single_column(&self) -> bool {
        self.columns.len() == 1
    }

    /// Check if this is a composite (multi-column) index.
    #[must_use]
    pub fn is_composite(&self) -> bool {
        self.columns.len() > 1
    }

    /// Encode this definition for storage.
    ///
    /// Format: `id|name|table_name|columns_csv|index_type|is_unique|created_at`
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let columns_csv = self.columns.join(",");
        let data = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            self.id.as_u64(),
            self.name,
            self.table_name,
            columns_csv,
            self.index_type.as_str(),
            self.is_unique,
            self.created_at
        );
        data.into_bytes()
    }

    /// Decode a definition from storage.
    ///
    /// Returns `None` if the data is malformed.
    #[must_use]
    pub fn decode(data: &[u8]) -> Option<Self> {
        let s = std::str::from_utf8(data).ok()?;
        let parts: Vec<&str> = s.split('|').collect();
        if parts.len() != 7 {
            return None;
        }

        let id = parts[0].parse::<u64>().ok()?;
        let name = parts[1].to_string();
        let table_name = parts[2].to_string();
        let columns: Vec<String> = if parts[3].is_empty() {
            Vec::new()
        } else {
            parts[3].split(',').map(String::from).collect()
        };
        let index_type = IndexType::from_str(parts[4])?;
        let is_unique = parts[5].parse::<bool>().ok()?;
        let created_at = parts[6].parse::<u64>().ok()?;

        Some(Self {
            id: CatalogIndexId::new(id),
            name,
            table_name,
            columns,
            index_type,
            is_unique,
            created_at,
        })
    }
}

/// Builder for constructing [`IndexDef`] instances.
///
/// Use [`IndexDef::builder`] to create a new builder.
#[derive(Debug)]
pub struct IndexDefBuilder {
    name: String,
    table_name: String,
    columns: Vec<String>,
    index_type: IndexType,
    is_unique: bool,
    created_at: Option<u64>,
}

impl IndexDefBuilder {
    /// Create a new builder with required fields.
    fn new(name: impl Into<String>, table_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            table_name: table_name.into(),
            columns: Vec::new(),
            index_type: IndexType::default(),
            is_unique: false,
            created_at: None,
        }
    }

    /// Add a column to the index.
    #[must_use]
    pub fn column(mut self, column: impl Into<String>) -> Self {
        self.columns.push(column.into());
        self
    }

    /// Add multiple columns to the index.
    #[must_use]
    pub fn columns<I, S>(mut self, columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.columns.extend(columns.into_iter().map(Into::into));
        self
    }

    /// Set the index type.
    #[must_use]
    pub fn index_type(mut self, index_type: IndexType) -> Self {
        self.index_type = index_type;
        self
    }

    /// Set whether the index enforces uniqueness.
    #[must_use]
    pub fn unique(mut self, is_unique: bool) -> Self {
        self.is_unique = is_unique;
        self
    }

    /// Set the creation timestamp (for loading from storage).
    #[must_use]
    pub fn created_at(mut self, timestamp: u64) -> Self {
        self.created_at = Some(timestamp);
        self
    }

    /// Build the index definition.
    ///
    /// The ID will be set to 0 - it gets assigned when registered with the catalog.
    #[must_use]
    pub fn build(self) -> IndexDef {
        let created_at = self.created_at.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        });

        IndexDef {
            id: CatalogIndexId::new(0), // Assigned by catalog
            name: self.name,
            table_name: self.table_name,
            columns: self.columns,
            index_type: self.index_type,
            is_unique: self.is_unique,
            created_at,
        }
    }
}

/// Error type for index catalog operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    /// An index with this name already exists.
    IndexAlreadyExists(String),
    /// No index found with the given name.
    IndexNotFound(String),
    /// Index definition is invalid.
    InvalidDefinition(String),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::IndexAlreadyExists(name) => {
                write!(f, "index already exists: {name}")
            }
            CatalogError::IndexNotFound(name) => {
                write!(f, "index not found: {name}")
            }
            CatalogError::InvalidDefinition(msg) => {
                write!(f, "invalid index definition: {msg}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

/// In-memory catalog of index definitions.
///
/// The catalog tracks all user-created indexes and provides fast lookups
/// by name, table, and column. It is designed to be loaded into memory
/// on database startup for efficient query planning.
///
/// # Thread Safety
///
/// The catalog itself is not thread-safe. Wrap it in appropriate
/// synchronization primitives (e.g., `RwLock`) for concurrent access.
#[derive(Debug, Default)]
pub struct IndexCatalog {
    /// Map from index name to definition.
    by_name: HashMap<String, IndexDef>,
    /// Map from table name to index names.
    by_table: HashMap<String, Vec<String>>,
    /// Next ID to assign.
    next_id: u64,
}

impl IndexCatalog {
    /// Create a new empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a catalog and load existing definitions.
    ///
    /// # Arguments
    ///
    /// * `definitions` - Iterator of encoded index definitions
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::index::{IndexCatalog, IndexDef};
    ///
    /// // In practice, you'd load these from storage
    /// let stored: Vec<Vec<u8>> = Vec::new();
    /// let catalog = IndexCatalog::load(stored.iter().map(|v| v.as_slice()));
    /// ```
    #[must_use]
    pub fn load<'a>(definitions: impl Iterator<Item = &'a [u8]>) -> Self {
        let mut catalog = Self::new();
        for data in definitions {
            if let Some(def) = IndexDef::decode(data) {
                // Track max ID for next_id
                if def.id.as_u64() >= catalog.next_id {
                    catalog.next_id = def.id.as_u64() + 1;
                }
                // Insert into indexes
                catalog.by_table.entry(def.table_name.clone()).or_default().push(def.name.clone());
                catalog.by_name.insert(def.name.clone(), def);
            }
        }
        catalog
    }

    /// Register a new index definition.
    ///
    /// # Arguments
    ///
    /// * `def` - The index definition to register
    ///
    /// # Returns
    ///
    /// The assigned `CatalogIndexId` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - An index with the same name already exists
    /// - The definition is invalid (no columns, empty name, etc.)
    pub fn create_index(&mut self, mut def: IndexDef) -> Result<CatalogIndexId, CatalogError> {
        // Validate the definition
        if def.name.is_empty() {
            return Err(CatalogError::InvalidDefinition("index name cannot be empty".into()));
        }
        if def.table_name.is_empty() {
            return Err(CatalogError::InvalidDefinition("table name cannot be empty".into()));
        }
        if def.columns.is_empty() {
            return Err(CatalogError::InvalidDefinition("index must have at least one column".into()));
        }

        // Check for duplicate name
        if self.by_name.contains_key(&def.name) {
            return Err(CatalogError::IndexAlreadyExists(def.name));
        }

        // Assign ID
        let id = CatalogIndexId::new(self.next_id);
        self.next_id += 1;
        def.id = id;

        // Insert into indexes
        self.by_table.entry(def.table_name.clone()).or_default().push(def.name.clone());
        self.by_name.insert(def.name.clone(), def);

        Ok(id)
    }

    /// Remove an index by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the index to remove
    ///
    /// # Returns
    ///
    /// The removed index definition on success.
    ///
    /// # Errors
    ///
    /// Returns an error if no index exists with the given name.
    pub fn drop_index(&mut self, name: &str) -> Result<IndexDef, CatalogError> {
        let def = self.by_name.remove(name).ok_or_else(|| CatalogError::IndexNotFound(name.into()))?;

        // Remove from table index
        if let Some(indexes) = self.by_table.get_mut(&def.table_name) {
            indexes.retain(|n| n != name);
            if indexes.is_empty() {
                self.by_table.remove(&def.table_name);
            }
        }

        Ok(def)
    }

    /// Get an index definition by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the index
    ///
    /// # Returns
    ///
    /// A reference to the index definition, or `None` if not found.
    #[must_use]
    pub fn get_index(&self, name: &str) -> Option<&IndexDef> {
        self.by_name.get(name)
    }

    /// List all indexes for a table.
    ///
    /// # Arguments
    ///
    /// * `table_name` - The table to list indexes for
    ///
    /// # Returns
    ///
    /// A vector of references to index definitions.
    #[must_use]
    pub fn list_indexes(&self, table_name: &str) -> Vec<&IndexDef> {
        self.by_table
            .get(table_name)
            .map(|names| names.iter().filter_map(|n| self.by_name.get(n)).collect())
            .unwrap_or_default()
    }

    /// Find indexes that cover a specific column.
    ///
    /// Returns all indexes for the given table that include the specified
    /// column in their column list.
    ///
    /// # Arguments
    ///
    /// * `table_name` - The table to search
    /// * `column` - The column name to find
    ///
    /// # Returns
    ///
    /// A vector of references to matching index definitions.
    #[must_use]
    pub fn find_indexes_for_column(&self, table_name: &str, column: &str) -> Vec<&IndexDef> {
        self.list_indexes(table_name)
            .into_iter()
            .filter(|def| def.covers_column(column))
            .collect()
    }

    /// Get all index definitions.
    ///
    /// # Returns
    ///
    /// An iterator over all index definitions in the catalog.
    pub fn all_indexes(&self) -> impl Iterator<Item = &IndexDef> {
        self.by_name.values()
    }

    /// Check if an index with the given name exists.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Get the number of indexes in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Check if the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn index_def_builder_basic() {
        let def = IndexDef::builder("idx_users_email", "users")
            .column("email")
            .unique(true)
            .build();

        assert_eq!(def.name, "idx_users_email");
        assert_eq!(def.table_name, "users");
        assert_eq!(def.columns, vec!["email"]);
        assert!(def.is_unique);
        assert_eq!(def.index_type, IndexType::BTree);
    }

    #[test]
    fn index_def_builder_composite() {
        let def = IndexDef::builder("idx_orders_customer_date", "orders")
            .columns(["customer_id", "order_date"])
            .index_type(IndexType::BTree)
            .build();

        assert_eq!(def.columns, vec!["customer_id", "order_date"]);
        assert!(def.is_composite());
        assert!(!def.is_single_column());
    }

    #[test]
    fn index_def_covers_column() {
        let def = IndexDef::builder("idx_test", "test")
            .columns(["a", "b", "c"])
            .build();

        assert!(def.covers_column("a"));
        assert!(def.covers_column("b"));
        assert!(def.covers_column("c"));
        assert!(!def.covers_column("d"));
    }

    #[test]
    fn index_def_encode_decode_roundtrip() {
        let def = IndexDef::builder("idx_users_email", "users")
            .column("email")
            .index_type(IndexType::BTree)
            .unique(true)
            .created_at(1234567890)
            .build();

        // Need to set ID for encoding
        let mut def_with_id = def;
        def_with_id.id = CatalogIndexId::new(42);

        let encoded = def_with_id.encode();
        let decoded = IndexDef::decode(&encoded).unwrap();

        assert_eq!(decoded.id, CatalogIndexId::new(42));
        assert_eq!(decoded.name, "idx_users_email");
        assert_eq!(decoded.table_name, "users");
        assert_eq!(decoded.columns, vec!["email"]);
        assert_eq!(decoded.index_type, IndexType::BTree);
        assert!(decoded.is_unique);
        assert_eq!(decoded.created_at, 1234567890);
    }

    #[test]
    fn index_def_encode_decode_composite() {
        let def = IndexDef::builder("idx_composite", "test")
            .columns(["a", "b", "c"])
            .index_type(IndexType::Hash)
            .created_at(9999999999)
            .build();

        let mut def_with_id = def;
        def_with_id.id = CatalogIndexId::new(100);

        let encoded = def_with_id.encode();
        let decoded = IndexDef::decode(&encoded).unwrap();

        assert_eq!(decoded.columns, vec!["a", "b", "c"]);
        assert_eq!(decoded.index_type, IndexType::Hash);
    }

    #[test]
    fn catalog_create_and_get() {
        let mut catalog = IndexCatalog::new();

        let def = IndexDef::builder("idx_users_email", "users")
            .column("email")
            .build();

        let id = catalog.create_index(def).unwrap();
        assert_eq!(id, CatalogIndexId::new(0));

        let retrieved = catalog.get_index("idx_users_email").unwrap();
        assert_eq!(retrieved.name, "idx_users_email");
        assert_eq!(retrieved.id, id);
    }

    #[test]
    fn catalog_create_duplicate_fails() {
        let mut catalog = IndexCatalog::new();

        let def1 = IndexDef::builder("idx_test", "users").column("email").build();
        let def2 = IndexDef::builder("idx_test", "users").column("name").build();

        catalog.create_index(def1).unwrap();
        let result = catalog.create_index(def2);

        assert!(matches!(result, Err(CatalogError::IndexAlreadyExists(_))));
    }

    #[test]
    fn catalog_create_invalid_fails() {
        let mut catalog = IndexCatalog::new();

        // Empty name
        let def1 = IndexDef::builder("", "users").column("email").build();
        assert!(matches!(catalog.create_index(def1), Err(CatalogError::InvalidDefinition(_))));

        // Empty table
        let def2 = IndexDef::builder("idx_test", "").column("email").build();
        assert!(matches!(catalog.create_index(def2), Err(CatalogError::InvalidDefinition(_))));

        // No columns
        let def3 = IndexDef::builder("idx_test", "users").build();
        assert!(matches!(catalog.create_index(def3), Err(CatalogError::InvalidDefinition(_))));
    }

    #[test]
    fn catalog_drop_index() {
        let mut catalog = IndexCatalog::new();

        let def = IndexDef::builder("idx_test", "users").column("email").build();
        catalog.create_index(def).unwrap();

        assert!(catalog.contains("idx_test"));

        let dropped = catalog.drop_index("idx_test").unwrap();
        assert_eq!(dropped.name, "idx_test");

        assert!(!catalog.contains("idx_test"));
        assert!(catalog.get_index("idx_test").is_none());
    }

    #[test]
    fn catalog_drop_nonexistent_fails() {
        let mut catalog = IndexCatalog::new();

        let result = catalog.drop_index("nonexistent");
        assert!(matches!(result, Err(CatalogError::IndexNotFound(_))));
    }

    #[test]
    fn catalog_list_indexes() {
        let mut catalog = IndexCatalog::new();

        let def1 = IndexDef::builder("idx_users_email", "users").column("email").build();
        let def2 = IndexDef::builder("idx_users_name", "users").column("name").build();
        let def3 = IndexDef::builder("idx_orders_id", "orders").column("id").build();

        catalog.create_index(def1).unwrap();
        catalog.create_index(def2).unwrap();
        catalog.create_index(def3).unwrap();

        let user_indexes = catalog.list_indexes("users");
        assert_eq!(user_indexes.len(), 2);

        let order_indexes = catalog.list_indexes("orders");
        assert_eq!(order_indexes.len(), 1);

        let empty_indexes = catalog.list_indexes("nonexistent");
        assert!(empty_indexes.is_empty());
    }

    #[test]
    fn catalog_find_indexes_for_column() {
        let mut catalog = IndexCatalog::new();

        let def1 = IndexDef::builder("idx_single", "users").column("email").build();
        let def2 = IndexDef::builder("idx_composite", "users")
            .columns(["email", "name"])
            .build();
        let def3 = IndexDef::builder("idx_other", "users").column("age").build();

        catalog.create_index(def1).unwrap();
        catalog.create_index(def2).unwrap();
        catalog.create_index(def3).unwrap();

        let email_indexes = catalog.find_indexes_for_column("users", "email");
        assert_eq!(email_indexes.len(), 2);

        let name_indexes = catalog.find_indexes_for_column("users", "name");
        assert_eq!(name_indexes.len(), 1);

        let age_indexes = catalog.find_indexes_for_column("users", "age");
        assert_eq!(age_indexes.len(), 1);

        let missing_indexes = catalog.find_indexes_for_column("users", "nonexistent");
        assert!(missing_indexes.is_empty());
    }

    #[test]
    fn catalog_load() {
        // Create some definitions and encode them
        let mut def1 = IndexDef::builder("idx_a", "users").column("email").build();
        def1.id = CatalogIndexId::new(5);

        let mut def2 = IndexDef::builder("idx_b", "users").column("name").build();
        def2.id = CatalogIndexId::new(10);

        let encoded1 = def1.encode();
        let encoded2 = def2.encode();
        let data: Vec<&[u8]> = vec![&encoded1, &encoded2];

        let catalog = IndexCatalog::load(data.into_iter());

        assert_eq!(catalog.len(), 2);
        assert!(catalog.contains("idx_a"));
        assert!(catalog.contains("idx_b"));

        // Next ID should be max + 1
        let retrieved = catalog.get_index("idx_b").unwrap();
        assert_eq!(retrieved.id, CatalogIndexId::new(10));
    }

    #[test]
    fn catalog_all_indexes() {
        let mut catalog = IndexCatalog::new();

        let def1 = IndexDef::builder("idx_a", "users").column("email").build();
        let def2 = IndexDef::builder("idx_b", "orders").column("id").build();

        catalog.create_index(def1).unwrap();
        catalog.create_index(def2).unwrap();

        let all: Vec<_> = catalog.all_indexes().collect();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn catalog_len_and_is_empty() {
        let mut catalog = IndexCatalog::new();

        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);

        let def = IndexDef::builder("idx_test", "users").column("email").build();
        catalog.create_index(def).unwrap();

        assert!(!catalog.is_empty());
        assert_eq!(catalog.len(), 1);
    }

    #[test]
    fn index_type_as_str() {
        assert_eq!(IndexType::BTree.as_str(), "btree");
        assert_eq!(IndexType::Hash.as_str(), "hash");
        assert_eq!(IndexType::FullText.as_str(), "fulltext");
        assert_eq!(IndexType::Vector.as_str(), "vector");
    }

    #[test]
    fn index_type_from_str() {
        assert_eq!(IndexType::from_str("btree"), Some(IndexType::BTree));
        assert_eq!(IndexType::from_str("BTREE"), Some(IndexType::BTree));
        assert_eq!(IndexType::from_str("hash"), Some(IndexType::Hash));
        assert_eq!(IndexType::from_str("fulltext"), Some(IndexType::FullText));
        assert_eq!(IndexType::from_str("vector"), Some(IndexType::Vector));
        assert_eq!(IndexType::from_str("unknown"), None);
    }

    #[test]
    fn catalog_index_id() {
        let id = CatalogIndexId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn catalog_error_display() {
        let err1 = CatalogError::IndexAlreadyExists("idx_test".into());
        assert_eq!(format!("{err1}"), "index already exists: idx_test");

        let err2 = CatalogError::IndexNotFound("idx_missing".into());
        assert_eq!(format!("{err2}"), "index not found: idx_missing");

        let err3 = CatalogError::InvalidDefinition("no columns".into());
        assert_eq!(format!("{err3}"), "invalid index definition: no columns");
    }
}
