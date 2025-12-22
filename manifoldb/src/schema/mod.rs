//! Schema metadata storage for DDL operations.
//!
//! This module provides persistence for table and index definitions
//! using the metadata table in the storage layer.

// Allow missing docs for internal schema types
#![allow(missing_docs)]

use manifoldb_core::TransactionError;
use manifoldb_query::ast::{ColumnConstraint, ColumnDef, DataType, IndexColumn, TableConstraint};
use manifoldb_query::plan::logical::{CreateIndexNode, CreateTableNode};
use manifoldb_storage::Transaction;
use serde::{Deserialize, Serialize};

use crate::transaction::DatabaseTransaction;

/// Prefix for table schema metadata keys.
const TABLE_PREFIX: &[u8] = b"schema:table:";
/// Prefix for index metadata keys.
const INDEX_PREFIX: &[u8] = b"schema:index:";
/// Key for the list of all tables.
const TABLES_LIST_KEY: &[u8] = b"schema:tables_list";
/// Key for the list of all indexes.
const INDEXES_LIST_KEY: &[u8] = b"schema:indexes_list";
/// Key for the schema version counter.
const SCHEMA_VERSION_KEY: &[u8] = b"schema:version";

/// Stored table schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableSchema {
    /// The table name.
    pub name: String,
    /// Column definitions.
    pub columns: Vec<StoredColumnDef>,
    /// Table constraints.
    pub constraints: Vec<StoredTableConstraint>,
}

/// Stored column definition (serializable version of ColumnDef).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredColumnDef {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub data_type: StoredDataType,
    /// Column constraints.
    pub constraints: Vec<StoredColumnConstraint>,
}

/// Stored data type (simplified for serialization).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StoredDataType {
    Boolean,
    SmallInt,
    Integer,
    BigInt,
    Real,
    DoublePrecision,
    Numeric { precision: Option<u32>, scale: Option<u32> },
    Varchar { length: Option<u32> },
    Text,
    Bytea,
    Timestamp,
    Date,
    Time,
    Interval,
    Json,
    Jsonb,
    Uuid,
    Vector { dimensions: Option<u32> },
    Array { element_type: Box<StoredDataType> },
    Custom { name: String },
}

/// Stored column constraint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StoredColumnConstraint {
    NotNull,
    Unique,
    PrimaryKey,
    ForeignKey { table: String, column: String },
    Check { expression: String },
    Default { expression: String },
}

/// Stored table constraint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StoredTableConstraint {
    PrimaryKey {
        columns: Vec<String>,
        name: Option<String>,
    },
    Unique {
        columns: Vec<String>,
        name: Option<String>,
    },
    ForeignKey {
        columns: Vec<String>,
        ref_table: String,
        ref_columns: Vec<String>,
        name: Option<String>,
    },
    Check {
        expression: String,
        name: Option<String>,
    },
}

/// Stored index definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexSchema {
    /// Index name.
    pub name: String,
    /// Table the index is on.
    pub table: String,
    /// Whether this is a unique index.
    pub unique: bool,
    /// Columns/expressions in the index.
    pub columns: Vec<StoredIndexColumn>,
    /// Index method (btree, hash, hnsw, ivfflat).
    pub using: Option<String>,
    /// Index options.
    pub with_options: Vec<(String, String)>,
    /// Partial index WHERE clause.
    pub where_clause: Option<String>,
}

/// Stored index column.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredIndexColumn {
    /// Column name or expression.
    pub expr: String,
    /// Sort order (ascending/descending).
    pub ascending: bool,
    /// Nulls first/last.
    pub nulls_first: Option<bool>,
}

impl TableSchema {
    /// Create a new table schema from a CREATE TABLE node.
    pub fn from_create_table(node: &CreateTableNode) -> Self {
        Self {
            name: node.name.clone(),
            columns: node.columns.iter().map(StoredColumnDef::from_column_def).collect(),
            constraints: node
                .constraints
                .iter()
                .map(StoredTableConstraint::from_table_constraint)
                .collect(),
        }
    }
}

impl StoredColumnDef {
    /// Create from an AST ColumnDef.
    pub fn from_column_def(def: &ColumnDef) -> Self {
        Self {
            name: def.name.name.clone(),
            data_type: StoredDataType::from_data_type(&def.data_type),
            constraints: def
                .constraints
                .iter()
                .map(StoredColumnConstraint::from_constraint)
                .collect(),
        }
    }
}

impl StoredDataType {
    /// Create from an AST DataType.
    pub fn from_data_type(dt: &DataType) -> Self {
        match dt {
            DataType::Boolean => Self::Boolean,
            DataType::SmallInt => Self::SmallInt,
            DataType::Integer => Self::Integer,
            DataType::BigInt => Self::BigInt,
            DataType::Real => Self::Real,
            DataType::DoublePrecision => Self::DoublePrecision,
            DataType::Numeric { precision, scale } => {
                Self::Numeric { precision: *precision, scale: *scale }
            }
            DataType::Varchar(length) => Self::Varchar { length: *length },
            DataType::Text => Self::Text,
            DataType::Bytea => Self::Bytea,
            DataType::Timestamp => Self::Timestamp,
            DataType::Date => Self::Date,
            DataType::Time => Self::Time,
            DataType::Interval => Self::Interval,
            DataType::Json => Self::Json,
            DataType::Jsonb => Self::Jsonb,
            DataType::Uuid => Self::Uuid,
            DataType::Vector(dimensions) => Self::Vector { dimensions: *dimensions },
            DataType::Array(element_type) => {
                Self::Array { element_type: Box::new(Self::from_data_type(element_type)) }
            }
            DataType::Custom(name) => Self::Custom { name: name.clone() },
        }
    }
}

impl StoredColumnConstraint {
    /// Create from an AST ColumnConstraint.
    pub fn from_constraint(c: &ColumnConstraint) -> Self {
        match c {
            ColumnConstraint::NotNull => Self::NotNull,
            ColumnConstraint::Null => Self::NotNull, // Treat explicit NULL as not-null for now
            ColumnConstraint::Unique => Self::Unique,
            ColumnConstraint::PrimaryKey => Self::PrimaryKey,
            ColumnConstraint::References { table, column } => Self::ForeignKey {
                table: table.parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join("."),
                column: column.as_ref().map_or(String::new(), |c| c.name.clone()),
            },
            ColumnConstraint::Check(expr) => Self::Check { expression: format!("{expr:?}") },
            ColumnConstraint::Default(expr) => Self::Default { expression: format!("{expr:?}") },
        }
    }
}

impl StoredTableConstraint {
    /// Create from an AST TableConstraint.
    pub fn from_table_constraint(tc: &TableConstraint) -> Self {
        match tc {
            TableConstraint::PrimaryKey { columns, name } => Self::PrimaryKey {
                columns: columns.iter().map(|c| c.name.clone()).collect(),
                name: name.as_ref().map(|n| n.name.clone()),
            },
            TableConstraint::Unique { columns, name } => Self::Unique {
                columns: columns.iter().map(|c| c.name.clone()).collect(),
                name: name.as_ref().map(|n| n.name.clone()),
            },
            TableConstraint::ForeignKey { columns, references_table, references_columns, name } => {
                Self::ForeignKey {
                    columns: columns.iter().map(|c| c.name.clone()).collect(),
                    ref_table: references_table
                        .parts
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join("."),
                    ref_columns: references_columns.iter().map(|c| c.name.clone()).collect(),
                    name: name.as_ref().map(|n| n.name.clone()),
                }
            }
            TableConstraint::Check { expr, name } => Self::Check {
                expression: format!("{expr:?}"),
                name: name.as_ref().map(|n| n.name.clone()),
            },
        }
    }
}

impl IndexSchema {
    /// Create a new index schema from a CREATE INDEX node.
    pub fn from_create_index(node: &CreateIndexNode) -> Self {
        Self {
            name: node.name.clone(),
            table: node.table.clone(),
            unique: node.unique,
            columns: node.columns.iter().map(StoredIndexColumn::from_index_column).collect(),
            using: node.using.clone(),
            with_options: node.with.clone(),
            where_clause: node.where_clause.as_ref().map(|e| format!("{e:?}")),
        }
    }
}

impl StoredIndexColumn {
    /// Create from an AST IndexColumn.
    pub fn from_index_column(ic: &IndexColumn) -> Self {
        Self {
            expr: format!("{:?}", ic.expr),
            ascending: ic.asc.unwrap_or(true),
            nulls_first: ic.nulls_first,
        }
    }
}

/// Schema manager for DDL operations.
pub struct SchemaManager;

impl SchemaManager {
    /// Get the current schema version.
    ///
    /// Returns 0 if no schema version has been set yet.
    pub fn get_version<T: Transaction>(tx: &DatabaseTransaction<T>) -> Result<u64, SchemaError> {
        match tx.get_metadata(SCHEMA_VERSION_KEY)? {
            Some(bytes) => {
                let (version, _): (u64, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                Ok(version)
            }
            None => Ok(0),
        }
    }

    /// Increment and return the new schema version.
    ///
    /// This should be called after any DDL operation that modifies the schema.
    fn increment_version<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
    ) -> Result<u64, SchemaError> {
        let current = Self::get_version(tx)?;
        let new_version = current + 1;
        let value = bincode::serde::encode_to_vec(new_version, bincode::config::standard())
            .map_err(|e| SchemaError::Serialization(e.to_string()))?;
        tx.put_metadata(SCHEMA_VERSION_KEY, &value)?;
        Ok(new_version)
    }

    /// Create a new table schema.
    pub fn create_table<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        node: &CreateTableNode,
    ) -> Result<(), SchemaError> {
        let table_name = &node.name;

        // Check if table already exists
        if Self::table_exists(tx, table_name)? {
            if node.if_not_exists {
                return Ok(());
            }
            return Err(SchemaError::TableExists(table_name.clone()));
        }

        // Create and store the schema
        let schema = TableSchema::from_create_table(node);
        let key = Self::table_key(table_name);
        let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
            .map_err(|e| SchemaError::Serialization(e.to_string()))?;

        tx.put_metadata(&key, &value)?;

        // Add to tables list
        Self::add_to_list(tx, TABLES_LIST_KEY, table_name)?;

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Drop a table schema.
    pub fn drop_table<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        table_name: &str,
        if_exists: bool,
    ) -> Result<(), SchemaError> {
        // Check if table exists
        if !Self::table_exists(tx, table_name)? {
            if if_exists {
                return Ok(());
            }
            return Err(SchemaError::TableNotFound(table_name.to_string()));
        }

        // Remove the schema
        let key = Self::table_key(table_name);
        tx.delete_metadata(&key)?;

        // Remove from tables list
        Self::remove_from_list(tx, TABLES_LIST_KEY, table_name)?;

        // Also remove any indexes on this table
        let indexes = Self::list_indexes_for_table(tx, table_name)?;
        for idx_name in indexes {
            Self::drop_index(tx, &idx_name, true)?;
        }

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Create an index schema.
    pub fn create_index<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        node: &CreateIndexNode,
    ) -> Result<(), SchemaError> {
        let index_name = &node.name;

        // Check if index already exists
        if Self::index_exists(tx, index_name)? {
            if node.if_not_exists {
                return Ok(());
            }
            return Err(SchemaError::IndexExists(index_name.clone()));
        }

        // Check if table exists
        if !Self::table_exists(tx, &node.table)? {
            return Err(SchemaError::TableNotFound(node.table.clone()));
        }

        // Create and store the index schema
        let schema = IndexSchema::from_create_index(node);
        let key = Self::index_key(index_name);
        let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
            .map_err(|e| SchemaError::Serialization(e.to_string()))?;

        tx.put_metadata(&key, &value)?;

        // Add to indexes list
        Self::add_to_list(tx, INDEXES_LIST_KEY, index_name)?;

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Drop an index schema.
    pub fn drop_index<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        index_name: &str,
        if_exists: bool,
    ) -> Result<(), SchemaError> {
        // Check if index exists
        if !Self::index_exists(tx, index_name)? {
            if if_exists {
                return Ok(());
            }
            return Err(SchemaError::IndexNotFound(index_name.to_string()));
        }

        // Remove the schema
        let key = Self::index_key(index_name);
        tx.delete_metadata(&key)?;

        // Remove from indexes list
        Self::remove_from_list(tx, INDEXES_LIST_KEY, index_name)?;

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Check if a table exists.
    pub fn table_exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<bool, SchemaError> {
        let key = Self::table_key(name);
        Ok(tx.get_metadata(&key)?.is_some())
    }

    /// Check if an index exists.
    pub fn index_exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<bool, SchemaError> {
        let key = Self::index_key(name);
        Ok(tx.get_metadata(&key)?.is_some())
    }

    /// Get a table schema.
    pub fn get_table<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<Option<TableSchema>, SchemaError> {
        let key = Self::table_key(name);
        match tx.get_metadata(&key)? {
            Some(bytes) => {
                let (schema, _): (TableSchema, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    /// Get an index schema.
    pub fn get_index<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<Option<IndexSchema>, SchemaError> {
        let key = Self::index_key(name);
        match tx.get_metadata(&key)? {
            Some(bytes) => {
                let (schema, _): (IndexSchema, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    /// List all tables.
    pub fn list_tables<T: Transaction>(
        tx: &DatabaseTransaction<T>,
    ) -> Result<Vec<String>, SchemaError> {
        Self::get_list(tx, TABLES_LIST_KEY)
    }

    /// List all indexes.
    pub fn list_indexes<T: Transaction>(
        tx: &DatabaseTransaction<T>,
    ) -> Result<Vec<String>, SchemaError> {
        Self::get_list(tx, INDEXES_LIST_KEY)
    }

    /// List all indexes for a specific table.
    pub fn list_indexes_for_table<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table_name: &str,
    ) -> Result<Vec<String>, SchemaError> {
        let all_indexes = Self::list_indexes(tx)?;
        let mut result = Vec::new();

        for idx_name in all_indexes {
            if let Some(schema) = Self::get_index(tx, &idx_name)? {
                if schema.table == table_name {
                    result.push(idx_name);
                }
            }
        }

        Ok(result)
    }

    // Helper methods

    fn table_key(name: &str) -> Vec<u8> {
        let mut key = TABLE_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn index_key(name: &str) -> Vec<u8> {
        let mut key = INDEX_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn get_list<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        list_key: &[u8],
    ) -> Result<Vec<String>, SchemaError> {
        match tx.get_metadata(list_key)? {
            Some(bytes) => {
                let (list, _): (Vec<String>, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                Ok(list)
            }
            None => Ok(Vec::new()),
        }
    }

    fn add_to_list<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        list_key: &[u8],
        name: &str,
    ) -> Result<(), SchemaError> {
        let mut list = Self::get_list(tx, list_key)?;
        if !list.contains(&name.to_string()) {
            list.push(name.to_string());
            let value = bincode::serde::encode_to_vec(&list, bincode::config::standard())
                .map_err(|e| SchemaError::Serialization(e.to_string()))?;
            tx.put_metadata(list_key, &value)?;
        }
        Ok(())
    }

    fn remove_from_list<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        list_key: &[u8],
        name: &str,
    ) -> Result<(), SchemaError> {
        let mut list = Self::get_list(tx, list_key)?;
        list.retain(|n| n != name);
        let value = bincode::serde::encode_to_vec(&list, bincode::config::standard())
            .map_err(|e| SchemaError::Serialization(e.to_string()))?;
        tx.put_metadata(list_key, &value)?;
        Ok(())
    }
}

/// Errors that can occur during schema operations.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("Table already exists: {0}")]
    TableExists(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Index already exists: {0}")]
    IndexExists(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    #[error("Serialization error: {0}")]
    Serialization(String),
}
