//! Schema metadata storage for DDL operations.
//!
//! This module provides persistence for table, index, and view definitions
//! using the metadata table in the storage layer.

// Allow missing docs for internal schema types
#![allow(missing_docs)]

use manifoldb_core::TransactionError;
use manifoldb_query::ast::{
    AlterColumnAction, AlterTableAction, ColumnConstraint, ColumnDef, DataType, IndexColumn,
    TableConstraint,
};
use manifoldb_query::plan::logical::{
    AlterTableNode, CreateIndexNode, CreateTableNode, CreateViewNode,
};
use manifoldb_storage::Transaction;
use serde::{Deserialize, Serialize};

use crate::transaction::DatabaseTransaction;

/// Prefix for table schema metadata keys.
const TABLE_PREFIX: &[u8] = b"schema:table:";
/// Prefix for index metadata keys.
const INDEX_PREFIX: &[u8] = b"schema:index:";
/// Prefix for view metadata keys.
const VIEW_PREFIX: &[u8] = b"schema:view:";
/// Key for the list of all tables.
const TABLES_LIST_KEY: &[u8] = b"schema:tables_list";
/// Key for the list of all indexes.
const INDEXES_LIST_KEY: &[u8] = b"schema:indexes_list";
/// Key for the list of all views.
const VIEWS_LIST_KEY: &[u8] = b"schema:views_list";
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

/// Stored view definition.
///
/// Views are stored query definitions that can be used like tables.
/// The query is stored as a string representation of the SQL that can be
/// re-parsed when the view is accessed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ViewSchema {
    /// View name.
    pub name: String,
    /// Optional column aliases for the view.
    pub columns: Vec<String>,
    /// The SQL query defining the view (stored as string for flexibility).
    pub query_sql: String,
}

impl ViewSchema {
    /// Create a new view schema from a CREATE VIEW node.
    pub fn from_create_view(node: &CreateViewNode) -> Self {
        // Store the query as its debug representation
        // In production, we would want a proper SQL serializer
        let query_sql = format!("{:?}", node.query);

        Self {
            name: node.name.clone(),
            columns: node.columns.iter().map(|c| c.name.clone()).collect(),
            query_sql,
        }
    }

    /// Create a new view schema with raw query SQL.
    pub fn new(name: String, columns: Vec<String>, query_sql: String) -> Self {
        Self { name, columns, query_sql }
    }

    /// Get the stored SELECT statement if available.
    ///
    /// Note: This returns the raw query stored with the view.
    /// The query needs to be re-parsed from the stored SQL representation.
    pub fn query(&self) -> &str {
        &self.query_sql
    }
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

    /// Alter a table schema.
    ///
    /// Supports the following operations:
    /// - ADD COLUMN
    /// - DROP COLUMN
    /// - ALTER COLUMN (SET NOT NULL, DROP NOT NULL, SET DEFAULT, DROP DEFAULT, SET DATA TYPE)
    /// - RENAME COLUMN
    /// - RENAME TABLE
    /// - ADD CONSTRAINT
    /// - DROP CONSTRAINT
    pub fn alter_table<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        node: &AlterTableNode,
    ) -> Result<(), SchemaError> {
        let table_name = &node.name;

        // Check if table exists
        let mut schema = match Self::get_table(tx, table_name)? {
            Some(s) => s,
            None => {
                if node.if_exists {
                    return Ok(());
                }
                return Err(SchemaError::TableNotFound(table_name.clone()));
            }
        };

        // Track if we need to rename the table
        let mut new_table_name: Option<String> = None;

        // Apply each action
        for action in &node.actions {
            match action {
                AlterTableAction::AddColumn { if_not_exists, column } => {
                    let col_name = &column.name.name;
                    // Check if column already exists
                    if schema.columns.iter().any(|c| c.name == *col_name) {
                        if *if_not_exists {
                            continue;
                        }
                        return Err(SchemaError::ColumnExists(col_name.clone()));
                    }
                    schema.columns.push(StoredColumnDef::from_column_def(column));
                }

                AlterTableAction::DropColumn { if_exists, column_name, cascade: _ } => {
                    let col_name = &column_name.name;
                    let col_idx = schema.columns.iter().position(|c| c.name == *col_name);
                    match col_idx {
                        Some(idx) => {
                            schema.columns.remove(idx);
                        }
                        None => {
                            if !*if_exists {
                                return Err(SchemaError::ColumnNotFound(col_name.clone()));
                            }
                        }
                    }
                }

                AlterTableAction::AlterColumn { column_name, action: col_action } => {
                    let col_name = &column_name.name;
                    let col = schema
                        .columns
                        .iter_mut()
                        .find(|c| c.name == *col_name)
                        .ok_or_else(|| SchemaError::ColumnNotFound(col_name.clone()))?;

                    match col_action {
                        AlterColumnAction::SetNotNull => {
                            // Add NOT NULL constraint if not present
                            if !col
                                .constraints
                                .iter()
                                .any(|c| matches!(c, StoredColumnConstraint::NotNull))
                            {
                                col.constraints.push(StoredColumnConstraint::NotNull);
                            }
                        }
                        AlterColumnAction::DropNotNull => {
                            // Remove NOT NULL constraint
                            col.constraints
                                .retain(|c| !matches!(c, StoredColumnConstraint::NotNull));
                        }
                        AlterColumnAction::SetDefault(expr) => {
                            // Remove any existing DEFAULT, then add new one
                            col.constraints
                                .retain(|c| !matches!(c, StoredColumnConstraint::Default { .. }));
                            col.constraints.push(StoredColumnConstraint::Default {
                                expression: format!("{expr:?}"),
                            });
                        }
                        AlterColumnAction::DropDefault => {
                            // Remove DEFAULT constraint
                            col.constraints
                                .retain(|c| !matches!(c, StoredColumnConstraint::Default { .. }));
                        }
                        AlterColumnAction::SetType { data_type, using: _ } => {
                            // Change the column type
                            col.data_type = StoredDataType::from_data_type(data_type);
                        }
                    }
                }

                AlterTableAction::RenameColumn { old_name, new_name } => {
                    let col = schema
                        .columns
                        .iter_mut()
                        .find(|c| c.name == old_name.name)
                        .ok_or_else(|| SchemaError::ColumnNotFound(old_name.name.clone()))?;
                    col.name.clone_from(&new_name.name);
                }

                AlterTableAction::RenameTable { new_name } => {
                    // Just store for later - we'll handle the actual rename at the end
                    new_table_name = Some(
                        new_name
                            .parts
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>()
                            .join("."),
                    );
                }

                AlterTableAction::AddConstraint(constraint) => {
                    schema
                        .constraints
                        .push(StoredTableConstraint::from_table_constraint(constraint));
                }

                AlterTableAction::DropConstraint { if_exists, constraint_name, cascade: _ } => {
                    let name = &constraint_name.name;
                    let original_len = schema.constraints.len();
                    schema.constraints.retain(|c| {
                        let constraint_name = match c {
                            StoredTableConstraint::PrimaryKey { name, .. } => name.as_ref(),
                            StoredTableConstraint::Unique { name, .. } => name.as_ref(),
                            StoredTableConstraint::ForeignKey { name, .. } => name.as_ref(),
                            StoredTableConstraint::Check { name, .. } => name.as_ref(),
                        };
                        constraint_name != Some(name)
                    });
                    if schema.constraints.len() == original_len && !*if_exists {
                        return Err(SchemaError::ConstraintNotFound(name.clone()));
                    }
                }
            }
        }

        // Handle table rename if requested
        if let Some(new_name) = new_table_name {
            // Check if target table name already exists
            if Self::table_exists(tx, &new_name)? {
                return Err(SchemaError::TableExists(new_name));
            }

            // Delete old entry
            let old_key = Self::table_key(table_name);
            tx.delete_metadata(&old_key)?;
            Self::remove_from_list(tx, TABLES_LIST_KEY, table_name)?;

            // Update schema name
            schema.name.clone_from(&new_name);

            // Store with new name
            let new_key = Self::table_key(&new_name);
            let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
                .map_err(|e| SchemaError::Serialization(e.to_string()))?;
            tx.put_metadata(&new_key, &value)?;
            Self::add_to_list(tx, TABLES_LIST_KEY, &new_name)?;

            // Update indexes to point to new table name
            let indexes = Self::list_indexes_for_table(tx, table_name)?;
            for idx_name in indexes {
                if let Some(mut idx_schema) = Self::get_index(tx, &idx_name)? {
                    idx_schema.table.clone_from(&new_name);
                    let idx_key = Self::index_key(&idx_name);
                    let idx_value =
                        bincode::serde::encode_to_vec(&idx_schema, bincode::config::standard())
                            .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                    tx.put_metadata(&idx_key, &idx_value)?;
                }
            }
        } else {
            // Just update the existing schema in place
            let key = Self::table_key(table_name);
            let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
                .map_err(|e| SchemaError::Serialization(e.to_string()))?;
            tx.put_metadata(&key, &value)?;
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

    /// Create a view schema.
    pub fn create_view<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        node: &CreateViewNode,
    ) -> Result<(), SchemaError> {
        let view_name = &node.name;

        // Check if view already exists
        if Self::view_exists(tx, view_name)? {
            if node.or_replace {
                // OR REPLACE: drop and recreate
                Self::drop_view(tx, view_name, true)?;
            } else {
                return Err(SchemaError::ViewExists(view_name.clone()));
            }
        }

        // Create and store the schema
        let schema = ViewSchema::from_create_view(node);
        let key = Self::view_key(view_name);
        let value = bincode::serde::encode_to_vec(&schema, bincode::config::standard())
            .map_err(|e| SchemaError::Serialization(e.to_string()))?;

        tx.put_metadata(&key, &value)?;

        // Add to views list
        Self::add_to_list(tx, VIEWS_LIST_KEY, view_name)?;

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Drop a view schema.
    pub fn drop_view<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        view_name: &str,
        if_exists: bool,
    ) -> Result<(), SchemaError> {
        // Check if view exists
        if !Self::view_exists(tx, view_name)? {
            if if_exists {
                return Ok(());
            }
            return Err(SchemaError::ViewNotFound(view_name.to_string()));
        }

        // Remove the schema
        let key = Self::view_key(view_name);
        tx.delete_metadata(&key)?;

        // Remove from views list
        Self::remove_from_list(tx, VIEWS_LIST_KEY, view_name)?;

        // Increment schema version
        Self::increment_version(tx)?;

        Ok(())
    }

    /// Check if a view exists.
    pub fn view_exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<bool, SchemaError> {
        let key = Self::view_key(name);
        Ok(tx.get_metadata(&key)?.is_some())
    }

    /// Get a view schema.
    pub fn get_view<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &str,
    ) -> Result<Option<ViewSchema>, SchemaError> {
        let key = Self::view_key(name);
        match tx.get_metadata(&key)? {
            Some(bytes) => {
                let (schema, _): (ViewSchema, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| SchemaError::Serialization(e.to_string()))?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    /// List all views.
    pub fn list_views<T: Transaction>(
        tx: &DatabaseTransaction<T>,
    ) -> Result<Vec<String>, SchemaError> {
        Self::get_list(tx, VIEWS_LIST_KEY)
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

    fn view_key(name: &str) -> Vec<u8> {
        let mut key = VIEW_PREFIX.to_vec();
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

    #[error("Column already exists: {0}")]
    ColumnExists(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Constraint not found: {0}")]
    ConstraintNotFound(String),

    #[error("Index already exists: {0}")]
    IndexExists(String),

    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("View already exists: {0}")]
    ViewExists(String),

    #[error("View not found: {0}")]
    ViewNotFound(String),

    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    #[error("Serialization error: {0}")]
    Serialization(String),
}
