//! DDL (Data Definition Language) plan nodes.
//!
//! This module defines logical plan nodes for DDL operations:
//! - CREATE TABLE
//! - DROP TABLE
//! - CREATE INDEX
//! - DROP INDEX
//! - CREATE COLLECTION
//! - DROP COLLECTION
//! - CREATE VIEW
//! - DROP VIEW

use crate::ast::{
    ColumnDef, Expr, Identifier, IndexColumn, SelectStatement, TableConstraint, VectorDef,
};

/// A CREATE TABLE operation.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableNode {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The table name.
    pub name: String,
    /// Column definitions.
    pub columns: Vec<ColumnDef>,
    /// Table constraints.
    pub constraints: Vec<TableConstraint>,
}

impl CreateTableNode {
    /// Creates a new CREATE TABLE node.
    #[must_use]
    pub fn new(name: impl Into<String>, columns: Vec<ColumnDef>) -> Self {
        Self { if_not_exists: false, name: name.into(), columns, constraints: vec![] }
    }

    /// Sets the IF NOT EXISTS flag.
    #[must_use]
    pub const fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    /// Adds table constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: Vec<TableConstraint>) -> Self {
        self.constraints = constraints;
        self
    }
}

/// A DROP TABLE operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTableNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The table names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

impl DropTableNode {
    /// Creates a new DROP TABLE node.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { if_exists: false, names, cascade: false }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }

    /// Sets the CASCADE flag.
    #[must_use]
    pub const fn with_cascade(mut self, cascade: bool) -> Self {
        self.cascade = cascade;
        self
    }
}

/// A CREATE INDEX operation.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexNode {
    /// Whether UNIQUE is specified.
    pub unique: bool,
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The index name.
    pub name: String,
    /// The table to index.
    pub table: String,
    /// The columns/expressions to index.
    pub columns: Vec<IndexColumn>,
    /// The index method (btree, hash, gin, hnsw, ivfflat).
    pub using: Option<String>,
    /// Index-specific options.
    pub with: Vec<(String, String)>,
    /// Optional WHERE clause for partial indexes.
    pub where_clause: Option<Expr>,
}

impl CreateIndexNode {
    /// Creates a new CREATE INDEX node.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        table: impl Into<String>,
        columns: Vec<IndexColumn>,
    ) -> Self {
        Self {
            unique: false,
            if_not_exists: false,
            name: name.into(),
            table: table.into(),
            columns,
            using: None,
            with: vec![],
            where_clause: None,
        }
    }

    /// Sets the UNIQUE flag.
    #[must_use]
    pub const fn with_unique(mut self, unique: bool) -> Self {
        self.unique = unique;
        self
    }

    /// Sets the IF NOT EXISTS flag.
    #[must_use]
    pub const fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    /// Sets the index method.
    #[must_use]
    pub fn with_using(mut self, using: Option<String>) -> Self {
        self.using = using;
        self
    }

    /// Sets index options.
    #[must_use]
    pub fn with_options(mut self, options: Vec<(String, String)>) -> Self {
        self.with = options;
        self
    }

    /// Sets the WHERE clause for partial indexes.
    #[must_use]
    pub fn with_where_clause(mut self, where_clause: Option<Expr>) -> Self {
        self.where_clause = where_clause;
        self
    }
}

/// A DROP INDEX operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropIndexNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The index names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

impl DropIndexNode {
    /// Creates a new DROP INDEX node.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { if_exists: false, names, cascade: false }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }

    /// Sets the CASCADE flag.
    #[must_use]
    pub const fn with_cascade(mut self, cascade: bool) -> Self {
        self.cascade = cascade;
        self
    }
}

/// A CREATE COLLECTION operation for vector collections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateCollectionNode {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The collection name.
    pub name: String,
    /// Named vector definitions.
    pub vectors: Vec<VectorDef>,
}

impl CreateCollectionNode {
    /// Creates a new CREATE COLLECTION node.
    #[must_use]
    pub fn new(name: impl Into<String>, vectors: Vec<VectorDef>) -> Self {
        Self { if_not_exists: false, name: name.into(), vectors }
    }

    /// Sets the IF NOT EXISTS flag.
    #[must_use]
    pub const fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }
}

/// A DROP COLLECTION operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropCollectionNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The collection names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified (drops associated data and indexes).
    pub cascade: bool,
}

impl DropCollectionNode {
    /// Creates a new DROP COLLECTION node.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { if_exists: false, names, cascade: false }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }

    /// Sets the CASCADE flag.
    #[must_use]
    pub const fn with_cascade(mut self, cascade: bool) -> Self {
        self.cascade = cascade;
        self
    }
}

/// A CREATE VIEW operation.
///
/// Views are stored query definitions that can be used like tables.
/// They expand to their defining query when referenced.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateViewNode {
    /// Whether OR REPLACE is specified.
    pub or_replace: bool,
    /// The view name.
    pub name: String,
    /// Optional column aliases for the view.
    pub columns: Vec<Identifier>,
    /// The query defining the view.
    pub query: Box<SelectStatement>,
}

impl CreateViewNode {
    /// Creates a new CREATE VIEW node.
    #[must_use]
    pub fn new(name: impl Into<String>, query: SelectStatement) -> Self {
        Self { or_replace: false, name: name.into(), columns: vec![], query: Box::new(query) }
    }

    /// Sets the OR REPLACE flag.
    #[must_use]
    pub const fn with_or_replace(mut self, or_replace: bool) -> Self {
        self.or_replace = or_replace;
        self
    }

    /// Sets the column aliases.
    #[must_use]
    pub fn with_columns(mut self, columns: Vec<Identifier>) -> Self {
        self.columns = columns;
        self
    }
}

/// A DROP VIEW operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropViewNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The view names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified (drops dependent objects).
    pub cascade: bool,
}

impl DropViewNode {
    /// Creates a new DROP VIEW node.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { if_exists: false, names, cascade: false }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }

    /// Sets the CASCADE flag.
    #[must_use]
    pub const fn with_cascade(mut self, cascade: bool) -> Self {
        self.cascade = cascade;
        self
    }
}
