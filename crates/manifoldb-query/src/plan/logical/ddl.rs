//! DDL (Data Definition Language) plan nodes.
//!
//! This module defines logical plan nodes for DDL operations:
//! - CREATE TABLE
//! - ALTER TABLE
//! - DROP TABLE
//! - CREATE INDEX
//! - DROP INDEX
//! - CREATE COLLECTION
//! - DROP COLLECTION
//! - CREATE VIEW
//! - DROP VIEW
//! - CREATE SCHEMA
//! - ALTER SCHEMA
//! - DROP SCHEMA
//! - CREATE FUNCTION
//! - DROP FUNCTION
//! - CREATE TRIGGER
//! - DROP TRIGGER

use crate::ast::{
    AlterSchemaAction, AlterTableAction, ColumnDef, DataType, Expr, FunctionLanguage,
    FunctionParameter, FunctionVolatility, Identifier, IndexColumn, SelectStatement,
    TableConstraint, TriggerEvent, TriggerForEach, TriggerTiming, VectorDef,
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

/// An ALTER TABLE operation.
///
/// Supports various schema modifications including:
/// - ADD COLUMN
/// - DROP COLUMN
/// - ALTER COLUMN (SET NOT NULL, DROP NOT NULL, SET DEFAULT, DROP DEFAULT, SET DATA TYPE)
/// - RENAME COLUMN
/// - RENAME TABLE
/// - ADD CONSTRAINT
/// - DROP CONSTRAINT
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The table name.
    pub name: String,
    /// The alter actions to perform.
    pub actions: Vec<AlterTableAction>,
}

impl AlterTableNode {
    /// Creates a new ALTER TABLE node.
    #[must_use]
    pub fn new(name: impl Into<String>, actions: Vec<AlterTableAction>) -> Self {
        Self { if_exists: false, name: name.into(), actions }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }

    /// Adds an action.
    #[must_use]
    pub fn add_action(mut self, action: AlterTableAction) -> Self {
        self.actions.push(action);
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

/// An ALTER INDEX operation.
///
/// Supports modifications to existing indexes:
/// - RENAME TO: Rename an index
/// - SET: Set index options
/// - RESET: Reset index options to defaults
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterIndexNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The index name.
    pub name: String,
    /// The action to perform.
    pub action: AlterIndexAction,
}

impl AlterIndexNode {
    /// Creates a new ALTER INDEX node.
    #[must_use]
    pub fn new(name: impl Into<String>, action: AlterIndexAction) -> Self {
        Self { if_exists: false, name: name.into(), action }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn with_if_exists(mut self, if_exists: bool) -> Self {
        self.if_exists = if_exists;
        self
    }
}

/// An action to perform in an ALTER INDEX statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlterIndexAction {
    /// RENAME TO: Rename the index.
    RenameIndex {
        /// The new index name.
        new_name: String,
    },
    /// SET: Set index options.
    SetOptions {
        /// The options to set (key-value pairs).
        options: Vec<(String, String)>,
    },
    /// RESET: Reset index options to defaults.
    ResetOptions {
        /// The option names to reset.
        options: Vec<String>,
    },
}

/// A TRUNCATE TABLE operation.
///
/// Quickly removes all rows from one or more tables without logging
/// individual row deletions. More efficient than DELETE for removing all rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncateTableNode {
    /// The table names to truncate.
    pub names: Vec<String>,
    /// Whether to restart identity columns.
    pub restart_identity: bool,
    /// Whether to cascade to dependent tables.
    pub cascade: bool,
}

impl TruncateTableNode {
    /// Creates a new TRUNCATE TABLE node.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { names, restart_identity: false, cascade: false }
    }

    /// Sets the restart identity flag.
    #[must_use]
    pub const fn with_restart_identity(mut self, restart_identity: bool) -> Self {
        self.restart_identity = restart_identity;
        self
    }

    /// Sets the cascade flag.
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

// ============================================================================
// Materialized View DDL Nodes
// ============================================================================

/// A CREATE MATERIALIZED VIEW operation.
///
/// Materialized views store query results persistently and must be
/// explicitly refreshed to update their data.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateMaterializedViewNode {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The materialized view name.
    pub name: String,
    /// Optional column aliases for the view.
    pub columns: Vec<Identifier>,
    /// The query defining the materialized view.
    pub query: Box<SelectStatement>,
}

impl CreateMaterializedViewNode {
    /// Creates a new CREATE MATERIALIZED VIEW node.
    #[must_use]
    pub fn new(name: impl Into<String>, query: SelectStatement) -> Self {
        Self { if_not_exists: false, name: name.into(), columns: vec![], query: Box::new(query) }
    }

    /// Sets the IF NOT EXISTS flag.
    #[must_use]
    pub const fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    /// Sets the column aliases.
    #[must_use]
    pub fn with_columns(mut self, columns: Vec<Identifier>) -> Self {
        self.columns = columns;
        self
    }
}

/// A DROP MATERIALIZED VIEW operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropMaterializedViewNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The materialized view names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified (drops dependent objects).
    pub cascade: bool,
}

impl DropMaterializedViewNode {
    /// Creates a new DROP MATERIALIZED VIEW node.
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

/// A REFRESH MATERIALIZED VIEW operation.
///
/// Re-executes the materialized view's defining query and updates
/// the stored data with the new results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefreshMaterializedViewNode {
    /// The materialized view name to refresh.
    pub name: String,
    /// Whether CONCURRENTLY is specified (allows concurrent reads during refresh).
    pub concurrently: bool,
}

impl RefreshMaterializedViewNode {
    /// Creates a new REFRESH MATERIALIZED VIEW node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), concurrently: false }
    }

    /// Sets the CONCURRENTLY flag.
    #[must_use]
    pub const fn with_concurrently(mut self, concurrently: bool) -> Self {
        self.concurrently = concurrently;
        self
    }
}

// ============================================================================
// Schema DDL Nodes
// ============================================================================

/// A CREATE SCHEMA operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateSchemaNode {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The schema name.
    pub name: String,
    /// Optional authorization (owner) for the schema.
    pub authorization: Option<String>,
}

impl CreateSchemaNode {
    /// Creates a new CREATE SCHEMA node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { if_not_exists: false, name: name.into(), authorization: None }
    }

    /// Sets the IF NOT EXISTS flag.
    #[must_use]
    pub const fn with_if_not_exists(mut self, if_not_exists: bool) -> Self {
        self.if_not_exists = if_not_exists;
        self
    }

    /// Sets the authorization (owner) for the schema.
    #[must_use]
    pub fn with_authorization(mut self, auth: impl Into<String>) -> Self {
        self.authorization = Some(auth.into());
        self
    }
}

/// An ALTER SCHEMA operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlterSchemaNode {
    /// The schema name to alter.
    pub name: String,
    /// The action to perform on the schema.
    pub action: AlterSchemaAction,
}

impl AlterSchemaNode {
    /// Creates a new ALTER SCHEMA node.
    #[must_use]
    pub fn new(name: impl Into<String>, action: AlterSchemaAction) -> Self {
        Self { name: name.into(), action }
    }
}

/// A DROP SCHEMA operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropSchemaNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The schema names to drop.
    pub names: Vec<String>,
    /// Whether CASCADE is specified (drops all contained objects).
    pub cascade: bool,
}

impl DropSchemaNode {
    /// Creates a new DROP SCHEMA node.
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

// ============================================================================
// Function DDL Nodes
// ============================================================================

/// A CREATE FUNCTION operation.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct CreateFunctionNode {
    /// Whether OR REPLACE is specified.
    pub or_replace: bool,
    /// The function name (may be schema-qualified).
    pub name: String,
    /// Function parameters.
    pub parameters: Vec<FunctionParameter>,
    /// Return type.
    pub returns: DataType,
    /// Whether the function returns a set of values (SETOF).
    pub returns_setof: bool,
    /// The function body.
    pub body: String,
    /// The language of the function body.
    pub language: FunctionLanguage,
    /// Function volatility (IMMUTABLE, STABLE, VOLATILE).
    pub volatility: Option<FunctionVolatility>,
    /// Whether the function is STRICT (returns NULL on NULL input).
    pub strict: bool,
    /// Security definer mode (SECURITY DEFINER vs SECURITY INVOKER).
    pub security_definer: bool,
}

impl CreateFunctionNode {
    /// Creates a new CREATE FUNCTION node.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        parameters: Vec<FunctionParameter>,
        returns: DataType,
        body: impl Into<String>,
        language: FunctionLanguage,
    ) -> Self {
        Self {
            or_replace: false,
            name: name.into(),
            parameters,
            returns,
            returns_setof: false,
            body: body.into(),
            language,
            volatility: None,
            strict: false,
            security_definer: false,
        }
    }

    /// Sets the OR REPLACE flag.
    #[must_use]
    pub const fn with_or_replace(mut self, or_replace: bool) -> Self {
        self.or_replace = or_replace;
        self
    }

    /// Sets whether the function returns a set.
    #[must_use]
    pub const fn with_returns_setof(mut self, returns_setof: bool) -> Self {
        self.returns_setof = returns_setof;
        self
    }

    /// Sets the function volatility.
    #[must_use]
    pub fn with_volatility(mut self, volatility: FunctionVolatility) -> Self {
        self.volatility = Some(volatility);
        self
    }

    /// Sets the function as STRICT.
    #[must_use]
    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Sets SECURITY DEFINER mode.
    #[must_use]
    pub const fn with_security_definer(mut self, security_definer: bool) -> Self {
        self.security_definer = security_definer;
        self
    }
}

/// A DROP FUNCTION operation.
#[derive(Debug, Clone, PartialEq)]
pub struct DropFunctionNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The function name (may be schema-qualified).
    pub name: String,
    /// Optional argument types (for overload resolution).
    pub arg_types: Vec<DataType>,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

impl DropFunctionNode {
    /// Creates a new DROP FUNCTION node.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { if_exists: false, name: name.into(), arg_types: vec![], cascade: false }
    }

    /// Creates a DROP FUNCTION node with argument types.
    #[must_use]
    pub fn with_args(name: impl Into<String>, arg_types: Vec<DataType>) -> Self {
        Self { if_exists: false, name: name.into(), arg_types, cascade: false }
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

// ============================================================================
// Trigger DDL Nodes
// ============================================================================

/// A CREATE TRIGGER operation.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTriggerNode {
    /// Whether OR REPLACE is specified.
    pub or_replace: bool,
    /// The trigger name.
    pub name: String,
    /// When the trigger fires (BEFORE, AFTER, INSTEAD OF).
    pub timing: TriggerTiming,
    /// Events that fire the trigger (INSERT, UPDATE, DELETE, TRUNCATE).
    pub events: Vec<TriggerEvent>,
    /// The table the trigger is on.
    pub table: String,
    /// Whether this is a row-level or statement-level trigger.
    pub for_each: TriggerForEach,
    /// Optional WHEN condition.
    pub when_clause: Option<Expr>,
    /// The function to execute.
    pub function: String,
    /// Arguments to pass to the function.
    pub function_args: Vec<String>,
}

impl CreateTriggerNode {
    /// Creates a new CREATE TRIGGER node.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        timing: TriggerTiming,
        events: Vec<TriggerEvent>,
        table: impl Into<String>,
        function: impl Into<String>,
    ) -> Self {
        Self {
            or_replace: false,
            name: name.into(),
            timing,
            events,
            table: table.into(),
            for_each: TriggerForEach::Row,
            when_clause: None,
            function: function.into(),
            function_args: vec![],
        }
    }

    /// Sets the OR REPLACE flag.
    #[must_use]
    pub const fn with_or_replace(mut self, or_replace: bool) -> Self {
        self.or_replace = or_replace;
        self
    }

    /// Sets the FOR EACH clause.
    #[must_use]
    pub const fn with_for_each(mut self, for_each: TriggerForEach) -> Self {
        self.for_each = for_each;
        self
    }

    /// Sets the WHEN clause.
    #[must_use]
    pub fn with_when(mut self, condition: Expr) -> Self {
        self.when_clause = Some(condition);
        self
    }

    /// Sets the function arguments.
    #[must_use]
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.function_args = args;
        self
    }
}

/// A DROP TRIGGER operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTriggerNode {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The trigger name.
    pub name: String,
    /// The table the trigger is on.
    pub table: String,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

impl DropTriggerNode {
    /// Creates a new DROP TRIGGER node.
    #[must_use]
    pub fn new(name: impl Into<String>, table: impl Into<String>) -> Self {
        Self { if_exists: false, name: name.into(), table: table.into(), cascade: false }
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
