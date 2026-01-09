//! Statement AST types.
//!
//! This module defines the top-level statement types for parsed queries.

use super::expr::{Expr, Identifier, OrderByExpr, QualifiedName};
use super::pattern::GraphPattern;

/// A parsed SQL statement.
///
/// Large statement types are boxed to reduce enum size overhead.
/// This improves memory efficiency when many Statement instances
/// are created (e.g., in query planning).
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// SELECT statement (boxed - 744 bytes unboxed).
    Select(Box<SelectStatement>),
    /// INSERT statement (boxed - 304 bytes unboxed).
    Insert(Box<InsertStatement>),
    /// UPDATE statement (boxed - 328 bytes unboxed).
    Update(Box<UpdateStatement>),
    /// DELETE statement (boxed - 304 bytes unboxed).
    Delete(Box<DeleteStatement>),
    /// CREATE TABLE statement.
    CreateTable(CreateTableStatement),
    /// ALTER TABLE statement.
    AlterTable(AlterTableStatement),
    /// CREATE INDEX statement (boxed - 288 bytes unboxed).
    CreateIndex(Box<CreateIndexStatement>),
    /// CREATE COLLECTION statement for vector collections.
    CreateCollection(Box<CreateCollectionStatement>),
    /// DROP TABLE statement.
    DropTable(DropTableStatement),
    /// DROP INDEX statement.
    DropIndex(DropIndexStatement),
    /// DROP COLLECTION statement.
    DropCollection(DropCollectionStatement),
    /// CREATE VIEW statement.
    CreateView(Box<CreateViewStatement>),
    /// DROP VIEW statement.
    DropView(DropViewStatement),
    /// MATCH statement (Cypher-style graph query).
    Match(Box<MatchStatement>),
    /// Cypher CREATE statement for creating nodes and relationships.
    Create(Box<CreateGraphStatement>),
    /// Cypher MERGE statement for upserting nodes and relationships.
    Merge(Box<MergeGraphStatement>),
    /// CALL statement for procedure invocation.
    Call(Box<CallStatement>),
    /// Cypher SET statement for updating properties.
    Set(Box<SetGraphStatement>),
    /// Cypher DELETE statement for removing nodes/relationships.
    DeleteGraph(Box<DeleteGraphStatement>),
    /// Cypher REMOVE statement for removing properties/labels.
    Remove(Box<RemoveGraphStatement>),
    /// Cypher FOREACH statement for iterating over lists.
    Foreach(Box<ForeachStatement>),
    /// EXPLAIN statement (basic - just shows plan).
    Explain(Box<Statement>),
    /// EXPLAIN ANALYZE statement (with execution statistics).
    ExplainAnalyze(Box<ExplainAnalyzeStatement>),
    /// Transaction control statements.
    Transaction(TransactionStatement),
    /// Utility statements (VACUUM, ANALYZE, COPY, SET/SHOW/RESET).
    Utility(Box<UtilityStatement>),
}

// ============================================================================
// Transaction Statements
// ============================================================================

/// A transaction control statement.
///
/// Supports SQL transaction control statements for managing ACID transactions:
///
/// - `BEGIN` / `START TRANSACTION` - Start a new transaction
/// - `COMMIT` - Commit the current transaction
/// - `ROLLBACK` - Roll back the current transaction
/// - `SAVEPOINT` - Create a named savepoint
/// - `RELEASE SAVEPOINT` - Remove a savepoint
/// - `ROLLBACK TO SAVEPOINT` - Roll back to a named savepoint
///
/// # Examples
///
/// Basic transaction:
/// ```sql
/// BEGIN;
/// INSERT INTO users VALUES (1, 'Alice');
/// COMMIT;
/// ```
///
/// Using savepoints:
/// ```sql
/// BEGIN;
/// INSERT INTO users VALUES (1, 'Alice');
/// SAVEPOINT sp1;
/// INSERT INTO users VALUES (2, 'Bob');
/// ROLLBACK TO SAVEPOINT sp1;  -- Undoes Bob insert
/// COMMIT;  -- Only Alice committed
/// ```
///
/// Transaction with options:
/// ```sql
/// BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
/// SET TRANSACTION READ ONLY;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionStatement {
    /// BEGIN or START TRANSACTION.
    Begin(BeginTransaction),
    /// COMMIT.
    Commit,
    /// ROLLBACK.
    Rollback(RollbackTransaction),
    /// SAVEPOINT <name>.
    Savepoint(SavepointStatement),
    /// RELEASE SAVEPOINT <name>.
    ReleaseSavepoint(ReleaseSavepointStatement),
    /// SET TRANSACTION.
    SetTransaction(SetTransactionStatement),
}

/// A BEGIN or START TRANSACTION statement.
///
/// Starts a new transaction with optional configuration.
///
/// # Examples
///
/// ```sql
/// BEGIN;
/// START TRANSACTION;
/// BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
/// BEGIN TRANSACTION READ ONLY;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BeginTransaction {
    /// Whether the TRANSACTION keyword was used.
    pub has_transaction_keyword: bool,
    /// Transaction isolation level.
    pub isolation_level: Option<IsolationLevel>,
    /// Transaction access mode.
    pub access_mode: Option<TransactionAccessMode>,
    /// Whether this is a deferred transaction (for database-level locking).
    pub deferred: bool,
}

impl BeginTransaction {
    /// Creates a new BEGIN statement with default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the isolation level.
    #[must_use]
    pub fn with_isolation_level(mut self, level: IsolationLevel) -> Self {
        self.isolation_level = Some(level);
        self
    }

    /// Sets the access mode to READ ONLY.
    #[must_use]
    pub fn read_only(mut self) -> Self {
        self.access_mode = Some(TransactionAccessMode::ReadOnly);
        self
    }

    /// Sets the access mode to READ WRITE.
    #[must_use]
    pub fn read_write(mut self) -> Self {
        self.access_mode = Some(TransactionAccessMode::ReadWrite);
        self
    }

    /// Sets the deferred flag.
    #[must_use]
    pub fn deferred(mut self) -> Self {
        self.deferred = true;
        self
    }
}

/// A ROLLBACK statement.
///
/// # Examples
///
/// ```sql
/// ROLLBACK;
/// ROLLBACK TO SAVEPOINT sp1;
/// ROLLBACK TRANSACTION TO SAVEPOINT sp1;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RollbackTransaction {
    /// Whether the TRANSACTION keyword was used.
    pub has_transaction_keyword: bool,
    /// Optional savepoint name to roll back to.
    pub to_savepoint: Option<Identifier>,
}

impl RollbackTransaction {
    /// Creates a ROLLBACK statement that rolls back the entire transaction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a ROLLBACK TO SAVEPOINT statement.
    #[must_use]
    pub fn to_savepoint(name: impl Into<Identifier>) -> Self {
        Self { has_transaction_keyword: false, to_savepoint: Some(name.into()) }
    }
}

/// A SAVEPOINT statement.
///
/// Creates a savepoint within the current transaction that can be
/// rolled back to later without affecting previous work.
///
/// # Examples
///
/// ```sql
/// SAVEPOINT my_savepoint;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavepointStatement {
    /// The savepoint name.
    pub name: Identifier,
}

impl SavepointStatement {
    /// Creates a new SAVEPOINT statement.
    #[must_use]
    pub fn new(name: impl Into<Identifier>) -> Self {
        Self { name: name.into() }
    }
}

/// A RELEASE SAVEPOINT statement.
///
/// Removes a savepoint without affecting the transaction state.
/// After releasing, the savepoint can no longer be rolled back to.
///
/// # Examples
///
/// ```sql
/// RELEASE SAVEPOINT my_savepoint;
/// RELEASE my_savepoint;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseSavepointStatement {
    /// The savepoint name to release.
    pub name: Identifier,
}

impl ReleaseSavepointStatement {
    /// Creates a new RELEASE SAVEPOINT statement.
    #[must_use]
    pub fn new(name: impl Into<Identifier>) -> Self {
        Self { name: name.into() }
    }
}

/// A SET TRANSACTION statement.
///
/// Changes the characteristics of the current transaction.
///
/// # Examples
///
/// ```sql
/// SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
/// SET TRANSACTION READ ONLY;
/// SET TRANSACTION READ WRITE, ISOLATION LEVEL REPEATABLE READ;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SetTransactionStatement {
    /// Transaction isolation level.
    pub isolation_level: Option<IsolationLevel>,
    /// Transaction access mode.
    pub access_mode: Option<TransactionAccessMode>,
}

impl SetTransactionStatement {
    /// Creates a new SET TRANSACTION statement.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the isolation level.
    #[must_use]
    pub fn with_isolation_level(mut self, level: IsolationLevel) -> Self {
        self.isolation_level = Some(level);
        self
    }

    /// Sets the access mode.
    #[must_use]
    pub fn with_access_mode(mut self, mode: TransactionAccessMode) -> Self {
        self.access_mode = Some(mode);
        self
    }
}

/// Transaction isolation levels.
///
/// Controls the visibility of changes made by other concurrent transactions.
///
/// # Isolation Levels
///
/// - **READ UNCOMMITTED**: Allows dirty reads (reading uncommitted data).
/// - **READ COMMITTED**: Only sees committed data (PostgreSQL default).
/// - **REPEATABLE READ**: Snapshot isolation - same query returns same results.
/// - **SERIALIZABLE**: Strongest isolation - transactions execute as if serial.
///
/// Note: redb provides serializable isolation by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// READ UNCOMMITTED - allows dirty reads.
    ReadUncommitted,
    /// READ COMMITTED - default for many databases.
    ReadCommitted,
    /// REPEATABLE READ - snapshot isolation.
    RepeatableRead,
    /// SERIALIZABLE - full serializable isolation (redb default).
    #[default]
    Serializable,
}

impl std::fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadUncommitted => write!(f, "READ UNCOMMITTED"),
            Self::ReadCommitted => write!(f, "READ COMMITTED"),
            Self::RepeatableRead => write!(f, "REPEATABLE READ"),
            Self::Serializable => write!(f, "SERIALIZABLE"),
        }
    }
}

/// Transaction access mode.
///
/// Controls whether the transaction can modify data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionAccessMode {
    /// READ ONLY - transaction cannot modify data.
    ReadOnly,
    /// READ WRITE - transaction can read and modify data (default).
    ReadWrite,
}

impl std::fmt::Display for TransactionAccessMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadOnly => write!(f, "READ ONLY"),
            Self::ReadWrite => write!(f, "READ WRITE"),
        }
    }
}

/// A Common Table Expression (CTE) defined in a WITH clause.
///
/// CTEs allow defining named subqueries that can be referenced multiple times
/// in the main query, similar to temporary views.
///
/// # Examples
///
/// Non-recursive CTE:
/// ```sql
/// WITH active_users AS (
///     SELECT * FROM users WHERE status = 'active'
/// )
/// SELECT * FROM active_users WHERE age > 21;
/// ```
///
/// Recursive CTE:
/// ```sql
/// WITH RECURSIVE hierarchy AS (
///     -- Base case
///     SELECT id, name, manager_id, 1 as level
///     FROM employees WHERE manager_id IS NULL
///     UNION ALL
///     -- Recursive case
///     SELECT e.id, e.name, e.manager_id, h.level + 1
///     FROM employees e
///     JOIN hierarchy h ON e.manager_id = h.id
/// )
/// SELECT * FROM hierarchy;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    /// The name of the CTE.
    pub name: Identifier,
    /// Optional column aliases for the CTE.
    pub columns: Vec<Identifier>,
    /// The subquery that defines the CTE.
    pub query: Box<SelectStatement>,
    /// Whether this is a recursive CTE (WITH RECURSIVE).
    pub recursive: bool,
}

impl WithClause {
    /// Creates a new non-recursive CTE with the given name and query.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, query: SelectStatement) -> Self {
        Self { name: name.into(), columns: vec![], query: Box::new(query), recursive: false }
    }

    /// Creates a new CTE with column aliases.
    #[must_use]
    pub fn with_columns(
        name: impl Into<Identifier>,
        columns: Vec<Identifier>,
        query: SelectStatement,
    ) -> Self {
        Self { name: name.into(), columns, query: Box::new(query), recursive: false }
    }

    /// Creates a new recursive CTE with the given name and query.
    ///
    /// The query should contain a UNION or UNION ALL combining the base case
    /// and the recursive case that references this CTE by name.
    #[must_use]
    pub fn recursive(name: impl Into<Identifier>, query: SelectStatement) -> Self {
        Self { name: name.into(), columns: vec![], query: Box::new(query), recursive: true }
    }

    /// Creates a new recursive CTE with column aliases.
    #[must_use]
    pub fn recursive_with_columns(
        name: impl Into<Identifier>,
        columns: Vec<Identifier>,
        query: SelectStatement,
    ) -> Self {
        Self { name: name.into(), columns, query: Box::new(query), recursive: true }
    }
}

/// A SELECT statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    /// Common Table Expressions (WITH clause).
    pub with_clauses: Vec<WithClause>,
    /// Whether DISTINCT is specified.
    pub distinct: bool,
    /// The projection (SELECT list).
    pub projection: Vec<SelectItem>,
    /// The FROM clause.
    pub from: Vec<TableRef>,
    /// Optional MATCH clause for graph patterns.
    pub match_clause: Option<GraphPattern>,
    /// OPTIONAL MATCH clauses for left outer join graph patterns.
    /// Each pattern is joined with the main query using a LEFT OUTER JOIN,
    /// returning NULL for unmatched optional patterns.
    pub optional_match_clauses: Vec<GraphPattern>,
    /// Optional WHERE clause.
    pub where_clause: Option<Expr>,
    /// Optional GROUP BY clause.
    pub group_by: Vec<Expr>,
    /// Optional HAVING clause.
    pub having: Option<Expr>,
    /// Optional ORDER BY clause.
    pub order_by: Vec<OrderByExpr>,
    /// Optional LIMIT clause.
    pub limit: Option<Expr>,
    /// Optional OFFSET clause.
    pub offset: Option<Expr>,
    /// Set operations (UNION, INTERSECT, EXCEPT).
    pub set_op: Option<Box<SetOperation>>,
}

impl SelectStatement {
    /// Creates a new SELECT statement with the given projection.
    #[must_use]
    pub const fn new(projection: Vec<SelectItem>) -> Self {
        Self {
            with_clauses: vec![],
            distinct: false,
            projection,
            from: vec![],
            match_clause: None,
            optional_match_clauses: vec![],
            where_clause: None,
            group_by: vec![],
            having: None,
            order_by: vec![],
            limit: None,
            offset: None,
            set_op: None,
        }
    }

    /// Adds a WITH clause (CTE) to the statement.
    #[must_use]
    pub fn with_cte(mut self, cte: WithClause) -> Self {
        self.with_clauses.push(cte);
        self
    }

    /// Adds a FROM clause.
    #[must_use]
    pub fn from(mut self, table: TableRef) -> Self {
        self.from.push(table);
        self
    }

    /// Sets the WHERE clause.
    #[must_use]
    pub fn where_clause(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Sets the MATCH clause.
    #[must_use]
    pub fn match_clause(mut self, pattern: GraphPattern) -> Self {
        self.match_clause = Some(pattern);
        self
    }

    /// Adds an OPTIONAL MATCH clause.
    ///
    /// OPTIONAL MATCH clauses are joined using LEFT OUTER JOIN semantics,
    /// returning NULL for variables in the optional pattern when no match exists.
    #[must_use]
    pub fn optional_match_clause(mut self, pattern: GraphPattern) -> Self {
        self.optional_match_clauses.push(pattern);
        self
    }

    /// Adds ORDER BY.
    #[must_use]
    pub fn order_by(mut self, orders: Vec<OrderByExpr>) -> Self {
        self.order_by = orders;
        self
    }

    /// Sets the LIMIT.
    #[must_use]
    pub fn limit(mut self, limit: Expr) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the OFFSET.
    #[must_use]
    pub fn offset(mut self, offset: Expr) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// A standalone MATCH statement (Cypher-style graph query).
///
/// This provides pure Cypher-like syntax for graph pattern matching:
///
/// ```text
/// MATCH (a:User)-[:FOLLOWS]->(b:User)
/// WHERE a.name = 'Alice'
/// RETURN b.name, b.email
/// ORDER BY b.name
/// LIMIT 10;
/// ```
///
/// Internally, this is converted to a SELECT statement during planning.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchStatement {
    /// The graph pattern to match.
    pub pattern: GraphPattern,
    /// Optional WHERE clause.
    pub where_clause: Option<Expr>,
    /// The RETURN clause (required in Cypher).
    pub return_clause: Vec<ReturnItem>,
    /// Whether RETURN DISTINCT is specified.
    pub distinct: bool,
    /// Optional ORDER BY clause.
    pub order_by: Vec<OrderByExpr>,
    /// Optional SKIP clause (equivalent to OFFSET).
    pub skip: Option<Expr>,
    /// Optional LIMIT clause.
    pub limit: Option<Expr>,
}

impl MatchStatement {
    /// Creates a new MATCH statement with a pattern and return items.
    #[must_use]
    pub const fn new(pattern: GraphPattern, return_clause: Vec<ReturnItem>) -> Self {
        Self {
            pattern,
            where_clause: None,
            return_clause,
            distinct: false,
            order_by: vec![],
            skip: None,
            limit: None,
        }
    }

    /// Sets the WHERE clause.
    #[must_use]
    pub fn where_clause(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Sets DISTINCT on the RETURN clause.
    #[must_use]
    pub const fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// Sets the ORDER BY clause.
    #[must_use]
    pub fn order_by(mut self, orders: Vec<OrderByExpr>) -> Self {
        self.order_by = orders;
        self
    }

    /// Sets the SKIP clause.
    #[must_use]
    pub fn skip(mut self, skip: Expr) -> Self {
        self.skip = Some(skip);
        self
    }

    /// Sets the LIMIT clause.
    #[must_use]
    pub fn limit(mut self, limit: Expr) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Converts this MATCH statement to an equivalent SELECT statement.
    ///
    /// This is used during planning to leverage the existing SELECT infrastructure.
    #[must_use]
    pub fn to_select(&self) -> SelectStatement {
        // Convert return items to select items
        let projection: Vec<SelectItem> = self
            .return_clause
            .iter()
            .map(|item| match item {
                ReturnItem::Wildcard => SelectItem::Wildcard,
                ReturnItem::Expr { expr, alias } => {
                    SelectItem::Expr { expr: expr.clone(), alias: alias.clone() }
                }
            })
            .collect();

        SelectStatement {
            with_clauses: vec![], // MATCH statements don't have CTEs
            distinct: self.distinct,
            projection,
            from: vec![], // Graph patterns don't need a FROM clause
            match_clause: Some(self.pattern.clone()),
            optional_match_clauses: vec![], // TODO: Add support for OPTIONAL MATCH in standalone Cypher
            where_clause: self.where_clause.clone(),
            group_by: vec![],
            having: None,
            order_by: self.order_by.clone(),
            limit: self.limit.clone(),
            offset: self.skip.clone(), // Cypher SKIP = SQL OFFSET
            set_op: None,
        }
    }
}

/// An item in a RETURN clause.
#[derive(Debug, Clone, PartialEq)]
pub enum ReturnItem {
    /// A wildcard (*) - return all bound variables.
    Wildcard,
    /// An expression, optionally aliased.
    Expr {
        /// The expression to return.
        expr: Expr,
        /// Optional alias (AS name).
        alias: Option<Identifier>,
    },
}

impl ReturnItem {
    /// Creates a wildcard return item.
    #[must_use]
    pub const fn wildcard() -> Self {
        Self::Wildcard
    }

    /// Creates an unaliased expression return item.
    #[must_use]
    pub const fn expr(expr: Expr) -> Self {
        Self::Expr { expr, alias: None }
    }

    /// Creates an aliased expression return item.
    #[must_use]
    pub fn aliased(expr: Expr, alias: impl Into<Identifier>) -> Self {
        Self::Expr { expr, alias: Some(alias.into()) }
    }
}

impl From<Expr> for ReturnItem {
    fn from(expr: Expr) -> Self {
        Self::expr(expr)
    }
}

/// A CALL statement for invoking procedures.
///
/// Supports both SQL and Cypher-style procedure invocation:
///
/// ```text
/// -- SQL style
/// CALL algo.pageRank('nodes', 'edges', {damping: 0.85}) YIELD node, score
/// WHERE score > 0.1
///
/// -- Cypher style
/// CALL algo.shortestPath(source, target) YIELD path, cost
/// ```
///
/// Procedures can be built-in graph algorithms or user-defined functions.
#[derive(Debug, Clone, PartialEq)]
pub struct CallStatement {
    /// The fully-qualified procedure name (e.g., "algo.pageRank").
    pub procedure_name: QualifiedName,
    /// Arguments passed to the procedure.
    pub arguments: Vec<Expr>,
    /// Items to yield from the procedure result.
    /// Empty vec means no YIELD clause (standalone CALL).
    /// Contains `YieldItem::Wildcard` for `YIELD *`.
    pub yield_items: Vec<YieldItem>,
    /// Optional WHERE clause to filter yielded results.
    pub where_clause: Option<Expr>,
}

impl CallStatement {
    /// Creates a new CALL statement.
    #[must_use]
    pub fn new(procedure_name: impl Into<QualifiedName>, arguments: Vec<Expr>) -> Self {
        Self {
            procedure_name: procedure_name.into(),
            arguments,
            yield_items: vec![],
            where_clause: None,
        }
    }

    /// Adds YIELD items to the statement.
    #[must_use]
    pub fn yield_items(mut self, items: Vec<YieldItem>) -> Self {
        self.yield_items = items;
        self
    }

    /// Adds a YIELD * clause.
    #[must_use]
    pub fn yield_all(mut self) -> Self {
        self.yield_items = vec![YieldItem::Wildcard];
        self
    }

    /// Sets the WHERE clause for filtering yielded results.
    #[must_use]
    pub fn where_clause(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }
}

/// An item in a YIELD clause.
///
/// YIELD clauses specify which columns to extract from procedure results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YieldItem {
    /// A wildcard (*) - yield all columns from the procedure.
    Wildcard,
    /// A named column, optionally aliased.
    Column {
        /// The column name from the procedure result.
        name: Identifier,
        /// Optional alias (AS name).
        alias: Option<Identifier>,
    },
}

impl YieldItem {
    /// Creates a wildcard yield item.
    #[must_use]
    pub const fn wildcard() -> Self {
        Self::Wildcard
    }

    /// Creates an unaliased column yield item.
    #[must_use]
    pub fn column(name: impl Into<Identifier>) -> Self {
        Self::Column { name: name.into(), alias: None }
    }

    /// Creates an aliased column yield item.
    #[must_use]
    pub fn aliased(name: impl Into<Identifier>, alias: impl Into<Identifier>) -> Self {
        Self::Column { name: name.into(), alias: Some(alias.into()) }
    }
}

impl From<&str> for YieldItem {
    fn from(s: &str) -> Self {
        Self::column(s)
    }
}

impl From<Identifier> for YieldItem {
    fn from(id: Identifier) -> Self {
        Self::Column { name: id, alias: None }
    }
}

/// An item in a SELECT list.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectItem {
    /// An expression, optionally aliased.
    Expr {
        /// The expression.
        expr: Expr,
        /// Optional alias.
        alias: Option<Identifier>,
    },
    /// Wildcard (*).
    Wildcard,
    /// Qualified wildcard (table.*).
    QualifiedWildcard(QualifiedName),
}

impl SelectItem {
    /// Creates an unaliased expression item.
    #[must_use]
    pub const fn expr(expr: Expr) -> Self {
        Self::Expr { expr, alias: None }
    }

    /// Creates an aliased expression item.
    #[must_use]
    pub fn aliased(expr: Expr, alias: impl Into<Identifier>) -> Self {
        Self::Expr { expr, alias: Some(alias.into()) }
    }
}

impl From<Expr> for SelectItem {
    fn from(expr: Expr) -> Self {
        Self::expr(expr)
    }
}

/// A table reference in a FROM clause.
#[derive(Debug, Clone, PartialEq)]
pub enum TableRef {
    /// A simple table reference.
    Table {
        /// The table name.
        name: QualifiedName,
        /// Optional alias.
        alias: Option<TableAlias>,
    },
    /// A subquery.
    Subquery {
        /// The subquery.
        query: Box<SelectStatement>,
        /// Required alias for subqueries.
        alias: TableAlias,
    },
    /// A join between two table references.
    Join(Box<JoinClause>),
    /// A table function call.
    TableFunction {
        /// The function name.
        name: QualifiedName,
        /// Function arguments.
        args: Vec<Expr>,
        /// Optional alias.
        alias: Option<TableAlias>,
    },
}

impl TableRef {
    /// Creates a simple table reference.
    #[must_use]
    pub fn table(name: impl Into<QualifiedName>) -> Self {
        Self::Table { name: name.into(), alias: None }
    }

    /// Creates an aliased table reference.
    #[must_use]
    pub fn aliased(name: impl Into<QualifiedName>, alias: impl Into<TableAlias>) -> Self {
        Self::Table { name: name.into(), alias: Some(alias.into()) }
    }
}

/// A table alias with optional column aliases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableAlias {
    /// The alias name.
    pub name: Identifier,
    /// Optional column aliases.
    pub columns: Vec<Identifier>,
}

impl TableAlias {
    /// Creates a simple alias.
    #[must_use]
    pub fn new(name: impl Into<Identifier>) -> Self {
        Self { name: name.into(), columns: vec![] }
    }

    /// Creates an alias with column names.
    #[must_use]
    pub fn with_columns(name: impl Into<Identifier>, columns: Vec<Identifier>) -> Self {
        Self { name: name.into(), columns }
    }
}

impl From<&str> for TableAlias {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for TableAlias {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<Identifier> for TableAlias {
    fn from(id: Identifier) -> Self {
        Self::new(id)
    }
}

/// A JOIN clause.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinClause {
    /// Left side of the join.
    pub left: TableRef,
    /// Right side of the join.
    pub right: TableRef,
    /// Join type.
    pub join_type: JoinType,
    /// Join condition.
    pub condition: JoinCondition,
}

impl JoinClause {
    /// Creates an inner join.
    #[must_use]
    pub const fn inner(left: TableRef, right: TableRef, on: Expr) -> Self {
        Self { left, right, join_type: JoinType::Inner, condition: JoinCondition::On(on) }
    }

    /// Creates a left outer join.
    #[must_use]
    pub const fn left_join(left: TableRef, right: TableRef, on: Expr) -> Self {
        Self { left, right, join_type: JoinType::LeftOuter, condition: JoinCondition::On(on) }
    }
}

/// Type of JOIN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// INNER JOIN.
    Inner,
    /// LEFT OUTER JOIN.
    LeftOuter,
    /// RIGHT OUTER JOIN.
    RightOuter,
    /// FULL OUTER JOIN.
    FullOuter,
    /// CROSS JOIN.
    Cross,
}

/// Condition for a JOIN.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinCondition {
    /// ON clause.
    On(Expr),
    /// USING clause.
    Using(Vec<Identifier>),
    /// NATURAL join (no explicit condition).
    Natural,
    /// No condition (for CROSS JOIN).
    None,
}

/// A set operation (UNION, INTERSECT, EXCEPT).
#[derive(Debug, Clone, PartialEq)]
pub struct SetOperation {
    /// The type of set operation.
    pub op: SetOperator,
    /// Whether ALL is specified.
    pub all: bool,
    /// The right-hand SELECT statement.
    pub right: SelectStatement,
}

/// Set operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOperator {
    /// UNION.
    Union,
    /// INTERSECT.
    Intersect,
    /// EXCEPT.
    Except,
}

/// An INSERT statement.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStatement {
    /// The target table.
    pub table: QualifiedName,
    /// Optional column list.
    pub columns: Vec<Identifier>,
    /// The source of values.
    pub source: InsertSource,
    /// Optional ON CONFLICT clause.
    pub on_conflict: Option<OnConflict>,
    /// Optional RETURNING clause.
    pub returning: Vec<SelectItem>,
}

impl InsertStatement {
    /// Creates a new INSERT statement with VALUES.
    #[must_use]
    pub fn values(table: impl Into<QualifiedName>, values: Vec<Vec<Expr>>) -> Self {
        Self {
            table: table.into(),
            columns: vec![],
            source: InsertSource::Values(values),
            on_conflict: None,
            returning: vec![],
        }
    }

    /// Creates a new INSERT ... SELECT statement.
    #[must_use]
    pub fn select(table: impl Into<QualifiedName>, query: SelectStatement) -> Self {
        Self {
            table: table.into(),
            columns: vec![],
            source: InsertSource::Query(Box::new(query)),
            on_conflict: None,
            returning: vec![],
        }
    }

    /// Sets the column list.
    #[must_use]
    pub fn columns(mut self, columns: Vec<Identifier>) -> Self {
        self.columns = columns;
        self
    }

    /// Sets the RETURNING clause.
    #[must_use]
    pub fn returning(mut self, items: Vec<SelectItem>) -> Self {
        self.returning = items;
        self
    }
}

/// Source of values for INSERT.
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// VALUES clause with rows of expressions.
    Values(Vec<Vec<Expr>>),
    /// SELECT subquery.
    Query(Box<SelectStatement>),
    /// DEFAULT VALUES.
    DefaultValues,
}

/// ON CONFLICT clause for INSERT.
#[derive(Debug, Clone, PartialEq)]
pub struct OnConflict {
    /// The conflict target (columns or constraint).
    pub target: ConflictTarget,
    /// The action to take on conflict.
    pub action: ConflictAction,
}

/// Target for ON CONFLICT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictTarget {
    /// Specific columns.
    Columns(Vec<Identifier>),
    /// A named constraint.
    Constraint(Identifier),
}

/// Action for ON CONFLICT.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictAction {
    /// DO NOTHING.
    DoNothing,
    /// DO UPDATE SET ...
    DoUpdate {
        /// The assignments.
        assignments: Vec<Assignment>,
        /// Optional WHERE clause.
        where_clause: Option<Expr>,
    },
}

/// An UPDATE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateStatement {
    /// The target table.
    pub table: QualifiedName,
    /// Optional alias for the table.
    pub alias: Option<TableAlias>,
    /// The assignments (SET clause).
    pub assignments: Vec<Assignment>,
    /// Optional FROM clause (for UPDATE ... FROM).
    pub from: Vec<TableRef>,
    /// Optional MATCH clause for graph patterns.
    pub match_clause: Option<GraphPattern>,
    /// Optional WHERE clause.
    pub where_clause: Option<Expr>,
    /// Optional RETURNING clause.
    pub returning: Vec<SelectItem>,
}

impl UpdateStatement {
    /// Creates a new UPDATE statement.
    #[must_use]
    pub fn new(table: impl Into<QualifiedName>, assignments: Vec<Assignment>) -> Self {
        Self {
            table: table.into(),
            alias: None,
            assignments,
            from: vec![],
            match_clause: None,
            where_clause: None,
            returning: vec![],
        }
    }

    /// Sets the WHERE clause.
    #[must_use]
    pub fn where_clause(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Sets the RETURNING clause.
    #[must_use]
    pub fn returning(mut self, items: Vec<SelectItem>) -> Self {
        self.returning = items;
        self
    }
}

/// An assignment in an UPDATE or INSERT ON CONFLICT.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// The column to assign to.
    pub column: Identifier,
    /// The value expression.
    pub value: Expr,
}

impl Assignment {
    /// Creates a new assignment.
    #[must_use]
    pub fn new(column: impl Into<Identifier>, value: Expr) -> Self {
        Self { column: column.into(), value }
    }
}

/// A DELETE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStatement {
    /// The target table.
    pub table: QualifiedName,
    /// Optional alias for the table.
    pub alias: Option<TableAlias>,
    /// Optional USING clause.
    pub using: Vec<TableRef>,
    /// Optional MATCH clause for graph patterns.
    pub match_clause: Option<GraphPattern>,
    /// Optional WHERE clause.
    pub where_clause: Option<Expr>,
    /// Optional RETURNING clause.
    pub returning: Vec<SelectItem>,
}

impl DeleteStatement {
    /// Creates a new DELETE statement.
    #[must_use]
    pub fn new(table: impl Into<QualifiedName>) -> Self {
        Self {
            table: table.into(),
            alias: None,
            using: vec![],
            match_clause: None,
            where_clause: None,
            returning: vec![],
        }
    }

    /// Sets the WHERE clause.
    #[must_use]
    pub fn where_clause(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Sets the RETURNING clause.
    #[must_use]
    pub fn returning(mut self, items: Vec<SelectItem>) -> Self {
        self.returning = items;
        self
    }
}

/// A CREATE TABLE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTableStatement {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The table name.
    pub name: QualifiedName,
    /// Column definitions.
    pub columns: Vec<ColumnDef>,
    /// Table constraints.
    pub constraints: Vec<TableConstraint>,
}

impl CreateTableStatement {
    /// Creates a new CREATE TABLE statement.
    #[must_use]
    pub fn new(name: impl Into<QualifiedName>, columns: Vec<ColumnDef>) -> Self {
        Self { if_not_exists: false, name: name.into(), columns, constraints: vec![] }
    }
}

/// A column definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    /// The column name.
    pub name: Identifier,
    /// The data type.
    pub data_type: DataType,
    /// Column constraints.
    pub constraints: Vec<ColumnConstraint>,
}

impl ColumnDef {
    /// Creates a new column definition.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, data_type: DataType) -> Self {
        Self { name: name.into(), data_type, constraints: vec![] }
    }

    /// Adds a NOT NULL constraint.
    #[must_use]
    pub fn not_null(mut self) -> Self {
        self.constraints.push(ColumnConstraint::NotNull);
        self
    }

    /// Adds a PRIMARY KEY constraint.
    #[must_use]
    pub fn primary_key(mut self) -> Self {
        self.constraints.push(ColumnConstraint::PrimaryKey);
        self
    }

    /// Adds a UNIQUE constraint.
    #[must_use]
    pub fn unique(mut self) -> Self {
        self.constraints.push(ColumnConstraint::Unique);
        self
    }

    /// Adds a DEFAULT value.
    #[must_use]
    pub fn default(mut self, value: Expr) -> Self {
        self.constraints.push(ColumnConstraint::Default(value));
        self
    }
}

/// SQL data types.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// BOOLEAN.
    Boolean,
    /// SMALLINT (16-bit).
    SmallInt,
    /// INTEGER (32-bit).
    Integer,
    /// BIGINT (64-bit).
    BigInt,
    /// REAL (32-bit float).
    Real,
    /// DOUBLE PRECISION (64-bit float).
    DoublePrecision,
    /// NUMERIC with optional precision and scale.
    Numeric {
        /// Total digits.
        precision: Option<u32>,
        /// Digits after decimal point.
        scale: Option<u32>,
    },
    /// VARCHAR with optional length.
    Varchar(Option<u32>),
    /// TEXT (unlimited length string).
    Text,
    /// BYTEA (binary data).
    Bytea,
    /// TIMESTAMP.
    Timestamp,
    /// DATE.
    Date,
    /// TIME.
    Time,
    /// INTERVAL.
    Interval,
    /// JSON.
    Json,
    /// JSONB.
    Jsonb,
    /// UUID.
    Uuid,
    /// VECTOR with dimension.
    Vector(Option<u32>),
    /// Array of another type.
    Array(Box<DataType>),
    /// Custom/user-defined type.
    Custom(String),
}

/// Column constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraint {
    /// NOT NULL.
    NotNull,
    /// NULL (explicit).
    Null,
    /// UNIQUE.
    Unique,
    /// PRIMARY KEY.
    PrimaryKey,
    /// REFERENCES (foreign key).
    References {
        /// Referenced table.
        table: QualifiedName,
        /// Referenced column.
        column: Option<Identifier>,
    },
    /// CHECK constraint.
    Check(Expr),
    /// DEFAULT value.
    Default(Expr),
}

/// Table-level constraints.
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraint {
    /// PRIMARY KEY.
    PrimaryKey {
        /// Optional constraint name.
        name: Option<Identifier>,
        /// Columns in the key.
        columns: Vec<Identifier>,
    },
    /// UNIQUE.
    Unique {
        /// Optional constraint name.
        name: Option<Identifier>,
        /// Columns in the constraint.
        columns: Vec<Identifier>,
    },
    /// FOREIGN KEY.
    ForeignKey {
        /// Optional constraint name.
        name: Option<Identifier>,
        /// Local columns.
        columns: Vec<Identifier>,
        /// Referenced table.
        references_table: QualifiedName,
        /// Referenced columns.
        references_columns: Vec<Identifier>,
    },
    /// CHECK.
    Check {
        /// Optional constraint name.
        name: Option<Identifier>,
        /// The check expression.
        expr: Expr,
    },
}

/// An ALTER TABLE statement.
///
/// Supports adding, dropping, and modifying columns, as well as renaming
/// tables and columns.
///
/// # Examples
///
/// Add a column:
/// ```sql
/// ALTER TABLE users ADD COLUMN email VARCHAR(255);
/// ```
///
/// Drop a column:
/// ```sql
/// ALTER TABLE users DROP COLUMN temporary_field;
/// ```
///
/// Alter a column's nullability:
/// ```sql
/// ALTER TABLE users ALTER COLUMN name SET NOT NULL;
/// ```
///
/// Change column type:
/// ```sql
/// ALTER TABLE users ALTER COLUMN score TYPE FLOAT;
/// ```
///
/// Rename a column:
/// ```sql
/// ALTER TABLE users RENAME COLUMN old_name TO new_name;
/// ```
///
/// Rename the table:
/// ```sql
/// ALTER TABLE old_table RENAME TO new_table;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTableStatement {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The table to alter.
    pub name: QualifiedName,
    /// The alterations to perform.
    pub actions: Vec<AlterTableAction>,
}

impl AlterTableStatement {
    /// Creates a new ALTER TABLE statement.
    #[must_use]
    pub fn new(name: impl Into<QualifiedName>, actions: Vec<AlterTableAction>) -> Self {
        Self { if_exists: false, name: name.into(), actions }
    }

    /// Sets the IF EXISTS flag.
    #[must_use]
    pub const fn if_exists(mut self) -> Self {
        self.if_exists = true;
        self
    }

    /// Adds an action to the statement.
    #[must_use]
    pub fn add_action(mut self, action: AlterTableAction) -> Self {
        self.actions.push(action);
        self
    }
}

/// An action to perform in an ALTER TABLE statement.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableAction {
    /// ADD COLUMN: Add a new column to the table.
    AddColumn {
        /// Whether IF NOT EXISTS is specified.
        if_not_exists: bool,
        /// The column definition.
        column: ColumnDef,
    },
    /// DROP COLUMN: Remove a column from the table.
    DropColumn {
        /// Whether IF EXISTS is specified.
        if_exists: bool,
        /// The column name to drop.
        column_name: Identifier,
        /// Whether CASCADE is specified (drops dependent objects).
        cascade: bool,
    },
    /// ALTER COLUMN: Modify an existing column.
    AlterColumn {
        /// The column name to alter.
        column_name: Identifier,
        /// The alteration to perform.
        action: AlterColumnAction,
    },
    /// RENAME COLUMN: Rename a column.
    RenameColumn {
        /// The current column name.
        old_name: Identifier,
        /// The new column name.
        new_name: Identifier,
    },
    /// RENAME TO: Rename the table.
    RenameTable {
        /// The new table name.
        new_name: QualifiedName,
    },
    /// ADD CONSTRAINT: Add a table constraint.
    AddConstraint(TableConstraint),
    /// DROP CONSTRAINT: Remove a constraint.
    DropConstraint {
        /// Whether IF EXISTS is specified.
        if_exists: bool,
        /// The constraint name to drop.
        constraint_name: Identifier,
        /// Whether CASCADE is specified.
        cascade: bool,
    },
}

impl AlterTableAction {
    /// Creates an ADD COLUMN action.
    #[must_use]
    pub fn add_column(column: ColumnDef) -> Self {
        Self::AddColumn { if_not_exists: false, column }
    }

    /// Creates an ADD COLUMN IF NOT EXISTS action.
    #[must_use]
    pub fn add_column_if_not_exists(column: ColumnDef) -> Self {
        Self::AddColumn { if_not_exists: true, column }
    }

    /// Creates a DROP COLUMN action.
    #[must_use]
    pub fn drop_column(column_name: impl Into<Identifier>) -> Self {
        Self::DropColumn { if_exists: false, column_name: column_name.into(), cascade: false }
    }

    /// Creates a DROP COLUMN IF EXISTS action.
    #[must_use]
    pub fn drop_column_if_exists(column_name: impl Into<Identifier>) -> Self {
        Self::DropColumn { if_exists: true, column_name: column_name.into(), cascade: false }
    }

    /// Creates a DROP COLUMN CASCADE action.
    #[must_use]
    pub fn drop_column_cascade(column_name: impl Into<Identifier>) -> Self {
        Self::DropColumn { if_exists: false, column_name: column_name.into(), cascade: true }
    }

    /// Creates an ALTER COLUMN action.
    #[must_use]
    pub fn alter_column(column_name: impl Into<Identifier>, action: AlterColumnAction) -> Self {
        Self::AlterColumn { column_name: column_name.into(), action }
    }

    /// Creates a RENAME COLUMN action.
    #[must_use]
    pub fn rename_column(old_name: impl Into<Identifier>, new_name: impl Into<Identifier>) -> Self {
        Self::RenameColumn { old_name: old_name.into(), new_name: new_name.into() }
    }

    /// Creates a RENAME TO action.
    #[must_use]
    pub fn rename_table(new_name: impl Into<QualifiedName>) -> Self {
        Self::RenameTable { new_name: new_name.into() }
    }

    /// Creates an ADD CONSTRAINT action.
    #[must_use]
    pub const fn add_constraint(constraint: TableConstraint) -> Self {
        Self::AddConstraint(constraint)
    }

    /// Creates a DROP CONSTRAINT action.
    #[must_use]
    pub fn drop_constraint(constraint_name: impl Into<Identifier>) -> Self {
        Self::DropConstraint {
            if_exists: false,
            constraint_name: constraint_name.into(),
            cascade: false,
        }
    }
}

/// An action to perform on a column in ALTER COLUMN.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterColumnAction {
    /// SET NOT NULL: Make the column non-nullable.
    SetNotNull,
    /// DROP NOT NULL: Make the column nullable.
    DropNotNull,
    /// SET DEFAULT: Set a default value.
    SetDefault(Expr),
    /// DROP DEFAULT: Remove the default value.
    DropDefault,
    /// SET DATA TYPE / TYPE: Change the column's data type.
    SetType {
        /// The new data type.
        data_type: DataType,
        /// Optional USING expression for type conversion.
        using: Option<Expr>,
    },
}

impl AlterColumnAction {
    /// Creates a SET NOT NULL action.
    #[must_use]
    pub const fn set_not_null() -> Self {
        Self::SetNotNull
    }

    /// Creates a DROP NOT NULL action.
    #[must_use]
    pub const fn drop_not_null() -> Self {
        Self::DropNotNull
    }

    /// Creates a SET DEFAULT action.
    #[must_use]
    pub const fn set_default(expr: Expr) -> Self {
        Self::SetDefault(expr)
    }

    /// Creates a DROP DEFAULT action.
    #[must_use]
    pub const fn drop_default() -> Self {
        Self::DropDefault
    }

    /// Creates a SET TYPE action.
    #[must_use]
    pub const fn set_type(data_type: DataType) -> Self {
        Self::SetType { data_type, using: None }
    }

    /// Creates a SET TYPE action with a USING clause.
    #[must_use]
    pub const fn set_type_using(data_type: DataType, using: Expr) -> Self {
        Self::SetType { data_type, using: Some(using) }
    }
}

/// A CREATE INDEX statement.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndexStatement {
    /// Whether UNIQUE is specified.
    pub unique: bool,
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The index name.
    pub name: Identifier,
    /// The table to index.
    pub table: QualifiedName,
    /// The columns/expressions to index.
    pub columns: Vec<IndexColumn>,
    /// The index method (btree, hash, gin, hnsw, ivfflat).
    pub using: Option<String>,
    /// Index-specific options.
    pub with: Vec<(String, String)>,
    /// Optional WHERE clause for partial indexes.
    pub where_clause: Option<Expr>,
}

/// A column in an index.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexColumn {
    /// The column or expression.
    pub expr: Expr,
    /// Sort order (ASC/DESC).
    pub asc: Option<bool>,
    /// NULLS FIRST/LAST.
    pub nulls_first: Option<bool>,
    /// Operator class.
    pub opclass: Option<String>,
}

impl IndexColumn {
    /// Creates a new index column.
    #[must_use]
    pub const fn new(expr: Expr) -> Self {
        Self { expr, asc: None, nulls_first: None, opclass: None }
    }
}

/// A DROP TABLE statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTableStatement {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The table(s) to drop.
    pub names: Vec<QualifiedName>,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

/// A DROP INDEX statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropIndexStatement {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The index(es) to drop.
    pub names: Vec<QualifiedName>,
    /// Whether CASCADE is specified.
    pub cascade: bool,
}

// ============================================================================
// VIEW Statements
// ============================================================================

/// A CREATE VIEW statement.
///
/// Views are stored query definitions that can be used like tables.
/// They don't store data themselves, but expand to their defining query
/// when referenced in other queries.
///
/// # Examples
///
/// Basic view:
/// ```sql
/// CREATE VIEW active_users AS
/// SELECT * FROM users WHERE status = 'active';
/// ```
///
/// Replace existing view:
/// ```sql
/// CREATE OR REPLACE VIEW user_stats AS
/// SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
/// FROM employees
/// GROUP BY department;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CreateViewStatement {
    /// Whether OR REPLACE is specified.
    pub or_replace: bool,
    /// The view name.
    pub name: QualifiedName,
    /// Optional column aliases for the view.
    pub columns: Vec<Identifier>,
    /// The query defining the view.
    pub query: Box<SelectStatement>,
}

impl CreateViewStatement {
    /// Creates a new CREATE VIEW statement.
    #[must_use]
    pub fn new(name: impl Into<QualifiedName>, query: SelectStatement) -> Self {
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

/// A DROP VIEW statement.
///
/// Removes one or more views from the database.
///
/// # Examples
///
/// ```sql
/// DROP VIEW active_users;
/// DROP VIEW IF EXISTS maybe_exists;
/// DROP VIEW cascade_view CASCADE;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropViewStatement {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The view(s) to drop.
    pub names: Vec<QualifiedName>,
    /// Whether CASCADE is specified (drops dependent objects).
    pub cascade: bool,
}

impl DropViewStatement {
    /// Creates a new DROP VIEW statement.
    #[must_use]
    pub fn new(names: Vec<QualifiedName>) -> Self {
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

/// A CREATE COLLECTION statement for vector collections.
///
/// Creates a collection with named vector configurations and optional payload fields:
///
/// ```sql
/// CREATE COLLECTION documents (
///     title TEXT,
///     content TEXT,
///     VECTOR text_embedding DIMENSION 1536,
///     VECTOR image_embedding DIMENSION 512,
///     VECTOR summary_embedding DIMENSION 1536
/// );
/// ```
///
/// Alternative syntax with USING and WITH clauses:
///
/// ```sql
/// CREATE COLLECTION documents (
///     dense VECTOR(768) USING hnsw WITH (distance = 'cosine'),
///     sparse SPARSE_VECTOR USING inverted,
///     colbert MULTI_VECTOR(128) USING hnsw WITH (aggregation = 'maxsim')
/// );
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CreateCollectionStatement {
    /// Whether IF NOT EXISTS is specified.
    pub if_not_exists: bool,
    /// The collection name.
    pub name: Identifier,
    /// Named vector definitions.
    pub vectors: Vec<VectorDef>,
    /// Payload field definitions.
    pub payload_fields: Vec<PayloadFieldDef>,
}

/// A payload field definition in a collection.
///
/// Defines a field in the collection's payload schema.
#[derive(Debug, Clone, PartialEq)]
pub struct PayloadFieldDef {
    /// The field name.
    pub name: Identifier,
    /// The field data type.
    pub data_type: DataType,
    /// Whether the field is indexed.
    pub indexed: bool,
}

impl CreateCollectionStatement {
    /// Creates a new CREATE COLLECTION statement.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, vectors: Vec<VectorDef>) -> Self {
        Self { if_not_exists: false, name: name.into(), vectors, payload_fields: vec![] }
    }

    /// Creates a CREATE COLLECTION statement with vectors and payload fields.
    #[must_use]
    pub fn with_payload(
        name: impl Into<Identifier>,
        vectors: Vec<VectorDef>,
        payload_fields: Vec<PayloadFieldDef>,
    ) -> Self {
        Self { if_not_exists: false, name: name.into(), vectors, payload_fields }
    }

    /// Set IF NOT EXISTS flag.
    #[must_use]
    pub const fn if_not_exists(mut self) -> Self {
        self.if_not_exists = true;
        self
    }

    /// Add a payload field definition.
    #[must_use]
    pub fn with_field(mut self, field: PayloadFieldDef) -> Self {
        self.payload_fields.push(field);
        self
    }
}

impl PayloadFieldDef {
    /// Creates a new payload field definition.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, data_type: DataType) -> Self {
        Self { name: name.into(), data_type, indexed: false }
    }

    /// Set the field as indexed.
    #[must_use]
    pub const fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }
}

/// A named vector definition in a collection.
///
/// Defines a single named vector space with its type, index method, and options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorDef {
    /// The vector name (e.g., "dense", "sparse", "colbert").
    pub name: Identifier,
    /// The vector type.
    pub vector_type: VectorTypeDef,
    /// The index method (e.g., "hnsw", "inverted", "flat").
    pub using: Option<String>,
    /// Index and vector options (e.g., distance, aggregation, m, ef_construction).
    pub with_options: Vec<(String, String)>,
}

impl VectorDef {
    /// Creates a new vector definition.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, vector_type: VectorTypeDef) -> Self {
        Self { name: name.into(), vector_type, using: None, with_options: vec![] }
    }

    /// Set the index method.
    #[must_use]
    pub fn using(mut self, method: impl Into<String>) -> Self {
        self.using = Some(method.into());
        self
    }

    /// Add a WITH option.
    #[must_use]
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.with_options.push((key.into(), value.into()));
        self
    }
}

/// Vector type definition in DDL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorTypeDef {
    /// Dense vector with fixed dimension: VECTOR(768).
    Vector {
        /// The dimension of the vector.
        dimension: u32,
    },
    /// Sparse vector with optional max dimension: SPARSE_VECTOR or SPARSE_VECTOR(30522).
    SparseVector {
        /// Maximum vocabulary size (optional).
        max_dimension: Option<u32>,
    },
    /// Multi-vector with per-token dimension: MULTI_VECTOR(128).
    MultiVector {
        /// The dimension of each token embedding.
        token_dim: u32,
    },
    /// Binary vector with bit count: BINARY_VECTOR(1024).
    BinaryVector {
        /// The number of bits.
        bits: u32,
    },
}

impl VectorTypeDef {
    /// Create a dense vector type.
    #[must_use]
    pub const fn dense(dimension: u32) -> Self {
        Self::Vector { dimension }
    }

    /// Create a sparse vector type.
    #[must_use]
    pub const fn sparse(max_dimension: Option<u32>) -> Self {
        Self::SparseVector { max_dimension }
    }

    /// Create a multi-vector type.
    #[must_use]
    pub const fn multi(token_dim: u32) -> Self {
        Self::MultiVector { token_dim }
    }

    /// Create a binary vector type.
    #[must_use]
    pub const fn binary(bits: u32) -> Self {
        Self::BinaryVector { bits }
    }
}

/// A DROP COLLECTION statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropCollectionStatement {
    /// Whether IF EXISTS is specified.
    pub if_exists: bool,
    /// The collection(s) to drop.
    pub names: Vec<Identifier>,
    /// Whether CASCADE is specified (drops associated data and indexes).
    pub cascade: bool,
}

impl DropCollectionStatement {
    /// Creates a new DROP COLLECTION statement.
    #[must_use]
    pub fn new(names: Vec<Identifier>) -> Self {
        Self { if_exists: false, names, cascade: false }
    }

    /// Set IF EXISTS flag.
    #[must_use]
    pub const fn if_exists(mut self) -> Self {
        self.if_exists = true;
        self
    }

    /// Set CASCADE flag.
    #[must_use]
    pub const fn cascade(mut self) -> Self {
        self.cascade = true;
        self
    }
}

// ============================================================================
// Cypher Graph Mutation Statements
// ============================================================================

/// A Cypher CREATE statement for creating nodes and relationships.
///
/// The CREATE clause is used to create new nodes and relationships in the graph.
///
/// # Examples
///
/// Create a single node:
/// ```text
/// CREATE (u:User {name: 'Alice', age: 30})
/// ```
///
/// Create a node and return it:
/// ```text
/// CREATE (u:User {name: 'Bob'}) RETURN u
/// ```
///
/// Create a relationship after MATCH:
/// ```text
/// MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'})
/// CREATE (a)-[:FOLLOWS {since: '2024-01-01'}]->(b)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CreateGraphStatement {
    /// Optional preceding MATCH clause to bind variables.
    pub match_clause: Option<GraphPattern>,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// Patterns to create (nodes and relationships).
    pub patterns: Vec<CreatePattern>,
    /// Optional RETURN clause.
    pub return_clause: Vec<ReturnItem>,
}

impl CreateGraphStatement {
    /// Creates a new CREATE statement with the given patterns.
    #[must_use]
    pub fn new(patterns: Vec<CreatePattern>) -> Self {
        Self { match_clause: None, where_clause: None, patterns, return_clause: vec![] }
    }

    /// Adds a MATCH clause to this CREATE statement.
    #[must_use]
    pub fn with_match(mut self, pattern: GraphPattern) -> Self {
        self.match_clause = Some(pattern);
        self
    }

    /// Adds a WHERE clause to this CREATE statement.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Adds a RETURN clause to this CREATE statement.
    #[must_use]
    pub fn with_return(mut self, items: Vec<ReturnItem>) -> Self {
        self.return_clause = items;
        self
    }
}

/// A pattern to create (node or relationship).
#[derive(Debug, Clone, PartialEq)]
pub enum CreatePattern {
    /// Create a node.
    Node {
        /// Variable name for the created node.
        variable: Option<Identifier>,
        /// Labels for the node.
        labels: Vec<Identifier>,
        /// Properties to set on the node.
        properties: Vec<(Identifier, Expr)>,
    },
    /// Create a relationship between two nodes.
    Relationship {
        /// Start node variable (must be bound from MATCH or earlier CREATE).
        start: Identifier,
        /// Relationship variable (optional).
        rel_variable: Option<Identifier>,
        /// Relationship type.
        rel_type: Identifier,
        /// Properties to set on the relationship.
        properties: Vec<(Identifier, Expr)>,
        /// End node variable (must be bound from MATCH or earlier CREATE).
        end: Identifier,
    },
    /// Create a full path pattern (e.g., (a)-[:KNOWS]->(b)-[:LIKES]->(c)).
    Path {
        /// The starting node (can be new or existing).
        start: CreateNodeRef,
        /// Steps in the path (each step is a relationship and destination node).
        steps: Vec<CreatePathStep>,
    },
}

/// Reference to a node in a CREATE path pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateNodeRef {
    /// Reference an existing variable (bound from MATCH).
    Variable(Identifier),
    /// Create a new node.
    New {
        /// Variable name for the created node.
        variable: Option<Identifier>,
        /// Labels for the node.
        labels: Vec<Identifier>,
        /// Properties to set on the node.
        properties: Vec<(Identifier, Expr)>,
    },
}

/// A step in a CREATE path pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct CreatePathStep {
    /// Relationship variable (optional).
    pub rel_variable: Option<Identifier>,
    /// Relationship type.
    pub rel_type: Identifier,
    /// Properties for the relationship.
    pub rel_properties: Vec<(Identifier, Expr)>,
    /// Direction (true = outgoing ->, false = incoming <-).
    pub outgoing: bool,
    /// The destination node.
    pub destination: CreateNodeRef,
}

/// A Cypher MERGE statement for upserting nodes and relationships.
///
/// MERGE creates nodes/relationships if they don't exist, or matches them if they do.
/// It supports ON CREATE and ON MATCH clauses to conditionally set properties.
///
/// # Examples
///
/// MERGE with ON CREATE/ON MATCH:
/// ```text
/// MERGE (u:User {email: 'alice@example.com'})
/// ON CREATE SET u.created_at = timestamp()
/// ON MATCH SET u.last_seen = timestamp()
/// ```
///
/// MERGE a relationship:
/// ```text
/// MATCH (a:User), (b:User)
/// WHERE a.name = 'Alice' AND b.name = 'Bob'
/// MERGE (a)-[:KNOWS]->(b)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MergeGraphStatement {
    /// Optional preceding MATCH clause to bind variables.
    pub match_clause: Option<GraphPattern>,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// The pattern to merge (match or create).
    pub pattern: MergePattern,
    /// Actions to perform when creating a new entity.
    pub on_create: Vec<SetAction>,
    /// Actions to perform when matching an existing entity.
    pub on_match: Vec<SetAction>,
    /// Optional RETURN clause.
    pub return_clause: Vec<ReturnItem>,
}

impl MergeGraphStatement {
    /// Creates a new MERGE statement with the given pattern.
    #[must_use]
    pub fn new(pattern: MergePattern) -> Self {
        Self {
            match_clause: None,
            where_clause: None,
            pattern,
            on_create: vec![],
            on_match: vec![],
            return_clause: vec![],
        }
    }

    /// Adds a MATCH clause to this MERGE statement.
    #[must_use]
    pub fn with_match(mut self, pattern: GraphPattern) -> Self {
        self.match_clause = Some(pattern);
        self
    }

    /// Adds a WHERE clause to this MERGE statement.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Adds ON CREATE actions.
    #[must_use]
    pub fn on_create(mut self, actions: Vec<SetAction>) -> Self {
        self.on_create = actions;
        self
    }

    /// Adds ON MATCH actions.
    #[must_use]
    pub fn on_match(mut self, actions: Vec<SetAction>) -> Self {
        self.on_match = actions;
        self
    }

    /// Adds a RETURN clause.
    #[must_use]
    pub fn with_return(mut self, items: Vec<ReturnItem>) -> Self {
        self.return_clause = items;
        self
    }
}

/// A pattern for MERGE operation.
#[derive(Debug, Clone, PartialEq)]
pub enum MergePattern {
    /// Merge a node.
    Node {
        /// Variable name for the merged node.
        variable: Identifier,
        /// Labels for matching/creating.
        labels: Vec<Identifier>,
        /// Properties to match on (key properties for upsert).
        match_properties: Vec<(Identifier, Expr)>,
    },
    /// Merge a relationship between two bound nodes.
    Relationship {
        /// Start node variable (must be bound).
        start: Identifier,
        /// Relationship variable (optional).
        rel_variable: Option<Identifier>,
        /// Relationship type.
        rel_type: Identifier,
        /// Properties to match on.
        match_properties: Vec<(Identifier, Expr)>,
        /// End node variable (must be bound).
        end: Identifier,
    },
}

/// An action to perform in SET clause or ON CREATE/ON MATCH.
#[derive(Debug, Clone, PartialEq)]
pub enum SetAction {
    /// Set a single property: SET n.prop = value
    Property {
        /// The variable to set the property on.
        variable: Identifier,
        /// The property name.
        property: Identifier,
        /// The value expression.
        value: Expr,
    },
    /// Set multiple properties from a map: SET n = {props} or SET n += {props}
    Properties {
        /// The variable to set properties on.
        variable: Identifier,
        /// The properties as an expression (should evaluate to a map).
        properties: Expr,
        /// Whether to replace all properties (=) or merge (+= ).
        replace: bool,
    },
    /// Add a label: SET n:Label
    Label {
        /// The variable to add the label to.
        variable: Identifier,
        /// The label to add.
        label: Identifier,
    },
}

impl SetAction {
    /// Creates a property assignment action.
    #[must_use]
    pub fn property(
        variable: impl Into<Identifier>,
        property: impl Into<Identifier>,
        value: Expr,
    ) -> Self {
        Self::Property { variable: variable.into(), property: property.into(), value }
    }

    /// Creates a label assignment action.
    #[must_use]
    pub fn label(variable: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self::Label { variable: variable.into(), label: label.into() }
    }
}

// ============================================================================
// Cypher SET, DELETE, and REMOVE Statements
// ============================================================================

/// A Cypher SET statement for updating properties.
///
/// SET can update properties on nodes or relationships matched by a MATCH clause.
///
/// # Examples
///
/// Set a property:
/// ```text
/// MATCH (u:User {name: 'Alice'})
/// SET u.verified = true, u.updated_at = timestamp()
/// ```
///
/// Add a label:
/// ```text
/// MATCH (u:User {name: 'Alice'})
/// SET u:Admin
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SetGraphStatement {
    /// The MATCH pattern to find entities to update.
    pub match_clause: GraphPattern,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// The SET actions to apply.
    pub set_actions: Vec<SetAction>,
    /// Optional RETURN clause.
    pub return_clause: Vec<ReturnItem>,
}

impl SetGraphStatement {
    /// Creates a new SET statement.
    #[must_use]
    pub fn new(match_clause: GraphPattern, set_actions: Vec<SetAction>) -> Self {
        Self { match_clause, where_clause: None, set_actions, return_clause: vec![] }
    }

    /// Adds a WHERE clause.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Adds a RETURN clause.
    #[must_use]
    pub fn with_return(mut self, items: Vec<ReturnItem>) -> Self {
        self.return_clause = items;
        self
    }
}

/// A Cypher DELETE statement for removing nodes and relationships.
///
/// DELETE removes nodes/relationships matched by a MATCH clause.
/// DETACH DELETE also removes all connected relationships.
///
/// # Examples
///
/// Delete a node (fails if has relationships):
/// ```text
/// MATCH (u:User {name: 'Alice'})
/// DELETE u
/// ```
///
/// Detach delete (removes node and all its relationships):
/// ```text
/// MATCH (u:User {name: 'Alice'})
/// DETACH DELETE u
/// ```
///
/// Delete a relationship:
/// ```text
/// MATCH (a:User)-[r:FOLLOWS]->(b:User)
/// WHERE a.name = 'Alice' AND b.name = 'Bob'
/// DELETE r
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteGraphStatement {
    /// The MATCH pattern to find entities to delete.
    pub match_clause: GraphPattern,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// Variables to delete (node or relationship variables).
    pub variables: Vec<Identifier>,
    /// Whether this is a DETACH DELETE (also deletes relationships).
    pub detach: bool,
    /// Optional RETURN clause (returns deleted entities).
    pub return_clause: Vec<ReturnItem>,
}

impl DeleteGraphStatement {
    /// Creates a new DELETE statement.
    #[must_use]
    pub fn new(match_clause: GraphPattern, variables: Vec<Identifier>) -> Self {
        Self { match_clause, where_clause: None, variables, detach: false, return_clause: vec![] }
    }

    /// Creates a new DETACH DELETE statement.
    #[must_use]
    pub fn detach(match_clause: GraphPattern, variables: Vec<Identifier>) -> Self {
        Self { match_clause, where_clause: None, variables, detach: true, return_clause: vec![] }
    }

    /// Adds a WHERE clause.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Adds a RETURN clause.
    #[must_use]
    pub fn with_return(mut self, items: Vec<ReturnItem>) -> Self {
        self.return_clause = items;
        self
    }
}

/// An item to remove in a REMOVE clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoveItem {
    /// Remove a property: REMOVE n.property
    Property {
        /// The variable to remove the property from.
        variable: Identifier,
        /// The property name to remove.
        property: Identifier,
    },
    /// Remove a label: REMOVE n:Label
    Label {
        /// The variable to remove the label from.
        variable: Identifier,
        /// The label to remove.
        label: Identifier,
    },
}

impl RemoveItem {
    /// Creates a property removal item.
    #[must_use]
    pub fn property(variable: impl Into<Identifier>, property: impl Into<Identifier>) -> Self {
        Self::Property { variable: variable.into(), property: property.into() }
    }

    /// Creates a label removal item.
    #[must_use]
    pub fn label(variable: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self::Label { variable: variable.into(), label: label.into() }
    }
}

/// A Cypher REMOVE statement for removing properties and labels.
///
/// REMOVE removes properties or labels from nodes/relationships.
///
/// # Examples
///
/// Remove a property:
/// ```text
/// MATCH (u:User {name: 'Alice'})
/// REMOVE u.temporary_field
/// ```
///
/// Remove a label:
/// ```text
/// MATCH (u:User:Admin {name: 'Alice'})
/// REMOVE u:Admin
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveGraphStatement {
    /// The MATCH pattern to find entities to modify.
    pub match_clause: GraphPattern,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// Items to remove (properties and labels).
    pub items: Vec<RemoveItem>,
    /// Optional RETURN clause.
    pub return_clause: Vec<ReturnItem>,
}

impl RemoveGraphStatement {
    /// Creates a new REMOVE statement.
    #[must_use]
    pub fn new(match_clause: GraphPattern, items: Vec<RemoveItem>) -> Self {
        Self { match_clause, where_clause: None, items, return_clause: vec![] }
    }

    /// Adds a WHERE clause.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }

    /// Adds a RETURN clause.
    #[must_use]
    pub fn with_return(mut self, items: Vec<ReturnItem>) -> Self {
        self.return_clause = items;
        self
    }
}

// ============================================================================
// Cypher FOREACH Statement
// ============================================================================

/// An action that can be performed inside a FOREACH clause.
///
/// FOREACH supports a subset of Cypher mutation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ForeachAction {
    /// SET action: SET n.prop = value or SET n:Label
    Set(SetAction),
    /// CREATE action: CREATE (n:Label {props})
    Create(CreatePattern),
    /// MERGE action: MERGE (n:Label {props})
    Merge(MergePattern),
    /// DELETE action: DELETE n or DETACH DELETE n
    Delete {
        /// Variables to delete.
        variables: Vec<Identifier>,
        /// Whether this is a DETACH DELETE.
        detach: bool,
    },
    /// REMOVE action: REMOVE n.prop or REMOVE n:Label
    Remove(RemoveItem),
    /// Nested FOREACH
    Foreach(Box<ForeachStatement>),
}

impl ForeachAction {
    /// Creates a SET property action.
    #[must_use]
    pub fn set_property(
        variable: impl Into<Identifier>,
        property: impl Into<Identifier>,
        value: Expr,
    ) -> Self {
        Self::Set(SetAction::property(variable, property, value))
    }

    /// Creates a SET label action.
    #[must_use]
    pub fn set_label(variable: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self::Set(SetAction::label(variable, label))
    }

    /// Creates a DELETE action.
    #[must_use]
    pub fn delete(variables: Vec<Identifier>) -> Self {
        Self::Delete { variables, detach: false }
    }

    /// Creates a DETACH DELETE action.
    #[must_use]
    pub fn detach_delete(variables: Vec<Identifier>) -> Self {
        Self::Delete { variables, detach: true }
    }

    /// Creates a REMOVE property action.
    #[must_use]
    pub fn remove_property(
        variable: impl Into<Identifier>,
        property: impl Into<Identifier>,
    ) -> Self {
        Self::Remove(RemoveItem::property(variable, property))
    }

    /// Creates a REMOVE label action.
    #[must_use]
    pub fn remove_label(variable: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self::Remove(RemoveItem::label(variable, label))
    }
}

/// A Cypher FOREACH statement for iterating over lists and performing mutations.
///
/// FOREACH iterates over a list expression and executes mutation operations
/// for each element. The variable is bound to each element in turn.
///
/// # Examples
///
/// Set a property on all matched nodes:
/// ```text
/// MATCH (n:Person)
/// FOREACH (x IN n.friends | SET x.contacted = true)
/// ```
///
/// Create nodes from a list:
/// ```text
/// FOREACH (name IN ['Alice', 'Bob', 'Carol'] |
///     CREATE (n:Person {name: name})
/// )
/// ```
///
/// Nested FOREACH:
/// ```text
/// FOREACH (i IN range(0, 10) |
///     FOREACH (j IN range(0, 10) |
///         CREATE (n:Cell {x: i, y: j})
///     )
/// )
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ForeachStatement {
    /// Optional preceding MATCH clause to bind variables.
    pub match_clause: Option<GraphPattern>,
    /// Optional WHERE clause on the MATCH.
    pub where_clause: Option<Expr>,
    /// The iteration variable bound to each list element.
    pub variable: Identifier,
    /// The list expression to iterate over.
    pub list_expr: Expr,
    /// Actions to perform for each element.
    pub actions: Vec<ForeachAction>,
}

impl ForeachStatement {
    /// Creates a new FOREACH statement.
    #[must_use]
    pub fn new(
        variable: impl Into<Identifier>,
        list_expr: Expr,
        actions: Vec<ForeachAction>,
    ) -> Self {
        Self {
            match_clause: None,
            where_clause: None,
            variable: variable.into(),
            list_expr,
            actions,
        }
    }

    /// Adds a MATCH clause to this FOREACH statement.
    #[must_use]
    pub fn with_match(mut self, pattern: GraphPattern) -> Self {
        self.match_clause = Some(pattern);
        self
    }

    /// Adds a WHERE clause to this FOREACH statement.
    #[must_use]
    pub fn with_where(mut self, condition: Expr) -> Self {
        self.where_clause = Some(condition);
        self
    }
}

// ============================================================================
// Utility Statements
// ============================================================================

/// A utility statement (VACUUM, ANALYZE, COPY, SET, SHOW, RESET).
///
/// These statements perform database maintenance and configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum UtilityStatement {
    /// VACUUM statement for table maintenance.
    Vacuum(VacuumStatement),
    /// ANALYZE statement for statistics collection.
    Analyze(AnalyzeStatement),
    /// COPY statement for data import/export.
    Copy(CopyStatement),
    /// SET statement for session variables.
    Set(SetSessionStatement),
    /// SHOW statement for viewing configuration.
    Show(ShowStatement),
    /// RESET statement for resetting variables.
    Reset(ResetStatement),
}

/// An EXPLAIN ANALYZE statement.
///
/// EXPLAIN ANALYZE executes the statement and shows actual execution statistics.
///
/// # Examples
///
/// Basic usage:
/// ```sql
/// EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1;
/// ```
///
/// With options:
/// ```sql
/// EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT * FROM users;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct ExplainAnalyzeStatement {
    /// The statement to analyze.
    pub statement: Statement,
    /// Whether to include buffer usage statistics.
    pub buffers: bool,
    /// Whether to include timing information (default true).
    pub timing: bool,
    /// Output format for the plan.
    pub format: ExplainFormat,
    /// Whether to show verbose output.
    pub verbose: bool,
    /// Whether to show cost estimates.
    pub costs: bool,
    /// Whether to show settings.
    pub settings: bool,
}

impl ExplainAnalyzeStatement {
    /// Creates a new EXPLAIN ANALYZE statement with default options.
    #[must_use]
    pub fn new(statement: Statement) -> Self {
        Self {
            statement,
            buffers: false,
            timing: true,
            format: ExplainFormat::Text,
            verbose: false,
            costs: true,
            settings: false,
        }
    }

    /// Sets the buffers flag.
    #[must_use]
    pub const fn with_buffers(mut self, buffers: bool) -> Self {
        self.buffers = buffers;
        self
    }

    /// Sets the timing flag.
    #[must_use]
    pub const fn with_timing(mut self, timing: bool) -> Self {
        self.timing = timing;
        self
    }

    /// Sets the output format.
    #[must_use]
    pub const fn with_format(mut self, format: ExplainFormat) -> Self {
        self.format = format;
        self
    }

    /// Sets the verbose flag.
    #[must_use]
    pub const fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the costs flag.
    #[must_use]
    pub const fn with_costs(mut self, costs: bool) -> Self {
        self.costs = costs;
        self
    }
}

/// Output format for EXPLAIN.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExplainFormat {
    /// Plain text format (default).
    #[default]
    Text,
    /// JSON format.
    Json,
    /// XML format.
    Xml,
    /// YAML format.
    Yaml,
}

impl std::fmt::Display for ExplainFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "TEXT"),
            Self::Json => write!(f, "JSON"),
            Self::Xml => write!(f, "XML"),
            Self::Yaml => write!(f, "YAML"),
        }
    }
}

/// A VACUUM statement for table maintenance.
///
/// VACUUM reclaims storage and optionally updates statistics.
///
/// # Examples
///
/// ```sql
/// VACUUM users;
/// VACUUM FULL users;
/// VACUUM ANALYZE users;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VacuumStatement {
    /// Whether FULL vacuum is requested (reclaims more space).
    pub full: bool,
    /// Whether to also collect statistics (VACUUM ANALYZE).
    pub analyze: bool,
    /// Target table (None means all tables).
    pub table: Option<QualifiedName>,
    /// Specific columns to analyze (only with analyze=true).
    pub columns: Vec<Identifier>,
}

impl VacuumStatement {
    /// Creates a new VACUUM statement for all tables.
    #[must_use]
    pub fn all() -> Self {
        Self { full: false, analyze: false, table: None, columns: vec![] }
    }

    /// Creates a VACUUM statement for a specific table.
    #[must_use]
    pub fn table(name: impl Into<QualifiedName>) -> Self {
        Self { full: false, analyze: false, table: Some(name.into()), columns: vec![] }
    }

    /// Sets the FULL flag.
    #[must_use]
    pub const fn full(mut self) -> Self {
        self.full = true;
        self
    }

    /// Sets the ANALYZE flag.
    #[must_use]
    pub const fn analyze(mut self) -> Self {
        self.analyze = true;
        self
    }

    /// Sets specific columns to analyze.
    #[must_use]
    pub fn columns(mut self, columns: Vec<Identifier>) -> Self {
        self.columns = columns;
        self
    }
}

/// An ANALYZE statement for statistics collection.
///
/// ANALYZE collects statistics about table contents for the query planner.
///
/// # Examples
///
/// ```sql
/// ANALYZE;           -- All tables
/// ANALYZE users;     -- Specific table
/// ANALYZE users (name, email);  -- Specific columns
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzeStatement {
    /// Target table (None means all tables).
    pub table: Option<QualifiedName>,
    /// Specific columns to analyze.
    pub columns: Vec<Identifier>,
}

impl AnalyzeStatement {
    /// Creates an ANALYZE statement for all tables.
    #[must_use]
    pub fn all() -> Self {
        Self { table: None, columns: vec![] }
    }

    /// Creates an ANALYZE statement for a specific table.
    #[must_use]
    pub fn table(name: impl Into<QualifiedName>) -> Self {
        Self { table: Some(name.into()), columns: vec![] }
    }

    /// Sets specific columns to analyze.
    #[must_use]
    pub fn columns(mut self, columns: Vec<Identifier>) -> Self {
        self.columns = columns;
        self
    }
}

/// A COPY statement for data import/export.
///
/// COPY moves data between tables and files or standard output.
///
/// # Examples
///
/// Export to file:
/// ```sql
/// COPY users TO '/tmp/users.csv' WITH (FORMAT CSV, HEADER);
/// ```
///
/// Import from file:
/// ```sql
/// COPY users FROM '/tmp/users.csv' WITH (FORMAT CSV, HEADER);
/// ```
///
/// Export query to stdout:
/// ```sql
/// COPY (SELECT * FROM users WHERE active) TO STDOUT;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CopyStatement {
    /// The copy target (table or query).
    pub target: CopyTarget,
    /// The copy direction.
    pub direction: CopyDirection,
    /// Copy options.
    pub options: CopyOptions,
}

impl CopyStatement {
    /// Creates a COPY TO statement for a table.
    #[must_use]
    pub fn table_to(table: impl Into<QualifiedName>, destination: impl Into<String>) -> Self {
        Self {
            target: CopyTarget::Table { name: table.into(), columns: vec![] },
            direction: CopyDirection::To(CopyDestination::File(destination.into())),
            options: CopyOptions::default(),
        }
    }

    /// Creates a COPY FROM statement for a table.
    #[must_use]
    pub fn table_from(table: impl Into<QualifiedName>, source: impl Into<String>) -> Self {
        Self {
            target: CopyTarget::Table { name: table.into(), columns: vec![] },
            direction: CopyDirection::From(CopySource::File(source.into())),
            options: CopyOptions::default(),
        }
    }

    /// Creates a COPY TO statement for a query.
    #[must_use]
    pub fn query_to(query: SelectStatement, destination: impl Into<String>) -> Self {
        Self {
            target: CopyTarget::Query(Box::new(query)),
            direction: CopyDirection::To(CopyDestination::File(destination.into())),
            options: CopyOptions::default(),
        }
    }

    /// Sets copy options.
    #[must_use]
    pub fn with_options(mut self, options: CopyOptions) -> Self {
        self.options = options;
        self
    }
}

/// The target of a COPY statement.
#[derive(Debug, Clone, PartialEq)]
pub enum CopyTarget {
    /// A table (optionally with specific columns).
    Table {
        /// Table name.
        name: QualifiedName,
        /// Specific columns (empty means all).
        columns: Vec<Identifier>,
    },
    /// A query (for COPY TO only).
    Query(Box<SelectStatement>),
}

/// The direction of a COPY statement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyDirection {
    /// COPY TO (export).
    To(CopyDestination),
    /// COPY FROM (import).
    From(CopySource),
}

/// The destination for COPY TO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyDestination {
    /// A file path.
    File(String),
    /// Standard output.
    Stdout,
    /// A program to pipe to.
    Program(String),
}

/// The source for COPY FROM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopySource {
    /// A file path.
    File(String),
    /// Standard input.
    Stdin,
    /// A program to pipe from.
    Program(String),
}

/// Options for COPY statements.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CopyOptions {
    /// Data format (CSV, TEXT, BINARY).
    pub format: CopyFormat,
    /// Whether the file has a header row.
    pub header: bool,
    /// Field delimiter (default comma for CSV, tab for TEXT).
    pub delimiter: Option<char>,
    /// NULL string representation.
    pub null_string: Option<String>,
    /// Quote character for CSV.
    pub quote: Option<char>,
    /// Escape character for CSV.
    pub escape: Option<char>,
    /// Encoding of the file.
    pub encoding: Option<String>,
    /// Whether to force quote all columns (COPY TO only).
    pub force_quote: Vec<Identifier>,
    /// Whether to not quote specific columns.
    pub force_not_null: Vec<Identifier>,
}

impl CopyOptions {
    /// Creates default CSV options.
    #[must_use]
    pub fn csv() -> Self {
        Self { format: CopyFormat::Csv, header: true, ..Default::default() }
    }

    /// Creates default text options.
    #[must_use]
    pub fn text() -> Self {
        Self { format: CopyFormat::Text, ..Default::default() }
    }

    /// Creates binary options.
    #[must_use]
    pub fn binary() -> Self {
        Self { format: CopyFormat::Binary, ..Default::default() }
    }

    /// Sets the header flag.
    #[must_use]
    pub const fn with_header(mut self, header: bool) -> Self {
        self.header = header;
        self
    }

    /// Sets the delimiter.
    #[must_use]
    pub const fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Sets the null string.
    #[must_use]
    pub fn with_null(mut self, null_string: impl Into<String>) -> Self {
        self.null_string = Some(null_string.into());
        self
    }
}

/// Data format for COPY.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyFormat {
    /// Comma-separated values.
    #[default]
    Csv,
    /// Tab-separated text.
    Text,
    /// Binary format.
    Binary,
}

impl std::fmt::Display for CopyFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Csv => write!(f, "CSV"),
            Self::Text => write!(f, "TEXT"),
            Self::Binary => write!(f, "BINARY"),
        }
    }
}

/// A SET statement for session variables.
///
/// Sets a session-level configuration parameter.
///
/// # Examples
///
/// ```sql
/// SET search_path TO myschema, public;
/// SET timezone TO 'UTC';
/// SET statement_timeout TO '30s';
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SetSessionStatement {
    /// The variable name.
    pub name: Identifier,
    /// The value to set (None means DEFAULT).
    pub value: Option<SetValue>,
    /// Whether this is SET LOCAL (transaction-scoped) vs SET (session-scoped).
    pub local: bool,
}

impl SetSessionStatement {
    /// Creates a SET statement with a value.
    #[must_use]
    pub fn new(name: impl Into<Identifier>, value: SetValue) -> Self {
        Self { name: name.into(), value: Some(value), local: false }
    }

    /// Creates a SET statement to DEFAULT.
    #[must_use]
    pub fn to_default(name: impl Into<Identifier>) -> Self {
        Self { name: name.into(), value: None, local: false }
    }

    /// Sets the LOCAL flag.
    #[must_use]
    pub const fn local(mut self) -> Self {
        self.local = true;
        self
    }
}

/// The value for a SET statement.
#[derive(Debug, Clone, PartialEq)]
pub enum SetValue {
    /// A single value.
    Single(Expr),
    /// A list of values (e.g., search_path).
    List(Vec<Expr>),
    /// DEFAULT keyword.
    Default,
}

impl SetValue {
    /// Creates a single string value.
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::Single(Expr::string(s.into()))
    }

    /// Creates a single identifier value.
    #[must_use]
    pub fn identifier(name: impl Into<String>) -> Self {
        Self::Single(Expr::Column(QualifiedName::simple(name.into())))
    }
}

impl std::fmt::Display for SetValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(expr) => write!(f, "{expr:?}"),
            Self::List(exprs) => {
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr:?}")?;
                }
                Ok(())
            }
            Self::Default => write!(f, "DEFAULT"),
        }
    }
}

/// A SHOW statement for viewing configuration.
///
/// Displays the current value of a configuration parameter.
///
/// # Examples
///
/// ```sql
/// SHOW search_path;
/// SHOW ALL;
/// SHOW timezone;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowStatement {
    /// The variable to show (None means ALL).
    pub name: Option<Identifier>,
}

impl ShowStatement {
    /// Creates a SHOW ALL statement.
    #[must_use]
    pub fn all() -> Self {
        Self { name: None }
    }

    /// Creates a SHOW statement for a specific variable.
    #[must_use]
    pub fn variable(name: impl Into<Identifier>) -> Self {
        Self { name: Some(name.into()) }
    }
}

/// A RESET statement for resetting variables.
///
/// Resets a configuration parameter to its default value.
///
/// # Examples
///
/// ```sql
/// RESET search_path;
/// RESET ALL;
/// RESET timezone;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResetStatement {
    /// The variable to reset (None means ALL).
    pub name: Option<Identifier>,
}

impl ResetStatement {
    /// Creates a RESET ALL statement.
    #[must_use]
    pub fn all() -> Self {
        Self { name: None }
    }

    /// Creates a RESET statement for a specific variable.
    #[must_use]
    pub fn variable(name: impl Into<Identifier>) -> Self {
        Self { name: Some(name.into()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_builder() {
        let stmt = SelectStatement::new(vec![SelectItem::Wildcard])
            .from(TableRef::table(QualifiedName::simple("users")))
            .where_clause(Expr::column(QualifiedName::simple("id")).eq(Expr::integer(1)));

        assert!(stmt.where_clause.is_some());
        assert_eq!(stmt.from.len(), 1);
    }

    #[test]
    fn insert_builder() {
        let stmt = InsertStatement::values(
            QualifiedName::simple("users"),
            vec![vec![Expr::string("Alice"), Expr::integer(30)]],
        )
        .columns(vec![Identifier::new("name"), Identifier::new("age")]);

        assert_eq!(stmt.columns.len(), 2);
    }

    #[test]
    fn column_def_builder() {
        let col = ColumnDef::new("id", DataType::BigInt).primary_key().not_null();

        assert_eq!(col.constraints.len(), 2);
    }

    #[test]
    fn assignment() {
        let assign = Assignment::new("status", Expr::string("active"));
        assert_eq!(assign.column.name, "status");
    }

    #[test]
    fn call_statement_builder() {
        let stmt = CallStatement::new(
            QualifiedName::qualified("algo", "pageRank"),
            vec![Expr::string("nodes"), Expr::string("edges")],
        )
        .yield_items(vec![YieldItem::column("node"), YieldItem::aliased("score", "rank")])
        .where_clause(Expr::column(QualifiedName::simple("score")).gt(Expr::float(0.1)));

        assert_eq!(stmt.procedure_name.parts.len(), 2);
        assert_eq!(stmt.arguments.len(), 2);
        assert_eq!(stmt.yield_items.len(), 2);
        assert!(stmt.where_clause.is_some());
    }

    #[test]
    fn call_statement_yield_all() {
        let stmt = CallStatement::new(QualifiedName::simple("listProcedures"), vec![]).yield_all();

        assert_eq!(stmt.yield_items.len(), 1);
        assert!(matches!(stmt.yield_items[0], YieldItem::Wildcard));
    }

    #[test]
    fn yield_item_conversions() {
        let from_str: YieldItem = "column_name".into();
        assert!(
            matches!(from_str, YieldItem::Column { name, alias: None } if name.name == "column_name")
        );

        let from_id: YieldItem = Identifier::new("other").into();
        assert!(matches!(from_id, YieldItem::Column { name, alias: None } if name.name == "other"));
    }
}
