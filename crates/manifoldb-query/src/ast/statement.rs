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
    /// MATCH statement (Cypher-style graph query).
    Match(Box<MatchStatement>),
    /// Cypher CREATE statement for creating nodes and relationships.
    Create(Box<CreateGraphStatement>),
    /// Cypher MERGE statement for upserting nodes and relationships.
    Merge(Box<MergeGraphStatement>),
    /// EXPLAIN statement.
    Explain(Box<Statement>),
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
}
