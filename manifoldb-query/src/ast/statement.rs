//! Statement AST types.
//!
//! This module defines the top-level statement types for parsed queries.

use super::expr::{Expr, Identifier, OrderByExpr, QualifiedName};
use super::pattern::GraphPattern;

/// A parsed SQL statement.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum Statement {
    /// SELECT statement.
    Select(SelectStatement),
    /// INSERT statement.
    Insert(InsertStatement),
    /// UPDATE statement.
    Update(UpdateStatement),
    /// DELETE statement.
    Delete(DeleteStatement),
    /// CREATE TABLE statement.
    CreateTable(CreateTableStatement),
    /// CREATE INDEX statement.
    CreateIndex(CreateIndexStatement),
    /// DROP TABLE statement.
    DropTable(DropTableStatement),
    /// DROP INDEX statement.
    DropIndex(DropIndexStatement),
    /// EXPLAIN statement.
    Explain(Box<Statement>),
}

/// A SELECT statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    /// Whether DISTINCT is specified.
    pub distinct: bool,
    /// The projection (SELECT list).
    pub projection: Vec<SelectItem>,
    /// The FROM clause.
    pub from: Vec<TableRef>,
    /// Optional MATCH clause for graph patterns.
    pub match_clause: Option<GraphPattern>,
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
            distinct: false,
            projection,
            from: vec![],
            match_clause: None,
            where_clause: None,
            group_by: vec![],
            having: None,
            order_by: vec![],
            limit: None,
            offset: None,
            set_op: None,
        }
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
        Self::Expr {
            expr,
            alias: Some(alias.into()),
        }
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
        Self::Table {
            name: name.into(),
            alias: None,
        }
    }

    /// Creates an aliased table reference.
    #[must_use]
    pub fn aliased(name: impl Into<QualifiedName>, alias: impl Into<TableAlias>) -> Self {
        Self::Table {
            name: name.into(),
            alias: Some(alias.into()),
        }
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
        Self {
            name: name.into(),
            columns: vec![],
        }
    }

    /// Creates an alias with column names.
    #[must_use]
    pub fn with_columns(name: impl Into<Identifier>, columns: Vec<Identifier>) -> Self {
        Self {
            name: name.into(),
            columns,
        }
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
        Self {
            left,
            right,
            join_type: JoinType::Inner,
            condition: JoinCondition::On(on),
        }
    }

    /// Creates a left outer join.
    #[must_use]
    pub const fn left_join(left: TableRef, right: TableRef, on: Expr) -> Self {
        Self {
            left,
            right,
            join_type: JoinType::LeftOuter,
            condition: JoinCondition::On(on),
        }
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
        Self {
            column: column.into(),
            value,
        }
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
        Self {
            if_not_exists: false,
            name: name.into(),
            columns,
            constraints: vec![],
        }
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
        Self {
            name: name.into(),
            data_type,
            constraints: vec![],
        }
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
        Self {
            expr,
            asc: None,
            nulls_first: None,
            opclass: None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_builder() {
        let stmt = SelectStatement::new(vec![SelectItem::Wildcard])
            .from(TableRef::table(QualifiedName::simple("users")))
            .where_clause(
                Expr::column(QualifiedName::simple("id")).eq(Expr::integer(1)),
            );

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
        let col = ColumnDef::new("id", DataType::BigInt)
            .primary_key()
            .not_null();

        assert_eq!(col.constraints.len(), 2);
    }

    #[test]
    fn assignment() {
        let assign = Assignment::new("status", Expr::string("active"));
        assert_eq!(assign.column.name, "status");
    }
}
