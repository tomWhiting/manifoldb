//! Relational plan nodes.
//!
//! This module defines the standard relational algebra operators:
//! `Scan`, `Filter`, `Project`, `Join`, `Aggregate`, `Sort`, `Limit`, etc.

// Allow missing_const_for_fn - const fn with Vec isn't stable
#![allow(clippy::missing_const_for_fn)]

use super::expr::{LogicalExpr, SortOrder};

/// A table scan node.
///
/// Represents reading all rows from a table or subquery.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanNode {
    /// Table name.
    pub table_name: String,
    /// Optional table alias.
    pub alias: Option<String>,
    /// Optional projection (column indices or names to read).
    pub projection: Option<Vec<String>>,
    /// Optional filter to push down to storage.
    pub filter: Option<LogicalExpr>,
}

impl ScanNode {
    /// Creates a new scan node for the given table.
    #[must_use]
    pub fn new(table_name: impl Into<String>) -> Self {
        Self { table_name: table_name.into(), alias: None, projection: None, filter: None }
    }

    /// Sets the table alias.
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Sets the projection columns.
    #[must_use]
    pub fn with_projection(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }

    /// Sets the filter predicate.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Returns the effective table reference name (alias or table name).
    #[must_use]
    pub fn reference_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.table_name)
    }
}

/// A filter node.
///
/// Represents a selection operation that filters rows based on a predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterNode {
    /// The predicate to filter by.
    pub predicate: LogicalExpr,
}

impl FilterNode {
    /// Creates a new filter node.
    #[must_use]
    pub const fn new(predicate: LogicalExpr) -> Self {
        Self { predicate }
    }
}

/// A projection node.
///
/// Represents selecting/computing specific columns from the input.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectNode {
    /// The expressions to project.
    pub exprs: Vec<LogicalExpr>,
}

impl ProjectNode {
    /// Creates a new projection node.
    #[must_use]
    pub const fn new(exprs: Vec<LogicalExpr>) -> Self {
        Self { exprs }
    }
}

/// Join type for join operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// INNER JOIN.
    Inner,
    /// LEFT OUTER JOIN.
    Left,
    /// RIGHT OUTER JOIN.
    Right,
    /// FULL OUTER JOIN.
    Full,
    /// CROSS JOIN.
    Cross,
    /// LEFT SEMI JOIN (returns left rows that have a match).
    LeftSemi,
    /// LEFT ANTI JOIN (returns left rows that don't have a match).
    LeftAnti,
    /// RIGHT SEMI JOIN.
    RightSemi,
    /// RIGHT ANTI JOIN.
    RightAnti,
}

impl std::fmt::Display for JoinType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Inner => "INNER",
            Self::Left => "LEFT OUTER",
            Self::Right => "RIGHT OUTER",
            Self::Full => "FULL OUTER",
            Self::Cross => "CROSS",
            Self::LeftSemi => "LEFT SEMI",
            Self::LeftAnti => "LEFT ANTI",
            Self::RightSemi => "RIGHT SEMI",
            Self::RightAnti => "RIGHT ANTI",
        };
        write!(f, "{name}")
    }
}

/// A join node.
///
/// Represents joining two relations based on a condition.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinNode {
    /// The type of join.
    pub join_type: JoinType,
    /// The join condition (for non-cross joins).
    pub condition: Option<LogicalExpr>,
    /// Columns to use for USING clause (alternative to condition).
    pub using_columns: Vec<String>,
}

impl JoinNode {
    /// Creates a new inner join.
    #[must_use]
    pub fn inner(condition: LogicalExpr) -> Self {
        Self { join_type: JoinType::Inner, condition: Some(condition), using_columns: vec![] }
    }

    /// Creates a new left outer join.
    #[must_use]
    pub fn left(condition: LogicalExpr) -> Self {
        Self { join_type: JoinType::Left, condition: Some(condition), using_columns: vec![] }
    }

    /// Creates a new right outer join.
    #[must_use]
    pub fn right(condition: LogicalExpr) -> Self {
        Self { join_type: JoinType::Right, condition: Some(condition), using_columns: vec![] }
    }

    /// Creates a new full outer join.
    #[must_use]
    pub fn full(condition: LogicalExpr) -> Self {
        Self { join_type: JoinType::Full, condition: Some(condition), using_columns: vec![] }
    }

    /// Creates a new cross join.
    #[must_use]
    pub const fn cross() -> Self {
        Self { join_type: JoinType::Cross, condition: None, using_columns: vec![] }
    }

    /// Creates a join with USING clause.
    #[must_use]
    pub fn using(join_type: JoinType, columns: Vec<String>) -> Self {
        Self { join_type, condition: None, using_columns: columns }
    }
}

/// An aggregate node.
///
/// Represents a grouping/aggregation operation.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregateNode {
    /// GROUP BY expressions.
    pub group_by: Vec<LogicalExpr>,
    /// Aggregate expressions.
    pub aggregates: Vec<LogicalExpr>,
    /// Optional HAVING clause.
    pub having: Option<LogicalExpr>,
}

impl AggregateNode {
    /// Creates a new aggregate node.
    #[must_use]
    pub const fn new(group_by: Vec<LogicalExpr>, aggregates: Vec<LogicalExpr>) -> Self {
        Self { group_by, aggregates, having: None }
    }

    /// Sets the HAVING clause.
    #[must_use]
    pub fn with_having(mut self, having: LogicalExpr) -> Self {
        self.having = Some(having);
        self
    }

    /// Returns true if this is a simple aggregation (no GROUP BY).
    #[must_use]
    pub fn is_simple(&self) -> bool {
        self.group_by.is_empty()
    }
}

/// A sort node.
///
/// Represents an ORDER BY operation.
#[derive(Debug, Clone, PartialEq)]
pub struct SortNode {
    /// Sort specifications.
    pub order_by: Vec<SortOrder>,
}

impl SortNode {
    /// Creates a new sort node.
    #[must_use]
    pub const fn new(order_by: Vec<SortOrder>) -> Self {
        Self { order_by }
    }
}

/// A limit node.
///
/// Represents LIMIT and OFFSET clauses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitNode {
    /// Maximum number of rows to return.
    pub limit: Option<usize>,
    /// Number of rows to skip.
    pub offset: Option<usize>,
}

impl LimitNode {
    /// Creates a limit-only node.
    #[must_use]
    pub const fn limit(n: usize) -> Self {
        Self { limit: Some(n), offset: None }
    }

    /// Creates an offset-only node.
    #[must_use]
    pub const fn offset(n: usize) -> Self {
        Self { limit: None, offset: Some(n) }
    }

    /// Creates a limit with offset node.
    #[must_use]
    pub const fn limit_offset(limit: usize, offset: usize) -> Self {
        Self { limit: Some(limit), offset: Some(offset) }
    }
}

/// A distinct node.
///
/// Represents SELECT DISTINCT.
#[derive(Debug, Clone, PartialEq)]
pub struct DistinctNode {
    /// Optional columns to distinct on (for DISTINCT ON).
    pub on_columns: Option<Vec<LogicalExpr>>,
}

impl DistinctNode {
    /// Creates a simple DISTINCT node.
    #[must_use]
    pub const fn all() -> Self {
        Self { on_columns: None }
    }

    /// Creates a DISTINCT ON node.
    #[must_use]
    pub const fn on(columns: Vec<LogicalExpr>) -> Self {
        Self { on_columns: Some(columns) }
    }
}

/// Set operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpType {
    /// UNION.
    Union,
    /// UNION ALL.
    UnionAll,
    /// INTERSECT.
    Intersect,
    /// INTERSECT ALL.
    IntersectAll,
    /// EXCEPT.
    Except,
    /// EXCEPT ALL.
    ExceptAll,
}

impl std::fmt::Display for SetOpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Union => "UNION",
            Self::UnionAll => "UNION ALL",
            Self::Intersect => "INTERSECT",
            Self::IntersectAll => "INTERSECT ALL",
            Self::Except => "EXCEPT",
            Self::ExceptAll => "EXCEPT ALL",
        };
        write!(f, "{name}")
    }
}

/// A set operation node.
///
/// Represents UNION, INTERSECT, or EXCEPT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetOpNode {
    /// The type of set operation.
    pub op_type: SetOpType,
}

impl SetOpNode {
    /// Creates a new set operation node.
    #[must_use]
    pub const fn new(op_type: SetOpType) -> Self {
        Self { op_type }
    }
}

/// A union node (special case for UNION ALL).
///
/// Represents concatenating multiple inputs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnionNode {
    /// Whether to preserve duplicates (UNION ALL vs UNION).
    pub all: bool,
}

impl UnionNode {
    /// Creates a UNION node.
    #[must_use]
    pub const fn distinct() -> Self {
        Self { all: false }
    }

    /// Creates a UNION ALL node.
    #[must_use]
    pub const fn all() -> Self {
        Self { all: true }
    }
}

/// A window node.
///
/// Represents window functions (ROW_NUMBER, RANK, DENSE_RANK, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct WindowNode {
    /// Window expressions paired with their output column names.
    /// Each element is (window_expr, column_alias).
    pub window_exprs: Vec<(LogicalExpr, String)>,
}

impl WindowNode {
    /// Creates a new window node.
    #[must_use]
    pub const fn new(window_exprs: Vec<(LogicalExpr, String)>) -> Self {
        Self { window_exprs }
    }

    /// Adds a window expression with an alias.
    #[must_use]
    pub fn add_window_expr(mut self, expr: LogicalExpr, alias: impl Into<String>) -> Self {
        self.window_exprs.push((expr, alias.into()));
        self
    }
}

/// A values node.
///
/// Represents inline row data (VALUES clause).
#[derive(Debug, Clone, PartialEq)]
pub struct ValuesNode {
    /// The rows of values.
    pub rows: Vec<Vec<LogicalExpr>>,
    /// Optional column names.
    pub column_names: Option<Vec<String>>,
}

impl ValuesNode {
    /// Creates a new values node.
    #[must_use]
    pub const fn new(rows: Vec<Vec<LogicalExpr>>) -> Self {
        Self { rows, column_names: None }
    }

    /// Sets column names for the values.
    #[must_use]
    pub fn with_column_names(mut self, names: Vec<String>) -> Self {
        self.column_names = Some(names);
        self
    }
}

/// An unwind node.
///
/// Represents unwinding a list into individual rows.
/// This is the Cypher UNWIND clause equivalent.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwindNode {
    /// The expression that produces a list to unwind.
    pub list_expr: LogicalExpr,
    /// The variable name to bind each unwound element to.
    pub alias: String,
}

impl UnwindNode {
    /// Creates a new unwind node.
    #[must_use]
    pub fn new(list_expr: LogicalExpr, alias: impl Into<String>) -> Self {
        Self { list_expr, alias: alias.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_node() {
        let scan = ScanNode::new("users")
            .with_alias("u")
            .with_projection(vec!["id".to_string(), "name".to_string()]);

        assert_eq!(scan.table_name, "users");
        assert_eq!(scan.reference_name(), "u");
        assert_eq!(scan.projection.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn join_types() {
        let inner = JoinNode::inner(LogicalExpr::column("a").eq(LogicalExpr::column("b")));
        assert_eq!(inner.join_type, JoinType::Inner);

        let cross = JoinNode::cross();
        assert!(cross.condition.is_none());

        let using = JoinNode::using(JoinType::Inner, vec!["id".to_string()]);
        assert_eq!(using.using_columns, vec!["id"]);
    }

    #[test]
    fn aggregate_node() {
        let agg = AggregateNode::new(
            vec![LogicalExpr::column("category")],
            vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
        );

        assert!(!agg.is_simple());
        assert_eq!(agg.group_by.len(), 1);
    }

    #[test]
    fn limit_offset() {
        let limit = LimitNode::limit(10);
        assert_eq!(limit.limit, Some(10));
        assert!(limit.offset.is_none());

        let both = LimitNode::limit_offset(20, 10);
        assert_eq!(both.limit, Some(20));
        assert_eq!(both.offset, Some(10));
    }

    #[test]
    fn set_op_display() {
        assert_eq!(SetOpType::Union.to_string(), "UNION");
        assert_eq!(SetOpType::UnionAll.to_string(), "UNION ALL");
        assert_eq!(SetOpType::Intersect.to_string(), "INTERSECT");
    }

    #[test]
    fn values_node() {
        let values = ValuesNode::new(vec![
            vec![LogicalExpr::integer(1), LogicalExpr::string("a")],
            vec![LogicalExpr::integer(2), LogicalExpr::string("b")],
        ])
        .with_column_names(vec!["id".to_string(), "name".to_string()]);

        assert_eq!(values.rows.len(), 2);
        assert_eq!(values.column_names.as_ref().unwrap().len(), 2);
    }
}
