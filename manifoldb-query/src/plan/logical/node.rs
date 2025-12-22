//! Logical plan node.
//!
//! This module defines the main `LogicalPlan` enum that represents
//! the tree structure of a logical query plan.

// Allow matching arms with identical bodies - intentional for grouping
#![allow(clippy::match_same_arms)]
// Allow long Display impl
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]
// Allow use_self in builders
#![allow(clippy::use_self)]
// Allow missing_const_for_fn - const fn with Vec isn't stable
#![allow(clippy::missing_const_for_fn)]
// Allow unused_self in helper methods
#![allow(clippy::unused_self)]

use std::fmt;

use super::expr::{LogicalExpr, SortOrder};
use super::graph::{ExpandNode, PathScanNode};
use super::relational::{
    AggregateNode, DistinctNode, FilterNode, JoinNode, LimitNode, ProjectNode, ScanNode, SetOpNode,
    SortNode, UnionNode, ValuesNode,
};
use super::vector::{AnnSearchNode, VectorDistanceNode};

/// A logical query plan.
///
/// This is a tree structure where each node represents an operation,
/// and children represent inputs to that operation.
///
/// Large node types are boxed to reduce enum size overhead. This improves
/// memory efficiency when many `LogicalPlan` instances are created during
/// query planning.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    // ========== Leaf Nodes (no inputs) ==========
    /// Table scan (boxed - 120 bytes unboxed).
    Scan(Box<ScanNode>),

    /// Inline values (VALUES clause).
    Values(ValuesNode),

    /// Empty relation (no rows).
    Empty {
        /// Column names for the empty relation.
        columns: Vec<String>,
    },

    // ========== Unary Nodes (single input) ==========
    /// Filter (WHERE clause).
    Filter {
        /// The filter node.
        node: FilterNode,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Projection (SELECT list).
    Project {
        /// The projection node.
        node: ProjectNode,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Aggregation (GROUP BY) (boxed node - 96 bytes unboxed).
    Aggregate {
        /// The aggregate node.
        node: Box<AggregateNode>,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Sort (ORDER BY).
    Sort {
        /// The sort node.
        node: SortNode,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Limit and/or offset.
    Limit {
        /// The limit node.
        node: LimitNode,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Distinct (SELECT DISTINCT).
    Distinct {
        /// The distinct node.
        node: DistinctNode,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    /// Subquery alias.
    Alias {
        /// The alias name.
        alias: String,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    // ========== Binary Nodes (two inputs) ==========
    /// Join two relations (boxed node - 80 bytes unboxed).
    Join {
        /// The join node.
        node: Box<JoinNode>,
        /// The left input.
        left: Box<LogicalPlan>,
        /// The right input.
        right: Box<LogicalPlan>,
    },

    /// Set operation (UNION, INTERSECT, EXCEPT).
    SetOp {
        /// The set operation node.
        node: SetOpNode,
        /// The left input.
        left: Box<LogicalPlan>,
        /// The right input.
        right: Box<LogicalPlan>,
    },

    // ========== N-ary Nodes (multiple inputs) ==========
    /// Union of multiple inputs.
    Union {
        /// The union node.
        node: UnionNode,
        /// The input plans.
        inputs: Vec<LogicalPlan>,
    },

    // ========== Graph Nodes ==========
    /// Graph edge expansion (boxed node - 248 bytes unboxed).
    Expand {
        /// The expand node.
        node: Box<ExpandNode>,
        /// The input plan (provides source nodes).
        input: Box<LogicalPlan>,
    },

    /// Path pattern scan (boxed node - 80 bytes unboxed).
    PathScan {
        /// The path scan node.
        node: Box<PathScanNode>,
        /// The input plan (provides starting nodes).
        input: Box<LogicalPlan>,
    },

    // ========== Vector Nodes ==========
    /// Approximate nearest neighbor search (boxed node - 208 bytes unboxed).
    AnnSearch {
        /// The ANN search node.
        node: Box<AnnSearchNode>,
        /// The input plan (table to search).
        input: Box<LogicalPlan>,
    },

    /// Vector distance computation (boxed node - 128 bytes unboxed).
    VectorDistance {
        /// The distance node.
        node: Box<VectorDistanceNode>,
        /// The input plan.
        input: Box<LogicalPlan>,
    },

    // ========== DML Nodes ==========
    /// INSERT operation.
    Insert {
        /// Target table name.
        table: String,
        /// Column names.
        columns: Vec<String>,
        /// The input providing rows to insert.
        input: Box<LogicalPlan>,
        /// Whether to return inserted rows.
        returning: Vec<LogicalExpr>,
    },

    /// UPDATE operation.
    Update {
        /// Target table name.
        table: String,
        /// Assignments (column, value).
        assignments: Vec<(String, LogicalExpr)>,
        /// Filter for rows to update.
        filter: Option<LogicalExpr>,
        /// Whether to return updated rows.
        returning: Vec<LogicalExpr>,
    },

    /// DELETE operation.
    Delete {
        /// Target table name.
        table: String,
        /// Filter for rows to delete.
        filter: Option<LogicalExpr>,
        /// Whether to return deleted rows.
        returning: Vec<LogicalExpr>,
    },
}

impl LogicalPlan {
    // ========== Constructors ==========

    /// Creates a table scan.
    #[must_use]
    pub fn scan(table: impl Into<String>) -> Self {
        Self::Scan(Box::new(ScanNode::new(table)))
    }

    /// Creates a table scan with alias.
    #[must_use]
    pub fn scan_aliased(table: impl Into<String>, alias: impl Into<String>) -> Self {
        Self::Scan(Box::new(ScanNode::new(table).with_alias(alias)))
    }

    /// Creates an empty relation.
    #[must_use]
    pub fn empty(columns: Vec<String>) -> Self {
        Self::Empty { columns }
    }

    /// Creates a values node.
    #[must_use]
    pub fn values(rows: Vec<Vec<LogicalExpr>>) -> Self {
        Self::Values(ValuesNode::new(rows))
    }

    // ========== Builder Methods ==========

    /// Adds a filter to this plan.
    #[must_use]
    pub fn filter(self, predicate: LogicalExpr) -> Self {
        Self::Filter { node: FilterNode::new(predicate), input: Box::new(self) }
    }

    /// Adds a projection to this plan.
    #[must_use]
    pub fn project(self, exprs: Vec<LogicalExpr>) -> Self {
        Self::Project { node: ProjectNode::new(exprs), input: Box::new(self) }
    }

    /// Adds an aggregation to this plan.
    #[must_use]
    pub fn aggregate(self, group_by: Vec<LogicalExpr>, aggregates: Vec<LogicalExpr>) -> Self {
        Self::Aggregate { node: Box::new(AggregateNode::new(group_by, aggregates)), input: Box::new(self) }
    }

    /// Adds a sort to this plan.
    #[must_use]
    pub fn sort(self, order_by: Vec<SortOrder>) -> Self {
        Self::Sort { node: SortNode::new(order_by), input: Box::new(self) }
    }

    /// Adds a limit to this plan.
    #[must_use]
    pub fn limit(self, n: usize) -> Self {
        Self::Limit { node: LimitNode::limit(n), input: Box::new(self) }
    }

    /// Adds an offset to this plan.
    #[must_use]
    pub fn offset(self, n: usize) -> Self {
        Self::Limit { node: LimitNode::offset(n), input: Box::new(self) }
    }

    /// Adds limit and offset to this plan.
    #[must_use]
    pub fn limit_offset(self, limit: usize, offset: usize) -> Self {
        Self::Limit { node: LimitNode::limit_offset(limit, offset), input: Box::new(self) }
    }

    /// Adds DISTINCT to this plan.
    #[must_use]
    pub fn distinct(self) -> Self {
        Self::Distinct { node: DistinctNode::all(), input: Box::new(self) }
    }

    /// Adds an alias to this plan.
    #[must_use]
    pub fn alias(self, name: impl Into<String>) -> Self {
        Self::Alias { alias: name.into(), input: Box::new(self) }
    }

    /// Creates an inner join with another plan.
    #[must_use]
    pub fn inner_join(self, right: LogicalPlan, on: LogicalExpr) -> Self {
        Self::Join { node: Box::new(JoinNode::inner(on)), left: Box::new(self), right: Box::new(right) }
    }

    /// Creates a left outer join with another plan.
    #[must_use]
    pub fn left_join(self, right: LogicalPlan, on: LogicalExpr) -> Self {
        Self::Join { node: Box::new(JoinNode::left(on)), left: Box::new(self), right: Box::new(right) }
    }

    /// Creates a cross join with another plan.
    #[must_use]
    pub fn cross_join(self, right: LogicalPlan) -> Self {
        Self::Join { node: Box::new(JoinNode::cross()), left: Box::new(self), right: Box::new(right) }
    }

    /// Creates a UNION ALL with another plan.
    #[must_use]
    pub fn union_all(self, other: LogicalPlan) -> Self {
        Self::Union { node: UnionNode::all(), inputs: vec![self, other] }
    }

    /// Adds a graph expand operation.
    #[must_use]
    pub fn expand(self, node: ExpandNode) -> Self {
        Self::Expand { node: Box::new(node), input: Box::new(self) }
    }

    /// Adds an ANN search operation.
    #[must_use]
    pub fn ann_search(self, node: AnnSearchNode) -> Self {
        Self::AnnSearch { node: Box::new(node), input: Box::new(self) }
    }

    // ========== Utility Methods ==========

    /// Returns the children of this plan node.
    #[must_use]
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            // Leaf nodes
            Self::Scan(_) | Self::Values(_) | Self::Empty { .. } => vec![],

            // Unary nodes
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Aggregate { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::Distinct { input, .. }
            | Self::Alias { input, .. }
            | Self::Expand { input, .. }
            | Self::PathScan { input, .. }
            | Self::AnnSearch { input, .. }
            | Self::VectorDistance { input, .. }
            | Self::Insert { input, .. } => vec![input.as_ref()],

            // Binary nodes
            Self::Join { left, right, .. } | Self::SetOp { left, right, .. } => {
                vec![left.as_ref(), right.as_ref()]
            }

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter().collect(),

            // DML without input
            Self::Update { .. } | Self::Delete { .. } => vec![],
        }
    }

    /// Returns the mutable children of this plan node.
    #[must_use]
    pub fn children_mut(&mut self) -> Vec<&mut LogicalPlan> {
        match self {
            // Leaf nodes
            Self::Scan(_) | Self::Values(_) | Self::Empty { .. } => vec![],

            // Unary nodes
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Aggregate { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::Distinct { input, .. }
            | Self::Alias { input, .. }
            | Self::Expand { input, .. }
            | Self::PathScan { input, .. }
            | Self::AnnSearch { input, .. }
            | Self::VectorDistance { input, .. }
            | Self::Insert { input, .. } => vec![input.as_mut()],

            // Binary nodes
            Self::Join { left, right, .. } | Self::SetOp { left, right, .. } => {
                vec![left.as_mut(), right.as_mut()]
            }

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter_mut().collect(),

            // DML without input
            Self::Update { .. } | Self::Delete { .. } => vec![],
        }
    }

    /// Returns true if this is a leaf node (no children).
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Scan(_) | Self::Values(_) | Self::Empty { .. })
    }

    /// Returns the node type name (for display/debugging).
    #[must_use]
    pub fn node_type(&self) -> &'static str {
        match self {
            Self::Scan(_) => "Scan",
            Self::Values(_) => "Values",
            Self::Empty { .. } => "Empty",
            Self::Filter { .. } => "Filter",
            Self::Project { .. } => "Project",
            Self::Aggregate { .. } => "Aggregate",
            Self::Sort { .. } => "Sort",
            Self::Limit { .. } => "Limit",
            Self::Distinct { .. } => "Distinct",
            Self::Alias { .. } => "Alias",
            Self::Join { .. } => "Join",
            Self::SetOp { .. } => "SetOp",
            Self::Union { .. } => "Union",
            Self::Expand { .. } => "Expand",
            Self::PathScan { .. } => "PathScan",
            Self::AnnSearch { .. } => "AnnSearch",
            Self::VectorDistance { .. } => "VectorDistance",
            Self::Insert { .. } => "Insert",
            Self::Update { .. } => "Update",
            Self::Delete { .. } => "Delete",
        }
    }

    /// Pretty prints the plan as a tree.
    #[must_use]
    pub fn display_tree(&self) -> DisplayTree<'_> {
        DisplayTree { plan: self }
    }
}

/// Helper for tree-style plan display.
pub struct DisplayTree<'a> {
    plan: &'a LogicalPlan,
}

impl fmt::Display for DisplayTree<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_node(f, self.plan, "", true)
    }
}

impl DisplayTree<'_> {
    fn fmt_node(
        &self,
        f: &mut fmt::Formatter<'_>,
        plan: &LogicalPlan,
        prefix: &str,
        is_last: bool,
    ) -> fmt::Result {
        let connector = if is_last { "└── " } else { "├── " };

        write!(f, "{prefix}{connector}")?;
        self.fmt_node_content(f, plan)?;
        writeln!(f)?;

        let children = plan.children();
        let new_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });

        for (i, child) in children.iter().enumerate() {
            self.fmt_node(f, child, &new_prefix, i == children.len() - 1)?;
        }

        Ok(())
    }

    fn fmt_node_content(&self, f: &mut fmt::Formatter<'_>, plan: &LogicalPlan) -> fmt::Result {
        match plan {
            LogicalPlan::Scan(node) => {
                write!(f, "Scan: {}", node.table_name)?;
                if let Some(alias) = &node.alias {
                    write!(f, " AS {alias}")?;
                }
                if let Some(filter) = &node.filter {
                    write!(f, " [filter: {filter}]")?;
                }
            }
            LogicalPlan::Values(node) => {
                write!(f, "Values: {} rows", node.rows.len())?;
            }
            LogicalPlan::Empty { columns } => {
                write!(f, "Empty: {} columns", columns.len())?;
            }
            LogicalPlan::Filter { node, .. } => {
                write!(f, "Filter: {}", node.predicate)?;
            }
            LogicalPlan::Project { node, .. } => {
                write!(f, "Project: ")?;
                for (i, expr) in node.exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr}")?;
                }
            }
            LogicalPlan::Aggregate { node, .. } => {
                write!(f, "Aggregate: ")?;
                if !node.group_by.is_empty() {
                    write!(f, "GROUP BY ")?;
                    for (i, expr) in node.group_by.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{expr}")?;
                    }
                    write!(f, " ")?;
                }
                for (i, agg) in node.aggregates.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{agg}")?;
                }
            }
            LogicalPlan::Sort { node, .. } => {
                write!(f, "Sort: ")?;
                for (i, order) in node.order_by.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{order}")?;
                }
            }
            LogicalPlan::Limit { node, .. } => {
                write!(f, "Limit: ")?;
                if let Some(n) = node.limit {
                    write!(f, "{n}")?;
                }
                if let Some(off) = node.offset {
                    write!(f, " OFFSET {off}")?;
                }
            }
            LogicalPlan::Distinct { node, .. } => {
                write!(f, "Distinct")?;
                if let Some(cols) = &node.on_columns {
                    write!(f, " ON ")?;
                    for (i, col) in cols.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{col}")?;
                    }
                }
            }
            LogicalPlan::Alias { alias, .. } => {
                write!(f, "Alias: {alias}")?;
            }
            LogicalPlan::Join { node, .. } => {
                write!(f, "Join: {} JOIN", node.join_type)?;
                if let Some(cond) = &node.condition {
                    write!(f, " ON {cond}")?;
                }
                if !node.using_columns.is_empty() {
                    write!(f, " USING ({})", node.using_columns.join(", "))?;
                }
            }
            LogicalPlan::SetOp { node, .. } => {
                write!(f, "SetOp: {}", node.op_type)?;
            }
            LogicalPlan::Union { node, inputs, .. } => {
                write!(f, "Union{}: {} inputs", if node.all { " All" } else { "" }, inputs.len())?;
            }
            LogicalPlan::Expand { node, .. } => {
                write!(f, "Expand: ({}){}({})", node.src_var, node.direction, node.dst_var)?;
                if !node.edge_types.is_empty() {
                    write!(f, " [types: {}]", node.edge_types.join("|"))?;
                }
            }
            LogicalPlan::PathScan { node, .. } => {
                write!(f, "PathScan: {} steps", node.steps.len())?;
            }
            LogicalPlan::AnnSearch { node, .. } => {
                write!(
                    f,
                    "AnnSearch: {} {} k={}",
                    node.vector_column,
                    node.metric.operator(),
                    node.k
                )?;
            }
            LogicalPlan::VectorDistance { node, .. } => {
                write!(f, "VectorDistance: {} {}", node.metric.operator(), node.left)?;
            }
            LogicalPlan::Insert { table, columns, .. } => {
                write!(f, "Insert: {table}")?;
                if !columns.is_empty() {
                    write!(f, " ({})", columns.join(", "))?;
                }
            }
            LogicalPlan::Update { table, assignments, .. } => {
                write!(f, "Update: {table} SET ")?;
                for (i, (col, _)) in assignments.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{col}")?;
                }
            }
            LogicalPlan::Delete { table, filter, .. } => {
                write!(f, "Delete: {table}")?;
                if let Some(filt) = filter {
                    write!(f, " WHERE {filt}")?;
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_tree())
    }
}

#[cfg(test)]
mod tests {
    use super::super::expr::LogicalExpr;
    use super::*;

    #[test]
    fn simple_scan() {
        let plan = LogicalPlan::scan("users");
        assert_eq!(plan.node_type(), "Scan");
        assert!(plan.is_leaf());
        assert!(plan.children().is_empty());
    }

    #[test]
    fn filter_on_scan() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        assert_eq!(plan.node_type(), "Filter");
        assert!(!plan.is_leaf());
        assert_eq!(plan.children().len(), 1);
    }

    #[test]
    fn complex_query() {
        // SELECT name, COUNT(*) FROM users WHERE active = true GROUP BY name ORDER BY name LIMIT 10
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)))
            .aggregate(
                vec![LogicalExpr::column("name")],
                vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
            )
            .sort(vec![SortOrder::asc(LogicalExpr::column("name"))])
            .limit(10);

        assert_eq!(plan.node_type(), "Limit");

        // Verify the tree structure
        let children: Vec<_> = plan.children();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].node_type(), "Sort");
    }

    #[test]
    fn join_query() {
        let users = LogicalPlan::scan("users");
        let orders = LogicalPlan::scan("orders");

        let plan = users.inner_join(
            orders,
            LogicalExpr::qualified_column("users", "id")
                .eq(LogicalExpr::qualified_column("orders", "user_id")),
        );

        assert_eq!(plan.node_type(), "Join");
        assert_eq!(plan.children().len(), 2);
    }

    #[test]
    fn display_tree() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)))
            .project(vec![LogicalExpr::column("id"), LogicalExpr::column("name")])
            .limit(10);

        let output = format!("{plan}");
        assert!(output.contains("Limit"));
        assert!(output.contains("Project"));
        assert!(output.contains("Filter"));
        assert!(output.contains("Scan"));
    }
}
