//! Physical plan node types.
//!
//! This module defines the concrete execution operators that make up
//! a physical query plan.

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

use crate::ast::DistanceMetric;
use crate::ast::WindowFunction;
use crate::plan::logical::{
    CreateCollectionNode, CreateIndexNode, CreateTableNode, DropCollectionNode, DropIndexNode,
    DropTableNode, ExpandDirection, ExpandLength, GraphCreateNode, GraphMergeNode, JoinType,
    LogicalExpr, ProcedureCallNode, SetOpType, SortOrder,
};

use super::cost::Cost;

/// A physical query plan.
///
/// This is a tree structure where each node represents a concrete
/// execution operator with a specific algorithm choice.
///
/// Large node types are boxed to reduce enum size overhead. This improves
/// memory efficiency when many `PhysicalPlan` instances are created during
/// query optimization and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalPlan {
    // ========== Scan Operations ==========
    /// Full table scan (boxed - 136 bytes unboxed).
    FullScan(Box<FullScanNode>),

    /// Index point lookup (boxed - 136 bytes unboxed).
    IndexScan(Box<IndexScanNode>),

    /// Index range scan (boxed - 216 bytes unboxed).
    IndexRangeScan(Box<IndexRangeScanNode>),

    /// Values source (inline data).
    Values {
        /// The rows of values.
        rows: Vec<Vec<LogicalExpr>>,
        /// Estimated cost.
        cost: Cost,
    },

    /// Empty source (no rows).
    Empty {
        /// Column names.
        columns: Vec<String>,
    },

    // ========== Unary Operators ==========
    /// Filter operator.
    Filter {
        /// Filter configuration.
        node: FilterExecNode,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Projection operator.
    Project {
        /// Projection configuration.
        node: ProjectExecNode,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Sort operator.
    Sort {
        /// Sort configuration.
        node: SortExecNode,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Limit/offset operator.
    Limit {
        /// Limit configuration.
        node: LimitExecNode,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Distinct operator (hash-based).
    HashDistinct {
        /// Columns to distinct on.
        on_columns: Option<Vec<LogicalExpr>>,
        /// Estimated cost.
        cost: Cost,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Window operator for ROW_NUMBER, RANK, DENSE_RANK.
    Window {
        /// Window configuration.
        node: Box<WindowExecNode>,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Hash-based aggregation (boxed node - 112 bytes unboxed).
    HashAggregate {
        /// Aggregation configuration.
        node: Box<HashAggregateNode>,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Sort-merge based aggregation (boxed node - 112 bytes unboxed).
    SortMergeAggregate {
        /// Aggregation configuration.
        node: Box<SortMergeAggregateNode>,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    // ========== Binary Operators (Joins) ==========
    /// Nested loop join.
    NestedLoopJoin {
        /// Join configuration.
        node: NestedLoopJoinNode,
        /// Left input (outer).
        left: Box<PhysicalPlan>,
        /// Right input (inner).
        right: Box<PhysicalPlan>,
    },

    /// Hash join (boxed node - 120 bytes unboxed).
    HashJoin {
        /// Join configuration.
        node: Box<HashJoinNode>,
        /// Build side input.
        build: Box<PhysicalPlan>,
        /// Probe side input.
        probe: Box<PhysicalPlan>,
    },

    /// Sort-merge join (boxed node - 120 bytes unboxed).
    MergeJoin {
        /// Join configuration.
        node: Box<MergeJoinNode>,
        /// Left input (must be sorted).
        left: Box<PhysicalPlan>,
        /// Right input (must be sorted).
        right: Box<PhysicalPlan>,
    },

    // ========== Set Operations ==========
    /// Set operation (UNION, INTERSECT, EXCEPT).
    SetOp {
        /// The type of set operation.
        op_type: SetOpType,
        /// Estimated cost.
        cost: Cost,
        /// Left input.
        left: Box<PhysicalPlan>,
        /// Right input.
        right: Box<PhysicalPlan>,
    },

    /// Union of multiple inputs.
    Union {
        /// Whether to preserve duplicates.
        all: bool,
        /// Estimated cost.
        cost: Cost,
        /// Input plans.
        inputs: Vec<PhysicalPlan>,
    },

    // ========== Unwind Operation ==========
    /// Unwind operator (expands list to rows).
    Unwind {
        /// Unwind configuration.
        node: UnwindExecNode,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    // ========== Recursive Operation ==========
    /// Recursive CTE operator (WITH RECURSIVE).
    ///
    /// Iteratively executes the recursive query until a fixed point is reached.
    RecursiveCTE {
        /// Recursive CTE configuration.
        node: Box<RecursiveCTEExecNode>,
        /// The initial (base case) query plan.
        initial: Box<PhysicalPlan>,
        /// The recursive query plan.
        recursive: Box<PhysicalPlan>,
    },

    // ========== Vector Operations ==========
    /// HNSW-based approximate nearest neighbor search (boxed node - 184 bytes unboxed).
    HnswSearch {
        /// HNSW search configuration.
        node: Box<HnswSearchNode>,
        /// Input plan (source table).
        input: Box<PhysicalPlan>,
    },

    /// Brute-force vector search (boxed node - 176 bytes unboxed).
    BruteForceSearch {
        /// Brute-force search configuration.
        node: Box<BruteForceSearchNode>,
        /// Input plan (source table).
        input: Box<PhysicalPlan>,
    },

    /// Hybrid vector search combining multiple search types.
    HybridSearch {
        /// Hybrid search configuration.
        node: Box<HybridSearchNode>,
        /// Input plan (source table).
        input: Box<PhysicalPlan>,
    },

    // ========== Graph Operations ==========
    /// Graph edge expansion (boxed node - 264 bytes unboxed).
    GraphExpand {
        /// Expand configuration.
        node: Box<GraphExpandExecNode>,
        /// Input plan (source nodes).
        input: Box<PhysicalPlan>,
    },

    /// Graph path scan (boxed node - 96 bytes unboxed).
    GraphPathScan {
        /// Path scan configuration.
        node: Box<GraphPathScanExecNode>,
        /// Input plan (starting nodes).
        input: Box<PhysicalPlan>,
    },

    // ========== DML Operations ==========
    /// Insert operation.
    Insert {
        /// Target table.
        table: String,
        /// Column names.
        columns: Vec<String>,
        /// Returning columns.
        returning: Vec<LogicalExpr>,
        /// Estimated cost.
        cost: Cost,
        /// Input plan.
        input: Box<PhysicalPlan>,
    },

    /// Update operation.
    Update {
        /// Target table.
        table: String,
        /// Assignments.
        assignments: Vec<(String, LogicalExpr)>,
        /// Filter predicate.
        filter: Option<LogicalExpr>,
        /// Returning columns.
        returning: Vec<LogicalExpr>,
        /// Estimated cost.
        cost: Cost,
    },

    /// Delete operation.
    Delete {
        /// Target table.
        table: String,
        /// Filter predicate.
        filter: Option<LogicalExpr>,
        /// Returning columns.
        returning: Vec<LogicalExpr>,
        /// Estimated cost.
        cost: Cost,
    },

    // ========== DDL Operations ==========
    /// CREATE TABLE operation.
    CreateTable(CreateTableNode),

    /// DROP TABLE operation.
    DropTable(DropTableNode),

    /// CREATE INDEX operation.
    CreateIndex(CreateIndexNode),

    /// DROP INDEX operation.
    DropIndex(DropIndexNode),

    /// CREATE COLLECTION operation.
    CreateCollection(CreateCollectionNode),

    /// DROP COLLECTION operation.
    DropCollection(DropCollectionNode),

    // ========== Graph DML Operations ==========
    /// Cypher CREATE operation.
    GraphCreate {
        /// The create node configuration.
        node: Box<GraphCreateNode>,
        /// Optional input plan (from MATCH clause).
        input: Option<Box<PhysicalPlan>>,
    },

    /// Cypher MERGE operation.
    GraphMerge {
        /// The merge node configuration.
        node: Box<GraphMergeNode>,
        /// Optional input plan (from MATCH clause).
        input: Option<Box<PhysicalPlan>>,
    },

    // ========== Procedure Operations ==========
    /// Procedure call (CALL/YIELD).
    ProcedureCall(Box<ProcedureCallNode>),
}

// ============================================================================
// Scan Node Types
// ============================================================================

/// Full table scan node.
///
/// Reads all rows from a table without using any index.
#[derive(Debug, Clone, PartialEq)]
pub struct FullScanNode {
    /// Table name to scan.
    pub table_name: String,
    /// Optional table alias.
    pub alias: Option<String>,
    /// Columns to read (None means all).
    pub projection: Option<Vec<String>>,
    /// Optional filter to evaluate during scan.
    pub filter: Option<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl FullScanNode {
    /// Creates a new full scan node.
    #[must_use]
    pub fn new(table_name: impl Into<String>) -> Self {
        Self {
            table_name: table_name.into(),
            alias: None,
            projection: None,
            filter: None,
            cost: Cost::default(),
        }
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

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }

    /// Returns the effective table reference name.
    #[must_use]
    pub fn reference_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.table_name)
    }
}

/// Index point lookup scan node.
///
/// Uses an index to look up rows matching exact key values.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexScanNode {
    /// Table name.
    pub table_name: String,
    /// Index name.
    pub index_name: String,
    /// Key column(s) for the index.
    pub key_columns: Vec<String>,
    /// Key value(s) to look up.
    pub key_values: Vec<LogicalExpr>,
    /// Additional columns to read.
    pub projection: Option<Vec<String>>,
    /// Estimated cost.
    pub cost: Cost,
}

impl IndexScanNode {
    /// Creates a new index scan node.
    #[must_use]
    pub fn new(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_columns: Vec<String>,
        key_values: Vec<LogicalExpr>,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_columns,
            key_values,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Sets the projection columns.
    #[must_use]
    pub fn with_projection(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Index range scan node.
///
/// Uses an index to scan rows within a key range.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexRangeScanNode {
    /// Table name.
    pub table_name: String,
    /// Index name.
    pub index_name: String,
    /// Key column for the range.
    pub key_column: String,
    /// Lower bound (None means unbounded).
    pub lower_bound: Option<LogicalExpr>,
    /// Whether lower bound is inclusive.
    pub lower_inclusive: bool,
    /// Upper bound (None means unbounded).
    pub upper_bound: Option<LogicalExpr>,
    /// Whether upper bound is inclusive.
    pub upper_inclusive: bool,
    /// Additional columns to read.
    pub projection: Option<Vec<String>>,
    /// Estimated cost.
    pub cost: Cost,
}

impl IndexRangeScanNode {
    /// Creates a new range scan for values >= lower.
    #[must_use]
    pub fn gte(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_column: impl Into<String>,
        lower: LogicalExpr,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_column: key_column.into(),
            lower_bound: Some(lower),
            lower_inclusive: true,
            upper_bound: None,
            upper_inclusive: false,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Creates a new range scan for values > lower.
    #[must_use]
    pub fn gt(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_column: impl Into<String>,
        lower: LogicalExpr,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_column: key_column.into(),
            lower_bound: Some(lower),
            lower_inclusive: false,
            upper_bound: None,
            upper_inclusive: false,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Creates a new range scan for values <= upper.
    #[must_use]
    pub fn lte(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_column: impl Into<String>,
        upper: LogicalExpr,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_column: key_column.into(),
            lower_bound: None,
            lower_inclusive: false,
            upper_bound: Some(upper),
            upper_inclusive: true,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Creates a new range scan for values < upper.
    #[must_use]
    pub fn lt(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_column: impl Into<String>,
        upper: LogicalExpr,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_column: key_column.into(),
            lower_bound: None,
            lower_inclusive: false,
            upper_bound: Some(upper),
            upper_inclusive: false,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Creates a bounded range scan.
    #[must_use]
    pub fn between(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        key_column: impl Into<String>,
        lower: LogicalExpr,
        upper: LogicalExpr,
        inclusive: bool,
    ) -> Self {
        Self {
            table_name: table_name.into(),
            index_name: index_name.into(),
            key_column: key_column.into(),
            lower_bound: Some(lower),
            lower_inclusive: inclusive,
            upper_bound: Some(upper),
            upper_inclusive: inclusive,
            projection: None,
            cost: Cost::default(),
        }
    }

    /// Sets the projection columns.
    #[must_use]
    pub fn with_projection(mut self, columns: Vec<String>) -> Self {
        self.projection = Some(columns);
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// Unary Operator Node Types
// ============================================================================

/// Filter execution node.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterExecNode {
    /// The predicate to evaluate.
    pub predicate: LogicalExpr,
    /// Estimated selectivity (0.0 to 1.0).
    pub selectivity: f64,
    /// Estimated cost.
    pub cost: Cost,
}

impl FilterExecNode {
    /// Creates a new filter execution node.
    #[must_use]
    pub fn new(predicate: LogicalExpr) -> Self {
        Self {
            predicate,
            selectivity: 0.5, // Default 50% selectivity
            cost: Cost::default(),
        }
    }

    /// Sets the estimated selectivity.
    #[must_use]
    pub const fn with_selectivity(mut self, selectivity: f64) -> Self {
        self.selectivity = selectivity;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Projection execution node.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectExecNode {
    /// Expressions to project.
    pub exprs: Vec<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl ProjectExecNode {
    /// Creates a new projection execution node.
    #[must_use]
    pub fn new(exprs: Vec<LogicalExpr>) -> Self {
        Self { exprs, cost: Cost::default() }
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Sort execution node.
#[derive(Debug, Clone, PartialEq)]
pub struct SortExecNode {
    /// Sort specifications.
    pub order_by: Vec<SortOrder>,
    /// Whether the input might already be sorted.
    pub preserve_partitioning: bool,
    /// Estimated cost.
    pub cost: Cost,
}

impl SortExecNode {
    /// Creates a new sort execution node.
    #[must_use]
    pub fn new(order_by: Vec<SortOrder>) -> Self {
        Self { order_by, preserve_partitioning: false, cost: Cost::default() }
    }

    /// Sets preserve partitioning flag.
    #[must_use]
    pub const fn with_preserve_partitioning(mut self, preserve: bool) -> Self {
        self.preserve_partitioning = preserve;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Limit execution node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitExecNode {
    /// Maximum rows to return.
    pub limit: Option<usize>,
    /// Rows to skip.
    pub offset: Option<usize>,
}

impl LimitExecNode {
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

// ============================================================================
// Window Node Types
// ============================================================================

/// A single window function expression with its output alias.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFunctionExpr {
    /// The window function type (ROW_NUMBER, RANK, DENSE_RANK).
    pub func: WindowFunction,
    /// Partition by expressions.
    pub partition_by: Vec<LogicalExpr>,
    /// Order by expressions.
    pub order_by: Vec<SortOrder>,
    /// Output column alias.
    pub alias: String,
}

impl WindowFunctionExpr {
    /// Creates a new window function expression.
    #[must_use]
    pub fn new(
        func: WindowFunction,
        partition_by: Vec<LogicalExpr>,
        order_by: Vec<SortOrder>,
        alias: impl Into<String>,
    ) -> Self {
        Self { func, partition_by, order_by, alias: alias.into() }
    }
}

/// Window execution node.
///
/// Computes window functions (ROW_NUMBER, RANK, DENSE_RANK) over input rows.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExecNode {
    /// Window function expressions.
    pub window_exprs: Vec<WindowFunctionExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl WindowExecNode {
    /// Creates a new window execution node.
    #[must_use]
    pub fn new(window_exprs: Vec<WindowFunctionExpr>) -> Self {
        Self { window_exprs, cost: Cost::default() }
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// Aggregation Node Types
// ============================================================================

/// Hash-based aggregation node.
#[derive(Debug, Clone, PartialEq)]
pub struct HashAggregateNode {
    /// GROUP BY expressions.
    pub group_by: Vec<LogicalExpr>,
    /// Aggregate expressions.
    pub aggregates: Vec<LogicalExpr>,
    /// Optional HAVING clause.
    pub having: Option<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl HashAggregateNode {
    /// Creates a new hash aggregate node.
    #[must_use]
    pub fn new(group_by: Vec<LogicalExpr>, aggregates: Vec<LogicalExpr>) -> Self {
        Self { group_by, aggregates, having: None, cost: Cost::default() }
    }

    /// Sets the HAVING clause.
    #[must_use]
    pub fn with_having(mut self, having: LogicalExpr) -> Self {
        self.having = Some(having);
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Sort-merge based aggregation node.
///
/// Assumes input is already sorted by group-by keys.
#[derive(Debug, Clone, PartialEq)]
pub struct SortMergeAggregateNode {
    /// GROUP BY expressions.
    pub group_by: Vec<LogicalExpr>,
    /// Aggregate expressions.
    pub aggregates: Vec<LogicalExpr>,
    /// Optional HAVING clause.
    pub having: Option<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl SortMergeAggregateNode {
    /// Creates a new sort-merge aggregate node.
    #[must_use]
    pub fn new(group_by: Vec<LogicalExpr>, aggregates: Vec<LogicalExpr>) -> Self {
        Self { group_by, aggregates, having: None, cost: Cost::default() }
    }

    /// Sets the HAVING clause.
    #[must_use]
    pub fn with_having(mut self, having: LogicalExpr) -> Self {
        self.having = Some(having);
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// Join Node Types
// ============================================================================

/// Join order for hash/merge joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JoinOrder {
    /// Left is build side, right is probe side.
    #[default]
    LeftBuild,
    /// Right is build side, left is probe side.
    RightBuild,
}

/// Nested loop join node.
///
/// Simple O(n*m) join that iterates through all pairs.
/// Best for small tables or when no equijoin condition exists.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedLoopJoinNode {
    /// Join type.
    pub join_type: JoinType,
    /// Join condition (evaluated for each pair).
    pub condition: Option<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl NestedLoopJoinNode {
    /// Creates a new nested loop join.
    #[must_use]
    pub fn new(join_type: JoinType, condition: Option<LogicalExpr>) -> Self {
        Self { join_type, condition, cost: Cost::default() }
    }

    /// Creates an inner nested loop join.
    #[must_use]
    pub fn inner(condition: LogicalExpr) -> Self {
        Self::new(JoinType::Inner, Some(condition))
    }

    /// Creates a cross join.
    #[must_use]
    pub fn cross() -> Self {
        Self::new(JoinType::Cross, None)
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Hash join node.
///
/// Builds a hash table on one side, probes with the other.
/// Requires equijoin conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct HashJoinNode {
    /// Join type.
    pub join_type: JoinType,
    /// Equijoin keys from build side.
    pub build_keys: Vec<LogicalExpr>,
    /// Equijoin keys from probe side.
    pub probe_keys: Vec<LogicalExpr>,
    /// Additional non-equijoin conditions.
    pub filter: Option<LogicalExpr>,
    /// Which side is build vs probe.
    pub join_order: JoinOrder,
    /// Estimated cost.
    pub cost: Cost,
}

impl HashJoinNode {
    /// Creates a new hash join.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        build_keys: Vec<LogicalExpr>,
        probe_keys: Vec<LogicalExpr>,
    ) -> Self {
        Self {
            join_type,
            build_keys,
            probe_keys,
            filter: None,
            join_order: JoinOrder::default(),
            cost: Cost::default(),
        }
    }

    /// Creates an inner hash join on a single key pair.
    #[must_use]
    pub fn inner_on(build_key: LogicalExpr, probe_key: LogicalExpr) -> Self {
        Self::new(JoinType::Inner, vec![build_key], vec![probe_key])
    }

    /// Sets additional filter.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the join order.
    #[must_use]
    pub const fn with_join_order(mut self, order: JoinOrder) -> Self {
        self.join_order = order;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Merge join node.
///
/// Merges two sorted inputs. Requires both sides sorted on join keys.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeJoinNode {
    /// Join type.
    pub join_type: JoinType,
    /// Join keys from left side.
    pub left_keys: Vec<LogicalExpr>,
    /// Join keys from right side.
    pub right_keys: Vec<LogicalExpr>,
    /// Additional non-equijoin conditions.
    pub filter: Option<LogicalExpr>,
    /// Estimated cost.
    pub cost: Cost,
}

impl MergeJoinNode {
    /// Creates a new merge join.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        left_keys: Vec<LogicalExpr>,
        right_keys: Vec<LogicalExpr>,
    ) -> Self {
        Self { join_type, left_keys, right_keys, filter: None, cost: Cost::default() }
    }

    /// Sets additional filter.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// Vector Search Node Types
// ============================================================================

/// HNSW-based vector search node.
///
/// Uses Hierarchical Navigable Small World graphs for fast ANN search.
#[derive(Debug, Clone, PartialEq)]
pub struct HnswSearchNode {
    /// Vector column to search.
    pub vector_column: String,
    /// Query vector.
    pub query_vector: LogicalExpr,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Number of results.
    pub k: usize,
    /// HNSW ef_search parameter.
    pub ef_search: usize,
    /// Optional pre-filter.
    pub filter: Option<LogicalExpr>,
    /// Whether to include distance in output.
    pub include_distance: bool,
    /// Distance column alias.
    pub distance_alias: Option<String>,
    /// Name of the HNSW index to use for the search.
    /// Format: `{collection}_{vector_name}_hnsw`
    pub index_name: Option<String>,
    /// Collection name for named vector lookup.
    pub collection_name: Option<String>,
    /// Estimated cost.
    pub cost: Cost,
}

impl HnswSearchNode {
    /// Creates a new HNSW search node.
    #[must_use]
    pub fn new(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
    ) -> Self {
        Self {
            vector_column: vector_column.into(),
            query_vector,
            metric,
            k,
            ef_search: k * 10, // Default ef_search
            filter: None,
            include_distance: true,
            distance_alias: None,
            index_name: None,
            collection_name: None,
            cost: Cost::default(),
        }
    }

    /// Creates a new HNSW search node for a named vector in a collection.
    #[must_use]
    pub fn for_named_vector(
        collection_name: impl Into<String>,
        vector_name: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
    ) -> Self {
        let collection = collection_name.into();
        let vector = vector_name.into();
        let index_name = format!("{collection}_{vector}_hnsw");
        Self {
            vector_column: vector,
            query_vector,
            metric,
            k,
            ef_search: k * 10,
            filter: None,
            include_distance: true,
            distance_alias: None,
            index_name: Some(index_name),
            collection_name: Some(collection),
            cost: Cost::default(),
        }
    }

    /// Sets ef_search parameter.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Sets pre-filter.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets include_distance flag.
    #[must_use]
    pub const fn with_include_distance(mut self, include: bool) -> Self {
        self.include_distance = include;
        self
    }

    /// Sets distance alias.
    #[must_use]
    pub fn with_distance_alias(mut self, alias: impl Into<String>) -> Self {
        self.distance_alias = Some(alias.into());
        self
    }

    /// Sets the HNSW index name to use for this search.
    #[must_use]
    pub fn with_index_name(mut self, name: impl Into<String>) -> Self {
        self.index_name = Some(name.into());
        self
    }

    /// Sets the collection name for named vector lookup.
    #[must_use]
    pub fn with_collection_name(mut self, name: impl Into<String>) -> Self {
        self.collection_name = Some(name.into());
        self
    }

    /// Sets both collection and generates the index name.
    #[must_use]
    pub fn with_collection(mut self, collection: impl Into<String>) -> Self {
        let collection_name = collection.into();
        let index_name = format!("{}_{}_hnsw", collection_name, self.vector_column);
        self.collection_name = Some(collection_name);
        self.index_name = Some(index_name);
        self
    }

    /// Returns the index name if configured.
    #[must_use]
    pub fn index_name(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    /// Returns the collection name if configured.
    #[must_use]
    pub fn collection_name(&self) -> Option<&str> {
        self.collection_name.as_deref()
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Brute-force vector search node.
///
/// Computes distance to all vectors. Exact but slow for large datasets.
#[derive(Debug, Clone, PartialEq)]
pub struct BruteForceSearchNode {
    /// Vector column to search.
    pub vector_column: String,
    /// Query vector.
    pub query_vector: LogicalExpr,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Number of results.
    pub k: usize,
    /// Optional filter.
    pub filter: Option<LogicalExpr>,
    /// Whether to include distance in output.
    pub include_distance: bool,
    /// Distance column alias.
    pub distance_alias: Option<String>,
    /// Estimated cost.
    pub cost: Cost,
}

impl BruteForceSearchNode {
    /// Creates a new brute-force search node.
    #[must_use]
    pub fn new(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
    ) -> Self {
        Self {
            vector_column: vector_column.into(),
            query_vector,
            metric,
            k,
            filter: None,
            include_distance: true,
            distance_alias: None,
            cost: Cost::default(),
        }
    }

    /// Sets filter.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets include_distance flag.
    #[must_use]
    pub const fn with_include_distance(mut self, include: bool) -> Self {
        self.include_distance = include;
        self
    }

    /// Sets distance alias.
    #[must_use]
    pub fn with_distance_alias(mut self, alias: impl Into<String>) -> Self {
        self.distance_alias = Some(alias.into());
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Score combination method for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalScoreCombinationMethod {
    /// Weighted linear combination: `w1*s1 + w2*s2`
    WeightedSum,
    /// Reciprocal Rank Fusion with configurable k parameter.
    ReciprocalRankFusion {
        /// RRF k parameter (typically 60).
        k_param: u32,
    },
}

impl Default for PhysicalScoreCombinationMethod {
    fn default() -> Self {
        Self::WeightedSum
    }
}

/// A component search in a hybrid search.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchComponentNode {
    /// Vector column to search.
    pub vector_column: String,
    /// Query vector.
    pub query_vector: LogicalExpr,
    /// Distance metric.
    pub metric: DistanceMetric,
    /// Weight for this component.
    pub weight: f32,
    /// HNSW ef_search parameter.
    pub ef_search: usize,
    /// HNSW index name if available.
    pub index_name: Option<String>,
    /// Whether to use HNSW (true) or brute-force (false).
    pub use_hnsw: bool,
}

impl HybridSearchComponentNode {
    /// Creates a new component node.
    #[must_use]
    pub fn new(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        weight: f32,
    ) -> Self {
        Self {
            vector_column: vector_column.into(),
            query_vector,
            metric,
            weight,
            ef_search: 100,
            index_name: None,
            use_hnsw: false,
        }
    }

    /// Sets ef_search parameter.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Sets the HNSW index name.
    #[must_use]
    pub fn with_index_name(mut self, name: impl Into<String>) -> Self {
        self.index_name = Some(name.into());
        self.use_hnsw = true;
        self
    }

    /// Sets whether to use HNSW.
    #[must_use]
    pub const fn with_use_hnsw(mut self, use_hnsw: bool) -> Self {
        self.use_hnsw = use_hnsw;
        self
    }
}

/// Hybrid vector search node.
///
/// Combines multiple vector searches and merges results using score combination.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchNode {
    /// Search components (each represents a vector search).
    pub components: Vec<HybridSearchComponentNode>,
    /// Number of results.
    pub k: usize,
    /// Score combination method.
    pub combination_method: PhysicalScoreCombinationMethod,
    /// Whether to normalize scores before combining.
    pub normalize_scores: bool,
    /// Optional filter.
    pub filter: Option<LogicalExpr>,
    /// Whether to include the combined score in output.
    pub include_score: bool,
    /// Score column alias.
    pub score_alias: Option<String>,
    /// Collection name for named vector lookup.
    pub collection_name: Option<String>,
    /// Estimated cost.
    pub cost: Cost,
}

impl HybridSearchNode {
    /// Creates a new hybrid search node.
    #[must_use]
    pub fn new(components: Vec<HybridSearchComponentNode>, k: usize) -> Self {
        Self {
            components,
            k,
            combination_method: PhysicalScoreCombinationMethod::default(),
            normalize_scores: true,
            filter: None,
            include_score: true,
            score_alias: None,
            collection_name: None,
            cost: Cost::default(),
        }
    }

    /// Creates a two-component hybrid search.
    #[must_use]
    pub fn two(
        first: HybridSearchComponentNode,
        second: HybridSearchComponentNode,
        k: usize,
    ) -> Self {
        Self::new(vec![first, second], k)
    }

    /// Sets the combination method.
    #[must_use]
    pub const fn with_combination_method(mut self, method: PhysicalScoreCombinationMethod) -> Self {
        self.combination_method = method;
        self
    }

    /// Uses RRF combination.
    #[must_use]
    pub const fn with_rrf(mut self, k_param: u32) -> Self {
        self.combination_method = PhysicalScoreCombinationMethod::ReciprocalRankFusion { k_param };
        self
    }

    /// Disables score normalization.
    #[must_use]
    pub const fn without_normalization(mut self) -> Self {
        self.normalize_scores = false;
        self
    }

    /// Sets the filter.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets whether to include the score in output.
    #[must_use]
    pub const fn with_include_score(mut self, include: bool) -> Self {
        self.include_score = include;
        self
    }

    /// Sets the score alias.
    #[must_use]
    pub fn with_score_alias(mut self, alias: impl Into<String>) -> Self {
        self.score_alias = Some(alias.into());
        self
    }

    /// Sets the collection name.
    #[must_use]
    pub fn with_collection_name(mut self, name: impl Into<String>) -> Self {
        self.collection_name = Some(name.into());
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }

    /// Returns the number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

// ============================================================================
// Graph Operation Node Types
// ============================================================================

/// Graph expand execution node.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphExpandExecNode {
    /// Direction of expansion.
    pub direction: ExpandDirection,
    /// Source node variable.
    pub src_var: String,
    /// Destination node variable.
    pub dst_var: String,
    /// Edge variable.
    pub edge_var: Option<String>,
    /// Edge type filters.
    pub edge_types: Vec<String>,
    /// Variable length specification.
    pub length: ExpandLength,
    /// Edge filter.
    pub edge_filter: Option<LogicalExpr>,
    /// Node filter.
    pub node_filter: Option<LogicalExpr>,
    /// Node label filters.
    pub node_labels: Vec<String>,
    /// Estimated cost.
    pub cost: Cost,
}

impl GraphExpandExecNode {
    /// Creates a new graph expand execution node.
    #[must_use]
    pub fn new(
        src_var: impl Into<String>,
        dst_var: impl Into<String>,
        direction: ExpandDirection,
    ) -> Self {
        Self {
            direction,
            src_var: src_var.into(),
            dst_var: dst_var.into(),
            edge_var: None,
            edge_types: vec![],
            length: ExpandLength::Single,
            edge_filter: None,
            node_filter: None,
            node_labels: vec![],
            cost: Cost::default(),
        }
    }

    /// Sets edge variable.
    #[must_use]
    pub fn with_edge_var(mut self, var: impl Into<String>) -> Self {
        self.edge_var = Some(var.into());
        self
    }

    /// Sets edge types.
    #[must_use]
    pub fn with_edge_types(mut self, types: Vec<String>) -> Self {
        self.edge_types = types;
        self
    }

    /// Sets expansion length.
    #[must_use]
    pub const fn with_length(mut self, length: ExpandLength) -> Self {
        self.length = length;
        self
    }

    /// Sets edge filter.
    #[must_use]
    pub fn with_edge_filter(mut self, filter: LogicalExpr) -> Self {
        self.edge_filter = Some(filter);
        self
    }

    /// Sets node filter.
    #[must_use]
    pub fn with_node_filter(mut self, filter: LogicalExpr) -> Self {
        self.node_filter = Some(filter);
        self
    }

    /// Sets node labels.
    #[must_use]
    pub fn with_node_labels(mut self, labels: Vec<String>) -> Self {
        self.node_labels = labels;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Graph path scan execution node.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPathScanExecNode {
    /// Path steps.
    pub steps: Vec<GraphExpandExecNode>,
    /// Starting node filter.
    pub start_filter: Option<LogicalExpr>,
    /// Return all paths or just distinct end nodes.
    pub all_paths: bool,
    /// Track full path for path expressions.
    pub track_path: bool,
    /// Estimated cost.
    pub cost: Cost,
}

impl GraphPathScanExecNode {
    /// Creates a new path scan execution node.
    #[must_use]
    pub fn new(steps: Vec<GraphExpandExecNode>) -> Self {
        Self {
            steps,
            start_filter: None,
            all_paths: false,
            track_path: false,
            cost: Cost::default(),
        }
    }

    /// Sets start filter.
    #[must_use]
    pub fn with_start_filter(mut self, filter: LogicalExpr) -> Self {
        self.start_filter = Some(filter);
        self
    }

    /// Enables all paths mode.
    #[must_use]
    pub const fn with_all_paths(mut self) -> Self {
        self.all_paths = true;
        self
    }

    /// Enables path tracking.
    #[must_use]
    pub const fn with_track_path(mut self) -> Self {
        self.track_path = true;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// Unwind Node Type
// ============================================================================

/// Unwind execution node.
///
/// Expands a list expression into multiple rows, one per element.
#[derive(Debug, Clone, PartialEq)]
pub struct UnwindExecNode {
    /// The expression that produces a list to unwind.
    pub list_expr: LogicalExpr,
    /// The variable name to bind each unwound element to.
    pub alias: String,
    /// Estimated cost.
    pub cost: Cost,
}

impl UnwindExecNode {
    /// Creates a new unwind execution node.
    #[must_use]
    pub fn new(list_expr: LogicalExpr, alias: impl Into<String>) -> Self {
        Self { list_expr, alias: alias.into(), cost: Cost::default() }
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

/// Recursive CTE execution node.
///
/// Represents a recursive Common Table Expression that iteratively executes
/// until a fixed point is reached.
#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveCTEExecNode {
    /// The name of the CTE.
    pub name: String,
    /// Column names for the CTE result.
    pub columns: Vec<String>,
    /// Whether to use UNION ALL (true) or UNION (false) semantics.
    /// UNION ALL keeps all rows, UNION deduplicates.
    pub union_all: bool,
    /// Maximum number of iterations allowed (safety limit).
    pub max_iterations: usize,
    /// Estimated cost.
    pub cost: Cost,
}

impl RecursiveCTEExecNode {
    /// Default maximum iterations if not specified.
    pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

    /// Creates a new recursive CTE execution node.
    #[must_use]
    pub fn new(name: impl Into<String>, columns: Vec<String>, union_all: bool) -> Self {
        Self {
            name: name.into(),
            columns,
            union_all,
            max_iterations: Self::DEFAULT_MAX_ITERATIONS,
            cost: Cost::default(),
        }
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Sets the cost estimate.
    #[must_use]
    pub const fn with_cost(mut self, cost: Cost) -> Self {
        self.cost = cost;
        self
    }
}

// ============================================================================
// PhysicalPlan Implementation
// ============================================================================

impl PhysicalPlan {
    /// Returns the children of this plan node.
    #[must_use]
    pub fn children(&self) -> Vec<&PhysicalPlan> {
        match self {
            // Leaf nodes
            Self::FullScan(_)
            | Self::IndexScan(_)
            | Self::IndexRangeScan(_)
            | Self::Values { .. }
            | Self::Empty { .. }
            | Self::Update { .. }
            | Self::Delete { .. }
            | Self::CreateTable(_)
            | Self::DropTable(_)
            | Self::CreateIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::ProcedureCall(_) => vec![],

            // Unary nodes
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::HashDistinct { input, .. }
            | Self::Window { input, .. }
            | Self::HashAggregate { input, .. }
            | Self::SortMergeAggregate { input, .. }
            | Self::Unwind { input, .. }
            | Self::HnswSearch { input, .. }
            | Self::BruteForceSearch { input, .. }
            | Self::HybridSearch { input, .. }
            | Self::GraphExpand { input, .. }
            | Self::GraphPathScan { input, .. }
            | Self::Insert { input, .. } => vec![input.as_ref()],

            // Binary nodes
            Self::NestedLoopJoin { left, right, .. }
            | Self::SetOp { left, right, .. }
            | Self::RecursiveCTE { initial: left, recursive: right, .. } => {
                vec![left.as_ref(), right.as_ref()]
            }

            Self::HashJoin { build, probe, .. } => vec![build.as_ref(), probe.as_ref()],

            Self::MergeJoin { left, right, .. } => vec![left.as_ref(), right.as_ref()],

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter().collect(),

            // Graph DML operations (optional input)
            Self::GraphCreate { input, .. } | Self::GraphMerge { input, .. } => {
                input.as_ref().map_or(vec![], |i| vec![i.as_ref()])
            }
        }
    }

    /// Returns the mutable children of this plan node.
    #[must_use]
    pub fn children_mut(&mut self) -> Vec<&mut PhysicalPlan> {
        match self {
            // Leaf nodes
            Self::FullScan(_)
            | Self::IndexScan(_)
            | Self::IndexRangeScan(_)
            | Self::Values { .. }
            | Self::Empty { .. }
            | Self::Update { .. }
            | Self::Delete { .. }
            | Self::CreateTable(_)
            | Self::DropTable(_)
            | Self::CreateIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::ProcedureCall(_) => vec![],

            // Unary nodes
            Self::Filter { input, .. }
            | Self::Project { input, .. }
            | Self::Sort { input, .. }
            | Self::Limit { input, .. }
            | Self::HashDistinct { input, .. }
            | Self::Window { input, .. }
            | Self::HashAggregate { input, .. }
            | Self::SortMergeAggregate { input, .. }
            | Self::Unwind { input, .. }
            | Self::HnswSearch { input, .. }
            | Self::BruteForceSearch { input, .. }
            | Self::HybridSearch { input, .. }
            | Self::GraphExpand { input, .. }
            | Self::GraphPathScan { input, .. }
            | Self::Insert { input, .. } => vec![input.as_mut()],

            // Binary nodes
            Self::NestedLoopJoin { left, right, .. }
            | Self::SetOp { left, right, .. }
            | Self::RecursiveCTE { initial: left, recursive: right, .. } => {
                vec![left.as_mut(), right.as_mut()]
            }

            Self::HashJoin { build, probe, .. } => vec![build.as_mut(), probe.as_mut()],

            Self::MergeJoin { left, right, .. } => vec![left.as_mut(), right.as_mut()],

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter_mut().collect(),

            // Graph DML operations (optional input)
            Self::GraphCreate { input, .. } | Self::GraphMerge { input, .. } => {
                input.as_mut().map_or(vec![], |i| vec![i.as_mut()])
            }
        }
    }

    /// Returns true if this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(
            self,
            Self::FullScan(_)
                | Self::IndexScan(_)
                | Self::IndexRangeScan(_)
                | Self::Values { .. }
                | Self::Empty { .. }
                | Self::CreateTable(_)
                | Self::DropTable(_)
                | Self::CreateIndex(_)
                | Self::DropIndex(_)
                | Self::CreateCollection(_)
                | Self::DropCollection(_)
                | Self::ProcedureCall(_)
        )
    }

    /// Returns the node type name.
    #[must_use]
    pub fn node_type(&self) -> &'static str {
        match self {
            Self::FullScan(_) => "FullScan",
            Self::IndexScan(_) => "IndexScan",
            Self::IndexRangeScan(_) => "IndexRangeScan",
            Self::Values { .. } => "Values",
            Self::Empty { .. } => "Empty",
            Self::Filter { .. } => "Filter",
            Self::Project { .. } => "Project",
            Self::Sort { .. } => "Sort",
            Self::Limit { .. } => "Limit",
            Self::HashDistinct { .. } => "HashDistinct",
            Self::Window { .. } => "Window",
            Self::HashAggregate { .. } => "HashAggregate",
            Self::SortMergeAggregate { .. } => "SortMergeAggregate",
            Self::NestedLoopJoin { .. } => "NestedLoopJoin",
            Self::HashJoin { .. } => "HashJoin",
            Self::MergeJoin { .. } => "MergeJoin",
            Self::SetOp { .. } => "SetOp",
            Self::Union { .. } => "Union",
            Self::Unwind { .. } => "Unwind",
            Self::RecursiveCTE { .. } => "RecursiveCTE",
            Self::HnswSearch { .. } => "HnswSearch",
            Self::BruteForceSearch { .. } => "BruteForceSearch",
            Self::HybridSearch { .. } => "HybridSearch",
            Self::GraphExpand { .. } => "GraphExpand",
            Self::GraphPathScan { .. } => "GraphPathScan",
            Self::Insert { .. } => "Insert",
            Self::Update { .. } => "Update",
            Self::Delete { .. } => "Delete",
            Self::CreateTable(_) => "CreateTable",
            Self::DropTable(_) => "DropTable",
            Self::CreateIndex(_) => "CreateIndex",
            Self::DropIndex(_) => "DropIndex",
            Self::CreateCollection(_) => "CreateCollection",
            Self::DropCollection(_) => "DropCollection",
            Self::GraphCreate { .. } => "GraphCreate",
            Self::GraphMerge { .. } => "GraphMerge",
            Self::ProcedureCall(_) => "ProcedureCall",
        }
    }

    /// Returns the estimated cost of this node.
    #[must_use]
    pub fn cost(&self) -> Cost {
        match self {
            Self::FullScan(node) => node.cost,
            Self::IndexScan(node) => node.cost,
            Self::IndexRangeScan(node) => node.cost,
            Self::Values { cost, .. } => *cost,
            Self::Empty { .. } => Cost::zero(),
            Self::Filter { node, .. } => node.cost,
            Self::Project { node, .. } => node.cost,
            Self::Sort { node, .. } => node.cost,
            Self::Limit { .. } => Cost::zero(),
            Self::HashDistinct { cost, .. } => *cost,
            Self::Window { node, .. } => node.cost,
            Self::HashAggregate { node, .. } => node.cost,
            Self::SortMergeAggregate { node, .. } => node.cost,
            Self::NestedLoopJoin { node, .. } => node.cost,
            Self::HashJoin { node, .. } => node.cost,
            Self::MergeJoin { node, .. } => node.cost,
            Self::SetOp { cost, .. } => *cost,
            Self::Union { cost, .. } => *cost,
            Self::Unwind { node, .. } => node.cost,
            Self::RecursiveCTE { node, .. } => node.cost,
            Self::HnswSearch { node, .. } => node.cost,
            Self::BruteForceSearch { node, .. } => node.cost,
            Self::HybridSearch { node, .. } => node.cost,
            Self::GraphExpand { node, .. } => node.cost,
            Self::GraphPathScan { node, .. } => node.cost,
            Self::Insert { cost, .. } => *cost,
            Self::Update { cost, .. } => *cost,
            Self::Delete { cost, .. } => *cost,
            // DDL and procedure operations have zero query cost
            Self::CreateTable(_)
            | Self::DropTable(_)
            | Self::CreateIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::ProcedureCall(_) => Cost::zero(),
            // Graph DML operations - cost is based on input if present
            Self::GraphCreate { .. } | Self::GraphMerge { .. } => Cost::default(),
        }
    }

    /// Returns the total cost including all children.
    #[must_use]
    pub fn total_cost(&self) -> Cost {
        let mut total = self.cost();
        for child in self.children() {
            total = total + child.total_cost();
        }
        total
    }

    /// Pretty prints the plan as a tree.
    #[must_use]
    pub fn display_tree(&self) -> DisplayTree<'_> {
        DisplayTree { plan: self }
    }
}

/// Helper for tree-style plan display.
pub struct DisplayTree<'a> {
    plan: &'a PhysicalPlan,
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
        plan: &PhysicalPlan,
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

    fn fmt_node_content(&self, f: &mut fmt::Formatter<'_>, plan: &PhysicalPlan) -> fmt::Result {
        match plan {
            PhysicalPlan::FullScan(node) => {
                write!(f, "FullScan: {}", node.table_name)?;
                if let Some(alias) = &node.alias {
                    write!(f, " AS {alias}")?;
                }
                if let Some(filter) = &node.filter {
                    write!(f, " [filter: {filter}]")?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::IndexScan(node) => {
                write!(
                    f,
                    "IndexScan: {} using {} (cost: {:.2})",
                    node.table_name,
                    node.index_name,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::IndexRangeScan(node) => {
                write!(f, "IndexRangeScan: {} using {}", node.table_name, node.index_name)?;
                if let Some(lower) = &node.lower_bound {
                    let op = if node.lower_inclusive { ">=" } else { ">" };
                    write!(f, " [{} {op} {lower}]", node.key_column)?;
                }
                if let Some(upper) = &node.upper_bound {
                    let op = if node.upper_inclusive { "<=" } else { "<" };
                    write!(f, " [{} {op} {upper}]", node.key_column)?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::Values { rows, cost } => {
                write!(f, "Values: {} rows (cost: {:.2})", rows.len(), cost.value())?;
            }
            PhysicalPlan::Empty { columns } => {
                write!(f, "Empty: {} columns", columns.len())?;
            }
            PhysicalPlan::Filter { node, .. } => {
                write!(
                    f,
                    "Filter: {} (selectivity: {:.2}, cost: {:.2})",
                    node.predicate,
                    node.selectivity,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::Project { node, .. } => {
                write!(f, "Project: ")?;
                for (i, expr) in node.exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::Sort { node, .. } => {
                write!(f, "Sort: ")?;
                for (i, order) in node.order_by.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{order}")?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::Limit { node, .. } => {
                write!(f, "Limit: ")?;
                if let Some(n) = node.limit {
                    write!(f, "{n}")?;
                }
                if let Some(off) = node.offset {
                    write!(f, " OFFSET {off}")?;
                }
            }
            PhysicalPlan::HashDistinct { on_columns, cost, .. } => {
                write!(f, "HashDistinct")?;
                if let Some(cols) = on_columns {
                    write!(f, " ON ")?;
                    for (i, col) in cols.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{col}")?;
                    }
                }
                write!(f, " (cost: {:.2})", cost.value())?;
            }
            PhysicalPlan::Window { node, .. } => {
                write!(f, "Window: ")?;
                for (i, expr) in node.window_exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}() OVER (", expr.func)?;
                    if !expr.partition_by.is_empty() {
                        write!(f, "PARTITION BY ")?;
                        for (j, p) in expr.partition_by.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{p}")?;
                        }
                        if !expr.order_by.is_empty() {
                            write!(f, " ")?;
                        }
                    }
                    if !expr.order_by.is_empty() {
                        write!(f, "ORDER BY ")?;
                        for (j, o) in expr.order_by.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{o}")?;
                        }
                    }
                    write!(f, ") AS {}", expr.alias)?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::HashAggregate { node, .. } => {
                write!(f, "HashAggregate: ")?;
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
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::SortMergeAggregate { node, .. } => {
                write!(f, "SortMergeAggregate: ")?;
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
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::NestedLoopJoin { node, .. } => {
                write!(
                    f,
                    "NestedLoopJoin: {} JOIN (cost: {:.2})",
                    node.join_type,
                    node.cost.value()
                )?;
                if let Some(cond) = &node.condition {
                    write!(f, " ON {cond}")?;
                }
            }
            PhysicalPlan::HashJoin { node, .. } => {
                write!(f, "HashJoin: {} JOIN (cost: {:.2})", node.join_type, node.cost.value())?;
                write!(f, " [")?;
                for (i, (build, probe)) in node.build_keys.iter().zip(&node.probe_keys).enumerate()
                {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{build} = {probe}")?;
                }
                write!(f, "]")?;
            }
            PhysicalPlan::MergeJoin { node, .. } => {
                write!(f, "MergeJoin: {} JOIN (cost: {:.2})", node.join_type, node.cost.value())?;
            }
            PhysicalPlan::SetOp { op_type, cost, .. } => {
                write!(f, "SetOp: {} (cost: {:.2})", op_type, cost.value())?;
            }
            PhysicalPlan::Union { all, cost, inputs, .. } => {
                write!(
                    f,
                    "Union{}: {} inputs (cost: {:.2})",
                    if *all { " All" } else { "" },
                    inputs.len(),
                    cost.value()
                )?;
            }
            PhysicalPlan::Unwind { node, .. } => {
                write!(
                    f,
                    "Unwind: {} AS {} (cost: {:.2})",
                    node.list_expr,
                    node.alias,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::RecursiveCTE { node, .. } => {
                write!(f, "RecursiveCTE: {}", node.name)?;
                if !node.columns.is_empty() {
                    write!(f, " ({})", node.columns.join(", "))?;
                }
                write!(f, " {}", if node.union_all { "UNION ALL" } else { "UNION" })?;
                write!(f, " [max_iter: {}]", node.max_iterations)?;
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::HnswSearch { node, .. } => {
                write!(
                    f,
                    "HnswSearch: {} {} k={} ef={} (cost: {:.2})",
                    node.vector_column,
                    node.metric.operator(),
                    node.k,
                    node.ef_search,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::BruteForceSearch { node, .. } => {
                write!(
                    f,
                    "BruteForceSearch: {} {} k={} (cost: {:.2})",
                    node.vector_column,
                    node.metric.operator(),
                    node.k,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::HybridSearch { node, .. } => {
                write!(f, "HybridSearch: {} components, k={}", node.num_components(), node.k)?;
                for (i, comp) in node.components.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    } else {
                        write!(f, " [")?;
                    }
                    write!(
                        f,
                        "{} {} (w={:.2}{})",
                        comp.vector_column,
                        comp.metric.operator(),
                        comp.weight,
                        if comp.use_hnsw { ", HNSW" } else { "" }
                    )?;
                }
                if !node.components.is_empty() {
                    write!(f, "]")?;
                }
                write!(f, " (cost: {:.2})", node.cost.value())?;
            }
            PhysicalPlan::GraphExpand { node, .. } => {
                write!(
                    f,
                    "GraphExpand: ({}){}({}) (cost: {:.2})",
                    node.src_var,
                    node.direction,
                    node.dst_var,
                    node.cost.value()
                )?;
            }
            PhysicalPlan::GraphPathScan { node, .. } => {
                write!(
                    f,
                    "GraphPathScan: {} steps (cost: {:.2})",
                    node.steps.len(),
                    node.cost.value()
                )?;
            }
            PhysicalPlan::Insert { table, cost, .. } => {
                write!(f, "Insert: {} (cost: {:.2})", table, cost.value())?;
            }
            PhysicalPlan::Update { table, cost, .. } => {
                write!(f, "Update: {} (cost: {:.2})", table, cost.value())?;
            }
            PhysicalPlan::Delete { table, cost, .. } => {
                write!(f, "Delete: {} (cost: {:.2})", table, cost.value())?;
            }
            PhysicalPlan::CreateTable(node) => {
                write!(f, "CreateTable: {}", node.name)?;
                if node.if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " ({} columns)", node.columns.len())?;
            }
            PhysicalPlan::DropTable(node) => {
                write!(f, "DropTable: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            PhysicalPlan::CreateIndex(node) => {
                write!(f, "CreateIndex: {} ON {}", node.name, node.table)?;
                if node.unique {
                    write!(f, " UNIQUE")?;
                }
                if let Some(method) = &node.using {
                    write!(f, " USING {method}")?;
                }
            }
            PhysicalPlan::DropIndex(node) => {
                write!(f, "DropIndex: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
            }
            PhysicalPlan::CreateCollection(node) => {
                write!(f, "CreateCollection: {}", node.name)?;
                if node.if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " ({} vectors)", node.vectors.len())?;
            }
            PhysicalPlan::DropCollection(node) => {
                write!(f, "DropCollection: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            PhysicalPlan::GraphCreate { node, .. } => {
                write!(
                    f,
                    "GraphCreate: {} nodes, {} relationships",
                    node.nodes.len(),
                    node.relationships.len()
                )?;
                if !node.returning.is_empty() {
                    write!(f, " RETURN {} items", node.returning.len())?;
                }
            }
            PhysicalPlan::GraphMerge { node, .. } => {
                write!(f, "GraphMerge")?;
                if !node.on_create.is_empty() {
                    write!(f, " ON CREATE {} SET", node.on_create.len())?;
                }
                if !node.on_match.is_empty() {
                    write!(f, " ON MATCH {} SET", node.on_match.len())?;
                }
                if !node.returning.is_empty() {
                    write!(f, " RETURN {} items", node.returning.len())?;
                }
            }
            PhysicalPlan::ProcedureCall(node) => {
                write!(f, "ProcedureCall: {}", node.procedure_name)?;
                if !node.arguments.is_empty() {
                    write!(f, "(")?;
                    for (i, arg) in node.arguments.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{arg}")?;
                    }
                    write!(f, ")")?;
                }
                if !node.yield_columns.is_empty() {
                    write!(f, " YIELD ")?;
                    for (i, col) in node.yield_columns.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", col.name)?;
                        if let Some(alias) = &col.alias {
                            write!(f, " AS {alias}")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_tree())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::LogicalExpr;

    #[test]
    fn full_scan_basic() {
        let scan = FullScanNode::new("users")
            .with_alias("u")
            .with_projection(vec!["id".to_string(), "name".to_string()]);

        assert_eq!(scan.table_name, "users");
        assert_eq!(scan.reference_name(), "u");
        assert!(scan.projection.is_some());
    }

    #[test]
    fn index_scan_basic() {
        let scan = IndexScanNode::new(
            "users",
            "users_pk",
            vec!["id".to_string()],
            vec![LogicalExpr::integer(42)],
        );

        assert_eq!(scan.table_name, "users");
        assert_eq!(scan.index_name, "users_pk");
    }

    #[test]
    fn hash_join_basic() {
        let join =
            HashJoinNode::inner_on(LogicalExpr::column("user_id"), LogicalExpr::column("id"));

        assert_eq!(join.join_type, JoinType::Inner);
        assert_eq!(join.build_keys.len(), 1);
        assert_eq!(join.probe_keys.len(), 1);
    }

    #[test]
    fn physical_plan_tree() {
        let plan = PhysicalPlan::Filter {
            node: FilterExecNode::new(LogicalExpr::column("age").gt(LogicalExpr::integer(21))),
            input: Box::new(PhysicalPlan::FullScan(Box::new(FullScanNode::new("users")))),
        };

        assert_eq!(plan.node_type(), "Filter");
        assert_eq!(plan.children().len(), 1);
        assert!(!plan.is_leaf());
    }

    #[test]
    fn display_physical_plan() {
        let plan = PhysicalPlan::Project {
            node: ProjectExecNode::new(vec![
                LogicalExpr::column("id"),
                LogicalExpr::column("name"),
            ]),
            input: Box::new(PhysicalPlan::Filter {
                node: FilterExecNode::new(
                    LogicalExpr::column("active").eq(LogicalExpr::boolean(true)),
                ),
                input: Box::new(PhysicalPlan::FullScan(Box::new(FullScanNode::new("users")))),
            }),
        };

        let output = format!("{plan}");
        assert!(output.contains("Project"));
        assert!(output.contains("Filter"));
        assert!(output.contains("FullScan"));
    }

    #[test]
    fn hnsw_search_node() {
        let node =
            HnswSearchNode::new("embedding", LogicalExpr::param(1), DistanceMetric::Cosine, 10)
                .with_ef_search(100);

        assert_eq!(node.k, 10);
        assert_eq!(node.ef_search, 100);
        assert_eq!(node.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn cost_propagation() {
        let plan = PhysicalPlan::Filter {
            node: FilterExecNode::new(LogicalExpr::column("a").eq(LogicalExpr::integer(1)))
                .with_cost(Cost::new(10.0, 100)),
            input: Box::new(PhysicalPlan::FullScan(Box::new(
                FullScanNode::new("t").with_cost(Cost::new(100.0, 1000)),
            ))),
        };

        let total = plan.total_cost();
        assert!(total.value() > 0.0);
    }
}
