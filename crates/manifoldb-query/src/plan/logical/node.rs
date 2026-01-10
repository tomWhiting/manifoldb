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

use super::ddl::{
    AlterIndexNode, AlterSchemaNode, AlterTableNode, CreateCollectionNode, CreateFunctionNode,
    CreateIndexNode, CreateSchemaNode, CreateTableNode, CreateTriggerNode, CreateViewNode,
    DropCollectionNode, DropFunctionNode, DropIndexNode, DropSchemaNode, DropTableNode,
    DropTriggerNode, DropViewNode, TruncateTableNode,
};
use super::expr::{LogicalExpr, SortOrder};
use super::graph::{
    ExpandNode, GraphCreateNode, GraphDeleteNode, GraphForeachNode, GraphMergeNode,
    GraphRemoveNode, GraphSetNode, PathScanNode, ShortestPathNode,
};
use super::procedure::ProcedureCallNode;
use super::relational::{
    AggregateNode, CallSubqueryNode, DistinctNode, FilterNode, JoinNode, LimitNode, ProjectNode,
    RecursiveCTENode, ScanNode, SetOpNode, SortNode, UnionNode, UnwindNode, ValuesNode, WindowNode,
};
use super::transaction::{
    BeginTransactionNode, CommitNode, ReleaseSavepointNode, RollbackNode, SavepointNode,
    SetTransactionNode,
};
use super::utility::{
    AnalyzeNode, CopyNode, ExplainAnalyzeNode, ResetNode, SetSessionNode, ShowNode,
    ShowProceduresNode, VacuumNode,
};
use super::vector::{AnnSearchNode, HybridSearchNode, VectorDistanceNode};

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

    /// Window functions (ROW_NUMBER, RANK, DENSE_RANK, etc.).
    Window {
        /// The window node.
        node: WindowNode,
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

    /// Unwind (expand a list into rows).
    Unwind {
        /// The unwind node.
        node: UnwindNode,
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

    // ========== CALL { } Subquery Nodes ==========
    /// Cypher CALL { } inline subquery.
    ///
    /// For each input row, executes the inner subquery with imported variables
    /// bound to the outer row's values. Returns the cross product of input rows
    /// with their corresponding subquery results (like LATERAL join in SQL).
    CallSubquery {
        /// The CALL subquery node metadata.
        node: CallSubqueryNode,
        /// The inner subquery plan.
        subquery: Box<LogicalPlan>,
        /// The input plan (outer query rows).
        input: Box<LogicalPlan>,
    },

    // ========== Recursive Nodes ==========
    /// Recursive CTE (WITH RECURSIVE).
    ///
    /// Iteratively executes the recursive query until a fixed point is reached.
    /// The initial query seeds the result, and the recursive query is executed
    /// repeatedly using the working table until no new rows are produced.
    RecursiveCTE {
        /// The recursive CTE node metadata.
        node: RecursiveCTENode,
        /// The initial (base case) query.
        initial: Box<LogicalPlan>,
        /// The recursive query (references the CTE).
        recursive: Box<LogicalPlan>,
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

    /// Shortest path pattern function (shortestPath/allShortestPaths).
    ///
    /// Finds the shortest path(s) between two nodes using BFS (unweighted)
    /// or Dijkstra's algorithm (weighted).
    ShortestPath {
        /// The shortest path node configuration.
        node: Box<ShortestPathNode>,
        /// The input plan (provides source/target nodes to match against).
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

    /// Hybrid vector search combining multiple distance types.
    HybridSearch {
        /// The hybrid search node.
        node: Box<HybridSearchNode>,
        /// The input plan (table to search).
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
        /// ON CONFLICT clause for upsert behavior.
        on_conflict: Option<LogicalOnConflict>,
        /// Whether to return inserted rows.
        returning: Vec<LogicalExpr>,
    },

    /// UPDATE operation.
    Update {
        /// Target table name.
        table: String,
        /// Assignments (column, value).
        assignments: Vec<(String, LogicalExpr)>,
        /// Optional source tables (for UPDATE ... FROM).
        /// When present, the update uses rows from the source joined with target.
        source: Option<Box<LogicalPlan>>,
        /// Filter for rows to update (WHERE clause).
        filter: Option<LogicalExpr>,
        /// Whether to return updated rows.
        returning: Vec<LogicalExpr>,
    },

    /// DELETE operation.
    Delete {
        /// Target table name.
        table: String,
        /// Optional source tables (for DELETE ... USING).
        /// When present, the delete uses rows from the source joined with target.
        source: Option<Box<LogicalPlan>>,
        /// Filter for rows to delete (WHERE clause).
        filter: Option<LogicalExpr>,
        /// Whether to return deleted rows.
        returning: Vec<LogicalExpr>,
    },

    /// SQL MERGE operation (conditional INSERT/UPDATE/DELETE).
    MergeSql {
        /// Target table name.
        target_table: String,
        /// Source plan (can be a scan, subquery, etc.).
        source: Box<LogicalPlan>,
        /// Join condition between source and target.
        on_condition: LogicalExpr,
        /// WHEN clauses specifying actions.
        clauses: Vec<LogicalMergeClause>,
    },

    // ========== DDL Nodes ==========
    /// CREATE TABLE operation.
    CreateTable(CreateTableNode),

    /// ALTER TABLE operation.
    AlterTable(AlterTableNode),

    /// DROP TABLE operation.
    DropTable(DropTableNode),

    /// TRUNCATE TABLE operation.
    TruncateTable(TruncateTableNode),

    /// CREATE INDEX operation.
    CreateIndex(CreateIndexNode),

    /// ALTER INDEX operation.
    AlterIndex(AlterIndexNode),

    /// DROP INDEX operation.
    DropIndex(DropIndexNode),

    /// CREATE COLLECTION operation.
    CreateCollection(CreateCollectionNode),

    /// DROP COLLECTION operation.
    DropCollection(DropCollectionNode),

    /// CREATE VIEW operation.
    CreateView(CreateViewNode),

    /// DROP VIEW operation.
    DropView(DropViewNode),

    /// CREATE SCHEMA operation.
    CreateSchema(CreateSchemaNode),

    /// ALTER SCHEMA operation.
    AlterSchema(AlterSchemaNode),

    /// DROP SCHEMA operation.
    DropSchema(DropSchemaNode),

    /// CREATE FUNCTION operation.
    CreateFunction(Box<CreateFunctionNode>),

    /// DROP FUNCTION operation.
    DropFunction(DropFunctionNode),

    /// CREATE TRIGGER operation.
    CreateTrigger(Box<CreateTriggerNode>),

    /// DROP TRIGGER operation.
    DropTrigger(DropTriggerNode),

    // ========== Graph DML Nodes ==========
    /// Cypher CREATE operation (nodes and/or relationships).
    GraphCreate {
        /// The create node.
        node: Box<GraphCreateNode>,
        /// Optional input plan (from MATCH clause).
        input: Option<Box<LogicalPlan>>,
    },

    /// Cypher MERGE operation (upsert semantics).
    GraphMerge {
        /// The merge node.
        node: Box<GraphMergeNode>,
        /// Optional input plan (from MATCH clause).
        input: Option<Box<LogicalPlan>>,
    },

    /// Cypher SET operation (update properties/labels).
    GraphSet {
        /// The SET node.
        node: Box<GraphSetNode>,
        /// Input plan (from MATCH clause).
        input: Box<LogicalPlan>,
    },

    /// Cypher DELETE operation (remove nodes/relationships).
    GraphDelete {
        /// The DELETE node.
        node: Box<GraphDeleteNode>,
        /// Input plan (from MATCH clause).
        input: Box<LogicalPlan>,
    },

    /// Cypher REMOVE operation (remove properties/labels).
    GraphRemove {
        /// The REMOVE node.
        node: Box<GraphRemoveNode>,
        /// Input plan (from MATCH clause).
        input: Box<LogicalPlan>,
    },

    /// Cypher FOREACH operation (iterate over list with mutations).
    GraphForeach {
        /// The FOREACH node.
        node: Box<GraphForeachNode>,
        /// Input plan (from MATCH clause).
        input: Box<LogicalPlan>,
    },

    // ========== Procedure Nodes ==========
    /// Procedure call (CALL/YIELD).
    ProcedureCall(Box<ProcedureCallNode>),

    // ========== Transaction Nodes ==========
    /// BEGIN / START TRANSACTION.
    BeginTransaction(BeginTransactionNode),

    /// COMMIT.
    Commit(CommitNode),

    /// ROLLBACK (optionally to a savepoint).
    Rollback(RollbackNode),

    /// SAVEPOINT <name>.
    Savepoint(SavepointNode),

    /// RELEASE SAVEPOINT <name>.
    ReleaseSavepoint(ReleaseSavepointNode),

    /// SET TRANSACTION.
    SetTransaction(SetTransactionNode),

    // ========== Utility Nodes ==========
    /// EXPLAIN ANALYZE (executes and collects statistics).
    ExplainAnalyze(ExplainAnalyzeNode),

    /// VACUUM operation.
    Vacuum(VacuumNode),

    /// ANALYZE operation.
    Analyze(AnalyzeNode),

    /// COPY operation (import/export).
    Copy(CopyNode),

    /// SET session variable.
    SetSession(SetSessionNode),

    /// SHOW session variable.
    Show(ShowNode),

    /// RESET session variable.
    Reset(ResetNode),

    /// SHOW PROCEDURES command.
    ShowProcedures(ShowProceduresNode),
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

    /// Creates a procedure call node.
    #[must_use]
    pub fn procedure_call(node: ProcedureCallNode) -> Self {
        Self::ProcedureCall(Box::new(node))
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
        Self::Aggregate {
            node: Box::new(AggregateNode::new(group_by, aggregates)),
            input: Box::new(self),
        }
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

    /// Adds a window operation to this plan.
    #[must_use]
    pub fn window(self, window_exprs: Vec<(LogicalExpr, String)>) -> Self {
        Self::Window { node: WindowNode::new(window_exprs), input: Box::new(self) }
    }

    /// Adds an alias to this plan.
    #[must_use]
    pub fn alias(self, name: impl Into<String>) -> Self {
        Self::Alias { alias: name.into(), input: Box::new(self) }
    }

    /// Adds an unwind operation to this plan.
    #[must_use]
    pub fn unwind(self, list_expr: LogicalExpr, alias: impl Into<String>) -> Self {
        Self::Unwind { node: UnwindNode::new(list_expr, alias), input: Box::new(self) }
    }

    /// Creates an inner join with another plan.
    #[must_use]
    pub fn inner_join(self, right: LogicalPlan, on: LogicalExpr) -> Self {
        Self::Join {
            node: Box::new(JoinNode::inner(on)),
            left: Box::new(self),
            right: Box::new(right),
        }
    }

    /// Creates a left outer join with another plan.
    #[must_use]
    pub fn left_join(self, right: LogicalPlan, on: LogicalExpr) -> Self {
        Self::Join {
            node: Box::new(JoinNode::left(on)),
            left: Box::new(self),
            right: Box::new(right),
        }
    }

    /// Creates a cross join with another plan.
    #[must_use]
    pub fn cross_join(self, right: LogicalPlan) -> Self {
        Self::Join {
            node: Box::new(JoinNode::cross()),
            left: Box::new(self),
            right: Box::new(right),
        }
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

    /// Adds a hybrid search operation.
    #[must_use]
    pub fn hybrid_search(self, node: HybridSearchNode) -> Self {
        Self::HybridSearch { node: Box::new(node), input: Box::new(self) }
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
            | Self::Window { input, .. }
            | Self::Alias { input, .. }
            | Self::Unwind { input, .. }
            | Self::Expand { input, .. }
            | Self::PathScan { input, .. }
            | Self::ShortestPath { input, .. }
            | Self::AnnSearch { input, .. }
            | Self::VectorDistance { input, .. }
            | Self::HybridSearch { input, .. }
            | Self::Insert { input, .. }
            | Self::MergeSql { source: input, .. } => vec![input.as_ref()],

            // Binary nodes
            Self::Join { left, right, .. }
            | Self::SetOp { left, right, .. }
            | Self::RecursiveCTE { initial: left, recursive: right, .. }
            | Self::CallSubquery { input: left, subquery: right, .. } => {
                vec![left.as_ref(), right.as_ref()]
            }

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter().collect(),

            // DML with optional source input
            Self::Update { source: Some(source), .. }
            | Self::Delete { source: Some(source), .. } => vec![source.as_ref()],
            Self::Update { source: None, .. } | Self::Delete { source: None, .. } => vec![],

            // DDL nodes (no inputs)
            Self::CreateTable(_)
            | Self::AlterTable(_)
            | Self::DropTable(_)
            | Self::TruncateTable(_)
            | Self::CreateIndex(_)
            | Self::AlterIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::CreateView(_)
            | Self::DropView(_)
            | Self::CreateSchema(_)
            | Self::AlterSchema(_)
            | Self::DropSchema(_)
            | Self::CreateFunction(_)
            | Self::DropFunction(_)
            | Self::CreateTrigger(_)
            | Self::DropTrigger(_) => vec![],

            // Graph DML nodes (optional input)
            Self::GraphCreate { input: Some(input), .. }
            | Self::GraphMerge { input: Some(input), .. } => vec![input.as_ref()],
            Self::GraphCreate { input: None, .. } | Self::GraphMerge { input: None, .. } => vec![],

            // Graph DML nodes (required input)
            Self::GraphSet { input, .. }
            | Self::GraphDelete { input, .. }
            | Self::GraphRemove { input, .. }
            | Self::GraphForeach { input, .. } => vec![input.as_ref()],

            // Procedure nodes (leaf - no inputs)
            Self::ProcedureCall(_) => vec![],

            // Transaction nodes (leaf - no inputs, control flow only)
            Self::BeginTransaction(_)
            | Self::Commit(_)
            | Self::Rollback(_)
            | Self::Savepoint(_)
            | Self::ReleaseSavepoint(_)
            | Self::SetTransaction(_) => vec![],

            // Utility nodes
            Self::ExplainAnalyze(node) => vec![node.input.as_ref()],
            Self::Vacuum(_)
            | Self::Analyze(_)
            | Self::Copy(_)
            | Self::SetSession(_)
            | Self::Show(_)
            | Self::Reset(_)
            | Self::ShowProcedures(_) => vec![],
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
            | Self::Window { input, .. }
            | Self::Alias { input, .. }
            | Self::Unwind { input, .. }
            | Self::Expand { input, .. }
            | Self::PathScan { input, .. }
            | Self::ShortestPath { input, .. }
            | Self::AnnSearch { input, .. }
            | Self::VectorDistance { input, .. }
            | Self::HybridSearch { input, .. }
            | Self::Insert { input, .. }
            | Self::MergeSql { source: input, .. }
            | Self::GraphSet { input, .. }
            | Self::GraphDelete { input, .. }
            | Self::GraphRemove { input, .. }
            | Self::GraphForeach { input, .. } => vec![input.as_mut()],

            // Binary nodes
            Self::Join { left, right, .. }
            | Self::SetOp { left, right, .. }
            | Self::RecursiveCTE { initial: left, recursive: right, .. }
            | Self::CallSubquery { input: left, subquery: right, .. } => {
                vec![left.as_mut(), right.as_mut()]
            }

            // N-ary nodes
            Self::Union { inputs, .. } => inputs.iter_mut().collect(),

            // DML with optional source input
            Self::Update { source: Some(source), .. }
            | Self::Delete { source: Some(source), .. } => vec![source.as_mut()],
            Self::Update { source: None, .. } | Self::Delete { source: None, .. } => vec![],

            // DDL nodes (no inputs)
            Self::CreateTable(_)
            | Self::AlterTable(_)
            | Self::DropTable(_)
            | Self::TruncateTable(_)
            | Self::CreateIndex(_)
            | Self::AlterIndex(_)
            | Self::DropIndex(_)
            | Self::CreateCollection(_)
            | Self::DropCollection(_)
            | Self::CreateView(_)
            | Self::DropView(_)
            | Self::CreateSchema(_)
            | Self::AlterSchema(_)
            | Self::DropSchema(_)
            | Self::CreateFunction(_)
            | Self::DropFunction(_)
            | Self::CreateTrigger(_)
            | Self::DropTrigger(_) => vec![],

            // Graph DML nodes (optional input)
            Self::GraphCreate { input: Some(input), .. }
            | Self::GraphMerge { input: Some(input), .. } => vec![input.as_mut()],
            Self::GraphCreate { input: None, .. } | Self::GraphMerge { input: None, .. } => vec![],

            // Graph DML nodes (required input) - already handled above in unary nodes

            // Procedure nodes (leaf - no inputs)
            Self::ProcedureCall(_) => vec![],

            // Transaction nodes (leaf - no inputs, control flow only)
            Self::BeginTransaction(_)
            | Self::Commit(_)
            | Self::Rollback(_)
            | Self::Savepoint(_)
            | Self::ReleaseSavepoint(_)
            | Self::SetTransaction(_) => vec![],

            // Utility nodes
            Self::ExplainAnalyze(node) => vec![node.input.as_mut()],
            Self::Vacuum(_)
            | Self::Analyze(_)
            | Self::Copy(_)
            | Self::SetSession(_)
            | Self::Show(_)
            | Self::Reset(_)
            | Self::ShowProcedures(_) => vec![],
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
            Self::Window { .. } => "Window",
            Self::Alias { .. } => "Alias",
            Self::Unwind { .. } => "Unwind",
            Self::Join { .. } => "Join",
            Self::SetOp { .. } => "SetOp",
            Self::Union { .. } => "Union",
            Self::CallSubquery { .. } => "CallSubquery",
            Self::RecursiveCTE { .. } => "RecursiveCTE",
            Self::Expand { .. } => "Expand",
            Self::PathScan { .. } => "PathScan",
            Self::ShortestPath { .. } => "ShortestPath",
            Self::AnnSearch { .. } => "AnnSearch",
            Self::VectorDistance { .. } => "VectorDistance",
            Self::HybridSearch { .. } => "HybridSearch",
            Self::Insert { .. } => "Insert",
            Self::Update { .. } => "Update",
            Self::Delete { .. } => "Delete",
            Self::MergeSql { .. } => "MergeSql",
            Self::CreateTable(_) => "CreateTable",
            Self::AlterTable(_) => "AlterTable",
            Self::DropTable(_) => "DropTable",
            Self::TruncateTable(_) => "TruncateTable",
            Self::CreateIndex(_) => "CreateIndex",
            Self::AlterIndex(_) => "AlterIndex",
            Self::DropIndex(_) => "DropIndex",
            Self::CreateCollection(_) => "CreateCollection",
            Self::DropCollection(_) => "DropCollection",
            Self::CreateView(_) => "CreateView",
            Self::DropView(_) => "DropView",
            Self::CreateSchema(_) => "CreateSchema",
            Self::AlterSchema(_) => "AlterSchema",
            Self::DropSchema(_) => "DropSchema",
            Self::CreateFunction(_) => "CreateFunction",
            Self::DropFunction(_) => "DropFunction",
            Self::CreateTrigger(_) => "CreateTrigger",
            Self::DropTrigger(_) => "DropTrigger",
            Self::GraphCreate { .. } => "GraphCreate",
            Self::GraphMerge { .. } => "GraphMerge",
            Self::GraphSet { .. } => "GraphSet",
            Self::GraphDelete { .. } => "GraphDelete",
            Self::GraphRemove { .. } => "GraphRemove",
            Self::GraphForeach { .. } => "GraphForeach",
            Self::ProcedureCall(_) => "ProcedureCall",
            Self::BeginTransaction(_) => "BeginTransaction",
            Self::Commit(_) => "Commit",
            Self::Rollback(_) => "Rollback",
            Self::Savepoint(_) => "Savepoint",
            Self::ReleaseSavepoint(_) => "ReleaseSavepoint",
            Self::SetTransaction(_) => "SetTransaction",
            Self::ExplainAnalyze(_) => "ExplainAnalyze",
            Self::Vacuum(_) => "Vacuum",
            Self::Analyze(_) => "Analyze",
            Self::Copy(_) => "Copy",
            Self::SetSession(_) => "SetSession",
            Self::Show(_) => "Show",
            Self::Reset(_) => "Reset",
            Self::ShowProcedures(_) => "ShowProcedures",
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
            LogicalPlan::Window { node, .. } => {
                write!(f, "Window: ")?;
                for (i, (expr, alias)) in node.window_exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr} AS {alias}")?;
                }
            }
            LogicalPlan::Alias { alias, .. } => {
                write!(f, "Alias: {alias}")?;
            }
            LogicalPlan::Unwind { node, .. } => {
                write!(f, "Unwind: {} AS {}", node.list_expr, node.alias)?;
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
            LogicalPlan::CallSubquery { node, .. } => {
                write!(f, "CallSubquery")?;
                if !node.imported_variables.is_empty() {
                    write!(f, " WITH {}", node.imported_variables.join(", "))?;
                }
            }
            LogicalPlan::RecursiveCTE { node, .. } => {
                write!(f, "RecursiveCTE: {}", node.name)?;
                if !node.columns.is_empty() {
                    write!(f, " ({})", node.columns.join(", "))?;
                }
                write!(f, " {}", if node.union_all { "UNION ALL" } else { "UNION" })?;
                if let Some(max) = node.max_iterations {
                    write!(f, " [max_iter: {max}]")?;
                }
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
            LogicalPlan::ShortestPath { node, .. } => {
                let func = if node.find_all { "allShortestPaths" } else { "shortestPath" };
                write!(f, "{}: ({}){}({})", func, node.src_var, node.direction, node.dst_var)?;
                if !node.edge_types.is_empty() {
                    write!(f, " [types: {}]", node.edge_types.join("|"))?;
                }
                if let Some(max) = node.max_length {
                    write!(f, " [max: {max}]")?;
                }
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
            LogicalPlan::HybridSearch { node, .. } => {
                write!(f, "HybridSearch: {} components, k={}", node.num_components(), node.k)?;
                for (i, comp) in node.components.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    } else {
                        write!(f, " [")?;
                    }
                    write!(
                        f,
                        "{} {} (w={:.2})",
                        comp.vector_column,
                        comp.metric.operator(),
                        comp.weight
                    )?;
                }
                if !node.components.is_empty() {
                    write!(f, "]")?;
                }
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
            LogicalPlan::MergeSql { target_table, on_condition, clauses, .. } => {
                write!(f, "MergeSql: {target_table} ON {on_condition}")?;
                write!(f, " ({} clauses)", clauses.len())?;
            }
            LogicalPlan::CreateTable(node) => {
                write!(f, "CreateTable: {}", node.name)?;
                if node.if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " ({} columns)", node.columns.len())?;
            }
            LogicalPlan::AlterTable(node) => {
                write!(f, "AlterTable: {}", node.name)?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                write!(f, " ({} actions)", node.actions.len())?;
            }
            LogicalPlan::DropTable(node) => {
                write!(f, "DropTable: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::TruncateTable(node) => {
                write!(f, "TruncateTable: {}", node.names.join(", "))?;
                if node.restart_identity {
                    write!(f, " RESTART IDENTITY")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::CreateIndex(node) => {
                write!(f, "CreateIndex: {} ON {}", node.name, node.table)?;
                if node.unique {
                    write!(f, " UNIQUE")?;
                }
                if let Some(method) = &node.using {
                    write!(f, " USING {method}")?;
                }
            }
            LogicalPlan::AlterIndex(node) => {
                write!(f, "AlterIndex: {}", node.name)?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
            }
            LogicalPlan::DropIndex(node) => {
                write!(f, "DropIndex: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
            }
            LogicalPlan::CreateCollection(node) => {
                write!(f, "CreateCollection: {}", node.name)?;
                if node.if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                write!(f, " ({} vectors)", node.vectors.len())?;
            }
            LogicalPlan::DropCollection(node) => {
                write!(f, "DropCollection: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::CreateView(node) => {
                write!(f, "CreateView: {}", node.name)?;
                if node.or_replace {
                    write!(f, " OR REPLACE")?;
                }
                if !node.columns.is_empty() {
                    let cols: Vec<_> = node.columns.iter().map(|c| c.name.as_str()).collect();
                    write!(f, " ({})", cols.join(", "))?;
                }
            }
            LogicalPlan::DropView(node) => {
                write!(f, "DropView: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::CreateSchema(node) => {
                write!(f, "CreateSchema: {}", node.name)?;
                if node.if_not_exists {
                    write!(f, " IF NOT EXISTS")?;
                }
                if let Some(ref auth) = node.authorization {
                    write!(f, " AUTHORIZATION {auth}")?;
                }
            }
            LogicalPlan::AlterSchema(node) => {
                write!(f, "AlterSchema: {}", node.name)?;
                match &node.action {
                    crate::ast::AlterSchemaAction::OwnerTo(owner) => {
                        write!(f, " OWNER TO {}", owner.name)?;
                    }
                    crate::ast::AlterSchemaAction::RenameTo(new_name) => {
                        write!(f, " RENAME TO {}", new_name.name)?;
                    }
                }
            }
            LogicalPlan::DropSchema(node) => {
                write!(f, "DropSchema: {}", node.names.join(", "))?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::CreateFunction(node) => {
                write!(f, "CreateFunction: {}", node.name)?;
                if node.or_replace {
                    write!(f, " OR REPLACE")?;
                }
                write!(f, " ({} params)", node.parameters.len())?;
                write!(f, " RETURNS {:?}", node.returns)?;
                write!(f, " LANGUAGE {}", node.language)?;
            }
            LogicalPlan::DropFunction(node) => {
                write!(f, "DropFunction: {}", node.name)?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if !node.arg_types.is_empty() {
                    let types: Vec<_> = node.arg_types.iter().map(|t| format!("{t:?}")).collect();
                    write!(f, " ({})", types.join(", "))?;
                }
            }
            LogicalPlan::CreateTrigger(node) => {
                write!(f, "CreateTrigger: {}", node.name)?;
                if node.or_replace {
                    write!(f, " OR REPLACE")?;
                }
                write!(f, " {} ", node.timing)?;
                let events: Vec<_> = node.events.iter().map(ToString::to_string).collect();
                write!(f, "{}", events.join(" OR "))?;
                write!(f, " ON {}", node.table)?;
                write!(f, " {}", node.for_each)?;
                write!(f, " EXECUTE {}", node.function)?;
            }
            LogicalPlan::DropTrigger(node) => {
                write!(f, "DropTrigger: {} ON {}", node.name, node.table)?;
                if node.if_exists {
                    write!(f, " IF EXISTS")?;
                }
                if node.cascade {
                    write!(f, " CASCADE")?;
                }
            }
            LogicalPlan::GraphCreate { node, .. } => {
                write!(
                    f,
                    "GraphCreate: {} nodes, {} relationships",
                    node.nodes.len(),
                    node.relationships.len()
                )?;
            }
            LogicalPlan::GraphMerge { node, .. } => {
                let pattern_desc = match &node.pattern {
                    super::graph::MergePatternSpec::Node { variable, labels, .. } => {
                        let labels_str = if labels.is_empty() {
                            String::new()
                        } else {
                            format!(":{}", labels.join(":"))
                        };
                        format!("({}{})", variable, labels_str)
                    }
                    super::graph::MergePatternSpec::Relationship {
                        start_var,
                        rel_type,
                        end_var,
                        ..
                    } => {
                        format!("({start_var})-[:{rel_type}]->({end_var})")
                    }
                };
                write!(f, "GraphMerge: {pattern_desc}")?;
            }
            LogicalPlan::GraphSet { node, .. } => {
                write!(f, "GraphSet: {} actions", node.set_actions.len())?;
            }
            LogicalPlan::GraphDelete { node, .. } => {
                write!(f, "GraphDelete: {}", node.variables.join(", "))?;
                if node.detach {
                    write!(f, " DETACH")?;
                }
            }
            LogicalPlan::GraphRemove { node, .. } => {
                write!(f, "GraphRemove: {} actions", node.remove_actions.len())?;
            }
            LogicalPlan::GraphForeach { node, .. } => {
                write!(
                    f,
                    "GraphForeach: {} IN {} ({} actions)",
                    node.variable,
                    node.list_expr,
                    node.actions.len()
                )?;
            }
            LogicalPlan::ProcedureCall(node) => {
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
            // Transaction nodes
            LogicalPlan::BeginTransaction(node) => {
                write!(f, "BeginTransaction")?;
                if let Some(level) = &node.isolation_level {
                    write!(f, " ISOLATION LEVEL {level}")?;
                }
                if let Some(mode) = &node.access_mode {
                    write!(f, " {mode}")?;
                }
                if node.deferred {
                    write!(f, " DEFERRED")?;
                }
            }
            LogicalPlan::Commit(_) => {
                write!(f, "Commit")?;
            }
            LogicalPlan::Rollback(node) => {
                write!(f, "Rollback")?;
                if let Some(sp) = &node.to_savepoint {
                    write!(f, " TO SAVEPOINT {sp}")?;
                }
            }
            LogicalPlan::Savepoint(node) => {
                write!(f, "Savepoint: {}", node.name)?;
            }
            LogicalPlan::ReleaseSavepoint(node) => {
                write!(f, "ReleaseSavepoint: {}", node.name)?;
            }
            LogicalPlan::SetTransaction(node) => {
                write!(f, "SetTransaction")?;
                if let Some(level) = &node.isolation_level {
                    write!(f, " ISOLATION LEVEL {level}")?;
                }
                if let Some(mode) = &node.access_mode {
                    write!(f, " {mode}")?;
                }
            }
            // Utility nodes
            LogicalPlan::ExplainAnalyze(node) => {
                write!(f, "ExplainAnalyze")?;
                write!(f, " (format: {})", node.format)?;
                if node.verbose {
                    write!(f, " VERBOSE")?;
                }
                if node.buffers {
                    write!(f, " BUFFERS")?;
                }
            }
            LogicalPlan::Vacuum(node) => {
                write!(f, "Vacuum")?;
                if node.full {
                    write!(f, " FULL")?;
                }
                if node.analyze {
                    write!(f, " ANALYZE")?;
                }
                if let Some(table) = &node.table {
                    write!(f, " {table}")?;
                }
            }
            LogicalPlan::Analyze(node) => {
                write!(f, "Analyze")?;
                if let Some(table) = &node.table {
                    write!(f, " {table}")?;
                }
                if !node.columns.is_empty() {
                    write!(f, " ({})", node.columns.join(", "))?;
                }
            }
            LogicalPlan::Copy(node) => {
                write!(f, "Copy")?;
                if node.is_export() {
                    write!(f, " TO")?;
                } else {
                    write!(f, " FROM")?;
                }
                write!(f, " (format: {})", node.options.format)?;
            }
            LogicalPlan::SetSession(node) => {
                write!(f, "SetSession: {}", node.name)?;
                if node.local {
                    write!(f, " LOCAL")?;
                }
            }
            LogicalPlan::Show(node) => {
                write!(f, "Show")?;
                if let Some(name) = &node.name {
                    write!(f, ": {name}")?;
                } else {
                    write!(f, " ALL")?;
                }
            }
            LogicalPlan::Reset(node) => {
                write!(f, "Reset")?;
                if let Some(name) = &node.name {
                    write!(f, ": {name}")?;
                } else {
                    write!(f, " ALL")?;
                }
            }
            LogicalPlan::ShowProcedures(node) => {
                write!(f, "ShowProcedures")?;
                if node.executable {
                    write!(f, " EXECUTABLE")?;
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

// ========== ON CONFLICT Types ==========

/// ON CONFLICT clause for INSERT statements (logical plan version).
///
/// This represents the PostgreSQL-style upsert syntax:
/// - `ON CONFLICT DO NOTHING` - skip conflicting rows
/// - `ON CONFLICT DO UPDATE SET ...` - update existing rows
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalOnConflict {
    /// The conflict target (columns or constraint).
    pub target: LogicalConflictTarget,
    /// The action to take on conflict.
    pub action: LogicalConflictAction,
}

impl LogicalOnConflict {
    /// Creates a new ON CONFLICT DO NOTHING clause.
    #[must_use]
    pub fn do_nothing(target: LogicalConflictTarget) -> Self {
        Self { target, action: LogicalConflictAction::DoNothing }
    }

    /// Creates a new ON CONFLICT DO UPDATE clause.
    #[must_use]
    pub fn do_update(
        target: LogicalConflictTarget,
        assignments: Vec<(String, LogicalExpr)>,
        where_clause: Option<LogicalExpr>,
    ) -> Self {
        Self { target, action: LogicalConflictAction::DoUpdate { assignments, where_clause } }
    }
}

/// Target for ON CONFLICT clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogicalConflictTarget {
    /// Specific columns that form a unique constraint.
    Columns(Vec<String>),
    /// A named constraint.
    Constraint(String),
}

impl LogicalConflictTarget {
    /// Creates a column-based conflict target.
    #[must_use]
    pub fn columns(cols: Vec<String>) -> Self {
        Self::Columns(cols)
    }

    /// Creates a constraint-based conflict target.
    #[must_use]
    pub fn constraint(name: impl Into<String>) -> Self {
        Self::Constraint(name.into())
    }
}

/// Action for ON CONFLICT clause.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalConflictAction {
    /// DO NOTHING - skip conflicting rows.
    DoNothing,
    /// DO UPDATE SET ... - update existing rows.
    DoUpdate {
        /// The assignments (column, expression).
        assignments: Vec<(String, LogicalExpr)>,
        /// Optional WHERE clause for the update.
        where_clause: Option<LogicalExpr>,
    },
}

// ============================================================================
// MERGE Statement Types
// ============================================================================

/// A WHEN clause in a SQL MERGE statement.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalMergeClause {
    /// Whether this is a MATCHED or NOT MATCHED clause.
    pub match_type: LogicalMergeMatchType,
    /// Optional additional condition (AND clause).
    pub condition: Option<LogicalExpr>,
    /// The action to perform.
    pub action: LogicalMergeAction,
}

impl LogicalMergeClause {
    /// Creates a new WHEN MATCHED ... UPDATE clause.
    #[must_use]
    pub fn matched_update(assignments: Vec<(String, LogicalExpr)>) -> Self {
        Self {
            match_type: LogicalMergeMatchType::Matched,
            condition: None,
            action: LogicalMergeAction::Update { assignments },
        }
    }

    /// Creates a new WHEN MATCHED ... DELETE clause.
    #[must_use]
    pub fn matched_delete() -> Self {
        Self {
            match_type: LogicalMergeMatchType::Matched,
            condition: None,
            action: LogicalMergeAction::Delete,
        }
    }

    /// Creates a new WHEN NOT MATCHED ... INSERT clause.
    #[must_use]
    pub fn not_matched_insert(columns: Vec<String>, values: Vec<LogicalExpr>) -> Self {
        Self {
            match_type: LogicalMergeMatchType::NotMatched,
            condition: None,
            action: LogicalMergeAction::Insert { columns, values },
        }
    }

    /// Creates a new WHEN NOT MATCHED BY SOURCE ... DELETE clause.
    #[must_use]
    pub fn not_matched_by_source_delete() -> Self {
        Self {
            match_type: LogicalMergeMatchType::NotMatchedBySource,
            condition: None,
            action: LogicalMergeAction::Delete,
        }
    }

    /// Creates a new WHEN NOT MATCHED BY SOURCE ... UPDATE clause.
    #[must_use]
    pub fn not_matched_by_source_update(assignments: Vec<(String, LogicalExpr)>) -> Self {
        Self {
            match_type: LogicalMergeMatchType::NotMatchedBySource,
            condition: None,
            action: LogicalMergeAction::Update { assignments },
        }
    }

    /// Adds a condition to this clause.
    #[must_use]
    pub fn with_condition(mut self, condition: LogicalExpr) -> Self {
        self.condition = Some(condition);
        self
    }
}

/// The match type for a MERGE WHEN clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalMergeMatchType {
    /// WHEN MATCHED - source row matches target row.
    Matched,
    /// WHEN NOT MATCHED (BY TARGET) - source row has no match in target.
    NotMatched,
    /// WHEN NOT MATCHED BY SOURCE - target row has no match in source.
    NotMatchedBySource,
}

/// The action to perform in a MERGE WHEN clause.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalMergeAction {
    /// UPDATE SET ... action.
    Update {
        /// The assignments (column, expression).
        assignments: Vec<(String, LogicalExpr)>,
    },
    /// DELETE action.
    Delete,
    /// INSERT (columns) VALUES (values) action.
    Insert {
        /// Target columns for the insert.
        columns: Vec<String>,
        /// Values to insert.
        values: Vec<LogicalExpr>,
    },
    /// DO NOTHING action (skip the row).
    DoNothing,
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
