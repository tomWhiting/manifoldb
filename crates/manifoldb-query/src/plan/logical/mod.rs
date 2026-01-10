//! Logical query plan.
//!
//! This module defines the logical query plan that represents
//! query semantics independent of execution strategy.
//!
//! # Overview
//!
//! A logical plan is a tree of operators that describes how to compute
//! a query result. It focuses on *what* to compute, not *how* to compute it.
//!
//! # Plan Nodes
//!
//! The plan supports three categories of operations:
//!
//! - **Relational**: Standard SQL operations (`Scan`, `Filter`, `Project`, `Join`, etc.)
//! - **Graph**: Graph traversal operations (`Expand`, `PathScan`)
//! - **Vector**: Vector similarity search (`AnnSearch`, `VectorDistance`)
//!
//! # Example
//!
//! ```
//! use manifoldb_query::plan::logical::{LogicalPlan, LogicalExpr, SortOrder};
//!
//! // SELECT * FROM users WHERE age > 21 ORDER BY name LIMIT 10
//! let plan = LogicalPlan::scan("users")
//!     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)))
//!     .sort(vec![SortOrder::asc(LogicalExpr::column("name"))])
//!     .limit(10);
//! ```

mod builder;
mod ddl;
mod expr;
mod graph;
mod node;
mod procedure;
mod relational;
mod schema;
mod transaction;
mod type_infer;
mod types;
pub mod utility;
mod validate;
mod vector;

pub use builder::{PlanBuilder, ViewDefinition};
pub use ddl::{
    AlterIndexAction, AlterIndexNode, AlterSchemaNode, AlterTableNode, CreateCollectionNode,
    CreateFunctionNode, CreateIndexNode, CreateSchemaNode, CreateTableNode, CreateTriggerNode,
    CreateViewNode, DropCollectionNode, DropFunctionNode, DropIndexNode, DropSchemaNode,
    DropTableNode, DropTriggerNode, DropViewNode, TruncateTableNode,
};
pub use expr::{
    AggregateFunction, HybridCombinationMethod, HybridExprComponent, LogicalExpr,
    LogicalMapProjectionItem, ScalarFunction, SortOrder,
};
pub use graph::{
    CreateNodeSpec, CreateRelSpec, ExpandDirection, ExpandLength, ExpandNode, GraphCreateNode,
    GraphDeleteNode, GraphForeachAction, GraphForeachNode, GraphMergeNode, GraphRemoveAction,
    GraphRemoveNode, GraphSetAction, GraphSetNode, MergePatternSpec, PathScanNode, PathStep,
    ShortestPathNode, ShortestPathWeight,
};
pub use node::LogicalPlan;
pub use procedure::{ProcedureCallNode, YieldColumn};
pub use relational::{
    AggregateNode, CallSubqueryNode, DistinctNode, FilterNode, JoinNode, JoinType, LimitNode,
    ProjectNode, RecursiveCTENode, ScanNode, SetOpNode, SetOpType, SortNode, UnionNode, UnwindNode,
    ValuesNode, WindowNode,
};
pub use schema::{EmptyCatalog, SchemaCatalog, SchemaProvider};
pub use transaction::{
    BeginTransactionNode, CommitNode, ReleaseSavepointNode, RollbackNode, SavepointNode,
    SetTransactionNode,
};
pub use type_infer::{TypeError, TypeResult};
pub use types::{PlanType, Schema, TypeContext, TypedColumn};
pub use utility::{
    AnalyzeNode, CopyNode, ExplainAnalyzeNode, ExplainFormat, ResetNode, SetSessionNode, ShowNode,
    VacuumNode,
};
pub use validate::{check_no_cycles, validate_plan, validate_with_schema, PlanError, PlanResult};
pub use vector::{
    AnnSearchNode, AnnSearchParams, HybridSearchComponent, HybridSearchNode,
    ScoreCombinationMethod, VectorDistanceNode,
};
