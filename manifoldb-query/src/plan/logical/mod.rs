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
mod expr;
mod graph;
mod node;
mod relational;
mod validate;
mod vector;

pub use builder::PlanBuilder;
pub use expr::{AggregateFunction, LogicalExpr, ScalarFunction, SortOrder};
pub use graph::{ExpandDirection, ExpandLength, ExpandNode, PathScanNode, PathStep};
pub use node::LogicalPlan;
pub use relational::{
    AggregateNode, DistinctNode, FilterNode, JoinNode, JoinType, LimitNode, ProjectNode, ScanNode,
    SetOpNode, SetOpType, SortNode, UnionNode, ValuesNode,
};
pub use validate::{check_no_cycles, validate_plan, PlanError, PlanResult};
pub use vector::{AnnSearchNode, AnnSearchParams, VectorDistanceNode};
