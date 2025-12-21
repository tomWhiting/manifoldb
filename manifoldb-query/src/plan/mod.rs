//! Query planning.
//!
//! This module provides logical and physical query planning.
//!
//! # Overview
//!
//! Query planning transforms parsed SQL AST into executable plans:
//!
//! 1. **Logical Plan**: Represents what to compute (relations, predicates)
//! 2. **Physical Plan**: Represents how to compute it (algorithms, access paths)
//!
//! # Example
//!
//! ```
//! use manifoldb_query::parser::parse_single_statement;
//! use manifoldb_query::plan::logical::PlanBuilder;
//!
//! let stmt = parse_single_statement("SELECT * FROM users WHERE id = 1").unwrap();
//! let plan = PlanBuilder::new().build_statement(&stmt).unwrap();
//! println!("{}", plan.display_tree());
//! ```

pub mod logical;
pub mod physical;

// Re-export commonly used types
pub use logical::{
    AggregateFunction, AggregateNode, AnnSearchNode, DistinctNode, ExpandDirection, ExpandNode,
    FilterNode, JoinNode, JoinType, LimitNode, LogicalExpr, LogicalPlan, PathScanNode, PlanBuilder,
    PlanError, PlanResult, ProjectNode, ScanNode, SetOpNode, SetOpType, SortNode, SortOrder,
    UnionNode, ValuesNode, VectorDistanceNode,
};
