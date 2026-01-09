//! Physical query plan.
//!
//! This module defines the physical query plan that represents
//! the concrete execution strategy for a query.
//!
//! # Overview
//!
//! Physical plans describe *how* to execute a query, in contrast to
//! logical plans which describe *what* to compute. The physical plan
//! specifies concrete algorithms (e.g., hash join vs nested loop),
//! access methods (e.g., index scan vs full scan), and execution order.
//!
//! # Plan Types
//!
//! - **Scan operations**: `FullScan`, `IndexScan`, `IndexRangeScan`
//! - **Join operations**: `NestedLoopJoin`, `HashJoin`, `MergeJoin`
//! - **Vector operations**: `HnswSearch`, `BruteForceSearch`
//! - **Graph operations**: `GraphExpand`, `GraphPathScan`
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_query::plan::physical::{PhysicalPlan, PhysicalPlanner};
//! use manifoldb_query::plan::LogicalPlan;
//!
//! let logical = LogicalPlan::scan("users")
//!     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
//!
//! let physical = PhysicalPlanner::new().plan(&logical)?;
//! println!("{}", physical.display_tree());
//! ```

mod builder;
mod cost;
mod node;

pub use builder::{IndexInfo, IndexType, PhysicalPlanner, PlannerCatalog, TableStats};
pub use cost::{Cost, CostModel};
pub use node::{
    BruteForceSearchNode, FilterExecNode, FullScanNode, GraphExpandExecNode, HashAggregateNode,
    HashJoinNode, HnswSearchNode, HybridSearchComponentNode, HybridSearchNode, IndexRangeScanNode,
    IndexScanNode, JoinOrder, LimitExecNode, MergeJoinNode, NestedLoopJoinNode, PhysicalPlan,
    PhysicalScoreCombinationMethod, ProjectExecNode, SortExecNode, SortMergeAggregateNode,
    WindowExecNode, WindowFunctionExpr,
};
