//! Concrete operator implementations.
//!
//! This module contains the implementations of all physical operators.
//!
//! # Operator Categories
//!
//! - **Scan operators**: [`scan`] - Table and index scans
//! - **Filter operators**: [`filter`] - Predicate evaluation
//! - **Project operators**: [`project`] - Column projection
//! - **Join operators**: [`join`] - Join algorithms
//! - **Aggregate operators**: [`aggregate`] - Aggregations
//! - **Sort operators**: [`sort`] - Sorting
//! - **Limit operators**: [`limit`] - Limit/offset
//! - **Set operators**: [`set_ops`] - UNION, INTERSECT, EXCEPT
//! - **Graph operators**: [`graph`] - Graph traversal
//! - **Graph mutation operators**: [`graph_create`] - CREATE, MERGE, etc.
//! - **Vector operators**: [`vector`] - Vector search
//! - **Analytics operators**: [`analytics`] - Graph analytics (PageRank, centrality, community)

pub mod aggregate;
pub mod analytics;
pub mod filter;
pub mod graph;
pub mod graph_create;
pub mod join;
pub mod limit;
pub mod project;
pub mod recursive_cte;
pub mod scan;
pub mod set_ops;
pub mod sort;
pub mod unwind;
pub mod values;
pub mod vector;
pub mod window;

// Re-exports for convenience
pub use aggregate::{HashAggregateOp, SortMergeAggregateOp};
pub use analytics::{
    BetweennessCentralityOp, BetweennessCentralityOpConfig, CommunityDetectionOp,
    CommunityDetectionOpConfig, PageRankOp, PageRankOpConfig,
};
pub use filter::FilterOp;
pub use graph::{GraphExpandOp, GraphPathScanOp};
pub use graph_create::GraphCreateOp;
pub use join::{HashJoinOp, MergeJoinOp, NestedLoopJoinOp};
pub use limit::LimitOp;
pub use project::ProjectOp;
pub use recursive_cte::RecursiveCTEOp;
pub use scan::{FullScanOp, IndexRangeScanOp, IndexScanOp};
pub use set_ops::{SetOpOp, UnionOp};
pub use sort::SortOp;
pub use unwind::UnwindOp;
pub use values::{EmptyOp, ValuesOp};
pub use vector::{BruteForceSearchOp, HnswSearchOp};
pub use window::WindowOp;
