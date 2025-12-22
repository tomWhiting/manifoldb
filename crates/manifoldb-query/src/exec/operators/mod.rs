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
//! - **Vector operators**: [`vector`] - Vector search
//! - **Analytics operators**: [`analytics`] - Graph analytics (PageRank, centrality, community)

pub mod aggregate;
pub mod analytics;
pub mod filter;
pub mod graph;
pub mod join;
pub mod limit;
pub mod project;
pub mod scan;
pub mod set_ops;
pub mod sort;
pub mod values;
pub mod vector;

// Re-exports for convenience
pub use aggregate::{HashAggregateOp, SortMergeAggregateOp};
pub use analytics::{
    BetweennessCentralityOp, BetweennessCentralityOpConfig, CommunityDetectionOp,
    CommunityDetectionOpConfig, PageRankOp, PageRankOpConfig,
};
pub use filter::FilterOp;
pub use graph::{GraphExpandOp, GraphPathScanOp};
pub use join::{HashJoinOp, MergeJoinOp, NestedLoopJoinOp};
pub use limit::LimitOp;
pub use project::ProjectOp;
pub use scan::{FullScanOp, IndexRangeScanOp, IndexScanOp};
pub use set_ops::{SetOpOp, UnionOp};
pub use sort::SortOp;
pub use values::{EmptyOp, ValuesOp};
pub use vector::{BruteForceSearchOp, HnswSearchOp};
