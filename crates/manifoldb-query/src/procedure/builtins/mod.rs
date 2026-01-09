//! Built-in procedures for ManifoldDB.
//!
//! This module provides built-in procedures that can be called via CALL statements:
//!
//! ## Centrality Algorithms
//!
//! - `algo.pageRank` - Computes PageRank scores for graph nodes
//! - `algo.betweennessCentrality` - Measures bridge/bottleneck nodes
//! - `algo.closenessCentrality` - Measures distance-based centrality
//! - `algo.degreeCentrality` - Measures node connection counts
//! - `algo.eigenvectorCentrality` - Measures influence in networks
//!
//! ## Path Finding
//!
//! - `algo.shortestPath` - Finds the shortest path between two nodes

mod betweenness;
mod closeness;
mod degree;
mod eigenvector;
mod pagerank;
mod shortest_path;

pub use betweenness::{execute_betweenness_with_tx, BetweennessCentralityProcedure};
pub use closeness::{execute_closeness_with_tx, ClosenessCentralityProcedure};
pub use degree::{execute_degree_with_tx, DegreeCentralityProcedure};
pub use eigenvector::{execute_eigenvector_with_tx, EigenvectorCentralityProcedure};
pub use pagerank::{execute_pagerank_with_tx, PageRankProcedure};
pub use shortest_path::{execute_shortest_path_with_tx, ShortestPathProcedure};

#[allow(unused_imports)] // Trait is used for Arc<dyn Procedure> coercion
use super::traits::Procedure;
use super::ProcedureRegistry;
use std::sync::Arc;

/// Registers all built-in procedures with the given registry.
pub fn register_builtins(registry: &mut ProcedureRegistry) {
    // Centrality algorithms
    registry.register(Arc::new(PageRankProcedure));
    registry.register(Arc::new(BetweennessCentralityProcedure));
    registry.register(Arc::new(ClosenessCentralityProcedure));
    registry.register(Arc::new(DegreeCentralityProcedure));
    registry.register(Arc::new(EigenvectorCentralityProcedure));

    // Path finding
    registry.register(Arc::new(ShortestPathProcedure));
}
