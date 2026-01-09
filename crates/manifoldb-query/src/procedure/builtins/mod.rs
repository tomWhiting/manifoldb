//! Built-in procedures for ManifoldDB.
//!
//! This module provides built-in procedures that can be called via CALL statements:
//!
//! - `algo.pageRank` - Computes PageRank scores for graph nodes
//! - `algo.shortestPath` - Finds the shortest path between two nodes

mod pagerank;
mod shortest_path;

pub use pagerank::{execute_pagerank_with_tx, PageRankProcedure};
pub use shortest_path::{execute_shortest_path_with_tx, ShortestPathProcedure};

#[allow(unused_imports)] // Trait is used for Arc<dyn Procedure> coercion
use super::traits::Procedure;
use super::ProcedureRegistry;
use std::sync::Arc;

/// Registers all built-in procedures with the given registry.
pub fn register_builtins(registry: &mut ProcedureRegistry) {
    registry.register(Arc::new(PageRankProcedure));
    registry.register(Arc::new(ShortestPathProcedure));
}
