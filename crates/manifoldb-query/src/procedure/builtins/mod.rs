//! Built-in procedures for ManifoldDB.
//!
//! This module provides built-in procedures that can be called via CALL statements:
//!
//! ## Traversal Algorithms
//!
//! - `algo.bfs` - Breadth-first search traversal (level by level)
//! - `algo.dfs` - Depth-first search traversal (branch exploration)
//!
//! ## Centrality Algorithms
//!
//! - `algo.pageRank` - Computes PageRank scores for graph nodes
//! - `algo.betweennessCentrality` - Measures bridge/bottleneck nodes
//! - `algo.closenessCentrality` - Measures distance-based centrality
//! - `algo.degreeCentrality` - Measures node connection counts
//! - `algo.eigenvectorCentrality` - Measures influence in networks
//!
//! ## Community Detection
//!
//! - `algo.louvain` - Detects communities using Louvain algorithm (modularity optimization)
//! - `algo.labelPropagation` - Detects communities using Label Propagation
//! - `algo.connectedComponents` - Finds weakly or strongly connected components
//! - `algo.stronglyConnectedComponents` - Finds strongly connected components
//!
//! ## Path Finding
//!
//! - `algo.shortestPath` - Finds the shortest path between two nodes (BFS)
//! - `algo.dijkstra` - Weighted shortest path using Dijkstra's algorithm
//! - `algo.astar` - Weighted shortest path using A* with heuristics
//! - `algo.allShortestPaths` - Finds all shortest paths between two nodes
//! - `algo.sssp` - Single-source shortest paths to all reachable nodes

mod all_shortest_paths;
mod astar;
mod betweenness;
mod bfs;
mod closeness;
mod community;
mod connected;
mod degree;
mod dfs;
mod dijkstra;
mod eigenvector;
mod louvain;
mod pagerank;
mod shortest_path;
mod sssp;

pub use all_shortest_paths::{execute_all_shortest_paths_with_tx, AllShortestPathsProcedure};
pub use astar::{execute_astar_with_tx, AStarProcedure};
pub use betweenness::{execute_betweenness_with_tx, BetweennessCentralityProcedure};
pub use bfs::{execute_bfs_with_tx, BfsProcedure};
pub use closeness::{execute_closeness_with_tx, ClosenessCentralityProcedure};
pub use community::{execute_label_propagation_with_tx, LabelPropagationProcedure};
pub use connected::{
    execute_connected_components_with_tx, execute_strongly_connected_with_tx,
    ConnectedComponentsProcedure, StronglyConnectedComponentsProcedure,
};
pub use degree::{execute_degree_with_tx, DegreeCentralityProcedure};
pub use dfs::{execute_dfs_with_tx, DfsProcedure};
pub use dijkstra::{execute_dijkstra_with_tx, DijkstraProcedure};
pub use eigenvector::{execute_eigenvector_with_tx, EigenvectorCentralityProcedure};
pub use louvain::{execute_louvain_with_tx, LouvainProcedure};
pub use pagerank::{execute_pagerank_with_tx, PageRankProcedure};
pub use shortest_path::{execute_shortest_path_with_tx, ShortestPathProcedure};
pub use sssp::{execute_sssp_with_tx, SSSPProcedure};

#[allow(unused_imports)] // Trait is used for Arc<dyn Procedure> coercion
use super::traits::Procedure;
use super::ProcedureRegistry;
use std::sync::Arc;

/// Registers all built-in procedures with the given registry.
pub fn register_builtins(registry: &mut ProcedureRegistry) {
    // Traversal algorithms
    registry.register(Arc::new(BfsProcedure));
    registry.register(Arc::new(DfsProcedure));

    // Centrality algorithms
    registry.register(Arc::new(PageRankProcedure));
    registry.register(Arc::new(BetweennessCentralityProcedure));
    registry.register(Arc::new(ClosenessCentralityProcedure));
    registry.register(Arc::new(DegreeCentralityProcedure));
    registry.register(Arc::new(EigenvectorCentralityProcedure));

    // Community detection
    registry.register(Arc::new(LouvainProcedure));
    registry.register(Arc::new(LabelPropagationProcedure));
    registry.register(Arc::new(ConnectedComponentsProcedure));
    registry.register(Arc::new(StronglyConnectedComponentsProcedure));

    // Path finding
    registry.register(Arc::new(ShortestPathProcedure));
    registry.register(Arc::new(DijkstraProcedure));
    registry.register(Arc::new(AStarProcedure));
    registry.register(Arc::new(AllShortestPathsProcedure));
    registry.register(Arc::new(SSSPProcedure));
}
