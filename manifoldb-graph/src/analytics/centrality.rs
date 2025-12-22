//! Betweenness Centrality implementation using Brandes algorithm.
//!
//! Betweenness centrality measures the extent to which a node lies on paths
//! between other nodes. Nodes with high betweenness centrality act as bridges
//! or bottlenecks in the network.
//!
//! # Algorithm
//!
//! This module implements Brandes' algorithm (2001) for efficient computation
//! of betweenness centrality. The algorithm runs in O(V*E) time for unweighted
//! graphs and O(V*E + V^2*log(V)) for weighted graphs.
//!
//! # Formula
//!
//! BC(v) = Σ (σ_st(v) / σ_st) for all s≠v≠t
//!
//! Where:
//! - σ_st is the total number of shortest paths from s to t
//! - σ_st(v) is the number of those paths passing through v
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{BetweennessCentrality, BetweennessCentralityConfig};
//!
//! let config = BetweennessCentralityConfig::default();
//! let result = BetweennessCentrality::compute(&tx, &config)?;
//!
//! // Find the most central nodes (bridges/bottlenecks)
//! for (node, score) in result.top_n(10) {
//!     println!("Node {:?} has betweenness centrality {:.4}", node, score);
//! }
//! ```

use std::collections::{HashMap, VecDeque};

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};
use crate::traversal::Direction;

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Betweenness Centrality computation.
#[derive(Debug, Clone)]
pub struct BetweennessCentralityConfig {
    /// Whether to normalize centrality values to [0, 1].
    /// Default: true
    pub normalize: bool,

    /// Direction of edges to follow.
    /// Default: Both (treat as undirected)
    pub direction: Direction,

    /// Whether to include endpoints in the centrality calculation.
    /// Default: false
    pub include_endpoints: bool,

    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,
}

impl Default for BetweennessCentralityConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            direction: Direction::Both,
            include_endpoints: false,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
        }
    }
}

impl BetweennessCentralityConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to normalize centrality values.
    ///
    /// When normalized, values are scaled to [0, 1] by dividing by
    /// (n-1)*(n-2)/2 for undirected or (n-1)*(n-2) for directed graphs.
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the direction to follow edges.
    ///
    /// - `Outgoing`: Follow edges in their natural direction
    /// - `Incoming`: Follow edges in reverse
    /// - `Both`: Treat graph as undirected (default)
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set whether to include endpoints in centrality calculation.
    ///
    /// When true, source and target nodes also contribute to the
    /// centrality of intermediate nodes.
    pub const fn with_include_endpoints(mut self, include: bool) -> Self {
        self.include_endpoints = include;
        self
    }

    /// Set the maximum number of nodes allowed.
    ///
    /// If the graph has more nodes than this limit, the algorithm will
    /// return a [`GraphError::GraphTooLarge`] error instead of attempting
    /// to allocate potentially gigabytes of memory.
    ///
    /// Set to `None` to disable the check (use with caution).
    ///
    /// [`GraphError::GraphTooLarge`]: crate::store::GraphError::GraphTooLarge
    pub const fn with_max_graph_nodes(mut self, limit: Option<usize>) -> Self {
        self.max_graph_nodes = limit;
        self
    }
}

/// Result of Betweenness Centrality computation.
#[derive(Debug, Clone)]
pub struct CentralityResult {
    /// Centrality scores for each node.
    pub scores: HashMap<EntityId, f64>,

    /// Whether scores are normalized.
    pub normalized: bool,
}

impl CentralityResult {
    /// Get the centrality score for a specific node.
    pub fn score(&self, node: EntityId) -> Option<f64> {
        self.scores.get(&node).copied()
    }

    /// Get nodes sorted by centrality score (descending).
    pub fn sorted(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.scores.iter().map(|(&id, &score)| (id, score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by centrality score.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted().into_iter().take(n).collect()
    }

    /// Get the node with the highest centrality score.
    pub fn max(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the node with the lowest centrality score.
    pub fn min(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the mean centrality score.
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.values().sum::<f64>() / self.scores.len() as f64
    }
}

/// Betweenness Centrality algorithm implementation.
///
/// Betweenness centrality quantifies the number of times a node acts as a
/// bridge along the shortest path between two other nodes.
pub struct BetweennessCentrality;

impl BetweennessCentrality {
    /// Compute betweenness centrality for all nodes in the graph.
    ///
    /// Uses Brandes' algorithm for efficient O(V*E) computation.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `CentralityResult` containing scores for all nodes.
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &BetweennessCentralityConfig,
    ) -> GraphResult<CentralityResult> {
        // Check graph size before allocating large data structures
        if let Some(limit) = config.max_graph_nodes {
            let node_count = NodeStore::count(tx)?;
            if node_count > limit {
                return Err(GraphError::GraphTooLarge { node_count, limit });
            }
        }

        // Collect all nodes
        let mut nodes: Vec<EntityId> = Vec::new();
        NodeStore::for_each(tx, |entity| {
            nodes.push(entity.id);
            true
        })?;

        let n = nodes.len();
        if n == 0 {
            return Ok(CentralityResult { scores: HashMap::new(), normalized: config.normalize });
        }

        // Build node index for fast lookup
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency lists
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &node) in nodes.iter().enumerate() {
            let neighbor_ids = Self::get_neighbors(tx, node, config.direction)?;
            for neighbor_id in neighbor_ids {
                if let Some(&j) = node_index.get(&neighbor_id) {
                    neighbors[i].push(j);
                }
            }
        }

        // Initialize centrality scores
        let mut centrality: Vec<f64> = vec![0.0; n];

        // Brandes' algorithm: process each node as source
        for s in 0..n {
            // Single-source shortest-paths problem
            let mut stack: Vec<usize> = Vec::new();
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma: Vec<f64> = vec![0.0; n]; // Number of shortest paths
            sigma[s] = 1.0;
            let mut dist: Vec<i64> = vec![-1; n]; // Distance from source
            dist[s] = 0;

            // BFS
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &w in &neighbors[v] {
                    // Path discovery
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }

                    // Path counting
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }

            // Accumulation phase
            let mut delta: Vec<f64> = vec![0.0; n];

            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // For undirected graphs, divide by 2 (each path counted twice)
        if config.direction == Direction::Both {
            for score in &mut centrality {
                *score /= 2.0;
            }
        }

        // Normalize if requested
        if config.normalize && n > 2 {
            let normalization_factor = if config.direction == Direction::Both {
                // Undirected: (n-1)(n-2)/2
                2.0 / ((n - 1) * (n - 2)) as f64
            } else {
                // Directed: (n-1)(n-2)
                1.0 / ((n - 1) * (n - 2)) as f64
            };

            for score in &mut centrality {
                *score *= normalization_factor;
            }
        }

        // Build result map
        let scores: HashMap<EntityId, f64> = nodes.into_iter().zip(centrality).collect();

        Ok(CentralityResult { scores, normalized: config.normalize })
    }

    /// Compute betweenness centrality for a subset of nodes.
    ///
    /// Only considers shortest paths within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn compute_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &BetweennessCentralityConfig,
    ) -> GraphResult<CentralityResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(CentralityResult { scores: HashMap::new(), normalized: config.normalize });
        }

        // Build node index and set for fast lookup
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency lists (only including edges within the subgraph)
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &node) in nodes.iter().enumerate() {
            let neighbor_ids = Self::get_neighbors(tx, node, config.direction)?;
            for neighbor_id in neighbor_ids {
                if node_set.contains(&neighbor_id) {
                    if let Some(&j) = node_index.get(&neighbor_id) {
                        neighbors[i].push(j);
                    }
                }
            }
        }

        // Initialize centrality scores
        let mut centrality: Vec<f64> = vec![0.0; n];

        // Brandes' algorithm
        for s in 0..n {
            let mut stack: Vec<usize> = Vec::new();
            let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma: Vec<f64> = vec![0.0; n];
            sigma[s] = 1.0;
            let mut dist: Vec<i64> = vec![-1; n];
            dist[s] = 0;

            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                for &w in &neighbors[v] {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        queue.push_back(w);
                    }

                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        predecessors[w].push(v);
                    }
                }
            }

            let mut delta: Vec<f64> = vec![0.0; n];

            while let Some(w) = stack.pop() {
                for &v in &predecessors[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // For undirected graphs, divide by 2
        if config.direction == Direction::Both {
            for score in &mut centrality {
                *score /= 2.0;
            }
        }

        // Normalize if requested
        if config.normalize && n > 2 {
            let normalization_factor = if config.direction == Direction::Both {
                2.0 / ((n - 1) * (n - 2)) as f64
            } else {
                1.0 / ((n - 1) * (n - 2)) as f64
            };

            for score in &mut centrality {
                *score *= normalization_factor;
            }
        }

        // Build result map
        let scores: HashMap<EntityId, f64> = nodes.iter().copied().zip(centrality).collect();

        Ok(CentralityResult { scores, normalized: config.normalize })
    }

    /// Get neighbors of a node based on direction.
    fn get_neighbors<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<Vec<EntityId>> {
        let mut neighbors = Vec::new();

        if direction.includes_outgoing() {
            let edges = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in edges {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    neighbors.push(edge.target);
                }
            }
        }

        if direction.includes_incoming() {
            let edges = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in edges {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    neighbors.push(edge.source);
                }
            }
        }

        Ok(neighbors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = BetweennessCentralityConfig::default();
        assert!(config.normalize);
        assert_eq!(config.direction, Direction::Both);
        assert!(!config.include_endpoints);
    }

    #[test]
    fn config_builder() {
        let config = BetweennessCentralityConfig::new()
            .with_normalize(false)
            .with_direction(Direction::Outgoing)
            .with_include_endpoints(true);

        assert!(!config.normalize);
        assert_eq!(config.direction, Direction::Outgoing);
        assert!(config.include_endpoints);
    }

    #[test]
    fn result_empty() {
        let result = CentralityResult { scores: HashMap::new(), normalized: true };

        assert!(result.score(EntityId::new(1)).is_none());
        assert!(result.sorted().is_empty());
        assert!(result.top_n(10).is_empty());
        assert!(result.max().is_none());
        assert!(result.min().is_none());
        assert!((result.mean() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn result_sorted() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.5);
        scores.insert(EntityId::new(3), 0.2);

        let result = CentralityResult { scores, normalized: true };

        let sorted = result.sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, EntityId::new(2)); // highest
        assert_eq!(sorted[1].0, EntityId::new(1));
        assert_eq!(sorted[2].0, EntityId::new(3)); // lowest
    }

    #[test]
    fn result_mean() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.6);
        scores.insert(EntityId::new(3), 0.9);

        let result = CentralityResult { scores, normalized: true };

        assert!((result.mean() - 0.6).abs() < f64::EPSILON);
    }
}
