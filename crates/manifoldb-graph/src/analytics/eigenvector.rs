//! Eigenvector Centrality implementation.
//!
//! Eigenvector centrality measures node importance based on connections to
//! important nodes. A node is important if it's connected to other important
//! nodes. This creates a recursive definition that can be computed using
//! the power iteration method.
//!
//! # Algorithm
//!
//! This module implements eigenvector centrality using the power iteration
//! method to find the dominant eigenvector of the adjacency matrix.
//!
//! # Formula
//!
//! For node v:
//! EC(v) = (1/λ) * Σ EC(u) for all u connected to v
//!
//! where λ is the largest eigenvalue of the adjacency matrix.
//!
//! The algorithm iteratively updates scores until convergence:
//! x(k+1) = A * x(k) / ||A * x(k)||
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{EigenvectorCentrality, EigenvectorCentralityConfig};
//!
//! let config = EigenvectorCentralityConfig::default()
//!     .with_tolerance(1e-6)
//!     .with_max_iterations(100);
//! let result = EigenvectorCentrality::compute(&tx, &config)?;
//!
//! // Find the most important nodes
//! for (node, score) in result.top_n(10) {
//!     println!("Node {:?} has eigenvector centrality {:.4}", node, score);
//! }
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};
use crate::traversal::Direction;

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Eigenvector Centrality computation.
#[derive(Debug, Clone)]
pub struct EigenvectorCentralityConfig {
    /// Direction of edges to follow.
    /// Default: Both (treat as undirected)
    pub direction: Direction,

    /// Maximum number of iterations before stopping.
    /// Default: 100
    pub max_iterations: usize,

    /// Convergence tolerance. Algorithm stops when max score change < tolerance.
    /// Default: 1e-6
    pub tolerance: f64,

    /// Whether to normalize scores to have unit L2 norm.
    /// Default: true
    pub normalize: bool,

    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,
}

impl Default for EigenvectorCentralityConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            max_iterations: 100,
            tolerance: 1e-6,
            normalize: true,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
        }
    }
}

impl EigenvectorCentralityConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the direction for following edges.
    ///
    /// - `Outgoing`: Follow edges in their natural direction
    /// - `Incoming`: Follow edges in reverse
    /// - `Both`: Treat graph as undirected (default)
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set the maximum number of iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the convergence tolerance.
    ///
    /// The algorithm stops when the maximum change in any node's score
    /// between iterations is less than this value.
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set whether to normalize scores to have unit L2 norm.
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the maximum number of nodes allowed.
    ///
    /// If the graph has more nodes than this limit, the algorithm will
    /// return a [`GraphError::GraphTooLarge`] error.
    ///
    /// Set to `None` to disable the check (use with caution).
    ///
    /// [`GraphError::GraphTooLarge`]: crate::store::GraphError::GraphTooLarge
    pub const fn with_max_graph_nodes(mut self, limit: Option<usize>) -> Self {
        self.max_graph_nodes = limit;
        self
    }
}

/// Result of Eigenvector Centrality computation.
#[derive(Debug, Clone)]
pub struct EigenvectorCentralityResult {
    /// Eigenvector centrality scores for each node.
    pub scores: HashMap<EntityId, f64>,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged within tolerance.
    pub converged: bool,

    /// Final convergence delta (max change in last iteration).
    pub final_delta: f64,
}

impl EigenvectorCentralityResult {
    /// Get the eigenvector centrality score for a specific node.
    pub fn score(&self, node: EntityId) -> Option<f64> {
        self.scores.get(&node).copied()
    }

    /// Get nodes sorted by eigenvector centrality (descending).
    pub fn sorted(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.scores.iter().map(|(&id, &score)| (id, score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by eigenvector centrality.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted().into_iter().take(n).collect()
    }

    /// Get the node with the highest eigenvector centrality.
    pub fn max(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the node with the lowest eigenvector centrality.
    pub fn min(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the mean eigenvector centrality.
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.values().sum::<f64>() / self.scores.len() as f64
    }
}

/// Eigenvector Centrality algorithm implementation.
///
/// Eigenvector centrality assigns importance scores to nodes based on
/// connections to important nodes. The algorithm uses the power iteration
/// method to compute the dominant eigenvector of the adjacency matrix.
pub struct EigenvectorCentrality;

impl EigenvectorCentrality {
    /// Compute eigenvector centrality for all nodes in the graph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// An `EigenvectorCentralityResult` containing scores for all nodes.
    ///
    /// # Algorithm
    ///
    /// Uses the power iteration method:
    /// 1. Initialize all nodes with score 1/√n
    /// 2. Iteratively update scores: x(k+1) = A * x(k)
    /// 3. Normalize after each iteration
    /// 4. Check for convergence
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &EigenvectorCentralityConfig,
    ) -> GraphResult<EigenvectorCentralityResult> {
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
            return Ok(EigenvectorCentralityResult {
                scores: HashMap::new(),
                iterations: 0,
                converged: true,
                final_delta: 0.0,
            });
        }

        // Build node index for fast lookup
        let mut node_index: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &id) in nodes.iter().enumerate() {
            node_index.insert(id, i);
        }

        // Build adjacency lists
        const AVG_DEGREE_ESTIMATE: usize = 8;
        let mut neighbors: Vec<Vec<usize>> =
            (0..n).map(|_| Vec::with_capacity(AVG_DEGREE_ESTIMATE)).collect();
        for (i, &node) in nodes.iter().enumerate() {
            let neighbor_ids = Self::get_neighbors(tx, node, config.direction)?;
            for neighbor_id in neighbor_ids {
                if let Some(&j) = node_index.get(&neighbor_id) {
                    neighbors[i].push(j);
                }
            }
        }

        // Initialize scores with uniform distribution
        let initial_score = 1.0 / (n as f64).sqrt();
        let mut scores: Vec<f64> = vec![initial_score; n];
        let mut new_scores: Vec<f64> = vec![0.0; n];

        let mut iterations = 0;
        let mut converged = false;
        let mut final_delta = f64::MAX;

        // Power iteration
        while iterations < config.max_iterations {
            iterations += 1;

            // Reset new scores
            new_scores.fill(0.0);

            // Compute new scores: x(k+1) = A * x(k)
            // For each node, sum the scores of its neighbors
            for (i, neighbor_list) in neighbors.iter().enumerate() {
                for &j in neighbor_list {
                    new_scores[i] += scores[j];
                }
            }

            // Compute L2 norm for normalization
            let norm: f64 = new_scores.iter().map(|&x| x * x).sum::<f64>().sqrt();

            // Handle the case where all scores are zero (disconnected graph)
            if norm < f64::EPSILON {
                // Fall back to uniform distribution
                let uniform_score = 1.0 / (n as f64).sqrt();
                new_scores.fill(uniform_score);
            } else {
                // Normalize to unit L2 norm
                for score in &mut new_scores {
                    *score /= norm;
                }
            }

            // Check convergence
            let mut max_delta = 0.0f64;
            for i in 0..n {
                let delta = (new_scores[i] - scores[i]).abs();
                max_delta = max_delta.max(delta);
            }

            final_delta = max_delta;

            // Swap scores
            std::mem::swap(&mut scores, &mut new_scores);

            if max_delta < config.tolerance {
                converged = true;
                break;
            }
        }

        // Final normalization if not already normalized
        if config.normalize {
            let norm: f64 = scores.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > f64::EPSILON {
                for score in &mut scores {
                    *score /= norm;
                }
            }
        }

        // Build result map
        let scores_map: HashMap<EntityId, f64> = nodes.into_iter().zip(scores).collect();

        Ok(EigenvectorCentralityResult { scores: scores_map, iterations, converged, final_delta })
    }

    /// Compute eigenvector centrality for a subset of nodes.
    ///
    /// Only considers edges within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn compute_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &EigenvectorCentralityConfig,
    ) -> GraphResult<EigenvectorCentralityResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(EigenvectorCentralityResult {
                scores: HashMap::new(),
                iterations: 0,
                converged: true,
                final_delta: 0.0,
            });
        }

        // Build node index and set for fast lookup
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();
        let mut node_index: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &id) in nodes.iter().enumerate() {
            node_index.insert(id, i);
        }

        // Build adjacency lists (only including edges within the subgraph)
        const AVG_DEGREE_ESTIMATE: usize = 8;
        let mut neighbors: Vec<Vec<usize>> =
            (0..n).map(|_| Vec::with_capacity(AVG_DEGREE_ESTIMATE)).collect();
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

        // Initialize scores
        let initial_score = 1.0 / (n as f64).sqrt();
        let mut scores: Vec<f64> = vec![initial_score; n];
        let mut new_scores: Vec<f64> = vec![0.0; n];

        let mut iterations = 0;
        let mut converged = false;
        let mut final_delta = f64::MAX;

        // Power iteration
        while iterations < config.max_iterations {
            iterations += 1;

            new_scores.fill(0.0);

            for (i, neighbor_list) in neighbors.iter().enumerate() {
                for &j in neighbor_list {
                    new_scores[i] += scores[j];
                }
            }

            let norm: f64 = new_scores.iter().map(|&x| x * x).sum::<f64>().sqrt();

            if norm < f64::EPSILON {
                let uniform_score = 1.0 / (n as f64).sqrt();
                new_scores.fill(uniform_score);
            } else {
                for score in &mut new_scores {
                    *score /= norm;
                }
            }

            let mut max_delta = 0.0f64;
            for i in 0..n {
                let delta = (new_scores[i] - scores[i]).abs();
                max_delta = max_delta.max(delta);
            }

            final_delta = max_delta;
            std::mem::swap(&mut scores, &mut new_scores);

            if max_delta < config.tolerance {
                converged = true;
                break;
            }
        }

        if config.normalize {
            let norm: f64 = scores.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > f64::EPSILON {
                for score in &mut scores {
                    *score /= norm;
                }
            }
        }

        let scores_map: HashMap<EntityId, f64> = nodes.iter().copied().zip(scores).collect();

        Ok(EigenvectorCentralityResult { scores: scores_map, iterations, converged, final_delta })
    }

    /// Get neighbors of a node based on direction.
    fn get_neighbors<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<Vec<EntityId>> {
        const INITIAL_NEIGHBORS_CAPACITY: usize = 16;
        let mut neighbors = Vec::with_capacity(INITIAL_NEIGHBORS_CAPACITY);

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
        let config = EigenvectorCentralityConfig::default();
        assert_eq!(config.direction, Direction::Both);
        assert_eq!(config.max_iterations, 100);
        assert!((config.tolerance - 1e-6).abs() < f64::EPSILON);
        assert!(config.normalize);
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
    }

    #[test]
    fn config_builder() {
        let config = EigenvectorCentralityConfig::new()
            .with_direction(Direction::Outgoing)
            .with_max_iterations(50)
            .with_tolerance(1e-8)
            .with_normalize(false)
            .with_max_graph_nodes(Some(1000));

        assert_eq!(config.direction, Direction::Outgoing);
        assert_eq!(config.max_iterations, 50);
        assert!((config.tolerance - 1e-8).abs() < f64::EPSILON);
        assert!(!config.normalize);
        assert_eq!(config.max_graph_nodes, Some(1000));
    }

    #[test]
    fn result_empty() {
        let result = EigenvectorCentralityResult {
            scores: HashMap::new(),
            iterations: 0,
            converged: true,
            final_delta: 0.0,
        };

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

        let result = EigenvectorCentralityResult {
            scores,
            iterations: 10,
            converged: true,
            final_delta: 1e-7,
        };

        let sorted = result.sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, EntityId::new(2)); // highest
        assert_eq!(sorted[1].0, EntityId::new(1));
        assert_eq!(sorted[2].0, EntityId::new(3)); // lowest
    }

    #[test]
    fn result_top_n() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.5);
        scores.insert(EntityId::new(3), 0.2);

        let result = EigenvectorCentralityResult {
            scores,
            iterations: 10,
            converged: true,
            final_delta: 1e-7,
        };

        let top2 = result.top_n(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, EntityId::new(2));
        assert_eq!(top2[1].0, EntityId::new(1));
    }

    #[test]
    fn result_max_min() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.5);
        scores.insert(EntityId::new(3), 0.2);

        let result = EigenvectorCentralityResult {
            scores,
            iterations: 10,
            converged: true,
            final_delta: 1e-7,
        };

        let max = result.max().unwrap();
        assert_eq!(max.0, EntityId::new(2));
        assert!((max.1 - 0.5).abs() < f64::EPSILON);

        let min = result.min().unwrap();
        assert_eq!(min.0, EntityId::new(3));
        assert!((min.1 - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn result_mean() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.6);
        scores.insert(EntityId::new(3), 0.9);

        let result = EigenvectorCentralityResult {
            scores,
            iterations: 10,
            converged: true,
            final_delta: 1e-7,
        };

        assert!((result.mean() - 0.6).abs() < f64::EPSILON);
    }
}
