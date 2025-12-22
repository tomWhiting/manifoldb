//! PageRank algorithm implementation.
//!
//! This module implements the PageRank algorithm using the iterative power method.
//! PageRank assigns an importance score to each node based on the link structure
//! of the graph.
//!
//! # Algorithm
//!
//! PageRank uses an iterative approach:
//! 1. Initialize all nodes with equal rank (1/N)
//! 2. For each iteration:
//!    - Each node distributes its rank equally among outgoing neighbors
//!    - Apply damping factor to handle random jumps
//!    - Check for convergence
//! 3. Repeat until convergence or max iterations reached
//!
//! # Formula
//!
//! PR(u) = (1-d)/N + d * Σ(PR(v)/L(v)) for all v linking to u
//!
//! Where:
//! - d is the damping factor (typically 0.85)
//! - N is the total number of nodes
//! - L(v) is the out-degree of node v
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{PageRank, PageRankConfig};
//!
//! let config = PageRankConfig::default()
//!     .with_damping_factor(0.85)
//!     .with_max_iterations(100)
//!     .with_tolerance(1e-6);
//!
//! let result = PageRank::compute(&tx, &config)?;
//!
//! // Get top nodes by PageRank
//! for (node, score) in result.top_n(10) {
//!     println!("Node {:?}: {:.6}", node, score);
//! }
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult, NodeStore};

/// Configuration for PageRank algorithm.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    /// Damping factor (probability of following a link vs random jump).
    /// Default: 0.85
    pub damping_factor: f64,

    /// Maximum number of iterations before stopping.
    /// Default: 100
    pub max_iterations: usize,

    /// Convergence tolerance. Algorithm stops when max score change < tolerance.
    /// Default: 1e-6
    pub tolerance: f64,

    /// Whether to normalize scores to sum to 1.0.
    /// Default: true
    pub normalize: bool,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self { damping_factor: 0.85, max_iterations: 100, tolerance: 1e-6, normalize: true }
    }
}

impl PageRankConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the damping factor.
    ///
    /// The damping factor represents the probability that a random walker
    /// follows a link instead of jumping to a random node.
    /// Common values are 0.85 (default) or 0.9.
    pub const fn with_damping_factor(mut self, damping_factor: f64) -> Self {
        self.damping_factor = damping_factor;
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

    /// Set whether to normalize scores to sum to 1.0.
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

/// Result of PageRank computation.
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// PageRank scores for each node.
    pub scores: HashMap<EntityId, f64>,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged within tolerance.
    pub converged: bool,

    /// Final convergence delta (max change in last iteration).
    pub final_delta: f64,
}

impl PageRankResult {
    /// Get the PageRank score for a specific node.
    pub fn score(&self, node: EntityId) -> Option<f64> {
        self.scores.get(&node).copied()
    }

    /// Get nodes sorted by PageRank score (descending).
    pub fn sorted(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.scores.iter().map(|(&id, &score)| (id, score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by PageRank score.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted().into_iter().take(n).collect()
    }

    /// Get the node with the highest PageRank score.
    pub fn max(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the node with the lowest PageRank score.
    pub fn min(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }
}

/// PageRank algorithm implementation.
///
/// PageRank assigns importance scores to nodes based on the link structure
/// of the graph. Nodes with many incoming links from important nodes will
/// have higher scores.
pub struct PageRank;

impl PageRank {
    /// Compute PageRank scores for all nodes in the graph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `PageRankResult` containing scores for all nodes.
    ///
    /// # Algorithm
    ///
    /// Uses the iterative power method:
    /// 1. Initialize all nodes with score 1/N
    /// 2. Iteratively update scores based on incoming links
    /// 3. Handle dangling nodes (nodes with no outgoing edges)
    /// 4. Check for convergence after each iteration
    pub fn compute<T: Transaction>(tx: &T, config: &PageRankConfig) -> GraphResult<PageRankResult> {
        // Collect all nodes and build adjacency information
        let mut nodes: Vec<EntityId> = Vec::new();
        NodeStore::for_each(tx, |entity| {
            nodes.push(entity.id);
            true
        })?;

        let n = nodes.len();
        if n == 0 {
            return Ok(PageRankResult {
                scores: HashMap::new(),
                iterations: 0,
                converged: true,
                final_delta: 0.0,
            });
        }

        // Build node index for fast lookup
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build outgoing edge lists and compute out-degrees
        let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_degrees: Vec<usize> = vec![0; n];

        for (i, &node) in nodes.iter().enumerate() {
            // Get outgoing edges
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            out_degrees[i] = outgoing.len();

            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if let Some(&target_idx) = node_index.get(&edge.target) {
                        out_neighbors[i].push(target_idx);
                    }
                }
            }
        }

        // Build incoming edge lists for efficient iteration
        let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in out_neighbors.iter().enumerate() {
            for &j in neighbors {
                in_neighbors[j].push(i);
            }
        }

        // Initialize scores
        let initial_score = 1.0 / n as f64;
        let mut scores: Vec<f64> = vec![initial_score; n];
        let mut new_scores: Vec<f64> = vec![0.0; n];

        let d = config.damping_factor;
        let base_score = (1.0 - d) / n as f64;

        let mut iterations = 0;
        let mut converged = false;
        let mut final_delta = f64::MAX;

        // Iterative computation
        while iterations < config.max_iterations {
            iterations += 1;

            // Calculate dangling node contribution (nodes with no outgoing edges)
            let dangling_sum: f64 = nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| out_degrees[*i] == 0)
                .map(|(i, _)| scores[i])
                .sum();
            let dangling_contribution = d * dangling_sum / n as f64;

            // Calculate new scores
            for (i, incoming) in in_neighbors.iter().enumerate() {
                let mut link_sum = 0.0;
                for &j in incoming {
                    if out_degrees[j] > 0 {
                        link_sum += scores[j] / out_degrees[j] as f64;
                    }
                }
                new_scores[i] = base_score + d * link_sum + dangling_contribution;
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

        // Normalize if requested
        if config.normalize {
            let total: f64 = scores.iter().sum();
            if total > 0.0 {
                for score in &mut scores {
                    *score /= total;
                }
            }
        }

        // Build result map
        let scores_map: HashMap<EntityId, f64> = nodes.into_iter().zip(scores).collect();

        Ok(PageRankResult { scores: scores_map, iterations, converged, final_delta })
    }

    /// Compute PageRank for a subset of nodes.
    ///
    /// This is useful for computing PageRank on a subgraph defined by
    /// a specific set of nodes.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn compute_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &PageRankConfig,
    ) -> GraphResult<PageRankResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(PageRankResult {
                scores: HashMap::new(),
                iterations: 0,
                converged: true,
                final_delta: 0.0,
            });
        }

        // Build node index for fast lookup
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency information for the subgraph
        let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_degrees: Vec<usize> = vec![0; n];

        for (i, &node) in nodes.iter().enumerate() {
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;

            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    // Only include edges to nodes in the subgraph
                    if node_set.contains(&edge.target) {
                        if let Some(&target_idx) = node_index.get(&edge.target) {
                            out_neighbors[i].push(target_idx);
                            out_degrees[i] += 1;
                        }
                    }
                }
            }
        }

        // Build incoming edge lists
        let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in out_neighbors.iter().enumerate() {
            for &j in neighbors {
                in_neighbors[j].push(i);
            }
        }

        // Initialize scores
        let initial_score = 1.0 / n as f64;
        let mut scores: Vec<f64> = vec![initial_score; n];
        let mut new_scores: Vec<f64> = vec![0.0; n];

        let d = config.damping_factor;
        let base_score = (1.0 - d) / n as f64;

        let mut iterations = 0;
        let mut converged = false;
        let mut final_delta = f64::MAX;

        while iterations < config.max_iterations {
            iterations += 1;

            // Calculate dangling node contribution
            let dangling_sum: f64 =
                (0..n).filter(|&i| out_degrees[i] == 0).map(|i| scores[i]).sum();
            let dangling_contribution = d * dangling_sum / n as f64;

            // Calculate new scores
            for (i, incoming) in in_neighbors.iter().enumerate() {
                let mut link_sum = 0.0;
                for &j in incoming {
                    if out_degrees[j] > 0 {
                        link_sum += scores[j] / out_degrees[j] as f64;
                    }
                }
                new_scores[i] = base_score + d * link_sum + dangling_contribution;
            }

            // Check convergence
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

        // Normalize if requested
        if config.normalize {
            let total: f64 = scores.iter().sum();
            if total > 0.0 {
                for score in &mut scores {
                    *score /= total;
                }
            }
        }

        // Build result map
        let scores_map: HashMap<EntityId, f64> = nodes.iter().copied().zip(scores).collect();

        Ok(PageRankResult { scores: scores_map, iterations, converged, final_delta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = PageRankConfig::default();
        assert!((config.damping_factor - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 100);
        assert!((config.tolerance - 1e-6).abs() < f64::EPSILON);
        assert!(config.normalize);
    }

    #[test]
    fn config_builder() {
        let config = PageRankConfig::new()
            .with_damping_factor(0.9)
            .with_max_iterations(50)
            .with_tolerance(1e-8)
            .with_normalize(false);

        assert!((config.damping_factor - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.max_iterations, 50);
        assert!((config.tolerance - 1e-8).abs() < f64::EPSILON);
        assert!(!config.normalize);
    }

    #[test]
    fn result_empty() {
        let result = PageRankResult {
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
    }

    #[test]
    fn result_sorted() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 0.3);
        scores.insert(EntityId::new(2), 0.5);
        scores.insert(EntityId::new(3), 0.2);

        let result = PageRankResult { scores, iterations: 10, converged: true, final_delta: 1e-7 };

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

        let result = PageRankResult { scores, iterations: 10, converged: true, final_delta: 1e-7 };

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

        let result = PageRankResult { scores, iterations: 10, converged: true, final_delta: 1e-7 };

        let max = result.max().unwrap();
        assert_eq!(max.0, EntityId::new(2));
        assert!((max.1 - 0.5).abs() < f64::EPSILON);

        let min = result.min().unwrap();
        assert_eq!(min.0, EntityId::new(3));
        assert!((min.1 - 0.2).abs() < f64::EPSILON);
    }
}
