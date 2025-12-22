//! Closeness Centrality implementation.
//!
//! Closeness centrality measures how close a node is to all other nodes in
//! the graph, based on the sum of shortest path distances. Nodes with high
//! closeness centrality can reach other nodes quickly.
//!
//! # Algorithm
//!
//! This module implements closeness centrality using BFS for shortest paths.
//! For large graphs, sampling can be enabled to approximate the result.
//!
//! # Formula
//!
//! Standard closeness centrality:
//! CC(v) = (n - 1) / Σ d(v, u) for all u ≠ v
//!
//! where d(v, u) is the shortest path distance from v to u.
//!
//! Harmonic centrality (for disconnected graphs):
//! HC(v) = Σ 1/d(v, u) for all u ≠ v where d(v, u) < ∞
//!
//! The harmonic variant handles disconnected graphs better since it sums
//! reciprocals rather than dividing by the sum.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{ClosenessCentrality, ClosenessCentralityConfig};
//!
//! // Compute standard closeness centrality
//! let config = ClosenessCentralityConfig::default();
//! let result = ClosenessCentrality::compute(&tx, &config)?;
//!
//! // Compute harmonic centrality for disconnected graphs
//! let config = ClosenessCentralityConfig::default()
//!     .with_harmonic(true);
//! let result = ClosenessCentrality::compute(&tx, &config)?;
//!
//! // Find the most central nodes
//! for (node, score) in result.top_n(10) {
//!     println!("Node {:?} has closeness {:.4}", node, score);
//! }
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};
use crate::traversal::Direction;

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Closeness Centrality computation.
#[derive(Debug, Clone)]
pub struct ClosenessCentralityConfig {
    /// Direction of edges to follow for path finding.
    /// Default: Both (treat as undirected)
    pub direction: Direction,

    /// Whether to compute harmonic centrality instead of standard closeness.
    /// Harmonic centrality handles disconnected graphs better.
    /// Default: false
    pub harmonic: bool,

    /// Whether to normalize centrality values.
    /// Default: true
    pub normalize: bool,

    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,

    /// Optional sampling ratio for large graphs (0.0 to 1.0).
    /// When set, only a sample of nodes are used as sources for BFS.
    /// Default: None (compute exact values)
    pub sample_ratio: Option<f64>,

    /// Random seed for sampling (for reproducibility).
    /// Default: None
    pub seed: Option<u64>,
}

impl Default for ClosenessCentralityConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            harmonic: false,
            normalize: true,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
            sample_ratio: None,
            seed: None,
        }
    }
}

impl ClosenessCentralityConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the direction for path finding.
    ///
    /// - `Outgoing`: Follow edges in their natural direction
    /// - `Incoming`: Follow edges in reverse
    /// - `Both`: Treat graph as undirected (default)
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set whether to compute harmonic centrality.
    ///
    /// Harmonic centrality is defined as the sum of reciprocals of distances,
    /// which handles disconnected graphs better than standard closeness.
    pub const fn with_harmonic(mut self, harmonic: bool) -> Self {
        self.harmonic = harmonic;
        self
    }

    /// Set whether to normalize centrality values.
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

    /// Set sampling ratio for approximate computation on large graphs.
    ///
    /// A ratio of 0.1 means 10% of nodes will be sampled as sources.
    /// Results will be approximate but computed faster.
    pub const fn with_sample_ratio(mut self, ratio: f64) -> Self {
        self.sample_ratio = Some(ratio);
        self
    }

    /// Set random seed for reproducible sampling.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Result of Closeness Centrality computation.
#[derive(Debug, Clone)]
pub struct ClosenessCentralityResult {
    /// Closeness centrality scores for each node.
    pub scores: HashMap<EntityId, f64>,

    /// Whether scores are normalized.
    pub normalized: bool,

    /// Whether harmonic centrality was used.
    pub harmonic: bool,

    /// Whether results are approximate (from sampling).
    pub approximate: bool,
}

impl ClosenessCentralityResult {
    /// Get the closeness centrality score for a specific node.
    pub fn score(&self, node: EntityId) -> Option<f64> {
        self.scores.get(&node).copied()
    }

    /// Get nodes sorted by closeness centrality (descending).
    pub fn sorted(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.scores.iter().map(|(&id, &score)| (id, score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by closeness centrality.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted().into_iter().take(n).collect()
    }

    /// Get the node with the highest closeness centrality.
    pub fn max(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the node with the lowest closeness centrality.
    pub fn min(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the mean closeness centrality.
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.values().sum::<f64>() / self.scores.len() as f64
    }
}

/// Closeness Centrality algorithm implementation.
///
/// Closeness centrality quantifies how close a node is to all other nodes
/// in the graph, based on shortest path distances.
pub struct ClosenessCentrality;

impl ClosenessCentrality {
    /// Compute closeness centrality for all nodes in the graph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `ClosenessCentralityResult` containing scores for all nodes.
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &ClosenessCentralityConfig,
    ) -> GraphResult<ClosenessCentralityResult> {
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
            return Ok(ClosenessCentralityResult {
                scores: HashMap::new(),
                normalized: config.normalize,
                harmonic: config.harmonic,
                approximate: false,
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

        // Determine which nodes to use as sources (for sampling)
        let (sources, approximate) = if let Some(ratio) = config.sample_ratio {
            let sample_size = ((n as f64 * ratio).ceil() as usize).max(1).min(n);
            let mut indices: Vec<usize> = (0..n).collect();

            // Simple deterministic shuffle if seed is provided
            if let Some(seed) = config.seed {
                Self::shuffle_with_seed(&mut indices, seed);
            }

            indices.truncate(sample_size);
            (indices, true)
        } else {
            ((0..n).collect(), false)
        };

        // Compute closeness for each node
        let mut scores: Vec<f64> = vec![0.0; n];

        if config.harmonic {
            // Harmonic centrality: sum of reciprocals of distances
            for &s in &sources {
                let distances = Self::bfs_distances(&neighbors, s, n);
                for (t, &dist) in distances.iter().enumerate() {
                    if s != t && dist > 0 {
                        scores[t] += 1.0 / dist as f64;
                    }
                }
            }

            // Scale if sampling
            if approximate {
                let scale = n as f64 / sources.len() as f64;
                for score in &mut scores {
                    *score *= scale;
                }
            }

            // Normalize if requested
            if config.normalize && n > 1 {
                let normalization_factor = 1.0 / (n - 1) as f64;
                for score in &mut scores {
                    *score *= normalization_factor;
                }
            }
        } else {
            // Standard closeness: (n-1) / sum of distances
            for i in 0..n {
                let distances = Self::bfs_distances(&neighbors, i, n);

                let mut total_distance: u64 = 0;
                let mut reachable_count: usize = 0;

                for (j, &dist) in distances.iter().enumerate() {
                    if i != j && dist > 0 {
                        total_distance += dist as u64;
                        reachable_count += 1;
                    }
                }

                if total_distance > 0 {
                    scores[i] = reachable_count as f64 / total_distance as f64;

                    // Normalize by (reachable_count / (n-1)) to handle disconnected graphs
                    if config.normalize && n > 1 {
                        scores[i] *= reachable_count as f64 / (n - 1) as f64;
                    }
                } else {
                    scores[i] = 0.0;
                }
            }
        }

        // Build result map
        let scores_map: HashMap<EntityId, f64> = nodes.into_iter().zip(scores).collect();

        Ok(ClosenessCentralityResult {
            scores: scores_map,
            normalized: config.normalize,
            harmonic: config.harmonic,
            approximate,
        })
    }

    /// Compute closeness centrality for a subset of nodes.
    ///
    /// Only considers paths within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn compute_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &ClosenessCentralityConfig,
    ) -> GraphResult<ClosenessCentralityResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(ClosenessCentralityResult {
                scores: HashMap::new(),
                normalized: config.normalize,
                harmonic: config.harmonic,
                approximate: false,
            });
        }

        // Build node index and set for fast lookup
        let node_set: HashSet<EntityId> = nodes.iter().copied().collect();
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

        // Compute closeness for each node
        let mut scores: Vec<f64> = vec![0.0; n];

        if config.harmonic {
            // Harmonic centrality
            for s in 0..n {
                let distances = Self::bfs_distances(&neighbors, s, n);
                for (t, &dist) in distances.iter().enumerate() {
                    if s != t && dist > 0 {
                        scores[s] += 1.0 / dist as f64;
                    }
                }
            }

            if config.normalize && n > 1 {
                let normalization_factor = 1.0 / (n - 1) as f64;
                for score in &mut scores {
                    *score *= normalization_factor;
                }
            }
        } else {
            // Standard closeness
            for i in 0..n {
                let distances = Self::bfs_distances(&neighbors, i, n);

                let mut total_distance: u64 = 0;
                let mut reachable_count: usize = 0;

                for (j, &dist) in distances.iter().enumerate() {
                    if i != j && dist > 0 {
                        total_distance += dist as u64;
                        reachable_count += 1;
                    }
                }

                if total_distance > 0 {
                    scores[i] = reachable_count as f64 / total_distance as f64;

                    if config.normalize && n > 1 {
                        scores[i] *= reachable_count as f64 / (n - 1) as f64;
                    }
                } else {
                    scores[i] = 0.0;
                }
            }
        }

        // Build result map
        let scores_map: HashMap<EntityId, f64> = nodes.iter().copied().zip(scores).collect();

        Ok(ClosenessCentralityResult {
            scores: scores_map,
            normalized: config.normalize,
            harmonic: config.harmonic,
            approximate: false,
        })
    }

    /// Compute BFS distances from a source node to all other nodes.
    fn bfs_distances(neighbors: &[Vec<usize>], source: usize, n: usize) -> Vec<usize> {
        let mut distances = vec![0; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();

        visited[source] = true;
        queue.push_back((source, 0));

        while let Some((node, dist)) = queue.pop_front() {
            distances[node] = dist;

            for &neighbor in &neighbors[node] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        distances
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

    /// Simple deterministic shuffle using a seed.
    fn shuffle_with_seed(vec: &mut [usize], seed: u64) {
        let n = vec.len();
        if n <= 1 {
            return;
        }

        let mut state = seed;
        for i in (1..n).rev() {
            // Simple LCG random number generator
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (state as usize) % (i + 1);
            vec.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = ClosenessCentralityConfig::default();
        assert_eq!(config.direction, Direction::Both);
        assert!(!config.harmonic);
        assert!(config.normalize);
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
        assert!(config.sample_ratio.is_none());
        assert!(config.seed.is_none());
    }

    #[test]
    fn config_builder() {
        let config = ClosenessCentralityConfig::new()
            .with_direction(Direction::Outgoing)
            .with_harmonic(true)
            .with_normalize(false)
            .with_sample_ratio(0.5)
            .with_seed(42)
            .with_max_graph_nodes(Some(1000));

        assert_eq!(config.direction, Direction::Outgoing);
        assert!(config.harmonic);
        assert!(!config.normalize);
        assert_eq!(config.sample_ratio, Some(0.5));
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.max_graph_nodes, Some(1000));
    }

    #[test]
    fn result_empty() {
        let result = ClosenessCentralityResult {
            scores: HashMap::new(),
            normalized: true,
            harmonic: false,
            approximate: false,
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

        let result = ClosenessCentralityResult {
            scores,
            normalized: true,
            harmonic: false,
            approximate: false,
        };

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

        let result = ClosenessCentralityResult {
            scores,
            normalized: true,
            harmonic: false,
            approximate: false,
        };

        assert!((result.mean() - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn bfs_distances_simple() {
        // Simple graph: 0 -> 1 -> 2
        let neighbors = vec![vec![1], vec![2], vec![]];
        let distances = ClosenessCentrality::bfs_distances(&neighbors, 0, 3);

        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 1);
        assert_eq!(distances[2], 2);
    }

    #[test]
    fn shuffle_with_seed_deterministic() {
        let mut vec1 = vec![0, 1, 2, 3, 4, 5];
        let mut vec2 = vec![0, 1, 2, 3, 4, 5];

        ClosenessCentrality::shuffle_with_seed(&mut vec1, 42);
        ClosenessCentrality::shuffle_with_seed(&mut vec2, 42);

        assert_eq!(vec1, vec2);
    }
}
