//! Degree Centrality implementation.
//!
//! Degree centrality is a simple measure based on the number of connections
//! a node has. It can measure in-degree (incoming connections), out-degree
//! (outgoing connections), or total degree (both).
//!
//! # Formula
//!
//! For a node v:
//! - In-degree: DC_in(v) = number of incoming edges
//! - Out-degree: DC_out(v) = number of outgoing edges
//! - Total degree: DC(v) = in-degree + out-degree
//!
//! When normalized:
//! - DC_normalized(v) = DC(v) / (n - 1)
//!
//! where n is the total number of nodes.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{DegreeCentrality, DegreeCentralityConfig};
//! use manifoldb_graph::traversal::Direction;
//!
//! // Compute out-degree centrality
//! let config = DegreeCentralityConfig::default()
//!     .with_direction(Direction::Outgoing);
//! let result = DegreeCentrality::compute(&tx, &config)?;
//!
//! // Find the most connected nodes
//! for (node, degree) in result.top_n(10) {
//!     println!("Node {:?} has degree {:.4}", node, degree);
//! }
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{GraphError, GraphResult, NodeStore};
use crate::traversal::Direction;

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Degree Centrality computation.
#[derive(Debug, Clone)]
pub struct DegreeCentralityConfig {
    /// Direction of edges to count.
    /// - `Outgoing`: Count outgoing edges (out-degree)
    /// - `Incoming`: Count incoming edges (in-degree)
    /// - `Both`: Count both directions (total degree)
    ///
    /// Default: Both
    pub direction: Direction,

    /// Whether to normalize centrality values.
    /// When normalized, values are divided by (n-1) where n is the number of nodes.
    /// Default: false
    pub normalize: bool,

    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,
}

impl Default for DegreeCentralityConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            normalize: false,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
        }
    }
}

impl DegreeCentralityConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the direction for degree counting.
    ///
    /// - `Outgoing`: Count outgoing edges (out-degree)
    /// - `Incoming`: Count incoming edges (in-degree)
    /// - `Both`: Count both directions (total degree, default)
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set whether to normalize degree values.
    ///
    /// When normalized, values are divided by (n-1) where n is the number
    /// of nodes, giving a value in the range [0, 1].
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

/// Result of Degree Centrality computation.
#[derive(Debug, Clone)]
pub struct DegreeCentralityResult {
    /// Degree centrality scores for each node.
    pub scores: HashMap<EntityId, f64>,

    /// Whether scores are normalized.
    pub normalized: bool,

    /// The direction used for computation.
    pub direction: Direction,
}

impl DegreeCentralityResult {
    /// Get the degree centrality score for a specific node.
    pub fn score(&self, node: EntityId) -> Option<f64> {
        self.scores.get(&node).copied()
    }

    /// Get nodes sorted by degree centrality (descending).
    pub fn sorted(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.scores.iter().map(|(&id, &score)| (id, score)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by degree centrality.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted().into_iter().take(n).collect()
    }

    /// Get the node with the highest degree centrality.
    pub fn max(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the node with the lowest degree centrality.
    pub fn min(&self) -> Option<(EntityId, f64)> {
        self.scores
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &score)| (id, score))
    }

    /// Get the mean degree centrality.
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.values().sum::<f64>() / self.scores.len() as f64
    }
}

/// Degree Centrality algorithm implementation.
///
/// Degree centrality measures importance based on the number of direct
/// connections a node has. It's a simple but often effective measure.
pub struct DegreeCentrality;

impl DegreeCentrality {
    /// Compute degree centrality for all nodes in the graph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `DegreeCentralityResult` containing scores for all nodes.
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &DegreeCentralityConfig,
    ) -> GraphResult<DegreeCentralityResult> {
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
            return Ok(DegreeCentralityResult {
                scores: HashMap::new(),
                normalized: config.normalize,
                direction: config.direction,
            });
        }

        // Compute degree for each node
        let mut scores: HashMap<EntityId, f64> = HashMap::with_capacity(n);

        for &node in &nodes {
            let degree = Self::compute_degree(tx, node, config.direction)?;
            scores.insert(node, degree as f64);
        }

        // Normalize if requested
        if config.normalize && n > 1 {
            let normalization_factor = 1.0 / (n - 1) as f64;
            for score in scores.values_mut() {
                *score *= normalization_factor;
            }
        }

        Ok(DegreeCentralityResult {
            scores,
            normalized: config.normalize,
            direction: config.direction,
        })
    }

    /// Compute degree centrality for a subset of nodes.
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
        config: &DegreeCentralityConfig,
    ) -> GraphResult<DegreeCentralityResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(DegreeCentralityResult {
                scores: HashMap::new(),
                normalized: config.normalize,
                direction: config.direction,
            });
        }

        // Build node set for fast lookup
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();

        // Compute degree for each node (only counting edges within the subgraph)
        let mut scores: HashMap<EntityId, f64> = HashMap::with_capacity(n);

        for &node in nodes {
            let degree = Self::compute_degree_in_subgraph(tx, node, config.direction, &node_set)?;
            scores.insert(node, degree as f64);
        }

        // Normalize if requested
        if config.normalize && n > 1 {
            let normalization_factor = 1.0 / (n - 1) as f64;
            for score in scores.values_mut() {
                *score *= normalization_factor;
            }
        }

        Ok(DegreeCentralityResult {
            scores,
            normalized: config.normalize,
            direction: config.direction,
        })
    }

    /// Compute the degree of a single node.
    fn compute_degree<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<usize> {
        let mut degree = 0;

        if direction.includes_outgoing() {
            degree += AdjacencyIndex::count_outgoing(tx, node)?;
        }

        if direction.includes_incoming() {
            degree += AdjacencyIndex::count_incoming(tx, node)?;
        }

        Ok(degree)
    }

    /// Compute the degree of a node within a subgraph.
    fn compute_degree_in_subgraph<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
        node_set: &std::collections::HashSet<EntityId>,
    ) -> GraphResult<usize> {
        use crate::store::EdgeStore;

        let mut degree = 0;

        if direction.includes_outgoing() {
            let edges = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in edges {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.target) {
                        degree += 1;
                    }
                }
            }
        }

        if direction.includes_incoming() {
            let edges = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in edges {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.source) {
                        degree += 1;
                    }
                }
            }
        }

        Ok(degree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = DegreeCentralityConfig::default();
        assert_eq!(config.direction, Direction::Both);
        assert!(!config.normalize);
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
    }

    #[test]
    fn config_builder() {
        let config = DegreeCentralityConfig::new()
            .with_direction(Direction::Outgoing)
            .with_normalize(true)
            .with_max_graph_nodes(Some(1000));

        assert_eq!(config.direction, Direction::Outgoing);
        assert!(config.normalize);
        assert_eq!(config.max_graph_nodes, Some(1000));
    }

    #[test]
    fn result_empty() {
        let result = DegreeCentralityResult {
            scores: HashMap::new(),
            normalized: false,
            direction: Direction::Both,
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
        scores.insert(EntityId::new(1), 3.0);
        scores.insert(EntityId::new(2), 5.0);
        scores.insert(EntityId::new(3), 2.0);

        let result =
            DegreeCentralityResult { scores, normalized: false, direction: Direction::Both };

        let sorted = result.sorted();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, EntityId::new(2)); // highest
        assert_eq!(sorted[1].0, EntityId::new(1));
        assert_eq!(sorted[2].0, EntityId::new(3)); // lowest
    }

    #[test]
    fn result_mean() {
        let mut scores = HashMap::new();
        scores.insert(EntityId::new(1), 3.0);
        scores.insert(EntityId::new(2), 6.0);
        scores.insert(EntityId::new(3), 9.0);

        let result =
            DegreeCentralityResult { scores, normalized: false, direction: Direction::Both };

        assert!((result.mean() - 6.0).abs() < f64::EPSILON);
    }
}
