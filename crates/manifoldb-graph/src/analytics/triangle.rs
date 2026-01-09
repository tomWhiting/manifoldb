//! Triangle counting and clustering coefficient algorithms.
//!
//! This module implements triangle counting and local clustering coefficient
//! algorithms for graph analysis.
//!
//! # Triangles
//!
//! A triangle is a set of three nodes that are all connected to each other.
//! Triangle counting is useful for understanding the cohesiveness of a network.
//!
//! # Clustering Coefficient
//!
//! The local clustering coefficient of a node measures how connected its
//! neighbors are to each other. It's defined as:
//!
//! ```text
//! C(v) = 2 * T(v) / (k(v) * (k(v) - 1))
//! ```
//!
//! Where:
//! - T(v) is the number of triangles containing node v
//! - k(v) is the degree of node v
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{TriangleCount, TriangleCountConfig};
//!
//! let config = TriangleCountConfig::default();
//! let result = TriangleCount::compute(&tx, &config)?;
//!
//! // Get total triangles in graph
//! println!("Total triangles: {}", result.total_triangles);
//!
//! // Get triangles for a specific node
//! if let Some(count) = result.triangles_for(node_id) {
//!     println!("Node {:?} participates in {} triangles", node_id, count);
//! }
//! ```

use std::collections::{HashMap, HashSet};

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Triangle Count algorithm.
#[derive(Debug, Clone)]
pub struct TriangleCountConfig {
    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,
}

impl Default for TriangleCountConfig {
    fn default() -> Self {
        Self { max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES) }
    }
}

impl TriangleCountConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
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

/// Result of Triangle Count computation.
#[derive(Debug, Clone)]
pub struct TriangleCountResult {
    /// Number of triangles each node participates in.
    pub node_triangles: HashMap<EntityId, usize>,

    /// Total number of triangles in the graph.
    /// (Each triangle is counted once, not three times.)
    pub total_triangles: usize,

    /// Local clustering coefficient for each node.
    pub coefficients: HashMap<EntityId, f64>,

    /// Global (average) clustering coefficient.
    pub global_coefficient: f64,
}

impl TriangleCountResult {
    /// Get the number of triangles a specific node participates in.
    pub fn triangles_for(&self, node: EntityId) -> Option<usize> {
        self.node_triangles.get(&node).copied()
    }

    /// Get the local clustering coefficient for a specific node.
    pub fn coefficient_for(&self, node: EntityId) -> Option<f64> {
        self.coefficients.get(&node).copied()
    }

    /// Get nodes sorted by triangle count (descending).
    pub fn sorted_by_triangles(&self) -> Vec<(EntityId, usize)> {
        let mut pairs: Vec<_> =
            self.node_triangles.iter().map(|(&id, &count)| (id, count)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs
    }

    /// Get nodes sorted by clustering coefficient (descending).
    pub fn sorted_by_coefficient(&self) -> Vec<(EntityId, f64)> {
        let mut pairs: Vec<_> = self.coefficients.iter().map(|(&id, &coef)| (id, coef)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get the top N nodes by triangle count.
    pub fn top_n_by_triangles(&self, n: usize) -> Vec<(EntityId, usize)> {
        self.sorted_by_triangles().into_iter().take(n).collect()
    }

    /// Get the top N nodes by clustering coefficient.
    pub fn top_n_by_coefficient(&self, n: usize) -> Vec<(EntityId, f64)> {
        self.sorted_by_coefficient().into_iter().take(n).collect()
    }

    /// Get the node with the most triangles.
    pub fn max_triangles(&self) -> Option<(EntityId, usize)> {
        self.node_triangles.iter().max_by_key(|(_, &count)| count).map(|(&id, &count)| (id, count))
    }

    /// Get the node with the highest clustering coefficient.
    pub fn max_coefficient(&self) -> Option<(EntityId, f64)> {
        self.coefficients
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&id, &coef)| (id, coef))
    }
}

/// Triangle counting algorithm implementation.
///
/// This algorithm counts triangles in an undirected graph and computes
/// local clustering coefficients for each node.
pub struct TriangleCount;

impl TriangleCount {
    /// Compute triangle counts and clustering coefficients for all nodes.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `TriangleCountResult` containing triangle counts and coefficients.
    ///
    /// # Algorithm
    ///
    /// The algorithm treats the graph as undirected (considers both incoming
    /// and outgoing edges). For each node:
    /// 1. Get all neighbors (both directions)
    /// 2. For each pair of neighbors, check if they're connected
    /// 3. Count triangles (each counted once per node that participates)
    /// 4. Compute local clustering coefficient
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &TriangleCountConfig,
    ) -> GraphResult<TriangleCountResult> {
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
            return Ok(TriangleCountResult {
                node_triangles: HashMap::new(),
                total_triangles: 0,
                coefficients: HashMap::new(),
                global_coefficient: 0.0,
            });
        }

        // Build node index and neighbor sets (treating graph as undirected)
        let node_set: HashSet<EntityId> = nodes.iter().copied().collect();
        let mut neighbors: HashMap<EntityId, HashSet<EntityId>> = HashMap::with_capacity(n);

        for &node in &nodes {
            let mut node_neighbors = HashSet::new();

            // Get outgoing neighbors
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.target) && edge.target != node {
                        node_neighbors.insert(edge.target);
                    }
                }
            }

            // Get incoming neighbors
            let incoming = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in incoming {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.source) && edge.source != node {
                        node_neighbors.insert(edge.source);
                    }
                }
            }

            neighbors.insert(node, node_neighbors);
        }

        // Count triangles for each node
        let mut node_triangles: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        let mut total_triangle_count: usize = 0;

        for &node in &nodes {
            let node_neighbors = &neighbors[&node];
            let degree = node_neighbors.len();

            if degree < 2 {
                // Need at least 2 neighbors to form a triangle
                node_triangles.insert(node, 0);
                continue;
            }

            // Count triangles: for each pair of neighbors, check if they're connected
            let neighbor_vec: Vec<EntityId> = node_neighbors.iter().copied().collect();
            let mut triangle_count = 0;

            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    let neighbor_i = neighbor_vec[i];
                    let neighbor_j = neighbor_vec[j];

                    // Check if neighbor_i and neighbor_j are connected
                    if neighbors[&neighbor_i].contains(&neighbor_j) {
                        triangle_count += 1;
                    }
                }
            }

            node_triangles.insert(node, triangle_count);
            total_triangle_count += triangle_count;
        }

        // Each triangle is counted 3 times (once per vertex)
        let total_triangles = total_triangle_count / 3;

        // Compute clustering coefficients
        let mut coefficients: HashMap<EntityId, f64> = HashMap::with_capacity(n);
        let mut coefficient_sum = 0.0;
        let mut valid_node_count = 0;

        for &node in &nodes {
            let degree = neighbors[&node].len();
            let triangles = node_triangles[&node];

            let coefficient = if degree < 2 {
                // Nodes with degree 0 or 1 cannot have triangles
                // Convention: clustering coefficient is 0
                0.0
            } else {
                // C(v) = 2 * T(v) / (k(v) * (k(v) - 1))
                let max_triangles = degree * (degree - 1) / 2;
                triangles as f64 / max_triangles as f64
            };

            coefficients.insert(node, coefficient);

            // Only include nodes with degree >= 2 in global average
            if degree >= 2 {
                coefficient_sum += coefficient;
                valid_node_count += 1;
            }
        }

        let global_coefficient =
            if valid_node_count > 0 { coefficient_sum / valid_node_count as f64 } else { 0.0 };

        Ok(TriangleCountResult {
            node_triangles,
            total_triangles,
            coefficients,
            global_coefficient,
        })
    }

    /// Compute triangle counts for a subset of nodes.
    ///
    /// Only considers triangles formed entirely within the specified nodes.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn compute_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &TriangleCountConfig,
    ) -> GraphResult<TriangleCountResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(TriangleCountResult {
                node_triangles: HashMap::new(),
                total_triangles: 0,
                coefficients: HashMap::new(),
                global_coefficient: 0.0,
            });
        }

        // Check size limit
        if let Some(limit) = config.max_graph_nodes {
            if n > limit {
                return Err(GraphError::GraphTooLarge { node_count: n, limit });
            }
        }

        // Build node set for fast lookup
        let node_set: HashSet<EntityId> = nodes.iter().copied().collect();

        // Build neighbor sets (only considering nodes in the subgraph)
        let mut neighbors: HashMap<EntityId, HashSet<EntityId>> = HashMap::with_capacity(n);

        for &node in nodes {
            let mut node_neighbors = HashSet::new();

            // Get outgoing neighbors
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.target) && edge.target != node {
                        node_neighbors.insert(edge.target);
                    }
                }
            }

            // Get incoming neighbors
            let incoming = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in incoming {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.source) && edge.source != node {
                        node_neighbors.insert(edge.source);
                    }
                }
            }

            neighbors.insert(node, node_neighbors);
        }

        // Count triangles
        let mut node_triangles: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        let mut total_triangle_count: usize = 0;

        for &node in nodes {
            let node_neighbors = &neighbors[&node];
            let degree = node_neighbors.len();

            if degree < 2 {
                node_triangles.insert(node, 0);
                continue;
            }

            let neighbor_vec: Vec<EntityId> = node_neighbors.iter().copied().collect();
            let mut triangle_count = 0;

            for i in 0..neighbor_vec.len() {
                for j in (i + 1)..neighbor_vec.len() {
                    let neighbor_i = neighbor_vec[i];
                    let neighbor_j = neighbor_vec[j];

                    if neighbors[&neighbor_i].contains(&neighbor_j) {
                        triangle_count += 1;
                    }
                }
            }

            node_triangles.insert(node, triangle_count);
            total_triangle_count += triangle_count;
        }

        let total_triangles = total_triangle_count / 3;

        // Compute clustering coefficients
        let mut coefficients: HashMap<EntityId, f64> = HashMap::with_capacity(n);
        let mut coefficient_sum = 0.0;
        let mut valid_node_count = 0;

        for &node in nodes {
            let degree = neighbors[&node].len();
            let triangles = node_triangles[&node];

            let coefficient = if degree < 2 {
                0.0
            } else {
                let max_triangles = degree * (degree - 1) / 2;
                triangles as f64 / max_triangles as f64
            };

            coefficients.insert(node, coefficient);

            if degree >= 2 {
                coefficient_sum += coefficient;
                valid_node_count += 1;
            }
        }

        let global_coefficient =
            if valid_node_count > 0 { coefficient_sum / valid_node_count as f64 } else { 0.0 };

        Ok(TriangleCountResult {
            node_triangles,
            total_triangles,
            coefficients,
            global_coefficient,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = TriangleCountConfig::default();
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
    }

    #[test]
    fn config_builder() {
        let config = TriangleCountConfig::new().with_max_graph_nodes(Some(1000));
        assert_eq!(config.max_graph_nodes, Some(1000));

        let config = TriangleCountConfig::new().with_max_graph_nodes(None);
        assert_eq!(config.max_graph_nodes, None);
    }

    #[test]
    fn result_empty() {
        let result = TriangleCountResult {
            node_triangles: HashMap::new(),
            total_triangles: 0,
            coefficients: HashMap::new(),
            global_coefficient: 0.0,
        };

        assert!(result.triangles_for(EntityId::new(1)).is_none());
        assert!(result.coefficient_for(EntityId::new(1)).is_none());
        assert!(result.sorted_by_triangles().is_empty());
        assert!(result.sorted_by_coefficient().is_empty());
        assert!(result.max_triangles().is_none());
        assert!(result.max_coefficient().is_none());
    }

    #[test]
    fn result_sorted_by_triangles() {
        let mut node_triangles = HashMap::new();
        node_triangles.insert(EntityId::new(1), 3);
        node_triangles.insert(EntityId::new(2), 5);
        node_triangles.insert(EntityId::new(3), 1);

        let result = TriangleCountResult {
            node_triangles,
            total_triangles: 3,
            coefficients: HashMap::new(),
            global_coefficient: 0.0,
        };

        let sorted = result.sorted_by_triangles();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, EntityId::new(2)); // highest
        assert_eq!(sorted[0].1, 5);
        assert_eq!(sorted[1].0, EntityId::new(1));
        assert_eq!(sorted[2].0, EntityId::new(3)); // lowest
    }

    #[test]
    fn result_sorted_by_coefficient() {
        let mut coefficients = HashMap::new();
        coefficients.insert(EntityId::new(1), 0.3);
        coefficients.insert(EntityId::new(2), 0.8);
        coefficients.insert(EntityId::new(3), 0.5);

        let result = TriangleCountResult {
            node_triangles: HashMap::new(),
            total_triangles: 0,
            coefficients,
            global_coefficient: 0.0,
        };

        let sorted = result.sorted_by_coefficient();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, EntityId::new(2)); // highest
        assert!((sorted[0].1 - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn result_top_n() {
        let mut node_triangles = HashMap::new();
        node_triangles.insert(EntityId::new(1), 3);
        node_triangles.insert(EntityId::new(2), 5);
        node_triangles.insert(EntityId::new(3), 1);
        node_triangles.insert(EntityId::new(4), 4);

        let result = TriangleCountResult {
            node_triangles,
            total_triangles: 0,
            coefficients: HashMap::new(),
            global_coefficient: 0.0,
        };

        let top2 = result.top_n_by_triangles(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, EntityId::new(2)); // 5 triangles
        assert_eq!(top2[1].0, EntityId::new(4)); // 4 triangles
    }

    #[test]
    fn result_max() {
        let mut node_triangles = HashMap::new();
        node_triangles.insert(EntityId::new(1), 3);
        node_triangles.insert(EntityId::new(2), 5);
        node_triangles.insert(EntityId::new(3), 1);

        let mut coefficients = HashMap::new();
        coefficients.insert(EntityId::new(1), 0.3);
        coefficients.insert(EntityId::new(2), 0.8);
        coefficients.insert(EntityId::new(3), 0.5);

        let result = TriangleCountResult {
            node_triangles,
            total_triangles: 3,
            coefficients,
            global_coefficient: 0.53,
        };

        let max_tri = result.max_triangles().unwrap();
        assert_eq!(max_tri.0, EntityId::new(2));
        assert_eq!(max_tri.1, 5);

        let max_coef = result.max_coefficient().unwrap();
        assert_eq!(max_coef.0, EntityId::new(2));
        assert!((max_coef.1 - 0.8).abs() < f64::EPSILON);
    }
}
