//! Community Detection using Label Propagation Algorithm.
//!
//! This module implements community detection using the Label Propagation
//! Algorithm (LPA). LPA is a fast, near-linear time algorithm for detecting
//! communities in large networks.
//!
//! # Algorithm
//!
//! Label Propagation works by:
//! 1. Initialize each node with a unique community label
//! 2. For each iteration:
//!    - Update each node's label to the most frequent label among neighbors
//!    - If there are ties, randomly select one of the most frequent labels
//! 3. Repeat until no labels change (convergence) or max iterations reached
//!
//! # Properties
//!
//! - Near-linear time complexity O(m) where m is the number of edges
//! - No prior knowledge of the number of communities required
//! - Non-deterministic (different runs may produce different results)
//! - Works well for networks with clear community structure
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{CommunityDetection, CommunityDetectionConfig};
//!
//! let config = CommunityDetectionConfig::default()
//!     .with_max_iterations(100);
//!
//! let result = CommunityDetection::label_propagation(&tx, &config)?;
//!
//! // Get community assignments
//! for (node, community) in result.assignments.iter() {
//!     println!("Node {:?} belongs to community {}", node, community);
//! }
//!
//! // Get community sizes
//! for (community_id, size) in result.community_sizes() {
//!     println!("Community {} has {} members", community_id, size);
//! }
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult, NodeStore};
use crate::traversal::Direction;

/// Configuration for Community Detection.
#[derive(Debug, Clone)]
pub struct CommunityDetectionConfig {
    /// Maximum number of iterations.
    /// Default: 100
    pub max_iterations: usize,

    /// Direction of edges to follow.
    /// Default: Both (treat as undirected)
    pub direction: Direction,

    /// Seed for random number generation (for reproducibility).
    /// Default: None (use system entropy)
    pub seed: Option<u64>,

    /// Minimum improvement in modularity to continue iterating.
    /// Set to 0.0 to only check for label stability.
    /// Default: 0.0
    pub min_improvement: f64,
}

impl Default for CommunityDetectionConfig {
    fn default() -> Self {
        Self { max_iterations: 100, direction: Direction::Both, seed: None, min_improvement: 0.0 }
    }
}

impl CommunityDetectionConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the direction to follow edges.
    pub const fn with_direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    /// Set the seed for reproducible results.
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the minimum improvement threshold.
    pub const fn with_min_improvement(mut self, min_improvement: f64) -> Self {
        self.min_improvement = min_improvement;
        self
    }
}

/// Result of community detection.
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Community assignments: node -> community ID
    pub assignments: HashMap<EntityId, u64>,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged.
    pub converged: bool,

    /// Number of distinct communities found.
    pub num_communities: usize,
}

impl CommunityResult {
    /// Get the community ID for a specific node.
    pub fn community(&self, node: EntityId) -> Option<u64> {
        self.assignments.get(&node).copied()
    }

    /// Get all nodes in a specific community.
    pub fn members(&self, community_id: u64) -> Vec<EntityId> {
        self.assignments.iter().filter(|(_, &c)| c == community_id).map(|(&node, _)| node).collect()
    }

    /// Get community sizes.
    pub fn community_sizes(&self) -> HashMap<u64, usize> {
        let mut sizes: HashMap<u64, usize> = HashMap::new();
        for &community in self.assignments.values() {
            *sizes.entry(community).or_insert(0) += 1;
        }
        sizes
    }

    /// Get communities sorted by size (descending).
    pub fn communities_by_size(&self) -> Vec<(u64, usize)> {
        let mut sizes: Vec<_> = self.community_sizes().into_iter().collect();
        sizes.sort_by(|a, b| b.1.cmp(&a.1));
        sizes
    }

    /// Get the largest community.
    pub fn largest_community(&self) -> Option<(u64, usize)> {
        self.communities_by_size().into_iter().next()
    }

    /// Get the smallest community.
    pub fn smallest_community(&self) -> Option<(u64, usize)> {
        self.communities_by_size().into_iter().last()
    }

    /// Check if two nodes are in the same community.
    pub fn same_community(&self, node1: EntityId, node2: EntityId) -> bool {
        match (self.community(node1), self.community(node2)) {
            (Some(c1), Some(c2)) => c1 == c2,
            _ => false,
        }
    }
}

/// Community Detection algorithm implementations.
pub struct CommunityDetection;

impl CommunityDetection {
    /// Detect communities using Label Propagation Algorithm.
    ///
    /// This is a fast, near-linear time algorithm that doesn't require
    /// knowing the number of communities in advance.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `CommunityResult` containing community assignments for all nodes.
    pub fn label_propagation<T: Transaction>(
        tx: &T,
        config: &CommunityDetectionConfig,
    ) -> GraphResult<CommunityResult> {
        // Collect all nodes
        let mut nodes: Vec<EntityId> = Vec::new();
        NodeStore::for_each(tx, |entity| {
            nodes.push(entity.id);
            true
        })?;

        let n = nodes.len();
        if n == 0 {
            return Ok(CommunityResult {
                assignments: HashMap::new(),
                iterations: 0,
                converged: true,
                num_communities: 0,
            });
        }

        // Build node index
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

        // Initialize labels (each node starts with its own label)
        let mut labels: Vec<u64> = (0..n as u64).collect();

        // Simple pseudo-random number generator
        let mut rng_state = config.seed.unwrap_or(12345);
        let simple_random = |state: &mut u64| -> u64 {
            // Linear congruential generator (constants from PCG)
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *state
        };

        let mut iterations = 0;
        let mut converged = false;

        // Create a shuffled order for processing nodes
        let mut order: Vec<usize> = (0..n).collect();

        while iterations < config.max_iterations {
            iterations += 1;

            // Shuffle node processing order for better convergence
            for i in (1..n).rev() {
                let j = (simple_random(&mut rng_state) as usize) % (i + 1);
                order.swap(i, j);
            }

            let mut changed = false;

            for &i in &order {
                if neighbors[i].is_empty() {
                    continue;
                }

                // Count neighbor labels
                let mut label_counts: HashMap<u64, usize> = HashMap::new();
                for &j in &neighbors[i] {
                    *label_counts.entry(labels[j]).or_insert(0) += 1;
                }

                // Find maximum count
                let max_count = *label_counts.values().max().unwrap_or(&0);

                // Get all labels with maximum count
                let max_labels: Vec<u64> = label_counts
                    .iter()
                    .filter(|(_, &count)| count == max_count)
                    .map(|(&label, _)| label)
                    .collect();

                // Select one of the max labels (preferring current label if tied)
                let new_label = if max_labels.contains(&labels[i]) {
                    labels[i]
                } else if max_labels.len() == 1 {
                    max_labels[0]
                } else {
                    // Random selection among tied labels
                    let idx = (simple_random(&mut rng_state) as usize) % max_labels.len();
                    max_labels[idx]
                };

                if new_label != labels[i] {
                    labels[i] = new_label;
                    changed = true;
                }
            }

            if !changed {
                converged = true;
                break;
            }
        }

        // Renumber communities to be contiguous starting from 0
        let mut community_map: HashMap<u64, u64> = HashMap::new();
        let mut next_id = 0u64;
        for label in &mut labels {
            let new_id = *community_map.entry(*label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *label = new_id;
        }

        // Build result
        let assignments: HashMap<EntityId, u64> = nodes.into_iter().zip(labels).collect();

        let num_communities = community_map.len();

        Ok(CommunityResult { assignments, iterations, converged, num_communities })
    }

    /// Detect communities for a subset of nodes.
    ///
    /// Only considers edges within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn label_propagation_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &CommunityDetectionConfig,
    ) -> GraphResult<CommunityResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(CommunityResult {
                assignments: HashMap::new(),
                iterations: 0,
                converged: true,
                num_communities: 0,
            });
        }

        // Build node index and set
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

        // Initialize labels
        let mut labels: Vec<u64> = (0..n as u64).collect();

        // Simple pseudo-random number generator
        let mut rng_state = config.seed.unwrap_or(12345);
        let simple_random = |state: &mut u64| -> u64 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *state
        };

        let mut iterations = 0;
        let mut converged = false;
        let mut order: Vec<usize> = (0..n).collect();

        while iterations < config.max_iterations {
            iterations += 1;

            // Shuffle order
            for i in (1..n).rev() {
                let j = (simple_random(&mut rng_state) as usize) % (i + 1);
                order.swap(i, j);
            }

            let mut changed = false;

            for &i in &order {
                if neighbors[i].is_empty() {
                    continue;
                }

                let mut label_counts: HashMap<u64, usize> = HashMap::new();
                for &j in &neighbors[i] {
                    *label_counts.entry(labels[j]).or_insert(0) += 1;
                }

                let max_count = *label_counts.values().max().unwrap_or(&0);
                let max_labels: Vec<u64> = label_counts
                    .iter()
                    .filter(|(_, &count)| count == max_count)
                    .map(|(&label, _)| label)
                    .collect();

                let new_label = if max_labels.contains(&labels[i]) {
                    labels[i]
                } else if max_labels.len() == 1 {
                    max_labels[0]
                } else {
                    let idx = (simple_random(&mut rng_state) as usize) % max_labels.len();
                    max_labels[idx]
                };

                if new_label != labels[i] {
                    labels[i] = new_label;
                    changed = true;
                }
            }

            if !changed {
                converged = true;
                break;
            }
        }

        // Renumber communities
        let mut community_map: HashMap<u64, u64> = HashMap::new();
        let mut next_id = 0u64;
        for label in &mut labels {
            let new_id = *community_map.entry(*label).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *label = new_id;
        }

        let assignments: HashMap<EntityId, u64> = nodes.iter().copied().zip(labels).collect();

        let num_communities = community_map.len();

        Ok(CommunityResult { assignments, iterations, converged, num_communities })
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
        let config = CommunityDetectionConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert_eq!(config.direction, Direction::Both);
        assert!(config.seed.is_none());
        assert!((config.min_improvement - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_builder() {
        let config = CommunityDetectionConfig::new()
            .with_max_iterations(50)
            .with_direction(Direction::Outgoing)
            .with_seed(42)
            .with_min_improvement(0.001);

        assert_eq!(config.max_iterations, 50);
        assert_eq!(config.direction, Direction::Outgoing);
        assert_eq!(config.seed, Some(42));
        assert!((config.min_improvement - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn result_empty() {
        let result = CommunityResult {
            assignments: HashMap::new(),
            iterations: 0,
            converged: true,
            num_communities: 0,
        };

        assert!(result.community(EntityId::new(1)).is_none());
        assert!(result.members(0).is_empty());
        assert!(result.community_sizes().is_empty());
        assert!(result.largest_community().is_none());
        assert!(result.smallest_community().is_none());
    }

    #[test]
    fn result_community_operations() {
        let mut assignments = HashMap::new();
        assignments.insert(EntityId::new(1), 0);
        assignments.insert(EntityId::new(2), 0);
        assignments.insert(EntityId::new(3), 1);
        assignments.insert(EntityId::new(4), 1);
        assignments.insert(EntityId::new(5), 1);

        let result =
            CommunityResult { assignments, iterations: 10, converged: true, num_communities: 2 };

        // Test community lookup
        assert_eq!(result.community(EntityId::new(1)), Some(0));
        assert_eq!(result.community(EntityId::new(3)), Some(1));
        assert_eq!(result.community(EntityId::new(99)), None);

        // Test members
        let mut members_0 = result.members(0);
        members_0.sort_by_key(|e| e.as_u64());
        assert_eq!(members_0.len(), 2);

        let mut members_1 = result.members(1);
        members_1.sort_by_key(|e| e.as_u64());
        assert_eq!(members_1.len(), 3);

        // Test sizes
        let sizes = result.community_sizes();
        assert_eq!(sizes.get(&0), Some(&2));
        assert_eq!(sizes.get(&1), Some(&3));

        // Test largest/smallest
        assert_eq!(result.largest_community(), Some((1, 3)));
        assert_eq!(result.smallest_community(), Some((0, 2)));

        // Test same_community
        assert!(result.same_community(EntityId::new(1), EntityId::new(2)));
        assert!(result.same_community(EntityId::new(3), EntityId::new(4)));
        assert!(!result.same_community(EntityId::new(1), EntityId::new(3)));
    }
}
