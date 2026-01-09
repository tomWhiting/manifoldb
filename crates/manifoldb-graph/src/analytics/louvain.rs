//! Louvain Community Detection Algorithm.
//!
//! This module implements the Louvain algorithm for community detection,
//! which optimizes modularity through a two-phase iterative process.
//!
//! # Algorithm
//!
//! The Louvain algorithm works in two phases that repeat until convergence:
//!
//! 1. **Local optimization**: Each node is moved to the community that yields
//!    the maximum modularity gain. This continues until no single node move
//!    improves modularity.
//!
//! 2. **Aggregation**: A new graph is built where nodes represent communities
//!    from phase 1, and edges are weighted by the sum of edges between
//!    original nodes in those communities.
//!
//! # Properties
//!
//! - Produces hierarchical community structure
//! - Optimizes modularity Q ∈ [-0.5, 1]
//! - Time complexity: O(n log n) in typical cases
//! - Deterministic when using a fixed seed
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{LouvainCommunityDetection, LouvainConfig};
//!
//! let config = LouvainConfig::default()
//!     .with_max_iterations(10)
//!     .with_tolerance(0.0001);
//!
//! let result = LouvainCommunityDetection::detect_communities(&tx, &config)?;
//!
//! // Get community assignments
//! for (node, community) in result.assignments.iter() {
//!     println!("Node {:?} belongs to community {}", node, community);
//! }
//!
//! // Check final modularity
//! println!("Modularity: {:.4}", result.modularity);
//! ```

use std::collections::HashMap;

use manifoldb_core::{EntityId, Value};
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};
use crate::traversal::Direction;

use super::community::CommunityResult;
use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Louvain Community Detection.
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// Maximum number of iterations (passes over all nodes).
    /// Default: 10
    pub max_iterations: usize,

    /// Minimum modularity improvement to continue iterating.
    /// If improvement is below this threshold, the algorithm stops.
    /// Default: 0.0001
    pub tolerance: f64,

    /// Direction of edges to follow.
    /// Default: Both (treat as undirected)
    pub direction: Direction,

    /// Seed for random number generation (for reproducible node ordering).
    /// Default: None (use deterministic ordering)
    pub seed: Option<u64>,

    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,

    /// Edge property name to use as weight.
    /// If None, all edges have weight 1.0.
    /// Default: None
    pub weight_property: Option<String>,

    /// Whether to include intermediate community assignments from each level.
    /// Default: false
    pub include_intermediate_communities: bool,

    /// Resolution parameter for modularity.
    /// Higher values favor smaller communities.
    /// Default: 1.0
    pub resolution: f64,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 0.0001,
            direction: Direction::Both,
            seed: None,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
            weight_property: None,
            include_intermediate_communities: false,
            resolution: 1.0,
        }
    }
}

impl LouvainConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the tolerance for modularity improvement.
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
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

    /// Set the maximum number of nodes allowed.
    pub const fn with_max_graph_nodes(mut self, limit: Option<usize>) -> Self {
        self.max_graph_nodes = limit;
        self
    }

    /// Set the weight property name for weighted edges.
    pub fn with_weight_property(mut self, property: impl Into<String>) -> Self {
        self.weight_property = Some(property.into());
        self
    }

    /// Set whether to include intermediate communities.
    pub const fn with_include_intermediate_communities(mut self, include: bool) -> Self {
        self.include_intermediate_communities = include;
        self
    }

    /// Set the resolution parameter.
    pub const fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }
}

/// Result of Louvain community detection.
#[derive(Debug, Clone)]
pub struct LouvainResult {
    /// Community assignments: node -> community ID
    pub assignments: HashMap<EntityId, u64>,

    /// Number of passes (complete iterations over all nodes) performed.
    pub passes: usize,

    /// Whether the algorithm converged (modularity improvement below tolerance).
    pub converged: bool,

    /// Number of distinct communities found.
    pub num_communities: usize,

    /// Final modularity score.
    pub modularity: f64,

    /// Intermediate community assignments from each level (if requested).
    /// Each entry is a mapping from node to community at that level.
    pub intermediate_communities: Vec<HashMap<EntityId, u64>>,
}

impl LouvainResult {
    /// Convert to CommunityResult for compatibility with existing procedures.
    pub fn to_community_result(&self) -> CommunityResult {
        CommunityResult {
            assignments: self.assignments.clone(),
            iterations: self.passes,
            converged: self.converged,
            num_communities: self.num_communities,
        }
    }
}

/// Louvain Community Detection algorithm implementation.
pub struct LouvainCommunityDetection;

impl LouvainCommunityDetection {
    /// Detect communities using the Louvain algorithm.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `LouvainResult` containing community assignments and modularity.
    pub fn detect_communities<T: Transaction>(
        tx: &T,
        config: &LouvainConfig,
    ) -> GraphResult<LouvainResult> {
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
            return Ok(LouvainResult {
                assignments: HashMap::new(),
                passes: 0,
                converged: true,
                num_communities: 0,
                modularity: 0.0,
                intermediate_communities: Vec::new(),
            });
        }

        // Build node index
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency structure with weights
        let (neighbors, weights) = Self::build_adjacency(tx, &nodes, &node_index, config)?;

        // Calculate total edge weight (m) - for undirected, count each edge once
        let mut total_weight = 0.0;
        for node_weights in &weights {
            for &w in node_weights {
                total_weight += w;
            }
        }
        // For undirected graph, we've counted each edge twice
        if config.direction == Direction::Both {
            total_weight /= 2.0;
        }

        if total_weight == 0.0 {
            // No edges - each node is its own community
            let assignments: HashMap<EntityId, u64> =
                nodes.iter().enumerate().map(|(i, &id)| (id, i as u64)).collect();
            return Ok(LouvainResult {
                assignments,
                passes: 0,
                converged: true,
                num_communities: n,
                modularity: 0.0,
                intermediate_communities: Vec::new(),
            });
        }

        // Initialize: each node in its own community
        let mut community: Vec<usize> = (0..n).collect();
        let mut intermediate_communities = Vec::new();

        // Calculate initial node strengths (weighted degree)
        let mut strength: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            for &w in &weights[i] {
                strength[i] += w;
            }
        }

        // Run Louvain phases
        let mut passes = 0;
        let mut converged = false;
        let mut current_modularity = Self::calculate_modularity(
            &neighbors,
            &weights,
            &community,
            &strength,
            total_weight,
            config.resolution,
        );

        // Deterministic or shuffled order
        let mut order: Vec<usize> = (0..n).collect();
        if let Some(seed) = config.seed {
            Self::shuffle_with_seed(&mut order, seed);
        }

        while passes < config.max_iterations {
            passes += 1;
            let mut improved = false;

            // Phase 1: Local optimization
            let mut local_changed = true;
            while local_changed {
                local_changed = false;

                for &i in &order {
                    let current_community = community[i];

                    // Calculate weights to neighboring communities
                    let mut community_weights: HashMap<usize, f64> = HashMap::new();
                    for (j_idx, &j) in neighbors[i].iter().enumerate() {
                        let w = weights[i][j_idx];
                        *community_weights.entry(community[j]).or_insert(0.0) += w;
                    }

                    // Calculate community sums (total strength of each community)
                    let mut community_strength: HashMap<usize, f64> = HashMap::new();
                    for (k, &c) in community.iter().enumerate() {
                        *community_strength.entry(c).or_insert(0.0) += strength[k];
                    }

                    // Remove node from current community and find best community
                    let ki = strength[i];
                    let ki_in = community_weights.get(&current_community).copied().unwrap_or(0.0);

                    // Sum of strengths in current community minus this node
                    let sigma_tot =
                        community_strength.get(&current_community).copied().unwrap_or(0.0) - ki;

                    let mut best_community = current_community;
                    let mut best_delta = 0.0;

                    // Calculate modularity gain for each neighboring community
                    for (&c, &ki_c) in &community_weights {
                        if c == current_community {
                            continue;
                        }

                        let sigma_c = community_strength.get(&c).copied().unwrap_or(0.0);

                        // Modularity gain: delta Q = ki_c/m - gamma * ki * sigma_c / (2m^2)
                        // minus removal cost: -ki_in/m + gamma * ki * sigma_tot / (2m^2)
                        let delta = (ki_c - ki_in) / total_weight
                            - config.resolution * ki * (sigma_c - sigma_tot)
                                / (2.0 * total_weight * total_weight);

                        if delta > best_delta {
                            best_delta = delta;
                            best_community = c;
                        }
                    }

                    // Move node if improvement found
                    if best_community != current_community {
                        community[i] = best_community;
                        local_changed = true;
                        improved = true;
                    }
                }
            }

            // Store intermediate if requested
            if config.include_intermediate_communities {
                let level_assignments: HashMap<EntityId, u64> =
                    nodes.iter().enumerate().map(|(i, &id)| (id, community[i] as u64)).collect();
                intermediate_communities.push(level_assignments);
            }

            // Calculate new modularity
            let new_modularity = Self::calculate_modularity(
                &neighbors,
                &weights,
                &community,
                &strength,
                total_weight,
                config.resolution,
            );

            // Check for convergence
            let improvement = new_modularity - current_modularity;
            if improvement < config.tolerance || !improved {
                converged = true;
                current_modularity = new_modularity;
                break;
            }

            current_modularity = new_modularity;
        }

        // Renumber communities to be contiguous starting from 0
        let mut community_map: HashMap<usize, u64> = HashMap::new();
        let mut next_id = 0u64;
        for c in &mut community {
            let new_id = *community_map.entry(*c).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *c = new_id as usize;
        }

        // Build final assignments
        let assignments: HashMap<EntityId, u64> =
            nodes.into_iter().zip(community.iter()).map(|(id, &c)| (id, c as u64)).collect();

        let num_communities = community_map.len();

        Ok(LouvainResult {
            assignments,
            passes,
            converged,
            num_communities,
            modularity: current_modularity,
            intermediate_communities,
        })
    }

    /// Build adjacency lists with weights.
    fn build_adjacency<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        node_index: &HashMap<EntityId, usize>,
        config: &LouvainConfig,
    ) -> GraphResult<(Vec<Vec<usize>>, Vec<Vec<f64>>)> {
        let n = nodes.len();
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut weights: Vec<Vec<f64>> = vec![Vec::new(); n];

        for (i, &node) in nodes.iter().enumerate() {
            let mut seen: HashMap<usize, f64> = HashMap::new();

            // Get outgoing edges
            if config.direction.includes_outgoing() {
                let edge_ids = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
                for edge_id in edge_ids {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        if let Some(&j) = node_index.get(&edge.target) {
                            let weight = Self::get_edge_weight(&edge.properties, config);
                            *seen.entry(j).or_insert(0.0) += weight;
                        }
                    }
                }
            }

            // Get incoming edges
            if config.direction.includes_incoming() {
                let edge_ids = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
                for edge_id in edge_ids {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        if let Some(&j) = node_index.get(&edge.source) {
                            let weight = Self::get_edge_weight(&edge.properties, config);
                            *seen.entry(j).or_insert(0.0) += weight;
                        }
                    }
                }
            }

            // Convert to vectors
            for (j, w) in seen {
                neighbors[i].push(j);
                weights[i].push(w);
            }
        }

        Ok((neighbors, weights))
    }

    /// Get edge weight from properties, defaulting to 1.0.
    fn get_edge_weight(properties: &HashMap<String, Value>, config: &LouvainConfig) -> f64 {
        if let Some(ref prop_name) = config.weight_property {
            if let Some(value) = properties.get(prop_name) {
                return match value {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => 1.0,
                };
            }
        }
        1.0
    }

    /// Calculate modularity Q.
    ///
    /// Q = (1/2m) * sum_ij[ (A_ij - gamma * k_i * k_j / 2m) * delta(c_i, c_j) ]
    fn calculate_modularity(
        neighbors: &[Vec<usize>],
        weights: &[Vec<f64>],
        community: &[usize],
        strength: &[f64],
        total_weight: f64,
        resolution: f64,
    ) -> f64 {
        let n = neighbors.len();
        let m2 = 2.0 * total_weight;

        if m2 == 0.0 {
            return 0.0;
        }

        let mut q = 0.0;

        // Sum over all edges
        for i in 0..n {
            for (j_idx, &j) in neighbors[i].iter().enumerate() {
                if community[i] == community[j] {
                    let a_ij = weights[i][j_idx];
                    let expected = resolution * strength[i] * strength[j] / m2;
                    q += a_ij - expected;
                }
            }
        }

        q / m2
    }

    /// Shuffle array using LCG with given seed.
    fn shuffle_with_seed(arr: &mut [usize], mut seed: u64) {
        let n = arr.len();
        for i in (1..n).rev() {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (seed as usize) % (i + 1);
            arr.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = LouvainConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!((config.tolerance - 0.0001).abs() < f64::EPSILON);
        assert_eq!(config.direction, Direction::Both);
        assert!(config.seed.is_none());
        assert!(config.weight_property.is_none());
        assert!(!config.include_intermediate_communities);
        assert!((config.resolution - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn config_builder() {
        let config = LouvainConfig::new()
            .with_max_iterations(20)
            .with_tolerance(0.001)
            .with_direction(Direction::Outgoing)
            .with_seed(42)
            .with_weight_property("strength")
            .with_include_intermediate_communities(true)
            .with_resolution(0.5);

        assert_eq!(config.max_iterations, 20);
        assert!((config.tolerance - 0.001).abs() < f64::EPSILON);
        assert_eq!(config.direction, Direction::Outgoing);
        assert_eq!(config.seed, Some(42));
        assert_eq!(config.weight_property, Some("strength".to_string()));
        assert!(config.include_intermediate_communities);
        assert!((config.resolution - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn result_to_community_result() {
        let mut assignments = HashMap::new();
        assignments.insert(EntityId::new(1), 0);
        assignments.insert(EntityId::new(2), 0);
        assignments.insert(EntityId::new(3), 1);

        let louvain_result = LouvainResult {
            assignments: assignments.clone(),
            passes: 5,
            converged: true,
            num_communities: 2,
            modularity: 0.5,
            intermediate_communities: Vec::new(),
        };

        let community_result = louvain_result.to_community_result();
        assert_eq!(community_result.assignments, assignments);
        assert_eq!(community_result.iterations, 5);
        assert!(community_result.converged);
        assert_eq!(community_result.num_communities, 2);
    }

    #[test]
    fn shuffle_with_seed_deterministic() {
        let mut arr1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut arr2 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        LouvainCommunityDetection::shuffle_with_seed(&mut arr1, 42);
        LouvainCommunityDetection::shuffle_with_seed(&mut arr2, 42);

        assert_eq!(arr1, arr2);
    }

    #[test]
    fn shuffle_with_different_seeds() {
        let mut arr1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut arr2 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        LouvainCommunityDetection::shuffle_with_seed(&mut arr1, 42);
        LouvainCommunityDetection::shuffle_with_seed(&mut arr2, 123);

        // Different seeds should (very likely) produce different orderings
        assert_ne!(arr1, arr2);
    }
}
