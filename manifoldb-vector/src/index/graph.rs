//! HNSW graph data structure.
//!
//! This module contains the in-memory graph representation for HNSW.
//! The graph is a multi-layer structure where each node can have connections
//! to other nodes in the same layer.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use manifoldb_core::EntityId;

use crate::distance::DistanceMetric;
use crate::types::Embedding;

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// The entity ID this node represents.
    pub entity_id: EntityId,
    /// The embedding vector.
    pub embedding: Embedding,
    /// The maximum layer this node appears in.
    pub max_layer: usize,
    /// Connections to other nodes, indexed by layer.
    /// `connections[layer]` = list of neighbor entity IDs
    pub connections: Vec<Vec<EntityId>>,
}

impl HnswNode {
    /// Create a new HNSW node.
    #[inline]
    pub fn new(entity_id: EntityId, embedding: Embedding, max_layer: usize) -> Self {
        let connections = vec![Vec::new(); max_layer + 1];
        Self { entity_id, embedding, max_layer, connections }
    }

    /// Get the connections at a specific layer.
    #[inline]
    #[must_use]
    pub fn connections_at(&self, layer: usize) -> &[EntityId] {
        self.connections.get(layer).map_or(&[], |c| c.as_slice())
    }

    /// Add a connection at a specific layer.
    #[inline]
    pub fn add_connection(&mut self, layer: usize, neighbor: EntityId) {
        if layer < self.connections.len() && !self.connections[layer].contains(&neighbor) {
            self.connections[layer].push(neighbor);
        }
    }

    /// Remove a connection at a specific layer.
    #[inline]
    pub fn remove_connection(&mut self, layer: usize, neighbor: EntityId) {
        if layer < self.connections.len() {
            self.connections[layer].retain(|&id| id != neighbor);
        }
    }

    /// Set the connections at a specific layer, replacing existing ones.
    #[inline]
    pub fn set_connections(&mut self, layer: usize, neighbors: Vec<EntityId>) {
        if layer < self.connections.len() {
            self.connections[layer] = neighbors;
        }
    }
}

/// The HNSW graph structure.
#[derive(Debug)]
pub struct HnswGraph {
    /// All nodes in the graph, keyed by entity ID.
    pub nodes: HashMap<EntityId, HnswNode>,
    /// The entry point node (highest level node).
    pub entry_point: Option<EntityId>,
    /// The current maximum layer in the graph.
    pub max_layer: usize,
    /// The distance metric used for similarity.
    pub distance_metric: DistanceMetric,
    /// The dimension of embeddings in this graph.
    pub dimension: usize,
}

impl HnswGraph {
    /// Create a new empty HNSW graph.
    #[must_use]
    pub fn new(dimension: usize, distance_metric: DistanceMetric) -> Self {
        Self { nodes: HashMap::new(), entry_point: None, max_layer: 0, distance_metric, dimension }
    }

    /// Get a node by entity ID.
    #[inline]
    #[must_use]
    pub fn get_node(&self, entity_id: EntityId) -> Option<&HnswNode> {
        self.nodes.get(&entity_id)
    }

    /// Get a mutable node by entity ID.
    #[inline]
    pub fn get_node_mut(&mut self, entity_id: EntityId) -> Option<&mut HnswNode> {
        self.nodes.get_mut(&entity_id)
    }

    /// Check if a node exists in the graph.
    #[inline]
    #[must_use]
    pub fn contains(&self, entity_id: EntityId) -> bool {
        self.nodes.contains_key(&entity_id)
    }

    /// Get the number of nodes in the graph.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Calculate the distance between two embeddings.
    #[inline]
    #[must_use]
    pub fn distance(&self, a: &Embedding, b: &Embedding) -> f32 {
        match self.distance_metric {
            DistanceMetric::Euclidean => crate::distance::euclidean_distance(a, b),
            DistanceMetric::Cosine => crate::distance::cosine_distance(a, b),
            DistanceMetric::DotProduct => -crate::distance::dot_product(a, b), // Negate for min-distance
        }
    }

    /// Calculate the distance from a query to a node.
    #[inline]
    #[must_use]
    pub fn distance_to_node(&self, query: &Embedding, entity_id: EntityId) -> Option<f32> {
        self.nodes.get(&entity_id).map(|node| self.distance(query, &node.embedding))
    }

    /// Insert a node into the graph.
    pub fn insert_node(&mut self, node: HnswNode) {
        let entity_id = node.entity_id;
        let max_layer = node.max_layer;

        // Update entry point if this is the first node or has a higher layer
        if self.entry_point.is_none() || max_layer > self.max_layer {
            self.entry_point = Some(entity_id);
            self.max_layer = max_layer;
        }

        self.nodes.insert(entity_id, node);
    }

    /// Remove a node from the graph.
    pub fn remove_node(&mut self, entity_id: EntityId) -> Option<HnswNode> {
        let node = self.nodes.remove(&entity_id)?;

        // Remove connections to this node from all neighbors
        for layer in 0..=node.max_layer {
            for &neighbor_id in &node.connections[layer] {
                if let Some(neighbor) = self.nodes.get_mut(&neighbor_id) {
                    neighbor.remove_connection(layer, entity_id);
                }
            }
        }

        // Update entry point if we removed it
        if self.entry_point == Some(entity_id) {
            self.update_entry_point();
        }

        Some(node)
    }

    /// Find a new entry point after removal.
    fn update_entry_point(&mut self) {
        // Find the node with the highest max_layer
        let new_entry = self
            .nodes
            .iter()
            .max_by_key(|(_, node)| node.max_layer)
            .map(|(&id, node)| (id, node.max_layer));

        if let Some((id, max_layer)) = new_entry {
            self.entry_point = Some(id);
            self.max_layer = max_layer;
        } else {
            self.entry_point = None;
            self.max_layer = 0;
        }
    }
}

/// A candidate during HNSW search.
///
/// Used in the priority queue for greedy search.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// The entity ID of this candidate.
    pub entity_id: EntityId,
    /// The distance to the query.
    pub distance: f32,
}

impl Candidate {
    /// Create a new candidate.
    #[inline]
    #[must_use]
    pub const fn new(entity_id: EntityId, distance: f32) -> Self {
        Self { entity_id, distance }
    }
}

impl PartialEq for Candidate {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.entity_id == other.entity_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smallest distance first)
        // NaN values are treated as equal to maintain a total ordering for the heap.
        // In practice, NaN distances should not occur from valid distance calculations.
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// A max-heap candidate for tracking the worst element in the result set.
#[derive(Debug, Clone, Copy)]
pub struct MaxCandidate(pub Candidate);

impl PartialEq for MaxCandidate {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal ordering for max-heap (largest distance first)
        // NaN values are treated as equal to maintain a total ordering for the heap.
        // In practice, NaN distances should not occur from valid distance calculations.
        self.0.distance.partial_cmp(&other.0.distance).unwrap_or(Ordering::Equal)
    }
}

/// Search layer for a single entry point.
///
/// Performs a greedy search starting from the entry point, returning
/// the ef closest candidates to the query.
pub fn search_layer(
    graph: &HnswGraph,
    query: &Embedding,
    entry_points: &[EntityId],
    ef: usize,
    layer: usize,
) -> Vec<Candidate> {
    if entry_points.is_empty() {
        return Vec::new();
    }

    // Initialize candidates with entry points
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();
    let mut visited: HashSet<EntityId> = HashSet::new();

    for &ep in entry_points {
        if let Some(dist) = graph.distance_to_node(query, ep) {
            visited.insert(ep);
            let candidate = Candidate::new(ep, dist);
            candidates.push(candidate);
            results.push(MaxCandidate(candidate));
        }
    }

    // Greedy search
    while let Some(current) = candidates.pop() {
        // Get the furthest result
        let furthest_result = results.peek().map_or(f32::INFINITY, |c| c.0.distance);

        // If the closest candidate is further than the furthest result, we're done
        if current.distance > furthest_result {
            break;
        }

        // Explore neighbors
        if let Some(node) = graph.get_node(current.entity_id) {
            for &neighbor_id in node.connections_at(layer) {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                if let Some(neighbor_dist) = graph.distance_to_node(query, neighbor_id) {
                    let furthest_result = results.peek().map_or(f32::INFINITY, |c| c.0.distance);

                    // Only add if better than worst result or results not full
                    if results.len() < ef || neighbor_dist < furthest_result {
                        let neighbor_candidate = Candidate::new(neighbor_id, neighbor_dist);
                        candidates.push(neighbor_candidate);
                        results.push(MaxCandidate(neighbor_candidate));

                        // Trim results to ef size
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
    }

    // Convert results to vector, sorted by distance
    let mut result_vec: Vec<Candidate> = results.into_iter().map(|mc| mc.0).collect();
    result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result_vec
}

/// Select the best neighbors for a node using a simple heuristic.
///
/// This uses a simple approach: keep the M closest neighbors.
/// This is an alternative to [`select_neighbors_heuristic`] that
/// may be faster for small candidate sets.
pub fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<EntityId> {
    candidates.iter().take(m).map(|c| c.entity_id).collect()
}

/// Select neighbors using the heuristic algorithm (Algorithm 4 from the paper).
///
/// This algorithm tries to ensure diversity in the neighborhood by preferring
/// neighbors that are not too close to each other.
pub fn select_neighbors_heuristic(
    graph: &HnswGraph,
    _query: &Embedding,
    candidates: &[Candidate],
    m: usize,
    _extend_candidates: bool,
) -> Vec<EntityId> {
    if candidates.len() <= m {
        return candidates.iter().map(|c| c.entity_id).collect();
    }

    let mut selected: Vec<EntityId> = Vec::with_capacity(m);
    let mut remaining: Vec<Candidate> = candidates.to_vec();

    // Sort by distance
    remaining.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

    for candidate in remaining {
        if selected.len() >= m {
            break;
        }

        // Check if this candidate is good (diverse enough from already selected)
        let mut is_good = true;
        let candidate_embedding = match graph.get_node(candidate.entity_id) {
            Some(node) => &node.embedding,
            None => continue,
        };

        for &selected_id in &selected {
            if let Some(selected_node) = graph.get_node(selected_id) {
                let dist_to_selected =
                    graph.distance(candidate_embedding, &selected_node.embedding);
                // If candidate is closer to an already selected node than to the query,
                // it might not provide diverse coverage
                if dist_to_selected < candidate.distance {
                    is_good = false;
                    break;
                }
            }
        }

        if is_good || selected.is_empty() {
            selected.push(candidate.entity_id);
        }
    }

    // If we didn't get enough diverse neighbors, fill with closest remaining
    if selected.len() < m {
        let remaining: Vec<Candidate> =
            candidates.iter().filter(|c| !selected.contains(&c.entity_id)).copied().collect();

        for candidate in remaining {
            if selected.len() >= m {
                break;
            }
            selected.push(candidate.entity_id);
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding(dim: usize, value: f32) -> Embedding {
        Embedding::new(vec![value; dim]).unwrap()
    }

    #[test]
    fn test_hnsw_node_creation() {
        let embedding = create_test_embedding(4, 1.0);
        let node = HnswNode::new(EntityId::new(1), embedding.clone(), 2);

        assert_eq!(node.entity_id, EntityId::new(1));
        assert_eq!(node.max_layer, 2);
        assert_eq!(node.connections.len(), 3); // layers 0, 1, 2
    }

    #[test]
    fn test_node_connections() {
        let embedding = create_test_embedding(4, 1.0);
        let mut node = HnswNode::new(EntityId::new(1), embedding, 1);

        node.add_connection(0, EntityId::new(2));
        node.add_connection(0, EntityId::new(3));
        node.add_connection(1, EntityId::new(4));

        assert_eq!(node.connections_at(0), &[EntityId::new(2), EntityId::new(3)]);
        assert_eq!(node.connections_at(1), &[EntityId::new(4)]);

        node.remove_connection(0, EntityId::new(2));
        assert_eq!(node.connections_at(0), &[EntityId::new(3)]);
    }

    #[test]
    fn test_graph_insert_and_remove() {
        let mut graph = HnswGraph::new(4, DistanceMetric::Euclidean);

        let node1 = HnswNode::new(EntityId::new(1), create_test_embedding(4, 1.0), 2);
        let node2 = HnswNode::new(EntityId::new(2), create_test_embedding(4, 2.0), 1);

        graph.insert_node(node1);
        assert_eq!(graph.entry_point, Some(EntityId::new(1)));
        assert_eq!(graph.max_layer, 2);

        graph.insert_node(node2);
        assert_eq!(graph.entry_point, Some(EntityId::new(1))); // Still node 1 (higher layer)
        assert_eq!(graph.len(), 2);

        graph.remove_node(EntityId::new(1));
        assert_eq!(graph.entry_point, Some(EntityId::new(2)));
        assert_eq!(graph.max_layer, 1);
    }

    #[test]
    fn test_candidate_ordering() {
        let c1 = Candidate::new(EntityId::new(1), 1.0);
        let c2 = Candidate::new(EntityId::new(2), 2.0);
        let c3 = Candidate::new(EntityId::new(3), 0.5);

        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        heap.push(c1);
        heap.push(c2);
        heap.push(c3);

        // Min-heap: should pop smallest first
        assert_eq!(heap.pop().unwrap().entity_id, EntityId::new(3));
        assert_eq!(heap.pop().unwrap().entity_id, EntityId::new(1));
        assert_eq!(heap.pop().unwrap().entity_id, EntityId::new(2));
    }

    #[test]
    fn test_search_layer_empty() {
        let graph = HnswGraph::new(4, DistanceMetric::Euclidean);
        let query = create_test_embedding(4, 1.0);

        let results = search_layer(&graph, &query, &[], 10, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_layer_single_node() {
        let mut graph = HnswGraph::new(4, DistanceMetric::Euclidean);
        let node = HnswNode::new(EntityId::new(1), create_test_embedding(4, 1.0), 0);
        graph.insert_node(node);

        let query = create_test_embedding(4, 2.0);
        let results = search_layer(&graph, &query, &[EntityId::new(1)], 10, 0);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, EntityId::new(1));
    }
}
