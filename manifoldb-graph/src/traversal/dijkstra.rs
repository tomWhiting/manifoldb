//! Dijkstra's algorithm for weighted shortest path finding.
//!
//! This module provides Dijkstra's algorithm for finding the shortest weighted
//! path between two nodes in a graph. Unlike BFS-based shortest path which
//! treats all edges as having weight 1, Dijkstra's algorithm considers edge
//! weights stored as properties.
//!
//! # Edge Weights
//!
//! Weights are extracted from edge properties using a configurable weight
//! function. The default behavior looks for a "weight" property, falling back
//! to 1.0 if not present.
//!
//! # Negative Weights
//!
//! This implementation detects negative edge weights and returns an error.
//! For graphs with negative weights, Bellman-Ford algorithm would be needed,
//! which is not currently implemented.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::traversal::{Dijkstra, Direction, WeightedPathResult};
//!
//! // Find shortest weighted path between two locations
//! let path = Dijkstra::new(city_a, city_b, Direction::Outgoing)
//!     .with_weight_property("distance")
//!     .find(&tx)?;
//!
//! if let Some(result) = path {
//!     println!("Total distance: {}", result.total_weight);
//!     println!("Path: {:?}", result.nodes);
//! }
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use manifoldb_core::{Edge, EdgeId, EdgeType, EntityId, Value};
use manifoldb_storage::Transaction;

use super::{Direction, TraversalFilter};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult};

/// A weighted path through the graph.
///
/// Represents a sequence of nodes and edges from a source to a target,
/// along with the total accumulated weight.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedPathResult {
    /// The nodes in the path, from source to target.
    pub nodes: Vec<EntityId>,
    /// The edges connecting the nodes.
    /// Length is `nodes.len() - 1`.
    pub edges: Vec<EdgeId>,
    /// The total weight of the path.
    pub total_weight: f64,
    /// The number of edges in the path.
    pub length: usize,
}

impl WeightedPathResult {
    /// Create a new weighted path result.
    pub fn new(nodes: Vec<EntityId>, edges: Vec<EdgeId>, total_weight: f64) -> Self {
        let length = edges.len();
        Self { nodes, edges, total_weight, length }
    }

    /// Create a path for a single node (source == target).
    pub fn single_node(node: EntityId) -> Self {
        Self { nodes: vec![node], edges: Vec::new(), total_weight: 0.0, length: 0 }
    }

    /// Get the source node.
    pub fn source(&self) -> EntityId {
        self.nodes[0]
    }

    /// Get the target node.
    ///
    /// # Panics
    /// Panics if the path is empty (should not happen with valid construction).
    pub fn target(&self) -> EntityId {
        *self.nodes.last().expect("path has at least one node")
    }

    /// Check if the path is empty (source == target).
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Weight extraction configuration.
///
/// Defines how edge weights are extracted from edge properties.
#[derive(Debug, Clone)]
pub enum WeightConfig {
    /// Use a specific property name as the weight.
    /// Falls back to `default_weight` if the property is not found.
    Property {
        /// The name of the property to use as weight.
        name: String,
        /// The default weight if the property is not found.
        default_weight: f64,
    },
    /// Use a constant weight for all edges.
    Constant(f64),
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self::Property { name: "weight".to_owned(), default_weight: 1.0 }
    }
}

impl WeightConfig {
    /// Create a weight config that uses a specific property.
    pub fn property(name: impl Into<String>) -> Self {
        Self::Property { name: name.into(), default_weight: 1.0 }
    }

    /// Create a weight config that uses a specific property with a default value.
    pub fn property_with_default(name: impl Into<String>, default: f64) -> Self {
        Self::Property { name: name.into(), default_weight: default }
    }

    /// Create a weight config that uses a constant weight.
    pub const fn constant(weight: f64) -> Self {
        Self::Constant(weight)
    }

    /// Extract the weight from an edge based on this configuration.
    ///
    /// # Returns
    /// - `Ok(weight)` - The extracted weight
    /// - `Err` - If the weight value is invalid (non-numeric)
    pub fn extract_weight(&self, edge: &Edge) -> GraphResult<f64> {
        match self {
            Self::Constant(w) => Ok(*w),
            Self::Property { name, default_weight } => match edge.get_property(name) {
                Some(Value::Float(f)) => Ok(*f),
                Some(Value::Int(i)) => Ok(*i as f64),
                Some(other) => Err(GraphError::InvalidWeight {
                    edge_id: edge.id,
                    message: format!("non-numeric weight property '{}': {:?}", name, other),
                }),
                None => Ok(*default_weight),
            },
        }
    }
}

/// Entry in the priority queue for Dijkstra's algorithm.
///
/// Ordered by distance (lower distance = higher priority).
#[derive(Debug, Clone)]
struct DijkstraEntry {
    node: EntityId,
    distance: f64,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node == other.node
    }
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        // Lower distance = higher priority
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.as_u64().cmp(&other.node.as_u64()))
    }
}

/// Dijkstra's algorithm for weighted shortest path finding.
///
/// Finds the shortest weighted path between two nodes using Dijkstra's
/// algorithm with a binary heap.
///
/// # Features
///
/// - Configurable edge weight extraction
/// - Detects negative weights (returns error)
/// - Configurable traversal direction
/// - Optional edge type filtering
/// - Optional maximum weight limit
///
/// # Example
///
/// ```ignore
/// // Find shortest weighted path
/// let path = Dijkstra::new(start, end, Direction::Outgoing)
///     .with_weight_property("cost")
///     .find(&tx)?;
///
/// // With maximum weight constraint
/// let path = Dijkstra::new(start, end, Direction::Both)
///     .with_max_weight(100.0)
///     .find(&tx)?;
/// ```
pub struct Dijkstra {
    /// Source node.
    source: EntityId,
    /// Target node.
    target: EntityId,
    /// Traversal direction.
    direction: Direction,
    /// Weight configuration.
    weight_config: WeightConfig,
    /// Maximum total weight to search (acts like a distance cutoff).
    max_weight: Option<f64>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl Dijkstra {
    /// Create a new Dijkstra shortest path finder.
    ///
    /// # Arguments
    ///
    /// * `source` - The starting node
    /// * `target` - The destination node
    /// * `direction` - Which direction to traverse edges
    pub fn new(source: EntityId, target: EntityId, direction: Direction) -> Self {
        Self {
            source,
            target,
            direction,
            weight_config: WeightConfig::default(),
            max_weight: None,
            filter: TraversalFilter::new(),
        }
    }

    /// Set the weight property name to use.
    ///
    /// The weight will be extracted from this edge property.
    /// Defaults to "weight" with a fallback value of 1.0.
    pub fn with_weight_property(mut self, property: impl Into<String>) -> Self {
        self.weight_config = WeightConfig::property(property);
        self
    }

    /// Set the weight property name with a custom default value.
    pub fn with_weight_property_default(
        mut self,
        property: impl Into<String>,
        default: f64,
    ) -> Self {
        self.weight_config = WeightConfig::property_with_default(property, default);
        self
    }

    /// Use a constant weight for all edges.
    ///
    /// This effectively makes Dijkstra behave like BFS when weight is 1.0.
    pub fn with_constant_weight(mut self, weight: f64) -> Self {
        self.weight_config = WeightConfig::constant(weight);
        self
    }

    /// Set a custom weight configuration.
    pub fn with_weight_config(mut self, config: WeightConfig) -> Self {
        self.weight_config = config;
        self
    }

    /// Set the maximum total weight to search.
    ///
    /// Paths with total weight exceeding this value will not be considered.
    pub fn with_max_weight(mut self, max_weight: f64) -> Self {
        self.max_weight = Some(max_weight);
        self
    }

    /// Filter to only traverse edges of the specified type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = self.filter.with_edge_type(edge_type);
        self
    }

    /// Filter to only traverse edges of the specified types.
    pub fn with_edge_types(mut self, edge_types: impl IntoIterator<Item = EdgeType>) -> Self {
        self.filter = self.filter.with_edge_types(edge_types);
        self
    }

    /// Exclude specific nodes from the path.
    pub fn exclude_nodes(mut self, nodes: impl IntoIterator<Item = EntityId>) -> Self {
        self.filter = self.filter.exclude_nodes(nodes);
        self
    }

    /// Find the shortest weighted path.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(WeightedPathResult))` if a path exists
    /// - `Ok(None)` if no path exists within the constraints
    /// - `Err(GraphError)` if a negative weight is detected or other error occurs
    pub fn find<T: Transaction>(self, tx: &T) -> GraphResult<Option<WeightedPathResult>> {
        // Handle same source and target
        if self.source == self.target {
            return Ok(Some(WeightedPathResult::single_node(self.source)));
        }

        // Distance from source to each node
        let mut distances: HashMap<EntityId, f64> = HashMap::new();
        // Track parent and edge used to reach each node
        let mut parent: HashMap<EntityId, (EntityId, EdgeId)> = HashMap::new();
        // Nodes that have been finalized
        let mut finalized: HashSet<EntityId> = HashSet::new();
        // Priority queue (min-heap by distance)
        let mut heap: BinaryHeap<DijkstraEntry> = BinaryHeap::new();

        // Initialize source
        distances.insert(self.source, 0.0);
        heap.push(DijkstraEntry { node: self.source, distance: 0.0 });

        while let Some(DijkstraEntry { node: current, distance: current_dist }) = heap.pop() {
            // Skip if already finalized
            if finalized.contains(&current) {
                continue;
            }

            // Check if we've found the target
            if current == self.target {
                return Ok(Some(self.reconstruct_path(&parent, current_dist)));
            }

            // Mark as finalized
            finalized.insert(current);

            // Skip if we already have a better path to this node
            // (can happen due to duplicate entries in heap)
            if let Some(&known_dist) = distances.get(&current) {
                if current_dist > known_dist {
                    continue;
                }
            }

            // Explore neighbors
            let neighbors = self.get_neighbors_with_weights(tx, current)?;

            for (neighbor, edge_id, weight) in neighbors {
                // Check for negative weights
                if weight < 0.0 {
                    return Err(GraphError::InvalidWeight {
                        edge_id,
                        message: format!(
                            "negative weight {} not supported by Dijkstra's algorithm",
                            weight
                        ),
                    });
                }

                // Skip finalized nodes
                if finalized.contains(&neighbor) {
                    continue;
                }

                // Check node filter
                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                let new_dist = current_dist + weight;

                // Check max weight constraint
                if let Some(max) = self.max_weight {
                    if new_dist > max {
                        continue;
                    }
                }

                // Update if this is a better path
                let is_better = match distances.get(&neighbor) {
                    None => true,
                    Some(&existing) => new_dist < existing,
                };

                if is_better {
                    distances.insert(neighbor, new_dist);
                    parent.insert(neighbor, (current, edge_id));
                    heap.push(DijkstraEntry { node: neighbor, distance: new_dist });
                }
            }
        }

        Ok(None)
    }

    /// Get neighbors with their edge weights.
    fn get_neighbors_with_weights<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
    ) -> GraphResult<Vec<(EntityId, EdgeId, f64)>> {
        let mut neighbors = Vec::new();

        // Get outgoing neighbors
        if self.direction.includes_outgoing() {
            self.add_neighbors_outgoing(tx, node, &mut neighbors)?;
        }

        // Get incoming neighbors
        if self.direction.includes_incoming() {
            self.add_neighbors_incoming(tx, node, &mut neighbors)?;
        }

        Ok(neighbors)
    }

    fn add_neighbors_outgoing<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
        neighbors: &mut Vec<(EntityId, EdgeId, f64)>,
    ) -> GraphResult<()> {
        match &self.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_outgoing_by_type(tx, node, edge_type, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            let weight = self.weight_config.extract_weight(&edge)?;
                            neighbors.push((edge.target, edge_id, weight));
                        }
                        Ok(true)
                    })?;
                }
            }
            None => {
                AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        let weight = self.weight_config.extract_weight(&edge)?;
                        neighbors.push((edge.target, edge_id, weight));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    fn add_neighbors_incoming<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
        neighbors: &mut Vec<(EntityId, EdgeId, f64)>,
    ) -> GraphResult<()> {
        match &self.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_incoming_by_type(tx, node, edge_type, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            let weight = self.weight_config.extract_weight(&edge)?;
                            neighbors.push((edge.source, edge_id, weight));
                        }
                        Ok(true)
                    })?;
                }
            }
            None => {
                AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        let weight = self.weight_config.extract_weight(&edge)?;
                        neighbors.push((edge.source, edge_id, weight));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    /// Reconstruct the path from source to target using the parent map.
    fn reconstruct_path(
        &self,
        parent: &HashMap<EntityId, (EntityId, EdgeId)>,
        total_weight: f64,
    ) -> WeightedPathResult {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut current = self.target;

        // Trace back from target to source
        while let Some(&(prev, edge_id)) = parent.get(&current) {
            nodes.push(current);
            edges.push(edge_id);
            current = prev;
        }

        // Add source
        nodes.push(self.source);

        // Reverse to get source -> target order
        nodes.reverse();
        edges.reverse();

        WeightedPathResult::new(nodes, edges, total_weight)
    }

    /// Convenience method: find shortest weighted path with default settings.
    pub fn find_path<T: Transaction>(
        tx: &T,
        source: EntityId,
        target: EntityId,
        direction: Direction,
    ) -> GraphResult<Option<WeightedPathResult>> {
        Self::new(source, target, direction).find(tx)
    }

    /// Find the shortest distance (total weight) to the target.
    ///
    /// This is more efficient than `find()` when you only need the distance.
    pub fn distance<T: Transaction>(self, tx: &T) -> GraphResult<Option<f64>> {
        if self.source == self.target {
            return Ok(Some(0.0));
        }

        let mut distances: HashMap<EntityId, f64> = HashMap::new();
        let mut finalized: HashSet<EntityId> = HashSet::new();
        let mut heap: BinaryHeap<DijkstraEntry> = BinaryHeap::new();

        distances.insert(self.source, 0.0);
        heap.push(DijkstraEntry { node: self.source, distance: 0.0 });

        while let Some(DijkstraEntry { node: current, distance: current_dist }) = heap.pop() {
            if finalized.contains(&current) {
                continue;
            }

            if current == self.target {
                return Ok(Some(current_dist));
            }

            finalized.insert(current);

            if let Some(&known_dist) = distances.get(&current) {
                if current_dist > known_dist {
                    continue;
                }
            }

            let neighbors = self.get_neighbors_with_weights(tx, current)?;

            for (neighbor, edge_id, weight) in neighbors {
                if weight < 0.0 {
                    return Err(GraphError::InvalidWeight {
                        edge_id,
                        message: format!(
                            "negative weight {} not supported by Dijkstra's algorithm",
                            weight
                        ),
                    });
                }

                if finalized.contains(&neighbor) {
                    continue;
                }

                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                let new_dist = current_dist + weight;

                if let Some(max) = self.max_weight {
                    if new_dist > max {
                        continue;
                    }
                }

                let is_better = match distances.get(&neighbor) {
                    None => true,
                    Some(&existing) => new_dist < existing,
                };

                if is_better {
                    distances.insert(neighbor, new_dist);
                    heap.push(DijkstraEntry { node: neighbor, distance: new_dist });
                }
            }
        }

        Ok(None)
    }

    /// Check if a path exists within the weight constraints.
    pub fn exists<T: Transaction>(self, tx: &T) -> GraphResult<bool> {
        Ok(self.distance(tx)?.is_some())
    }
}

/// Find single-source shortest paths to all reachable nodes.
///
/// This runs Dijkstra's algorithm from a single source and returns
/// the shortest distance to all reachable nodes.
pub struct SingleSourceDijkstra {
    /// Source node.
    source: EntityId,
    /// Traversal direction.
    direction: Direction,
    /// Weight configuration.
    weight_config: WeightConfig,
    /// Maximum total weight to search.
    max_weight: Option<f64>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl SingleSourceDijkstra {
    /// Create a new single-source Dijkstra finder.
    pub fn new(source: EntityId, direction: Direction) -> Self {
        Self {
            source,
            direction,
            weight_config: WeightConfig::default(),
            max_weight: None,
            filter: TraversalFilter::new(),
        }
    }

    /// Set the weight property name to use.
    pub fn with_weight_property(mut self, property: impl Into<String>) -> Self {
        self.weight_config = WeightConfig::property(property);
        self
    }

    /// Set the maximum total weight to search.
    pub fn with_max_weight(mut self, max_weight: f64) -> Self {
        self.max_weight = Some(max_weight);
        self
    }

    /// Filter to only traverse edges of the specified type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = self.filter.with_edge_type(edge_type);
        self
    }

    /// Compute shortest distances to all reachable nodes.
    ///
    /// # Returns
    ///
    /// A map from node ID to (distance, parent node, edge used).
    /// The source node maps to (0.0, None).
    pub fn compute<T: Transaction>(
        self,
        tx: &T,
    ) -> GraphResult<HashMap<EntityId, (f64, Option<(EntityId, EdgeId)>)>> {
        let mut results: HashMap<EntityId, (f64, Option<(EntityId, EdgeId)>)> = HashMap::new();
        let mut finalized: HashSet<EntityId> = HashSet::new();
        let mut heap: BinaryHeap<DijkstraEntry> = BinaryHeap::new();

        // Initialize source
        results.insert(self.source, (0.0, None));
        heap.push(DijkstraEntry { node: self.source, distance: 0.0 });

        while let Some(DijkstraEntry { node: current, distance: current_dist }) = heap.pop() {
            if finalized.contains(&current) {
                continue;
            }

            finalized.insert(current);

            // Skip if we already have a better path
            if let Some(&(known_dist, _)) = results.get(&current) {
                if current_dist > known_dist {
                    continue;
                }
            }

            // Explore neighbors
            let neighbors = self.get_neighbors_with_weights(tx, current)?;

            for (neighbor, edge_id, weight) in neighbors {
                if weight < 0.0 {
                    return Err(GraphError::InvalidWeight {
                        edge_id,
                        message: format!(
                            "negative weight {} not supported by Dijkstra's algorithm",
                            weight
                        ),
                    });
                }

                if finalized.contains(&neighbor) {
                    continue;
                }

                if !self.filter.should_include_node(neighbor) {
                    continue;
                }

                let new_dist = current_dist + weight;

                if let Some(max) = self.max_weight {
                    if new_dist > max {
                        continue;
                    }
                }

                let is_better = match results.get(&neighbor) {
                    None => true,
                    Some((existing, _)) => new_dist < *existing,
                };

                if is_better {
                    results.insert(neighbor, (new_dist, Some((current, edge_id))));
                    heap.push(DijkstraEntry { node: neighbor, distance: new_dist });
                }
            }
        }

        Ok(results)
    }

    /// Get neighbors with their edge weights.
    fn get_neighbors_with_weights<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
    ) -> GraphResult<Vec<(EntityId, EdgeId, f64)>> {
        let mut neighbors = Vec::new();

        if self.direction.includes_outgoing() {
            match &self.filter.edge_types {
                Some(types) => {
                    for edge_type in types {
                        AdjacencyIndex::for_each_outgoing_by_type(
                            tx,
                            node,
                            edge_type,
                            |edge_id| {
                                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                                    let weight = self.weight_config.extract_weight(&edge)?;
                                    neighbors.push((edge.target, edge_id, weight));
                                }
                                Ok(true)
                            },
                        )?;
                    }
                }
                None => {
                    AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            let weight = self.weight_config.extract_weight(&edge)?;
                            neighbors.push((edge.target, edge_id, weight));
                        }
                        Ok(true)
                    })?;
                }
            }
        }

        if self.direction.includes_incoming() {
            match &self.filter.edge_types {
                Some(types) => {
                    for edge_type in types {
                        AdjacencyIndex::for_each_incoming_by_type(
                            tx,
                            node,
                            edge_type,
                            |edge_id| {
                                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                                    let weight = self.weight_config.extract_weight(&edge)?;
                                    neighbors.push((edge.source, edge_id, weight));
                                }
                                Ok(true)
                            },
                        )?;
                    }
                }
                None => {
                    AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            let weight = self.weight_config.extract_weight(&edge)?;
                            neighbors.push((edge.source, edge_id, weight));
                        }
                        Ok(true)
                    })?;
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
    fn weighted_path_result_single_node() {
        let path = WeightedPathResult::single_node(EntityId::new(1));
        assert_eq!(path.source(), EntityId::new(1));
        assert_eq!(path.target(), EntityId::new(1));
        assert_eq!(path.length, 0);
        assert_eq!(path.total_weight, 0.0);
        assert!(path.is_empty());
    }

    #[test]
    fn weighted_path_result_multi_node() {
        let nodes = vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)];
        let edges = vec![EdgeId::new(10), EdgeId::new(20)];
        let path = WeightedPathResult::new(nodes, edges, 5.5);

        assert_eq!(path.source(), EntityId::new(1));
        assert_eq!(path.target(), EntityId::new(3));
        assert_eq!(path.length, 2);
        assert_eq!(path.total_weight, 5.5);
        assert!(!path.is_empty());
    }

    #[test]
    fn weight_config_default() {
        let config = WeightConfig::default();
        match config {
            WeightConfig::Property { name, default_weight } => {
                assert_eq!(name, "weight");
                assert_eq!(default_weight, 1.0);
            }
            _ => panic!("expected Property variant"),
        }
    }

    #[test]
    fn weight_config_constant() {
        let config = WeightConfig::constant(2.5);
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST");
        assert_eq!(config.extract_weight(&edge).unwrap(), 2.5);
    }

    #[test]
    fn weight_config_property_float() {
        let config = WeightConfig::property("cost");
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST")
            .with_property("cost", 3.5f64);
        assert_eq!(config.extract_weight(&edge).unwrap(), 3.5);
    }

    #[test]
    fn weight_config_property_int() {
        let config = WeightConfig::property("cost");
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST")
            .with_property("cost", 42i64);
        assert_eq!(config.extract_weight(&edge).unwrap(), 42.0);
    }

    #[test]
    fn weight_config_property_missing_uses_default() {
        let config = WeightConfig::property_with_default("cost", 1.5);
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST");
        assert_eq!(config.extract_weight(&edge).unwrap(), 1.5);
    }

    #[test]
    fn weight_config_property_non_numeric_error() {
        let config = WeightConfig::property("cost");
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST")
            .with_property("cost", "not a number");
        assert!(config.extract_weight(&edge).is_err());
    }

    #[test]
    fn dijkstra_builder() {
        let d = Dijkstra::new(EntityId::new(1), EntityId::new(10), Direction::Both)
            .with_weight_property("distance")
            .with_max_weight(100.0)
            .with_edge_type("ROAD");

        assert_eq!(d.source, EntityId::new(1));
        assert_eq!(d.target, EntityId::new(10));
        assert_eq!(d.direction, Direction::Both);
        assert_eq!(d.max_weight, Some(100.0));
    }

    #[test]
    fn dijkstra_entry_ordering() {
        let entry1 = DijkstraEntry { node: EntityId::new(1), distance: 5.0 };
        let entry2 = DijkstraEntry { node: EntityId::new(2), distance: 3.0 };
        let entry3 = DijkstraEntry { node: EntityId::new(3), distance: 7.0 };

        // Min-heap: smaller distance should have higher priority
        assert!(entry2 > entry1); // 3.0 has higher priority than 5.0
        assert!(entry1 > entry3); // 5.0 has higher priority than 7.0
    }
}
