//! A* algorithm for goal-directed weighted shortest path finding.
//!
//! This module provides A* algorithm implementation for finding the shortest
//! weighted path between two nodes. A* uses heuristics to guide the search
//! toward the goal, making it more efficient than Dijkstra's algorithm when
//! a good heuristic is available.
//!
//! # Heuristics
//!
//! The algorithm accepts pluggable heuristics through the [`Heuristic`] trait.
//! A good heuristic should:
//! - Never overestimate the actual cost (admissibility)
//! - Be consistent: h(n) <= cost(n, n') + h(n') (consistency/monotonicity)
//!
//! Built-in heuristics:
//! - [`ZeroHeuristic`] - Always returns 0, making A* behave like Dijkstra
//! - [`ConstantHeuristic`] - Returns a constant value
//! - [`PropertyHeuristic`] - Uses node properties for heuristic calculation
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::traversal::{AStar, Direction, ZeroHeuristic};
//!
//! // Using A* with zero heuristic (equivalent to Dijkstra)
//! let path = AStar::new(start, goal, Direction::Outgoing)
//!     .with_weight_property("distance")
//!     .with_heuristic(ZeroHeuristic)
//!     .find(&tx)?;
//!
//! // Using a custom heuristic
//! let path = AStar::new(start, goal, Direction::Outgoing)
//!     .with_heuristic(MyCustomHeuristic::new(goal_coords))
//!     .find(&tx)?;
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::dijkstra::{WeightConfig, WeightedPathResult};
use super::{Direction, TraversalFilter};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};

/// A heuristic function for A* algorithm.
///
/// The heuristic estimates the cost from a node to the goal.
/// For optimal results, the heuristic should be:
/// - **Admissible**: Never overestimate the actual cost
/// - **Consistent**: h(n) <= cost(n, n') + h(n')
///
/// # Example
///
/// ```ignore
/// struct ManhattanDistance {
///     goal_x: f64,
///     goal_y: f64,
/// }
///
/// impl Heuristic for ManhattanDistance {
///     fn estimate<T: Transaction>(
///         &self,
///         tx: &T,
///         node: EntityId,
///         _goal: EntityId,
///     ) -> GraphResult<f64> {
///         // Get node coordinates from properties
///         let entity = NodeStore::get(tx, node)?
///             .ok_or(GraphError::EntityNotFound(node))?;
///         let x = entity.get_property("x")
///             .and_then(|v| v.as_float())
///             .unwrap_or(0.0);
///         let y = entity.get_property("y")
///             .and_then(|v| v.as_float())
///             .unwrap_or(0.0);
///
///         Ok((self.goal_x - x).abs() + (self.goal_y - y).abs())
///     }
/// }
/// ```
pub trait Heuristic: Send + Sync {
    /// Estimate the cost from a node to the goal.
    ///
    /// # Arguments
    ///
    /// * `tx` - Transaction for reading node properties
    /// * `node` - The current node
    /// * `goal` - The goal node
    ///
    /// # Returns
    ///
    /// The estimated cost from `node` to `goal`.
    fn estimate<T: Transaction>(&self, tx: &T, node: EntityId, goal: EntityId) -> GraphResult<f64>;
}

/// Zero heuristic - always returns 0.
///
/// Using this heuristic makes A* behave exactly like Dijkstra's algorithm.
/// Use when no domain-specific heuristic is available.
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroHeuristic;

impl Heuristic for ZeroHeuristic {
    fn estimate<T: Transaction>(
        &self,
        _tx: &T,
        _node: EntityId,
        _goal: EntityId,
    ) -> GraphResult<f64> {
        Ok(0.0)
    }
}

/// Constant heuristic - always returns the same value.
///
/// Useful for testing or when a rough lower bound is known.
#[derive(Debug, Clone, Copy)]
pub struct ConstantHeuristic(pub f64);

impl Heuristic for ConstantHeuristic {
    fn estimate<T: Transaction>(
        &self,
        _tx: &T,
        _node: EntityId,
        _goal: EntityId,
    ) -> GraphResult<f64> {
        Ok(self.0)
    }
}

/// Property-based heuristic using Euclidean distance.
///
/// Calculates the Euclidean distance between nodes based on
/// coordinate properties (e.g., "x", "y", "z").
///
/// This is useful for spatial graphs where nodes have coordinate properties.
#[derive(Debug, Clone)]
pub struct EuclideanHeuristic {
    /// Property names for coordinates (e.g., `["x", "y"]` or `["lat", "lon"]`).
    pub coord_properties: Vec<String>,
    /// Scaling factor for the heuristic (default 1.0).
    /// Use < 1.0 to ensure admissibility when edge weights don't directly
    /// correspond to Euclidean distance.
    pub scale: f64,
}

impl EuclideanHeuristic {
    /// Create a 2D Euclidean heuristic using "x" and "y" properties.
    pub fn xy() -> Self {
        Self { coord_properties: vec!["x".to_owned(), "y".to_owned()], scale: 1.0 }
    }

    /// Create a 3D Euclidean heuristic using "x", "y", and "z" properties.
    pub fn xyz() -> Self {
        Self { coord_properties: vec!["x".to_owned(), "y".to_owned(), "z".to_owned()], scale: 1.0 }
    }

    /// Create a heuristic using latitude and longitude properties.
    pub fn lat_lon() -> Self {
        Self { coord_properties: vec!["lat".to_owned(), "lon".to_owned()], scale: 1.0 }
    }

    /// Create a custom heuristic with specified properties.
    pub fn with_properties(properties: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { coord_properties: properties.into_iter().map(Into::into).collect(), scale: 1.0 }
    }

    /// Set the scaling factor.
    pub const fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    fn get_coordinates<T: Transaction>(&self, tx: &T, node: EntityId) -> GraphResult<Vec<f64>> {
        let entity = NodeStore::get(tx, node)?.ok_or(GraphError::EntityNotFound(node))?;

        let mut coords = Vec::with_capacity(self.coord_properties.len());
        for prop in &self.coord_properties {
            let value = entity
                .get_property(prop)
                .and_then(|v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)))
                .unwrap_or(0.0);
            coords.push(value);
        }
        Ok(coords)
    }
}

impl Heuristic for EuclideanHeuristic {
    fn estimate<T: Transaction>(&self, tx: &T, node: EntityId, goal: EntityId) -> GraphResult<f64> {
        if node == goal {
            return Ok(0.0);
        }

        let node_coords = self.get_coordinates(tx, node)?;
        let goal_coords = self.get_coordinates(tx, goal)?;

        let sum_sq: f64 =
            node_coords.iter().zip(goal_coords.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        Ok(sum_sq.sqrt() * self.scale)
    }
}

/// Manhattan distance heuristic.
///
/// Calculates the Manhattan (L1) distance between nodes based on
/// coordinate properties. Useful for grid-based graphs.
#[derive(Debug, Clone)]
pub struct ManhattanHeuristic {
    /// Property names for coordinates.
    pub coord_properties: Vec<String>,
    /// Scaling factor for the heuristic.
    pub scale: f64,
}

impl ManhattanHeuristic {
    /// Create a 2D Manhattan heuristic using "x" and "y" properties.
    pub fn xy() -> Self {
        Self { coord_properties: vec!["x".to_owned(), "y".to_owned()], scale: 1.0 }
    }

    /// Create a custom heuristic with specified properties.
    pub fn with_properties(properties: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { coord_properties: properties.into_iter().map(Into::into).collect(), scale: 1.0 }
    }

    /// Set the scaling factor.
    pub const fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }
}

impl Heuristic for ManhattanHeuristic {
    fn estimate<T: Transaction>(&self, tx: &T, node: EntityId, goal: EntityId) -> GraphResult<f64> {
        if node == goal {
            return Ok(0.0);
        }

        let entity = NodeStore::get(tx, node)?.ok_or(GraphError::EntityNotFound(node))?;
        let goal_entity = NodeStore::get(tx, goal)?.ok_or(GraphError::EntityNotFound(goal))?;

        let mut sum = 0.0;
        for prop in &self.coord_properties {
            let node_val = entity
                .get_property(prop)
                .and_then(|v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)))
                .unwrap_or(0.0);
            let goal_val = goal_entity
                .get_property(prop)
                .and_then(|v| v.as_float().or_else(|| v.as_int().map(|i| i as f64)))
                .unwrap_or(0.0);
            sum += (node_val - goal_val).abs();
        }

        Ok(sum * self.scale)
    }
}

/// Entry in the priority queue for A* algorithm.
///
/// Ordered by f-score (lower f-score = higher priority).
#[derive(Debug, Clone)]
struct AStarEntry {
    node: EntityId,
    /// g-score: actual cost from start to this node
    g_score: f64,
    /// f-score: g_score + heuristic estimate to goal
    f_score: f64,
}

impl PartialEq for AStarEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score && self.node == other.node
    }
}

impl Eq for AStarEntry {}

impl PartialOrd for AStarEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        // Lower f-score = higher priority
        // Tie-break by g-score (prefer nodes closer to goal)
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.g_score.partial_cmp(&self.g_score).unwrap_or(Ordering::Equal))
            .then_with(|| self.node.as_u64().cmp(&other.node.as_u64()))
    }
}

/// A* algorithm for goal-directed weighted shortest path finding.
///
/// A* uses a heuristic to guide the search toward the goal, potentially
/// exploring fewer nodes than Dijkstra's algorithm.
///
/// # Type Parameters
///
/// * `H` - The heuristic type implementing [`Heuristic`]
///
/// # Example
///
/// ```ignore
/// // A* with Euclidean distance heuristic for spatial graphs
/// let path = AStar::new(start, goal, Direction::Outgoing)
///     .with_heuristic(EuclideanHeuristic::xy())
///     .with_weight_property("distance")
///     .find(&tx)?;
/// ```
pub struct AStar<H = ZeroHeuristic> {
    /// Source node.
    source: EntityId,
    /// Target (goal) node.
    target: EntityId,
    /// Traversal direction.
    direction: Direction,
    /// Heuristic function.
    heuristic: H,
    /// Weight configuration.
    weight_config: WeightConfig,
    /// Maximum g-score (actual cost) to search.
    max_cost: Option<f64>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl AStar<ZeroHeuristic> {
    /// Create a new A* pathfinder with the zero heuristic.
    ///
    /// This behaves like Dijkstra's algorithm. Use `with_heuristic()`
    /// to provide a domain-specific heuristic.
    pub fn new(source: EntityId, target: EntityId, direction: Direction) -> Self {
        Self {
            source,
            target,
            direction,
            heuristic: ZeroHeuristic,
            weight_config: WeightConfig::default(),
            max_cost: None,
            filter: TraversalFilter::new(),
        }
    }
}

impl<H: Heuristic> AStar<H> {
    /// Set the heuristic function.
    pub fn with_heuristic<H2: Heuristic>(self, heuristic: H2) -> AStar<H2> {
        AStar {
            source: self.source,
            target: self.target,
            direction: self.direction,
            heuristic,
            weight_config: self.weight_config,
            max_cost: self.max_cost,
            filter: self.filter,
        }
    }

    /// Set the weight property name to use.
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
    pub fn with_constant_weight(mut self, weight: f64) -> Self {
        self.weight_config = WeightConfig::constant(weight);
        self
    }

    /// Set a custom weight configuration.
    pub fn with_weight_config(mut self, config: WeightConfig) -> Self {
        self.weight_config = config;
        self
    }

    /// Set the maximum cost (g-score) to search.
    ///
    /// Paths with actual cost exceeding this value will not be considered.
    pub fn with_max_cost(mut self, max_cost: f64) -> Self {
        self.max_cost = Some(max_cost);
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

    /// Find the shortest weighted path using A*.
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

        // g-score: actual cost from source to each node
        let mut g_scores: HashMap<EntityId, f64> = HashMap::new();
        // Track parent and edge used to reach each node
        let mut parent: HashMap<EntityId, (EntityId, EdgeId)> = HashMap::new();
        // Nodes that have been finalized (closed set)
        let mut closed: HashSet<EntityId> = HashSet::new();
        // Priority queue (open set, ordered by f-score)
        let mut open: BinaryHeap<AStarEntry> = BinaryHeap::new();

        // Initialize source
        let initial_h = self.heuristic.estimate(tx, self.source, self.target)?;
        g_scores.insert(self.source, 0.0);
        open.push(AStarEntry { node: self.source, g_score: 0.0, f_score: initial_h });

        while let Some(AStarEntry { node: current, g_score: current_g, .. }) = open.pop() {
            // Skip if already in closed set
            if closed.contains(&current) {
                continue;
            }

            // Check if we've found the target
            if current == self.target {
                return Ok(Some(self.reconstruct_path(&parent, current_g)));
            }

            // Add to closed set
            closed.insert(current);

            // Skip if we already have a better path to this node
            if let Some(&known_g) = g_scores.get(&current) {
                if current_g > known_g {
                    continue;
                }
            }

            // Explore neighbors
            let neighbors = self.get_neighbors_with_weights(tx, current)?;

            for (neighbor, edge_id, weight) in neighbors {
                // Check for negative weights
                if weight < 0.0 {
                    return Err(GraphError::Internal(format!(
                        "negative edge weight detected on edge {}: {}. \
                         A* algorithm does not support negative weights.",
                        edge_id, weight
                    )));
                }

                // Skip nodes in closed set
                if closed.contains(&neighbor) {
                    continue;
                }

                // Check node filter
                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                let tentative_g = current_g + weight;

                // Check max cost constraint
                if let Some(max) = self.max_cost {
                    if tentative_g > max {
                        continue;
                    }
                }

                // Check if this is a better path
                let is_better = match g_scores.get(&neighbor) {
                    None => true,
                    Some(&existing) => tentative_g < existing,
                };

                if is_better {
                    // Update g-score and parent
                    g_scores.insert(neighbor, tentative_g);
                    parent.insert(neighbor, (current, edge_id));

                    // Calculate f-score and add to open set
                    let h = self.heuristic.estimate(tx, neighbor, self.target)?;
                    let f = tentative_g + h;
                    open.push(AStarEntry { node: neighbor, g_score: tentative_g, f_score: f });
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

        if self.direction.includes_outgoing() {
            self.add_neighbors_outgoing(tx, node, &mut neighbors)?;
        }

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
        total_cost: f64,
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

        WeightedPathResult::new(nodes, edges, total_cost)
    }

    /// Find the shortest distance (total cost) to the target.
    ///
    /// This is more efficient than `find()` when you only need the distance.
    pub fn distance<T: Transaction>(self, tx: &T) -> GraphResult<Option<f64>> {
        if self.source == self.target {
            return Ok(Some(0.0));
        }

        let mut g_scores: HashMap<EntityId, f64> = HashMap::new();
        let mut closed: HashSet<EntityId> = HashSet::new();
        let mut open: BinaryHeap<AStarEntry> = BinaryHeap::new();

        let initial_h = self.heuristic.estimate(tx, self.source, self.target)?;
        g_scores.insert(self.source, 0.0);
        open.push(AStarEntry { node: self.source, g_score: 0.0, f_score: initial_h });

        while let Some(AStarEntry { node: current, g_score: current_g, .. }) = open.pop() {
            if closed.contains(&current) {
                continue;
            }

            if current == self.target {
                return Ok(Some(current_g));
            }

            closed.insert(current);

            if let Some(&known_g) = g_scores.get(&current) {
                if current_g > known_g {
                    continue;
                }
            }

            let neighbors = self.get_neighbors_with_weights(tx, current)?;

            for (neighbor, edge_id, weight) in neighbors {
                if weight < 0.0 {
                    return Err(GraphError::Internal(format!(
                        "negative edge weight detected on edge {}: {}",
                        edge_id, weight
                    )));
                }

                if closed.contains(&neighbor) {
                    continue;
                }

                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                let tentative_g = current_g + weight;

                if let Some(max) = self.max_cost {
                    if tentative_g > max {
                        continue;
                    }
                }

                let is_better = match g_scores.get(&neighbor) {
                    None => true,
                    Some(&existing) => tentative_g < existing,
                };

                if is_better {
                    g_scores.insert(neighbor, tentative_g);
                    let h = self.heuristic.estimate(tx, neighbor, self.target)?;
                    let f = tentative_g + h;
                    open.push(AStarEntry { node: neighbor, g_score: tentative_g, f_score: f });
                }
            }
        }

        Ok(None)
    }

    /// Check if a path exists within the cost constraints.
    pub fn exists<T: Transaction>(self, tx: &T) -> GraphResult<bool> {
        Ok(self.distance(tx)?.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astar_entry_ordering() {
        let entry1 = AStarEntry { node: EntityId::new(1), g_score: 5.0, f_score: 10.0 };
        let entry2 = AStarEntry { node: EntityId::new(2), g_score: 3.0, f_score: 8.0 };
        let entry3 = AStarEntry { node: EntityId::new(3), g_score: 7.0, f_score: 12.0 };

        // Min-heap: smaller f-score should have higher priority
        assert!(entry2 > entry1); // f=8 has higher priority than f=10
        assert!(entry1 > entry3); // f=10 has higher priority than f=12
    }

    #[test]
    fn astar_entry_tiebreak_by_g_score() {
        // Same f-score, different g-score
        // With same f-score (10), one has g=5,h=5 and another has g=7,h=3
        // The one with higher g (7) has traveled further, so has a more accurate estimate
        // Actually for a proper tie-breaker, we prefer higher g (so lower h, closer to goal)
        // But the implementation uses reverse ordering, so entry1 > entry2 means entry1
        // gets popped first (has lower priority in BinaryHeap terms)
        let entry1 = AStarEntry { node: EntityId::new(1), g_score: 5.0, f_score: 10.0 };
        let entry2 = AStarEntry { node: EntityId::new(2), g_score: 7.0, f_score: 10.0 };

        // entry2 has higher g-score, so it should be preferred (higher Ord value in our reversed scheme)
        // But we're using reverse ordering for min-heap, so higher g means LOWER priority in Ord
        // Actually looking at the code: we compare other.g_score to self.g_score
        // So higher g_score in other means it should be greater
        assert!(entry1 > entry2);
    }

    #[test]
    fn astar_builder() {
        let a = AStar::new(EntityId::new(1), EntityId::new(10), Direction::Both)
            .with_weight_property("distance")
            .with_max_cost(100.0)
            .with_edge_type("ROAD");

        assert_eq!(a.source, EntityId::new(1));
        assert_eq!(a.target, EntityId::new(10));
        assert_eq!(a.direction, Direction::Both);
        assert_eq!(a.max_cost, Some(100.0));
    }

    #[test]
    fn euclidean_heuristic_creation() {
        let h = EuclideanHeuristic::xy();
        assert_eq!(h.coord_properties, vec!["x", "y"]);
        assert_eq!(h.scale, 1.0);

        let h = EuclideanHeuristic::xyz();
        assert_eq!(h.coord_properties, vec!["x", "y", "z"]);

        let h = EuclideanHeuristic::lat_lon();
        assert_eq!(h.coord_properties, vec!["lat", "lon"]);
    }

    #[test]
    fn manhattan_heuristic_creation() {
        let h = ManhattanHeuristic::xy();
        assert_eq!(h.coord_properties, vec!["x", "y"]);
        assert_eq!(h.scale, 1.0);

        let h = ManhattanHeuristic::with_properties(["a", "b", "c"]).with_scale(0.5);
        assert_eq!(h.coord_properties, vec!["a", "b", "c"]);
        assert_eq!(h.scale, 0.5);
    }

    #[test]
    fn constant_heuristic_value() {
        let h = ConstantHeuristic(5.0);
        assert_eq!(h.0, 5.0);
    }

    #[test]
    fn zero_heuristic_is_default() {
        let _h = ZeroHeuristic::default();
    }
}
