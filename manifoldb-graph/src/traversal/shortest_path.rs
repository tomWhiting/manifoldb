//! Shortest path finding algorithms.
//!
//! This module provides BFS-based shortest path algorithms for finding
//! the shortest path between two nodes in a graph.

// Allow expect - the invariant is guaranteed by the data structure
#![allow(clippy::expect_used)]

use std::collections::{HashMap, HashSet, VecDeque};

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::{Direction, TraversalFilter};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult};

/// A path through the graph.
///
/// Represents a sequence of nodes and edges from a source to a target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathResult {
    /// The nodes in the path, from source to target.
    pub nodes: Vec<EntityId>,
    /// The edges connecting the nodes.
    /// Length is `nodes.len() - 1`.
    pub edges: Vec<EdgeId>,
    /// The total length of the path (number of edges).
    pub length: usize,
}

impl PathResult {
    /// Create a new path result.
    fn new(nodes: Vec<EntityId>, edges: Vec<EdgeId>) -> Self {
        let length = edges.len();
        Self { nodes, edges, length }
    }

    /// Create a path for a single node (source == target).
    fn single_node(node: EntityId) -> Self {
        Self { nodes: vec![node], edges: Vec::new(), length: 0 }
    }

    /// Get the source node.
    pub fn source(&self) -> EntityId {
        self.nodes[0]
    }

    /// Get the target node.
    pub fn target(&self) -> EntityId {
        *self.nodes.last().expect("path has at least one node")
    }

    /// Check if the path is empty (source == target).
    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// BFS-based shortest path finder.
///
/// Finds the shortest unweighted path between two nodes using
/// breadth-first search.
///
/// # Features
///
/// - Unweighted shortest path (all edges have weight 1)
/// - Configurable traversal direction
/// - Optional edge type filtering
/// - Maximum depth limit for bounded searches
///
/// # Example
///
/// ```ignore
/// // Find shortest path between two users
/// let path = ShortestPath::find(&tx, user_a, user_b, Direction::Both)?;
///
/// if let Some(result) = path {
///     println!("Path length: {}", result.length);
///     println!("Path: {:?}", result.nodes);
/// }
///
/// // Find path following only FRIEND edges
/// let path = ShortestPath::new(user_a, user_b, Direction::Both)
///     .with_edge_type("FRIEND")
///     .find(&tx)?;
/// ```
pub struct ShortestPath {
    /// Source node.
    source: EntityId,
    /// Target node.
    target: EntityId,
    /// Traversal direction.
    direction: Direction,
    /// Maximum path length to search.
    max_depth: Option<usize>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl ShortestPath {
    /// Create a new shortest path finder.
    ///
    /// # Arguments
    ///
    /// * `source` - The starting node
    /// * `target` - The destination node
    /// * `direction` - Which direction to traverse edges
    pub fn new(source: EntityId, target: EntityId, direction: Direction) -> Self {
        Self { source, target, direction, max_depth: None, filter: TraversalFilter::new() }
    }

    /// Set the maximum path length to search.
    ///
    /// If no path of this length or shorter is found, returns None.
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
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

    /// Find the shortest path.
    ///
    /// # Returns
    ///
    /// - `Some(PathResult)` if a path exists
    /// - `None` if no path exists within the constraints
    pub fn find<T: Transaction>(self, tx: &T) -> GraphResult<Option<PathResult>> {
        // Handle same source and target
        if self.source == self.target {
            return Ok(Some(PathResult::single_node(self.source)));
        }

        // BFS from source
        let mut visited: HashSet<EntityId> = HashSet::new();
        // Maps each node to (previous_node, edge_used)
        let mut parent: HashMap<EntityId, (EntityId, EdgeId)> = HashMap::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(self.source);
        queue.push_back((self.source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            // Check depth limit
            if let Some(max) = self.max_depth {
                if depth >= max {
                    continue;
                }
            }

            // Get neighbors
            let neighbors = self.get_neighbors(tx, current)?;

            for (neighbor, edge_id) in neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }

                // Check node filter
                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                visited.insert(neighbor);
                parent.insert(neighbor, (current, edge_id));

                // Found target
                if neighbor == self.target {
                    return Ok(Some(self.reconstruct_path(&parent)));
                }

                queue.push_back((neighbor, depth + 1));
            }
        }

        Ok(None)
    }

    /// Get neighbors considering direction and edge type filters.
    fn get_neighbors<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
    ) -> GraphResult<Vec<(EntityId, EdgeId)>> {
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
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &self.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_outgoing_by_type(tx, node, edge_type, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.target, edge_id));
                        }
                        Ok(true)
                    })?;
                }
            }
            None => {
                AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        neighbors.push((edge.target, edge_id));
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
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &self.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_incoming_by_type(tx, node, edge_type, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.source, edge_id));
                        }
                        Ok(true)
                    })?;
                }
            }
            None => {
                AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        neighbors.push((edge.source, edge_id));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    /// Reconstruct the path from source to target using the parent map.
    fn reconstruct_path(&self, parent: &HashMap<EntityId, (EntityId, EdgeId)>) -> PathResult {
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

        PathResult::new(nodes, edges)
    }

    /// Convenience method: find shortest path with default settings.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The starting node
    /// * `target` - The destination node
    /// * `direction` - Which direction to traverse
    pub fn find_path<T: Transaction>(
        tx: &T,
        source: EntityId,
        target: EntityId,
        direction: Direction,
    ) -> GraphResult<Option<PathResult>> {
        Self::new(source, target, direction).find(tx)
    }

    /// Check if a path exists between two nodes.
    ///
    /// This is more efficient than `find()` when you only need to know
    /// if a path exists, not what it is.
    pub fn exists<T: Transaction>(self, tx: &T) -> GraphResult<bool> {
        // Handle same source and target
        if self.source == self.target {
            return Ok(true);
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(self.source);
        queue.push_back((self.source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            if let Some(max) = self.max_depth {
                if depth >= max {
                    continue;
                }
            }

            let neighbors = self.get_neighbors(tx, current)?;

            for (neighbor, _) in neighbors {
                if neighbor == self.target {
                    return Ok(true);
                }

                if visited.contains(&neighbor) {
                    continue;
                }

                if !self.filter.should_include_node(neighbor) {
                    continue;
                }

                visited.insert(neighbor);
                queue.push_back((neighbor, depth + 1));
            }
        }

        Ok(false)
    }

    /// Find the distance between two nodes (path length).
    ///
    /// This is more efficient than `find()` when you only need the distance.
    pub fn distance<T: Transaction>(self, tx: &T) -> GraphResult<Option<usize>> {
        if self.source == self.target {
            return Ok(Some(0));
        }

        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(self.source);
        queue.push_back((self.source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            if let Some(max) = self.max_depth {
                if depth >= max {
                    continue;
                }
            }

            let neighbors = self.get_neighbors(tx, current)?;

            for (neighbor, _) in neighbors {
                if neighbor == self.target {
                    return Ok(Some(depth + 1));
                }

                if visited.contains(&neighbor) {
                    continue;
                }

                if !self.filter.should_include_node(neighbor) {
                    continue;
                }

                visited.insert(neighbor);
                queue.push_back((neighbor, depth + 1));
            }
        }

        Ok(None)
    }
}

/// Find all shortest paths between two nodes.
///
/// When multiple paths of the same shortest length exist, this
/// function returns all of them.
pub struct AllShortestPaths {
    /// Source node.
    source: EntityId,
    /// Target node.
    target: EntityId,
    /// Traversal direction.
    direction: Direction,
    /// Maximum path length to search.
    max_depth: Option<usize>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl AllShortestPaths {
    /// Create a new finder for all shortest paths.
    pub fn new(source: EntityId, target: EntityId, direction: Direction) -> Self {
        Self { source, target, direction, max_depth: None, filter: TraversalFilter::new() }
    }

    /// Set the maximum path length to search.
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Filter to only traverse edges of the specified type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = self.filter.with_edge_type(edge_type);
        self
    }

    /// Find all shortest paths.
    ///
    /// # Returns
    ///
    /// A vector of all paths with the shortest length.
    /// Empty if no path exists.
    pub fn find<T: Transaction>(self, tx: &T) -> GraphResult<Vec<PathResult>> {
        if self.source == self.target {
            return Ok(vec![PathResult::single_node(self.source)]);
        }

        // BFS with tracking all parents at shortest distance
        let mut visited_at_depth: HashMap<EntityId, usize> = HashMap::new();
        // Maps node -> list of (parent, edge)
        let mut parents: HashMap<EntityId, Vec<(EntityId, EdgeId)>> = HashMap::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();
        let mut target_depth: Option<usize> = None;

        visited_at_depth.insert(self.source, 0);
        queue.push_back((self.source, 0));

        while let Some((current, depth)) = queue.pop_front() {
            // If we've found target and current depth exceeds target depth, stop
            if let Some(td) = target_depth {
                if depth >= td {
                    continue;
                }
            }

            // Check max depth
            if let Some(max) = self.max_depth {
                if depth >= max {
                    continue;
                }
            }

            let neighbors = self.get_neighbors(tx, current)?;

            for (neighbor, edge_id) in neighbors {
                let next_depth = depth + 1;

                // Check if we've seen this node before
                if let Some(&prev_depth) = visited_at_depth.get(&neighbor) {
                    // Only add parent if at same depth (for multiple shortest paths)
                    if prev_depth == next_depth {
                        parents.entry(neighbor).or_default().push((current, edge_id));
                    }
                    continue;
                }

                // Check node filter
                if neighbor != self.target && !self.filter.should_include_node(neighbor) {
                    continue;
                }

                visited_at_depth.insert(neighbor, next_depth);
                parents.entry(neighbor).or_default().push((current, edge_id));

                if neighbor == self.target {
                    target_depth = Some(next_depth);
                } else {
                    queue.push_back((neighbor, next_depth));
                }
            }
        }

        // Reconstruct all paths
        if target_depth.is_some() {
            Ok(self.reconstruct_all_paths(&parents))
        } else {
            Ok(Vec::new())
        }
    }

    fn get_neighbors<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
    ) -> GraphResult<Vec<(EntityId, EdgeId)>> {
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
                                    neighbors.push((edge.target, edge_id));
                                }
                                Ok(true)
                            },
                        )?;
                    }
                }
                None => {
                    AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.target, edge_id));
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
                                    neighbors.push((edge.source, edge_id));
                                }
                                Ok(true)
                            },
                        )?;
                    }
                }
                None => {
                    AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.source, edge_id));
                        }
                        Ok(true)
                    })?;
                }
            }
        }

        Ok(neighbors)
    }

    fn reconstruct_all_paths(
        &self,
        parents: &HashMap<EntityId, Vec<(EntityId, EdgeId)>>,
    ) -> Vec<PathResult> {
        let mut paths = Vec::new();
        let mut current_path_nodes = vec![self.target];
        let mut current_path_edges = Vec::new();

        self.backtrack_paths(
            parents,
            self.target,
            &mut current_path_nodes,
            &mut current_path_edges,
            &mut paths,
        );

        paths
    }

    fn backtrack_paths(
        &self,
        parents: &HashMap<EntityId, Vec<(EntityId, EdgeId)>>,
        current: EntityId,
        path_nodes: &mut Vec<EntityId>,
        path_edges: &mut Vec<EdgeId>,
        results: &mut Vec<PathResult>,
    ) {
        if current == self.source {
            // We've reached the source - build path by iterating in reverse
            // without cloning and then reversing
            let path_len = path_edges.len();
            let mut nodes = Vec::with_capacity(path_nodes.len());
            let mut edges = Vec::with_capacity(path_len);

            // Iterate in reverse order to build forward path
            for &node in path_nodes.iter().rev() {
                nodes.push(node);
            }
            for &edge in path_edges.iter().rev() {
                edges.push(edge);
            }

            results.push(PathResult::new(nodes, edges));
            return;
        }

        if let Some(parent_list) = parents.get(&current) {
            for &(parent, edge_id) in parent_list {
                path_nodes.push(parent);
                path_edges.push(edge_id);

                self.backtrack_paths(parents, parent, path_nodes, path_edges, results);

                path_nodes.pop();
                path_edges.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_result_single_node() {
        let path = PathResult::single_node(EntityId::new(1));
        assert_eq!(path.source(), EntityId::new(1));
        assert_eq!(path.target(), EntityId::new(1));
        assert_eq!(path.length, 0);
        assert!(path.is_empty());
    }

    #[test]
    fn path_result_multi_node() {
        let nodes = vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)];
        let edges = vec![EdgeId::new(10), EdgeId::new(20)];
        let path = PathResult::new(nodes, edges);

        assert_eq!(path.source(), EntityId::new(1));
        assert_eq!(path.target(), EntityId::new(3));
        assert_eq!(path.length, 2);
        assert!(!path.is_empty());
    }

    #[test]
    fn shortest_path_builder() {
        let sp = ShortestPath::new(EntityId::new(1), EntityId::new(10), Direction::Both)
            .with_max_depth(5)
            .with_edge_type("FRIEND");

        assert_eq!(sp.source, EntityId::new(1));
        assert_eq!(sp.target, EntityId::new(10));
        assert_eq!(sp.direction, Direction::Both);
        assert_eq!(sp.max_depth, Some(5));
    }

    #[test]
    fn all_shortest_paths_builder() {
        let asp = AllShortestPaths::new(EntityId::new(1), EntityId::new(10), Direction::Outgoing)
            .with_max_depth(3)
            .with_edge_type("FOLLOWS");

        assert_eq!(asp.source, EntityId::new(1));
        assert_eq!(asp.target, EntityId::new(10));
        assert_eq!(asp.max_depth, Some(3));
    }
}
