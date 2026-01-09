//! Breadth-first search (BFS) graph traversal.
//!
//! This module provides a BFS traversal algorithm that visits nodes level by level,
//! starting from a given node. It supports:
//!
//! - Configurable maximum depth
//! - Edge type filtering
//! - Direction control (outgoing, incoming, both)
//! - Path tracking for each visited node
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::traversal::{BfsTraversal, Direction};
//!
//! // Find all nodes within 3 hops
//! let results = BfsTraversal::new(start_node, Direction::Outgoing)
//!     .with_max_depth(3)
//!     .with_edge_type("KNOWS")
//!     .execute(&tx)?;
//!
//! for result in results {
//!     println!("Node {} at depth {}", result.node.as_u64(), result.depth);
//! }
//! ```

use std::collections::{HashSet, VecDeque};

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::{Direction, TraversalFilter};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult};

/// Result of a BFS traversal for a single node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BfsResult {
    /// The visited node.
    pub node: EntityId,
    /// The depth at which this node was discovered.
    pub depth: usize,
    /// The path from start node to this node (node IDs).
    /// Empty if path tracking is disabled.
    pub path: Vec<EntityId>,
}

impl BfsResult {
    /// Create a new BFS result.
    #[inline]
    pub fn new(node: EntityId, depth: usize, path: Vec<EntityId>) -> Self {
        Self { node, depth, path }
    }

    /// Create a BFS result without path tracking.
    #[inline]
    pub fn without_path(node: EntityId, depth: usize) -> Self {
        Self { node, depth, path: Vec::new() }
    }
}

/// BFS traversal configuration and executor.
///
/// Performs a breadth-first search starting from a given node,
/// visiting all reachable nodes level by level.
pub struct BfsTraversal {
    /// Starting node for traversal.
    start: EntityId,
    /// Direction to traverse.
    direction: Direction,
    /// Maximum depth to traverse.
    max_depth: Option<usize>,
    /// Filter for traversal.
    filter: TraversalFilter,
    /// Whether to track paths to each node.
    track_paths: bool,
}

impl BfsTraversal {
    /// Create a new BFS traversal starting from the given node.
    pub fn new(start: EntityId, direction: Direction) -> Self {
        Self {
            start,
            direction,
            max_depth: None,
            filter: TraversalFilter::new(),
            track_paths: false,
        }
    }

    /// Set the maximum depth to traverse.
    ///
    /// Nodes beyond this depth will not be visited.
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

    /// Exclude specific nodes from traversal.
    pub fn exclude_nodes(mut self, nodes: impl IntoIterator<Item = EntityId>) -> Self {
        self.filter = self.filter.exclude_nodes(nodes);
        self
    }

    /// Set a result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.filter = self.filter.with_limit(limit);
        self
    }

    /// Enable path tracking for each visited node.
    ///
    /// When enabled, each result will include the full path from
    /// the start node to that node.
    pub const fn with_path_tracking(mut self) -> Self {
        self.track_paths = true;
        self
    }

    /// Execute the BFS traversal.
    ///
    /// Returns all visited nodes with their depth and optionally paths.
    pub fn execute<T: Transaction>(self, tx: &T) -> GraphResult<Vec<BfsResult>> {
        const INITIAL_CAPACITY: usize = 256;

        let mut visited: HashSet<EntityId> = HashSet::with_capacity(INITIAL_CAPACITY);
        let mut results: Vec<BfsResult> = Vec::with_capacity(INITIAL_CAPACITY);

        // Queue entries: (node, depth, path_to_node)
        let mut queue: VecDeque<(EntityId, usize, Vec<EntityId>)> =
            VecDeque::with_capacity(INITIAL_CAPACITY);

        // Start with the initial node
        visited.insert(self.start);
        let initial_path = if self.track_paths { vec![self.start] } else { Vec::new() };
        queue.push_back((self.start, 0, initial_path));

        // Add start node to results at depth 0
        results.push(BfsResult::new(
            self.start,
            0,
            if self.track_paths { vec![self.start] } else { Vec::new() },
        ));

        while let Some((current, depth, current_path)) = queue.pop_front() {
            // Check if we've hit the result limit
            if self.filter.is_at_limit(results.len()) {
                break;
            }

            // Check if we should continue expanding
            let should_expand = self.max_depth.map_or(true, |max| depth < max);
            if !should_expand {
                continue;
            }

            // Get neighbors based on direction and edge type filter
            let neighbors = self.get_neighbors(tx, current)?;

            for (neighbor, _edge_id) in neighbors {
                // Check limit before adding more results
                if self.filter.is_at_limit(results.len()) {
                    break;
                }

                if visited.contains(&neighbor) {
                    continue;
                }

                // Check node filter
                if !self.filter.should_include_node(neighbor) {
                    continue;
                }

                visited.insert(neighbor);
                let next_depth = depth + 1;

                // Build path to neighbor
                let neighbor_path = if self.track_paths {
                    let mut path = current_path.clone();
                    path.push(neighbor);
                    path
                } else {
                    Vec::new()
                };

                // Add to results
                results.push(BfsResult::new(neighbor, next_depth, neighbor_path.clone()));

                // Add to queue for further expansion
                queue.push_back((neighbor, next_depth, neighbor_path));
            }
        }

        Ok(results)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfs_result_creation() {
        let result = BfsResult::new(EntityId::new(1), 2, vec![EntityId::new(0), EntityId::new(1)]);
        assert_eq!(result.node, EntityId::new(1));
        assert_eq!(result.depth, 2);
        assert_eq!(result.path.len(), 2);
    }

    #[test]
    fn bfs_result_without_path() {
        let result = BfsResult::without_path(EntityId::new(5), 3);
        assert_eq!(result.node, EntityId::new(5));
        assert_eq!(result.depth, 3);
        assert!(result.path.is_empty());
    }

    #[test]
    fn bfs_traversal_builder() {
        let traversal = BfsTraversal::new(EntityId::new(1), Direction::Both)
            .with_max_depth(5)
            .with_edge_type("KNOWS")
            .with_limit(100)
            .with_path_tracking();

        assert_eq!(traversal.start, EntityId::new(1));
        assert_eq!(traversal.direction, Direction::Both);
        assert_eq!(traversal.max_depth, Some(5));
        assert_eq!(traversal.filter.limit, Some(100));
        assert!(traversal.track_paths);
    }

    #[test]
    fn bfs_traversal_default_values() {
        let traversal = BfsTraversal::new(EntityId::new(1), Direction::Outgoing);

        assert_eq!(traversal.start, EntityId::new(1));
        assert_eq!(traversal.direction, Direction::Outgoing);
        assert!(traversal.max_depth.is_none());
        assert!(traversal.filter.limit.is_none());
        assert!(!traversal.track_paths);
    }
}
