//! Single-hop and multi-hop graph expansion.
//!
//! This module provides the [`Expand`] and [`ExpandAll`] operators for
//! traversing from a node to its neighbors.

use std::collections::{HashSet, VecDeque};

use manifoldb_core::{Edge, EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::{Direction, TraversalFilter, TraversalNode};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult};

/// Result of an expansion operation.
///
/// Contains the neighbor node and the edge used to reach it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandResult {
    /// The neighbor node's entity ID.
    pub node: EntityId,
    /// The ID of the edge connecting to this neighbor.
    pub edge_id: EdgeId,
    /// The direction in which this edge was traversed.
    pub direction: Direction,
}

impl ExpandResult {
    /// Create a new expansion result.
    #[inline]
    pub const fn new(node: EntityId, edge_id: EdgeId, direction: Direction) -> Self {
        Self { node, edge_id, direction }
    }
}

/// Single-hop graph expansion.
///
/// `Expand` traverses from a single node to its immediate neighbors,
/// optionally filtered by edge type or other criteria.
///
/// # Example
///
/// ```ignore
/// // Find all outgoing neighbors
/// let neighbors = Expand::neighbors(&tx, node_id, Direction::Outgoing)?;
///
/// // Find neighbors through specific edge type
/// let friends = Expand::neighbors_by_type(&tx, node_id, Direction::Outgoing, &EdgeType::new("FRIEND"))?;
/// ```
pub struct Expand;

impl Expand {
    /// Get all neighbors of a node in the specified direction.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `node` - The starting node
    /// * `direction` - Which direction to traverse
    ///
    /// # Returns
    ///
    /// A vector of expansion results with neighbor nodes and connecting edges.
    pub fn neighbors<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<Vec<ExpandResult>> {
        let mut results = Vec::new();

        // Get outgoing neighbors
        if direction.includes_outgoing() {
            AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(ExpandResult::new(edge.target, edge_id, Direction::Outgoing));
                }
                Ok(true)
            })?;
        }

        // Get incoming neighbors
        if direction.includes_incoming() {
            AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(ExpandResult::new(edge.source, edge_id, Direction::Incoming));
                }
                Ok(true)
            })?;
        }

        Ok(results)
    }

    /// Get neighbors of a node filtered by edge type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `node` - The starting node
    /// * `direction` - Which direction to traverse
    /// * `edge_type` - The edge type to filter by
    ///
    /// # Returns
    ///
    /// A vector of expansion results with neighbor nodes and connecting edges.
    pub fn neighbors_by_type<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
        edge_type: &EdgeType,
    ) -> GraphResult<Vec<ExpandResult>> {
        let mut results = Vec::new();

        // Get outgoing neighbors by type
        if direction.includes_outgoing() {
            AdjacencyIndex::for_each_outgoing_by_type(tx, node, edge_type, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(ExpandResult::new(edge.target, edge_id, Direction::Outgoing));
                }
                Ok(true)
            })?;
        }

        // Get incoming neighbors by type
        if direction.includes_incoming() {
            AdjacencyIndex::for_each_incoming_by_type(tx, node, edge_type, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(ExpandResult::new(edge.source, edge_id, Direction::Incoming));
                }
                Ok(true)
            })?;
        }

        Ok(results)
    }

    /// Get neighbors with full edge data.
    ///
    /// This is more expensive than `neighbors()` as it fetches full edge objects.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `node` - The starting node
    /// * `direction` - Which direction to traverse
    ///
    /// # Returns
    ///
    /// A vector of (`neighbor_id`, edge) tuples.
    pub fn neighbors_with_edges<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<Vec<(EntityId, Edge)>> {
        let mut results = Vec::new();

        // Get outgoing neighbors
        if direction.includes_outgoing() {
            let edges = EdgeStore::get_outgoing(tx, node)?;
            for edge in edges {
                results.push((edge.target, edge));
            }
        }

        // Get incoming neighbors
        if direction.includes_incoming() {
            let edges = EdgeStore::get_incoming(tx, node)?;
            for edge in edges {
                results.push((edge.source, edge));
            }
        }

        Ok(results)
    }

    /// Get only the neighbor node IDs without edge information.
    ///
    /// This is more efficient when edge data is not needed.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `node` - The starting node
    /// * `direction` - Which direction to traverse
    ///
    /// # Returns
    ///
    /// A vector of neighbor entity IDs.
    pub fn neighbor_ids<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
    ) -> GraphResult<Vec<EntityId>> {
        let mut results = Vec::new();

        // Get outgoing neighbors
        if direction.includes_outgoing() {
            AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(edge.target);
                }
                Ok(true)
            })?;
        }

        // Get incoming neighbors
        if direction.includes_incoming() {
            AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    results.push(edge.source);
                }
                Ok(true)
            })?;
        }

        Ok(results)
    }

    /// Get neighbors with filtering and early termination.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `node` - The starting node
    /// * `direction` - Which direction to traverse
    /// * `filter` - Filter to apply during traversal
    ///
    /// # Returns
    ///
    /// A vector of expansion results respecting the filter.
    pub fn neighbors_filtered<T: Transaction>(
        tx: &T,
        node: EntityId,
        direction: Direction,
        filter: &TraversalFilter,
    ) -> GraphResult<Vec<ExpandResult>> {
        let mut results = Vec::new();

        // Helper to process edges based on filter
        let process_edges = |edge_ids: &[EdgeId],
                             direction: Direction,
                             results: &mut Vec<ExpandResult>|
         -> GraphResult<bool> {
            for &edge_id in edge_ids {
                if filter.is_at_limit(results.len()) {
                    return Ok(false);
                }

                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    // Check edge type filter
                    if !filter.should_include_edge_type(&edge.edge_type) {
                        continue;
                    }

                    let neighbor = match direction {
                        Direction::Outgoing => edge.target,
                        Direction::Incoming => edge.source,
                        Direction::Both => unreachable!(),
                    };

                    // Check node filter
                    if !filter.should_include_node(neighbor) {
                        continue;
                    }

                    results.push(ExpandResult::new(neighbor, edge_id, direction));
                }
            }
            Ok(true)
        };

        // Process outgoing edges
        if direction.includes_outgoing() {
            let edge_ids = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            if !process_edges(&edge_ids, Direction::Outgoing, &mut results)? {
                return Ok(results);
            }
        }

        // Process incoming edges
        if direction.includes_incoming() && !filter.is_at_limit(results.len()) {
            let edge_ids = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            process_edges(&edge_ids, Direction::Incoming, &mut results)?;
        }

        Ok(results)
    }

    /// Expand from multiple starting nodes at once.
    ///
    /// This is useful for BFS-style traversals where you need to expand
    /// from multiple frontier nodes.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `nodes` - The starting nodes
    /// * `direction` - Which direction to traverse
    ///
    /// # Returns
    ///
    /// A vector of (`source_node`, `expansion_result`) tuples.
    pub fn expand_all<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        direction: Direction,
    ) -> GraphResult<Vec<(EntityId, ExpandResult)>> {
        let mut results = Vec::new();

        for &node in nodes {
            let neighbors = Self::neighbors(tx, node, direction)?;
            for result in neighbors {
                results.push((node, result));
            }
        }

        Ok(results)
    }
}

/// Multi-hop graph expansion with depth control.
///
/// `ExpandAll` performs a breadth-first traversal from a starting node,
/// visiting all nodes within a specified depth limit.
///
/// # Features
///
/// - Configurable depth limit (min/max)
/// - Cycle detection (each node visited once)
/// - Optional edge type filtering
/// - Memory-efficient BFS implementation
///
/// # Example
///
/// ```ignore
/// // Find all nodes within 3 hops
/// let nodes = ExpandAll::new(user_id, Direction::Outgoing)
///     .with_max_depth(3)
///     .execute(&tx)?;
///
/// // Find nodes between 2 and 4 hops away
/// let distant = ExpandAll::new(user_id, Direction::Both)
///     .with_min_depth(2)
///     .with_max_depth(4)
///     .execute(&tx)?;
/// ```
pub struct ExpandAll {
    /// Starting node for traversal.
    start: EntityId,
    /// Direction to traverse.
    direction: Direction,
    /// Minimum depth to include in results (default: 1).
    min_depth: usize,
    /// Maximum depth to traverse (default: unlimited).
    max_depth: Option<usize>,
    /// Filter for traversal.
    filter: TraversalFilter,
}

impl ExpandAll {
    /// Create a new multi-hop expansion.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting node
    /// * `direction` - Which direction to traverse
    pub fn new(start: EntityId, direction: Direction) -> Self {
        Self { start, direction, min_depth: 1, max_depth: None, filter: TraversalFilter::new() }
    }

    /// Set the minimum depth to include in results.
    ///
    /// Nodes closer than this depth will be traversed but not included
    /// in the results.
    pub const fn with_min_depth(mut self, min_depth: usize) -> Self {
        self.min_depth = min_depth;
        self
    }

    /// Set the maximum depth to traverse.
    ///
    /// Traversal will stop at this depth. Setting to 0 returns only the
    /// starting node.
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set both min and max depth.
    pub const fn with_depth_range(mut self, min: usize, max: usize) -> Self {
        self.min_depth = min;
        self.max_depth = Some(max);
        self
    }

    /// Filter to only traverse edges of the specified type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = self.filter.with_edge_type(edge_type);
        self
    }

    /// Set a result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.filter = self.filter.with_limit(limit);
        self
    }

    /// Exclude specific nodes from traversal.
    pub fn exclude_nodes(mut self, nodes: impl IntoIterator<Item = EntityId>) -> Self {
        self.filter = self.filter.exclude_nodes(nodes);
        self
    }

    /// Execute the multi-hop expansion.
    ///
    /// Returns nodes with their discovery depth.
    pub fn execute<T: Transaction>(self, tx: &T) -> GraphResult<Vec<TraversalNode>> {
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut results: Vec<TraversalNode> = Vec::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        // Start with the initial node
        visited.insert(self.start);
        queue.push_back((self.start, 0));

        // Include start node if min_depth is 0
        if self.min_depth == 0 {
            results.push(TraversalNode::new(self.start, 0));
        }

        while let Some((current, depth)) = queue.pop_front() {
            // Check if we've hit the result limit
            if self.filter.is_at_limit(results.len()) {
                break;
            }

            // Check if we should continue expanding
            let should_expand = self.max_depth.map_or(true, |max| depth < max);
            if !should_expand {
                continue;
            }

            // Get neighbors based on edge type filter
            let neighbors = if self.filter.edge_types.is_some() {
                Expand::neighbors_filtered(tx, current, self.direction, &self.filter)?
            } else {
                Expand::neighbors(tx, current, self.direction)?
            };

            for result in neighbors {
                // Check limit before adding more results
                if self.filter.is_at_limit(results.len()) {
                    break;
                }

                if visited.contains(&result.node) {
                    continue;
                }

                visited.insert(result.node);
                let next_depth = depth + 1;

                // Add to queue for further expansion
                queue.push_back((result.node, next_depth));

                // Add to results if within depth range
                if next_depth >= self.min_depth {
                    // Check node filter
                    if self.filter.should_include_node(result.node) {
                        results.push(TraversalNode::new(result.node, next_depth));
                    }
                }
            }
        }

        Ok(results)
    }

    /// Execute and return only the node IDs.
    pub fn collect_node_ids<T: Transaction>(self, tx: &T) -> GraphResult<Vec<EntityId>> {
        let nodes = self.execute(tx)?;
        Ok(nodes.into_iter().map(|n| n.id).collect())
    }

    /// Execute and count results without collecting.
    pub fn count<T: Transaction>(self, tx: &T) -> GraphResult<usize> {
        // For counting, we still need to traverse but don't need to store TraversalNodes
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut count = 0usize;
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(self.start);
        queue.push_back((self.start, 0));

        if self.min_depth == 0 {
            count += 1;
        }

        while let Some((current, depth)) = queue.pop_front() {
            if self.filter.is_at_limit(count) {
                break;
            }

            let should_expand = self.max_depth.map_or(true, |max| depth < max);
            if !should_expand {
                continue;
            }

            let neighbors = if self.filter.edge_types.is_some() {
                Expand::neighbors_filtered(tx, current, self.direction, &self.filter)?
            } else {
                Expand::neighbors(tx, current, self.direction)?
            };

            for result in neighbors {
                if self.filter.is_at_limit(count) {
                    break;
                }

                if visited.contains(&result.node) {
                    continue;
                }

                visited.insert(result.node);
                let next_depth = depth + 1;

                queue.push_back((result.node, next_depth));

                if next_depth >= self.min_depth && self.filter.should_include_node(result.node) {
                    count += 1;
                }
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_result_creation() {
        let result = ExpandResult::new(EntityId::new(1), EdgeId::new(10), Direction::Outgoing);
        assert_eq!(result.node, EntityId::new(1));
        assert_eq!(result.edge_id, EdgeId::new(10));
        assert_eq!(result.direction, Direction::Outgoing);
    }

    #[test]
    fn expand_all_builder() {
        let expand = ExpandAll::new(EntityId::new(1), Direction::Both)
            .with_min_depth(2)
            .with_max_depth(5)
            .with_edge_type("FRIEND")
            .with_limit(100);

        assert_eq!(expand.start, EntityId::new(1));
        assert_eq!(expand.direction, Direction::Both);
        assert_eq!(expand.min_depth, 2);
        assert_eq!(expand.max_depth, Some(5));
        assert_eq!(expand.filter.limit, Some(100));
    }

    #[test]
    fn expand_all_depth_range() {
        let expand = ExpandAll::new(EntityId::new(1), Direction::Outgoing).with_depth_range(1, 3);

        assert_eq!(expand.min_depth, 1);
        assert_eq!(expand.max_depth, Some(3));
    }
}
