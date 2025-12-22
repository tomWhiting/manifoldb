//! Graph traversal algorithms.
//!
//! This module provides traversal primitives for graph exploration, including
//! single-hop expansion, multi-hop traversal, shortest path finding, and
//! path pattern matching.
//!
//! # Overview
//!
//! Graph traversal is fundamental to querying graph databases. This module
//! provides the building blocks used by the query engine:
//!
//! - [`Expand`] - Single-hop traversal from a node to its neighbors
//! - [`ExpandAll`] - Multi-hop traversal with depth control
//! - [`ShortestPath`] - BFS-based shortest path finding
//! - [`PathPattern`] - Pattern matching for paths
//! - [`TraversalIterator`] - Lazy iteration over traversal results
//!
//! # Direction
//!
//! All traversal operations support three directions:
//!
//! - [`Direction::Outgoing`] - Follow edges from source to target
//! - [`Direction::Incoming`] - Follow edges from target to source
//! - [`Direction::Both`] - Follow edges in both directions
//!
//! # Memory Efficiency
//!
//! Traversal operations are designed for memory efficiency on large graphs:
//!
//! - Use iterators for lazy evaluation
//! - Support early termination with limits
//! - Allow filtering during traversal to prune the search space
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::traversal::{Expand, ExpandAll, Direction, TraversalConfig};
//!
//! // Single-hop: find all friends of a user
//! let friends = Expand::neighbors(&tx, user_id, Direction::Outgoing)?
//!     .with_edge_type("FRIEND")
//!     .collect()?;
//!
//! // Multi-hop: find all users within 3 hops
//! let nearby = ExpandAll::new(&tx, user_id, Direction::Outgoing)
//!     .with_max_depth(3)
//!     .collect_nodes()?;
//!
//! // Shortest path between two users
//! let path = ShortestPath::find(&tx, user_a, user_b, Direction::Both)?;
//! ```

mod expand;
mod iterator;
mod pattern;
mod shortest_path;

pub use expand::{Expand, ExpandAll, ExpandResult};
pub use iterator::{
    NeighborIterator, NeighborIteratorAdapter, TraversalConfig, TraversalIterator,
    TraversalIteratorAdapter,
};
pub use pattern::{PathPattern, PathStep, PatternBuilder, PatternMatch, StepFilter};
pub use shortest_path::{AllShortestPaths, PathResult, ShortestPath};

use manifoldb_core::{EdgeType, EntityId};

/// Direction for graph traversal.
///
/// Specifies which edges to follow when traversing from a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Direction {
    /// Follow outgoing edges (source -> target).
    ///
    /// When traversing from node A, find nodes B where an edge A -> B exists.
    #[default]
    Outgoing,

    /// Follow incoming edges (target <- source).
    ///
    /// When traversing from node A, find nodes B where an edge B -> A exists.
    Incoming,

    /// Follow edges in both directions.
    ///
    /// When traversing from node A, find nodes B where either A -> B or B -> A exists.
    Both,
}

impl Direction {
    /// Returns true if this direction includes outgoing edges.
    #[inline]
    pub const fn includes_outgoing(self) -> bool {
        matches!(self, Self::Outgoing | Self::Both)
    }

    /// Returns true if this direction includes incoming edges.
    #[inline]
    pub const fn includes_incoming(self) -> bool {
        matches!(self, Self::Incoming | Self::Both)
    }
}

/// Filter for traversal operations.
///
/// Used to limit which edges and nodes are considered during traversal.
#[derive(Debug, Clone, Default)]
pub struct TraversalFilter {
    /// Only traverse edges of these types.
    pub edge_types: Option<Vec<EdgeType>>,
    /// Exclude these nodes from traversal.
    pub exclude_nodes: Option<Vec<EntityId>>,
    /// Maximum number of results to return.
    pub limit: Option<usize>,
}

impl TraversalFilter {
    /// Create a new empty filter.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter to only traverse edges of the specified type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        let types = self.edge_types.get_or_insert_with(Vec::new);
        types.push(edge_type.into());
        self
    }

    /// Filter to only traverse edges of the specified types.
    pub fn with_edge_types(mut self, edge_types: impl IntoIterator<Item = EdgeType>) -> Self {
        let types = self.edge_types.get_or_insert_with(Vec::new);
        types.extend(edge_types);
        self
    }

    /// Exclude the specified node from traversal results.
    pub fn exclude_node(mut self, node: EntityId) -> Self {
        let nodes = self.exclude_nodes.get_or_insert_with(Vec::new);
        nodes.push(node);
        self
    }

    /// Exclude the specified nodes from traversal results.
    pub fn exclude_nodes(mut self, nodes: impl IntoIterator<Item = EntityId>) -> Self {
        let exclude = self.exclude_nodes.get_or_insert_with(Vec::new);
        exclude.extend(nodes);
        self
    }

    /// Limit the number of results.
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if a node should be included based on exclusion list.
    #[inline]
    pub fn should_include_node(&self, node: EntityId) -> bool {
        match &self.exclude_nodes {
            Some(excluded) => !excluded.contains(&node),
            None => true,
        }
    }

    /// Check if an edge type should be included.
    #[inline]
    pub fn should_include_edge_type(&self, edge_type: &EdgeType) -> bool {
        match &self.edge_types {
            Some(types) => types.contains(edge_type),
            None => true,
        }
    }

    /// Check if we've hit the result limit.
    #[inline]
    pub fn is_at_limit(&self, count: usize) -> bool {
        self.limit.is_some_and(|l| count >= l)
    }
}

/// A node with traversal metadata.
///
/// Returned by traversal operations to provide context about how a node
/// was reached during traversal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalNode {
    /// The entity ID of the node.
    pub id: EntityId,
    /// The depth at which this node was discovered (0 = starting node).
    pub depth: usize,
}

impl TraversalNode {
    /// Create a new traversal node.
    #[inline]
    pub const fn new(id: EntityId, depth: usize) -> Self {
        Self { id, depth }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_includes_outgoing() {
        assert!(Direction::Outgoing.includes_outgoing());
        assert!(!Direction::Incoming.includes_outgoing());
        assert!(Direction::Both.includes_outgoing());
    }

    #[test]
    fn direction_includes_incoming() {
        assert!(!Direction::Outgoing.includes_incoming());
        assert!(Direction::Incoming.includes_incoming());
        assert!(Direction::Both.includes_incoming());
    }

    #[test]
    fn direction_default() {
        assert_eq!(Direction::default(), Direction::Outgoing);
    }

    #[test]
    fn filter_edge_types() {
        let filter = TraversalFilter::new().with_edge_type("FRIEND").with_edge_type("FOLLOWS");

        assert!(filter.should_include_edge_type(&EdgeType::new("FRIEND")));
        assert!(filter.should_include_edge_type(&EdgeType::new("FOLLOWS")));
        assert!(!filter.should_include_edge_type(&EdgeType::new("BLOCKS")));
    }

    #[test]
    fn filter_exclude_nodes() {
        let node1 = EntityId::new(1);
        let node2 = EntityId::new(2);
        let node3 = EntityId::new(3);

        let filter = TraversalFilter::new().exclude_node(node1).exclude_node(node2);

        assert!(!filter.should_include_node(node1));
        assert!(!filter.should_include_node(node2));
        assert!(filter.should_include_node(node3));
    }

    #[test]
    fn filter_limit() {
        let filter = TraversalFilter::new().with_limit(10);

        assert!(!filter.is_at_limit(0));
        assert!(!filter.is_at_limit(9));
        assert!(filter.is_at_limit(10));
        assert!(filter.is_at_limit(11));
    }

    #[test]
    fn empty_filter_allows_all() {
        let filter = TraversalFilter::new();

        assert!(filter.should_include_node(EntityId::new(1)));
        assert!(filter.should_include_edge_type(&EdgeType::new("ANY")));
        assert!(!filter.is_at_limit(1000));
    }

    #[test]
    fn traversal_node_creation() {
        let node = TraversalNode::new(EntityId::new(42), 3);
        assert_eq!(node.id, EntityId::new(42));
        assert_eq!(node.depth, 3);
    }
}
