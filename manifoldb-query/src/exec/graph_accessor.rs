//! Graph storage accessor trait for query execution.
//!
//! This module provides the [`GraphAccessor`] trait that abstracts
//! graph traversal operations for use in query execution. The trait
//! is object-safe and can be stored in the execution context.

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_graph::traversal::Direction;

/// Result of a neighbor expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeighborResult {
    /// The neighbor entity ID.
    pub node: EntityId,
    /// The edge ID connecting to this neighbor.
    pub edge_id: EdgeId,
    /// The direction the edge was traversed.
    pub direction: Direction,
}

impl NeighborResult {
    /// Create a new neighbor result.
    pub const fn new(node: EntityId, edge_id: EdgeId, direction: Direction) -> Self {
        Self { node, edge_id, direction }
    }
}

/// Result of a multi-hop expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalResult {
    /// The reached entity ID.
    pub node: EntityId,
    /// The edge ID used to reach this node (from the previous node).
    pub edge_id: Option<EdgeId>,
    /// The depth at which this node was discovered.
    pub depth: usize,
}

impl TraversalResult {
    /// Create a new traversal result.
    pub const fn new(node: EntityId, edge_id: Option<EdgeId>, depth: usize) -> Self {
        Self { node, edge_id, depth }
    }
}

/// Graph accessor error type.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GraphAccessError {
    /// No graph storage is available.
    #[error("no graph storage available")]
    NoStorage,
    /// An internal error occurred.
    #[error("graph error: {0}")]
    Internal(String),
}

/// Result type for graph accessor operations.
pub type GraphAccessResult<T> = Result<T, GraphAccessError>;

/// A trait for accessing graph storage during query execution.
///
/// This trait is object-safe and provides the graph operations
/// needed by the query executor. It abstracts over the actual
/// storage transaction type.
pub trait GraphAccessor: Send + Sync {
    /// Get neighbors of a node in the specified direction.
    ///
    /// Returns immediate neighbors (single hop).
    fn neighbors(
        &self,
        node: EntityId,
        direction: Direction,
    ) -> GraphAccessResult<Vec<NeighborResult>>;

    /// Get neighbors filtered by edge type.
    fn neighbors_by_type(
        &self,
        node: EntityId,
        direction: Direction,
        edge_type: &EdgeType,
    ) -> GraphAccessResult<Vec<NeighborResult>>;

    /// Get neighbors filtered by multiple edge types.
    fn neighbors_by_types(
        &self,
        node: EntityId,
        direction: Direction,
        edge_types: &[EdgeType],
    ) -> GraphAccessResult<Vec<NeighborResult>>;

    /// Perform multi-hop expansion from a node.
    ///
    /// Returns all nodes reachable within the depth range.
    fn expand_all(
        &self,
        node: EntityId,
        direction: Direction,
        min_depth: usize,
        max_depth: Option<usize>,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphAccessResult<Vec<TraversalResult>>;
}

/// A null implementation of `GraphAccessor` that returns no results.
///
/// Used when no graph storage is configured.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullGraphAccessor;

impl GraphAccessor for NullGraphAccessor {
    fn neighbors(
        &self,
        _node: EntityId,
        _direction: Direction,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        Err(GraphAccessError::NoStorage)
    }

    fn neighbors_by_type(
        &self,
        _node: EntityId,
        _direction: Direction,
        _edge_type: &EdgeType,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        Err(GraphAccessError::NoStorage)
    }

    fn neighbors_by_types(
        &self,
        _node: EntityId,
        _direction: Direction,
        _edge_types: &[EdgeType],
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        Err(GraphAccessError::NoStorage)
    }

    fn expand_all(
        &self,
        _node: EntityId,
        _direction: Direction,
        _min_depth: usize,
        _max_depth: Option<usize>,
        _edge_types: Option<&[EdgeType]>,
    ) -> GraphAccessResult<Vec<TraversalResult>> {
        Err(GraphAccessError::NoStorage)
    }
}

/// A concrete implementation of `GraphAccessor` backed by a storage transaction.
///
/// This wraps a transaction reference and delegates to the actual graph traversal code.
pub struct TransactionGraphAccessor<T> {
    tx: T,
}

impl<T> TransactionGraphAccessor<T> {
    /// Create a new accessor wrapping a transaction.
    pub fn new(tx: T) -> Self {
        Self { tx }
    }
}

impl<T> GraphAccessor for TransactionGraphAccessor<T>
where
    T: manifoldb_storage::Transaction + Send + Sync,
{
    fn neighbors(
        &self,
        node: EntityId,
        direction: Direction,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        manifoldb_graph::traversal::Expand::neighbors(&self.tx, node, direction)
            .map(|results| {
                results
                    .into_iter()
                    .map(|r| NeighborResult::new(r.node, r.edge_id, r.direction))
                    .collect()
            })
            .map_err(|e| GraphAccessError::Internal(e.to_string()))
    }

    fn neighbors_by_type(
        &self,
        node: EntityId,
        direction: Direction,
        edge_type: &EdgeType,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        manifoldb_graph::traversal::Expand::neighbors_by_type(&self.tx, node, direction, edge_type)
            .map(|results| {
                results
                    .into_iter()
                    .map(|r| NeighborResult::new(r.node, r.edge_id, r.direction))
                    .collect()
            })
            .map_err(|e| GraphAccessError::Internal(e.to_string()))
    }

    fn neighbors_by_types(
        &self,
        node: EntityId,
        direction: Direction,
        edge_types: &[EdgeType],
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        // For multiple edge types, we collect results from each type
        let mut results = Vec::new();
        for edge_type in edge_types {
            let neighbors = manifoldb_graph::traversal::Expand::neighbors_by_type(
                &self.tx, node, direction, edge_type,
            )
            .map_err(|e| GraphAccessError::Internal(e.to_string()))?;

            results.extend(
                neighbors.into_iter().map(|r| NeighborResult::new(r.node, r.edge_id, r.direction)),
            );
        }
        Ok(results)
    }

    fn expand_all(
        &self,
        node: EntityId,
        direction: Direction,
        min_depth: usize,
        max_depth: Option<usize>,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphAccessResult<Vec<TraversalResult>> {
        let mut expander =
            manifoldb_graph::traversal::ExpandAll::new(node, direction).with_min_depth(min_depth);

        if let Some(max) = max_depth {
            expander = expander.with_max_depth(max);
        }

        // Add edge type filters
        if let Some(types) = edge_types {
            for edge_type in types {
                expander = expander.with_edge_type(edge_type.clone());
            }
        }

        expander
            .execute(&self.tx)
            .map(|results| {
                results.into_iter().map(|r| TraversalResult::new(r.id, None, r.depth)).collect()
            })
            .map_err(|e| GraphAccessError::Internal(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_accessor_returns_no_storage() {
        let accessor = NullGraphAccessor;

        let result = accessor.neighbors(EntityId::new(1), Direction::Outgoing);
        assert!(matches!(result, Err(GraphAccessError::NoStorage)));
    }

    #[test]
    fn neighbor_result_creation() {
        let result = NeighborResult::new(EntityId::new(1), EdgeId::new(10), Direction::Outgoing);
        assert_eq!(result.node, EntityId::new(1));
        assert_eq!(result.edge_id, EdgeId::new(10));
        assert_eq!(result.direction, Direction::Outgoing);
    }

    #[test]
    fn traversal_result_creation() {
        let result = TraversalResult::new(EntityId::new(2), Some(EdgeId::new(20)), 3);
        assert_eq!(result.node, EntityId::new(2));
        assert_eq!(result.edge_id, Some(EdgeId::new(20)));
        assert_eq!(result.depth, 3);
    }
}
