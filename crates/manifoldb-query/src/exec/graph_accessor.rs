//! Graph storage accessor trait for query execution.
//!
//! This module provides the [`GraphAccessor`] trait that abstracts
//! graph traversal operations for use in query execution. The trait
//! is object-safe and can be stored in the execution context.

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_graph::traversal::{Direction, PathPattern, PathStep, PatternMatch, StepFilter};

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

/// Result of a path pattern match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMatchResult {
    /// The nodes traversed, in order.
    pub nodes: Vec<EntityId>,
    /// The edges used for each step.
    /// Each inner vector contains edges for one step (may have multiple for variable-length steps).
    pub step_edges: Vec<Vec<EdgeId>>,
}

impl PathMatchResult {
    /// Create a new path match result.
    pub fn new(nodes: Vec<EntityId>, step_edges: Vec<Vec<EdgeId>>) -> Self {
        Self { nodes, step_edges }
    }

    /// Get the starting node.
    pub fn source(&self) -> Option<EntityId> {
        self.nodes.first().copied()
    }

    /// Get the ending node.
    pub fn target(&self) -> Option<EntityId> {
        self.nodes.last().copied()
    }

    /// Get all edges as a flat list.
    pub fn all_edges(&self) -> Vec<EdgeId> {
        self.step_edges.iter().flatten().copied().collect()
    }

    /// Get the total path length (number of edges).
    pub fn length(&self) -> usize {
        self.step_edges.iter().map(std::vec::Vec::len).sum()
    }
}

impl From<PatternMatch> for PathMatchResult {
    fn from(pm: PatternMatch) -> Self {
        Self { nodes: pm.nodes, step_edges: pm.step_edges }
    }
}

/// Configuration for path finding.
#[derive(Debug, Clone)]
pub struct PathFindConfig {
    /// Path steps to match.
    pub steps: Vec<PathStepConfig>,
    /// Maximum number of results to return (None for unlimited).
    pub limit: Option<usize>,
    /// Whether to allow cycles in the path.
    pub allow_cycles: bool,
}

/// Configuration for a single step in a path pattern.
#[derive(Debug, Clone)]
pub struct PathStepConfig {
    /// Direction to traverse.
    pub direction: Direction,
    /// Edge type filters (empty means any).
    pub edge_types: Vec<EdgeType>,
    /// Minimum number of hops.
    pub min_hops: usize,
    /// Maximum number of hops (None for unlimited).
    pub max_hops: Option<usize>,
}

impl PathStepConfig {
    /// Convert to a `PathStep` for the traversal module.
    pub fn to_path_step(&self) -> PathStep {
        let filter = if self.edge_types.is_empty() {
            StepFilter::Any
        } else if self.edge_types.len() == 1 {
            StepFilter::EdgeType(self.edge_types[0].clone())
        } else {
            StepFilter::EdgeTypes(self.edge_types.clone())
        };

        PathStep::new(self.direction, filter).with_hops(self.min_hops, self.max_hops)
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

    /// Find paths matching a pattern from a starting node.
    ///
    /// Executes multi-hop path patterns with support for variable-length steps.
    fn find_paths(
        &self,
        start: EntityId,
        config: &PathFindConfig,
    ) -> GraphAccessResult<Vec<PathMatchResult>>;
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

    fn find_paths(
        &self,
        _start: EntityId,
        _config: &PathFindConfig,
    ) -> GraphAccessResult<Vec<PathMatchResult>> {
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

    fn find_paths(
        &self,
        start: EntityId,
        config: &PathFindConfig,
    ) -> GraphAccessResult<Vec<PathMatchResult>> {
        // Build the path pattern from the configuration
        let mut pattern = PathPattern::new();

        for step in &config.steps {
            pattern = pattern.add_step(step.to_path_step());
        }

        // Apply limit if configured
        if let Some(limit) = config.limit {
            pattern = pattern.with_limit(limit);
        }

        // Apply cycle policy
        if config.allow_cycles {
            pattern = pattern.allow_cycles();
        }

        // Execute the pattern match
        pattern
            .find_from(&self.tx, start)
            .map(|matches| matches.into_iter().map(PathMatchResult::from).collect())
            .map_err(|e| GraphAccessError::Internal(e.to_string()))
    }
}

// ============================================================================
// Graph Mutation Support
// ============================================================================

use manifoldb_core::{Edge, Entity, Label, Value};
use std::collections::HashMap;

/// Specification for creating a node.
#[derive(Debug, Clone)]
pub struct CreateNodeRequest {
    /// Labels to assign to the node.
    pub labels: Vec<Label>,
    /// Properties to set on the node.
    pub properties: HashMap<String, Value>,
}

impl CreateNodeRequest {
    /// Create a new empty node request.
    pub fn new() -> Self {
        Self { labels: Vec::new(), properties: HashMap::new() }
    }

    /// Add a label.
    pub fn with_label(mut self, label: impl Into<Label>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

impl Default for CreateNodeRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Specification for creating an edge.
#[derive(Debug, Clone)]
pub struct CreateEdgeRequest {
    /// The source entity ID.
    pub source: EntityId,
    /// The target entity ID.
    pub target: EntityId,
    /// The edge type.
    pub edge_type: EdgeType,
    /// Properties to set on the edge.
    pub properties: HashMap<String, Value>,
}

impl CreateEdgeRequest {
    /// Create a new edge request.
    pub fn new(source: EntityId, target: EntityId, edge_type: impl Into<EdgeType>) -> Self {
        Self { source, target, edge_type: edge_type.into(), properties: HashMap::new() }
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
}

/// A trait for mutating the graph during query execution.
///
/// This trait provides write operations for Cypher CREATE, MERGE, SET, DELETE, etc.
/// It is separate from `GraphAccessor` because write operations require mutable access.
pub trait GraphMutator: Send + Sync {
    /// Create a new node in the graph.
    ///
    /// Returns the created entity with its generated ID.
    fn create_node(&self, request: &CreateNodeRequest) -> GraphAccessResult<Entity>;

    /// Create a new edge in the graph.
    ///
    /// Returns the created edge with its generated ID.
    fn create_edge(&self, request: &CreateEdgeRequest) -> GraphAccessResult<Edge>;
}

/// A null implementation of `GraphMutator` that always returns an error.
///
/// Used when no graph storage is configured for writes.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullGraphMutator;

impl GraphMutator for NullGraphMutator {
    fn create_node(&self, _request: &CreateNodeRequest) -> GraphAccessResult<Entity> {
        Err(GraphAccessError::NoStorage)
    }

    fn create_edge(&self, _request: &CreateEdgeRequest) -> GraphAccessResult<Edge> {
        Err(GraphAccessError::NoStorage)
    }
}

/// A concrete implementation of `GraphMutator` backed by a mutable storage transaction.
///
/// This wraps a transaction reference and delegates to the actual graph storage code.
/// Uses interior mutability via `RwLock` for thread-safe mutation.
pub struct TransactionGraphMutator<T> {
    tx: std::sync::RwLock<T>,
    id_gen: std::sync::Arc<manifoldb_graph::store::IdGenerator>,
}

impl<T> TransactionGraphMutator<T> {
    /// Create a new mutator wrapping a transaction.
    pub fn new(tx: T, id_gen: std::sync::Arc<manifoldb_graph::store::IdGenerator>) -> Self {
        Self { tx: std::sync::RwLock::new(tx), id_gen }
    }
}

impl<T> GraphMutator for TransactionGraphMutator<T>
where
    T: manifoldb_storage::Transaction + Send + Sync,
{
    fn create_node(&self, request: &CreateNodeRequest) -> GraphAccessResult<Entity> {
        let mut tx = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        manifoldb_graph::store::NodeStore::create(&mut *tx, &self.id_gen, |id| {
            let mut entity = Entity::new(id);
            for label in &request.labels {
                entity = entity.with_label(label.clone());
            }
            for (key, value) in &request.properties {
                entity = entity.with_property(key.clone(), value.clone());
            }
            entity
        })
        .map_err(|e| GraphAccessError::Internal(e.to_string()))
    }

    fn create_edge(&self, request: &CreateEdgeRequest) -> GraphAccessResult<Edge> {
        let mut tx = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        manifoldb_graph::store::EdgeStore::create(
            &mut *tx,
            &self.id_gen,
            request.source,
            request.target,
            request.edge_type.clone(),
            |id| {
                let mut edge =
                    Edge::new(id, request.source, request.target, request.edge_type.clone());
                for (key, value) in &request.properties {
                    edge = edge.with_property(key.clone(), value.clone());
                }
                edge
            },
        )
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

    #[test]
    fn path_match_result_creation() {
        let result = PathMatchResult::new(
            vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)],
            vec![vec![EdgeId::new(10)], vec![EdgeId::new(20)]],
        );
        assert_eq!(result.source(), Some(EntityId::new(1)));
        assert_eq!(result.target(), Some(EntityId::new(3)));
        assert_eq!(result.length(), 2);
        assert_eq!(result.all_edges(), vec![EdgeId::new(10), EdgeId::new(20)]);
    }

    #[test]
    fn path_step_config_to_path_step() {
        let config = PathStepConfig {
            direction: Direction::Outgoing,
            edge_types: vec![EdgeType::new("FRIEND")],
            min_hops: 1,
            max_hops: Some(3),
        };
        let step = config.to_path_step();
        assert_eq!(step.direction, Direction::Outgoing);
        assert_eq!(step.min_hops, 1);
        assert_eq!(step.max_hops, Some(3));
    }

    #[test]
    fn null_accessor_find_paths_returns_no_storage() {
        let accessor = NullGraphAccessor;
        let config = PathFindConfig { steps: vec![], limit: None, allow_cycles: false };
        let result = accessor.find_paths(EntityId::new(1), &config);
        assert!(matches!(result, Err(GraphAccessError::NoStorage)));
    }
}
