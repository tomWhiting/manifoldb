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

/// Configuration for shortest path finding.
#[derive(Debug, Clone)]
pub struct ShortestPathConfig {
    /// Direction to traverse edges.
    pub direction: Direction,
    /// Edge type filters (empty means any).
    pub edge_types: Vec<EdgeType>,
    /// Maximum path length (None for unlimited).
    pub max_depth: Option<usize>,
    /// Whether to find all shortest paths.
    pub find_all: bool,
}

impl Default for ShortestPathConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Both,
            edge_types: Vec::new(),
            max_depth: None,
            find_all: false,
        }
    }
}

impl ShortestPathConfig {
    /// Create a new shortest path configuration.
    #[must_use]
    pub fn new(direction: Direction) -> Self {
        Self { direction, ..Default::default() }
    }

    /// Set edge type filters.
    #[must_use]
    pub fn with_edge_types(mut self, types: Vec<EdgeType>) -> Self {
        self.edge_types = types;
        self
    }

    /// Set maximum path depth.
    #[must_use]
    pub fn with_max_depth(mut self, max: usize) -> Self {
        self.max_depth = Some(max);
        self
    }

    /// Set whether to find all shortest paths.
    #[must_use]
    pub fn with_find_all(mut self, find_all: bool) -> Self {
        self.find_all = find_all;
        self
    }
}

/// Result of a shortest path search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortestPathResult {
    /// The nodes in the path, from source to target.
    pub nodes: Vec<EntityId>,
    /// The edges connecting the nodes.
    pub edges: Vec<EdgeId>,
    /// The total length of the path (number of edges).
    pub length: usize,
}

impl ShortestPathResult {
    /// Create a new shortest path result.
    pub fn new(nodes: Vec<EntityId>, edges: Vec<EdgeId>) -> Self {
        let length = edges.len();
        Self { nodes, edges, length }
    }

    /// Get the source node.
    pub fn source(&self) -> Option<EntityId> {
        self.nodes.first().copied()
    }

    /// Get the target node.
    pub fn target(&self) -> Option<EntityId> {
        self.nodes.last().copied()
    }
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

    /// Find the shortest path(s) between two nodes.
    ///
    /// Uses BFS for unweighted shortest path.
    /// Returns `None` if no path exists.
    fn shortest_path(
        &self,
        source: EntityId,
        target: EntityId,
        config: &ShortestPathConfig,
    ) -> GraphAccessResult<Vec<ShortestPathResult>>;
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

    fn shortest_path(
        &self,
        _source: EntityId,
        _target: EntityId,
        _config: &ShortestPathConfig,
    ) -> GraphAccessResult<Vec<ShortestPathResult>> {
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

    fn shortest_path(
        &self,
        source: EntityId,
        target: EntityId,
        config: &ShortestPathConfig,
    ) -> GraphAccessResult<Vec<ShortestPathResult>> {
        use manifoldb_graph::traversal::{AllShortestPaths, ShortestPath};

        if config.find_all {
            // Find all shortest paths
            let mut finder = AllShortestPaths::new(source, target, config.direction);

            if let Some(max) = config.max_depth {
                finder = finder.with_max_depth(max);
            }

            for edge_type in &config.edge_types {
                finder = finder.with_edge_type(edge_type.clone());
            }

            finder
                .find(&self.tx)
                .map(|paths| {
                    paths.into_iter().map(|p| ShortestPathResult::new(p.nodes, p.edges)).collect()
                })
                .map_err(|e| GraphAccessError::Internal(e.to_string()))
        } else {
            // Find single shortest path
            let mut finder = ShortestPath::new(source, target, config.direction);

            if let Some(max) = config.max_depth {
                finder = finder.with_max_depth(max);
            }

            if !config.edge_types.is_empty() {
                finder = finder.with_edge_types(config.edge_types.iter().cloned());
            }

            finder
                .find(&self.tx)
                .map(|opt| {
                    opt.map(|p| vec![ShortestPathResult::new(p.nodes, p.edges)]).unwrap_or_default()
                })
                .map_err(|e| GraphAccessError::Internal(e.to_string()))
        }
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

    #[test]
    fn null_accessor_shortest_path_returns_no_storage() {
        let accessor = NullGraphAccessor;
        let config = ShortestPathConfig::default();
        let result = accessor.shortest_path(EntityId::new(1), EntityId::new(2), &config);
        assert!(matches!(result, Err(GraphAccessError::NoStorage)));
    }

    #[test]
    fn shortest_path_config_default() {
        let config = ShortestPathConfig::default();
        assert_eq!(config.direction, Direction::Both);
        assert!(config.edge_types.is_empty());
        assert!(config.max_depth.is_none());
        assert!(!config.find_all);
    }

    #[test]
    fn shortest_path_config_builder() {
        let config = ShortestPathConfig::new(Direction::Outgoing)
            .with_edge_types(vec![EdgeType::new("KNOWS")])
            .with_max_depth(5)
            .with_find_all(true);

        assert_eq!(config.direction, Direction::Outgoing);
        assert_eq!(config.edge_types.len(), 1);
        assert_eq!(config.max_depth, Some(5));
        assert!(config.find_all);
    }

    #[test]
    fn shortest_path_result_creation() {
        let result = ShortestPathResult::new(
            vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)],
            vec![EdgeId::new(10), EdgeId::new(20)],
        );
        assert_eq!(result.source(), Some(EntityId::new(1)));
        assert_eq!(result.target(), Some(EntityId::new(3)));
        assert_eq!(result.length, 2);
        assert_eq!(result.nodes.len(), 3);
        assert_eq!(result.edges.len(), 2);
    }

    #[test]
    fn shortest_path_result_empty_path() {
        let result = ShortestPathResult::new(vec![], vec![]);
        assert_eq!(result.source(), None);
        assert_eq!(result.target(), None);
        assert_eq!(result.length, 0);
    }

    #[test]
    fn shortest_path_result_single_node() {
        let result = ShortestPathResult::new(vec![EntityId::new(42)], vec![]);
        assert_eq!(result.source(), Some(EntityId::new(42)));
        assert_eq!(result.target(), Some(EntityId::new(42)));
        assert_eq!(result.length, 0);
    }
}
