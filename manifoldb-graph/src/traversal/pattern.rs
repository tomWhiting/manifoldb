//! Path pattern matching.
//!
//! This module provides pattern-based path matching for complex graph queries.
//! Patterns define sequences of edge types and optional node/edge filters
//! that paths must match.

// Allow expect - the invariant is guaranteed by the data structure
#![allow(clippy::expect_used)]

use std::collections::{HashSet, VecDeque};

use manifoldb_core::{Edge, EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::Direction;
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult};

/// A filter for edges in a path step.
#[derive(Debug, Clone)]
pub enum StepFilter {
    /// Match any edge.
    Any,
    /// Match edges of a specific type.
    EdgeType(EdgeType),
    /// Match edges of any of these types.
    EdgeTypes(Vec<EdgeType>),
    /// Custom filter with a predicate function.
    /// Note: This variant stores a descriptive string for debugging.
    Custom(String),
}

impl Default for StepFilter {
    fn default() -> Self {
        Self::Any
    }
}

impl StepFilter {
    /// Create a filter for a specific edge type.
    pub fn edge_type(edge_type: impl Into<EdgeType>) -> Self {
        Self::EdgeType(edge_type.into())
    }

    /// Create a filter for multiple edge types.
    pub fn edge_types(types: impl IntoIterator<Item = EdgeType>) -> Self {
        Self::EdgeTypes(types.into_iter().collect())
    }

    /// Check if an edge matches this filter.
    pub fn matches(&self, edge: &Edge) -> bool {
        match self {
            Self::Any => true,
            Self::EdgeType(et) => &edge.edge_type == et,
            Self::EdgeTypes(types) => types.contains(&edge.edge_type),
            Self::Custom(_) => true, // Custom filters need external predicate
        }
    }
}

impl From<&str> for StepFilter {
    fn from(s: &str) -> Self {
        Self::EdgeType(EdgeType::new(s))
    }
}

impl From<EdgeType> for StepFilter {
    fn from(et: EdgeType) -> Self {
        Self::EdgeType(et)
    }
}

/// A single step in a path pattern.
///
/// Each step specifies a direction, edge filter, and optional hop range.
#[derive(Debug, Clone)]
pub struct PathStep {
    /// Direction to traverse for this step.
    pub direction: Direction,
    /// Filter for edges in this step.
    pub filter: StepFilter,
    /// Minimum number of hops (default: 1).
    pub min_hops: usize,
    /// Maximum number of hops (default: 1).
    /// Use `None` for unlimited (variable length patterns).
    pub max_hops: Option<usize>,
}

impl PathStep {
    /// Create a new single-hop step.
    pub fn new(direction: Direction, filter: impl Into<StepFilter>) -> Self {
        Self { direction, filter: filter.into(), min_hops: 1, max_hops: Some(1) }
    }

    /// Create a step that matches any edge in the given direction.
    pub fn any(direction: Direction) -> Self {
        Self::new(direction, StepFilter::Any)
    }

    /// Create an outgoing step with the given edge type.
    pub fn outgoing(edge_type: impl Into<EdgeType>) -> Self {
        Self::new(Direction::Outgoing, StepFilter::edge_type(edge_type))
    }

    /// Create an incoming step with the given edge type.
    pub fn incoming(edge_type: impl Into<EdgeType>) -> Self {
        Self::new(Direction::Incoming, StepFilter::edge_type(edge_type))
    }

    /// Create a bidirectional step with the given edge type.
    pub fn both(edge_type: impl Into<EdgeType>) -> Self {
        Self::new(Direction::Both, StepFilter::edge_type(edge_type))
    }

    /// Set the hop range for this step.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum number of hops
    /// * `max` - Maximum number of hops (None for unlimited)
    pub const fn with_hops(mut self, min: usize, max: Option<usize>) -> Self {
        self.min_hops = min;
        self.max_hops = max;
        self
    }

    /// Make this a variable-length step with the given range.
    ///
    /// Equivalent to Cypher's `*min..max` syntax.
    pub const fn variable_length(mut self, min: usize, max: usize) -> Self {
        self.min_hops = min;
        self.max_hops = Some(max);
        self
    }

    /// Make this step optional (0 or 1 hops).
    pub const fn optional(mut self) -> Self {
        self.min_hops = 0;
        self.max_hops = Some(1);
        self
    }

    /// Make this step repeat any number of times (0 or more).
    pub const fn zero_or_more(mut self) -> Self {
        self.min_hops = 0;
        self.max_hops = None;
        self
    }

    /// Make this step repeat one or more times.
    pub const fn one_or_more(mut self) -> Self {
        self.min_hops = 1;
        self.max_hops = None;
        self
    }

    /// Check if this step is variable length.
    pub fn is_variable_length(&self) -> bool {
        self.min_hops != 1 || self.max_hops != Some(1)
    }
}

/// A match result from pattern matching.
#[derive(Debug, Clone)]
pub struct PatternMatch {
    /// The nodes traversed, in order.
    pub nodes: Vec<EntityId>,
    /// The edges used for each step.
    /// Each inner vector contains edges for one step (may have multiple for variable-length steps).
    pub step_edges: Vec<Vec<EdgeId>>,
}

impl PatternMatch {
    /// Get the starting node.
    pub fn source(&self) -> EntityId {
        self.nodes[0]
    }

    /// Get the ending node.
    pub fn target(&self) -> EntityId {
        *self.nodes.last().expect("pattern match has at least one node")
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

/// Context for pattern matching operations.
///
/// Bundles related parameters to reduce function argument counts
/// and improve code readability.
struct MatchContext<'a, T: Transaction> {
    /// Transaction for storage operations.
    tx: &'a T,
    /// Current node being processed.
    current: EntityId,
    /// Current step index in the pattern.
    step_idx: usize,
    /// Accumulated path nodes.
    path_nodes: Vec<EntityId>,
    /// Accumulated path edges for each step.
    path_edges: Vec<Vec<EdgeId>>,
    /// Set of visited nodes (for cycle detection).
    visited: HashSet<EntityId>,
    /// Accumulated match results.
    results: &'a mut Vec<PatternMatch>,
}

impl<'a, T: Transaction> MatchContext<'a, T> {
    /// Create a new match context.
    fn new(
        tx: &'a T,
        start: EntityId,
        allow_cycles: bool,
        results: &'a mut Vec<PatternMatch>,
    ) -> Self {
        let visited = if allow_cycles {
            HashSet::new()
        } else {
            let mut set = HashSet::new();
            set.insert(start);
            set
        };

        Self {
            tx,
            current: start,
            step_idx: 0,
            path_nodes: vec![start],
            path_edges: Vec::new(),
            visited,
            results,
        }
    }
}

/// Path pattern matcher.
///
/// Matches paths against a sequence of steps, supporting variable-length
/// patterns and filters.
///
/// # Example
///
/// ```ignore
/// // Pattern: (a)-[:KNOWS]->(b)-[:WORKS_AT]->(c)
/// let pattern = PathPattern::new()
///     .add_step(PathStep::outgoing("KNOWS"))
///     .add_step(PathStep::outgoing("WORKS_AT"));
///
/// let matches = pattern.find_from(&tx, start_node)?;
///
/// // Variable-length: (a)-[:FRIEND*1..3]->(b)
/// let pattern = PathPattern::new()
///     .add_step(PathStep::outgoing("FRIEND").variable_length(1, 3));
/// ```
#[derive(Debug, Clone, Default)]
pub struct PathPattern {
    /// The steps in this pattern.
    steps: Vec<PathStep>,
    /// Maximum number of matches to return.
    limit: Option<usize>,
    /// Whether to allow cycles in matches.
    allow_cycles: bool,
}

impl PathPattern {
    /// Create a new empty pattern.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a step to the pattern.
    pub fn add_step(mut self, step: PathStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add an outgoing step with the given edge type.
    pub fn outgoing(self, edge_type: impl Into<EdgeType>) -> Self {
        self.add_step(PathStep::outgoing(edge_type))
    }

    /// Add an incoming step with the given edge type.
    pub fn incoming(self, edge_type: impl Into<EdgeType>) -> Self {
        self.add_step(PathStep::incoming(edge_type))
    }

    /// Add a bidirectional step with the given edge type.
    pub fn both(self, edge_type: impl Into<EdgeType>) -> Self {
        self.add_step(PathStep::both(edge_type))
    }

    /// Set the maximum number of matches to return.
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Allow cycles in matched paths.
    ///
    /// By default, nodes cannot be visited more than once in a path.
    pub const fn allow_cycles(mut self) -> Self {
        self.allow_cycles = true;
        self
    }

    /// Get the steps in this pattern.
    pub fn steps(&self) -> &[PathStep] {
        &self.steps
    }

    /// Check if this pattern is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Find all paths matching this pattern from a starting node.
    pub fn find_from<T: Transaction>(
        &self,
        tx: &T,
        start: EntityId,
    ) -> GraphResult<Vec<PatternMatch>> {
        if self.steps.is_empty() {
            // Empty pattern matches only the start node
            return Ok(vec![PatternMatch { nodes: vec![start], step_edges: Vec::new() }]);
        }

        let mut results = Vec::new();
        let mut ctx = MatchContext::new(tx, start, self.allow_cycles, &mut results);

        self.match_from_step(&mut ctx)?;

        Ok(results)
    }

    /// Find paths matching this pattern that end at a specific node.
    pub fn find_between<T: Transaction>(
        &self,
        tx: &T,
        start: EntityId,
        end: EntityId,
    ) -> GraphResult<Vec<PatternMatch>> {
        let all_matches = self.find_from(tx, start)?;
        Ok(all_matches.into_iter().filter(|m| m.target() == end).collect())
    }

    /// Check if any path matches this pattern from the given start.
    pub fn matches<T: Transaction>(&self, tx: &T, start: EntityId) -> GraphResult<bool> {
        let pattern_with_limit =
            Self { steps: self.steps.clone(), limit: Some(1), allow_cycles: self.allow_cycles };
        let matches = pattern_with_limit.find_from(tx, start)?;
        Ok(!matches.is_empty())
    }

    /// Recursive helper to match pattern steps.
    fn match_from_step<T: Transaction>(&self, ctx: &mut MatchContext<'_, T>) -> GraphResult<()> {
        // Check limit
        if let Some(limit) = self.limit {
            if ctx.results.len() >= limit {
                return Ok(());
            }
        }

        // If we've processed all steps, we have a match
        if ctx.step_idx >= self.steps.len() {
            ctx.results.push(PatternMatch {
                nodes: ctx.path_nodes.clone(),
                step_edges: ctx.path_edges.clone(),
            });
            return Ok(());
        }

        let step = &self.steps[ctx.step_idx];

        // Handle variable-length steps
        if step.is_variable_length() {
            self.match_variable_step(ctx, step)?;
        } else {
            // Single-hop step
            self.match_single_step(ctx, step)?;
        }

        Ok(())
    }

    fn match_single_step<T: Transaction>(
        &self,
        ctx: &mut MatchContext<'_, T>,
        step: &PathStep,
    ) -> GraphResult<()> {
        let neighbors = self.get_filtered_neighbors(ctx.tx, ctx.current, step)?;

        for (neighbor, edge_id) in neighbors {
            if !self.allow_cycles && ctx.visited.contains(&neighbor) {
                continue;
            }

            // Save current state for backtracking
            let prev_current = ctx.current;
            let prev_step_idx = ctx.step_idx;
            let prev_nodes_len = ctx.path_nodes.len();
            let prev_edges_len = ctx.path_edges.len();

            // Build new path
            ctx.path_nodes.push(neighbor);
            ctx.path_edges.push(vec![edge_id]);
            ctx.current = neighbor;
            ctx.step_idx += 1;

            // Track visited
            let was_new = if self.allow_cycles { false } else { ctx.visited.insert(neighbor) };

            // Continue to next step
            self.match_from_step(ctx)?;

            // Restore state for backtracking
            ctx.path_nodes.truncate(prev_nodes_len);
            ctx.path_edges.truncate(prev_edges_len);
            ctx.current = prev_current;
            ctx.step_idx = prev_step_idx;

            if was_new {
                ctx.visited.remove(&neighbor);
            }
        }

        Ok(())
    }

    fn match_variable_step<T: Transaction>(
        &self,
        ctx: &mut MatchContext<'_, T>,
        step: &PathStep,
    ) -> GraphResult<()> {
        // Use BFS to explore variable-length paths
        let mut queue: VecDeque<(EntityId, Vec<EntityId>, Vec<EdgeId>, HashSet<EntityId>)> =
            VecDeque::new();

        // Initialize with current state
        queue.push_back((ctx.current, vec![ctx.current], Vec::new(), ctx.visited.clone()));

        while let Some((node, step_nodes, step_edges, step_visited)) = queue.pop_front() {
            let hop_count = step_edges.len();

            // If within valid hop range, try continuing with next step
            if hop_count >= step.min_hops {
                // Save current state
                let prev_current = ctx.current;
                let prev_step_idx = ctx.step_idx;
                let prev_nodes_len = ctx.path_nodes.len();
                let prev_edges_len = ctx.path_edges.len();
                let prev_visited = ctx.visited.clone();

                // Add intermediate nodes (skip first which is already in path)
                ctx.path_nodes.extend(step_nodes.iter().skip(1));
                ctx.path_edges.push(step_edges.clone());
                ctx.current = node;
                ctx.step_idx += 1;
                ctx.visited.clone_from(&step_visited);

                self.match_from_step(ctx)?;

                // Restore state
                ctx.path_nodes.truncate(prev_nodes_len);
                ctx.path_edges.truncate(prev_edges_len);
                ctx.current = prev_current;
                ctx.step_idx = prev_step_idx;
                ctx.visited = prev_visited;
            }

            // Check if we can expand further
            let can_expand = step.max_hops.map_or(true, |max| hop_count < max);
            if !can_expand {
                continue;
            }

            // Expand to neighbors
            let neighbors = self.get_filtered_neighbors(ctx.tx, node, step)?;

            for (neighbor, edge_id) in neighbors {
                if !self.allow_cycles && step_visited.contains(&neighbor) {
                    continue;
                }

                let mut new_step_nodes = step_nodes.clone();
                let mut new_step_edges = step_edges.clone();
                let mut new_step_visited = step_visited.clone();

                new_step_nodes.push(neighbor);
                new_step_edges.push(edge_id);

                if !self.allow_cycles {
                    new_step_visited.insert(neighbor);
                }

                queue.push_back((neighbor, new_step_nodes, new_step_edges, new_step_visited));
            }
        }

        Ok(())
    }

    fn get_filtered_neighbors<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
        step: &PathStep,
    ) -> GraphResult<Vec<(EntityId, EdgeId)>> {
        let mut neighbors = Vec::new();

        if step.direction.includes_outgoing() {
            self.add_filtered_outgoing(tx, node, step, &mut neighbors)?;
        }

        if step.direction.includes_incoming() {
            self.add_filtered_incoming(tx, node, step, &mut neighbors)?;
        }

        Ok(neighbors)
    }

    fn add_filtered_outgoing<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
        step: &PathStep,
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &step.filter {
            StepFilter::EdgeType(et) => {
                AdjacencyIndex::for_each_outgoing_by_type(tx, node, et, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        neighbors.push((edge.target, edge_id));
                    }
                    Ok(true)
                })?;
            }
            StepFilter::EdgeTypes(types) => {
                for et in types {
                    AdjacencyIndex::for_each_outgoing_by_type(tx, node, et, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.target, edge_id));
                        }
                        Ok(true)
                    })?;
                }
            }
            StepFilter::Any | StepFilter::Custom(_) => {
                AdjacencyIndex::for_each_outgoing(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        if step.filter.matches(&edge) {
                            neighbors.push((edge.target, edge_id));
                        }
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    fn add_filtered_incoming<T: Transaction>(
        &self,
        tx: &T,
        node: EntityId,
        step: &PathStep,
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &step.filter {
            StepFilter::EdgeType(et) => {
                AdjacencyIndex::for_each_incoming_by_type(tx, node, et, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        neighbors.push((edge.source, edge_id));
                    }
                    Ok(true)
                })?;
            }
            StepFilter::EdgeTypes(types) => {
                for et in types {
                    AdjacencyIndex::for_each_incoming_by_type(tx, node, et, |edge_id| {
                        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                            neighbors.push((edge.source, edge_id));
                        }
                        Ok(true)
                    })?;
                }
            }
            StepFilter::Any | StepFilter::Custom(_) => {
                AdjacencyIndex::for_each_incoming(tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                        if step.filter.matches(&edge) {
                            neighbors.push((edge.source, edge_id));
                        }
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }
}

/// Builder for creating path patterns with a fluent API.
pub struct PatternBuilder {
    pattern: PathPattern,
}

impl PatternBuilder {
    /// Start building a new pattern.
    pub fn new() -> Self {
        Self { pattern: PathPattern::new() }
    }

    /// Add an outgoing edge of the specified type.
    pub fn out(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.pattern = self.pattern.outgoing(edge_type);
        self
    }

    /// Add an incoming edge of the specified type.
    pub fn inc(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.pattern = self.pattern.incoming(edge_type);
        self
    }

    /// Add a bidirectional edge of the specified type.
    pub fn rel(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.pattern = self.pattern.both(edge_type);
        self
    }

    /// Add any outgoing edge.
    pub fn out_any(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Outgoing));
        self
    }

    /// Add any incoming edge.
    pub fn in_any(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Incoming));
        self
    }

    /// Add any edge in either direction.
    pub fn any(mut self) -> Self {
        self.pattern = self.pattern.add_step(PathStep::any(Direction::Both));
        self
    }

    /// Add a variable-length outgoing path.
    pub fn out_var(mut self, edge_type: impl Into<EdgeType>, min: usize, max: usize) -> Self {
        self.pattern =
            self.pattern.add_step(PathStep::outgoing(edge_type).variable_length(min, max));
        self
    }

    /// Add a variable-length incoming path.
    pub fn in_var(mut self, edge_type: impl Into<EdgeType>, min: usize, max: usize) -> Self {
        self.pattern =
            self.pattern.add_step(PathStep::incoming(edge_type).variable_length(min, max));
        self
    }

    /// Set result limit.
    pub fn limit(mut self, limit: usize) -> Self {
        self.pattern = self.pattern.with_limit(limit);
        self
    }

    /// Allow cycles in matches.
    pub fn with_cycles(mut self) -> Self {
        self.pattern = self.pattern.allow_cycles();
        self
    }

    /// Build the pattern.
    pub fn build(self) -> PathPattern {
        self.pattern
    }
}

impl Default for PatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_filter_matches_any() {
        let filter = StepFilter::Any;
        let edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "TEST");
        assert!(filter.matches(&edge));
    }

    #[test]
    fn step_filter_matches_edge_type() {
        let filter = StepFilter::edge_type("FRIEND");
        let friend_edge = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "FRIEND");
        let work_edge = Edge::new(EdgeId::new(2), EntityId::new(1), EntityId::new(2), "WORKS_AT");

        assert!(filter.matches(&friend_edge));
        assert!(!filter.matches(&work_edge));
    }

    #[test]
    fn step_filter_matches_multiple_types() {
        let filter = StepFilter::edge_types([EdgeType::new("FRIEND"), EdgeType::new("FOLLOWS")]);
        let friend = Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "FRIEND");
        let follows = Edge::new(EdgeId::new(2), EntityId::new(1), EntityId::new(2), "FOLLOWS");
        let blocks = Edge::new(EdgeId::new(3), EntityId::new(1), EntityId::new(2), "BLOCKS");

        assert!(filter.matches(&friend));
        assert!(filter.matches(&follows));
        assert!(!filter.matches(&blocks));
    }

    #[test]
    fn path_step_variable_length() {
        let step = PathStep::outgoing("FRIEND").variable_length(1, 3);
        assert!(step.is_variable_length());
        assert_eq!(step.min_hops, 1);
        assert_eq!(step.max_hops, Some(3));
    }

    #[test]
    fn path_step_optional() {
        let step = PathStep::outgoing("FRIEND").optional();
        assert!(step.is_variable_length());
        assert_eq!(step.min_hops, 0);
        assert_eq!(step.max_hops, Some(1));
    }

    #[test]
    fn path_step_zero_or_more() {
        let step = PathStep::outgoing("FRIEND").zero_or_more();
        assert!(step.is_variable_length());
        assert_eq!(step.min_hops, 0);
        assert_eq!(step.max_hops, None);
    }

    #[test]
    fn path_pattern_builder() {
        let pattern = PatternBuilder::new().out("KNOWS").out("WORKS_AT").limit(10).build();

        assert_eq!(pattern.steps().len(), 2);
        assert_eq!(pattern.limit, Some(10));
    }

    #[test]
    fn pattern_match_helpers() {
        let pm = PatternMatch {
            nodes: vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)],
            step_edges: vec![vec![EdgeId::new(10)], vec![EdgeId::new(20)]],
        };

        assert_eq!(pm.source(), EntityId::new(1));
        assert_eq!(pm.target(), EntityId::new(3));
        assert_eq!(pm.length(), 2);
        assert_eq!(pm.all_edges(), vec![EdgeId::new(10), EdgeId::new(20)]);
    }

    #[test]
    fn empty_pattern() {
        let pattern = PathPattern::new();
        assert!(pattern.is_empty());
    }
}
