//! Lazy traversal iterators.
//!
//! This module provides lazy iterators for graph traversal that enable
//! memory-efficient processing of large result sets.

use std::collections::{HashSet, VecDeque};

use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use super::{Direction, ExpandResult, TraversalFilter, TraversalNode};
use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphResult};

/// Configuration for traversal iterators.
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    /// Direction to traverse.
    pub direction: Direction,
    /// Minimum depth to include (default: 1).
    pub min_depth: usize,
    /// Maximum depth to traverse.
    pub max_depth: Option<usize>,
    /// Filter for traversal.
    pub filter: TraversalFilter,
    /// Whether to include the starting node in results.
    pub include_start: bool,
}

impl Default for TraversalConfig {
    fn default() -> Self {
        Self {
            direction: Direction::Outgoing,
            min_depth: 1,
            max_depth: None,
            filter: TraversalFilter::new(),
            include_start: false,
        }
    }
}

impl TraversalConfig {
    /// Create a new configuration with the given direction.
    pub fn new(direction: Direction) -> Self {
        Self { direction, ..Default::default() }
    }

    /// Set the minimum depth.
    pub const fn with_min_depth(mut self, min_depth: usize) -> Self {
        self.min_depth = min_depth;
        self
    }

    /// Set the maximum depth.
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Filter to specific edge types.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = self.filter.with_edge_type(edge_type);
        self
    }

    /// Set result limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.filter = self.filter.with_limit(limit);
        self
    }

    /// Include the starting node in results.
    pub const fn include_start(mut self) -> Self {
        self.include_start = true;
        self.min_depth = 0;
        self
    }
}

/// State for the traversal iterator.
struct TraversalState {
    /// Queue of (node, depth) pairs for BFS.
    queue: VecDeque<(EntityId, usize)>,
    /// Visited nodes to prevent cycles.
    visited: HashSet<EntityId>,
    /// Number of results returned so far.
    count: usize,
}

impl TraversalState {
    fn new(start: EntityId) -> Self {
        let mut visited = HashSet::new();
        visited.insert(start);

        let mut queue = VecDeque::new();
        queue.push_back((start, 0));

        Self { queue, visited, count: 0 }
    }
}

/// A lazy BFS iterator over graph traversal.
///
/// This iterator performs breadth-first traversal, yielding nodes
/// one at a time without materializing the entire result set.
///
/// # Example
///
/// ```ignore
/// let config = TraversalConfig::new(Direction::Outgoing)
///     .with_max_depth(3)
///     .with_limit(100);
///
/// let iter = TraversalIterator::new(&tx, start_node, config);
///
/// for result in iter {
///     let node = result?;
///     println!("Found node {:?} at depth {}", node.id, node.depth);
/// }
/// ```
pub struct TraversalIterator<'a, T: Transaction> {
    tx: &'a T,
    config: TraversalConfig,
    state: TraversalState,
    /// Pending results that haven't been yielded yet.
    pending: VecDeque<TraversalNode>,
    /// Whether we've yielded the start node.
    yielded_start: bool,
    /// The starting node.
    start: EntityId,
}

impl<'a, T: Transaction> TraversalIterator<'a, T> {
    /// Create a new traversal iterator.
    pub fn new(tx: &'a T, start: EntityId, config: TraversalConfig) -> Self {
        Self {
            tx,
            config,
            state: TraversalState::new(start),
            pending: VecDeque::new(),
            yielded_start: false,
            start,
        }
    }

    /// Expand from the current node and queue neighbors.
    fn expand_current(&mut self) -> GraphResult<()> {
        let Some((current, depth)) = self.state.queue.pop_front() else {
            return Ok(());
        };

        // Check if we should continue expanding
        let should_expand = self.config.max_depth.map_or(true, |max| depth < max);
        if !should_expand {
            return Ok(());
        }

        // Get neighbors
        let neighbors = self.get_neighbors(current)?;

        for (neighbor, _edge_id) in neighbors {
            if self.state.visited.contains(&neighbor) {
                continue;
            }

            self.state.visited.insert(neighbor);
            let next_depth = depth + 1;

            // Queue for further expansion
            self.state.queue.push_back((neighbor, next_depth));

            // Add to pending if within depth range
            if next_depth >= self.config.min_depth
                && self.config.filter.should_include_node(neighbor)
            {
                self.pending.push_back(TraversalNode::new(neighbor, next_depth));
            }
        }

        Ok(())
    }

    fn get_neighbors(&self, node: EntityId) -> GraphResult<Vec<(EntityId, EdgeId)>> {
        let mut neighbors = Vec::new();

        if self.config.direction.includes_outgoing() {
            self.add_outgoing_neighbors(node, &mut neighbors)?;
        }

        if self.config.direction.includes_incoming() {
            self.add_incoming_neighbors(node, &mut neighbors)?;
        }

        Ok(neighbors)
    }

    fn add_outgoing_neighbors(
        &self,
        node: EntityId,
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &self.config.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_outgoing_by_type(
                        self.tx,
                        node,
                        edge_type,
                        |edge_id| {
                            if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                                neighbors.push((edge.target, edge_id));
                            }
                            Ok(true)
                        },
                    )?;
                }
            }
            None => {
                AdjacencyIndex::for_each_outgoing(self.tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        neighbors.push((edge.target, edge_id));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    fn add_incoming_neighbors(
        &self,
        node: EntityId,
        neighbors: &mut Vec<(EntityId, EdgeId)>,
    ) -> GraphResult<()> {
        match &self.config.filter.edge_types {
            Some(types) => {
                for edge_type in types {
                    AdjacencyIndex::for_each_incoming_by_type(
                        self.tx,
                        node,
                        edge_type,
                        |edge_id| {
                            if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                                neighbors.push((edge.source, edge_id));
                            }
                            Ok(true)
                        },
                    )?;
                }
            }
            None => {
                AdjacencyIndex::for_each_incoming(self.tx, node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        neighbors.push((edge.source, edge_id));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    /// Try to get the next node from the iterator.
    fn try_next(&mut self) -> GraphResult<Option<TraversalNode>> {
        // Check limit
        if self.config.filter.is_at_limit(self.state.count) {
            return Ok(None);
        }

        // Handle start node if needed
        if !self.yielded_start && self.config.include_start {
            self.yielded_start = true;
            self.state.count += 1;
            return Ok(Some(TraversalNode::new(self.start, 0)));
        }
        self.yielded_start = true;

        // Check pending queue first
        if let Some(node) = self.pending.pop_front() {
            self.state.count += 1;
            return Ok(Some(node));
        }

        // Expand nodes until we find results or exhaust the queue
        while !self.state.queue.is_empty() {
            self.expand_current()?;

            if let Some(node) = self.pending.pop_front() {
                self.state.count += 1;
                return Ok(Some(node));
            }
        }

        Ok(None)
    }

    /// Collect all remaining results into a vector.
    pub fn collect_all(mut self) -> GraphResult<Vec<TraversalNode>> {
        let mut results = Vec::new();
        while let Some(node) = self.try_next()? {
            results.push(node);
        }
        Ok(results)
    }

    /// Collect only node IDs.
    pub fn collect_ids(mut self) -> GraphResult<Vec<EntityId>> {
        let mut results = Vec::new();
        while let Some(node) = self.try_next()? {
            results.push(node.id);
        }
        Ok(results)
    }

    /// Count results without collecting.
    pub fn count_all(mut self) -> GraphResult<usize> {
        let mut count = 0;
        while self.try_next()?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    /// Take up to n results.
    pub fn take(mut self, n: usize) -> GraphResult<Vec<TraversalNode>> {
        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            match self.try_next()? {
                Some(node) => results.push(node),
                None => break,
            }
        }
        Ok(results)
    }

    /// Check if there are any more results.
    pub fn has_next(&mut self) -> GraphResult<bool> {
        if !self.pending.is_empty() {
            return Ok(true);
        }

        if self.config.filter.is_at_limit(self.state.count) {
            return Ok(false);
        }

        // Try to expand and see if we get any results
        while !self.state.queue.is_empty() && self.pending.is_empty() {
            self.expand_current()?;
        }

        Ok(!self.pending.is_empty())
    }
}

/// A wrapper that converts `GraphResult` into Iterator Item.
pub struct TraversalIteratorAdapter<'a, T: Transaction> {
    inner: TraversalIterator<'a, T>,
    errored: bool,
}

impl<'a, T: Transaction> TraversalIteratorAdapter<'a, T> {
    /// Create a new adapter.
    pub fn new(tx: &'a T, start: EntityId, config: TraversalConfig) -> Self {
        Self { inner: TraversalIterator::new(tx, start, config), errored: false }
    }
}

impl<T: Transaction> Iterator for TraversalIteratorAdapter<'_, T> {
    type Item = GraphResult<TraversalNode>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }

        match self.inner.try_next() {
            Ok(Some(node)) => Some(Ok(node)),
            Ok(None) => None,
            Err(e) => {
                self.errored = true;
                Some(Err(e))
            }
        }
    }
}

/// Iterator over single-hop neighbors.
pub struct NeighborIterator<'a, T: Transaction> {
    tx: &'a T,
    node: EntityId,
    direction: Direction,
    filter: Option<EdgeType>,
    /// Pending neighbors to yield.
    pending: VecDeque<ExpandResult>,
    /// Whether we've loaded neighbors yet.
    loaded: bool,
}

impl<'a, T: Transaction> NeighborIterator<'a, T> {
    /// Create a new neighbor iterator.
    pub const fn new(tx: &'a T, node: EntityId, direction: Direction) -> Self {
        Self { tx, node, direction, filter: None, pending: VecDeque::new(), loaded: false }
    }

    /// Filter to a specific edge type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.filter = Some(edge_type.into());
        self
    }

    fn load_neighbors(&mut self) -> GraphResult<()> {
        if self.loaded {
            return Ok(());
        }
        self.loaded = true;

        if self.direction.includes_outgoing() {
            self.load_outgoing()?;
        }

        if self.direction.includes_incoming() {
            self.load_incoming()?;
        }

        Ok(())
    }

    fn load_outgoing(&mut self) -> GraphResult<()> {
        match &self.filter {
            Some(et) => {
                AdjacencyIndex::for_each_outgoing_by_type(self.tx, self.node, et, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        self.pending.push_back(ExpandResult::new(
                            edge.target,
                            edge_id,
                            Direction::Outgoing,
                        ));
                    }
                    Ok(true)
                })?;
            }
            None => {
                AdjacencyIndex::for_each_outgoing(self.tx, self.node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        self.pending.push_back(ExpandResult::new(
                            edge.target,
                            edge_id,
                            Direction::Outgoing,
                        ));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    fn load_incoming(&mut self) -> GraphResult<()> {
        match &self.filter {
            Some(et) => {
                AdjacencyIndex::for_each_incoming_by_type(self.tx, self.node, et, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        self.pending.push_back(ExpandResult::new(
                            edge.source,
                            edge_id,
                            Direction::Incoming,
                        ));
                    }
                    Ok(true)
                })?;
            }
            None => {
                AdjacencyIndex::for_each_incoming(self.tx, self.node, |edge_id| {
                    if let Some(edge) = EdgeStore::get(self.tx, edge_id)? {
                        self.pending.push_back(ExpandResult::new(
                            edge.source,
                            edge_id,
                            Direction::Incoming,
                        ));
                    }
                    Ok(true)
                })?;
            }
        }
        Ok(())
    }

    /// Try to get the next neighbor.
    fn try_next(&mut self) -> GraphResult<Option<ExpandResult>> {
        self.load_neighbors()?;
        Ok(self.pending.pop_front())
    }

    /// Collect all neighbors.
    pub fn collect_all(mut self) -> GraphResult<Vec<ExpandResult>> {
        self.load_neighbors()?;
        Ok(self.pending.into_iter().collect())
    }
}

/// Adapter to make `NeighborIterator` a standard Iterator.
pub struct NeighborIteratorAdapter<'a, T: Transaction> {
    inner: NeighborIterator<'a, T>,
    errored: bool,
}

impl<'a, T: Transaction> NeighborIteratorAdapter<'a, T> {
    /// Create a new adapter.
    pub fn new(tx: &'a T, node: EntityId, direction: Direction) -> Self {
        Self { inner: NeighborIterator::new(tx, node, direction), errored: false }
    }

    /// Filter to a specific edge type.
    pub fn with_edge_type(mut self, edge_type: impl Into<EdgeType>) -> Self {
        self.inner = self.inner.with_edge_type(edge_type);
        self
    }
}

impl<T: Transaction> Iterator for NeighborIteratorAdapter<'_, T> {
    type Item = GraphResult<ExpandResult>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.errored {
            return None;
        }

        match self.inner.try_next() {
            Ok(Some(result)) => Some(Ok(result)),
            Ok(None) => None,
            Err(e) => {
                self.errored = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_config_builder() {
        let config = TraversalConfig::new(Direction::Both)
            .with_min_depth(2)
            .with_max_depth(5)
            .with_edge_type("FRIEND")
            .with_limit(100);

        assert_eq!(config.direction, Direction::Both);
        assert_eq!(config.min_depth, 2);
        assert_eq!(config.max_depth, Some(5));
        assert_eq!(config.filter.limit, Some(100));
    }

    #[test]
    fn traversal_config_include_start() {
        let config = TraversalConfig::new(Direction::Outgoing).include_start();

        assert!(config.include_start);
        assert_eq!(config.min_depth, 0);
    }

    #[test]
    fn traversal_state_initialization() {
        let start = EntityId::new(1);
        let state = TraversalState::new(start);

        assert!(state.visited.contains(&start));
        assert_eq!(state.queue.len(), 1);
        assert_eq!(state.count, 0);
    }
}
