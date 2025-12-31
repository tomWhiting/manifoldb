//! Connected Components algorithms.
//!
//! This module implements algorithms for finding connected components in graphs:
//!
//! - **Weakly Connected Components (WCC)**: Treats the graph as undirected and
//!   finds sets of nodes that are reachable from each other ignoring edge direction.
//!   Uses a Union-Find data structure for O(V + E) time complexity.
//!
//! - **Strongly Connected Components (SCC)**: For directed graphs, finds sets of
//!   nodes where every node is reachable from every other node following edge
//!   directions. Uses Tarjan's algorithm for O(V + E) time complexity.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::{ConnectedComponents, ConnectedComponentsConfig};
//!
//! // Find weakly connected components (treats graph as undirected)
//! let config = ConnectedComponentsConfig::default();
//! let wcc = ConnectedComponents::weakly_connected(&tx, &config)?;
//!
//! println!("Found {} weakly connected components", wcc.num_components);
//! for (component_id, size) in wcc.components_by_size().iter().take(5) {
//!     println!("Component {} has {} nodes", component_id, size);
//! }
//!
//! // Find strongly connected components (respects edge direction)
//! let scc = ConnectedComponents::strongly_connected(&tx, &config)?;
//!
//! println!("Found {} strongly connected components", scc.num_components);
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for Connected Components algorithms.
#[derive(Debug, Clone)]
pub struct ConnectedComponentsConfig {
    /// Maximum number of nodes allowed before returning an error.
    /// Set to `None` to disable the check.
    /// Default: 10,000,000 (10M nodes)
    pub max_graph_nodes: Option<usize>,
}

impl Default for ConnectedComponentsConfig {
    fn default() -> Self {
        Self { max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES) }
    }
}

impl ConnectedComponentsConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of nodes allowed.
    ///
    /// If the graph has more nodes than this limit, the algorithm will
    /// return a [`GraphError::GraphTooLarge`] error instead of attempting
    /// to allocate potentially gigabytes of memory.
    ///
    /// Set to `None` to disable the check (use with caution).
    ///
    /// [`GraphError::GraphTooLarge`]: crate::store::GraphError::GraphTooLarge
    pub const fn with_max_graph_nodes(mut self, limit: Option<usize>) -> Self {
        self.max_graph_nodes = limit;
        self
    }
}

/// Result of a connected components computation.
///
/// Contains the component assignment for each node, along with
/// metadata about the computation.
#[derive(Debug, Clone)]
pub struct ComponentResult {
    /// Component assignments: node -> component ID.
    /// Component IDs are contiguous integers starting from 0.
    pub assignments: HashMap<EntityId, usize>,

    /// Number of distinct components found.
    pub num_components: usize,
}

impl ComponentResult {
    /// Get the component ID for a specific node.
    pub fn component(&self, node: EntityId) -> Option<usize> {
        self.assignments.get(&node).copied()
    }

    /// Get all nodes in a specific component.
    pub fn nodes_in_component(&self, component_id: usize) -> Vec<EntityId> {
        self.assignments.iter().filter(|(_, &c)| c == component_id).map(|(&node, _)| node).collect()
    }

    /// Get component sizes.
    pub fn component_sizes(&self) -> HashMap<usize, usize> {
        let mut sizes: HashMap<usize, usize> = HashMap::new();
        for &component in self.assignments.values() {
            *sizes.entry(component).or_insert(0) += 1;
        }
        sizes
    }

    /// Get components sorted by size (descending).
    pub fn components_by_size(&self) -> Vec<(usize, usize)> {
        let mut sizes: Vec<_> = self.component_sizes().into_iter().collect();
        sizes.sort_by(|a, b| b.1.cmp(&a.1));
        sizes
    }

    /// Get the largest component.
    pub fn largest_component(&self) -> Option<(usize, usize)> {
        self.components_by_size().into_iter().next()
    }

    /// Get the smallest component.
    pub fn smallest_component(&self) -> Option<(usize, usize)> {
        self.components_by_size().into_iter().last()
    }

    /// Check if two nodes are in the same component.
    pub fn same_component(&self, node1: EntityId, node2: EntityId) -> bool {
        match (self.component(node1), self.component(node2)) {
            (Some(c1), Some(c2)) => c1 == c2,
            _ => false,
        }
    }

    /// Get the number of nodes in a specific component.
    pub fn component_size(&self, component_id: usize) -> usize {
        self.component_sizes().get(&component_id).copied().unwrap_or(0)
    }
}

/// Union-Find data structure with path compression and union by rank.
///
/// This provides near-constant time operations for union and find,
/// making it ideal for computing weakly connected components.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure with n elements.
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    /// Find the root of the set containing x, with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing x and y, using union by rank.
    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            match self.rank[root_x].cmp(&self.rank[root_y]) {
                std::cmp::Ordering::Less => {
                    self.parent[root_x] = root_y;
                }
                std::cmp::Ordering::Greater => {
                    self.parent[root_y] = root_x;
                }
                std::cmp::Ordering::Equal => {
                    self.parent[root_y] = root_x;
                    self.rank[root_x] += 1;
                }
            }
        }
    }
}

/// Connected Components algorithm implementations.
pub struct ConnectedComponents;

impl ConnectedComponents {
    /// Compute weakly connected components.
    ///
    /// This treats the graph as undirected - two nodes are in the same
    /// weakly connected component if there's a path between them ignoring
    /// edge direction.
    ///
    /// Uses Union-Find with path compression for O(V + E * α(V)) time complexity,
    /// where α is the inverse Ackermann function (effectively constant).
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `ComponentResult` containing component assignments for all nodes.
    pub fn weakly_connected<T: Transaction>(
        tx: &T,
        config: &ConnectedComponentsConfig,
    ) -> GraphResult<ComponentResult> {
        // Check graph size before allocating large data structures
        if let Some(limit) = config.max_graph_nodes {
            let node_count = NodeStore::count(tx)?;
            if node_count > limit {
                return Err(GraphError::GraphTooLarge { node_count, limit });
            }
        }

        // Collect all nodes
        let mut nodes: Vec<EntityId> = Vec::new();
        NodeStore::for_each(tx, |entity| {
            nodes.push(entity.id);
            true
        })?;

        let n = nodes.len();
        if n == 0 {
            return Ok(ComponentResult { assignments: HashMap::new(), num_components: 0 });
        }

        // Build node index for fast lookup
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Initialize Union-Find
        let mut uf = UnionFind::new(n);

        // Process all edges (both outgoing and incoming to treat as undirected)
        for (i, &node) in nodes.iter().enumerate() {
            // Get outgoing edges
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if let Some(&j) = node_index.get(&edge.target) {
                        uf.union(i, j);
                    }
                }
            }

            // Get incoming edges
            let incoming = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in incoming {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if let Some(&j) = node_index.get(&edge.source) {
                        uf.union(i, j);
                    }
                }
            }
        }

        // Extract components and renumber to be contiguous from 0
        let mut root_to_component: HashMap<usize, usize> = HashMap::new();
        let mut next_component = 0usize;

        let mut assignments: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            let root = uf.find(i);
            let component = *root_to_component.entry(root).or_insert_with(|| {
                let c = next_component;
                next_component += 1;
                c
            });
            assignments.insert(node, component);
        }

        Ok(ComponentResult { assignments, num_components: next_component })
    }

    /// Compute strongly connected components using Tarjan's algorithm.
    ///
    /// In a strongly connected component, every node is reachable from
    /// every other node following edge directions. This is a stricter
    /// property than weak connectivity.
    ///
    /// Uses Tarjan's algorithm for O(V + E) time complexity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `config` - Configuration parameters for the algorithm
    ///
    /// # Returns
    ///
    /// A `ComponentResult` containing component assignments for all nodes.
    pub fn strongly_connected<T: Transaction>(
        tx: &T,
        config: &ConnectedComponentsConfig,
    ) -> GraphResult<ComponentResult> {
        // Check graph size before allocating large data structures
        if let Some(limit) = config.max_graph_nodes {
            let node_count = NodeStore::count(tx)?;
            if node_count > limit {
                return Err(GraphError::GraphTooLarge { node_count, limit });
            }
        }

        // Collect all nodes
        let mut nodes: Vec<EntityId> = Vec::new();
        NodeStore::for_each(tx, |entity| {
            nodes.push(entity.id);
            true
        })?;

        let n = nodes.len();
        if n == 0 {
            return Ok(ComponentResult { assignments: HashMap::new(), num_components: 0 });
        }

        // Build node index for fast lookup
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency list (outgoing edges only for SCC)
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &node) in nodes.iter().enumerate() {
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if let Some(&j) = node_index.get(&edge.target) {
                        adjacency[i].push(j);
                    }
                }
            }
        }

        // Run Tarjan's algorithm
        let mut state = TarjanState::new(n);

        for i in 0..n {
            if state.index[i].is_none() {
                tarjan_dfs(i, &adjacency, &mut state);
            }
        }

        // Build result
        let mut assignments: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            if let Some(component) = state.component[i] {
                assignments.insert(node, component);
            }
        }

        Ok(ComponentResult { assignments, num_components: state.num_components })
    }

    /// Compute weakly connected components for a subset of nodes.
    ///
    /// Only considers edges within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn weakly_connected_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &ConnectedComponentsConfig,
    ) -> GraphResult<ComponentResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(ComponentResult { assignments: HashMap::new(), num_components: 0 });
        }

        // Check size limit
        if let Some(limit) = config.max_graph_nodes {
            if n > limit {
                return Err(GraphError::GraphTooLarge { node_count: n, limit });
            }
        }

        // Build node index and set
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Initialize Union-Find
        let mut uf = UnionFind::new(n);

        // Process all edges within the subgraph
        for (i, &node) in nodes.iter().enumerate() {
            // Get outgoing edges
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.target) {
                        if let Some(&j) = node_index.get(&edge.target) {
                            uf.union(i, j);
                        }
                    }
                }
            }

            // Get incoming edges
            let incoming = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
            for edge_id in incoming {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.source) {
                        if let Some(&j) = node_index.get(&edge.source) {
                            uf.union(i, j);
                        }
                    }
                }
            }
        }

        // Extract components
        let mut root_to_component: HashMap<usize, usize> = HashMap::new();
        let mut next_component = 0usize;

        let mut assignments: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            let root = uf.find(i);
            let component = *root_to_component.entry(root).or_insert_with(|| {
                let c = next_component;
                next_component += 1;
                c
            });
            assignments.insert(node, component);
        }

        Ok(ComponentResult { assignments, num_components: next_component })
    }

    /// Compute strongly connected components for a subset of nodes.
    ///
    /// Only considers edges within the specified subgraph.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use for graph access
    /// * `nodes` - The nodes to include in the computation
    /// * `config` - Configuration parameters for the algorithm
    pub fn strongly_connected_for_nodes<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        config: &ConnectedComponentsConfig,
    ) -> GraphResult<ComponentResult> {
        let n = nodes.len();
        if n == 0 {
            return Ok(ComponentResult { assignments: HashMap::new(), num_components: 0 });
        }

        // Check size limit
        if let Some(limit) = config.max_graph_nodes {
            if n > limit {
                return Err(GraphError::GraphTooLarge { node_count: n, limit });
            }
        }

        // Build node index and set
        let node_set: std::collections::HashSet<EntityId> = nodes.iter().copied().collect();
        let node_index: HashMap<EntityId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build adjacency list (outgoing edges only, within subgraph)
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &node) in nodes.iter().enumerate() {
            let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
            for edge_id in outgoing {
                if let Some(edge) = EdgeStore::get(tx, edge_id)? {
                    if node_set.contains(&edge.target) {
                        if let Some(&j) = node_index.get(&edge.target) {
                            adjacency[i].push(j);
                        }
                    }
                }
            }
        }

        // Run Tarjan's algorithm
        let mut state = TarjanState::new(n);

        for i in 0..n {
            if state.index[i].is_none() {
                tarjan_dfs(i, &adjacency, &mut state);
            }
        }

        // Build result
        let mut assignments: HashMap<EntityId, usize> = HashMap::with_capacity(n);
        for (i, &node) in nodes.iter().enumerate() {
            if let Some(component) = state.component[i] {
                assignments.insert(node, component);
            }
        }

        Ok(ComponentResult { assignments, num_components: state.num_components })
    }
}

/// State for Tarjan's SCC algorithm.
struct TarjanState {
    /// Discovery index for each node
    index: Vec<Option<usize>>,
    /// Low-link value for each node
    lowlink: Vec<usize>,
    /// Whether node is on the stack
    on_stack: Vec<bool>,
    /// The stack of nodes being processed
    stack: Vec<usize>,
    /// Component assignment for each node
    component: Vec<Option<usize>>,
    /// Current index counter
    current_index: usize,
    /// Number of components found
    num_components: usize,
}

impl TarjanState {
    fn new(n: usize) -> Self {
        Self {
            index: vec![None; n],
            lowlink: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::new(),
            component: vec![None; n],
            current_index: 0,
            num_components: 0,
        }
    }
}

/// Non-recursive Tarjan's DFS using explicit stack to avoid stack overflow on large graphs.
fn tarjan_dfs(start: usize, adjacency: &[Vec<usize>], state: &mut TarjanState) {
    // Work stack contains (node, neighbor_index, phase)
    // phase 0: initial visit
    // phase 1: processing neighbors
    // phase 2: post-processing after returning from neighbor
    let mut work_stack: Vec<(usize, usize, u8)> = vec![(start, 0, 0)];

    while let Some((v, neighbor_idx, phase)) = work_stack.pop() {
        match phase {
            0 => {
                // Initial visit
                state.index[v] = Some(state.current_index);
                state.lowlink[v] = state.current_index;
                state.current_index += 1;
                state.on_stack[v] = true;
                state.stack.push(v);

                // Move to phase 1
                work_stack.push((v, 0, 1));
            }
            1 => {
                // Processing neighbors
                if neighbor_idx < adjacency[v].len() {
                    let w = adjacency[v][neighbor_idx];

                    if state.index[w].is_none() {
                        // w not yet visited, recurse
                        work_stack.push((v, neighbor_idx + 1, 2)); // Return here after w
                        work_stack.push((w, 0, 0)); // Visit w
                    } else if state.on_stack[w] {
                        // w is on stack, update lowlink
                        // index[w] is Some because we checked is_none() above
                        if let Some(w_index) = state.index[w] {
                            state.lowlink[v] = state.lowlink[v].min(w_index);
                        }
                        work_stack.push((v, neighbor_idx + 1, 1)); // Continue with next neighbor
                    } else {
                        // w already processed, continue
                        work_stack.push((v, neighbor_idx + 1, 1));
                    }
                } else {
                    // All neighbors processed, check if v is a root
                    // index[v] is Some because v was visited in phase 0
                    if let Some(v_index) = state.index[v] {
                        if state.lowlink[v] == v_index {
                            // v is root of an SCC, pop the component
                            let component_id = state.num_components;
                            state.num_components += 1;

                            // Pop nodes from stack until we reach v
                            while let Some(w) = state.stack.pop() {
                                state.on_stack[w] = false;
                                state.component[w] = Some(component_id);
                                if w == v {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            2 => {
                // Returning from neighbor - update lowlink
                // The neighbor we just returned from is at neighbor_idx - 1
                let w = adjacency[v][neighbor_idx - 1];
                state.lowlink[v] = state.lowlink[v].min(state.lowlink[w]);

                // Continue with remaining neighbors
                work_stack.push((v, neighbor_idx, 1));
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = ConnectedComponentsConfig::default();
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
    }

    #[test]
    fn config_builder() {
        let config = ConnectedComponentsConfig::new().with_max_graph_nodes(Some(1000));
        assert_eq!(config.max_graph_nodes, Some(1000));

        let config = ConnectedComponentsConfig::new().with_max_graph_nodes(None);
        assert_eq!(config.max_graph_nodes, None);
    }

    #[test]
    fn result_empty() {
        let result = ComponentResult { assignments: HashMap::new(), num_components: 0 };

        assert!(result.component(EntityId::new(1)).is_none());
        assert!(result.nodes_in_component(0).is_empty());
        assert!(result.component_sizes().is_empty());
        assert!(result.largest_component().is_none());
        assert!(result.smallest_component().is_none());
    }

    #[test]
    fn result_operations() {
        let mut assignments = HashMap::new();
        assignments.insert(EntityId::new(1), 0);
        assignments.insert(EntityId::new(2), 0);
        assignments.insert(EntityId::new(3), 1);
        assignments.insert(EntityId::new(4), 1);
        assignments.insert(EntityId::new(5), 1);

        let result = ComponentResult { assignments, num_components: 2 };

        // Test component lookup
        assert_eq!(result.component(EntityId::new(1)), Some(0));
        assert_eq!(result.component(EntityId::new(3)), Some(1));
        assert_eq!(result.component(EntityId::new(99)), None);

        // Test nodes_in_component
        let nodes_0 = result.nodes_in_component(0);
        assert_eq!(nodes_0.len(), 2);

        let nodes_1 = result.nodes_in_component(1);
        assert_eq!(nodes_1.len(), 3);

        // Test sizes
        let sizes = result.component_sizes();
        assert_eq!(sizes.get(&0), Some(&2));
        assert_eq!(sizes.get(&1), Some(&3));

        // Test largest/smallest
        assert_eq!(result.largest_component(), Some((1, 3)));
        assert_eq!(result.smallest_component(), Some((0, 2)));

        // Test same_component
        assert!(result.same_component(EntityId::new(1), EntityId::new(2)));
        assert!(result.same_component(EntityId::new(3), EntityId::new(4)));
        assert!(!result.same_component(EntityId::new(1), EntityId::new(3)));

        // Test component_size
        assert_eq!(result.component_size(0), 2);
        assert_eq!(result.component_size(1), 3);
        assert_eq!(result.component_size(99), 0);
    }

    #[test]
    fn union_find_basic() {
        let mut uf = UnionFind::new(5);

        // Initially all separate
        assert_ne!(uf.find(0), uf.find(1));

        // Union 0 and 1
        uf.union(0, 1);
        assert_eq!(uf.find(0), uf.find(1));

        // Union 2 and 3
        uf.union(2, 3);
        assert_eq!(uf.find(2), uf.find(3));

        // 0-1 and 2-3 still separate
        assert_ne!(uf.find(0), uf.find(2));

        // Union them
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(2));
        assert_eq!(uf.find(0), uf.find(3));
    }

    #[test]
    fn union_find_chain() {
        let mut uf = UnionFind::new(10);

        // Create a chain: 0-1-2-3-4-5-6-7-8-9
        for i in 0..9 {
            uf.union(i, i + 1);
        }

        // All should be in same set
        let root = uf.find(0);
        for i in 1..10 {
            assert_eq!(uf.find(i), root);
        }
    }
}
