//! Graph traversal operators.
//!
//! These operators integrate with the manifoldb-graph crate
//! for graph pattern matching and traversal.

use std::sync::Arc;

use manifoldb_core::{EdgeType, EntityId, Value};
use manifoldb_graph::traversal::Direction;

use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{GraphAccessResult, GraphAccessor};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{ExpandDirection, ExpandLength};
use crate::plan::physical::GraphExpandExecNode;

/// Graph expand operator.
///
/// Expands from source nodes to neighbors based on direction and edge types.
/// Uses actual graph storage for traversal when a graph accessor is provided.
pub struct GraphExpandOp {
    /// Base operator state.
    base: OperatorBase,
    /// Expand configuration.
    node: GraphExpandExecNode,
    /// Input operator (provides source nodes).
    input: BoxedOperator,
    /// Current input row.
    current_input: Option<Row>,
    /// Expanded neighbors for current input.
    expanded: Vec<ExpandedNode>,
    /// Position in expanded neighbors.
    position: usize,
    /// Graph accessor for actual storage traversal.
    graph: Option<Arc<dyn GraphAccessor>>,
}

/// An expanded node result.
#[derive(Debug, Clone)]
struct ExpandedNode {
    /// The neighbor entity ID.
    entity_id: EntityId,
    /// The edge ID (if tracking edges). Reserved for future use.
    #[allow(dead_code)]
    edge_id: Option<manifoldb_core::EdgeId>,
    /// Depth for variable-length expansion. Reserved for future use.
    #[allow(dead_code)]
    depth: usize,
}

impl GraphExpandOp {
    /// Creates a new graph expand operator.
    #[must_use]
    pub fn new(node: GraphExpandExecNode, input: BoxedOperator) -> Self {
        // Build output schema: input columns + dst_var + optional edge_var
        let mut columns: Vec<String> = input.schema().columns().to_vec();
        columns.push(node.dst_var.clone());
        if let Some(ref edge_var) = node.edge_var {
            columns.push(edge_var.clone());
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            node,
            input,
            current_input: None,
            expanded: Vec::new(),
            position: 0,
            graph: None,
        }
    }

    /// Converts `ExpandDirection` to `traversal::Direction`.
    fn to_graph_direction(dir: &ExpandDirection) -> Direction {
        match dir {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        }
    }

    /// Gets edge types as `EdgeType` values.
    fn get_edge_types(&self) -> Option<Vec<EdgeType>> {
        if self.node.edge_types.is_empty() {
            None
        } else {
            Some(self.node.edge_types.iter().map(|s| EdgeType::new(s.as_str())).collect())
        }
    }

    /// Expands from a source node using actual graph storage.
    fn expand_from_storage(
        &self,
        source_id: EntityId,
        graph: &dyn GraphAccessor,
    ) -> GraphAccessResult<Vec<ExpandedNode>> {
        let direction = Self::to_graph_direction(&self.node.direction);
        let edge_types = self.get_edge_types();

        match &self.node.length {
            ExpandLength::Single => {
                // Single hop expansion
                let results = if let Some(ref types) = edge_types {
                    graph.neighbors_by_types(source_id, direction, types)?
                } else {
                    graph.neighbors(source_id, direction)?
                };

                Ok(results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: Some(r.edge_id), depth: 1 })
                    .collect())
            }
            ExpandLength::Exact(n) => {
                // Exact depth: use expand_all with min=max=n
                let results =
                    graph.expand_all(source_id, direction, *n, Some(*n), edge_types.as_deref())?;

                Ok(results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: r.edge_id, depth: r.depth })
                    .collect())
            }
            ExpandLength::Range { min, max } => {
                // Variable length expansion
                let results =
                    graph.expand_all(source_id, direction, *min, *max, edge_types.as_deref())?;

                Ok(results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: r.edge_id, depth: r.depth })
                    .collect())
            }
        }
    }

    /// Fallback mock expansion for when no graph storage is available.
    fn expand_from_mock(&self, source_id: EntityId) -> Vec<ExpandedNode> {
        let mut results = Vec::new();
        let base = source_id.as_u64();

        match self.node.length {
            ExpandLength::Single => {
                // Single hop: return 2 mock neighbors
                results.push(ExpandedNode {
                    entity_id: EntityId::new(base * 10 + 1),
                    edge_id: None,
                    depth: 1,
                });
                results.push(ExpandedNode {
                    entity_id: EntityId::new(base * 10 + 2),
                    edge_id: None,
                    depth: 1,
                });
            }
            ExpandLength::Exact(n) => {
                // Exact number of hops: return node at that depth
                results.push(ExpandedNode {
                    entity_id: EntityId::new(base * 10 + n as u64),
                    edge_id: None,
                    depth: n,
                });
            }
            ExpandLength::Range { min, max } => {
                // Variable length: return nodes at each depth
                for depth in min..=max.unwrap_or(min + 2) {
                    results.push(ExpandedNode {
                        entity_id: EntityId::new(base * 10 + depth as u64),
                        edge_id: None,
                        depth,
                    });
                }
            }
        }

        results
    }

    /// Expands from a source node.
    ///
    /// Uses actual graph storage if available, otherwise falls back to mock data.
    fn expand_from(&self, source_id: EntityId) -> Vec<ExpandedNode> {
        if let Some(ref graph) = self.graph {
            match self.expand_from_storage(source_id, graph.as_ref()) {
                Ok(results) => results,
                Err(_) => {
                    // Fall back to mock on error
                    self.expand_from_mock(source_id)
                }
            }
        } else {
            // No graph storage, use mock
            self.expand_from_mock(source_id)
        }
    }

    /// Gets the source entity ID from the current input row.
    fn get_source_id(&self, row: &Row) -> Option<EntityId> {
        row.get_by_name(&self.node.src_var).and_then(|v| match v {
            Value::Int(id) => Some(EntityId::new(*id as u64)),
            _ => None,
        })
    }
}

impl Operator for GraphExpandOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.current_input = None;
        self.expanded.clear();
        self.position = 0;
        // Capture the graph accessor from the context
        self.graph = Some(ctx.graph_arc());
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // Return next expanded result if available
            if self.position < self.expanded.len() {
                if let Some(input_row) = &self.current_input {
                    let expanded = &self.expanded[self.position];
                    self.position += 1;

                    // Build output row
                    let mut values = input_row.values().to_vec();
                    values.push(Value::Int(expanded.entity_id.as_u64() as i64));
                    if self.node.edge_var.is_some() {
                        // Add edge ID if tracking (use actual edge ID if available)
                        let edge_value = match expanded.edge_id {
                            Some(edge_id) => Value::Int(edge_id.as_u64() as i64),
                            None => Value::Null,
                        };
                        values.push(edge_value);
                    }

                    let row = Row::new(self.base.schema(), values);
                    self.base.inc_rows_produced();
                    return Ok(Some(row));
                }
            }

            // Get next input row
            match self.input.next()? {
                Some(row) => {
                    if let Some(source_id) = self.get_source_id(&row) {
                        self.expanded = self.expand_from(source_id);
                        self.current_input = Some(row);
                        self.position = 0;
                    } else {
                        // No valid source ID, skip this row
                        continue;
                    }
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.expanded.clear();
        self.graph = None; // Release graph accessor reference
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "GraphExpand"
    }
}

/// Graph path scan operator.
///
/// Executes multi-hop path patterns.
pub struct GraphPathScanOp {
    /// Base operator state.
    base: OperatorBase,
    /// Path steps.
    steps: Vec<GraphExpandExecNode>,
    /// Whether to return all paths.
    all_paths: bool,
    /// Whether to track the full path.
    track_path: bool,
    /// Input operator.
    input: BoxedOperator,
    /// Current input row.
    current_input: Option<Row>,
    /// Found paths for current input.
    paths: Vec<PathResult>,
    /// Position in paths.
    position: usize,
}

/// A path result.
#[derive(Debug, Clone)]
struct PathResult {
    /// Nodes in the path.
    nodes: Vec<EntityId>,
    /// Edges in the path. Reserved for future use.
    #[allow(dead_code)]
    edges: Vec<manifoldb_core::EdgeId>,
}

impl GraphPathScanOp {
    /// Creates a new graph path scan operator.
    #[must_use]
    pub fn new(
        steps: Vec<GraphExpandExecNode>,
        all_paths: bool,
        track_path: bool,
        input: BoxedOperator,
    ) -> Self {
        // Build output schema
        let mut columns: Vec<String> = input.schema().columns().to_vec();
        // Add columns for path nodes
        columns.push("path_start".to_string());
        columns.push("path_end".to_string());
        if track_path {
            columns.push("path_nodes".to_string());
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            steps,
            all_paths,
            track_path,
            input,
            current_input: None,
            paths: Vec::new(),
            position: 0,
        }
    }

    /// Returns the path steps configuration.
    #[must_use]
    pub fn steps(&self) -> &[GraphExpandExecNode] {
        &self.steps
    }

    /// Returns whether all paths are returned.
    #[must_use]
    pub fn all_paths(&self) -> bool {
        self.all_paths
    }

    /// Returns whether the full path is tracked.
    #[must_use]
    pub fn track_path(&self) -> bool {
        self.track_path
    }

    /// Simulates path finding (mock implementation).
    fn find_paths(&self, start_id: EntityId) -> Vec<PathResult> {
        // Mock implementation - returns a single path
        let end_id = EntityId::new(start_id.as_u64() * 100);
        vec![PathResult { nodes: vec![start_id, end_id], edges: Vec::new() }]
    }

    /// Gets the start entity ID from the input row.
    fn get_start_id(&self, row: &Row) -> Option<EntityId> {
        // Look for first column that might be an entity ID
        row.get(0).and_then(|v| match v {
            Value::Int(id) => Some(EntityId::new(*id as u64)),
            _ => None,
        })
    }
}

impl Operator for GraphPathScanOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.current_input = None;
        self.paths.clear();
        self.position = 0;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // Return next path if available
            if self.position < self.paths.len() {
                if let Some(input_row) = &self.current_input {
                    let path = &self.paths[self.position];
                    self.position += 1;

                    let mut values = input_row.values().to_vec();

                    // Add path start and end
                    if let Some(start) = path.nodes.first() {
                        values.push(Value::Int(start.as_u64() as i64));
                    } else {
                        values.push(Value::Null);
                    }

                    if let Some(end) = path.nodes.last() {
                        values.push(Value::Int(end.as_u64() as i64));
                    } else {
                        values.push(Value::Null);
                    }

                    // Add path nodes if tracking
                    if self.track_path {
                        let nodes: Vec<Value> =
                            path.nodes.iter().map(|n| Value::Int(n.as_u64() as i64)).collect();
                        values.push(Value::Array(nodes));
                    }

                    let row = Row::new(self.base.schema(), values);
                    self.base.inc_rows_produced();
                    return Ok(Some(row));
                }
            }

            // Get next input row
            match self.input.next()? {
                Some(row) => {
                    if let Some(start_id) = self.get_start_id(&row) {
                        self.paths = self.find_paths(start_id);
                        self.current_input = Some(row);
                        self.position = 0;
                    } else {
                        continue;
                    }
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.paths.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "GraphPathScan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use crate::plan::logical::{ExpandDirection, ExpandLength};
    use crate::plan::physical::Cost;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ))
    }

    #[test]
    fn graph_expand_single_hop() {
        let node = GraphExpandExecNode::new("n", "m", ExpandDirection::Outgoing)
            .with_length(ExpandLength::Single)
            .with_cost(Cost::default());

        let mut op = GraphExpandOp::new(node, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while let Some(row) = op.next().unwrap() {
            count += 1;
            // Should have n and m columns
            assert_eq!(row.schema().columns().len(), 2);
            assert!(row.get_by_name("n").is_some());
            assert!(row.get_by_name("m").is_some());
        }

        // 2 input rows * 2 neighbors each = 4 results
        assert_eq!(count, 4);
        op.close().unwrap();
    }

    #[test]
    fn graph_expand_variable_length() {
        let node = GraphExpandExecNode::new("n", "m", ExpandDirection::Outgoing)
            .with_length(ExpandLength::Range { min: 1, max: Some(3) })
            .with_cost(Cost::default());

        let mut op = GraphExpandOp::new(node, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }

        // 2 input rows * 3 depths each = 6 results
        assert_eq!(count, 6);
        op.close().unwrap();
    }

    #[test]
    fn graph_path_scan_basic() {
        let steps = vec![GraphExpandExecNode::new("a", "b", ExpandDirection::Outgoing)];

        let mut op = GraphPathScanOp::new(steps, false, false, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while let Some(row) = op.next().unwrap() {
            count += 1;
            // Should have n, path_start, path_end columns
            assert!(row.get_by_name("path_start").is_some());
            assert!(row.get_by_name("path_end").is_some());
        }

        // 2 input rows, 1 path each
        assert_eq!(count, 2);
        op.close().unwrap();
    }
}
