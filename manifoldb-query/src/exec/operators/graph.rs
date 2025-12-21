//! Graph traversal operators.
//!
//! These operators integrate with the manifoldb-graph crate
//! for graph pattern matching and traversal.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::ExpandLength;
use crate::plan::physical::GraphExpandExecNode;

/// Graph expand operator.
///
/// Expands from source nodes to neighbors based on direction and edge types.
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
}

/// An expanded node result.
#[derive(Debug, Clone)]
struct ExpandedNode {
    /// The neighbor entity ID.
    entity_id: EntityId,
    /// The edge ID (if tracking edges).
    #[allow(dead_code)]
    edge_id: Option<manifoldb_core::EdgeId>,
    /// Depth for variable-length expansion.
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
        }
    }

    /// Simulates graph expansion (in a real implementation, this would query the graph store).
    fn expand_from(&self, source_id: EntityId) -> Vec<ExpandedNode> {
        // This is a mock implementation for testing.
        // In production, this would call into manifoldb-graph traversal.
        let mut results = Vec::new();

        // Simulate some neighbors based on source ID
        let base = source_id.as_u64();
        match self.node.length {
            ExpandLength::Single => {
                // Single hop: return 2 neighbors
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
                        // Add edge ID if tracking
                        values.push(Value::Null);
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
    #[allow(dead_code)]
    steps: Vec<GraphExpandExecNode>,
    /// Whether to return all paths.
    #[allow(dead_code)]
    all_paths: bool,
    /// Whether to track the full path.
    #[allow(dead_code)]
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
    /// Edges in the path.
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
