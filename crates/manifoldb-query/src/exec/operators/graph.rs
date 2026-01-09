//! Graph traversal operators.
//!
//! These operators integrate with the manifoldb-graph crate
//! for graph pattern matching and traversal.

use std::sync::Arc;

use manifoldb_core::{EdgeType, EntityId, Value};
use manifoldb_graph::traversal::Direction;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{
    GraphAccessError, GraphAccessResult, GraphAccessor, PathFindConfig, PathStepConfig,
    ShortestPathConfig, ShortestPathResult,
};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{ExpandDirection, ExpandLength, LogicalExpr};
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
    /// The edge ID (if tracking edges).
    edge_id: Option<manifoldb_core::EdgeId>,
    /// Depth for variable-length expansion.
    /// TODO(v0.2): Use this field for variable-length path traversal (e.g., [:1..5]).
    #[allow(dead_code)]
    depth: usize,
}

impl GraphExpandOp {
    /// Creates a new graph expand operator.
    #[must_use]
    pub fn new(node: GraphExpandExecNode, input: BoxedOperator) -> Self {
        // Build output schema: input columns + dst_var + optional edge_var
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();
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
    ///
    /// Returns an error if no graph storage is configured.
    fn expand_from(&self, source_id: EntityId) -> GraphAccessResult<Vec<ExpandedNode>> {
        let graph = self.graph.as_ref().ok_or(GraphAccessError::NoStorage)?;
        let direction = Self::to_graph_direction(&self.node.direction);
        let edge_types = self.get_edge_types();

        let expanded: Vec<ExpandedNode> = match &self.node.length {
            ExpandLength::Single => {
                // Single hop expansion
                let results = if let Some(ref types) = edge_types {
                    graph.neighbors_by_types(source_id, direction, types)?
                } else {
                    graph.neighbors(source_id, direction)?
                };

                results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: Some(r.edge_id), depth: 1 })
                    .collect()
            }
            ExpandLength::Exact(n) => {
                // Exact depth: use expand_all with min=max=n
                let results =
                    graph.expand_all(source_id, direction, *n, Some(*n), edge_types.as_deref())?;

                results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: r.edge_id, depth: r.depth })
                    .collect()
            }
            ExpandLength::Range { min, max } => {
                // Variable length expansion
                let results =
                    graph.expand_all(source_id, direction, *min, *max, edge_types.as_deref())?;

                results
                    .into_iter()
                    .map(|r| ExpandedNode { entity_id: r.node, edge_id: r.edge_id, depth: r.depth })
                    .collect()
            }
        };

        // Apply node label filter if specified
        let expanded = if self.node.node_labels.is_empty() {
            expanded
        } else {
            expanded
                .into_iter()
                .filter(|e| {
                    graph.entity_has_labels(e.entity_id, &self.node.node_labels).unwrap_or(false)
                })
                .collect()
        };

        // Apply node property filter if specified
        let expanded = if let Some(ref filter) = self.node.node_filter {
            expanded
                .into_iter()
                .filter(|e| self.evaluate_node_filter(e.entity_id, filter).unwrap_or(false))
                .collect()
        } else {
            expanded
        };

        // Apply edge property filter if specified
        let expanded = if let Some(ref filter) = self.node.edge_filter {
            expanded
                .into_iter()
                .filter(|e| {
                    e.edge_id
                        .map(|eid| self.evaluate_edge_filter(eid, filter).unwrap_or(false))
                        .unwrap_or(false)
                })
                .collect()
        } else {
            expanded
        };

        Ok(expanded)
    }

    /// Evaluates a node property filter predicate against an entity.
    fn evaluate_node_filter(
        &self,
        entity_id: EntityId,
        filter: &LogicalExpr,
    ) -> Result<bool, GraphAccessError> {
        let graph = self.graph.as_ref().ok_or(GraphAccessError::NoStorage)?;

        // Get the entity's properties
        let props = graph.get_entity_properties(entity_id)?;
        let props = match props {
            Some(p) => p,
            None => return Ok(false), // Entity doesn't exist
        };

        // Build a row with the node properties for filter evaluation
        // The filter references properties using the destination variable name
        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (key, value) in props {
            // Add both qualified (n.prop) and unqualified (prop) column names
            columns.push(format!("{}.{}", self.node.dst_var, key));
            values.push(value);
        }

        let schema = Arc::new(Schema::new(columns));
        let row = Row::new(schema, values);

        // Evaluate the filter expression
        match evaluate_expr(filter, &row) {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(Value::Null) => Ok(false),
            Ok(_) => Ok(false),
            Err(_) => Ok(false), // If filter evaluation fails, exclude the node
        }
    }

    /// Evaluates an edge property filter predicate against an edge.
    fn evaluate_edge_filter(
        &self,
        edge_id: manifoldb_core::EdgeId,
        filter: &LogicalExpr,
    ) -> Result<bool, GraphAccessError> {
        let graph = self.graph.as_ref().ok_or(GraphAccessError::NoStorage)?;

        // Get the edge's properties
        let props = graph.get_edge_properties(edge_id)?;
        let props = match props {
            Some(p) => p,
            None => return Ok(false), // Edge doesn't exist
        };

        // Build a row with the edge properties for filter evaluation
        // The filter references properties using the edge variable name
        let edge_var = self.node.edge_var.as_deref().unwrap_or("_edge");

        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (key, value) in props {
            // Add both qualified (r.prop) and unqualified (prop) column names
            columns.push(format!("{}.{}", edge_var, key));
            values.push(value);
        }

        let schema = Arc::new(Schema::new(columns));
        let row = Row::new(schema, values);

        // Evaluate the filter expression
        match evaluate_expr(filter, &row) {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(Value::Null) => Ok(false),
            Ok(_) => Ok(false),
            Err(_) => Ok(false), // If filter evaluation fails, exclude the edge
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
                        self.expanded = self.expand_from(source_id).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("graph expand failed: {e}"))
                        })?;
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
/// Executes multi-hop path patterns using actual graph storage for traversal.
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
    /// Graph accessor for actual storage traversal.
    graph: Option<Arc<dyn GraphAccessor>>,
}

/// A path result.
#[derive(Debug, Clone)]
struct PathResult {
    /// Nodes in the path.
    nodes: Vec<EntityId>,
    /// Edges in the path.
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
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();
        // Add columns for path nodes
        columns.push("path_start".to_string());
        columns.push("path_end".to_string());
        if track_path {
            columns.push("path_nodes".to_string());
            columns.push("path_edges".to_string());
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
            graph: None,
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

    /// Converts `ExpandDirection` to `traversal::Direction`.
    fn to_graph_direction(dir: &ExpandDirection) -> Direction {
        match dir {
            ExpandDirection::Outgoing => Direction::Outgoing,
            ExpandDirection::Incoming => Direction::Incoming,
            ExpandDirection::Both => Direction::Both,
        }
    }

    /// Converts a `GraphExpandExecNode` step to a `PathStepConfig`.
    fn step_to_config(step: &GraphExpandExecNode) -> PathStepConfig {
        let direction = Self::to_graph_direction(&step.direction);
        let edge_types: Vec<EdgeType> =
            step.edge_types.iter().map(|s| EdgeType::new(s.as_str())).collect();

        let (min_hops, max_hops) = match &step.length {
            ExpandLength::Single => (1, Some(1)),
            ExpandLength::Exact(n) => (*n, Some(*n)),
            ExpandLength::Range { min, max } => (*min, *max),
        };

        PathStepConfig { direction, edge_types, min_hops, max_hops }
    }

    /// Builds a `PathFindConfig` from the operator's steps.
    fn build_path_config(&self) -> PathFindConfig {
        let steps: Vec<PathStepConfig> = self.steps.iter().map(Self::step_to_config).collect();

        // Use limit of 1 if not all_paths mode
        let limit = if self.all_paths { None } else { Some(1) };

        PathFindConfig {
            steps,
            limit,
            allow_cycles: false, // Default: no cycles
        }
    }

    /// Finds paths from a start node using actual graph storage.
    ///
    /// Returns an error if no graph storage is configured.
    fn find_paths(&self, start_id: EntityId) -> GraphAccessResult<Vec<PathResult>> {
        let graph = self.graph.as_ref().ok_or(GraphAccessError::NoStorage)?;
        let config = self.build_path_config();
        let matches = graph.find_paths(start_id, &config)?;

        Ok(matches
            .into_iter()
            .map(|pm| {
                let edges = pm.all_edges();
                PathResult { nodes: pm.nodes, edges }
            })
            .collect())
    }

    /// Gets the start entity ID from the input row.
    fn get_start_id(&self, row: &Row) -> Option<EntityId> {
        // First check if there's a src_var in the first step
        if let Some(first_step) = self.steps.first() {
            if let Some(val) = row.get_by_name(&first_step.src_var) {
                return match val {
                    Value::Int(id) => Some(EntityId::new(*id as u64)),
                    _ => None,
                };
            }
        }

        // Fall back to first column that might be an entity ID
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
        // Capture the graph accessor from the context
        self.graph = Some(ctx.graph_arc());
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

                    // Add path nodes and edges if tracking
                    if self.track_path {
                        let nodes: Vec<Value> =
                            path.nodes.iter().map(|n| Value::Int(n.as_u64() as i64)).collect();
                        values.push(Value::Array(nodes));

                        let edges: Vec<Value> =
                            path.edges.iter().map(|e| Value::Int(e.as_u64() as i64)).collect();
                        values.push(Value::Array(edges));
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
                        self.paths = self.find_paths(start_id).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("path find failed: {e}"))
                        })?;
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
    fn graph_expand_requires_graph_storage() {
        // Tests that GraphExpandOp returns an error when no graph storage is configured
        let node = GraphExpandExecNode::new("n", "m", ExpandDirection::Outgoing)
            .with_length(ExpandLength::Single)
            .with_cost(Cost::default());

        let mut op = GraphExpandOp::new(node, make_input());

        // ExecutionContext::new() creates context with NullGraphAccessor
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on first next() since NullGraphAccessor returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("graph expand failed"));

        op.close().unwrap();
    }

    #[test]
    fn graph_path_scan_requires_graph_storage() {
        // Tests that GraphPathScanOp returns an error when no graph storage is configured
        let steps = vec![GraphExpandExecNode::new("a", "b", ExpandDirection::Outgoing)];

        let mut op = GraphPathScanOp::new(steps, false, false, make_input());

        // ExecutionContext::new() creates context with NullGraphAccessor
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on first next() since NullGraphAccessor returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("path find failed"));

        op.close().unwrap();
    }

    #[test]
    fn graph_expand_schema_construction() {
        // Test that schema is correctly constructed even without graph storage
        let node = GraphExpandExecNode::new("n", "m", ExpandDirection::Outgoing)
            .with_length(ExpandLength::Single)
            .with_edge_var("e")
            .with_cost(Cost::default());

        let op = GraphExpandOp::new(node, make_input());

        // Should have n, m, and e columns
        assert_eq!(op.schema().columns().len(), 3);
        assert_eq!(op.schema().columns(), &["n".to_string(), "m".to_string(), "e".to_string()]);
    }

    #[test]
    fn graph_path_scan_schema_construction() {
        // Test that schema is correctly constructed even without graph storage
        let steps = vec![
            GraphExpandExecNode::new("n", "m", ExpandDirection::Outgoing),
            GraphExpandExecNode::new("m", "o", ExpandDirection::Outgoing),
        ];

        // Enable track_path
        let op = GraphPathScanOp::new(steps, false, true, make_input());

        // Should have n, path_start, path_end, path_nodes, path_edges columns
        assert_eq!(op.schema().columns().len(), 5);
        assert_eq!(
            op.schema().columns(),
            &[
                "n".to_string(),
                "path_start".to_string(),
                "path_end".to_string(),
                "path_nodes".to_string(),
                "path_edges".to_string()
            ]
        );
    }
}

// ============================================================================
// Shortest Path Operator
// ============================================================================

use crate::plan::physical::ShortestPathExecNode;

/// Shortest path operator.
///
/// Finds the shortest path(s) between source and target nodes using BFS.
/// Supports both single path and all shortest paths modes.
pub struct ShortestPathOp {
    /// Base operator state.
    base: OperatorBase,
    /// Shortest path execution configuration.
    node: ShortestPathExecNode,
    /// Input operator (provides source and target nodes).
    input: BoxedOperator,
    /// Current input row.
    current_input: Option<Row>,
    /// Found paths for current input.
    paths: Vec<ShortestPathResult>,
    /// Position in paths.
    position: usize,
    /// Graph accessor for actual storage traversal.
    graph: Option<Arc<dyn GraphAccessor>>,
}

impl ShortestPathOp {
    /// Creates a new shortest path operator.
    #[must_use]
    pub fn new(node: ShortestPathExecNode, input: BoxedOperator) -> Self {
        // Build output schema: input columns + path_variable (as a map containing nodes/edges)
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();

        // Add the path variable column
        if let Some(ref path_var) = node.path_variable {
            columns.push(path_var.clone());
        } else {
            columns.push("path".to_string());
        }

        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            node,
            input,
            current_input: None,
            paths: Vec::new(),
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

    /// Gets the source entity ID from the current input row.
    fn get_source_id(&self, row: &Row) -> Option<EntityId> {
        row.get_by_name(&self.node.src_var).and_then(|v| match v {
            Value::Int(id) => Some(EntityId::new(*id as u64)),
            _ => None,
        })
    }

    /// Gets the target entity ID from the current input row.
    fn get_target_id(&self, row: &Row) -> Option<EntityId> {
        row.get_by_name(&self.node.dst_var).and_then(|v| match v {
            Value::Int(id) => Some(EntityId::new(*id as u64)),
            _ => None,
        })
    }

    /// Builds the shortest path configuration from the exec node.
    fn build_config(&self) -> ShortestPathConfig {
        let direction = Self::to_graph_direction(&self.node.direction);
        let edge_types: Vec<EdgeType> =
            self.node.edge_types.iter().map(|s| EdgeType::new(s.as_str())).collect();

        let mut config = ShortestPathConfig::new(direction)
            .with_edge_types(edge_types)
            .with_find_all(self.node.find_all);

        if let Some(max) = self.node.max_length {
            config = config.with_max_depth(max);
        }

        config
    }

    /// Finds shortest path(s) between source and target.
    fn find_shortest_paths(
        &self,
        source: EntityId,
        target: EntityId,
    ) -> GraphAccessResult<Vec<ShortestPathResult>> {
        let graph = self.graph.as_ref().ok_or(GraphAccessError::NoStorage)?;
        let config = self.build_config();
        graph.shortest_path(source, target, &config)
    }

    /// Converts a path result to a Value (JSON string with _nodes and _edges).
    fn path_to_value(&self, path: &ShortestPathResult) -> Value {
        // The nodes(), relationships(), and length() functions expect a JSON object string
        // with _nodes and _edges arrays containing node/edge IDs
        let nodes: Vec<i64> = path.nodes.iter().map(|n| n.as_u64() as i64).collect();
        let edges: Vec<i64> = path.edges.iter().map(|e| e.as_u64() as i64).collect();

        // Format as JSON string that can be parsed by path functions
        let json = format!(
            r#"{{"_nodes": {:?}, "_edges": {:?}, "_length": {}}}"#,
            nodes, edges, path.length
        );

        Value::String(json)
    }
}

impl Operator for ShortestPathOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.current_input = None;
        self.paths.clear();
        self.position = 0;
        // Capture the graph accessor from the context
        self.graph = Some(ctx.graph_arc());
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

                    // Build output row: input values + path value
                    let mut values = input_row.values().to_vec();
                    values.push(self.path_to_value(path));

                    let row = Row::new(self.base.schema(), values);
                    self.base.inc_rows_produced();
                    return Ok(Some(row));
                }
            }

            // Get next input row
            match self.input.next()? {
                Some(row) => {
                    let source_id = self.get_source_id(&row);
                    let target_id = self.get_target_id(&row);

                    if let (Some(src), Some(tgt)) = (source_id, target_id) {
                        self.paths = self.find_shortest_paths(src, tgt).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("shortest path failed: {e}"))
                        })?;
                        self.current_input = Some(row);
                        self.position = 0;
                    } else {
                        // Missing source or target, skip this row
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
        "ShortestPath"
    }
}

#[cfg(test)]
mod shortest_path_tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use crate::plan::physical::Cost;

    fn make_two_node_input() -> BoxedOperator {
        // Input with source and target node IDs
        Box::new(ValuesOp::with_columns(
            vec!["a".to_string(), "b".to_string()],
            vec![vec![Value::Int(1), Value::Int(5)], vec![Value::Int(2), Value::Int(6)]],
        ))
    }

    #[test]
    fn shortest_path_requires_graph_storage() {
        // Tests that ShortestPathOp returns an error when no graph storage is configured
        let node = ShortestPathExecNode::new("a", "b")
            .with_path_variable("p")
            .with_direction(ExpandDirection::Both)
            .with_cost(Cost::default());

        let mut op = ShortestPathOp::new(node, make_two_node_input());

        // ExecutionContext::new() creates context with NullGraphAccessor
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on first next() since NullGraphAccessor returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("shortest path failed"));

        op.close().unwrap();
    }

    #[test]
    fn shortest_path_schema_construction() {
        // Test that schema is correctly constructed
        let node = ShortestPathExecNode::new("a", "b")
            .with_path_variable("mypath")
            .with_direction(ExpandDirection::Outgoing)
            .with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Should have a, b, and mypath columns
        assert_eq!(op.schema().columns().len(), 3);
        assert_eq!(
            op.schema().columns(),
            &["a".to_string(), "b".to_string(), "mypath".to_string()]
        );
    }

    #[test]
    fn shortest_path_default_path_variable() {
        // Test that default path variable is "path"
        let node = ShortestPathExecNode::new("a", "b").with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Should have a, b, and default "path" columns
        assert_eq!(op.schema().columns().len(), 3);
        assert_eq!(op.schema().columns(), &["a".to_string(), "b".to_string(), "path".to_string()]);
    }

    #[test]
    fn shortest_path_with_edge_types() {
        // Test that edge types are properly configured
        let node = ShortestPathExecNode::new("a", "b")
            .with_path_variable("p")
            .with_edge_types(vec!["KNOWS".to_string(), "WORKS_WITH".to_string()])
            .with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Just check it builds correctly - schema construction doesn't fail
        assert_eq!(op.schema().columns().len(), 3);
    }

    #[test]
    fn shortest_path_with_max_length() {
        // Test that max length is properly configured
        let node = ShortestPathExecNode::new("a", "b")
            .with_path_variable("p")
            .with_max_length(10)
            .with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Just check it builds correctly
        assert_eq!(op.schema().columns().len(), 3);
    }

    #[test]
    fn shortest_path_find_all() {
        // Test that find_all mode is properly configured
        let node = ShortestPathExecNode::new("a", "b")
            .with_path_variable("p")
            .with_find_all(true)
            .with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Just check it builds correctly
        assert_eq!(op.schema().columns().len(), 3);
    }

    #[test]
    fn path_to_value_format() {
        // Test that path_to_value produces correct JSON format
        let node =
            ShortestPathExecNode::new("a", "b").with_path_variable("p").with_cost(Cost::default());

        let op = ShortestPathOp::new(node, make_two_node_input());

        // Create a test path
        let test_path = ShortestPathResult::new(
            vec![EntityId::new(1), EntityId::new(2), EntityId::new(3)],
            vec![manifoldb_core::EdgeId::new(10), manifoldb_core::EdgeId::new(20)],
        );

        let value = op.path_to_value(&test_path);

        // Should be a JSON string
        if let Value::String(json) = value {
            assert!(json.contains("\"_nodes\""));
            assert!(json.contains("\"_edges\""));
            assert!(json.contains("\"_length\": 2"));
            assert!(json.contains("[1, 2, 3]"));
            assert!(json.contains("[10, 20]"));
        } else {
            panic!("Expected Value::String, got {:?}", value);
        }
    }

    #[test]
    fn operator_lifecycle() {
        // Test that the operator lifecycle (open/close) works correctly
        let node =
            ShortestPathExecNode::new("a", "b").with_path_variable("p").with_cost(Cost::default());

        let mut op = ShortestPathOp::new(node, make_two_node_input());

        // Initially not open
        assert_eq!(op.state(), OperatorState::Created);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Now should be open
        assert_eq!(op.state(), OperatorState::Open);

        // Close should work
        op.close().unwrap();
        assert_eq!(op.state(), OperatorState::Closed);
    }

    #[test]
    fn operator_name() {
        let node = ShortestPathExecNode::new("a", "b").with_cost(Cost::default());
        let op = ShortestPathOp::new(node, make_two_node_input());

        assert_eq!(op.name(), "ShortestPath");
    }
}
