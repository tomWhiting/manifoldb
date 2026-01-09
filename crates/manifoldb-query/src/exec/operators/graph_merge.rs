//! Graph MERGE operator for Cypher MERGE statements.
//!
//! This operator implements the get-or-create semantics of MERGE.
//! It tries to match an existing pattern and creates it if not found.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{EdgeId, EdgeType, EntityId, Label, Value};
use manifoldb_graph::traversal::Direction;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{
    CreateEdgeRequest, CreateNodeRequest, GraphAccessor, GraphMutator, UpdateEdgeRequest,
    UpdateNodeRequest,
};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{GraphMergeNode, GraphSetAction, LogicalExpr, MergePatternSpec};

/// Graph MERGE operator.
///
/// Implements MERGE (get-or-create) semantics for nodes and relationships.
/// Supports:
/// - MERGE (n:Label {key: value}) - merge node
/// - MERGE (a)-[r:TYPE]->(b) - merge relationship
/// - ON CREATE SET - actions when creating
/// - ON MATCH SET - actions when matching existing
pub struct GraphMergeOp {
    /// Base operator state.
    base: OperatorBase,
    /// The MERGE node specification.
    merge_node: GraphMergeNode,
    /// Optional input operator (from MATCH clause for relationship variables).
    input: Option<BoxedOperator>,
    /// Graph accessor for read operations.
    graph_accessor: Option<Arc<dyn GraphAccessor>>,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
    /// Whether we've produced the output row yet (for no-input case).
    produced: bool,
}

impl GraphMergeOp {
    /// Creates a new graph MERGE operator.
    #[must_use]
    pub fn new(merge_node: GraphMergeNode, input: Option<BoxedOperator>) -> Self {
        // Build output schema from RETURN clause or merged variable
        let columns: Vec<String> = if merge_node.returning.is_empty() {
            // Return the merged variable
            match &merge_node.pattern {
                MergePatternSpec::Node { variable, .. } => vec![variable.clone()],
                MergePatternSpec::Relationship { rel_variable, .. } => {
                    rel_variable.as_ref().map(|v| vec![v.clone()]).unwrap_or_default()
                }
            }
        } else {
            // Use RETURN clause expressions as column names
            merge_node
                .returning
                .iter()
                .enumerate()
                .map(|(i, expr)| expr_to_column_name(expr, i))
                .collect()
        };

        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            merge_node,
            input,
            graph_accessor: None,
            graph_mutator: None,
            produced: false,
        }
    }

    /// Executes the MERGE operation for a single input row.
    fn execute_merge(&self, input_row: Option<&Row>) -> OperatorResult<Row> {
        match &self.merge_node.pattern.clone() {
            MergePatternSpec::Node { variable, labels, match_properties } => {
                self.merge_node_pattern(variable, labels, match_properties, input_row)
            }
            MergePatternSpec::Relationship {
                start_var,
                rel_variable,
                rel_type,
                match_properties,
                end_var,
            } => self.merge_relationship_pattern(
                start_var,
                rel_variable.as_deref(),
                rel_type,
                match_properties,
                end_var,
                input_row,
            ),
        }
    }

    /// Merges a node pattern.
    fn merge_node_pattern(
        &self,
        variable: &str,
        labels: &[String],
        match_properties: &[(String, LogicalExpr)],
        input_row: Option<&Row>,
    ) -> OperatorResult<Row> {
        let accessor = self
            .graph_accessor
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Evaluate match properties
        let empty_schema = Arc::new(Schema::empty());
        let empty_row = Row::new(Arc::clone(&empty_schema), vec![]);
        let row_for_eval = input_row.unwrap_or(&empty_row);

        let mut evaluated_props: Vec<(String, Value)> = Vec::new();
        for (key, expr) in match_properties {
            let val = evaluate_expr(expr, row_for_eval)?;
            evaluated_props.push((key.clone(), val));
        }

        // Try to find existing node with matching labels and properties
        let primary_label = labels.first().map(String::as_str);
        let existing_nodes = accessor
            .scan_nodes(primary_label)
            .map_err(|e| ParseError::InvalidGraphOp(format!("failed to scan nodes: {e}")))?;

        // Find a node that matches all labels and properties
        let matched_node = existing_nodes.into_iter().find(|node| {
            // Check all labels
            if !labels.iter().all(|l| node.labels.contains(l)) {
                return false;
            }
            // Check all properties
            evaluated_props.iter().all(|(key, val)| node.properties.get(key) == Some(val))
        });

        let (entity_id, was_created) = if let Some(node) = matched_node {
            // Found existing - execute ON MATCH actions
            (node.id, false)
        } else {
            // Not found - create new node
            let mut request = CreateNodeRequest::new();

            // Add labels
            for label in labels {
                request = request.with_label(Label::new(label));
            }

            // Add match properties (these become the node's initial properties)
            for (key, val) in &evaluated_props {
                request = request.with_property(key.clone(), val.clone());
            }

            let entity = mutator
                .create_node(&request)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to create node: {e}")))?;

            (entity.id, true)
        };

        // Execute appropriate actions
        if was_created {
            self.execute_on_create_actions(entity_id, None, input_row)?;
        } else {
            self.execute_on_match_actions(entity_id, None, input_row)?;
        }

        // Build output row
        self.build_output_row(variable, entity_id, input_row)
    }

    /// Merges a relationship pattern.
    fn merge_relationship_pattern(
        &self,
        start_var: &str,
        rel_variable: Option<&str>,
        rel_type: &str,
        match_properties: &[(String, LogicalExpr)],
        end_var: &str,
        input_row: Option<&Row>,
    ) -> OperatorResult<Row> {
        let accessor = self
            .graph_accessor
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // We need input row for relationship merging (start and end nodes must be bound)
        let input_row = input_row.ok_or_else(|| {
            ParseError::InvalidGraphOp(
                "relationship MERGE requires bound start and end nodes".to_string(),
            )
        })?;

        // Resolve start and end node IDs from input row
        let start_id = self.resolve_entity_id(start_var, input_row)?;
        let end_id = self.resolve_entity_id(end_var, input_row)?;

        // Evaluate match properties
        let mut evaluated_props: Vec<(String, Value)> = Vec::new();
        for (key, expr) in match_properties {
            let val = evaluate_expr(expr, input_row)?;
            evaluated_props.push((key.clone(), val));
        }

        // Try to find existing edge with matching type and properties
        let edge_type = EdgeType::new(rel_type);
        let neighbors = accessor
            .neighbors_by_type(start_id, Direction::Outgoing, &edge_type)
            .map_err(|e| ParseError::InvalidGraphOp(format!("failed to get neighbors: {e}")))?;

        // Find edge that connects to end_id and matches properties
        let matched_edge = neighbors.into_iter().find(|n| {
            if n.node != end_id {
                return false;
            }
            // Check properties on the edge
            if let Ok(Some(edge_props)) = accessor.get_edge_properties(n.edge_id) {
                evaluated_props.iter().all(|(key, val)| edge_props.get(key) == Some(val))
            } else {
                evaluated_props.is_empty()
            }
        });

        let (edge_id, was_created) = if let Some(neighbor) = matched_edge {
            // Found existing - execute ON MATCH actions
            (neighbor.edge_id, false)
        } else {
            // Not found - create new edge
            let mut request = CreateEdgeRequest::new(start_id, end_id, rel_type.to_string());

            // Add match properties
            for (key, val) in &evaluated_props {
                request = request.with_property(key.clone(), val.clone());
            }

            let edge = mutator
                .create_edge(&request)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to create edge: {e}")))?;

            (edge.id, true)
        };

        // Execute appropriate actions
        if was_created {
            self.execute_on_create_actions(
                EntityId::new(edge_id.as_u64()),
                Some(edge_id),
                Some(input_row),
            )?;
        } else {
            self.execute_on_match_actions(
                EntityId::new(edge_id.as_u64()),
                Some(edge_id),
                Some(input_row),
            )?;
        }

        // Build output row with edge ID
        self.build_edge_output_row(rel_variable, edge_id, input_row)
    }

    /// Resolves an entity ID from a variable name in the input row.
    fn resolve_entity_id(&self, var_name: &str, input_row: &Row) -> OperatorResult<EntityId> {
        let value = input_row.get_by_name(var_name).ok_or_else(|| {
            ParseError::InvalidGraphOp(format!("variable '{}' not found in row", var_name))
        })?;

        match value {
            Value::Int(id) => Ok(EntityId::new(*id as u64)),
            _ => Err(ParseError::InvalidGraphOp(format!(
                "variable '{}' is not an entity ID",
                var_name
            ))),
        }
    }

    /// Executes ON CREATE SET actions.
    fn execute_on_create_actions(
        &self,
        entity_id: EntityId,
        edge_id: Option<EdgeId>,
        input_row: Option<&Row>,
    ) -> OperatorResult<()> {
        self.execute_set_actions(&self.merge_node.on_create.clone(), entity_id, edge_id, input_row)
    }

    /// Executes ON MATCH SET actions.
    fn execute_on_match_actions(
        &self,
        entity_id: EntityId,
        edge_id: Option<EdgeId>,
        input_row: Option<&Row>,
    ) -> OperatorResult<()> {
        self.execute_set_actions(&self.merge_node.on_match.clone(), entity_id, edge_id, input_row)
    }

    /// Executes a list of SET actions on an entity or edge.
    fn execute_set_actions(
        &self,
        actions: &[GraphSetAction],
        entity_id: EntityId,
        edge_id: Option<EdgeId>,
        input_row: Option<&Row>,
    ) -> OperatorResult<()> {
        if actions.is_empty() {
            return Ok(());
        }

        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Build evaluation row with the merged entity/edge ID
        let eval_row = self.build_eval_row(entity_id, edge_id, input_row);

        // Group actions into node and edge updates
        let mut node_updates: HashMap<EntityId, UpdateNodeRequest> = HashMap::new();
        let mut edge_updates: HashMap<EdgeId, UpdateEdgeRequest> = HashMap::new();

        for action in actions {
            match action {
                GraphSetAction::Property { variable, property, value } => {
                    let val = evaluate_expr(value, &eval_row)?;

                    // Determine if this is targeting the merged entity/edge or something else
                    let target_id = if let Some(target_val) = eval_row.get_by_name(variable) {
                        match target_val {
                            Value::Int(id) => EntityId::new(*id as u64),
                            _ => continue,
                        }
                    } else {
                        continue;
                    };

                    // If this is the merged edge, use edge updates
                    if let Some(eid) = edge_id {
                        if target_id.as_u64() == eid.as_u64() {
                            let update = edge_updates
                                .entry(eid)
                                .or_insert_with(|| UpdateEdgeRequest::new(eid));
                            update.set_properties.insert(property.clone(), val);
                            continue;
                        }
                    }

                    // Otherwise treat as node update
                    let update = node_updates
                        .entry(target_id)
                        .or_insert_with(|| UpdateNodeRequest::new(target_id));
                    update.set_properties.insert(property.clone(), val);
                }
                GraphSetAction::Label { variable, label } => {
                    // Labels only apply to nodes
                    let target_id = if let Some(target_val) = eval_row.get_by_name(variable) {
                        match target_val {
                            Value::Int(id) => EntityId::new(*id as u64),
                            _ => continue,
                        }
                    } else {
                        continue;
                    };

                    let update = node_updates
                        .entry(target_id)
                        .or_insert_with(|| UpdateNodeRequest::new(target_id));
                    update.add_labels.push(Label::new(label));
                }
            }
        }

        // Execute updates
        for (_, update) in node_updates {
            mutator
                .update_node(&update)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to update node: {e}")))?;
        }

        for (_, update) in edge_updates {
            mutator
                .update_edge(&update)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to update edge: {e}")))?;
        }

        Ok(())
    }

    /// Builds an evaluation row containing the merged entity/edge ID.
    fn build_eval_row(
        &self,
        entity_id: EntityId,
        edge_id: Option<EdgeId>,
        input_row: Option<&Row>,
    ) -> Row {
        let mut columns = Vec::new();
        let mut values = Vec::new();

        // Add input row columns if present
        if let Some(row) = input_row {
            for col in row.schema().columns() {
                columns.push(col.to_string());
            }
            values.extend(row.values().to_vec());
        }

        // Add the merged variable
        match &self.merge_node.pattern {
            MergePatternSpec::Node { variable, .. } => {
                columns.push(variable.clone());
                values.push(Value::Int(entity_id.as_u64() as i64));
            }
            MergePatternSpec::Relationship { rel_variable, .. } => {
                if let Some(var) = rel_variable {
                    let id = edge_id.map(|e| e.as_u64()).unwrap_or(entity_id.as_u64());
                    columns.push(var.clone());
                    values.push(Value::Int(id as i64));
                }
            }
        }

        let schema = Arc::new(Schema::new(columns));
        Row::new(schema, values)
    }

    /// Builds the output row for a node merge.
    fn build_output_row(
        &self,
        variable: &str,
        entity_id: EntityId,
        input_row: Option<&Row>,
    ) -> OperatorResult<Row> {
        let schema = self.base.schema();

        if self.merge_node.returning.is_empty() {
            // Return just the entity ID
            Ok(Row::new(schema, vec![Value::Int(entity_id.as_u64() as i64)]))
        } else {
            // Evaluate RETURN expressions
            let mut columns = Vec::new();
            let mut values = Vec::new();

            // Add input row columns
            if let Some(row) = input_row {
                for col in row.schema().columns() {
                    columns.push(col.to_string());
                }
                values.extend(row.values().to_vec());
            }

            // Add the merged variable
            columns.push(variable.to_string());
            values.push(Value::Int(entity_id.as_u64() as i64));

            let eval_schema = Arc::new(Schema::new(columns));
            let eval_row = Row::new(eval_schema, values);

            let result_values: Vec<Value> = self
                .merge_node
                .returning
                .iter()
                .map(|expr| evaluate_expr(expr, &eval_row))
                .collect::<OperatorResult<Vec<_>>>()?;

            Ok(Row::new(schema, result_values))
        }
    }

    /// Builds the output row for an edge merge.
    fn build_edge_output_row(
        &self,
        rel_variable: Option<&str>,
        edge_id: EdgeId,
        input_row: &Row,
    ) -> OperatorResult<Row> {
        let schema = self.base.schema();

        if self.merge_node.returning.is_empty() {
            // Return just the edge ID (if variable exists)
            if rel_variable.is_some() {
                Ok(Row::new(schema, vec![Value::Int(edge_id.as_u64() as i64)]))
            } else {
                Ok(Row::new(schema, vec![]))
            }
        } else {
            // Evaluate RETURN expressions
            let mut columns = Vec::new();
            let mut values = Vec::new();

            // Add input row columns
            for col in input_row.schema().columns() {
                columns.push(col.to_string());
            }
            values.extend(input_row.values().to_vec());

            // Add the merged relationship variable
            if let Some(var) = rel_variable {
                columns.push(var.to_string());
                values.push(Value::Int(edge_id.as_u64() as i64));
            }

            let eval_schema = Arc::new(Schema::new(columns));
            let eval_row = Row::new(eval_schema, values);

            let result_values: Vec<Value> = self
                .merge_node
                .returning
                .iter()
                .map(|expr| evaluate_expr(expr, &eval_row))
                .collect::<OperatorResult<Vec<_>>>()?;

            Ok(Row::new(schema, result_values))
        }
    }
}

impl Operator for GraphMergeOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open input operator if present
        if let Some(ref mut input) = self.input {
            input.open(ctx)?;
        }

        // Capture both accessor and mutator from context
        self.graph_accessor = Some(ctx.graph_arc());
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.produced = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if let Some(ref mut input) = self.input {
            // With input: merge for each input row
            match input.next()? {
                Some(input_row) => {
                    let row = self.execute_merge(Some(&input_row))?;
                    self.base.inc_rows_produced();
                    Ok(Some(row))
                }
                None => {
                    self.base.set_finished();
                    Ok(None)
                }
            }
        } else {
            // Without input: merge once
            if self.produced {
                self.base.set_finished();
                return Ok(None);
            }

            self.produced = true;
            let row = self.execute_merge(None)?;
            self.base.inc_rows_produced();
            Ok(Some(row))
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.close()?;
        }
        self.graph_accessor = None;
        self.graph_mutator = None;
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
        "GraphMerge"
    }
}

/// Derives a column name from an expression.
fn expr_to_column_name(expr: &LogicalExpr, index: usize) -> String {
    match expr {
        LogicalExpr::Column { name, .. } => name.clone(),
        LogicalExpr::Alias { alias, .. } => alias.clone(),
        _ => format!("col_{index}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_merge_schema_node() {
        let merge_node = GraphMergeNode::new(MergePatternSpec::Node {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            match_properties: vec![],
        });

        let op = GraphMergeOp::new(merge_node, None);

        assert_eq!(op.schema().columns(), &["n"]);
    }

    #[test]
    fn graph_merge_schema_relationship() {
        let merge_node = GraphMergeNode::new(MergePatternSpec::Relationship {
            start_var: "a".to_string(),
            rel_variable: Some("r".to_string()),
            rel_type: "KNOWS".to_string(),
            match_properties: vec![],
            end_var: "b".to_string(),
        });

        let op = GraphMergeOp::new(merge_node, None);

        assert_eq!(op.schema().columns(), &["r"]);
    }

    #[test]
    fn graph_merge_requires_storage() {
        let merge_node = GraphMergeNode::new(MergePatternSpec::Node {
            variable: "n".to_string(),
            labels: vec!["Person".to_string()],
            match_properties: vec![],
        });

        let mut op = GraphMergeOp::new(merge_node, None);

        // ExecutionContext::new() creates context with NullGraphAccessor/Mutator
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on next() since NullGraphAccessor returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("failed to scan nodes")
                || err.to_string().contains("no graph storage")
        );

        op.close().unwrap();
    }
}
