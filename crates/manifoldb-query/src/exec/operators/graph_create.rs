//! Graph CREATE operator for Cypher CREATE statements.
//!
//! This operator creates nodes and relationships in the graph.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{EntityId, Label, Value};

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{CreateEdgeRequest, CreateNodeRequest, GraphMutator};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::operators::values::ValuesOp;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{CreateNodeSpec, CreateRelSpec, GraphCreateNode};

/// Graph CREATE operator.
///
/// Creates nodes and/or relationships in the graph based on the logical plan.
/// Supports:
/// - Creating nodes with labels and properties
/// - Creating relationships between nodes
/// - Multiple patterns in a single CREATE
/// - MATCH ... CREATE patterns (relationship creation)
/// - RETURN clause for returning created entities
pub struct GraphCreateOp {
    /// Base operator state.
    base: OperatorBase,
    /// The CREATE node specification.
    create_node: GraphCreateNode,
    /// Optional input operator (from MATCH clause).
    input: Option<BoxedOperator>,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
    /// Whether we've produced the output row yet (for no-input case).
    produced: bool,
    /// Created entities mapped by variable name (full Value::Node or Value::Edge).
    created_entities: HashMap<String, Value>,
}

impl GraphCreateOp {
    /// Creates a new graph CREATE operator.
    #[must_use]
    pub fn new(create_node: GraphCreateNode, input: Option<BoxedOperator>) -> Self {
        // Build output schema from RETURN clause or created entities
        let columns: Vec<String> = if create_node.returning.is_empty() {
            // If no RETURN clause, return all created variables
            let mut cols = Vec::new();
            for node_spec in &create_node.nodes {
                if let Some(ref var) = node_spec.variable {
                    cols.push(var.clone());
                }
            }
            for rel_spec in &create_node.relationships {
                if let Some(ref var) = rel_spec.rel_variable {
                    cols.push(var.clone());
                }
            }
            cols
        } else {
            // Use RETURN clause expressions as column names
            create_node
                .returning
                .iter()
                .enumerate()
                .map(|(i, expr)| expr_to_column_name(expr, i))
                .collect()
        };

        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            create_node,
            input,
            graph_mutator: None,
            produced: false,
            created_entities: HashMap::new(),
        }
    }

    /// Creates a node from a specification.
    fn create_node_from_spec(
        &mut self,
        spec: &CreateNodeSpec,
        input_row: Option<&Row>,
    ) -> OperatorResult<EntityId> {
        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Build the node request
        let mut request = CreateNodeRequest::new();

        // Add labels
        for label in &spec.labels {
            request = request.with_label(Label::new(label));
        }

        // Evaluate and add properties
        // For properties, we need to evaluate expressions against the input row
        // If there's no input row, we use an empty row for expression evaluation
        let empty_schema = Arc::new(Schema::empty());
        let empty_row = Row::new(empty_schema, vec![]);
        let row_for_eval = input_row.unwrap_or(&empty_row);

        let mut evaluated_properties = HashMap::new();
        for (key, expr) in &spec.properties {
            let value = evaluate_expr(expr, row_for_eval)?;
            evaluated_properties.insert(key.clone(), value.clone());
            request = request.with_property(key.clone(), value);
        }

        // Create the node
        let entity = mutator
            .create_node(&request)
            .map_err(|e| ParseError::InvalidGraphOp(format!("failed to create node: {e}")))?;

        // Store the full node value if there's a variable
        if let Some(ref var) = spec.variable {
            let node_value = Value::Node {
                id: entity.id.as_u64() as i64,
                labels: spec.labels.clone(),
                properties: evaluated_properties,
            };
            self.created_entities.insert(var.clone(), node_value);
        }

        Ok(entity.id)
    }

    /// Creates a relationship from a specification.
    fn create_relationship_from_spec(
        &mut self,
        spec: &CreateRelSpec,
        input_row: Option<&Row>,
    ) -> OperatorResult<manifoldb_core::EdgeId> {
        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Resolve source and target entity IDs
        let source_id = self.resolve_entity_id(&spec.start_var, input_row)?;
        let target_id = self.resolve_entity_id(&spec.end_var, input_row)?;

        // Build the edge request
        let mut request = CreateEdgeRequest::new(source_id, target_id, spec.rel_type.clone());

        // Evaluate and add properties
        let empty_schema = Arc::new(Schema::empty());
        let empty_row = Row::new(empty_schema, vec![]);
        let row_for_eval = input_row.unwrap_or(&empty_row);

        let mut evaluated_properties = HashMap::new();
        for (key, expr) in &spec.properties {
            let value = evaluate_expr(expr, row_for_eval)?;
            evaluated_properties.insert(key.clone(), value.clone());
            request = request.with_property(key.clone(), value);
        }

        // Create the edge
        let edge = mutator.create_edge(&request).map_err(|e| {
            ParseError::InvalidGraphOp(format!("failed to create relationship: {e}"))
        })?;

        // Store the full edge value if there's a variable
        if let Some(ref var) = spec.rel_variable {
            let edge_value = Value::Edge {
                id: edge.id.as_u64() as i64,
                edge_type: spec.rel_type.clone(),
                source: source_id.as_u64() as i64,
                target: target_id.as_u64() as i64,
                properties: evaluated_properties,
            };
            self.created_entities.insert(var.clone(), edge_value);
        }

        Ok(edge.id)
    }

    /// Resolves an entity ID from a variable name.
    ///
    /// First checks created entities, then checks the input row.
    fn resolve_entity_id(
        &self,
        var_name: &str,
        input_row: Option<&Row>,
    ) -> OperatorResult<EntityId> {
        // Check if we created this entity in this CREATE
        if let Some(value) = self.created_entities.get(var_name) {
            match value {
                Value::Node { id, .. } => return Ok(EntityId::new(*id as u64)),
                Value::Edge { id, .. } => return Ok(EntityId::new(*id as u64)),
                Value::Int(id) => return Ok(EntityId::new(*id as u64)),
                _ => {
                    return Err(ParseError::InvalidGraphOp(format!(
                        "variable '{var_name}' is not an entity"
                    )));
                }
            }
        }

        // Check the input row (from MATCH clause)
        if let Some(row) = input_row {
            if let Some(value) = row.get_by_name(var_name) {
                match value {
                    Value::Node { id, .. } => return Ok(EntityId::new(*id as u64)),
                    Value::Edge { id, .. } => return Ok(EntityId::new(*id as u64)),
                    Value::Int(id) => return Ok(EntityId::new(*id as u64)),
                    _ => {
                        return Err(ParseError::InvalidGraphOp(format!(
                            "variable '{var_name}' is not an entity ID"
                        )));
                    }
                }
            }
        }

        Err(ParseError::InvalidGraphOp(format!("unbound variable: {var_name}")))
    }

    /// Executes the CREATE operation for a single input row.
    fn execute_create(&mut self, input_row: Option<&Row>) -> OperatorResult<Row> {
        // Clear previously created entities for this execution
        self.created_entities.clear();

        // Create all nodes first
        for node_spec in self.create_node.nodes.clone() {
            self.create_node_from_spec(&node_spec, input_row)?;
        }

        // Then create all relationships
        for rel_spec in self.create_node.relationships.clone() {
            self.create_relationship_from_spec(&rel_spec, input_row)?;
        }

        // Build the output row
        self.build_output_row(input_row)
    }

    /// Builds the output row from created entities.
    fn build_output_row(&self, input_row: Option<&Row>) -> OperatorResult<Row> {
        let schema = self.base.schema();

        if self.create_node.returning.is_empty() {
            // Return full entity values for all created variables
            let cols = schema.columns();
            let values: Vec<Value> = cols
                .into_iter()
                .map(|col: &str| self.created_entities.get(col).cloned().unwrap_or(Value::Null))
                .collect();
            Ok(Row::new(schema, values))
        } else {
            // Evaluate RETURN expressions
            // Build a merged row with input + created entities
            let merged_row = self.build_merged_row(input_row);

            let values: Vec<Value> = self
                .create_node
                .returning
                .iter()
                .map(|expr| evaluate_expr(expr, &merged_row))
                .collect::<OperatorResult<Vec<_>>>()?;

            Ok(Row::new(schema, values))
        }
    }

    /// Builds a merged row containing input columns + created entities.
    fn build_merged_row(&self, input_row: Option<&Row>) -> Row {
        let mut columns = Vec::new();
        let mut values = Vec::new();

        // Add input row columns if present
        if let Some(row) = input_row {
            for col in row.schema().columns() {
                columns.push(col.to_string());
            }
            values.extend(row.values().to_vec());
        }

        // Add created entities (full Value::Node or Value::Edge)
        for (var, value) in &self.created_entities {
            columns.push(var.clone());
            values.push(value.clone());
        }

        let schema = Arc::new(Schema::new(columns));
        Row::new(schema, values)
    }
}

impl Operator for GraphCreateOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open input operator if present
        if let Some(ref mut input) = self.input {
            input.open(ctx)?;
        }

        // Capture the graph mutator from the context
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.produced = false;
        self.created_entities.clear();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if let Some(ref mut input) = self.input {
            // With input: create for each input row
            match input.next()? {
                Some(input_row) => {
                    let row = self.execute_create(Some(&input_row))?;
                    self.base.inc_rows_produced();
                    Ok(Some(row))
                }
                None => {
                    self.base.set_finished();
                    Ok(None)
                }
            }
        } else {
            // Without input: create once
            if self.produced {
                self.base.set_finished();
                return Ok(None);
            }

            self.produced = true;
            let row = self.execute_create(None)?;
            self.base.inc_rows_produced();
            Ok(Some(row))
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        if let Some(ref mut input) = self.input {
            input.close()?;
        }
        self.graph_mutator = None;
        self.created_entities.clear();
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
        "GraphCreate"
    }
}

/// Derives a column name from an expression.
fn expr_to_column_name(expr: &crate::plan::logical::LogicalExpr, index: usize) -> String {
    match expr {
        crate::plan::logical::LogicalExpr::Column { name, .. } => name.clone(),
        crate::plan::logical::LogicalExpr::Alias { alias, .. } => alias.clone(),
        _ => format!("col_{index}"),
    }
}

/// Creates a simple values input for CREATE without MATCH.
///
/// This provides a single empty row to trigger the CREATE once.
#[must_use]
pub fn single_row_input() -> BoxedOperator {
    Box::new(ValuesOp::new(Arc::new(Schema::empty()), vec![vec![]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_create_schema_from_nodes() {
        let create_node = GraphCreateNode::new()
            .with_node(CreateNodeSpec::new(Some("n".to_string()), vec!["Person".to_string()]))
            .with_node(CreateNodeSpec::new(Some("m".to_string()), vec!["Company".to_string()]));

        let op = GraphCreateOp::new(create_node, None);

        assert_eq!(op.schema().columns(), &["n", "m"]);
    }

    #[test]
    fn graph_create_schema_from_relationships() {
        let create_node = GraphCreateNode::new()
            .with_node(CreateNodeSpec::new(Some("a".to_string()), vec![]))
            .with_node(CreateNodeSpec::new(Some("b".to_string()), vec![]))
            .with_relationship(
                CreateRelSpec::new("a".to_string(), "KNOWS".to_string(), "b".to_string())
                    .with_variable("r".to_string()),
            );

        let op = GraphCreateOp::new(create_node, None);

        assert_eq!(op.schema().columns(), &["a", "b", "r"]);
    }

    #[test]
    fn graph_create_requires_storage() {
        let create_node = GraphCreateNode::new()
            .with_node(CreateNodeSpec::new(Some("n".to_string()), vec!["Person".to_string()]));

        let mut op = GraphCreateOp::new(create_node, None);

        // ExecutionContext::new() creates context with NullGraphMutator
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on next() since NullGraphMutator returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("failed to create node"));

        op.close().unwrap();
    }

    #[test]
    fn graph_create_anonymous_node() {
        // CREATE (:Person) - node without variable
        let create_node =
            GraphCreateNode::new().with_node(CreateNodeSpec::new(None, vec!["Person".to_string()]));

        let op = GraphCreateOp::new(create_node, None);

        // No columns since there's no variable
        assert!(op.schema().columns().is_empty());
    }
}
