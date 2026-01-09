//! Graph SET operator for Cypher SET statements.
//!
//! This operator updates properties and labels on nodes and relationships.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{EdgeId, EntityId, Label, Value};

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{GraphMutator, UpdateEdgeRequest, UpdateNodeRequest};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{GraphSetAction, GraphSetNode, LogicalExpr};

/// Graph SET operator.
///
/// Updates properties and labels on matched nodes/relationships.
/// Supports:
/// - Setting a single property: SET n.prop = value
/// - Setting multiple properties: SET n.prop1 = val1, n.prop2 = val2
/// - Adding labels: SET n:Label
/// - Updating relationship properties: SET r.prop = value
pub struct GraphSetOp {
    /// Base operator state.
    base: OperatorBase,
    /// The SET node specification.
    set_node: GraphSetNode,
    /// Input operator (from MATCH clause).
    input: BoxedOperator,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
}

impl GraphSetOp {
    /// Creates a new graph SET operator.
    #[must_use]
    pub fn new(set_node: GraphSetNode, input: BoxedOperator) -> Self {
        // Build output schema from RETURN clause or input schema
        let schema = if set_node.returning.is_empty() {
            // If no RETURN clause, pass through the input schema
            input.schema()
        } else {
            // Use RETURN clause expressions as column names
            let columns: Vec<String> = set_node
                .returning
                .iter()
                .enumerate()
                .map(|(i, expr)| expr_to_column_name(expr, i))
                .collect();
            Arc::new(Schema::new(columns))
        };

        Self { base: OperatorBase::new(schema), set_node, input, graph_mutator: None }
    }

    /// Executes the SET operations for a single input row.
    fn execute_set(&self, input_row: &Row) -> OperatorResult<Row> {
        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Group SET actions by variable name
        let mut node_updates: HashMap<String, UpdateNodeRequest> = HashMap::new();
        let mut edge_updates: HashMap<String, UpdateEdgeRequest> = HashMap::new();

        for action in &self.set_node.set_actions {
            match action {
                GraphSetAction::Property { variable, property, value } => {
                    // Evaluate the value expression
                    let val = evaluate_expr(value, input_row)?;

                    // Resolve the entity/edge ID from the input row
                    let id_value = input_row.get_by_name(variable).ok_or_else(|| {
                        ParseError::InvalidGraphOp(format!(
                            "variable '{}' not found in row",
                            variable
                        ))
                    })?;

                    match id_value {
                        Value::Int(id) => {
                            // Could be either a node or an edge - we need to determine which
                            // For now, we'll try node first, and if that fails, try edge
                            let entity_id = EntityId::new(*id as u64);

                            // Check if we can find it as a node
                            if let Ok(Some(_)) = mutator.get_node(entity_id) {
                                // It's a node - add to node updates
                                let update = node_updates
                                    .entry(variable.clone())
                                    .or_insert_with(|| UpdateNodeRequest::new(entity_id));
                                update.set_properties.insert(property.clone(), val);
                            } else {
                                // Try as an edge
                                let edge_id = EdgeId::new(*id as u64);
                                if mutator
                                    .get_edge(edge_id)
                                    .map_err(|e| {
                                        ParseError::InvalidGraphOp(format!(
                                            "failed to check edge: {e}"
                                        ))
                                    })?
                                    .is_some()
                                {
                                    // It's an edge - add to edge updates
                                    let update = edge_updates
                                        .entry(variable.clone())
                                        .or_insert_with(|| UpdateEdgeRequest::new(edge_id));
                                    update.set_properties.insert(property.clone(), val);
                                } else {
                                    return Err(ParseError::InvalidGraphOp(format!(
                                        "entity with ID {} not found",
                                        id
                                    )));
                                }
                            }
                        }
                        _ => {
                            return Err(ParseError::InvalidGraphOp(format!(
                                "variable '{}' is not an entity ID",
                                variable
                            )));
                        }
                    }
                }
                GraphSetAction::Label { variable, label } => {
                    // Labels can only be added to nodes
                    let id_value = input_row.get_by_name(variable).ok_or_else(|| {
                        ParseError::InvalidGraphOp(format!(
                            "variable '{}' not found in row",
                            variable
                        ))
                    })?;

                    match id_value {
                        Value::Int(id) => {
                            let entity_id = EntityId::new(*id as u64);
                            let update = node_updates
                                .entry(variable.clone())
                                .or_insert_with(|| UpdateNodeRequest::new(entity_id));
                            update.add_labels.push(Label::new(label));
                        }
                        _ => {
                            return Err(ParseError::InvalidGraphOp(format!(
                                "variable '{}' is not an entity ID",
                                variable
                            )));
                        }
                    }
                }
            }
        }

        // Execute node updates
        for (_var, update) in node_updates {
            mutator
                .update_node(&update)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to update node: {e}")))?;
        }

        // Execute edge updates
        for (_var, update) in edge_updates {
            mutator
                .update_edge(&update)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to update edge: {e}")))?;
        }

        // Build output row
        self.build_output_row(input_row)
    }

    /// Builds the output row.
    fn build_output_row(&self, input_row: &Row) -> OperatorResult<Row> {
        let schema = self.base.schema();

        if self.set_node.returning.is_empty() {
            // Pass through the input row
            Ok(Row::new(schema, input_row.values().to_vec()))
        } else {
            // Evaluate RETURN expressions
            let values: Vec<Value> = self
                .set_node
                .returning
                .iter()
                .map(|expr| evaluate_expr(expr, input_row))
                .collect::<OperatorResult<Vec<_>>>()?;

            Ok(Row::new(schema, values))
        }
    }
}

impl Operator for GraphSetOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        match self.input.next()? {
            Some(input_row) => {
                let row = self.execute_set(&input_row)?;
                self.base.inc_rows_produced();
                Ok(Some(row))
            }
            None => {
                self.base.set_finished();
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
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
        "GraphSet"
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
    fn graph_set_schema_passthrough() {
        use crate::exec::operators::values::EmptyOp;

        let set_node = GraphSetNode::new(vec![]);
        let input = Box::new(EmptyOp::with_columns(vec!["n".to_string(), "m".to_string()]));
        let op = GraphSetOp::new(set_node, input);

        assert_eq!(op.schema().columns(), &["n", "m"]);
    }

    #[test]
    fn graph_set_schema_from_return() {
        use crate::exec::operators::values::EmptyOp;

        let set_node = GraphSetNode::new(vec![]).with_returning(vec![
            LogicalExpr::column("n"),
            LogicalExpr::Alias {
                expr: Box::new(LogicalExpr::column("m")),
                alias: "node".to_string(),
            },
        ]);
        let input = Box::new(EmptyOp::with_columns(vec!["n".to_string(), "m".to_string()]));
        let op = GraphSetOp::new(set_node, input);

        assert_eq!(op.schema().columns(), &["n", "node"]);
    }

    #[test]
    fn graph_set_requires_storage() {
        use crate::exec::operators::values::ValuesOp;

        let set_node = GraphSetNode::new(vec![GraphSetAction::Property {
            variable: "n".to_string(),
            property: "name".to_string(),
            value: LogicalExpr::string("Alice"),
        }]);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(schema, vec![vec![Value::Int(1)]]));

        let mut op = GraphSetOp::new(set_node, input);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on next() since NullGraphMutator returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("no graph storage"));

        op.close().unwrap();
    }
}
