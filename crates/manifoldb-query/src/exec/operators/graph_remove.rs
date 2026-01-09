//! Graph REMOVE operator for Cypher REMOVE statements.
//!
//! This operator removes properties and labels from nodes and relationships.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::GraphMutator;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{GraphRemoveAction, GraphRemoveNode, LogicalExpr};

/// Graph REMOVE operator.
///
/// Removes properties and/or labels from nodes/relationships based on the logical plan.
/// Supports:
/// - Removing a single property: `REMOVE n.age`
/// - Removing multiple properties: `REMOVE n.age, n.city`
/// - Removing labels: `REMOVE n:Employee`
/// - Combinations of the above
///
/// The entity must already be bound from a MATCH clause or previous CREATE.
pub struct GraphRemoveOp {
    /// Base operator state.
    base: OperatorBase,
    /// The REMOVE node specification.
    remove_node: GraphRemoveNode,
    /// Input operator (from MATCH clause).
    input: BoxedOperator,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
}

impl GraphRemoveOp {
    /// Creates a new graph REMOVE operator.
    #[must_use]
    pub fn new(remove_node: GraphRemoveNode, input: BoxedOperator) -> Self {
        // Build output schema from RETURN clause or pass through input schema
        let schema = if remove_node.returning.is_empty() {
            // Pass through input schema
            input.schema()
        } else {
            // Use RETURN clause expressions as column names
            let columns: Vec<String> = remove_node
                .returning
                .iter()
                .enumerate()
                .map(|(i, expr)| expr_to_column_name(expr, i))
                .collect();
            Arc::new(Schema::new(columns))
        };

        Self { base: OperatorBase::new(schema), remove_node, input, graph_mutator: None }
    }

    /// Executes the REMOVE operation for a single input row.
    fn execute_remove(&self, input_row: &Row) -> OperatorResult<Row> {
        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Process each REMOVE action
        for action in &self.remove_node.remove_actions {
            match action {
                GraphRemoveAction::Property { variable, property } => {
                    // Resolve the entity ID from the variable
                    let entity_id = self.resolve_entity_id(variable, input_row)?;

                    // Check if this is an edge or entity
                    // For now, we assume all REMOVE property targets are entities
                    // TODO: Support edge property removal (need to track which variables are edges)
                    mutator.remove_entity_property(entity_id, property).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to remove property: {e}"))
                    })?;
                }
                GraphRemoveAction::Label { variable, label } => {
                    // Resolve the entity ID from the variable
                    let entity_id = self.resolve_entity_id(variable, input_row)?;

                    mutator.remove_entity_label(entity_id, label).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to remove label: {e}"))
                    })?;
                }
            }
        }

        // Build the output row
        self.build_output_row(input_row)
    }

    /// Resolves an entity ID from a variable name in the input row.
    fn resolve_entity_id(&self, var_name: &str, input_row: &Row) -> OperatorResult<EntityId> {
        if let Some(value) = input_row.get_by_name(var_name) {
            match value {
                Value::Int(id) => return Ok(EntityId::new(*id as u64)),
                _ => {
                    return Err(ParseError::InvalidGraphOp(format!(
                        "variable '{var_name}' is not an entity ID"
                    )));
                }
            }
        }

        Err(ParseError::InvalidGraphOp(format!("unbound variable: {var_name}")))
    }

    /// Builds the output row.
    fn build_output_row(&self, input_row: &Row) -> OperatorResult<Row> {
        let schema = self.base.schema();

        if self.remove_node.returning.is_empty() {
            // Pass through the input row (but with possibly modified values)
            // The entity has been modified in storage, but we return the original row values
            Ok(input_row.clone())
        } else {
            // Evaluate RETURN expressions
            let values: Vec<Value> = self
                .remove_node
                .returning
                .iter()
                .map(|expr| evaluate_expr(expr, input_row))
                .collect::<OperatorResult<Vec<_>>>()?;

            Ok(Row::new(schema, values))
        }
    }
}

impl Operator for GraphRemoveOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open input operator
        self.input.open(ctx)?;

        // Capture the graph mutator from the context
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // For each input row, execute the REMOVE operations
        match self.input.next()? {
            Some(input_row) => {
                let row = self.execute_remove(&input_row)?;
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
        "GraphRemove"
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
    use crate::exec::operators::values::ValuesOp;

    #[test]
    fn graph_remove_schema_passthrough() {
        // Test that schema passes through when no RETURN clause
        let input_schema = Arc::new(Schema::new(vec!["n".to_string(), "m".to_string()]));
        let input = Box::new(ValuesOp::new(input_schema.clone(), vec![]));

        let remove_node = GraphRemoveNode::new(vec![]);
        let op = GraphRemoveOp::new(remove_node, input);

        assert_eq!(op.schema().columns(), &["n", "m"]);
    }

    #[test]
    fn graph_remove_requires_storage() {
        let input_schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(input_schema, vec![vec![Value::Int(1)]]));

        let remove_node = GraphRemoveNode::new(vec![GraphRemoveAction::Property {
            variable: "n".to_string(),
            property: "age".to_string(),
        }]);

        let mut op = GraphRemoveOp::new(remove_node, input);

        // ExecutionContext::new() creates context with NullGraphMutator
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on next() since NullGraphMutator returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("failed to remove property"));

        op.close().unwrap();
    }

    #[test]
    fn graph_remove_with_return_clause() {
        let input_schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(input_schema, vec![]));

        let remove_node =
            GraphRemoveNode::new(vec![]).with_returning(vec![LogicalExpr::column("n")]);

        let op = GraphRemoveOp::new(remove_node, input);

        assert_eq!(op.schema().columns(), &["n"]);
    }
}
