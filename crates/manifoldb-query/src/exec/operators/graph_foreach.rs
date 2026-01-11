//! Graph FOREACH operator for Cypher FOREACH statements.
//!
//! This operator iterates over a list expression and executes writing
//! clauses (SET, CREATE, DELETE, REMOVE, MERGE) for each element.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{EdgeId, Entity, EntityId, Label, Value};

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::{
    CreateEdgeRequest, CreateNodeRequest, DeleteResult, GraphAccessor, GraphMutator,
    UpdateNodeRequest,
};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{
    GraphCreateNode, GraphDeleteNode, GraphForeachAction, GraphForeachNode, GraphMergeNode,
    GraphRemoveAction, GraphSetAction, LogicalExpr, MergePatternSpec,
};

/// Graph FOREACH operator.
///
/// Iterates over a list expression and executes writing clauses for each element.
/// This is a side-effect only operator - it passes through input rows unchanged.
///
/// Supports:
/// - FOREACH (x IN list | SET ...)
/// - FOREACH (x IN list | CREATE ...)
/// - FOREACH (x IN list | DELETE ...)
/// - FOREACH (x IN list | REMOVE ...)
/// - FOREACH (x IN list | MERGE ...)
/// - Nested FOREACH
pub struct GraphForeachOp {
    /// Base operator state.
    base: OperatorBase,
    /// The FOREACH node specification.
    foreach_node: GraphForeachNode,
    /// Input operator (from MATCH clause).
    input: BoxedOperator,
    /// Graph accessor for read operations (used by MERGE to find existing nodes).
    graph_accessor: Option<Arc<dyn GraphAccessor>>,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
}

impl GraphForeachOp {
    /// Creates a new graph FOREACH operator.
    #[must_use]
    pub fn new(foreach_node: GraphForeachNode, input: BoxedOperator) -> Self {
        // FOREACH passes through the input schema - it doesn't modify the output columns
        let schema = input.schema();

        Self {
            base: OperatorBase::new(schema),
            foreach_node,
            input,
            graph_accessor: None,
            graph_mutator: None,
        }
    }

    /// Executes the FOREACH operation for a single input row.
    fn execute_foreach(&self, input_row: &Row) -> OperatorResult<()> {
        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Evaluate the list expression
        let list_value = evaluate_expr(&self.foreach_node.list_expr, input_row)?;

        // Extract the list elements
        let elements = match list_value {
            Value::Array(elements) => elements,
            Value::Null => {
                // FOREACH over NULL is a no-op
                return Ok(());
            }
            other => {
                return Err(ParseError::InvalidGraphOp(format!(
                    "FOREACH requires a list, got {:?}",
                    other
                )));
            }
        };

        // For each element in the list, execute the nested actions
        for element in elements {
            // Create a new row with the iteration variable bound
            let iteration_row = self.bind_variable(input_row, &self.foreach_node.variable, element);

            // Execute each action
            for action in &self.foreach_node.actions {
                self.execute_action(action, &iteration_row, mutator.as_ref())?;
            }
        }

        Ok(())
    }

    /// Binds the iteration variable to a value, creating a new row.
    fn bind_variable(&self, input_row: &Row, variable: &str, value: Value) -> Row {
        let mut columns: Vec<String> =
            input_row.schema().columns().iter().map(|s| (*s).to_string()).collect();
        let mut values: Vec<Value> = input_row.values().to_vec();

        // Add or replace the iteration variable
        if let Some(pos) = columns.iter().position(|c| c == variable) {
            values[pos] = value;
        } else {
            columns.push(variable.to_string());
            values.push(value);
        }

        let schema = Arc::new(Schema::new(columns));
        Row::new(schema, values)
    }

    /// Executes a single FOREACH action.
    fn execute_action(
        &self,
        action: &GraphForeachAction,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        match action {
            GraphForeachAction::Set(set_action) => self.execute_set(set_action, row, mutator),
            GraphForeachAction::Create(create_node) => {
                self.execute_create(create_node, row, mutator)
            }
            GraphForeachAction::Merge(merge_node) => self.execute_merge(merge_node, row, mutator),
            GraphForeachAction::Delete(delete_node) => {
                self.execute_delete(delete_node, row, mutator)
            }
            GraphForeachAction::Remove(remove_action) => {
                self.execute_remove(remove_action, row, mutator)
            }
            GraphForeachAction::Foreach(nested_foreach) => {
                self.execute_nested_foreach(nested_foreach, row, mutator)
            }
        }
    }

    /// Executes a SET action.
    fn execute_set(
        &self,
        action: &GraphSetAction,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        match action {
            GraphSetAction::Property { variable, property, value } => {
                // Evaluate the value expression
                let val = evaluate_expr(value, row)?;

                // Resolve the entity ID from the row
                let id_value = row.get_by_name(variable).ok_or_else(|| {
                    ParseError::InvalidGraphOp(format!("variable '{}' not found in row", variable))
                })?;

                match id_value {
                    Value::Node { id, .. } => {
                        let entity_id = EntityId::new(*id as u64);
                        let mut update = UpdateNodeRequest::new(entity_id);
                        update.set_properties.insert(property.clone(), val);
                        mutator.update_node(&update).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                        })?;
                    }
                    Value::Edge { id, .. } => {
                        let edge_id = EdgeId::new(*id as u64);
                        let mut update =
                            crate::exec::graph_accessor::UpdateEdgeRequest::new(edge_id);
                        update.set_properties.insert(property.clone(), val);
                        mutator.update_edge(&update).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to update edge: {e}"))
                        })?;
                    }
                    Value::Int(id) => {
                        let entity_id = EntityId::new(*id as u64);

                        // Check if it's a node
                        if let Ok(Some(_)) = mutator.get_node(entity_id) {
                            let mut update = UpdateNodeRequest::new(entity_id);
                            update.set_properties.insert(property.clone(), val);
                            mutator.update_node(&update).map_err(|e| {
                                ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                            })?;
                        } else {
                            // Try as an edge
                            let edge_id = EdgeId::new(*id as u64);
                            if mutator
                                .get_edge(edge_id)
                                .map_err(|e| {
                                    ParseError::InvalidGraphOp(format!("failed to check edge: {e}"))
                                })?
                                .is_some()
                            {
                                let mut update =
                                    crate::exec::graph_accessor::UpdateEdgeRequest::new(edge_id);
                                update.set_properties.insert(property.clone(), val);
                                mutator.update_edge(&update).map_err(|e| {
                                    ParseError::InvalidGraphOp(format!(
                                        "failed to update edge: {e}"
                                    ))
                                })?;
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
                let id_value = row.get_by_name(variable).ok_or_else(|| {
                    ParseError::InvalidGraphOp(format!("variable '{}' not found in row", variable))
                })?;

                match id_value {
                    Value::Node { id, .. } => {
                        let entity_id = EntityId::new(*id as u64);
                        let mut update = UpdateNodeRequest::new(entity_id);
                        update.add_labels.push(Label::new(label));
                        mutator.update_node(&update).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                        })?;
                    }
                    Value::Int(id) => {
                        let entity_id = EntityId::new(*id as u64);
                        let mut update = UpdateNodeRequest::new(entity_id);
                        update.add_labels.push(Label::new(label));
                        mutator.update_node(&update).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                        })?;
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
        Ok(())
    }

    /// Executes a CREATE action.
    fn execute_create(
        &self,
        create_node: &GraphCreateNode,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        let mut created_entities: HashMap<String, EntityId> = HashMap::new();

        // Create nodes first
        for node_spec in &create_node.nodes {
            let mut request = CreateNodeRequest::new();

            // Add labels
            for label in &node_spec.labels {
                request = request.with_label(Label::new(label));
            }

            // Evaluate and add properties
            for (key, expr) in &node_spec.properties {
                let value = evaluate_expr(expr, row)?;
                request = request.with_property(key.clone(), value);
            }

            // Create the node
            let entity = mutator
                .create_node(&request)
                .map_err(|e| ParseError::InvalidGraphOp(format!("failed to create node: {e}")))?;

            // Store if there's a variable
            if let Some(ref var) = node_spec.variable {
                created_entities.insert(var.clone(), entity.id);
            }
        }

        // Then create relationships
        for rel_spec in &create_node.relationships {
            // Resolve source and target
            let source_id = resolve_entity_id(&rel_spec.start_var, row, &created_entities)?;
            let target_id = resolve_entity_id(&rel_spec.end_var, row, &created_entities)?;

            let mut request =
                CreateEdgeRequest::new(source_id, target_id, rel_spec.rel_type.clone());

            // Evaluate and add properties
            for (key, expr) in &rel_spec.properties {
                let value = evaluate_expr(expr, row)?;
                request = request.with_property(key.clone(), value);
            }

            mutator.create_edge(&request).map_err(|e| {
                ParseError::InvalidGraphOp(format!("failed to create relationship: {e}"))
            })?;
        }

        Ok(())
    }

    /// Executes a MERGE action.
    fn execute_merge(
        &self,
        merge_node: &GraphMergeNode,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        match &merge_node.pattern {
            MergePatternSpec::Node { variable, labels, match_properties } => {
                // Try to find an existing node matching the pattern
                let existing = self.find_matching_node(labels, match_properties, row)?;

                let entity = if let Some(entity) = existing {
                    // Execute ON MATCH actions
                    self.execute_on_match_actions(&merge_node.on_match, &entity, row, mutator)?;
                    entity
                } else {
                    // Create new node
                    let mut request = CreateNodeRequest::new();
                    for label in labels {
                        request = request.with_label(Label::new(label));
                    }
                    for (key, expr) in match_properties {
                        let value = evaluate_expr(expr, row)?;
                        request = request.with_property(key.clone(), value);
                    }
                    let entity = mutator.create_node(&request).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to create node: {e}"))
                    })?;

                    // Execute ON CREATE actions
                    self.execute_on_create_actions(&merge_node.on_create, &entity, row, mutator)?;
                    entity
                };

                // Note: We don't bind the variable here since FOREACH doesn't produce output rows
                let _ = (variable, entity);
            }
            MergePatternSpec::Relationship {
                start_var,
                rel_type,
                end_var,
                match_properties,
                ..
            } => {
                // Resolve start and end nodes
                let start_id = resolve_entity_id_simple(start_var, row)?;
                let end_id = resolve_entity_id_simple(end_var, row)?;

                // Try to find an existing relationship
                // For now, we always create - proper MERGE semantics would check for existing
                let mut request = CreateEdgeRequest::new(start_id, end_id, rel_type.clone());
                for (key, expr) in match_properties {
                    let value = evaluate_expr(expr, row)?;
                    request = request.with_property(key.clone(), value);
                }
                mutator.create_edge(&request).map_err(|e| {
                    ParseError::InvalidGraphOp(format!("failed to create relationship: {e}"))
                })?;
            }
        }
        Ok(())
    }

    /// Finds a node matching the given labels and properties.
    fn find_matching_node(
        &self,
        labels: &[String],
        properties: &[(String, LogicalExpr)],
        row: &Row,
    ) -> OperatorResult<Option<Entity>> {
        let accessor = self
            .graph_accessor
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        // Evaluate properties to match
        let mut match_props: HashMap<String, Value> = HashMap::new();
        for (key, expr) in properties {
            let value = evaluate_expr(expr, row)?;
            match_props.insert(key.clone(), value);
        }

        // Get primary label for the scan (use first label if available)
        let primary_label = labels.first().map(String::as_str);

        // Scan nodes by primary label
        let nodes = accessor
            .scan_nodes(primary_label)
            .map_err(|e| ParseError::InvalidGraphOp(format!("failed to scan nodes: {e}")))?;

        // Find a node that matches all labels and all properties
        for node in nodes {
            // Check all labels are present
            if !labels.iter().all(|l| node.labels.contains(l)) {
                continue;
            }

            // Check all properties match
            if match_props.iter().all(|(key, val)| node.properties.get(key) == Some(val)) {
                // Found a matching node - convert NodeScanResult to Entity
                let entity = Entity {
                    id: node.id,
                    labels: node.labels.into_iter().map(Label::new).collect(),
                    properties: node.properties,
                };
                return Ok(Some(entity));
            }
        }

        // No match found
        Ok(None)
    }

    /// Executes ON MATCH actions.
    fn execute_on_match_actions(
        &self,
        actions: &[GraphSetAction],
        entity: &Entity,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        for action in actions {
            match action {
                GraphSetAction::Property { property, value, .. } => {
                    let val = evaluate_expr(value, row)?;
                    let mut update = UpdateNodeRequest::new(entity.id);
                    update.set_properties.insert(property.clone(), val);
                    mutator.update_node(&update).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                    })?;
                }
                GraphSetAction::Label { label, .. } => {
                    let mut update = UpdateNodeRequest::new(entity.id);
                    update.add_labels.push(Label::new(label));
                    mutator.update_node(&update).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to update node: {e}"))
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Executes ON CREATE actions.
    fn execute_on_create_actions(
        &self,
        actions: &[GraphSetAction],
        entity: &Entity,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        // Same as on_match for now
        self.execute_on_match_actions(actions, entity, row, mutator)
    }

    /// Executes a DELETE action.
    fn execute_delete(
        &self,
        delete_node: &GraphDeleteNode,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        for variable in &delete_node.variables {
            let id_value = row.get_by_name(variable).ok_or_else(|| {
                ParseError::InvalidGraphOp(format!("variable '{}' not found in row", variable))
            })?;

            match id_value {
                Value::Node { id, .. } | Value::Int(id) => {
                    let entity_id = EntityId::new(*id as u64);

                    let result: DeleteResult = if delete_node.detach {
                        mutator.delete_node_detach(entity_id)
                    } else {
                        mutator.delete_node(entity_id)
                    }
                    .map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to delete node: {e}"))
                    })?;

                    // Ignore result - delete is best effort in FOREACH
                    let _ = result;
                }
                Value::Edge { id, .. } => {
                    let edge_id = EdgeId::new(*id as u64);
                    mutator.delete_edge(edge_id).map_err(|e| {
                        ParseError::InvalidGraphOp(format!("failed to delete edge: {e}"))
                    })?;
                }
                _ => {
                    return Err(ParseError::InvalidGraphOp(format!(
                        "variable '{}' is not an entity ID",
                        variable
                    )));
                }
            }
        }
        Ok(())
    }

    /// Executes a REMOVE action.
    fn execute_remove(
        &self,
        action: &GraphRemoveAction,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        match action {
            GraphRemoveAction::Property { variable, property } => {
                let id_value = row.get_by_name(variable).ok_or_else(|| {
                    ParseError::InvalidGraphOp(format!("variable '{}' not found in row", variable))
                })?;

                match id_value {
                    Value::Node { id, .. } | Value::Int(id) => {
                        let entity_id = EntityId::new(*id as u64);
                        mutator.remove_entity_property(entity_id, property).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to remove property: {e}"))
                        })?;
                    }
                    Value::Edge { id, .. } => {
                        let edge_id = EdgeId::new(*id as u64);
                        mutator.remove_edge_property(edge_id, property).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to remove property: {e}"))
                        })?;
                    }
                    _ => {
                        return Err(ParseError::InvalidGraphOp(format!(
                            "variable '{}' is not an entity ID",
                            variable
                        )));
                    }
                }
            }
            GraphRemoveAction::Label { variable, label } => {
                let id_value = row.get_by_name(variable).ok_or_else(|| {
                    ParseError::InvalidGraphOp(format!("variable '{}' not found in row", variable))
                })?;

                match id_value {
                    Value::Node { id, .. } | Value::Int(id) => {
                        let entity_id = EntityId::new(*id as u64);
                        mutator.remove_entity_label(entity_id, label).map_err(|e| {
                            ParseError::InvalidGraphOp(format!("failed to remove label: {e}"))
                        })?;
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
        Ok(())
    }

    /// Executes a nested FOREACH.
    fn execute_nested_foreach(
        &self,
        nested: &GraphForeachNode,
        row: &Row,
        mutator: &dyn GraphMutator,
    ) -> OperatorResult<()> {
        // Evaluate the nested list expression
        let list_value = evaluate_expr(&nested.list_expr, row)?;

        let elements = match list_value {
            Value::Array(elements) => elements,
            Value::Null => return Ok(()), // FOREACH over NULL is a no-op
            other => {
                return Err(ParseError::InvalidGraphOp(format!(
                    "FOREACH requires a list, got {:?}",
                    other
                )));
            }
        };

        // For each element, execute the nested actions
        for element in elements {
            let iteration_row = self.bind_variable(row, &nested.variable, element);

            for action in &nested.actions {
                self.execute_action(action, &iteration_row, mutator)?;
            }
        }

        Ok(())
    }
}

impl Operator for GraphForeachOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.graph_accessor = Some(ctx.graph_arc());
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        match self.input.next()? {
            Some(input_row) => {
                // Execute FOREACH side effects
                self.execute_foreach(&input_row)?;

                // Pass through the input row unchanged
                self.base.inc_rows_produced();
                Ok(Some(input_row))
            }
            None => {
                self.base.set_finished();
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
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
        "GraphForeach"
    }
}

/// Resolves an entity ID from a variable name.
fn resolve_entity_id(
    var_name: &str,
    row: &Row,
    created_entities: &HashMap<String, EntityId>,
) -> OperatorResult<EntityId> {
    // Check created entities first
    if let Some(&id) = created_entities.get(var_name) {
        return Ok(id);
    }

    // Then check the row
    resolve_entity_id_simple(var_name, row)
}

/// Resolves an entity ID from a variable in a row.
fn resolve_entity_id_simple(var_name: &str, row: &Row) -> OperatorResult<EntityId> {
    let value = row
        .get_by_name(var_name)
        .ok_or_else(|| ParseError::InvalidGraphOp(format!("unbound variable: {var_name}")))?;

    match value {
        Value::Node { id, .. } => Ok(EntityId::new(*id as u64)),
        Value::Edge { id, .. } => Ok(EntityId::new(*id as u64)),
        Value::Int(id) => Ok(EntityId::new(*id as u64)),
        _ => Err(ParseError::InvalidGraphOp(format!("variable '{var_name}' is not an entity ID"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::EmptyOp;

    #[test]
    fn graph_foreach_schema_passthrough() {
        let foreach_node = GraphForeachNode::new("x", LogicalExpr::ListLiteral(vec![]), vec![]);

        let input = Box::new(EmptyOp::with_columns(vec!["n".to_string(), "m".to_string()]));
        let op = GraphForeachOp::new(foreach_node, input);

        // FOREACH passes through the input schema
        assert_eq!(op.schema().columns(), &["n", "m"]);
    }

    #[test]
    fn graph_foreach_empty_list() {
        use crate::exec::operators::values::ValuesOp;

        let foreach_node = GraphForeachNode::new(
            "x",
            LogicalExpr::ListLiteral(vec![]),
            vec![GraphForeachAction::Set(GraphSetAction::Property {
                variable: "n".to_string(),
                property: "touched".to_string(),
                value: LogicalExpr::boolean(true),
            })],
        );

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(schema, vec![vec![Value::Int(1)]]));

        let mut op = GraphForeachOp::new(foreach_node, input);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // With empty list, FOREACH is a no-op and passes through rows
        let result = op.next();
        // The result depends on whether the set action succeeds (which requires storage)
        // Since we use NullGraphMutator, this should succeed because empty list means no actions
        assert!(result.is_ok());

        op.close().unwrap();
    }

    #[test]
    fn graph_foreach_requires_storage() {
        use crate::exec::operators::values::ValuesOp;

        let foreach_node = GraphForeachNode::new(
            "x",
            LogicalExpr::ListLiteral(vec![LogicalExpr::integer(1)]),
            vec![GraphForeachAction::Set(GraphSetAction::Property {
                variable: "x".to_string(),
                property: "touched".to_string(),
                value: LogicalExpr::boolean(true),
            })],
        );

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(schema, vec![vec![Value::Int(1)]]));

        let mut op = GraphForeachOp::new(foreach_node, input);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should fail because NullGraphMutator returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("no graph storage"));

        op.close().unwrap();
    }

    #[test]
    fn graph_foreach_null_list() {
        use crate::exec::operators::values::ValuesOp;

        let foreach_node = GraphForeachNode::new(
            "x",
            LogicalExpr::null(),
            vec![GraphForeachAction::Set(GraphSetAction::Property {
                variable: "x".to_string(),
                property: "touched".to_string(),
                value: LogicalExpr::boolean(true),
            })],
        );

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let input = Box::new(ValuesOp::new(schema, vec![vec![Value::Int(1)]]));

        let mut op = GraphForeachOp::new(foreach_node, input);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // FOREACH over NULL is a no-op - should pass through successfully
        let result = op.next();
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());

        op.close().unwrap();
    }
}
