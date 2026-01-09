//! Graph DELETE operator for Cypher DELETE statements.
//!
//! This operator deletes nodes and relationships from the graph.

use std::collections::HashSet;
use std::sync::Arc;

use manifoldb_core::{EdgeId, EntityId, Value};

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::graph_accessor::GraphMutator;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::GraphDeleteNode;

/// Graph DELETE operator.
///
/// Deletes nodes and/or relationships from the graph based on the logical plan.
/// Supports:
/// - DELETE node (fails if node has relationships)
/// - DELETE relationship
/// - DETACH DELETE (deletes node and all its relationships)
pub struct GraphDeleteOp {
    /// Base operator state.
    base: OperatorBase,
    /// The DELETE node specification.
    delete_node: GraphDeleteNode,
    /// Input operator (from MATCH clause).
    input: BoxedOperator,
    /// Graph mutator for write operations.
    graph_mutator: Option<Arc<dyn GraphMutator>>,
    /// Track which entities we've already deleted (to avoid double-deletion).
    deleted_nodes: HashSet<EntityId>,
    /// Track which edges we've already deleted.
    deleted_edges: HashSet<EdgeId>,
    /// Count of nodes deleted.
    nodes_deleted_count: usize,
    /// Count of edges deleted.
    edges_deleted_count: usize,
    /// Whether we've finished processing all input rows.
    finished: bool,
}

impl GraphDeleteOp {
    /// Creates a new graph DELETE operator.
    #[must_use]
    pub fn new(delete_node: GraphDeleteNode, input: BoxedOperator) -> Self {
        // DELETE always returns an empty result set (no columns)
        // unless there's a RETURN clause
        let schema = Arc::new(Schema::empty());

        Self {
            base: OperatorBase::new(schema),
            delete_node,
            input,
            graph_mutator: None,
            deleted_nodes: HashSet::new(),
            deleted_edges: HashSet::new(),
            nodes_deleted_count: 0,
            edges_deleted_count: 0,
            finished: false,
        }
    }

    /// Delete an edge by ID.
    fn delete_edge(&mut self, id: EdgeId) -> OperatorResult<()> {
        // Skip if already deleted
        if self.deleted_edges.contains(&id) {
            return Ok(());
        }

        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        let deleted = mutator
            .delete_edge(id)
            .map_err(|e| ParseError::InvalidGraphOp(format!("failed to delete edge: {e}")))?;

        if deleted {
            self.deleted_edges.insert(id);
            self.edges_deleted_count += 1;
        }

        Ok(())
    }

    /// Process a single input row and delete all specified variables.
    fn process_row(&mut self, row: &Row) -> OperatorResult<()> {
        // Clone the variables to avoid borrow issues
        let variables = self.delete_node.variables.clone();

        for var_name in &variables {
            // Look up the variable in the row
            if let Some(value) = row.get_by_name(var_name) {
                match value {
                    Value::Int(id) => {
                        // Could be either a node ID or an edge ID
                        // We need to determine which based on the context
                        let entity_id = EntityId::new(*id as u64);
                        let edge_id = EdgeId::new(*id as u64);

                        // First, try to delete as a node
                        match self.try_delete_as_node(entity_id) {
                            Ok(deleted) if deleted => {
                                // Node was successfully deleted
                            }
                            Ok(_not_deleted) => {
                                // Node wasn't deleted (not found), try as edge
                                // Ignore edge errors - it might not exist either
                                let _ = self.delete_edge(edge_id);
                            }
                            Err(e) => {
                                // Node deletion failed with an error (e.g., has relationships)
                                // Propagate the error
                                return Err(e);
                            }
                        }
                    }
                    Value::Null => {
                        // Skip null values - nothing to delete
                    }
                    other => {
                        return Err(ParseError::InvalidGraphOp(format!(
                            "variable '{var_name}' has invalid type for DELETE: expected Int, got {other:?}"
                        )));
                    }
                }
            }
            // Variable not found in row - this could be an edge variable
            // We'll skip it since it might have been deleted with a node
        }

        Ok(())
    }

    /// Try to delete a value as a node, returning Ok(true) if deleted, Ok(false) if not found.
    fn try_delete_as_node(&mut self, id: EntityId) -> OperatorResult<bool> {
        // Skip if already deleted
        if self.deleted_nodes.contains(&id) {
            return Ok(true);
        }

        let mutator = self
            .graph_mutator
            .as_ref()
            .ok_or_else(|| ParseError::InvalidGraphOp("no graph storage available".to_string()))?;

        let result = if self.delete_node.detach {
            mutator.delete_node_detach(id)
        } else {
            mutator.delete_node(id)
        };

        match result {
            Ok(delete_result) => {
                if delete_result.nodes_deleted > 0 {
                    self.deleted_nodes.insert(id);
                    self.nodes_deleted_count += delete_result.nodes_deleted;
                    self.edges_deleted_count += delete_result.edges_deleted;
                    Ok(true)
                } else {
                    // Node wasn't found
                    Ok(false)
                }
            }
            Err(e) => Err(ParseError::InvalidGraphOp(format!("failed to delete node: {e}"))),
        }
    }
}

impl Operator for GraphDeleteOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open input operator
        self.input.open(ctx)?;

        // Capture the graph mutator from the context
        self.graph_mutator = Some(ctx.graph_mutator_arc());
        self.deleted_nodes.clear();
        self.deleted_edges.clear();
        self.nodes_deleted_count = 0;
        self.edges_deleted_count = 0;
        self.finished = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.finished {
            self.base.set_finished();
            return Ok(None);
        }

        // Process all input rows and delete entities
        loop {
            match self.input.next()? {
                Some(row) => {
                    self.process_row(&row)?;
                }
                None => {
                    // All input rows processed
                    self.finished = true;
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.graph_mutator = None;
        self.deleted_nodes.clear();
        self.deleted_edges.clear();
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
        "GraphDelete"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    #[test]
    fn graph_delete_empty_schema() {
        let delete_node = GraphDeleteNode::new(vec!["n".to_string()]);
        let input = Box::new(ValuesOp::new(
            Arc::new(Schema::new(vec!["n".to_string()])),
            vec![vec![Value::Int(1)]],
        ));
        let op = GraphDeleteOp::new(delete_node, input);

        // DELETE returns empty schema
        assert!(op.schema().columns().is_empty());
    }

    #[test]
    fn graph_delete_detach_flag() {
        let delete_node = GraphDeleteNode::detach(vec!["n".to_string()]);
        assert!(delete_node.detach);

        let regular_delete = GraphDeleteNode::new(vec!["n".to_string()]);
        assert!(!regular_delete.detach);
    }

    #[test]
    fn graph_delete_requires_storage() {
        let delete_node = GraphDeleteNode::new(vec!["n".to_string()]);
        let input = Box::new(ValuesOp::new(
            Arc::new(Schema::new(vec!["n".to_string()])),
            vec![vec![Value::Int(1)]],
        ));
        let mut op = GraphDeleteOp::new(delete_node, input);

        // ExecutionContext::new() creates context with NullGraphMutator
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should return an error on next() since NullGraphMutator returns NoStorage
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("failed to delete"));

        op.close().unwrap();
    }
}
