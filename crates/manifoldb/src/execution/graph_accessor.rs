//! Graph accessor implementation for query execution.
//!
//! This module provides utilities for executing graph traversals using
//! the `DatabaseTransaction` edge traversal methods directly.
//!
//! This approach uses the `DatabaseTransaction`'s own edge storage and indexing,
//! ensuring compatibility with the edge key format used by `put_edge`/`get_edge`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use manifoldb_core::{Edge, EdgeId, EdgeType, Entity, EntityId, Value};
use manifoldb_query::exec::operators::filter::evaluate_expr;
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::ResultSet;
use manifoldb_query::plan::logical::{ExpandDirection, ExpandLength, ExpandNode, LogicalExpr};
use manifoldb_storage::Transaction;

use crate::error::{Error, Result};
use crate::transaction::DatabaseTransaction;

/// Execute a single expand operation and return the results as a ResultSet.
///
/// This function executes a graph expansion from source nodes to destination nodes,
/// following edges according to the expand configuration.
///
/// This implementation uses `DatabaseTransaction`'s edge traversal methods directly,
/// which ensures compatibility with the edge storage key format.
pub fn execute_expand_operation<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    expand: &ExpandNode,
    source_nodes: Vec<(EntityId, Row)>,
) -> Result<ResultSet> {
    // Convert edge types to EdgeType
    let edge_types: Vec<EdgeType> = expand.edge_types.iter().map(|s| EdgeType::new(s)).collect();

    // Build the output schema - includes source variable + destination variable + optional edge
    let mut output_columns = Vec::new();
    output_columns.push(expand.src_var.clone());
    output_columns.push(expand.dst_var.clone());
    if let Some(ref edge_var) = expand.edge_var {
        output_columns.push(edge_var.clone());
    }

    let schema = Arc::new(Schema::new(output_columns.clone()));
    let mut result_rows = Vec::new();

    // Process each source node
    for (src_id, src_row) in source_nodes {
        // Get source entity for building Value::Node
        // First try to get it from the source row, then fall back to lookup
        let src_node_value = src_row
            .get_by_name(&expand.src_var)
            .filter(|v| matches!(v, Value::Node { .. }))
            .cloned()
            .or_else(|| {
                tx.get_entity(src_id).ok().flatten().map(|e| Value::Node {
                    id: e.id.as_u64() as i64,
                    labels: e.labels.iter().map(|l| l.as_str().to_string()).collect(),
                    properties: e.properties.clone(),
                })
            })
            .unwrap_or_else(|| Value::Int(src_id.as_u64() as i64));

        match &expand.length {
            ExpandLength::Single => {
                // Single hop expansion using DatabaseTransaction's edge methods
                let neighbors =
                    get_single_hop_neighbors(tx, src_id, &expand.direction, &edge_types)?;

                for (neighbor_id, edge) in neighbors {
                    // Apply node label filter if specified
                    let dest_entity = if !expand.node_labels.is_empty() || expand.node_filter.is_some() {
                        match tx.get_entity(neighbor_id).map_err(Error::Transaction)? {
                            Some(e) => {
                                // Check labels
                                if !expand.node_labels.is_empty() {
                                    let has_label =
                                        expand.node_labels.iter().any(|label| e.has_label(label));
                                    if !has_label {
                                        continue;
                                    }
                                }
                                Some(e)
                            }
                            None => continue,
                        }
                    } else {
                        // Look up the destination entity for Value::Node
                        tx.get_entity(neighbor_id).map_err(Error::Transaction)?
                    };

                    // Apply node property filter if specified
                    if let Some(ref filter) = expand.node_filter {
                        if let Some(ref e) = dest_entity {
                            if !evaluate_entity_filter(e, filter, &expand.dst_var) {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    // Apply edge property filter if specified
                    if let Some(ref filter) = expand.edge_filter {
                        let edge_var = expand.edge_var.as_deref().unwrap_or("_edge");
                        if !evaluate_edge_filter(&edge, filter, edge_var) {
                            continue;
                        }
                    }

                    // Build destination node value
                    let dst_node_value = dest_entity
                        .map(|e| Value::Node {
                            id: e.id.as_u64() as i64,
                            labels: e.labels.iter().map(|l| l.as_str().to_string()).collect(),
                            properties: e.properties.clone(),
                        })
                        .unwrap_or_else(|| Value::Int(neighbor_id.as_u64() as i64));

                    // Build row with full Value::Node for nodes and Value::Edge for edges
                    let mut values = Vec::new();
                    values.push(src_node_value.clone());
                    values.push(dst_node_value);

                    if expand.edge_var.is_some() {
                        values.push(Value::Edge {
                            id: edge.id.as_u64() as i64,
                            edge_type: edge.edge_type.as_str().to_string(),
                            source: edge.source.as_u64() as i64,
                            target: edge.target.as_u64() as i64,
                            properties: edge.properties.clone(),
                        });
                    }

                    result_rows.push(Row::new(Arc::clone(&schema), values));
                }
            }

            ExpandLength::Range { .. } | ExpandLength::Exact(_) => {
                // Variable length expansion using BFS
                let (min_depth, max_depth) = match &expand.length {
                    ExpandLength::Range { min, max } => (*min, *max),
                    ExpandLength::Exact(n) => (*n, Some(*n)),
                    _ => (1, None),
                };

                let traversal_results = execute_variable_length_expansion(
                    tx,
                    src_id,
                    &expand.direction,
                    &edge_types,
                    min_depth,
                    max_depth,
                )?;

                for (neighbor_id, _depth) in traversal_results {
                    // Get destination entity for Value::Node and filtering
                    let dest_entity = tx.get_entity(neighbor_id).map_err(Error::Transaction)?;

                    // Apply node label filter and property filter if specified
                    if !expand.node_labels.is_empty() || expand.node_filter.is_some() {
                        if let Some(ref entity) = dest_entity {
                            // Check labels
                            if !expand.node_labels.is_empty() {
                                let has_label =
                                    expand.node_labels.iter().any(|label| entity.has_label(label));
                                if !has_label {
                                    continue;
                                }
                            }

                            // Check property filter
                            if let Some(ref filter) = expand.node_filter {
                                if !evaluate_entity_filter(entity, filter, &expand.dst_var) {
                                    continue;
                                }
                            }
                        } else {
                            continue;
                        }
                    }

                    // Build destination node value
                    let dst_node_value = dest_entity
                        .map(|e| Value::Node {
                            id: e.id.as_u64() as i64,
                            labels: e.labels.iter().map(|l| l.as_str().to_string()).collect(),
                            properties: e.properties.clone(),
                        })
                        .unwrap_or_else(|| Value::Int(neighbor_id.as_u64() as i64));

                    // Build row
                    let mut values = Vec::new();
                    values.push(src_node_value.clone());
                    values.push(dst_node_value);

                    // For variable length, we don't have a single edge ID
                    if expand.edge_var.is_some() {
                        values.push(Value::Null);
                    }

                    result_rows.push(Row::new(Arc::clone(&schema), values));
                }
            }
        }
    }

    Ok(ResultSet::with_rows(schema, result_rows))
}

/// Get single-hop neighbors using DatabaseTransaction's edge methods.
fn get_single_hop_neighbors<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    node: EntityId,
    direction: &ExpandDirection,
    edge_types: &[EdgeType],
) -> Result<Vec<(EntityId, Edge)>> {
    let mut results = Vec::new();

    // Get outgoing edges
    if matches!(direction, ExpandDirection::Outgoing | ExpandDirection::Both) {
        let outgoing = tx.get_outgoing_edges(node).map_err(Error::Transaction)?;
        for edge in outgoing {
            // Filter by edge type if specified
            if edge_types.is_empty()
                || edge_types.iter().any(|et| et.as_str() == edge.edge_type.as_str())
            {
                results.push((edge.target, edge));
            }
        }
    }

    // Get incoming edges
    if matches!(direction, ExpandDirection::Incoming | ExpandDirection::Both) {
        let incoming = tx.get_incoming_edges(node).map_err(Error::Transaction)?;
        for edge in incoming {
            // Filter by edge type if specified
            if edge_types.is_empty()
                || edge_types.iter().any(|et| et.as_str() == edge.edge_type.as_str())
            {
                results.push((edge.source, edge));
            }
        }
    }

    Ok(results)
}

/// Execute variable-length graph traversal using BFS.
fn execute_variable_length_expansion<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    start: EntityId,
    direction: &ExpandDirection,
    edge_types: &[EdgeType],
    min_depth: usize,
    max_depth: Option<usize>,
) -> Result<Vec<(EntityId, usize)>> {
    let mut visited: HashSet<EntityId> = HashSet::new();
    let mut results: Vec<(EntityId, usize)> = Vec::new();
    let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

    // Start with the initial node
    visited.insert(start);
    queue.push_back((start, 0));

    // Include start node if min_depth is 0
    if min_depth == 0 {
        results.push((start, 0));
    }

    while let Some((current, depth)) = queue.pop_front() {
        // Check if we should continue expanding
        let should_expand = max_depth.map_or(true, |max| depth < max);
        if !should_expand {
            continue;
        }

        // Get neighbors at this step
        let neighbors = get_single_hop_neighbors(tx, current, direction, edge_types)?;

        for (neighbor_id, _edge) in neighbors {
            if visited.contains(&neighbor_id) {
                continue;
            }

            visited.insert(neighbor_id);
            let next_depth = depth + 1;

            // Add to queue for further expansion
            queue.push_back((neighbor_id, next_depth));

            // Add to results if within depth range
            if next_depth >= min_depth {
                results.push((neighbor_id, next_depth));
            }
        }
    }

    Ok(results)
}

/// Evaluates a property filter predicate against an entity.
///
/// Builds a row containing the entity's properties and evaluates
/// the filter expression against it.
fn evaluate_entity_filter(entity: &Entity, filter: &LogicalExpr, var_name: &str) -> bool {
    // Build a row with the entity's properties
    let mut columns = Vec::new();
    let mut values = Vec::new();

    for (key, value) in &entity.properties {
        // Add qualified column name (e.g., "n.name")
        columns.push(format!("{}.{}", var_name, key));
        values.push(value.clone());
    }

    let schema = Arc::new(Schema::new(columns));
    let row = Row::new(schema, values);

    // Evaluate the filter expression
    match evaluate_expr(filter, &row) {
        Ok(Value::Bool(b)) => b,
        Ok(Value::Null) => false,
        Ok(_) => false,
        Err(_) => false, // If filter evaluation fails, exclude the entity
    }
}

/// Evaluates a property filter predicate against an edge.
///
/// Builds a row containing the edge's properties and evaluates
/// the filter expression against it.
fn evaluate_edge_filter(edge: &Edge, filter: &LogicalExpr, var_name: &str) -> bool {
    // Build a row with the edge's properties
    let mut columns = Vec::new();
    let mut values = Vec::new();

    for (key, value) in &edge.properties {
        // Add qualified column name (e.g., "r.since")
        columns.push(format!("{}.{}", var_name, key));
        values.push(value.clone());
    }

    let schema = Arc::new(Schema::new(columns));
    let row = Row::new(schema, values);

    // Evaluate the filter expression
    match evaluate_expr(filter, &row) {
        Ok(Value::Bool(b)) => b,
        Ok(Value::Null) => false,
        Ok(_) => false,
        Err(_) => false, // If filter evaluation fails, exclude the edge
    }
}

/// Extract entity IDs and their corresponding rows from a ResultSet.
///
/// This function looks for an ID column (matching the source variable pattern)
/// and returns pairs of (EntityId, Row).
pub fn extract_source_nodes(result: ResultSet, src_var: &str) -> Vec<(EntityId, Row)> {
    let schema = result.schema_arc();
    let columns = schema.columns();

    // Find the column that represents the source variable
    // It might be "p", "p._rowid", or just "_rowid"
    let id_col_idx = columns
        .iter()
        .position(|c| {
            *c == src_var
                || *c == format!("{}._rowid", src_var)
                || *c == "_rowid"
                || c.ends_with("._rowid")
        })
        .unwrap_or(0);

    result
        .into_rows()
        .into_iter()
        .filter_map(|row| {
            let id_value = row.get(id_col_idx)?;
            let entity_id = match id_value {
                Value::Node { id, .. } => Some(EntityId::new(*id as u64)),
                Value::Edge { id, .. } => Some(EntityId::new(*id as u64)),
                Value::Int(id) => Some(EntityId::new(*id as u64)),
                _ => None,
            }?;
            Some((entity_id, row))
        })
        .collect()
}

// ============================================================================
// Graph Mutator Implementation for DatabaseTransaction
// ============================================================================

use manifoldb_graph::traversal::Direction;
use manifoldb_query::exec::graph_accessor::{
    CreateEdgeRequest, CreateNodeRequest, DeleteResult, EdgeResult, GraphAccessError,
    GraphAccessResult, GraphAccessor, GraphMutator, NeighborResult, NodeScanResult, PathFindConfig,
    PathMatchResult, ShortestPathConfig, ShortestPathResult, TraversalResult, UpdateEdgeRequest,
    UpdateNodeRequest,
};

/// A `GraphMutator` implementation that wraps a `DatabaseTransaction`.
///
/// This allows CREATE operations to use the database's entity and edge storage
/// directly, using the proper transaction semantics.
///
/// The transaction is stored in an `Option` inside an `RwLock`, allowing
/// it to be "taken" out after execution completes (for commit/rollback).
pub struct DatabaseGraphMutator<T: Transaction> {
    tx: std::sync::Arc<std::sync::RwLock<Option<DatabaseTransaction<T>>>>,
}

impl<T: Transaction> DatabaseGraphMutator<T> {
    /// Create a new mutator wrapping a database transaction.
    pub fn new(tx: DatabaseTransaction<T>) -> Self {
        Self { tx: std::sync::Arc::new(std::sync::RwLock::new(Some(tx))) }
    }

    /// Get a reference to the inner Arc for sharing with the context.
    ///
    /// This allows multiple references to exist while still being able to
    /// take the transaction out at the end.
    pub fn transaction_arc(
        &self,
    ) -> std::sync::Arc<std::sync::RwLock<Option<DatabaseTransaction<T>>>> {
        self.tx.clone()
    }

    /// Take the underlying transaction out of the mutator.
    ///
    /// This is used to commit or rollback the transaction after CREATE operations.
    /// Returns `None` if the transaction was already taken.
    pub fn take_transaction(&self) -> Option<DatabaseTransaction<T>> {
        self.tx.write().ok()?.take()
    }
}

impl<T: Transaction> Clone for DatabaseGraphMutator<T> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone() }
    }
}

impl<T> GraphMutator for DatabaseGraphMutator<T>
where
    T: Transaction + Send + Sync,
{
    fn create_node(
        &self,
        request: &CreateNodeRequest,
    ) -> GraphAccessResult<manifoldb_core::Entity> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Use DatabaseTransaction's create_entity method
        let mut entity = tx
            .create_entity()
            .map_err(|e| GraphAccessError::Internal(format!("failed to create entity: {e}")))?;

        // Add labels
        for label in &request.labels {
            entity = entity.with_label(label.as_str());
        }

        // Add properties
        for (key, value) in &request.properties {
            entity = entity.with_property(key.clone(), value.clone());
        }

        // Save the entity
        tx.put_entity(&entity)
            .map_err(|e| GraphAccessError::Internal(format!("failed to save entity: {e}")))?;

        Ok(entity)
    }

    fn create_edge(&self, request: &CreateEdgeRequest) -> GraphAccessResult<Edge> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Use DatabaseTransaction's create_edge method
        let mut edge = tx
            .create_edge(request.source, request.target, request.edge_type.clone())
            .map_err(|e| GraphAccessError::Internal(format!("failed to create edge: {e}")))?;

        // Add properties
        for (key, value) in &request.properties {
            edge = edge.with_property(key.clone(), value.clone());
        }

        // Save the edge
        tx.put_edge(&edge)
            .map_err(|e| GraphAccessError::Internal(format!("failed to save edge: {e}")))?;

        Ok(edge)
    }

    fn update_node(&self, request: &UpdateNodeRequest) -> GraphAccessResult<Entity> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Get the existing entity
        let mut entity = tx
            .get_entity(request.id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get entity: {e}")))?
            .ok_or_else(|| {
                GraphAccessError::Internal(format!("entity {} not found", request.id.as_u64()))
            })?;

        // Set new properties
        for (key, value) in &request.set_properties {
            entity.properties.insert(key.clone(), value.clone());
        }

        // Remove properties
        for key in &request.remove_properties {
            entity.properties.remove(key);
        }

        // Add labels
        for label in &request.add_labels {
            if !entity.labels.contains(label) {
                entity.labels.push(label.clone());
            }
        }

        // Remove labels
        for label in &request.remove_labels {
            entity.labels.retain(|l| l != label);
        }

        // Save the updated entity
        tx.put_entity(&entity)
            .map_err(|e| GraphAccessError::Internal(format!("failed to save entity: {e}")))?;

        Ok(entity)
    }

    fn update_edge(&self, request: &UpdateEdgeRequest) -> GraphAccessResult<Edge> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Get the existing edge
        let mut edge = tx
            .get_edge(request.id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get edge: {e}")))?
            .ok_or_else(|| {
                GraphAccessError::Internal(format!("edge {} not found", request.id.as_u64()))
            })?;

        // Set new properties
        for (key, value) in &request.set_properties {
            edge.properties.insert(key.clone(), value.clone());
        }

        // Remove properties
        for key in &request.remove_properties {
            edge.properties.remove(key);
        }

        // Save the updated edge
        tx.put_edge(&edge)
            .map_err(|e| GraphAccessError::Internal(format!("failed to save edge: {e}")))?;

        Ok(edge)
    }

    fn get_node(&self, id: EntityId) -> GraphAccessResult<Option<Entity>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire read lock: {e}")))?;

        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        tx.get_entity(id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get entity: {e}")))
    }

    fn get_edge(&self, id: manifoldb_core::EdgeId) -> GraphAccessResult<Option<Edge>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire read lock: {e}")))?;

        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        tx.get_edge(id).map_err(|e| GraphAccessError::Internal(format!("failed to get edge: {e}")))
    }

    fn remove_entity_property(&self, entity_id: EntityId, property: &str) -> GraphAccessResult<()> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Get current entity
        if let Some(mut entity) = tx
            .get_entity(entity_id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get entity: {e}")))?
        {
            // Remove the property
            entity.properties.remove(property);

            // Save the updated entity
            tx.put_entity(&entity)
                .map_err(|e| GraphAccessError::Internal(format!("failed to save entity: {e}")))?;
        }

        Ok(())
    }

    fn remove_entity_label(&self, entity_id: EntityId, label: &str) -> GraphAccessResult<()> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Get current entity
        if let Some(mut entity) = tx
            .get_entity(entity_id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get entity: {e}")))?
        {
            // Remove the label
            entity.labels.retain(|l| l.as_str() != label);

            // Save the updated entity (put_entity handles label index updates)
            tx.put_entity(&entity)
                .map_err(|e| GraphAccessError::Internal(format!("failed to save entity: {e}")))?;
        }

        Ok(())
    }

    fn remove_edge_property(
        &self,
        edge_id: manifoldb_core::EdgeId,
        property: &str,
    ) -> GraphAccessResult<()> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Get current edge
        if let Some(mut edge) = tx
            .get_edge(edge_id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to get edge: {e}")))?
        {
            // Remove the property
            edge.properties.remove(property);

            // Save the updated edge
            tx.put_edge(&edge)
                .map_err(|e| GraphAccessError::Internal(format!("failed to save edge: {e}")))?;
        }

        Ok(())
    }

    fn delete_node(&self, id: EntityId) -> GraphAccessResult<DeleteResult> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Use delete_entity_checked which fails if the node has edges
        match tx.delete_entity_checked(id) {
            Ok(true) => Ok(DeleteResult::node_deleted()),
            Ok(false) => Ok(DeleteResult::not_found()),
            Err(e) => {
                // Check if it's the "has edges" error
                let err_msg = e.to_string();
                if err_msg.contains("still has edges") || err_msg.contains("HasEdges") {
                    Err(GraphAccessError::Internal(format!(
                        "cannot delete node {:?} because it still has relationships. \
                         Use DETACH DELETE to delete the node and its relationships.",
                        id
                    )))
                } else {
                    Err(GraphAccessError::Internal(format!("failed to delete entity: {e}")))
                }
            }
        }
    }

    fn delete_node_detach(&self, id: EntityId) -> GraphAccessResult<DeleteResult> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        // Use delete_entity_cascade which deletes edges first
        let core_result = tx
            .delete_entity_cascade(id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to delete entity: {e}")))?;

        if core_result.entity_deleted {
            Ok(DeleteResult::detach_deleted(core_result.edges_deleted_count()))
        } else {
            // Node didn't exist, but we may have deleted some edges
            Ok(DeleteResult { nodes_deleted: 0, edges_deleted: core_result.edges_deleted_count() })
        }
    }

    fn delete_edge(&self, id: manifoldb_core::EdgeId) -> GraphAccessResult<bool> {
        let mut guard = self.tx.write().map_err(|e| {
            GraphAccessError::Internal(format!("failed to acquire write lock: {e}"))
        })?;

        let tx = guard
            .as_mut()
            .ok_or_else(|| GraphAccessError::Internal("transaction already taken".to_string()))?;

        tx.delete_edge(id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to delete edge: {e}")))
    }

    fn node_has_edges(&self, id: EntityId) -> GraphAccessResult<bool> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire read lock: {e}")))?;

        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        tx.has_edges(id)
            .map_err(|e| GraphAccessError::Internal(format!("failed to check edges: {e}")))
    }
}

// ============================================================================
// Graph Accessor Implementation for DatabaseTransaction
// ============================================================================

/// A `GraphAccessor` implementation that wraps a `DatabaseTransaction`.
///
/// This allows MATCH operations to read from the database's entity and edge storage.
/// The transaction is shared with `DatabaseGraphMutator` for MATCH + CREATE patterns.
pub struct DatabaseGraphAccessor<T: Transaction> {
    tx: std::sync::Arc<std::sync::RwLock<Option<DatabaseTransaction<T>>>>,
}

impl<T: Transaction> DatabaseGraphAccessor<T> {
    /// Create a new accessor sharing a transaction arc (typically from a DatabaseGraphMutator).
    pub fn from_arc(tx: std::sync::Arc<std::sync::RwLock<Option<DatabaseTransaction<T>>>>) -> Self {
        Self { tx }
    }

    /// Create a new accessor owning a transaction (for read-only queries).
    ///
    /// This wraps the transaction in an Arc<RwLock<Option<...>>> for compatibility
    /// with the existing interface, but doesn't share with a mutator.
    pub fn from_transaction(tx: DatabaseTransaction<T>) -> Self {
        Self {
            tx: std::sync::Arc::new(std::sync::RwLock::new(Some(tx))),
        }
    }
}

impl<T: Transaction> Clone for DatabaseGraphAccessor<T> {
    fn clone(&self) -> Self {
        Self { tx: self.tx.clone() }
    }
}

impl<T> GraphAccessor for DatabaseGraphAccessor<T>
where
    T: Transaction + Send + Sync,
{
    fn neighbors(
        &self,
        node: EntityId,
        direction: Direction,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        let mut results = Vec::new();

        if matches!(direction, Direction::Outgoing | Direction::Both) {
            let edges = tx
                .get_outgoing_edges(node)
                .map_err(|e| GraphAccessError::Internal(e.to_string()))?;
            for edge in edges {
                results.push(NeighborResult::new(edge.target, edge.id, Direction::Outgoing));
            }
        }

        if matches!(direction, Direction::Incoming | Direction::Both) {
            let edges = tx
                .get_incoming_edges(node)
                .map_err(|e| GraphAccessError::Internal(e.to_string()))?;
            for edge in edges {
                results.push(NeighborResult::new(edge.source, edge.id, Direction::Incoming));
            }
        }

        Ok(results)
    }

    fn neighbors_by_type(
        &self,
        node: EntityId,
        direction: Direction,
        edge_type: &EdgeType,
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        self.neighbors_by_types(node, direction, std::slice::from_ref(edge_type))
    }

    fn neighbors_by_types(
        &self,
        node: EntityId,
        direction: Direction,
        edge_types: &[EdgeType],
    ) -> GraphAccessResult<Vec<NeighborResult>> {
        let all_neighbors = self.neighbors(node, direction)?;
        if edge_types.is_empty() {
            return Ok(all_neighbors);
        }

        // Filter by edge types - we need to get the edge to check its type
        // For efficiency we'd need an index, but for now just filter all neighbors
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        let filtered: Vec<NeighborResult> = all_neighbors
            .into_iter()
            .filter(|n| {
                if let Ok(Some(edge)) = tx.get_edge(n.edge_id) {
                    edge_types.iter().any(|et| et.as_str() == edge.edge_type.as_str())
                } else {
                    false
                }
            })
            .collect();

        Ok(filtered)
    }

    fn expand_all(
        &self,
        node: EntityId,
        direction: Direction,
        min_depth: usize,
        max_depth: Option<usize>,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphAccessResult<Vec<TraversalResult>> {
        // Simple BFS implementation
        let mut visited: HashSet<EntityId> = HashSet::new();
        let mut results: Vec<TraversalResult> = Vec::new();
        let mut queue: VecDeque<(EntityId, usize)> = VecDeque::new();

        visited.insert(node);
        queue.push_back((node, 0));

        if min_depth == 0 {
            results.push(TraversalResult::new(node, None, 0));
        }

        while let Some((current, depth)) = queue.pop_front() {
            let should_expand = max_depth.map_or(true, |max| depth < max);
            if !should_expand {
                continue;
            }

            let neighbors = if let Some(types) = edge_types {
                self.neighbors_by_types(current, direction, types)?
            } else {
                self.neighbors(current, direction)?
            };

            for neighbor in neighbors {
                if visited.contains(&neighbor.node) {
                    continue;
                }

                visited.insert(neighbor.node);
                let next_depth = depth + 1;

                queue.push_back((neighbor.node, next_depth));

                if next_depth >= min_depth {
                    results.push(TraversalResult::new(
                        neighbor.node,
                        Some(neighbor.edge_id),
                        next_depth,
                    ));
                }
            }
        }

        Ok(results)
    }

    fn find_paths(
        &self,
        _start: EntityId,
        _config: &PathFindConfig,
    ) -> GraphAccessResult<Vec<PathMatchResult>> {
        // Not implemented for graph DML context
        Ok(vec![])
    }

    fn shortest_path(
        &self,
        _source: EntityId,
        _target: EntityId,
        _config: &ShortestPathConfig,
    ) -> GraphAccessResult<Vec<ShortestPathResult>> {
        // Not implemented for graph DML context
        Ok(vec![])
    }

    fn get_entity_properties(
        &self,
        entity_id: EntityId,
    ) -> GraphAccessResult<Option<HashMap<String, Value>>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        match tx.get_entity(entity_id) {
            Ok(Some(entity)) => Ok(Some(entity.properties.clone())),
            Ok(None) => Ok(None),
            Err(e) => Err(GraphAccessError::Internal(e.to_string())),
        }
    }

    fn get_edge_properties(
        &self,
        edge_id: manifoldb_core::EdgeId,
    ) -> GraphAccessResult<Option<HashMap<String, Value>>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        match tx.get_edge(edge_id) {
            Ok(Some(edge)) => Ok(Some(edge.properties.clone())),
            Ok(None) => Ok(None),
            Err(e) => Err(GraphAccessError::Internal(e.to_string())),
        }
    }

    fn entity_has_labels(&self, entity_id: EntityId, labels: &[String]) -> GraphAccessResult<bool> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        match tx.get_entity(entity_id) {
            Ok(Some(entity)) => Ok(labels.iter().all(|l| entity.has_label(l))),
            Ok(None) => Ok(false),
            Err(e) => Err(GraphAccessError::Internal(e.to_string())),
        }
    }

    fn scan_nodes(&self, label: Option<&str>) -> GraphAccessResult<Vec<NodeScanResult>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        // Use iter_entities to get all entities with the given label
        let entities =
            tx.iter_entities(label).map_err(|e| GraphAccessError::Internal(e.to_string()))?;

        Ok(entities.iter().map(NodeScanResult::from_entity).collect())
    }

    fn get_entity(&self, entity_id: EntityId) -> GraphAccessResult<Option<NodeScanResult>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        match tx.get_entity(entity_id) {
            Ok(Some(entity)) => Ok(Some(NodeScanResult::from_entity(&entity))),
            Ok(None) => Ok(None),
            Err(e) => Err(GraphAccessError::Internal(e.to_string())),
        }
    }

    fn get_edge(&self, edge_id: EdgeId) -> GraphAccessResult<Option<EdgeResult>> {
        let guard = self
            .tx
            .read()
            .map_err(|e| GraphAccessError::Internal(format!("failed to acquire lock: {e}")))?;
        let tx = guard
            .as_ref()
            .ok_or_else(|| GraphAccessError::Internal("transaction not available".to_string()))?;

        match tx.get_edge(edge_id) {
            Ok(Some(edge)) => Ok(Some(EdgeResult {
                id: edge.id,
                edge_type: edge.edge_type.as_str().to_string(),
                source: edge.source,
                target: edge.target,
                properties: edge.properties.clone(),
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(GraphAccessError::Internal(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        // Basic compilation test
    }
}
