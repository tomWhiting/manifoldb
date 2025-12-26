//! Graph accessor implementation for query execution.
//!
//! This module provides utilities for executing graph traversals using
//! the `DatabaseTransaction` edge traversal methods directly.
//!
//! This approach uses the `DatabaseTransaction`'s own edge storage and indexing,
//! ensuring compatibility with the edge key format used by `put_edge`/`get_edge`.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use manifoldb_core::{Edge, EdgeType, EntityId, Value};
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::ResultSet;
use manifoldb_query::plan::logical::{ExpandDirection, ExpandLength, ExpandNode};
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
    for (src_id, _src_row) in source_nodes {
        match &expand.length {
            ExpandLength::Single => {
                // Single hop expansion using DatabaseTransaction's edge methods
                let neighbors =
                    get_single_hop_neighbors(tx, src_id, &expand.direction, &edge_types)?;

                for (neighbor_id, edge) in neighbors {
                    // Apply node label filter if specified
                    if !expand.node_labels.is_empty() {
                        if let Some(entity) =
                            tx.get_entity(neighbor_id).map_err(Error::Transaction)?
                        {
                            let has_label =
                                expand.node_labels.iter().any(|label| entity.has_label(label));
                            if !has_label {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    // Build row
                    let mut values = Vec::new();
                    values.push(Value::Int(src_id.as_u64() as i64));
                    values.push(Value::Int(neighbor_id.as_u64() as i64));

                    if expand.edge_var.is_some() {
                        values.push(Value::Int(edge.id.as_u64() as i64));
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
                    // Apply node label filter if specified
                    if !expand.node_labels.is_empty() {
                        if let Some(entity) =
                            tx.get_entity(neighbor_id).map_err(Error::Transaction)?
                        {
                            let has_label =
                                expand.node_labels.iter().any(|label| entity.has_label(label));
                            if !has_label {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }

                    // Build row
                    let mut values = Vec::new();
                    values.push(Value::Int(src_id.as_u64() as i64));
                    values.push(Value::Int(neighbor_id.as_u64() as i64));

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
                Value::Int(id) => Some(EntityId::new(*id as u64)),
                _ => None,
            }?;
            Some((entity_id, row))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        // Basic compilation test
    }
}
