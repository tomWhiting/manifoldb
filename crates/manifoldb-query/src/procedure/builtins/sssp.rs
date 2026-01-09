//! Single-Source Shortest Paths (SSSP) procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::traversal::{Direction, SingleSourceDijkstra};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Single-Source Shortest Paths (SSSP) procedure.
///
/// Computes shortest paths from a source node to all reachable nodes using
/// Dijkstra's algorithm.
///
/// # Usage
///
/// ```sql
/// CALL algo.sssp(sourceId) YIELD nodeId, distance
/// CALL algo.sssp(1, 'weight', 100.0) YIELD nodeId, distance, pathNodeIds
/// ```
///
/// # Parameters
///
/// - `sourceId` (required, INTEGER): The source node ID
/// - `weightProperty` (optional, STRING): Name of the edge property to use as weight (default: "weight")
/// - `maxWeight` (optional, FLOAT): Maximum distance to search (cutoff)
///
/// # Returns
///
/// Returns one row per reachable node:
/// - `nodeId` (INTEGER): The reachable node ID
/// - `distance` (FLOAT): Total weight/distance from source to this node
/// - `pathNodeIds` (ARRAY): Array of node IDs in the path from source
pub struct SSSPProcedure;

impl Procedure for SSSPProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.sssp")
            .with_description("Computes single-source shortest paths to all reachable nodes")
            .with_parameter(
                ProcedureParameter::required("sourceId", "INTEGER")
                    .with_description("The source node ID"),
            )
            .with_parameter(
                ProcedureParameter::optional("weightProperty", "STRING")
                    .with_description("Edge property to use as weight (default: 'weight')"),
            )
            .with_parameter(
                ProcedureParameter::optional("maxWeight", "FLOAT")
                    .with_description("Maximum distance to search (cutoff)"),
            )
            .with_return(
                ReturnColumn::new("nodeId", "INTEGER").with_description("The reachable node ID"),
            )
            .with_return(
                ReturnColumn::new("distance", "FLOAT")
                    .with_description("Total distance from source"),
            )
            .with_return(
                ReturnColumn::new("pathNodeIds", "ARRAY")
                    .with_description("Node IDs in path from source"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed("algo.sssp requires graph storage context".to_string()))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get required parameters
        let source_id = args.get_int(0, "sourceId")?;
        let weight_property = args.get_string_opt(1);
        let max_weight = get_float_opt(&args, 2);

        let _ = ctx;
        let _ = source_id;
        let _ = weight_property;
        let _ = max_weight;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "SSSP execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            "nodeId".to_string(),
            "distance".to_string(),
            "pathNodeIds".to_string(),
        ]))
    }
}

/// Helper function to get an optional float argument.
fn get_float_opt(args: &ProcedureArgs, index: usize) -> Option<f64> {
    match args.get(index) {
        Some(Value::Float(f)) => Some(*f),
        Some(Value::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

/// Helper function to execute SSSP with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_sssp_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    source_id: i64,
    weight_property: Option<&str>,
    max_weight: Option<f64>,
) -> ProcedureResult<RowBatch> {
    let source = EntityId::new(source_id as u64);

    // Build the single-source Dijkstra finder
    let mut finder = SingleSourceDijkstra::new(source, Direction::Both);

    if let Some(prop) = weight_property {
        finder = finder.with_weight_property(prop);
    }

    if let Some(max) = max_weight {
        finder = finder.with_max_weight(max);
    }

    // Execute the search
    let results = finder.compute(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec![
        "nodeId".to_string(),
        "distance".to_string(),
        "pathNodeIds".to_string(),
    ]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    // Add one row per reachable node
    for (node_id, (distance, _parent_info)) in &results {
        // Reconstruct path from source to this node
        let path_node_ids = reconstruct_path_to_node(&results, source, *node_id);

        let path_values: Vec<Value> =
            path_node_ids.iter().map(|id| Value::Int(id.as_u64() as i64)).collect();

        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Int(node_id.as_u64() as i64),
                Value::Float(*distance),
                Value::Array(path_values),
            ],
        );
        batch.push(row);
    }

    Ok(batch)
}

/// Reconstruct the path from source to a target node using the SSSP results.
fn reconstruct_path_to_node(
    results: &std::collections::HashMap<
        EntityId,
        (f64, Option<(EntityId, manifoldb_core::EdgeId)>),
    >,
    source: EntityId,
    target: EntityId,
) -> Vec<EntityId> {
    let mut path = Vec::new();
    let mut current = target;

    // Trace back from target to source
    loop {
        path.push(current);
        if current == source {
            break;
        }

        match results.get(&current) {
            Some((_, Some((parent, _)))) => {
                current = *parent;
            }
            _ => break, // Should not happen in valid SSSP results
        }
    }

    // Reverse to get source -> target order
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature() {
        let proc = SSSPProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.sssp");
        assert_eq!(sig.parameters.len(), 3);
        assert_eq!(sig.returns.len(), 3);
        assert_eq!(sig.required_param_count(), 1);
    }

    #[test]
    fn output_schema() {
        let proc = SSSPProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "distance", "pathNodeIds"]);
    }

    #[test]
    fn requires_context() {
        let proc = SSSPProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn get_float_opt_values() {
        let args = ProcedureArgs::new(vec![Value::Float(3.14)]);
        assert_eq!(get_float_opt(&args, 0), Some(3.14));
        assert_eq!(get_float_opt(&args, 1), None);
    }
}
