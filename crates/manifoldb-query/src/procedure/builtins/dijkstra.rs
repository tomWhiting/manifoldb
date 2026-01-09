//! Dijkstra's weighted shortest path procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::traversal::{Dijkstra, Direction};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Dijkstra's weighted shortest path procedure.
///
/// Finds the shortest weighted path between two nodes using Dijkstra's algorithm.
///
/// # Usage
///
/// ```sql
/// CALL algo.dijkstra(sourceId, targetId) YIELD path, totalCost
/// CALL algo.dijkstra(1, 10, 'weight') YIELD path, totalCost
/// ```
///
/// # Parameters
///
/// - `sourceId` (required, INTEGER): The source node ID
/// - `targetId` (required, INTEGER): The target node ID
/// - `weightProperty` (optional, STRING): Name of the edge property to use as weight (default: "weight")
/// - `defaultWeight` (optional, FLOAT): Default weight for edges without the property (default: 1.0)
/// - `maxWeight` (optional, FLOAT): Maximum total weight to search
///
/// # Returns
///
/// - `path` (ARRAY): Array of node IDs in the path
/// - `totalCost` (FLOAT): Total weight of the path
/// - `nodeIds` (ARRAY): Array of node IDs (same as path)
/// - `edgeIds` (ARRAY): Array of edge IDs in the path
pub struct DijkstraProcedure;

impl Procedure for DijkstraProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.dijkstra")
            .with_description("Finds the shortest weighted path using Dijkstra's algorithm")
            .with_parameter(
                ProcedureParameter::required("sourceId", "INTEGER")
                    .with_description("The source node ID"),
            )
            .with_parameter(
                ProcedureParameter::required("targetId", "INTEGER")
                    .with_description("The target node ID"),
            )
            .with_parameter(
                ProcedureParameter::optional("weightProperty", "STRING")
                    .with_description("Edge property to use as weight (default: 'weight')"),
            )
            .with_parameter(
                ProcedureParameter::optional("defaultWeight", "FLOAT")
                    .with_description("Default weight if property missing (default: 1.0)"),
            )
            .with_parameter(
                ProcedureParameter::optional("maxWeight", "FLOAT")
                    .with_description("Maximum total weight to search"),
            )
            .with_return(
                ReturnColumn::new("path", "ARRAY").with_description("Array of node IDs in path"),
            )
            .with_return(
                ReturnColumn::new("totalCost", "FLOAT")
                    .with_description("Total weight of the path"),
            )
            .with_return(
                ReturnColumn::new("nodeIds", "ARRAY").with_description("Array of node IDs"),
            )
            .with_return(
                ReturnColumn::new("edgeIds", "ARRAY").with_description("Array of edge IDs"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.dijkstra requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get required parameters
        let source_id = args.get_int(0, "sourceId")?;
        let target_id = args.get_int(1, "targetId")?;
        let weight_property = args.get_string_opt(2);
        let default_weight = args.get_float_or(3, 1.0);
        let max_weight = get_float_opt(&args, 4);

        let _ = ctx;
        let _ = source_id;
        let _ = target_id;
        let _ = weight_property;
        let _ = default_weight;
        let _ = max_weight;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "Dijkstra execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            "path".to_string(),
            "totalCost".to_string(),
            "nodeIds".to_string(),
            "edgeIds".to_string(),
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

/// Helper function to execute Dijkstra with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_dijkstra_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    source_id: i64,
    target_id: i64,
    weight_property: Option<&str>,
    default_weight: f64,
    max_weight: Option<f64>,
) -> ProcedureResult<RowBatch> {
    let source = EntityId::new(source_id as u64);
    let target = EntityId::new(target_id as u64);

    // Build the Dijkstra finder
    let mut finder = Dijkstra::new(source, target, Direction::Both);

    // Set weight property
    if let Some(prop) = weight_property {
        finder = finder.with_weight_property_default(prop, default_weight);
    } else {
        finder = finder.with_weight_property_default("weight", default_weight);
    }

    // Set max weight constraint
    if let Some(max) = max_weight {
        finder = finder.with_max_weight(max);
    }

    // Execute the search
    let result = finder.find(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec![
        "path".to_string(),
        "totalCost".to_string(),
        "nodeIds".to_string(),
        "edgeIds".to_string(),
    ]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    if let Some(path) = result {
        // Convert node IDs to array of integers
        let path_values: Vec<Value> =
            path.nodes.iter().map(|id| Value::Int(id.as_u64() as i64)).collect();

        // Convert edge IDs to array of integers
        let edge_values: Vec<Value> =
            path.edges.iter().map(|id| Value::Int(id.as_u64() as i64)).collect();

        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Array(path_values.clone()),
                Value::Float(path.total_weight),
                Value::Array(path_values),
                Value::Array(edge_values),
            ],
        );
        batch.push(row);
    }
    // If no path found, return empty batch

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature() {
        let proc = DijkstraProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.dijkstra");
        assert_eq!(sig.parameters.len(), 5);
        assert_eq!(sig.returns.len(), 4);
        assert_eq!(sig.required_param_count(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = DijkstraProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["path", "totalCost", "nodeIds", "edgeIds"]);
    }

    #[test]
    fn requires_context() {
        let proc = DijkstraProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn get_float_opt_float() {
        let args = ProcedureArgs::new(vec![Value::Float(3.14)]);
        assert_eq!(get_float_opt(&args, 0), Some(3.14));
    }

    #[test]
    fn get_float_opt_int() {
        let args = ProcedureArgs::new(vec![Value::Int(42)]);
        assert_eq!(get_float_opt(&args, 0), Some(42.0));
    }

    #[test]
    fn get_float_opt_missing() {
        let args = ProcedureArgs::new(vec![]);
        assert_eq!(get_float_opt(&args, 0), None);
    }
}
