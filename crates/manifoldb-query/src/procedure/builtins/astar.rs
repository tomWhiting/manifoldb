//! A* weighted shortest path procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::traversal::{AStar, Direction, EuclideanHeuristic};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// A* weighted shortest path procedure.
///
/// Finds the shortest weighted path between two nodes using A* algorithm
/// with an optional heuristic based on node properties.
///
/// # Usage
///
/// ```sql
/// CALL algo.astar(sourceId, targetId) YIELD path, totalCost
/// CALL algo.astar(1, 10, 'distance', 'latitude', 'longitude') YIELD path, totalCost
/// ```
///
/// # Parameters
///
/// - `sourceId` (required, INTEGER): The source node ID
/// - `targetId` (required, INTEGER): The target node ID
/// - `weightProperty` (optional, STRING): Name of the edge property to use as weight (default: "weight")
/// - `latProperty` (optional, STRING): Latitude property for Euclidean heuristic
/// - `lonProperty` (optional, STRING): Longitude property for Euclidean heuristic
/// - `maxCost` (optional, FLOAT): Maximum total cost to search
///
/// # Returns
///
/// - `path` (ARRAY): Array of node IDs in the path
/// - `totalCost` (FLOAT): Total weight of the path
/// - `nodeIds` (ARRAY): Array of node IDs (same as path)
/// - `edgeIds` (ARRAY): Array of edge IDs in the path
pub struct AStarProcedure;

impl Procedure for AStarProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.astar")
            .with_description("Finds the shortest weighted path using A* algorithm")
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
                ProcedureParameter::optional("latProperty", "STRING")
                    .with_description("Latitude property for Euclidean heuristic"),
            )
            .with_parameter(
                ProcedureParameter::optional("lonProperty", "STRING")
                    .with_description("Longitude property for Euclidean heuristic"),
            )
            .with_parameter(
                ProcedureParameter::optional("maxCost", "FLOAT")
                    .with_description("Maximum total cost to search"),
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
            "algo.astar requires graph storage context".to_string(),
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
        let lat_property = args.get_string_opt(3);
        let lon_property = args.get_string_opt(4);
        let max_cost = get_float_opt(&args, 5);

        let _ = ctx;
        let _ = source_id;
        let _ = target_id;
        let _ = weight_property;
        let _ = lat_property;
        let _ = lon_property;
        let _ = max_cost;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "A* execution requires direct transaction access. \
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

/// Helper function to execute A* with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_astar_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    source_id: i64,
    target_id: i64,
    weight_property: Option<&str>,
    lat_property: Option<&str>,
    lon_property: Option<&str>,
    max_cost: Option<f64>,
) -> ProcedureResult<RowBatch> {
    let source = EntityId::new(source_id as u64);
    let target = EntityId::new(target_id as u64);

    // Build the A* finder
    let finder = AStar::new(source, target, Direction::Both);

    // Set weight property
    let finder = if let Some(prop) = weight_property {
        finder.with_weight_property(prop)
    } else {
        finder.with_weight_property("weight")
    };

    // Set heuristic if geographic properties are provided
    let result = if let (Some(lat), Some(lon)) = (lat_property, lon_property) {
        let heuristic = EuclideanHeuristic::with_properties([lat, lon]);
        let finder = finder.with_heuristic(heuristic);

        // Set max cost constraint
        let finder = if let Some(max) = max_cost { finder.with_max_cost(max) } else { finder };

        finder.find(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?
    } else {
        // No heuristic (behaves like Dijkstra)
        // Set max cost constraint
        let finder = if let Some(max) = max_cost { finder.with_max_cost(max) } else { finder };

        finder.find(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?
    };

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
        let proc = AStarProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.astar");
        assert_eq!(sig.parameters.len(), 6);
        assert_eq!(sig.returns.len(), 4);
        assert_eq!(sig.required_param_count(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = AStarProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["path", "totalCost", "nodeIds", "edgeIds"]);
    }

    #[test]
    fn requires_context() {
        let proc = AStarProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn get_float_opt_values() {
        let args = ProcedureArgs::new(vec![Value::Float(3.14), Value::Int(42)]);
        assert_eq!(get_float_opt(&args, 0), Some(3.14));
        assert_eq!(get_float_opt(&args, 1), Some(42.0));
        assert_eq!(get_float_opt(&args, 2), None);
    }
}
