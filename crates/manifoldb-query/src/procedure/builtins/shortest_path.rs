//! Shortest path procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::traversal::{Direction, ShortestPath};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Shortest path procedure.
///
/// Finds the shortest path between two nodes using BFS.
///
/// # Usage
///
/// ```sql
/// CALL algo.shortestPath(sourceId, targetId) YIELD nodeId, distance
/// CALL algo.shortestPath(1, 10, 'FRIEND') YIELD nodeId, distance
/// ```
///
/// # Parameters
///
/// - `sourceId` (required, INTEGER): The source node ID
/// - `targetId` (required, INTEGER): The target node ID
/// - `edge_type` (optional, STRING): Filter by edge type
/// - `max_depth` (optional, INTEGER): Maximum path length
///
/// # Returns
///
/// - `path` (ARRAY): Array of node IDs in the path
/// - `length` (INTEGER): Length of the path (number of edges)
pub struct ShortestPathProcedure;

impl Procedure for ShortestPathProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.shortestPath")
            .with_description("Finds the shortest path between two nodes")
            .with_parameter(
                ProcedureParameter::required("sourceId", "INTEGER")
                    .with_description("The source node ID"),
            )
            .with_parameter(
                ProcedureParameter::required("targetId", "INTEGER")
                    .with_description("The target node ID"),
            )
            .with_parameter(
                ProcedureParameter::optional("edge_type", "STRING")
                    .with_description("Optional edge type filter"),
            )
            .with_parameter(
                ProcedureParameter::optional("max_depth", "INTEGER")
                    .with_description("Maximum path length to search"),
            )
            .with_return(
                ReturnColumn::new("path", "ARRAY").with_description("Array of node IDs in path"),
            )
            .with_return(
                ReturnColumn::new("length", "INTEGER").with_description("Number of edges in path"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.shortestPath requires graph storage context".to_string(),
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
        let edge_type = args.get_string_opt(2);
        let max_depth = args.get_int_opt(3);

        let _ = ctx;
        let _ = source_id;
        let _ = target_id;
        let _ = edge_type;
        let _ = max_depth;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "ShortestPath execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["path".to_string(), "length".to_string()]))
    }
}

/// Helper function to execute shortest path with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_shortest_path_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    source_id: i64,
    target_id: i64,
    edge_type: Option<&str>,
    max_depth: Option<i64>,
) -> ProcedureResult<RowBatch> {
    let source = EntityId::new(source_id as u64);
    let target = EntityId::new(target_id as u64);

    // Build the shortest path finder
    let mut finder = ShortestPath::new(source, target, Direction::Both);

    if let Some(et) = edge_type {
        finder = finder.with_edge_type(et);
    }

    if let Some(depth) = max_depth {
        finder = finder.with_max_depth(depth as usize);
    }

    // Execute the search
    let result = finder.find(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec!["path".to_string(), "length".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    if let Some(path) = result {
        // Convert node IDs to array of integers
        let path_values: Vec<Value> =
            path.nodes.iter().map(|id| Value::Int(id.as_u64() as i64)).collect();

        let row = Row::new(
            Arc::clone(&schema),
            vec![Value::Array(path_values), Value::Int(path.length as i64)],
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
        let proc = ShortestPathProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.shortestPath");
        assert_eq!(sig.parameters.len(), 4);
        assert_eq!(sig.returns.len(), 2);
        assert_eq!(sig.required_param_count(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = ShortestPathProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["path", "length"]);
    }

    #[test]
    fn requires_context() {
        let proc = ShortestPathProcedure;
        assert!(proc.requires_context());
    }
}
