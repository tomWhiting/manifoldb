//! DFS (Depth-First Search) procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::traversal::{DfsTraversal, Direction};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// DFS traversal procedure.
///
/// Performs a depth-first search starting from a given node, exploring
/// as far as possible along each branch before backtracking.
///
/// # Usage
///
/// ```sql
/// CALL algo.dfs(startNode, edge_type, direction, maxDepth) YIELD node, depth, path
/// ```
///
/// # Parameters
///
/// - `startNode` (required, INTEGER): The starting node ID
/// - `edge_type` (optional, STRING): Filter by edge type (null for all types)
/// - `direction` (optional, STRING): 'OUTGOING', 'INCOMING', or 'BOTH' (default: 'BOTH')
/// - `maxDepth` (optional, INTEGER): Maximum depth to traverse (default: unlimited)
///
/// # Returns
///
/// - `node` (INTEGER): Node ID of each visited node
/// - `depth` (INTEGER): Depth at which the node was discovered
/// - `path` (ARRAY): Array of node IDs from start to this node
pub struct DfsProcedure;

impl Procedure for DfsProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.dfs")
            .with_description("Depth-first search traversal from a starting node")
            .with_parameter(
                ProcedureParameter::required("startNode", "INTEGER")
                    .with_description("The starting node ID"),
            )
            .with_parameter(
                ProcedureParameter::optional("edge_type", "STRING")
                    .with_description("Edge type to traverse (null for all types)"),
            )
            .with_parameter(
                ProcedureParameter::optional("direction", "STRING")
                    .with_description("Traversal direction: OUTGOING, INCOMING, or BOTH"),
            )
            .with_parameter(
                ProcedureParameter::optional("maxDepth", "INTEGER")
                    .with_description("Maximum depth to traverse"),
            )
            .with_return(
                ReturnColumn::new("node", "INTEGER").with_description("Node ID of visited node"),
            )
            .with_return(ReturnColumn::new("depth", "INTEGER").with_description("Discovery depth"))
            .with_return(
                ReturnColumn::new("path", "ARRAY")
                    .with_description("Path from start node to this node"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed("algo.dfs requires graph storage context".to_string()))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        _ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get required parameters
        let _start_id = args.get_int(0, "startNode")?;
        let _edge_type = args.get_string_opt(1);
        let _direction_str = args.get_string_opt(2);
        let _max_depth = args.get_int_opt(3);

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "DFS execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["node".to_string(), "depth".to_string(), "path".to_string()]))
    }
}

/// Parse direction string to Direction enum.
fn parse_direction(direction_str: Option<&str>) -> Direction {
    match direction_str.map(|s| s.to_uppercase()).as_deref() {
        Some("OUTGOING") => Direction::Outgoing,
        Some("INCOMING") => Direction::Incoming,
        Some("BOTH") | None => Direction::Both,
        _ => Direction::Both, // Default to BOTH for invalid values
    }
}

/// Helper function to execute DFS with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_dfs_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    start_id: i64,
    edge_type: Option<&str>,
    direction_str: Option<&str>,
    max_depth: Option<i64>,
) -> ProcedureResult<RowBatch> {
    let start = EntityId::new(start_id as u64);
    let direction = parse_direction(direction_str);

    // Build the DFS traversal
    let mut traversal = DfsTraversal::new(start, direction).with_path_tracking();

    if let Some(et) = edge_type {
        traversal = traversal.with_edge_type(et);
    }

    if let Some(depth) = max_depth {
        traversal = traversal.with_max_depth(depth as usize);
    }

    // Execute the traversal
    let results = traversal.execute(tx).map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema =
        Arc::new(Schema::new(vec!["node".to_string(), "depth".to_string(), "path".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    for result in results {
        // Convert path to array of integers
        let path_values: Vec<Value> =
            result.path.iter().map(|id| Value::Int(id.as_u64() as i64)).collect();

        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Int(result.node.as_u64() as i64),
                Value::Int(result.depth as i64),
                Value::Array(path_values),
            ],
        );
        batch.push(row);
    }

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature() {
        let proc = DfsProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.dfs");
        assert_eq!(sig.parameters.len(), 4);
        assert_eq!(sig.returns.len(), 3);
        assert_eq!(sig.required_param_count(), 1);
    }

    #[test]
    fn output_schema() {
        let proc = DfsProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["node", "depth", "path"]);
    }

    #[test]
    fn requires_context() {
        let proc = DfsProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn parse_direction_variants() {
        assert_eq!(parse_direction(Some("OUTGOING")), Direction::Outgoing);
        assert_eq!(parse_direction(Some("outgoing")), Direction::Outgoing);
        assert_eq!(parse_direction(Some("INCOMING")), Direction::Incoming);
        assert_eq!(parse_direction(Some("incoming")), Direction::Incoming);
        assert_eq!(parse_direction(Some("BOTH")), Direction::Both);
        assert_eq!(parse_direction(Some("both")), Direction::Both);
        assert_eq!(parse_direction(None), Direction::Both);
        assert_eq!(parse_direction(Some("invalid")), Direction::Both);
    }
}
