//! Degree Centrality procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{DegreeCentrality, DegreeCentralityConfig};
use manifoldb_graph::traversal::Direction;

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Degree Centrality procedure.
///
/// Computes degree centrality scores for all nodes in the graph.
/// Degree centrality measures importance based on the number of direct connections.
///
/// # Usage
///
/// ```sql
/// CALL algo.degreeCentrality() YIELD nodeId, inDegree, outDegree, totalDegree
/// CALL algo.degreeCentrality('out') YIELD nodeId, inDegree, outDegree, totalDegree
/// ```
///
/// # Parameters
///
/// - `direction` (optional, STRING): Direction to count - 'in', 'out', or 'both' (default 'both')
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `inDegree` (INTEGER): Number of incoming edges
/// - `outDegree` (INTEGER): Number of outgoing edges
/// - `totalDegree` (INTEGER): Total degree (in + out)
pub struct DegreeCentralityProcedure;

impl Procedure for DegreeCentralityProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.degreeCentrality")
            .with_description("Computes degree centrality scores for all nodes (connection counts)")
            .with_parameter(
                ProcedureParameter::optional("direction", "STRING").with_description(
                    "Direction to count: 'in', 'out', or 'both' (default 'both')",
                ),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("inDegree", "INTEGER")
                    .with_description("Number of incoming edges"),
            )
            .with_return(
                ReturnColumn::new("outDegree", "INTEGER")
                    .with_description("Number of outgoing edges"),
            )
            .with_return(
                ReturnColumn::new("totalDegree", "INTEGER")
                    .with_description("Total degree (in + out)"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.degreeCentrality requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional direction parameter
        let direction = parse_direction(args.get_string_opt(0))?;

        // Build config
        let config = DegreeCentralityConfig::default().with_direction(direction);

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "DegreeCentrality execution requires direct transaction access. \
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
            "inDegree".to_string(),
            "outDegree".to_string(),
            "totalDegree".to_string(),
        ]))
    }
}

/// Parse direction string to Direction enum.
fn parse_direction(direction_str: Option<&str>) -> ProcedureResult<Direction> {
    match direction_str {
        None | Some("both" | "BOTH") => Ok(Direction::Both),
        Some("in" | "IN" | "incoming" | "INCOMING") => Ok(Direction::Incoming),
        Some("out" | "OUT" | "outgoing" | "OUTGOING") => Ok(Direction::Outgoing),
        Some(other) => Err(ProcedureError::InvalidArgType {
            param: "direction".to_string(),
            expected: "'in', 'out', or 'both'".to_string(),
            actual: other.to_string(),
        }),
    }
}

/// Helper function to execute Degree Centrality with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
///
/// Unlike other centrality algorithms, this returns in/out/total degree for each node.
pub fn execute_degree_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    direction_str: Option<&str>,
) -> ProcedureResult<RowBatch> {
    // We compute all three directions to return comprehensive degree info
    let config_in = DegreeCentralityConfig::default().with_direction(Direction::Incoming);
    let config_out = DegreeCentralityConfig::default().with_direction(Direction::Outgoing);

    let result_in = DegreeCentrality::compute(tx, &config_in)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;
    let result_out = DegreeCentrality::compute(tx, &config_out)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Validate direction parameter (for error reporting)
    let _ = parse_direction(direction_str)?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec![
        "nodeId".to_string(),
        "inDegree".to_string(),
        "outDegree".to_string(),
        "totalDegree".to_string(),
    ]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    // Collect all node IDs (use result_in as reference)
    for (node_id, in_degree) in &result_in.scores {
        let out_degree = result_out.scores.get(node_id).copied().unwrap_or(0.0);
        let total_degree = in_degree + out_degree;

        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Int(node_id.as_u64() as i64),
                Value::Int(*in_degree as i64),
                Value::Int(out_degree as i64),
                Value::Int(total_degree as i64),
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
        let proc = DegreeCentralityProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.degreeCentrality");
        assert_eq!(sig.parameters.len(), 1);
        assert_eq!(sig.returns.len(), 4);
    }

    #[test]
    fn output_schema() {
        let proc = DegreeCentralityProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "inDegree", "outDegree", "totalDegree"]);
    }

    #[test]
    fn requires_context() {
        let proc = DegreeCentralityProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn parse_direction_valid() {
        assert_eq!(parse_direction(None).unwrap(), Direction::Both);
        assert_eq!(parse_direction(Some("both")).unwrap(), Direction::Both);
        assert_eq!(parse_direction(Some("BOTH")).unwrap(), Direction::Both);
        assert_eq!(parse_direction(Some("in")).unwrap(), Direction::Incoming);
        assert_eq!(parse_direction(Some("IN")).unwrap(), Direction::Incoming);
        assert_eq!(parse_direction(Some("incoming")).unwrap(), Direction::Incoming);
        assert_eq!(parse_direction(Some("out")).unwrap(), Direction::Outgoing);
        assert_eq!(parse_direction(Some("OUT")).unwrap(), Direction::Outgoing);
        assert_eq!(parse_direction(Some("outgoing")).unwrap(), Direction::Outgoing);
    }

    #[test]
    fn parse_direction_invalid() {
        assert!(parse_direction(Some("invalid")).is_err());
    }
}
