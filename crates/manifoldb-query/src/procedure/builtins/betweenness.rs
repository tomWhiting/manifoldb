//! Betweenness Centrality procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{BetweennessCentrality, BetweennessCentralityConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Betweenness Centrality procedure.
///
/// Computes betweenness centrality scores for all nodes in the graph.
/// Betweenness centrality measures the extent to which a node lies on paths
/// between other nodes (bridge/bottleneck detection).
///
/// # Usage
///
/// ```sql
/// CALL algo.betweennessCentrality() YIELD nodeId, score
/// CALL algo.betweennessCentrality(true, false) YIELD nodeId, score
/// ```
///
/// # Parameters
///
/// - `normalized` (optional, BOOLEAN): Whether to normalize scores to [0, 1], default true
/// - `endpoints` (optional, BOOLEAN): Include endpoints in calculation, default false
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `score` (FLOAT): The betweenness centrality score
pub struct BetweennessCentralityProcedure;

impl Procedure for BetweennessCentralityProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.betweennessCentrality")
            .with_description(
                "Computes betweenness centrality scores for all nodes (bridge/bottleneck detection)",
            )
            .with_parameter(
                ProcedureParameter::optional("normalized", "BOOLEAN")
                    .with_description("Whether to normalize scores to [0, 1] (default true)"),
            )
            .with_parameter(
                ProcedureParameter::optional("endpoints", "BOOLEAN")
                    .with_description("Include endpoints in calculation (default false)"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("score", "FLOAT")
                    .with_description("The betweenness centrality score"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.betweennessCentrality requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let normalized = get_bool_or(&args, 0, true);
        let endpoints = get_bool_or(&args, 1, false);

        // Build config
        let config = BetweennessCentralityConfig::default()
            .with_normalize(normalized)
            .with_include_endpoints(endpoints);

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "BetweennessCentrality execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["nodeId".to_string(), "score".to_string()]))
    }
}

/// Helper function to get a boolean argument with a default value.
fn get_bool_or(args: &ProcedureArgs, index: usize, default: bool) -> bool {
    match args.get(index) {
        Some(Value::Bool(b)) => *b,
        _ => default,
    }
}

/// Helper function to execute Betweenness Centrality with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_betweenness_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    normalized: bool,
    endpoints: bool,
) -> ProcedureResult<RowBatch> {
    let config = BetweennessCentralityConfig::default()
        .with_normalize(normalized)
        .with_include_endpoints(endpoints);

    let result = BetweennessCentrality::compute(tx, &config)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec!["nodeId".to_string(), "score".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    for (node_id, score) in result.scores {
        let row = Row::new(
            Arc::clone(&schema),
            vec![Value::Int(node_id.as_u64() as i64), Value::Float(score)],
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
        let proc = BetweennessCentralityProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.betweennessCentrality");
        assert_eq!(sig.parameters.len(), 2);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = BetweennessCentralityProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "score"]);
    }

    #[test]
    fn requires_context() {
        let proc = BetweennessCentralityProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn get_bool_or_default() {
        let args = ProcedureArgs::new(vec![]);
        assert!(get_bool_or(&args, 0, true));
        assert!(!get_bool_or(&args, 0, false));
    }

    #[test]
    fn get_bool_or_value() {
        let args = ProcedureArgs::new(vec![Value::Bool(false)]);
        assert!(!get_bool_or(&args, 0, true));
    }
}
