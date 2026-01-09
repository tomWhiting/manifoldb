//! Eigenvector Centrality procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{EigenvectorCentrality, EigenvectorCentralityConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Eigenvector Centrality procedure.
///
/// Computes eigenvector centrality scores for all nodes in the graph.
/// Eigenvector centrality measures node importance based on connections
/// to other important nodes (influence networks).
///
/// # Usage
///
/// ```sql
/// CALL algo.eigenvectorCentrality() YIELD nodeId, score
/// CALL algo.eigenvectorCentrality(100, 0.000001) YIELD nodeId, score
/// ```
///
/// # Parameters
///
/// - `maxIterations` (optional, INTEGER): Maximum iterations, default 100
/// - `tolerance` (optional, FLOAT): Convergence tolerance, default 1e-6
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `score` (FLOAT): The eigenvector centrality score
pub struct EigenvectorCentralityProcedure;

impl Procedure for EigenvectorCentralityProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.eigenvectorCentrality")
            .with_description(
                "Computes eigenvector centrality scores for all nodes (influence networks)",
            )
            .with_parameter(
                ProcedureParameter::optional("maxIterations", "INTEGER")
                    .with_description("Maximum iterations (default 100)"),
            )
            .with_parameter(
                ProcedureParameter::optional("tolerance", "FLOAT")
                    .with_description("Convergence tolerance (default 1e-6)"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("score", "FLOAT")
                    .with_description("The eigenvector centrality score"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.eigenvectorCentrality requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let max_iterations = args.get_int_opt(0).unwrap_or(100) as usize;
        let tolerance = args.get_float_or(1, 1e-6);

        // Build config
        let config = EigenvectorCentralityConfig::default()
            .with_max_iterations(max_iterations)
            .with_tolerance(tolerance);

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "EigenvectorCentrality execution requires direct transaction access. \
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

/// Helper function to execute Eigenvector Centrality with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_eigenvector_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    max_iterations: usize,
    tolerance: f64,
) -> ProcedureResult<RowBatch> {
    let config = EigenvectorCentralityConfig::default()
        .with_max_iterations(max_iterations)
        .with_tolerance(tolerance);

    let result = EigenvectorCentrality::compute(tx, &config)
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
        let proc = EigenvectorCentralityProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.eigenvectorCentrality");
        assert_eq!(sig.parameters.len(), 2);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = EigenvectorCentralityProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "score"]);
    }

    #[test]
    fn requires_context() {
        let proc = EigenvectorCentralityProcedure;
        assert!(proc.requires_context());
    }
}
