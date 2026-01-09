//! PageRank procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{PageRank, PageRankConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// PageRank procedure.
///
/// Computes PageRank scores for all nodes in the graph.
///
/// # Usage
///
/// ```sql
/// CALL algo.pageRank() YIELD nodeId, score
/// CALL algo.pageRank(0.85, 100) YIELD nodeId, score
/// ```
///
/// # Parameters
///
/// - `damping_factor` (optional, FLOAT): Damping factor, default 0.85
/// - `max_iterations` (optional, INTEGER): Maximum iterations, default 100
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `score` (FLOAT): The PageRank score
pub struct PageRankProcedure;

impl Procedure for PageRankProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.pageRank")
            .with_description("Computes PageRank scores for all nodes in the graph")
            .with_parameter(
                ProcedureParameter::optional("damping_factor", "FLOAT")
                    .with_description("Damping factor (default 0.85)"),
            )
            .with_parameter(
                ProcedureParameter::optional("max_iterations", "INTEGER")
                    .with_description("Maximum iterations (default 100)"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(ReturnColumn::new("score", "FLOAT").with_description("The PageRank score"))
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.pageRank requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let damping_factor = args.get_float_or(0, 0.85);
        let max_iterations = args.get_int_opt(1).unwrap_or(100) as usize;

        // Build PageRank config
        let config = PageRankConfig::default()
            .with_damping_factor(damping_factor)
            .with_max_iterations(max_iterations);

        // We need to access the graph storage through the context.
        // The ExecutionContext has a graph() method that returns a GraphAccessor,
        // but PageRank needs a Transaction. We need to work around this.
        //
        // For now, we'll return an error indicating this needs to be executed
        // at a higher level where the transaction is available.
        //
        // In a real implementation, the procedure execution would happen in the
        // main manifoldb crate where we have access to the transaction.
        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "PageRank execution requires direct transaction access. \
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

/// Helper function to execute PageRank with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_pagerank_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    damping_factor: f64,
    max_iterations: usize,
) -> ProcedureResult<RowBatch> {
    let config = PageRankConfig::default()
        .with_damping_factor(damping_factor)
        .with_max_iterations(max_iterations);

    let result =
        PageRank::compute(tx, &config).map_err(|e| ProcedureError::GraphError(e.to_string()))?;

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
        let proc = PageRankProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.pageRank");
        assert_eq!(sig.parameters.len(), 2);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = PageRankProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "score"]);
    }

    #[test]
    fn requires_context() {
        let proc = PageRankProcedure;
        assert!(proc.requires_context());
    }
}
