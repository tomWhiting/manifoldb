//! Closeness Centrality procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{ClosenessCentrality, ClosenessCentralityConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Closeness Centrality procedure.
///
/// Computes closeness centrality scores for all nodes in the graph.
/// Closeness centrality measures how close a node is to all other nodes
/// based on shortest path distances.
///
/// # Usage
///
/// ```sql
/// CALL algo.closenessCentrality() YIELD nodeId, score
/// CALL algo.closenessCentrality(true) YIELD nodeId, score
/// ```
///
/// # Parameters
///
/// - `harmonic` (optional, BOOLEAN): Use harmonic centrality (better for disconnected graphs), default false
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `score` (FLOAT): The closeness centrality score
pub struct ClosenessCentralityProcedure;

impl Procedure for ClosenessCentralityProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.closenessCentrality")
            .with_description(
                "Computes closeness centrality scores for all nodes (distance-based centrality)",
            )
            .with_parameter(ProcedureParameter::optional("harmonic", "BOOLEAN").with_description(
                "Use harmonic centrality for disconnected graphs (default false)",
            ))
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("score", "FLOAT")
                    .with_description("The closeness centrality score"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.closenessCentrality requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let harmonic = get_bool_or(&args, 0, false);

        // Build config
        let config = ClosenessCentralityConfig::default().with_harmonic(harmonic);

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "ClosenessCentrality execution requires direct transaction access. \
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

/// Helper function to execute Closeness Centrality with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_closeness_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    harmonic: bool,
) -> ProcedureResult<RowBatch> {
    let config = ClosenessCentralityConfig::default().with_harmonic(harmonic);

    let result = ClosenessCentrality::compute(tx, &config)
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
        let proc = ClosenessCentralityProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.closenessCentrality");
        assert_eq!(sig.parameters.len(), 1);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = ClosenessCentralityProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "score"]);
    }

    #[test]
    fn requires_context() {
        let proc = ClosenessCentralityProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn get_bool_or_default() {
        let args = ProcedureArgs::new(vec![]);
        assert!(!get_bool_or(&args, 0, false));
        assert!(get_bool_or(&args, 0, true));
    }

    #[test]
    fn get_bool_or_value() {
        let args = ProcedureArgs::new(vec![Value::Bool(true)]);
        assert!(get_bool_or(&args, 0, false));
    }
}
