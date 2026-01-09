//! Label Propagation community detection procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{CommunityDetection, CommunityDetectionConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Label Propagation community detection procedure.
///
/// Detects communities using the Label Propagation Algorithm (LPA).
/// This is a fast, near-linear time algorithm that doesn't require
/// knowing the number of communities in advance.
///
/// # Usage
///
/// ```sql
/// CALL algo.labelPropagation() YIELD nodeId, communityId
/// CALL algo.labelPropagation(100) YIELD nodeId, communityId
/// ```
///
/// # Parameters
///
/// - `maxIterations` (optional, INTEGER): Maximum number of iterations, default 100
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `communityId` (INTEGER): The community ID assigned to the node
pub struct LabelPropagationProcedure;

impl Procedure for LabelPropagationProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.labelPropagation")
            .with_description("Detects communities using the Label Propagation Algorithm (LPA)")
            .with_parameter(
                ProcedureParameter::optional("maxIterations", "INTEGER")
                    .with_description("Maximum number of iterations (default 100)"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("communityId", "INTEGER")
                    .with_description("The community ID assigned to the node"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.labelPropagation requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let max_iterations = args.get_int_opt(0).unwrap_or(100) as usize;

        // Build config
        let config = CommunityDetectionConfig::default().with_max_iterations(max_iterations);

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "LabelPropagation execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["nodeId".to_string(), "communityId".to_string()]))
    }
}

/// Helper function to execute Label Propagation with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_label_propagation_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    max_iterations: usize,
) -> ProcedureResult<RowBatch> {
    let config = CommunityDetectionConfig::default().with_max_iterations(max_iterations);

    let result = CommunityDetection::label_propagation(tx, &config)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec!["nodeId".to_string(), "communityId".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    for (node_id, community_id) in result.assignments {
        let row = Row::new(
            Arc::clone(&schema),
            vec![Value::Int(node_id.as_u64() as i64), Value::Int(community_id as i64)],
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
        let proc = LabelPropagationProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.labelPropagation");
        assert_eq!(sig.parameters.len(), 1);
        assert_eq!(sig.returns.len(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = LabelPropagationProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "communityId"]);
    }

    #[test]
    fn requires_context() {
        let proc = LabelPropagationProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn signature_parameters() {
        let proc = LabelPropagationProcedure;
        let sig = proc.signature();

        // Check parameter details
        let max_iter_param = &sig.parameters[0];
        assert_eq!(max_iter_param.name, "maxIterations");
        assert_eq!(max_iter_param.type_hint, "INTEGER");
        assert!(!max_iter_param.required);
    }

    #[test]
    fn signature_returns() {
        let proc = LabelPropagationProcedure;
        let sig = proc.signature();

        // Check return column details
        let node_id_col = &sig.returns[0];
        assert_eq!(node_id_col.name, "nodeId");
        assert_eq!(node_id_col.type_hint, "INTEGER");

        let community_id_col = &sig.returns[1];
        assert_eq!(community_id_col.name, "communityId");
        assert_eq!(community_id_col.type_hint, "INTEGER");
    }
}
