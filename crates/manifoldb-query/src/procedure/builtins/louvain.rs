//! Louvain community detection procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{LouvainCommunityDetection, LouvainConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Louvain community detection procedure.
///
/// Detects communities using the Louvain algorithm, which optimizes
/// modularity through iterative local optimization and graph aggregation.
///
/// # Usage
///
/// ```sql
/// -- Basic usage
/// CALL algo.louvain() YIELD nodeId, communityId
///
/// -- With max iterations
/// CALL algo.louvain(10) YIELD nodeId, communityId
///
/// -- With max iterations and tolerance
/// CALL algo.louvain(10, 0.0001) YIELD nodeId, communityId
///
/// -- With all parameters: maxIterations, tolerance, weightProperty
/// CALL algo.louvain(10, 0.0001, 'strength') YIELD nodeId, communityId, modularity
/// ```
///
/// # Parameters
///
/// - `maxIterations` (optional, INTEGER): Maximum number of passes, default 10
/// - `tolerance` (optional, FLOAT): Minimum modularity improvement to continue, default 0.0001
/// - `weightProperty` (optional, STRING): Edge property name for weights
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `communityId` (INTEGER): The community ID assigned to the node
/// - `modularity` (FLOAT): The final modularity score (same for all rows)
pub struct LouvainProcedure;

impl Procedure for LouvainProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.louvain")
            .with_description(
                "Detects communities using the Louvain algorithm (modularity optimization)",
            )
            .with_parameter(
                ProcedureParameter::optional("maxIterations", "INTEGER")
                    .with_description("Maximum number of passes (default 10)"),
            )
            .with_parameter(
                ProcedureParameter::optional("tolerance", "FLOAT").with_description(
                    "Minimum modularity improvement to continue (default 0.0001)",
                ),
            )
            .with_parameter(
                ProcedureParameter::optional("weightProperty", "STRING")
                    .with_description("Edge property name for weights"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("communityId", "INTEGER")
                    .with_description("The community ID assigned to the node"),
            )
            .with_return(
                ReturnColumn::new("modularity", "FLOAT")
                    .with_description("The final modularity score"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.louvain requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let max_iterations = args.get_int_opt(0).unwrap_or(10) as usize;
        let tolerance = args.get_float_or(1, 0.0001);
        let weight_property = args.get_string_opt(2).map(String::from);

        // Build config
        let mut config =
            LouvainConfig::default().with_max_iterations(max_iterations).with_tolerance(tolerance);

        if let Some(prop) = weight_property {
            config = config.with_weight_property(prop);
        }

        let _ = ctx;
        let _ = config;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "Louvain execution requires direct transaction access. \
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
            "communityId".to_string(),
            "modularity".to_string(),
        ]))
    }
}

/// Helper function to execute Louvain with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
///
/// # Arguments
///
/// * `tx` - The transaction to use for graph access
/// * `max_iterations` - Maximum number of passes
/// * `tolerance` - Minimum modularity improvement threshold
/// * `weight_property` - Optional edge property name for weights
pub fn execute_louvain_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    max_iterations: usize,
    tolerance: f64,
    weight_property: Option<&str>,
) -> ProcedureResult<RowBatch> {
    let mut config =
        LouvainConfig::default().with_max_iterations(max_iterations).with_tolerance(tolerance);

    if let Some(prop) = weight_property {
        config = config.with_weight_property(prop);
    }

    let result = LouvainCommunityDetection::detect_communities(tx, &config)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec![
        "nodeId".to_string(),
        "communityId".to_string(),
        "modularity".to_string(),
    ]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    let modularity = result.modularity;

    for (node_id, community_id) in result.assignments {
        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Int(node_id.as_u64() as i64),
                Value::Int(community_id as i64),
                Value::Float(modularity),
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
        let proc = LouvainProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.louvain");
        assert_eq!(sig.parameters.len(), 3);
        assert_eq!(sig.returns.len(), 3);
    }

    #[test]
    fn output_schema() {
        let proc = LouvainProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["nodeId", "communityId", "modularity"]);
    }

    #[test]
    fn requires_context() {
        let proc = LouvainProcedure;
        assert!(proc.requires_context());
    }

    #[test]
    fn signature_parameters() {
        let proc = LouvainProcedure;
        let sig = proc.signature();

        // Check parameter details
        let max_iter_param = &sig.parameters[0];
        assert_eq!(max_iter_param.name, "maxIterations");
        assert_eq!(max_iter_param.type_hint, "INTEGER");
        assert!(!max_iter_param.required);

        let tolerance_param = &sig.parameters[1];
        assert_eq!(tolerance_param.name, "tolerance");
        assert_eq!(tolerance_param.type_hint, "FLOAT");
        assert!(!tolerance_param.required);

        let weight_param = &sig.parameters[2];
        assert_eq!(weight_param.name, "weightProperty");
        assert_eq!(weight_param.type_hint, "STRING");
        assert!(!weight_param.required);
    }

    #[test]
    fn signature_returns() {
        let proc = LouvainProcedure;
        let sig = proc.signature();

        // Check return column details
        let node_id_col = &sig.returns[0];
        assert_eq!(node_id_col.name, "nodeId");
        assert_eq!(node_id_col.type_hint, "INTEGER");

        let community_id_col = &sig.returns[1];
        assert_eq!(community_id_col.name, "communityId");
        assert_eq!(community_id_col.type_hint, "INTEGER");

        let modularity_col = &sig.returns[2];
        assert_eq!(modularity_col.name, "modularity");
        assert_eq!(modularity_col.type_hint, "FLOAT");
    }
}
