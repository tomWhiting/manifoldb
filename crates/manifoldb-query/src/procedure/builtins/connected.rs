//! Connected Components procedures implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{ConnectedComponents, ConnectedComponentsConfig};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Connected Components procedure.
///
/// Finds weakly or strongly connected components in the graph.
/// Weak connectivity treats the graph as undirected.
///
/// # Usage
///
/// ```sql
/// CALL algo.connectedComponents() YIELD nodeId, componentId
/// CALL algo.connectedComponents('weak') YIELD nodeId, componentId
/// CALL algo.connectedComponents('strong') YIELD nodeId, componentId
/// ```
///
/// # Parameters
///
/// - `mode` (optional, STRING): Connectivity mode - 'weak' (default) or 'strong'
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `componentId` (INTEGER): The component ID assigned to the node
pub struct ConnectedComponentsProcedure;

impl Procedure for ConnectedComponentsProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.connectedComponents")
            .with_description("Finds weakly or strongly connected components in the graph")
            .with_parameter(
                ProcedureParameter::optional("mode", "STRING")
                    .with_description("Connectivity mode: 'weak' (default) or 'strong'"),
            )
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("componentId", "INTEGER")
                    .with_description("The component ID assigned to the node"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.connectedComponents requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional mode parameter
        let mode = args.get_string_opt(0).unwrap_or("weak");

        // Validate mode
        if mode != "weak" && mode != "strong" {
            return Err(ProcedureError::InvalidArgType {
                param: "mode".to_string(),
                expected: "'weak' or 'strong'".to_string(),
                actual: mode.to_string(),
            });
        }

        let _ = ctx;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "ConnectedComponents execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["nodeId".to_string(), "componentId".to_string()]))
    }
}

/// Strongly Connected Components procedure.
///
/// Finds strongly connected components in a directed graph.
/// In a strongly connected component, every node is reachable from
/// every other node following edge directions.
///
/// # Usage
///
/// ```sql
/// CALL algo.stronglyConnectedComponents() YIELD nodeId, componentId
/// ```
///
/// # Returns
///
/// - `nodeId` (INTEGER): The node ID
/// - `componentId` (INTEGER): The component ID assigned to the node
pub struct StronglyConnectedComponentsProcedure;

impl Procedure for StronglyConnectedComponentsProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.stronglyConnectedComponents")
            .with_description("Finds strongly connected components in a directed graph")
            .with_return(ReturnColumn::new("nodeId", "INTEGER").with_description("The node ID"))
            .with_return(
                ReturnColumn::new("componentId", "INTEGER")
                    .with_description("The component ID assigned to the node"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.stronglyConnectedComponents requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        _args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        let _ = ctx;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "StronglyConnectedComponents execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["nodeId".to_string(), "componentId".to_string()]))
    }
}

/// Helper function to execute Connected Components with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
///
/// # Arguments
///
/// * `tx` - The transaction to use for graph access
/// * `mode` - Connectivity mode: "weak" or "strong"
pub fn execute_connected_components_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    mode: &str,
) -> ProcedureResult<RowBatch> {
    let config = ConnectedComponentsConfig::default();

    let result = match mode {
        "weak" => ConnectedComponents::weakly_connected(tx, &config)
            .map_err(|e| ProcedureError::GraphError(e.to_string()))?,
        "strong" => ConnectedComponents::strongly_connected(tx, &config)
            .map_err(|e| ProcedureError::GraphError(e.to_string()))?,
        _ => {
            return Err(ProcedureError::InvalidArgType {
                param: "mode".to_string(),
                expected: "'weak' or 'strong'".to_string(),
                actual: mode.to_string(),
            })
        }
    };

    // Build result rows
    let schema = Arc::new(Schema::new(vec!["nodeId".to_string(), "componentId".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    for (node_id, component_id) in result.assignments {
        let row = Row::new(
            Arc::clone(&schema),
            vec![Value::Int(node_id.as_u64() as i64), Value::Int(component_id as i64)],
        );
        batch.push(row);
    }

    Ok(batch)
}

/// Helper function to execute Strongly Connected Components with a transaction.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_strongly_connected_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
) -> ProcedureResult<RowBatch> {
    execute_connected_components_with_tx(tx, "strong")
}

#[cfg(test)]
mod tests {
    use super::*;

    mod connected_components {
        use super::*;

        #[test]
        fn signature() {
            let proc = ConnectedComponentsProcedure;
            let sig = proc.signature();
            assert_eq!(sig.name, "algo.connectedComponents");
            assert_eq!(sig.parameters.len(), 1);
            assert_eq!(sig.returns.len(), 2);
        }

        #[test]
        fn output_schema() {
            let proc = ConnectedComponentsProcedure;
            let schema = proc.output_schema();
            assert_eq!(schema.columns(), vec!["nodeId", "componentId"]);
        }

        #[test]
        fn requires_context() {
            let proc = ConnectedComponentsProcedure;
            assert!(proc.requires_context());
        }

        #[test]
        fn signature_parameters() {
            let proc = ConnectedComponentsProcedure;
            let sig = proc.signature();

            // Check parameter details
            let mode_param = &sig.parameters[0];
            assert_eq!(mode_param.name, "mode");
            assert_eq!(mode_param.type_hint, "STRING");
            assert!(!mode_param.required);
        }

        #[test]
        fn signature_returns() {
            let proc = ConnectedComponentsProcedure;
            let sig = proc.signature();

            // Check return column details
            let node_id_col = &sig.returns[0];
            assert_eq!(node_id_col.name, "nodeId");
            assert_eq!(node_id_col.type_hint, "INTEGER");

            let component_id_col = &sig.returns[1];
            assert_eq!(component_id_col.name, "componentId");
            assert_eq!(component_id_col.type_hint, "INTEGER");
        }
    }

    mod strongly_connected_components {
        use super::*;

        #[test]
        fn signature() {
            let proc = StronglyConnectedComponentsProcedure;
            let sig = proc.signature();
            assert_eq!(sig.name, "algo.stronglyConnectedComponents");
            assert_eq!(sig.parameters.len(), 0);
            assert_eq!(sig.returns.len(), 2);
        }

        #[test]
        fn output_schema() {
            let proc = StronglyConnectedComponentsProcedure;
            let schema = proc.output_schema();
            assert_eq!(schema.columns(), vec!["nodeId", "componentId"]);
        }

        #[test]
        fn requires_context() {
            let proc = StronglyConnectedComponentsProcedure;
            assert!(proc.requires_context());
        }

        #[test]
        fn signature_returns() {
            let proc = StronglyConnectedComponentsProcedure;
            let sig = proc.signature();

            // Check return column details
            let node_id_col = &sig.returns[0];
            assert_eq!(node_id_col.name, "nodeId");
            assert_eq!(node_id_col.type_hint, "INTEGER");

            let component_id_col = &sig.returns[1];
            assert_eq!(component_id_col.name, "componentId");
            assert_eq!(component_id_col.type_hint, "INTEGER");
        }
    }
}
