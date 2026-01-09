//! Jaccard similarity procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::analytics::jaccard_similarity;

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Jaccard similarity procedure.
///
/// Computes Jaccard similarity between two nodes based on their neighborhoods.
///
/// Jaccard coefficient = |A ∩ B| / |A ∪ B|
///
/// Where A and B are the neighbor sets of the two nodes.
///
/// # Usage
///
/// ```sql
/// MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
/// CALL algo.jaccard(a, b) YIELD similarity
/// RETURN similarity
///
/// CALL algo.jaccard(1, 2, 'KNOWS') YIELD similarity
/// RETURN similarity
/// ```
///
/// # Parameters
///
/// - `node1` (required, INTEGER): The first node ID
/// - `node2` (required, INTEGER): The second node ID
/// - `edge_type` (optional, STRING): Filter by edge type
///
/// # Returns
///
/// - `similarity` (FLOAT): The Jaccard similarity coefficient (0.0 to 1.0)
pub struct JaccardProcedure;

impl Procedure for JaccardProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.jaccard")
            .with_description(
                "Computes Jaccard similarity between two nodes based on their neighborhoods",
            )
            .with_parameter(
                ProcedureParameter::required("node1", "INTEGER")
                    .with_description("The first node ID"),
            )
            .with_parameter(
                ProcedureParameter::required("node2", "INTEGER")
                    .with_description("The second node ID"),
            )
            .with_parameter(
                ProcedureParameter::optional("edge_type", "STRING")
                    .with_description("Optional edge type filter"),
            )
            .with_return(
                ReturnColumn::new("similarity", "FLOAT")
                    .with_description("The Jaccard similarity coefficient (0.0 to 1.0)"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.jaccard requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get required parameters
        let node1 = args.get_int(0, "node1")?;
        let node2 = args.get_int(1, "node2")?;
        let edge_type = args.get_string_opt(2);

        let _ = ctx;
        let _ = node1;
        let _ = node2;
        let _ = edge_type;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "Jaccard execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec!["similarity".to_string()]))
    }
}

/// Helper function to execute Jaccard similarity with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_jaccard_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    node1_id: i64,
    node2_id: i64,
    edge_type: Option<&str>,
) -> ProcedureResult<RowBatch> {
    let node1 = EntityId::new(node1_id as u64);
    let node2 = EntityId::new(node2_id as u64);

    let similarity = jaccard_similarity(tx, node1, node2, edge_type)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec!["similarity".to_string()]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    let row = Row::new(Arc::clone(&schema), vec![Value::Float(similarity)]);
    batch.push(row);

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature() {
        let proc = JaccardProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.jaccard");
        assert_eq!(sig.parameters.len(), 3);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.required_param_count(), 2);
    }

    #[test]
    fn output_schema() {
        let proc = JaccardProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["similarity"]);
    }

    #[test]
    fn requires_context() {
        let proc = JaccardProcedure;
        assert!(proc.requires_context());
    }
}
