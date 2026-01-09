//! Cosine similarity procedure implementation.

use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_graph::analytics::cosine_similarity_properties;

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Cosine similarity procedure.
///
/// Computes Cosine similarity between two nodes based on their property values.
///
/// The specified properties are treated as dimensions of a vector, and
/// cosine similarity is computed between the two vectors.
///
/// # Usage
///
/// ```sql
/// MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
/// CALL algo.cosine(a, b, ['age', 'income', 'score']) YIELD similarity
/// RETURN similarity
///
/// CALL algo.cosine(1, 2, ['x', 'y', 'z']) YIELD similarity
/// RETURN similarity
/// ```
///
/// # Parameters
///
/// - `node1` (required, INTEGER): The first node ID
/// - `node2` (required, INTEGER): The second node ID
/// - `properties` (required, ARRAY): List of property names to use as vector dimensions
///
/// # Returns
///
/// - `similarity` (FLOAT): The Cosine similarity (-1.0 to 1.0, typically 0.0 to 1.0)
pub struct CosineProcedure;

impl Procedure for CosineProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.cosine")
            .with_description(
                "Computes Cosine similarity between two nodes based on their property values",
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
                ProcedureParameter::required("properties", "ARRAY")
                    .with_description("List of property names to use as vector dimensions"),
            )
            .with_return(
                ReturnColumn::new("similarity", "FLOAT")
                    .with_description("The Cosine similarity (-1.0 to 1.0)"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.cosine requires graph storage context".to_string(),
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
        let properties = args.get_array(2, "properties")?;

        let _ = ctx;
        let _ = node1;
        let _ = node2;
        let _ = properties;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "Cosine execution requires direct transaction access. \
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

/// Helper function to execute Cosine similarity with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_cosine_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    node1_id: i64,
    node2_id: i64,
    properties: &[Value],
) -> ProcedureResult<RowBatch> {
    let node1 = EntityId::new(node1_id as u64);
    let node2 = EntityId::new(node2_id as u64);

    // Convert property values to strings
    let property_names: Vec<String> =
        properties.iter().filter_map(|v| v.as_str().map(String::from)).collect();

    if property_names.is_empty() {
        return Err(ProcedureError::ExecutionFailed(
            "Properties array must contain at least one property name".to_string(),
        ));
    }

    let similarity = cosine_similarity_properties(tx, node1, node2, &property_names)
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
        let proc = CosineProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.cosine");
        assert_eq!(sig.parameters.len(), 3);
        assert_eq!(sig.returns.len(), 1);
        assert_eq!(sig.required_param_count(), 3);
    }

    #[test]
    fn output_schema() {
        let proc = CosineProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["similarity"]);
    }

    #[test]
    fn requires_context() {
        let proc = CosineProcedure;
        assert!(proc.requires_context());
    }
}
