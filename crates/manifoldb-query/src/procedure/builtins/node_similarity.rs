//! Node similarity procedure implementation.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_graph::analytics::{NodeSimilarity, NodeSimilarityConfig, SimilarityAlgorithm};

use crate::exec::{ExecutionContext, Row, RowBatch, Schema};
use crate::procedure::signature::ProcedureParameter;
use crate::procedure::traits::Procedure;
use crate::procedure::{
    ProcedureArgs, ProcedureError, ProcedureResult, ProcedureSignature, ReturnColumn,
};

/// Node similarity procedure.
///
/// Computes similarity scores for all pairs of nodes based on their neighborhoods.
/// Uses Jaccard similarity by default.
///
/// # Usage
///
/// ```sql
/// -- Basic usage with defaults
/// CALL algo.nodeSimilarity() YIELD node1, node2, similarity
/// RETURN node1, node2, similarity
/// ORDER BY similarity DESC
///
/// -- With label and edge type filters
/// CALL algo.nodeSimilarity('Person', 'KNOWS', {topK: 10})
/// YIELD node1, node2, similarity
/// RETURN node1, node2, similarity
/// ```
///
/// # Parameters
///
/// - `label` (optional, STRING): Only consider nodes with this label
/// - `edge_type` (optional, STRING): Only consider edges of this type
/// - `topK` (optional, INTEGER): Return only top K most similar pairs
/// - `similarityCutoff` (optional, FLOAT): Minimum similarity threshold
///
/// # Returns
///
/// - `node1` (INTEGER): First node ID
/// - `node2` (INTEGER): Second node ID
/// - `similarity` (FLOAT): Similarity score (0.0 to 1.0)
pub struct NodeSimilarityProcedure;

impl Procedure for NodeSimilarityProcedure {
    fn signature(&self) -> ProcedureSignature {
        ProcedureSignature::new("algo.nodeSimilarity")
            .with_description("Computes similarity scores for all pairs of nodes")
            .with_parameter(
                ProcedureParameter::optional("label", "STRING")
                    .with_description("Only consider nodes with this label"),
            )
            .with_parameter(
                ProcedureParameter::optional("edge_type", "STRING")
                    .with_description("Only consider edges of this type"),
            )
            .with_parameter(
                ProcedureParameter::optional("topK", "INTEGER")
                    .with_description("Return only top K most similar pairs"),
            )
            .with_parameter(
                ProcedureParameter::optional("similarityCutoff", "FLOAT")
                    .with_description("Minimum similarity threshold (default: 0.0)"),
            )
            .with_return(ReturnColumn::new("node1", "INTEGER").with_description("First node ID"))
            .with_return(ReturnColumn::new("node2", "INTEGER").with_description("Second node ID"))
            .with_return(
                ReturnColumn::new("similarity", "FLOAT")
                    .with_description("Similarity score (0.0 to 1.0)"),
            )
    }

    fn execute(&self, _args: ProcedureArgs) -> ProcedureResult<RowBatch> {
        Err(ProcedureError::ExecutionFailed(
            "algo.nodeSimilarity requires graph storage context".to_string(),
        ))
    }

    fn execute_with_context(
        &self,
        args: ProcedureArgs,
        ctx: &ExecutionContext,
    ) -> ProcedureResult<RowBatch> {
        // Get optional parameters
        let label = args.get_string_opt(0);
        let edge_type = args.get_string_opt(1);
        let top_k = args.get_int_opt(2);
        let similarity_cutoff = args.get_float_or(3, 0.0);

        let _ = ctx;
        let _ = label;
        let _ = edge_type;
        let _ = top_k;
        let _ = similarity_cutoff;

        // Return a placeholder error - the actual execution will be done
        // in the manifoldb crate's executor where the transaction is available.
        Err(ProcedureError::ExecutionFailed(
            "NodeSimilarity execution requires direct transaction access. \
             Use the higher-level executor in manifoldb crate."
                .to_string(),
        ))
    }

    fn requires_context(&self) -> bool {
        true
    }

    fn output_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            "node1".to_string(),
            "node2".to_string(),
            "similarity".to_string(),
        ]))
    }
}

/// Helper function to execute NodeSimilarity with a transaction and return rows.
///
/// This function is intended to be called from the main manifoldb executor
/// where the transaction is available.
pub fn execute_node_similarity_with_tx<T: manifoldb_storage::Transaction>(
    tx: &T,
    label: Option<&str>,
    edge_type: Option<&str>,
    top_k: Option<i64>,
    similarity_cutoff: f64,
) -> ProcedureResult<RowBatch> {
    // Build configuration
    let mut config = NodeSimilarityConfig::default()
        .with_algorithm(SimilarityAlgorithm::Jaccard)
        .with_similarity_cutoff(similarity_cutoff);

    if let Some(l) = label {
        config = config.with_label_filter(l);
    }

    if let Some(et) = edge_type {
        config = config.with_edge_type_filter(et);
    }

    if let Some(k) = top_k {
        config = config.with_top_k(k as usize);
    }

    let result = NodeSimilarity::compute(tx, &config)
        .map_err(|e| ProcedureError::GraphError(e.to_string()))?;

    // Build result rows
    let schema = Arc::new(Schema::new(vec![
        "node1".to_string(),
        "node2".to_string(),
        "similarity".to_string(),
    ]));
    let mut batch = RowBatch::new(Arc::clone(&schema));

    for (node1, node2, similarity) in result.similarities {
        let row = Row::new(
            Arc::clone(&schema),
            vec![
                Value::Int(node1.as_u64() as i64),
                Value::Int(node2.as_u64() as i64),
                Value::Float(similarity),
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
        let proc = NodeSimilarityProcedure;
        let sig = proc.signature();
        assert_eq!(sig.name, "algo.nodeSimilarity");
        assert_eq!(sig.parameters.len(), 4);
        assert_eq!(sig.returns.len(), 3);
        assert_eq!(sig.required_param_count(), 0);
    }

    #[test]
    fn output_schema() {
        let proc = NodeSimilarityProcedure;
        let schema = proc.output_schema();
        assert_eq!(schema.columns(), vec!["node1", "node2", "similarity"]);
    }

    #[test]
    fn requires_context() {
        let proc = NodeSimilarityProcedure;
        assert!(proc.requires_context());
    }
}
