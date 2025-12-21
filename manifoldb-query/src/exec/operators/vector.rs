//! Vector search operators.
//!
//! These operators integrate with the manifoldb-vector crate
//! for similarity search operations.

use std::sync::Arc;

use manifoldb_core::Value;
use manifoldb_vector::Embedding;

use crate::ast::DistanceMetric;
use crate::exec::context::{ExecutionContext, VectorIndexProvider};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::LogicalExpr;

/// Computes distance between two vectors using the specified metric.
fn compute_distance_with_metric(a: &[f32], b: &[f32], metric: &DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
        }
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                f32::MAX
            } else {
                1.0 - (dot / (norm_a * norm_b))
            }
        }
        DistanceMetric::InnerProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
        DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::Hamming => {
            // For Hamming distance, we count the number of positions where values differ.
            // With f32 values, we use a tolerance-based comparison.
            a.iter().zip(b.iter()).filter(|(x, y)| (*x - *y).abs() > f32::EPSILON).count() as f32
        }
    }
}

/// HNSW-based vector search operator.
///
/// Uses the HNSW index for approximate nearest neighbor search when available.
/// Falls back to brute-force search if no index is provided in the execution context.
pub struct HnswSearchOp {
    /// Base operator state.
    base: OperatorBase,
    /// Name of the HNSW index to use (optional).
    index_name: Option<String>,
    /// Vector column name.
    vector_column: String,
    /// Query vector expression.
    query_vector: LogicalExpr,
    /// Distance metric.
    metric: DistanceMetric,
    /// Number of results.
    k: usize,
    /// HNSW ef_search parameter.
    ef_search: usize,
    /// Whether to include distance in output.
    include_distance: bool,
    /// Distance column alias.
    distance_alias: String,
    /// Input operator.
    input: BoxedOperator,
    /// Collected candidates with distances.
    candidates: Vec<(Row, f32)>,
    /// Position in results.
    position: usize,
    /// Whether search is complete.
    searched: bool,
    /// Cached reference to the vector index provider.
    vector_index_provider: Option<Arc<dyn VectorIndexProvider>>,
}

impl HnswSearchOp {
    /// Creates a new HNSW search operator.
    #[must_use]
    pub fn new(
        vector_column: String,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
        ef_search: usize,
        include_distance: bool,
        distance_alias: Option<String>,
        input: BoxedOperator,
    ) -> Self {
        Self::with_index(
            None,
            vector_column,
            query_vector,
            metric,
            k,
            ef_search,
            include_distance,
            distance_alias,
            input,
        )
    }

    /// Creates a new HNSW search operator with a specified index name.
    ///
    /// When an index name is provided and a vector index provider is available
    /// in the execution context, the operator will use the HNSW index for
    /// efficient approximate nearest neighbor search.
    #[must_use]
    pub fn with_index(
        index_name: Option<String>,
        vector_column: String,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
        ef_search: usize,
        include_distance: bool,
        distance_alias: Option<String>,
        input: BoxedOperator,
    ) -> Self {
        // Build output schema
        let mut columns: Vec<String> = input.schema().columns().to_vec();
        let distance_name = distance_alias.clone().unwrap_or_else(|| "distance".to_string());
        if include_distance {
            columns.push(distance_name.clone());
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            index_name,
            vector_column,
            query_vector,
            metric,
            k,
            ef_search,
            include_distance,
            distance_alias: distance_name,
            input,
            candidates: Vec::new(),
            position: 0,
            searched: false,
            vector_index_provider: None,
        }
    }

    /// Returns the index name if one is configured.
    #[must_use]
    pub fn index_name(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    /// Returns the HNSW ef_search parameter.
    #[must_use]
    pub fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Returns the distance column alias.
    #[must_use]
    pub fn distance_alias(&self) -> &str {
        &self.distance_alias
    }

    /// Returns the number of results to return.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the vector column name.
    #[must_use]
    pub fn vector_column(&self) -> &str {
        &self.vector_column
    }

    /// Computes distance between two vectors.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        compute_distance_with_metric(a, b, &self.metric)
    }

    /// Performs the vector search.
    ///
    /// If a vector index provider is available and an index name is configured,
    /// uses the HNSW index for efficient approximate nearest neighbor search.
    /// Otherwise, falls back to brute-force search over the input rows.
    fn search(&mut self) -> OperatorResult<()> {
        // Get query vector (using a dummy row for evaluation)
        let dummy_schema = Arc::new(Schema::empty());
        let dummy_row = Row::new(dummy_schema, vec![]);
        let query_value = evaluate_expr(&self.query_vector, &dummy_row)?;

        let query = match query_value {
            Value::Vector(v) => v,
            _ => return Ok(()), // No valid query vector
        };

        // Try to use HNSW index if available
        // Clone the values to avoid borrow issues
        let index_name = self.index_name.clone();
        let provider = self.vector_index_provider.clone();

        if let (Some(idx_name), Some(prov)) = (index_name, provider) {
            if prov.has_index(&idx_name) {
                return self.search_with_hnsw_index(&idx_name, &query, prov.as_ref());
            }
        }

        // Fall back to brute-force search
        self.search_brute_force(&query)
    }

    /// Performs vector search using the HNSW index.
    fn search_with_hnsw_index(
        &mut self,
        index_name: &str,
        query: &[f32],
        provider: &dyn VectorIndexProvider,
    ) -> OperatorResult<()> {
        // Create embedding from query vector
        let embedding = Embedding::new(query.to_vec()).map_err(|e| {
            crate::error::ParseError::InvalidVectorOp(format!("Failed to create embedding: {e}"))
        })?;

        // Search the HNSW index
        let results =
            provider.search(index_name, &embedding, self.k, Some(self.ef_search)).map_err(|e| {
                crate::error::ParseError::InvalidVectorOp(format!("HNSW search failed: {e}"))
            })?;

        // Build a map from entity IDs to input rows for efficient lookup
        // First, collect all input rows
        let mut rows_by_id: std::collections::HashMap<i64, Row> = std::collections::HashMap::new();
        while let Some(row) = self.input.next()? {
            // Try to extract an entity ID from the row
            // We look for "id" or "_id" columns
            if let Some(id) = row.get_by_name("id").or_else(|| row.get_by_name("_id")) {
                if let Value::Int(id_val) = id {
                    rows_by_id.insert(*id_val, row);
                }
            }
        }

        // Match search results to input rows
        for result in results {
            let entity_id = result.entity_id.as_u64() as i64;
            if let Some(row) = rows_by_id.remove(&entity_id) {
                self.candidates.push((row, result.distance));
            }
        }

        // Results from HNSW are already sorted by distance
        self.searched = true;
        Ok(())
    }

    /// Performs brute-force vector search.
    fn search_brute_force(&mut self, query: &[f32]) -> OperatorResult<()> {
        // Collect all rows with distances
        while let Some(row) = self.input.next()? {
            if let Some(Value::Vector(v)) = row.get_by_name(&self.vector_column) {
                let distance = self.compute_distance(v, query);
                self.candidates.push((row, distance));
            }
        }

        // Sort by distance and keep top k
        self.candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        self.candidates.truncate(self.k);

        self.searched = true;
        Ok(())
    }
}

impl Operator for HnswSearchOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.candidates.clear();
        self.position = 0;
        self.searched = false;
        // Capture the vector index provider for use during search
        self.vector_index_provider = ctx.vector_index_provider_arc();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if !self.searched {
            self.search()?;
        }

        if self.position >= self.candidates.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let (row, distance) = &self.candidates[self.position];
        self.position += 1;

        // Build output row
        let mut values = row.values().to_vec();
        if self.include_distance {
            values.push(Value::Float(f64::from(*distance)));
        }

        let result = Row::new(self.base.schema(), values);
        self.base.inc_rows_produced();
        Ok(Some(result))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.candidates.clear();
        self.vector_index_provider = None;
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "HnswSearch"
    }
}

/// Brute-force vector search operator.
///
/// Computes distances to all vectors. Exact but slow.
pub struct BruteForceSearchOp {
    /// Base operator state.
    base: OperatorBase,
    /// Vector column name.
    vector_column: String,
    /// Query vector expression.
    query_vector: LogicalExpr,
    /// Distance metric.
    metric: DistanceMetric,
    /// Number of results.
    k: usize,
    /// Whether to include distance in output.
    include_distance: bool,
    /// Distance column alias.
    distance_alias: String,
    /// Input operator.
    input: BoxedOperator,
    /// Collected candidates with distances.
    candidates: Vec<(Row, f32)>,
    /// Position in results.
    position: usize,
    /// Whether search is complete.
    searched: bool,
}

impl BruteForceSearchOp {
    /// Creates a new brute-force search operator.
    #[must_use]
    pub fn new(
        vector_column: String,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
        include_distance: bool,
        distance_alias: Option<String>,
        input: BoxedOperator,
    ) -> Self {
        let mut columns: Vec<String> = input.schema().columns().to_vec();
        let distance_name = distance_alias.clone().unwrap_or_else(|| "distance".to_string());
        if include_distance {
            columns.push(distance_name.clone());
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            vector_column,
            query_vector,
            metric,
            k,
            include_distance,
            distance_alias: distance_name,
            input,
            candidates: Vec::new(),
            position: 0,
            searched: false,
        }
    }

    /// Returns the distance column alias.
    #[must_use]
    pub fn distance_alias(&self) -> &str {
        &self.distance_alias
    }

    /// Returns the number of results to return.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the vector column name.
    #[must_use]
    pub fn vector_column(&self) -> &str {
        &self.vector_column
    }

    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        compute_distance_with_metric(a, b, &self.metric)
    }

    fn search(&mut self) -> OperatorResult<()> {
        let dummy_schema = Arc::new(Schema::empty());
        let dummy_row = Row::new(dummy_schema, vec![]);
        let query_value = evaluate_expr(&self.query_vector, &dummy_row)?;

        let query = match query_value {
            Value::Vector(v) => v,
            _ => return Ok(()),
        };

        while let Some(row) = self.input.next()? {
            if let Some(Value::Vector(v)) = row.get_by_name(&self.vector_column) {
                let distance = self.compute_distance(v, &query);
                self.candidates.push((row, distance));
            }
        }

        self.candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        self.candidates.truncate(self.k);

        self.searched = true;
        Ok(())
    }
}

impl Operator for BruteForceSearchOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.candidates.clear();
        self.position = 0;
        self.searched = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if !self.searched {
            self.search()?;
        }

        if self.position >= self.candidates.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let (row, distance) = &self.candidates[self.position];
        self.position += 1;

        let mut values = row.values().to_vec();
        if self.include_distance {
            values.push(Value::Float(f64::from(*distance)));
        }

        let result = Row::new(self.base.schema(), values);
        self.base.inc_rows_produced();
        Ok(Some(result))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.candidates.clear();
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "BruteForceSearch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_vector_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "embedding".to_string()],
            vec![
                vec![Value::Int(1), Value::Vector(vec![1.0, 0.0, 0.0])],
                vec![Value::Int(2), Value::Vector(vec![0.0, 1.0, 0.0])],
                vec![Value::Int(3), Value::Vector(vec![0.0, 0.0, 1.0])],
                vec![Value::Int(4), Value::Vector(vec![0.5, 0.5, 0.0])],
                vec![Value::Int(5), Value::Vector(vec![0.0, 0.5, 0.5])],
            ],
        ))
    }

    #[test]
    fn brute_force_euclidean() {
        let query = LogicalExpr::vector(vec![1.0, 0.0, 0.0]);

        let mut op = BruteForceSearchOp::new(
            "embedding".to_string(),
            query,
            DistanceMetric::Euclidean,
            3,
            true,
            None,
            make_vector_input(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        // First result should be the exact match (id=1)
        assert_eq!(row1.get_by_name("id"), Some(&Value::Int(1)));
        // Distance should be 0
        if let Some(Value::Float(d)) = row1.get_by_name("distance") {
            assert!(*d < 0.001);
        }

        let row2 = op.next().unwrap().unwrap();
        // Second closest should be id=4 ([0.5, 0.5, 0.0])
        assert_eq!(row2.get_by_name("id"), Some(&Value::Int(4)));

        let row3 = op.next().unwrap();
        assert!(row3.is_some());

        // Only k=3 results
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn brute_force_cosine() {
        let query = LogicalExpr::vector(vec![0.0, 1.0, 0.0]);

        let mut op = BruteForceSearchOp::new(
            "embedding".to_string(),
            query,
            DistanceMetric::Cosine,
            2,
            true,
            Some("dist".to_string()),
            make_vector_input(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        // Exact match is id=2
        assert_eq!(row1.get_by_name("id"), Some(&Value::Int(2)));
        // Cosine distance should be 0 for exact match
        if let Some(Value::Float(d)) = row1.get_by_name("dist") {
            assert!(*d < 0.001);
        }

        op.close().unwrap();
    }

    #[test]
    fn hnsw_search_basic() {
        let query = LogicalExpr::vector(vec![0.0, 0.0, 1.0]);

        let mut op = HnswSearchOp::new(
            "embedding".to_string(),
            query,
            DistanceMetric::Euclidean,
            2,
            100, // ef_search
            true,
            None,
            make_vector_input(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        // Closest is id=3
        assert_eq!(row1.get_by_name("id"), Some(&Value::Int(3)));

        let row2 = op.next().unwrap();
        assert!(row2.is_some());

        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }
}
