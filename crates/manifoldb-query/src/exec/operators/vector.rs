//! Vector search operators.
//!
//! These operators integrate with the manifoldb-vector crate
//! for similarity search operations.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{EntityId, Value};
use manifoldb_vector::Embedding;

use crate::ast::DistanceMetric;
use crate::exec::context::{ExecutionContext, VectorIndexProvider};
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::LogicalExpr;
use crate::plan::physical::{HybridSearchComponentNode, PhysicalScoreCombinationMethod};

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
/// Uses the HNSW index for approximate nearest neighbor search.
/// Requires a vector index provider and valid index name to be configured.
pub struct HnswSearchOp {
    /// Base operator state.
    base: OperatorBase,
    /// Name of the HNSW index to use.
    index_name: Option<String>,
    /// Vector column name.
    vector_column: String,
    /// Query vector expression.
    query_vector: LogicalExpr,
    /// Distance metric (stored for potential future validation with index metric).
    /// TODO(v0.2): Use this field to validate query metric matches index metric.
    #[allow(dead_code)]
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
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();
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

    /// Performs the vector search using the HNSW index.
    ///
    /// Requires a vector index provider and a valid index name to be configured.
    /// Returns an error if no index is available.
    fn search(&mut self) -> OperatorResult<()> {
        // Get query vector (using a dummy row for evaluation)
        let dummy_schema = Arc::new(Schema::empty());
        let dummy_row = Row::new(dummy_schema, vec![]);
        let query_value = evaluate_expr(&self.query_vector, &dummy_row)?;

        let query = match query_value {
            Value::Vector(v) => v,
            _ => {
                return Err(crate::error::ParseError::InvalidVectorOp(
                    "query expression did not evaluate to a vector".to_string(),
                )
                .into())
            }
        };

        // Require both index name and provider - clone to avoid borrow issues
        let index_name = self.index_name.clone().ok_or_else(|| {
            crate::error::ParseError::InvalidVectorOp(
                "HnswSearchOp requires an index name".to_string(),
            )
        })?;

        let provider = self.vector_index_provider.clone().ok_or_else(|| {
            crate::error::ParseError::InvalidVectorOp(
                "no vector index provider configured".to_string(),
            )
        })?;

        if !provider.has_index(&index_name) {
            return Err(crate::error::ParseError::InvalidVectorOp(format!(
                "vector index '{}' not found",
                index_name
            ))
            .into());
        }

        self.search_with_hnsw_index(&index_name, &query, provider.as_ref())
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
        // First, collect all input rows - pre-allocate for typical result sizes
        const INITIAL_CAPACITY: usize = 1000;
        let mut rows_by_id: std::collections::HashMap<i64, Row> =
            std::collections::HashMap::with_capacity(INITIAL_CAPACITY);
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
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();
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

/// Hybrid vector search operator.
///
/// Combines multiple vector searches (dense + sparse) and merges results
/// using weighted combination or reciprocal rank fusion.
pub struct HybridSearchOp {
    /// Base operator state.
    base: OperatorBase,
    /// Search components.
    components: Vec<HybridSearchComponentNode>,
    /// Number of results.
    k: usize,
    /// Score combination method.
    combination_method: PhysicalScoreCombinationMethod,
    /// Whether to normalize scores.
    normalize_scores: bool,
    /// Whether to include score in output.
    include_score: bool,
    /// Score column alias (used for schema naming).
    /// TODO(v0.2): Use this field for customizable score column naming in output.
    #[allow(dead_code)]
    score_alias: String,
    /// Input operator.
    input: BoxedOperator,
    /// Collected candidates with scores.
    candidates: Vec<(Row, f32)>,
    /// Position in results.
    position: usize,
    /// Whether search is complete.
    searched: bool,
    /// Cached reference to the vector index provider.
    vector_index_provider: Option<Arc<dyn VectorIndexProvider>>,
}

impl HybridSearchOp {
    /// Creates a new hybrid search operator.
    #[must_use]
    pub fn new(
        components: Vec<HybridSearchComponentNode>,
        k: usize,
        combination_method: PhysicalScoreCombinationMethod,
        normalize_scores: bool,
        include_score: bool,
        score_alias: Option<String>,
        input: BoxedOperator,
    ) -> Self {
        let mut columns: Vec<String> =
            input.schema().columns().into_iter().map(|s| s.to_owned()).collect();
        let score_name = score_alias.clone().unwrap_or_else(|| "score".to_string());
        if include_score {
            columns.push(score_name.clone());
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            components,
            k,
            combination_method,
            normalize_scores,
            include_score,
            score_alias: score_name,
            input,
            candidates: Vec::new(),
            position: 0,
            searched: false,
            vector_index_provider: None,
        }
    }

    /// Performs the hybrid search by executing each component and merging results.
    fn search(&mut self) -> OperatorResult<()> {
        // Collect all input rows first
        let mut all_rows: Vec<Row> = Vec::new();
        while let Some(row) = self.input.next()? {
            all_rows.push(row);
        }

        if all_rows.is_empty() {
            self.searched = true;
            return Ok(());
        }

        // Execute each component search
        let mut component_results: Vec<Vec<(EntityId, f32)>> = Vec::new();

        for comp in &self.components {
            let results = self.search_component(comp, &all_rows)?;
            component_results.push(results);
        }

        // Merge results using the specified combination method
        let merged = self.merge_component_results(&component_results, &all_rows)?;

        // Match merged results back to rows
        let rows_by_id: HashMap<i64, Row> = all_rows
            .into_iter()
            .filter_map(|row| {
                row.get_by_name("id").or_else(|| row.get_by_name("_id")).and_then(|v| {
                    if let Value::Int(id) = v {
                        Some((*id, row.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();

        for (entity_id, score) in merged {
            if let Some(row) = rows_by_id.get(&(entity_id.as_u64() as i64)) {
                self.candidates.push((row.clone(), score));
            }
        }

        self.searched = true;
        Ok(())
    }

    /// Searches a single component using either HNSW or brute-force.
    fn search_component(
        &self,
        comp: &HybridSearchComponentNode,
        rows: &[Row],
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        // Evaluate query vector
        let dummy_schema = Arc::new(Schema::empty());
        let dummy_row = Row::new(dummy_schema, vec![]);
        let query_value = evaluate_expr(&comp.query_vector, &dummy_row)?;

        let query = match query_value {
            Value::Vector(v) => v,
            _ => {
                return Err(crate::error::ParseError::InvalidVectorOp(
                    "query expression did not evaluate to a vector".to_string(),
                )
                .into())
            }
        };

        // Try to use HNSW if configured
        if comp.use_hnsw {
            if let (Some(index_name), Some(provider)) =
                (&comp.index_name, &self.vector_index_provider)
            {
                if provider.has_index(index_name) {
                    return self.search_with_hnsw(index_name, &query, comp.ef_search, provider);
                }
            }
        }

        // Fall back to brute-force search
        self.search_brute_force(&query, &comp.vector_column, &comp.metric, rows)
    }

    /// Searches using HNSW index.
    fn search_with_hnsw(
        &self,
        index_name: &str,
        query: &[f32],
        ef_search: usize,
        provider: &Arc<dyn VectorIndexProvider>,
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        let embedding = Embedding::new(query.to_vec()).map_err(|e| {
            crate::error::ParseError::InvalidVectorOp(format!("Failed to create embedding: {e}"))
        })?;

        // Fetch more candidates for hybrid merging (2x k per component)
        let search_k = self.k * 2;

        let results =
            provider.search(index_name, &embedding, search_k, Some(ef_search)).map_err(|e| {
                crate::error::ParseError::InvalidVectorOp(format!("HNSW search failed: {e}"))
            })?;

        Ok(results.into_iter().map(|m| (m.entity_id, m.distance)).collect())
    }

    /// Searches using brute-force.
    fn search_brute_force(
        &self,
        query: &[f32],
        vector_column: &str,
        metric: &DistanceMetric,
        rows: &[Row],
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        let mut results: Vec<(EntityId, f32)> = Vec::new();

        for row in rows {
            if let Some(Value::Vector(v)) = row.get_by_name(vector_column) {
                if v.len() == query.len() {
                    let distance = compute_distance_with_metric(v, query, metric);

                    // Extract entity ID from row
                    if let Some(Value::Int(id)) =
                        row.get_by_name("id").or_else(|| row.get_by_name("_id"))
                    {
                        let entity_id = EntityId::new(*id as u64);
                        results.push((entity_id, distance));
                    }
                }
            }
        }

        // Sort by distance and take top 2*k for merging
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.k * 2);

        Ok(results)
    }

    /// Merges results from all components using the specified combination method.
    fn merge_component_results(
        &self,
        component_results: &[Vec<(EntityId, f32)>],
        _rows: &[Row],
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        if component_results.is_empty() {
            return Ok(Vec::new());
        }

        match self.combination_method {
            PhysicalScoreCombinationMethod::WeightedSum => {
                self.merge_weighted_sum(component_results)
            }
            PhysicalScoreCombinationMethod::ReciprocalRankFusion { k_param } => {
                self.merge_rrf(component_results, k_param)
            }
        }
    }

    /// Merges using weighted sum combination.
    fn merge_weighted_sum(
        &self,
        component_results: &[Vec<(EntityId, f32)>],
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        // Collect all scores by entity ID
        let mut scores_by_entity: HashMap<EntityId, Vec<(usize, f32)>> = HashMap::new();

        for (idx, results) in component_results.iter().enumerate() {
            for (entity_id, distance) in results {
                scores_by_entity.entry(*entity_id).or_default().push((idx, *distance));
            }
        }

        // Normalize if configured
        let normalization_params: Vec<(f32, f32)> = if self.normalize_scores {
            component_results
                .iter()
                .map(|results| {
                    if results.is_empty() {
                        (0.0, 1.0)
                    } else {
                        let min = results.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
                        let max = results.iter().map(|(_, d)| *d).fold(f32::NEG_INFINITY, f32::max);
                        (min, max)
                    }
                })
                .collect()
        } else {
            vec![(0.0, 1.0); component_results.len()]
        };

        // Compute combined scores
        let mut results: Vec<(EntityId, f32)> = scores_by_entity
            .into_iter()
            .map(|(entity_id, component_scores)| {
                let mut total_score = 0.0;
                let mut total_weight = 0.0;

                // Track which components have scores
                let mut has_score_for: Vec<bool> = vec![false; self.components.len()];

                for (idx, distance) in &component_scores {
                    let weight = self.components[*idx].weight;
                    let (min, max) = normalization_params[*idx];

                    let normalized = if max - min > f32::EPSILON {
                        (*distance - min) / (max - min)
                    } else {
                        0.0
                    };

                    total_score += weight * normalized;
                    total_weight += weight;
                    has_score_for[*idx] = true;
                }

                // Handle missing component scores (use 1.0 as worst normalized distance)
                for (idx, comp) in self.components.iter().enumerate() {
                    if !has_score_for[idx] {
                        total_score += comp.weight * 1.0;
                        total_weight += comp.weight;
                    }
                }

                // Normalize by total weight
                let combined = if total_weight > 0.0 { total_score / total_weight } else { 1.0 };

                (entity_id, combined)
            })
            .collect();

        // Sort by combined score (lower is better)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.k);

        Ok(results)
    }

    /// Merges using Reciprocal Rank Fusion.
    fn merge_rrf(
        &self,
        component_results: &[Vec<(EntityId, f32)>],
        k_param: u32,
    ) -> OperatorResult<Vec<(EntityId, f32)>> {
        let mut rrf_scores: HashMap<EntityId, f32> = HashMap::new();

        for results in component_results {
            for (rank, (entity_id, _)) in results.iter().enumerate() {
                let score = 1.0 / (k_param as f32 + rank as f32 + 1.0);
                *rrf_scores.entry(*entity_id).or_insert(0.0) += score;
            }
        }

        // Convert RRF scores to distances (higher RRF score = lower distance)
        let max_score = rrf_scores.values().fold(0.0f32, |a, &b| a.max(b));

        let mut results: Vec<(EntityId, f32)> = rrf_scores
            .into_iter()
            .map(|(entity_id, score)| {
                let distance = if max_score > 0.0 { 1.0 - (score / max_score) } else { 1.0 };
                (entity_id, distance)
            })
            .collect();

        // Sort by distance (lower is better)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.k);

        Ok(results)
    }
}

impl Operator for HybridSearchOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.candidates.clear();
        self.position = 0;
        self.searched = false;
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

        let (row, score) = &self.candidates[self.position];
        self.position += 1;

        // Build output row
        let mut values = row.values().to_vec();
        if self.include_score {
            values.push(Value::Float(f64::from(*score)));
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
        "HybridSearch"
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
    fn hnsw_search_requires_index_name() {
        // Test that HnswSearchOp requires an index name
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

        // Should error because no index name is configured
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("requires an index name"));

        op.close().unwrap();
    }

    #[test]
    fn hnsw_search_requires_provider() {
        // Test that HnswSearchOp requires a vector index provider
        let query = LogicalExpr::vector(vec![0.0, 0.0, 1.0]);

        let mut op = HnswSearchOp::with_index(
            Some("test_index".to_string()),
            "embedding".to_string(),
            query,
            DistanceMetric::Euclidean,
            2,
            100, // ef_search
            true,
            None,
            make_vector_input(),
        );

        let ctx = ExecutionContext::new(); // No vector index provider
        op.open(&ctx).unwrap();

        // Should error because no vector index provider is configured
        let result = op.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("no vector index provider"));

        op.close().unwrap();
    }

    #[test]
    fn hnsw_search_schema_construction() {
        // Test that schema is correctly constructed even without a provider
        let query = LogicalExpr::vector(vec![0.0, 0.0, 1.0]);

        let op = HnswSearchOp::with_index(
            Some("test_index".to_string()),
            "embedding".to_string(),
            query,
            DistanceMetric::Euclidean,
            2,
            100,
            true,
            Some("dist".to_string()),
            make_vector_input(),
        );

        // Should have id, embedding, and dist columns
        assert_eq!(op.schema().columns().len(), 3);
        assert_eq!(
            op.schema().columns(),
            &["id".to_string(), "embedding".to_string(), "dist".to_string()]
        );
    }
}
