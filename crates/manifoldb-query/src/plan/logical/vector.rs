//! Vector-specific plan nodes.
//!
//! This module defines plan nodes for vector similarity search:
//! `AnnSearch` (approximate nearest neighbor), `VectorDistance`, and `HybridSearch`.

// Allow missing_const_for_fn - const fn with Vec isn't stable
#![allow(clippy::missing_const_for_fn)]

use super::expr::LogicalExpr;
use crate::ast::DistanceMetric;

/// An approximate nearest neighbor search node.
///
/// Represents a vector similarity search operation that finds
/// the K nearest neighbors to a query vector.
///
/// # Example
///
/// For the query:
/// ```sql
/// SELECT * FROM documents
/// ORDER BY embedding <-> $query_vector
/// LIMIT 10
/// ```
///
/// This becomes an ANN search with k=10 using Euclidean distance.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnSearchNode {
    /// The vector column/expression to search.
    pub vector_column: String,

    /// The query vector expression.
    pub query_vector: LogicalExpr,

    /// The distance metric to use.
    pub metric: DistanceMetric,

    /// Maximum number of results to return (K).
    pub k: usize,

    /// Optional filter to apply before/during search.
    pub filter: Option<LogicalExpr>,

    /// Search parameters for index configuration.
    pub params: AnnSearchParams,

    /// Whether to include distance in output.
    pub include_distance: bool,

    /// Optional alias for the distance column.
    pub distance_alias: Option<String>,
}

impl AnnSearchNode {
    /// Creates a new ANN search node.
    #[must_use]
    pub fn new(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        k: usize,
    ) -> Self {
        Self {
            vector_column: vector_column.into(),
            query_vector,
            metric,
            k,
            filter: None,
            params: AnnSearchParams::default(),
            include_distance: true,
            distance_alias: None,
        }
    }

    /// Creates an Euclidean distance search.
    #[must_use]
    pub fn euclidean(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        k: usize,
    ) -> Self {
        Self::new(vector_column, query_vector, DistanceMetric::Euclidean, k)
    }

    /// Creates a cosine distance search.
    #[must_use]
    pub fn cosine(vector_column: impl Into<String>, query_vector: LogicalExpr, k: usize) -> Self {
        Self::new(vector_column, query_vector, DistanceMetric::Cosine, k)
    }

    /// Creates an inner product search.
    #[must_use]
    pub fn inner_product(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        k: usize,
    ) -> Self {
        Self::new(vector_column, query_vector, DistanceMetric::InnerProduct, k)
    }

    /// Sets a filter predicate.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Sets search parameters.
    #[must_use]
    pub const fn with_params(mut self, params: AnnSearchParams) -> Self {
        self.params = params;
        self
    }

    /// Configures whether to include distance in output.
    #[must_use]
    pub const fn include_distance(mut self, include: bool) -> Self {
        self.include_distance = include;
        self
    }

    /// Sets the distance column alias.
    #[must_use]
    pub fn with_distance_alias(mut self, alias: impl Into<String>) -> Self {
        self.distance_alias = Some(alias.into());
        self
    }

    /// Sets the `ef_search` parameter (for HNSW indexes).
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.params.ef_search = Some(ef);
        self
    }

    /// Sets the `n_probe` parameter (for IVF indexes).
    #[must_use]
    pub fn with_n_probe(mut self, n_probe: usize) -> Self {
        self.params.n_probe = Some(n_probe);
        self
    }

    /// Sets a distance threshold for filtering.
    #[must_use]
    pub fn with_distance_threshold(mut self, threshold: f32) -> Self {
        self.params.distance_threshold = Some(threshold);
        self
    }
}

/// Parameters for ANN search operations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnnSearchParams {
    /// `ef_search` parameter for HNSW indexes.
    pub ef_search: Option<usize>,

    /// Number of probes for IVF indexes.
    pub n_probe: Option<usize>,

    /// Maximum distance threshold.
    pub distance_threshold: Option<f32>,

    /// Whether to use exact search instead of approximate.
    pub exact: bool,
}

impl AnnSearchParams {
    /// Creates default search parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self { ef_search: None, n_probe: None, distance_threshold: None, exact: false }
    }

    /// Enables exact search.
    #[must_use]
    pub const fn exact(mut self) -> Self {
        self.exact = true;
        self
    }

    /// Sets `ef_search`.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Sets `n_probe`.
    #[must_use]
    pub const fn with_n_probe(mut self, n: usize) -> Self {
        self.n_probe = Some(n);
        self
    }

    /// Sets distance threshold.
    #[must_use]
    pub const fn with_distance_threshold(mut self, threshold: f32) -> Self {
        self.distance_threshold = Some(threshold);
        self
    }
}

/// A vector distance computation node.
///
/// Computes the distance between two vectors without the full
/// ANN search semantics. Used for filtering and ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDistanceNode {
    /// The left vector expression.
    pub left: LogicalExpr,

    /// The right vector expression.
    pub right: LogicalExpr,

    /// The distance metric.
    pub metric: DistanceMetric,

    /// Optional alias for the result.
    pub alias: Option<String>,
}

impl VectorDistanceNode {
    /// Creates a new vector distance node.
    #[must_use]
    pub fn new(left: LogicalExpr, right: LogicalExpr, metric: DistanceMetric) -> Self {
        Self { left, right, metric, alias: None }
    }

    /// Creates an Euclidean distance node.
    #[must_use]
    pub fn euclidean(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self::new(left, right, DistanceMetric::Euclidean)
    }

    /// Creates a cosine distance node.
    #[must_use]
    pub fn cosine(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self::new(left, right, DistanceMetric::Cosine)
    }

    /// Creates an inner product node.
    #[must_use]
    pub fn inner_product(left: LogicalExpr, right: LogicalExpr) -> Self {
        Self::new(left, right, DistanceMetric::InnerProduct)
    }

    /// Sets an alias for the result.
    #[must_use]
    pub fn with_alias(mut self, alias: impl Into<String>) -> Self {
        self.alias = Some(alias.into());
        self
    }

    /// Returns the SQL operator for this distance.
    #[must_use]
    pub fn operator(&self) -> &'static str {
        self.metric.operator()
    }
}

/// Score combination method for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoreCombinationMethod {
    /// Weighted linear combination: `w1*s1 + w2*s2`
    WeightedSum,
    /// Reciprocal Rank Fusion with configurable k parameter.
    ReciprocalRankFusion {
        /// RRF k parameter (typically 60).
        k_param: u32,
    },
}

impl Default for ScoreCombinationMethod {
    fn default() -> Self {
        Self::WeightedSum
    }
}

impl ScoreCombinationMethod {
    /// Creates an RRF combination with the default k=60.
    #[must_use]
    pub const fn rrf() -> Self {
        Self::ReciprocalRankFusion { k_param: 60 }
    }

    /// Creates an RRF combination with a custom k parameter.
    #[must_use]
    pub const fn rrf_with_k(k_param: u32) -> Self {
        Self::ReciprocalRankFusion { k_param }
    }
}

/// A component in a hybrid search (one vector search with weight).
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchComponent {
    /// The vector column to search.
    pub vector_column: String,
    /// The query vector expression.
    pub query_vector: LogicalExpr,
    /// Distance metric for this component.
    pub metric: DistanceMetric,
    /// Weight for this component (0.0 to 1.0).
    pub weight: f32,
    /// Search parameters.
    pub params: AnnSearchParams,
}

impl HybridSearchComponent {
    /// Creates a new hybrid search component.
    #[must_use]
    pub fn new(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        metric: DistanceMetric,
        weight: f32,
    ) -> Self {
        Self {
            vector_column: vector_column.into(),
            query_vector,
            metric,
            weight,
            params: AnnSearchParams::default(),
        }
    }

    /// Creates a dense vector component with cosine distance.
    #[must_use]
    pub fn dense(vector_column: impl Into<String>, query_vector: LogicalExpr, weight: f32) -> Self {
        Self::new(vector_column, query_vector, DistanceMetric::Cosine, weight)
    }

    /// Creates a sparse vector component with inner product.
    #[must_use]
    pub fn sparse(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        weight: f32,
    ) -> Self {
        Self::new(vector_column, query_vector, DistanceMetric::InnerProduct, weight)
    }

    /// Sets search parameters.
    #[must_use]
    pub fn with_params(mut self, params: AnnSearchParams) -> Self {
        self.params = params;
        self
    }

    /// Sets ef_search parameter.
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.params.ef_search = Some(ef);
        self
    }
}

/// A hybrid vector search node.
///
/// Combines multiple vector searches (e.g., dense + sparse) into a single
/// query using weighted score combination or reciprocal rank fusion.
///
/// # Example
///
/// ```sql
/// SELECT * FROM documents
/// ORDER BY HYBRID(dense <=> $q1, 0.7, sparse <#> $q2, 0.3)
/// LIMIT 10;
/// ```
///
/// This combines dense vector search (70% weight) with sparse vector
/// search (30% weight) using weighted sum combination.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchNode {
    /// The search components (each with a vector column, query, and weight).
    pub components: Vec<HybridSearchComponent>,

    /// Maximum number of results to return (K).
    pub k: usize,

    /// Score combination method.
    pub combination_method: ScoreCombinationMethod,

    /// Whether to normalize scores before combining.
    pub normalize_scores: bool,

    /// Optional filter to apply.
    pub filter: Option<LogicalExpr>,

    /// Whether to include the combined score in output.
    pub include_score: bool,

    /// Optional alias for the score column.
    pub score_alias: Option<String>,
}

impl HybridSearchNode {
    /// Creates a new hybrid search node.
    #[must_use]
    pub fn new(components: Vec<HybridSearchComponent>, k: usize) -> Self {
        Self {
            components,
            k,
            combination_method: ScoreCombinationMethod::default(),
            normalize_scores: true,
            filter: None,
            include_score: true,
            score_alias: None,
        }
    }

    /// Creates a hybrid search with two components.
    #[must_use]
    pub fn two(first: HybridSearchComponent, second: HybridSearchComponent, k: usize) -> Self {
        Self::new(vec![first, second], k)
    }

    /// Creates a dense+sparse hybrid search.
    ///
    /// This is a convenience constructor for the common case of combining
    /// a dense embedding (cosine distance) with a sparse embedding (inner product).
    #[must_use]
    pub fn dense_sparse(
        dense_column: impl Into<String>,
        dense_query: LogicalExpr,
        dense_weight: f32,
        sparse_column: impl Into<String>,
        sparse_query: LogicalExpr,
        sparse_weight: f32,
        k: usize,
    ) -> Self {
        Self::two(
            HybridSearchComponent::dense(dense_column, dense_query, dense_weight),
            HybridSearchComponent::sparse(sparse_column, sparse_query, sparse_weight),
            k,
        )
    }

    /// Sets the score combination method.
    #[must_use]
    pub const fn with_combination_method(mut self, method: ScoreCombinationMethod) -> Self {
        self.combination_method = method;
        self
    }

    /// Uses reciprocal rank fusion for score combination.
    #[must_use]
    pub const fn with_rrf(self) -> Self {
        self.with_combination_method(ScoreCombinationMethod::rrf())
    }

    /// Uses reciprocal rank fusion with a custom k parameter.
    #[must_use]
    pub const fn with_rrf_k(self, k_param: u32) -> Self {
        self.with_combination_method(ScoreCombinationMethod::rrf_with_k(k_param))
    }

    /// Disables score normalization.
    #[must_use]
    pub const fn without_normalization(mut self) -> Self {
        self.normalize_scores = false;
        self
    }

    /// Sets a filter predicate.
    #[must_use]
    pub fn with_filter(mut self, filter: LogicalExpr) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Configures whether to include the score in output.
    #[must_use]
    pub const fn include_score(mut self, include: bool) -> Self {
        self.include_score = include;
        self
    }

    /// Sets the score column alias.
    #[must_use]
    pub fn with_score_alias(mut self, alias: impl Into<String>) -> Self {
        self.score_alias = Some(alias.into());
        self
    }

    /// Returns the total weight of all components.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.components.iter().map(|c| c.weight).sum()
    }

    /// Returns the number of components.
    #[must_use]
    pub fn num_components(&self) -> usize {
        self.components.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ann_search_basic() {
        let search = AnnSearchNode::euclidean("embedding", LogicalExpr::param(1), 10);

        assert_eq!(search.vector_column, "embedding");
        assert_eq!(search.k, 10);
        assert_eq!(search.metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn ann_search_with_params() {
        let search = AnnSearchNode::cosine("embedding", LogicalExpr::param(1), 20)
            .with_ef_search(100)
            .with_distance_threshold(0.5)
            .with_filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)));

        assert_eq!(search.params.ef_search, Some(100));
        assert_eq!(search.params.distance_threshold, Some(0.5));
        assert!(search.filter.is_some());
    }

    #[test]
    fn vector_distance_basic() {
        let dist =
            VectorDistanceNode::euclidean(LogicalExpr::column("embedding"), LogicalExpr::param(1))
                .with_alias("distance");

        assert_eq!(dist.operator(), "<->");
        assert_eq!(dist.alias.as_deref(), Some("distance"));
    }

    #[test]
    fn ann_search_params() {
        let params = AnnSearchParams::new().exact().with_ef_search(200).with_n_probe(16);

        assert!(params.exact);
        assert_eq!(params.ef_search, Some(200));
        assert_eq!(params.n_probe, Some(16));
    }

    #[test]
    fn distance_metrics() {
        let euclidean =
            VectorDistanceNode::euclidean(LogicalExpr::column("a"), LogicalExpr::column("b"));
        assert_eq!(euclidean.metric, DistanceMetric::Euclidean);

        let cosine = VectorDistanceNode::cosine(LogicalExpr::column("a"), LogicalExpr::column("b"));
        assert_eq!(cosine.metric, DistanceMetric::Cosine);

        let inner =
            VectorDistanceNode::inner_product(LogicalExpr::column("a"), LogicalExpr::column("b"));
        assert_eq!(inner.metric, DistanceMetric::InnerProduct);
    }

    #[test]
    fn hybrid_search_basic() {
        let dense = HybridSearchComponent::dense("dense_vec", LogicalExpr::param(1), 0.7);
        let sparse = HybridSearchComponent::sparse("sparse_vec", LogicalExpr::param(2), 0.3);

        let search = HybridSearchNode::two(dense, sparse, 10);

        assert_eq!(search.num_components(), 2);
        assert_eq!(search.k, 10);
        assert!((search.total_weight() - 1.0).abs() < 0.001);
        assert_eq!(search.combination_method, ScoreCombinationMethod::WeightedSum);
        assert!(search.normalize_scores);
    }

    #[test]
    fn hybrid_search_dense_sparse_constructor() {
        let search = HybridSearchNode::dense_sparse(
            "dense",
            LogicalExpr::param(1),
            0.6,
            "sparse",
            LogicalExpr::param(2),
            0.4,
            20,
        );

        assert_eq!(search.num_components(), 2);
        assert_eq!(search.k, 20);
        assert_eq!(search.components[0].vector_column, "dense");
        assert_eq!(search.components[0].metric, DistanceMetric::Cosine);
        assert!((search.components[0].weight - 0.6).abs() < 0.001);
        assert_eq!(search.components[1].vector_column, "sparse");
        assert_eq!(search.components[1].metric, DistanceMetric::InnerProduct);
        assert!((search.components[1].weight - 0.4).abs() < 0.001);
    }

    #[test]
    fn hybrid_search_with_rrf() {
        let search = HybridSearchNode::dense_sparse(
            "dense",
            LogicalExpr::param(1),
            0.5,
            "sparse",
            LogicalExpr::param(2),
            0.5,
            10,
        )
        .with_rrf();

        assert!(matches!(
            search.combination_method,
            ScoreCombinationMethod::ReciprocalRankFusion { k_param: 60 }
        ));
    }

    #[test]
    fn hybrid_search_with_custom_rrf_k() {
        let search = HybridSearchNode::dense_sparse(
            "dense",
            LogicalExpr::param(1),
            0.5,
            "sparse",
            LogicalExpr::param(2),
            0.5,
            10,
        )
        .with_rrf_k(30);

        assert!(matches!(
            search.combination_method,
            ScoreCombinationMethod::ReciprocalRankFusion { k_param: 30 }
        ));
    }

    #[test]
    fn hybrid_search_with_options() {
        let search = HybridSearchNode::dense_sparse(
            "dense",
            LogicalExpr::param(1),
            0.7,
            "sparse",
            LogicalExpr::param(2),
            0.3,
            10,
        )
        .without_normalization()
        .with_filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)))
        .with_score_alias("hybrid_score");

        assert!(!search.normalize_scores);
        assert!(search.filter.is_some());
        assert_eq!(search.score_alias.as_deref(), Some("hybrid_score"));
    }

    #[test]
    fn score_combination_methods() {
        assert_eq!(ScoreCombinationMethod::default(), ScoreCombinationMethod::WeightedSum);
        assert!(matches!(
            ScoreCombinationMethod::rrf(),
            ScoreCombinationMethod::ReciprocalRankFusion { k_param: 60 }
        ));
        assert!(matches!(
            ScoreCombinationMethod::rrf_with_k(100),
            ScoreCombinationMethod::ReciprocalRankFusion { k_param: 100 }
        ));
    }
}
