//! Vector-specific plan nodes.
//!
//! This module defines plan nodes for vector similarity search:
//! `AnnSearch` (approximate nearest neighbor) and `VectorDistance`.

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
    pub fn cosine(
        vector_column: impl Into<String>,
        query_vector: LogicalExpr,
        k: usize,
    ) -> Self {
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
        Self {
            ef_search: None,
            n_probe: None,
            distance_threshold: None,
            exact: false,
        }
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
        Self {
            left,
            right,
            metric,
            alias: None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ann_search_basic() {
        let search = AnnSearchNode::euclidean(
            "embedding",
            LogicalExpr::param(1),
            10,
        );

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
        let dist = VectorDistanceNode::euclidean(
            LogicalExpr::column("embedding"),
            LogicalExpr::param(1),
        )
        .with_alias("distance");

        assert_eq!(dist.operator(), "<->");
        assert_eq!(dist.alias.as_deref(), Some("distance"));
    }

    #[test]
    fn ann_search_params() {
        let params = AnnSearchParams::new()
            .exact()
            .with_ef_search(200)
            .with_n_probe(16);

        assert!(params.exact);
        assert_eq!(params.ef_search, Some(200));
        assert_eq!(params.n_probe, Some(16));
    }

    #[test]
    fn distance_metrics() {
        let euclidean = VectorDistanceNode::euclidean(
            LogicalExpr::column("a"),
            LogicalExpr::column("b"),
        );
        assert_eq!(euclidean.metric, DistanceMetric::Euclidean);

        let cosine = VectorDistanceNode::cosine(
            LogicalExpr::column("a"),
            LogicalExpr::column("b"),
        );
        assert_eq!(cosine.metric, DistanceMetric::Cosine);

        let inner = VectorDistanceNode::inner_product(
            LogicalExpr::column("a"),
            LogicalExpr::column("b"),
        );
        assert_eq!(inner.metric, DistanceMetric::InnerProduct);
    }
}
