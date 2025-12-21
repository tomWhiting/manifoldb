//! Vector operation AST types.
//!
//! This module defines types for representing vector similarity search operations
//! and related constructs.

use super::expr::{Expr, QualifiedName};
use std::fmt;

/// A vector similarity search operation.
///
/// This represents operations like `embedding <-> $query` for similarity search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSearch {
    /// The vector column or expression.
    pub vector_expr: Box<Expr>,
    /// The query vector expression.
    pub query_expr: Box<Expr>,
    /// The distance metric to use.
    pub metric: DistanceMetric,
}

impl VectorSearch {
    /// Creates a new vector search with the specified metric.
    #[must_use]
    pub fn new(vector_expr: Expr, query_expr: Expr, metric: DistanceMetric) -> Self {
        Self { vector_expr: Box::new(vector_expr), query_expr: Box::new(query_expr), metric }
    }

    /// Creates an Euclidean distance search.
    #[must_use]
    pub fn euclidean(vector_expr: Expr, query_expr: Expr) -> Self {
        Self::new(vector_expr, query_expr, DistanceMetric::Euclidean)
    }

    /// Creates a cosine distance search.
    #[must_use]
    pub fn cosine(vector_expr: Expr, query_expr: Expr) -> Self {
        Self::new(vector_expr, query_expr, DistanceMetric::Cosine)
    }

    /// Creates an inner product search.
    #[must_use]
    pub fn inner_product(vector_expr: Expr, query_expr: Expr) -> Self {
        Self::new(vector_expr, query_expr, DistanceMetric::InnerProduct)
    }
}

impl fmt::Display for VectorSearch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vector_search({}, ...)", self.metric)
    }
}

/// Distance metrics for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance: `<->`.
    Euclidean,
    /// Cosine distance: `<=>`.
    Cosine,
    /// Inner product (negative for similarity): `<#>`.
    InnerProduct,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Hamming distance (for binary vectors).
    Hamming,
}

impl DistanceMetric {
    /// Returns the SQL operator for this metric.
    #[must_use]
    pub const fn operator(&self) -> &'static str {
        match self {
            Self::Euclidean => "<->",
            Self::Cosine => "<=>",
            Self::InnerProduct => "<#>",
            Self::Manhattan => "<~>", // Custom operator
            Self::Hamming => "<%>",   // Custom operator
        }
    }

    /// Returns the function name for this metric.
    #[must_use]
    pub const fn function_name(&self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean_distance",
            Self::Cosine => "cosine_distance",
            Self::InnerProduct => "inner_product",
            Self::Manhattan => "manhattan_distance",
            Self::Hamming => "hamming_distance",
        }
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.operator())
    }
}

/// A vector index hint for query optimization.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorIndexHint {
    /// The index name to use.
    pub index_name: QualifiedName,
    /// Optional search parameters.
    pub params: VectorSearchParams,
}

/// Parameters for vector search operations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VectorSearchParams {
    /// Maximum number of results to return (for ANN search).
    pub limit: Option<u32>,
    /// Number of candidates to consider (for IVF/HNSW indexes).
    pub ef_search: Option<u32>,
    /// Number of probes for IVF indexes.
    pub n_probe: Option<u32>,
    /// Distance threshold for filtering results.
    pub distance_threshold: Option<f32>,
}

impl VectorSearchParams {
    /// Creates empty search parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self { limit: None, ef_search: None, n_probe: None, distance_threshold: None }
    }

    /// Sets the result limit.
    #[must_use]
    pub const fn with_limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Sets the `ef_search` parameter.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: u32) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Sets the number of probes.
    #[must_use]
    pub const fn with_n_probe(mut self, n: u32) -> Self {
        self.n_probe = Some(n);
        self
    }

    /// Sets the distance threshold.
    #[must_use]
    pub const fn with_distance_threshold(mut self, threshold: f32) -> Self {
        self.distance_threshold = Some(threshold);
        self
    }
}

/// Vector aggregation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAggregateOp {
    /// Average of vectors.
    Avg,
    /// Sum of vectors.
    Sum,
    /// Find the centroid of a set of vectors.
    Centroid,
}

impl fmt::Display for VectorAggregateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Avg => "VECTOR_AVG",
            Self::Sum => "VECTOR_SUM",
            Self::Centroid => "VECTOR_CENTROID",
        };
        write!(f, "{name}")
    }
}

/// A vector aggregate function call.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorAggregate {
    /// The aggregation operation.
    pub op: VectorAggregateOp,
    /// The vector expression to aggregate.
    pub expr: Box<Expr>,
}

impl VectorAggregate {
    /// Creates a new vector aggregate.
    #[must_use]
    pub fn new(op: VectorAggregateOp, expr: Expr) -> Self {
        Self { op, expr: Box::new(expr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_metric_operators() {
        assert_eq!(DistanceMetric::Euclidean.operator(), "<->");
        assert_eq!(DistanceMetric::Cosine.operator(), "<=>");
        assert_eq!(DistanceMetric::InnerProduct.operator(), "<#>");
    }

    #[test]
    fn distance_metric_function_names() {
        assert_eq!(DistanceMetric::Euclidean.function_name(), "euclidean_distance");
        assert_eq!(DistanceMetric::Cosine.function_name(), "cosine_distance");
        assert_eq!(DistanceMetric::InnerProduct.function_name(), "inner_product");
    }

    #[test]
    fn vector_search_params_builder() {
        let params = VectorSearchParams::new()
            .with_limit(10)
            .with_ef_search(100)
            .with_distance_threshold(0.5);

        assert_eq!(params.limit, Some(10));
        assert_eq!(params.ef_search, Some(100));
        assert_eq!(params.distance_threshold, Some(0.5));
    }

    #[test]
    fn vector_aggregate_display() {
        assert_eq!(VectorAggregateOp::Avg.to_string(), "VECTOR_AVG");
        assert_eq!(VectorAggregateOp::Centroid.to_string(), "VECTOR_CENTROID");
    }
}
