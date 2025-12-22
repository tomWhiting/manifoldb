//! Cost model for query optimization.
//!
//! This module provides cost estimation for physical plan operators,
//! enabling the optimizer to choose between alternative execution strategies.

use std::ops::Add;

/// Estimated cost of a plan operator.
///
/// Cost is an abstract unit combining CPU and I/O costs.
/// Lower costs are better.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cost {
    /// Abstract cost value (combines CPU + I/O).
    value: f64,
    /// Estimated output cardinality (rows).
    cardinality: usize,
}

impl Cost {
    /// Creates a new cost estimate.
    #[must_use]
    pub const fn new(value: f64, cardinality: usize) -> Self {
        Self { value, cardinality }
    }

    /// Creates a zero cost.
    #[must_use]
    pub const fn zero() -> Self {
        Self { value: 0.0, cardinality: 0 }
    }

    /// Returns the cost value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Returns the estimated cardinality.
    #[must_use]
    pub const fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns true if this cost is less than another.
    #[must_use]
    pub fn is_less_than(&self, other: &Cost) -> bool {
        self.value < other.value
    }

    /// Scales cost by a factor.
    #[must_use]
    pub fn scale(self, factor: f64) -> Self {
        Self { value: self.value * factor, cardinality: self.cardinality }
    }

    /// Sets the cardinality.
    #[must_use]
    pub const fn with_cardinality(mut self, cardinality: usize) -> Self {
        self.cardinality = cardinality;
        self
    }
}

impl Add for Cost {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self { value: self.value + rhs.value, cardinality: self.cardinality.max(rhs.cardinality) }
    }
}

/// Cost model for estimating operator costs.
///
/// Provides cost estimates based on table statistics and operator characteristics.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Cost of reading a single row from storage.
    pub seq_scan_cost_per_row: f64,
    /// Cost of an index lookup.
    pub index_lookup_cost: f64,
    /// Cost of reading a row via index.
    pub index_scan_cost_per_row: f64,
    /// Cost of evaluating a filter predicate per row.
    pub filter_cost_per_row: f64,
    /// Cost of computing a projection expression per row.
    pub project_cost_per_row: f64,
    /// Cost of hashing a row (for hash join/aggregate).
    pub hash_cost_per_row: f64,
    /// Cost of comparing during merge join.
    pub merge_cost_per_row: f64,
    /// Cost of sorting per row (amortized).
    pub sort_cost_per_row: f64,
    /// Cost of vector distance computation.
    pub vector_distance_cost: f64,
    /// Cost of HNSW graph traversal per hop.
    pub hnsw_hop_cost: f64,
    /// Cost of graph edge traversal.
    pub graph_edge_cost: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            seq_scan_cost_per_row: 1.0,
            index_lookup_cost: 4.0,
            index_scan_cost_per_row: 0.5,
            filter_cost_per_row: 0.1,
            project_cost_per_row: 0.05,
            hash_cost_per_row: 0.2,
            merge_cost_per_row: 0.1,
            sort_cost_per_row: 0.5,
            vector_distance_cost: 0.3,
            hnsw_hop_cost: 0.5,
            graph_edge_cost: 0.2,
        }
    }
}

impl CostModel {
    /// Creates a new cost model with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimates cost of a full table scan.
    #[must_use]
    pub fn full_scan_cost(&self, row_count: usize) -> Cost {
        let value = row_count as f64 * self.seq_scan_cost_per_row;
        Cost::new(value, row_count)
    }

    /// Estimates cost of an index point lookup.
    #[must_use]
    pub fn index_lookup_cost(&self, result_count: usize) -> Cost {
        let value = self.index_lookup_cost + result_count as f64 * self.index_scan_cost_per_row;
        Cost::new(value, result_count)
    }

    /// Estimates cost of an index range scan.
    #[must_use]
    pub fn index_range_cost(&self, row_count: usize, selectivity: f64) -> Cost {
        let result_count = (row_count as f64 * selectivity).ceil() as usize;
        let value = self.index_lookup_cost + result_count as f64 * self.index_scan_cost_per_row;
        Cost::new(value, result_count)
    }

    /// Estimates cost of a filter operation.
    #[must_use]
    pub fn filter_cost(&self, input_rows: usize, selectivity: f64) -> Cost {
        let output_rows = (input_rows as f64 * selectivity).ceil() as usize;
        let value = input_rows as f64 * self.filter_cost_per_row;
        Cost::new(value, output_rows)
    }

    /// Estimates cost of a projection operation.
    #[must_use]
    pub fn project_cost(&self, row_count: usize, expr_count: usize) -> Cost {
        let value = row_count as f64 * expr_count as f64 * self.project_cost_per_row;
        Cost::new(value, row_count)
    }

    /// Estimates cost of a hash join.
    #[must_use]
    pub fn hash_join_cost(&self, build_rows: usize, probe_rows: usize, output_rows: usize) -> Cost {
        // Build phase: hash all build rows
        let build_cost = build_rows as f64 * self.hash_cost_per_row;
        // Probe phase: hash and lookup all probe rows
        let probe_cost = probe_rows as f64 * (self.hash_cost_per_row + 0.1);
        Cost::new(build_cost + probe_cost, output_rows)
    }

    /// Estimates cost of a nested loop join.
    #[must_use]
    pub fn nested_loop_cost(
        &self,
        outer_rows: usize,
        inner_rows: usize,
        output_rows: usize,
    ) -> Cost {
        // O(n*m) comparisons
        let value = outer_rows as f64 * inner_rows as f64 * self.filter_cost_per_row;
        Cost::new(value, output_rows)
    }

    /// Estimates cost of a merge join.
    #[must_use]
    pub fn merge_join_cost(&self, left_rows: usize, right_rows: usize, output_rows: usize) -> Cost {
        // O(n+m) comparisons
        let value = (left_rows + right_rows) as f64 * self.merge_cost_per_row;
        Cost::new(value, output_rows)
    }

    /// Estimates cost of a sort operation.
    #[must_use]
    pub fn sort_cost(&self, row_count: usize) -> Cost {
        if row_count == 0 {
            return Cost::zero();
        }
        // O(n log n) comparison cost
        let log_n = (row_count as f64).log2().max(1.0);
        let value = row_count as f64 * log_n * self.sort_cost_per_row;
        Cost::new(value, row_count)
    }

    /// Estimates cost of a hash aggregate.
    #[must_use]
    pub fn hash_aggregate_cost(&self, input_rows: usize, group_count: usize) -> Cost {
        let value = input_rows as f64 * self.hash_cost_per_row;
        Cost::new(value, group_count)
    }

    /// Estimates cost of a hash distinct.
    #[must_use]
    pub fn hash_distinct_cost(&self, input_rows: usize, distinct_count: usize) -> Cost {
        let value = input_rows as f64 * self.hash_cost_per_row;
        Cost::new(value, distinct_count)
    }

    /// Estimates cost of HNSW vector search.
    #[must_use]
    pub fn hnsw_search_cost(&self, table_rows: usize, k: usize, ef_search: usize) -> Cost {
        // HNSW is O(log n) on average
        let log_n = if table_rows > 0 { (table_rows as f64).log2() } else { 1.0 };
        let hops = log_n * ef_search as f64;
        let value = hops * (self.hnsw_hop_cost + self.vector_distance_cost);
        Cost::new(value, k)
    }

    /// Estimates cost of brute-force vector search.
    #[must_use]
    pub fn brute_force_search_cost(&self, table_rows: usize, k: usize) -> Cost {
        // Must compute distance to all vectors
        let value = table_rows as f64 * self.vector_distance_cost;
        Cost::new(value, k)
    }

    /// Estimates cost of graph edge expansion.
    #[must_use]
    pub fn graph_expand_cost(&self, input_nodes: usize, avg_degree: f64) -> Cost {
        let edges_traversed = input_nodes as f64 * avg_degree;
        let value = edges_traversed * self.graph_edge_cost;
        Cost::new(value, edges_traversed.ceil() as usize)
    }

    /// Compares costs and returns the better (lower cost) option.
    #[must_use]
    pub fn better<T>(&self, a: (Cost, T), b: (Cost, T)) -> (Cost, T) {
        if a.0.is_less_than(&b.0) {
            a
        } else {
            b
        }
    }

    /// Decides between HNSW and brute-force search based on data size.
    ///
    /// For small tables, brute-force may be faster due to lower overhead.
    #[must_use]
    pub fn prefer_hnsw(&self, table_rows: usize, k: usize, ef_search: usize) -> bool {
        let hnsw_cost = self.hnsw_search_cost(table_rows, k, ef_search);
        let brute_cost = self.brute_force_search_cost(table_rows, k);
        hnsw_cost.is_less_than(&brute_cost)
    }

    /// Decides between hash join and nested loop based on table sizes.
    #[must_use]
    pub fn prefer_hash_join(
        &self,
        left_rows: usize,
        right_rows: usize,
        output_rows: usize,
    ) -> bool {
        let hash_cost =
            self.hash_join_cost(left_rows.min(right_rows), left_rows.max(right_rows), output_rows);
        let nl_cost = self.nested_loop_cost(left_rows, right_rows, output_rows);
        hash_cost.is_less_than(&nl_cost)
    }

    /// Decides which side should be the build side for hash join.
    ///
    /// Generally, the smaller table should be the build side.
    #[must_use]
    pub fn hash_join_build_side(&self, left_rows: usize, right_rows: usize) -> BuildSide {
        if left_rows <= right_rows {
            BuildSide::Left
        } else {
            BuildSide::Right
        }
    }
}

/// Which side of a join should be the build side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildSide {
    /// Left side is build, right is probe.
    Left,
    /// Right side is build, left is probe.
    Right,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_basics() {
        let cost = Cost::new(100.0, 1000);
        assert_eq!(cost.value(), 100.0);
        assert_eq!(cost.cardinality(), 1000);
    }

    #[test]
    fn cost_addition() {
        let a = Cost::new(50.0, 100);
        let b = Cost::new(30.0, 200);
        let sum = a + b;
        assert_eq!(sum.value(), 80.0);
        assert_eq!(sum.cardinality(), 200); // max
    }

    #[test]
    fn cost_comparison() {
        let low = Cost::new(10.0, 100);
        let high = Cost::new(100.0, 100);
        assert!(low.is_less_than(&high));
        assert!(!high.is_less_than(&low));
    }

    #[test]
    fn cost_model_full_scan() {
        let model = CostModel::new();
        let cost = model.full_scan_cost(1000);
        assert!(cost.value() > 0.0);
        assert_eq!(cost.cardinality(), 1000);
    }

    #[test]
    fn cost_model_index_vs_scan() {
        let model = CostModel::new();

        // For a large table with low selectivity, index should be cheaper
        let scan_cost = model.full_scan_cost(100_000);
        let index_cost = model.index_range_cost(100_000, 0.01); // 1% selectivity

        assert!(index_cost.is_less_than(&scan_cost));
    }

    #[test]
    fn cost_model_hash_join_decision() {
        let model = CostModel::new();

        // For large tables, hash join should be preferred
        assert!(model.prefer_hash_join(10_000, 10_000, 10_000));

        // For very small tables, nested loop may be competitive
        // (but usually hash is still better due to O(n*m) vs O(n+m))
    }

    #[test]
    fn cost_model_hnsw_decision() {
        let model = CostModel::new();

        // For large tables, HNSW should be preferred
        assert!(model.prefer_hnsw(1_000_000, 10, 100));

        // For small tables, brute force may be faster
        assert!(!model.prefer_hnsw(100, 10, 100));
    }

    #[test]
    fn cost_model_sort() {
        let model = CostModel::new();
        let cost = model.sort_cost(10_000);
        assert!(cost.value() > 0.0);
        assert_eq!(cost.cardinality(), 10_000);
    }

    #[test]
    fn cost_model_build_side() {
        let model = CostModel::new();
        assert_eq!(model.hash_join_build_side(100, 10_000), BuildSide::Left);
        assert_eq!(model.hash_join_build_side(10_000, 100), BuildSide::Right);
    }
}
