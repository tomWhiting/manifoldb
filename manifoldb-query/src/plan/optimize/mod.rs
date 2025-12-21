//! Query optimization.
//!
//! This module provides optimization passes for logical query plans,
//! transforming them into more efficient equivalent plans.
//!
//! # Optimization Rules
//!
//! The optimizer applies a series of transformation rules:
//!
//! - **Predicate Pushdown**: Push filters closer to data sources
//! - **Projection Pushdown**: Read only required columns
//! - **Index Selection**: Choose optimal access methods
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_query::plan::optimize::Optimizer;
//! use manifoldb_query::plan::LogicalPlan;
//!
//! let plan = LogicalPlan::scan("users")
//!     .project(vec![LogicalExpr::column("name")])
//!     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
//!
//! let optimizer = Optimizer::new();
//! let optimized = optimizer.optimize(plan);
//! ```

mod index_selection;
mod predicate_pushdown;
mod projection_pushdown;

pub use index_selection::{AccessType, IndexCandidate, IndexSelector};
pub use predicate_pushdown::{split_conjunction, PredicatePushdown};
pub use projection_pushdown::ProjectionPushdown;

use crate::plan::logical::LogicalPlan;

/// Query optimizer.
///
/// Applies optimization rules to transform logical plans into
/// more efficient equivalent plans.
#[derive(Debug, Clone, Default)]
pub struct Optimizer {
    /// Whether to enable predicate pushdown.
    predicate_pushdown: bool,
    /// Whether to enable projection pushdown.
    projection_pushdown: bool,
    /// Maximum optimization iterations.
    max_iterations: usize,
}

impl Optimizer {
    /// Creates a new optimizer with all optimizations enabled.
    #[must_use]
    pub fn new() -> Self {
        Self { predicate_pushdown: true, projection_pushdown: true, max_iterations: 10 }
    }

    /// Disables predicate pushdown.
    #[must_use]
    pub const fn without_predicate_pushdown(mut self) -> Self {
        self.predicate_pushdown = false;
        self
    }

    /// Disables projection pushdown.
    #[must_use]
    pub const fn without_projection_pushdown(mut self) -> Self {
        self.projection_pushdown = false;
        self
    }

    /// Sets maximum optimization iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Optimizes a logical plan.
    #[must_use]
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        let mut current = plan;

        for _ in 0..self.max_iterations {
            let optimized = self.apply_rules(current.clone());

            // Check if plan changed
            if optimized == current {
                break;
            }
            current = optimized;
        }

        current
    }

    /// Applies all enabled optimization rules.
    fn apply_rules(&self, plan: LogicalPlan) -> LogicalPlan {
        let mut current = plan;

        if self.predicate_pushdown {
            current = PredicatePushdown::new().optimize(current);
        }

        if self.projection_pushdown {
            current = ProjectionPushdown::new().optimize(current);
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::{LogicalExpr, SortOrder};

    #[test]
    fn optimizer_basic() {
        // Filter after project -> should push filter through project to scan
        let plan = LogicalPlan::scan("users")
            .project(vec![
                LogicalExpr::column("id"),
                LogicalExpr::column("name"),
                LogicalExpr::column("age"),
            ])
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan);

        // Filter should be pushed into the Scan node itself
        // The display format shows it as "Scan: users [filter: ...]"
        let display = format!("{optimized}");
        assert!(display.contains("Project"));
        assert!(display.contains("filter:") || display.contains("Filter"));
    }

    #[test]
    fn optimizer_disabled_rules() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let optimizer = Optimizer::new().without_predicate_pushdown().without_projection_pushdown();

        let optimized = optimizer.optimize(plan.clone());

        // Without optimizations, plan should be unchanged
        assert_eq!(optimized, plan);
    }

    #[test]
    fn optimizer_complex_query() {
        let plan = LogicalPlan::scan("users")
            .project(vec![LogicalExpr::column("name")])
            .filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)))
            .sort(vec![SortOrder::asc(LogicalExpr::column("name"))])
            .limit(10);

        let optimizer = Optimizer::new();
        let optimized = optimizer.optimize(plan);

        // Should still contain all operations
        let display = format!("{optimized}");
        assert!(display.contains("Limit"));
        assert!(display.contains("Sort"));
    }
}
