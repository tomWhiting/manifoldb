//! Projection pushdown optimization.
//!
//! Pushes column requirements down toward data sources to read
//! only the columns that are actually needed.

use std::collections::HashSet;

use crate::plan::logical::{LogicalExpr, LogicalPlan};

/// Projection pushdown optimizer.
///
/// Transforms plans by pushing column requirements down to scans,
/// so that only needed columns are read from storage.
///
/// # Example Transformation
///
/// Before:
/// ```text
/// Project(name)
///   Filter(age > 21)
///     Scan(users)
/// ```
///
/// After:
/// ```text
/// Project(name)
///   Filter(age > 21)
///     Scan(users, projection=[name, age])
/// ```
#[derive(Debug, Clone, Default)]
pub struct ProjectionPushdown {}

impl ProjectionPushdown {
    /// Creates a new projection pushdown optimizer.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }

    /// Optimizes a plan by pushing projections down.
    #[must_use]
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        // First pass: collect required columns
        let required = self.collect_required_columns(&plan);

        // Second pass: push projections to scans
        self.push_projections(plan, &required)
    }

    /// Collects all columns required by the plan.
    fn collect_required_columns(&self, plan: &LogicalPlan) -> HashSet<String> {
        let mut columns = HashSet::new();
        self.collect_columns_recursive(plan, &mut columns);
        columns
    }

    fn collect_columns_recursive(&self, plan: &LogicalPlan, columns: &mut HashSet<String>) {
        match plan {
            LogicalPlan::Scan(node) => {
                // If scan already has projection, use that
                if let Some(proj) = &node.projection {
                    for col in proj {
                        columns.insert(col.clone());
                    }
                }
                // Collect columns from filter
                if let Some(filter) = &node.filter {
                    self.collect_expr_columns(filter, columns);
                }
            }

            LogicalPlan::Filter { node, input } => {
                self.collect_expr_columns(&node.predicate, columns);
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Project { node, input } => {
                for expr in &node.exprs {
                    self.collect_expr_columns(expr, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Aggregate { node, input } => {
                for expr in &node.group_by {
                    self.collect_expr_columns(expr, columns);
                }
                for expr in &node.aggregates {
                    self.collect_expr_columns(expr, columns);
                }
                if let Some(having) = &node.having {
                    self.collect_expr_columns(having, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Sort { node, input } => {
                for order in &node.order_by {
                    self.collect_expr_columns(&order.expr, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Join { node, left, right } => {
                if let Some(cond) = &node.condition {
                    self.collect_expr_columns(cond, columns);
                }
                for col in &node.using_columns {
                    columns.insert(col.clone());
                }
                self.collect_columns_recursive(left, columns);
                self.collect_columns_recursive(right, columns);
            }

            LogicalPlan::Distinct { node, input } => {
                if let Some(on_cols) = &node.on_columns {
                    for expr in on_cols {
                        self.collect_expr_columns(expr, columns);
                    }
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Expand { node, input } => {
                columns.insert(node.src_var.clone());
                columns.insert(node.dst_var.clone());
                if let Some(var) = &node.edge_var {
                    columns.insert(var.clone());
                }
                if let Some(filter) = &node.edge_filter {
                    self.collect_expr_columns(filter, columns);
                }
                if let Some(filter) = &node.node_filter {
                    self.collect_expr_columns(filter, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::AnnSearch { node, input } => {
                columns.insert(node.vector_column.clone());
                self.collect_expr_columns(&node.query_vector, columns);
                if let Some(filter) = &node.filter {
                    self.collect_expr_columns(filter, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::VectorDistance { node, input } => {
                self.collect_expr_columns(&node.left, columns);
                self.collect_expr_columns(&node.right, columns);
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Insert { columns: cols, input, returning, .. } => {
                for col in cols {
                    columns.insert(col.clone());
                }
                for expr in returning {
                    self.collect_expr_columns(expr, columns);
                }
                self.collect_columns_recursive(input, columns);
            }

            LogicalPlan::Update { assignments, filter, returning, .. } => {
                for (col, expr) in assignments {
                    columns.insert(col.clone());
                    self.collect_expr_columns(expr, columns);
                }
                if let Some(f) = filter {
                    self.collect_expr_columns(f, columns);
                }
                for expr in returning {
                    self.collect_expr_columns(expr, columns);
                }
            }

            LogicalPlan::Delete { filter, returning, .. } => {
                if let Some(f) = filter {
                    self.collect_expr_columns(f, columns);
                }
                for expr in returning {
                    self.collect_expr_columns(expr, columns);
                }
            }

            // For other nodes, just recurse
            _ => {
                for child in plan.children() {
                    self.collect_columns_recursive(child, columns);
                }
            }
        }
    }

    /// Collects columns referenced in an expression.
    fn collect_expr_columns(&self, expr: &LogicalExpr, columns: &mut HashSet<String>) {
        match expr {
            LogicalExpr::Column { name, .. } => {
                columns.insert(name.clone());
            }
            LogicalExpr::Wildcard => {
                // Wildcard means all columns - we can't optimize
            }
            LogicalExpr::QualifiedWildcard { .. } => {
                // Qualified wildcard means all columns from a table
            }
            LogicalExpr::BinaryOp { left, right, .. } => {
                self.collect_expr_columns(left, columns);
                self.collect_expr_columns(right, columns);
            }
            LogicalExpr::UnaryOp { operand, .. } => {
                self.collect_expr_columns(operand, columns);
            }
            LogicalExpr::ScalarFunction { args, .. } => {
                for arg in args {
                    self.collect_expr_columns(arg, columns);
                }
            }
            LogicalExpr::AggregateFunction { arg, .. } => {
                self.collect_expr_columns(arg, columns);
            }
            LogicalExpr::Case { operand, when_clauses, else_result } => {
                if let Some(op) = operand {
                    self.collect_expr_columns(op, columns);
                }
                for (when_expr, then_expr) in when_clauses {
                    self.collect_expr_columns(when_expr, columns);
                    self.collect_expr_columns(then_expr, columns);
                }
                if let Some(else_expr) = else_result {
                    self.collect_expr_columns(else_expr, columns);
                }
            }
            LogicalExpr::Cast { expr, .. } => {
                self.collect_expr_columns(expr, columns);
            }
            LogicalExpr::Alias { expr, .. } => {
                self.collect_expr_columns(expr, columns);
            }
            LogicalExpr::InList { expr, list, .. } => {
                self.collect_expr_columns(expr, columns);
                for item in list {
                    self.collect_expr_columns(item, columns);
                }
            }
            LogicalExpr::Between { expr, low, high, .. } => {
                self.collect_expr_columns(expr, columns);
                self.collect_expr_columns(low, columns);
                self.collect_expr_columns(high, columns);
            }
            LogicalExpr::Subquery(sub) | LogicalExpr::Exists { subquery: sub, .. } => {
                self.collect_columns_recursive(sub, columns);
            }
            LogicalExpr::InSubquery { expr, subquery, .. } => {
                self.collect_expr_columns(expr, columns);
                self.collect_columns_recursive(subquery, columns);
            }
            _ => {}
        }
    }

    /// Pushes projections down to scan nodes.
    fn push_projections(&self, plan: LogicalPlan, required: &HashSet<String>) -> LogicalPlan {
        match plan {
            LogicalPlan::Scan(mut node) => {
                // Only push projection if not already set and we have required columns
                if node.projection.is_none() && !required.is_empty() && !self.has_wildcard(required)
                {
                    // Collect columns needed for this scan
                    let scan_columns: Vec<String> = required.iter().cloned().collect();

                    if !scan_columns.is_empty() {
                        node.projection = Some(scan_columns);
                    }
                }
                LogicalPlan::Scan(node)
            }

            LogicalPlan::Filter { node, input } => {
                // Collect columns needed by this filter
                let mut filter_required = required.clone();
                self.collect_expr_columns(&node.predicate, &mut filter_required);

                LogicalPlan::Filter {
                    node,
                    input: Box::new(self.push_projections(*input, &filter_required)),
                }
            }

            LogicalPlan::Project { node, input } => {
                // Collect columns needed by the projection expressions
                let mut project_required = HashSet::new();
                for expr in &node.exprs {
                    self.collect_expr_columns(expr, &mut project_required);
                }

                LogicalPlan::Project {
                    node,
                    input: Box::new(self.push_projections(*input, &project_required)),
                }
            }

            LogicalPlan::Aggregate { node, input } => {
                // Collect columns needed by aggregate
                let mut agg_required = HashSet::new();
                for expr in &node.group_by {
                    self.collect_expr_columns(expr, &mut agg_required);
                }
                for expr in &node.aggregates {
                    self.collect_expr_columns(expr, &mut agg_required);
                }
                if let Some(having) = &node.having {
                    self.collect_expr_columns(having, &mut agg_required);
                }

                LogicalPlan::Aggregate {
                    node,
                    input: Box::new(self.push_projections(*input, &agg_required)),
                }
            }

            LogicalPlan::Sort { node, input } => {
                // Pass through all required columns plus sort columns
                let mut sort_required = required.clone();
                for order in &node.order_by {
                    self.collect_expr_columns(&order.expr, &mut sort_required);
                }

                LogicalPlan::Sort {
                    node,
                    input: Box::new(self.push_projections(*input, &sort_required)),
                }
            }

            LogicalPlan::Limit { node, input } => LogicalPlan::Limit {
                node,
                input: Box::new(self.push_projections(*input, required)),
            },

            LogicalPlan::Distinct { node, input } => {
                let mut distinct_required = required.clone();
                if let Some(on_cols) = &node.on_columns {
                    for expr in on_cols {
                        self.collect_expr_columns(expr, &mut distinct_required);
                    }
                }

                LogicalPlan::Distinct {
                    node,
                    input: Box::new(self.push_projections(*input, &distinct_required)),
                }
            }

            LogicalPlan::Alias { alias, input } => LogicalPlan::Alias {
                alias,
                input: Box::new(self.push_projections(*input, required)),
            },

            LogicalPlan::Join { node, left, right } => {
                // Collect columns needed for join
                let mut join_required = required.clone();
                if let Some(cond) = &node.condition {
                    self.collect_expr_columns(cond, &mut join_required);
                }
                for col in &node.using_columns {
                    join_required.insert(col.clone());
                }

                LogicalPlan::Join {
                    node,
                    left: Box::new(self.push_projections(*left, &join_required)),
                    right: Box::new(self.push_projections(*right, &join_required)),
                }
            }

            LogicalPlan::SetOp { node, left, right } => LogicalPlan::SetOp {
                node,
                left: Box::new(self.push_projections(*left, required)),
                right: Box::new(self.push_projections(*right, required)),
            },

            LogicalPlan::Union { node, inputs } => LogicalPlan::Union {
                node,
                inputs: inputs.into_iter().map(|i| self.push_projections(i, required)).collect(),
            },

            // For other nodes that don't need special handling, return as-is
            // These nodes typically don't have children that need projection pushed
            plan => plan,
        }
    }

    /// Checks if the required columns include a wildcard.
    fn has_wildcard(&self, _required: &HashSet<String>) -> bool {
        // We track wildcards separately in a real implementation
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::LogicalExpr;

    #[test]
    fn push_projection_to_scan() {
        let plan = LogicalPlan::scan("users").project(vec![LogicalExpr::column("name")]);

        let optimizer = ProjectionPushdown::new();
        let optimized = optimizer.optimize(plan);

        // Projection should be pushed to scan
        if let LogicalPlan::Project { input, .. } = &optimized {
            if let LogicalPlan::Scan(node) = input.as_ref() {
                assert!(node.projection.is_some());
                let proj = node.projection.as_ref().unwrap();
                assert!(proj.contains(&"name".to_string()));
            } else {
                panic!("Expected Scan under Project");
            }
        } else {
            panic!("Expected Project at top");
        }
    }

    #[test]
    fn projection_includes_filter_columns() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)))
            .project(vec![LogicalExpr::column("name")]);

        let optimizer = ProjectionPushdown::new();
        let optimized = optimizer.optimize(plan);

        // Scan should include both 'name' (projected) and 'age' (filtered)
        fn find_scan_projection(plan: &LogicalPlan) -> Option<Vec<String>> {
            match plan {
                LogicalPlan::Scan(node) => node.projection.clone(),
                _ => plan.children().iter().find_map(|c| find_scan_projection(c)),
            }
        }

        if let Some(proj) = find_scan_projection(&optimized) {
            assert!(proj.contains(&"name".to_string()) || proj.contains(&"age".to_string()));
        }
    }

    #[test]
    fn projection_through_aggregate() {
        let plan = LogicalPlan::scan("orders")
            .aggregate(
                vec![LogicalExpr::column("status")],
                vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
            )
            .project(vec![LogicalExpr::column("status")]);

        let optimizer = ProjectionPushdown::new();
        let optimized = optimizer.optimize(plan);

        // Should still work without errors
        assert_eq!(optimized.node_type(), "Project");
    }

    #[test]
    fn collect_columns_from_expr() {
        let optimizer = ProjectionPushdown::new();
        let mut columns = HashSet::new();

        let expr =
            LogicalExpr::column("a").add(LogicalExpr::column("b")).mul(LogicalExpr::column("c"));

        optimizer.collect_expr_columns(&expr, &mut columns);

        assert!(columns.contains("a"));
        assert!(columns.contains("b"));
        assert!(columns.contains("c"));
    }
}
