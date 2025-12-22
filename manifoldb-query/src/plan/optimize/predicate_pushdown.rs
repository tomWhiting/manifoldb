//! Predicate pushdown optimization.
//!
//! Pushes filter predicates down toward data sources to reduce
//! the amount of data processed by upper operators.

// Allow cognitive complexity for the main optimization function
#![allow(clippy::cognitive_complexity)]
// Allow expect - invariant guaranteed by the preceding filter
#![allow(clippy::expect_used)]

use crate::ast::BinaryOp;
use crate::plan::logical::{
    FilterNode, JoinNode, JoinType, LogicalExpr, LogicalPlan, ProjectNode, ScanNode,
};

/// Predicate pushdown optimizer.
///
/// Transforms plans by pushing filter predicates as close to
/// data sources as possible.
///
/// # Example Transformation
///
/// Before:
/// ```text
/// Project
///   Filter(age > 21)
///     Scan(users)
/// ```
///
/// After:
/// ```text
/// Project
///   Scan(users, filter=age > 21)
/// ```
#[derive(Debug, Clone, Default)]
pub struct PredicatePushdown {}

impl PredicatePushdown {
    /// Creates a new predicate pushdown optimizer.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }

    /// Optimizes a plan by pushing predicates down.
    #[must_use]
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        self.push_down(plan, Vec::new())
    }

    /// Recursively pushes predicates down through the plan tree.
    fn push_down(&self, plan: LogicalPlan, mut predicates: Vec<LogicalExpr>) -> LogicalPlan {
        match plan {
            // For Filter nodes, collect the predicate and continue pushing down
            LogicalPlan::Filter { node, input } => {
                predicates.push(node.predicate);
                self.push_down(*input, predicates)
            }

            // For Scan nodes, push predicates into the scan if possible
            LogicalPlan::Scan(node) => self.push_to_scan(*node, predicates),

            // For Project, push through but keep predicates that use projected columns
            LogicalPlan::Project { node, input } => {
                self.push_through_project(node, *input, predicates)
            }

            // For Aggregate, some predicates can be pushed, others must stay above
            LogicalPlan::Aggregate { node, input } => {
                self.push_through_aggregate(*node, *input, predicates)
            }

            // For Sort, push all predicates through
            LogicalPlan::Sort { node, input } => {
                let optimized_input = self.push_down(*input, predicates);
                LogicalPlan::Sort { node, input: Box::new(optimized_input) }
            }

            // For Limit, push all predicates through
            LogicalPlan::Limit { node, input } => {
                let optimized_input = self.push_down(*input, predicates);
                LogicalPlan::Limit { node, input: Box::new(optimized_input) }
            }

            // For Distinct, push all predicates through
            LogicalPlan::Distinct { node, input } => {
                let optimized_input = self.push_down(*input, predicates);
                LogicalPlan::Distinct { node, input: Box::new(optimized_input) }
            }

            // For Alias, push through
            LogicalPlan::Alias { alias, input } => {
                let optimized_input = self.push_down(*input, predicates);
                LogicalPlan::Alias { alias, input: Box::new(optimized_input) }
            }

            // For Join, push predicates to appropriate sides
            LogicalPlan::Join { node, left, right } => {
                self.push_through_join(*node, *left, *right, predicates)
            }

            // For SetOp, predicates cannot be pushed through
            LogicalPlan::SetOp { node, left, right } => {
                let result = LogicalPlan::SetOp {
                    node,
                    left: Box::new(self.push_down(*left, Vec::new())),
                    right: Box::new(self.push_down(*right, Vec::new())),
                };
                self.apply_predicates(result, predicates)
            }

            // For Union, push predicates to all inputs
            LogicalPlan::Union { node, inputs } => {
                // Clone predicates for each input
                let optimized_inputs: Vec<LogicalPlan> =
                    inputs.into_iter().map(|i| self.push_down(i, predicates.clone())).collect();

                LogicalPlan::Union { node, inputs: optimized_inputs }
            }

            // For graph and vector operations, apply predicates above
            LogicalPlan::Expand { node, input } => {
                let optimized_input = self.push_down(*input, Vec::new());
                let result = LogicalPlan::Expand { node, input: Box::new(optimized_input) };
                self.apply_predicates(result, predicates)
            }

            LogicalPlan::PathScan { node, input } => {
                let optimized_input = self.push_down(*input, Vec::new());
                let result = LogicalPlan::PathScan { node, input: Box::new(optimized_input) };
                self.apply_predicates(result, predicates)
            }

            LogicalPlan::AnnSearch { node, input } => {
                // For ANN search, some filters can be pushed into the search
                let (pushable, remaining) = self.partition_predicates(&predicates, |p| {
                    // Check if predicate only references columns from input
                    self.references_only_scan_columns(p)
                });

                let mut search_node = *node;
                if !pushable.is_empty() {
                    let combined = Self::combine_predicates(pushable);
                    search_node.filter = Some(match search_node.filter {
                        Some(existing) => existing.and(combined),
                        None => combined,
                    });
                }

                let optimized_input = self.push_down(*input, Vec::new());
                let result = LogicalPlan::AnnSearch {
                    node: Box::new(search_node),
                    input: Box::new(optimized_input),
                };
                self.apply_predicates(result, remaining)
            }

            LogicalPlan::VectorDistance { node, input } => {
                let optimized_input = self.push_down(*input, Vec::new());
                let result = LogicalPlan::VectorDistance { node, input: Box::new(optimized_input) };
                self.apply_predicates(result, predicates)
            }

            // For DML operations, apply predicates above
            LogicalPlan::Insert { table, columns, input, returning } => {
                let optimized_input = self.push_down(*input, Vec::new());
                let result = LogicalPlan::Insert {
                    table,
                    columns,
                    input: Box::new(optimized_input),
                    returning,
                };
                self.apply_predicates(result, predicates)
            }

            // Update and Delete can have their filters combined
            LogicalPlan::Update { table, assignments, filter, returning } => {
                let combined = self.combine_with_existing(filter, predicates);
                LogicalPlan::Update { table, assignments, filter: combined, returning }
            }

            LogicalPlan::Delete { table, filter, returning } => {
                let combined = self.combine_with_existing(filter, predicates);
                LogicalPlan::Delete { table, filter: combined, returning }
            }

            // Leaf nodes without filter support
            LogicalPlan::Values(node) => {
                self.apply_predicates(LogicalPlan::Values(node), predicates)
            }

            LogicalPlan::Empty { columns } => {
                self.apply_predicates(LogicalPlan::Empty { columns }, predicates)
            }

            // DDL operations - no predicate pushdown
            LogicalPlan::CreateTable(_)
            | LogicalPlan::DropTable(_)
            | LogicalPlan::CreateIndex(_)
            | LogicalPlan::DropIndex(_) => {
                // DDL statements don't have predicates to push
                plan
            }
        }
    }

    /// Pushes predicates into a scan node.
    fn push_to_scan(&self, mut node: ScanNode, predicates: Vec<LogicalExpr>) -> LogicalPlan {
        if predicates.is_empty() {
            return LogicalPlan::Scan(Box::new(node));
        }

        // Combine all predicates
        let combined = Self::combine_predicates(predicates);

        // Merge with existing filter
        node.filter = Some(match node.filter {
            Some(existing) => existing.and(combined),
            None => combined,
        });

        LogicalPlan::Scan(Box::new(node))
    }

    /// Pushes predicates through a projection.
    fn push_through_project(
        &self,
        node: ProjectNode,
        input: LogicalPlan,
        predicates: Vec<LogicalExpr>,
    ) -> LogicalPlan {
        // All predicates can be pushed through projection
        // (assuming they reference columns that exist in the input)
        let optimized_input = self.push_down(input, predicates);

        LogicalPlan::Project { node, input: Box::new(optimized_input) }
    }

    /// Pushes predicates through an aggregate.
    fn push_through_aggregate(
        &self,
        node: crate::plan::logical::AggregateNode,
        input: LogicalPlan,
        predicates: Vec<LogicalExpr>,
    ) -> LogicalPlan {
        // Only predicates on GROUP BY columns can be pushed below aggregate
        let (pushable, remaining) = self.partition_predicates(&predicates, |p| {
            self.references_only_group_by_columns(p, &node.group_by)
        });

        let optimized_input = self.push_down(input, pushable);

        let result =
            LogicalPlan::Aggregate { node: Box::new(node), input: Box::new(optimized_input) };

        self.apply_predicates(result, remaining)
    }

    /// Pushes predicates through a join.
    fn push_through_join(
        &self,
        node: JoinNode,
        left: LogicalPlan,
        right: LogicalPlan,
        predicates: Vec<LogicalExpr>,
    ) -> LogicalPlan {
        let left_tables = self.collect_tables(&left);
        let right_tables = self.collect_tables(&right);

        // Pre-allocate based on predicates size - most predicates go to one side
        let mut left_predicates = Vec::with_capacity(predicates.len());
        let mut right_predicates = Vec::with_capacity(predicates.len());
        let mut remaining = Vec::with_capacity(predicates.len());

        for pred in predicates {
            let pred_tables = self.collect_referenced_tables(&pred);

            if pred_tables.iter().all(|t| left_tables.contains(t)) {
                // Predicate only references left side
                match node.join_type {
                    JoinType::Inner | JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => {
                        left_predicates.push(pred);
                    }
                    _ => remaining.push(pred),
                }
            } else if pred_tables.iter().all(|t| right_tables.contains(t)) {
                // Predicate only references right side
                match node.join_type {
                    JoinType::Inner
                    | JoinType::Right
                    | JoinType::RightSemi
                    | JoinType::RightAnti => {
                        right_predicates.push(pred);
                    }
                    _ => remaining.push(pred),
                }
            } else {
                // Predicate references both sides
                remaining.push(pred);
            }
        }

        let optimized_left = self.push_down(left, left_predicates);
        let optimized_right = self.push_down(right, right_predicates);

        let result = LogicalPlan::Join {
            node: Box::new(node),
            left: Box::new(optimized_left),
            right: Box::new(optimized_right),
        };

        self.apply_predicates(result, remaining)
    }

    /// Applies remaining predicates as Filter nodes above the plan.
    fn apply_predicates(&self, plan: LogicalPlan, predicates: Vec<LogicalExpr>) -> LogicalPlan {
        if predicates.is_empty() {
            return plan;
        }

        let combined = Self::combine_predicates(predicates);

        LogicalPlan::Filter { node: FilterNode::new(combined), input: Box::new(plan) }
    }

    /// Combines multiple predicates with AND.
    fn combine_predicates(predicates: Vec<LogicalExpr>) -> LogicalExpr {
        predicates.into_iter().reduce(|a, b| a.and(b)).expect("predicates should not be empty")
    }

    /// Combines predicates with an existing optional filter.
    fn combine_with_existing(
        &self,
        existing: Option<LogicalExpr>,
        predicates: Vec<LogicalExpr>,
    ) -> Option<LogicalExpr> {
        if predicates.is_empty() {
            return existing;
        }

        let combined = Self::combine_predicates(predicates);
        Some(match existing {
            Some(e) => e.and(combined),
            None => combined,
        })
    }

    /// Partitions predicates based on a condition.
    fn partition_predicates<F>(
        &self,
        predicates: &[LogicalExpr],
        can_push: F,
    ) -> (Vec<LogicalExpr>, Vec<LogicalExpr>)
    where
        F: Fn(&LogicalExpr) -> bool,
    {
        // Pre-allocate assuming roughly half go to each partition
        let mut pushable = Vec::with_capacity(predicates.len());
        let mut remaining = Vec::with_capacity(predicates.len());

        for pred in predicates {
            if can_push(pred) {
                pushable.push(pred.clone());
            } else {
                remaining.push(pred.clone());
            }
        }

        (pushable, remaining)
    }

    /// Checks if a predicate only references columns (no aggregates).
    fn references_only_scan_columns(&self, _expr: &LogicalExpr) -> bool {
        // Simplified: assume all predicates can be pushed to scan
        // A real implementation would check for aggregate functions
        true
    }

    /// Checks if a predicate only references GROUP BY columns.
    fn references_only_group_by_columns(
        &self,
        expr: &LogicalExpr,
        group_by: &[LogicalExpr],
    ) -> bool {
        let expr_columns = self.collect_columns(expr);
        let group_columns: Vec<String> = group_by
            .iter()
            .filter_map(|e| {
                if let LogicalExpr::Column { name, .. } = e {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        expr_columns.iter().all(|c| group_columns.iter().any(|g| g == c))
    }

    /// Collects column names referenced by an expression.
    fn collect_columns(&self, expr: &LogicalExpr) -> Vec<String> {
        let mut columns = Vec::new();
        self.collect_columns_recursive(expr, &mut columns);
        columns
    }

    fn collect_columns_recursive(&self, expr: &LogicalExpr, columns: &mut Vec<String>) {
        match expr {
            LogicalExpr::Column { name, .. } => {
                columns.push(name.clone());
            }
            LogicalExpr::BinaryOp { left, right, .. } => {
                self.collect_columns_recursive(left, columns);
                self.collect_columns_recursive(right, columns);
            }
            LogicalExpr::UnaryOp { operand, .. } => {
                self.collect_columns_recursive(operand, columns);
            }
            LogicalExpr::ScalarFunction { args, .. } => {
                for arg in args {
                    self.collect_columns_recursive(arg, columns);
                }
            }
            LogicalExpr::AggregateFunction { arg, .. } => {
                self.collect_columns_recursive(arg, columns);
            }
            LogicalExpr::Case { operand, when_clauses, else_result } => {
                if let Some(op) = operand {
                    self.collect_columns_recursive(op, columns);
                }
                for (when_expr, then_expr) in when_clauses {
                    self.collect_columns_recursive(when_expr, columns);
                    self.collect_columns_recursive(then_expr, columns);
                }
                if let Some(else_expr) = else_result {
                    self.collect_columns_recursive(else_expr, columns);
                }
            }
            LogicalExpr::Cast { expr, .. } => {
                self.collect_columns_recursive(expr, columns);
            }
            LogicalExpr::Alias { expr, .. } => {
                self.collect_columns_recursive(expr, columns);
            }
            LogicalExpr::InList { expr, list, .. } => {
                self.collect_columns_recursive(expr, columns);
                for item in list {
                    self.collect_columns_recursive(item, columns);
                }
            }
            LogicalExpr::Between { expr, low, high, .. } => {
                self.collect_columns_recursive(expr, columns);
                self.collect_columns_recursive(low, columns);
                self.collect_columns_recursive(high, columns);
            }
            _ => {}
        }
    }

    /// Collects table names from a plan.
    fn collect_tables(&self, plan: &LogicalPlan) -> Vec<String> {
        let mut tables = Vec::new();
        self.collect_tables_recursive(plan, &mut tables);
        tables
    }

    fn collect_tables_recursive(&self, plan: &LogicalPlan, tables: &mut Vec<String>) {
        match plan {
            LogicalPlan::Scan(node) => {
                tables.push(node.alias.clone().unwrap_or_else(|| node.table_name.clone()));
            }
            LogicalPlan::Alias { alias, input } => {
                tables.push(alias.clone());
                self.collect_tables_recursive(input, tables);
            }
            _ => {
                for child in plan.children() {
                    self.collect_tables_recursive(child, tables);
                }
            }
        }
    }

    /// Collects table references from an expression.
    fn collect_referenced_tables(&self, expr: &LogicalExpr) -> Vec<String> {
        let mut tables = Vec::new();
        self.collect_expr_tables(expr, &mut tables);
        tables
    }

    fn collect_expr_tables(&self, expr: &LogicalExpr, tables: &mut Vec<String>) {
        match expr {
            LogicalExpr::Column { qualifier, .. } => {
                if let Some(t) = qualifier {
                    if !tables.contains(t) {
                        tables.push(t.clone());
                    }
                }
            }
            LogicalExpr::BinaryOp { left, right, .. } => {
                self.collect_expr_tables(left, tables);
                self.collect_expr_tables(right, tables);
            }
            LogicalExpr::UnaryOp { operand, .. } => {
                self.collect_expr_tables(operand, tables);
            }
            _ => {}
        }
    }
}

/// Splits a conjunctive predicate into its AND components.
pub fn split_conjunction(expr: &LogicalExpr) -> Vec<LogicalExpr> {
    let mut parts = Vec::new();
    split_conjunction_recursive(expr, &mut parts);
    parts
}

fn split_conjunction_recursive(expr: &LogicalExpr, parts: &mut Vec<LogicalExpr>) {
    if let LogicalExpr::BinaryOp { left, op, right } = expr {
        if matches!(op, BinaryOp::And) {
            split_conjunction_recursive(left, parts);
            split_conjunction_recursive(right, parts);
            return;
        }
    }
    parts.push(expr.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::LogicalExpr;

    #[test]
    fn push_filter_to_scan() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let optimizer = PredicatePushdown::new();
        let optimized = optimizer.optimize(plan);

        // Filter should be pushed into the scan
        if let LogicalPlan::Scan(node) = &optimized {
            assert!(node.filter.is_some());
        } else {
            panic!("Expected Scan node, got {:?}", optimized.node_type());
        }
    }

    #[test]
    fn push_filter_through_project() {
        let plan = LogicalPlan::scan("users")
            .project(vec![LogicalExpr::column("name"), LogicalExpr::column("age")])
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let optimizer = PredicatePushdown::new();
        let optimized = optimizer.optimize(plan);

        // Filter should be pushed through project into scan
        if let LogicalPlan::Project { input, .. } = &optimized {
            if let LogicalPlan::Scan(node) = input.as_ref() {
                assert!(node.filter.is_some());
            } else {
                panic!("Expected Scan under Project");
            }
        } else {
            panic!("Expected Project at top");
        }
    }

    #[test]
    fn push_filter_through_sort() {
        let plan = LogicalPlan::scan("users")
            .sort(vec![crate::plan::logical::SortOrder::asc(LogicalExpr::column("name"))])
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let optimizer = PredicatePushdown::new();
        let optimized = optimizer.optimize(plan);

        // Sort should be at top, filter pushed to scan
        assert_eq!(optimized.node_type(), "Sort");
    }

    #[test]
    fn multiple_filters_combined() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)))
            .filter(LogicalExpr::column("active").eq(LogicalExpr::boolean(true)));

        let optimizer = PredicatePushdown::new();
        let optimized = optimizer.optimize(plan);

        // Both filters should be combined into scan
        if let LogicalPlan::Scan(node) = &optimized {
            let filter = node.filter.as_ref().expect("should have filter");
            // Should be an AND expression
            if let LogicalExpr::BinaryOp { op, .. } = filter {
                assert!(matches!(op, BinaryOp::And));
            } else {
                panic!("Expected AND expression");
            }
        }
    }

    #[test]
    fn split_conjunction_test() {
        let expr = LogicalExpr::column("a")
            .eq(LogicalExpr::integer(1))
            .and(LogicalExpr::column("b").gt(LogicalExpr::integer(2)));

        let parts = split_conjunction(&expr);
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn push_to_join_sides() {
        let left = LogicalPlan::scan("users");
        let right = LogicalPlan::scan("orders");

        // Filter on left table
        let plan = LogicalPlan::Join {
            node: Box::new(crate::plan::logical::JoinNode::inner(
                LogicalExpr::qualified_column("users", "id")
                    .eq(LogicalExpr::qualified_column("orders", "user_id")),
            )),
            left: Box::new(left),
            right: Box::new(right),
        }
        .filter(LogicalExpr::qualified_column("users", "active").eq(LogicalExpr::boolean(true)));

        let optimizer = PredicatePushdown::new();
        let optimized = optimizer.optimize(plan);

        // The filter on users.active should be pushed to the left side
        if let LogicalPlan::Join { left, .. } = &optimized {
            // Check that left side has the filter
            if let LogicalPlan::Scan(node) = left.as_ref() {
                assert!(node.filter.is_some());
            }
        }
    }
}
