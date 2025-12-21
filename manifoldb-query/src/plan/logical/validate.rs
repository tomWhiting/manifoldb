//! Plan validation.
//!
//! This module provides validation for logical query plans,
//! catching errors before execution.

// Allow long validation function - it's a big match but straightforward
#![allow(clippy::too_many_lines)]
// Allow collapsible if - the nested structure is clearer
#![allow(clippy::collapsible_if)]

use thiserror::Error;

use super::node::LogicalPlan;
use super::relational::JoinType;

/// Errors that can occur during plan validation.
#[derive(Debug, Error)]
pub enum PlanError {
    /// Empty plan (no operations).
    #[error("empty plan")]
    EmptyPlan,

    /// Invalid join condition for the join type.
    #[error("invalid join: {0}")]
    InvalidJoin(String),

    /// Invalid aggregation (e.g., non-aggregate in SELECT with GROUP BY).
    #[error("invalid aggregation: {0}")]
    InvalidAggregation(String),

    /// Reference to unknown column.
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    /// Reference to unknown table.
    #[error("unknown table: {0}")]
    UnknownTable(String),

    /// Invalid ORDER BY expression.
    #[error("invalid order by: {0}")]
    InvalidOrderBy(String),

    /// Invalid LIMIT/OFFSET values.
    #[error("invalid limit/offset: {0}")]
    InvalidLimit(String),

    /// Invalid graph pattern.
    #[error("invalid graph pattern: {0}")]
    InvalidGraphPattern(String),

    /// Invalid vector operation.
    #[error("invalid vector operation: {0}")]
    InvalidVectorOp(String),

    /// Type mismatch.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    /// Unsupported operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// Invalid subquery.
    #[error("invalid subquery: {0}")]
    InvalidSubquery(String),

    /// Cyclic dependency in plan.
    #[error("cyclic dependency detected")]
    CyclicDependency,
}

/// Result type for plan operations.
pub type PlanResult<T> = Result<T, PlanError>;

/// Validates a logical plan.
///
/// This performs basic structural validation without schema information.
/// For full validation (including column references), use the planner's
/// validate method with catalog context.
pub fn validate_plan(plan: &LogicalPlan) -> PlanResult<()> {
    match plan {
        LogicalPlan::Scan(node) => {
            if node.table_name.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
        }

        LogicalPlan::Values(node) => {
            if node.rows.is_empty() {
                // Empty VALUES is allowed but might want to warn
            } else {
                // Check all rows have the same number of columns
                let first_len = node.rows[0].len();
                for (i, row) in node.rows.iter().enumerate().skip(1) {
                    if row.len() != first_len {
                        return Err(PlanError::TypeMismatch(format!(
                            "VALUES row {} has {} columns, expected {}",
                            i + 1,
                            row.len(),
                            first_len
                        )));
                    }
                }
            }
        }

        LogicalPlan::Empty { .. } => {
            // Empty relation is always valid
        }

        LogicalPlan::Filter { node, input } => {
            validate_plan(input)?;
            // Could validate that predicate doesn't contain aggregates
            // but that requires expression analysis
            let _ = node; // Predicate validation would go here
        }

        LogicalPlan::Project { node, input } => {
            validate_plan(input)?;
            if node.exprs.is_empty() {
                return Err(PlanError::InvalidAggregation("empty projection".to_string()));
            }
        }

        LogicalPlan::Aggregate { node, input } => {
            validate_plan(input)?;
            // At least one aggregate or group by expression required
            if node.group_by.is_empty() && node.aggregates.is_empty() {
                return Err(PlanError::InvalidAggregation(
                    "aggregate must have GROUP BY or aggregate expressions".to_string(),
                ));
            }
        }

        LogicalPlan::Sort { node, input } => {
            validate_plan(input)?;
            if node.order_by.is_empty() {
                return Err(PlanError::InvalidOrderBy("empty ORDER BY".to_string()));
            }
        }

        LogicalPlan::Limit { node, input } => {
            validate_plan(input)?;
            if node.limit.is_none() && node.offset.is_none() {
                return Err(PlanError::InvalidLimit(
                    "LIMIT/OFFSET must specify at least one value".to_string(),
                ));
            }
        }

        LogicalPlan::Distinct { input, .. } => {
            validate_plan(input)?;
        }

        LogicalPlan::Alias { alias, input } => {
            validate_plan(input)?;
            if alias.is_empty() {
                return Err(PlanError::UnknownTable("empty alias".to_string()));
            }
        }

        LogicalPlan::Join { node, left, right } => {
            validate_plan(left)?;
            validate_plan(right)?;

            // Cross joins don't need a condition
            if !matches!(node.join_type, JoinType::Cross) {
                if node.condition.is_none() && node.using_columns.is_empty() {
                    return Err(PlanError::InvalidJoin(format!(
                        "{} JOIN requires ON or USING clause",
                        node.join_type
                    )));
                }
            }
        }

        LogicalPlan::SetOp { left, right, .. } => {
            validate_plan(left)?;
            validate_plan(right)?;
            // Could validate that both sides have the same number of columns
        }

        LogicalPlan::Union { inputs, .. } => {
            if inputs.is_empty() {
                return Err(PlanError::EmptyPlan);
            }
            for input in inputs {
                validate_plan(input)?;
            }
        }

        LogicalPlan::Expand { node, input } => {
            validate_plan(input)?;
            if node.src_var.is_empty() {
                return Err(PlanError::InvalidGraphPattern(
                    "expand source variable is empty".to_string(),
                ));
            }
            if node.dst_var.is_empty() {
                return Err(PlanError::InvalidGraphPattern(
                    "expand destination variable is empty".to_string(),
                ));
            }
        }

        LogicalPlan::PathScan { node, input } => {
            validate_plan(input)?;
            if node.steps.is_empty() {
                return Err(PlanError::InvalidGraphPattern(
                    "path scan must have at least one step".to_string(),
                ));
            }
        }

        LogicalPlan::AnnSearch { node, input } => {
            validate_plan(input)?;
            if node.k == 0 {
                return Err(PlanError::InvalidVectorOp(
                    "ANN search k must be greater than 0".to_string(),
                ));
            }
            if node.vector_column.is_empty() {
                return Err(PlanError::InvalidVectorOp("vector column name is empty".to_string()));
            }
        }

        LogicalPlan::VectorDistance { input, .. } => {
            validate_plan(input)?;
        }

        LogicalPlan::Insert { table, input, .. } => {
            validate_plan(input)?;
            if table.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
        }

        LogicalPlan::Update { table, assignments, .. } => {
            if table.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
            if assignments.is_empty() {
                return Err(PlanError::Unsupported(
                    "UPDATE must have at least one assignment".to_string(),
                ));
            }
        }

        LogicalPlan::Delete { table, .. } => {
            if table.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
        }
    }

    Ok(())
}

/// Validates that a plan doesn't have cycles.
///
/// This is mainly a sanity check since the tree structure should
/// prevent cycles, but it's useful for debugging plan transformations.
pub fn check_no_cycles(plan: &LogicalPlan) -> PlanResult<()> {
    // Simple depth limit check (could use proper cycle detection with IDs)
    const MAX_DEPTH: usize = 1000;

    fn check_depth(plan: &LogicalPlan, depth: usize) -> PlanResult<()> {
        if depth > MAX_DEPTH {
            return Err(PlanError::CyclicDependency);
        }
        for child in plan.children() {
            check_depth(child, depth + 1)?;
        }
        Ok(())
    }

    check_depth(plan, 0)
}

#[cfg(test)]
mod tests {
    use super::super::expr::LogicalExpr;
    use super::super::relational::{AggregateNode, LimitNode, SortNode, ValuesNode};
    use super::super::vector::AnnSearchNode;
    use super::*;

    #[test]
    fn valid_simple_scan() {
        let plan = LogicalPlan::scan("users");
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn invalid_empty_table_name() {
        let plan = LogicalPlan::scan("");
        assert!(matches!(validate_plan(&plan), Err(PlanError::UnknownTable(_))));
    }

    #[test]
    fn valid_filter() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn invalid_empty_projection() {
        let plan = LogicalPlan::scan("users").project(vec![]);
        assert!(matches!(validate_plan(&plan), Err(PlanError::InvalidAggregation(_))));
    }

    #[test]
    fn valid_aggregate() {
        let plan = LogicalPlan::Aggregate {
            node: AggregateNode::new(
                vec![LogicalExpr::column("category")],
                vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
            ),
            input: Box::new(LogicalPlan::scan("products")),
        };
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn invalid_empty_aggregate() {
        let plan = LogicalPlan::Aggregate {
            node: AggregateNode::new(vec![], vec![]),
            input: Box::new(LogicalPlan::scan("products")),
        };
        assert!(matches!(validate_plan(&plan), Err(PlanError::InvalidAggregation(_))));
    }

    #[test]
    fn invalid_empty_order_by() {
        let plan = LogicalPlan::Sort {
            node: SortNode::new(vec![]),
            input: Box::new(LogicalPlan::scan("users")),
        };
        assert!(matches!(validate_plan(&plan), Err(PlanError::InvalidOrderBy(_))));
    }

    #[test]
    fn invalid_empty_limit() {
        let plan = LogicalPlan::Limit {
            node: LimitNode { limit: None, offset: None },
            input: Box::new(LogicalPlan::scan("users")),
        };
        assert!(matches!(validate_plan(&plan), Err(PlanError::InvalidLimit(_))));
    }

    #[test]
    fn valid_join() {
        let plan = LogicalPlan::scan("users").inner_join(
            LogicalPlan::scan("orders"),
            LogicalExpr::column("users.id").eq(LogicalExpr::column("orders.user_id")),
        );
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn valid_cross_join() {
        let plan = LogicalPlan::scan("a").cross_join(LogicalPlan::scan("b"));
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn invalid_values_row_length() {
        let plan = LogicalPlan::Values(ValuesNode::new(vec![
            vec![LogicalExpr::integer(1), LogicalExpr::string("a")],
            vec![LogicalExpr::integer(2)], // Wrong length
        ]));
        assert!(matches!(validate_plan(&plan), Err(PlanError::TypeMismatch(_))));
    }

    #[test]
    fn invalid_ann_search_k_zero() {
        let plan = LogicalPlan::AnnSearch {
            node: AnnSearchNode::euclidean("embedding", LogicalExpr::param(1), 0),
            input: Box::new(LogicalPlan::scan("documents")),
        };
        assert!(matches!(validate_plan(&plan), Err(PlanError::InvalidVectorOp(_))));
    }

    #[test]
    fn check_cycles() {
        let plan = LogicalPlan::scan("users")
            .filter(LogicalExpr::boolean(true))
            .project(vec![LogicalExpr::wildcard()]);
        assert!(check_no_cycles(&plan).is_ok());
    }
}
