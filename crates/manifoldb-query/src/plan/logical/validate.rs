//! Plan validation.
//!
//! This module provides validation for logical query plans,
//! catching errors before execution.
//!
//! # Validation Levels
//!
//! - [`validate_plan`]: Basic structural validation (no schema needed)
//! - [`validate_with_schema`]: Full validation including type checking
//! - [`check_no_cycles`]: Sanity check for cyclic dependencies

// Allow long validation function - it's a big match but straightforward
#![allow(clippy::too_many_lines)]
// Allow collapsible if - the nested structure is clearer
#![allow(clippy::collapsible_if)]

use thiserror::Error;

use super::expr::LogicalExpr;
use super::node::LogicalPlan;
use super::relational::JoinType;
use super::schema::SchemaCatalog;
use super::type_infer::TypeError;
use super::types::{PlanType, TypeContext};

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
    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        /// The expected type or value.
        expected: String,
        /// The actual type or value.
        actual: String,
    },

    /// Unsupported operation.
    #[error("unsupported operation: {0}")]
    Unsupported(String),

    /// Invalid subquery.
    #[error("invalid subquery: {0}")]
    InvalidSubquery(String),

    /// Cyclic dependency in plan.
    #[error("cyclic dependency detected")]
    CyclicDependency,

    /// Type inference error.
    #[error("type error: {0}")]
    TypeError(#[from] TypeError),

    /// Expression returns wrong type.
    #[error("expression type error: expected {expected}, found {found}")]
    ExpressionTypeMismatch {
        /// The expected type.
        expected: PlanType,
        /// The actual type found.
        found: PlanType,
    },

    /// Incompatible schema in set operation.
    #[error("incompatible schemas in {operation}: left has {left_count} columns, right has {right_count} columns")]
    SchemaColumnCountMismatch {
        /// The operation (UNION, INTERSECT, etc.).
        operation: String,
        /// Number of columns on the left.
        left_count: usize,
        /// Number of columns on the right.
        right_count: usize,
    },
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
                        return Err(PlanError::TypeMismatch {
                            expected: format!("{} columns", first_len),
                            actual: format!("{} columns in VALUES row {}", row.len(), i + 1),
                        });
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

        LogicalPlan::Window { node, input } => {
            validate_plan(input)?;
            if node.window_exprs.is_empty() {
                return Err(PlanError::InvalidAggregation(
                    "window node must have at least one window expression".to_string(),
                ));
            }
        }

        LogicalPlan::Alias { alias, input } => {
            validate_plan(input)?;
            if alias.is_empty() {
                return Err(PlanError::UnknownTable("empty alias".to_string()));
            }
        }

        LogicalPlan::Unwind { node, input } => {
            validate_plan(input)?;
            if node.alias.is_empty() {
                return Err(PlanError::UnknownColumn("UNWIND alias cannot be empty".to_string()));
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

        LogicalPlan::RecursiveCTE { node, initial, recursive } => {
            validate_plan(initial)?;
            validate_plan(recursive)?;
            if node.name.is_empty() {
                return Err(PlanError::UnknownTable("recursive CTE name is empty".to_string()));
            }
            // Max iterations check - if specified, must be > 0
            if let Some(max) = node.max_iterations {
                if max == 0 {
                    return Err(PlanError::InvalidLimit(
                        "recursive CTE max_iterations must be > 0".to_string(),
                    ));
                }
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

        LogicalPlan::ShortestPath { node, input } => {
            validate_plan(input)?;
            if node.src_var.is_empty() {
                return Err(PlanError::InvalidGraphPattern(
                    "shortest path source variable is empty".to_string(),
                ));
            }
            if node.dst_var.is_empty() {
                return Err(PlanError::InvalidGraphPattern(
                    "shortest path destination variable is empty".to_string(),
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

        LogicalPlan::CreateTable(node) => {
            if node.name.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
            if node.columns.is_empty() {
                return Err(PlanError::Unsupported(
                    "CREATE TABLE must have at least one column".to_string(),
                ));
            }
        }

        LogicalPlan::AlterTable(node) => {
            if node.name.is_empty() {
                return Err(PlanError::UnknownTable("empty table name".to_string()));
            }
            if node.actions.is_empty() {
                return Err(PlanError::Unsupported(
                    "ALTER TABLE must have at least one action".to_string(),
                ));
            }
        }

        LogicalPlan::DropTable(node) => {
            if node.names.is_empty() {
                return Err(PlanError::UnknownTable("DROP TABLE requires table name".to_string()));
            }
        }

        LogicalPlan::CreateIndex(node) => {
            if node.name.is_empty() {
                return Err(PlanError::Unsupported("CREATE INDEX requires index name".to_string()));
            }
            if node.table.is_empty() {
                return Err(PlanError::UnknownTable(
                    "CREATE INDEX requires table name".to_string(),
                ));
            }
            if node.columns.is_empty() {
                return Err(PlanError::Unsupported(
                    "CREATE INDEX must specify at least one column".to_string(),
                ));
            }
        }

        LogicalPlan::DropIndex(node) => {
            if node.names.is_empty() {
                return Err(PlanError::Unsupported("DROP INDEX requires index name".to_string()));
            }
        }

        LogicalPlan::CreateCollection(node) => {
            if node.name.is_empty() {
                return Err(PlanError::Unsupported(
                    "CREATE COLLECTION requires a name".to_string(),
                ));
            }
        }

        LogicalPlan::DropCollection(node) => {
            if node.names.is_empty() {
                return Err(PlanError::Unsupported(
                    "DROP COLLECTION requires collection name".to_string(),
                ));
            }
        }

        LogicalPlan::CreateView(node) => {
            if node.name.is_empty() {
                return Err(PlanError::Unsupported("CREATE VIEW requires a name".to_string()));
            }
        }

        LogicalPlan::DropView(node) => {
            if node.names.is_empty() {
                return Err(PlanError::Unsupported("DROP VIEW requires view name".to_string()));
            }
        }

        LogicalPlan::HybridSearch { node, input } => {
            validate_plan(input)?;
            if node.components.is_empty() {
                return Err(PlanError::InvalidAggregation(
                    "hybrid search requires at least one component".to_string(),
                ));
            }
            if node.k == 0 {
                return Err(PlanError::InvalidAggregation(
                    "hybrid search k must be > 0".to_string(),
                ));
            }
        }
        LogicalPlan::GraphCreate { input, .. } => {
            if let Some(input) = input {
                validate_plan(input)?;
            }
        }
        LogicalPlan::GraphMerge { input, .. } => {
            if let Some(input) = input {
                validate_plan(input)?;
            }
        }
        LogicalPlan::GraphSet { input, .. } => {
            validate_plan(input)?;
        }
        LogicalPlan::GraphDelete { input, .. } => {
            validate_plan(input)?;
        }
        LogicalPlan::GraphRemove { input, .. } => {
            validate_plan(input)?;
        }
        LogicalPlan::GraphForeach { input, .. } => {
            validate_plan(input)?;
        }

        LogicalPlan::ProcedureCall(node) => {
            if node.procedure_name.is_empty() {
                return Err(PlanError::Unsupported("CALL requires procedure name".to_string()));
            }
        }

        // Transaction control statements are always valid structurally
        LogicalPlan::BeginTransaction(_)
        | LogicalPlan::Commit(_)
        | LogicalPlan::Rollback(_)
        | LogicalPlan::Savepoint(_)
        | LogicalPlan::ReleaseSavepoint(_)
        | LogicalPlan::SetTransaction(_) => {
            // Transaction statements have no structural validation requirements
        }

        // CALL { } subquery validation
        LogicalPlan::CallSubquery { node, subquery, input } => {
            // Validate input plan
            validate_plan(input)?;
            // Validate subquery plan
            validate_plan(subquery)?;
            // Imported variables should be valid identifiers (basic check)
            for var in &node.imported_variables {
                if var.is_empty() {
                    return Err(PlanError::Unsupported(
                        "CALL WITH requires non-empty variable names".to_string(),
                    ));
                }
            }
        }

        // Utility statements are always valid structurally
        LogicalPlan::ExplainAnalyze(node) => {
            // Validate the inner plan
            validate_plan(&node.input)?;
        }
        LogicalPlan::Vacuum(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Copy(_)
        | LogicalPlan::SetSession(_)
        | LogicalPlan::Show(_)
        | LogicalPlan::Reset(_) => {
            // Utility statements have no structural validation requirements
        }
    }

    Ok(())
}

/// Validates a logical plan with schema information.
///
/// This performs full validation including:
/// - Structural validation (same as [`validate_plan`])
/// - Column reference validation (ensures columns exist)
/// - Type checking for expressions
/// - Schema compatibility for set operations
///
/// # Arguments
///
/// * `plan` - The logical plan to validate
/// * `catalog` - A catalog for looking up table schemas
///
/// # Example
///
/// ```ignore
/// use manifoldb_query::plan::logical::{LogicalPlan, validate_with_schema, EmptyCatalog};
///
/// let plan = LogicalPlan::scan("users")
///     .filter(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));
///
/// validate_with_schema(&plan, &EmptyCatalog)?;
/// ```
pub fn validate_with_schema(plan: &LogicalPlan, catalog: &dyn SchemaCatalog) -> PlanResult<()> {
    // First do structural validation
    validate_plan(plan)?;

    // Then do schema-aware validation
    validate_schema_recursive(plan, catalog)
}

/// Recursively validates schema information through the plan tree.
fn validate_schema_recursive(plan: &LogicalPlan, catalog: &dyn SchemaCatalog) -> PlanResult<()> {
    match plan {
        LogicalPlan::Filter { node, input } => {
            validate_schema_recursive(input, catalog)?;

            // Validate that the filter predicate is well-typed and returns boolean
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            let pred_type = node.predicate.infer_type(&ctx)?;
            if !matches!(pred_type, PlanType::Boolean | PlanType::Any | PlanType::Null) {
                return Err(PlanError::ExpressionTypeMismatch {
                    expected: PlanType::Boolean,
                    found: pred_type,
                });
            }
        }

        LogicalPlan::Project { node, input } => {
            validate_schema_recursive(input, catalog)?;

            // Validate that all projection expressions are well-typed
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            for expr in &node.exprs {
                validate_expression(expr, &ctx)?;
            }
        }

        LogicalPlan::Aggregate { node, input } => {
            validate_schema_recursive(input, catalog)?;

            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);

            // Validate group by expressions
            for expr in &node.group_by {
                validate_expression(expr, &ctx)?;
            }

            // Validate aggregate expressions
            for expr in &node.aggregates {
                validate_expression(expr, &ctx)?;
            }
        }

        LogicalPlan::Sort { node, input } => {
            validate_schema_recursive(input, catalog)?;

            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            for sort_order in &node.order_by {
                validate_expression(&sort_order.expr, &ctx)?;
            }
        }

        LogicalPlan::Join { node, left, right } => {
            validate_schema_recursive(left, catalog)?;
            validate_schema_recursive(right, catalog)?;

            // Validate join condition if present
            if let Some(ref condition) = node.condition {
                let left_schema = left.output_schema(catalog)?;
                let right_schema = right.output_schema(catalog)?;
                let combined = left_schema.merge(&right_schema);
                let ctx = TypeContext::with_schema(combined);
                let cond_type = condition.infer_type(&ctx)?;
                if !matches!(cond_type, PlanType::Boolean | PlanType::Any | PlanType::Null) {
                    return Err(PlanError::ExpressionTypeMismatch {
                        expected: PlanType::Boolean,
                        found: cond_type,
                    });
                }
            }
        }

        LogicalPlan::SetOp { node, left, right } => {
            validate_schema_recursive(left, catalog)?;
            validate_schema_recursive(right, catalog)?;

            // Validate that both sides have the same number of columns
            let left_schema = left.output_schema(catalog)?;
            let right_schema = right.output_schema(catalog)?;
            if left_schema.len() != right_schema.len() {
                return Err(PlanError::SchemaColumnCountMismatch {
                    operation: format!("{:?}", node.op_type),
                    left_count: left_schema.len(),
                    right_count: right_schema.len(),
                });
            }
        }

        LogicalPlan::Union { inputs, .. } => {
            if inputs.is_empty() {
                return Ok(());
            }

            for input in inputs {
                validate_schema_recursive(input, catalog)?;
            }

            // Validate all inputs have the same column count
            let first_schema = inputs[0].output_schema(catalog)?;
            for (i, input) in inputs.iter().enumerate().skip(1) {
                let schema = input.output_schema(catalog)?;
                if schema.len() != first_schema.len() {
                    return Err(PlanError::SchemaColumnCountMismatch {
                        operation: format!("UNION (input {})", i + 1),
                        left_count: first_schema.len(),
                        right_count: schema.len(),
                    });
                }
            }
        }

        // Recurse through other plan nodes
        LogicalPlan::Scan(_) | LogicalPlan::Values(_) | LogicalPlan::Empty { .. } => {
            // Leaf nodes - no further validation needed
        }

        LogicalPlan::Limit { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Alias { input, .. } => {
            validate_schema_recursive(input, catalog)?;
        }

        LogicalPlan::Window { node, input } => {
            validate_schema_recursive(input, catalog)?;
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            for (expr, _) in &node.window_exprs {
                validate_expression(expr, &ctx)?;
            }
        }

        LogicalPlan::Unwind { node, input } => {
            validate_schema_recursive(input, catalog)?;
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            // Validate the list expression
            let list_type = node.list_expr.infer_type(&ctx)?;
            if !list_type.is_collection() && !matches!(list_type, PlanType::Any) {
                return Err(PlanError::TypeMismatch {
                    expected: "list or array".to_string(),
                    actual: format!("{}", list_type),
                });
            }
        }

        LogicalPlan::RecursiveCTE { initial, recursive, .. } => {
            validate_schema_recursive(initial, catalog)?;
            validate_schema_recursive(recursive, catalog)?;
        }

        LogicalPlan::CallSubquery { subquery, input, .. } => {
            validate_schema_recursive(input, catalog)?;
            validate_schema_recursive(subquery, catalog)?;
        }

        // Graph operations
        LogicalPlan::Expand { input, .. }
        | LogicalPlan::PathScan { input, .. }
        | LogicalPlan::ShortestPath { input, .. } => {
            validate_schema_recursive(input, catalog)?;
        }

        // Vector operations
        LogicalPlan::AnnSearch { node, input } => {
            validate_schema_recursive(input, catalog)?;
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            validate_expression(&node.query_vector, &ctx)?;
        }

        LogicalPlan::VectorDistance { node, input } => {
            validate_schema_recursive(input, catalog)?;
            let input_schema = input.output_schema(catalog)?;
            let ctx = TypeContext::with_schema(input_schema);
            validate_expression(&node.left, &ctx)?;
            validate_expression(&node.right, &ctx)?;
        }

        LogicalPlan::HybridSearch { input, .. } => {
            validate_schema_recursive(input, catalog)?;
        }

        // DML operations
        LogicalPlan::Insert { input, .. } => {
            validate_schema_recursive(input, catalog)?;
        }

        LogicalPlan::Update { filter, .. } => {
            if let Some(ref f) = filter {
                // We don't have the table schema here without catalog lookup
                // So we just do basic validation
                let ctx = TypeContext::new();
                let filter_type = f.infer_type(&ctx)?;
                if !matches!(filter_type, PlanType::Boolean | PlanType::Any | PlanType::Null) {
                    return Err(PlanError::ExpressionTypeMismatch {
                        expected: PlanType::Boolean,
                        found: filter_type,
                    });
                }
            }
        }

        LogicalPlan::Delete { filter, .. } => {
            if let Some(ref f) = filter {
                let ctx = TypeContext::new();
                let filter_type = f.infer_type(&ctx)?;
                if !matches!(filter_type, PlanType::Boolean | PlanType::Any | PlanType::Null) {
                    return Err(PlanError::ExpressionTypeMismatch {
                        expected: PlanType::Boolean,
                        found: filter_type,
                    });
                }
            }
        }

        // DDL operations - no schema validation needed
        LogicalPlan::CreateTable(_)
        | LogicalPlan::AlterTable(_)
        | LogicalPlan::DropTable(_)
        | LogicalPlan::CreateIndex(_)
        | LogicalPlan::DropIndex(_)
        | LogicalPlan::CreateCollection(_)
        | LogicalPlan::DropCollection(_)
        | LogicalPlan::CreateView(_)
        | LogicalPlan::DropView(_) => {}

        // Graph DML
        LogicalPlan::GraphCreate { input, .. } | LogicalPlan::GraphMerge { input, .. } => {
            if let Some(input) = input {
                validate_schema_recursive(input, catalog)?;
            }
        }

        LogicalPlan::GraphSet { input, .. }
        | LogicalPlan::GraphDelete { input, .. }
        | LogicalPlan::GraphRemove { input, .. }
        | LogicalPlan::GraphForeach { input, .. } => {
            validate_schema_recursive(input, catalog)?;
        }

        // Procedure calls - no schema validation
        LogicalPlan::ProcedureCall(_) => {}

        // Transaction control - no schema validation
        LogicalPlan::BeginTransaction(_)
        | LogicalPlan::Commit(_)
        | LogicalPlan::Rollback(_)
        | LogicalPlan::Savepoint(_)
        | LogicalPlan::ReleaseSavepoint(_)
        | LogicalPlan::SetTransaction(_) => {}

        // Utility statements
        LogicalPlan::ExplainAnalyze(node) => {
            validate_schema_recursive(&node.input, catalog)?;
        }

        LogicalPlan::Vacuum(_)
        | LogicalPlan::Analyze(_)
        | LogicalPlan::Copy(_)
        | LogicalPlan::SetSession(_)
        | LogicalPlan::Show(_)
        | LogicalPlan::Reset(_) => {}
    }

    Ok(())
}

/// Validates an expression in the given type context.
fn validate_expression(expr: &LogicalExpr, ctx: &TypeContext) -> PlanResult<()> {
    // Try to infer the type - this will catch unknown columns, etc.
    let _ = expr.infer_type(ctx)?;
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
            node: Box::new(AggregateNode::new(
                vec![LogicalExpr::column("category")],
                vec![LogicalExpr::count(LogicalExpr::wildcard(), false)],
            )),
            input: Box::new(LogicalPlan::scan("products")),
        };
        assert!(validate_plan(&plan).is_ok());
    }

    #[test]
    fn invalid_empty_aggregate() {
        let plan = LogicalPlan::Aggregate {
            node: Box::new(AggregateNode::new(vec![], vec![])),
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
        assert!(matches!(
            validate_plan(&plan),
            Err(PlanError::TypeMismatch { expected: _, actual: _ })
        ));
    }

    #[test]
    fn invalid_ann_search_k_zero() {
        let plan = LogicalPlan::AnnSearch {
            node: Box::new(AnnSearchNode::euclidean("embedding", LogicalExpr::param(1), 0)),
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
