//! SQL query and statement execution.
//!
//! This module provides functions to execute SQL queries and statements
//! against the storage layer.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::{Entity, Value};
use manifoldb_query::ast::Literal;
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::{ExecutionContext, ResultSet};
use manifoldb_query::parse_single_statement;
use manifoldb_query::plan::logical::LogicalExpr;
use manifoldb_query::plan::{LogicalPlan, PhysicalPlan, PhysicalPlanner, PlanBuilder};
use manifoldb_storage::Transaction;

use super::StorageScan;
use crate::error::{Error, Result};
use crate::transaction::DatabaseTransaction;

/// Execute a SELECT query and return the result set.
pub fn execute_query<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
) -> Result<ResultSet> {
    // Parse SQL
    let stmt = parse_single_statement(sql)?;

    // Build logical plan
    let mut builder = PlanBuilder::new();
    let logical_plan = builder.build_statement(&stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Build physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(&logical_plan);

    // Create execution context with parameters
    let ctx = create_context(params);

    // Execute the plan against storage
    execute_physical_plan(tx, &physical_plan, &logical_plan, &ctx)
}

/// Execute a DML statement (INSERT, UPDATE, DELETE) and return the affected row count.
pub fn execute_statement<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    sql: &str,
    params: &[Value],
) -> Result<u64> {
    // Parse SQL
    let stmt = parse_single_statement(sql)?;

    // Build logical plan
    let mut builder = PlanBuilder::new();
    let logical_plan = builder.build_statement(&stmt).map_err(|e| Error::Parse(e.to_string()))?;

    // Create execution context with parameters
    let ctx = create_context(params);

    // Execute based on the statement type
    match &logical_plan {
        LogicalPlan::Insert { table, columns, input, .. } => {
            execute_insert(tx, table, columns, input, &ctx)
        }
        LogicalPlan::Update { table, assignments, filter, .. } => {
            execute_update(tx, table, assignments, filter, &ctx)
        }
        LogicalPlan::Delete { table, filter, .. } => execute_delete(tx, table, filter, &ctx),
        _ => {
            // For SELECT, we shouldn't be here but handle gracefully
            Err(Error::Execution("Expected DML statement".to_string()))
        }
    }
}

/// Create an execution context with bound parameters.
fn create_context(params: &[Value]) -> ExecutionContext {
    let mut param_map = HashMap::new();
    for (i, value) in params.iter().enumerate() {
        param_map.insert((i + 1) as u32, value.clone());
    }
    ExecutionContext::with_parameters(param_map)
}

/// Execute a physical plan and return the result set.
fn execute_physical_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    _physical: &PhysicalPlan,
    logical: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<ResultSet> {
    // For now, we'll implement a simple interpreter that handles basic queries
    // A full implementation would convert the physical plan to operators with storage access
    // The physical plan is currently unused but will be needed for optimization

    match logical {
        LogicalPlan::Project { node, input } => {
            // First execute the input
            let input_result = execute_logical_plan(tx, input, ctx)?;

            // Check if we have a wildcard projection (SELECT *)
            let has_wildcard = node.exprs.iter().any(|e| matches!(e, LogicalExpr::Wildcard));

            if has_wildcard {
                // For SELECT *, expand to all columns from the entities
                let columns = collect_all_columns(&input_result);
                let scan = StorageScan::new(input_result, columns);
                let schema = scan.schema();
                let rows = scan.collect_rows();
                Ok(ResultSet::with_rows(schema, rows))
            } else {
                // Normal projection
                let projected_columns: Vec<String> =
                    node.exprs.iter().map(|e| expr_to_column_name(e)).collect();

                let schema = Arc::new(Schema::new(projected_columns.clone()));
                let mut rows = Vec::new();

                for entity in &input_result {
                    let values: Vec<Value> =
                        node.exprs.iter().map(|expr| evaluate_expr(expr, entity, ctx)).collect();
                    rows.push(Row::new(Arc::clone(&schema), values));
                }

                Ok(ResultSet::with_rows(schema, rows))
            }
        }
        _ => {
            // For other plan types, execute and return
            let entities = execute_logical_plan(tx, logical, ctx)?;
            let columns = collect_all_columns(&entities);
            let scan = StorageScan::new(entities, columns);
            let schema = scan.schema();
            let rows = scan.collect_rows();

            Ok(ResultSet::with_rows(schema, rows))
        }
    }
}

/// Collect all unique column names from a set of entities.
fn collect_all_columns(entities: &[Entity]) -> Vec<String> {
    if entities.is_empty() {
        return vec![];
    }

    // Start with id, then add all unique property keys
    let mut cols: Vec<String> = vec!["id".to_string()];
    for entity in entities {
        for key in entity.properties.keys() {
            if !cols.contains(key) {
                cols.push(key.clone());
            }
        }
    }
    cols
}

/// Execute a logical plan and return matching entities.
fn execute_logical_plan<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<Vec<Entity>> {
    match plan {
        LogicalPlan::Scan(scan_node) => {
            // The table name is the entity label
            let label = &scan_node.table_name;
            let entities = tx.iter_entities(Some(label)).map_err(Error::Transaction)?;
            Ok(entities)
        }

        LogicalPlan::Filter { node, input } => {
            let entities = execute_logical_plan(tx, input, ctx)?;

            // Filter entities based on predicate
            let filtered: Vec<Entity> = entities
                .into_iter()
                .filter(|entity| evaluate_predicate(&node.predicate, entity, ctx))
                .collect();

            Ok(filtered)
        }

        LogicalPlan::Project { input, .. } => {
            // Projection doesn't change the entities, just what we extract
            execute_logical_plan(tx, input, ctx)
        }

        LogicalPlan::Limit { node, input } => {
            let entities = execute_logical_plan(tx, input, ctx)?;

            let start = node.offset.unwrap_or(0);
            let end = node.limit.map(|l| start + l).unwrap_or(entities.len());

            Ok(entities.into_iter().skip(start).take(end - start).collect())
        }

        LogicalPlan::Sort { node, input } => {
            let mut entities = execute_logical_plan(tx, input, ctx)?;

            // Sort by the first order-by expression
            if let Some(order) = node.order_by.first() {
                entities.sort_by(|a, b| {
                    let va = evaluate_expr(&order.expr, a, ctx);
                    let vb = evaluate_expr(&order.expr, b, ctx);
                    let cmp = compare_values(&va, &vb);
                    if order.ascending {
                        cmp
                    } else {
                        cmp.reverse()
                    }
                });
            }

            Ok(entities)
        }

        LogicalPlan::Values(_) => {
            // VALUES clause doesn't read from storage
            // Just return empty for now (used for INSERT)
            Ok(Vec::new())
        }

        LogicalPlan::Empty { .. } => Ok(Vec::new()),

        _ => {
            // For unimplemented plan types, return empty
            Ok(Vec::new())
        }
    }
}

/// Execute an INSERT statement.
fn execute_insert<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    columns: &[String],
    input: &LogicalPlan,
    ctx: &ExecutionContext,
) -> Result<u64> {
    let mut count = 0;

    // Extract values from the input plan
    if let LogicalPlan::Values(values_node) = input {
        for row_exprs in &values_node.rows {
            // Create a new entity with the table name as label
            let mut entity = tx.create_entity().map_err(Error::Transaction)?;
            entity = entity.with_label(table);

            // Set properties from columns and values
            for (i, col) in columns.iter().enumerate() {
                if let Some(expr) = row_exprs.get(i) {
                    let value = evaluate_literal_expr(expr, ctx);
                    entity = entity.with_property(col, value);
                }
            }

            tx.put_entity(&entity).map_err(Error::Transaction)?;
            count += 1;
        }
    }

    Ok(count)
}

/// Execute an UPDATE statement.
fn execute_update<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    assignments: &[(String, LogicalExpr)],
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    // Get all entities with this label
    let entities = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    let mut count = 0;

    for mut entity in entities {
        // Check if entity matches filter
        let matches = match filter {
            Some(pred) => evaluate_predicate(pred, &entity, ctx),
            None => true,
        };

        if matches {
            // Apply assignments
            for (col, expr) in assignments {
                let value = evaluate_expr(expr, &entity, ctx);
                entity.set_property(col, value);
            }

            tx.put_entity(&entity).map_err(Error::Transaction)?;
            count += 1;
        }
    }

    Ok(count)
}

/// Execute a DELETE statement.
fn execute_delete<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    table: &str,
    filter: &Option<LogicalExpr>,
    ctx: &ExecutionContext,
) -> Result<u64> {
    // Get all entities with this label
    let entities = tx.iter_entities(Some(table)).map_err(Error::Transaction)?;

    let mut count = 0;

    for entity in entities {
        // Check if entity matches filter
        let matches = match filter {
            Some(pred) => evaluate_predicate(&pred, &entity, ctx),
            None => true,
        };

        if matches {
            tx.delete_entity(entity.id).map_err(Error::Transaction)?;
            count += 1;
        }
    }

    Ok(count)
}

/// Evaluate a logical expression to a value.
fn evaluate_expr(expr: &LogicalExpr, entity: &Entity, ctx: &ExecutionContext) -> Value {
    match expr {
        LogicalExpr::Literal(lit) => literal_to_value(lit),

        LogicalExpr::Column { name, .. } => {
            if name == "id" || name == "_id" {
                Value::Int(entity.id.as_u64() as i64)
            } else {
                entity.get_property(name).cloned().unwrap_or(Value::Null)
            }
        }

        LogicalExpr::Parameter(idx) => {
            ctx.get_parameter(*idx as u32).cloned().unwrap_or(Value::Null)
        }

        LogicalExpr::Alias { expr, .. } => evaluate_expr(expr, entity, ctx),

        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr(left, entity, ctx);
            let rval = evaluate_expr(right, entity, ctx);
            evaluate_binary_op(op, &lval, &rval)
        }

        LogicalExpr::Wildcard => Value::Null, // Wildcard returns null in value context

        _ => Value::Null,
    }
}

/// Evaluate a literal expression (for INSERT VALUES).
fn evaluate_literal_expr(expr: &LogicalExpr, ctx: &ExecutionContext) -> Value {
    match expr {
        LogicalExpr::Literal(lit) => literal_to_value(lit),
        LogicalExpr::Parameter(idx) => {
            ctx.get_parameter(*idx as u32).cloned().unwrap_or(Value::Null)
        }
        _ => Value::Null,
    }
}

/// Convert an AST literal to a Value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Boolean(b) => Value::Bool(*b),
        Literal::Integer(n) => Value::Int(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Vector(v) => Value::Vector(v.clone()),
    }
}

/// Evaluate a predicate expression to a boolean.
fn evaluate_predicate(expr: &LogicalExpr, entity: &Entity, ctx: &ExecutionContext) -> bool {
    match expr {
        LogicalExpr::Literal(Literal::Boolean(b)) => *b,

        LogicalExpr::BinaryOp { left, op, right } => {
            let lval = evaluate_expr(left, entity, ctx);
            let rval = evaluate_expr(right, entity, ctx);

            use manifoldb_query::ast::BinaryOp;
            match op {
                BinaryOp::Eq => values_equal(&lval, &rval),
                BinaryOp::NotEq => !values_equal(&lval, &rval),
                BinaryOp::Lt => compare_values(&lval, &rval) == std::cmp::Ordering::Less,
                BinaryOp::LtEq => {
                    matches!(
                        compare_values(&lval, &rval),
                        std::cmp::Ordering::Less | std::cmp::Ordering::Equal
                    )
                }
                BinaryOp::Gt => compare_values(&lval, &rval) == std::cmp::Ordering::Greater,
                BinaryOp::GtEq => {
                    matches!(
                        compare_values(&lval, &rval),
                        std::cmp::Ordering::Greater | std::cmp::Ordering::Equal
                    )
                }
                BinaryOp::And => {
                    evaluate_predicate(left, entity, ctx) && evaluate_predicate(right, entity, ctx)
                }
                BinaryOp::Or => {
                    evaluate_predicate(left, entity, ctx) || evaluate_predicate(right, entity, ctx)
                }
                BinaryOp::Like => {
                    // Simple LIKE implementation (just prefix/suffix for now)
                    if let (Value::String(s), Value::String(pattern)) = (&lval, &rval) {
                        simple_like_match(s, pattern)
                    } else {
                        false
                    }
                }
                _ => false,
            }
        }

        LogicalExpr::UnaryOp { op, operand } => {
            use manifoldb_query::ast::UnaryOp;
            match op {
                UnaryOp::Not => !evaluate_predicate(operand, entity, ctx),
                UnaryOp::IsNull => {
                    matches!(evaluate_expr(operand, entity, ctx), Value::Null)
                }
                UnaryOp::IsNotNull => !matches!(evaluate_expr(operand, entity, ctx), Value::Null),
                _ => false,
            }
        }

        LogicalExpr::InList { expr, list, negated } => {
            let val = evaluate_expr(expr, entity, ctx);
            let in_list = list.iter().any(|item| {
                let item_val = evaluate_expr(item, entity, ctx);
                values_equal(&val, &item_val)
            });
            if *negated {
                !in_list
            } else {
                in_list
            }
        }

        LogicalExpr::Between { expr, low, high, negated } => {
            let val = evaluate_expr(expr, entity, ctx);
            let low_val = evaluate_expr(low, entity, ctx);
            let high_val = evaluate_expr(high, entity, ctx);

            let in_range = compare_values(&val, &low_val) != std::cmp::Ordering::Less
                && compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

            if *negated {
                !in_range
            } else {
                in_range
            }
        }

        _ => true, // Default to true for unhandled expressions
    }
}

/// Evaluate a binary operation on two values.
fn evaluate_binary_op(op: &manifoldb_query::ast::BinaryOp, lval: &Value, rval: &Value) -> Value {
    use manifoldb_query::ast::BinaryOp;

    match op {
        BinaryOp::Add => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a + b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 + b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a + *b as f64),
            (Value::String(a), Value::String(b)) => Value::String(format!("{a}{b}")),
            _ => Value::Null,
        },
        BinaryOp::Sub => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a - b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 - b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a - *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Mul => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
            (Value::Float(a), Value::Float(b)) => Value::Float(a * b),
            (Value::Int(a), Value::Float(b)) => Value::Float(*a as f64 * b),
            (Value::Float(a), Value::Int(b)) => Value::Float(a * *b as f64),
            _ => Value::Null,
        },
        BinaryOp::Div => match (lval, rval) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Value::Int(a / b),
            (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
            (Value::Int(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
            (Value::Float(a), Value::Int(b)) if *b != 0 => Value::Float(a / *b as f64),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

/// Check if two values are equal.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bytes(a), Value::Bytes(b)) => a == b,
        (Value::Vector(a), Value::Vector(b)) => a == b,
        _ => false,
    }
}

/// Compare two values for ordering.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Simple LIKE pattern matching.
fn simple_like_match(s: &str, pattern: &str) -> bool {
    if pattern.starts_with('%') && pattern.ends_with('%') {
        // Contains
        let needle = &pattern[1..pattern.len() - 1];
        s.contains(needle)
    } else if pattern.starts_with('%') {
        // Ends with
        s.ends_with(&pattern[1..])
    } else if pattern.ends_with('%') {
        // Starts with
        s.starts_with(&pattern[..pattern.len() - 1])
    } else {
        // Exact match
        s == pattern
    }
}

/// Extract column name from an expression.
fn expr_to_column_name(expr: &LogicalExpr) -> String {
    match expr {
        LogicalExpr::Column { name, .. } => name.clone(),
        LogicalExpr::Alias { alias, .. } => alias.clone(),
        LogicalExpr::Wildcard => "*".to_string(),
        LogicalExpr::AggregateFunction { func, .. } => format!("{func:?}"),
        LogicalExpr::Literal(lit) => format!("{lit:?}"),
        _ => "?".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_to_value() {
        assert_eq!(literal_to_value(&Literal::Null), Value::Null);
        assert_eq!(literal_to_value(&Literal::Boolean(true)), Value::Bool(true));
        assert_eq!(literal_to_value(&Literal::Integer(42)), Value::Int(42));
        assert_eq!(literal_to_value(&Literal::Float(1.5)), Value::Float(1.5));
        assert_eq!(
            literal_to_value(&Literal::String("hello".to_string())),
            Value::String("hello".to_string())
        );
    }

    #[test]
    fn test_simple_like_match() {
        assert!(simple_like_match("hello", "hello"));
        assert!(simple_like_match("hello", "hel%"));
        assert!(simple_like_match("hello", "%llo"));
        assert!(simple_like_match("hello", "%ell%"));
        assert!(!simple_like_match("hello", "world"));
        assert!(!simple_like_match("hello", "hi%"));
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(42), &Value::Int(42)));
        assert!(values_equal(&Value::Float(1.5), &Value::Float(1.5)));
        assert!(values_equal(&Value::Int(42), &Value::Float(42.0)));
        assert!(!values_equal(&Value::Int(42), &Value::Int(43)));
        assert!(values_equal(&Value::Null, &Value::Null));
    }

    #[test]
    fn test_compare_values() {
        use std::cmp::Ordering;

        assert_eq!(compare_values(&Value::Int(1), &Value::Int(2)), Ordering::Less);
        assert_eq!(compare_values(&Value::Int(2), &Value::Int(1)), Ordering::Greater);
        assert_eq!(compare_values(&Value::Int(1), &Value::Int(1)), Ordering::Equal);
        assert_eq!(compare_values(&Value::Null, &Value::Int(1)), Ordering::Less);
    }
}
