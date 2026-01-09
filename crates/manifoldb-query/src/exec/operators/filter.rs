//! Filter operator for predicate evaluation.

use std::sync::Arc;

use manifoldb_core::{EdgeType, EntityId, Value};
use manifoldb_graph::traversal::Direction;

use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::exec::context::ExecutionContext;
use crate::exec::executor::build_operator_tree;
use crate::exec::graph_accessor::GraphAccessor;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{
    ExpandDirection, ExpandLength, ExpandNode, HybridCombinationMethod, LogicalExpr,
    LogicalMapProjectionItem, LogicalPlan,
};
use crate::plan::physical::PhysicalPlanner;

/// Filter operator.
///
/// Evaluates a predicate for each input row and only passes through
/// rows where the predicate evaluates to true.
pub struct FilterOp {
    /// Base operator state.
    base: OperatorBase,
    /// The predicate to evaluate.
    predicate: LogicalExpr,
    /// Input operator.
    input: BoxedOperator,
    /// Graph accessor for EXISTS/COUNT subqueries.
    graph: Option<Arc<dyn GraphAccessor>>,
    /// Execution context for SQL subquery execution.
    ctx: Option<ExecutionContext>,
}

impl FilterOp {
    /// Creates a new filter operator.
    #[must_use]
    pub fn new(predicate: LogicalExpr, input: BoxedOperator) -> Self {
        let schema = input.schema();
        Self { base: OperatorBase::new(schema), predicate, input, graph: None, ctx: None }
    }

    /// Returns the predicate.
    #[must_use]
    pub fn predicate(&self) -> &LogicalExpr {
        &self.predicate
    }

    /// Evaluates the predicate against a row.
    fn evaluate_predicate(&self, row: &Row) -> OperatorResult<bool> {
        let value =
            evaluate_expr_with_subquery(&self.predicate, row, self.graph.as_deref(), &self.ctx)?;
        match value {
            Value::Bool(b) => Ok(b),
            Value::Null => Ok(false), // NULL is treated as false
            _ => Ok(false),
        }
    }
}

impl Operator for FilterOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        // Capture the graph accessor for EXISTS/COUNT subquery evaluation
        self.graph = Some(ctx.graph_arc());
        // Clone the execution context for SQL subquery execution
        // Note: We create a new context with the same graph accessor
        self.ctx = Some(ExecutionContext::new().with_graph(ctx.graph_arc()));
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            match self.input.next()? {
                Some(row) => {
                    if self.evaluate_predicate(&row)? {
                        self.base.inc_rows_produced();
                        return Ok(Some(row));
                    }
                    // Row filtered out, continue to next
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.graph = None; // Release graph accessor reference
        self.ctx = None; // Release execution context
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "Filter"
    }
}

/// Evaluates a logical expression against a row.
///
/// # NULL semantics
///
/// This function follows SQL NULL semantics:
/// - Missing columns return NULL (supports schema evolution and sparse data)
/// - Unresolved parameters return NULL (should be resolved before reaching filter)
pub fn evaluate_expr(expr: &LogicalExpr, row: &Row) -> OperatorResult<Value> {
    evaluate_expr_with_graph(expr, row, None)
}

/// Evaluates a logical expression with support for SQL subqueries.
///
/// This function extends `evaluate_expr_with_graph` by also supporting
/// SQL EXISTS and IN subquery execution through the provided execution context.
///
/// # Arguments
///
/// * `expr` - The logical expression to evaluate
/// * `row` - The current row providing variable bindings
/// * `graph` - Optional graph accessor for Cypher pattern subqueries
/// * `ctx` - Optional execution context for SQL subquery execution
///
/// # NULL semantics
///
/// - EXISTS returns boolean (never NULL)
/// - IN with NULL in the value being checked returns NULL if not found
/// - NOT IN with NULL values in the subquery has special semantics (UNKNOWN)
pub fn evaluate_expr_with_subquery(
    expr: &LogicalExpr,
    row: &Row,
    graph: Option<&dyn GraphAccessor>,
    ctx: &Option<ExecutionContext>,
) -> OperatorResult<Value> {
    match expr {
        // SQL EXISTS subquery
        LogicalExpr::Exists { subquery, negated } => {
            let result = evaluate_sql_exists_subquery(subquery, row, ctx)?;
            let exists = match result {
                Value::Bool(b) => b,
                _ => false,
            };
            Ok(Value::Bool(if *negated { !exists } else { exists }))
        }

        // SQL IN subquery
        LogicalExpr::InSubquery { expr, subquery, negated } => {
            let val = evaluate_expr_with_subquery(expr, row, graph, ctx)?;

            // NULL value handling: NULL IN (...) is NULL, NULL NOT IN (...) is NULL
            if matches!(val, Value::Null) {
                return Ok(Value::Null);
            }

            evaluate_sql_in_subquery(&val, subquery, *negated, row, ctx)
        }

        // SQL scalar subquery (returns single value)
        LogicalExpr::Subquery(subquery) => evaluate_sql_scalar_subquery(subquery, row, ctx),

        // For UnaryOp, we need special handling to properly propagate NOT over EXISTS/IN
        LogicalExpr::UnaryOp { op, operand } => {
            let val = evaluate_expr_with_subquery(operand, row, graph, ctx)?;
            evaluate_unary_op(op, &val)
        }

        // Binary operations need subquery support in their operands
        LogicalExpr::BinaryOp { left, op, right } => {
            let left_val = evaluate_expr_with_subquery(left, row, graph, ctx)?;
            let right_val = evaluate_expr_with_subquery(right, row, graph, ctx)?;
            evaluate_binary_op(&left_val, op, &right_val)
        }

        // Delegate to the graph-aware evaluator for all other expressions
        _ => evaluate_expr_with_graph(expr, row, graph),
    }
}

/// Evaluates a logical expression against a row with optional graph access.
///
/// This extended version of `evaluate_expr` accepts an optional graph accessor
/// for evaluating EXISTS { } and COUNT { } subqueries that require graph traversal.
///
/// # NULL semantics
///
/// This function follows SQL NULL semantics:
/// - Missing columns return NULL (supports schema evolution and sparse data)
/// - Unresolved parameters return NULL (should be resolved before reaching filter)
pub fn evaluate_expr_with_graph(
    expr: &LogicalExpr,
    row: &Row,
    graph: Option<&dyn GraphAccessor>,
) -> OperatorResult<Value> {
    match expr {
        LogicalExpr::Literal(lit) => Ok(literal_to_value(lit)),

        LogicalExpr::Column { qualifier, name } => {
            // Try qualified name first (e.g., "u.id"), then unqualified (e.g., "id")
            // This supports both regular queries and joins with aliased tables
            let value = if let Some(qual) = qualifier {
                let qualified_name = format!("{}.{}", qual, name);
                let direct_value =
                    row.get_by_name(&qualified_name).or_else(|| row.get_by_name(name));

                // If we have a graph accessor and didn't find the qualified name directly,
                // check if the qualifier refers to an entity ID column and look up the property
                // from the graph. This enables MATCH (n:Label {prop: value}) patterns where
                // the row only contains entity IDs and properties must be fetched from storage.
                if direct_value.is_none() {
                    if let Some(graph) = graph {
                        // Check if there's a column with just the qualifier name that contains an entity ID
                        if let Some(entity_value) = row.get_by_name(qual) {
                            if let Value::Int(id) = entity_value {
                                let entity_id = EntityId::new(*id as u64);
                                // Try to get the property from the graph
                                if let Ok(Some(properties)) = graph.get_entity_properties(entity_id)
                                {
                                    if let Some(prop_value) = properties.get(name) {
                                        return Ok(prop_value.clone());
                                    }
                                }
                            }
                        }
                    }
                }

                direct_value
            } else {
                // For unqualified names, also try finding a match that ends with ".name"
                row.get_by_name(name).or_else(|| {
                    // Search for a column that ends with ".{name}"
                    let suffix = format!(".{}", name);
                    row.schema()
                        .columns()
                        .iter()
                        .find(|col| col.ends_with(&suffix))
                        .and_then(|col| row.get_by_name(col))
                })
            };
            // Missing columns return NULL - follows SQL semantics for sparse/dynamic schemas
            Ok(value.cloned().unwrap_or(Value::Null))
        }

        LogicalExpr::BinaryOp { left, op, right } => {
            let left_val = evaluate_expr_with_graph(left, row, graph)?;
            let right_val = evaluate_expr_with_graph(right, row, graph)?;
            evaluate_binary_op(&left_val, op, &right_val)
        }

        LogicalExpr::UnaryOp { op, operand } => {
            let val = evaluate_expr_with_graph(operand, row, graph)?;
            evaluate_unary_op(op, &val)
        }

        LogicalExpr::Parameter(_idx) => {
            // Parameters should be resolved before evaluation - return NULL as fallback
            Ok(Value::Null)
        }

        LogicalExpr::InList { expr, list, negated } => {
            let val = evaluate_expr(expr, row)?;
            let mut found = false;
            for item in list {
                let item_val = evaluate_expr(item, row)?;
                if values_equal(&val, &item_val) {
                    found = true;
                    break;
                }
            }
            let result = if *negated { !found } else { found };
            Ok(Value::Bool(result))
        }

        LogicalExpr::Between { expr, low, high, negated } => {
            let val = evaluate_expr(expr, row)?;
            let low_val = evaluate_expr(low, row)?;
            let high_val = evaluate_expr(high, row)?;
            let in_range =
                compare_values(&val, &low_val) >= 0 && compare_values(&val, &high_val) <= 0;
            let result = if *negated { !in_range } else { in_range };
            Ok(Value::Bool(result))
        }

        LogicalExpr::Case { operand, when_clauses, else_result } => {
            if let Some(operand_expr) = operand {
                // Simple CASE
                let operand_val = evaluate_expr(operand_expr, row)?;
                for (when, then) in when_clauses {
                    let when_val = evaluate_expr(when, row)?;
                    if values_equal(&operand_val, &when_val) {
                        return evaluate_expr(then, row);
                    }
                }
            } else {
                // Searched CASE
                for (when, then) in when_clauses {
                    let when_val = evaluate_expr(when, row)?;
                    if matches!(when_val, Value::Bool(true)) {
                        return evaluate_expr(then, row);
                    }
                }
            }
            if let Some(else_expr) = else_result {
                evaluate_expr(else_expr, row)
            } else {
                Ok(Value::Null)
            }
        }

        LogicalExpr::Alias { expr, .. } => evaluate_expr(expr, row),

        LogicalExpr::Cast { expr, data_type: _ } => {
            // Simple cast implementation
            evaluate_expr(expr, row)
        }

        // These require subquery execution which is complex
        LogicalExpr::Subquery(_) | LogicalExpr::Exists { .. } | LogicalExpr::InSubquery { .. } => {
            Ok(Value::Null)
        }

        // Aggregates should be evaluated at the aggregate operator level
        LogicalExpr::AggregateFunction { .. } => Ok(Value::Null),

        // Scalar functions
        LogicalExpr::ScalarFunction { func, args } => {
            let arg_values: Vec<Value> =
                args.iter().map(|a| evaluate_expr(a, row)).collect::<OperatorResult<Vec<_>>>()?;
            evaluate_scalar_function(func, &arg_values)
        }

        LogicalExpr::Wildcard | LogicalExpr::QualifiedWildcard(_) => Ok(Value::Null),

        LogicalExpr::HybridSearch { components, method } => {
            // Evaluate hybrid search by combining component distances
            if components.is_empty() {
                return Ok(Value::Null);
            }

            // Evaluate each component's distance expression
            let mut distances: Vec<(f64, f64)> = Vec::with_capacity(components.len());
            for comp in components {
                let dist_value = evaluate_expr(&comp.distance_expr, row)?;
                let dist = match dist_value {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    Value::Null => continue, // Skip null distances
                    _ => continue,
                };
                distances.push((dist, comp.weight));
            }

            if distances.is_empty() {
                return Ok(Value::Null);
            }

            // Combine distances based on method
            let combined_score = match method {
                HybridCombinationMethod::WeightedSum => {
                    // Weighted sum: sum(weight_i * distance_i)
                    // For distances (lower is better), we compute weighted average
                    let total_weight: f64 = distances.iter().map(|(_, w)| w).sum();
                    if total_weight == 0.0 {
                        return Ok(Value::Null);
                    }
                    let weighted_sum: f64 =
                        distances.iter().map(|(d, w)| d * w).sum::<f64>() / total_weight;
                    weighted_sum
                }
                HybridCombinationMethod::RRF { k } => {
                    // For RRF at the row level (without global ranking), we use a simplified approach:
                    // Convert distances to pseudo-ranks by normalizing and inverting
                    // RRF formula: sum(1 / (k + rank_i))
                    // Since we don't have actual ranks, we use the distance as a proxy
                    // Lower distance = better, so we use 1 / (k + distance * scale_factor)
                    let k_f64 = f64::from(*k);
                    let rrf_score: f64 = distances
                        .iter()
                        .map(|(d, w)| {
                            // Use weight to scale the contribution
                            // Lower distance = higher contribution
                            w / (k_f64 + d.abs())
                        })
                        .sum();
                    // RRF produces higher scores for better matches, but we need
                    // lower = better for sorting, so we negate
                    -rrf_score
                }
            };

            Ok(Value::Float(combined_score))
        }

        // Window functions should be evaluated by the window operator, not at filter level
        LogicalExpr::WindowFunction { .. } => Ok(Value::Null),

        // List comprehension: [x IN list WHERE predicate | expression]
        LogicalExpr::ListComprehension {
            variable,
            list_expr,
            filter_predicate,
            transform_expr,
        } => {
            // First evaluate the list expression to get the source list
            let list_value = evaluate_expr(list_expr, row)?;

            // Extract elements from the list value
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                // If it's a single value, treat it as a one-element list
                other => vec![other],
            };

            // Process each element through the comprehension
            let mut result = Vec::new();
            for element in elements {
                // Create a temporary row with the variable bound to the current element
                // We'll add the variable as a column to the row for evaluation
                let temp_row = row.with_binding(variable.as_str(), element.clone());

                // Apply filter predicate if present
                let passes_filter = if let Some(predicate) = filter_predicate {
                    let filter_result = evaluate_expr(predicate, &temp_row)?;
                    match filter_result {
                        Value::Bool(b) => b,
                        Value::Null => false,
                        _ => false,
                    }
                } else {
                    true
                };

                if passes_filter {
                    // Apply transform expression if present, otherwise use the element itself
                    let output = if let Some(transform) = transform_expr {
                        evaluate_expr(transform, &temp_row)?
                    } else {
                        element
                    };
                    result.push(output);
                }
            }

            Ok(Value::Array(result))
        }

        // List literal: [expr1, expr2, ...]
        LogicalExpr::ListLiteral(exprs) => {
            let elements: Vec<Value> =
                exprs.iter().map(|e| evaluate_expr(e, row)).collect::<OperatorResult<Vec<_>>>()?;
            Ok(Value::Array(elements))
        }

        // List predicate: all(variable IN list WHERE predicate)
        // Returns true if ALL elements in the list satisfy the predicate
        LogicalExpr::ListPredicateAll { variable, list_expr, predicate } => {
            let list_value = evaluate_expr(list_expr, row)?;
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                other => vec![other],
            };

            // Empty list: all() returns true (vacuous truth)
            if elements.is_empty() {
                return Ok(Value::Bool(true));
            }

            for element in elements {
                let temp_row = row.with_binding(variable.as_str(), element);
                let result = evaluate_expr(predicate, &temp_row)?;
                match result {
                    Value::Bool(false) => return Ok(Value::Bool(false)),
                    Value::Bool(true) => continue,
                    Value::Null => return Ok(Value::Null), // NULL propagation
                    _ => return Ok(Value::Bool(false)),    // Non-boolean treated as false
                }
            }
            Ok(Value::Bool(true))
        }

        // List predicate: any(variable IN list WHERE predicate)
        // Returns true if ANY element in the list satisfies the predicate
        LogicalExpr::ListPredicateAny { variable, list_expr, predicate } => {
            let list_value = evaluate_expr(list_expr, row)?;
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                other => vec![other],
            };

            // Empty list: any() returns false
            if elements.is_empty() {
                return Ok(Value::Bool(false));
            }

            let mut has_null = false;
            for element in elements {
                let temp_row = row.with_binding(variable.as_str(), element);
                let result = evaluate_expr(predicate, &temp_row)?;
                match result {
                    Value::Bool(true) => return Ok(Value::Bool(true)),
                    Value::Bool(false) => continue,
                    Value::Null => has_null = true,
                    _ => continue, // Non-boolean treated as false
                }
            }
            // If we had any NULL and no true, return NULL
            if has_null {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(false))
            }
        }

        // List predicate: none(variable IN list WHERE predicate)
        // Returns true if NO elements in the list satisfy the predicate
        LogicalExpr::ListPredicateNone { variable, list_expr, predicate } => {
            let list_value = evaluate_expr(list_expr, row)?;
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                other => vec![other],
            };

            // Empty list: none() returns true
            if elements.is_empty() {
                return Ok(Value::Bool(true));
            }

            let mut has_null = false;
            for element in elements {
                let temp_row = row.with_binding(variable.as_str(), element);
                let result = evaluate_expr(predicate, &temp_row)?;
                match result {
                    Value::Bool(true) => return Ok(Value::Bool(false)),
                    Value::Bool(false) => continue,
                    Value::Null => has_null = true,
                    _ => continue, // Non-boolean treated as false
                }
            }
            // If we had any NULL and no true, return NULL
            if has_null {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(true))
            }
        }

        // List predicate: single(variable IN list WHERE predicate)
        // Returns true if EXACTLY ONE element in the list satisfies the predicate
        LogicalExpr::ListPredicateSingle { variable, list_expr, predicate } => {
            let list_value = evaluate_expr(list_expr, row)?;
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                other => vec![other],
            };

            // Empty list: single() returns false (need exactly one)
            if elements.is_empty() {
                return Ok(Value::Bool(false));
            }

            let mut count = 0;
            let mut has_null = false;
            for element in elements {
                let temp_row = row.with_binding(variable.as_str(), element);
                let result = evaluate_expr(predicate, &temp_row)?;
                match result {
                    Value::Bool(true) => {
                        count += 1;
                        if count > 1 {
                            return Ok(Value::Bool(false)); // More than one match
                        }
                    }
                    Value::Bool(false) => continue,
                    Value::Null => has_null = true,
                    _ => continue, // Non-boolean treated as false
                }
            }
            // If count is exactly 1 and no NULLs, return true
            // If count is 0 and we had NULLs, return NULL (unknown)
            // If count is 1 and we had NULLs, return NULL (could be more matches)
            if count == 1 && !has_null {
                Ok(Value::Bool(true))
            } else if count == 0 && !has_null {
                Ok(Value::Bool(false))
            } else {
                // has_null is true, result is uncertain (could be 0 or more matches among NULLs)
                Ok(Value::Null)
            }
        }

        // Reduce: reduce(accumulator = initial, variable IN list | expression)
        // Performs a fold operation over a list
        LogicalExpr::ListReduce { accumulator, initial, variable, list_expr, expression } => {
            let list_value = evaluate_expr(list_expr, row)?;
            let elements = match list_value {
                Value::Array(arr) => arr,
                Value::Null => return Ok(Value::Null),
                other => vec![other],
            };

            // Start with initial value
            let mut acc_value = evaluate_expr(initial, row)?;

            for element in elements {
                // Create a temp row with both accumulator and variable bindings
                let temp_row = row
                    .with_binding(accumulator.as_str(), acc_value.clone())
                    .with_binding(variable.as_str(), element);
                acc_value = evaluate_expr(expression, &temp_row)?;
            }

            Ok(acc_value)
        }

        // Map projection: node{.property1, .property2, key: expression, .*}
        LogicalExpr::MapProjection { source, items } => {
            // Evaluate the source expression to get the source entity's properties
            let source_value = evaluate_expr(source, row)?;

            // The source should be an entity (map-like structure) or we extract properties
            // from the row using the source as a prefix/qualifier.
            // We'll build the result map based on the projection items.
            let mut result_properties: Vec<(String, Value)> = Vec::new();

            // If source is a column reference, we get the prefix for property lookup
            let source_prefix = if let LogicalExpr::Column { qualifier, name } = source.as_ref() {
                if let Some(q) = qualifier {
                    format!("{q}.{name}")
                } else {
                    name.clone()
                }
            } else {
                String::new()
            };

            for item in items {
                match item {
                    LogicalMapProjectionItem::Property(prop_name) => {
                        // Extract property from source: try qualified name first, then direct
                        let prop_key = if source_prefix.is_empty() {
                            prop_name.clone()
                        } else {
                            format!("{source_prefix}.{prop_name}")
                        };
                        let value = row
                            .get_by_name(&prop_key)
                            .or_else(|| row.get_by_name(prop_name))
                            .cloned()
                            .unwrap_or(Value::Null);
                        result_properties.push((prop_name.clone(), value));
                    }
                    LogicalMapProjectionItem::Computed { key, value } => {
                        // Evaluate the expression for the computed value
                        let computed_value = evaluate_expr(value, row)?;
                        result_properties.push((key.clone(), computed_value));
                    }
                    LogicalMapProjectionItem::AllProperties => {
                        // Include all properties from the source
                        // If we have a source prefix, include all columns that start with it
                        // Otherwise, just include the source value if it's map-like
                        if !source_prefix.is_empty() {
                            let prefix_with_dot = format!("{source_prefix}.");
                            for col_name in row.schema().columns() {
                                if col_name.starts_with(&prefix_with_dot) {
                                    let prop_name =
                                        col_name.strip_prefix(&prefix_with_dot).unwrap_or(col_name);
                                    if let Some(val) = row.get_by_name(col_name) {
                                        result_properties
                                            .push((prop_name.to_string(), val.clone()));
                                    }
                                }
                            }
                        } else if let Value::Array(arr) = &source_value {
                            // If source is an array of key-value pairs, extract them
                            for (i, val) in arr.iter().enumerate() {
                                result_properties.push((i.to_string(), val.clone()));
                            }
                        }
                    }
                }
            }

            // Return as an Array of pairs (since we don't have a Map value type)
            // Each pair is [key, value] represented as a nested array
            let pairs: Vec<Value> = result_properties
                .into_iter()
                .map(|(k, v)| Value::Array(vec![Value::String(k), v]))
                .collect();
            Ok(Value::Array(pairs))
        }

        // Pattern comprehension: [(pattern) WHERE predicate | expression]
        // This expression type requires graph access to execute the pattern matching.
        // At the expression evaluation level, we cannot execute graph traversals directly.
        // Pattern comprehensions should be planned as subquery-like operations that
        // execute the pattern match and collect results.
        //
        // For now, we return NULL and pattern comprehensions should be handled
        // at the operator level where graph access is available.
        LogicalExpr::PatternComprehension { .. } => {
            // TODO: Pattern comprehensions require graph access for execution.
            // They should be transformed into a plan that:
            // 1. Executes the graph pattern match for each input row
            // 2. Filters matches based on the WHERE predicate
            // 3. Projects the result expression for each match
            // 4. Collects all projections into a list
            //
            // This is similar to a correlated subquery over graph data.
            // For now, return an empty list as a placeholder.
            Ok(Value::Array(vec![]))
        }

        // EXISTS { } subquery: Returns boolean based on pattern existence
        LogicalExpr::ExistsSubquery { expand_steps, filter_predicate } => {
            // EXISTS requires graph access for execution
            let Some(graph) = graph else {
                // No graph accessor available - return false as fallback
                return Ok(Value::Bool(false));
            };

            // Execute the EXISTS subquery and return true if any match exists
            evaluate_exists_subquery(graph, expand_steps, filter_predicate.as_deref(), row)
        }

        // COUNT { } subquery: Returns count of pattern matches
        LogicalExpr::CountSubquery { expand_steps, filter_predicate } => {
            // COUNT requires graph access for execution
            let Some(graph) = graph else {
                // No graph accessor available - return 0 as fallback
                return Ok(Value::Int(0));
            };

            // Execute the COUNT subquery and return the count of matches
            evaluate_count_subquery(graph, expand_steps, filter_predicate.as_deref(), row)
        }

        // CALL { } subquery: Executes inner plan and returns results
        // This requires full subquery execution with variable binding.
        // At the expression level, return null as placeholder.
        LogicalExpr::CallSubquery { .. } => {
            // TODO: CALL subqueries require building and executing a sub-plan.
            // They should:
            // 1. Import specified variables from the outer context
            // 2. Execute the inner plan
            // 3. Return the results (typically from the inner RETURN clause)
            //
            // For now, return NULL as a placeholder.
            Ok(Value::Null)
        }
    }
}

/// Evaluates an EXISTS { } subquery by executing graph pattern matching.
///
/// This function implements correlated EXISTS subqueries for Cypher.
/// It executes the graph pattern match starting from the source node
/// referenced in the current row, and returns true if any match exists.
///
/// # Short-circuit optimization
///
/// This implementation short-circuits on the first match found - it does not
/// enumerate all possible matches, making it efficient for existence checks.
///
/// # Arguments
///
/// * `graph` - The graph accessor for traversal operations
/// * `expand_steps` - The expand nodes representing the pattern to match
/// * `filter_predicate` - Optional WHERE filter to apply to matches
/// * `row` - The current row (provides correlated variable bindings)
fn evaluate_exists_subquery(
    graph: &dyn GraphAccessor,
    expand_steps: &[ExpandNode],
    filter_predicate: Option<&LogicalExpr>,
    row: &Row,
) -> OperatorResult<Value> {
    // Empty pattern matches nothing
    if expand_steps.is_empty() {
        return Ok(Value::Bool(false));
    }

    // Get the source node ID from the first expand step
    let first_step = &expand_steps[0];
    let source_id = match row.get_by_name(&first_step.src_var) {
        Some(Value::Int(id)) => EntityId::new(*id as u64),
        Some(_) | None => {
            // Source variable not found or not an integer - no match
            return Ok(Value::Bool(false));
        }
    };

    // Execute the pattern matching with short-circuit semantics
    // For EXISTS, we only need to find one match
    execute_exists_pattern(graph, expand_steps, filter_predicate, row, source_id, 0)
}

/// Recursively executes pattern matching for EXISTS subquery.
///
/// This function handles multi-hop patterns by recursively expanding each step.
/// It short-circuits on the first complete match to optimize performance.
fn execute_exists_pattern(
    graph: &dyn GraphAccessor,
    expand_steps: &[ExpandNode],
    filter_predicate: Option<&LogicalExpr>,
    base_row: &Row,
    current_node: EntityId,
    step_index: usize,
) -> OperatorResult<Value> {
    // Base case: all steps completed - check the filter if present
    if step_index >= expand_steps.len() {
        if let Some(predicate) = filter_predicate {
            // Evaluate the filter predicate with the current bindings
            let result = evaluate_expr(predicate, base_row)?;
            return Ok(match result {
                Value::Bool(true) => Value::Bool(true),
                _ => Value::Bool(false),
            });
        }
        // No filter - pattern matched
        return Ok(Value::Bool(true));
    }

    let step = &expand_steps[step_index];
    let direction = expand_direction_to_graph_direction(&step.direction);

    // Get edge type filters
    let edge_types: Option<Vec<EdgeType>> = if step.edge_types.is_empty() {
        None
    } else {
        Some(step.edge_types.iter().map(|s| EdgeType::new(s.as_str())).collect())
    };

    // Execute expansion based on length specification
    let neighbors = match &step.length {
        ExpandLength::Single => {
            // Single hop expansion
            let results = if let Some(ref types) = edge_types {
                graph.neighbors_by_types(current_node, direction, types)
            } else {
                graph.neighbors(current_node, direction)
            };

            match results {
                Ok(neighbors) => neighbors,
                Err(_) => return Ok(Value::Bool(false)),
            }
        }
        ExpandLength::Exact(n) => {
            // Exact depth: use expand_all with min=max=n
            let results =
                graph.expand_all(current_node, direction, *n, Some(*n), edge_types.as_deref());
            match results {
                Ok(traversals) => traversals
                    .into_iter()
                    .map(|t| {
                        crate::exec::graph_accessor::NeighborResult::new(
                            t.node,
                            t.edge_id.unwrap_or(manifoldb_core::EdgeId::new(0)),
                            direction,
                        )
                    })
                    .collect(),
                Err(_) => return Ok(Value::Bool(false)),
            }
        }
        ExpandLength::Range { min, max } => {
            // Variable length expansion
            let results =
                graph.expand_all(current_node, direction, *min, *max, edge_types.as_deref());
            match results {
                Ok(traversals) => traversals
                    .into_iter()
                    .map(|t| {
                        crate::exec::graph_accessor::NeighborResult::new(
                            t.node,
                            t.edge_id.unwrap_or(manifoldb_core::EdgeId::new(0)),
                            direction,
                        )
                    })
                    .collect(),
                Err(_) => return Ok(Value::Bool(false)),
            }
        }
    };

    // For each neighbor, continue the pattern matching
    for neighbor in neighbors {
        // Apply node label filter if present
        if !step.node_labels.is_empty() {
            // TODO: Check if the neighbor node has the required labels
            // For now, we skip label filtering as it requires entity access
        }

        // Create a new row with the destination variable bound
        let row_with_binding =
            base_row.with_binding(&step.dst_var, Value::Int(neighbor.node.as_u64() as i64));

        // Optionally bind edge variable
        let row_with_edge = if let Some(ref edge_var) = step.edge_var {
            row_with_binding.with_binding(edge_var, Value::Int(neighbor.edge_id.as_u64() as i64))
        } else {
            row_with_binding
        };

        // Recursively check the next step
        let result = execute_exists_pattern(
            graph,
            expand_steps,
            filter_predicate,
            &row_with_edge,
            neighbor.node,
            step_index + 1,
        )?;

        // Short-circuit: if we found a match, return immediately
        if matches!(result, Value::Bool(true)) {
            return Ok(Value::Bool(true));
        }
    }

    // No match found
    Ok(Value::Bool(false))
}

/// Evaluates a COUNT { } subquery by executing graph pattern matching.
///
/// This function implements correlated COUNT subqueries for Cypher.
/// It executes the graph pattern match starting from the source node
/// referenced in the current row, and returns the count of all matches.
///
/// # Full enumeration
///
/// Unlike EXISTS which short-circuits on the first match, COUNT must
/// enumerate all possible matches to return an accurate count.
///
/// # Arguments
///
/// * `graph` - The graph accessor for traversal operations
/// * `expand_steps` - The expand nodes representing the pattern to match
/// * `filter_predicate` - Optional WHERE filter to apply to matches
/// * `row` - The current row (provides correlated variable bindings)
fn evaluate_count_subquery(
    graph: &dyn GraphAccessor,
    expand_steps: &[ExpandNode],
    filter_predicate: Option<&LogicalExpr>,
    row: &Row,
) -> OperatorResult<Value> {
    // Empty pattern has zero matches
    if expand_steps.is_empty() {
        return Ok(Value::Int(0));
    }

    // Get the source node ID from the first expand step
    let first_step = &expand_steps[0];
    let source_id = match row.get_by_name(&first_step.src_var) {
        Some(Value::Int(id)) => EntityId::new(*id as u64),
        Some(_) | None => {
            // Source variable not found or not an integer - no matches
            return Ok(Value::Int(0));
        }
    };

    // Execute the pattern matching and count all matches
    let count = execute_count_pattern(graph, expand_steps, filter_predicate, row, source_id, 0)?;
    Ok(Value::Int(count))
}

/// Recursively executes pattern matching for COUNT subquery.
///
/// This function handles multi-hop patterns by recursively expanding each step.
/// Unlike EXISTS, it does NOT short-circuit - it counts ALL matching paths.
fn execute_count_pattern(
    graph: &dyn GraphAccessor,
    expand_steps: &[ExpandNode],
    filter_predicate: Option<&LogicalExpr>,
    base_row: &Row,
    current_node: EntityId,
    step_index: usize,
) -> OperatorResult<i64> {
    // Base case: all steps completed - check the filter if present
    if step_index >= expand_steps.len() {
        if let Some(predicate) = filter_predicate {
            // Evaluate the filter predicate with the current bindings
            let result = evaluate_expr(predicate, base_row)?;
            return Ok(match result {
                Value::Bool(true) => 1,
                _ => 0,
            });
        }
        // No filter - pattern matched, count as 1
        return Ok(1);
    }

    let step = &expand_steps[step_index];
    let direction = expand_direction_to_graph_direction(&step.direction);

    // Get edge type filters
    let edge_types: Option<Vec<EdgeType>> = if step.edge_types.is_empty() {
        None
    } else {
        Some(step.edge_types.iter().map(|s| EdgeType::new(s.as_str())).collect())
    };

    // Execute expansion based on length specification
    let neighbors = match &step.length {
        ExpandLength::Single => {
            // Single hop expansion
            let results = if let Some(ref types) = edge_types {
                graph.neighbors_by_types(current_node, direction, types)
            } else {
                graph.neighbors(current_node, direction)
            };

            match results {
                Ok(neighbors) => neighbors,
                Err(_) => return Ok(0),
            }
        }
        ExpandLength::Exact(n) => {
            // Exact depth: use expand_all with min=max=n
            let results =
                graph.expand_all(current_node, direction, *n, Some(*n), edge_types.as_deref());
            match results {
                Ok(traversals) => traversals
                    .into_iter()
                    .map(|t| {
                        crate::exec::graph_accessor::NeighborResult::new(
                            t.node,
                            t.edge_id.unwrap_or(manifoldb_core::EdgeId::new(0)),
                            direction,
                        )
                    })
                    .collect(),
                Err(_) => return Ok(0),
            }
        }
        ExpandLength::Range { min, max } => {
            // Variable length expansion
            let results =
                graph.expand_all(current_node, direction, *min, *max, edge_types.as_deref());
            match results {
                Ok(traversals) => traversals
                    .into_iter()
                    .map(|t| {
                        crate::exec::graph_accessor::NeighborResult::new(
                            t.node,
                            t.edge_id.unwrap_or(manifoldb_core::EdgeId::new(0)),
                            direction,
                        )
                    })
                    .collect(),
                Err(_) => return Ok(0),
            }
        }
    };

    // Accumulate count from all neighbors
    let mut total_count = 0i64;
    for neighbor in neighbors {
        // Apply node label filter if present
        if !step.node_labels.is_empty() {
            // TODO: Check if the neighbor node has the required labels
            // For now, we skip label filtering as it requires entity access
        }

        // Create a new row with the destination variable bound
        let row_with_binding =
            base_row.with_binding(&step.dst_var, Value::Int(neighbor.node.as_u64() as i64));

        // Optionally bind edge variable
        let row_with_edge = if let Some(ref edge_var) = step.edge_var {
            row_with_binding.with_binding(edge_var, Value::Int(neighbor.edge_id.as_u64() as i64))
        } else {
            row_with_binding
        };

        // Recursively count matches from this neighbor
        let count = execute_count_pattern(
            graph,
            expand_steps,
            filter_predicate,
            &row_with_edge,
            neighbor.node,
            step_index + 1,
        )?;

        total_count += count;
    }

    Ok(total_count)
}

// ===========================================================================
// SQL Subquery Execution Functions
// ===========================================================================

/// Executes an SQL EXISTS subquery.
///
/// EXISTS returns TRUE if the subquery returns at least one row, FALSE otherwise.
/// For correlated subqueries, the outer row values are available through variable binding.
///
/// # Arguments
///
/// * `subquery` - The logical plan representing the subquery
/// * `outer_row` - The current outer row (for correlated subqueries)
/// * `ctx` - Optional execution context (required for subquery execution)
///
/// # Returns
///
/// `Value::Bool(true)` if subquery returns any rows, `Value::Bool(false)` otherwise.
fn evaluate_sql_exists_subquery(
    subquery: &LogicalPlan,
    _outer_row: &Row,
    ctx: &Option<ExecutionContext>,
) -> OperatorResult<Value> {
    // Without an execution context, we cannot execute subqueries
    let Some(ctx) = ctx else {
        return Ok(Value::Bool(false));
    };

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let mut op = build_operator_tree(&physical_plan)?;
    op.open(ctx)?;

    // For EXISTS, we only need to check if there's at least one row
    let has_row = op.next()?.is_some();
    op.close()?;

    Ok(Value::Bool(has_row))
}

/// Executes an SQL IN subquery.
///
/// IN returns TRUE if the value matches any value in the subquery result set.
/// NOT IN returns TRUE if the value does NOT match any value in the result set.
///
/// # NULL Semantics (SQL Standard)
///
/// - `val IN (...)`: TRUE if val equals any subquery value, FALSE if val doesn't equal
///   any and no NULLs exist in results, NULL (UNKNOWN) if val doesn't equal any but
///   NULLs exist in results.
/// - `val NOT IN (...)`: FALSE if val equals any subquery value, TRUE if val doesn't
///   equal any and no NULLs exist in results, NULL (UNKNOWN) if val doesn't equal any
///   but NULLs exist in results.
///
/// # Arguments
///
/// * `val` - The value to search for
/// * `subquery` - The logical plan representing the subquery
/// * `negated` - Whether this is NOT IN (true) or IN (false)
/// * `outer_row` - The current outer row (for correlated subqueries)
/// * `ctx` - Optional execution context (required for subquery execution)
fn evaluate_sql_in_subquery(
    val: &Value,
    subquery: &LogicalPlan,
    negated: bool,
    _outer_row: &Row,
    ctx: &Option<ExecutionContext>,
) -> OperatorResult<Value> {
    // Without an execution context, we cannot execute subqueries
    let Some(ctx) = ctx else {
        return Ok(Value::Null);
    };

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let mut op = build_operator_tree(&physical_plan)?;
    op.open(ctx)?;

    // Collect subquery results (first column only for IN)
    let mut found = false;
    let mut has_null = false;

    while let Some(row) = op.next()? {
        // Get the first column value
        let subquery_val = row.get(0).cloned().unwrap_or(Value::Null);

        // Track if we see any NULLs
        if matches!(subquery_val, Value::Null) {
            has_null = true;
            continue;
        }

        // Check for equality
        if values_equal(val, &subquery_val) {
            found = true;
            break; // Found a match, can stop early
        }
    }

    op.close()?;

    // Apply SQL NULL semantics for IN/NOT IN
    if negated {
        // NOT IN semantics
        if found {
            // Value was found - NOT IN returns FALSE
            Ok(Value::Bool(false))
        } else if has_null {
            // Value not found but NULLs exist - return NULL (UNKNOWN)
            Ok(Value::Null)
        } else {
            // Value not found and no NULLs - NOT IN returns TRUE
            Ok(Value::Bool(true))
        }
    } else {
        // IN semantics
        if found {
            // Value was found - IN returns TRUE
            Ok(Value::Bool(true))
        } else if has_null {
            // Value not found but NULLs exist - return NULL (UNKNOWN)
            Ok(Value::Null)
        } else {
            // Value not found and no NULLs - IN returns FALSE
            Ok(Value::Bool(false))
        }
    }
}

/// Executes an SQL scalar subquery.
///
/// A scalar subquery must return exactly one row with one column.
/// If it returns no rows, the result is NULL.
/// If it returns more than one row, this is an error (but we return NULL for now).
///
/// # Arguments
///
/// * `subquery` - The logical plan representing the subquery
/// * `outer_row` - The current outer row (for correlated subqueries)
/// * `ctx` - Optional execution context (required for subquery execution)
fn evaluate_sql_scalar_subquery(
    subquery: &LogicalPlan,
    _outer_row: &Row,
    ctx: &Option<ExecutionContext>,
) -> OperatorResult<Value> {
    // Without an execution context, we cannot execute subqueries
    let Some(ctx) = ctx else {
        return Ok(Value::Null);
    };

    // Convert logical plan to physical plan
    let planner = PhysicalPlanner::new();
    let physical_plan = planner.plan(subquery);

    // Build and execute the operator tree
    let mut op = build_operator_tree(&physical_plan)?;
    op.open(ctx)?;

    // Get the first row
    let result = if let Some(row) = op.next()? {
        // Get the first column value
        row.get(0).cloned().unwrap_or(Value::Null)
    } else {
        // No rows - return NULL
        Value::Null
    };

    op.close()?;
    Ok(result)
}

/// Converts `ExpandDirection` to `manifoldb_graph::traversal::Direction`.
fn expand_direction_to_graph_direction(dir: &ExpandDirection) -> Direction {
    match dir {
        ExpandDirection::Outgoing => Direction::Outgoing,
        ExpandDirection::Incoming => Direction::Incoming,
        ExpandDirection::Both => Direction::Both,
    }
}

/// Converts a literal to a value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Boolean(b) => Value::Bool(*b),
        Literal::Integer(i) => Value::Int(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Vector(v) => Value::Vector(v.clone()),
        Literal::MultiVector(v) => Value::MultiVector(v.clone()),
    }
}

/// Evaluates a binary operation.
fn evaluate_binary_op(left: &Value, op: &BinaryOp, right: &Value) -> OperatorResult<Value> {
    // Handle NULL propagation
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return match op {
            BinaryOp::And => {
                // NULL AND FALSE = FALSE
                if matches!(left, Value::Bool(false)) || matches!(right, Value::Bool(false)) {
                    Ok(Value::Bool(false))
                } else {
                    Ok(Value::Null)
                }
            }
            BinaryOp::Or => {
                // NULL OR TRUE = TRUE
                if matches!(left, Value::Bool(true)) || matches!(right, Value::Bool(true)) {
                    Ok(Value::Bool(true))
                } else {
                    Ok(Value::Null)
                }
            }
            _ => Ok(Value::Null),
        };
    }

    match op {
        // Comparison operators - return NULL if either operand is NULL (SQL semantics)
        BinaryOp::Eq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(values_equal(left, right)))
            }
        }
        BinaryOp::NotEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(!values_equal(left, right)))
            }
        }
        BinaryOp::Lt => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) < 0))
            }
        }
        BinaryOp::LtEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) <= 0))
            }
        }
        BinaryOp::Gt => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) > 0))
            }
        }
        BinaryOp::GtEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) >= 0))
            }
        }

        // Logical operators
        BinaryOp::And => {
            let l = value_to_bool(left);
            let r = value_to_bool(right);
            Ok(Value::Bool(l && r))
        }
        BinaryOp::Or => {
            let l = value_to_bool(left);
            let r = value_to_bool(right);
            Ok(Value::Bool(l || r))
        }

        // Arithmetic operators
        BinaryOp::Add => evaluate_arithmetic(left, right, |a, b| a + b, |a, b| a + b),
        BinaryOp::Sub => evaluate_arithmetic(left, right, |a, b| a - b, |a, b| a - b),
        BinaryOp::Mul => evaluate_arithmetic(left, right, |a, b| a * b, |a, b| a * b),
        BinaryOp::Div => {
            // Check for division by zero
            let is_zero = match right {
                Value::Int(0) => true,
                Value::Float(f) => *f == 0.0,
                _ => false,
            };
            if is_zero {
                Ok(Value::Null)
            } else {
                evaluate_arithmetic(left, right, |a, b| a / b, |a, b| a / b)
            }
        }
        BinaryOp::Mod => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Ok(Value::Int(a % b)),
            _ => Ok(Value::Null),
        },

        // String operators
        BinaryOp::Like => {
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                Ok(Value::Bool(like_match(s, pattern)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::ILike => {
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                Ok(Value::Bool(like_match(&s.to_lowercase(), &pattern.to_lowercase())))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::NotLike | BinaryOp::NotILike => {
            // Handle negated variants
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                let s = if matches!(op, BinaryOp::NotILike) { s.to_lowercase() } else { s.clone() };
                let pattern = if matches!(op, BinaryOp::NotILike) {
                    pattern.to_lowercase()
                } else {
                    pattern.clone()
                };
                Ok(Value::Bool(!like_match(&s, &pattern)))
            } else {
                Ok(Value::Null)
            }
        }

        // Vector operators
        BinaryOp::EuclideanDistance => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let dist = euclidean_distance(a, b);
                Ok(Value::Float(f64::from(dist)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::CosineDistance => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let dist = cosine_distance(a, b);
                Ok(Value::Float(f64::from(dist)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::InnerProduct => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let prod = inner_product(a, b);
                Ok(Value::Float(f64::from(prod)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::MaxSim => {
            // MaxSim operates on multi-vectors (Vec<Vec<f32>>)
            if let (Value::MultiVector(query), Value::MultiVector(doc)) = (left, right) {
                let score = maxsim_score(query, doc);
                Ok(Value::Float(f64::from(score)))
            } else {
                Ok(Value::Null)
            }
        }
    }
}

/// Evaluates a unary operation.
fn evaluate_unary_op(op: &UnaryOp, operand: &Value) -> OperatorResult<Value> {
    match op {
        UnaryOp::Not => match operand {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            Value::Null => Ok(Value::Null),
            _ => Ok(Value::Bool(false)),
        },
        UnaryOp::Neg => match operand {
            Value::Int(i) => Ok(Value::Int(-i)),
            Value::Float(f) => Ok(Value::Float(-f)),
            _ => Ok(Value::Null),
        },
        UnaryOp::IsNull => Ok(Value::Bool(matches!(operand, Value::Null))),
        UnaryOp::IsNotNull => Ok(Value::Bool(!matches!(operand, Value::Null))),
    }
}

/// Evaluates a scalar function.
fn evaluate_scalar_function(
    func: &crate::plan::logical::ScalarFunction,
    args: &[Value],
) -> OperatorResult<Value> {
    use crate::plan::logical::ScalarFunction;

    // Helper to convert value to f64
    fn value_to_f64(v: &Value) -> Option<f64> {
        match v {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    match func {
        // ========== String Functions ==========
        ScalarFunction::Upper => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Lower => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Length => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::Int(s.chars().count() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Concat => {
            let result: String = args.iter().map(value_to_string).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Substring => {
            // SUBSTRING(string, start, [length])
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let start = match args.get(1) {
                Some(Value::Int(i)) => (*i as usize).saturating_sub(1), // SQL is 1-indexed
                _ => return Ok(Value::Null),
            };
            let chars: Vec<char> = s.chars().collect();
            if start >= chars.len() {
                return Ok(Value::String(String::new()));
            }
            let len = match args.get(2) {
                Some(Value::Int(l)) => *l as usize,
                _ => chars.len() - start,
            };
            let result: String = chars.iter().skip(start).take(len).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Trim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Ltrim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_start().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Rtrim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_end().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Lpad => {
            // LPAD(string, length, fill) - Left-pad string to length with fill characters
            // If string is longer than length, truncates from the right
            // Default fill is ' ' (single space) if not provided
            let s = match args.first() {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => return Ok(Value::Null),
                Some(other) => value_to_string(other),
            };
            let length = match args.get(1) {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(n)) => *n as i64,
                _ => return Ok(Value::Null),
            };
            // Handle negative length - return empty string (PostgreSQL behavior)
            if length <= 0 {
                return Ok(Value::String(String::new()));
            }
            let length = length as usize;
            let fill = match args.get(2) {
                Some(Value::String(f)) => f.clone(),
                Some(Value::Null) => return Ok(Value::Null),
                None => " ".to_string(), // Default fill is space
                Some(other) => value_to_string(other),
            };
            // Handle empty fill string - return original string up to length
            if fill.is_empty() {
                return Ok(Value::String(s.chars().take(length).collect()));
            }
            let s_chars: Vec<char> = s.chars().collect();
            // If string is already longer or equal to target length, truncate from right
            if s_chars.len() >= length {
                return Ok(Value::String(s_chars[..length].iter().collect()));
            }
            // Calculate padding needed
            let pad_len = length - s_chars.len();
            let fill_chars: Vec<char> = fill.chars().collect();
            // Build padding by cycling through fill characters
            let mut padding = String::with_capacity(pad_len);
            let mut fill_idx = 0;
            for _ in 0..pad_len {
                padding.push(fill_chars[fill_idx]);
                fill_idx = (fill_idx + 1) % fill_chars.len();
            }
            Ok(Value::String(format!("{}{}", padding, s)))
        }
        ScalarFunction::Rpad => {
            // RPAD(string, length, fill) - Right-pad string to length with fill characters
            // If string is longer than length, truncates from the right
            // Default fill is ' ' (single space) if not provided
            let s = match args.first() {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => return Ok(Value::Null),
                Some(other) => value_to_string(other),
            };
            let length = match args.get(1) {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(n)) => *n as i64,
                _ => return Ok(Value::Null),
            };
            // Handle negative length - return empty string (PostgreSQL behavior)
            if length <= 0 {
                return Ok(Value::String(String::new()));
            }
            let length = length as usize;
            let fill = match args.get(2) {
                Some(Value::String(f)) => f.clone(),
                Some(Value::Null) => return Ok(Value::Null),
                None => " ".to_string(), // Default fill is space
                Some(other) => value_to_string(other),
            };
            // Handle empty fill string - return original string up to length
            if fill.is_empty() {
                return Ok(Value::String(s.chars().take(length).collect()));
            }
            let s_chars: Vec<char> = s.chars().collect();
            // If string is already longer or equal to target length, truncate from right
            if s_chars.len() >= length {
                return Ok(Value::String(s_chars[..length].iter().collect()));
            }
            // Calculate padding needed
            let pad_len = length - s_chars.len();
            let fill_chars: Vec<char> = fill.chars().collect();
            // Build padding by cycling through fill characters
            let mut padding = String::with_capacity(pad_len);
            let mut fill_idx = 0;
            for _ in 0..pad_len {
                padding.push(fill_chars[fill_idx]);
                fill_idx = (fill_idx + 1) % fill_chars.len();
            }
            Ok(Value::String(format!("{}{}", s, padding)))
        }
        ScalarFunction::Replace => {
            // REPLACE(string, from, to)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let from = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            let to = match args.get(2) {
                Some(Value::String(t)) => t,
                _ => return Ok(Value::Null),
            };
            Ok(Value::String(s.replace(from.as_str(), to.as_str())))
        }
        ScalarFunction::Position => {
            // POSITION(substring IN string) - typically called as strpos(string, substring)
            // Args: [substring, string] or [string, substring] depending on call style
            // We'll support both: strpos(string, substring) and position with two args
            let (haystack, needle) = if args.len() >= 2 {
                match (&args[0], &args[1]) {
                    (Value::String(h), Value::String(n)) => (h, n),
                    _ => return Ok(Value::Null),
                }
            } else {
                return Ok(Value::Null);
            };
            match haystack.find(needle.as_str()) {
                Some(pos) => Ok(Value::Int((pos + 1) as i64)), // 1-indexed
                None => Ok(Value::Int(0)),
            }
        }
        ScalarFunction::ConcatWs => {
            // CONCAT_WS(separator, string, ...)
            let sep = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let parts: Vec<String> = args
                .iter()
                .skip(1)
                .filter(|v| !matches!(v, Value::Null))
                .map(value_to_string)
                .collect();
            Ok(Value::String(parts.join(sep)))
        }
        ScalarFunction::SplitPart => {
            // SPLIT_PART(string, delimiter, position)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let delim = match args.get(1) {
                Some(Value::String(d)) => d,
                _ => return Ok(Value::Null),
            };
            let pos = match args.get(2) {
                Some(Value::Int(p)) => *p,
                _ => return Ok(Value::Null),
            };
            if pos <= 0 {
                return Ok(Value::String(String::new()));
            }
            let parts: Vec<&str> = s.split(delim.as_str()).collect();
            let idx = (pos - 1) as usize; // 1-indexed to 0-indexed
            if idx < parts.len() {
                Ok(Value::String(parts[idx].to_string()))
            } else {
                Ok(Value::String(String::new()))
            }
        }
        ScalarFunction::Format => {
            // FORMAT(template, args...)
            // Simple %s substitution (PostgreSQL-style format)
            let template = match args.first() {
                Some(Value::String(s)) => s.clone(),
                _ => return Ok(Value::Null),
            };
            let mut result = template;
            for arg in args.iter().skip(1) {
                if let Some(pos) = result.find("%s") {
                    let replacement = value_to_string(arg);
                    result = format!("{}{}{}", &result[..pos], replacement, &result[pos + 2..]);
                }
            }
            Ok(Value::String(result))
        }
        ScalarFunction::RegexpMatch => {
            // REGEXP_MATCH(string, pattern) - returns first match or null
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let pattern = match args.get(1) {
                Some(Value::String(p)) => p,
                _ => return Ok(Value::Null),
            };
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    if let Some(caps) = re.captures(s) {
                        // Return the full match or first capture group
                        let matched = caps.get(1).or_else(|| caps.get(0));
                        match matched {
                            Some(m) => Ok(Value::String(m.as_str().to_string())),
                            None => Ok(Value::Null),
                        }
                    } else {
                        Ok(Value::Null)
                    }
                }
                Err(_) => Ok(Value::Null),
            }
        }
        ScalarFunction::RegexpReplace => {
            // REGEXP_REPLACE(string, pattern, replacement)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let pattern = match args.get(1) {
                Some(Value::String(p)) => p,
                _ => return Ok(Value::Null),
            };
            let replacement = match args.get(2) {
                Some(Value::String(r)) => r,
                _ => return Ok(Value::Null),
            };
            match regex::Regex::new(pattern) {
                Ok(re) => Ok(Value::String(re.replace_all(s, replacement.as_str()).to_string())),
                Err(_) => Ok(Value::Null),
            }
        }
        ScalarFunction::Left => {
            // LEFT(string, length) - returns the leftmost n characters
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let len = match args.get(1) {
                Some(Value::Int(l)) => {
                    if *l < 0 {
                        return Ok(Value::String(String::new()));
                    }
                    *l as usize
                }
                _ => return Ok(Value::Null),
            };
            let chars: Vec<char> = s.chars().collect();
            let result: String = chars.iter().take(len).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Right => {
            // RIGHT(string, length) - returns the rightmost n characters
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let len = match args.get(1) {
                Some(Value::Int(l)) => {
                    if *l < 0 {
                        return Ok(Value::String(String::new()));
                    }
                    *l as usize
                }
                _ => return Ok(Value::Null),
            };
            let chars: Vec<char> = s.chars().collect();
            let skip_count = chars.len().saturating_sub(len);
            let result: String = chars.iter().skip(skip_count).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Coalesce => {
            for arg in args {
                if !matches!(arg, Value::Null) {
                    return Ok(arg.clone());
                }
            }
            Ok(Value::Null)
        }
        ScalarFunction::NullIf => {
            // NULLIF(expr1, expr2) - returns NULL if expr1 = expr2, otherwise expr1
            if args.len() < 2 {
                return Ok(Value::Null);
            }
            if values_equal(&args[0], &args[1]) {
                Ok(Value::Null)
            } else {
                Ok(args[0].clone())
            }
        }

        // ========== Numeric Functions ==========
        ScalarFunction::Abs => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Int(i.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Ceil => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.ceil())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Floor => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.floor())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Round => {
            let val = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => return Ok(Value::Int(*i)),
                _ => return Ok(Value::Null),
            };
            // Optional precision argument
            let precision = match args.get(1) {
                Some(Value::Int(p)) => *p,
                _ => 0,
            };
            if precision == 0 {
                Ok(Value::Float(val.round()))
            } else {
                let factor = 10_f64.powi(precision as i32);
                Ok(Value::Float((val * factor).round() / factor))
            }
        }
        ScalarFunction::Trunc => {
            let val = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => return Ok(Value::Int(*i)),
                _ => return Ok(Value::Null),
            };
            let precision = match args.get(1) {
                Some(Value::Int(p)) => *p,
                _ => 0,
            };
            if precision == 0 {
                Ok(Value::Float(val.trunc()))
            } else {
                let factor = 10_f64.powi(precision as i32);
                Ok(Value::Float((val * factor).trunc() / factor))
            }
        }
        ScalarFunction::Sqrt => match args.first() {
            Some(Value::Float(f)) if *f >= 0.0 => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(i)) if *i >= 0 => Ok(Value::Float((*i as f64).sqrt())),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Power => {
            let base = value_to_f64(args.first().unwrap_or(&Value::Null));
            let exp = value_to_f64(args.get(1).unwrap_or(&Value::Null));
            match (base, exp) {
                (Some(b), Some(e)) => Ok(Value::Float(b.powf(e))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Exp => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.exp())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Ln => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if x > 0.0 => Ok(Value::Float(x.ln())),
                _ => Ok(Value::Null), // ln of non-positive is undefined
            }
        }
        ScalarFunction::Log => {
            // LOG(base, x) or LOG(x) for log base 10
            let (base, x) = if args.len() >= 2 {
                (
                    value_to_f64(args.first().unwrap_or(&Value::Null)),
                    value_to_f64(args.get(1).unwrap_or(&Value::Null)),
                )
            } else {
                (Some(10.0), value_to_f64(args.first().unwrap_or(&Value::Null)))
            };
            match (base, x) {
                (Some(b), Some(v)) if b > 0.0 && b != 1.0 && v > 0.0 => Ok(Value::Float(v.log(b))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Log10 => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if x > 0.0 => Ok(Value::Float(x.log10())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Sin => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.sin())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Cos => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.cos())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Tan => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.tan())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Asin => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if (-1.0..=1.0).contains(&x) => Ok(Value::Float(x.asin())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Acos => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if (-1.0..=1.0).contains(&x) => Ok(Value::Float(x.acos())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Atan => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.atan())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Atan2 => {
            let y = value_to_f64(args.first().unwrap_or(&Value::Null));
            let x = value_to_f64(args.get(1).unwrap_or(&Value::Null));
            match (y, x) {
                (Some(y_val), Some(x_val)) => Ok(Value::Float(y_val.atan2(x_val))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Degrees => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.to_degrees())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Radians => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.to_radians())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Sign => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Int(i.signum())),
            Some(Value::Float(f)) => {
                if *f > 0.0 {
                    Ok(Value::Int(1))
                } else if *f < 0.0 {
                    Ok(Value::Int(-1))
                } else {
                    Ok(Value::Int(0))
                }
            }
            _ => Ok(Value::Null),
        },
        ScalarFunction::Pi => Ok(Value::Float(std::f64::consts::PI)),
        ScalarFunction::Random => Ok(Value::Float(rand_float())),

        // ========== Date/Time Functions ==========
        ScalarFunction::Now | ScalarFunction::CurrentDate | ScalarFunction::CurrentTime => {
            use chrono::Utc;
            let now = Utc::now();
            Ok(Value::String(now.format("%Y-%m-%d %H:%M:%S%.6f+00").to_string()))
        }
        ScalarFunction::Extract | ScalarFunction::DatePart => {
            // EXTRACT(field FROM datetime) or DATE_PART('field', datetime)
            let field = match args.first() {
                Some(Value::String(s)) => s.to_lowercase(),
                _ => return Ok(Value::Null),
            };
            let datetime_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            extract_date_part(&field, datetime_str)
        }
        ScalarFunction::DateTrunc => {
            // DATE_TRUNC('field', datetime)
            let field = match args.first() {
                Some(Value::String(s)) => s.to_lowercase(),
                _ => return Ok(Value::Null),
            };
            let datetime_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            truncate_date(&field, datetime_str)
        }
        ScalarFunction::ToTimestamp => {
            // TO_TIMESTAMP(string, format) or TO_TIMESTAMP(epoch)
            if args.len() == 1 {
                // Epoch seconds
                let epoch = value_to_f64(args.first().unwrap_or(&Value::Null));
                if let Some(secs) = epoch {
                    use chrono::{TimeZone, Utc};
                    let whole_secs = secs.trunc() as i64;
                    let nanos = ((secs.fract()) * 1_000_000_000.0) as u32;
                    if let Some(dt) = Utc.timestamp_opt(whole_secs, nanos).single() {
                        return Ok(Value::String(dt.format("%Y-%m-%d %H:%M:%S+00").to_string()));
                    }
                }
                Ok(Value::Null)
            } else {
                // String with format
                let s = match args.first() {
                    Some(Value::String(s)) => s,
                    _ => return Ok(Value::Null),
                };
                let format = match args.get(1) {
                    Some(Value::String(f)) => f,
                    _ => return Ok(Value::Null),
                };
                parse_datetime_with_format(s, format)
            }
        }
        ScalarFunction::ToDate => {
            // TO_DATE(string, format)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let format = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            parse_date_with_format(s, format)
        }
        ScalarFunction::ToChar => {
            // TO_CHAR(datetime, format)
            let datetime_str = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let format = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            format_datetime(datetime_str, format)
        }
        ScalarFunction::Age => {
            // AGE(timestamp2, timestamp1) or AGE(timestamp) - calculates from now
            use chrono::Utc;
            if args.is_empty() {
                return Ok(Value::Null);
            }

            let (dt2, dt1) = if args.len() == 1 {
                // Single argument: age from now
                let ts = match args.first() {
                    Some(Value::String(s)) => s,
                    _ => return Ok(Value::Null),
                };
                let dt1 = match parse_naive_datetime(ts) {
                    Ok(dt) => dt,
                    Err(_) => return Ok(Value::Null),
                };
                (Utc::now().naive_utc(), dt1)
            } else {
                // Two arguments: age(timestamp2, timestamp1) = timestamp2 - timestamp1
                let ts2 = match args.first() {
                    Some(Value::String(s)) => s,
                    _ => return Ok(Value::Null),
                };
                let ts1 = match args.get(1) {
                    Some(Value::String(s)) => s,
                    _ => return Ok(Value::Null),
                };
                let dt2 = match parse_naive_datetime(ts2) {
                    Ok(dt) => dt,
                    Err(_) => return Ok(Value::Null),
                };
                let dt1 = match parse_naive_datetime(ts1) {
                    Ok(dt) => dt,
                    Err(_) => return Ok(Value::Null),
                };
                (dt2, dt1)
            };

            // Calculate age as interval string
            let interval_str = calculate_age_interval(dt2, dt1);
            Ok(Value::String(interval_str))
        }
        ScalarFunction::DateAdd => {
            // DATE_ADD(date, interval) - Add interval to date
            if args.len() < 2 {
                return Ok(Value::Null);
            }
            let datetime_str = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let interval_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            add_interval_to_datetime(datetime_str, interval_str, true)
        }
        ScalarFunction::DateSubtract => {
            // DATE_SUBTRACT(date, interval) - Subtract interval from date
            if args.len() < 2 {
                return Ok(Value::Null);
            }
            let datetime_str = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let interval_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            add_interval_to_datetime(datetime_str, interval_str, false)
        }
        ScalarFunction::MakeTimestamp => {
            // MAKE_TIMESTAMP(year, month, day, hour, minute, second)
            // All 6 arguments are required
            if args.len() < 6 {
                return Ok(Value::Null);
            }
            let year = value_to_i64(args.first().unwrap_or(&Value::Null)).unwrap_or(0) as i32;
            let month = value_to_i64(args.get(1).unwrap_or(&Value::Null)).unwrap_or(1) as u32;
            let day = value_to_i64(args.get(2).unwrap_or(&Value::Null)).unwrap_or(1) as u32;
            let hour = value_to_i64(args.get(3).unwrap_or(&Value::Null)).unwrap_or(0) as u32;
            let minute = value_to_i64(args.get(4).unwrap_or(&Value::Null)).unwrap_or(0) as u32;
            let second = value_to_f64(args.get(5).unwrap_or(&Value::Null)).unwrap_or(0.0);

            make_timestamp_from_parts(year, month, day, hour, minute, second)
        }
        ScalarFunction::MakeDate => {
            // MAKE_DATE(year, month, day)
            if args.len() < 3 {
                return Ok(Value::Null);
            }
            let year = value_to_i64(args.first().unwrap_or(&Value::Null)).unwrap_or(0) as i32;
            let month = value_to_i64(args.get(1).unwrap_or(&Value::Null)).unwrap_or(1) as u32;
            let day = value_to_i64(args.get(2).unwrap_or(&Value::Null)).unwrap_or(1) as u32;

            make_date_from_parts(year, month, day)
        }
        ScalarFunction::MakeTime => {
            // MAKE_TIME(hour, minute, second)
            if args.len() < 3 {
                return Ok(Value::Null);
            }
            let hour = value_to_i64(args.first().unwrap_or(&Value::Null)).unwrap_or(0) as u32;
            let minute = value_to_i64(args.get(1).unwrap_or(&Value::Null)).unwrap_or(0) as u32;
            let second = value_to_f64(args.get(2).unwrap_or(&Value::Null)).unwrap_or(0.0);

            make_time_from_parts(hour, minute, second)
        }
        ScalarFunction::Timezone => {
            // TIMEZONE(zone, timestamp) - Convert timestamp to timezone
            // For now, we support basic UTC offset timezones like 'UTC', 'America/New_York', etc.
            if args.len() < 2 {
                return Ok(Value::Null);
            }
            let timezone = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let datetime_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            convert_timezone(datetime_str, timezone)
        }

        // ========== Vector Functions ==========
        ScalarFunction::VectorDimension => {
            if let Some(Value::Vector(v)) = args.first() {
                Ok(Value::Int(v.len() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::VectorNorm => {
            if let Some(Value::Vector(v)) = args.first() {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                Ok(Value::Float(f64::from(norm)))
            } else {
                Ok(Value::Null)
            }
        }

        // List/Collection functions
        ScalarFunction::Range => {
            // RANGE(start, end) or RANGE(start, end, step)
            // Returns a list of integers from start to end (inclusive in Cypher)
            match args {
                [Value::Int(start), Value::Int(end)] => {
                    let (start, end) = (*start, *end);
                    let range: Vec<Value> = if start <= end {
                        (start..=end).map(Value::Int).collect()
                    } else {
                        (end..=start).rev().map(Value::Int).collect()
                    };
                    Ok(Value::Array(range))
                }
                [Value::Int(start), Value::Int(end), Value::Int(step)] => {
                    let (start, end, step) = (*start, *end, *step);
                    if step == 0 {
                        return Err(crate::error::ParseError::Execution(
                            "range step cannot be zero".into(),
                        ));
                    }
                    let mut range = Vec::new();
                    let mut current = start;
                    if step > 0 {
                        while current <= end {
                            range.push(Value::Int(current));
                            current += step;
                        }
                    } else {
                        while current >= end {
                            range.push(Value::Int(current));
                            current += step;
                        }
                    }
                    Ok(Value::Array(range))
                }
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Size => {
            // SIZE(list) or SIZE(string)
            match args.first() {
                Some(Value::Array(arr)) => Ok(Value::Int(arr.len() as i64)),
                Some(Value::String(s)) => Ok(Value::Int(s.len() as i64)),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Head => {
            // HEAD(list) - returns first element
            match args.first() {
                Some(Value::Array(arr)) => Ok(arr.first().cloned().unwrap_or(Value::Null)),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Tail => {
            // TAIL(list) - returns list without first element
            match args.first() {
                Some(Value::Array(arr)) => {
                    if arr.is_empty() {
                        Ok(Value::Array(vec![]))
                    } else {
                        Ok(Value::Array(arr[1..].to_vec()))
                    }
                }
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Last => {
            // LAST(list) - returns last element
            match args.first() {
                Some(Value::Array(arr)) => Ok(arr.last().cloned().unwrap_or(Value::Null)),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Reverse => {
            // REVERSE(list) or REVERSE(string)
            match args.first() {
                Some(Value::Array(arr)) => {
                    let reversed: Vec<Value> = arr.iter().rev().cloned().collect();
                    Ok(Value::Array(reversed))
                }
                Some(Value::String(s)) => Ok(Value::String(s.chars().rev().collect())),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        // ========== Array Functions (PostgreSQL-compatible) ==========
        ScalarFunction::ArrayLength => {
            // ARRAY_LENGTH(array, dimension)
            // For 1D arrays, dimension is 1. Returns NULL for empty arrays.
            match args {
                [Value::Array(arr), Value::Int(dim)] => {
                    if *dim != 1 {
                        // Only 1D arrays are currently supported
                        Ok(Value::Null)
                    } else if arr.is_empty() {
                        Ok(Value::Null)
                    } else {
                        Ok(Value::Int(arr.len() as i64))
                    }
                }
                [Value::Null, _] | [_, Value::Null] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Cardinality => {
            // CARDINALITY(array)
            // Returns total number of elements in the array (across all dimensions)
            match args.first() {
                Some(Value::Array(arr)) => Ok(Value::Int(arr.len() as i64)),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayAppend => {
            // ARRAY_APPEND(array, element)
            // Appends element to the end of array
            match args {
                [Value::Array(arr), element] => {
                    let mut new_arr = arr.clone();
                    new_arr.push(element.clone());
                    Ok(Value::Array(new_arr))
                }
                [Value::Null, _] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayPrepend => {
            // ARRAY_PREPEND(element, array)
            // Prepends element to the beginning of array
            match args {
                [element, Value::Array(arr)] => {
                    let mut new_arr = vec![element.clone()];
                    new_arr.extend(arr.iter().cloned());
                    Ok(Value::Array(new_arr))
                }
                [_, Value::Null] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayCat => {
            // ARRAY_CAT(array1, array2)
            // Concatenates two arrays
            match args {
                [Value::Array(arr1), Value::Array(arr2)] => {
                    let mut new_arr = arr1.clone();
                    new_arr.extend(arr2.iter().cloned());
                    Ok(Value::Array(new_arr))
                }
                [Value::Null, _] | [_, Value::Null] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayRemove => {
            // ARRAY_REMOVE(array, element)
            // Removes all occurrences of element from array
            match args {
                [Value::Array(arr), element] => {
                    let new_arr: Vec<Value> =
                        arr.iter().filter(|v| !values_equal(v, element)).cloned().collect();
                    Ok(Value::Array(new_arr))
                }
                [Value::Null, _] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayReplace => {
            // ARRAY_REPLACE(array, from, to)
            // Replaces all occurrences of 'from' with 'to'
            match args {
                [Value::Array(arr), from, to] => {
                    let new_arr: Vec<Value> = arr
                        .iter()
                        .map(|v| if values_equal(v, from) { to.clone() } else { v.clone() })
                        .collect();
                    Ok(Value::Array(new_arr))
                }
                [Value::Null, _, _] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayPosition => {
            // ARRAY_POSITION(array, element)
            // Returns 1-based index of first occurrence, or NULL if not found
            match args {
                [Value::Array(arr), element] => {
                    for (idx, v) in arr.iter().enumerate() {
                        if values_equal(v, element) {
                            return Ok(Value::Int((idx + 1) as i64));
                        }
                    }
                    Ok(Value::Null)
                }
                [Value::Null, _] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::ArrayPositions => {
            // ARRAY_POSITIONS(array, element)
            // Returns array of 1-based indices for all occurrences
            match args {
                [Value::Array(arr), element] => {
                    let positions: Vec<Value> = arr
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| values_equal(v, element))
                        .map(|(idx, _)| Value::Int((idx + 1) as i64))
                        .collect();
                    Ok(Value::Array(positions))
                }
                [Value::Null, _] => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        ScalarFunction::Unnest => {
            // UNNEST(array)
            // Note: UNNEST is typically a set-returning function (SRF) that expands
            // an array to rows. When used as a scalar function, it returns the first
            // element of the array. For proper SRF behavior, use UNWIND in Cypher.
            match args.first() {
                Some(Value::Array(arr)) => Ok(arr.first().cloned().unwrap_or(Value::Null)),
                Some(Value::Null) | None => Ok(Value::Null),
                _ => Ok(Value::Null),
            }
        }

        // ========== JSON Functions ==========
        ScalarFunction::JsonExtractPath | ScalarFunction::JsonbExtractPath => {
            // JSON_EXTRACT_PATH(json, VARIADIC path_elements)
            // Returns the JSON value at the specified path
            json_extract_path(args, false)
        }

        ScalarFunction::JsonExtractPathText | ScalarFunction::JsonbExtractPathText => {
            // JSON_EXTRACT_PATH_TEXT(json, VARIADIC path_elements)
            // Returns the JSON value at the specified path as text
            json_extract_path(args, true)
        }

        ScalarFunction::JsonBuildObject | ScalarFunction::JsonbBuildObject => {
            // JSON_BUILD_OBJECT(key1, val1, key2, val2, ...)
            // Builds a JSON object from alternating key/value pairs
            json_build_object(args)
        }

        ScalarFunction::JsonBuildArray | ScalarFunction::JsonbBuildArray => {
            // JSON_BUILD_ARRAY(val1, val2, ...)
            // Builds a JSON array from the arguments
            json_build_array(args)
        }

        ScalarFunction::JsonbSet => {
            // JSONB_SET(target, path, new_value, create_missing)
            // Sets a value at the specified path within a JSONB document
            jsonb_set(args)
        }

        ScalarFunction::JsonbInsert => {
            // JSONB_INSERT(target, path, new_value, insert_after)
            // Inserts a value at the specified path within a JSONB document
            jsonb_insert(args)
        }

        ScalarFunction::JsonbStripNulls => {
            // JSONB_STRIP_NULLS(jsonb)
            // Removes null values from a JSONB document
            jsonb_strip_nulls(args)
        }

        // ========== Cypher Entity Functions ==========
        ScalarFunction::Type => {
            // TYPE(relationship)
            // Returns the type (string) of a relationship.
            // For now, we expect the argument to be a map with an "_edge_type" key,
            // or return NULL if not found.
            cypher_type(args)
        }
        ScalarFunction::Labels => {
            // LABELS(node)
            // Returns a list of labels for a node.
            // Expects a map with a "_labels" key or returns NULL.
            cypher_labels(args)
        }
        ScalarFunction::Id => {
            // ID(entity)
            // Returns the internal ID of a node or relationship.
            // Expects an integer or a map with "_id" key.
            cypher_id(args)
        }
        ScalarFunction::Properties => {
            // PROPERTIES(entity)
            // Returns a map of all properties of a node or relationship.
            // Expects a map and returns properties (excluding internal keys).
            cypher_properties(args)
        }
        ScalarFunction::Keys => {
            // KEYS(map_or_entity)
            // Returns a list of property keys from a map or entity.
            cypher_keys(args)
        }

        // ========== Cypher Type Conversion Functions ==========
        ScalarFunction::ToBoolean => {
            // toBoolean(expression)
            // Converts to boolean per openCypher spec.
            match args.first() {
                Some(Value::Null) | None => Ok(Value::Null),
                Some(Value::Bool(b)) => Ok(Value::Bool(*b)),
                Some(Value::Int(i)) => Ok(Value::Bool(*i != 0)),
                Some(Value::Float(f)) => Ok(Value::Bool(*f != 0.0)),
                Some(Value::String(s)) => {
                    let s_lower = s.to_lowercase();
                    match s_lower.as_str() {
                        "true" => Ok(Value::Bool(true)),
                        "false" => Ok(Value::Bool(false)),
                        _ => Ok(Value::Null), // Invalid string returns null
                    }
                }
                _ => Ok(Value::Null), // Other types return null
            }
        }
        ScalarFunction::ToInteger => {
            // toInteger(expression)
            // Converts to integer per openCypher spec.
            match args.first() {
                Some(Value::Null) | None => Ok(Value::Null),
                Some(Value::Int(i)) => Ok(Value::Int(*i)),
                Some(Value::Float(f)) => {
                    // Truncate towards zero (as per openCypher spec)
                    Ok(Value::Int(f.trunc() as i64))
                }
                Some(Value::Bool(b)) => Ok(Value::Int(if *b { 1 } else { 0 })),
                Some(Value::String(s)) => {
                    // Try parsing as integer first, then as float (truncating)
                    if let Ok(i) = s.trim().parse::<i64>() {
                        Ok(Value::Int(i))
                    } else if let Ok(f) = s.trim().parse::<f64>() {
                        Ok(Value::Int(f.trunc() as i64))
                    } else {
                        Ok(Value::Null) // Invalid string returns null
                    }
                }
                _ => Ok(Value::Null), // Other types return null
            }
        }
        ScalarFunction::ToFloat => {
            // toFloat(expression)
            // Converts to float per openCypher spec.
            match args.first() {
                Some(Value::Null) | None => Ok(Value::Null),
                Some(Value::Float(f)) => Ok(Value::Float(*f)),
                Some(Value::Int(i)) => Ok(Value::Float(*i as f64)),
                Some(Value::Bool(b)) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                Some(Value::String(s)) => {
                    if let Ok(f) = s.trim().parse::<f64>() {
                        Ok(Value::Float(f))
                    } else {
                        Ok(Value::Null) // Invalid string returns null
                    }
                }
                _ => Ok(Value::Null), // Other types return null
            }
        }
        ScalarFunction::CypherToString => {
            // toString(expression)
            // Converts to string per openCypher spec.
            match args.first() {
                Some(Value::Null) | None => Ok(Value::Null),
                Some(Value::String(s)) => Ok(Value::String(s.clone())),
                Some(Value::Bool(b)) => Ok(Value::String(b.to_string())),
                Some(Value::Int(i)) => Ok(Value::String(i.to_string())),
                Some(Value::Float(f)) => {
                    // Format float to remove unnecessary trailing zeros
                    let s = if f.fract() == 0.0 {
                        format!("{:.1}", f) // Keep at least one decimal for floats like 3.0
                    } else {
                        f.to_string()
                    };
                    Ok(Value::String(s))
                }
                // Lists and arrays are also converted per Cypher
                Some(Value::Array(arr)) => {
                    let elements: Vec<String> = arr.iter().map(value_to_string).collect();
                    Ok(Value::String(format!("[{}]", elements.join(", "))))
                }
                _ => Ok(Value::Null), // Other types return null
            }
        }

        // ========== Cypher Path Functions ==========
        ScalarFunction::Nodes => {
            // nodes(path)
            // Returns a list of all nodes in a path.
            cypher_nodes(args)
        }
        ScalarFunction::Relationships => {
            // relationships(path)
            // Returns a list of all relationships/edges in a path.
            cypher_relationships(args)
        }
        ScalarFunction::StartNode => {
            // startNode(relationship)
            // Returns the start node ID of a relationship.
            cypher_start_node(args)
        }
        ScalarFunction::EndNode => {
            // endNode(relationship)
            // Returns the end node ID of a relationship.
            cypher_end_node(args)
        }
        ScalarFunction::PathLength => {
            // length(path)
            // Returns the length of a path (number of relationships).
            cypher_path_length(args)
        }

        // ========== Cypher Temporal Functions ==========
        ScalarFunction::CypherDatetime => {
            // datetime() - returns current datetime
            // datetime('2024-01-15T10:30:00') - parse ISO 8601 string
            // datetime({year: 2024, month: 1, day: 15}) - construct from map
            cypher_datetime(args)
        }
        ScalarFunction::CypherDate => {
            // date() - returns current date
            // date('2024-01-15') - parse ISO 8601 string
            // date({year: 2024, month: 1, day: 15}) - construct from map
            cypher_date(args)
        }
        ScalarFunction::CypherTime => {
            // time() - returns current time
            // time('10:30:00') - parse ISO 8601 string
            // time({hour: 10, minute: 30}) - construct from map
            cypher_time(args)
        }
        ScalarFunction::CypherLocalDatetime => {
            // localdatetime() - returns current local datetime (without timezone)
            // localdatetime('2024-01-15T10:30:00') - parse ISO 8601 string
            // localdatetime({year: 2024, month: 1, day: 15}) - construct from map
            cypher_localdatetime(args)
        }
        ScalarFunction::CypherLocalTime => {
            // localtime() - returns current local time (without timezone)
            // localtime('10:30:00') - parse ISO 8601 string
            // localtime({hour: 10, minute: 30}) - construct from map
            cypher_localtime(args)
        }
        ScalarFunction::CypherDuration => {
            // duration('P1Y2M3D') - parse ISO 8601 duration
            // duration({days: 14, hours: 16}) - construct from map
            cypher_duration(args)
        }
        ScalarFunction::CypherDatetimeTruncate => {
            // datetime.truncate('day', datetime()) - truncate to start of day
            // datetime.truncate('month', datetime()) - truncate to start of month
            cypher_datetime_truncate(args)
        }

        // Custom functions (not implemented)
        ScalarFunction::Custom(_) => Ok(Value::Null),
    }
}

/// Generates a random float between 0.0 and 1.0.
fn rand_float() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    (hasher.finish() as f64) / (u64::MAX as f64)
}

/// Extracts a date part from a datetime string.
fn extract_date_part(field: &str, datetime_str: &str) -> OperatorResult<Value> {
    use chrono::{Datelike, Timelike};

    // Try parsing common formats
    let dt = parse_naive_datetime(datetime_str)?;

    let value = match field {
        "year" => dt.year() as f64,
        "month" => dt.month() as f64,
        "day" => dt.day() as f64,
        "hour" => dt.hour() as f64,
        "minute" => dt.minute() as f64,
        "second" => dt.second() as f64,
        "millisecond" | "milliseconds" => (dt.nanosecond() / 1_000_000) as f64,
        "microsecond" | "microseconds" => (dt.nanosecond() / 1_000) as f64,
        "dow" | "dayofweek" => dt.weekday().num_days_from_sunday() as f64,
        "doy" | "dayofyear" => dt.ordinal() as f64,
        "week" => dt.iso_week().week() as f64,
        "quarter" => ((dt.month() - 1) / 3 + 1) as f64,
        "epoch" => dt.and_utc().timestamp() as f64,
        _ => return Ok(Value::Null),
    };

    Ok(Value::Float(value))
}

/// Truncates a datetime to the specified precision.
fn truncate_date(field: &str, datetime_str: &str) -> OperatorResult<Value> {
    use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};

    let dt = parse_naive_datetime(datetime_str)?;

    let truncated = match field {
        "year" => NaiveDateTime::new(
            NaiveDate::from_ymd_opt(dt.year(), 1, 1).unwrap_or_else(|| dt.date()),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
        ),
        "month" => NaiveDateTime::new(
            NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1).unwrap_or_else(|| dt.date()),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
        ),
        "day" => {
            NaiveDateTime::new(dt.date(), NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default())
        }
        "hour" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), 0, 0).unwrap_or_default(),
        ),
        "minute" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), dt.minute(), 0).unwrap_or_default(),
        ),
        "second" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), dt.minute(), dt.second()).unwrap_or_default(),
        ),
        "week" => {
            let days_since_monday = dt.weekday().num_days_from_monday();
            let monday = dt.date() - chrono::Duration::days(days_since_monday as i64);
            NaiveDateTime::new(monday, NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default())
        }
        "quarter" => {
            let quarter_month = ((dt.month() - 1) / 3) * 3 + 1;
            NaiveDateTime::new(
                NaiveDate::from_ymd_opt(dt.year(), quarter_month, 1).unwrap_or_else(|| dt.date()),
                NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
            )
        }
        _ => return Ok(Value::Null),
    };

    Ok(Value::String(truncated.format("%Y-%m-%d %H:%M:%S").to_string()))
}

/// Parses a naive datetime from various formats.
fn parse_naive_datetime(s: &str) -> OperatorResult<chrono::NaiveDateTime> {
    use chrono::NaiveDateTime;

    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ];

    // Strip timezone suffix if present
    let s = s.trim_end_matches("+00").trim_end_matches('Z');

    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(dt);
        }
    }

    // Try date-only and add midnight
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date.and_hms_opt(0, 0, 0).unwrap_or_default());
    }

    Err(crate::error::ParseError::Execution(format!("Cannot parse datetime: {}", s)))
}

/// Parses a datetime string with PostgreSQL-style format.
fn parse_datetime_with_format(s: &str, pg_format: &str) -> OperatorResult<Value> {
    let chrono_format = pg_format_to_chrono(pg_format);
    use chrono::NaiveDateTime;

    match NaiveDateTime::parse_from_str(s, &chrono_format) {
        Ok(dt) => Ok(Value::String(dt.format("%Y-%m-%d %H:%M:%S").to_string())),
        Err(_) => Ok(Value::Null),
    }
}

/// Parses a date string with PostgreSQL-style format.
fn parse_date_with_format(s: &str, pg_format: &str) -> OperatorResult<Value> {
    let chrono_format = pg_format_to_chrono(pg_format);
    use chrono::NaiveDate;

    match NaiveDate::parse_from_str(s, &chrono_format) {
        Ok(d) => Ok(Value::String(d.format("%Y-%m-%d").to_string())),
        Err(_) => Ok(Value::Null),
    }
}

/// Formats a datetime with PostgreSQL-style format.
fn format_datetime(datetime_str: &str, pg_format: &str) -> OperatorResult<Value> {
    let dt = parse_naive_datetime(datetime_str)?;
    let chrono_format = pg_format_to_chrono(pg_format);
    Ok(Value::String(dt.format(&chrono_format).to_string()))
}

/// Converts PostgreSQL format specifiers to chrono format specifiers.
fn pg_format_to_chrono(pg_format: &str) -> String {
    pg_format
        .replace("YYYY", "%Y")
        .replace("YY", "%y")
        .replace("MM", "%m")
        .replace("DD", "%d")
        .replace("HH24", "%H")
        .replace("HH12", "%I")
        .replace("HH", "%H")
        .replace("MI", "%M")
        .replace("SS", "%S")
        .replace("MS", "%3f")
        .replace("US", "%6f")
        .replace("AM", "%p")
        .replace("PM", "%p")
        .replace("am", "%P")
        .replace("pm", "%P")
        .replace("TZ", "%Z")
        .replace("Day", "%A")
        .replace("Mon", "%b")
        .replace("Month", "%B")
}

// ========== Tier 2 Date/Time Helper Functions ==========

/// Converts a Value to i64 if possible.
fn value_to_i64(val: &Value) -> Option<i64> {
    match val {
        Value::Int(i) => Some(*i),
        Value::Float(f) => Some(*f as i64),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Calculates age between two datetimes as an interval string.
/// Returns a PostgreSQL-style interval like "4 years 2 mons 10 days".
fn calculate_age_interval(dt2: chrono::NaiveDateTime, dt1: chrono::NaiveDateTime) -> String {
    use chrono::Datelike;

    // Calculate the difference
    let (later, earlier, negative) = if dt2 >= dt1 { (dt2, dt1, false) } else { (dt1, dt2, true) };

    // Calculate years, months, days
    let mut years = later.year() - earlier.year();
    let mut months = later.month() as i32 - earlier.month() as i32;
    let mut days = later.day() as i32 - earlier.day() as i32;

    // Adjust for day underflow
    if days < 0 {
        months -= 1;
        // Add days from previous month
        let prev_month = if later.month() == 1 { 12 } else { later.month() - 1 };
        let prev_year = if later.month() == 1 { later.year() - 1 } else { later.year() };
        let days_in_prev_month = days_in_month(prev_year, prev_month);
        days += days_in_prev_month as i32;
    }

    // Adjust for month underflow
    if months < 0 {
        years -= 1;
        months += 12;
    }

    // Calculate time difference
    let time_diff = later.time() - earlier.time();
    let total_seconds = time_diff.num_seconds();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;

    // Build the interval string
    let mut parts = Vec::new();
    if years != 0 {
        parts.push(format!("{} year{}", years, if years == 1 { "" } else { "s" }));
    }
    if months != 0 {
        parts.push(format!("{} mon{}", months, if months == 1 { "" } else { "s" }));
    }
    if days != 0 {
        parts.push(format!("{} day{}", days, if days == 1 { "" } else { "s" }));
    }
    if hours != 0 || minutes != 0 || seconds != 0 {
        parts.push(format!("{:02}:{:02}:{:02}", hours, minutes, seconds));
    }

    let interval = if parts.is_empty() { "00:00:00".to_string() } else { parts.join(" ") };

    if negative {
        format!("-{}", interval)
    } else {
        interval
    }
}

/// Returns the number of days in a given month.
fn days_in_month(year: i32, month: u32) -> u32 {
    use chrono::NaiveDate;
    let next_month = if month == 12 { 1 } else { month + 1 };
    let next_year = if month == 12 { year + 1 } else { year };
    let first_of_next = NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap_or_default();
    let first_of_this = NaiveDate::from_ymd_opt(year, month, 1).unwrap_or_default();
    (first_of_next - first_of_this).num_days() as u32
}

/// Parses an interval string and adds/subtracts it from a datetime.
/// Supports formats like: "1 year", "2 months", "3 days", "4 hours", "5 minutes", "6 seconds"
/// Also supports compound: "1 year 2 months 3 days"
fn add_interval_to_datetime(
    datetime_str: &str,
    interval_str: &str,
    add: bool,
) -> OperatorResult<Value> {
    use chrono::{Datelike, NaiveDate, NaiveDateTime};

    let dt = parse_naive_datetime(datetime_str)?;
    let interval_str = interval_str.to_lowercase();

    // Parse the interval components
    let mut years: i32 = 0;
    let mut months: i32 = 0;
    let mut days: i64 = 0;
    let mut hours: i64 = 0;
    let mut minutes: i64 = 0;
    let mut seconds: i64 = 0;

    // Parse using simple word-by-word approach
    let words: Vec<&str> = interval_str.split_whitespace().collect();
    let mut i = 0;
    while i < words.len() {
        if let Ok(n) = words[i].parse::<i64>() {
            if i + 1 < words.len() {
                let unit = words[i + 1];
                match unit {
                    u if u.starts_with("year") => years = n as i32,
                    u if u.starts_with("mon") => months = n as i32,
                    u if u.starts_with("day") => days = n,
                    u if u.starts_with("hour") => hours = n,
                    u if u.starts_with("min") => minutes = n,
                    u if u.starts_with("sec") => seconds = n,
                    _ => {}
                }
                i += 2;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    // Apply the sign for subtraction
    let sign: i32 = if add { 1 } else { -1 };
    years *= sign;
    months *= sign;
    days *= sign as i64;
    hours *= sign as i64;
    minutes *= sign as i64;
    seconds *= sign as i64;

    // Add years and months (these require special handling)
    let mut new_year = dt.year() + years;
    let mut new_month = dt.month() as i32 + months;

    // Handle month overflow/underflow
    while new_month > 12 {
        new_month -= 12;
        new_year += 1;
    }
    while new_month < 1 {
        new_month += 12;
        new_year -= 1;
    }

    // Handle day clamping (e.g., Jan 31 + 1 month = Feb 28/29)
    let days_in_new_month = days_in_month(new_year, new_month as u32);
    let new_day = std::cmp::min(dt.day(), days_in_new_month);

    let new_date =
        NaiveDate::from_ymd_opt(new_year, new_month as u32, new_day).unwrap_or_else(|| dt.date());

    let new_dt = NaiveDateTime::new(new_date, dt.time());

    // Add days, hours, minutes, seconds using Duration
    let duration = chrono::Duration::days(days)
        + chrono::Duration::hours(hours)
        + chrono::Duration::minutes(minutes)
        + chrono::Duration::seconds(seconds);

    let result = new_dt + duration;

    Ok(Value::String(result.format("%Y-%m-%d %H:%M:%S").to_string()))
}

/// Creates a timestamp from individual components.
fn make_timestamp_from_parts(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: f64,
) -> OperatorResult<Value> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    let date = match NaiveDate::from_ymd_opt(year, month, day) {
        Some(d) => d,
        None => return Ok(Value::Null),
    };

    let whole_secs = second.trunc() as u32;
    let nanos = ((second.fract()) * 1_000_000_000.0) as u32;

    let time = match NaiveTime::from_hms_nano_opt(hour, minute, whole_secs, nanos) {
        Some(t) => t,
        None => return Ok(Value::Null),
    };

    let dt = NaiveDateTime::new(date, time);
    Ok(Value::String(dt.format("%Y-%m-%d %H:%M:%S%.6f").to_string()))
}

/// Creates a date from individual components.
fn make_date_from_parts(year: i32, month: u32, day: u32) -> OperatorResult<Value> {
    use chrono::NaiveDate;

    match NaiveDate::from_ymd_opt(year, month, day) {
        Some(d) => Ok(Value::String(d.format("%Y-%m-%d").to_string())),
        None => Ok(Value::Null),
    }
}

/// Creates a time from individual components.
fn make_time_from_parts(hour: u32, minute: u32, second: f64) -> OperatorResult<Value> {
    use chrono::NaiveTime;

    let whole_secs = second.trunc() as u32;
    let nanos = ((second.fract()) * 1_000_000_000.0) as u32;

    match NaiveTime::from_hms_nano_opt(hour, minute, whole_secs, nanos) {
        Some(t) => Ok(Value::String(t.format("%H:%M:%S%.6f").to_string())),
        None => Ok(Value::Null),
    }
}

/// Converts a timestamp to a different timezone.
/// Supports common timezone names and UTC offsets.
fn convert_timezone(datetime_str: &str, timezone: &str) -> OperatorResult<Value> {
    use chrono::{FixedOffset, TimeZone};

    let dt = parse_naive_datetime(datetime_str)?;

    // Parse timezone - support common formats
    let offset = match timezone.to_uppercase().as_str() {
        "UTC" | "GMT" => FixedOffset::east_opt(0),
        // Common US timezones (standard time offsets)
        "EST" => FixedOffset::west_opt(5 * 3600),
        "EDT" => FixedOffset::west_opt(4 * 3600),
        "CST" => FixedOffset::west_opt(6 * 3600),
        "CDT" => FixedOffset::west_opt(5 * 3600),
        "MST" => FixedOffset::west_opt(7 * 3600),
        "MDT" => FixedOffset::west_opt(6 * 3600),
        "PST" => FixedOffset::west_opt(8 * 3600),
        "PDT" => FixedOffset::west_opt(7 * 3600),
        // Common international timezones
        "CET" => FixedOffset::east_opt(3600),
        "CEST" => FixedOffset::east_opt(2 * 3600),
        "JST" => FixedOffset::east_opt(9 * 3600),
        "IST" => FixedOffset::east_opt(5 * 3600 + 30 * 60), // India Standard Time
        _ => {
            // Try to parse as offset like "+05:00" or "-08:00"
            parse_timezone_offset(timezone)
        }
    };

    match offset {
        Some(tz) => {
            // Assume input is UTC, convert to target timezone
            let utc_dt = chrono::Utc.from_utc_datetime(&dt);
            let local_dt = utc_dt.with_timezone(&tz);
            Ok(Value::String(local_dt.format("%Y-%m-%d %H:%M:%S%:z").to_string()))
        }
        None => Ok(Value::Null),
    }
}

/// Parses a timezone offset string like "+05:00" or "-08:00".
fn parse_timezone_offset(s: &str) -> Option<chrono::FixedOffset> {
    use chrono::FixedOffset;

    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    let (sign, rest) = if s.starts_with('+') {
        (1, &s[1..])
    } else if s.starts_with('-') {
        (-1, &s[1..])
    } else {
        (1, s)
    };

    // Try formats: "05:00", "0500", "05"
    let (hours, minutes) = if rest.contains(':') {
        let parts: Vec<&str> = rest.split(':').collect();
        if parts.len() >= 2 {
            (parts[0].parse::<i32>().ok()?, parts[1].parse::<i32>().ok()?)
        } else {
            return None;
        }
    } else if rest.len() == 4 {
        // Format: "0500"
        (rest[0..2].parse::<i32>().ok()?, rest[2..4].parse::<i32>().ok()?)
    } else if rest.len() <= 2 {
        // Format: "5" or "05"
        (rest.parse::<i32>().ok()?, 0)
    } else {
        return None;
    };

    let total_seconds = sign * (hours * 3600 + minutes * 60);
    FixedOffset::east_opt(total_seconds)
}

// ========== JSON Functions ==========

/// Converts a `Value` to a `serde_json::Value`.
fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        Value::Vector(v) => serde_json::Value::Array(
            v.iter()
                .filter_map(|f| serde_json::Number::from_f64(f64::from(*f)))
                .map(serde_json::Value::Number)
                .collect(),
        ),
        Value::SparseVector(sv) => serde_json::Value::Object(
            sv.iter()
                .filter_map(|(idx, val)| {
                    serde_json::Number::from_f64(f64::from(*val))
                        .map(|n| (idx.to_string(), serde_json::Value::Number(n)))
                })
                .collect(),
        ),
        Value::MultiVector(mv) => serde_json::Value::Array(
            mv.iter()
                .map(|v| {
                    serde_json::Value::Array(
                        v.iter()
                            .filter_map(|f| serde_json::Number::from_f64(f64::from(*f)))
                            .map(serde_json::Value::Number)
                            .collect(),
                    )
                })
                .collect(),
        ),
        Value::Bytes(b) => serde_json::Value::String(base64_encode(b)),
    }
}

/// Encodes bytes as base64 string.
fn base64_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = chunk.first().copied().unwrap_or(0);
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);

        let _ = write!(result, "{}", BASE64_CHARS[(b0 >> 2) as usize] as char);
        let _ =
            write!(result, "{}", BASE64_CHARS[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            let _ = write!(
                result,
                "{}",
                BASE64_CHARS[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char
            );
        } else {
            let _ = write!(result, "=");
        }
        if chunk.len() > 2 {
            let _ = write!(result, "{}", BASE64_CHARS[(b2 & 0x3f) as usize] as char);
        } else {
            let _ = write!(result, "=");
        }
    }
    result
}

/// Converts a `serde_json::Value` back to a `Value`.
#[allow(dead_code)]
fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => Value::Array(arr.into_iter().map(json_to_value).collect()),
        serde_json::Value::Object(obj) => {
            // Convert object to JSON string representation
            Value::String(
                serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default(),
            )
        }
    }
}

/// Parses a JSON string into a `serde_json::Value`.
fn parse_json(s: &str) -> Option<serde_json::Value> {
    serde_json::from_str(s).ok()
}

/// Extracts a value from JSON at the specified path.
/// If `as_text` is true, returns the extracted value as a string.
fn json_extract_path(args: &[Value], as_text: bool) -> OperatorResult<Value> {
    // First arg is the JSON document
    let json_str = match args.first() {
        Some(Value::String(s)) => s,
        Some(Value::Null) | None => return Ok(Value::Null),
        // If not a string, try to convert to JSON
        Some(other) => {
            let json = value_to_json(other);
            let json_str = serde_json::to_string(&json).unwrap_or_default();
            return json_extract_path_impl(&json_str, &args[1..], as_text);
        }
    };

    json_extract_path_impl(json_str, &args[1..], as_text)
}

/// Implementation of JSON path extraction.
fn json_extract_path_impl(json_str: &str, path: &[Value], as_text: bool) -> OperatorResult<Value> {
    // Parse the JSON
    let mut current = match parse_json(json_str) {
        Some(v) => v,
        None => return Ok(Value::Null),
    };

    // Navigate through the path
    for path_elem in path {
        let key = match path_elem {
            Value::String(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::Null => return Ok(Value::Null),
            _ => return Ok(Value::Null),
        };

        current = match current {
            serde_json::Value::Object(mut obj) => {
                obj.remove(&key).unwrap_or(serde_json::Value::Null)
            }
            serde_json::Value::Array(arr) => {
                // Try to parse key as array index
                if let Ok(idx) = key.parse::<usize>() {
                    arr.into_iter().nth(idx).unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                }
            }
            _ => serde_json::Value::Null,
        };

        if current.is_null() {
            return Ok(Value::Null);
        }
    }

    // Return result
    if as_text {
        // Return as text (no quotes for strings)
        match current {
            serde_json::Value::Null => Ok(Value::Null),
            serde_json::Value::String(s) => Ok(Value::String(s)),
            other => Ok(Value::String(other.to_string())),
        }
    } else {
        // Return as JSON string representation
        match current {
            serde_json::Value::Null => Ok(Value::Null),
            other => Ok(Value::String(serde_json::to_string(&other).unwrap_or_default())),
        }
    }
}

/// Builds a JSON object from key/value pairs.
fn json_build_object(args: &[Value]) -> OperatorResult<Value> {
    // Arguments should come in pairs: key1, val1, key2, val2, ...
    if args.len() % 2 != 0 {
        // Odd number of arguments - error
        return Ok(Value::Null);
    }

    let mut obj = serde_json::Map::new();

    for chunk in args.chunks(2) {
        // First element is the key
        let key = match &chunk[0] {
            Value::String(s) => s.clone(),
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
            _ => continue,
        };

        // Second element is the value
        let val = value_to_json(&chunk[1]);
        obj.insert(key, val);
    }

    let json_str = serde_json::to_string(&serde_json::Value::Object(obj)).unwrap_or_default();
    Ok(Value::String(json_str))
}

/// Builds a JSON array from values.
fn json_build_array(args: &[Value]) -> OperatorResult<Value> {
    let arr: Vec<serde_json::Value> = args.iter().map(value_to_json).collect();
    let json_str = serde_json::to_string(&serde_json::Value::Array(arr)).unwrap_or_default();
    Ok(Value::String(json_str))
}

/// Sets a value at the specified path within a JSONB document.
/// Args: target, path (as array), new_value, create_missing (optional, default true)
fn jsonb_set(args: &[Value]) -> OperatorResult<Value> {
    // Extract arguments
    let target_str = match args.first() {
        Some(Value::String(s)) => s,
        Some(Value::Null) | None => return Ok(Value::Null),
        _ => return Ok(Value::Null),
    };

    // Path should be an array of strings/integers
    let path_values = match args.get(1) {
        Some(Value::Array(arr)) => arr,
        Some(Value::String(s)) => {
            // If it's a string, it might be a JSON array representation
            if let Some(serde_json::Value::Array(arr)) = parse_json(s) {
                let path: Vec<String> = arr
                    .into_iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        serde_json::Value::Number(n) => Some(n.to_string()),
                        _ => None,
                    })
                    .collect();
                return jsonb_set_impl(target_str, &path, args.get(2), args.get(3));
            }
            return Ok(Value::Null);
        }
        _ => return Ok(Value::Null),
    };

    // Convert path values to strings
    let path: Vec<String> = path_values
        .iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Int(i) => Some(i.to_string()),
            _ => None,
        })
        .collect();

    jsonb_set_impl(target_str, &path, args.get(2), args.get(3))
}

/// Implementation of jsonb_set.
fn jsonb_set_impl(
    target_str: &str,
    path: &[String],
    new_value: Option<&Value>,
    create_missing: Option<&Value>,
) -> OperatorResult<Value> {
    // Parse target JSON
    let mut target = match parse_json(target_str) {
        Some(v) => v,
        None => return Ok(Value::Null),
    };

    // Get new value
    let new_val = match new_value {
        Some(v) => value_to_json(v),
        None => return Ok(Value::Null),
    };

    // Check create_missing flag (default true)
    let should_create = match create_missing {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.to_lowercase() == "true",
        _ => true, // Default to true
    };

    // Navigate and set
    if path.is_empty() {
        return Ok(Value::String(serde_json::to_string(&new_val).unwrap_or_default()));
    }

    set_json_path(&mut target, path, new_val, should_create);

    Ok(Value::String(serde_json::to_string(&target).unwrap_or_default()))
}

/// Recursively sets a value at a JSON path.
fn set_json_path(
    current: &mut serde_json::Value,
    path: &[String],
    new_val: serde_json::Value,
    create_missing: bool,
) {
    if path.is_empty() {
        return;
    }

    let key = &path[0];
    let remaining = &path[1..];

    if remaining.is_empty() {
        // We're at the target location, set the value
        match current {
            serde_json::Value::Object(obj) => {
                if create_missing || obj.contains_key(key) {
                    obj.insert(key.clone(), new_val);
                }
            }
            serde_json::Value::Array(arr) => {
                if let Ok(idx) = key.parse::<usize>() {
                    if idx < arr.len() {
                        arr[idx] = new_val;
                    } else if create_missing {
                        // Extend array with nulls and then set
                        while arr.len() < idx {
                            arr.push(serde_json::Value::Null);
                        }
                        arr.push(new_val);
                    }
                }
            }
            _ => {}
        }
    } else {
        // Navigate deeper
        match current {
            serde_json::Value::Object(obj) => {
                if let Some(child) = obj.get_mut(key) {
                    set_json_path(child, remaining, new_val, create_missing);
                } else if create_missing {
                    // Create intermediate object or array based on next key
                    let next_is_index =
                        remaining.first().map(|k| k.parse::<usize>().is_ok()).unwrap_or(false);
                    let mut new_child = if next_is_index {
                        serde_json::Value::Array(vec![])
                    } else {
                        serde_json::Value::Object(serde_json::Map::new())
                    };
                    set_json_path(&mut new_child, remaining, new_val, create_missing);
                    obj.insert(key.clone(), new_child);
                }
            }
            serde_json::Value::Array(arr) => {
                if let Ok(idx) = key.parse::<usize>() {
                    if idx < arr.len() {
                        set_json_path(&mut arr[idx], remaining, new_val, create_missing);
                    } else if create_missing {
                        // Extend array
                        while arr.len() <= idx {
                            arr.push(serde_json::Value::Null);
                        }
                        let next_is_index =
                            remaining.first().map(|k| k.parse::<usize>().is_ok()).unwrap_or(false);
                        arr[idx] = if next_is_index {
                            serde_json::Value::Array(vec![])
                        } else {
                            serde_json::Value::Object(serde_json::Map::new())
                        };
                        set_json_path(&mut arr[idx], remaining, new_val, create_missing);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Inserts a value at the specified path within a JSONB document.
/// Args: target, path (as array), new_value, insert_after (optional, default false)
fn jsonb_insert(args: &[Value]) -> OperatorResult<Value> {
    // Extract arguments
    let target_str = match args.first() {
        Some(Value::String(s)) => s,
        Some(Value::Null) | None => return Ok(Value::Null),
        _ => return Ok(Value::Null),
    };

    // Path should be an array
    let path_values = match args.get(1) {
        Some(Value::Array(arr)) => arr,
        Some(Value::String(s)) => {
            if let Some(serde_json::Value::Array(arr)) = parse_json(s) {
                let path: Vec<String> = arr
                    .into_iter()
                    .filter_map(|v| match v {
                        serde_json::Value::String(s) => Some(s),
                        serde_json::Value::Number(n) => Some(n.to_string()),
                        _ => None,
                    })
                    .collect();
                return jsonb_insert_impl(target_str, &path, args.get(2), args.get(3));
            }
            return Ok(Value::Null);
        }
        _ => return Ok(Value::Null),
    };

    let path: Vec<String> = path_values
        .iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Int(i) => Some(i.to_string()),
            _ => None,
        })
        .collect();

    jsonb_insert_impl(target_str, &path, args.get(2), args.get(3))
}

/// Implementation of jsonb_insert.
fn jsonb_insert_impl(
    target_str: &str,
    path: &[String],
    new_value: Option<&Value>,
    insert_after: Option<&Value>,
) -> OperatorResult<Value> {
    let mut target = match parse_json(target_str) {
        Some(v) => v,
        None => return Ok(Value::Null),
    };

    let new_val = match new_value {
        Some(v) => value_to_json(v),
        None => return Ok(Value::Null),
    };

    let after = match insert_after {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => s.to_lowercase() == "true",
        _ => false,
    };

    if path.is_empty() {
        return Ok(Value::String(target_str.to_string()));
    }

    insert_json_path(&mut target, path, new_val, after);

    Ok(Value::String(serde_json::to_string(&target).unwrap_or_default()))
}

/// Recursively inserts a value at a JSON path (for arrays).
fn insert_json_path(
    current: &mut serde_json::Value,
    path: &[String],
    new_val: serde_json::Value,
    after: bool,
) {
    if path.is_empty() {
        return;
    }

    let key = &path[0];
    let remaining = &path[1..];

    if remaining.is_empty() {
        // We're at the target location
        match current {
            serde_json::Value::Object(obj) => {
                // For objects, just insert (like set)
                obj.insert(key.clone(), new_val);
            }
            serde_json::Value::Array(arr) => {
                if let Ok(idx) = key.parse::<usize>() {
                    let insert_idx = if after { idx + 1 } else { idx };
                    let insert_idx = insert_idx.min(arr.len());
                    arr.insert(insert_idx, new_val);
                }
            }
            _ => {}
        }
    } else {
        // Navigate deeper
        match current {
            serde_json::Value::Object(obj) => {
                if let Some(child) = obj.get_mut(key) {
                    insert_json_path(child, remaining, new_val, after);
                }
            }
            serde_json::Value::Array(arr) => {
                if let Ok(idx) = key.parse::<usize>() {
                    if idx < arr.len() {
                        insert_json_path(&mut arr[idx], remaining, new_val, after);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Removes null values from a JSONB document recursively.
fn jsonb_strip_nulls(args: &[Value]) -> OperatorResult<Value> {
    let json_str = match args.first() {
        Some(Value::String(s)) => s,
        Some(Value::Null) | None => return Ok(Value::Null),
        Some(other) => {
            // Convert non-string value to JSON
            let json = value_to_json(other);
            let stripped = strip_nulls_recursive(json);
            return Ok(Value::String(serde_json::to_string(&stripped).unwrap_or_default()));
        }
    };

    let json = match parse_json(json_str) {
        Some(v) => v,
        None => return Ok(Value::Null),
    };

    let stripped = strip_nulls_recursive(json);
    Ok(Value::String(serde_json::to_string(&stripped).unwrap_or_default()))
}

/// Recursively strips null values from a JSON value.
fn strip_nulls_recursive(val: serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(obj) => {
            let filtered: serde_json::Map<String, serde_json::Value> = obj
                .into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, strip_nulls_recursive(v)))
                .collect();
            serde_json::Value::Object(filtered)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(strip_nulls_recursive).collect())
        }
        other => other,
    }
}

// ========== Cypher Temporal Functions ==========

/// datetime() - Returns current datetime or constructs from string/map.
///
/// Formats:
/// - datetime() - current datetime with timezone
/// - datetime('2024-01-15T10:30:00') - parse ISO 8601 string
/// - datetime({year: 2024, month: 1, day: 15, ...}) - construct from map
fn cypher_datetime(args: &[Value]) -> OperatorResult<Value> {
    use chrono::Utc;

    if args.is_empty() {
        // No arguments: return current datetime with timezone
        let now = Utc::now();
        return Ok(Value::String(now.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()));
    }

    match args.first() {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 datetime or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return datetime_from_json_map(&json);
                }
            }
            // Parse ISO 8601 datetime string
            parse_cypher_datetime_string(s)
        }
        // Map construction not directly supported via Value yet,
        // but we handle JSON strings above
        _ => Ok(Value::Null),
    }
}

/// date() - Returns current date or constructs from string/map.
///
/// Formats:
/// - date() - current date
/// - date('2024-01-15') - parse ISO 8601 date string
/// - date({year: 2024, month: 1, day: 15}) - construct from map
fn cypher_date(args: &[Value]) -> OperatorResult<Value> {
    use chrono::Utc;

    if args.is_empty() {
        // No arguments: return current date
        let now = Utc::now();
        return Ok(Value::String(now.format("%Y-%m-%d").to_string()));
    }

    match args.first() {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 date or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return date_from_json_map(&json);
                }
            }
            // Parse ISO 8601 date string
            parse_cypher_date_string(s)
        }
        _ => Ok(Value::Null),
    }
}

/// time() - Returns current time or constructs from string/map.
///
/// Formats:
/// - time() - current time with timezone
/// - time('10:30:00') - parse ISO 8601 time string
/// - time({hour: 10, minute: 30, second: 0}) - construct from map
fn cypher_time(args: &[Value]) -> OperatorResult<Value> {
    use chrono::Utc;

    if args.is_empty() {
        // No arguments: return current time with timezone
        let now = Utc::now();
        return Ok(Value::String(now.format("%H:%M:%S%.6f+00:00").to_string()));
    }

    match args.first() {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 time or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return time_from_json_map(&json);
                }
            }
            // Parse ISO 8601 time string
            parse_cypher_time_string(s)
        }
        _ => Ok(Value::Null),
    }
}

/// localdatetime() - Returns current local datetime or constructs from string/map.
///
/// Similar to datetime() but without timezone.
fn cypher_localdatetime(args: &[Value]) -> OperatorResult<Value> {
    use chrono::Utc;

    if args.is_empty() {
        // No arguments: return current datetime without timezone
        let now = Utc::now();
        return Ok(Value::String(now.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()));
    }

    match args.first() {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 datetime or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return localdatetime_from_json_map(&json);
                }
            }
            // Parse ISO 8601 datetime string (stripping timezone if present)
            parse_cypher_localdatetime_string(s)
        }
        _ => Ok(Value::Null),
    }
}

/// localtime() - Returns current local time or constructs from string/map.
///
/// Similar to time() but without timezone.
fn cypher_localtime(args: &[Value]) -> OperatorResult<Value> {
    use chrono::Utc;

    if args.is_empty() {
        // No arguments: return current time without timezone
        let now = Utc::now();
        return Ok(Value::String(now.format("%H:%M:%S%.6f").to_string()));
    }

    match args.first() {
        Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 time or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return localtime_from_json_map(&json);
                }
            }
            // Parse ISO 8601 time string (stripping timezone if present)
            parse_cypher_localtime_string(s)
        }
        _ => Ok(Value::Null),
    }
}

/// duration() - Creates a duration from ISO 8601 string or map.
///
/// Formats:
/// - duration('P1Y2M3D') - ISO 8601 duration (years, months, days)
/// - duration('PT4H5M6S') - ISO 8601 duration (hours, minutes, seconds)
/// - duration('P1Y2M3DT4H5M6S') - combined
/// - duration({years: 1, months: 2, days: 3, hours: 4, minutes: 5, seconds: 6}) - map
fn cypher_duration(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Null) | None => Ok(Value::Null),
        Some(Value::String(s)) => {
            // Try to parse as ISO 8601 duration or as JSON map
            if s.starts_with('{') {
                // Try parsing as JSON map
                if let Some(json) = parse_json(s) {
                    return duration_from_json_map(&json);
                }
            }
            // Parse ISO 8601 duration string
            parse_iso8601_duration(s)
        }
        _ => Ok(Value::Null),
    }
}

/// datetime.truncate(unit, datetime) - Truncates datetime to specified unit.
///
/// Units: millennium, century, decade, year, quarter, month, week, day, hour, minute, second, millisecond, microsecond
fn cypher_datetime_truncate(args: &[Value]) -> OperatorResult<Value> {
    match args {
        [Value::String(unit), Value::String(datetime_str)] => {
            truncate_cypher_datetime(unit, datetime_str)
        }
        [Value::String(unit), Value::String(datetime_str), Value::String(_timezone)] => {
            // Optionally handle timezone parameter
            truncate_cypher_datetime(unit, datetime_str)
        }
        [Value::Null, _] | [_, Value::Null] => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

// ========== Cypher Temporal Helper Functions ==========

/// Parses an ISO 8601 datetime string and returns the Cypher datetime format.
fn parse_cypher_datetime_string(s: &str) -> OperatorResult<Value> {
    use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};

    // Try parsing with timezone first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Ok(Value::String(dt.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string()));
    }

    // Try parsing ISO 8601 formats
    let formats = [
        "%Y-%m-%dT%H:%M:%S%.f%:z",
        "%Y-%m-%dT%H:%M:%S%:z",
        "%Y-%m-%dT%H:%M:%S%.fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ];

    for fmt in &formats {
        // Try with timezone
        if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
            return Ok(Value::String(dt.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string()));
        }
        // Try as naive datetime
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            // Assume UTC if no timezone
            let with_tz = Utc.from_utc_datetime(&dt);
            return Ok(Value::String(with_tz.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()));
        }
    }

    // Try date-only
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap_or_default();
        let with_tz = Utc.from_utc_datetime(&dt);
        return Ok(Value::String(with_tz.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()));
    }

    Ok(Value::Null)
}

/// Parses an ISO 8601 date string and returns the Cypher date format.
fn parse_cypher_date_string(s: &str) -> OperatorResult<Value> {
    use chrono::NaiveDate;

    // Try parsing date
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(Value::String(date.format("%Y-%m-%d").to_string()));
    }

    // Try parsing datetime and extract date
    if let Ok(dt) = parse_naive_datetime(s) {
        return Ok(Value::String(dt.format("%Y-%m-%d").to_string()));
    }

    Ok(Value::Null)
}

/// Parses an ISO 8601 time string with timezone.
fn parse_cypher_time_string(s: &str) -> OperatorResult<Value> {
    use chrono::NaiveTime;

    // Parse time formats
    let formats = [
        "%H:%M:%S%.f%:z",
        "%H:%M:%S%:z",
        "%H:%M:%S%.fZ",
        "%H:%M:%SZ",
        "%H:%M:%S%.f",
        "%H:%M:%S",
        "%H:%M",
    ];

    for fmt in &formats {
        if let Ok(time) = NaiveTime::parse_from_str(s, fmt) {
            // Return with UTC timezone if none specified
            return Ok(Value::String(format!("{}+00:00", time.format("%H:%M:%S%.6f"))));
        }
    }

    Ok(Value::Null)
}

/// Parses an ISO 8601 datetime string without timezone.
fn parse_cypher_localdatetime_string(s: &str) -> OperatorResult<Value> {
    use chrono::NaiveDateTime;

    let formats = [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ];

    // Strip timezone if present (e.g., "+00:00", "-05:00", "Z")
    let s = strip_timezone_suffix(s);

    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(Value::String(dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()));
        }
    }

    // Try date-only
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let dt = date.and_hms_opt(0, 0, 0).unwrap_or_default();
        return Ok(Value::String(dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()));
    }

    Ok(Value::Null)
}

/// Parses an ISO 8601 time string without timezone.
fn parse_cypher_localtime_string(s: &str) -> OperatorResult<Value> {
    use chrono::NaiveTime;

    // Strip timezone if present (e.g., "+00:00", "-05:00", "Z")
    let s = strip_timezone_suffix(s);

    let formats = ["%H:%M:%S%.f", "%H:%M:%S", "%H:%M"];

    for fmt in &formats {
        if let Ok(time) = NaiveTime::parse_from_str(s, fmt) {
            return Ok(Value::String(time.format("%H:%M:%S%.6f").to_string()));
        }
    }

    Ok(Value::Null)
}

/// Strips timezone suffix from a datetime or time string.
fn strip_timezone_suffix(s: &str) -> &str {
    let s = s.trim();

    // Handle Z suffix
    if s.ends_with('Z') {
        return &s[..s.len() - 1];
    }

    // Look for +HH:MM or -HH:MM pattern at the end
    // Also handle +HHMM or +HH
    if let Some(pos) = s.rfind(['+', '-']) {
        // Check if the part after +/- looks like a timezone offset
        let after = &s[pos + 1..];
        // Timezone offsets are like: "00:00", "05:30", "0000", "05"
        let is_timezone = after.len() >= 2
            && after.chars().next().is_some_and(|c| c.is_ascii_digit())
            && after.chars().all(|c| c.is_ascii_digit() || c == ':');

        if is_timezone {
            return &s[..pos];
        }
    }

    s
}

/// Constructs a datetime from a JSON map.
fn datetime_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let year = obj.get("year").and_then(|v| v.as_i64()).unwrap_or(1970) as i32;
    let month = obj.get("month").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let day = obj.get("day").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let hour = obj.get("hour").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let minute = obj.get("minute").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let second = obj.get("second").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let nanosecond = obj.get("nanosecond").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let date = match NaiveDate::from_ymd_opt(year, month, day) {
        Some(d) => d,
        None => return Ok(Value::Null),
    };

    let time = match NaiveTime::from_hms_nano_opt(hour, minute, second, nanosecond) {
        Some(t) => t,
        None => return Ok(Value::Null),
    };

    let dt = NaiveDateTime::new(date, time);
    let with_tz = Utc.from_utc_datetime(&dt);
    Ok(Value::String(with_tz.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()))
}

/// Constructs a date from a JSON map.
fn date_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use chrono::NaiveDate;

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let year = obj.get("year").and_then(|v| v.as_i64()).unwrap_or(1970) as i32;
    let month = obj.get("month").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let day = obj.get("day").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

    match NaiveDate::from_ymd_opt(year, month, day) {
        Some(d) => Ok(Value::String(d.format("%Y-%m-%d").to_string())),
        None => Ok(Value::Null),
    }
}

/// Constructs a time from a JSON map.
fn time_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use chrono::NaiveTime;

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let hour = obj.get("hour").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let minute = obj.get("minute").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let second = obj.get("second").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let nanosecond = obj.get("nanosecond").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    match NaiveTime::from_hms_nano_opt(hour, minute, second, nanosecond) {
        Some(t) => Ok(Value::String(format!("{}+00:00", t.format("%H:%M:%S%.6f")))),
        None => Ok(Value::Null),
    }
}

/// Constructs a local datetime from a JSON map.
fn localdatetime_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let year = obj.get("year").and_then(|v| v.as_i64()).unwrap_or(1970) as i32;
    let month = obj.get("month").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let day = obj.get("day").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let hour = obj.get("hour").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let minute = obj.get("minute").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let second = obj.get("second").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let nanosecond = obj.get("nanosecond").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let date = match NaiveDate::from_ymd_opt(year, month, day) {
        Some(d) => d,
        None => return Ok(Value::Null),
    };

    let time = match NaiveTime::from_hms_nano_opt(hour, minute, second, nanosecond) {
        Some(t) => t,
        None => return Ok(Value::Null),
    };

    let dt = NaiveDateTime::new(date, time);
    Ok(Value::String(dt.format("%Y-%m-%dT%H:%M:%S%.6f").to_string()))
}

/// Constructs a local time from a JSON map.
fn localtime_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use chrono::NaiveTime;

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let hour = obj.get("hour").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let minute = obj.get("minute").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let second = obj.get("second").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let nanosecond = obj.get("nanosecond").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    match NaiveTime::from_hms_nano_opt(hour, minute, second, nanosecond) {
        Some(t) => Ok(Value::String(t.format("%H:%M:%S%.6f").to_string())),
        None => Ok(Value::Null),
    }
}

/// Constructs a duration from a JSON map.
fn duration_from_json_map(json: &serde_json::Value) -> OperatorResult<Value> {
    use std::fmt::Write;

    let obj = match json {
        serde_json::Value::Object(obj) => obj,
        _ => return Ok(Value::Null),
    };

    let years = obj.get("years").and_then(|v| v.as_i64()).unwrap_or(0);
    let months = obj.get("months").and_then(|v| v.as_i64()).unwrap_or(0);
    let weeks = obj.get("weeks").and_then(|v| v.as_i64()).unwrap_or(0);
    let days = obj.get("days").and_then(|v| v.as_i64()).unwrap_or(0);
    let hours = obj.get("hours").and_then(|v| v.as_i64()).unwrap_or(0);
    let minutes = obj.get("minutes").and_then(|v| v.as_i64()).unwrap_or(0);
    let seconds = obj.get("seconds").and_then(|v| v.as_i64()).unwrap_or(0);
    let nanoseconds = obj.get("nanoseconds").and_then(|v| v.as_i64()).unwrap_or(0);

    // Total days including weeks
    let total_days = days + weeks * 7;

    // Build ISO 8601 duration string
    let mut result = String::from("P");

    if years != 0 {
        let _ = write!(result, "{years}Y");
    }
    if months != 0 {
        let _ = write!(result, "{months}M");
    }
    if total_days != 0 {
        let _ = write!(result, "{total_days}D");
    }

    let has_time = hours != 0 || minutes != 0 || seconds != 0 || nanoseconds != 0;
    if has_time {
        result.push('T');
        if hours != 0 {
            let _ = write!(result, "{hours}H");
        }
        if minutes != 0 {
            let _ = write!(result, "{minutes}M");
        }
        if seconds != 0 || nanoseconds != 0 {
            if nanoseconds != 0 {
                let frac_seconds = seconds as f64 + nanoseconds as f64 / 1_000_000_000.0;
                let _ = write!(result, "{frac_seconds}S");
            } else {
                let _ = write!(result, "{seconds}S");
            }
        }
    }

    // If no components, use P0D
    if result == "P" {
        result.push_str("0D");
    }

    Ok(Value::String(result))
}

/// Parses an ISO 8601 duration string.
///
/// Format: P[n]Y[n]M[n]DT[n]H[n]M[n]S
/// Examples:
/// - P1Y - 1 year
/// - P1M - 1 month
/// - P1D - 1 day
/// - PT1H - 1 hour
/// - PT1M - 1 minute
/// - PT1S - 1 second
/// - P1Y2M3DT4H5M6S - combined
fn parse_iso8601_duration(s: &str) -> OperatorResult<Value> {
    use std::fmt::Write;

    let s = s.trim();

    // Check if it starts with P
    if !s.starts_with('P') && !s.starts_with('p') {
        return Ok(Value::Null);
    }

    let s = &s[1..]; // Skip 'P'

    // Split into date and time parts
    let (date_part, time_part) = if let Some(t_pos) = s.find(['T', 't']) {
        (&s[..t_pos], Some(&s[t_pos + 1..]))
    } else {
        (s, None)
    };

    // Parse date components
    let (years, months, weeks, days) = parse_duration_date_part(date_part);

    // Parse time components
    let (hours, minutes, seconds) = if let Some(time_str) = time_part {
        parse_duration_time_part(time_str)
    } else {
        (0, 0, 0.0)
    };

    // Build ISO 8601 duration string (normalized)
    let total_days = days + weeks * 7;
    let mut result = String::from("P");

    if years != 0 {
        let _ = write!(result, "{years}Y");
    }
    if months != 0 {
        let _ = write!(result, "{months}M");
    }
    if total_days != 0 {
        let _ = write!(result, "{total_days}D");
    }

    let has_time = hours != 0 || minutes != 0 || seconds != 0.0;
    if has_time {
        result.push('T');
        if hours != 0 {
            let _ = write!(result, "{hours}H");
        }
        if minutes != 0 {
            let _ = write!(result, "{minutes}M");
        }
        if seconds != 0.0 {
            if seconds.fract() == 0.0 {
                let secs_i64 = seconds as i64;
                let _ = write!(result, "{secs_i64}S");
            } else {
                let _ = write!(result, "{seconds}S");
            }
        }
    }

    // If no components, use P0D
    if result == "P" {
        result.push_str("0D");
    }

    Ok(Value::String(result))
}

/// Parses the date part of an ISO 8601 duration (years, months, weeks, days).
fn parse_duration_date_part(s: &str) -> (i64, i64, i64, i64) {
    let mut years = 0i64;
    let mut months = 0i64;
    let mut weeks = 0i64;
    let mut days = 0i64;

    let mut current_num = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            current_num.push(c);
        } else {
            let num: i64 = current_num.parse().unwrap_or(0);
            current_num.clear();
            match c {
                'Y' | 'y' => years = num,
                'M' | 'm' => months = num, // In date part, M is months
                'W' | 'w' => weeks = num,
                'D' | 'd' => days = num,
                _ => {}
            }
        }
    }

    (years, months, weeks, days)
}

/// Parses the time part of an ISO 8601 duration (hours, minutes, seconds).
fn parse_duration_time_part(s: &str) -> (i64, i64, f64) {
    let mut hours = 0i64;
    let mut minutes = 0i64;
    let mut seconds = 0.0f64;

    let mut current_num = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' {
            current_num.push(c);
        } else {
            match c {
                'H' | 'h' => {
                    hours = current_num.parse().unwrap_or(0);
                    current_num.clear();
                }
                'M' | 'm' => {
                    // In time part, M is minutes
                    minutes = current_num.parse().unwrap_or(0);
                    current_num.clear();
                }
                'S' | 's' => {
                    seconds = current_num.parse().unwrap_or(0.0);
                    current_num.clear();
                }
                _ => {}
            }
        }
    }

    (hours, minutes, seconds)
}

/// Truncates a datetime to the specified unit.
fn truncate_cypher_datetime(unit: &str, datetime_str: &str) -> OperatorResult<Value> {
    use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Timelike, Utc};

    // Parse the datetime
    let dt = parse_naive_datetime(datetime_str)?;

    let truncated = match unit.to_lowercase().as_str() {
        "millennium" => {
            let year = (dt.year() / 1000) * 1000;
            NaiveDate::from_ymd_opt(year, 1, 1).map(|d| NaiveDateTime::new(d, NaiveTime::default()))
        }
        "century" => {
            let year = (dt.year() / 100) * 100;
            NaiveDate::from_ymd_opt(year, 1, 1).map(|d| NaiveDateTime::new(d, NaiveTime::default()))
        }
        "decade" => {
            let year = (dt.year() / 10) * 10;
            NaiveDate::from_ymd_opt(year, 1, 1).map(|d| NaiveDateTime::new(d, NaiveTime::default()))
        }
        "year" => NaiveDate::from_ymd_opt(dt.year(), 1, 1)
            .map(|d| NaiveDateTime::new(d, NaiveTime::default())),
        "quarter" => {
            let quarter_month = ((dt.month() - 1) / 3) * 3 + 1;
            NaiveDate::from_ymd_opt(dt.year(), quarter_month, 1)
                .map(|d| NaiveDateTime::new(d, NaiveTime::default()))
        }
        "month" => NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1)
            .map(|d| NaiveDateTime::new(d, NaiveTime::default())),
        "week" => {
            // Truncate to Monday of the current week
            let weekday = dt.weekday().num_days_from_monday();
            let monday = dt.date() - chrono::Duration::days(weekday as i64);
            Some(NaiveDateTime::new(monday, NaiveTime::default()))
        }
        "day" => Some(NaiveDateTime::new(dt.date(), NaiveTime::default())),
        "hour" => NaiveTime::from_hms_opt(dt.time().hour(), 0, 0)
            .map(|t| NaiveDateTime::new(dt.date(), t)),
        "minute" => NaiveTime::from_hms_opt(dt.time().hour(), dt.time().minute(), 0)
            .map(|t| NaiveDateTime::new(dt.date(), t)),
        "second" => {
            NaiveTime::from_hms_opt(dt.time().hour(), dt.time().minute(), dt.time().second())
                .map(|t| NaiveDateTime::new(dt.date(), t))
        }
        "millisecond" => {
            let millis = dt.time().nanosecond() / 1_000_000;
            NaiveTime::from_hms_nano_opt(
                dt.time().hour(),
                dt.time().minute(),
                dt.time().second(),
                millis * 1_000_000,
            )
            .map(|t| NaiveDateTime::new(dt.date(), t))
        }
        "microsecond" => {
            let micros = dt.time().nanosecond() / 1_000;
            NaiveTime::from_hms_nano_opt(
                dt.time().hour(),
                dt.time().minute(),
                dt.time().second(),
                micros * 1_000,
            )
            .map(|t| NaiveDateTime::new(dt.date(), t))
        }
        _ => return Ok(Value::Null),
    };

    match truncated {
        Some(dt) => {
            let with_tz = Utc.from_utc_datetime(&dt);
            Ok(Value::String(with_tz.format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()))
        }
        None => Ok(Value::Null),
    }
}

// ========== Cypher Entity Functions ==========

/// TYPE(relationship) - Returns the type of a relationship.
///
/// In Cypher, relationships have a type (e.g., "KNOWS", "FRIENDS_WITH").
/// This function extracts that type from the relationship entity.
///
/// Currently supports:
/// - Maps with an "_edge_type" or "_type" key (from internal representation)
/// - String values (returns as-is, assumed to already be the type)
fn cypher_type(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::String(s)) => {
            // If it's a JSON object string, try to extract _edge_type or _type
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    if let Some(serde_json::Value::String(t)) = obj.get("_edge_type") {
                        return Ok(Value::String(t.clone()));
                    }
                    if let Some(serde_json::Value::String(t)) = obj.get("_type") {
                        return Ok(Value::String(t.clone()));
                    }
                }
            }
            // If it's a plain string, it might be the type itself
            Ok(Value::String(s.clone()))
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// LABELS(node) - Returns a list of labels for a node.
///
/// In Cypher, nodes can have one or more labels (e.g., "Person", "Employee").
/// This function extracts those labels from the node entity.
///
/// Currently supports:
/// - Maps with a "_labels" key containing an array
/// - Arrays (returns as-is, assumed to be labels)
fn cypher_labels(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Array(labels)) => {
            // Already an array, return as-is
            Ok(Value::Array(labels.clone()))
        }
        Some(Value::String(s)) => {
            // If it's a JSON object string, try to extract _labels
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    if let Some(serde_json::Value::Array(labels)) = obj.get("_labels") {
                        let label_values: Vec<Value> = labels
                            .iter()
                            .filter_map(|l| {
                                if let serde_json::Value::String(s) = l {
                                    Some(Value::String(s.clone()))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        return Ok(Value::Array(label_values));
                    }
                }
            }
            // If it's just a string, wrap it as a single label
            Ok(Value::Array(vec![Value::String(s.clone())]))
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// ID(entity) - Returns the internal ID of a node or relationship.
///
/// In Cypher, every node and relationship has a unique internal ID.
/// This function extracts that ID from the entity.
///
/// Currently supports:
/// - Integer values (returns as-is, assumed to be the ID)
/// - Maps with an "_id" key
fn cypher_id(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Int(id)) => {
            // Already an integer, return as-is
            Ok(Value::Int(*id))
        }
        Some(Value::String(s)) => {
            // If it's a JSON object string, try to extract _id
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    if let Some(serde_json::Value::Number(n)) = obj.get("_id") {
                        if let Some(id) = n.as_i64() {
                            return Ok(Value::Int(id));
                        }
                    }
                }
            }
            // Try parsing the string as an integer
            if let Ok(id) = s.parse::<i64>() {
                return Ok(Value::Int(id));
            }
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// PROPERTIES(entity) - Returns a map of all properties of a node or relationship.
///
/// In Cypher, nodes and relationships can have properties (key-value pairs).
/// This function returns all properties as a map, excluding internal keys
/// (those starting with "_").
///
/// Currently supports:
/// - Maps/JSON objects - returns the properties as a JSON string
fn cypher_properties(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::String(s)) => {
            // If it's a JSON object string, filter out internal keys
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    // Filter out internal keys (those starting with "_")
                    let properties: serde_json::Map<String, serde_json::Value> =
                        obj.into_iter().filter(|(k, _)| !k.starts_with('_')).collect();
                    let result = serde_json::Value::Object(properties);
                    return Ok(Value::String(serde_json::to_string(&result).unwrap_or_default()));
                }
            }
            // If it's not parseable as JSON, return empty object
            Ok(Value::String("{}".to_string()))
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => {
            // For other types, return empty object
            Ok(Value::String("{}".to_string()))
        }
    }
}

/// KEYS(map_or_entity) - Returns a list of property keys from a map or entity.
///
/// In Cypher, this function returns the keys of a map or the property keys
/// of a node/relationship. It excludes internal keys (those starting with "_").
///
/// Currently supports:
/// - Maps/JSON objects - returns the keys as a list
fn cypher_keys(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::String(s)) => {
            // If it's a JSON object string, extract keys
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    // Filter out internal keys and collect remaining keys
                    let keys: Vec<Value> = obj
                        .keys()
                        .filter(|k| !k.starts_with('_'))
                        .map(|k| Value::String(k.clone()))
                        .collect();
                    return Ok(Value::Array(keys));
                }
            }
            // If it's not parseable as JSON, return empty list
            Ok(Value::Array(vec![]))
        }
        Some(Value::Array(_)) => {
            // Arrays don't have keys in Cypher semantics
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => {
            // For other types, return empty list
            Ok(Value::Array(vec![]))
        }
    }
}

// ========== Cypher Path Functions ==========

/// nodes(path) - Returns a list of all nodes in a path.
///
/// In Cypher, paths consist of alternating nodes and relationships.
/// This function extracts all the nodes from a path.
///
/// Supports:
/// - Arrays (assumed to be node IDs or node objects directly)
/// - JSON objects with `_nodes` key containing an array of node IDs
/// - JSON objects with `path_nodes` key (from internal path representation)
fn cypher_nodes(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Array(nodes)) => {
            // Already an array, return as-is
            Ok(Value::Array(nodes.clone()))
        }
        Some(Value::String(s)) => {
            // Try to parse as JSON object with _nodes or path_nodes key
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(ref obj) = json {
                    // Try _nodes first
                    if let Some(serde_json::Value::Array(nodes)) = obj.get("_nodes") {
                        let node_values: Vec<Value> =
                            nodes.iter().map(json_value_to_value).collect();
                        return Ok(Value::Array(node_values));
                    }
                    // Try path_nodes (internal representation)
                    if let Some(serde_json::Value::Array(nodes)) = obj.get("path_nodes") {
                        let node_values: Vec<Value> =
                            nodes.iter().map(json_value_to_value).collect();
                        return Ok(Value::Array(node_values));
                    }
                }
                // If it's a JSON array directly
                if let serde_json::Value::Array(ref nodes) = json {
                    let node_values: Vec<Value> = nodes.iter().map(json_value_to_value).collect();
                    return Ok(Value::Array(node_values));
                }
            }
            // Not a valid path representation
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// relationships(path) - Returns a list of all relationships in a path.
///
/// In Cypher, paths consist of alternating nodes and relationships.
/// This function extracts all the relationships/edges from a path.
///
/// Supports:
/// - Arrays (assumed to be edge IDs or edge objects directly)
/// - JSON objects with `_edges`, `_relationships`, or `path_edges` key
fn cypher_relationships(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Array(rels)) => {
            // Already an array, return as-is
            Ok(Value::Array(rels.clone()))
        }
        Some(Value::String(s)) => {
            // Try to parse as JSON object with _edges, _relationships, or path_edges key
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(ref obj) = json {
                    // Try _edges first
                    if let Some(serde_json::Value::Array(edges)) = obj.get("_edges") {
                        let edge_values: Vec<Value> =
                            edges.iter().map(json_value_to_value).collect();
                        return Ok(Value::Array(edge_values));
                    }
                    // Try _relationships
                    if let Some(serde_json::Value::Array(edges)) = obj.get("_relationships") {
                        let edge_values: Vec<Value> =
                            edges.iter().map(json_value_to_value).collect();
                        return Ok(Value::Array(edge_values));
                    }
                    // Try path_edges (internal representation)
                    if let Some(serde_json::Value::Array(edges)) = obj.get("path_edges") {
                        let edge_values: Vec<Value> =
                            edges.iter().map(json_value_to_value).collect();
                        return Ok(Value::Array(edge_values));
                    }
                }
                // If it's a JSON array directly
                if let serde_json::Value::Array(ref edges) = json {
                    let edge_values: Vec<Value> = edges.iter().map(json_value_to_value).collect();
                    return Ok(Value::Array(edge_values));
                }
            }
            // Not a valid path representation
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// startNode(relationship) - Returns the start node of a relationship.
///
/// In Cypher, relationships have a start (source) node and an end (target) node.
/// This function extracts the start node ID from a relationship.
///
/// Supports:
/// - JSON objects with `_source`, `_start`, `source`, or `start` field
fn cypher_start_node(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::String(s)) => {
            // Try to parse as JSON object with source node field
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    // Try _source first (internal representation)
                    if let Some(source) = obj.get("_source") {
                        return Ok(json_value_to_value(source));
                    }
                    // Try _start
                    if let Some(start) = obj.get("_start") {
                        return Ok(json_value_to_value(start));
                    }
                    // Try source (user-friendly name)
                    if let Some(source) = obj.get("source") {
                        return Ok(json_value_to_value(source));
                    }
                    // Try start
                    if let Some(start) = obj.get("start") {
                        return Ok(json_value_to_value(start));
                    }
                }
            }
            // Not a valid relationship representation
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// endNode(relationship) - Returns the end node of a relationship.
///
/// In Cypher, relationships have a start (source) node and an end (target) node.
/// This function extracts the end node ID from a relationship.
///
/// Supports:
/// - JSON objects with `_target`, `_end`, `target`, or `end` field
fn cypher_end_node(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::String(s)) => {
            // Try to parse as JSON object with target node field
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(obj) = json {
                    // Try _target first (internal representation)
                    if let Some(target) = obj.get("_target") {
                        return Ok(json_value_to_value(target));
                    }
                    // Try _end
                    if let Some(end) = obj.get("_end") {
                        return Ok(json_value_to_value(end));
                    }
                    // Try target (user-friendly name)
                    if let Some(target) = obj.get("target") {
                        return Ok(json_value_to_value(target));
                    }
                    // Try end
                    if let Some(end) = obj.get("end") {
                        return Ok(json_value_to_value(end));
                    }
                }
            }
            // Not a valid relationship representation
            Ok(Value::Null)
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// length(path) - Returns the length of a path (number of relationships).
///
/// In Cypher, the length of a path is the number of relationships it contains.
/// This is always one less than the number of nodes in the path.
///
/// Supports:
/// - Arrays (returns the length of the array, assuming it's edges)
/// - JSON objects with `_edges`, `_relationships`, or `path_edges` key
/// - For strings (non-JSON), this delegates to string LENGTH behavior
fn cypher_path_length(args: &[Value]) -> OperatorResult<Value> {
    match args.first() {
        Some(Value::Array(arr)) => {
            // If it's an array, return its length
            Ok(Value::Int(arr.len() as i64))
        }
        Some(Value::String(s)) => {
            // Try to parse as JSON object with edges/relationships
            if let Some(json) = parse_json(s) {
                if let serde_json::Value::Object(ref obj) = json {
                    // Try _edges first
                    if let Some(serde_json::Value::Array(edges)) = obj.get("_edges") {
                        return Ok(Value::Int(edges.len() as i64));
                    }
                    // Try _relationships
                    if let Some(serde_json::Value::Array(edges)) = obj.get("_relationships") {
                        return Ok(Value::Int(edges.len() as i64));
                    }
                    // Try path_edges (internal representation)
                    if let Some(serde_json::Value::Array(edges)) = obj.get("path_edges") {
                        return Ok(Value::Int(edges.len() as i64));
                    }
                    // If we have _nodes, length is nodes - 1
                    if let Some(serde_json::Value::Array(nodes)) = obj.get("_nodes") {
                        if !nodes.is_empty() {
                            return Ok(Value::Int((nodes.len() - 1) as i64));
                        }
                        return Ok(Value::Int(0));
                    }
                    // If we have path_nodes, length is nodes - 1
                    if let Some(serde_json::Value::Array(nodes)) = obj.get("path_nodes") {
                        if !nodes.is_empty() {
                            return Ok(Value::Int((nodes.len() - 1) as i64));
                        }
                        return Ok(Value::Int(0));
                    }
                }
                // If it's a JSON array directly (assumed to be edges)
                if let serde_json::Value::Array(ref arr) = json {
                    return Ok(Value::Int(arr.len() as i64));
                }
            }
            // For non-JSON strings, return string length (same as LENGTH function)
            Ok(Value::Int(s.chars().count() as i64))
        }
        Some(Value::Null) | None => Ok(Value::Null),
        _ => Ok(Value::Null),
    }
}

/// Helper to convert serde_json::Value to manifoldb_core::Value
fn json_value_to_value(json: &serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Value::Array(arr.iter().map(json_value_to_value).collect())
        }
        serde_json::Value::Object(_) => {
            // Return the JSON object as a string
            Value::String(json.to_string())
        }
    }
}

/// Compares two values for equality.
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            ((*a as f64) - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Vector(a), Value::Vector(b)) => a == b,
        _ => false,
    }
}

/// Compares two values, returning -1, 0, or 1.
fn compare_values(left: &Value, right: &Value) -> i32 {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b) as i32,
        (Value::Float(a), Value::Float(b)) => {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        }
        (Value::Int(a), Value::Float(b)) => {
            let a = *a as f64;
            if a < *b {
                -1
            } else if a > *b {
                1
            } else {
                0
            }
        }
        (Value::Float(a), Value::Int(b)) => {
            let b = *b as f64;
            if *a < b {
                -1
            } else if *a > b {
                1
            } else {
                0
            }
        }
        (Value::String(a), Value::String(b)) => a.cmp(b) as i32,
        _ => 0,
    }
}

/// Converts a value to boolean.
fn value_to_bool(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::String(s) => !s.is_empty(),
        _ => false,
    }
}

/// Converts a value to string.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.clone(),
        Value::Vector(v) => format!("{v:?}"),
        Value::SparseVector(v) => format!("{v:?}"),
        Value::MultiVector(v) => format!("{v:?}"),
        Value::Bytes(b) => format!("{b:?}"),
        Value::Array(a) => format!("{a:?}"),
    }
}

/// Evaluates arithmetic operations.
fn evaluate_arithmetic<F1, F2>(
    left: &Value,
    right: &Value,
    int_op: F1,
    float_op: F2,
) -> OperatorResult<Value>
where
    F1: Fn(i64, i64) -> i64,
    F2: Fn(f64, f64) -> f64,
{
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
        _ => Ok(Value::Null),
    }
}

/// SQL LIKE pattern matching.
///
/// Supports:
/// - `%` matches any sequence of characters (including empty)
/// - `_` matches exactly one character
fn like_match(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();

    let s_len = s_chars.len();
    let p_len = p_chars.len();

    // dp[i][j] = true if s[0..i] matches pattern[0..j]
    let mut dp = vec![vec![false; p_len + 1]; s_len + 1];

    // Empty pattern matches empty string
    dp[0][0] = true;

    // Handle patterns starting with %
    for j in 1..=p_len {
        if p_chars[j - 1] == '%' {
            dp[0][j] = dp[0][j - 1];
        } else {
            break;
        }
    }

    for i in 1..=s_len {
        for j in 1..=p_len {
            let p_char = p_chars[j - 1];

            if p_char == '%' {
                // % matches zero or more characters
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if p_char == '_' {
                // _ matches exactly one character
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                // Regular character - must match exactly
                dp[i][j] = dp[i - 1][j - 1] && s_chars[i - 1] == p_char;
            }
        }
    }

    dp[s_len][p_len]
}

/// Euclidean distance between vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Cosine distance between vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return f32::MAX;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Inner product of vectors.
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// MaxSim score between multi-vectors.
///
/// Computes the sum of maximum similarities between each query token and all document tokens.
fn maxsim_score(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }

    // Verify dimensions match
    let dim = query[0].len();
    if doc[0].len() != dim {
        return 0.0;
    }

    let mut total_score = 0.0_f32;

    // For each query token, find the max dot product with any document token
    for q in query {
        let mut max_sim = f32::NEG_INFINITY;
        for d in doc {
            let sim: f32 = q.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim.is_finite() {
            total_score += max_sim;
        }
    }

    total_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use crate::plan::logical::ScalarFunction;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string(), "age".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice"), Value::Int(30)],
                vec![Value::Int(2), Value::from("Bob"), Value::Int(25)],
                vec![Value::Int(3), Value::from("Carol"), Value::Int(35)],
            ],
        ))
    }

    #[test]
    fn filter_basic() {
        // Filter: age > 28
        let predicate = LogicalExpr::column("age").gt(LogicalExpr::integer(28));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        // Should return Alice (30) and Carol (35)
        let row1 = filter.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("name"), Some(&Value::from("Alice")));

        let row2 = filter.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("name"), Some(&Value::from("Carol")));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn filter_equality() {
        // Filter: id = 2
        let predicate = LogicalExpr::column("id").eq(LogicalExpr::integer(2));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        let row = filter.next().unwrap().unwrap();
        assert_eq!(row.get_by_name("name"), Some(&Value::from("Bob")));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn filter_and() {
        // Filter: age >= 25 AND age <= 30
        let predicate = LogicalExpr::column("age")
            .gt_eq(LogicalExpr::integer(25))
            .and(LogicalExpr::column("age").lt_eq(LogicalExpr::integer(30)));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        let row1 = filter.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("age"), Some(&Value::Int(30)));

        let row2 = filter.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("age"), Some(&Value::Int(25)));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn evaluate_arithmetic_ops() {
        let schema = Arc::new(Schema::new(vec!["x".to_string()]));
        let row = Row::new(schema, vec![Value::Int(10)]);

        // x + 5
        let expr = LogicalExpr::column("x").add(LogicalExpr::integer(5));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Int(15));

        // x * 2
        let expr = LogicalExpr::column("x").mul(LogicalExpr::integer(2));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Int(20));
    }

    #[test]
    fn evaluate_in_list() {
        let schema = Arc::new(Schema::new(vec!["status".to_string()]));
        let row = Row::new(schema, vec![Value::from("active")]);

        let expr = LogicalExpr::column("status")
            .in_list(vec![LogicalExpr::string("active"), LogicalExpr::string("pending")], false);
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(true));

        let expr =
            LogicalExpr::column("status").in_list(vec![LogicalExpr::string("inactive")], false);
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(false));
    }

    #[test]
    fn evaluate_vector_distance() {
        let schema = Arc::new(Schema::new(vec!["v".to_string()]));
        let row = Row::new(schema, vec![Value::Vector(vec![1.0, 0.0, 0.0])]);

        let expr =
            LogicalExpr::column("v").euclidean_distance(LogicalExpr::vector(vec![0.0, 0.0, 0.0]));
        let result = evaluate_expr(&expr, &row).unwrap();

        if let Value::Float(d) = result {
            assert!((d - 1.0).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn evaluate_maxsim_operator() {
        // Test MaxSim operator for ColBERT-style multi-vector scoring
        // Query: 2 tokens, Document: 3 tokens, dimension: 2
        let query_tokens = vec![
            vec![1.0_f32, 0.0], // Query token 1
            vec![0.0_f32, 1.0], // Query token 2
        ];
        let doc_tokens = vec![
            vec![1.0_f32, 0.0], // Doc token 1 (matches query token 1 perfectly)
            vec![0.5_f32, 0.5], // Doc token 2
            vec![0.0_f32, 0.8], // Doc token 3 (close to query token 2)
        ];

        let schema = Arc::new(Schema::new(vec!["doc".to_string(), "query".to_string()]));
        let row = Row::new(
            schema,
            vec![Value::MultiVector(doc_tokens), Value::MultiVector(query_tokens)],
        );

        // Create MaxSim expression: query <##> doc
        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("query")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("doc")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();

        // Expected MaxSim score:
        // Query token 1 [1,0]: max dot products with doc tokens = max(1.0, 0.5, 0.0) = 1.0
        // Query token 2 [0,1]: max dot products with doc tokens = max(0.0, 0.5, 0.8) = 0.8
        // Total MaxSim = 1.0 + 0.8 = 1.8
        if let Value::Float(score) = result {
            assert!((score - 1.8).abs() < 0.001, "Expected MaxSim score ~1.8, got {score}");
        } else {
            panic!("Expected float result from MaxSim");
        }
    }

    #[test]
    fn evaluate_maxsim_identical_vectors() {
        // When query and document are identical, MaxSim should equal num_tokens
        let tokens = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];

        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));
        let row =
            Row::new(schema, vec![Value::MultiVector(tokens.clone()), Value::MultiVector(tokens)]);

        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("a")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("b")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();

        // Each query token matches itself perfectly: 1.0 + 1.0 = 2.0
        if let Value::Float(score) = result {
            assert!(
                (score - 2.0).abs() < 0.001,
                "Expected MaxSim score 2.0 for identical vectors, got {score}"
            );
        } else {
            panic!("Expected float result from MaxSim");
        }
    }

    #[test]
    fn evaluate_maxsim_null_on_type_mismatch() {
        // MaxSim should return Null when operands are not MultiVector
        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));
        let row = Row::new(
            schema,
            vec![
                Value::Vector(vec![1.0, 0.0]), // Dense vector, not multi-vector
                Value::Vector(vec![0.0, 1.0]),
            ],
        );

        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("a")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("b")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        assert!(matches!(result, Value::Null), "Expected Null for type mismatch");
    }

    #[test]
    fn like_match_percent_wildcard() {
        // % matches any sequence of characters
        assert!(like_match("hello", "hello"));
        assert!(like_match("hello", "%"));
        assert!(like_match("hello", "h%"));
        assert!(like_match("hello", "%o"));
        assert!(like_match("hello", "%ell%"));
        assert!(like_match("hello", "h%o"));
        assert!(like_match("hello world", "hello%world"));
        assert!(!like_match("hello", "world%"));
        assert!(!like_match("hello", "%world"));
    }

    #[test]
    fn like_match_underscore_wildcard() {
        // _ matches exactly one character
        assert!(like_match("hello", "h_llo"));
        assert!(like_match("hello", "_ello"));
        assert!(like_match("hello", "hell_"));
        assert!(like_match("hello", "_____"));
        assert!(like_match("ab", "a_"));
        assert!(like_match("ab", "_b"));
        assert!(!like_match("hello", "h_lo")); // _ is one char, not two
        assert!(!like_match("hello", "______")); // too many underscores
        assert!(!like_match("hello", "____")); // too few underscores
    }

    #[test]
    fn like_match_mixed_wildcards() {
        // Mix of % and _
        assert!(like_match("hello", "h_%"));
        assert!(like_match("hello", "%_o"));
        assert!(like_match("hello", "h_ll%"));
        assert!(like_match("hello world", "h%_d"));
        assert!(like_match("abc", "a%_"));
        assert!(like_match("abc", "_%c"));
        assert!(!like_match("a", "a%_")); // needs at least one char after a
    }

    #[test]
    fn like_match_empty_strings() {
        assert!(like_match("", ""));
        assert!(like_match("", "%"));
        assert!(!like_match("", "_"));
        assert!(!like_match("", "a"));
        assert!(!like_match("a", ""));
    }

    #[test]
    fn like_match_special_cases() {
        // Multiple consecutive wildcards
        assert!(like_match("hello", "%%"));
        assert!(like_match("hello", "%%%"));
        assert!(like_match("hello", "%_%"));
        assert!(like_match("h", "%_%"));
        assert!(!like_match("", "%_%")); // needs at least one char

        // Pattern at exact boundaries
        assert!(like_match("test", "test"));
        assert!(!like_match("test", "Test")); // case sensitive
    }

    #[test]
    fn null_comparison_returns_null() {
        // SQL semantics: any comparison with NULL returns NULL
        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));

        // NULL = NULL should return NULL (not true)
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Null]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL = value should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Int(5)]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // value = NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Int(5), Value::Null]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL != NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Null]);
        let expr = LogicalExpr::column("a").not_eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL < value should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Int(5)]);
        let expr = LogicalExpr::column("a").lt(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // value > NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Int(5), Value::Null]);
        let expr = LogicalExpr::column("a").gt(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // Non-null comparison should still work
        let row = Row::new(schema, vec![Value::Int(5), Value::Int(5)]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(true));
    }

    // ========== String Function Tests ==========

    fn eval_fn(func: crate::plan::logical::ScalarFunction, args: Vec<Value>) -> Value {
        evaluate_scalar_function(&func, &args).unwrap()
    }

    #[test]
    fn test_string_position() {
        use crate::plan::logical::ScalarFunction;

        // POSITION('lo' IN 'hello') = 4
        let result =
            eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("lo")]);
        assert_eq!(result, Value::Int(4));

        // Not found returns 0
        let result =
            eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("xyz")]);
        assert_eq!(result, Value::Int(0));

        // Empty substring returns 1
        let result = eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("")]);
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_string_concat_ws() {
        use crate::plan::logical::ScalarFunction;

        // CONCAT_WS(', ', 'a', 'b', 'c') = 'a, b, c'
        let result = eval_fn(
            ScalarFunction::ConcatWs,
            vec![Value::from(", "), Value::from("a"), Value::from("b"), Value::from("c")],
        );
        assert_eq!(result, Value::String("a, b, c".to_string()));

        // NULLs are skipped
        let result = eval_fn(
            ScalarFunction::ConcatWs,
            vec![Value::from("-"), Value::from("a"), Value::Null, Value::from("c")],
        );
        assert_eq!(result, Value::String("a-c".to_string()));
    }

    #[test]
    fn test_string_split_part() {
        use crate::plan::logical::ScalarFunction;

        // SPLIT_PART('a,b,c', ',', 2) = 'b'
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(2)],
        );
        assert_eq!(result, Value::String("b".to_string()));

        // Position 1
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(1)],
        );
        assert_eq!(result, Value::String("a".to_string()));

        // Position beyond parts returns empty string
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(5)],
        );
        assert_eq!(result, Value::String(String::new()));
    }

    #[test]
    fn test_string_format() {
        use crate::plan::logical::ScalarFunction;

        // FORMAT('Hello %s, you have %s messages', 'Alice', 3)
        let result = eval_fn(
            ScalarFunction::Format,
            vec![
                Value::from("Hello %s, you have %s messages"),
                Value::from("Alice"),
                Value::Int(3),
            ],
        );
        assert_eq!(result, Value::String("Hello Alice, you have 3 messages".to_string()));
    }

    #[test]
    fn test_string_replace() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::Replace,
            vec![Value::from("hello world"), Value::from("world"), Value::from("there")],
        );
        assert_eq!(result, Value::String("hello there".to_string()));
    }

    #[test]
    fn test_string_trim_functions() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Trim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("hello".to_string()));

        let result = eval_fn(ScalarFunction::Ltrim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("hello  ".to_string()));

        let result = eval_fn(ScalarFunction::Rtrim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("  hello".to_string()));
    }

    #[test]
    fn test_lpad_basic() {
        use crate::plan::logical::ScalarFunction;

        // Basic padding: LPAD('hi', 5, 'x') = 'xxxhi'
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::from("hi"), Value::Int(5), Value::from("x")]);
        assert_eq!(result, Value::String("xxxhi".to_string()));

        // Multi-character fill string: LPAD('hi', 5, 'xy') = 'xyxhi'
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("hi"), Value::Int(5), Value::from("xy")],
        );
        assert_eq!(result, Value::String("xyxhi".to_string()));

        // Default fill (space) when not provided: LPAD('hi', 5) = '   hi'
        let result = eval_fn(ScalarFunction::Lpad, vec![Value::from("hi"), Value::Int(5)]);
        assert_eq!(result, Value::String("   hi".to_string()));
    }

    #[test]
    fn test_lpad_truncation() {
        use crate::plan::logical::ScalarFunction;

        // Truncation when string is longer than length: LPAD('hello', 3, 'x') = 'hel'
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("hello"), Value::Int(3), Value::from("x")],
        );
        assert_eq!(result, Value::String("hel".to_string()));

        // String exactly matches length: LPAD('hi', 2, 'x') = 'hi'
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::from("hi"), Value::Int(2), Value::from("x")]);
        assert_eq!(result, Value::String("hi".to_string()));
    }

    #[test]
    fn test_lpad_edge_cases() {
        use crate::plan::logical::ScalarFunction;

        // Negative length returns empty string
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("hello"), Value::Int(-1), Value::from("x")],
        );
        assert_eq!(result, Value::String(String::new()));

        // Zero length returns empty string
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("hello"), Value::Int(0), Value::from("x")],
        );
        assert_eq!(result, Value::String(String::new()));

        // Empty fill string returns original truncated to length
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("hello"), Value::Int(3), Value::from("")],
        );
        assert_eq!(result, Value::String("hel".to_string()));

        // Empty input string with padding
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::from(""), Value::Int(3), Value::from("x")]);
        assert_eq!(result, Value::String("xxx".to_string()));
    }

    #[test]
    fn test_lpad_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // NULL string returns NULL
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::Null, Value::Int(5), Value::from("x")]);
        assert_eq!(result, Value::Null);

        // NULL length returns NULL
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::from("hi"), Value::Null, Value::from("x")]);
        assert_eq!(result, Value::Null);

        // NULL fill returns NULL
        let result =
            eval_fn(ScalarFunction::Lpad, vec![Value::from("hi"), Value::Int(5), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_rpad_basic() {
        use crate::plan::logical::ScalarFunction;

        // Basic padding: RPAD('hi', 5, 'x') = 'hixxx'
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::from("hi"), Value::Int(5), Value::from("x")]);
        assert_eq!(result, Value::String("hixxx".to_string()));

        // Multi-character fill string: RPAD('hi', 5, 'xy') = 'hixyx'
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("hi"), Value::Int(5), Value::from("xy")],
        );
        assert_eq!(result, Value::String("hixyx".to_string()));

        // Default fill (space) when not provided: RPAD('hi', 5) = 'hi   '
        let result = eval_fn(ScalarFunction::Rpad, vec![Value::from("hi"), Value::Int(5)]);
        assert_eq!(result, Value::String("hi   ".to_string()));
    }

    #[test]
    fn test_rpad_truncation() {
        use crate::plan::logical::ScalarFunction;

        // Truncation when string is longer than length: RPAD('hello', 3, 'x') = 'hel'
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("hello"), Value::Int(3), Value::from("x")],
        );
        assert_eq!(result, Value::String("hel".to_string()));

        // String exactly matches length: RPAD('hi', 2, 'x') = 'hi'
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::from("hi"), Value::Int(2), Value::from("x")]);
        assert_eq!(result, Value::String("hi".to_string()));
    }

    #[test]
    fn test_rpad_edge_cases() {
        use crate::plan::logical::ScalarFunction;

        // Negative length returns empty string
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("hello"), Value::Int(-1), Value::from("x")],
        );
        assert_eq!(result, Value::String(String::new()));

        // Zero length returns empty string
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("hello"), Value::Int(0), Value::from("x")],
        );
        assert_eq!(result, Value::String(String::new()));

        // Empty fill string returns original truncated to length
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("hello"), Value::Int(3), Value::from("")],
        );
        assert_eq!(result, Value::String("hel".to_string()));

        // Empty input string with padding
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::from(""), Value::Int(3), Value::from("x")]);
        assert_eq!(result, Value::String("xxx".to_string()));
    }

    #[test]
    fn test_rpad_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // NULL string returns NULL
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::Null, Value::Int(5), Value::from("x")]);
        assert_eq!(result, Value::Null);

        // NULL length returns NULL
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::from("hi"), Value::Null, Value::from("x")]);
        assert_eq!(result, Value::Null);

        // NULL fill returns NULL
        let result =
            eval_fn(ScalarFunction::Rpad, vec![Value::from("hi"), Value::Int(5), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_lpad_rpad_unicode() {
        use crate::plan::logical::ScalarFunction;

        // Unicode characters (multi-byte): LPAD('日本', 5, '語') = '語語語日本'
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("日本"), Value::Int(5), Value::from("語")],
        );
        assert_eq!(result, Value::String("語語語日本".to_string()));

        // Unicode RPAD: RPAD('日本', 5, '語') = '日本語語語'
        let result = eval_fn(
            ScalarFunction::Rpad,
            vec![Value::from("日本"), Value::Int(5), Value::from("語")],
        );
        assert_eq!(result, Value::String("日本語語語".to_string()));

        // Unicode truncation: LPAD('日本語', 2, 'x') = '日本'
        let result = eval_fn(
            ScalarFunction::Lpad,
            vec![Value::from("日本語"), Value::Int(2), Value::from("x")],
        );
        assert_eq!(result, Value::String("日本".to_string()));
    }

    #[test]
    fn test_regexp_match() {
        use crate::plan::logical::ScalarFunction;

        // Match with capture group
        let result = eval_fn(
            ScalarFunction::RegexpMatch,
            vec![Value::from("hello123world"), Value::from(r"(\d+)")],
        );
        assert_eq!(result, Value::String("123".to_string()));

        // No match returns NULL
        let result =
            eval_fn(ScalarFunction::RegexpMatch, vec![Value::from("hello"), Value::from(r"\d+")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_regexp_replace() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::RegexpReplace,
            vec![Value::from("hello 123 world 456"), Value::from(r"\d+"), Value::from("X")],
        );
        assert_eq!(result, Value::String("hello X world X".to_string()));
    }

    // ========== Numeric Function Tests ==========

    #[test]
    fn test_numeric_exp() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Exp, vec![Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - std::f64::consts::E).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // e^0 = 1
        let result = eval_fn(ScalarFunction::Exp, vec![Value::Int(0)]);
        assert_eq!(result, Value::Float(1.0));
    }

    #[test]
    fn test_numeric_ln() {
        use crate::plan::logical::ScalarFunction;

        // ln(e) = 1
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(std::f64::consts::E)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // ln(1) = 0
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Int(1)]);
        assert_eq!(result, Value::Float(0.0));

        // ln(0) = NULL (undefined)
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(0.0)]);
        assert_eq!(result, Value::Null);

        // ln(-1) = NULL (undefined)
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(-1.0)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_numeric_log() {
        use crate::plan::logical::ScalarFunction;

        // LOG(10, 100) = 2
        let result = eval_fn(ScalarFunction::Log, vec![Value::Int(10), Value::Int(100)]);
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // LOG(2, 8) = 3
        let result = eval_fn(ScalarFunction::Log, vec![Value::Int(2), Value::Int(8)]);
        if let Value::Float(f) = result {
            assert!((f - 3.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_log10() {
        use crate::plan::logical::ScalarFunction;

        // log10(100) = 2
        let result = eval_fn(ScalarFunction::Log10, vec![Value::Int(100)]);
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_trig() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // sin(0) = 0
        let result = eval_fn(ScalarFunction::Sin, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!(f.abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // cos(0) = 1
        let result = eval_fn(ScalarFunction::Cos, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // tan(0) = 0
        let result = eval_fn(ScalarFunction::Tan, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!(f.abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // sin(PI/2) ≈ 1
        let result = eval_fn(ScalarFunction::Sin, vec![Value::Float(PI / 2.0)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_inverse_trig() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // asin(1) = PI/2
        let result = eval_fn(ScalarFunction::Asin, vec![Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // acos(0) = PI/2
        let result = eval_fn(ScalarFunction::Acos, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // atan(0) = 0
        let result = eval_fn(ScalarFunction::Atan, vec![Value::Float(0.0)]);
        assert_eq!(result, Value::Float(0.0));

        // atan2(1, 1) = PI/4
        let result = eval_fn(ScalarFunction::Atan2, vec![Value::Float(1.0), Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 4.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // asin(2) = NULL (out of domain)
        let result = eval_fn(ScalarFunction::Asin, vec![Value::Float(2.0)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_numeric_degrees_radians() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // degrees(PI) = 180
        let result = eval_fn(ScalarFunction::Degrees, vec![Value::Float(PI)]);
        if let Value::Float(f) = result {
            assert!((f - 180.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // radians(180) = PI
        let result = eval_fn(ScalarFunction::Radians, vec![Value::Float(180.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_sign() {
        use crate::plan::logical::ScalarFunction;

        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(42)]), Value::Int(1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(-42)]), Value::Int(-1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(0)]), Value::Int(0));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Float(3.14)]), Value::Int(1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Float(-3.14)]), Value::Int(-1));
    }

    #[test]
    fn test_numeric_pi() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Pi, vec![]);
        assert_eq!(result, Value::Float(std::f64::consts::PI));
    }

    #[test]
    fn test_numeric_trunc() {
        use crate::plan::logical::ScalarFunction;

        // TRUNC(3.789) = 3
        let result = eval_fn(ScalarFunction::Trunc, vec![Value::Float(3.789)]);
        assert_eq!(result, Value::Float(3.0));

        // TRUNC(3.789, 2) = 3.78
        let result = eval_fn(ScalarFunction::Trunc, vec![Value::Float(3.789), Value::Int(2)]);
        if let Value::Float(f) = result {
            assert!((f - 3.78).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_round_with_precision() {
        use crate::plan::logical::ScalarFunction;

        // ROUND(3.14159, 2) = 3.14
        let result = eval_fn(ScalarFunction::Round, vec![Value::Float(3.14159), Value::Int(2)]);
        if let Value::Float(f) = result {
            assert!((f - 3.14).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    // ========== Date/Time Function Tests ==========

    #[test]
    fn test_date_part() {
        use crate::plan::logical::ScalarFunction;

        let ts = "2024-01-15 10:30:45";

        // year
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("year"), Value::from(ts)]);
        assert_eq!(result, Value::Float(2024.0));

        // month
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("month"), Value::from(ts)]);
        assert_eq!(result, Value::Float(1.0));

        // day
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("day"), Value::from(ts)]);
        assert_eq!(result, Value::Float(15.0));

        // hour
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("hour"), Value::from(ts)]);
        assert_eq!(result, Value::Float(10.0));

        // minute
        let result =
            eval_fn(ScalarFunction::DatePart, vec![Value::from("minute"), Value::from(ts)]);
        assert_eq!(result, Value::Float(30.0));

        // second
        let result =
            eval_fn(ScalarFunction::DatePart, vec![Value::from("second"), Value::from(ts)]);
        assert_eq!(result, Value::Float(45.0));
    }

    #[test]
    fn test_date_trunc() {
        use crate::plan::logical::ScalarFunction;

        let ts = "2024-01-15 10:30:45";

        // trunc to year
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("year"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));

        // trunc to month
        let result =
            eval_fn(ScalarFunction::DateTrunc, vec![Value::from("month"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));

        // trunc to day
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("day"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-15 00:00:00".to_string()));

        // trunc to hour
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("hour"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-15 10:00:00".to_string()));
    }

    #[test]
    fn test_to_timestamp_epoch() {
        use crate::plan::logical::ScalarFunction;

        // TO_TIMESTAMP(0) = 1970-01-01 00:00:00
        let result = eval_fn(ScalarFunction::ToTimestamp, vec![Value::Int(0)]);
        assert_eq!(result, Value::String("1970-01-01 00:00:00+00".to_string()));
    }

    #[test]
    fn test_to_date() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::ToDate,
            vec![Value::from("2024-01-15"), Value::from("YYYY-MM-DD")],
        );
        assert_eq!(result, Value::String("2024-01-15".to_string()));
    }

    #[test]
    fn test_to_char() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::ToChar,
            vec![Value::from("2024-01-15 10:30:45"), Value::from("YYYY-MM-DD")],
        );
        assert_eq!(result, Value::String("2024-01-15".to_string()));
    }

    // ========== Tier 2 Date/Time Function Tests ==========

    #[test]
    fn test_age_two_args() {
        use crate::plan::logical::ScalarFunction;

        // age('2024-01-01', '2020-01-01') = 4 years
        let result = eval_fn(
            ScalarFunction::Age,
            vec![Value::from("2024-01-01"), Value::from("2020-01-01")],
        );
        assert_eq!(result, Value::String("4 years".to_string()));

        // age('2024-03-15', '2020-01-10') = 4 years 2 mons 5 days
        let result = eval_fn(
            ScalarFunction::Age,
            vec![Value::from("2024-03-15"), Value::from("2020-01-10")],
        );
        assert_eq!(result, Value::String("4 years 2 mons 5 days".to_string()));

        // age('2024-01-01 10:30:00', '2024-01-01 08:00:00') = 02:30:00
        let result = eval_fn(
            ScalarFunction::Age,
            vec![Value::from("2024-01-01 10:30:00"), Value::from("2024-01-01 08:00:00")],
        );
        assert_eq!(result, Value::String("02:30:00".to_string()));

        // Negative age (older than newer)
        let result = eval_fn(
            ScalarFunction::Age,
            vec![Value::from("2020-01-01"), Value::from("2024-01-01")],
        );
        assert_eq!(result, Value::String("-4 years".to_string()));
    }

    #[test]
    fn test_date_add() {
        use crate::plan::logical::ScalarFunction;

        // date_add('2024-01-01', '1 month')
        let result = eval_fn(
            ScalarFunction::DateAdd,
            vec![Value::from("2024-01-01"), Value::from("1 month")],
        );
        assert_eq!(result, Value::String("2024-02-01 00:00:00".to_string()));

        // date_add('2024-01-01', '1 year')
        let result = eval_fn(
            ScalarFunction::DateAdd,
            vec![Value::from("2024-01-01"), Value::from("1 year")],
        );
        assert_eq!(result, Value::String("2025-01-01 00:00:00".to_string()));

        // date_add('2024-01-01', '7 days')
        let result = eval_fn(
            ScalarFunction::DateAdd,
            vec![Value::from("2024-01-01"), Value::from("7 days")],
        );
        assert_eq!(result, Value::String("2024-01-08 00:00:00".to_string()));

        // date_add with compound interval
        let result = eval_fn(
            ScalarFunction::DateAdd,
            vec![Value::from("2024-01-01 12:00:00"), Value::from("1 year 2 months 3 days")],
        );
        assert_eq!(result, Value::String("2025-03-04 12:00:00".to_string()));

        // Adding hours
        let result = eval_fn(
            ScalarFunction::DateAdd,
            vec![Value::from("2024-01-01 12:00:00"), Value::from("5 hours")],
        );
        assert_eq!(result, Value::String("2024-01-01 17:00:00".to_string()));
    }

    #[test]
    fn test_date_subtract() {
        use crate::plan::logical::ScalarFunction;

        // date_subtract('2024-02-01', '1 month')
        let result = eval_fn(
            ScalarFunction::DateSubtract,
            vec![Value::from("2024-02-01"), Value::from("1 month")],
        );
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));

        // date_subtract('2024-01-08', '7 days')
        let result = eval_fn(
            ScalarFunction::DateSubtract,
            vec![Value::from("2024-01-08"), Value::from("7 days")],
        );
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));
    }

    #[test]
    fn test_make_timestamp() {
        use crate::plan::logical::ScalarFunction;

        // make_timestamp(2024, 1, 15, 14, 30, 0)
        let result = eval_fn(
            ScalarFunction::MakeTimestamp,
            vec![
                Value::Int(2024),
                Value::Int(1),
                Value::Int(15),
                Value::Int(14),
                Value::Int(30),
                Value::Int(0),
            ],
        );
        assert_eq!(result, Value::String("2024-01-15 14:30:00.000000".to_string()));

        // make_timestamp with fractional seconds
        let result = eval_fn(
            ScalarFunction::MakeTimestamp,
            vec![
                Value::Int(2024),
                Value::Int(6),
                Value::Int(15),
                Value::Int(10),
                Value::Int(30),
                Value::Float(45.5),
            ],
        );
        assert_eq!(result, Value::String("2024-06-15 10:30:45.500000".to_string()));

        // Invalid date returns null
        let result = eval_fn(
            ScalarFunction::MakeTimestamp,
            vec![
                Value::Int(2024),
                Value::Int(13), // Invalid month
                Value::Int(1),
                Value::Int(0),
                Value::Int(0),
                Value::Int(0),
            ],
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_make_date() {
        use crate::plan::logical::ScalarFunction;

        // make_date(2024, 1, 15)
        let result = eval_fn(
            ScalarFunction::MakeDate,
            vec![Value::Int(2024), Value::Int(1), Value::Int(15)],
        );
        assert_eq!(result, Value::String("2024-01-15".to_string()));

        // make_date(2024, 2, 29) - leap year
        let result = eval_fn(
            ScalarFunction::MakeDate,
            vec![Value::Int(2024), Value::Int(2), Value::Int(29)],
        );
        assert_eq!(result, Value::String("2024-02-29".to_string()));

        // make_date(2023, 2, 29) - not leap year, invalid
        let result = eval_fn(
            ScalarFunction::MakeDate,
            vec![Value::Int(2023), Value::Int(2), Value::Int(29)],
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_make_time() {
        use crate::plan::logical::ScalarFunction;

        // make_time(14, 30, 0)
        let result =
            eval_fn(ScalarFunction::MakeTime, vec![Value::Int(14), Value::Int(30), Value::Int(0)]);
        assert_eq!(result, Value::String("14:30:00.000000".to_string()));

        // make_time with fractional seconds
        let result = eval_fn(
            ScalarFunction::MakeTime,
            vec![Value::Int(10), Value::Int(30), Value::Float(45.5)],
        );
        assert_eq!(result, Value::String("10:30:45.500000".to_string()));

        // Invalid time returns null
        let result = eval_fn(
            ScalarFunction::MakeTime,
            vec![Value::Int(25), Value::Int(0), Value::Int(0)], // Invalid hour
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_timezone() {
        use crate::plan::logical::ScalarFunction;

        // Convert UTC to EST (-5 hours)
        let result = eval_fn(
            ScalarFunction::Timezone,
            vec![Value::from("EST"), Value::from("2024-01-01 12:00:00")],
        );
        assert_eq!(result, Value::String("2024-01-01 07:00:00-05:00".to_string()));

        // Convert UTC to UTC (no change)
        let result = eval_fn(
            ScalarFunction::Timezone,
            vec![Value::from("UTC"), Value::from("2024-01-01 12:00:00")],
        );
        assert_eq!(result, Value::String("2024-01-01 12:00:00+00:00".to_string()));

        // Convert using offset notation
        let result = eval_fn(
            ScalarFunction::Timezone,
            vec![Value::from("+05:30"), Value::from("2024-01-01 12:00:00")],
        );
        assert_eq!(result, Value::String("2024-01-01 17:30:00+05:30".to_string()));

        // Convert to JST (Japan Standard Time, +9)
        let result = eval_fn(
            ScalarFunction::Timezone,
            vec![Value::from("JST"), Value::from("2024-01-01 12:00:00")],
        );
        assert_eq!(result, Value::String("2024-01-01 21:00:00+09:00".to_string()));
    }

    #[test]
    fn test_nullif() {
        use crate::plan::logical::ScalarFunction;

        // NULLIF(5, 5) = NULL
        let result = eval_fn(ScalarFunction::NullIf, vec![Value::Int(5), Value::Int(5)]);
        assert_eq!(result, Value::Null);

        // NULLIF(5, 3) = 5
        let result = eval_fn(ScalarFunction::NullIf, vec![Value::Int(5), Value::Int(3)]);
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_substring() {
        use crate::plan::logical::ScalarFunction;

        // SUBSTRING('hello', 2, 3) = 'ell'
        let result = eval_fn(
            ScalarFunction::Substring,
            vec![Value::from("hello"), Value::Int(2), Value::Int(3)],
        );
        assert_eq!(result, Value::String("ell".to_string()));

        // SUBSTRING('hello', 2) = 'ello' (to end)
        let result = eval_fn(ScalarFunction::Substring, vec![Value::from("hello"), Value::Int(2)]);
        assert_eq!(result, Value::String("ello".to_string()));
    }

    #[test]
    fn test_left() {
        use crate::plan::logical::ScalarFunction;

        // LEFT('Hello World', 5) = 'Hello'
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("Hello World"), Value::Int(5)]);
        assert_eq!(result, Value::String("Hello".to_string()));

        // LEFT('abc', 5) = 'abc' (length > string length returns entire string)
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("abc"), Value::Int(5)]);
        assert_eq!(result, Value::String("abc".to_string()));

        // LEFT('hello', 0) = '' (zero length returns empty string)
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("hello"), Value::Int(0)]);
        assert_eq!(result, Value::String(String::new()));

        // LEFT('hello', -1) = '' (negative length returns empty string)
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("hello"), Value::Int(-1)]);
        assert_eq!(result, Value::String(String::new()));

        // LEFT with unicode characters
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("日本語"), Value::Int(2)]);
        assert_eq!(result, Value::String("日本".to_string()));

        // LEFT with null input returns null
        let result = eval_fn(ScalarFunction::Left, vec![Value::Null, Value::Int(5)]);
        assert_eq!(result, Value::Null);

        // LEFT with null length returns null
        let result = eval_fn(ScalarFunction::Left, vec![Value::from("hello"), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_right() {
        use crate::plan::logical::ScalarFunction;

        // RIGHT('Hello World', 5) = 'World'
        let result =
            eval_fn(ScalarFunction::Right, vec![Value::from("Hello World"), Value::Int(5)]);
        assert_eq!(result, Value::String("World".to_string()));

        // RIGHT('abc', 5) = 'abc' (length > string length returns entire string)
        let result = eval_fn(ScalarFunction::Right, vec![Value::from("abc"), Value::Int(5)]);
        assert_eq!(result, Value::String("abc".to_string()));

        // RIGHT('hello', 0) = '' (zero length returns empty string)
        let result = eval_fn(ScalarFunction::Right, vec![Value::from("hello"), Value::Int(0)]);
        assert_eq!(result, Value::String(String::new()));

        // RIGHT('hello', -1) = '' (negative length returns empty string)
        let result = eval_fn(ScalarFunction::Right, vec![Value::from("hello"), Value::Int(-1)]);
        assert_eq!(result, Value::String(String::new()));

        // RIGHT with unicode characters
        let result = eval_fn(ScalarFunction::Right, vec![Value::from("日本語"), Value::Int(2)]);
        assert_eq!(result, Value::String("本語".to_string()));

        // RIGHT with null input returns null
        let result = eval_fn(ScalarFunction::Right, vec![Value::Null, Value::Int(5)]);
        assert_eq!(result, Value::Null);

        // RIGHT with null length returns null
        let result = eval_fn(ScalarFunction::Right, vec![Value::from("hello"), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_range_function() {
        use crate::plan::logical::ScalarFunction;

        // RANGE(1, 5) = [1, 2, 3, 4, 5]
        let result = eval_fn(ScalarFunction::Range, vec![Value::Int(1), Value::Int(5)]);
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
                Value::Int(4),
                Value::Int(5)
            ])
        );

        // RANGE(5, 1) = [5, 4, 3, 2, 1] (reverse order)
        let result = eval_fn(ScalarFunction::Range, vec![Value::Int(5), Value::Int(1)]);
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Int(5),
                Value::Int(4),
                Value::Int(3),
                Value::Int(2),
                Value::Int(1)
            ])
        );

        // RANGE(0, 10, 2) = [0, 2, 4, 6, 8, 10]
        let result =
            eval_fn(ScalarFunction::Range, vec![Value::Int(0), Value::Int(10), Value::Int(2)]);
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Int(0),
                Value::Int(2),
                Value::Int(4),
                Value::Int(6),
                Value::Int(8),
                Value::Int(10)
            ])
        );

        // RANGE(10, 0, -3) = [10, 7, 4, 1]
        let result =
            eval_fn(ScalarFunction::Range, vec![Value::Int(10), Value::Int(0), Value::Int(-3)]);
        assert_eq!(
            result,
            Value::Array(vec![Value::Int(10), Value::Int(7), Value::Int(4), Value::Int(1)])
        );
    }

    #[test]
    fn test_size_function() {
        use crate::plan::logical::ScalarFunction;

        // SIZE([1, 2, 3]) = 3
        let result = eval_fn(
            ScalarFunction::Size,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Int(3));

        // SIZE("hello") = 5
        let result = eval_fn(ScalarFunction::Size, vec![Value::from("hello")]);
        assert_eq!(result, Value::Int(5));

        // SIZE([]) = 0
        let result = eval_fn(ScalarFunction::Size, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_head_function() {
        use crate::plan::logical::ScalarFunction;

        // HEAD([1, 2, 3]) = 1
        let result = eval_fn(
            ScalarFunction::Head,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Int(1));

        // HEAD([]) = null
        let result = eval_fn(ScalarFunction::Head, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_tail_function() {
        use crate::plan::logical::ScalarFunction;

        // TAIL([1, 2, 3]) = [2, 3]
        let result = eval_fn(
            ScalarFunction::Tail,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(2), Value::Int(3)]));

        // TAIL([1]) = []
        let result = eval_fn(ScalarFunction::Tail, vec![Value::Array(vec![Value::Int(1)])]);
        assert_eq!(result, Value::Array(vec![]));

        // TAIL([]) = []
        let result = eval_fn(ScalarFunction::Tail, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Array(vec![]));
    }

    #[test]
    fn test_last_function() {
        use crate::plan::logical::ScalarFunction;

        // LAST([1, 2, 3]) = 3
        let result = eval_fn(
            ScalarFunction::Last,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Int(3));

        // LAST([]) = null
        let result = eval_fn(ScalarFunction::Last, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_reverse_function() {
        use crate::plan::logical::ScalarFunction;

        // REVERSE([1, 2, 3]) = [3, 2, 1]
        let result = eval_fn(
            ScalarFunction::Reverse,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(3), Value::Int(2), Value::Int(1)]));

        // REVERSE("hello") = "olleh"
        let result = eval_fn(ScalarFunction::Reverse, vec![Value::from("hello")]);
        assert_eq!(result, Value::String("olleh".to_string()));
    }

    #[test]
    fn test_list_comprehension_filter_only() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: [x IN [1, 2, 3, 4, 5] WHERE x > 2]
        let comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
                LogicalExpr::integer(4),
                LogicalExpr::integer(5),
            ])),
            filter_predicate: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(2)),
            })),
            transform_expr: None,
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&comprehension, &row).unwrap();
        assert_eq!(result, Value::Array(vec![Value::Int(3), Value::Int(4), Value::Int(5)]));
    }

    #[test]
    fn test_list_comprehension_transform_only() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: [x IN [1, 2, 3] | x * 2]
        let comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            filter_predicate: None,
            transform_expr: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Mul,
                right: Box::new(LogicalExpr::integer(2)),
            })),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&comprehension, &row).unwrap();
        assert_eq!(result, Value::Array(vec![Value::Int(2), Value::Int(4), Value::Int(6)]));
    }

    #[test]
    fn test_list_comprehension_filter_and_transform() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: [x IN [1, 2, 3, 4, 5] WHERE x % 2 = 0 | x * x]
        // (even numbers squared: [4, 16])
        let comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
                LogicalExpr::integer(4),
                LogicalExpr::integer(5),
            ])),
            filter_predicate: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::BinaryOp {
                    left: Box::new(LogicalExpr::column("x")),
                    op: crate::ast::BinaryOp::Mod,
                    right: Box::new(LogicalExpr::integer(2)),
                }),
                op: crate::ast::BinaryOp::Eq,
                right: Box::new(LogicalExpr::integer(0)),
            })),
            transform_expr: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Mul,
                right: Box::new(LogicalExpr::column("x")),
            })),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&comprehension, &row).unwrap();
        assert_eq!(result, Value::Array(vec![Value::Int(4), Value::Int(16)]));
    }

    #[test]
    fn test_list_literal() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: [1, 2, 3]
        let list = LogicalExpr::ListLiteral(vec![
            LogicalExpr::integer(1),
            LogicalExpr::integer(2),
            LogicalExpr::integer(3),
        ]);

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&list, &row).unwrap();
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));
    }

    #[test]
    fn test_list_comprehension_with_range() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::{LogicalExpr, ScalarFunction};
        use std::sync::Arc;

        // Test: [x IN range(1, 5) | x * 2]
        let comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ScalarFunction {
                func: ScalarFunction::Range,
                args: vec![LogicalExpr::integer(1), LogicalExpr::integer(5)],
            }),
            filter_predicate: None,
            transform_expr: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Mul,
                right: Box::new(LogicalExpr::integer(2)),
            })),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&comprehension, &row).unwrap();
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Int(2),
                Value::Int(4),
                Value::Int(6),
                Value::Int(8),
                Value::Int(10)
            ])
        );
    }

    #[test]
    fn test_nested_list_comprehension() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: [x IN [x IN [1, 2, 3] | x + 1] | x * 2]
        // Inner: [2, 3, 4], Outer: [4, 6, 8]
        let inner_comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            filter_predicate: None,
            transform_expr: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Add,
                right: Box::new(LogicalExpr::integer(1)),
            })),
        };

        let outer_comprehension = LogicalExpr::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(inner_comprehension),
            filter_predicate: None,
            transform_expr: Some(Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Mul,
                right: Box::new(LogicalExpr::integer(2)),
            })),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&outer_comprehension, &row).unwrap();
        assert_eq!(result, Value::Array(vec![Value::Int(4), Value::Int(6), Value::Int(8)]));
    }

    // ========== List Predicate Function Tests ==========

    #[test]
    fn test_list_predicate_all_true() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: all(x IN [1, 2, 3] WHERE x > 0) -> true
        let expr = LogicalExpr::ListPredicateAll {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_all_false() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: all(x IN [1, -2, 3] WHERE x > 0) -> false
        let expr = LogicalExpr::ListPredicateAll {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(-2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_all_empty_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: all(x IN [] WHERE x > 0) -> true (vacuous truth)
        let expr = LogicalExpr::ListPredicateAll {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_any_true() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: any(x IN [1, 2, 3] WHERE x > 2) -> true
        let expr = LogicalExpr::ListPredicateAny {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_any_false() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: any(x IN [1, 2, 3] WHERE x > 5) -> false
        let expr = LogicalExpr::ListPredicateAny {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(5)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_any_empty_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: any(x IN [] WHERE x > 0) -> false
        let expr = LogicalExpr::ListPredicateAny {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_none_true() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: none(x IN [1, 2, 3] WHERE x < 0) -> true
        let expr = LogicalExpr::ListPredicateNone {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Lt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_none_false() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: none(x IN [1, 2, 3] WHERE x > 2) -> false
        let expr = LogicalExpr::ListPredicateNone {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_none_empty_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: none(x IN [] WHERE x > 0) -> true
        let expr = LogicalExpr::ListPredicateNone {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_single_true() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: single(x IN [1, 2, 3] WHERE x = 2) -> true
        let expr = LogicalExpr::ListPredicateSingle {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Eq,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_list_predicate_single_false_multiple_matches() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: single(x IN [1, 2, 2] WHERE x = 2) -> false (two matches)
        let expr = LogicalExpr::ListPredicateSingle {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(2),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Eq,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_single_false_no_matches() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: single(x IN [1, 3, 5] WHERE x = 2) -> false (no matches)
        let expr = LogicalExpr::ListPredicateSingle {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(3),
                LogicalExpr::integer(5),
            ])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Eq,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_predicate_single_empty_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: single(x IN [] WHERE x = 2) -> false
        let expr = LogicalExpr::ListPredicateSingle {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![])),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Eq,
                right: Box::new(LogicalExpr::integer(2)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_list_reduce_sum() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: reduce(sum = 0, x IN [1, 2, 3] | sum + x) -> 6
        let expr = LogicalExpr::ListReduce {
            accumulator: "sum".to_string(),
            initial: Box::new(LogicalExpr::integer(0)),
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(1),
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
            ])),
            expression: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("sum")),
                op: crate::ast::BinaryOp::Add,
                right: Box::new(LogicalExpr::column("x")),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Int(6));
    }

    #[test]
    fn test_list_reduce_product() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: reduce(product = 1, x IN [2, 3, 4] | product * x) -> 24
        let expr = LogicalExpr::ListReduce {
            accumulator: "product".to_string(),
            initial: Box::new(LogicalExpr::integer(1)),
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::integer(2),
                LogicalExpr::integer(3),
                LogicalExpr::integer(4),
            ])),
            expression: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("product")),
                op: crate::ast::BinaryOp::Mul,
                right: Box::new(LogicalExpr::column("x")),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Int(24));
    }

    #[test]
    fn test_list_reduce_empty_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: reduce(sum = 0, x IN [] | sum + x) -> 0 (initial value)
        let expr = LogicalExpr::ListReduce {
            accumulator: "sum".to_string(),
            initial: Box::new(LogicalExpr::integer(0)),
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![])),
            expression: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("sum")),
                op: crate::ast::BinaryOp::Add,
                right: Box::new(LogicalExpr::column("x")),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_list_reduce_with_string_concat() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::{LogicalExpr, ScalarFunction};
        use std::sync::Arc;

        // Test: reduce(s = '', x IN ['a', 'b', 'c'] | concat(s, x)) -> 'abc'
        let expr = LogicalExpr::ListReduce {
            accumulator: "s".to_string(),
            initial: Box::new(LogicalExpr::string("")),
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::ListLiteral(vec![
                LogicalExpr::string("a"),
                LogicalExpr::string("b"),
                LogicalExpr::string("c"),
            ])),
            expression: Box::new(LogicalExpr::ScalarFunction {
                func: ScalarFunction::Concat,
                args: vec![LogicalExpr::column("s"), LogicalExpr::column("x")],
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::String("abc".to_string()));
    }

    #[test]
    fn test_list_predicate_with_null_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: all(x IN null WHERE x > 0) -> null
        let expr = LogicalExpr::ListPredicateAll {
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::null()),
            predicate: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("x")),
                op: crate::ast::BinaryOp::Gt,
                right: Box::new(LogicalExpr::integer(0)),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_list_reduce_with_null_list() {
        use crate::exec::row::{Row, Schema};
        use crate::plan::logical::LogicalExpr;
        use std::sync::Arc;

        // Test: reduce(sum = 0, x IN null | sum + x) -> null
        let expr = LogicalExpr::ListReduce {
            accumulator: "sum".to_string(),
            initial: Box::new(LogicalExpr::integer(0)),
            variable: "x".to_string(),
            list_expr: Box::new(LogicalExpr::null()),
            expression: Box::new(LogicalExpr::BinaryOp {
                left: Box::new(LogicalExpr::column("sum")),
                op: crate::ast::BinaryOp::Add,
                right: Box::new(LogicalExpr::column("x")),
            }),
        };

        let schema = Arc::new(Schema::empty());
        let row = Row::new(schema, vec![]);

        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Null);
    }

    // ========== JSON Function Tests ==========

    #[test]
    fn test_json_extract_path() {
        use crate::plan::logical::ScalarFunction;

        // Simple object extraction
        let json = r#"{"name": "Alice", "age": 30}"#;
        let result =
            eval_fn(ScalarFunction::JsonExtractPath, vec![Value::from(json), Value::from("name")]);
        assert_eq!(result, Value::String("\"Alice\"".to_string()));

        // Nested object extraction
        let json = r#"{"user": {"name": "Bob", "address": {"city": "NYC"}}}"#;
        let result = eval_fn(
            ScalarFunction::JsonExtractPath,
            vec![
                Value::from(json),
                Value::from("user"),
                Value::from("address"),
                Value::from("city"),
            ],
        );
        assert_eq!(result, Value::String("\"NYC\"".to_string()));

        // Array index extraction
        let json = r#"{"items": ["a", "b", "c"]}"#;
        let result = eval_fn(
            ScalarFunction::JsonExtractPath,
            vec![Value::from(json), Value::from("items"), Value::from("1")],
        );
        assert_eq!(result, Value::String("\"b\"".to_string()));

        // Non-existent path returns null
        let result = eval_fn(
            ScalarFunction::JsonExtractPath,
            vec![Value::from(json), Value::from("nonexistent")],
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_extract_path_text() {
        use crate::plan::logical::ScalarFunction;

        // Text extraction returns unquoted string
        let json = r#"{"name": "Alice"}"#;
        let result = eval_fn(
            ScalarFunction::JsonExtractPathText,
            vec![Value::from(json), Value::from("name")],
        );
        assert_eq!(result, Value::String("Alice".to_string()));

        // Number extraction as text
        let json = r#"{"age": 30}"#;
        let result = eval_fn(
            ScalarFunction::JsonExtractPathText,
            vec![Value::from(json), Value::from("age")],
        );
        assert_eq!(result, Value::String("30".to_string()));
    }

    #[test]
    fn test_json_build_object() {
        use crate::plan::logical::ScalarFunction;

        // Simple object
        let result = eval_fn(
            ScalarFunction::JsonBuildObject,
            vec![Value::from("name"), Value::from("Alice"), Value::from("age"), Value::Int(30)],
        );
        // Parse the result to verify it's valid JSON with expected content
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert_eq!(parsed["age"], 30);

        // Empty object
        let result = eval_fn(ScalarFunction::JsonBuildObject, vec![]);
        assert_eq!(result, Value::String("{}".to_string()));

        // Odd number of args returns null
        let result = eval_fn(ScalarFunction::JsonBuildObject, vec![Value::from("key")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_build_array() {
        use crate::plan::logical::ScalarFunction;

        // Simple array
        let result = eval_fn(
            ScalarFunction::JsonBuildArray,
            vec![Value::Int(1), Value::from("two"), Value::Bool(true)],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed[0], 1);
        assert_eq!(parsed[1], "two");
        assert_eq!(parsed[2], true);

        // Empty array
        let result = eval_fn(ScalarFunction::JsonBuildArray, vec![]);
        assert_eq!(result, Value::String("[]".to_string()));
    }

    #[test]
    fn test_jsonb_set() {
        use crate::plan::logical::ScalarFunction;

        // Set existing key
        let json = r#"{"name": "Alice", "age": 30}"#;
        let result = eval_fn(
            ScalarFunction::JsonbSet,
            vec![Value::from(json), Value::Array(vec![Value::from("name")]), Value::from("Bob")],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["name"], "Bob");
        assert_eq!(parsed["age"], 30);

        // Set nested key (create_missing = true by default)
        let json = r#"{"user": {"name": "Alice"}}"#;
        let result = eval_fn(
            ScalarFunction::JsonbSet,
            vec![
                Value::from(json),
                Value::Array(vec![Value::from("user"), Value::from("age")]),
                Value::Int(25),
            ],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["user"]["name"], "Alice");
        assert_eq!(parsed["user"]["age"], 25);

        // Set with create_missing = false (should not create new key)
        let json = r#"{"name": "Alice"}"#;
        let result = eval_fn(
            ScalarFunction::JsonbSet,
            vec![
                Value::from(json),
                Value::Array(vec![Value::from("age")]),
                Value::Int(30),
                Value::Bool(false),
            ],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert!(parsed.get("age").is_none());
    }

    #[test]
    fn test_jsonb_insert() {
        use crate::plan::logical::ScalarFunction;

        // Insert into array (before index)
        let json = r"[1, 2, 3]";
        let result = eval_fn(
            ScalarFunction::JsonbInsert,
            vec![Value::from(json), Value::Array(vec![Value::from("1")]), Value::Int(99)],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!([1, 99, 2, 3]));

        // Insert into array (after index)
        let result = eval_fn(
            ScalarFunction::JsonbInsert,
            vec![
                Value::from(json),
                Value::Array(vec![Value::from("1")]),
                Value::Int(99),
                Value::Bool(true), // insert_after
            ],
        );
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!([1, 2, 99, 3]));
    }

    #[test]
    fn test_jsonb_strip_nulls() {
        use crate::plan::logical::ScalarFunction;

        // Strip nulls from object
        let json = r#"{"name": "Alice", "age": null, "city": "NYC"}"#;
        let result = eval_fn(ScalarFunction::JsonbStripNulls, vec![Value::from(json)]);
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert_eq!(parsed["city"], "NYC");
        assert!(parsed.get("age").is_none());

        // Strip nulls recursively
        let json = r#"{"user": {"name": "Alice", "age": null}, "active": null}"#;
        let result = eval_fn(ScalarFunction::JsonbStripNulls, vec![Value::from(json)]);
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["user"]["name"], "Alice");
        assert!(parsed["user"].get("age").is_none());
        assert!(parsed.get("active").is_none());

        // Arrays with nulls are preserved (only object keys are stripped)
        let json = r#"{"items": [1, null, 3]}"#;
        let result = eval_fn(ScalarFunction::JsonbStripNulls, vec![Value::from(json)]);
        let parsed: serde_json::Value = serde_json::from_str(result.as_str().unwrap()).unwrap();
        assert_eq!(parsed["items"], serde_json::json!([1, null, 3]));
    }

    #[test]
    fn test_jsonb_functions_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // Null input returns null
        let result =
            eval_fn(ScalarFunction::JsonExtractPath, vec![Value::Null, Value::from("key")]);
        assert_eq!(result, Value::Null);

        let result = eval_fn(ScalarFunction::JsonbStripNulls, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        let result = eval_fn(
            ScalarFunction::JsonbSet,
            vec![Value::Null, Value::Array(vec![Value::from("key")]), Value::from("value")],
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_json_extract_with_integer_path() {
        use crate::plan::logical::ScalarFunction;

        // Using integer for array index
        let json = r#"["a", "b", "c"]"#;
        let result =
            eval_fn(ScalarFunction::JsonExtractPath, vec![Value::from(json), Value::Int(1)]);
        assert_eq!(result, Value::String("\"b\"".to_string()));
    }

    // ========== Map Projection Evaluation Tests ==========

    #[test]
    fn evaluate_map_projection_single_property() {
        // Test: p{.name} where p has name="Alice" and age=30
        let schema = Arc::new(Schema::new(vec!["p.name".to_string(), "p.age".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::from("Alice"), Value::Int(30)]);

        let expr = LogicalExpr::MapProjection {
            source: Box::new(LogicalExpr::column("p")),
            items: vec![LogicalMapProjectionItem::Property("name".to_string())],
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        // Result should be an array of [key, value] pairs
        match result {
            Value::Array(pairs) => {
                assert_eq!(pairs.len(), 1);
                // First pair should be ["name", "Alice"]
                if let Value::Array(pair) = &pairs[0] {
                    assert_eq!(pair[0], Value::String("name".to_string()));
                    assert_eq!(pair[1], Value::String("Alice".to_string()));
                } else {
                    panic!("Expected array pair");
                }
            }
            _ => panic!("Expected Array result, got {:?}", result),
        }
    }

    #[test]
    fn evaluate_map_projection_multiple_properties() {
        // Test: p{.name, .age}
        let schema = Arc::new(Schema::new(vec!["p.name".to_string(), "p.age".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::from("Bob"), Value::Int(25)]);

        let expr = LogicalExpr::MapProjection {
            source: Box::new(LogicalExpr::column("p")),
            items: vec![
                LogicalMapProjectionItem::Property("name".to_string()),
                LogicalMapProjectionItem::Property("age".to_string()),
            ],
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        match result {
            Value::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn evaluate_map_projection_computed_value() {
        // Test: p{.name, doubled: p.age * 2} is challenging because we need arithmetic
        // Let's use a simpler computed value: p{computed: 42}
        let schema = Arc::new(Schema::new(vec!["p.name".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::from("Alice")]);

        let expr = LogicalExpr::MapProjection {
            source: Box::new(LogicalExpr::column("p")),
            items: vec![
                LogicalMapProjectionItem::Property("name".to_string()),
                LogicalMapProjectionItem::Computed {
                    key: "computed".to_string(),
                    value: Box::new(LogicalExpr::integer(42)),
                },
            ],
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        match result {
            Value::Array(pairs) => {
                assert_eq!(pairs.len(), 2);
                // Second pair should be ["computed", 42]
                if let Value::Array(pair) = &pairs[1] {
                    assert_eq!(pair[0], Value::String("computed".to_string()));
                    assert_eq!(pair[1], Value::Int(42));
                } else {
                    panic!("Expected array pair");
                }
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn evaluate_map_projection_all_properties() {
        // Test: p{.*} - should include all properties with the p. prefix
        let schema = Arc::new(Schema::new(vec![
            "p.name".to_string(),
            "p.age".to_string(),
            "other".to_string(), // Should not be included
        ]));
        let row = Row::new(
            Arc::clone(&schema),
            vec![Value::from("Carol"), Value::Int(35), Value::from("ignored")],
        );

        let expr = LogicalExpr::MapProjection {
            source: Box::new(LogicalExpr::column("p")),
            items: vec![LogicalMapProjectionItem::AllProperties],
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        match result {
            Value::Array(pairs) => {
                // Should have 2 pairs: name and age (not "other")
                assert_eq!(pairs.len(), 2);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn evaluate_map_projection_empty() {
        // Test: p{} - empty projection
        let schema = Arc::new(Schema::new(vec!["p.name".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::from("Alice")]);

        let expr = LogicalExpr::MapProjection {
            source: Box::new(LogicalExpr::column("p")),
            items: vec![],
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        match result {
            Value::Array(pairs) => {
                assert!(pairs.is_empty());
            }
            _ => panic!("Expected empty Array result"),
        }
    }

    // ========== Array Function Tests ==========

    #[test]
    fn test_array_length() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_LENGTH([1, 2, 3], 1) = 3
        let result = eval_fn(
            ScalarFunction::ArrayLength,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(1)],
        );
        assert_eq!(result, Value::Int(3));

        // ARRAY_LENGTH([], 1) = NULL (PostgreSQL returns NULL for empty arrays)
        let result =
            eval_fn(ScalarFunction::ArrayLength, vec![Value::Array(vec![]), Value::Int(1)]);
        assert_eq!(result, Value::Null);

        // ARRAY_LENGTH([1, 2, 3], 2) = NULL (dimension 2 not supported for 1D arrays)
        let result = eval_fn(
            ScalarFunction::ArrayLength,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(2)],
        );
        assert_eq!(result, Value::Null);

        // NULL array = NULL
        let result = eval_fn(ScalarFunction::ArrayLength, vec![Value::Null, Value::Int(1)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cardinality() {
        use crate::plan::logical::ScalarFunction;

        // CARDINALITY([1, 2, 3]) = 3
        let result = eval_fn(
            ScalarFunction::Cardinality,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Int(3));

        // CARDINALITY([]) = 0
        let result = eval_fn(ScalarFunction::Cardinality, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Int(0));

        // CARDINALITY(NULL) = NULL
        let result = eval_fn(ScalarFunction::Cardinality, vec![Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_append() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_APPEND([1, 2, 3], 4) = [1, 2, 3, 4]
        let result = eval_fn(
            ScalarFunction::ArrayAppend,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(4)],
        );
        assert_eq!(
            result,
            Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)])
        );

        // ARRAY_APPEND([], 1) = [1]
        let result =
            eval_fn(ScalarFunction::ArrayAppend, vec![Value::Array(vec![]), Value::Int(1)]);
        assert_eq!(result, Value::Array(vec![Value::Int(1)]));

        // ARRAY_APPEND([1], NULL) = [1, NULL]
        let result = eval_fn(
            ScalarFunction::ArrayAppend,
            vec![Value::Array(vec![Value::Int(1)]), Value::Null],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Null]));

        // ARRAY_APPEND(NULL, 1) = NULL
        let result = eval_fn(ScalarFunction::ArrayAppend, vec![Value::Null, Value::Int(1)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_prepend() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_PREPEND(0, [1, 2, 3]) = [0, 1, 2, 3]
        let result = eval_fn(
            ScalarFunction::ArrayPrepend,
            vec![Value::Int(0), Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(
            result,
            Value::Array(vec![Value::Int(0), Value::Int(1), Value::Int(2), Value::Int(3)])
        );

        // ARRAY_PREPEND(1, []) = [1]
        let result =
            eval_fn(ScalarFunction::ArrayPrepend, vec![Value::Int(1), Value::Array(vec![])]);
        assert_eq!(result, Value::Array(vec![Value::Int(1)]));

        // ARRAY_PREPEND(1, NULL) = NULL
        let result = eval_fn(ScalarFunction::ArrayPrepend, vec![Value::Int(1), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_cat() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_CAT([1, 2], [3, 4]) = [1, 2, 3, 4]
        let result = eval_fn(
            ScalarFunction::ArrayCat,
            vec![
                Value::Array(vec![Value::Int(1), Value::Int(2)]),
                Value::Array(vec![Value::Int(3), Value::Int(4)]),
            ],
        );
        assert_eq!(
            result,
            Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)])
        );

        // ARRAY_CAT([], [1, 2]) = [1, 2]
        let result = eval_fn(
            ScalarFunction::ArrayCat,
            vec![Value::Array(vec![]), Value::Array(vec![Value::Int(1), Value::Int(2)])],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2)]));

        // ARRAY_CAT(NULL, [1]) = NULL
        let result =
            eval_fn(ScalarFunction::ArrayCat, vec![Value::Null, Value::Array(vec![Value::Int(1)])]);
        assert_eq!(result, Value::Null);

        // ARRAY_CAT([1], NULL) = NULL
        let result =
            eval_fn(ScalarFunction::ArrayCat, vec![Value::Array(vec![Value::Int(1)]), Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_remove() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_REMOVE([1, 2, 3, 2, 4], 2) = [1, 3, 4]
        let result = eval_fn(
            ScalarFunction::ArrayRemove,
            vec![
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Int(2),
                    Value::Int(4),
                ]),
                Value::Int(2),
            ],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(3), Value::Int(4)]));

        // ARRAY_REMOVE([1, 2, 3], 5) = [1, 2, 3] (element not found)
        let result = eval_fn(
            ScalarFunction::ArrayRemove,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(5)],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

        // ARRAY_REMOVE(NULL, 1) = NULL
        let result = eval_fn(ScalarFunction::ArrayRemove, vec![Value::Null, Value::Int(1)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_replace() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_REPLACE([1, 2, 3, 2, 4], 2, 9) = [1, 9, 3, 9, 4]
        let result = eval_fn(
            ScalarFunction::ArrayReplace,
            vec![
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Int(2),
                    Value::Int(4),
                ]),
                Value::Int(2),
                Value::Int(9),
            ],
        );
        assert_eq!(
            result,
            Value::Array(vec![
                Value::Int(1),
                Value::Int(9),
                Value::Int(3),
                Value::Int(9),
                Value::Int(4)
            ])
        );

        // ARRAY_REPLACE([1, 2, 3], 5, 9) = [1, 2, 3] (element not found)
        let result = eval_fn(
            ScalarFunction::ArrayReplace,
            vec![
                Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
                Value::Int(5),
                Value::Int(9),
            ],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

        // ARRAY_REPLACE(NULL, 1, 2) = NULL
        let result =
            eval_fn(ScalarFunction::ArrayReplace, vec![Value::Null, Value::Int(1), Value::Int(2)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_position() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_POSITION([1, 2, 3], 2) = 2 (1-based index)
        let result = eval_fn(
            ScalarFunction::ArrayPosition,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(2)],
        );
        assert_eq!(result, Value::Int(2));

        // ARRAY_POSITION([1, 2, 3], 1) = 1
        let result = eval_fn(
            ScalarFunction::ArrayPosition,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(1)],
        );
        assert_eq!(result, Value::Int(1));

        // ARRAY_POSITION([1, 2, 3], 5) = NULL (not found)
        let result = eval_fn(
            ScalarFunction::ArrayPosition,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(5)],
        );
        assert_eq!(result, Value::Null);

        // ARRAY_POSITION(NULL, 1) = NULL
        let result = eval_fn(ScalarFunction::ArrayPosition, vec![Value::Null, Value::Int(1)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_positions() {
        use crate::plan::logical::ScalarFunction;

        // ARRAY_POSITIONS([1, 2, 3, 2, 4], 2) = [2, 4] (1-based indices)
        let result = eval_fn(
            ScalarFunction::ArrayPositions,
            vec![
                Value::Array(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Int(2),
                    Value::Int(4),
                ]),
                Value::Int(2),
            ],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(2), Value::Int(4)]));

        // ARRAY_POSITIONS([1, 2, 3], 5) = [] (not found)
        let result = eval_fn(
            ScalarFunction::ArrayPositions,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]), Value::Int(5)],
        );
        assert_eq!(result, Value::Array(vec![]));

        // ARRAY_POSITIONS([1, 1, 1], 1) = [1, 2, 3]
        let result = eval_fn(
            ScalarFunction::ArrayPositions,
            vec![Value::Array(vec![Value::Int(1), Value::Int(1), Value::Int(1)]), Value::Int(1)],
        );
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

        // ARRAY_POSITIONS(NULL, 1) = NULL
        let result = eval_fn(ScalarFunction::ArrayPositions, vec![Value::Null, Value::Int(1)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_unnest() {
        use crate::plan::logical::ScalarFunction;

        // UNNEST([1, 2, 3]) as scalar = 1 (returns first element)
        // Note: True UNNEST behavior requires set-returning function context (UNWIND)
        let result = eval_fn(
            ScalarFunction::Unnest,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::Int(1));

        // UNNEST([]) = NULL
        let result = eval_fn(ScalarFunction::Unnest, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::Null);

        // UNNEST(NULL) = NULL
        let result = eval_fn(ScalarFunction::Unnest, vec![Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_array_functions_with_strings() {
        use crate::plan::logical::ScalarFunction;

        // Test array functions work with string elements
        let result = eval_fn(
            ScalarFunction::ArrayAppend,
            vec![Value::Array(vec![Value::from("a"), Value::from("b")]), Value::from("c")],
        );
        assert_eq!(
            result,
            Value::Array(vec![Value::from("a"), Value::from("b"), Value::from("c")])
        );

        let result = eval_fn(
            ScalarFunction::ArrayPosition,
            vec![
                Value::Array(vec![Value::from("a"), Value::from("b"), Value::from("c")]),
                Value::from("b"),
            ],
        );
        assert_eq!(result, Value::Int(2));

        let result = eval_fn(
            ScalarFunction::ArrayRemove,
            vec![
                Value::Array(vec![Value::from("a"), Value::from("b"), Value::from("a")]),
                Value::from("a"),
            ],
        );
        assert_eq!(result, Value::Array(vec![Value::from("b")]));
    }

    #[test]
    fn test_array_functions_with_mixed_types() {
        use crate::plan::logical::ScalarFunction;

        // Arrays can contain mixed types
        let mixed = Value::Array(vec![Value::Int(1), Value::from("two"), Value::Bool(true)]);

        let result = eval_fn(ScalarFunction::Cardinality, vec![mixed.clone()]);
        assert_eq!(result, Value::Int(3));

        let result = eval_fn(ScalarFunction::ArrayAppend, vec![mixed.clone(), Value::Float(4.0)]);
        if let Value::Array(arr) = result {
            assert_eq!(arr.len(), 4);
            assert_eq!(arr[3], Value::Float(4.0));
        } else {
            panic!("Expected array");
        }
    }

    // ========== Cypher Entity Functions ==========

    #[test]
    fn test_cypher_type() {
        use crate::plan::logical::ScalarFunction;

        // TYPE on a JSON object with _edge_type key
        let relationship = r#"{"_edge_type": "KNOWS", "since": 2020}"#;
        let result = eval_fn(ScalarFunction::Type, vec![Value::from(relationship)]);
        assert_eq!(result, Value::from("KNOWS"));

        // TYPE on a JSON object with _type key
        let relationship = r#"{"_type": "FOLLOWS", "weight": 5}"#;
        let result = eval_fn(ScalarFunction::Type, vec![Value::from(relationship)]);
        assert_eq!(result, Value::from("FOLLOWS"));

        // TYPE on a plain string (returns the string itself)
        let result = eval_fn(ScalarFunction::Type, vec![Value::from("LIKES")]);
        assert_eq!(result, Value::from("LIKES"));

        // TYPE on NULL returns NULL
        let result = eval_fn(ScalarFunction::Type, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // TYPE with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Type, vec![]);
        assert_eq!(result, Value::Null);

        // TYPE on non-relationship (integer) returns NULL
        let result = eval_fn(ScalarFunction::Type, vec![Value::Int(42)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_labels() {
        use crate::plan::logical::ScalarFunction;

        // LABELS on a JSON object with _labels key
        let node = r#"{"_labels": ["Person", "Employee"], "name": "Alice"}"#;
        let result = eval_fn(ScalarFunction::Labels, vec![Value::from(node)]);
        assert_eq!(result, Value::Array(vec![Value::from("Person"), Value::from("Employee")]));

        // LABELS on an array returns the array itself
        let labels = Value::Array(vec![Value::from("User"), Value::from("Admin")]);
        let result = eval_fn(ScalarFunction::Labels, vec![labels.clone()]);
        assert_eq!(result, labels);

        // LABELS on a plain string wraps it as a single label
        let result = eval_fn(ScalarFunction::Labels, vec![Value::from("Manager")]);
        assert_eq!(result, Value::Array(vec![Value::from("Manager")]));

        // LABELS on NULL returns NULL
        let result = eval_fn(ScalarFunction::Labels, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // LABELS with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Labels, vec![]);
        assert_eq!(result, Value::Null);

        // LABELS on node with no labels
        let node_no_labels = r#"{"_labels": [], "name": "Bob"}"#;
        let result = eval_fn(ScalarFunction::Labels, vec![Value::from(node_no_labels)]);
        assert_eq!(result, Value::Array(vec![]));
    }

    #[test]
    fn test_cypher_id() {
        use crate::plan::logical::ScalarFunction;

        // ID on an integer returns the integer
        let result = eval_fn(ScalarFunction::Id, vec![Value::Int(42)]);
        assert_eq!(result, Value::Int(42));

        // ID on a JSON object with _id key
        let entity = r#"{"_id": 123, "name": "Alice"}"#;
        let result = eval_fn(ScalarFunction::Id, vec![Value::from(entity)]);
        assert_eq!(result, Value::Int(123));

        // ID on a string that's a number
        let result = eval_fn(ScalarFunction::Id, vec![Value::from("456")]);
        assert_eq!(result, Value::Int(456));

        // ID on NULL returns NULL
        let result = eval_fn(ScalarFunction::Id, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // ID with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Id, vec![]);
        assert_eq!(result, Value::Null);

        // ID on a non-numeric string returns NULL
        let result = eval_fn(ScalarFunction::Id, vec![Value::from("not-an-id")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_properties() {
        use crate::plan::logical::ScalarFunction;

        // PROPERTIES on a JSON object (filters out internal keys)
        let entity = r#"{"_id": 1, "_labels": ["Person"], "name": "Alice", "age": 30}"#;
        let result = eval_fn(ScalarFunction::Properties, vec![Value::from(entity)]);
        // Should only contain name and age
        if let Value::String(s) = result {
            let json: serde_json::Value = serde_json::from_str(&s).unwrap();
            assert_eq!(json.get("name"), Some(&serde_json::Value::from("Alice")));
            assert_eq!(json.get("age"), Some(&serde_json::Value::from(30)));
            assert!(json.get("_id").is_none());
            assert!(json.get("_labels").is_none());
        } else {
            panic!("Expected string result");
        }

        // PROPERTIES on NULL returns NULL
        let result = eval_fn(ScalarFunction::Properties, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // PROPERTIES with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Properties, vec![]);
        assert_eq!(result, Value::Null);

        // PROPERTIES on non-JSON string returns empty object
        let result = eval_fn(ScalarFunction::Properties, vec![Value::from("not-json")]);
        assert_eq!(result, Value::from("{}"));

        // PROPERTIES on entity with no properties (only internal keys)
        let entity = r#"{"_id": 1, "_labels": ["Person"]}"#;
        let result = eval_fn(ScalarFunction::Properties, vec![Value::from(entity)]);
        assert_eq!(result, Value::from("{}"));
    }

    #[test]
    fn test_cypher_keys() {
        use crate::plan::logical::ScalarFunction;

        // KEYS on a JSON object (filters out internal keys)
        let entity = r#"{"_id": 1, "name": "Alice", "age": 30, "city": "NYC"}"#;
        let result = eval_fn(ScalarFunction::Keys, vec![Value::from(entity)]);
        if let Value::Array(keys) = result {
            assert_eq!(keys.len(), 3);
            // Keys may be in any order
            assert!(keys.contains(&Value::from("name")));
            assert!(keys.contains(&Value::from("age")));
            assert!(keys.contains(&Value::from("city")));
            // Should not contain internal keys
            assert!(!keys.contains(&Value::from("_id")));
        } else {
            panic!("Expected array result");
        }

        // KEYS on NULL returns NULL
        let result = eval_fn(ScalarFunction::Keys, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // KEYS with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Keys, vec![]);
        assert_eq!(result, Value::Null);

        // KEYS on non-JSON string returns empty array
        let result = eval_fn(ScalarFunction::Keys, vec![Value::from("not-json")]);
        assert_eq!(result, Value::Array(vec![]));

        // KEYS on array returns NULL (per Cypher semantics)
        let result = eval_fn(ScalarFunction::Keys, vec![Value::Array(vec![Value::Int(1)])]);
        assert_eq!(result, Value::Null);

        // KEYS on entity with only internal keys
        let entity = r#"{"_id": 1, "_labels": ["Person"]}"#;
        let result = eval_fn(ScalarFunction::Keys, vec![Value::from(entity)]);
        assert_eq!(result, Value::Array(vec![]));
    }

    // ========== Cypher Type Conversion Function Tests ==========

    #[test]
    fn test_to_boolean_from_string() {
        use crate::plan::logical::ScalarFunction;

        // toBoolean("true") = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("true")]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean("TRUE") = true (case-insensitive)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("TRUE")]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean("True") = true (case-insensitive)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("True")]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean("false") = false
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("false")]);
        assert_eq!(result, Value::Bool(false));

        // toBoolean("FALSE") = false (case-insensitive)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("FALSE")]);
        assert_eq!(result, Value::Bool(false));

        // toBoolean("invalid") = null (invalid string)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("invalid")]);
        assert_eq!(result, Value::Null);

        // toBoolean("") = null (empty string is invalid)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("")]);
        assert_eq!(result, Value::Null);

        // toBoolean("1") = null (numeric string is invalid for boolean)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::from("1")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_boolean_from_integer() {
        use crate::plan::logical::ScalarFunction;

        // toBoolean(0) = false
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Int(0)]);
        assert_eq!(result, Value::Bool(false));

        // toBoolean(1) = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Int(1)]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean(-1) = true (any non-zero integer)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Int(-1)]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean(42) = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Int(42)]);
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_to_boolean_from_float() {
        use crate::plan::logical::ScalarFunction;

        // toBoolean(0.0) = false
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Float(0.0)]);
        assert_eq!(result, Value::Bool(false));

        // toBoolean(1.0) = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Float(1.0)]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean(-0.1) = true (any non-zero float)
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Float(-0.1)]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean(3.14) = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Float(3.14)]);
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_to_boolean_from_boolean() {
        use crate::plan::logical::ScalarFunction;

        // toBoolean(true) = true
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Bool(true)]);
        assert_eq!(result, Value::Bool(true));

        // toBoolean(false) = false
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Bool(false)]);
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_to_boolean_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // toBoolean(null) = null
        let result = eval_fn(ScalarFunction::ToBoolean, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // toBoolean() with no args = null
        let result = eval_fn(ScalarFunction::ToBoolean, vec![]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_integer_from_string() {
        use crate::plan::logical::ScalarFunction;

        // toInteger("123") = 123
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("123")]);
        assert_eq!(result, Value::Int(123));

        // toInteger("-456") = -456
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("-456")]);
        assert_eq!(result, Value::Int(-456));

        // toInteger("  789  ") = 789 (whitespace trimmed)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("  789  ")]);
        assert_eq!(result, Value::Int(789));

        // toInteger("3.14") = 3 (truncates decimal)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("3.14")]);
        assert_eq!(result, Value::Int(3));

        // toInteger("-3.99") = -3 (truncates towards zero)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("-3.99")]);
        assert_eq!(result, Value::Int(-3));

        // toInteger("invalid") = null
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("invalid")]);
        assert_eq!(result, Value::Null);

        // toInteger("") = null (empty string)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::from("")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_integer_from_float() {
        use crate::plan::logical::ScalarFunction;

        // toInteger(3.14) = 3 (truncates)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Float(3.14)]);
        assert_eq!(result, Value::Int(3));

        // toInteger(3.99) = 3 (truncates, not rounds)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Float(3.99)]);
        assert_eq!(result, Value::Int(3));

        // toInteger(-3.14) = -3 (truncates towards zero)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Float(-3.14)]);
        assert_eq!(result, Value::Int(-3));

        // toInteger(-3.99) = -3 (truncates towards zero)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Float(-3.99)]);
        assert_eq!(result, Value::Int(-3));

        // toInteger(5.0) = 5
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Float(5.0)]);
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_to_integer_from_boolean() {
        use crate::plan::logical::ScalarFunction;

        // toInteger(true) = 1
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Bool(true)]);
        assert_eq!(result, Value::Int(1));

        // toInteger(false) = 0
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Bool(false)]);
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_to_integer_from_integer() {
        use crate::plan::logical::ScalarFunction;

        // toInteger(42) = 42 (identity)
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Int(42)]);
        assert_eq!(result, Value::Int(42));

        // toInteger(-100) = -100
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Int(-100)]);
        assert_eq!(result, Value::Int(-100));
    }

    #[test]
    fn test_to_integer_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // toInteger(null) = null
        let result = eval_fn(ScalarFunction::ToInteger, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // toInteger() with no args = null
        let result = eval_fn(ScalarFunction::ToInteger, vec![]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_float_from_string() {
        use crate::plan::logical::ScalarFunction;

        // toFloat("3.14") = 3.14
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("3.14")]);
        assert_eq!(result, Value::Float(3.14));

        // toFloat("-2.5") = -2.5
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("-2.5")]);
        assert_eq!(result, Value::Float(-2.5));

        // toFloat("  1.0  ") = 1.0 (whitespace trimmed)
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("  1.0  ")]);
        assert_eq!(result, Value::Float(1.0));

        // toFloat("100") = 100.0 (integer string to float)
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("100")]);
        assert_eq!(result, Value::Float(100.0));

        // toFloat("invalid") = null
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("invalid")]);
        assert_eq!(result, Value::Null);

        // toFloat("") = null
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::from("")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_float_from_integer() {
        use crate::plan::logical::ScalarFunction;

        // toFloat(3) = 3.0
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Int(3)]);
        assert_eq!(result, Value::Float(3.0));

        // toFloat(-10) = -10.0
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Int(-10)]);
        assert_eq!(result, Value::Float(-10.0));

        // toFloat(0) = 0.0
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Int(0)]);
        assert_eq!(result, Value::Float(0.0));
    }

    #[test]
    fn test_to_float_from_boolean() {
        use crate::plan::logical::ScalarFunction;

        // toFloat(true) = 1.0
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Bool(true)]);
        assert_eq!(result, Value::Float(1.0));

        // toFloat(false) = 0.0
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Bool(false)]);
        assert_eq!(result, Value::Float(0.0));
    }

    #[test]
    fn test_to_float_from_float() {
        use crate::plan::logical::ScalarFunction;

        // toFloat(3.14) = 3.14 (identity)
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Float(3.14)]);
        assert_eq!(result, Value::Float(3.14));
    }

    #[test]
    fn test_to_float_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // toFloat(null) = null
        let result = eval_fn(ScalarFunction::ToFloat, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // toFloat() with no args = null
        let result = eval_fn(ScalarFunction::ToFloat, vec![]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_to_string_from_integer() {
        use crate::plan::logical::ScalarFunction;

        // toString(123) = "123"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Int(123)]);
        assert_eq!(result, Value::String("123".to_string()));

        // toString(-456) = "-456"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Int(-456)]);
        assert_eq!(result, Value::String("-456".to_string()));

        // toString(0) = "0"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Int(0)]);
        assert_eq!(result, Value::String("0".to_string()));
    }

    #[test]
    fn test_to_string_from_float() {
        use crate::plan::logical::ScalarFunction;

        // toString(3.14) = "3.14"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Float(3.14)]);
        assert_eq!(result, Value::String("3.14".to_string()));

        // toString(3.0) = "3.0" (keeps one decimal for floats)
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Float(3.0)]);
        assert_eq!(result, Value::String("3.0".to_string()));

        // toString(-2.5) = "-2.5"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Float(-2.5)]);
        assert_eq!(result, Value::String("-2.5".to_string()));
    }

    #[test]
    fn test_to_string_from_boolean() {
        use crate::plan::logical::ScalarFunction;

        // toString(true) = "true"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Bool(true)]);
        assert_eq!(result, Value::String("true".to_string()));

        // toString(false) = "false"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Bool(false)]);
        assert_eq!(result, Value::String("false".to_string()));
    }

    #[test]
    fn test_to_string_from_string() {
        use crate::plan::logical::ScalarFunction;

        // toString("hello") = "hello" (identity)
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::from("hello")]);
        assert_eq!(result, Value::String("hello".to_string()));

        // toString("") = "" (empty string)
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::from("")]);
        assert_eq!(result, Value::String(String::new()));
    }

    #[test]
    fn test_to_string_from_array() {
        use crate::plan::logical::ScalarFunction;

        // toString([1, 2, 3]) = "[1, 2, 3]"
        let result = eval_fn(
            ScalarFunction::CypherToString,
            vec![Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)])],
        );
        assert_eq!(result, Value::String("[1, 2, 3]".to_string()));

        // toString([]) = "[]"
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Array(vec![])]);
        assert_eq!(result, Value::String("[]".to_string()));
    }

    #[test]
    fn test_to_string_null_handling() {
        use crate::plan::logical::ScalarFunction;

        // toString(null) = null
        let result = eval_fn(ScalarFunction::CypherToString, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // toString() with no args = null
        let result = eval_fn(ScalarFunction::CypherToString, vec![]);
        assert_eq!(result, Value::Null);
    }

    // ========== Cypher Path Function Tests ==========

    #[test]
    fn test_cypher_nodes() {
        use crate::plan::logical::ScalarFunction;

        // nodes() on a JSON path object with _nodes key
        let path = r#"{"_nodes": [1, 2, 3]}"#;
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]));

        // nodes() on a JSON path object with path_nodes key (internal format)
        let path = r#"{"path_nodes": [10, 20, 30]}"#;
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]));

        // nodes() on a JSON array directly
        let path = r"[100, 200, 300]";
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(100), Value::Int(200), Value::Int(300)]));

        // nodes() on an array value directly
        let nodes = Value::Array(vec![Value::Int(1), Value::Int(2)]);
        let result = eval_fn(ScalarFunction::Nodes, vec![nodes.clone()]);
        assert_eq!(result, nodes);

        // nodes() on NULL returns NULL
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // nodes() with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Nodes, vec![]);
        assert_eq!(result, Value::Null);

        // nodes() on invalid JSON returns NULL
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::from("not-json")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_relationships() {
        use crate::plan::logical::ScalarFunction;

        // relationships() on a JSON path object with _edges key
        let path = r#"{"_edges": [101, 102]}"#;
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(101), Value::Int(102)]));

        // relationships() on a JSON path object with _relationships key
        let path = r#"{"_relationships": [201, 202, 203]}"#;
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(201), Value::Int(202), Value::Int(203)]));

        // relationships() on a JSON path object with path_edges key (internal format)
        let path = r#"{"path_edges": [301, 302]}"#;
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(301), Value::Int(302)]));

        // relationships() on a JSON array directly
        let path = r"[401, 402]";
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from(path)]);
        assert_eq!(result, Value::Array(vec![Value::Int(401), Value::Int(402)]));

        // relationships() on an array value directly
        let edges = Value::Array(vec![Value::Int(1), Value::Int(2)]);
        let result = eval_fn(ScalarFunction::Relationships, vec![edges.clone()]);
        assert_eq!(result, edges);

        // relationships() on NULL returns NULL
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // relationships() with no arguments returns NULL
        let result = eval_fn(ScalarFunction::Relationships, vec![]);
        assert_eq!(result, Value::Null);

        // relationships() on invalid JSON returns NULL
        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from("not-json")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_start_node() {
        use crate::plan::logical::ScalarFunction;

        // startNode() on a relationship with _source key
        let rel = r#"{"_source": 1, "_target": 2, "_edge_type": "KNOWS"}"#;
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(1));

        // startNode() on a relationship with _start key
        let rel = r#"{"_start": 10, "_end": 20}"#;
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(10));

        // startNode() on a relationship with source key (user-friendly)
        let rel = r#"{"source": 100, "target": 200}"#;
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(100));

        // startNode() on a relationship with start key (user-friendly)
        let rel = r#"{"start": 1000, "end": 2000}"#;
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(1000));

        // startNode() on NULL returns NULL
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // startNode() with no arguments returns NULL
        let result = eval_fn(ScalarFunction::StartNode, vec![]);
        assert_eq!(result, Value::Null);

        // startNode() on invalid JSON returns NULL
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from("not-json")]);
        assert_eq!(result, Value::Null);

        // startNode() on JSON without source field returns NULL
        let rel = r#"{"_edge_type": "KNOWS"}"#;
        let result = eval_fn(ScalarFunction::StartNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_end_node() {
        use crate::plan::logical::ScalarFunction;

        // endNode() on a relationship with _target key
        let rel = r#"{"_source": 1, "_target": 2, "_edge_type": "KNOWS"}"#;
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(2));

        // endNode() on a relationship with _end key
        let rel = r#"{"_start": 10, "_end": 20}"#;
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(20));

        // endNode() on a relationship with target key (user-friendly)
        let rel = r#"{"source": 100, "target": 200}"#;
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(200));

        // endNode() on a relationship with end key (user-friendly)
        let rel = r#"{"start": 1000, "end": 2000}"#;
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Int(2000));

        // endNode() on NULL returns NULL
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // endNode() with no arguments returns NULL
        let result = eval_fn(ScalarFunction::EndNode, vec![]);
        assert_eq!(result, Value::Null);

        // endNode() on invalid JSON returns NULL
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from("not-json")]);
        assert_eq!(result, Value::Null);

        // endNode() on JSON without target field returns NULL
        let rel = r#"{"_edge_type": "KNOWS"}"#;
        let result = eval_fn(ScalarFunction::EndNode, vec![Value::from(rel)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_path_length() {
        use crate::plan::logical::ScalarFunction;

        // length() on a JSON path object with _edges key
        let path = r#"{"_edges": [1, 2, 3]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(3));

        // length() on a JSON path object with _relationships key
        let path = r#"{"_relationships": [1, 2]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(2));

        // length() on a JSON path object with path_edges key (internal format)
        let path = r#"{"path_edges": [1, 2, 3, 4]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(4));

        // length() on a JSON path object with _nodes key (length = nodes - 1)
        let path = r#"{"_nodes": [1, 2, 3, 4]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(3)); // 4 nodes = 3 edges

        // length() on a JSON path object with path_nodes key (internal format)
        let path = r#"{"path_nodes": [1, 2]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(1)); // 2 nodes = 1 edge

        // length() on a JSON path with single node (length = 0)
        let path = r#"{"_nodes": [1]}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(0));

        // length() on a JSON path with empty nodes (length = 0)
        let path = r#"{"_nodes": []}"#;
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(0));

        // length() on a JSON array directly (assumed to be edges)
        let path = r"[1, 2, 3]";
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path)]);
        assert_eq!(result, Value::Int(3));

        // length() on an array value directly
        let edges = Value::Array(vec![Value::Int(1), Value::Int(2)]);
        let result = eval_fn(ScalarFunction::PathLength, vec![edges]);
        assert_eq!(result, Value::Int(2));

        // length() on a non-JSON string returns string length (like LENGTH function)
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from("hello")]);
        assert_eq!(result, Value::Int(5));

        // length() on NULL returns NULL
        let result = eval_fn(ScalarFunction::PathLength, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        // length() with no arguments returns NULL
        let result = eval_fn(ScalarFunction::PathLength, vec![]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_path_functions_with_complex_data() {
        use crate::plan::logical::ScalarFunction;

        // Test with a more complex path structure (nested objects as node/edge data)
        let path_with_nested =
            r#"{"_nodes": [{"id": 1}, {"id": 2}], "_edges": [{"type": "KNOWS"}]}"#;
        let result = eval_fn(ScalarFunction::Nodes, vec![Value::from(path_with_nested)]);
        if let Value::Array(nodes) = result {
            assert_eq!(nodes.len(), 2);
            // Node objects are returned as JSON strings
            assert!(matches!(nodes[0], Value::String(_)));
            assert!(matches!(nodes[1], Value::String(_)));
        } else {
            panic!("Expected array result");
        }

        let result = eval_fn(ScalarFunction::Relationships, vec![Value::from(path_with_nested)]);
        if let Value::Array(edges) = result {
            assert_eq!(edges.len(), 1);
            // Edge object is returned as JSON string
            assert!(matches!(edges[0], Value::String(_)));
        } else {
            panic!("Expected array result");
        }

        let result = eval_fn(ScalarFunction::PathLength, vec![Value::from(path_with_nested)]);
        assert_eq!(result, Value::Int(1));
    }

    // ========== EXISTS Subquery Evaluation Tests ==========

    /// Mock graph accessor for testing EXISTS subqueries.
    ///
    /// This mock provides controlled graph traversal results for testing
    /// the EXISTS execution logic without requiring actual graph storage.
    struct MockGraphAccessor {
        /// Neighbors to return for each node. Key is source node ID.
        neighbors: std::collections::HashMap<u64, Vec<(u64, u64)>>, // (neighbor_id, edge_id)
    }

    impl MockGraphAccessor {
        fn new() -> Self {
            Self { neighbors: std::collections::HashMap::new() }
        }

        fn add_edge(&mut self, from: u64, to: u64, edge_id: u64) {
            self.neighbors.entry(from).or_default().push((to, edge_id));
        }
    }

    impl crate::exec::graph_accessor::GraphAccessor for MockGraphAccessor {
        fn neighbors(
            &self,
            node: EntityId,
            direction: Direction,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::NeighborResult>,
        > {
            let results = self.neighbors.get(&node.as_u64()).map(|edges| {
                edges
                    .iter()
                    .map(|(neighbor, edge_id)| {
                        crate::exec::graph_accessor::NeighborResult::new(
                            EntityId::new(*neighbor),
                            manifoldb_core::EdgeId::new(*edge_id),
                            direction,
                        )
                    })
                    .collect()
            });
            Ok(results.unwrap_or_default())
        }

        fn neighbors_by_type(
            &self,
            node: EntityId,
            direction: Direction,
            _edge_type: &manifoldb_core::EdgeType,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::NeighborResult>,
        > {
            // For testing, treat all edges as matching the type
            self.neighbors(node, direction)
        }

        fn neighbors_by_types(
            &self,
            node: EntityId,
            direction: Direction,
            _edge_types: &[manifoldb_core::EdgeType],
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::NeighborResult>,
        > {
            // For testing, treat all edges as matching the types
            self.neighbors(node, direction)
        }

        fn expand_all(
            &self,
            _node: EntityId,
            _direction: Direction,
            _min_depth: usize,
            _max_depth: Option<usize>,
            _edge_types: Option<&[manifoldb_core::EdgeType]>,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::TraversalResult>,
        > {
            // For testing, return empty - variable length is tested separately
            Ok(vec![])
        }

        fn find_paths(
            &self,
            _start: EntityId,
            _config: &crate::exec::graph_accessor::PathFindConfig,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::PathMatchResult>,
        > {
            Ok(vec![])
        }

        fn shortest_path(
            &self,
            _source: EntityId,
            _target: EntityId,
            _config: &crate::exec::graph_accessor::ShortestPathConfig,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::ShortestPathResult>,
        > {
            Ok(vec![])
        }

        fn get_entity_properties(
            &self,
            _entity_id: EntityId,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Option<std::collections::HashMap<String, manifoldb_core::Value>>,
        > {
            Ok(Some(std::collections::HashMap::new()))
        }

        fn get_edge_properties(
            &self,
            _edge_id: manifoldb_core::EdgeId,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Option<std::collections::HashMap<String, manifoldb_core::Value>>,
        > {
            Ok(Some(std::collections::HashMap::new()))
        }

        fn entity_has_labels(
            &self,
            _entity_id: EntityId,
            _labels: &[String],
        ) -> crate::exec::graph_accessor::GraphAccessResult<bool> {
            Ok(true)
        }

        fn scan_nodes(
            &self,
            _label: Option<&str>,
        ) -> crate::exec::graph_accessor::GraphAccessResult<
            Vec<crate::exec::graph_accessor::NodeScanResult>,
        > {
            Ok(vec![])
        }
    }

    #[test]
    fn test_exists_subquery_no_graph() {
        // When no graph accessor is provided, EXISTS should return false
        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let expr =
            LogicalExpr::ExistsSubquery { expand_steps: vec![expand_step], filter_predicate: None };

        // Without graph accessor, should return false
        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_with_match() {
        // Test EXISTS { (n)-[:KNOWS]->(friend) } when path exists
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 KNOWS Node 2 via edge 100

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_subquery_no_match() {
        // Test EXISTS { (n)-[:KNOWS]->(friend) } when no path exists
        let mock_graph = MockGraphAccessor::new(); // Empty graph

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_with_filter_pass() {
        // Test EXISTS { (n)-[:KNOWS]->(friend) WHERE friend > 0 }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > 0 (friend will be bound to node 2, which is > 0)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::integer(0));

        let result =
            evaluate_exists_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_subquery_with_filter_fail() {
        // Test EXISTS { (n)-[:KNOWS]->(friend) WHERE friend > 100 }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Friend is node 2

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > 100 (friend will be bound to node 2, which is NOT > 100)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::integer(100));

        let result =
            evaluate_exists_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_multi_hop() {
        // Test EXISTS { (n)-[:KNOWS]->(m)-[:KNOWS]->(friend) }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 -> Node 2
        mock_graph.add_edge(2, 3, 101); // Node 2 -> Node 3

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let step1 = ExpandNode::outgoing("n", "m").with_edge_type("KNOWS");
        let step2 = ExpandNode::outgoing("m", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[step1, step2], None, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_subquery_multi_hop_no_path() {
        // Test EXISTS { (n)-[:KNOWS]->(m)-[:KNOWS]->(friend) } when second hop fails
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 -> Node 2
                                        // No edge from Node 2

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let step1 = ExpandNode::outgoing("n", "m").with_edge_type("KNOWS");
        let step2 = ExpandNode::outgoing("m", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[step1, step2], None, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_empty_pattern() {
        // Empty pattern should return false
        let mock_graph = MockGraphAccessor::new();

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let result = evaluate_exists_subquery(&mock_graph, &[], None, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_missing_source_var() {
        // When source variable is not in the row, should return false
        let mock_graph = MockGraphAccessor::new();

        let schema = Arc::new(Schema::new(vec!["other".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn test_exists_subquery_multiple_neighbors() {
        // Test short-circuit: should return true on first match
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Multiple neighbors
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(1, 4, 102);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_exists_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_subquery_correlated_filter() {
        // Test correlated EXISTS using outer variable in filter
        // EXISTS { (n)-[:KNOWS]->(friend) WHERE friend > n }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 -> Node 2

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > n (2 > 1 should be true)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::column("n"));

        let result =
            evaluate_exists_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_subquery_edge_variable_binding() {
        // Test that edge variables are bound correctly
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 999); // Edge ID 999

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step =
            ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS").with_edge_var("r");

        // Filter: r = 999 (checking edge ID)
        let filter = LogicalExpr::column("r").eq(LogicalExpr::integer(999));

        let result =
            evaluate_exists_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_evaluate_expr_with_graph_basic() {
        // Test that evaluate_expr_with_graph works for non-graph expressions
        let schema = Arc::new(Schema::new(vec!["x".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(10)]);

        let expr = LogicalExpr::column("x").gt(LogicalExpr::integer(5));
        let result = evaluate_expr_with_graph(&expr, &row, None).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn test_exists_in_filter_operator() {
        use crate::exec::context::ExecutionContext;

        // Create a mock graph
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 10, 100); // Node 1 has neighbor
                                         // Node 2 has no neighbors

        // Create input with two rows
        let input = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)], // Should pass EXISTS
                vec![Value::Int(2)], // Should fail EXISTS
            ],
        ));

        // Create EXISTS predicate
        let expand_step = ExpandNode::outgoing("n", "friend");
        let predicate =
            LogicalExpr::ExistsSubquery { expand_steps: vec![expand_step], filter_predicate: None };

        let mut filter_op = FilterOp::new(predicate, input);

        // Set up context with mock graph
        let ctx = ExecutionContext::new().with_graph(Arc::new(mock_graph));

        filter_op.open(&ctx).unwrap();

        // Only the first row (n=1) should pass
        let row1 = filter_op.next().unwrap();
        assert!(row1.is_some());
        assert_eq!(row1.unwrap().get_by_name("n"), Some(&Value::Int(1)));

        // Second row (n=2) should be filtered out
        let row2 = filter_op.next().unwrap();
        assert!(row2.is_none());

        filter_op.close().unwrap();
    }

    // ========== COUNT Subquery Evaluation Tests ==========

    #[test]
    fn test_count_subquery_no_graph() {
        // When no graph accessor is provided, COUNT should return 0
        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let expr =
            LogicalExpr::CountSubquery { expand_steps: vec![expand_step], filter_predicate: None };

        // Without graph accessor, should return 0
        let result = evaluate_expr(&expr, &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_count_subquery_single_match() {
        // Test COUNT { (n)-[:KNOWS]->(friend) } with one neighbor
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 KNOWS Node 2

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_count_subquery_multiple_matches() {
        // Test COUNT { (n)-[:KNOWS]->(friend) } with multiple neighbors
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // Node 1 -> Node 2
        mock_graph.add_edge(1, 3, 101); // Node 1 -> Node 3
        mock_graph.add_edge(1, 4, 102); // Node 1 -> Node 4

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Int(3)); // Should count all 3 neighbors
    }

    #[test]
    fn test_count_subquery_no_match() {
        // Test COUNT { (n)-[:KNOWS]->(friend) } with no neighbors
        let mock_graph = MockGraphAccessor::new(); // Empty graph

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_count_subquery_with_filter_pass() {
        // Test COUNT { (n)-[:KNOWS]->(friend) WHERE friend > 0 }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(1, 4, 102);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > 0 (all friends 2, 3, 4 are > 0)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::integer(0));

        let result =
            evaluate_count_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Int(3)); // All 3 should pass
    }

    #[test]
    fn test_count_subquery_with_filter_partial() {
        // Test COUNT { (n)-[:KNOWS]->(friend) WHERE friend > 2 }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // friend=2, NOT > 2
        mock_graph.add_edge(1, 3, 101); // friend=3, IS > 2
        mock_graph.add_edge(1, 4, 102); // friend=4, IS > 2

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > 2 (only friends 3 and 4 match)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::integer(2));

        let result =
            evaluate_count_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Int(2)); // Only 2 pass the filter
    }

    #[test]
    fn test_count_subquery_with_filter_none_pass() {
        // Test COUNT { (n)-[:KNOWS]->(friend) WHERE friend > 100 }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > 100 (no friends match)
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::integer(100));

        let result =
            evaluate_count_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_count_subquery_multi_hop() {
        // Test COUNT { (n)-[:KNOWS]->(m)-[:KNOWS]->(friend) }
        // Graph: 1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5, 3 -> 6
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(2, 4, 102);
        mock_graph.add_edge(2, 5, 103);
        mock_graph.add_edge(3, 6, 104);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let step1 = ExpandNode::outgoing("n", "m").with_edge_type("KNOWS");
        let step2 = ExpandNode::outgoing("m", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[step1, step2], None, &row).unwrap();
        // Paths: 1->2->4, 1->2->5, 1->3->6 = 3 matches
        assert_eq!(result, Value::Int(3));
    }

    #[test]
    fn test_count_subquery_multi_hop_partial() {
        // Test COUNT { (n)-[:KNOWS]->(m)-[:KNOWS]->(friend) } when some hops fail
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(2, 4, 102); // Only node 2 has second-hop neighbors

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let step1 = ExpandNode::outgoing("n", "m").with_edge_type("KNOWS");
        let step2 = ExpandNode::outgoing("m", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[step1, step2], None, &row).unwrap();
        // Only path 1->2->4 completes
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_count_subquery_empty_pattern() {
        // Empty pattern should return 0
        let mock_graph = MockGraphAccessor::new();

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let result = evaluate_count_subquery(&mock_graph, &[], None, &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_count_subquery_missing_source_var() {
        // When source variable is not in the row, should return 0
        let mock_graph = MockGraphAccessor::new();

        let schema = Arc::new(Schema::new(vec!["other".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        let result = evaluate_count_subquery(&mock_graph, &[expand_step], None, &row).unwrap();
        assert_eq!(result, Value::Int(0));
    }

    #[test]
    fn test_count_subquery_correlated_filter() {
        // Test correlated COUNT using outer variable in filter
        // COUNT { (n)-[:KNOWS]->(friend) WHERE friend > n }
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100); // 2 > 1, passes
        mock_graph.add_edge(1, 0, 101); // 0 > 1, fails

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // Filter: friend > n
        let filter = LogicalExpr::column("friend").gt(LogicalExpr::column("n"));

        let result =
            evaluate_count_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Int(1)); // Only friend=2 passes
    }

    #[test]
    fn test_count_subquery_edge_variable_binding() {
        // Test that edge variables are bound correctly
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 999); // Edge ID 999
        mock_graph.add_edge(1, 3, 888); // Edge ID 888

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step =
            ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS").with_edge_var("r");

        // Filter: r = 999 (only one edge has this ID)
        let filter = LogicalExpr::column("r").eq(LogicalExpr::integer(999));

        let result =
            evaluate_count_subquery(&mock_graph, &[expand_step], Some(&filter), &row).unwrap();
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_count_in_filter_operator_where_clause() {
        // Test COUNT { } > 5 in WHERE clause
        use crate::exec::context::ExecutionContext;

        let mut mock_graph = MockGraphAccessor::new();
        // Node 1 has 6 neighbors (should pass COUNT > 5)
        for i in 10..16 {
            mock_graph.add_edge(1, i, 100 + i);
        }
        // Node 2 has 3 neighbors (should fail COUNT > 5)
        for i in 20..23 {
            mock_graph.add_edge(2, i, 100 + i);
        }

        let input = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)], // 6 neighbors, should pass
                vec![Value::Int(2)], // 3 neighbors, should fail
            ],
        ));

        // Predicate: COUNT { (n)-[:KNOWS]->(friend) } > 5
        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");
        let count_expr =
            LogicalExpr::CountSubquery { expand_steps: vec![expand_step], filter_predicate: None };
        let predicate = count_expr.gt(LogicalExpr::integer(5));

        let mut filter_op = FilterOp::new(predicate, input);

        let ctx = ExecutionContext::new().with_graph(Arc::new(mock_graph));
        filter_op.open(&ctx).unwrap();

        // Only node 1 should pass
        let row1 = filter_op.next().unwrap();
        assert!(row1.is_some());
        assert_eq!(row1.unwrap().get_by_name("n"), Some(&Value::Int(1)));

        // Node 2 should be filtered out
        let row2 = filter_op.next().unwrap();
        assert!(row2.is_none());

        filter_op.close().unwrap();
    }

    #[test]
    fn test_count_subquery_in_complex_expression() {
        // Test COUNT { } used in arithmetic expression
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(1, 4, 102);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");
        let count_expr =
            LogicalExpr::CountSubquery { expand_steps: vec![expand_step], filter_predicate: None };

        // Test: COUNT { ... } = 3
        let eq_expr = count_expr.eq(LogicalExpr::integer(3));

        let result = evaluate_expr_with_graph(&eq_expr, &row, Some(&mock_graph)).unwrap();
        assert_eq!(result, Value::Bool(true)); // COUNT is 3, equals 3
    }

    #[test]
    fn test_count_vs_exists_difference() {
        // Demonstrate COUNT counts all, while EXISTS short-circuits
        let mut mock_graph = MockGraphAccessor::new();
        mock_graph.add_edge(1, 2, 100);
        mock_graph.add_edge(1, 3, 101);
        mock_graph.add_edge(1, 4, 102);

        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1)]);

        let expand_step = ExpandNode::outgoing("n", "friend").with_edge_type("KNOWS");

        // EXISTS returns bool, COUNT returns integer
        let exists_result =
            evaluate_exists_subquery(&mock_graph, std::slice::from_ref(&expand_step), None, &row)
                .unwrap();
        let count_result =
            evaluate_count_subquery(&mock_graph, std::slice::from_ref(&expand_step), None, &row)
                .unwrap();

        assert_eq!(exists_result, Value::Bool(true));
        assert_eq!(count_result, Value::Int(3));
    }

    // ========== Cypher Temporal Function Tests ==========

    #[test]
    fn test_cypher_datetime_no_args() {
        // datetime() without args returns current datetime
        let result = eval_fn(ScalarFunction::CypherDatetime, vec![]);
        if let Value::String(s) = result {
            // Should be ISO 8601 format with timezone
            assert!(s.contains('T'), "datetime should contain T separator");
            assert!(s.contains('+') || s.contains('Z'), "datetime should have timezone");
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_from_string() {
        // datetime('2024-01-15T10:30:00') - parse ISO 8601 string
        let result = eval_fn(
            ScalarFunction::CypherDatetime,
            vec![Value::String("2024-01-15T10:30:00".to_string())],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T10:30:00"));
        } else {
            panic!("Expected string result, got: {:?}", result);
        }

        // With timezone
        let result = eval_fn(
            ScalarFunction::CypherDatetime,
            vec![Value::String("2024-01-15T10:30:00Z".to_string())],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T10:30:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_from_map() {
        // datetime({year: 2024, month: 1, day: 15}) - construct from JSON map
        let result = eval_fn(
            ScalarFunction::CypherDatetime,
            vec![Value::String(
                r#"{"year": 2024, "month": 1, "day": 15, "hour": 10, "minute": 30}"#.to_string(),
            )],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T10:30:00"));
        } else {
            panic!("Expected string result, got: {:?}", result);
        }
    }

    #[test]
    fn test_cypher_datetime_null() {
        let result = eval_fn(ScalarFunction::CypherDatetime, vec![Value::Null]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_date_no_args() {
        // date() without args returns current date
        let result = eval_fn(ScalarFunction::CypherDate, vec![]);
        if let Value::String(s) = result {
            // Should be YYYY-MM-DD format
            assert!(s.len() == 10, "date should be 10 chars");
            assert!(s.contains('-'), "date should contain dashes");
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_date_from_string() {
        // date('2024-01-15') - parse ISO 8601 date string
        let result =
            eval_fn(ScalarFunction::CypherDate, vec![Value::String("2024-01-15".to_string())]);
        assert_eq!(result, Value::String("2024-01-15".to_string()));

        // Should also work with datetime string
        let result = eval_fn(
            ScalarFunction::CypherDate,
            vec![Value::String("2024-06-20T14:30:00".to_string())],
        );
        assert_eq!(result, Value::String("2024-06-20".to_string()));
    }

    #[test]
    fn test_cypher_date_from_map() {
        let result = eval_fn(
            ScalarFunction::CypherDate,
            vec![Value::String(r#"{"year": 2024, "month": 6, "day": 20}"#.to_string())],
        );
        assert_eq!(result, Value::String("2024-06-20".to_string()));
    }

    #[test]
    fn test_cypher_time_no_args() {
        // time() without args returns current time with timezone
        let result = eval_fn(ScalarFunction::CypherTime, vec![]);
        if let Value::String(s) = result {
            assert!(s.contains(':'), "time should contain colons");
            assert!(s.contains('+'), "time should have timezone offset");
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_time_from_string() {
        // time('10:30:00') - parse ISO 8601 time string
        let result =
            eval_fn(ScalarFunction::CypherTime, vec![Value::String("10:30:00".to_string())]);
        if let Value::String(s) = result {
            assert!(s.starts_with("10:30:00"));
        } else {
            panic!("Expected string result");
        }

        // With just hour:minute
        let result = eval_fn(ScalarFunction::CypherTime, vec![Value::String("14:45".to_string())]);
        if let Value::String(s) = result {
            assert!(s.starts_with("14:45:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_time_from_map() {
        let result = eval_fn(
            ScalarFunction::CypherTime,
            vec![Value::String(r#"{"hour": 14, "minute": 30, "second": 45}"#.to_string())],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("14:30:45"));
        } else {
            panic!("Expected string result, got: {:?}", result);
        }
    }

    #[test]
    fn test_cypher_localdatetime_no_args() {
        // localdatetime() without args returns current datetime without timezone
        let result = eval_fn(ScalarFunction::CypherLocalDatetime, vec![]);
        if let Value::String(s) = result {
            assert!(s.contains('T'), "localdatetime should contain T separator");
            // Should NOT have timezone
            assert!(
                !s.contains("+00:00") || s.ends_with(".000000"),
                "localdatetime should not end with timezone"
            );
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_localdatetime_from_string() {
        let result = eval_fn(
            ScalarFunction::CypherLocalDatetime,
            vec![Value::String("2024-01-15T10:30:00".to_string())],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T10:30:00"));
            // Should not have timezone
            assert!(!s.ends_with("+00:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_localtime_no_args() {
        let result = eval_fn(ScalarFunction::CypherLocalTime, vec![]);
        if let Value::String(s) = result {
            assert!(s.contains(':'), "localtime should contain colons");
            // Should NOT have timezone
            assert!(!s.ends_with("+00:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_localtime_from_string() {
        let result =
            eval_fn(ScalarFunction::CypherLocalTime, vec![Value::String("10:30:00".to_string())]);
        if let Value::String(s) = result {
            assert!(s.starts_with("10:30:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_duration_iso8601() {
        // Test various ISO 8601 duration formats

        // P1Y - 1 year
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("P1Y".to_string())]);
        assert_eq!(result, Value::String("P1Y".to_string()));

        // P2M - 2 months
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("P2M".to_string())]);
        assert_eq!(result, Value::String("P2M".to_string()));

        // P3D - 3 days
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("P3D".to_string())]);
        assert_eq!(result, Value::String("P3D".to_string()));

        // PT4H - 4 hours
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("PT4H".to_string())]);
        assert_eq!(result, Value::String("PT4H".to_string()));

        // PT5M - 5 minutes
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("PT5M".to_string())]);
        assert_eq!(result, Value::String("PT5M".to_string()));

        // PT6S - 6 seconds
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("PT6S".to_string())]);
        assert_eq!(result, Value::String("PT6S".to_string()));

        // Combined: P1Y2M3DT4H5M6S
        let result = eval_fn(
            ScalarFunction::CypherDuration,
            vec![Value::String("P1Y2M3DT4H5M6S".to_string())],
        );
        assert_eq!(result, Value::String("P1Y2M3DT4H5M6S".to_string()));
    }

    #[test]
    fn test_cypher_duration_with_weeks() {
        // P2W - 2 weeks (should be converted to 14 days)
        let result =
            eval_fn(ScalarFunction::CypherDuration, vec![Value::String("P2W".to_string())]);
        assert_eq!(result, Value::String("P14D".to_string()));
    }

    #[test]
    fn test_cypher_duration_from_map() {
        // duration({days: 14, hours: 16})
        let result = eval_fn(
            ScalarFunction::CypherDuration,
            vec![Value::String(r#"{"days": 14, "hours": 16}"#.to_string())],
        );
        assert_eq!(result, Value::String("P14DT16H".to_string()));

        // duration({years: 1, months: 2, days: 3})
        let result = eval_fn(
            ScalarFunction::CypherDuration,
            vec![Value::String(r#"{"years": 1, "months": 2, "days": 3}"#.to_string())],
        );
        assert_eq!(result, Value::String("P1Y2M3D".to_string()));

        // duration({weeks: 2, days: 3}) - weeks converted to days
        let result = eval_fn(
            ScalarFunction::CypherDuration,
            vec![Value::String(r#"{"weeks": 2, "days": 3}"#.to_string())],
        );
        assert_eq!(result, Value::String("P17D".to_string())); // 2*7 + 3 = 17
    }

    #[test]
    fn test_cypher_duration_null() {
        let result = eval_fn(ScalarFunction::CypherDuration, vec![Value::Null]);
        assert_eq!(result, Value::Null);

        let result = eval_fn(ScalarFunction::CypherDuration, vec![]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_datetime_truncate_day() {
        // datetime.truncate('day', datetime) - truncate to start of day
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("day".to_string()),
                Value::String("2024-01-15T14:30:45".to_string()),
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T00:00:00"));
        } else {
            panic!("Expected string result, got: {:?}", result);
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_month() {
        // datetime.truncate('month', datetime) - truncate to start of month
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("month".to_string()),
                Value::String("2024-06-15T14:30:45".to_string()),
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-06-01T00:00:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_year() {
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("year".to_string()),
                Value::String("2024-06-15T14:30:45".to_string()),
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-01T00:00:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_hour() {
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("hour".to_string()),
                Value::String("2024-01-15T14:30:45".to_string()),
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T14:00:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_minute() {
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("minute".to_string()),
                Value::String("2024-01-15T14:30:45".to_string()),
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T14:30:00"));
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_week() {
        // Truncate to Monday of the week
        // 2024-01-15 is a Monday
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("week".to_string()),
                Value::String("2024-01-17T14:30:45".to_string()), // Wednesday
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-01-15T00:00:00")); // Should be Monday
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_quarter() {
        // Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("quarter".to_string()),
                Value::String("2024-05-15T14:30:45".to_string()), // May = Q2
            ],
        );
        if let Value::String(s) = result {
            assert!(s.starts_with("2024-04-01T00:00:00")); // Q2 starts April 1
        } else {
            panic!("Expected string result");
        }
    }

    #[test]
    fn test_cypher_datetime_truncate_null() {
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![Value::Null, Value::String("2024-01-15T14:30:45".to_string())],
        );
        assert_eq!(result, Value::Null);

        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![Value::String("day".to_string()), Value::Null],
        );
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_cypher_datetime_truncate_invalid_unit() {
        let result = eval_fn(
            ScalarFunction::CypherDatetimeTruncate,
            vec![
                Value::String("invalid".to_string()),
                Value::String("2024-01-15T14:30:45".to_string()),
            ],
        );
        assert_eq!(result, Value::Null);
    }

    // =============================================================================
    // SQL Subquery Tests
    // =============================================================================

    mod sql_subquery_tests {
        use super::*;

        /// Creates a LogicalPlan that returns the given values.
        fn make_values_plan(values: Vec<Value>) -> LogicalPlan {
            LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: values.into_iter().map(|v| vec![value_to_logical_expr(&v)]).collect(),
                column_names: Some(vec!["col".to_string()]),
            })
        }

        /// Converts a Value to a LogicalExpr literal
        fn value_to_logical_expr(value: &Value) -> LogicalExpr {
            match value {
                Value::Null => LogicalExpr::Literal(crate::ast::Literal::Null),
                Value::Bool(b) => LogicalExpr::Literal(crate::ast::Literal::Boolean(*b)),
                Value::Int(i) => LogicalExpr::Literal(crate::ast::Literal::Integer(*i)),
                Value::Float(f) => LogicalExpr::Literal(crate::ast::Literal::Float(*f)),
                Value::String(s) => LogicalExpr::Literal(crate::ast::Literal::String(s.clone())),
                _ => LogicalExpr::Literal(crate::ast::Literal::Null),
            }
        }

        #[test]
        fn test_exists_with_rows() {
            // EXISTS (SELECT 1) where subquery returns rows should be TRUE
            let subquery = make_values_plan(vec![Value::Int(1)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_exists_subquery(&subquery, &empty_row, &ctx).unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_exists_without_rows() {
            // EXISTS (SELECT ...) where subquery returns no rows should be FALSE
            let subquery = LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![], // Empty - no rows
                column_names: Some(vec!["col".to_string()]),
            });
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_exists_subquery(&subquery, &empty_row, &ctx).unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_exists_without_context() {
            // EXISTS without execution context should return FALSE
            let subquery = make_values_plan(vec![Value::Int(1)]);
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_exists_subquery(&subquery, &empty_row, &None).unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_not_exists_with_rows() {
            // NOT EXISTS where subquery returns rows should be FALSE
            let subquery = Box::new(make_values_plan(vec![Value::Int(1)]));
            let expr = LogicalExpr::Exists { subquery, negated: true };
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_expr_with_subquery(&expr, &empty_row, None, &ctx).unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_not_exists_without_rows() {
            // NOT EXISTS where subquery returns no rows should be TRUE
            let subquery = Box::new(LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![], // Empty - no rows
                column_names: Some(vec!["col".to_string()]),
            }));
            let expr = LogicalExpr::Exists { subquery, negated: true };
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_expr_with_subquery(&expr, &empty_row, None, &ctx).unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_in_subquery_found() {
            // 2 IN (1, 2, 3) should be TRUE
            let subquery = make_values_plan(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(2), &subquery, false, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_in_subquery_not_found() {
            // 5 IN (1, 2, 3) should be FALSE
            let subquery = make_values_plan(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, false, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_not_in_subquery_found() {
            // 2 NOT IN (1, 2, 3) should be FALSE
            let subquery = make_values_plan(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(2), &subquery, true, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_not_in_subquery_not_found() {
            // 5 NOT IN (1, 2, 3) should be TRUE
            let subquery = make_values_plan(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, true, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_in_subquery_with_nulls_not_found() {
            // 5 IN (1, NULL, 3) where 5 is not found - should be NULL (UNKNOWN)
            let subquery = make_values_plan(vec![Value::Int(1), Value::Null, Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, false, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Null);
        }

        #[test]
        fn test_in_subquery_with_nulls_found() {
            // 2 IN (1, NULL, 2) where 2 is found - should be TRUE (NULL doesn't matter)
            let subquery = make_values_plan(vec![Value::Int(1), Value::Null, Value::Int(2)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(2), &subquery, false, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_not_in_subquery_with_nulls_not_found() {
            // 5 NOT IN (1, NULL, 3) where 5 is not found but NULLs exist - should be NULL
            let subquery = make_values_plan(vec![Value::Int(1), Value::Null, Value::Int(3)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, true, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Null);
        }

        #[test]
        fn test_null_value_in_subquery() {
            // NULL IN (1, 2, 3) should be NULL
            let subquery = make_values_plan(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let subquery_box = Box::new(subquery);

            let expr = LogicalExpr::InSubquery {
                expr: Box::new(LogicalExpr::Literal(crate::ast::Literal::Null)),
                subquery: subquery_box,
                negated: false,
            };
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_expr_with_subquery(&expr, &empty_row, None, &ctx).unwrap();
            assert_eq!(result, Value::Null);
        }

        #[test]
        fn test_scalar_subquery_returns_value() {
            // Scalar subquery that returns a single value
            let subquery = make_values_plan(vec![Value::Int(42)]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_scalar_subquery(&subquery, &empty_row, &ctx).unwrap();
            assert_eq!(result, Value::Int(42));
        }

        #[test]
        fn test_scalar_subquery_empty() {
            // Scalar subquery that returns no rows - should be NULL
            let subquery = LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![],
                column_names: Some(vec!["col".to_string()]),
            });
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_scalar_subquery(&subquery, &empty_row, &ctx).unwrap();
            assert_eq!(result, Value::Null);
        }

        #[test]
        fn test_in_subquery_empty() {
            // 5 IN (empty) should be FALSE
            let subquery = LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![],
                column_names: Some(vec!["col".to_string()]),
            });
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, false, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_not_in_subquery_empty() {
            // 5 NOT IN (empty) should be TRUE
            let subquery = LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![],
                column_names: Some(vec!["col".to_string()]),
            });
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result =
                evaluate_sql_in_subquery(&Value::Int(5), &subquery, true, &empty_row, &ctx)
                    .unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_in_subquery_string_values() {
            // 'Bob' IN ('Alice', 'Bob', 'Carol') should be TRUE
            let subquery = make_values_plan(vec![
                Value::String("Alice".to_string()),
                Value::String("Bob".to_string()),
                Value::String("Carol".to_string()),
            ]);
            let ctx = Some(ExecutionContext::new());
            let empty_row = Row::new(Arc::new(Schema::empty()), vec![]);

            let result = evaluate_sql_in_subquery(
                &Value::String("Bob".to_string()),
                &subquery,
                false,
                &empty_row,
                &ctx,
            )
            .unwrap();
            assert_eq!(result, Value::Bool(true));
        }

        #[test]
        fn test_filter_with_exists_subquery() {
            // Test FilterOp with an EXISTS predicate
            let input: BoxedOperator = Box::new(ValuesOp::with_columns(
                vec!["id".to_string()],
                vec![vec![Value::Int(1)], vec![Value::Int(2)]],
            ));

            // EXISTS (SELECT 1) - always true, should pass all rows
            let subquery = Box::new(make_values_plan(vec![Value::Int(1)]));
            let predicate = LogicalExpr::Exists { subquery, negated: false };

            let mut filter = FilterOp::new(predicate, input);
            let ctx = ExecutionContext::new();
            filter.open(&ctx).unwrap();

            // Both rows should pass
            assert!(filter.next().unwrap().is_some());
            assert!(filter.next().unwrap().is_some());
            assert!(filter.next().unwrap().is_none());
            filter.close().unwrap();
        }

        #[test]
        fn test_filter_with_not_exists_empty_subquery() {
            // Test FilterOp with a NOT EXISTS predicate on empty subquery
            let input: BoxedOperator = Box::new(ValuesOp::with_columns(
                vec!["id".to_string()],
                vec![vec![Value::Int(1)], vec![Value::Int(2)]],
            ));

            // NOT EXISTS (SELECT ...) where subquery is empty - always true
            let subquery = Box::new(LogicalPlan::Values(crate::plan::logical::ValuesNode {
                rows: vec![],
                column_names: Some(vec!["col".to_string()]),
            }));
            let predicate = LogicalExpr::Exists { subquery, negated: true };

            let mut filter = FilterOp::new(predicate, input);
            let ctx = ExecutionContext::new();
            filter.open(&ctx).unwrap();

            // Both rows should pass (NOT EXISTS on empty is TRUE)
            assert!(filter.next().unwrap().is_some());
            assert!(filter.next().unwrap().is_some());
            assert!(filter.next().unwrap().is_none());
            filter.close().unwrap();
        }
    }
}
