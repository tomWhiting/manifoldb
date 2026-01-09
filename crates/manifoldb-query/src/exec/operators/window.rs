//! Window operator for ranking and value functions.
//!
//! Supports:
//! - Ranking functions: ROW_NUMBER, RANK, DENSE_RANK
//! - Value functions: LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE

use std::cmp::Ordering;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::ast::WindowFunction;
use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::physical::WindowFunctionExpr;

/// Window operator.
///
/// Computes window functions (ROW_NUMBER, RANK, DENSE_RANK) over input rows.
/// This is a blocking operator that materializes all input rows.
pub struct WindowOp {
    /// Base operator state.
    base: OperatorBase,
    /// Window function expressions.
    window_exprs: Vec<WindowFunctionExpr>,
    /// Input operator.
    input: BoxedOperator,
    /// Iterator over rows with computed window values.
    result_iter: std::vec::IntoIter<Row>,
    /// Whether rows have been materialized.
    materialized: bool,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl WindowOp {
    /// Creates a new window operator.
    #[must_use]
    pub fn new(window_exprs: Vec<WindowFunctionExpr>, input: BoxedOperator) -> Self {
        // Create output schema by adding window columns to input schema
        let input_schema = input.schema();
        let mut column_names: Vec<String> =
            input_schema.columns().iter().map(|s| (*s).to_string()).collect();
        for expr in &window_exprs {
            column_names.push(expr.alias.clone());
        }
        let schema = Arc::new(Schema::new(column_names));

        Self {
            base: OperatorBase::new(schema),
            window_exprs,
            input,
            result_iter: Vec::new().into_iter(),
            materialized: false,
            max_rows_in_memory: 0,
        }
    }

    /// Materializes input and computes window functions.
    fn materialize_and_compute(&mut self) -> OperatorResult<()> {
        // Collect all rows with size limit check
        let mut rows: Vec<Row> = Vec::new();
        while let Some(row) = self.input.next()? {
            rows.push(row);

            // Check limit after each row (0 means no limit)
            if self.max_rows_in_memory > 0 && rows.len() > self.max_rows_in_memory {
                return Err(ParseError::QueryTooLarge {
                    actual: rows.len(),
                    limit: self.max_rows_in_memory,
                });
            }
        }

        if rows.is_empty() {
            self.result_iter = Vec::new().into_iter();
            self.materialized = true;
            return Ok(());
        }

        // Process each window expression
        let mut result_rows = rows;

        for window_expr in &self.window_exprs {
            result_rows = self.compute_window_function(result_rows, window_expr)?;
        }

        self.result_iter = result_rows.into_iter();
        self.materialized = true;
        Ok(())
    }

    /// Computes a single window function and appends results to rows.
    fn compute_window_function(
        &self,
        rows: Vec<Row>,
        expr: &WindowFunctionExpr,
    ) -> OperatorResult<Vec<Row>> {
        if rows.is_empty() {
            return Ok(rows);
        }

        // Create indices for sorting while preserving original order
        let mut indices: Vec<usize> = (0..rows.len()).collect();

        // Sort indices by partition keys then order keys
        indices.sort_by(|&a, &b| {
            // First compare by partition keys
            for partition_expr in &expr.partition_by {
                let val_a = evaluate_expr(partition_expr, &rows[a]).unwrap_or(Value::Null);
                let val_b = evaluate_expr(partition_expr, &rows[b]).unwrap_or(Value::Null);
                let cmp = compare_values(&val_a, &val_b);
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }

            // Then compare by order keys
            for sort in &expr.order_by {
                let val_a = evaluate_expr(&sort.expr, &rows[a]).unwrap_or(Value::Null);
                let val_b = evaluate_expr(&sort.expr, &rows[b]).unwrap_or(Value::Null);
                let cmp = compare_values(&val_a, &val_b);
                let cmp = if sort.ascending { cmp } else { cmp.reverse() };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }

            Ordering::Equal
        });

        // Compute window function values (using Value to support any type)
        let mut window_values: Vec<Value> = vec![Value::Null; rows.len()];

        // Choose computation strategy based on function type
        match &expr.func {
            WindowFunction::RowNumber | WindowFunction::Rank | WindowFunction::DenseRank => {
                self.compute_ranking_function(&rows, &indices, expr, &mut window_values);
            }
            WindowFunction::Lag { offset, .. } | WindowFunction::Lead { offset, .. } => {
                let is_lag = matches!(expr.func, WindowFunction::Lag { .. });
                self.compute_lag_lead(&rows, &indices, expr, *offset, is_lag, &mut window_values);
            }
            WindowFunction::FirstValue
            | WindowFunction::LastValue
            | WindowFunction::NthValue { .. } => {
                self.compute_frame_value(&rows, &indices, expr, &mut window_values);
            }
        }

        // Add window values to rows with updated schema
        let result: Vec<Row> = rows
            .into_iter()
            .zip(window_values.into_iter())
            .map(|(row, window_val)| {
                let mut values = row.values().to_vec();
                values.push(window_val);

                // Update schema with new column
                let mut col_names: Vec<String> =
                    row.schema().columns().iter().map(|s| (*s).to_string()).collect();
                if !col_names.contains(&expr.alias) {
                    col_names.push(expr.alias.clone());
                }
                let new_schema = Arc::new(Schema::new(col_names));
                Row::new(new_schema, values)
            })
            .collect();

        Ok(result)
    }

    /// Computes ranking window functions (ROW_NUMBER, RANK, DENSE_RANK).
    fn compute_ranking_function(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        window_values: &mut [Value],
    ) {
        let mut current_partition_key: Option<Vec<Value>> = None;
        let mut row_number = 0i64;
        let mut rank = 0i64;
        let mut dense_rank = 0i64;
        let mut prev_order_key: Option<Vec<Value>> = None;

        for &idx in indices {
            // Get partition key for current row
            let partition_key: Vec<Value> = expr
                .partition_by
                .iter()
                .map(|e| evaluate_expr(e, &rows[idx]).unwrap_or(Value::Null))
                .collect();

            // Get order key for current row
            let order_key: Vec<Value> = expr
                .order_by
                .iter()
                .map(|s| evaluate_expr(&s.expr, &rows[idx]).unwrap_or(Value::Null))
                .collect();

            // Check if partition changed
            let partition_changed = current_partition_key.as_ref() != Some(&partition_key);
            if partition_changed {
                current_partition_key = Some(partition_key);
                row_number = 0;
                rank = 0;
                dense_rank = 0;
                prev_order_key = None;
            }

            // Check if order key changed (for RANK/DENSE_RANK)
            let order_key_changed = prev_order_key.as_ref() != Some(&order_key);

            // Increment counters
            row_number += 1;

            if order_key_changed {
                rank = row_number;
                dense_rank += 1;
            }

            // Compute the window function value
            let value = match expr.func {
                WindowFunction::RowNumber => row_number,
                WindowFunction::Rank => rank,
                WindowFunction::DenseRank => dense_rank,
                _ => unreachable!(),
            };

            window_values[idx] = Value::Int(value);
            prev_order_key = Some(order_key);
        }
    }

    /// Computes LAG and LEAD window functions.
    fn compute_lag_lead(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        offset: u64,
        is_lag: bool,
        window_values: &mut [Value],
    ) {
        // Build partition boundaries: maps each sorted position to its partition start/end
        let partition_ranges = self.build_partition_ranges(rows, indices, expr);

        let offset_i64 = offset as i64;

        for (sorted_pos, &idx) in indices.iter().enumerate() {
            let (partition_start, partition_end) = partition_ranges[sorted_pos];

            // Calculate target position within the partition
            let sorted_pos_i64 = sorted_pos as i64;
            let target_sorted_pos =
                if is_lag { sorted_pos_i64 - offset_i64 } else { sorted_pos_i64 + offset_i64 };

            let value = if target_sorted_pos >= partition_start as i64
                && target_sorted_pos < partition_end as i64
            {
                // Target row is within partition - get value from that row
                let target_idx = indices[target_sorted_pos as usize];
                if let Some(arg) = &expr.arg {
                    evaluate_expr(arg, &rows[target_idx]).unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            } else {
                // Target row is outside partition - use default value
                if let Some(default) = &expr.default_value {
                    evaluate_expr(default, &rows[idx]).unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            };

            window_values[idx] = value;
        }
    }

    /// Computes FIRST_VALUE, LAST_VALUE, and NTH_VALUE window functions.
    fn compute_frame_value(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        window_values: &mut [Value],
    ) {
        // Build partition boundaries
        let partition_ranges = self.build_partition_ranges(rows, indices, expr);

        for (sorted_pos, &idx) in indices.iter().enumerate() {
            let (partition_start, partition_end) = partition_ranges[sorted_pos];

            // Determine which row to get the value from
            let target_sorted_pos = match &expr.func {
                WindowFunction::FirstValue => Some(partition_start),
                WindowFunction::LastValue => {
                    if partition_end > 0 {
                        Some(partition_end - 1)
                    } else {
                        None
                    }
                }
                WindowFunction::NthValue { n } => {
                    let nth_pos = partition_start + (*n as usize) - 1; // n is 1-indexed
                    if nth_pos < partition_end {
                        Some(nth_pos)
                    } else {
                        None
                    }
                }
                _ => unreachable!(),
            };

            let value = if let Some(target_pos) = target_sorted_pos {
                let target_idx = indices[target_pos];
                if let Some(arg) = &expr.arg {
                    evaluate_expr(arg, &rows[target_idx]).unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            } else {
                Value::Null
            };

            window_values[idx] = value;
        }
    }

    /// Builds partition ranges for sorted indices.
    /// Returns a vector where each position maps to (partition_start, partition_end) in sorted order.
    fn build_partition_ranges(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
    ) -> Vec<(usize, usize)> {
        let mut ranges = vec![(0, indices.len()); indices.len()];

        if expr.partition_by.is_empty() {
            // No partitioning - entire result set is one partition
            return ranges;
        }

        let mut partition_start = 0;
        let mut current_partition_key: Option<Vec<Value>> = None;

        for (sorted_pos, &idx) in indices.iter().enumerate() {
            let partition_key: Vec<Value> = expr
                .partition_by
                .iter()
                .map(|e| evaluate_expr(e, &rows[idx]).unwrap_or(Value::Null))
                .collect();

            let partition_changed = current_partition_key.as_ref() != Some(&partition_key);
            if partition_changed {
                // Update previous partition's end
                if sorted_pos > 0 {
                    for pos in partition_start..sorted_pos {
                        ranges[pos].1 = sorted_pos;
                    }
                }
                partition_start = sorted_pos;
                current_partition_key = Some(partition_key);
            }
        }

        // Set the final partition's end
        for pos in partition_start..indices.len() {
            ranges[pos].1 = indices.len();
        }

        // Set all partition starts
        let mut current_start = 0;
        current_partition_key = None;
        for (sorted_pos, &idx) in indices.iter().enumerate() {
            let partition_key: Vec<Value> = expr
                .partition_by
                .iter()
                .map(|e| evaluate_expr(e, &rows[idx]).unwrap_or(Value::Null))
                .collect();

            if current_partition_key.as_ref() != Some(&partition_key) {
                current_start = sorted_pos;
                current_partition_key = Some(partition_key);
            }
            ranges[sorted_pos].0 = current_start;
        }

        ranges
    }
}

/// Compares two values for sorting.
fn compare_values(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Greater, // NULLS LAST
        (_, Value::Null) => Ordering::Less,
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

impl Operator for WindowOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.result_iter = Vec::new().into_iter();
        self.materialized = false;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Materialize on first call
        if !self.materialized {
            self.materialize_and_compute()?;
        }

        match self.result_iter.next() {
            Some(row) => {
                self.base.inc_rows_produced();
                Ok(Some(row))
            }
            None => {
                self.base.set_finished();
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.result_iter = Vec::new().into_iter();
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
        "Window"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use crate::plan::logical::{LogicalExpr, SortOrder};

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("IT"), Value::Int(90)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(80)],
            ],
        ))
    }

    #[test]
    fn row_number_no_partition() {
        let window_expr = WindowFunctionExpr::new(
            WindowFunction::RowNumber,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "rn",
        );
        let mut op = WindowOp::new(vec![window_expr], make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Collect all rows and verify row numbers
        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 4);

        // Find the row numbers assigned
        let row_numbers: Vec<i64> = rows
            .iter()
            .map(|r| if let Some(Value::Int(v)) = r.get_by_name("rn") { *v } else { -1 })
            .collect();

        // All row numbers should be 1, 2, 3, 4
        let mut sorted_rn = row_numbers.clone();
        sorted_rn.sort_unstable();
        assert_eq!(sorted_rn, vec![1, 2, 3, 4]);

        op.close().unwrap();
    }

    #[test]
    fn rank_with_ties() {
        // Create input with ties in salary
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "score".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(90)],
                vec![Value::from("Dave"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::new(
            WindowFunction::Rank,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("score"))],
            "rank",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Collect results and map by name
        let mut results: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::Int(rank))) =
                (row.get_by_name("name"), row.get_by_name("rank"))
            {
                results.insert(name.clone(), *rank);
            }
        }

        // Alice: score 100 -> rank 1
        // Bob, Carol: score 90 -> rank 2 (tie)
        // Dave: score 80 -> rank 4 (gap after tie)
        assert_eq!(results.get("Alice"), Some(&1));
        assert_eq!(results.get("Bob"), Some(&2));
        assert_eq!(results.get("Carol"), Some(&2));
        assert_eq!(results.get("Dave"), Some(&4));

        op.close().unwrap();
    }

    #[test]
    fn dense_rank_no_gaps() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "score".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(90)],
                vec![Value::from("Dave"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::new(
            WindowFunction::DenseRank,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("score"))],
            "drank",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::Int(drank))) =
                (row.get_by_name("name"), row.get_by_name("drank"))
            {
                results.insert(name.clone(), *drank);
            }
        }

        // Dense rank has no gaps: 1, 2, 2, 3
        assert_eq!(results.get("Alice"), Some(&1));
        assert_eq!(results.get("Bob"), Some(&2));
        assert_eq!(results.get("Carol"), Some(&2));
        assert_eq!(results.get("Dave"), Some(&3)); // No gap!

        op.close().unwrap();
    }

    #[test]
    fn partition_by() {
        let window_expr = WindowFunctionExpr::new(
            WindowFunction::RowNumber,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "dept_rank",
        );
        let mut op = WindowOp::new(vec![window_expr], make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::Int(rank))) =
                (row.get_by_name("name"), row.get_by_name("dept_rank"))
            {
                results.insert(name.clone(), *rank);
            }
        }

        // Within Sales: Alice(100) -> 1, Bob(90) -> 2
        // Within IT: Carol(90) -> 1, Dave(80) -> 2
        assert_eq!(results.get("Alice"), Some(&1));
        assert_eq!(results.get("Bob"), Some(&2));
        assert_eq!(results.get("Carol"), Some(&1));
        assert_eq!(results.get("Dave"), Some(&2));

        op.close().unwrap();
    }

    // ========== LAG Tests ==========

    #[test]
    fn lag_basic() {
        // LAG(salary, 1) - get previous salary ordered by salary desc
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lag { offset: 1, has_default: false },
            LogicalExpr::column("salary"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "prev_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, Option<i64>> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(name)) = row.get_by_name("name") {
                let prev_salary = match row.get_by_name("prev_salary") {
                    Some(Value::Int(v)) => Some(*v),
                    Some(Value::Null) | None => None,
                    _ => None,
                };
                results.insert(name.clone(), prev_salary);
            }
        }

        // Sorted by salary DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // LAG(salary, 1): Alice->NULL, Bob->100, Carol->90, Dave->80
        assert_eq!(results.get("Alice"), Some(&None));
        assert_eq!(results.get("Bob"), Some(&Some(100)));
        assert_eq!(results.get("Carol"), Some(&Some(90)));
        assert_eq!(results.get("Dave"), Some(&Some(80)));

        op.close().unwrap();
    }

    #[test]
    fn lag_with_default() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lag { offset: 1, has_default: true },
            LogicalExpr::column("salary"),
            Some(LogicalExpr::integer(0)), // Default to 0
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "prev_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::Int(prev))) =
                (row.get_by_name("name"), row.get_by_name("prev_salary"))
            {
                results.insert(name.clone(), *prev);
            }
        }

        // Alice->0 (default), Bob->100
        assert_eq!(results.get("Alice"), Some(&0));
        assert_eq!(results.get("Bob"), Some(&100));

        op.close().unwrap();
    }

    #[test]
    fn lag_with_offset_2() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lag { offset: 2, has_default: false },
            LogicalExpr::column("salary"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "prev2_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, Option<i64>> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(name)) = row.get_by_name("name") {
                let prev2 = match row.get_by_name("prev2_salary") {
                    Some(Value::Int(v)) => Some(*v),
                    _ => None,
                };
                results.insert(name.clone(), prev2);
            }
        }

        // LAG(salary, 2): Alice->NULL, Bob->NULL, Carol->100, Dave->90
        assert_eq!(results.get("Alice"), Some(&None));
        assert_eq!(results.get("Bob"), Some(&None));
        assert_eq!(results.get("Carol"), Some(&Some(100)));
        assert_eq!(results.get("Dave"), Some(&Some(90)));

        op.close().unwrap();
    }

    #[test]
    fn lag_with_partition() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("IT"), Value::Int(95)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lag { offset: 1, has_default: false },
            LogicalExpr::column("salary"),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "prev_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, Option<i64>> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(name)) = row.get_by_name("name") {
                let prev = match row.get_by_name("prev_salary") {
                    Some(Value::Int(v)) => Some(*v),
                    _ => None,
                };
                results.insert(name.clone(), prev);
            }
        }

        // Sales partition: Alice(100)->NULL, Bob(90)->100
        // IT partition: Carol(95)->NULL, Dave(80)->95
        assert_eq!(results.get("Alice"), Some(&None));
        assert_eq!(results.get("Bob"), Some(&Some(100)));
        assert_eq!(results.get("Carol"), Some(&None));
        assert_eq!(results.get("Dave"), Some(&Some(95)));

        op.close().unwrap();
    }

    // ========== LEAD Tests ==========

    #[test]
    fn lead_basic() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lead { offset: 1, has_default: false },
            LogicalExpr::column("salary"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "next_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, Option<i64>> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(name)) = row.get_by_name("name") {
                let next = match row.get_by_name("next_salary") {
                    Some(Value::Int(v)) => Some(*v),
                    _ => None,
                };
                results.insert(name.clone(), next);
            }
        }

        // Sorted DESC: Alice(100)->90, Bob(90)->80, Carol(80)->70, Dave(70)->NULL
        assert_eq!(results.get("Alice"), Some(&Some(90)));
        assert_eq!(results.get("Bob"), Some(&Some(80)));
        assert_eq!(results.get("Carol"), Some(&Some(70)));
        assert_eq!(results.get("Dave"), Some(&None));

        op.close().unwrap();
    }

    #[test]
    fn lead_with_default() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lead { offset: 1, has_default: true },
            LogicalExpr::column("salary"),
            Some(LogicalExpr::integer(-1)), // Default to -1
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "next_salary",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::Int(next))) =
                (row.get_by_name("name"), row.get_by_name("next_salary"))
            {
                results.insert(name.clone(), *next);
            }
        }

        // Alice->90, Bob->-1 (default)
        assert_eq!(results.get("Alice"), Some(&90));
        assert_eq!(results.get("Bob"), Some(&-1));

        op.close().unwrap();
    }

    // ========== FIRST_VALUE Tests ==========

    #[test]
    fn first_value_basic() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::FirstValue,
            LogicalExpr::column("name"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "top_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut top_names: Vec<String> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(top)) = row.get_by_name("top_name") {
                top_names.push(top.clone());
            }
        }

        // All rows should show Alice (highest salary)
        assert!(top_names.iter().all(|n| n == "Alice"));
        assert_eq!(top_names.len(), 3);

        op.close().unwrap();
    }

    #[test]
    fn first_value_with_partition() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("IT"), Value::Int(95)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::FirstValue,
            LogicalExpr::column("name"),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "top_in_dept",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(top))) =
                (row.get_by_name("name"), row.get_by_name("top_in_dept"))
            {
                results.insert(name.clone(), top.clone());
            }
        }

        // Sales: Alice and Bob both see Alice
        // IT: Carol and Dave both see Carol
        assert_eq!(results.get("Alice"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Carol".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Carol".to_string()));

        op.close().unwrap();
    }

    // ========== LAST_VALUE Tests ==========

    #[test]
    fn last_value_basic() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::LastValue,
            LogicalExpr::column("name"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "last_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut last_names: Vec<String> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(last)) = row.get_by_name("last_name") {
                last_names.push(last.clone());
            }
        }

        // All rows should show Carol (lowest salary = last in desc order)
        assert!(last_names.iter().all(|n| n == "Carol"));
        assert_eq!(last_names.len(), 3);

        op.close().unwrap();
    }

    #[test]
    fn last_value_with_partition() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("IT"), Value::Int(95)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::LastValue,
            LogicalExpr::column("name"),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "lowest_in_dept",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(lowest))) =
                (row.get_by_name("name"), row.get_by_name("lowest_in_dept"))
            {
                results.insert(name.clone(), lowest.clone());
            }
        }

        // Sales: Alice and Bob both see Bob
        // IT: Carol and Dave both see Dave
        assert_eq!(results.get("Alice"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Dave".to_string()));

        op.close().unwrap();
    }

    // ========== NTH_VALUE Tests ==========

    #[test]
    fn nth_value_basic() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::NthValue { n: 2 },
            LogicalExpr::column("name"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "second_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut second_names: Vec<String> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(second)) = row.get_by_name("second_name") {
                second_names.push(second.clone());
            }
        }

        // All rows should show Bob (2nd highest salary)
        assert!(second_names.iter().all(|n| n == "Bob"));
        assert_eq!(second_names.len(), 4);

        op.close().unwrap();
    }

    #[test]
    fn nth_value_out_of_range() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
            ],
        ));

        // Try to get 5th value when there are only 2 rows
        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::NthValue { n: 5 },
            LogicalExpr::column("name"),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "fifth_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut null_count = 0;
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::Null) = row.get_by_name("fifth_name") {
                null_count += 1;
            }
        }

        // Both rows should have NULL for 5th value
        assert_eq!(null_count, 2);

        op.close().unwrap();
    }

    #[test]
    fn nth_value_with_partition() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("Sales"), Value::Int(80)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(95)],
                vec![Value::from("Eve"), Value::from("IT"), Value::Int(85)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::NthValue { n: 2 },
            LogicalExpr::column("name"),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            "second_in_dept",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(second))) =
                (row.get_by_name("name"), row.get_by_name("second_in_dept"))
            {
                results.insert(name.clone(), second.clone());
            }
        }

        // Sales (3 members): 2nd is Bob
        // IT (2 members): 2nd is Eve
        assert_eq!(results.get("Alice"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Eve".to_string()));
        assert_eq!(results.get("Eve"), Some(&"Eve".to_string()));

        op.close().unwrap();
    }

    // ========== NULL Handling Tests ==========

    #[test]
    fn lag_with_null_values() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "bonus".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Null],
                vec![Value::from("Carol"), Value::Int(80)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Lag { offset: 1, has_default: false },
            LogicalExpr::column("bonus"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("name"))],
            "prev_bonus",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::String(name)) = row.get_by_name("name") {
                let prev = row.get_by_name("prev_bonus").cloned().unwrap_or(Value::Null);
                results.insert(name.clone(), prev);
            }
        }

        // Ordered by name ASC: Alice, Bob, Carol
        // Alice -> NULL (no prev), Bob -> 100, Carol -> NULL (Bob's bonus was NULL)
        assert_eq!(results.get("Alice"), Some(&Value::Null));
        assert_eq!(results.get("Bob"), Some(&Value::Int(100)));
        assert_eq!(results.get("Carol"), Some(&Value::Null));

        op.close().unwrap();
    }
}
