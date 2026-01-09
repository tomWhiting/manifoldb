//! Window operator for ROW_NUMBER, RANK, DENSE_RANK.

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

        // Compute window function values
        let mut window_values: Vec<i64> = vec![0; rows.len()];
        let mut current_partition_key: Option<Vec<Value>> = None;
        let mut row_number = 0i64;
        let mut rank = 0i64;
        let mut dense_rank = 0i64;
        let mut prev_order_key: Option<Vec<Value>> = None;
        let mut _ties_count = 0i64;

        for &idx in &indices {
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
                _ties_count = 0;
            }

            // Check if order key changed (for RANK/DENSE_RANK)
            let order_key_changed = prev_order_key.as_ref() != Some(&order_key);

            // Increment counters
            row_number += 1;

            if order_key_changed {
                rank = row_number;
                dense_rank += 1;
                _ties_count = 1;
            } else {
                _ties_count += 1;
            }

            // Compute the window function value
            let value = match expr.func {
                WindowFunction::RowNumber => row_number,
                WindowFunction::Rank => rank,
                WindowFunction::DenseRank => dense_rank,
            };

            window_values[idx] = value;
            prev_order_key = Some(order_key);
        }

        // Add window values to rows with updated schema
        let result: Vec<Row> = rows
            .into_iter()
            .zip(window_values.into_iter())
            .map(|(row, window_val)| {
                let mut values = row.values().to_vec();
                values.push(Value::Int(window_val));

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
}
