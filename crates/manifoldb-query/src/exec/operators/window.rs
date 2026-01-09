//! Window operator for ranking and value functions.
//!
//! Supports:
//! - Ranking functions: ROW_NUMBER, RANK, DENSE_RANK
//! - Value functions: LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE

use std::cmp::Ordering;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::ast::{
    AggregateWindowFunction, Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits,
    WindowFunction,
};
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
            WindowFunction::Aggregate(agg_func) => {
                self.compute_aggregate_window(&rows, &indices, expr, *agg_func, &mut window_values);
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
    /// These functions respect window frame specifications.
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

            // Compute frame boundaries for the current row
            let (frame_start, frame_end) = self.compute_frame_bounds(
                rows,
                indices,
                expr,
                sorted_pos,
                partition_start,
                partition_end,
            );

            // Determine which row to get the value from within the frame
            let target_sorted_pos = match &expr.func {
                WindowFunction::FirstValue => {
                    if frame_start < frame_end {
                        Some(frame_start)
                    } else {
                        None
                    }
                }
                WindowFunction::LastValue => {
                    if frame_end > frame_start {
                        Some(frame_end - 1)
                    } else {
                        None
                    }
                }
                WindowFunction::NthValue { n } => {
                    let nth_pos = frame_start + (*n as usize) - 1; // n is 1-indexed
                    if nth_pos < frame_end {
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

    /// Computes aggregate window functions (SUM, AVG, COUNT, MIN, MAX) over frame.
    ///
    /// These functions compute aggregates over the window frame, enabling
    /// running totals, moving averages, and cumulative computations.
    fn compute_aggregate_window(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        agg_func: AggregateWindowFunction,
        window_values: &mut [Value],
    ) {
        // Build partition boundaries
        let partition_ranges = self.build_partition_ranges(rows, indices, expr);

        for (sorted_pos, &idx) in indices.iter().enumerate() {
            let (partition_start, partition_end) = partition_ranges[sorted_pos];

            // Compute frame boundaries for the current row
            let (frame_start, frame_end) = self.compute_frame_bounds(
                rows,
                indices,
                expr,
                sorted_pos,
                partition_start,
                partition_end,
            );

            // Collect values in the frame
            let frame_values: Vec<Value> = (frame_start..frame_end)
                .map(|pos| {
                    let row_idx = indices[pos];
                    if let Some(arg) = &expr.arg {
                        evaluate_expr(arg, &rows[row_idx]).unwrap_or(Value::Null)
                    } else {
                        // For COUNT(*), we count all rows
                        Value::Int(1)
                    }
                })
                .collect();

            // Compute the aggregate based on function type
            let result = match agg_func {
                AggregateWindowFunction::Count => {
                    self.compute_count(&frame_values, expr.arg.is_some())
                }
                AggregateWindowFunction::Sum => self.compute_sum(&frame_values),
                AggregateWindowFunction::Avg => self.compute_avg(&frame_values),
                AggregateWindowFunction::Min => self.compute_min(&frame_values),
                AggregateWindowFunction::Max => self.compute_max(&frame_values),
            };

            window_values[idx] = result;
        }
    }

    /// Computes COUNT over a set of values.
    /// If count_non_null is true, only counts non-NULL values (COUNT(expr)).
    /// If count_non_null is false, counts all rows (COUNT(*)).
    fn compute_count(&self, values: &[Value], count_non_null: bool) -> Value {
        let count = if count_non_null {
            values.iter().filter(|v| !matches!(v, Value::Null)).count()
        } else {
            values.len()
        };
        Value::Int(count as i64)
    }

    /// Computes SUM over a set of values.
    fn compute_sum(&self, values: &[Value]) -> Value {
        let mut int_sum: i64 = 0;
        let mut float_sum: f64 = 0.0;
        let mut has_float = false;
        let mut has_value = false;

        for v in values {
            match v {
                Value::Int(n) => {
                    int_sum += n;
                    has_value = true;
                }
                Value::Float(n) => {
                    float_sum += n;
                    has_float = true;
                    has_value = true;
                }
                Value::Null => {}
                _ => {}
            }
        }

        if !has_value {
            return Value::Null;
        }

        if has_float {
            Value::Float(float_sum + int_sum as f64)
        } else {
            Value::Int(int_sum)
        }
    }

    /// Computes AVG over a set of values.
    fn compute_avg(&self, values: &[Value]) -> Value {
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;

        for v in values {
            match v {
                Value::Int(n) => {
                    sum += *n as f64;
                    count += 1;
                }
                Value::Float(n) => {
                    sum += n;
                    count += 1;
                }
                Value::Null => {}
                _ => {}
            }
        }

        if count == 0 {
            Value::Null
        } else {
            Value::Float(sum / count as f64)
        }
    }

    /// Computes MIN over a set of values.
    fn compute_min(&self, values: &[Value]) -> Value {
        let mut min_int: Option<i64> = None;
        let mut min_float: Option<f64> = None;
        let mut min_string: Option<&str> = None;

        for v in values {
            match v {
                Value::Int(n) => {
                    min_int = Some(min_int.map_or(*n, |m| m.min(*n)));
                }
                Value::Float(n) => {
                    min_float = Some(min_float.map_or(*n, |m| m.min(*n)));
                }
                Value::String(s) => {
                    min_string = Some(min_string.map_or(s.as_str(), |m| {
                        if s.as_str() < m {
                            s.as_str()
                        } else {
                            m
                        }
                    }));
                }
                Value::Null => {}
                _ => {}
            }
        }

        // Return the appropriate type
        if let Some(s) = min_string {
            Value::String(s.to_string())
        } else if let Some(f) = min_float {
            if let Some(i) = min_int {
                Value::Float(f.min(i as f64))
            } else {
                Value::Float(f)
            }
        } else if let Some(i) = min_int {
            Value::Int(i)
        } else {
            Value::Null
        }
    }

    /// Computes MAX over a set of values.
    fn compute_max(&self, values: &[Value]) -> Value {
        let mut max_int: Option<i64> = None;
        let mut max_float: Option<f64> = None;
        let mut max_string: Option<&str> = None;

        for v in values {
            match v {
                Value::Int(n) => {
                    max_int = Some(max_int.map_or(*n, |m| m.max(*n)));
                }
                Value::Float(n) => {
                    max_float = Some(max_float.map_or(*n, |m| m.max(*n)));
                }
                Value::String(s) => {
                    max_string = Some(max_string.map_or(s.as_str(), |m| {
                        if s.as_str() > m {
                            s.as_str()
                        } else {
                            m
                        }
                    }));
                }
                Value::Null => {}
                _ => {}
            }
        }

        // Return the appropriate type
        if let Some(s) = max_string {
            Value::String(s.to_string())
        } else if let Some(f) = max_float {
            if let Some(i) = max_int {
                Value::Float(f.max(i as f64))
            } else {
                Value::Float(f)
            }
        } else if let Some(i) = max_int {
            Value::Int(i)
        } else {
            Value::Null
        }
    }

    /// Computes the frame boundaries (start, end) for a row in sorted order.
    /// Returns (frame_start, frame_end) as indices into the sorted indices array.
    fn compute_frame_bounds(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        sorted_pos: usize,
        partition_start: usize,
        partition_end: usize,
    ) -> (usize, usize) {
        // Get the frame specification or use defaults
        let frame = match &expr.frame {
            Some(f) => f.clone(),
            None => {
                // Default frame: if ORDER BY is present, use RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                // Otherwise, use the entire partition
                if expr.order_by.is_empty() {
                    return (partition_start, partition_end);
                }
                WindowFrame {
                    units: WindowFrameUnits::Range,
                    start: WindowFrameBound::UnboundedPreceding,
                    end: Some(WindowFrameBound::CurrentRow),
                }
            }
        };

        // Get the end bound (default to same as start if not BETWEEN)
        let end_bound = frame.end.clone().unwrap_or_else(|| frame.start.clone());

        // Compute frame start
        let frame_start = match &frame.start {
            WindowFrameBound::UnboundedPreceding => partition_start,
            WindowFrameBound::CurrentRow => {
                if frame.units == WindowFrameUnits::Rows {
                    sorted_pos
                } else {
                    // RANGE: find first row with same ORDER BY value
                    self.find_peers_start(rows, indices, expr, sorted_pos, partition_start)
                }
            }
            WindowFrameBound::Preceding(n_expr) => {
                let n = self.eval_bound_offset(n_expr);
                if frame.units == WindowFrameUnits::Rows {
                    sorted_pos.saturating_sub(n).max(partition_start)
                } else {
                    // RANGE: find rows within n value difference
                    self.find_range_start(rows, indices, expr, sorted_pos, partition_start, n)
                }
            }
            WindowFrameBound::Following(n_expr) => {
                let n = self.eval_bound_offset(n_expr);
                if frame.units == WindowFrameUnits::Rows {
                    (sorted_pos + n).min(partition_end)
                } else {
                    self.find_range_end(rows, indices, expr, sorted_pos, partition_end, n)
                }
            }
            WindowFrameBound::UnboundedFollowing => partition_end,
        };

        // Compute frame end (exclusive)
        let frame_end = match &end_bound {
            WindowFrameBound::UnboundedFollowing => partition_end,
            WindowFrameBound::CurrentRow => {
                if frame.units == WindowFrameUnits::Rows {
                    (sorted_pos + 1).min(partition_end)
                } else {
                    // RANGE: find last row with same ORDER BY value + 1
                    self.find_peers_end(rows, indices, expr, sorted_pos, partition_end)
                }
            }
            WindowFrameBound::Following(n_expr) => {
                let n = self.eval_bound_offset(n_expr);
                if frame.units == WindowFrameUnits::Rows {
                    (sorted_pos + n + 1).min(partition_end)
                } else {
                    self.find_range_end(rows, indices, expr, sorted_pos, partition_end, n)
                }
            }
            WindowFrameBound::Preceding(n_expr) => {
                let n = self.eval_bound_offset(n_expr);
                if frame.units == WindowFrameUnits::Rows {
                    (sorted_pos.saturating_sub(n) + 1).max(partition_start)
                } else {
                    self.find_range_start(rows, indices, expr, sorted_pos, partition_start, n)
                }
            }
            WindowFrameBound::UnboundedPreceding => partition_start,
        };

        // Ensure start <= end
        let frame_start = frame_start.max(partition_start);
        let frame_end = frame_end.min(partition_end);
        let frame_start = frame_start.min(frame_end);

        (frame_start, frame_end)
    }

    /// Evaluates a frame bound offset expression to get the numeric offset.
    fn eval_bound_offset(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Literal(Literal::Integer(n)) => (*n).try_into().unwrap_or(0),
            _ => 0, // Default to 0 for non-literal expressions
        }
    }

    /// Finds the start of peer rows (rows with the same ORDER BY value).
    fn find_peers_start(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        sorted_pos: usize,
        partition_start: usize,
    ) -> usize {
        if expr.order_by.is_empty() {
            return partition_start;
        }

        let current_idx = indices[sorted_pos];
        let current_key: Vec<Value> = expr
            .order_by
            .iter()
            .map(|s| evaluate_expr(&s.expr, &rows[current_idx]).unwrap_or(Value::Null))
            .collect();

        let mut start = sorted_pos;
        while start > partition_start {
            let prev_idx = indices[start - 1];
            let prev_key: Vec<Value> = expr
                .order_by
                .iter()
                .map(|s| evaluate_expr(&s.expr, &rows[prev_idx]).unwrap_or(Value::Null))
                .collect();
            if prev_key != current_key {
                break;
            }
            start -= 1;
        }
        start
    }

    /// Finds the end of peer rows (rows with the same ORDER BY value).
    fn find_peers_end(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        sorted_pos: usize,
        partition_end: usize,
    ) -> usize {
        if expr.order_by.is_empty() {
            return partition_end;
        }

        let current_idx = indices[sorted_pos];
        let current_key: Vec<Value> = expr
            .order_by
            .iter()
            .map(|s| evaluate_expr(&s.expr, &rows[current_idx]).unwrap_or(Value::Null))
            .collect();

        let mut end = sorted_pos + 1;
        while end < partition_end {
            let next_idx = indices[end];
            let next_key: Vec<Value> = expr
                .order_by
                .iter()
                .map(|s| evaluate_expr(&s.expr, &rows[next_idx]).unwrap_or(Value::Null))
                .collect();
            if next_key != current_key {
                break;
            }
            end += 1;
        }
        end
    }

    /// Finds the start position for RANGE with a preceding offset.
    fn find_range_start(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        sorted_pos: usize,
        partition_start: usize,
        offset: usize,
    ) -> usize {
        if expr.order_by.is_empty() {
            return partition_start;
        }

        // Get the current row's ORDER BY value
        let current_idx = indices[sorted_pos];
        let current_value = if let Some(sort) = expr.order_by.first() {
            evaluate_expr(&sort.expr, &rows[current_idx]).unwrap_or(Value::Null)
        } else {
            return partition_start;
        };

        // Find rows where value >= current_value - offset (for ASC sort)
        let offset_value = self.subtract_offset(&current_value, offset);

        let mut start = partition_start;
        for pos in partition_start..sorted_pos {
            let idx = indices[pos];
            let value = if let Some(sort) = expr.order_by.first() {
                evaluate_expr(&sort.expr, &rows[idx]).unwrap_or(Value::Null)
            } else {
                Value::Null
            };

            if self.value_in_range(
                &value,
                &offset_value,
                &current_value,
                expr.order_by.first().map(|s| s.ascending).unwrap_or(true),
            ) {
                start = pos;
                break;
            }
        }
        start
    }

    /// Finds the end position for RANGE with a following offset.
    fn find_range_end(
        &self,
        rows: &[Row],
        indices: &[usize],
        expr: &WindowFunctionExpr,
        sorted_pos: usize,
        partition_end: usize,
        offset: usize,
    ) -> usize {
        if expr.order_by.is_empty() {
            return partition_end;
        }

        // Get the current row's ORDER BY value
        let current_idx = indices[sorted_pos];
        let current_value = if let Some(sort) = expr.order_by.first() {
            evaluate_expr(&sort.expr, &rows[current_idx]).unwrap_or(Value::Null)
        } else {
            return partition_end;
        };

        // Find rows where value <= current_value + offset (for ASC sort)
        let offset_value = self.add_offset(&current_value, offset);

        let mut end = partition_end;
        for pos in (sorted_pos + 1)..partition_end {
            let idx = indices[pos];
            let value = if let Some(sort) = expr.order_by.first() {
                evaluate_expr(&sort.expr, &rows[idx]).unwrap_or(Value::Null)
            } else {
                Value::Null
            };

            if !self.value_in_range(
                &value,
                &current_value,
                &offset_value,
                expr.order_by.first().map(|s| s.ascending).unwrap_or(true),
            ) {
                end = pos;
                break;
            }
        }
        end
    }

    /// Subtracts an offset from a value (for RANGE frames).
    fn subtract_offset(&self, value: &Value, offset: usize) -> Value {
        match value {
            Value::Int(n) => Value::Int(n - offset as i64),
            Value::Float(n) => Value::Float(n - offset as f64),
            _ => value.clone(),
        }
    }

    /// Adds an offset to a value (for RANGE frames).
    fn add_offset(&self, value: &Value, offset: usize) -> Value {
        match value {
            Value::Int(n) => Value::Int(n + offset as i64),
            Value::Float(n) => Value::Float(n + offset as f64),
            _ => value.clone(),
        }
    }

    /// Checks if a value is within the range [low, high] based on sort order.
    fn value_in_range(&self, value: &Value, low: &Value, high: &Value, ascending: bool) -> bool {
        let cmp_low = compare_values(value, low);
        let cmp_high = compare_values(value, high);

        if ascending {
            cmp_low != Ordering::Less && cmp_high != Ordering::Greater
        } else {
            cmp_low != Ordering::Greater && cmp_high != Ordering::Less
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
        // LAST_VALUE over entire partition using explicit frame
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
            ],
        ));

        // Explicit frame to cover entire partition
        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::LastValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
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
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::from("Sales"), Value::Int(100)],
                vec![Value::from("Bob"), Value::from("Sales"), Value::Int(90)],
                vec![Value::from("Carol"), Value::from("IT"), Value::Int(95)],
                vec![Value::from("Dave"), Value::from("IT"), Value::Int(80)],
            ],
        ));

        // Explicit frame to cover entire partition
        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::LastValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
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
        // NTH_VALUE over entire partition using explicit frame
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        // Explicit frame to cover entire partition
        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::NthValue { n: 2 },
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
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
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
            ],
        ));

        // Explicit frame to cover entire partition
        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        // Try to get 5th value when there are only 2 rows
        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::NthValue { n: 5 },
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
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
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

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

        // Explicit frame to cover entire partition
        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::NthValue { n: 2 },
            Some(LogicalExpr::column("name")),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
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

    // ========== Window Frame Tests ==========

    #[test]
    fn first_value_with_rows_frame_unbounded_preceding_to_current() {
        // ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        // Should behave same as default for FIRST_VALUE
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::UnboundedPreceding,
            end: Some(WindowFrameBound::CurrentRow),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::FirstValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "first_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(first))) =
                (row.get_by_name("name"), row.get_by_name("first_name"))
            {
                results.insert(name.clone(), first.clone());
            }
        }

        // Sorted DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // All should return Alice (first in frame starting at unbounded preceding)
        assert_eq!(results.get("Alice"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Alice".to_string()));

        op.close().unwrap();
    }

    #[test]
    fn last_value_with_rows_frame_unbounded_following() {
        // ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        use crate::ast::{WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::CurrentRow,
            end: Some(WindowFrameBound::UnboundedFollowing),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::LastValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "last_name",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(last))) =
                (row.get_by_name("name"), row.get_by_name("last_name"))
            {
                results.insert(name.clone(), last.clone());
            }
        }

        // Sorted DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // All should return Dave (last in frame extending to unbounded following)
        assert_eq!(results.get("Alice"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Dave".to_string()));

        op.close().unwrap();
    }

    #[test]
    fn first_value_with_rows_frame_n_preceding() {
        // ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        // Creates a 2-row sliding window
        use crate::ast::{Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::Preceding(Box::new(Expr::Literal(Literal::Integer(1)))),
            end: Some(WindowFrameBound::CurrentRow),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::FirstValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "first_in_window",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(first))) =
                (row.get_by_name("name"), row.get_by_name("first_in_window"))
            {
                results.insert(name.clone(), first.clone());
            }
        }

        // Sorted DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // Frame is 1 preceding to current:
        // Alice: [Alice] -> Alice (no preceding)
        // Bob: [Alice, Bob] -> Alice
        // Carol: [Bob, Carol] -> Bob
        // Dave: [Carol, Dave] -> Carol
        assert_eq!(results.get("Alice"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Carol".to_string()));

        op.close().unwrap();
    }

    #[test]
    fn last_value_with_rows_frame_n_following() {
        // ROWS BETWEEN CURRENT ROW AND 1 FOLLOWING
        // Creates a 2-row sliding window looking forward
        use crate::ast::{Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::CurrentRow,
            end: Some(WindowFrameBound::Following(Box::new(Expr::Literal(Literal::Integer(1))))),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::LastValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "last_in_window",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(last))) =
                (row.get_by_name("name"), row.get_by_name("last_in_window"))
            {
                results.insert(name.clone(), last.clone());
            }
        }

        // Sorted DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // Frame is current to 1 following:
        // Alice: [Alice, Bob] -> Bob
        // Bob: [Bob, Carol] -> Carol
        // Carol: [Carol, Dave] -> Dave
        // Dave: [Dave] -> Dave (no following)
        assert_eq!(results.get("Alice"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Carol".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Dave".to_string()));

        op.close().unwrap();
    }

    #[test]
    fn nth_value_with_rows_frame_3_row_moving_window() {
        // ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        // Creates a 3-row moving window for NTH_VALUE(2)
        use crate::ast::{Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits};

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("Alice"), Value::Int(100)],
                vec![Value::from("Bob"), Value::Int(90)],
                vec![Value::from("Carol"), Value::Int(80)],
                vec![Value::from("Dave"), Value::Int(70)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::Preceding(Box::new(Expr::Literal(Literal::Integer(1)))),
            end: Some(WindowFrameBound::Following(Box::new(Expr::Literal(Literal::Integer(1))))),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::NthValue { n: 2 },
            Some(LogicalExpr::column("name")),
            None,
            vec![],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "second_in_window",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(second))) =
                (row.get_by_name("name"), row.get_by_name("second_in_window"))
            {
                results.insert(name.clone(), second.clone());
            }
        }

        // Sorted DESC: Alice(100), Bob(90), Carol(80), Dave(70)
        // Frame is 1 preceding to 1 following:
        // Alice: [Alice, Bob] -> 2nd is Bob
        // Bob: [Alice, Bob, Carol] -> 2nd is Bob
        // Carol: [Bob, Carol, Dave] -> 2nd is Carol
        // Dave: [Carol, Dave] -> 2nd is Dave
        assert_eq!(results.get("Alice"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Carol".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Dave".to_string()));

        op.close().unwrap();
    }

    #[test]
    fn first_value_with_frame_and_partition() {
        // Test that frame works correctly with partitions
        use crate::ast::{Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits};

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

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::Preceding(Box::new(Expr::Literal(Literal::Integer(1)))),
            end: Some(WindowFrameBound::CurrentRow),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::FirstValue,
            Some(LogicalExpr::column("name")),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::desc(LogicalExpr::column("salary"))],
            frame,
            "prev_or_current",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(name)), Some(Value::String(prev))) =
                (row.get_by_name("name"), row.get_by_name("prev_or_current"))
            {
                results.insert(name.clone(), prev.clone());
            }
        }

        // Sales (sorted DESC): Alice(100), Bob(90), Carol(80)
        // IT (sorted DESC): Dave(95), Eve(85)
        // Frame is 1 preceding to current:
        // Alice: [Alice] -> Alice
        // Bob: [Alice, Bob] -> Alice
        // Carol: [Bob, Carol] -> Bob
        // Dave: [Dave] -> Dave (partition boundary)
        // Eve: [Dave, Eve] -> Dave
        assert_eq!(results.get("Alice"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Bob"), Some(&"Alice".to_string()));
        assert_eq!(results.get("Carol"), Some(&"Bob".to_string()));
        assert_eq!(results.get("Dave"), Some(&"Dave".to_string()));
        assert_eq!(results.get("Eve"), Some(&"Dave".to_string()));

        op.close().unwrap();
    }

    // ========== Aggregate Window Function Tests ==========

    #[test]
    fn sum_running_total() {
        // SUM(amount) OVER (ORDER BY date) - running total
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "amount".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(100)],
                vec![Value::Int(2), Value::Int(50)],
                vec![Value::Int(3), Value::Int(75)],
                vec![Value::Int(4), Value::Int(25)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Aggregate(AggregateWindowFunction::Sum),
            LogicalExpr::column("amount"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "running_total",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(total))) =
                (row.get_by_name("date"), row.get_by_name("running_total"))
            {
                results.push((*date, *total));
            }
        }

        // Sort by date to get consistent ordering
        results.sort_by_key(|(d, _)| *d);

        // Running totals: 100, 150, 225, 250
        assert_eq!(results, vec![(1, 100), (2, 150), (3, 225), (4, 250)]);

        op.close().unwrap();
    }

    #[test]
    fn avg_moving_average() {
        // AVG(value) OVER (ORDER BY date ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)
        // 2-row moving average
        use crate::ast::{
            AggregateWindowFunction, Expr, Literal, WindowFrame, WindowFrameBound, WindowFrameUnits,
        };

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "value".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(100)],
                vec![Value::Int(2), Value::Int(80)],
                vec![Value::Int(3), Value::Int(120)],
                vec![Value::Int(4), Value::Int(60)],
            ],
        ));

        let frame = Some(WindowFrame {
            units: WindowFrameUnits::Rows,
            start: WindowFrameBound::Preceding(Box::new(Expr::Literal(Literal::Integer(1)))),
            end: Some(WindowFrameBound::CurrentRow),
        });

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::Aggregate(AggregateWindowFunction::Avg),
            Some(LogicalExpr::column("value")),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            frame,
            "moving_avg",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, f64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Float(avg))) =
                (row.get_by_name("date"), row.get_by_name("moving_avg"))
            {
                results.push((*date, *avg));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Moving averages (2-row window):
        // date 1: avg(100) = 100.0
        // date 2: avg(100, 80) = 90.0
        // date 3: avg(80, 120) = 100.0
        // date 4: avg(120, 60) = 90.0
        assert_eq!(results.len(), 4);
        assert!((results[0].1 - 100.0).abs() < 0.001);
        assert!((results[1].1 - 90.0).abs() < 0.001);
        assert!((results[2].1 - 100.0).abs() < 0.001);
        assert!((results[3].1 - 90.0).abs() < 0.001);

        op.close().unwrap();
    }

    #[test]
    fn count_cumulative() {
        // COUNT(*) OVER (ORDER BY date) - cumulative count
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "event".to_string()],
            vec![
                vec![Value::Int(1), Value::from("A")],
                vec![Value::Int(2), Value::from("B")],
                vec![Value::Int(3), Value::from("C")],
                vec![Value::Int(4), Value::from("D")],
            ],
        ));

        // COUNT(*) - no argument means count all rows
        let window_expr = WindowFunctionExpr::new(
            WindowFunction::Aggregate(AggregateWindowFunction::Count),
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "cumulative_count",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(count))) =
                (row.get_by_name("date"), row.get_by_name("cumulative_count"))
            {
                results.push((*date, *count));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Cumulative counts: 1, 2, 3, 4
        assert_eq!(results, vec![(1, 1), (2, 2), (3, 3), (4, 4)]);

        op.close().unwrap();
    }

    #[test]
    fn min_cumulative() {
        // MIN(value) OVER (ORDER BY date) - cumulative minimum
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "value".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(50)],
                vec![Value::Int(2), Value::Int(30)],
                vec![Value::Int(3), Value::Int(70)],
                vec![Value::Int(4), Value::Int(20)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Aggregate(AggregateWindowFunction::Min),
            LogicalExpr::column("value"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "cumulative_min",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(min))) =
                (row.get_by_name("date"), row.get_by_name("cumulative_min"))
            {
                results.push((*date, *min));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Cumulative minimums: 50, 30, 30, 20
        assert_eq!(results, vec![(1, 50), (2, 30), (3, 30), (4, 20)]);

        op.close().unwrap();
    }

    #[test]
    fn max_cumulative() {
        // MAX(value) OVER (ORDER BY date) - cumulative maximum
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "value".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(50)],
                vec![Value::Int(2), Value::Int(80)],
                vec![Value::Int(3), Value::Int(40)],
                vec![Value::Int(4), Value::Int(100)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Aggregate(AggregateWindowFunction::Max),
            LogicalExpr::column("value"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "cumulative_max",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(max))) =
                (row.get_by_name("date"), row.get_by_name("cumulative_max"))
            {
                results.push((*date, *max));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Cumulative maximums: 50, 80, 80, 100
        assert_eq!(results, vec![(1, 50), (2, 80), (3, 80), (4, 100)]);

        op.close().unwrap();
    }

    #[test]
    fn sum_with_partition() {
        // SUM(amount) OVER (PARTITION BY dept ORDER BY date) - running total per department
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "date".to_string(), "amount".to_string()],
            vec![
                vec![Value::from("Sales"), Value::Int(1), Value::Int(100)],
                vec![Value::from("Sales"), Value::Int(2), Value::Int(50)],
                vec![Value::from("IT"), Value::Int(1), Value::Int(200)],
                vec![Value::from("IT"), Value::Int(2), Value::Int(75)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::Aggregate(AggregateWindowFunction::Sum),
            Some(LogicalExpr::column("amount")),
            None,
            vec![LogicalExpr::column("dept")],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            None,
            "dept_running_total",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: std::collections::HashMap<(String, i64), i64> =
            std::collections::HashMap::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::String(dept)), Some(Value::Int(date)), Some(Value::Int(total))) = (
                row.get_by_name("dept"),
                row.get_by_name("date"),
                row.get_by_name("dept_running_total"),
            ) {
                results.insert((dept.clone(), *date), *total);
            }
        }

        // Sales: date 1 -> 100, date 2 -> 150
        // IT: date 1 -> 200, date 2 -> 275
        assert_eq!(results.get(&("Sales".to_string(), 1)), Some(&100));
        assert_eq!(results.get(&("Sales".to_string(), 2)), Some(&150));
        assert_eq!(results.get(&("IT".to_string(), 1)), Some(&200));
        assert_eq!(results.get(&("IT".to_string(), 2)), Some(&275));

        op.close().unwrap();
    }

    #[test]
    fn sum_with_null_values() {
        // SUM(amount) should ignore NULL values
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "amount".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(100)],
                vec![Value::Int(2), Value::Null],
                vec![Value::Int(3), Value::Int(50)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Aggregate(AggregateWindowFunction::Sum),
            LogicalExpr::column("amount"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "running_total",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(total))) =
                (row.get_by_name("date"), row.get_by_name("running_total"))
            {
                results.push((*date, *total));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Running totals ignoring NULL: 100, 100, 150
        assert_eq!(results, vec![(1, 100), (2, 100), (3, 150)]);

        op.close().unwrap();
    }

    #[test]
    fn count_with_expression() {
        // COUNT(column) - counts non-NULL values
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["date".to_string(), "value".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(100)],
                vec![Value::Int(2), Value::Null],
                vec![Value::Int(3), Value::Int(50)],
                vec![Value::Int(4), Value::Null],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_arg(
            WindowFunction::Aggregate(AggregateWindowFunction::Count),
            LogicalExpr::column("value"),
            None,
            vec![],
            vec![SortOrder::asc(LogicalExpr::column("date"))],
            "non_null_count",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results: Vec<(i64, i64)> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let (Some(Value::Int(date)), Some(Value::Int(count))) =
                (row.get_by_name("date"), row.get_by_name("non_null_count"))
            {
                results.push((*date, *count));
            }
        }

        results.sort_by_key(|(d, _)| *d);

        // Non-NULL counts: 1, 1, 2, 2 (skips NULL values)
        assert_eq!(results, vec![(1, 1), (2, 1), (3, 2), (4, 2)]);

        op.close().unwrap();
    }

    #[test]
    fn avg_with_entire_partition_frame() {
        // AVG(value) OVER () - average over entire partition (no ORDER BY, no frame)
        use crate::ast::AggregateWindowFunction;

        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "value".to_string()],
            vec![
                vec![Value::from("A"), Value::Int(10)],
                vec![Value::from("B"), Value::Int(20)],
                vec![Value::from("C"), Value::Int(30)],
                vec![Value::from("D"), Value::Int(40)],
            ],
        ));

        let window_expr = WindowFunctionExpr::with_frame(
            WindowFunction::Aggregate(AggregateWindowFunction::Avg),
            Some(LogicalExpr::column("value")),
            None,
            vec![],
            vec![], // No ORDER BY = entire partition as frame
            None,
            "total_avg",
        );
        let mut op = WindowOp::new(vec![window_expr], input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut avgs: Vec<f64> = Vec::new();
        while let Some(row) = op.next().unwrap() {
            if let Some(Value::Float(avg)) = row.get_by_name("total_avg") {
                avgs.push(*avg);
            }
        }

        // All rows should have avg = (10+20+30+40)/4 = 25.0
        assert_eq!(avgs.len(), 4);
        for avg in avgs {
            assert!((avg - 25.0).abs() < 0.001);
        }

        op.close().unwrap();
    }
}
