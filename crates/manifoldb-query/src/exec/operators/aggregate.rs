//! Aggregate operators for GROUP BY and aggregation functions.
//!
//! This module provides aggregate operators that support:
//! - Hash-based aggregation for efficient grouping
//! - Sort-merge aggregation for pre-sorted data
//! - Full HAVING clause support including complex expressions with multiple aggregates
//!
//! # HAVING Clause Support
//!
//! The HAVING clause can reference:
//! - Simple aggregate comparisons: `HAVING COUNT(*) > 10`
//! - Multiple aggregates with AND/OR: `HAVING COUNT(*) > 10 AND AVG(salary) > 50000`
//! - Group columns: `HAVING department = 'Engineering' AND COUNT(*) > 5`
//! - Complex expressions: `HAVING SUM(amount) / COUNT(*) > 100`

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::ast::{BinaryOp, UnaryOp};
use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{AggregateFunction, LogicalExpr};

/// Hash-based aggregate operator.
///
/// Groups rows by key expressions and computes aggregates.
pub struct HashAggregateOp {
    /// Base operator state.
    base: OperatorBase,
    /// GROUP BY expressions.
    group_by: Vec<LogicalExpr>,
    /// Aggregate expressions.
    aggregates: Vec<LogicalExpr>,
    /// Optional HAVING clause.
    having: Option<LogicalExpr>,
    /// Input operator.
    input: BoxedOperator,
    /// Aggregation state: group key -> accumulators.
    groups: HashMap<Vec<u8>, GroupState>,
    /// Results iterator (consumes rows without cloning).
    results_iter: std::vec::IntoIter<Row>,
    /// Whether aggregation is complete.
    aggregated: bool,
    /// Reusable buffer for computing group keys.
    key_buffer: Vec<u8>,
    /// Maximum rows allowed in memory (0 = no limit).
    max_rows_in_memory: usize,
}

impl HashAggregateOp {
    /// Creates a new hash aggregate operator.
    #[must_use]
    pub fn new(
        group_by: Vec<LogicalExpr>,
        aggregates: Vec<LogicalExpr>,
        having: Option<LogicalExpr>,
        input: BoxedOperator,
    ) -> Self {
        // Build output schema - pre-allocate for group_by + aggregates
        let mut columns = Vec::with_capacity(group_by.len() + aggregates.len());
        for (i, expr) in group_by.iter().enumerate() {
            columns.push(expr_to_name(expr, i));
        }
        for (i, expr) in aggregates.iter().enumerate() {
            columns.push(expr_to_name(expr, group_by.len() + i));
        }
        let schema = Arc::new(Schema::new(columns));

        // Pre-allocate groups HashMap for typical query sizes
        const INITIAL_GROUPS_CAPACITY: usize = 1000;

        Self {
            base: OperatorBase::new(schema),
            group_by,
            aggregates,
            having,
            input,
            groups: HashMap::with_capacity(INITIAL_GROUPS_CAPACITY),
            results_iter: Vec::new().into_iter(),
            aggregated: false,
            key_buffer: Vec::with_capacity(64), // Pre-allocate for typical key sizes
            max_rows_in_memory: 0,              // Set in open() from context
        }
    }

    /// Computes the group key values for a row.
    fn compute_group_values(&self, row: &Row) -> OperatorResult<Vec<Value>> {
        self.group_by.iter().map(|expr| evaluate_expr(expr, row)).collect()
    }

    /// Aggregates all input rows.
    fn aggregate_all(&mut self) -> OperatorResult<()> {
        // Use a local buffer to work around borrow checker issues
        let mut key_buffer = std::mem::take(&mut self.key_buffer);

        while let Some(row) = self.input.next()? {
            // Compute key into reusable buffer (avoids allocation per row)
            key_buffer.clear();
            for expr in &self.group_by {
                let value = evaluate_expr(expr, &row)?;
                encode_value(&value, &mut key_buffer);
            }

            // Check if this is a new group and check limit before inserting
            let is_new_group = !self.groups.contains_key(&key_buffer);
            if is_new_group
                && self.max_rows_in_memory > 0
                && self.groups.len() >= self.max_rows_in_memory
            {
                // Restore buffer before returning error
                self.key_buffer = key_buffer;
                return Err(ParseError::QueryTooLarge {
                    actual: self.groups.len() + 1,
                    limit: self.max_rows_in_memory,
                });
            }

            // Only clone the key when inserting a new group
            let state = if let Some(state) = self.groups.get_mut(&key_buffer) {
                state
            } else {
                let group_values = self.compute_group_values(&row)?;
                self.groups
                    .entry(key_buffer.clone())
                    .or_insert_with(|| GroupState::new(group_values, self.aggregates.len()))
            };

            // Update each aggregate
            for (i, agg_expr) in self.aggregates.iter().enumerate() {
                if let LogicalExpr::AggregateFunction { func, args, distinct: _ } = agg_expr {
                    let is_wildcard =
                        args.first().is_some_and(|a| matches!(a, LogicalExpr::Wildcard));
                    // Evaluate all arguments
                    let arg_values: Vec<Value> =
                        args.iter().map(|a| evaluate_expr(a, &row)).collect::<Result<_, _>>()?;
                    state.accumulators[i].update(func, &arg_values, is_wildcard);
                }
            }
        }

        // Restore the buffer for potential reuse
        self.key_buffer = key_buffer;

        // Build result rows
        let schema = self.base.schema();
        let mut results = Vec::with_capacity(self.groups.len());
        for state in self.groups.values() {
            let mut values = state.group_values.clone();
            for acc in &state.accumulators {
                values.push(acc.result());
            }

            let row = Row::new(Arc::clone(&schema), values);

            // Apply HAVING filter with full aggregate expression support
            if let Some(having) = &self.having {
                let result =
                    evaluate_having_expr(having, &row, &self.aggregates, self.group_by.len())?;
                if !matches!(result, Value::Bool(true)) {
                    continue;
                }
            }

            results.push(row);
        }

        // Handle case with no groups (scalar aggregation)
        if self.group_by.is_empty() && self.groups.is_empty() {
            // Return single row with default aggregate values
            let mut values = Vec::new();
            for agg_expr in &self.aggregates {
                if let LogicalExpr::AggregateFunction { func, .. } = agg_expr {
                    values.push(Accumulator::new().default_for(func));
                } else {
                    values.push(Value::Null);
                }
            }
            let row = Row::new(Arc::clone(&schema), values);
            results.push(row);
        }

        // Convert to iterator for zero-copy consumption
        self.results_iter = results.into_iter();
        self.aggregated = true;
        Ok(())
    }
}

impl Operator for HashAggregateOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.groups.clear();
        self.results_iter = Vec::new().into_iter();
        self.aggregated = false;
        self.max_rows_in_memory = ctx.max_rows_in_memory();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if !self.aggregated {
            self.aggregate_all()?;
        }

        // Iterator yields owned rows without cloning
        match self.results_iter.next() {
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
        self.groups.clear();
        self.results_iter = Vec::new().into_iter();
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
        "HashAggregate"
    }
}

/// Sort-merge based aggregate operator.
///
/// Assumes input is sorted by group keys.
pub struct SortMergeAggregateOp {
    /// Base operator state.
    base: OperatorBase,
    /// GROUP BY expressions.
    group_by: Vec<LogicalExpr>,
    /// Aggregate expressions.
    aggregates: Vec<LogicalExpr>,
    /// Optional HAVING clause.
    having: Option<LogicalExpr>,
    /// Input operator.
    input: BoxedOperator,
    /// Current group key.
    current_key: Option<Vec<u8>>,
    /// Current group values.
    current_values: Vec<Value>,
    /// Current accumulators.
    accumulators: Vec<Accumulator>,
    /// Pending row from previous iteration.
    pending_row: Option<Row>,
    /// Whether we've finished.
    finished: bool,
    /// Reusable buffer for computing group keys.
    key_buffer: Vec<u8>,
}

impl SortMergeAggregateOp {
    /// Creates a new sort-merge aggregate operator.
    #[must_use]
    pub fn new(
        group_by: Vec<LogicalExpr>,
        aggregates: Vec<LogicalExpr>,
        having: Option<LogicalExpr>,
        input: BoxedOperator,
    ) -> Self {
        // Pre-allocate for group_by + aggregates
        let mut columns = Vec::with_capacity(group_by.len() + aggregates.len());
        for (i, expr) in group_by.iter().enumerate() {
            columns.push(expr_to_name(expr, i));
        }
        for (i, expr) in aggregates.iter().enumerate() {
            columns.push(expr_to_name(expr, group_by.len() + i));
        }
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            group_by,
            aggregates,
            having,
            input,
            current_key: None,
            current_values: Vec::with_capacity(8), // Pre-allocate for typical group size
            accumulators: Vec::new(),
            pending_row: None,
            finished: false,
            key_buffer: Vec::with_capacity(64), // Pre-allocate for typical key sizes
        }
    }

    /// Computes the group key for a row into the provided buffer.
    fn compute_group_key_into(&self, row: &Row, buf: &mut Vec<u8>) -> OperatorResult<()> {
        buf.clear();
        for expr in &self.group_by {
            let value = evaluate_expr(expr, row)?;
            encode_value(&value, buf);
        }
        Ok(())
    }

    fn compute_group_values(&self, row: &Row) -> OperatorResult<Vec<Value>> {
        self.group_by.iter().map(|expr| evaluate_expr(expr, row)).collect()
    }

    fn init_accumulators(&mut self) {
        self.accumulators = (0..self.aggregates.len()).map(|_| Accumulator::new()).collect();
    }

    fn update_accumulators(&mut self, row: &Row) -> OperatorResult<()> {
        for (i, agg_expr) in self.aggregates.iter().enumerate() {
            if let LogicalExpr::AggregateFunction { func, args, .. } = agg_expr {
                let is_wildcard = args.first().is_some_and(|a| matches!(a, LogicalExpr::Wildcard));
                let arg_values: Vec<Value> =
                    args.iter().map(|a| evaluate_expr(a, row)).collect::<Result<_, _>>()?;
                self.accumulators[i].update(func, &arg_values, is_wildcard);
            }
        }
        Ok(())
    }

    fn build_result(&self) -> Row {
        let mut values = self.current_values.clone();
        for acc in &self.accumulators {
            values.push(acc.result());
        }
        Row::new(self.base.schema(), values)
    }
}

impl Operator for SortMergeAggregateOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.current_key = None;
        self.current_values.clear();
        self.accumulators.clear();
        self.pending_row = None;
        self.finished = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.finished {
            return Ok(None);
        }

        // Take the key buffer to avoid borrow checker issues
        let mut key_buffer = std::mem::take(&mut self.key_buffer);

        let result = self.next_inner(&mut key_buffer);

        // Restore the buffer
        self.key_buffer = key_buffer;

        result
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
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
        "SortMergeAggregate"
    }
}

impl SortMergeAggregateOp {
    /// Inner implementation of next() that uses a reusable key buffer.
    fn next_inner(&mut self, key_buffer: &mut Vec<u8>) -> OperatorResult<Option<Row>> {
        loop {
            // Get next row
            let row =
                if let Some(r) = self.pending_row.take() { Some(r) } else { self.input.next()? };

            match row {
                Some(row) => {
                    // Compute key into reusable buffer
                    self.compute_group_key_into(&row, key_buffer)?;

                    if self.current_key.as_deref() == Some(key_buffer.as_slice()) {
                        // Same group, accumulate
                        self.update_accumulators(&row)?;
                    } else if self.current_key.is_some() {
                        // New group, output previous group
                        self.pending_row = Some(row.clone());
                        let result = self.build_result();

                        // Start new group - only clone the key when starting a new group
                        self.current_key = Some(key_buffer.clone());
                        self.current_values = self.compute_group_values(&row)?;
                        self.init_accumulators();
                        self.update_accumulators(&row)?;

                        // Check HAVING with full aggregate expression support
                        if let Some(having) = &self.having {
                            let check = evaluate_having_expr(
                                having,
                                &result,
                                &self.aggregates,
                                self.group_by.len(),
                            )?;
                            if !matches!(check, Value::Bool(true)) {
                                continue;
                            }
                        }

                        self.base.inc_rows_produced();
                        return Ok(Some(result));
                    } else {
                        // First group - only clone the key here
                        self.current_key = Some(key_buffer.clone());
                        self.current_values = self.compute_group_values(&row)?;
                        self.init_accumulators();
                        self.update_accumulators(&row)?;
                    }
                }
                None => {
                    // End of input, output last group
                    self.finished = true;
                    if self.current_key.is_some() {
                        let result = self.build_result();

                        if let Some(having) = &self.having {
                            let check = evaluate_having_expr(
                                having,
                                &result,
                                &self.aggregates,
                                self.group_by.len(),
                            )?;
                            if !matches!(check, Value::Bool(true)) {
                                self.base.set_finished();
                                return Ok(None);
                            }
                        }

                        self.base.inc_rows_produced();
                        return Ok(Some(result));
                    }
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }
}

/// State for a single group.
#[derive(Debug)]
struct GroupState {
    /// Group key values.
    group_values: Vec<Value>,
    /// Accumulators for each aggregate.
    accumulators: Vec<Accumulator>,
}

impl GroupState {
    fn new(group_values: Vec<Value>, num_aggregates: usize) -> Self {
        Self {
            group_values,
            accumulators: (0..num_aggregates).map(|_| Accumulator::new()).collect(),
        }
    }
}

/// Accumulator for aggregate functions.
#[derive(Debug, Default)]
struct Accumulator {
    /// The aggregate function being computed.
    func: Option<AggregateFunction>,
    count: i64,
    sum: f64,
    /// Sum of squared values for variance/stddev calculations.
    sum_sq: f64,
    min: Option<Value>,
    max: Option<Value>,
    /// Collected values for array_agg and json_agg/jsonb_agg.
    array_values: Vec<Value>,
    /// Collected strings for string_agg.
    string_values: Vec<String>,
    /// Delimiter for string_agg (captured from first row).
    string_delimiter: Option<String>,
    /// Collected numeric values for percentile calculations.
    percentile_values: Vec<f64>,
    /// Percentile argument (0.0 to 1.0) for percentile functions.
    percentile_arg: Option<f64>,
    /// Collected key-value pairs for json_object_agg/jsonb_object_agg.
    object_entries: Vec<(String, Value)>,
    /// Boolean accumulator for bool_and (true if all true) and bool_or (true if any true).
    /// None means no non-NULL values seen yet.
    bool_result: Option<bool>,
}

impl Accumulator {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, func: &AggregateFunction, values: &[Value], is_wildcard: bool) {
        // Store the function type on first update
        if self.func.is_none() {
            self.func = Some(*func);
        }

        // Get the primary value (first argument)
        let value = values.first().unwrap_or(&Value::Null);

        // For COUNT(*), always count (even NULLs)
        if matches!(func, AggregateFunction::Count) && is_wildcard {
            self.count += 1;
            return;
        }

        // Skip NULLs for most aggregates (including COUNT(column))
        if matches!(value, Value::Null) {
            return;
        }

        self.count += 1;

        match func {
            AggregateFunction::Count => {
                // Already counted above
            }
            AggregateFunction::Sum | AggregateFunction::Avg => {
                self.sum += value_to_f64(value);
            }
            AggregateFunction::Min => {
                self.min = Some(match &self.min {
                    None => value.clone(),
                    Some(m) => {
                        if compare_values(value, m) < 0 {
                            value.clone()
                        } else {
                            m.clone()
                        }
                    }
                });
            }
            AggregateFunction::Max => {
                self.max = Some(match &self.max {
                    None => value.clone(),
                    Some(m) => {
                        if compare_values(value, m) > 0 {
                            value.clone()
                        } else {
                            m.clone()
                        }
                    }
                });
            }
            AggregateFunction::ArrayAgg => {
                // Collect values into array (skip NULLs already handled above)
                self.array_values.push(value.clone());
            }
            AggregateFunction::StringAgg => {
                // Get the delimiter from the second argument (capture on first call)
                if self.string_delimiter.is_none() {
                    self.string_delimiter = Some(
                        values
                            .get(1)
                            .map(|v| match v {
                                Value::String(s) => s.clone(),
                                _ => ",".to_string(), // Default delimiter
                            })
                            .unwrap_or_else(|| ",".to_string()),
                    );
                }
                // Convert value to string and collect
                let s = match value {
                    Value::String(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => String::new(),
                };
                if !s.is_empty() {
                    self.string_values.push(s);
                }
            }
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarianceSamp
            | AggregateFunction::VariancePop => {
                // For variance/stddev we need sum and sum of squares
                let v = value_to_f64(value);
                self.sum += v;
                self.sum_sq += v * v;
            }
            AggregateFunction::PercentileCont | AggregateFunction::PercentileDisc => {
                // Capture the percentile argument from the first argument on first call
                // Note: In Cypher, percentileCont(0.5, n.value) has percentile first
                if self.percentile_arg.is_none() {
                    self.percentile_arg = values.first().map(value_to_f64);
                }
                // Collect the numeric values from the second argument
                if let Some(expr_value) = values.get(1) {
                    if !matches!(expr_value, Value::Null) {
                        self.percentile_values.push(value_to_f64(expr_value));
                    }
                }
            }
            AggregateFunction::JsonAgg | AggregateFunction::JsonbAgg => {
                // Collect values into array for JSON conversion (skip NULLs already handled above)
                self.array_values.push(value.clone());
            }
            AggregateFunction::JsonObjectAgg | AggregateFunction::JsonbObjectAgg => {
                // json_object_agg(key, value) - collect key-value pairs
                // Skip if key is NULL (already handled above for first argument)
                // Convert key to string
                let key_str = match value {
                    Value::String(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => return, // Skip non-stringifiable keys
                };
                // Get value from second argument
                let val = values.get(1).cloned().unwrap_or(Value::Null);
                self.object_entries.push((key_str, val));
            }
            AggregateFunction::VectorAvg | AggregateFunction::VectorCentroid => {
                // Vector aggregates - not implemented yet
            }
            AggregateFunction::BoolAnd => {
                // bool_and: true if ALL non-NULL values are true
                let b = match value {
                    Value::Bool(b) => *b,
                    // For non-boolean values, treat non-NULL non-false as true
                    Value::Int(i) => *i != 0,
                    _ => true,
                };
                self.bool_result = Some(self.bool_result.unwrap_or(true) && b);
            }
            AggregateFunction::BoolOr => {
                // bool_or: true if ANY non-NULL value is true
                let b = match value {
                    Value::Bool(b) => *b,
                    // For non-boolean values, treat non-NULL non-zero as true
                    Value::Int(i) => *i != 0,
                    _ => true,
                };
                self.bool_result = Some(self.bool_result.unwrap_or(false) || b);
            }
        }
    }

    fn result(&self) -> Value {
        match &self.func {
            Some(AggregateFunction::Count) => Value::Int(self.count),
            Some(AggregateFunction::Sum) => {
                if self.count > 0 {
                    Value::Float(self.sum)
                } else {
                    Value::Null
                }
            }
            Some(AggregateFunction::Avg) => {
                if self.count > 0 {
                    Value::Float(self.sum / self.count as f64)
                } else {
                    Value::Null
                }
            }
            Some(AggregateFunction::Min) => self.min.clone().unwrap_or(Value::Null),
            Some(AggregateFunction::Max) => self.max.clone().unwrap_or(Value::Null),
            Some(AggregateFunction::ArrayAgg) => {
                if self.array_values.is_empty() {
                    Value::Null
                } else {
                    Value::Array(self.array_values.clone())
                }
            }
            Some(AggregateFunction::StringAgg) => {
                if self.string_values.is_empty() {
                    Value::Null
                } else {
                    let delimiter = self.string_delimiter.as_deref().unwrap_or(",");
                    Value::String(self.string_values.join(delimiter))
                }
            }
            Some(AggregateFunction::VarianceSamp) => {
                // Sample variance: sum((x - mean)^2) / (n - 1)
                // Using computational formula: (sum_sq - sum^2/n) / (n - 1)
                if self.count < 2 {
                    Value::Null
                } else {
                    let n = self.count as f64;
                    let variance = (self.sum_sq - (self.sum * self.sum) / n) / (n - 1.0);
                    Value::Float(variance)
                }
            }
            Some(AggregateFunction::VariancePop) => {
                // Population variance: sum((x - mean)^2) / n
                // Using computational formula: (sum_sq - sum^2/n) / n
                if self.count == 0 {
                    Value::Null
                } else {
                    let n = self.count as f64;
                    let variance = (self.sum_sq - (self.sum * self.sum) / n) / n;
                    Value::Float(variance)
                }
            }
            Some(AggregateFunction::StddevSamp) => {
                // Sample stddev: sqrt(sample variance)
                if self.count < 2 {
                    Value::Null
                } else {
                    let n = self.count as f64;
                    let variance = (self.sum_sq - (self.sum * self.sum) / n) / (n - 1.0);
                    Value::Float(variance.sqrt())
                }
            }
            Some(AggregateFunction::StddevPop) => {
                // Population stddev: sqrt(population variance)
                if self.count == 0 {
                    Value::Null
                } else {
                    let n = self.count as f64;
                    let variance = (self.sum_sq - (self.sum * self.sum) / n) / n;
                    Value::Float(variance.sqrt())
                }
            }
            Some(AggregateFunction::PercentileCont) => {
                // Continuous percentile: interpolates between values
                if self.percentile_values.is_empty() {
                    return Value::Null;
                }
                let percentile = self.percentile_arg.unwrap_or(0.5);
                let mut sorted = self.percentile_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = sorted.len();
                if n == 1 {
                    return Value::Float(sorted[0]);
                }

                // Calculate the index position (0-based)
                let pos = percentile * (n - 1) as f64;
                let lower_idx = pos.floor() as usize;
                let upper_idx = pos.ceil() as usize;
                let frac = pos - pos.floor();

                if lower_idx == upper_idx || upper_idx >= n {
                    Value::Float(sorted[lower_idx.min(n - 1)])
                } else {
                    // Linear interpolation
                    let result = sorted[lower_idx] + frac * (sorted[upper_idx] - sorted[lower_idx]);
                    Value::Float(result)
                }
            }
            Some(AggregateFunction::PercentileDisc) => {
                // Discrete percentile: returns exact value from set
                if self.percentile_values.is_empty() {
                    return Value::Null;
                }
                let percentile = self.percentile_arg.unwrap_or(0.5);
                let mut sorted = self.percentile_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let n = sorted.len();
                // Find the first value whose position >= percentile * n
                let idx = ((percentile * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
                Value::Float(sorted[idx])
            }
            Some(AggregateFunction::JsonAgg | AggregateFunction::JsonbAgg) => {
                if self.array_values.is_empty() {
                    Value::Null
                } else {
                    // Convert collected values to JSON array string
                    let json_arr: Vec<serde_json::Value> =
                        self.array_values.iter().map(value_to_json).collect();
                    serde_json::to_string(&json_arr).map(Value::String).unwrap_or(Value::Null)
                }
            }
            Some(AggregateFunction::JsonObjectAgg | AggregateFunction::JsonbObjectAgg) => {
                if self.object_entries.is_empty() {
                    Value::Null
                } else {
                    // Convert collected key-value pairs to JSON object string
                    let mut obj = serde_json::Map::new();
                    for (key, val) in &self.object_entries {
                        obj.insert(key.clone(), value_to_json(val));
                    }
                    serde_json::to_string(&serde_json::Value::Object(obj))
                        .map(Value::String)
                        .unwrap_or(Value::Null)
                }
            }
            Some(AggregateFunction::VectorAvg | AggregateFunction::VectorCentroid) => Value::Null,
            Some(AggregateFunction::BoolAnd | AggregateFunction::BoolOr) => {
                // Returns NULL if no non-NULL values were seen
                self.bool_result.map(Value::Bool).unwrap_or(Value::Null)
            }
            None => {
                // Fallback for unknown or unset function type
                if self.min.is_some() {
                    return self.min.clone().unwrap_or(Value::Null);
                }
                if self.max.is_some() {
                    return self.max.clone().unwrap_or(Value::Null);
                }
                Value::Int(self.count)
            }
        }
    }

    fn default_for(&self, func: &AggregateFunction) -> Value {
        match func {
            AggregateFunction::Count => Value::Int(0),
            AggregateFunction::Sum
            | AggregateFunction::Avg
            | AggregateFunction::Min
            | AggregateFunction::Max => Value::Null,
            AggregateFunction::ArrayAgg => Value::Null,
            AggregateFunction::StringAgg => Value::Null,
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarianceSamp
            | AggregateFunction::VariancePop => Value::Null,
            AggregateFunction::PercentileCont | AggregateFunction::PercentileDisc => Value::Null,
            AggregateFunction::JsonAgg | AggregateFunction::JsonbAgg => Value::Null,
            AggregateFunction::JsonObjectAgg | AggregateFunction::JsonbObjectAgg => Value::Null,
            AggregateFunction::VectorAvg | AggregateFunction::VectorCentroid => Value::Null,
            AggregateFunction::BoolAnd | AggregateFunction::BoolOr => Value::Null,
        }
    }
}

/// Encodes bytes as base64 string for JSON serialization.
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
            result.push('=');
        }
        if chunk.len() > 2 {
            let _ = write!(result, "{}", BASE64_CHARS[(b2 & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Converts a `Value` to a `serde_json::Value` for JSON aggregate functions.
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
        Value::Bytes(b) => {
            // Encode bytes as base64 string
            serde_json::Value::String(base64_encode(b))
        }
        Value::Point { x, y, z, srid } => {
            let mut map = serde_json::Map::new();
            map.insert("x".to_string(), serde_json::json!(*x));
            map.insert("y".to_string(), serde_json::json!(*y));
            if let Some(z_val) = z {
                map.insert("z".to_string(), serde_json::json!(*z_val));
            }
            map.insert("srid".to_string(), serde_json::json!(*srid));
            serde_json::Value::Object(map)
        }
    }
}

/// Encodes a value to bytes for hashing.
fn encode_value(value: &Value, buf: &mut Vec<u8>) {
    match value {
        Value::Null => buf.push(0),
        Value::Bool(b) => {
            buf.push(1);
            buf.push(u8::from(*b));
        }
        Value::Int(i) => {
            buf.push(2);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(3);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            buf.push(4);
            buf.extend_from_slice(s.as_bytes());
            buf.push(0);
        }
        _ => buf.push(0),
    }
}

/// Converts a value to f64 for numeric aggregation.
fn value_to_f64(value: &Value) -> f64 {
    match value {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        _ => 0.0,
    }
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> i32 {
    match (a, b) {
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
        (Value::String(a), Value::String(b)) => a.cmp(b) as i32,
        _ => 0,
    }
}

/// Gets a name from an expression.
fn expr_to_name(expr: &LogicalExpr, index: usize) -> String {
    match expr {
        LogicalExpr::Column { name, .. } => name.clone(),
        LogicalExpr::Alias { alias, .. } => alias.clone(),
        LogicalExpr::AggregateFunction { func, .. } => format!("{func}"),
        _ => format!("col_{index}"),
    }
}

/// Evaluates a HAVING expression against an aggregate result row.
///
/// This function handles aggregate function expressions by:
/// 1. Looking up the aggregate value from the computed column in the row
/// 2. Recursively evaluating complex expressions (AND, OR, comparisons)
/// 3. Falling back to standard expression evaluation for non-aggregate expressions
///
/// # Arguments
///
/// * `expr` - The HAVING expression to evaluate
/// * `row` - The aggregate result row containing group columns and computed aggregates
/// * `aggregates` - The list of aggregate expressions used in the query
/// * `group_by_len` - Number of GROUP BY columns (aggregates start after this)
///
/// # Complex Expression Support
///
/// Supports expressions like:
/// - `COUNT(*) > 10 AND AVG(salary) > 50000`
/// - `SUM(amount) / COUNT(*) > 100`
/// - `department = 'Engineering' AND COUNT(*) > 5`
fn evaluate_having_expr(
    expr: &LogicalExpr,
    row: &Row,
    aggregates: &[LogicalExpr],
    group_by_len: usize,
) -> OperatorResult<Value> {
    match expr {
        // Handle aggregate function references by looking up the computed value
        LogicalExpr::AggregateFunction { func, args, distinct } => {
            // Find matching aggregate in the aggregates list
            for (i, agg) in aggregates.iter().enumerate() {
                if let LogicalExpr::AggregateFunction {
                    func: agg_func,
                    args: agg_args,
                    distinct: agg_distinct,
                } = agg
                {
                    if func == agg_func && distinct == agg_distinct && args_match(args, agg_args) {
                        // Found the matching aggregate - get its value from the row
                        let col_idx = group_by_len + i;
                        return Ok(row.get(col_idx).cloned().unwrap_or(Value::Null));
                    }
                }
            }
            // Aggregate not found - return NULL
            Ok(Value::Null)
        }

        // Handle binary operations (AND, OR, comparisons)
        LogicalExpr::BinaryOp { left, op, right } => {
            let left_val = evaluate_having_expr(left, row, aggregates, group_by_len)?;
            let right_val = evaluate_having_expr(right, row, aggregates, group_by_len)?;
            evaluate_binary_op(&left_val, op, &right_val)
        }

        // Handle unary operations (NOT, etc.)
        LogicalExpr::UnaryOp { op, operand } => {
            let val = evaluate_having_expr(operand, row, aggregates, group_by_len)?;
            match op {
                UnaryOp::Not => match val {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    Value::Null => Ok(Value::Null),
                    _ => Ok(Value::Null),
                },
                UnaryOp::Neg => match val {
                    Value::Int(i) => Ok(Value::Int(-i)),
                    Value::Float(f) => Ok(Value::Float(-f)),
                    _ => Ok(Value::Null),
                },
                UnaryOp::IsNull => Ok(Value::Bool(matches!(val, Value::Null))),
                UnaryOp::IsNotNull => Ok(Value::Bool(!matches!(val, Value::Null))),
            }
        }

        // For column references and other expressions, use standard evaluation
        _ => evaluate_expr(expr, row),
    }
}

/// Checks if two sets of aggregate arguments match.
fn args_match(args1: &[LogicalExpr], args2: &[LogicalExpr]) -> bool {
    if args1.len() != args2.len() {
        return false;
    }
    for (a, b) in args1.iter().zip(args2.iter()) {
        if !expr_matches(a, b) {
            return false;
        }
    }
    true
}

/// Checks if two expressions are structurally equivalent for aggregate matching.
fn expr_matches(a: &LogicalExpr, b: &LogicalExpr) -> bool {
    match (a, b) {
        (LogicalExpr::Wildcard, LogicalExpr::Wildcard) => true,
        (
            LogicalExpr::Column { name: n1, qualifier: q1 },
            LogicalExpr::Column { name: n2, qualifier: q2 },
        ) => n1 == n2 && q1 == q2,
        (LogicalExpr::Literal(l1), LogicalExpr::Literal(l2)) => l1 == l2,
        _ => false,
    }
}

/// Evaluates a binary operation for HAVING expressions.
fn evaluate_binary_op(left: &Value, op: &BinaryOp, right: &Value) -> OperatorResult<Value> {
    match op {
        BinaryOp::And => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
            (Value::Bool(false), _) | (_, Value::Bool(false)) => Ok(Value::Bool(false)),
            _ => Ok(Value::Null),
        },
        BinaryOp::Or => match (left, right) {
            (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
            (Value::Bool(true), _) | (_, Value::Bool(true)) => Ok(Value::Bool(true)),
            _ => Ok(Value::Null),
        },
        BinaryOp::Eq => Ok(Value::Bool(values_equal(left, right))),
        BinaryOp::NotEq => Ok(Value::Bool(!values_equal(left, right))),
        BinaryOp::Lt => compare_values_op(left, right, |ord| ord == std::cmp::Ordering::Less),
        BinaryOp::LtEq => compare_values_op(left, right, |ord| {
            ord == std::cmp::Ordering::Less || ord == std::cmp::Ordering::Equal
        }),
        BinaryOp::Gt => compare_values_op(left, right, |ord| ord == std::cmp::Ordering::Greater),
        BinaryOp::GtEq => compare_values_op(left, right, |ord| {
            ord == std::cmp::Ordering::Greater || ord == std::cmp::Ordering::Equal
        }),
        BinaryOp::Add => numeric_op(left, right, |a, b| a + b, |a, b| a + b),
        BinaryOp::Sub => numeric_op(left, right, |a, b| a - b, |a, b| a - b),
        BinaryOp::Mul => numeric_op(left, right, |a, b| a * b, |a, b| a * b),
        BinaryOp::Div => {
            // Handle division by zero
            let is_zero =
                matches!(right, Value::Int(0)) || matches!(right, Value::Float(f) if *f == 0.0);
            if is_zero {
                Ok(Value::Null)
            } else {
                numeric_op(left, right, |a, b| a / b, |a, b| a / b)
            }
        }
        _ => Ok(Value::Null),
    }
}

/// Checks if two values are equal.
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => false, // NULL != NULL in SQL
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            ((*a as f64) - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        _ => false,
    }
}

/// Compares two values and applies a predicate to the ordering.
fn compare_values_op<F>(left: &Value, right: &Value, pred: F) -> OperatorResult<Value>
where
    F: Fn(std::cmp::Ordering) -> bool,
{
    let ord = match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => {
            (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        }
        (Value::Float(a), Value::Int(b)) => {
            a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal)
        }
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => return Ok(Value::Null),
    };
    Ok(Value::Bool(pred(ord)))
}

/// Performs a numeric operation on two values.
fn numeric_op<F, G>(left: &Value, right: &Value, int_op: F, float_op: G) -> OperatorResult<Value>
where
    F: Fn(i64, i64) -> i64,
    G: Fn(f64, f64) -> f64,
{
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
        _ => Ok(Value::Null),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("A"), Value::Int(100)],
                vec![Value::from("A"), Value::Int(150)],
                vec![Value::from("B"), Value::Int(200)],
                vec![Value::from("A"), Value::Int(125)],
                vec![Value::from("B"), Value::Int(180)],
            ],
        ))
    }

    #[test]
    fn hash_aggregate_count() {
        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::count(LogicalExpr::wildcard(), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        // Should have 2 groups (A and B)
        assert_eq!(rows.len(), 2);

        // Find and check each group
        for row in &rows {
            let dept = row.get(0).unwrap();
            let count = row.get(1).unwrap();
            if dept == &Value::from("A") {
                assert_eq!(count, &Value::Int(3));
            } else if dept == &Value::from("B") {
                assert_eq!(count, &Value::Int(2));
            }
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_sum() {
        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::sum(LogicalExpr::column("salary"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        // Should have 2 groups (A and B)
        assert_eq!(rows.len(), 2);

        // Find and check each group
        for row in &rows {
            let dept = row.get(0).unwrap();
            let sum = row.get(1).unwrap();
            if dept == &Value::from("A") {
                assert_eq!(sum, &Value::Float(375.0));
            } else if dept == &Value::from("B") {
                assert_eq!(sum, &Value::Float(380.0));
            }
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_min_max() {
        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![
            LogicalExpr::min(LogicalExpr::column("salary")),
            LogicalExpr::max(LogicalExpr::column("salary")),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut found_a = false;
        while let Some(row) = op.next().unwrap() {
            if row.get(0) == Some(&Value::from("A")) {
                assert_eq!(row.get(1), Some(&Value::Int(100))); // min
                assert_eq!(row.get(2), Some(&Value::Int(150))); // max
                found_a = true;
            }
        }
        assert!(found_a);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_no_groups() {
        // Scalar aggregation (no GROUP BY)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::count(LogicalExpr::wildcard(), false),
            LogicalExpr::sum(LogicalExpr::column("n"), false),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(3))); // count
        assert_eq!(row.get(1), Some(&Value::Float(6.0))); // sum

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_avg_with_nulls() {
        // Test AVG with NULL values: AVG(10, NULL, 20) = 15.0
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["val".to_string()],
            vec![vec![Value::Int(10)], vec![Value::Null], vec![Value::Int(20)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::avg(LogicalExpr::column("val"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // AVG should be (10 + 20) / 2 = 15.0, NOT 30.0 (the sum)
        assert_eq!(row.get(0), Some(&Value::Float(15.0)));

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_avg_vs_sum() {
        // Ensure AVG and SUM return different values
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(10)], vec![Value::Int(20)], vec![Value::Int(30)]],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::sum(LogicalExpr::column("n"), false),
            LogicalExpr::avg(LogicalExpr::column("n"), false),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Float(60.0))); // SUM = 60
        assert_eq!(row.get(1), Some(&Value::Float(20.0))); // AVG = 60/3 = 20

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    // ========== Tests for array_agg ==========

    #[test]
    fn hash_aggregate_array_agg_basic() {
        // Test basic array_agg - collects all values into an array
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string()],
            vec![
                vec![Value::from("Alice")],
                vec![Value::from("Bob")],
                vec![Value::from("Charlie")],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::array_agg(LogicalExpr::column("name"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        // Should be an array with all names
        if let Value::Array(arr) = result {
            assert_eq!(arr.len(), 3);
            assert!(arr.contains(&Value::from("Alice")));
            assert!(arr.contains(&Value::from("Bob")));
            assert!(arr.contains(&Value::from("Charlie")));
        } else {
            panic!("Expected Array, got {:?}", result);
        }

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_array_agg_with_group_by() {
        // Test array_agg with GROUP BY
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "name".to_string()],
            vec![
                vec![Value::from("Engineering"), Value::from("Alice")],
                vec![Value::from("Engineering"), Value::from("Bob")],
                vec![Value::from("Sales"), Value::from("Charlie")],
                vec![Value::from("Engineering"), Value::from("Dave")],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::array_agg(LogicalExpr::column("name"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 2);

        for row in &rows {
            let dept = row.get(0).unwrap();
            let names = row.get(1).unwrap();

            if dept == &Value::from("Engineering") {
                if let Value::Array(arr) = names {
                    assert_eq!(arr.len(), 3);
                } else {
                    panic!("Expected Array");
                }
            } else if dept == &Value::from("Sales") {
                if let Value::Array(arr) = names {
                    assert_eq!(arr.len(), 1);
                    assert_eq!(arr[0], Value::from("Charlie"));
                } else {
                    panic!("Expected Array");
                }
            }
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_array_agg_with_nulls() {
        // Test that array_agg skips NULLs
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string()],
            vec![vec![Value::from("Alice")], vec![Value::Null], vec![Value::from("Bob")]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::array_agg(LogicalExpr::column("name"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        // Should only have non-NULL values
        if let Value::Array(arr) = result {
            assert_eq!(arr.len(), 2);
            assert!(arr.contains(&Value::from("Alice")));
            assert!(arr.contains(&Value::from("Bob")));
        } else {
            panic!("Expected Array, got {:?}", result);
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_array_agg_integers() {
        // Test array_agg with integers
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(10)], vec![Value::Int(20)], vec![Value::Int(30)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::array_agg(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        if let Value::Array(arr) = result {
            assert_eq!(arr.len(), 3);
            assert!(arr.contains(&Value::Int(10)));
            assert!(arr.contains(&Value::Int(20)));
            assert!(arr.contains(&Value::Int(30)));
        } else {
            panic!("Expected Array, got {:?}", result);
        }

        op.close().unwrap();
    }

    // ========== Tests for string_agg ==========

    #[test]
    fn hash_aggregate_string_agg_basic() {
        // Test basic string_agg with comma delimiter
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string()],
            vec![
                vec![Value::from("Alice")],
                vec![Value::from("Bob")],
                vec![Value::from("Charlie")],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::string_agg(
            LogicalExpr::column("name"),
            LogicalExpr::Literal(crate::ast::Literal::String(", ".to_string())),
            false,
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        if let Value::String(s) = result {
            // Order is not guaranteed in hash aggregation, so check components
            let parts: Vec<&str> = s.split(", ").collect();
            assert_eq!(parts.len(), 3);
            assert!(parts.contains(&"Alice"));
            assert!(parts.contains(&"Bob"));
            assert!(parts.contains(&"Charlie"));
        } else {
            panic!("Expected String, got {:?}", result);
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_string_agg_with_group_by() {
        // Test string_agg with GROUP BY
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "name".to_string()],
            vec![
                vec![Value::from("A"), Value::from("Alice")],
                vec![Value::from("A"), Value::from("Bob")],
                vec![Value::from("B"), Value::from("Charlie")],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::string_agg(
            LogicalExpr::column("name"),
            LogicalExpr::Literal(crate::ast::Literal::String(" | ".to_string())),
            false,
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 2);

        for row in &rows {
            let dept = row.get(0).unwrap();
            let names = row.get(1).unwrap();

            if dept == &Value::from("A") {
                if let Value::String(s) = names {
                    // Should contain both Alice and Bob
                    assert!(s.contains("Alice"));
                    assert!(s.contains("Bob"));
                    assert!(s.contains(" | "));
                } else {
                    panic!("Expected String");
                }
            } else if dept == &Value::from("B") {
                assert_eq!(names, &Value::from("Charlie"));
            }
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_string_agg_with_nulls() {
        // Test that string_agg skips NULLs
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["name".to_string()],
            vec![vec![Value::from("Alice")], vec![Value::Null], vec![Value::from("Bob")]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::string_agg(
            LogicalExpr::column("name"),
            LogicalExpr::Literal(crate::ast::Literal::String(", ".to_string())),
            false,
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        if let Value::String(s) = result {
            let parts: Vec<&str> = s.split(", ").collect();
            assert_eq!(parts.len(), 2);
            assert!(parts.contains(&"Alice"));
            assert!(parts.contains(&"Bob"));
        } else {
            panic!("Expected String, got {:?}", result);
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_string_agg_with_integers() {
        // Test string_agg with integers (should convert to strings)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::string_agg(
            LogicalExpr::column("n"),
            LogicalExpr::Literal(crate::ast::Literal::String("-".to_string())),
            false,
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        if let Value::String(s) = result {
            let parts: Vec<&str> = s.split('-').collect();
            assert_eq!(parts.len(), 3);
            assert!(parts.contains(&"1"));
            assert!(parts.contains(&"2"));
            assert!(parts.contains(&"3"));
        } else {
            panic!("Expected String, got {:?}", result);
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_string_agg_empty() {
        // Test string_agg with all NULLs returns NULL
        let input: BoxedOperator =
            Box::new(ValuesOp::with_columns(vec!["name".to_string()], vec![vec![Value::Null]]));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::string_agg(
            LogicalExpr::column("name"),
            LogicalExpr::Literal(crate::ast::Literal::String(", ".to_string())),
            false,
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        let result = row.get(0).unwrap();

        assert_eq!(result, &Value::Null);

        op.close().unwrap();
    }

    // ========== Tests for variance functions ==========

    #[test]
    fn hash_aggregate_variance_samp() {
        // Test sample variance: values 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Sum of squared deviations = 32, n = 8, variance = 32/7 ≈ 4.571
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(2)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
                vec![Value::Int(5)],
                vec![Value::Int(7)],
                vec![Value::Int(9)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::variance_samp(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // Sample variance = 32/7 ≈ 4.571428
            assert!((v - 4.571_428_571).abs() < 0.0001, "Expected ~4.571, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_variance_pop() {
        // Test population variance: values 2, 4, 4, 4, 5, 5, 7, 9
        // Mean = 5, Sum of squared deviations = 32, n = 8, variance = 32/8 = 4.0
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(2)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
                vec![Value::Int(5)],
                vec![Value::Int(7)],
                vec![Value::Int(9)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::variance_pop(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // Population variance = 32/8 = 4.0
            assert!((v - 4.0).abs() < 0.0001, "Expected 4.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_variance_single_value() {
        // Test variance with single value - should return NULL for sample, 0 for population
        let input: BoxedOperator =
            Box::new(ValuesOp::with_columns(vec!["n".to_string()], vec![vec![Value::Int(5)]]));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::variance_samp(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // Sample variance with 1 value is NULL (division by n-1 = 0)
        assert_eq!(row.get(0).unwrap(), &Value::Null);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_variance_pop_single_value() {
        // Population variance with single value should be 0
        let input: BoxedOperator =
            Box::new(ValuesOp::with_columns(vec!["n".to_string()], vec![vec![Value::Int(5)]]));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::variance_pop(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            assert!((v - 0.0).abs() < 0.0001, "Expected 0.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    // ========== Tests for stddev functions ==========

    #[test]
    fn hash_aggregate_stddev_samp() {
        // Test sample standard deviation: values 2, 4, 4, 4, 5, 5, 7, 9
        // Sample variance ≈ 4.571, stddev ≈ 2.138
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(2)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
                vec![Value::Int(5)],
                vec![Value::Int(7)],
                vec![Value::Int(9)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::stddev_samp(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // sqrt(4.571428) ≈ 2.138
            assert!((v - 2.138_089_935).abs() < 0.0001, "Expected ~2.138, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_stddev_pop() {
        // Test population standard deviation: values 2, 4, 4, 4, 5, 5, 7, 9
        // Population variance = 4.0, stddev = 2.0
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(2)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
                vec![Value::Int(5)],
                vec![Value::Int(7)],
                vec![Value::Int(9)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::stddev_pop(LogicalExpr::column("n"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // sqrt(4.0) = 2.0
            assert!((v - 2.0).abs() < 0.0001, "Expected 2.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_stddev_with_nulls() {
        // Test that stddev skips NULLs
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(10)],
                vec![Value::Null],
                vec![Value::Int(20)],
                vec![Value::Int(30)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::stddev_pop(LogicalExpr::column("n"), false),
            LogicalExpr::avg(LogicalExpr::column("n"), false),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // Values 10, 20, 30 => mean = 20, variance = (100 + 0 + 100)/3 = 200/3
        // stddev_pop = sqrt(200/3) ≈ 8.165
        if let Value::Float(v) = row.get(0).unwrap() {
            assert!((v - 8.164_965_8).abs() < 0.001, "Expected ~8.165, got {}", v);
        } else {
            panic!("Expected Float for stddev");
        }
        // AVG should be 20
        assert_eq!(row.get(1).unwrap(), &Value::Float(20.0));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_stddev_with_group_by() {
        // Test stddev with GROUP BY
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "salary".to_string()],
            vec![
                vec![Value::from("A"), Value::Int(100)],
                vec![Value::from("A"), Value::Int(200)],
                vec![Value::from("A"), Value::Int(150)],
                vec![Value::from("B"), Value::Int(50)],
                vec![Value::from("B"), Value::Int(50)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::stddev_pop(LogicalExpr::column("salary"), false)];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 2);

        for row in &rows {
            let dept = row.get(0).unwrap();
            if let Value::Float(stddev) = row.get(1).unwrap() {
                if dept == &Value::from("A") {
                    // Values 100, 200, 150 => mean = 150
                    // variance = (2500 + 2500 + 0)/3 = 5000/3
                    // stddev = sqrt(5000/3) ≈ 40.82
                    assert!(
                        (stddev - 40.824_829).abs() < 0.01,
                        "Dept A: Expected ~40.82, got {}",
                        stddev
                    );
                } else if dept == &Value::from("B") {
                    // All same values => stddev = 0
                    assert!((stddev - 0.0).abs() < 0.01, "Dept B: Expected 0.0, got {}", stddev);
                }
            } else {
                panic!("Expected Float");
            }
        }

        op.close().unwrap();
    }

    // ========== Tests for percentile functions ==========

    #[test]
    fn hash_aggregate_percentile_cont_median() {
        // Test percentileCont at 0.5 (median) with odd count
        // Values: 1, 2, 3, 4, 5 => median = 3
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(2)],
                vec![Value::Int(3)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::percentile_cont(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            assert!((v - 3.0).abs() < 0.0001, "Expected 3.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_cont_interpolated() {
        // Test percentileCont at 0.5 with even count (interpolation needed)
        // Values: 1, 2, 3, 4 => median = (2 + 3) / 2 = 2.5
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(2)],
                vec![Value::Int(3)],
                vec![Value::Int(4)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::percentile_cont(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            assert!((v - 2.5).abs() < 0.0001, "Expected 2.5, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_cont_quartiles() {
        // Test percentileCont at various percentiles
        // Values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            (1..=10).map(|i| vec![Value::Int(i)]).collect(),
        ));

        let group_by = vec![];
        // Test 25th percentile
        let aggregates = vec![LogicalExpr::percentile_cont(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.25)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // 25th percentile at position 0.25 * 9 = 2.25
            // Interpolate between index 2 (value 3) and index 3 (value 4)
            // Result = 3 + 0.25 * (4 - 3) = 3.25
            assert!((v - 3.25).abs() < 0.0001, "Expected 3.25, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_disc_median() {
        // Test percentileDisc at 0.5 (median) - returns exact value
        // Values: 1, 2, 3, 4 => should return 2 (first value >= 50% position)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(2)],
                vec![Value::Int(3)],
                vec![Value::Int(4)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::percentile_disc(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // Should be 2 (discrete value at position ceil(0.5 * 4) - 1 = 1)
            assert!((v - 2.0).abs() < 0.0001, "Expected 2.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_disc_extremes() {
        // Test percentileDisc at 0 and 1
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(10)],
                vec![Value::Int(20)],
                vec![Value::Int(30)],
                vec![Value::Int(40)],
                vec![Value::Int(50)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::percentile_disc(
                LogicalExpr::Literal(crate::ast::Literal::Float(0.0)),
                LogicalExpr::column("n"),
            ),
            LogicalExpr::percentile_disc(
                LogicalExpr::Literal(crate::ast::Literal::Float(1.0)),
                LogicalExpr::column("n"),
            ),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // 0th percentile should be minimum (10)
        if let Value::Float(v) = row.get(0).unwrap() {
            assert!((v - 10.0).abs() < 0.0001, "0th percentile: Expected 10.0, got {}", v);
        }
        // 100th percentile should be maximum (50)
        if let Value::Float(v) = row.get(1).unwrap() {
            assert!((v - 50.0).abs() < 0.0001, "100th percentile: Expected 50.0, got {}", v);
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_with_nulls() {
        // Test that percentile functions skip NULLs
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)],
                vec![Value::Null],
                vec![Value::Int(3)],
                vec![Value::Null],
                vec![Value::Int(5)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::percentile_cont(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        if let Value::Float(v) = row.get(0).unwrap() {
            // Values 1, 3, 5 => median = 3
            assert!((v - 3.0).abs() < 0.0001, "Expected 3.0, got {}", v);
        } else {
            panic!("Expected Float");
        }

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_percentile_empty_returns_null() {
        // Test that percentile returns NULL for all-NULL input
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Null], vec![Value::Null]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::percentile_cont(
            LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
            LogicalExpr::column("n"),
        )];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Null);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_multiple_statistical_aggregates() {
        // Test multiple statistical aggregates together
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(10)],
                vec![Value::Int(20)],
                vec![Value::Int(30)],
                vec![Value::Int(40)],
            ],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::avg(LogicalExpr::column("n"), false),
            LogicalExpr::variance_pop(LogicalExpr::column("n"), false),
            LogicalExpr::stddev_pop(LogicalExpr::column("n"), false),
            LogicalExpr::percentile_cont(
                LogicalExpr::Literal(crate::ast::Literal::Float(0.5)),
                LogicalExpr::column("n"),
            ),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();

        // AVG = 25
        assert_eq!(row.get(0).unwrap(), &Value::Float(25.0));

        // Variance (pop) = ((10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2) / 4
        //                = (225 + 25 + 25 + 225) / 4 = 500/4 = 125
        if let Value::Float(v) = row.get(1).unwrap() {
            assert!((v - 125.0).abs() < 0.0001, "Variance: Expected 125.0, got {}", v);
        }

        // Stddev (pop) = sqrt(125) ≈ 11.18
        if let Value::Float(v) = row.get(2).unwrap() {
            assert!((v - 11.180_339_8).abs() < 0.001, "Stddev: Expected ~11.18, got {}", v);
        }

        // Median = (20 + 30) / 2 = 25
        if let Value::Float(v) = row.get(3).unwrap() {
            assert!((v - 25.0).abs() < 0.0001, "Median: Expected 25.0, got {}", v);
        }

        op.close().unwrap();
    }

    // ========== Tests for bool_and ==========

    #[test]
    fn hash_aggregate_bool_and_all_true() {
        // Test bool_and when all values are true
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_active".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Bool(true)], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("is_active"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_one_false() {
        // Test bool_and when one value is false
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_active".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Bool(false)], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("is_active"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_with_nulls() {
        // Test bool_and ignores NULLs - only considers non-NULL values
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_active".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Null], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("is_active"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // NULL is skipped, so result is true AND true = true
        assert_eq!(row.get(0).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_all_nulls() {
        // Test bool_and with all NULLs returns NULL
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_active".to_string()],
            vec![vec![Value::Null], vec![Value::Null]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("is_active"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Null);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_with_group_by() {
        // Test bool_and with GROUP BY
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "is_active".to_string()],
            vec![
                vec![Value::from("A"), Value::Bool(true)],
                vec![Value::from("A"), Value::Bool(true)],
                vec![Value::from("B"), Value::Bool(true)],
                vec![Value::from("B"), Value::Bool(false)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("is_active"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 2);

        for row in &rows {
            let dept = row.get(0).unwrap();
            let result = row.get(1).unwrap();

            if dept == &Value::from("A") {
                assert_eq!(result, &Value::Bool(true));
            } else if dept == &Value::from("B") {
                assert_eq!(result, &Value::Bool(false));
            }
        }

        op.close().unwrap();
    }

    // ========== Tests for bool_or ==========

    #[test]
    fn hash_aggregate_bool_or_all_false() {
        // Test bool_or when all values are false
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["has_error".to_string()],
            vec![vec![Value::Bool(false)], vec![Value::Bool(false)], vec![Value::Bool(false)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("has_error"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_or_one_true() {
        // Test bool_or when one value is true
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["has_error".to_string()],
            vec![vec![Value::Bool(false)], vec![Value::Bool(true)], vec![Value::Bool(false)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("has_error"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_or_with_nulls() {
        // Test bool_or ignores NULLs - only considers non-NULL values
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["has_error".to_string()],
            vec![vec![Value::Bool(false)], vec![Value::Null], vec![Value::Bool(false)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("has_error"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // NULL is skipped, so result is false OR false = false
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_or_all_nulls() {
        // Test bool_or with all NULLs returns NULL
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["has_error".to_string()],
            vec![vec![Value::Null], vec![Value::Null]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("has_error"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Null);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_or_with_group_by() {
        // Test bool_or with GROUP BY
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "has_error".to_string()],
            vec![
                vec![Value::from("A"), Value::Bool(false)],
                vec![Value::from("A"), Value::Bool(false)],
                vec![Value::from("B"), Value::Bool(false)],
                vec![Value::from("B"), Value::Bool(true)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("has_error"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut rows = Vec::new();
        while let Some(row) = op.next().unwrap() {
            rows.push(row);
        }

        assert_eq!(rows.len(), 2);

        for row in &rows {
            let dept = row.get(0).unwrap();
            let result = row.get(1).unwrap();

            if dept == &Value::from("A") {
                assert_eq!(result, &Value::Bool(false));
            } else if dept == &Value::from("B") {
                assert_eq!(result, &Value::Bool(true));
            }
        }

        op.close().unwrap();
    }

    // ========== Tests for every (synonym for bool_and) ==========

    #[test]
    fn hash_aggregate_every_all_true() {
        // Test every (synonym for bool_and) when all values are true
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_verified".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Bool(true)], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::every(LogicalExpr::column("is_verified"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_every_one_false() {
        // Test every (synonym for bool_and) when one value is false
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["is_verified".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Bool(false)], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::every(LogicalExpr::column("is_verified"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    // ========== Test combining boolean aggregates ==========

    #[test]
    fn hash_aggregate_bool_and_and_bool_or_combined() {
        // Test both bool_and and bool_or together
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["flag".to_string()],
            vec![vec![Value::Bool(true)], vec![Value::Bool(false)], vec![Value::Bool(true)]],
        ));

        let group_by = vec![];
        let aggregates = vec![
            LogicalExpr::bool_and(LogicalExpr::column("flag")),
            LogicalExpr::bool_or(LogicalExpr::column("flag")),
        ];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // bool_and: true AND false AND true = false
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));
        // bool_or: true OR false OR true = true
        assert_eq!(row.get(1).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_with_integers() {
        // Test bool_and with integer values (non-zero is truthy)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["val".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("val"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // All non-zero integers are truthy, so result is true
        assert_eq!(row.get(0).unwrap(), &Value::Bool(true));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_and_with_zero() {
        // Test bool_and with zero (falsy)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["val".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(0)], vec![Value::Int(2)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_and(LogicalExpr::column("val"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // 0 is falsy, so result is false
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_bool_or_with_all_zeros() {
        // Test bool_or with all zeros (falsy)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["val".to_string()],
            vec![vec![Value::Int(0)], vec![Value::Int(0)], vec![Value::Int(0)]],
        ));

        let group_by = vec![];
        let aggregates = vec![LogicalExpr::bool_or(LogicalExpr::column("val"))];

        let mut op = HashAggregateOp::new(group_by, aggregates, None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // All zeros are falsy, so result is false
        assert_eq!(row.get(0).unwrap(), &Value::Bool(false));

        op.close().unwrap();
    }

    // ========== Complex HAVING Expression Tests ==========

    #[test]
    fn hash_aggregate_complex_having_and() {
        // Test: HAVING COUNT(*) > 1 AND AVG(salary) > 100
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "salary".to_string()],
            vec![
                // Dept A: count=3, avg=125 -> passes (3 > 1 AND 125 > 100)
                vec![Value::from("A"), Value::Int(100)],
                vec![Value::from("A"), Value::Int(125)],
                vec![Value::from("A"), Value::Int(150)],
                // Dept B: count=2, avg=75 -> fails (75 < 100)
                vec![Value::from("B"), Value::Int(50)],
                vec![Value::from("B"), Value::Int(100)],
                // Dept C: count=1, avg=200 -> fails (1 <= 1)
                vec![Value::from("C"), Value::Int(200)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![
            LogicalExpr::count(LogicalExpr::wildcard(), false),
            LogicalExpr::avg(LogicalExpr::column("salary"), false),
        ];

        // HAVING COUNT(*) > 1 AND AVG(salary) > 100
        let having =
            LogicalExpr::count(LogicalExpr::wildcard(), false).gt(LogicalExpr::integer(1)).and(
                LogicalExpr::avg(LogicalExpr::column("salary"), false)
                    .gt(LogicalExpr::integer(100)),
            );

        let mut op = HashAggregateOp::new(group_by, aggregates, Some(having), input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Only dept A should pass
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get(0), Some(&Value::from("A")));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_complex_having_or() {
        // Test: HAVING COUNT(*) >= 3 OR SUM(amount) > 1000
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["category".to_string(), "amount".to_string()],
            vec![
                // Category X: count=3, sum=300 -> passes (count >= 3)
                vec![Value::from("X"), Value::Int(100)],
                vec![Value::from("X"), Value::Int(100)],
                vec![Value::from("X"), Value::Int(100)],
                // Category Y: count=2, sum=1500 -> passes (sum > 1000)
                vec![Value::from("Y"), Value::Int(500)],
                vec![Value::from("Y"), Value::Int(1000)],
                // Category Z: count=1, sum=200 -> fails (neither condition)
                vec![Value::from("Z"), Value::Int(200)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("category")];
        let aggregates = vec![
            LogicalExpr::count(LogicalExpr::wildcard(), false),
            LogicalExpr::sum(LogicalExpr::column("amount"), false),
        ];

        // HAVING COUNT(*) >= 3 OR SUM(amount) > 1000
        let having = LogicalExpr::count(LogicalExpr::wildcard(), false)
            .gt_eq(LogicalExpr::integer(3))
            .or(LogicalExpr::sum(LogicalExpr::column("amount"), false)
                .gt(LogicalExpr::integer(1000)));

        let mut op = HashAggregateOp::new(group_by, aggregates, Some(having), input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Categories X and Y should pass
        assert_eq!(results.len(), 2);

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_having_with_group_column() {
        // Test: HAVING department = 'Engineering' AND COUNT(*) > 2
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["dept".to_string(), "name".to_string()],
            vec![
                vec![Value::from("Engineering"), Value::from("Alice")],
                vec![Value::from("Engineering"), Value::from("Bob")],
                vec![Value::from("Engineering"), Value::from("Carol")],
                vec![Value::from("Sales"), Value::from("Dave")],
                vec![Value::from("Sales"), Value::from("Eve")],
                vec![Value::from("Sales"), Value::from("Frank")],
            ],
        ));

        let group_by = vec![LogicalExpr::column("dept")];
        let aggregates = vec![LogicalExpr::count(LogicalExpr::wildcard(), false)];

        // HAVING dept = 'Engineering' AND COUNT(*) > 2
        let having = LogicalExpr::column("dept")
            .eq(LogicalExpr::Literal(crate::ast::Literal::String("Engineering".to_string())))
            .and(LogicalExpr::count(LogicalExpr::wildcard(), false).gt(LogicalExpr::integer(2)));

        let mut op = HashAggregateOp::new(group_by, aggregates, Some(having), input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Only Engineering should pass (Sales count=3 but dept != Engineering)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get(0), Some(&Value::from("Engineering")));

        op.close().unwrap();
    }

    #[test]
    fn hash_aggregate_having_arithmetic() {
        // Test: HAVING SUM(amount) / COUNT(*) > 100 (average via division)
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["region".to_string(), "amount".to_string()],
            vec![
                // Region North: sum=600, count=3, avg=200 -> passes
                vec![Value::from("North"), Value::Int(100)],
                vec![Value::from("North"), Value::Int(200)],
                vec![Value::from("North"), Value::Int(300)],
                // Region South: sum=150, count=3, avg=50 -> fails
                vec![Value::from("South"), Value::Int(50)],
                vec![Value::from("South"), Value::Int(50)],
                vec![Value::from("South"), Value::Int(50)],
            ],
        ));

        let group_by = vec![LogicalExpr::column("region")];
        let aggregates = vec![
            LogicalExpr::sum(LogicalExpr::column("amount"), false),
            LogicalExpr::count(LogicalExpr::wildcard(), false),
        ];

        // HAVING SUM(amount) / COUNT(*) > 100
        let having = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::sum(LogicalExpr::column("amount"), false)),
            op: BinaryOp::Div,
            right: Box::new(LogicalExpr::count(LogicalExpr::wildcard(), false)),
        }
        .gt(LogicalExpr::integer(100));

        let mut op = HashAggregateOp::new(group_by, aggregates, Some(having), input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Only North should pass (avg 200 > 100)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get(0), Some(&Value::from("North")));

        op.close().unwrap();
    }
}
