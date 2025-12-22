//! Aggregate operators for GROUP BY and aggregation functions.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::Value;

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
                if let LogicalExpr::AggregateFunction { func, arg, distinct: _ } = agg_expr {
                    let is_wildcard = matches!(arg.as_ref(), LogicalExpr::Wildcard);
                    let arg_value = evaluate_expr(arg, &row)?;
                    state.accumulators[i].update(func, &arg_value, is_wildcard);
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

            // Apply HAVING filter
            if let Some(having) = &self.having {
                let result = evaluate_expr(having, &row)?;
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
            if let LogicalExpr::AggregateFunction { func, arg, .. } = agg_expr {
                let is_wildcard = matches!(arg.as_ref(), LogicalExpr::Wildcard);
                let arg_value = evaluate_expr(arg, row)?;
                self.accumulators[i].update(func, &arg_value, is_wildcard);
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

                        // Check HAVING
                        if let Some(having) = &self.having {
                            let check = evaluate_expr(having, &result)?;
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
                            let check = evaluate_expr(having, &result)?;
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
    count: i64,
    sum: f64,
    min: Option<Value>,
    max: Option<Value>,
}

impl Accumulator {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, func: &AggregateFunction, value: &Value, is_wildcard: bool) {
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
            _ => {}
        }
    }

    fn result(&self) -> Value {
        // Default result based on what was accumulated
        if self.min.is_some() {
            return self.min.clone().unwrap_or(Value::Null);
        }
        if self.max.is_some() {
            return self.max.clone().unwrap_or(Value::Null);
        }
        if self.count > 0 && self.sum != 0.0 {
            return Value::Float(self.sum);
        }
        Value::Int(self.count)
    }

    fn default_for(&self, func: &AggregateFunction) -> Value {
        match func {
            AggregateFunction::Count => Value::Int(0),
            AggregateFunction::Sum => Value::Null,
            AggregateFunction::Avg => Value::Null,
            AggregateFunction::Min | AggregateFunction::Max => Value::Null,
            _ => Value::Null,
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
}
