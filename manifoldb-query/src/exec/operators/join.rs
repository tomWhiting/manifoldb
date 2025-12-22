//! Join operators for combining data from multiple sources.

// Allow unwrap - state invariants guarantee these are Some when accessed
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{JoinType, LogicalExpr};

/// Nested loop join operator.
///
/// Simple O(n*m) join that evaluates a condition for each pair of rows.
pub struct NestedLoopJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Join condition.
    condition: Option<LogicalExpr>,
    /// Left (outer) input.
    left: BoxedOperator,
    /// Right (inner) input.
    right: BoxedOperator,
    /// Materialized right rows.
    right_rows: Vec<Row>,
    /// Current left row.
    current_left: Option<Row>,
    /// Current position in right rows.
    right_position: usize,
    /// Whether we've matched current left row.
    matched_left: bool,
    /// Whether right is materialized.
    right_materialized: bool,
}

impl NestedLoopJoinOp {
    /// Creates a new nested loop join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        condition: Option<LogicalExpr>,
        left: BoxedOperator,
        right: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(left.schema().merge(&right.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            condition,
            left,
            right,
            right_rows: Vec::new(),
            current_left: None,
            right_position: 0,
            matched_left: false,
            right_materialized: false,
        }
    }

    /// Evaluates the join condition.
    fn matches(&self, left: &Row, right: &Row) -> OperatorResult<bool> {
        match &self.condition {
            Some(cond) => {
                // Merge rows to evaluate condition
                let merged = left.merge(right);
                let result = evaluate_expr(cond, &merged)?;
                match result {
                    Value::Bool(b) => Ok(b),
                    Value::Null => Ok(false),
                    _ => Ok(false),
                }
            }
            None => Ok(true), // Cross join
        }
    }

    /// Creates a null row for the right side.
    fn right_null_row(&self) -> Row {
        Row::empty(self.right.schema())
    }

    /// Creates a null row for the left side.
    ///
    /// Used for RIGHT and FULL outer joins to output right rows
    /// that have no matching left rows.
    pub fn left_null_row(&self) -> Row {
        Row::empty(self.left.schema())
    }

    /// Returns the join type.
    #[must_use]
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }
}

impl Operator for NestedLoopJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.right_rows.clear();
        self.current_left = None;
        self.right_position = 0;
        self.matched_left = false;
        self.right_materialized = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Materialize right side on first call
        if !self.right_materialized {
            while let Some(row) = self.right.next()? {
                self.right_rows.push(row);
            }
            self.right_materialized = true;
        }

        loop {
            // Get next left row if needed
            if self.current_left.is_none() {
                match self.left.next()? {
                    Some(row) => {
                        self.current_left = Some(row);
                        self.right_position = 0;
                        self.matched_left = false;
                    }
                    None => {
                        self.base.set_finished();
                        return Ok(None);
                    }
                }
            }

            let left_row = self.current_left.as_ref().unwrap();

            // Try to find matching right row
            while self.right_position < self.right_rows.len() {
                let right_row = &self.right_rows[self.right_position];
                self.right_position += 1;

                if self.matches(left_row, right_row)? {
                    self.matched_left = true;
                    self.base.inc_rows_produced();
                    return Ok(Some(left_row.merge(right_row)));
                }
            }

            // No more right rows for this left row
            match self.join_type {
                JoinType::Left | JoinType::Full => {
                    if !self.matched_left {
                        // Output left row with NULL right
                        self.matched_left = true;
                        self.base.inc_rows_produced();
                        return Ok(Some(left_row.merge(&self.right_null_row())));
                    }
                }
                _ => {}
            }

            // Move to next left row
            self.current_left = None;
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.right_rows.clear();
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
        "NestedLoopJoin"
    }
}

/// Hash join operator.
///
/// Builds a hash table on the build side, probes with the probe side.
/// Efficient for equijoin conditions.
pub struct HashJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Build side key expressions.
    build_keys: Vec<LogicalExpr>,
    /// Probe side key expressions.
    probe_keys: Vec<LogicalExpr>,
    /// Additional filter.
    filter: Option<LogicalExpr>,
    /// Build (left) input.
    build: BoxedOperator,
    /// Probe (right) input.
    probe: BoxedOperator,
    /// Hash table: key -> shared list of matching rows (Arc avoids cloning on probe).
    hash_table: HashMap<Vec<u8>, Arc<Vec<Row>>>,
    /// Current probe row.
    current_probe: Option<Row>,
    /// Current matches for probe row (shared reference, no clone).
    current_matches: Option<Arc<Vec<Row>>>,
    /// Position in current matches.
    match_position: usize,
    /// Whether hash table is built.
    built: bool,
}

impl HashJoinOp {
    /// Creates a new hash join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        build_keys: Vec<LogicalExpr>,
        probe_keys: Vec<LogicalExpr>,
        filter: Option<LogicalExpr>,
        build: BoxedOperator,
        probe: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(build.schema().merge(&probe.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            build_keys,
            probe_keys,
            filter,
            build,
            probe,
            hash_table: HashMap::new(),
            current_probe: None,
            current_matches: None,
            match_position: 0,
            built: false,
        }
    }

    /// Computes hash key from expressions.
    fn compute_key(&self, row: &Row, exprs: &[LogicalExpr]) -> OperatorResult<Vec<u8>> {
        // Pre-allocate for typical key sizes (64 bytes handles most cases)
        let mut key = Vec::with_capacity(64);
        for expr in exprs {
            let value = evaluate_expr(expr, row)?;
            // Simple key encoding
            match &value {
                Value::Null => key.push(0),
                Value::Bool(b) => {
                    key.push(1);
                    key.push(u8::from(*b));
                }
                Value::Int(i) => {
                    key.push(2);
                    key.extend_from_slice(&i.to_le_bytes());
                }
                Value::Float(f) => {
                    key.push(3);
                    key.extend_from_slice(&f.to_le_bytes());
                }
                Value::String(s) => {
                    key.push(4);
                    key.extend_from_slice(s.as_bytes());
                    key.push(0); // Null terminator
                }
                _ => key.push(0),
            }
        }
        Ok(key)
    }

    /// Builds the hash table from the build side.
    fn build_hash_table(&mut self) -> OperatorResult<()> {
        // First, collect all rows into a temporary HashMap with Vec
        let mut temp_table: HashMap<Vec<u8>, Vec<Row>> = HashMap::new();
        while let Some(row) = self.build.next()? {
            let key = self.compute_key(&row, &self.build_keys)?;
            temp_table.entry(key).or_default().push(row);
        }
        // Convert to Arc<Vec<Row>> for sharing without cloning
        for (key, rows) in temp_table {
            self.hash_table.insert(key, Arc::new(rows));
        }
        self.built = true;
        Ok(())
    }

    /// Checks if filter passes.
    fn filter_passes(&self, left: &Row, right: &Row) -> OperatorResult<bool> {
        match &self.filter {
            Some(f) => {
                let merged = left.merge(right);
                let result = evaluate_expr(f, &merged)?;
                Ok(matches!(result, Value::Bool(true)))
            }
            None => Ok(true),
        }
    }

    /// Creates a null row for the build side.
    fn build_null_row(&self) -> Row {
        Row::empty(self.build.schema())
    }
}

impl Operator for HashJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.build.open(ctx)?;
        self.probe.open(ctx)?;
        self.hash_table.clear();
        self.current_probe = None;
        self.current_matches = None;
        self.match_position = 0;
        self.built = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Build hash table on first call
        if !self.built {
            self.build_hash_table()?;
        }

        loop {
            // Return next match if available
            if let Some(ref matches) = self.current_matches {
                while self.match_position < matches.len() {
                    let build_row = &matches[self.match_position];
                    self.match_position += 1;

                    if let Some(probe_row) = &self.current_probe {
                        if self.filter_passes(build_row, probe_row)? {
                            self.base.inc_rows_produced();
                            return Ok(Some(build_row.merge(probe_row)));
                        }
                    }
                }
            }

            // Get next probe row
            match self.probe.next()? {
                Some(probe_row) => {
                    let key = self.compute_key(&probe_row, &self.probe_keys)?;
                    // Clone only the Arc pointer, not the underlying Vec<Row>
                    self.current_matches = self.hash_table.get(&key).cloned();
                    self.current_probe = Some(probe_row);
                    self.match_position = 0;

                    // Handle left outer join with no matches
                    let has_no_matches =
                        self.current_matches.as_ref().map_or(true, |m| m.is_empty());
                    if has_no_matches && self.join_type == JoinType::Left {
                        let probe = self.current_probe.as_ref().unwrap();
                        self.base.inc_rows_produced();
                        return Ok(Some(self.build_null_row().merge(probe)));
                    }
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.build.close()?;
        self.probe.close()?;
        self.hash_table.clear();
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
        "HashJoin"
    }
}

/// Merge join operator.
///
/// Merges two sorted inputs. Requires both sides sorted on join keys.
pub struct MergeJoinOp {
    /// Base operator state.
    base: OperatorBase,
    /// Join type.
    join_type: JoinType,
    /// Left side key expressions.
    left_keys: Vec<LogicalExpr>,
    /// Right side key expressions.
    right_keys: Vec<LogicalExpr>,
    /// Left input.
    left: BoxedOperator,
    /// Right input.
    right: BoxedOperator,
    /// Current left row.
    current_left: Option<Row>,
    /// Current right row.
    current_right: Option<Row>,
    /// Buffer of right rows with same key.
    right_buffer: Vec<Row>,
    /// Position in right buffer.
    buffer_position: usize,
}

impl MergeJoinOp {
    /// Creates a new merge join operator.
    #[must_use]
    pub fn new(
        join_type: JoinType,
        left_keys: Vec<LogicalExpr>,
        right_keys: Vec<LogicalExpr>,
        left: BoxedOperator,
        right: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(left.schema().merge(&right.schema()));
        Self {
            base: OperatorBase::new(schema),
            join_type,
            left_keys,
            right_keys,
            left,
            right,
            current_left: None,
            current_right: None,
            right_buffer: Vec::new(),
            buffer_position: 0,
        }
    }

    /// Returns the join type.
    #[must_use]
    pub fn join_type(&self) -> JoinType {
        self.join_type
    }

    /// Compares keys from two rows.
    fn compare_keys(&self, left: &Row, right: &Row) -> OperatorResult<std::cmp::Ordering> {
        for (left_expr, right_expr) in self.left_keys.iter().zip(&self.right_keys) {
            let left_val = evaluate_expr(left_expr, left)?;
            let right_val = evaluate_expr(right_expr, right)?;

            let cmp = compare_values(&left_val, &right_val);
            if cmp != std::cmp::Ordering::Equal {
                return Ok(cmp);
            }
        }
        Ok(std::cmp::Ordering::Equal)
    }
}

impl Operator for MergeJoinOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.current_left = self.left.next()?;
        self.current_right = self.right.next()?;
        self.right_buffer.clear();
        self.buffer_position = 0;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // Return from buffer if available
            if self.buffer_position < self.right_buffer.len() {
                if let Some(left) = &self.current_left {
                    let right = &self.right_buffer[self.buffer_position];
                    self.buffer_position += 1;
                    self.base.inc_rows_produced();
                    return Ok(Some(left.merge(right)));
                }
            }

            // Need both sides to continue
            let (left, right) = match (&self.current_left, &self.current_right) {
                (Some(l), Some(r)) => (l, r),
                _ => {
                    self.base.set_finished();
                    return Ok(None);
                }
            };

            match self.compare_keys(left, right)? {
                std::cmp::Ordering::Less => {
                    // Advance left
                    self.current_left = self.left.next()?;
                    self.right_buffer.clear();
                    self.buffer_position = 0;
                }
                std::cmp::Ordering::Greater => {
                    // Advance right
                    self.current_right = self.right.next()?;
                }
                std::cmp::Ordering::Equal => {
                    // Buffer all right rows with same key
                    self.right_buffer.clear();
                    self.right_buffer.push(right.clone());

                    // Read more right rows with same key
                    loop {
                        self.current_right = self.right.next()?;
                        match &self.current_right {
                            Some(r) if self.compare_keys(left, r)? == std::cmp::Ordering::Equal => {
                                self.right_buffer.push(r.clone());
                            }
                            _ => break,
                        }
                    }

                    self.buffer_position = 0;
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.right_buffer.clear();
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
        "MergeJoin"
    }
}

/// Compares two values.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Less,
        (_, Value::Null) => Ordering::Greater,
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_left() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice")],
                vec![Value::Int(2), Value::from("Bob")],
                vec![Value::Int(3), Value::from("Carol")],
            ],
        ))
    }

    fn make_right() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["user_id".to_string(), "order".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Order1")],
                vec![Value::Int(1), Value::from("Order2")],
                vec![Value::Int(2), Value::from("Order3")],
            ],
        ))
    }

    #[test]
    fn nested_loop_inner_join() {
        let condition = LogicalExpr::column("id").eq(LogicalExpr::column("user_id"));
        let mut op =
            NestedLoopJoinOp::new(JoinType::Inner, Some(condition), make_left(), make_right());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Alice has 2 orders, Bob has 1 order, Carol has 0
        assert_eq!(results.len(), 3);

        op.close().unwrap();
    }

    #[test]
    fn nested_loop_cross_join() {
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["a".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ));
        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["b".to_string()],
            vec![vec![Value::Int(10)], vec![Value::Int(20)]],
        ));

        let mut op = NestedLoopJoinOp::new(JoinType::Cross, None, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }

        assert_eq!(count, 4); // 2 * 2
        op.close().unwrap();
    }

    #[test]
    fn hash_join_inner() {
        let build_keys = vec![LogicalExpr::column("id")];
        let probe_keys = vec![LogicalExpr::column("user_id")];

        let mut op = HashJoinOp::new(
            JoinType::Inner,
            build_keys,
            probe_keys,
            None,
            make_left(),
            make_right(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        assert_eq!(results.len(), 3);
        op.close().unwrap();
    }

    #[test]
    fn merge_join_inner() {
        // Sorted inputs
        let left: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));
        let right: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(4)]],
        ));

        let left_keys = vec![LogicalExpr::column("id")];
        let right_keys = vec![LogicalExpr::column("id")];

        let mut op = MergeJoinOp::new(JoinType::Inner, left_keys, right_keys, left, right);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row);
        }

        // Matches: 1, 2 (3 and 4 don't match)
        assert_eq!(results.len(), 2);
        op.close().unwrap();
    }
}
