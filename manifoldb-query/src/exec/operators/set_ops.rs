//! Set operation operators for UNION, INTERSECT, and EXCEPT.
//!
//! These operators combine results from multiple input operators using
//! set semantics (with or without duplicate elimination).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::SetOpType;

/// Union operator - concatenates results from multiple inputs.
///
/// For UNION (without ALL), duplicates are eliminated.
/// For UNION ALL, all rows are returned including duplicates.
pub struct UnionOp {
    /// Base operator state.
    base: OperatorBase,
    /// Input operators.
    inputs: Vec<BoxedOperator>,
    /// Whether to preserve duplicates (UNION ALL).
    all: bool,
    /// Current input index.
    current_input: usize,
    /// Seen rows for deduplication (only used when all=false).
    seen: HashSet<Vec<u8>>,
}

impl UnionOp {
    /// Creates a new union operator.
    #[must_use]
    pub fn new(inputs: Vec<BoxedOperator>, all: bool) -> Self {
        // Use schema from first input (all inputs should have compatible schemas)
        let schema = if inputs.is_empty() { Arc::new(Schema::empty()) } else { inputs[0].schema() };

        Self {
            base: OperatorBase::new(schema),
            inputs,
            all,
            current_input: 0,
            seen: HashSet::new(),
        }
    }

    /// Encodes a row to bytes for deduplication.
    fn encode_row(row: &Row) -> Vec<u8> {
        let mut buf = Vec::new();
        for value in row.values() {
            encode_value(value, &mut buf);
        }
        buf
    }
}

impl Operator for UnionOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        for input in &mut self.inputs {
            input.open(ctx)?;
        }
        self.current_input = 0;
        self.seen.clear();
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // If we've exhausted all inputs, we're done
            if self.current_input >= self.inputs.len() {
                self.base.set_finished();
                return Ok(None);
            }

            // Try to get next row from current input
            match self.inputs[self.current_input].next()? {
                Some(row) => {
                    if self.all {
                        // UNION ALL - return all rows
                        self.base.inc_rows_produced();
                        return Ok(Some(row));
                    }
                    // UNION - deduplicate
                    let key = Self::encode_row(&row);
                    if self.seen.insert(key) {
                        self.base.inc_rows_produced();
                        return Ok(Some(row));
                    }
                    // Duplicate - continue to next row
                }
                None => {
                    // Move to next input
                    self.current_input += 1;
                }
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        for input in &mut self.inputs {
            input.close()?;
        }
        self.seen.clear();
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
        "Union"
    }
}

/// Set operation operator for binary set operations (INTERSECT, EXCEPT).
///
/// Handles:
/// - INTERSECT: rows present in both inputs
/// - INTERSECT ALL: preserves duplicates based on minimum count
/// - EXCEPT: rows in left but not in right
/// - EXCEPT ALL: removes one occurrence for each match in right
pub struct SetOpOp {
    /// Base operator state.
    base: OperatorBase,
    /// The type of set operation.
    op_type: SetOpType,
    /// Left input operator.
    left: BoxedOperator,
    /// Right input operator.
    right: BoxedOperator,
    /// Materialized results.
    results: Vec<Row>,
    /// Current position in results.
    position: usize,
    /// Whether results have been computed.
    computed: bool,
}

impl SetOpOp {
    /// Creates a new set operation operator.
    #[must_use]
    pub fn new(op_type: SetOpType, left: BoxedOperator, right: BoxedOperator) -> Self {
        let schema = left.schema();
        Self {
            base: OperatorBase::new(schema),
            op_type,
            left,
            right,
            results: Vec::new(),
            position: 0,
            computed: false,
        }
    }

    /// Computes the set operation results.
    fn compute_results(&mut self) -> OperatorResult<()> {
        let schema = self.base.schema();

        match self.op_type {
            SetOpType::Union => {
                // For Union in SetOp, treat as UNION (with deduplication)
                self.compute_union(false)?;
            }
            SetOpType::UnionAll => {
                // For UnionAll in SetOp, treat as UNION ALL
                self.compute_union(true)?;
            }
            SetOpType::Intersect => {
                self.compute_intersect(false, &schema)?;
            }
            SetOpType::IntersectAll => {
                self.compute_intersect(true, &schema)?;
            }
            SetOpType::Except => {
                self.compute_except(false, &schema)?;
            }
            SetOpType::ExceptAll => {
                self.compute_except(true, &schema)?;
            }
        }

        self.computed = true;
        Ok(())
    }

    /// Computes UNION or UNION ALL.
    fn compute_union(&mut self, all: bool) -> OperatorResult<()> {
        let mut seen: HashSet<Vec<u8>> = HashSet::new();

        // Collect left rows
        while let Some(row) = self.left.next()? {
            if all {
                self.results.push(row);
            } else {
                let key = encode_row(&row);
                if seen.insert(key) {
                    self.results.push(row);
                }
            }
        }

        // Collect right rows
        while let Some(row) = self.right.next()? {
            if all {
                self.results.push(row);
            } else {
                let key = encode_row(&row);
                if seen.insert(key) {
                    self.results.push(row);
                }
            }
        }

        Ok(())
    }

    /// Computes INTERSECT or INTERSECT ALL.
    fn compute_intersect(&mut self, all: bool, _schema: &Arc<Schema>) -> OperatorResult<()> {
        // Build hash map of right side with counts
        // Pre-allocate for typical query sizes
        const INITIAL_CAPACITY: usize = 1000;
        let mut right_counts: HashMap<Vec<u8>, (usize, Vec<Value>)> =
            HashMap::with_capacity(INITIAL_CAPACITY);
        while let Some(row) = self.right.next()? {
            let key = encode_row(&row);
            let entry = right_counts.entry(key).or_insert_with(|| (0, row.values().to_vec()));
            entry.0 += 1;
        }

        if all {
            // INTERSECT ALL - for each left row, if it exists in right, output it
            // and decrement the right count
            while let Some(row) = self.left.next()? {
                let key = encode_row(&row);
                if let Some((count, _)) = right_counts.get_mut(&key) {
                    if *count > 0 {
                        *count -= 1;
                        self.results.push(row);
                    }
                }
            }
        } else {
            // INTERSECT - output each distinct row that appears in both
            let mut seen_left: HashSet<Vec<u8>> = HashSet::new();
            while let Some(row) = self.left.next()? {
                let key = encode_row(&row);
                if right_counts.contains_key(&key) && seen_left.insert(key) {
                    self.results.push(row);
                }
            }
        }

        Ok(())
    }

    /// Computes EXCEPT or EXCEPT ALL.
    fn compute_except(&mut self, all: bool, _schema: &Arc<Schema>) -> OperatorResult<()> {
        // Build hash map of right side with counts
        // Pre-allocate for typical query sizes
        const INITIAL_CAPACITY: usize = 1000;
        let mut right_counts: HashMap<Vec<u8>, usize> = HashMap::with_capacity(INITIAL_CAPACITY);
        while let Some(row) = self.right.next()? {
            let key = encode_row(&row);
            *right_counts.entry(key).or_insert(0) += 1;
        }

        if all {
            // EXCEPT ALL - for each left row, if it exists in right, decrement count
            // only output if count would go negative (more in left than right)
            while let Some(row) = self.left.next()? {
                let key = encode_row(&row);
                match right_counts.get_mut(&key) {
                    Some(count) if *count > 0 => {
                        *count -= 1;
                        // Don't output - this row is "removed" by the right side
                    }
                    _ => {
                        self.results.push(row);
                    }
                }
            }
        } else {
            // EXCEPT - output distinct rows from left that don't appear in right
            let mut seen_left: HashSet<Vec<u8>> = HashSet::new();
            while let Some(row) = self.left.next()? {
                let key = encode_row(&row);
                if !right_counts.contains_key(&key) && seen_left.insert(key) {
                    self.results.push(row);
                }
            }
        }

        Ok(())
    }
}

impl Operator for SetOpOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.left.open(ctx)?;
        self.right.open(ctx)?;
        self.results.clear();
        self.position = 0;
        self.computed = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Compute results on first call
        if !self.computed {
            self.compute_results()?;
        }

        if self.position >= self.results.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let row = self.results[self.position].clone();
        self.position += 1;
        self.base.inc_rows_produced();
        Ok(Some(row))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.left.close()?;
        self.right.close()?;
        self.results.clear();
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
        match self.op_type {
            SetOpType::Union | SetOpType::UnionAll => "Union",
            SetOpType::Intersect | SetOpType::IntersectAll => "Intersect",
            SetOpType::Except | SetOpType::ExceptAll => "Except",
        }
    }
}

/// Encodes a row to bytes for hashing.
fn encode_row(row: &Row) -> Vec<u8> {
    let mut buf = Vec::new();
    for value in row.values() {
        encode_value(value, &mut buf);
    }
    buf
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
            buf.push(0); // null terminator
        }
        _ => buf.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_values_op(columns: Vec<&str>, rows: Vec<Vec<Value>>) -> BoxedOperator {
        let schema = Arc::new(Schema::new(columns.into_iter().map(String::from).collect()));
        Box::new(ValuesOp::new(schema, rows))
    }

    #[test]
    fn union_all_combines_all_rows() {
        let left = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(1)]],
        );
        let right = make_values_op(vec!["x"], vec![vec![Value::Int(2)], vec![Value::Int(3)]]);

        let mut op = UnionOp::new(vec![left, right], true);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have all 5 rows including duplicates
        assert_eq!(results.len(), 5);
        assert_eq!(
            results,
            vec![Value::Int(1), Value::Int(2), Value::Int(1), Value::Int(2), Value::Int(3)]
        );
    }

    #[test]
    fn union_deduplicates() {
        let left = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(1)]],
        );
        let right = make_values_op(vec!["x"], vec![vec![Value::Int(2)], vec![Value::Int(3)]]);

        let mut op = UnionOp::new(vec![left, right], false);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have 3 distinct values
        assert_eq!(results.len(), 3);
        assert!(results.contains(&Value::Int(1)));
        assert!(results.contains(&Value::Int(2)));
        assert!(results.contains(&Value::Int(3)));
    }

    #[test]
    fn intersect_returns_common_rows() {
        let left = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        );
        let right = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(2)], vec![Value::Int(3)], vec![Value::Int(4)]],
        );

        let mut op = SetOpOp::new(SetOpType::Intersect, left, right);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have 2 and 3 (common to both)
        assert_eq!(results.len(), 2);
        assert!(results.contains(&Value::Int(2)));
        assert!(results.contains(&Value::Int(3)));
    }

    #[test]
    fn intersect_all_preserves_min_count() {
        let left = make_values_op(
            vec!["x"],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(1)],
                vec![Value::Int(1)],
                vec![Value::Int(2)],
            ],
        );
        let right = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(1)], vec![Value::Int(1)], vec![Value::Int(3)]],
        );

        let mut op = SetOpOp::new(SetOpType::IntersectAll, left, right);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have 1 twice (min of 3 and 2)
        assert_eq!(results.len(), 2);
        assert_eq!(results, vec![Value::Int(1), Value::Int(1)]);
    }

    #[test]
    fn except_returns_left_minus_right() {
        let left = make_values_op(
            vec!["x"],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        );
        let right = make_values_op(vec!["x"], vec![vec![Value::Int(2)], vec![Value::Int(4)]]);

        let mut op = SetOpOp::new(SetOpType::Except, left, right);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have 1 and 3 (in left but not in right)
        assert_eq!(results.len(), 2);
        assert!(results.contains(&Value::Int(1)));
        assert!(results.contains(&Value::Int(3)));
    }

    #[test]
    fn except_all_subtracts_counts() {
        let left = make_values_op(
            vec!["x"],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(1)],
                vec![Value::Int(1)],
                vec![Value::Int(2)],
            ],
        );
        let right = make_values_op(vec!["x"], vec![vec![Value::Int(1)], vec![Value::Int(2)]]);

        let mut op = SetOpOp::new(SetOpType::ExceptAll, left, right);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        // Should have 1 twice (3 - 1 = 2 occurrences), 2 is removed
        assert_eq!(results.len(), 2);
        assert_eq!(results, vec![Value::Int(1), Value::Int(1)]);
    }

    #[test]
    fn union_empty_inputs() {
        let inputs: Vec<BoxedOperator> = vec![];
        let mut op = UnionOp::new(inputs, true);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn set_op_with_strings() {
        let left = make_values_op(
            vec!["name"],
            vec![vec![Value::String("alice".into())], vec![Value::String("bob".into())]],
        );
        let right = make_values_op(
            vec!["name"],
            vec![vec![Value::String("bob".into())], vec![Value::String("charlie".into())]],
        );

        let mut op = SetOpOp::new(SetOpType::Intersect, left, right);
        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut results = Vec::new();
        while let Some(row) = op.next().unwrap() {
            results.push(row.values()[0].clone());
        }

        assert_eq!(results.len(), 1);
        assert_eq!(results[0], Value::String("bob".into()));
    }
}
