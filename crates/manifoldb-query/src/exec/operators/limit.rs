//! Limit operator for LIMIT and OFFSET.

use std::sync::Arc;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Limit operator.
///
/// Skips the first `offset` rows and returns at most `limit` rows.
pub struct LimitOp {
    /// Base operator state.
    base: OperatorBase,
    /// Maximum number of rows to return.
    limit: Option<usize>,
    /// Number of rows to skip.
    offset: usize,
    /// Input operator.
    input: BoxedOperator,
    /// Number of rows skipped so far.
    skipped: usize,
    /// Number of rows returned so far.
    returned: usize,
}

impl LimitOp {
    /// Creates a new limit operator.
    #[must_use]
    pub fn new(limit: Option<usize>, offset: Option<usize>, input: BoxedOperator) -> Self {
        let schema = input.schema();
        Self {
            base: OperatorBase::new(schema),
            limit,
            offset: offset.unwrap_or(0),
            input,
            skipped: 0,
            returned: 0,
        }
    }

    /// Creates a limit-only operator.
    #[must_use]
    pub fn limit(limit: usize, input: BoxedOperator) -> Self {
        Self::new(Some(limit), None, input)
    }

    /// Creates an offset-only operator.
    #[must_use]
    pub fn offset(offset: usize, input: BoxedOperator) -> Self {
        Self::new(None, Some(offset), input)
    }
}

impl Operator for LimitOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.skipped = 0;
        self.returned = 0;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Check if we've already returned the limit
        if let Some(limit) = self.limit {
            if self.returned >= limit {
                self.base.set_finished();
                return Ok(None);
            }
        }

        loop {
            match self.input.next()? {
                Some(row) => {
                    // Skip offset rows
                    if self.skipped < self.offset {
                        self.skipped += 1;
                        continue;
                    }

                    // Return this row
                    self.returned += 1;
                    self.base.inc_rows_produced();
                    return Ok(Some(row));
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
        "Limit"
    }
}

/// Limit WITH TIES operator.
///
/// Similar to `LimitOp`, but includes additional rows that have the same
/// ORDER BY values as the last row in the limit.
///
/// Note: This operator assumes the input is sorted. It works by:
/// 1. Returning rows up to the limit
/// 2. Storing the last returned row's values
/// 3. Continuing to return rows that match the last row's ordering
///
/// ## Known Limitation
///
/// The current implementation compares **all columns** of rows to determine
/// ties, rather than just the ORDER BY columns. This works correctly when
/// the ORDER BY columns are the only columns in the result set, or when all
/// rows with the same ORDER BY values also have the same values in other
/// columns. For full SQL compliance, the operator would need to receive
/// the ORDER BY column indices from the physical plan.
pub struct LimitWithTiesOp {
    /// Base operator state.
    base: OperatorBase,
    /// Maximum number of rows to return (before ties).
    limit: Option<usize>,
    /// Number of rows to skip.
    offset: usize,
    /// Input operator.
    input: BoxedOperator,
    /// Number of rows skipped so far.
    skipped: usize,
    /// Number of rows returned so far (before ties).
    returned: usize,
    /// The last row that was within the limit (used for tie comparison).
    last_limit_row: Option<Row>,
}

impl LimitWithTiesOp {
    /// Creates a new limit with ties operator.
    #[must_use]
    pub fn new(limit: Option<usize>, offset: Option<usize>, input: BoxedOperator) -> Self {
        let schema = input.schema();
        Self {
            base: OperatorBase::new(schema),
            limit,
            offset: offset.unwrap_or(0),
            input,
            skipped: 0,
            returned: 0,
            last_limit_row: None,
        }
    }

    /// Checks if two rows are equal for tie comparison.
    /// Since we assume the input is sorted by ORDER BY, we compare all columns.
    fn rows_are_tied(&self, a: &Row, b: &Row) -> bool {
        // Compare all values in the rows
        let len = a.len().min(b.len());
        for i in 0..len {
            if a.get(i) != b.get(i) {
                return false;
            }
        }
        true
    }
}

impl Operator for LimitWithTiesOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.skipped = 0;
        self.returned = 0;
        self.last_limit_row = None;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            match self.input.next()? {
                Some(row) => {
                    // Skip offset rows
                    if self.skipped < self.offset {
                        self.skipped += 1;
                        continue;
                    }

                    // Check if we've hit the limit
                    if let Some(limit) = self.limit {
                        if self.returned < limit {
                            // Still within limit, return the row
                            self.returned += 1;
                            self.last_limit_row = Some(row.clone());
                            self.base.inc_rows_produced();
                            return Ok(Some(row));
                        }
                        // We've hit the limit, check for ties
                        // Check if this row ties with the last row
                        if let Some(ref last_row) = self.last_limit_row {
                            if self.rows_are_tied(last_row, &row) {
                                self.base.inc_rows_produced();
                                return Ok(Some(row));
                            }
                        }

                        // No tie, we're done
                        self.base.set_finished();
                        return Ok(None);
                    }
                    // No limit specified, return all rows
                    self.base.inc_rows_produced();
                    return Ok(Some(row));
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
        self.last_limit_row = None;
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
        "LimitWithTies"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use manifoldb_core::Value;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![
                vec![Value::Int(1)],
                vec![Value::Int(2)],
                vec![Value::Int(3)],
                vec![Value::Int(4)],
                vec![Value::Int(5)],
            ],
        ))
    }

    #[test]
    fn limit_only() {
        let mut op = LimitOp::limit(3, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(1)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(2)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(3)));
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn offset_only() {
        let mut op = LimitOp::offset(2, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(3)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(4)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(5)));
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn limit_and_offset() {
        let mut op = LimitOp::new(Some(2), Some(1), make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(2)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(3)));
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn limit_exceeds_rows() {
        let mut op = LimitOp::limit(10, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);

        op.close().unwrap();
    }

    // ========================================================================
    // LimitWithTiesOp tests
    // ========================================================================

    #[allow(dead_code)]
    fn make_input_with_ties() -> BoxedOperator {
        // Input data that simulates sorted data with ties:
        // (1, "a"), (1, "b"), (2, "c"), (2, "d"), (3, "e")
        // If we limit to 2, we should get rows with value 1 only
        // But if values 2 tie, we get more
        Box::new(ValuesOp::with_columns(
            vec!["score".to_string(), "name".to_string()],
            vec![
                vec![Value::Int(10), Value::String("alice".into())],
                vec![Value::Int(10), Value::String("bob".into())],
                vec![Value::Int(20), Value::String("carol".into())],
                vec![Value::Int(20), Value::String("dave".into())],
                vec![Value::Int(30), Value::String("eve".into())],
            ],
        ))
    }

    #[test]
    fn limit_with_ties_no_ties() {
        // Limit 1 with no ties - should return just the first row
        let input = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)], vec![Value::Int(3)]],
        ));
        let mut op = LimitWithTiesOp::new(Some(1), None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(1)));
        // Next row (2) doesn't tie with last row (1), so stop
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn limit_with_ties_returns_tied_rows() {
        // Limit 2, but rows 0 and 1 have same values, and row 2 also ties
        let input = Box::new(ValuesOp::with_columns(
            vec!["score".to_string()],
            vec![
                vec![Value::Int(10)],
                vec![Value::Int(10)],
                vec![Value::Int(10)], // This ties with the limit boundary
                vec![Value::Int(20)], // This doesn't tie
            ],
        ));
        let mut op = LimitWithTiesOp::new(Some(2), None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should get 3 rows because the 3rd ties with the 2nd
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        // Row 4 (value=20) doesn't tie
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn limit_with_ties_and_offset() {
        // Skip 1, limit 2, with ties
        let input = Box::new(ValuesOp::with_columns(
            vec!["score".to_string()],
            vec![
                vec![Value::Int(5)],  // skipped
                vec![Value::Int(10)], // returned (1)
                vec![Value::Int(10)], // returned (2) - this is the boundary
                vec![Value::Int(10)], // returned (tie)
                vec![Value::Int(20)], // not returned (no tie)
            ],
        ));
        let mut op = LimitWithTiesOp::new(Some(2), Some(1), input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(10)));
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }

    #[test]
    fn limit_with_ties_no_limit() {
        // No limit set - should return all rows
        let mut op = LimitWithTiesOp::new(None, None, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 5);

        op.close().unwrap();
    }

    #[test]
    fn limit_with_ties_all_same_value() {
        // All rows have the same value - all should be returned
        let input = Box::new(ValuesOp::with_columns(
            vec!["score".to_string()],
            vec![vec![Value::Int(42)], vec![Value::Int(42)], vec![Value::Int(42)]],
        ));
        let mut op = LimitWithTiesOp::new(Some(1), None, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // All rows tie with the first one
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(42)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(42)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(42)));
        assert!(op.next().unwrap().is_none());

        op.close().unwrap();
    }
}
