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
}
