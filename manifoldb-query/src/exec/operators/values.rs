//! Values and Empty operators.
//!
//! These operators produce inline data without reading from storage.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Values operator - produces rows from inline data.
pub struct ValuesOp {
    /// Base operator state.
    base: OperatorBase,
    /// The rows to produce.
    rows: Vec<Vec<Value>>,
    /// Current row index.
    current: usize,
}

impl ValuesOp {
    /// Creates a new values operator.
    #[must_use]
    pub fn new(schema: Arc<Schema>, rows: Vec<Vec<Value>>) -> Self {
        Self { base: OperatorBase::new(schema), rows, current: 0 }
    }

    /// Creates a values operator with auto-generated column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Self {
        let schema = Arc::new(Schema::new(columns));
        Self::new(schema, rows)
    }
}

impl Operator for ValuesOp {
    fn open(&mut self, _ctx: &ExecutionContext) -> OperatorResult<()> {
        self.current = 0;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.current >= self.rows.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let values = self.rows[self.current].clone();
        self.current += 1;
        self.base.inc_rows_produced();

        let row = Row::new(self.base.schema(), values);
        Ok(Some(row))
    }

    fn close(&mut self) -> OperatorResult<()> {
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
        "Values"
    }
}

/// Empty operator - produces no rows.
pub struct EmptyOp {
    /// Base operator state.
    base: OperatorBase,
}

impl EmptyOp {
    /// Creates a new empty operator.
    #[must_use]
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { base: OperatorBase::new(schema) }
    }

    /// Creates an empty operator with column names.
    #[must_use]
    pub fn with_columns(columns: Vec<String>) -> Self {
        let schema = Arc::new(Schema::new(columns));
        Self::new(schema)
    }
}

impl Operator for EmptyOp {
    fn open(&mut self, _ctx: &ExecutionContext) -> OperatorResult<()> {
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        self.base.set_finished();
        Ok(None)
    }

    fn close(&mut self) -> OperatorResult<()> {
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
        "Empty"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_op_basic() {
        let mut op = ValuesOp::with_columns(
            vec!["x".to_string(), "y".to_string()],
            vec![vec![Value::Int(1), Value::Int(2)], vec![Value::Int(3), Value::Int(4)]],
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.values(), &[Value::Int(1), Value::Int(2)]);

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.values(), &[Value::Int(3), Value::Int(4)]);

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn empty_op_basic() {
        let mut op = EmptyOp::with_columns(vec!["id".to_string()]);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert!(op.next().unwrap().is_none());
        assert_eq!(op.state(), OperatorState::Finished);

        op.close().unwrap();
    }
}
