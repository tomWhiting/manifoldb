//! Operator trait and base types.
//!
//! This module defines the [`Operator`] trait that all execution
//! operators implement.

use std::sync::Arc;

use crate::error::ParseError;

use super::context::ExecutionContext;
use super::row::{Row, Schema};

/// Result type for operator operations.
pub type OperatorResult<T> = Result<T, ParseError>;

/// The state of an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorState {
    /// Operator has not been opened yet.
    Created,
    /// Operator is open and ready to produce rows.
    Open,
    /// Operator has finished producing rows.
    Finished,
    /// Operator has been closed.
    Closed,
}

impl OperatorState {
    /// Returns true if the operator is open.
    #[must_use]
    pub const fn is_open(self) -> bool {
        matches!(self, Self::Open)
    }

    /// Returns true if the operator has finished.
    #[must_use]
    pub const fn is_finished(self) -> bool {
        matches!(self, Self::Finished)
    }

    /// Returns true if the operator is closed.
    #[must_use]
    pub const fn is_closed(self) -> bool {
        matches!(self, Self::Closed)
    }
}

/// The operator trait for pull-based query execution.
///
/// Operators are organized in a tree structure matching the physical plan.
/// Data flows from leaf operators (scans) up through intermediate operators
/// (filter, project, join) to the root.
///
/// # Lifecycle
///
/// 1. **Created**: Initial state after construction
/// 2. **Open**: After `open()` is called; ready to produce rows
/// 3. **Finished**: After `next()` returns `None`; no more rows
/// 4. **Closed**: After `close()` is called; resources released
///
/// # Thread Safety
///
/// The `Send` bound allows operators to be passed between threads,
/// but operators are not required to be `Sync` - they maintain mutable
/// internal state.
pub trait Operator: Send {
    /// Opens the operator and prepares it to produce rows.
    ///
    /// This method recursively opens any child operators.
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()>;

    /// Returns the next row, or `None` if there are no more rows.
    ///
    /// This method should be called repeatedly until it returns `None`.
    fn next(&mut self) -> OperatorResult<Option<Row>>;

    /// Closes the operator and releases resources.
    ///
    /// This method recursively closes any child operators.
    fn close(&mut self) -> OperatorResult<()>;

    /// Returns the output schema of this operator.
    fn schema(&self) -> Arc<Schema>;

    /// Returns the current state of this operator.
    fn state(&self) -> OperatorState;

    /// Returns the name of this operator type.
    fn name(&self) -> &'static str;
}

/// A boxed operator for dynamic dispatch.
pub type BoxedOperator = Box<dyn Operator>;

/// Base implementation for operators.
///
/// This struct provides common functionality that operators can use.
#[derive(Debug)]
pub struct OperatorBase {
    /// The output schema.
    schema: Arc<Schema>,
    /// The current state.
    state: OperatorState,
    /// Number of rows produced.
    rows_produced: u64,
}

impl OperatorBase {
    /// Creates a new operator base with the given schema.
    #[must_use]
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema, state: OperatorState::Created, rows_produced: 0 }
    }

    /// Returns the schema.
    #[must_use]
    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Returns the current state.
    #[must_use]
    pub const fn state(&self) -> OperatorState {
        self.state
    }

    /// Sets the state to open.
    pub fn set_open(&mut self) {
        self.state = OperatorState::Open;
    }

    /// Sets the state to finished.
    pub fn set_finished(&mut self) {
        self.state = OperatorState::Finished;
    }

    /// Sets the state to closed.
    pub fn set_closed(&mut self) {
        self.state = OperatorState::Closed;
    }

    /// Increments the rows produced counter.
    pub fn inc_rows_produced(&mut self) {
        self.rows_produced += 1;
    }

    /// Returns the number of rows produced.
    #[must_use]
    pub const fn rows_produced(&self) -> u64 {
        self.rows_produced
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_state_transitions() {
        let mut base = OperatorBase::new(Arc::new(Schema::empty()));

        assert_eq!(base.state(), OperatorState::Created);

        base.set_open();
        assert!(base.state().is_open());

        base.set_finished();
        assert!(base.state().is_finished());

        base.set_closed();
        assert!(base.state().is_closed());
    }

    #[test]
    fn operator_base_rows() {
        let mut base = OperatorBase::new(Arc::new(Schema::empty()));
        assert_eq!(base.rows_produced(), 0);

        base.inc_rows_produced();
        base.inc_rows_produced();
        assert_eq!(base.rows_produced(), 2);
    }
}
