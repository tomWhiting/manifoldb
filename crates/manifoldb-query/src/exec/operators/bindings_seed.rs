//! Bindings seed operator for correlated subqueries.
//!
//! This operator materializes variable bindings from the execution context
//! into a single seed row, enabling correlated subqueries to access variables
//! from the outer query scope.
//!
//! # Usage
//!
//! When executing a correlated subquery like:
//!
//! ```cypher
//! MATCH (p:Person)
//! CALL {
//!   WITH p
//!   MATCH (p)-[:KNOWS]->(friend)
//!   RETURN friend
//! }
//! RETURN p, friend
//! ```
//!
//! The `BindingsSeedOp` is placed at the root of the subquery operator tree.
//! When opened, it reads the variable bindings from the execution context
//! (which were set by `CallSubqueryOp`) and produces a single row containing
//! those values. This row then flows through the subquery operators, allowing
//! them to access the bound variables normally via column lookups.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Bindings seed operator.
///
/// Produces a single row containing the values of variables bound in the
/// execution context. This is the entry point for correlated subqueries,
/// converting context-based variable bindings into row-based data that
/// downstream operators can consume.
pub struct BindingsSeedOp {
    /// Base operator state.
    base: OperatorBase,
    /// The variable names to extract from the context.
    variable_names: Vec<String>,
    /// The seed row (populated on open).
    seed_row: Option<Row>,
    /// Whether we've returned the seed row.
    returned: bool,
}

impl BindingsSeedOp {
    /// Creates a new bindings seed operator.
    ///
    /// # Arguments
    ///
    /// * `variable_names` - The names of variables to extract from the
    ///   execution context. These become the columns of the output row.
    #[must_use]
    pub fn new(variable_names: Vec<String>) -> Self {
        let schema = Arc::new(Schema::new(variable_names.clone()));
        Self { base: OperatorBase::new(schema), variable_names, seed_row: None, returned: false }
    }
}

impl Operator for BindingsSeedOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Extract values for each variable from the context's variable bindings
        let values: Vec<Value> = self
            .variable_names
            .iter()
            .map(|name| ctx.get_variable(name).cloned().unwrap_or(Value::Null))
            .collect();

        // Create the seed row
        self.seed_row = Some(Row::new(self.base.schema(), values));
        self.returned = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.returned {
            self.base.set_finished();
            return Ok(None);
        }

        self.returned = true;
        self.base.inc_rows_produced();
        Ok(self.seed_row.clone())
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.seed_row = None;
        self.returned = false;
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
        "BindingsSeed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn bindings_seed_basic() {
        let mut op = BindingsSeedOp::new(vec!["x".to_string(), "y".to_string()]);

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), Value::Int(42));
        bindings.insert("y".to_string(), Value::from("hello"));

        let ctx = ExecutionContext::with_variable_bindings(bindings);
        op.open(&ctx).unwrap();

        // Should produce exactly one row
        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get_by_name("x"), Some(&Value::Int(42)));
        assert_eq!(row.get_by_name("y"), Some(&Value::from("hello")));

        // Should return None after the first row
        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn bindings_seed_missing_variable() {
        let mut op = BindingsSeedOp::new(vec!["x".to_string(), "missing".to_string()]);

        let mut bindings = HashMap::new();
        bindings.insert("x".to_string(), Value::Int(42));
        // "missing" is not bound

        let ctx = ExecutionContext::with_variable_bindings(bindings);
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get_by_name("x"), Some(&Value::Int(42)));
        // Missing variable should be NULL
        assert_eq!(row.get_by_name("missing"), Some(&Value::Null));

        op.close().unwrap();
    }

    #[test]
    fn bindings_seed_schema() {
        let op = BindingsSeedOp::new(vec!["a".to_string(), "b".to_string(), "c".to_string()]);

        let schema = op.schema();
        let cols = schema.columns();
        assert_eq!(cols, vec!["a", "b", "c"]);
    }

    #[test]
    fn bindings_seed_reopen() {
        let mut op = BindingsSeedOp::new(vec!["value".to_string()]);

        // First execution with value = 1
        let mut bindings = HashMap::new();
        bindings.insert("value".to_string(), Value::Int(1));
        let ctx1 = ExecutionContext::with_variable_bindings(bindings);

        op.open(&ctx1).unwrap();
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("value"), Some(&Value::Int(1)));
        op.close().unwrap();

        // Second execution with value = 2
        let mut bindings = HashMap::new();
        bindings.insert("value".to_string(), Value::Int(2));
        let ctx2 = ExecutionContext::with_variable_bindings(bindings);

        op.open(&ctx2).unwrap();
        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("value"), Some(&Value::Int(2)));
        op.close().unwrap();
    }
}
