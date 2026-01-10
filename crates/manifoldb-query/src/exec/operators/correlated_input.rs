//! Correlated subquery input operator.
//!
//! This operator wraps a subquery and provides it with a seed row from
//! the execution context's variable bindings. It enables correlated
//! subqueries by materializing outer scope variables into a row that
//! the inner subquery operators can consume.
//!
//! # Architecture
//!
//! For a correlated subquery like:
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
//! The inner subquery's operators (e.g., `GraphExpandOp`) expect to receive
//! rows with the `p` column from their input. This operator:
//!
//! 1. Creates a seed row from the variable bindings in the execution context
//! 2. Combines this seed row with each row from the inner subquery
//! 3. Passes the combined rows through
//!
//! This allows the subquery to execute as if `p` came from an actual input
//! operator, while actually getting its value from the outer scope.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Correlated subquery input operator.
///
/// Wraps a subquery and provides it with seed row values from the
/// execution context's variable bindings. This enables correlated
/// subqueries where the inner query references variables from the outer scope.
pub struct CorrelatedInputOp {
    /// Base operator state.
    base: OperatorBase,
    /// Variable names to extract from the context as the seed row.
    imported_variables: Vec<String>,
    /// The inner subquery operator.
    inner: BoxedOperator,
    /// Schema of the inner subquery's output.
    inner_schema: Arc<Schema>,
    /// Whether the inner operator is currently open.
    inner_open: bool,
    /// Cached context for reopening.
    ctx: Option<ExecutionContext>,
}

impl CorrelatedInputOp {
    /// Creates a new correlated input operator.
    ///
    /// # Arguments
    ///
    /// * `imported_variables` - Names of variables to extract from the context
    /// * `inner` - The inner subquery operator
    #[must_use]
    pub fn new(imported_variables: Vec<String>, inner: BoxedOperator) -> Self {
        // The output schema includes both the imported variables and the inner subquery's columns
        let inner_schema = inner.schema();
        let inner_cols: Vec<&str> = inner_schema.columns();

        let mut combined_columns: Vec<String> = imported_variables.clone();
        for col in inner_cols {
            // Avoid duplicate column names
            if !imported_variables.contains(&col.to_string()) {
                combined_columns.push(col.to_string());
            }
        }

        let schema = Arc::new(Schema::new(combined_columns));

        Self {
            base: OperatorBase::new(schema),
            imported_variables,
            inner_schema,
            inner,
            inner_open: false,
            ctx: None,
        }
    }

    /// Creates the seed row from the current context's variable bindings.
    fn create_seed_row(&self) -> Row {
        let ctx = self.ctx.as_ref();
        let values: Vec<Value> = self
            .imported_variables
            .iter()
            .map(|name| ctx.and_then(|c| c.get_variable(name).cloned()).unwrap_or(Value::Null))
            .collect();

        // Create a schema for just the imported variables
        let seed_schema = Arc::new(Schema::new(self.imported_variables.clone()));
        Row::new(seed_schema, values)
    }

    /// Combines the seed row with an inner subquery result row.
    fn combine_rows(&self, seed_row: &Row, inner_row: &Row) -> Row {
        // Start with seed row values
        let mut values: Vec<Value> = seed_row.values().to_vec();

        // Add inner columns that aren't already in the seed row
        for (i, col) in self.inner_schema.columns().iter().enumerate() {
            if !self.imported_variables.contains(&(*col).to_string()) {
                if let Some(value) = inner_row.get(i) {
                    values.push(value.clone());
                } else {
                    values.push(Value::Null);
                }
            }
        }

        Row::new(self.base.schema(), values)
    }
}

impl Operator for CorrelatedInputOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Store the context for creating seed rows
        self.ctx = Some(
            ExecutionContext::with_variable_bindings(ctx.variable_bindings().clone())
                .with_graph(ctx.graph_arc())
                .with_graph_mutator(ctx.graph_mutator_arc()),
        );

        // Open the inner operator with the same context
        // The inner operator will use variable bindings for expression evaluation
        self.inner.open(ctx)?;
        self.inner_open = true;

        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if !self.inner_open {
            self.base.set_finished();
            return Ok(None);
        }

        // Get the seed row (values from outer scope)
        let seed_row = self.create_seed_row();

        // Get next row from the inner subquery
        match self.inner.next()? {
            Some(inner_row) => {
                // Combine seed row with inner result
                let combined = self.combine_rows(&seed_row, &inner_row);
                self.base.inc_rows_produced();
                Ok(Some(combined))
            }
            None => {
                self.base.set_finished();
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        if self.inner_open {
            self.inner.close()?;
            self.inner_open = false;
        }
        self.ctx = None;
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
        "CorrelatedInput"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use std::collections::HashMap;

    #[test]
    fn correlated_input_basic() {
        // Inner subquery produces two rows with "friend" column
        let inner = Box::new(ValuesOp::with_columns(
            vec!["friend".to_string()],
            vec![vec![Value::from("Alice")], vec![Value::from("Bob")]],
        ));

        let mut op = CorrelatedInputOp::new(vec!["person".to_string()], inner);

        // Create context with person = "Charlie" as a variable binding
        let mut bindings = HashMap::new();
        bindings.insert("person".to_string(), Value::from("Charlie"));
        let ctx = ExecutionContext::with_variable_bindings(bindings);

        op.open(&ctx).unwrap();

        // Should produce two rows, each with person and friend
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("person"), Some(&Value::from("Charlie")));
        assert_eq!(row1.get_by_name("friend"), Some(&Value::from("Alice")));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("person"), Some(&Value::from("Charlie")));
        assert_eq!(row2.get_by_name("friend"), Some(&Value::from("Bob")));

        // No more rows
        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn correlated_input_schema() {
        let inner =
            Box::new(ValuesOp::with_columns(vec!["a".to_string(), "b".to_string()], vec![]));

        let op = CorrelatedInputOp::new(vec!["x".to_string(), "y".to_string()], inner);

        let schema = op.schema();
        let cols = schema.columns();
        // Should have x, y (from imported), then a, b (from inner)
        assert_eq!(cols, vec!["x", "y", "a", "b"]);
    }

    #[test]
    fn correlated_input_overlapping_columns() {
        // Test when inner has same column name as imported variable
        let inner = Box::new(ValuesOp::with_columns(
            vec!["person".to_string(), "friend".to_string()],
            vec![vec![Value::from("Inner"), Value::from("Friend")]],
        ));

        let mut op = CorrelatedInputOp::new(vec!["person".to_string()], inner);

        let mut bindings = HashMap::new();
        bindings.insert("person".to_string(), Value::from("Outer"));
        let ctx = ExecutionContext::with_variable_bindings(bindings);

        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        // person should come from the outer scope (imported variable)
        assert_eq!(row.get_by_name("person"), Some(&Value::from("Outer")));
        // friend should come from inner
        assert_eq!(row.get_by_name("friend"), Some(&Value::from("Friend")));

        op.close().unwrap();
    }
}
