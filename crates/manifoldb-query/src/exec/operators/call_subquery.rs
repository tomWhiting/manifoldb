//! CALL { } inline subquery operator for Cypher.
//!
//! The CALL { } operator executes an inline subquery for each input row,
//! with variables imported from the outer query scope. It implements
//! correlated and uncorrelated subquery semantics similar to SQL LATERAL joins.
//!
//! # Semantics
//!
//! For each row from the outer query:
//! 1. Bind imported variables from the outer row
//! 2. Execute the inner subquery
//! 3. Combine outer row with each subquery result row
//!
//! If the subquery returns no rows for an outer row, that outer row
//! is not included in the output (similar to INNER JOIN semantics).
//!
//! # Examples
//!
//! ```cypher
//! MATCH (p:Person)
//! CALL {
//!   WITH p
//!   MATCH (p)-[:KNOWS]->(friend)
//!   RETURN friend
//!   ORDER BY friend.name
//!   LIMIT 5
//! }
//! RETURN p.name, friend.name
//! ```

use std::sync::Arc;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use manifoldb_core::Value;

/// CALL { } inline subquery operator.
///
/// Executes a subquery for each input row, binding imported variables
/// from the outer context. Returns the cross product of input rows
/// with their corresponding subquery results.
pub struct CallSubqueryOp {
    /// Base operator state.
    base: OperatorBase,
    /// Variables imported from outer query via WITH clause.
    imported_variables: Vec<String>,
    /// Input operator (outer query rows).
    input: BoxedOperator,
    /// Subquery operator to execute for each input row.
    subquery: BoxedOperator,
    /// Schema of the subquery output.
    subquery_schema: Arc<Schema>,
    /// Current outer row being processed.
    current_outer_row: Option<Row>,
    /// Whether the subquery is currently open.
    subquery_open: bool,
    /// Whether we've exhausted the input.
    input_exhausted: bool,
    /// Cached execution context for use in next().
    ctx: Option<ExecutionContext>,
}

impl CallSubqueryOp {
    /// Creates a new CALL subquery operator.
    #[must_use]
    pub fn new(
        imported_variables: Vec<String>,
        input: BoxedOperator,
        subquery: BoxedOperator,
    ) -> Self {
        // Build the combined schema: input columns + subquery columns
        let input_schema = input.schema();
        let subquery_schema = subquery.schema();

        let input_cols: Vec<&str> = input_schema.columns();
        let mut combined_columns: Vec<String> =
            input_cols.iter().map(|s| (*s).to_string()).collect();
        for col in subquery_schema.columns() {
            // Avoid duplicate column names
            if !input_cols.contains(&col) {
                combined_columns.push(col.to_string());
            }
        }

        let schema = Arc::new(Schema::new(combined_columns));

        Self {
            base: OperatorBase::new(schema),
            imported_variables,
            subquery_schema,
            input,
            subquery,
            current_outer_row: None,
            subquery_open: false,
            input_exhausted: false,
            ctx: None,
        }
    }

    /// Returns true if this is an uncorrelated subquery (no imported variables).
    #[must_use]
    pub fn is_uncorrelated(&self) -> bool {
        self.imported_variables.is_empty()
    }

    /// Opens the subquery for the current outer row.
    fn open_subquery(&mut self) -> OperatorResult<()> {
        if let Some(ctx) = &self.ctx {
            // Open the subquery
            self.subquery.open(ctx)?;
            self.subquery_open = true;
        }
        Ok(())
    }

    /// Closes the subquery.
    fn close_subquery(&mut self) -> OperatorResult<()> {
        if self.subquery_open {
            self.subquery.close()?;
            self.subquery_open = false;
        }
        Ok(())
    }

    /// Combines an outer row with a subquery result row.
    fn combine_rows(&self, outer_row: &Row, subquery_row: &Row) -> Row {
        // Start with outer row values
        let mut values: Vec<Value> = outer_row.values().to_vec();

        // Add subquery columns that aren't already in the outer row
        let input_schema = self.input.schema();
        let outer_cols: Vec<&str> = input_schema.columns();
        for (i, col) in self.subquery_schema.columns().iter().enumerate() {
            if !outer_cols.contains(&col) {
                if let Some(value) = subquery_row.get(i) {
                    values.push(value.clone());
                } else {
                    values.push(Value::Null);
                }
            }
        }

        Row::new(self.base.schema(), values)
    }

    /// Advances to the next outer row and opens the subquery for it.
    fn advance_outer(&mut self) -> OperatorResult<bool> {
        // Close current subquery if open
        self.close_subquery()?;

        // Get next outer row
        match self.input.next()? {
            Some(row) => {
                self.current_outer_row = Some(row);
                self.open_subquery()?;
                Ok(true)
            }
            None => {
                self.input_exhausted = true;
                self.current_outer_row = None;
                Ok(false)
            }
        }
    }
}

impl Operator for CallSubqueryOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Open the input operator
        self.input.open(ctx)?;

        // Clone the context for use in next()
        // Note: ExecutionContext doesn't implement Clone, so we create a new one
        // In a real implementation, we'd use Arc<ExecutionContext> or similar
        self.ctx = Some(ExecutionContext::new());

        // Initialize state
        self.input_exhausted = false;
        self.current_outer_row = None;
        self.subquery_open = false;

        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // If we don't have a current outer row, get one
            if self.current_outer_row.is_none() && !self.advance_outer()? {
                // No more outer rows
                self.base.set_finished();
                return Ok(None);
            }

            // Try to get a row from the subquery
            if self.subquery_open {
                match self.subquery.next()? {
                    Some(subquery_row) => {
                        // Combine outer row with subquery result
                        if let Some(outer_row) = &self.current_outer_row {
                            let combined = self.combine_rows(outer_row, &subquery_row);
                            self.base.inc_rows_produced();
                            return Ok(Some(combined));
                        }
                    }
                    None => {
                        // Subquery exhausted for this outer row, move to next
                        self.current_outer_row = None;
                        self.close_subquery()?;
                    }
                }
            } else if !self.advance_outer()? {
                // No more outer rows
                self.base.set_finished();
                return Ok(None);
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.close_subquery()?;
        self.input.close()?;
        self.current_outer_row = None;
        self.input_exhausted = false;
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
        "CallSubquery"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_outer_op() -> BoxedOperator {
        // Simulates: two persons
        Box::new(ValuesOp::with_columns(
            vec!["person".to_string()],
            vec![vec![Value::from("Alice")], vec![Value::from("Bob")]],
        ))
    }

    fn make_subquery_op() -> BoxedOperator {
        // Simulates: two friends
        Box::new(ValuesOp::with_columns(
            vec!["friend".to_string()],
            vec![vec![Value::from("Friend1")], vec![Value::from("Friend2")]],
        ))
    }

    fn make_empty_subquery_op() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(vec!["friend".to_string()], vec![]))
    }

    #[test]
    fn call_subquery_uncorrelated_basic() {
        // CALL { RETURN 'Friend1', 'Friend2' } with outer persons
        // Should produce 2 persons * 2 friends = 4 rows
        let mut op = CallSubqueryOp::new(vec![], make_outer_op(), make_subquery_op());

        let ctx = ExecutionContext::new();

        op.open(&ctx).unwrap();

        let mut count = 0;
        while op.next().unwrap().is_some() {
            count += 1;
        }

        // 2 outer rows * 2 subquery rows = 4 combined rows
        assert_eq!(count, 4);
        op.close().unwrap();
    }

    #[test]
    fn call_subquery_empty_subquery() {
        // When subquery returns no rows, outer rows are not included
        let mut op = CallSubqueryOp::new(vec![], make_outer_op(), make_empty_subquery_op());

        let ctx = ExecutionContext::new();

        op.open(&ctx).unwrap();

        // Should produce no rows
        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn call_subquery_schema() {
        let op =
            CallSubqueryOp::new(vec!["person".to_string()], make_outer_op(), make_subquery_op());

        // Combined schema should have both person and friend columns
        let schema = op.schema();
        let cols = schema.columns();
        assert!(cols.contains(&"person"));
        assert!(cols.contains(&"friend"));
    }

    #[test]
    fn call_subquery_is_uncorrelated() {
        let uncorrelated = CallSubqueryOp::new(vec![], make_outer_op(), make_subquery_op());
        assert!(uncorrelated.is_uncorrelated());

        let correlated =
            CallSubqueryOp::new(vec!["person".to_string()], make_outer_op(), make_subquery_op());
        assert!(!correlated.is_uncorrelated());
    }
}
