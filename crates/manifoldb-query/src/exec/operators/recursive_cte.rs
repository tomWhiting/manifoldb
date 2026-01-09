//! Recursive CTE operator for WITH RECURSIVE queries.
//!
//! The recursive CTE operator implements iterative evaluation of recursive
//! Common Table Expressions. It executes an initial (base case) query once,
//! then repeatedly executes a recursive query until no new rows are produced.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};

/// Recursive CTE operator.
///
/// Implements the iterative semantics of WITH RECURSIVE:
/// 1. Execute the initial (base case) query to seed the working table
/// 2. Repeatedly execute the recursive query using the working table
/// 3. Add new rows to the result set (UNION ALL) or deduplicate (UNION)
/// 4. Continue until no new rows are produced or max iterations reached
///
/// # Example
///
/// ```sql
/// WITH RECURSIVE hierarchy AS (
///     -- Base case: root employees (no manager)
///     SELECT id, name, manager_id, 1 as level
///     FROM employees WHERE manager_id IS NULL
///     UNION ALL
///     -- Recursive case: employees with managers in the hierarchy
///     SELECT e.id, e.name, e.manager_id, h.level + 1
///     FROM employees e
///     JOIN hierarchy h ON e.manager_id = h.id
/// )
/// SELECT * FROM hierarchy;
/// ```
pub struct RecursiveCTEOp {
    /// Base operator state.
    base: OperatorBase,
    /// CTE name (for debugging/error messages).
    name: String,
    /// Whether to use UNION ALL (keep duplicates) or UNION (deduplicate).
    union_all: bool,
    /// Maximum number of iterations allowed.
    max_iterations: usize,
    /// Initial (base case) operator.
    initial: BoxedOperator,
    /// Recursive operator (references the CTE).
    recursive: BoxedOperator,
    /// Accumulated result rows.
    result_rows: Vec<Row>,
    /// Current index into result_rows for output.
    result_index: usize,
    /// Working table rows (new rows from current iteration).
    working_table: Vec<Row>,
    /// Set of row hashes for deduplication when union_all is false.
    seen_rows: HashSet<u64>,
    /// Current iteration count.
    iteration: usize,
    /// Whether evaluation is complete.
    evaluation_done: bool,
}

impl RecursiveCTEOp {
    /// Creates a new recursive CTE operator.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        columns: Vec<String>,
        union_all: bool,
        max_iterations: usize,
        initial: BoxedOperator,
        recursive: BoxedOperator,
    ) -> Self {
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            name: name.into(),
            union_all,
            max_iterations,
            initial,
            recursive,
            result_rows: Vec::new(),
            result_index: 0,
            working_table: Vec::new(),
            seen_rows: HashSet::new(),
            iteration: 0,
            evaluation_done: false,
        }
    }

    /// Hash a row for deduplication.
    fn hash_row(row: &Row) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in row.values() {
            // Hash the value's debug representation as a simple approach
            // A production implementation would use a proper value hash
            format!("{value:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Checks if we should keep this row (handles deduplication for UNION).
    fn should_keep_row(&mut self, row: &Row) -> bool {
        if self.union_all {
            true
        } else {
            let hash = Self::hash_row(row);
            self.seen_rows.insert(hash)
        }
    }

    /// Runs the complete recursive evaluation.
    fn evaluate(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Step 1: Execute initial query
        self.initial.open(ctx)?;
        while let Some(row) = self.initial.next()? {
            // Reproject the row to our schema
            let row = Row::new(self.base.schema(), row.values().to_vec());
            if self.should_keep_row(&row) {
                self.working_table.push(row.clone());
                self.result_rows.push(row);
            }
        }
        self.initial.close()?;

        // Step 2: Iteratively execute recursive query
        while !self.working_table.is_empty() && self.iteration < self.max_iterations {
            self.iteration += 1;

            // The recursive query reads from the working table
            // In a real implementation, we'd inject the working table as a scan source
            // For now, we execute the recursive operator which should reference the CTE

            // Open the recursive operator for this iteration
            self.recursive.open(ctx)?;

            let mut new_rows = Vec::new();
            while let Some(row) = self.recursive.next()? {
                // Reproject the row to our schema
                let row = Row::new(self.base.schema(), row.values().to_vec());
                if self.should_keep_row(&row) {
                    new_rows.push(row.clone());
                    self.result_rows.push(row);
                }
            }
            self.recursive.close()?;

            // Update working table for next iteration
            self.working_table = new_rows;
        }

        // Check if we hit the iteration limit
        if !self.working_table.is_empty() && self.iteration >= self.max_iterations {
            return Err(ParseError::Execution(format!(
                "Recursive CTE '{}' exceeded maximum iterations ({})",
                self.name, self.max_iterations
            )));
        }

        self.evaluation_done = true;
        Ok(())
    }
}

impl Operator for RecursiveCTEOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Run the complete recursive evaluation during open
        self.evaluate(ctx)?;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.result_index < self.result_rows.len() {
            let row = self.result_rows[self.result_index].clone();
            self.result_index += 1;
            self.base.inc_rows_produced();
            Ok(Some(row))
        } else {
            self.base.set_finished();
            Ok(None)
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        // Clear working data
        self.result_rows.clear();
        self.working_table.clear();
        self.seen_rows.clear();
        self.result_index = 0;
        self.iteration = 0;
        self.evaluation_done = false;
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
        "RecursiveCTE"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use manifoldb_core::Value;

    fn make_initial_op() -> BoxedOperator {
        // Simulates: SELECT 1 as n
        Box::new(ValuesOp::with_columns(vec!["n".to_string()], vec![vec![Value::Int(1)]]))
    }

    fn make_recursive_op_empty() -> BoxedOperator {
        // Returns no rows - recursive case terminates immediately
        Box::new(ValuesOp::with_columns(vec!["n".to_string()], vec![]))
    }

    #[test]
    fn recursive_cte_base_case_only() {
        // When recursive query returns no rows, should just return base case
        let mut op = RecursiveCTEOp::new(
            "test_cte",
            vec!["n".to_string()],
            true, // UNION ALL
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(1)));

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn recursive_cte_schema() {
        let op = RecursiveCTEOp::new(
            "hierarchy",
            vec!["id".to_string(), "name".to_string(), "level".to_string()],
            true,
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        );

        assert_eq!(op.schema().columns(), &["id", "name", "level"]);
    }

    #[test]
    fn recursive_cte_union_deduplication() {
        // Test UNION (not UNION ALL) semantics - duplicates should be removed
        let initial = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(1)], vec![Value::Int(2)]],
        ));

        let mut op = RecursiveCTEOp::new(
            "test_cte",
            vec!["n".to_string()],
            false, // UNION (deduplicate)
            100,
            initial,
            make_recursive_op_empty(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should only get 2 unique rows (1 and 2), not 3
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(2)));

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn recursive_cte_union_all_keeps_duplicates() {
        // Test UNION ALL semantics - duplicates should be kept
        let initial = Box::new(ValuesOp::with_columns(
            vec!["n".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(1)], vec![Value::Int(2)]],
        ));

        let mut op = RecursiveCTEOp::new(
            "test_cte",
            vec!["n".to_string()],
            true, // UNION ALL (keep duplicates)
            100,
            initial,
            make_recursive_op_empty(),
        );

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Should get all 3 rows including duplicates
        assert!(op.next().unwrap().is_some()); // 1
        assert!(op.next().unwrap().is_some()); // 1
        assert!(op.next().unwrap().is_some()); // 2
        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }
}
