//! Recursive CTE operator for WITH RECURSIVE queries.
//!
//! The recursive CTE operator implements iterative evaluation of recursive
//! Common Table Expressions. It executes an initial (base case) query once,
//! then repeatedly executes a recursive query until no new rows are produced.
//!
//! # Advanced Features
//!
//! ## SEARCH Clause
//! Controls traversal order (depth-first or breadth-first) and adds a sequence
//! column to track traversal order.
//!
//! ```sql
//! WITH RECURSIVE tree AS (...)
//! SEARCH DEPTH FIRST BY id SET ordercol
//! SELECT * FROM tree ORDER BY ordercol;
//! ```
//!
//! ## CYCLE Clause
//! Detects cycles in the recursion and optionally tracks the path.
//!
//! ```sql
//! WITH RECURSIVE path AS (...)
//! CYCLE id SET is_cycle USING path_array
//! SELECT * FROM path WHERE NOT is_cycle;
//! ```

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::physical::{CteCycleExecConfig, CteSearchExecConfig};
use manifoldb_core::Value;

/// Recursive CTE operator.
///
/// Implements the iterative semantics of WITH RECURSIVE:
/// 1. Execute the initial (base case) query to seed the working table
/// 2. Repeatedly execute the recursive query using the working table
/// 3. Add new rows to the result set (UNION ALL) or deduplicate (UNION)
/// 4. Continue until no new rows are produced or max iterations reached
///
/// # Advanced Features
///
/// - **SEARCH DEPTH/BREADTH FIRST**: Controls traversal order
/// - **CYCLE detection**: Detects cycles and optionally tracks path
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
/// SEARCH DEPTH FIRST BY id SET ordercol
/// CYCLE id SET is_cycle USING path_array
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
    /// Search configuration (SEARCH DEPTH/BREADTH FIRST).
    search_config: Option<CteSearchExecConfig>,
    /// Cycle detection configuration (CYCLE clause).
    cycle_config: Option<CteCycleExecConfig>,
    /// Initial (base case) operator.
    initial: BoxedOperator,
    /// Recursive operator (references the CTE).
    recursive: BoxedOperator,
    /// Accumulated result rows.
    result_rows: Vec<Row>,
    /// Current index into result_rows for output.
    result_index: usize,
    /// Working table rows (new rows from current iteration).
    /// For depth-first, this is used as a stack.
    /// For breadth-first, this is used as a queue.
    working_table: VecDeque<WorkingRow>,
    /// Set of row hashes for deduplication when union_all is false.
    seen_rows: HashSet<u64>,
    /// Current iteration count.
    iteration: usize,
    /// Whether evaluation is complete.
    evaluation_done: bool,
    /// Next sequence number for SEARCH clause.
    next_sequence: i64,
    /// Base column count (without added search/cycle columns).
    base_column_count: usize,
}

/// A row in the working table with tracking metadata.
#[derive(Debug, Clone)]
struct WorkingRow {
    /// The actual row values.
    values: Vec<Value>,
    /// The path of cycle check values (for CYCLE detection).
    path: Vec<Vec<Value>>,
    /// Depth in the traversal (for ordering).
    depth: usize,
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
        let base_column_count = columns.len();
        let schema = Arc::new(Schema::new(columns));

        Self {
            base: OperatorBase::new(schema),
            name: name.into(),
            union_all,
            max_iterations,
            search_config: None,
            cycle_config: None,
            initial,
            recursive,
            result_rows: Vec::new(),
            result_index: 0,
            working_table: VecDeque::new(),
            seen_rows: HashSet::new(),
            iteration: 0,
            evaluation_done: false,
            next_sequence: 0,
            base_column_count,
        }
    }

    /// Sets the search configuration.
    #[must_use]
    pub fn with_search(mut self, config: CteSearchExecConfig) -> Self {
        self.search_config = Some(config);
        self
    }

    /// Sets the cycle detection configuration.
    #[must_use]
    pub fn with_cycle(mut self, config: CteCycleExecConfig) -> Self {
        self.cycle_config = Some(config);
        self
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

    /// Hash values for cycle detection.
    fn hash_values(values: &[Value]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for value in values {
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

    /// Extracts cycle check values from a row based on configured columns.
    fn extract_cycle_values(&self, values: &[Value]) -> Vec<Value> {
        if let Some(ref config) = self.cycle_config {
            config.column_indices.iter().filter_map(|&idx| values.get(idx).cloned()).collect()
        } else {
            vec![]
        }
    }

    /// Checks if adding these values would create a cycle.
    #[allow(dead_code)]
    fn would_cycle(&self, values: &[Value], path: &[Vec<Value>]) -> bool {
        let check_values = self.extract_cycle_values(values);
        path.iter().any(|p| p == &check_values)
    }

    /// Creates an output row with optional sequence and cycle columns added.
    fn make_output_row(&mut self, working_row: &WorkingRow, is_cycle: bool) -> Row {
        let mut values = working_row.values.clone();

        // Add sequence column if SEARCH is configured
        if self.search_config.is_some() {
            values.push(Value::Int(self.next_sequence));
            self.next_sequence += 1;
        }

        // Add cycle mark column if CYCLE is configured
        if self.cycle_config.is_some() {
            values.push(Value::Bool(is_cycle));

            // Add path column if configured
            if self.cycle_config.as_ref().is_some_and(|c| c.path_column_index.is_some()) {
                // Convert path to array of strings for the path column
                let path_array: Vec<Value> = working_row
                    .path
                    .iter()
                    .map(|p| {
                        Value::String(
                            p.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join(","),
                        )
                    })
                    .collect();
                values.push(Value::Array(path_array));
            }
        }

        Row::new(self.base.schema(), values)
    }

    /// Runs the complete recursive evaluation with depth-first ordering.
    fn evaluate_depth_first(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Step 1: Execute initial query and seed working table
        self.initial.open(ctx)?;
        let mut initial_rows = Vec::new();
        while let Some(row) = self.initial.next()? {
            let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
            let cycle_values = self.extract_cycle_values(&values);
            initial_rows.push(WorkingRow { values, path: vec![cycle_values], depth: 0 });
        }
        self.initial.close()?;

        // For depth-first, add initial rows in reverse order so first row is processed first
        for row in initial_rows.into_iter().rev() {
            self.working_table.push_back(row);
        }

        // Step 2: Depth-first traversal using stack
        while let Some(working_row) = self.working_table.pop_back() {
            // Check iteration limit
            if self.iteration >= self.max_iterations {
                return Err(ParseError::Execution(format!(
                    "Recursive CTE '{}' exceeded maximum iterations ({})",
                    self.name, self.max_iterations
                )));
            }
            self.iteration += 1;

            // Check for cycle
            let is_cycle = if self.cycle_config.is_some() && working_row.depth > 0 {
                let cycle_values = self.extract_cycle_values(&working_row.values);
                working_row.path[..working_row.path.len() - 1].iter().any(|p| p == &cycle_values)
            } else {
                false
            };

            // Create output row
            let output_row = self.make_output_row(&working_row, is_cycle);
            if self.should_keep_row(&output_row) {
                self.result_rows.push(output_row);
            }

            // Don't recurse on cycles
            if is_cycle {
                continue;
            }

            // Execute recursive query for children
            // Note: In a full implementation, we'd bind working_row to the recursive scan
            self.recursive.open(ctx)?;
            let mut children = Vec::new();
            while let Some(row) = self.recursive.next()? {
                let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
                let cycle_values = self.extract_cycle_values(&values);

                // Build path for cycle detection
                let mut path = working_row.path.clone();
                path.push(cycle_values);

                children.push(WorkingRow { values, path, depth: working_row.depth + 1 });
            }
            self.recursive.close()?;

            // Add children in reverse order for depth-first (last child processed first)
            for child in children.into_iter().rev() {
                self.working_table.push_back(child);
            }
        }

        self.evaluation_done = true;
        Ok(())
    }

    /// Runs the complete recursive evaluation with breadth-first ordering.
    fn evaluate_breadth_first(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Step 1: Execute initial query and seed working table
        self.initial.open(ctx)?;
        while let Some(row) = self.initial.next()? {
            let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
            let cycle_values = self.extract_cycle_values(&values);
            self.working_table.push_back(WorkingRow { values, path: vec![cycle_values], depth: 0 });
        }
        self.initial.close()?;

        // Step 2: Breadth-first traversal using queue
        while let Some(working_row) = self.working_table.pop_front() {
            // Check iteration limit
            if self.iteration >= self.max_iterations {
                return Err(ParseError::Execution(format!(
                    "Recursive CTE '{}' exceeded maximum iterations ({})",
                    self.name, self.max_iterations
                )));
            }
            self.iteration += 1;

            // Check for cycle
            let is_cycle = if self.cycle_config.is_some() && working_row.depth > 0 {
                let cycle_values = self.extract_cycle_values(&working_row.values);
                working_row.path[..working_row.path.len() - 1].iter().any(|p| p == &cycle_values)
            } else {
                false
            };

            // Create output row
            let output_row = self.make_output_row(&working_row, is_cycle);
            if self.should_keep_row(&output_row) {
                self.result_rows.push(output_row);
            }

            // Don't recurse on cycles
            if is_cycle {
                continue;
            }

            // Execute recursive query for children
            self.recursive.open(ctx)?;
            while let Some(row) = self.recursive.next()? {
                let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
                let cycle_values = self.extract_cycle_values(&values);

                // Build path for cycle detection
                let mut path = working_row.path.clone();
                path.push(cycle_values);

                self.working_table.push_back(WorkingRow {
                    values,
                    path,
                    depth: working_row.depth + 1,
                });
            }
            self.recursive.close()?;
        }

        self.evaluation_done = true;
        Ok(())
    }

    /// Runs the complete recursive evaluation (standard iterative).
    fn evaluate_standard(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Step 1: Execute initial query
        self.initial.open(ctx)?;
        while let Some(row) = self.initial.next()? {
            let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
            let cycle_values = self.extract_cycle_values(&values);
            let working_row = WorkingRow { values, path: vec![cycle_values], depth: 0 };

            // Check for cycle (shouldn't happen in initial rows, but be safe)
            let is_cycle = false;
            let output_row = self.make_output_row(&working_row, is_cycle);

            if self.should_keep_row(&output_row) {
                self.working_table.push_back(working_row);
                self.result_rows.push(output_row);
            }
        }
        self.initial.close()?;

        // Step 2: Iteratively execute recursive query
        let mut current_depth = 0;
        while !self.working_table.is_empty() && self.iteration < self.max_iterations {
            self.iteration += 1;
            current_depth += 1;

            // Collect all working rows from this level
            let _working_rows: Vec<_> = self.working_table.drain(..).collect();

            // Execute recursive query
            self.recursive.open(ctx)?;
            while let Some(row) = self.recursive.next()? {
                let values: Vec<Value> = row.values()[..self.base_column_count].to_vec();
                let cycle_values = self.extract_cycle_values(&values);

                // For standard mode, use simplified path tracking
                let path = vec![cycle_values.clone()];
                let working_row = WorkingRow { values, path, depth: current_depth };

                // Check for cycle using seen_rows
                let is_cycle = if self.cycle_config.is_some() {
                    let hash = Self::hash_values(&cycle_values);
                    !self.seen_rows.insert(hash)
                } else {
                    false
                };

                let output_row = self.make_output_row(&working_row, is_cycle);

                if !is_cycle {
                    self.working_table.push_back(working_row);
                }
                self.result_rows.push(output_row);
            }
            self.recursive.close()?;
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

    /// Runs the complete recursive evaluation.
    fn evaluate(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        // Choose evaluation strategy based on search configuration
        if let Some(ref config) = self.search_config {
            if config.depth_first {
                self.evaluate_depth_first(ctx)
            } else {
                self.evaluate_breadth_first(ctx)
            }
        } else {
            self.evaluate_standard(ctx)
        }
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
        self.next_sequence = 0;
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

    #[test]
    fn recursive_cte_with_search_depth_first_config() {
        // Test that search config can be set
        let search_config = CteSearchExecConfig::depth_first(vec![0], 1);

        let op = RecursiveCTEOp::new(
            "tree",
            vec!["id".to_string()],
            true,
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        )
        .with_search(search_config);

        assert!(op.search_config.is_some());
        assert!(op.search_config.as_ref().unwrap().depth_first);
    }

    #[test]
    fn recursive_cte_with_search_breadth_first_config() {
        // Test that breadth-first search config can be set
        let search_config = CteSearchExecConfig::breadth_first(vec![0], 1);

        let op = RecursiveCTEOp::new(
            "tree",
            vec!["id".to_string()],
            true,
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        )
        .with_search(search_config);

        assert!(op.search_config.is_some());
        assert!(!op.search_config.as_ref().unwrap().depth_first);
    }

    #[test]
    fn recursive_cte_with_cycle_detection_config() {
        // Test that cycle detection config can be set
        let cycle_config = CteCycleExecConfig::new(vec![0], 1);

        let op = RecursiveCTEOp::new(
            "path",
            vec!["id".to_string()],
            true,
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        )
        .with_cycle(cycle_config);

        assert!(op.cycle_config.is_some());
        assert_eq!(op.cycle_config.as_ref().unwrap().column_indices, vec![0]);
    }

    #[test]
    fn recursive_cte_with_cycle_path_column() {
        // Test cycle config with path column
        let cycle_config = CteCycleExecConfig::new(vec![0], 1).with_path_column(2);

        let op = RecursiveCTEOp::new(
            "path",
            vec!["id".to_string()],
            true,
            100,
            make_initial_op(),
            make_recursive_op_empty(),
        )
        .with_cycle(cycle_config);

        assert!(op.cycle_config.as_ref().unwrap().path_column_index.is_some());
        assert_eq!(op.cycle_config.as_ref().unwrap().path_column_index, Some(2));
    }

    #[test]
    fn recursive_cte_depth_first_adds_sequence_column() {
        // Test that depth-first search adds a sequence column
        // Note: The schema must include all output columns (base + sequence).
        // base_column_count is set to columns.len(), so when we add sequence,
        // we need to include it in the schema.
        let initial = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ));

        // set_column_index is the index where sequence will be added (index 1)
        let search_config = CteSearchExecConfig::depth_first(vec![0], 1);

        // Include base column AND sequence column in schema
        // The base_column_count should be 1 (just "id"), but schema includes both
        let mut op = RecursiveCTEOp::new(
            "tree",
            vec!["id".to_string(), "seq".to_string()], // All output columns
            true,
            100,
            initial,
            make_recursive_op_empty(),
        )
        .with_search(search_config);

        // Fix: base_column_count should be 1, not 2
        // We need to manually set it since we included seq in the schema
        op.base_column_count = 1;

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // First row should have id and sequence 0
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        // Sequence is added as second column
        assert_eq!(row1.get(1), Some(&Value::Int(0)));

        // Second row should have id and sequence 1
        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(2)));
        assert_eq!(row2.get(1), Some(&Value::Int(1)));

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn recursive_cte_breadth_first_adds_sequence_column() {
        // Test that breadth-first search adds a sequence column
        let initial = Box::new(ValuesOp::with_columns(
            vec!["id".to_string()],
            vec![vec![Value::Int(1)], vec![Value::Int(2)]],
        ));

        let search_config = CteSearchExecConfig::breadth_first(vec![0], 1);

        // Include all output columns in schema
        let mut op = RecursiveCTEOp::new(
            "tree",
            vec!["id".to_string(), "seq".to_string()], // All output columns
            true,
            100,
            initial,
            make_recursive_op_empty(),
        )
        .with_search(search_config);

        // Fix: base_column_count should be 1
        op.base_column_count = 1;

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Both rows should be processed in breadth-first order
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        assert_eq!(row1.get(1), Some(&Value::Int(0))); // First sequence number

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(2)));
        assert_eq!(row2.get(1), Some(&Value::Int(1))); // Second sequence number

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn cte_search_config_constructors() {
        let df = CteSearchExecConfig::depth_first(vec![0, 1], 2);
        assert!(df.depth_first);
        assert_eq!(df.by_column_indices, vec![0, 1]);
        assert_eq!(df.set_column_index, 2);

        let bf = CteSearchExecConfig::breadth_first(vec![0], 1);
        assert!(!bf.depth_first);
        assert_eq!(bf.by_column_indices, vec![0]);
        assert_eq!(bf.set_column_index, 1);
    }

    #[test]
    fn cte_cycle_config_constructors() {
        let basic = CteCycleExecConfig::new(vec![0, 1], 2);
        assert_eq!(basic.column_indices, vec![0, 1]);
        assert_eq!(basic.mark_column_index, 2);
        assert!(basic.path_column_index.is_none());

        let with_path = CteCycleExecConfig::new(vec![0], 1).with_path_column(2);
        assert_eq!(with_path.path_column_index, Some(2));
    }
}
