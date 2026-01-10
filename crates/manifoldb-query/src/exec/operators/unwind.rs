//! Unwind operator for expanding lists into rows.
//!
//! The UNWIND operator takes a list expression and produces one output row
//! for each element in the list, binding the element to the specified alias.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::error::ParseError;
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::LogicalExpr;

/// Unwind operator.
///
/// Expands a list expression into multiple rows, one per element.
/// For each input row, the list expression is evaluated and each element
/// becomes a separate output row with all original columns plus the unwound value.
///
/// # Semantics
///
/// - For `Value::Array`, produces one row per element with the element bound to the alias
/// - For `Value::Null`, produces zero rows (the input row is filtered out)
/// - For empty arrays, produces zero rows
/// - For non-array, non-null values, returns a type error
pub struct UnwindOp {
    /// Base operator state.
    base: OperatorBase,
    /// The expression that produces a list to unwind.
    list_expr: LogicalExpr,
    /// The variable name to bind each unwound element to.
    alias: String,
    /// Input operator.
    input: BoxedOperator,
    /// Current input row being processed (if any).
    current_row: Option<Row>,
    /// Elements from the current list being processed.
    current_elements: Vec<Value>,
    /// Index into current_elements.
    current_index: usize,
}

impl UnwindOp {
    /// Creates a new unwind operator.
    #[must_use]
    pub fn new(list_expr: LogicalExpr, alias: impl Into<String>, input: BoxedOperator) -> Self {
        let alias = alias.into();

        // Build output schema: input schema + the unwind alias
        let input_schema = input.schema();
        let output_schema = Arc::new(input_schema.with_column(&alias));

        Self {
            base: OperatorBase::new(output_schema),
            list_expr,
            alias,
            input,
            current_row: None,
            current_elements: Vec::new(),
            current_index: 0,
        }
    }

    /// Returns the list expression.
    #[must_use]
    pub fn list_expression(&self) -> &LogicalExpr {
        &self.list_expr
    }

    /// Returns the alias for the unwound elements.
    #[must_use]
    pub fn alias(&self) -> &str {
        &self.alias
    }

    /// Fetches the next input row and prepares its list elements.
    /// Returns Ok(true) if a new row with elements is ready, Ok(false) if no more input.
    fn fetch_next_row(&mut self) -> OperatorResult<bool> {
        loop {
            match self.input.next()? {
                Some(row) => {
                    // Evaluate the list expression
                    let list_value = evaluate_expr(&self.list_expr, &row)?;

                    match list_value {
                        Value::Array(elements) => {
                            if elements.is_empty() {
                                // Empty list - skip this row, try next
                                continue;
                            }
                            // Store the row and elements
                            self.current_row = Some(row);
                            self.current_elements = elements;
                            self.current_index = 0;
                            return Ok(true);
                        }
                        Value::Null => {
                            // Null list - skip this row, try next
                            continue;
                        }
                        other => {
                            // Non-list value is a type error
                            let type_name = match &other {
                                Value::Null => "null",
                                Value::Bool(_) => "boolean",
                                Value::Int(_) => "integer",
                                Value::Float(_) => "float",
                                Value::String(_) => "string",
                                Value::Bytes(_) => "bytes",
                                Value::Vector(_) => "vector",
                                Value::SparseVector(_) => "sparse_vector",
                                Value::MultiVector(_) => "multi_vector",
                                Value::Array(_) => "array",
                                Value::Point { .. } => "point",
                                Value::Node { .. } => "node",
                                Value::Edge { .. } => "edge",
                            };
                            return Err(ParseError::Execution(format!(
                                "UNWIND requires a list, but got {type_name}"
                            )));
                        }
                    }
                }
                None => {
                    // No more input rows
                    return Ok(false);
                }
            }
        }
    }
}

impl Operator for UnwindOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            // If we have elements remaining from the current row, produce the next one
            if self.current_index < self.current_elements.len() {
                let element = self.current_elements[self.current_index].clone();
                self.current_index += 1;

                // Get the current row - it must exist if we have elements
                let input_row = self.current_row.as_ref().ok_or_else(|| {
                    ParseError::Execution("Internal error: no current row".into())
                })?;

                // Build output row: input values + unwound element
                let mut values = input_row.values().to_vec();
                values.push(element);

                self.base.inc_rows_produced();
                return Ok(Some(Row::new(self.base.schema(), values)));
            }

            // No more elements from current row, fetch next input row
            if !self.fetch_next_row()? {
                // No more input rows
                self.base.set_finished();
                return Ok(None);
            }
            // Loop back to process the new row's elements
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
        "Unwind"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_input_with_list() -> BoxedOperator {
        // Input with columns: id, items (array)
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "items".to_string()],
            vec![
                vec![
                    Value::Int(1),
                    Value::Array(vec![Value::from("a"), Value::from("b"), Value::from("c")]),
                ],
                vec![Value::Int(2), Value::Array(vec![Value::from("x"), Value::from("y")])],
            ],
        ))
    }

    #[test]
    fn unwind_basic() {
        let mut unwind =
            UnwindOp::new(LogicalExpr::column("items"), "item", make_input_with_list());

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        // Row 1, element 1: id=1, items=[a,b,c], item=a
        let row1 = unwind.next().unwrap().unwrap();
        assert_eq!(row1.schema().columns(), &["id", "items", "item"]);
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        assert_eq!(row1.get(2), Some(&Value::from("a")));

        // Row 1, element 2: item=b
        let row2 = unwind.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(1)));
        assert_eq!(row2.get(2), Some(&Value::from("b")));

        // Row 1, element 3: item=c
        let row3 = unwind.next().unwrap().unwrap();
        assert_eq!(row3.get(0), Some(&Value::Int(1)));
        assert_eq!(row3.get(2), Some(&Value::from("c")));

        // Row 2, element 1: id=2, item=x
        let row4 = unwind.next().unwrap().unwrap();
        assert_eq!(row4.get(0), Some(&Value::Int(2)));
        assert_eq!(row4.get(2), Some(&Value::from("x")));

        // Row 2, element 2: item=y
        let row5 = unwind.next().unwrap().unwrap();
        assert_eq!(row5.get(0), Some(&Value::Int(2)));
        assert_eq!(row5.get(2), Some(&Value::from("y")));

        // No more rows
        assert!(unwind.next().unwrap().is_none());
        unwind.close().unwrap();
    }

    #[test]
    fn unwind_null_list() {
        // Input with a null list - should produce no output rows for that input
        let input = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "items".to_string()],
            vec![
                vec![Value::Int(1), Value::Null],
                vec![Value::Int(2), Value::Array(vec![Value::from("x")])],
            ],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("items"), "item", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        // Should skip row with null list, only get row 2
        let row = unwind.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(2)));
        assert_eq!(row.get(2), Some(&Value::from("x")));

        assert!(unwind.next().unwrap().is_none());
        unwind.close().unwrap();
    }

    #[test]
    fn unwind_empty_list() {
        // Input with an empty list - should produce no output rows for that input
        let input = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "items".to_string()],
            vec![
                vec![Value::Int(1), Value::Array(vec![])],
                vec![Value::Int(2), Value::Array(vec![Value::from("x")])],
            ],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("items"), "item", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        // Should skip row with empty list, only get row 2
        let row = unwind.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(2)));
        assert_eq!(row.get(2), Some(&Value::from("x")));

        assert!(unwind.next().unwrap().is_none());
        unwind.close().unwrap();
    }

    #[test]
    fn unwind_non_list_error() {
        // Input with a non-list value - should error
        let input = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "items".to_string()],
            vec![vec![Value::Int(1), Value::Int(42)]],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("items"), "item", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        let result = unwind.next();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("UNWIND requires a list"));
    }

    #[test]
    fn unwind_multiple_arrays() {
        // Unwind where each row has a different array
        let input = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "nums".to_string()],
            vec![
                vec![Value::Int(1), Value::Array(vec![Value::Int(10), Value::Int(20)])],
                vec![Value::Int(2), Value::Array(vec![Value::Int(30)])],
            ],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("nums"), "n", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        // Row 1 with n=10
        let row1 = unwind.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        assert_eq!(row1.get(2), Some(&Value::Int(10)));

        // Row 1 with n=20
        let row2 = unwind.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(1)));
        assert_eq!(row2.get(2), Some(&Value::Int(20)));

        // Row 2 with n=30
        let row3 = unwind.next().unwrap().unwrap();
        assert_eq!(row3.get(0), Some(&Value::Int(2)));
        assert_eq!(row3.get(2), Some(&Value::Int(30)));

        assert!(unwind.next().unwrap().is_none());
        unwind.close().unwrap();
    }

    #[test]
    fn unwind_nested_arrays() {
        // Unwind arrays containing nested arrays
        let input = Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "nested".to_string()],
            vec![vec![
                Value::Int(1),
                Value::Array(vec![
                    Value::Array(vec![Value::Int(1), Value::Int(2)]),
                    Value::Array(vec![Value::Int(3), Value::Int(4)]),
                ]),
            ]],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("nested"), "arr", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        // First nested array [1, 2]
        let row1 = unwind.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        assert_eq!(row1.get(2), Some(&Value::Array(vec![Value::Int(1), Value::Int(2)])));

        // Second nested array [3, 4]
        let row2 = unwind.next().unwrap().unwrap();
        assert_eq!(row2.get(2), Some(&Value::Array(vec![Value::Int(3), Value::Int(4)])));

        assert!(unwind.next().unwrap().is_none());
        unwind.close().unwrap();
    }

    #[test]
    fn unwind_preserves_all_columns() {
        // Verify that all input columns are preserved in output
        let input = Box::new(ValuesOp::with_columns(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![vec![Value::Int(1), Value::from("hello"), Value::Array(vec![Value::Int(10)])]],
        ));

        let mut unwind = UnwindOp::new(LogicalExpr::column("c"), "x", input);

        let ctx = ExecutionContext::new();
        unwind.open(&ctx).unwrap();

        let row = unwind.next().unwrap().unwrap();
        assert_eq!(row.schema().columns(), &["a", "b", "c", "x"]);
        assert_eq!(row.get(0), Some(&Value::Int(1)));
        assert_eq!(row.get(1), Some(&Value::from("hello")));
        assert_eq!(row.get(3), Some(&Value::Int(10)));

        unwind.close().unwrap();
    }
}
