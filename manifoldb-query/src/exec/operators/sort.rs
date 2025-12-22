//! Sort operator for ORDER BY.

use std::cmp::Ordering;
use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::SortOrder;

/// Sort operator.
///
/// Sorts input rows by the specified order expressions.
/// This is a blocking operator that materializes all input rows.
pub struct SortOp {
    /// Base operator state.
    base: OperatorBase,
    /// Sort specifications.
    order_by: Vec<SortOrder>,
    /// Input operator.
    input: BoxedOperator,
    /// Iterator over sorted rows (consumes without cloning).
    sorted_iter: std::vec::IntoIter<Row>,
    /// Whether rows have been materialized.
    materialized: bool,
}

impl SortOp {
    /// Creates a new sort operator.
    #[must_use]
    pub fn new(order_by: Vec<SortOrder>, input: BoxedOperator) -> Self {
        let schema = input.schema();
        Self {
            base: OperatorBase::new(schema),
            order_by,
            input,
            sorted_iter: Vec::new().into_iter(),
            materialized: false,
        }
    }

    /// Materializes and sorts all input rows.
    fn materialize_and_sort(&mut self) -> OperatorResult<()> {
        // Collect all rows
        let mut rows = Vec::new();
        while let Some(row) = self.input.next()? {
            rows.push(row);
        }

        // Sort rows
        let order_by = &self.order_by;
        rows.sort_by(|a, b| compare_rows(a, b, order_by));

        // Convert to iterator for zero-copy consumption
        self.sorted_iter = rows.into_iter();
        self.materialized = true;
        Ok(())
    }
}

impl Operator for SortOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.sorted_iter = Vec::new().into_iter();
        self.materialized = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        // Materialize on first call
        if !self.materialized {
            self.materialize_and_sort()?;
        }

        // Iterator yields owned rows without cloning
        match self.sorted_iter.next() {
            Some(row) => {
                self.base.inc_rows_produced();
                Ok(Some(row))
            }
            None => {
                self.base.set_finished();
                Ok(None)
            }
        }
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.input.close()?;
        self.sorted_iter = Vec::new().into_iter();
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
        "Sort"
    }
}

/// Compares two rows based on sort specifications.
///
/// # Error handling
///
/// Expression evaluation errors are treated as NULL values. This allows sorting
/// to continue even when some rows have invalid or missing sort key values.
fn compare_rows(a: &Row, b: &Row, order_by: &[SortOrder]) -> Ordering {
    for sort in order_by {
        // Expression evaluation errors become NULL - allows sorting to continue
        let val_a = evaluate_expr(&sort.expr, a).unwrap_or(Value::Null);
        let val_b = evaluate_expr(&sort.expr, b).unwrap_or(Value::Null);

        let cmp = compare_values(&val_a, &val_b, sort.nulls_first);

        let cmp = if sort.ascending { cmp } else { cmp.reverse() };

        if cmp != Ordering::Equal {
            return cmp;
        }
    }
    Ordering::Equal
}

/// Compares two values with NULL handling.
///
/// # Float comparison and NaN handling
///
/// For float comparisons, NaN values are treated as equal to maintain a stable sort order.
/// This follows the principle that NaN should not cause sort instability. Use `nulls_first`
/// to control whether NaN/NULL values sort to the beginning or end.
///
/// # Default behavior
///
/// - `nulls_first` defaults to `false` (NULLS LAST) per SQL standard
fn compare_values(a: &Value, b: &Value, nulls_first: Option<bool>) -> Ordering {
    // Default to NULLS LAST per SQL standard when not specified
    let nulls_first = nulls_first.unwrap_or(false);

    match (a, b) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => {
            if nulls_first {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
        (_, Value::Null) => {
            if nulls_first {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        }
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        // NaN comparison: treat NaN as Equal to avoid sort instability
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;
    use crate::plan::logical::LogicalExpr;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["name".to_string(), "age".to_string()],
            vec![
                vec![Value::from("Bob"), Value::Int(30)],
                vec![Value::from("Alice"), Value::Int(25)],
                vec![Value::from("Carol"), Value::Int(35)],
                vec![Value::from("Dave"), Value::Int(25)],
            ],
        ))
    }

    #[test]
    fn sort_ascending() {
        let order_by = vec![SortOrder::asc(LogicalExpr::column("name"))];
        let mut op = SortOp::new(order_by, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("name"), Some(&Value::from("Alice")));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("name"), Some(&Value::from("Bob")));

        let row3 = op.next().unwrap().unwrap();
        assert_eq!(row3.get_by_name("name"), Some(&Value::from("Carol")));

        let row4 = op.next().unwrap().unwrap();
        assert_eq!(row4.get_by_name("name"), Some(&Value::from("Dave")));

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }

    #[test]
    fn sort_descending() {
        let order_by = vec![SortOrder::desc(LogicalExpr::column("age"))];
        let mut op = SortOp::new(order_by, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("age"), Some(&Value::Int(35)));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("age"), Some(&Value::Int(30)));

        op.close().unwrap();
    }

    #[test]
    fn sort_multiple_keys() {
        // Sort by age ASC, then name ASC
        let order_by = vec![
            SortOrder::asc(LogicalExpr::column("age")),
            SortOrder::asc(LogicalExpr::column("name")),
        ];
        let mut op = SortOp::new(order_by, make_input());

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // Age 25: Alice, Dave
        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("name"), Some(&Value::from("Alice")));
        assert_eq!(row1.get_by_name("age"), Some(&Value::Int(25)));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("name"), Some(&Value::from("Dave")));
        assert_eq!(row2.get_by_name("age"), Some(&Value::Int(25)));

        // Age 30: Bob
        let row3 = op.next().unwrap().unwrap();
        assert_eq!(row3.get_by_name("name"), Some(&Value::from("Bob")));

        // Age 35: Carol
        let row4 = op.next().unwrap().unwrap();
        assert_eq!(row4.get_by_name("name"), Some(&Value::from("Carol")));

        op.close().unwrap();
    }

    #[test]
    fn sort_with_nulls() {
        let input: BoxedOperator = Box::new(ValuesOp::with_columns(
            vec!["x".to_string()],
            vec![
                vec![Value::Int(3)],
                vec![Value::Null],
                vec![Value::Int(1)],
                vec![Value::Null],
                vec![Value::Int(2)],
            ],
        ));

        let order_by = vec![SortOrder::asc(LogicalExpr::column("x")).nulls_first()];
        let mut op = SortOp::new(order_by, input);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        // NULLs first
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Null));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Null));
        // Then sorted values
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(1)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(2)));
        assert_eq!(op.next().unwrap().unwrap().get(0), Some(&Value::Int(3)));

        op.close().unwrap();
    }
}
