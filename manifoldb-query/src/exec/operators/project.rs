//! Projection operator for column projection and expression evaluation.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::operators::filter::evaluate_expr;
use crate::exec::row::{Row, Schema};
use crate::plan::logical::LogicalExpr;

/// Projection operator.
///
/// Projects input rows to a subset of columns or computed expressions.
pub struct ProjectOp {
    /// Base operator state.
    base: OperatorBase,
    /// Expressions to project.
    exprs: Vec<LogicalExpr>,
    /// Input operator.
    input: BoxedOperator,
}

impl ProjectOp {
    /// Creates a new projection operator.
    #[must_use]
    pub fn new(exprs: Vec<LogicalExpr>, input: BoxedOperator) -> Self {
        // Build output schema from expressions
        let columns: Vec<String> =
            exprs.iter().enumerate().map(|(i, expr)| expr_to_column_name(expr, i)).collect();
        let schema = Arc::new(Schema::new(columns));

        Self { base: OperatorBase::new(schema), exprs, input }
    }

    /// Returns the projection expressions.
    #[must_use]
    pub fn expressions(&self) -> &[LogicalExpr] {
        &self.exprs
    }
}

impl Operator for ProjectOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        match self.input.next()? {
            Some(input_row) => {
                // Evaluate each expression against the input row
                let values: Vec<Value> = self
                    .exprs
                    .iter()
                    .map(|expr| evaluate_expr(expr, &input_row))
                    .collect::<OperatorResult<Vec<_>>>()?;

                self.base.inc_rows_produced();
                let row = Row::new(self.base.schema(), values);
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
        "Project"
    }
}

/// Derives a column name from an expression.
fn expr_to_column_name(expr: &LogicalExpr, index: usize) -> String {
    match expr {
        LogicalExpr::Column { name, .. } => name.clone(),
        LogicalExpr::Alias { alias, .. } => alias.clone(),
        LogicalExpr::AggregateFunction { func, distinct, .. } => {
            if *distinct {
                format!("{func}_distinct")
            } else {
                format!("{func}")
            }
        }
        LogicalExpr::ScalarFunction { func, .. } => format!("{func}"),
        LogicalExpr::Literal(lit) => format!("{lit}"),
        _ => format!("col_{index}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec![
                vec![Value::Int(1), Value::Int(2), Value::Int(3)],
                vec![Value::Int(4), Value::Int(5), Value::Int(6)],
            ],
        ))
    }

    #[test]
    fn project_columns() {
        // Project only columns a and c
        let exprs = vec![LogicalExpr::column("a"), LogicalExpr::column("c")];
        let mut project = ProjectOp::new(exprs, make_input());

        let ctx = ExecutionContext::new();
        project.open(&ctx).unwrap();

        let row1 = project.next().unwrap().unwrap();
        assert_eq!(row1.schema().columns(), &["a", "c"]);
        assert_eq!(row1.values(), &[Value::Int(1), Value::Int(3)]);

        let row2 = project.next().unwrap().unwrap();
        assert_eq!(row2.values(), &[Value::Int(4), Value::Int(6)]);

        assert!(project.next().unwrap().is_none());
        project.close().unwrap();
    }

    #[test]
    fn project_expressions() {
        // Project: a, a + b, a * c
        let exprs = vec![
            LogicalExpr::column("a"),
            LogicalExpr::column("a").add(LogicalExpr::column("b")),
            LogicalExpr::column("a").mul(LogicalExpr::column("c")),
        ];
        let mut project = ProjectOp::new(exprs, make_input());

        let ctx = ExecutionContext::new();
        project.open(&ctx).unwrap();

        let row1 = project.next().unwrap().unwrap();
        assert_eq!(row1.values(), &[Value::Int(1), Value::Int(3), Value::Int(3)]);

        let row2 = project.next().unwrap().unwrap();
        assert_eq!(row2.values(), &[Value::Int(4), Value::Int(9), Value::Int(24)]);

        project.close().unwrap();
    }

    #[test]
    fn project_with_alias() {
        let exprs = vec![LogicalExpr::column("a").alias("x"), LogicalExpr::column("b").alias("y")];
        let mut project = ProjectOp::new(exprs, make_input());

        let ctx = ExecutionContext::new();
        project.open(&ctx).unwrap();

        let row = project.next().unwrap().unwrap();
        assert_eq!(row.schema().columns(), &["x", "y"]);

        project.close().unwrap();
    }

    #[test]
    fn project_with_literals() {
        let exprs =
            vec![LogicalExpr::column("a"), LogicalExpr::integer(100), LogicalExpr::string("hello")];
        let mut project = ProjectOp::new(exprs, make_input());

        let ctx = ExecutionContext::new();
        project.open(&ctx).unwrap();

        let row = project.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(1)));
        assert_eq!(row.get(1), Some(&Value::Int(100)));
        assert_eq!(row.get(2), Some(&Value::from("hello")));

        project.close().unwrap();
    }
}
