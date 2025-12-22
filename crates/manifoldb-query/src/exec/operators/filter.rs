//! Filter operator for predicate evaluation.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::LogicalExpr;

/// Filter operator.
///
/// Evaluates a predicate for each input row and only passes through
/// rows where the predicate evaluates to true.
pub struct FilterOp {
    /// Base operator state.
    base: OperatorBase,
    /// The predicate to evaluate.
    predicate: LogicalExpr,
    /// Input operator.
    input: BoxedOperator,
}

impl FilterOp {
    /// Creates a new filter operator.
    #[must_use]
    pub fn new(predicate: LogicalExpr, input: BoxedOperator) -> Self {
        let schema = input.schema();
        Self { base: OperatorBase::new(schema), predicate, input }
    }

    /// Returns the predicate.
    #[must_use]
    pub fn predicate(&self) -> &LogicalExpr {
        &self.predicate
    }

    /// Evaluates the predicate against a row.
    fn evaluate_predicate(&self, row: &Row) -> OperatorResult<bool> {
        let value = evaluate_expr(&self.predicate, row)?;
        match value {
            Value::Bool(b) => Ok(b),
            Value::Null => Ok(false), // NULL is treated as false
            _ => Ok(false),
        }
    }
}

impl Operator for FilterOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.input.open(ctx)?;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        loop {
            match self.input.next()? {
                Some(row) => {
                    if self.evaluate_predicate(&row)? {
                        self.base.inc_rows_produced();
                        return Ok(Some(row));
                    }
                    // Row filtered out, continue to next
                }
                None => {
                    self.base.set_finished();
                    return Ok(None);
                }
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
        "Filter"
    }
}

/// Evaluates a logical expression against a row.
///
/// # NULL semantics
///
/// This function follows SQL NULL semantics:
/// - Missing columns return NULL (supports schema evolution and sparse data)
/// - Unresolved parameters return NULL (should be resolved before reaching filter)
pub fn evaluate_expr(expr: &LogicalExpr, row: &Row) -> OperatorResult<Value> {
    match expr {
        LogicalExpr::Literal(lit) => Ok(literal_to_value(lit)),

        LogicalExpr::Column { qualifier, name } => {
            // Try qualified name first (e.g., "u.id"), then unqualified (e.g., "id")
            // This supports both regular queries and joins with aliased tables
            let value = if let Some(qual) = qualifier {
                let qualified_name = format!("{}.{}", qual, name);
                row.get_by_name(&qualified_name).or_else(|| row.get_by_name(name))
            } else {
                // For unqualified names, also try finding a match that ends with ".name"
                row.get_by_name(name).or_else(|| {
                    // Search for a column that ends with ".{name}"
                    let suffix = format!(".{}", name);
                    row.schema()
                        .columns()
                        .iter()
                        .find(|col| col.ends_with(&suffix))
                        .and_then(|col| row.get_by_name(col))
                })
            };
            // Missing columns return NULL - follows SQL semantics for sparse/dynamic schemas
            Ok(value.cloned().unwrap_or(Value::Null))
        }

        LogicalExpr::BinaryOp { left, op, right } => {
            let left_val = evaluate_expr(left, row)?;
            let right_val = evaluate_expr(right, row)?;
            evaluate_binary_op(&left_val, op, &right_val)
        }

        LogicalExpr::UnaryOp { op, operand } => {
            let val = evaluate_expr(operand, row)?;
            evaluate_unary_op(op, &val)
        }

        LogicalExpr::Parameter(_idx) => {
            // Parameters should be resolved before evaluation - return NULL as fallback
            Ok(Value::Null)
        }

        LogicalExpr::InList { expr, list, negated } => {
            let val = evaluate_expr(expr, row)?;
            let mut found = false;
            for item in list {
                let item_val = evaluate_expr(item, row)?;
                if values_equal(&val, &item_val) {
                    found = true;
                    break;
                }
            }
            let result = if *negated { !found } else { found };
            Ok(Value::Bool(result))
        }

        LogicalExpr::Between { expr, low, high, negated } => {
            let val = evaluate_expr(expr, row)?;
            let low_val = evaluate_expr(low, row)?;
            let high_val = evaluate_expr(high, row)?;
            let in_range =
                compare_values(&val, &low_val) >= 0 && compare_values(&val, &high_val) <= 0;
            let result = if *negated { !in_range } else { in_range };
            Ok(Value::Bool(result))
        }

        LogicalExpr::Case { operand, when_clauses, else_result } => {
            if let Some(operand_expr) = operand {
                // Simple CASE
                let operand_val = evaluate_expr(operand_expr, row)?;
                for (when, then) in when_clauses {
                    let when_val = evaluate_expr(when, row)?;
                    if values_equal(&operand_val, &when_val) {
                        return evaluate_expr(then, row);
                    }
                }
            } else {
                // Searched CASE
                for (when, then) in when_clauses {
                    let when_val = evaluate_expr(when, row)?;
                    if matches!(when_val, Value::Bool(true)) {
                        return evaluate_expr(then, row);
                    }
                }
            }
            if let Some(else_expr) = else_result {
                evaluate_expr(else_expr, row)
            } else {
                Ok(Value::Null)
            }
        }

        LogicalExpr::Alias { expr, .. } => evaluate_expr(expr, row),

        LogicalExpr::Cast { expr, data_type: _ } => {
            // Simple cast implementation
            evaluate_expr(expr, row)
        }

        // These require subquery execution which is complex
        LogicalExpr::Subquery(_) | LogicalExpr::Exists { .. } | LogicalExpr::InSubquery { .. } => {
            Ok(Value::Null)
        }

        // Aggregates should be evaluated at the aggregate operator level
        LogicalExpr::AggregateFunction { .. } => Ok(Value::Null),

        // Scalar functions
        LogicalExpr::ScalarFunction { func, args } => {
            let arg_values: Vec<Value> =
                args.iter().map(|a| evaluate_expr(a, row)).collect::<OperatorResult<Vec<_>>>()?;
            evaluate_scalar_function(func, &arg_values)
        }

        LogicalExpr::Wildcard | LogicalExpr::QualifiedWildcard(_) => Ok(Value::Null),
    }
}

/// Converts a literal to a value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Boolean(b) => Value::Bool(*b),
        Literal::Integer(i) => Value::Int(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Vector(v) => Value::Vector(v.clone()),
    }
}

/// Evaluates a binary operation.
fn evaluate_binary_op(left: &Value, op: &BinaryOp, right: &Value) -> OperatorResult<Value> {
    // Handle NULL propagation
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        return match op {
            BinaryOp::And => {
                // NULL AND FALSE = FALSE
                if matches!(left, Value::Bool(false)) || matches!(right, Value::Bool(false)) {
                    Ok(Value::Bool(false))
                } else {
                    Ok(Value::Null)
                }
            }
            BinaryOp::Or => {
                // NULL OR TRUE = TRUE
                if matches!(left, Value::Bool(true)) || matches!(right, Value::Bool(true)) {
                    Ok(Value::Bool(true))
                } else {
                    Ok(Value::Null)
                }
            }
            _ => Ok(Value::Null),
        };
    }

    match op {
        // Comparison operators
        BinaryOp::Eq => Ok(Value::Bool(values_equal(left, right))),
        BinaryOp::NotEq => Ok(Value::Bool(!values_equal(left, right))),
        BinaryOp::Lt => Ok(Value::Bool(compare_values(left, right) < 0)),
        BinaryOp::LtEq => Ok(Value::Bool(compare_values(left, right) <= 0)),
        BinaryOp::Gt => Ok(Value::Bool(compare_values(left, right) > 0)),
        BinaryOp::GtEq => Ok(Value::Bool(compare_values(left, right) >= 0)),

        // Logical operators
        BinaryOp::And => {
            let l = value_to_bool(left);
            let r = value_to_bool(right);
            Ok(Value::Bool(l && r))
        }
        BinaryOp::Or => {
            let l = value_to_bool(left);
            let r = value_to_bool(right);
            Ok(Value::Bool(l || r))
        }

        // Arithmetic operators
        BinaryOp::Add => evaluate_arithmetic(left, right, |a, b| a + b, |a, b| a + b),
        BinaryOp::Sub => evaluate_arithmetic(left, right, |a, b| a - b, |a, b| a - b),
        BinaryOp::Mul => evaluate_arithmetic(left, right, |a, b| a * b, |a, b| a * b),
        BinaryOp::Div => {
            // Check for division by zero
            let is_zero = match right {
                Value::Int(0) => true,
                Value::Float(f) => *f == 0.0,
                _ => false,
            };
            if is_zero {
                Ok(Value::Null)
            } else {
                evaluate_arithmetic(left, right, |a, b| a / b, |a, b| a / b)
            }
        }
        BinaryOp::Mod => match (left, right) {
            (Value::Int(a), Value::Int(b)) if *b != 0 => Ok(Value::Int(a % b)),
            _ => Ok(Value::Null),
        },

        // String operators
        BinaryOp::Like => {
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                Ok(Value::Bool(like_match(s, pattern)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::ILike => {
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                Ok(Value::Bool(like_match(&s.to_lowercase(), &pattern.to_lowercase())))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::NotLike | BinaryOp::NotILike => {
            // Handle negated variants
            if let (Value::String(s), Value::String(pattern)) = (left, right) {
                let s = if matches!(op, BinaryOp::NotILike) { s.to_lowercase() } else { s.clone() };
                let pattern = if matches!(op, BinaryOp::NotILike) {
                    pattern.to_lowercase()
                } else {
                    pattern.clone()
                };
                Ok(Value::Bool(!like_match(&s, &pattern)))
            } else {
                Ok(Value::Null)
            }
        }

        // Vector operators
        BinaryOp::EuclideanDistance => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let dist = euclidean_distance(a, b);
                Ok(Value::Float(f64::from(dist)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::CosineDistance => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let dist = cosine_distance(a, b);
                Ok(Value::Float(f64::from(dist)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::InnerProduct => {
            if let (Value::Vector(a), Value::Vector(b)) = (left, right) {
                let prod = inner_product(a, b);
                Ok(Value::Float(f64::from(prod)))
            } else {
                Ok(Value::Null)
            }
        }
        BinaryOp::MaxSim => {
            // MaxSim operates on multi-vectors (Vec<Vec<f32>>)
            if let (Value::MultiVector(query), Value::MultiVector(doc)) = (left, right) {
                let score = maxsim_score(query, doc);
                Ok(Value::Float(f64::from(score)))
            } else {
                Ok(Value::Null)
            }
        }
    }
}

/// Evaluates a unary operation.
fn evaluate_unary_op(op: &UnaryOp, operand: &Value) -> OperatorResult<Value> {
    match op {
        UnaryOp::Not => match operand {
            Value::Bool(b) => Ok(Value::Bool(!b)),
            Value::Null => Ok(Value::Null),
            _ => Ok(Value::Bool(false)),
        },
        UnaryOp::Neg => match operand {
            Value::Int(i) => Ok(Value::Int(-i)),
            Value::Float(f) => Ok(Value::Float(-f)),
            _ => Ok(Value::Null),
        },
        UnaryOp::IsNull => Ok(Value::Bool(matches!(operand, Value::Null))),
        UnaryOp::IsNotNull => Ok(Value::Bool(!matches!(operand, Value::Null))),
    }
}

/// Evaluates a scalar function.
fn evaluate_scalar_function(
    func: &crate::plan::logical::ScalarFunction,
    args: &[Value],
) -> OperatorResult<Value> {
    use crate::plan::logical::ScalarFunction;

    match func {
        ScalarFunction::Upper => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Lower => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Length => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::Int(s.len() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Concat => {
            let result: String = args.iter().map(value_to_string).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Coalesce => {
            for arg in args {
                if !matches!(arg, Value::Null) {
                    return Ok(arg.clone());
                }
            }
            Ok(Value::Null)
        }
        ScalarFunction::Abs => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Int(i.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Ceil => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.ceil())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Floor => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.floor())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Round => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.round())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Sqrt => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).sqrt())),
            _ => Ok(Value::Null),
        },
        ScalarFunction::VectorDimension => {
            if let Some(Value::Vector(v)) = args.first() {
                Ok(Value::Int(v.len() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::VectorNorm => {
            if let Some(Value::Vector(v)) = args.first() {
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                Ok(Value::Float(f64::from(norm)))
            } else {
                Ok(Value::Null)
            }
        }
        _ => Ok(Value::Null),
    }
}

/// Compares two values for equality.
fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => {
            ((*a as f64) - b).abs() < f64::EPSILON
        }
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Vector(a), Value::Vector(b)) => a == b,
        _ => false,
    }
}

/// Compares two values, returning -1, 0, or 1.
fn compare_values(left: &Value, right: &Value) -> i32 {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b) as i32,
        (Value::Float(a), Value::Float(b)) => {
            if a < b {
                -1
            } else if a > b {
                1
            } else {
                0
            }
        }
        (Value::Int(a), Value::Float(b)) => {
            let a = *a as f64;
            if a < *b {
                -1
            } else if a > *b {
                1
            } else {
                0
            }
        }
        (Value::Float(a), Value::Int(b)) => {
            let b = *b as f64;
            if *a < b {
                -1
            } else if *a > b {
                1
            } else {
                0
            }
        }
        (Value::String(a), Value::String(b)) => a.cmp(b) as i32,
        _ => 0,
    }
}

/// Converts a value to boolean.
fn value_to_bool(value: &Value) -> bool {
    match value {
        Value::Bool(b) => *b,
        Value::Int(i) => *i != 0,
        Value::Float(f) => *f != 0.0,
        Value::String(s) => !s.is_empty(),
        _ => false,
    }
}

/// Converts a value to string.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.clone(),
        Value::Vector(v) => format!("{v:?}"),
        Value::SparseVector(v) => format!("{v:?}"),
        Value::MultiVector(v) => format!("{v:?}"),
        Value::Bytes(b) => format!("{b:?}"),
        Value::Array(a) => format!("{a:?}"),
    }
}

/// Evaluates arithmetic operations.
fn evaluate_arithmetic<F1, F2>(
    left: &Value,
    right: &Value,
    int_op: F1,
    float_op: F2,
) -> OperatorResult<Value>
where
    F1: Fn(i64, i64) -> i64,
    F2: Fn(f64, f64) -> f64,
{
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
        _ => Ok(Value::Null),
    }
}

/// Simple LIKE pattern matching.
fn like_match(s: &str, pattern: &str) -> bool {
    // Convert SQL LIKE pattern to simple matching
    // % matches any sequence, _ matches single char
    let regex_pattern = pattern.replace('%', ".*").replace('_', ".");

    // Simple check - in production would use proper regex
    if pattern.starts_with('%') && pattern.ends_with('%') {
        let inner = &pattern[1..pattern.len() - 1];
        s.contains(inner)
    } else if pattern.starts_with('%') {
        let suffix = &pattern[1..];
        s.ends_with(suffix)
    } else if pattern.ends_with('%') {
        let prefix = &pattern[..pattern.len() - 1];
        s.starts_with(prefix)
    } else {
        s == pattern || regex_pattern == format!(".*{}.*", s)
    }
}

/// Euclidean distance between vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Cosine distance between vectors.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return f32::MAX;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Inner product of vectors.
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// MaxSim score between multi-vectors.
///
/// Computes the sum of maximum similarities between each query token and all document tokens.
fn maxsim_score(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }

    // Verify dimensions match
    let dim = query[0].len();
    if doc[0].len() != dim {
        return 0.0;
    }

    let mut total_score = 0.0_f32;

    // For each query token, find the max dot product with any document token
    for q in query {
        let mut max_sim = f32::NEG_INFINITY;
        for d in doc {
            let sim: f32 = q.iter().zip(d.iter()).map(|(x, y)| x * y).sum();
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim.is_finite() {
            total_score += max_sim;
        }
    }

    total_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::operators::values::ValuesOp;

    fn make_input() -> BoxedOperator {
        Box::new(ValuesOp::with_columns(
            vec!["id".to_string(), "name".to_string(), "age".to_string()],
            vec![
                vec![Value::Int(1), Value::from("Alice"), Value::Int(30)],
                vec![Value::Int(2), Value::from("Bob"), Value::Int(25)],
                vec![Value::Int(3), Value::from("Carol"), Value::Int(35)],
            ],
        ))
    }

    #[test]
    fn filter_basic() {
        // Filter: age > 28
        let predicate = LogicalExpr::column("age").gt(LogicalExpr::integer(28));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        // Should return Alice (30) and Carol (35)
        let row1 = filter.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("name"), Some(&Value::from("Alice")));

        let row2 = filter.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("name"), Some(&Value::from("Carol")));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn filter_equality() {
        // Filter: id = 2
        let predicate = LogicalExpr::column("id").eq(LogicalExpr::integer(2));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        let row = filter.next().unwrap().unwrap();
        assert_eq!(row.get_by_name("name"), Some(&Value::from("Bob")));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn filter_and() {
        // Filter: age >= 25 AND age <= 30
        let predicate = LogicalExpr::column("age")
            .gt_eq(LogicalExpr::integer(25))
            .and(LogicalExpr::column("age").lt_eq(LogicalExpr::integer(30)));
        let mut filter = FilterOp::new(predicate, make_input());

        let ctx = ExecutionContext::new();
        filter.open(&ctx).unwrap();

        let row1 = filter.next().unwrap().unwrap();
        assert_eq!(row1.get_by_name("age"), Some(&Value::Int(30)));

        let row2 = filter.next().unwrap().unwrap();
        assert_eq!(row2.get_by_name("age"), Some(&Value::Int(25)));

        assert!(filter.next().unwrap().is_none());
        filter.close().unwrap();
    }

    #[test]
    fn evaluate_arithmetic_ops() {
        let schema = Arc::new(Schema::new(vec!["x".to_string()]));
        let row = Row::new(schema, vec![Value::Int(10)]);

        // x + 5
        let expr = LogicalExpr::column("x").add(LogicalExpr::integer(5));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Int(15));

        // x * 2
        let expr = LogicalExpr::column("x").mul(LogicalExpr::integer(2));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Int(20));
    }

    #[test]
    fn evaluate_in_list() {
        let schema = Arc::new(Schema::new(vec!["status".to_string()]));
        let row = Row::new(schema, vec![Value::from("active")]);

        let expr = LogicalExpr::column("status")
            .in_list(vec![LogicalExpr::string("active"), LogicalExpr::string("pending")], false);
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(true));

        let expr =
            LogicalExpr::column("status").in_list(vec![LogicalExpr::string("inactive")], false);
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(false));
    }

    #[test]
    fn evaluate_vector_distance() {
        let schema = Arc::new(Schema::new(vec!["v".to_string()]));
        let row = Row::new(schema, vec![Value::Vector(vec![1.0, 0.0, 0.0])]);

        let expr =
            LogicalExpr::column("v").euclidean_distance(LogicalExpr::vector(vec![0.0, 0.0, 0.0]));
        let result = evaluate_expr(&expr, &row).unwrap();

        if let Value::Float(d) = result {
            assert!((d - 1.0).abs() < 0.001);
        } else {
            panic!("Expected float");
        }
    }
}
