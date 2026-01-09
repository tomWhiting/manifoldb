//! Filter operator for predicate evaluation.

use std::sync::Arc;

use manifoldb_core::Value;

use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::exec::context::ExecutionContext;
use crate::exec::operator::{BoxedOperator, Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::logical::{HybridCombinationMethod, LogicalExpr};

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

        LogicalExpr::HybridSearch { components, method } => {
            // Evaluate hybrid search by combining component distances
            if components.is_empty() {
                return Ok(Value::Null);
            }

            // Evaluate each component's distance expression
            let mut distances: Vec<(f64, f64)> = Vec::with_capacity(components.len());
            for comp in components {
                let dist_value = evaluate_expr(&comp.distance_expr, row)?;
                let dist = match dist_value {
                    Value::Float(f) => f,
                    Value::Int(i) => i as f64,
                    Value::Null => continue, // Skip null distances
                    _ => continue,
                };
                distances.push((dist, comp.weight));
            }

            if distances.is_empty() {
                return Ok(Value::Null);
            }

            // Combine distances based on method
            let combined_score = match method {
                HybridCombinationMethod::WeightedSum => {
                    // Weighted sum: sum(weight_i * distance_i)
                    // For distances (lower is better), we compute weighted average
                    let total_weight: f64 = distances.iter().map(|(_, w)| w).sum();
                    if total_weight == 0.0 {
                        return Ok(Value::Null);
                    }
                    let weighted_sum: f64 =
                        distances.iter().map(|(d, w)| d * w).sum::<f64>() / total_weight;
                    weighted_sum
                }
                HybridCombinationMethod::RRF { k } => {
                    // For RRF at the row level (without global ranking), we use a simplified approach:
                    // Convert distances to pseudo-ranks by normalizing and inverting
                    // RRF formula: sum(1 / (k + rank_i))
                    // Since we don't have actual ranks, we use the distance as a proxy
                    // Lower distance = better, so we use 1 / (k + distance * scale_factor)
                    let k_f64 = f64::from(*k);
                    let rrf_score: f64 = distances
                        .iter()
                        .map(|(d, w)| {
                            // Use weight to scale the contribution
                            // Lower distance = higher contribution
                            w / (k_f64 + d.abs())
                        })
                        .sum();
                    // RRF produces higher scores for better matches, but we need
                    // lower = better for sorting, so we negate
                    -rrf_score
                }
            };

            Ok(Value::Float(combined_score))
        }
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
        Literal::MultiVector(v) => Value::MultiVector(v.clone()),
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
        // Comparison operators - return NULL if either operand is NULL (SQL semantics)
        BinaryOp::Eq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(values_equal(left, right)))
            }
        }
        BinaryOp::NotEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(!values_equal(left, right)))
            }
        }
        BinaryOp::Lt => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) < 0))
            }
        }
        BinaryOp::LtEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) <= 0))
            }
        }
        BinaryOp::Gt => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) > 0))
            }
        }
        BinaryOp::GtEq => {
            if matches!(left, Value::Null) || matches!(right, Value::Null) {
                Ok(Value::Null)
            } else {
                Ok(Value::Bool(compare_values(left, right) >= 0))
            }
        }

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

    // Helper to convert value to f64
    fn value_to_f64(v: &Value) -> Option<f64> {
        match v {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    match func {
        // ========== String Functions ==========
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
                Ok(Value::Int(s.chars().count() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Concat => {
            let result: String = args.iter().map(value_to_string).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Substring => {
            // SUBSTRING(string, start, [length])
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let start = match args.get(1) {
                Some(Value::Int(i)) => (*i as usize).saturating_sub(1), // SQL is 1-indexed
                _ => return Ok(Value::Null),
            };
            let chars: Vec<char> = s.chars().collect();
            if start >= chars.len() {
                return Ok(Value::String(String::new()));
            }
            let len = match args.get(2) {
                Some(Value::Int(l)) => *l as usize,
                _ => chars.len() - start,
            };
            let result: String = chars.iter().skip(start).take(len).collect();
            Ok(Value::String(result))
        }
        ScalarFunction::Trim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Ltrim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_start().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Rtrim => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_end().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        ScalarFunction::Replace => {
            // REPLACE(string, from, to)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let from = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            let to = match args.get(2) {
                Some(Value::String(t)) => t,
                _ => return Ok(Value::Null),
            };
            Ok(Value::String(s.replace(from.as_str(), to.as_str())))
        }
        ScalarFunction::Position => {
            // POSITION(substring IN string) - typically called as strpos(string, substring)
            // Args: [substring, string] or [string, substring] depending on call style
            // We'll support both: strpos(string, substring) and position with two args
            let (haystack, needle) = if args.len() >= 2 {
                match (&args[0], &args[1]) {
                    (Value::String(h), Value::String(n)) => (h, n),
                    _ => return Ok(Value::Null),
                }
            } else {
                return Ok(Value::Null);
            };
            match haystack.find(needle.as_str()) {
                Some(pos) => Ok(Value::Int((pos + 1) as i64)), // 1-indexed
                None => Ok(Value::Int(0)),
            }
        }
        ScalarFunction::ConcatWs => {
            // CONCAT_WS(separator, string, ...)
            let sep = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let parts: Vec<String> = args
                .iter()
                .skip(1)
                .filter(|v| !matches!(v, Value::Null))
                .map(value_to_string)
                .collect();
            Ok(Value::String(parts.join(sep)))
        }
        ScalarFunction::SplitPart => {
            // SPLIT_PART(string, delimiter, position)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let delim = match args.get(1) {
                Some(Value::String(d)) => d,
                _ => return Ok(Value::Null),
            };
            let pos = match args.get(2) {
                Some(Value::Int(p)) => *p,
                _ => return Ok(Value::Null),
            };
            if pos <= 0 {
                return Ok(Value::String(String::new()));
            }
            let parts: Vec<&str> = s.split(delim.as_str()).collect();
            let idx = (pos - 1) as usize; // 1-indexed to 0-indexed
            if idx < parts.len() {
                Ok(Value::String(parts[idx].to_string()))
            } else {
                Ok(Value::String(String::new()))
            }
        }
        ScalarFunction::Format => {
            // FORMAT(template, args...)
            // Simple %s substitution (PostgreSQL-style format)
            let template = match args.first() {
                Some(Value::String(s)) => s.clone(),
                _ => return Ok(Value::Null),
            };
            let mut result = template;
            for arg in args.iter().skip(1) {
                if let Some(pos) = result.find("%s") {
                    let replacement = value_to_string(arg);
                    result = format!("{}{}{}", &result[..pos], replacement, &result[pos + 2..]);
                }
            }
            Ok(Value::String(result))
        }
        ScalarFunction::RegexpMatch => {
            // REGEXP_MATCH(string, pattern) - returns first match or null
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let pattern = match args.get(1) {
                Some(Value::String(p)) => p,
                _ => return Ok(Value::Null),
            };
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    if let Some(caps) = re.captures(s) {
                        // Return the full match or first capture group
                        let matched = caps.get(1).or_else(|| caps.get(0));
                        match matched {
                            Some(m) => Ok(Value::String(m.as_str().to_string())),
                            None => Ok(Value::Null),
                        }
                    } else {
                        Ok(Value::Null)
                    }
                }
                Err(_) => Ok(Value::Null),
            }
        }
        ScalarFunction::RegexpReplace => {
            // REGEXP_REPLACE(string, pattern, replacement)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let pattern = match args.get(1) {
                Some(Value::String(p)) => p,
                _ => return Ok(Value::Null),
            };
            let replacement = match args.get(2) {
                Some(Value::String(r)) => r,
                _ => return Ok(Value::Null),
            };
            match regex::Regex::new(pattern) {
                Ok(re) => Ok(Value::String(re.replace_all(s, replacement.as_str()).to_string())),
                Err(_) => Ok(Value::Null),
            }
        }
        ScalarFunction::Coalesce => {
            for arg in args {
                if !matches!(arg, Value::Null) {
                    return Ok(arg.clone());
                }
            }
            Ok(Value::Null)
        }
        ScalarFunction::NullIf => {
            // NULLIF(expr1, expr2) - returns NULL if expr1 = expr2, otherwise expr1
            if args.len() < 2 {
                return Ok(Value::Null);
            }
            if values_equal(&args[0], &args[1]) {
                Ok(Value::Null)
            } else {
                Ok(args[0].clone())
            }
        }

        // ========== Numeric Functions ==========
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
        ScalarFunction::Round => {
            let val = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => return Ok(Value::Int(*i)),
                _ => return Ok(Value::Null),
            };
            // Optional precision argument
            let precision = match args.get(1) {
                Some(Value::Int(p)) => *p,
                _ => 0,
            };
            if precision == 0 {
                Ok(Value::Float(val.round()))
            } else {
                let factor = 10_f64.powi(precision as i32);
                Ok(Value::Float((val * factor).round() / factor))
            }
        }
        ScalarFunction::Trunc => {
            let val = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => return Ok(Value::Int(*i)),
                _ => return Ok(Value::Null),
            };
            let precision = match args.get(1) {
                Some(Value::Int(p)) => *p,
                _ => 0,
            };
            if precision == 0 {
                Ok(Value::Float(val.trunc()))
            } else {
                let factor = 10_f64.powi(precision as i32);
                Ok(Value::Float((val * factor).trunc() / factor))
            }
        }
        ScalarFunction::Sqrt => match args.first() {
            Some(Value::Float(f)) if *f >= 0.0 => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(i)) if *i >= 0 => Ok(Value::Float((*i as f64).sqrt())),
            _ => Ok(Value::Null),
        },
        ScalarFunction::Power => {
            let base = value_to_f64(args.first().unwrap_or(&Value::Null));
            let exp = value_to_f64(args.get(1).unwrap_or(&Value::Null));
            match (base, exp) {
                (Some(b), Some(e)) => Ok(Value::Float(b.powf(e))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Exp => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.exp())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Ln => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if x > 0.0 => Ok(Value::Float(x.ln())),
                _ => Ok(Value::Null), // ln of non-positive is undefined
            }
        }
        ScalarFunction::Log => {
            // LOG(base, x) or LOG(x) for log base 10
            let (base, x) = if args.len() >= 2 {
                (
                    value_to_f64(args.first().unwrap_or(&Value::Null)),
                    value_to_f64(args.get(1).unwrap_or(&Value::Null)),
                )
            } else {
                (Some(10.0), value_to_f64(args.first().unwrap_or(&Value::Null)))
            };
            match (base, x) {
                (Some(b), Some(v)) if b > 0.0 && b != 1.0 && v > 0.0 => Ok(Value::Float(v.log(b))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Log10 => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if x > 0.0 => Ok(Value::Float(x.log10())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Sin => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.sin())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Cos => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.cos())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Tan => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.tan())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Asin => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if (-1.0..=1.0).contains(&x) => Ok(Value::Float(x.asin())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Acos => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) if (-1.0..=1.0).contains(&x) => Ok(Value::Float(x.acos())),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Atan => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.atan())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Atan2 => {
            let y = value_to_f64(args.first().unwrap_or(&Value::Null));
            let x = value_to_f64(args.get(1).unwrap_or(&Value::Null));
            match (y, x) {
                (Some(y_val), Some(x_val)) => Ok(Value::Float(y_val.atan2(x_val))),
                _ => Ok(Value::Null),
            }
        }
        ScalarFunction::Degrees => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.to_degrees())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Radians => {
            let val = value_to_f64(args.first().unwrap_or(&Value::Null));
            match val {
                Some(x) => Ok(Value::Float(x.to_radians())),
                None => Ok(Value::Null),
            }
        }
        ScalarFunction::Sign => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Int(i.signum())),
            Some(Value::Float(f)) => {
                if *f > 0.0 {
                    Ok(Value::Int(1))
                } else if *f < 0.0 {
                    Ok(Value::Int(-1))
                } else {
                    Ok(Value::Int(0))
                }
            }
            _ => Ok(Value::Null),
        },
        ScalarFunction::Pi => Ok(Value::Float(std::f64::consts::PI)),
        ScalarFunction::Random => Ok(Value::Float(rand_float())),

        // ========== Date/Time Functions ==========
        ScalarFunction::Now | ScalarFunction::CurrentDate | ScalarFunction::CurrentTime => {
            use chrono::Utc;
            let now = Utc::now();
            Ok(Value::String(now.format("%Y-%m-%d %H:%M:%S%.6f+00").to_string()))
        }
        ScalarFunction::Extract | ScalarFunction::DatePart => {
            // EXTRACT(field FROM datetime) or DATE_PART('field', datetime)
            let field = match args.first() {
                Some(Value::String(s)) => s.to_lowercase(),
                _ => return Ok(Value::Null),
            };
            let datetime_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            extract_date_part(&field, datetime_str)
        }
        ScalarFunction::DateTrunc => {
            // DATE_TRUNC('field', datetime)
            let field = match args.first() {
                Some(Value::String(s)) => s.to_lowercase(),
                _ => return Ok(Value::Null),
            };
            let datetime_str = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            truncate_date(&field, datetime_str)
        }
        ScalarFunction::ToTimestamp => {
            // TO_TIMESTAMP(string, format) or TO_TIMESTAMP(epoch)
            if args.len() == 1 {
                // Epoch seconds
                let epoch = value_to_f64(args.first().unwrap_or(&Value::Null));
                if let Some(secs) = epoch {
                    use chrono::{TimeZone, Utc};
                    let whole_secs = secs.trunc() as i64;
                    let nanos = ((secs.fract()) * 1_000_000_000.0) as u32;
                    if let Some(dt) = Utc.timestamp_opt(whole_secs, nanos).single() {
                        return Ok(Value::String(dt.format("%Y-%m-%d %H:%M:%S+00").to_string()));
                    }
                }
                Ok(Value::Null)
            } else {
                // String with format
                let s = match args.first() {
                    Some(Value::String(s)) => s,
                    _ => return Ok(Value::Null),
                };
                let format = match args.get(1) {
                    Some(Value::String(f)) => f,
                    _ => return Ok(Value::Null),
                };
                parse_datetime_with_format(s, format)
            }
        }
        ScalarFunction::ToDate => {
            // TO_DATE(string, format)
            let s = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let format = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            parse_date_with_format(s, format)
        }
        ScalarFunction::ToChar => {
            // TO_CHAR(datetime, format)
            let datetime_str = match args.first() {
                Some(Value::String(s)) => s,
                _ => return Ok(Value::Null),
            };
            let format = match args.get(1) {
                Some(Value::String(f)) => f,
                _ => return Ok(Value::Null),
            };
            format_datetime(datetime_str, format)
        }

        // ========== Vector Functions ==========
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

        // Custom functions (not implemented)
        ScalarFunction::Custom(_) => Ok(Value::Null),
    }
}

/// Generates a random float between 0.0 and 1.0.
fn rand_float() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    (hasher.finish() as f64) / (u64::MAX as f64)
}

/// Extracts a date part from a datetime string.
fn extract_date_part(field: &str, datetime_str: &str) -> OperatorResult<Value> {
    use chrono::{Datelike, Timelike};

    // Try parsing common formats
    let dt = parse_naive_datetime(datetime_str)?;

    let value = match field {
        "year" => dt.year() as f64,
        "month" => dt.month() as f64,
        "day" => dt.day() as f64,
        "hour" => dt.hour() as f64,
        "minute" => dt.minute() as f64,
        "second" => dt.second() as f64,
        "millisecond" | "milliseconds" => (dt.nanosecond() / 1_000_000) as f64,
        "microsecond" | "microseconds" => (dt.nanosecond() / 1_000) as f64,
        "dow" | "dayofweek" => dt.weekday().num_days_from_sunday() as f64,
        "doy" | "dayofyear" => dt.ordinal() as f64,
        "week" => dt.iso_week().week() as f64,
        "quarter" => ((dt.month() - 1) / 3 + 1) as f64,
        "epoch" => dt.and_utc().timestamp() as f64,
        _ => return Ok(Value::Null),
    };

    Ok(Value::Float(value))
}

/// Truncates a datetime to the specified precision.
fn truncate_date(field: &str, datetime_str: &str) -> OperatorResult<Value> {
    use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};

    let dt = parse_naive_datetime(datetime_str)?;

    let truncated = match field {
        "year" => NaiveDateTime::new(
            NaiveDate::from_ymd_opt(dt.year(), 1, 1).unwrap_or_else(|| dt.date()),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
        ),
        "month" => NaiveDateTime::new(
            NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1).unwrap_or_else(|| dt.date()),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
        ),
        "day" => {
            NaiveDateTime::new(dt.date(), NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default())
        }
        "hour" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), 0, 0).unwrap_or_default(),
        ),
        "minute" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), dt.minute(), 0).unwrap_or_default(),
        ),
        "second" => NaiveDateTime::new(
            dt.date(),
            NaiveTime::from_hms_opt(dt.hour(), dt.minute(), dt.second()).unwrap_or_default(),
        ),
        "week" => {
            let days_since_monday = dt.weekday().num_days_from_monday();
            let monday = dt.date() - chrono::Duration::days(days_since_monday as i64);
            NaiveDateTime::new(monday, NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default())
        }
        "quarter" => {
            let quarter_month = ((dt.month() - 1) / 3) * 3 + 1;
            NaiveDateTime::new(
                NaiveDate::from_ymd_opt(dt.year(), quarter_month, 1).unwrap_or_else(|| dt.date()),
                NaiveTime::from_hms_opt(0, 0, 0).unwrap_or_default(),
            )
        }
        _ => return Ok(Value::Null),
    };

    Ok(Value::String(truncated.format("%Y-%m-%d %H:%M:%S").to_string()))
}

/// Parses a naive datetime from various formats.
fn parse_naive_datetime(s: &str) -> OperatorResult<chrono::NaiveDateTime> {
    use chrono::NaiveDateTime;

    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ];

    // Strip timezone suffix if present
    let s = s.trim_end_matches("+00").trim_end_matches('Z');

    for fmt in &formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(dt);
        }
    }

    // Try date-only and add midnight
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date.and_hms_opt(0, 0, 0).unwrap_or_default());
    }

    Err(crate::error::ParseError::Execution(format!("Cannot parse datetime: {}", s)))
}

/// Parses a datetime string with PostgreSQL-style format.
fn parse_datetime_with_format(s: &str, pg_format: &str) -> OperatorResult<Value> {
    let chrono_format = pg_format_to_chrono(pg_format);
    use chrono::NaiveDateTime;

    match NaiveDateTime::parse_from_str(s, &chrono_format) {
        Ok(dt) => Ok(Value::String(dt.format("%Y-%m-%d %H:%M:%S").to_string())),
        Err(_) => Ok(Value::Null),
    }
}

/// Parses a date string with PostgreSQL-style format.
fn parse_date_with_format(s: &str, pg_format: &str) -> OperatorResult<Value> {
    let chrono_format = pg_format_to_chrono(pg_format);
    use chrono::NaiveDate;

    match NaiveDate::parse_from_str(s, &chrono_format) {
        Ok(d) => Ok(Value::String(d.format("%Y-%m-%d").to_string())),
        Err(_) => Ok(Value::Null),
    }
}

/// Formats a datetime with PostgreSQL-style format.
fn format_datetime(datetime_str: &str, pg_format: &str) -> OperatorResult<Value> {
    let dt = parse_naive_datetime(datetime_str)?;
    let chrono_format = pg_format_to_chrono(pg_format);
    Ok(Value::String(dt.format(&chrono_format).to_string()))
}

/// Converts PostgreSQL format specifiers to chrono format specifiers.
fn pg_format_to_chrono(pg_format: &str) -> String {
    pg_format
        .replace("YYYY", "%Y")
        .replace("YY", "%y")
        .replace("MM", "%m")
        .replace("DD", "%d")
        .replace("HH24", "%H")
        .replace("HH12", "%I")
        .replace("HH", "%H")
        .replace("MI", "%M")
        .replace("SS", "%S")
        .replace("MS", "%3f")
        .replace("US", "%6f")
        .replace("AM", "%p")
        .replace("PM", "%p")
        .replace("am", "%P")
        .replace("pm", "%P")
        .replace("TZ", "%Z")
        .replace("Day", "%A")
        .replace("Mon", "%b")
        .replace("Month", "%B")
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

/// SQL LIKE pattern matching.
///
/// Supports:
/// - `%` matches any sequence of characters (including empty)
/// - `_` matches exactly one character
fn like_match(s: &str, pattern: &str) -> bool {
    let s_chars: Vec<char> = s.chars().collect();
    let p_chars: Vec<char> = pattern.chars().collect();

    let s_len = s_chars.len();
    let p_len = p_chars.len();

    // dp[i][j] = true if s[0..i] matches pattern[0..j]
    let mut dp = vec![vec![false; p_len + 1]; s_len + 1];

    // Empty pattern matches empty string
    dp[0][0] = true;

    // Handle patterns starting with %
    for j in 1..=p_len {
        if p_chars[j - 1] == '%' {
            dp[0][j] = dp[0][j - 1];
        } else {
            break;
        }
    }

    for i in 1..=s_len {
        for j in 1..=p_len {
            let p_char = p_chars[j - 1];

            if p_char == '%' {
                // % matches zero or more characters
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if p_char == '_' {
                // _ matches exactly one character
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                // Regular character - must match exactly
                dp[i][j] = dp[i - 1][j - 1] && s_chars[i - 1] == p_char;
            }
        }
    }

    dp[s_len][p_len]
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

    #[test]
    fn evaluate_maxsim_operator() {
        // Test MaxSim operator for ColBERT-style multi-vector scoring
        // Query: 2 tokens, Document: 3 tokens, dimension: 2
        let query_tokens = vec![
            vec![1.0_f32, 0.0], // Query token 1
            vec![0.0_f32, 1.0], // Query token 2
        ];
        let doc_tokens = vec![
            vec![1.0_f32, 0.0], // Doc token 1 (matches query token 1 perfectly)
            vec![0.5_f32, 0.5], // Doc token 2
            vec![0.0_f32, 0.8], // Doc token 3 (close to query token 2)
        ];

        let schema = Arc::new(Schema::new(vec!["doc".to_string(), "query".to_string()]));
        let row = Row::new(
            schema,
            vec![Value::MultiVector(doc_tokens), Value::MultiVector(query_tokens)],
        );

        // Create MaxSim expression: query <##> doc
        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("query")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("doc")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();

        // Expected MaxSim score:
        // Query token 1 [1,0]: max dot products with doc tokens = max(1.0, 0.5, 0.0) = 1.0
        // Query token 2 [0,1]: max dot products with doc tokens = max(0.0, 0.5, 0.8) = 0.8
        // Total MaxSim = 1.0 + 0.8 = 1.8
        if let Value::Float(score) = result {
            assert!((score - 1.8).abs() < 0.001, "Expected MaxSim score ~1.8, got {score}");
        } else {
            panic!("Expected float result from MaxSim");
        }
    }

    #[test]
    fn evaluate_maxsim_identical_vectors() {
        // When query and document are identical, MaxSim should equal num_tokens
        let tokens = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];

        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));
        let row =
            Row::new(schema, vec![Value::MultiVector(tokens.clone()), Value::MultiVector(tokens)]);

        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("a")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("b")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();

        // Each query token matches itself perfectly: 1.0 + 1.0 = 2.0
        if let Value::Float(score) = result {
            assert!(
                (score - 2.0).abs() < 0.001,
                "Expected MaxSim score 2.0 for identical vectors, got {score}"
            );
        } else {
            panic!("Expected float result from MaxSim");
        }
    }

    #[test]
    fn evaluate_maxsim_null_on_type_mismatch() {
        // MaxSim should return Null when operands are not MultiVector
        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));
        let row = Row::new(
            schema,
            vec![
                Value::Vector(vec![1.0, 0.0]), // Dense vector, not multi-vector
                Value::Vector(vec![0.0, 1.0]),
            ],
        );

        let expr = LogicalExpr::BinaryOp {
            left: Box::new(LogicalExpr::column("a")),
            op: crate::ast::BinaryOp::MaxSim,
            right: Box::new(LogicalExpr::column("b")),
        };

        let result = evaluate_expr(&expr, &row).unwrap();
        assert!(matches!(result, Value::Null), "Expected Null for type mismatch");
    }

    #[test]
    fn like_match_percent_wildcard() {
        // % matches any sequence of characters
        assert!(like_match("hello", "hello"));
        assert!(like_match("hello", "%"));
        assert!(like_match("hello", "h%"));
        assert!(like_match("hello", "%o"));
        assert!(like_match("hello", "%ell%"));
        assert!(like_match("hello", "h%o"));
        assert!(like_match("hello world", "hello%world"));
        assert!(!like_match("hello", "world%"));
        assert!(!like_match("hello", "%world"));
    }

    #[test]
    fn like_match_underscore_wildcard() {
        // _ matches exactly one character
        assert!(like_match("hello", "h_llo"));
        assert!(like_match("hello", "_ello"));
        assert!(like_match("hello", "hell_"));
        assert!(like_match("hello", "_____"));
        assert!(like_match("ab", "a_"));
        assert!(like_match("ab", "_b"));
        assert!(!like_match("hello", "h_lo")); // _ is one char, not two
        assert!(!like_match("hello", "______")); // too many underscores
        assert!(!like_match("hello", "____")); // too few underscores
    }

    #[test]
    fn like_match_mixed_wildcards() {
        // Mix of % and _
        assert!(like_match("hello", "h_%"));
        assert!(like_match("hello", "%_o"));
        assert!(like_match("hello", "h_ll%"));
        assert!(like_match("hello world", "h%_d"));
        assert!(like_match("abc", "a%_"));
        assert!(like_match("abc", "_%c"));
        assert!(!like_match("a", "a%_")); // needs at least one char after a
    }

    #[test]
    fn like_match_empty_strings() {
        assert!(like_match("", ""));
        assert!(like_match("", "%"));
        assert!(!like_match("", "_"));
        assert!(!like_match("", "a"));
        assert!(!like_match("a", ""));
    }

    #[test]
    fn like_match_special_cases() {
        // Multiple consecutive wildcards
        assert!(like_match("hello", "%%"));
        assert!(like_match("hello", "%%%"));
        assert!(like_match("hello", "%_%"));
        assert!(like_match("h", "%_%"));
        assert!(!like_match("", "%_%")); // needs at least one char

        // Pattern at exact boundaries
        assert!(like_match("test", "test"));
        assert!(!like_match("test", "Test")); // case sensitive
    }

    #[test]
    fn null_comparison_returns_null() {
        // SQL semantics: any comparison with NULL returns NULL
        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string()]));

        // NULL = NULL should return NULL (not true)
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Null]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL = value should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Int(5)]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // value = NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Int(5), Value::Null]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL != NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Null]);
        let expr = LogicalExpr::column("a").not_eq(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // NULL < value should return NULL
        let row = Row::new(schema.clone(), vec![Value::Null, Value::Int(5)]);
        let expr = LogicalExpr::column("a").lt(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // value > NULL should return NULL
        let row = Row::new(schema.clone(), vec![Value::Int(5), Value::Null]);
        let expr = LogicalExpr::column("a").gt(LogicalExpr::column("b"));
        assert!(matches!(evaluate_expr(&expr, &row).unwrap(), Value::Null));

        // Non-null comparison should still work
        let row = Row::new(schema, vec![Value::Int(5), Value::Int(5)]);
        let expr = LogicalExpr::column("a").eq(LogicalExpr::column("b"));
        assert_eq!(evaluate_expr(&expr, &row).unwrap(), Value::Bool(true));
    }

    // ========== String Function Tests ==========

    fn eval_fn(func: crate::plan::logical::ScalarFunction, args: Vec<Value>) -> Value {
        evaluate_scalar_function(&func, &args).unwrap()
    }

    #[test]
    fn test_string_position() {
        use crate::plan::logical::ScalarFunction;

        // POSITION('lo' IN 'hello') = 4
        let result =
            eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("lo")]);
        assert_eq!(result, Value::Int(4));

        // Not found returns 0
        let result =
            eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("xyz")]);
        assert_eq!(result, Value::Int(0));

        // Empty substring returns 1
        let result = eval_fn(ScalarFunction::Position, vec![Value::from("hello"), Value::from("")]);
        assert_eq!(result, Value::Int(1));
    }

    #[test]
    fn test_string_concat_ws() {
        use crate::plan::logical::ScalarFunction;

        // CONCAT_WS(', ', 'a', 'b', 'c') = 'a, b, c'
        let result = eval_fn(
            ScalarFunction::ConcatWs,
            vec![Value::from(", "), Value::from("a"), Value::from("b"), Value::from("c")],
        );
        assert_eq!(result, Value::String("a, b, c".to_string()));

        // NULLs are skipped
        let result = eval_fn(
            ScalarFunction::ConcatWs,
            vec![Value::from("-"), Value::from("a"), Value::Null, Value::from("c")],
        );
        assert_eq!(result, Value::String("a-c".to_string()));
    }

    #[test]
    fn test_string_split_part() {
        use crate::plan::logical::ScalarFunction;

        // SPLIT_PART('a,b,c', ',', 2) = 'b'
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(2)],
        );
        assert_eq!(result, Value::String("b".to_string()));

        // Position 1
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(1)],
        );
        assert_eq!(result, Value::String("a".to_string()));

        // Position beyond parts returns empty string
        let result = eval_fn(
            ScalarFunction::SplitPart,
            vec![Value::from("a,b,c"), Value::from(","), Value::Int(5)],
        );
        assert_eq!(result, Value::String(String::new()));
    }

    #[test]
    fn test_string_format() {
        use crate::plan::logical::ScalarFunction;

        // FORMAT('Hello %s, you have %s messages', 'Alice', 3)
        let result = eval_fn(
            ScalarFunction::Format,
            vec![
                Value::from("Hello %s, you have %s messages"),
                Value::from("Alice"),
                Value::Int(3),
            ],
        );
        assert_eq!(result, Value::String("Hello Alice, you have 3 messages".to_string()));
    }

    #[test]
    fn test_string_replace() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::Replace,
            vec![Value::from("hello world"), Value::from("world"), Value::from("there")],
        );
        assert_eq!(result, Value::String("hello there".to_string()));
    }

    #[test]
    fn test_string_trim_functions() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Trim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("hello".to_string()));

        let result = eval_fn(ScalarFunction::Ltrim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("hello  ".to_string()));

        let result = eval_fn(ScalarFunction::Rtrim, vec![Value::from("  hello  ")]);
        assert_eq!(result, Value::String("  hello".to_string()));
    }

    #[test]
    fn test_regexp_match() {
        use crate::plan::logical::ScalarFunction;

        // Match with capture group
        let result = eval_fn(
            ScalarFunction::RegexpMatch,
            vec![Value::from("hello123world"), Value::from(r"(\d+)")],
        );
        assert_eq!(result, Value::String("123".to_string()));

        // No match returns NULL
        let result =
            eval_fn(ScalarFunction::RegexpMatch, vec![Value::from("hello"), Value::from(r"\d+")]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_regexp_replace() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::RegexpReplace,
            vec![Value::from("hello 123 world 456"), Value::from(r"\d+"), Value::from("X")],
        );
        assert_eq!(result, Value::String("hello X world X".to_string()));
    }

    // ========== Numeric Function Tests ==========

    #[test]
    fn test_numeric_exp() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Exp, vec![Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - std::f64::consts::E).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // e^0 = 1
        let result = eval_fn(ScalarFunction::Exp, vec![Value::Int(0)]);
        assert_eq!(result, Value::Float(1.0));
    }

    #[test]
    fn test_numeric_ln() {
        use crate::plan::logical::ScalarFunction;

        // ln(e) = 1
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(std::f64::consts::E)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // ln(1) = 0
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Int(1)]);
        assert_eq!(result, Value::Float(0.0));

        // ln(0) = NULL (undefined)
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(0.0)]);
        assert_eq!(result, Value::Null);

        // ln(-1) = NULL (undefined)
        let result = eval_fn(ScalarFunction::Ln, vec![Value::Float(-1.0)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_numeric_log() {
        use crate::plan::logical::ScalarFunction;

        // LOG(10, 100) = 2
        let result = eval_fn(ScalarFunction::Log, vec![Value::Int(10), Value::Int(100)]);
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // LOG(2, 8) = 3
        let result = eval_fn(ScalarFunction::Log, vec![Value::Int(2), Value::Int(8)]);
        if let Value::Float(f) = result {
            assert!((f - 3.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_log10() {
        use crate::plan::logical::ScalarFunction;

        // log10(100) = 2
        let result = eval_fn(ScalarFunction::Log10, vec![Value::Int(100)]);
        if let Value::Float(f) = result {
            assert!((f - 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_trig() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // sin(0) = 0
        let result = eval_fn(ScalarFunction::Sin, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!(f.abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // cos(0) = 1
        let result = eval_fn(ScalarFunction::Cos, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // tan(0) = 0
        let result = eval_fn(ScalarFunction::Tan, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!(f.abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // sin(PI/2) ≈ 1
        let result = eval_fn(ScalarFunction::Sin, vec![Value::Float(PI / 2.0)]);
        if let Value::Float(f) = result {
            assert!((f - 1.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_inverse_trig() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // asin(1) = PI/2
        let result = eval_fn(ScalarFunction::Asin, vec![Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // acos(0) = PI/2
        let result = eval_fn(ScalarFunction::Acos, vec![Value::Float(0.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 2.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // atan(0) = 0
        let result = eval_fn(ScalarFunction::Atan, vec![Value::Float(0.0)]);
        assert_eq!(result, Value::Float(0.0));

        // atan2(1, 1) = PI/4
        let result = eval_fn(ScalarFunction::Atan2, vec![Value::Float(1.0), Value::Float(1.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI / 4.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // asin(2) = NULL (out of domain)
        let result = eval_fn(ScalarFunction::Asin, vec![Value::Float(2.0)]);
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_numeric_degrees_radians() {
        use crate::plan::logical::ScalarFunction;
        use std::f64::consts::PI;

        // degrees(PI) = 180
        let result = eval_fn(ScalarFunction::Degrees, vec![Value::Float(PI)]);
        if let Value::Float(f) = result {
            assert!((f - 180.0).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }

        // radians(180) = PI
        let result = eval_fn(ScalarFunction::Radians, vec![Value::Float(180.0)]);
        if let Value::Float(f) = result {
            assert!((f - PI).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_sign() {
        use crate::plan::logical::ScalarFunction;

        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(42)]), Value::Int(1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(-42)]), Value::Int(-1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Int(0)]), Value::Int(0));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Float(3.14)]), Value::Int(1));
        assert_eq!(eval_fn(ScalarFunction::Sign, vec![Value::Float(-3.14)]), Value::Int(-1));
    }

    #[test]
    fn test_numeric_pi() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(ScalarFunction::Pi, vec![]);
        assert_eq!(result, Value::Float(std::f64::consts::PI));
    }

    #[test]
    fn test_numeric_trunc() {
        use crate::plan::logical::ScalarFunction;

        // TRUNC(3.789) = 3
        let result = eval_fn(ScalarFunction::Trunc, vec![Value::Float(3.789)]);
        assert_eq!(result, Value::Float(3.0));

        // TRUNC(3.789, 2) = 3.78
        let result = eval_fn(ScalarFunction::Trunc, vec![Value::Float(3.789), Value::Int(2)]);
        if let Value::Float(f) = result {
            assert!((f - 3.78).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    #[test]
    fn test_numeric_round_with_precision() {
        use crate::plan::logical::ScalarFunction;

        // ROUND(3.14159, 2) = 3.14
        let result = eval_fn(ScalarFunction::Round, vec![Value::Float(3.14159), Value::Int(2)]);
        if let Value::Float(f) = result {
            assert!((f - 3.14).abs() < 0.0001);
        } else {
            panic!("Expected float");
        }
    }

    // ========== Date/Time Function Tests ==========

    #[test]
    fn test_date_part() {
        use crate::plan::logical::ScalarFunction;

        let ts = "2024-01-15 10:30:45";

        // year
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("year"), Value::from(ts)]);
        assert_eq!(result, Value::Float(2024.0));

        // month
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("month"), Value::from(ts)]);
        assert_eq!(result, Value::Float(1.0));

        // day
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("day"), Value::from(ts)]);
        assert_eq!(result, Value::Float(15.0));

        // hour
        let result = eval_fn(ScalarFunction::DatePart, vec![Value::from("hour"), Value::from(ts)]);
        assert_eq!(result, Value::Float(10.0));

        // minute
        let result =
            eval_fn(ScalarFunction::DatePart, vec![Value::from("minute"), Value::from(ts)]);
        assert_eq!(result, Value::Float(30.0));

        // second
        let result =
            eval_fn(ScalarFunction::DatePart, vec![Value::from("second"), Value::from(ts)]);
        assert_eq!(result, Value::Float(45.0));
    }

    #[test]
    fn test_date_trunc() {
        use crate::plan::logical::ScalarFunction;

        let ts = "2024-01-15 10:30:45";

        // trunc to year
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("year"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));

        // trunc to month
        let result =
            eval_fn(ScalarFunction::DateTrunc, vec![Value::from("month"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-01 00:00:00".to_string()));

        // trunc to day
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("day"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-15 00:00:00".to_string()));

        // trunc to hour
        let result = eval_fn(ScalarFunction::DateTrunc, vec![Value::from("hour"), Value::from(ts)]);
        assert_eq!(result, Value::String("2024-01-15 10:00:00".to_string()));
    }

    #[test]
    fn test_to_timestamp_epoch() {
        use crate::plan::logical::ScalarFunction;

        // TO_TIMESTAMP(0) = 1970-01-01 00:00:00
        let result = eval_fn(ScalarFunction::ToTimestamp, vec![Value::Int(0)]);
        assert_eq!(result, Value::String("1970-01-01 00:00:00+00".to_string()));
    }

    #[test]
    fn test_to_date() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::ToDate,
            vec![Value::from("2024-01-15"), Value::from("YYYY-MM-DD")],
        );
        assert_eq!(result, Value::String("2024-01-15".to_string()));
    }

    #[test]
    fn test_to_char() {
        use crate::plan::logical::ScalarFunction;

        let result = eval_fn(
            ScalarFunction::ToChar,
            vec![Value::from("2024-01-15 10:30:45"), Value::from("YYYY-MM-DD")],
        );
        assert_eq!(result, Value::String("2024-01-15".to_string()));
    }

    #[test]
    fn test_nullif() {
        use crate::plan::logical::ScalarFunction;

        // NULLIF(5, 5) = NULL
        let result = eval_fn(ScalarFunction::NullIf, vec![Value::Int(5), Value::Int(5)]);
        assert_eq!(result, Value::Null);

        // NULLIF(5, 3) = 5
        let result = eval_fn(ScalarFunction::NullIf, vec![Value::Int(5), Value::Int(3)]);
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_substring() {
        use crate::plan::logical::ScalarFunction;

        // SUBSTRING('hello', 2, 3) = 'ell'
        let result = eval_fn(
            ScalarFunction::Substring,
            vec![Value::from("hello"), Value::Int(2), Value::Int(3)],
        );
        assert_eq!(result, Value::String("ell".to_string()));

        // SUBSTRING('hello', 2) = 'ello' (to end)
        let result = eval_fn(ScalarFunction::Substring, vec![Value::from("hello"), Value::Int(2)]);
        assert_eq!(result, Value::String("ello".to_string()));
    }
}
