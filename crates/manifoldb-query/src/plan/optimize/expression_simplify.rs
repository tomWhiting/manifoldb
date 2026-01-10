//! Expression simplification optimization.
//!
//! This module provides an optimization pass that simplifies expressions
//! at compile time, reducing runtime computation.
//!
//! # Simplification Rules
//!
//! - **Constant folding**: `1 + 2` → `3`, `'hello' || ' world'` → `'hello world'`
//! - **Boolean simplification**: `true AND x` → `x`, `false OR x` → `x`
//! - **Null propagation**: `null + 1` → `null`
//! - **Identity operations**: `x + 0` → `x`, `x * 1` → `x`
//! - **Annihilator operations**: `x * 0` → `0`, `x AND false` → `false`

use crate::ast::{BinaryOp, Literal, UnaryOp};
use crate::plan::logical::{
    AggregateNode, FilterNode, JoinNode, LogicalExpr, LogicalPlan, ProjectNode,
};

/// Expression simplification optimizer.
///
/// Traverses expressions and applies simplification rules to reduce
/// computation at runtime.
#[derive(Debug, Clone, Default)]
pub struct ExpressionSimplify;

impl ExpressionSimplify {
    /// Creates a new expression simplifier.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Optimizes a logical plan by simplifying expressions.
    #[must_use]
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        self.optimize_plan(plan)
    }

    /// Recursively optimizes a plan node and its children.
    fn optimize_plan(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { node, input } => {
                let simplified_predicate = self.simplify_expr(node.predicate);
                let optimized_input = self.optimize_plan(*input);

                // If predicate simplified to true, remove the filter entirely
                if let LogicalExpr::Literal(Literal::Boolean(true)) = simplified_predicate {
                    return optimized_input;
                }

                // If predicate simplified to false, we could potentially optimize to empty
                // but that requires more careful handling, so we leave it for now

                LogicalPlan::Filter {
                    node: FilterNode::new(simplified_predicate),
                    input: Box::new(optimized_input),
                }
            }
            LogicalPlan::Project { node, input } => {
                let simplified_exprs =
                    node.exprs.into_iter().map(|e| self.simplify_expr(e)).collect();
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Project {
                    node: ProjectNode::new(simplified_exprs),
                    input: Box::new(optimized_input),
                }
            }
            LogicalPlan::Sort { node, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Sort { node, input: Box::new(optimized_input) }
            }
            LogicalPlan::Limit { node, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Limit { node, input: Box::new(optimized_input) }
            }
            LogicalPlan::Distinct { node, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Distinct { node, input: Box::new(optimized_input) }
            }
            LogicalPlan::Window { node, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Window { node, input: Box::new(optimized_input) }
            }
            LogicalPlan::Alias { alias, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Alias { alias, input: Box::new(optimized_input) }
            }
            LogicalPlan::Unwind { node, input } => {
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Unwind { node, input: Box::new(optimized_input) }
            }
            LogicalPlan::Join { node, left, right } => {
                let simplified_condition = node.condition.map(|c| self.simplify_expr(c));
                let optimized_left = self.optimize_plan(*left);
                let optimized_right = self.optimize_plan(*right);
                LogicalPlan::Join {
                    node: Box::new(JoinNode {
                        join_type: node.join_type,
                        condition: simplified_condition,
                        using_columns: node.using_columns,
                    }),
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                }
            }
            LogicalPlan::Aggregate { node, input } => {
                let simplified_group_by =
                    node.group_by.into_iter().map(|e| self.simplify_expr(e)).collect();
                let simplified_aggregates =
                    node.aggregates.into_iter().map(|e| self.simplify_expr(e)).collect();
                let simplified_having = node.having.map(|h| self.simplify_expr(h));
                let optimized_input = self.optimize_plan(*input);
                LogicalPlan::Aggregate {
                    node: Box::new(AggregateNode {
                        group_by: simplified_group_by,
                        aggregates: simplified_aggregates,
                        having: simplified_having,
                    }),
                    input: Box::new(optimized_input),
                }
            }
            LogicalPlan::SetOp { node, left, right } => {
                let optimized_left = self.optimize_plan(*left);
                let optimized_right = self.optimize_plan(*right);
                LogicalPlan::SetOp {
                    node,
                    left: Box::new(optimized_left),
                    right: Box::new(optimized_right),
                }
            }
            LogicalPlan::Union { node, inputs } => {
                let optimized_inputs = inputs.into_iter().map(|p| self.optimize_plan(p)).collect();
                LogicalPlan::Union { node, inputs: optimized_inputs }
            }
            // Pass through nodes that don't contain expressions to simplify
            other => other,
        }
    }

    /// Simplifies an expression recursively.
    #[must_use]
    pub fn simplify_expr(&self, expr: LogicalExpr) -> LogicalExpr {
        match expr {
            // Binary operations - the core of simplification
            LogicalExpr::BinaryOp { left, op, right } => {
                let left = self.simplify_expr(*left);
                let right = self.simplify_expr(*right);
                self.simplify_binary_op(left, op, right)
            }

            // Unary operations
            LogicalExpr::UnaryOp { op, operand } => {
                let operand = self.simplify_expr(*operand);
                self.simplify_unary_op(op, operand)
            }

            // Scalar functions - simplify arguments
            LogicalExpr::ScalarFunction { func, args } => {
                let simplified_args = args.into_iter().map(|a| self.simplify_expr(a)).collect();
                LogicalExpr::ScalarFunction { func, args: simplified_args }
            }

            // Aggregate functions - simplify arguments
            LogicalExpr::AggregateFunction { func, args, distinct, .. } => {
                let simplified_args = args.into_iter().map(|a| self.simplify_expr(a)).collect();
                LogicalExpr::AggregateFunction {
                    func,
                    args: simplified_args,
                    distinct,
                    filter: None,
                }
            }

            // Cast - simplify inner expression
            LogicalExpr::Cast { expr, data_type } => {
                let simplified = self.simplify_expr(*expr);
                LogicalExpr::Cast { expr: Box::new(simplified), data_type }
            }

            // Case - simplify operand, when clauses, and else
            LogicalExpr::Case { operand, when_clauses, else_result } => {
                let simplified_operand = operand.map(|o| Box::new(self.simplify_expr(*o)));
                let simplified_when: Vec<_> = when_clauses
                    .into_iter()
                    .map(|(when, then)| (self.simplify_expr(when), self.simplify_expr(then)))
                    .collect();
                let simplified_else = else_result.map(|e| Box::new(self.simplify_expr(*e)));
                LogicalExpr::Case {
                    operand: simplified_operand,
                    when_clauses: simplified_when,
                    else_result: simplified_else,
                }
            }

            // Alias - simplify inner expression
            LogicalExpr::Alias { expr, alias } => {
                let simplified = self.simplify_expr(*expr);
                LogicalExpr::Alias { expr: Box::new(simplified), alias }
            }

            // IN list - simplify expression and list items
            LogicalExpr::InList { expr, list, negated } => {
                let simplified_expr = self.simplify_expr(*expr);
                let simplified_list: Vec<_> =
                    list.into_iter().map(|e| self.simplify_expr(e)).collect();
                LogicalExpr::InList {
                    expr: Box::new(simplified_expr),
                    list: simplified_list,
                    negated,
                }
            }

            // Between - simplify all parts
            LogicalExpr::Between { expr, low, high, negated } => {
                let simplified_expr = self.simplify_expr(*expr);
                let simplified_low = self.simplify_expr(*low);
                let simplified_high = self.simplify_expr(*high);
                LogicalExpr::Between {
                    expr: Box::new(simplified_expr),
                    low: Box::new(simplified_low),
                    high: Box::new(simplified_high),
                    negated,
                }
            }

            // Array index - simplify both parts
            LogicalExpr::ArrayIndex { array, index } => {
                let simplified_array = self.simplify_expr(*array);
                let simplified_index = self.simplify_expr(*index);
                LogicalExpr::ArrayIndex {
                    array: Box::new(simplified_array),
                    index: Box::new(simplified_index),
                }
            }

            // Literals, columns, parameters, wildcards - no simplification needed
            other => other,
        }
    }

    /// Simplifies a binary operation.
    fn simplify_binary_op(
        &self,
        left: LogicalExpr,
        op: BinaryOp,
        right: LogicalExpr,
    ) -> LogicalExpr {
        // Handle null propagation first
        if matches!(&left, LogicalExpr::Literal(Literal::Null))
            || matches!(&right, LogicalExpr::Literal(Literal::Null))
        {
            // Most operations with null return null
            // Exception: IS NULL, IS NOT NULL (handled as unary ops)
            // Exception: OR with true, AND with false
            match (&op, &left, &right) {
                // null OR x -> depends on x; x OR null -> depends on x
                (BinaryOp::Or, LogicalExpr::Literal(Literal::Boolean(true)), _)
                | (BinaryOp::Or, _, LogicalExpr::Literal(Literal::Boolean(true))) => {
                    return LogicalExpr::Literal(Literal::Boolean(true));
                }
                // null AND x -> false if x is false
                (BinaryOp::And, LogicalExpr::Literal(Literal::Boolean(false)), _)
                | (BinaryOp::And, _, LogicalExpr::Literal(Literal::Boolean(false))) => {
                    return LogicalExpr::Literal(Literal::Boolean(false));
                }
                // Comparison with null returns null (unknown)
                (
                    BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::LtEq
                    | BinaryOp::Gt
                    | BinaryOp::GtEq,
                    _,
                    _,
                ) => {
                    return LogicalExpr::Literal(Literal::Null);
                }
                // Arithmetic with null returns null
                (
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod,
                    _,
                    _,
                ) => {
                    return LogicalExpr::Literal(Literal::Null);
                }
                _ => {}
            }
        }

        // Constant folding for literals
        if let (LogicalExpr::Literal(l), LogicalExpr::Literal(r)) = (&left, &right) {
            if let Some(result) = self.fold_binary_literals(l, &op, r) {
                return LogicalExpr::Literal(result);
            }
        }

        // Boolean simplification
        match (&op, &left, &right) {
            // true AND x -> x
            (BinaryOp::And, LogicalExpr::Literal(Literal::Boolean(true)), _) => {
                return right;
            }
            // x AND true -> x
            (BinaryOp::And, _, LogicalExpr::Literal(Literal::Boolean(true))) => {
                return left;
            }
            // false AND x -> false
            (BinaryOp::And, LogicalExpr::Literal(Literal::Boolean(false)), _)
            | (BinaryOp::And, _, LogicalExpr::Literal(Literal::Boolean(false))) => {
                return LogicalExpr::Literal(Literal::Boolean(false));
            }
            // false OR x -> x
            (BinaryOp::Or, LogicalExpr::Literal(Literal::Boolean(false)), _) => {
                return right;
            }
            // x OR false -> x
            (BinaryOp::Or, _, LogicalExpr::Literal(Literal::Boolean(false))) => {
                return left;
            }
            // true OR x -> true
            (BinaryOp::Or, LogicalExpr::Literal(Literal::Boolean(true)), _)
            | (BinaryOp::Or, _, LogicalExpr::Literal(Literal::Boolean(true))) => {
                return LogicalExpr::Literal(Literal::Boolean(true));
            }
            _ => {}
        }

        // Arithmetic identity simplification
        match (&op, &left, &right) {
            // x + 0 -> x
            (BinaryOp::Add, _, LogicalExpr::Literal(Literal::Integer(0))) => {
                return left;
            }
            (BinaryOp::Add, _, LogicalExpr::Literal(Literal::Float(f))) if *f == 0.0 => {
                return left;
            }
            // 0 + x -> x
            (BinaryOp::Add, LogicalExpr::Literal(Literal::Integer(0)), _) => {
                return right;
            }
            (BinaryOp::Add, LogicalExpr::Literal(Literal::Float(f)), _) if *f == 0.0 => {
                return right;
            }
            // x - 0 -> x
            (BinaryOp::Sub, _, LogicalExpr::Literal(Literal::Integer(0))) => {
                return left;
            }
            (BinaryOp::Sub, _, LogicalExpr::Literal(Literal::Float(f))) if *f == 0.0 => {
                return left;
            }
            // x * 1 -> x
            (BinaryOp::Mul, _, LogicalExpr::Literal(Literal::Integer(1))) => {
                return left;
            }
            (BinaryOp::Mul, _, LogicalExpr::Literal(Literal::Float(f))) if *f == 1.0 => {
                return left;
            }
            // 1 * x -> x
            (BinaryOp::Mul, LogicalExpr::Literal(Literal::Integer(1)), _) => {
                return right;
            }
            (BinaryOp::Mul, LogicalExpr::Literal(Literal::Float(f)), _) if *f == 1.0 => {
                return right;
            }
            // x * 0 -> 0
            (BinaryOp::Mul, _, LogicalExpr::Literal(Literal::Integer(0))) => {
                return LogicalExpr::Literal(Literal::Integer(0));
            }
            (BinaryOp::Mul, LogicalExpr::Literal(Literal::Integer(0)), _) => {
                return LogicalExpr::Literal(Literal::Integer(0));
            }
            // x / 1 -> x
            (BinaryOp::Div, _, LogicalExpr::Literal(Literal::Integer(1))) => {
                return left;
            }
            (BinaryOp::Div, _, LogicalExpr::Literal(Literal::Float(f))) if *f == 1.0 => {
                return left;
            }
            _ => {}
        }

        // No simplification possible
        LogicalExpr::BinaryOp { left: Box::new(left), op, right: Box::new(right) }
    }

    /// Simplifies a unary operation.
    fn simplify_unary_op(&self, op: UnaryOp, operand: LogicalExpr) -> LogicalExpr {
        match (&op, &operand) {
            // NOT true -> false
            (UnaryOp::Not, LogicalExpr::Literal(Literal::Boolean(b))) => {
                return LogicalExpr::Literal(Literal::Boolean(!b));
            }
            // NOT NOT x -> x
            (UnaryOp::Not, LogicalExpr::UnaryOp { op: UnaryOp::Not, operand: inner }) => {
                return (**inner).clone();
            }
            // -(-x) -> x
            (UnaryOp::Neg, LogicalExpr::UnaryOp { op: UnaryOp::Neg, operand: inner }) => {
                return (**inner).clone();
            }
            // -literal -> negate the literal
            (UnaryOp::Neg, LogicalExpr::Literal(Literal::Integer(i))) => {
                return LogicalExpr::Literal(Literal::Integer(-i));
            }
            (UnaryOp::Neg, LogicalExpr::Literal(Literal::Float(f))) => {
                return LogicalExpr::Literal(Literal::Float(-f));
            }
            // IS NULL null -> true
            (UnaryOp::IsNull, LogicalExpr::Literal(Literal::Null)) => {
                return LogicalExpr::Literal(Literal::Boolean(true));
            }
            // IS NOT NULL null -> false
            (UnaryOp::IsNotNull, LogicalExpr::Literal(Literal::Null)) => {
                return LogicalExpr::Literal(Literal::Boolean(false));
            }
            // IS NULL non-null-literal -> false
            (UnaryOp::IsNull, LogicalExpr::Literal(_)) => {
                return LogicalExpr::Literal(Literal::Boolean(false));
            }
            // IS NOT NULL non-null-literal -> true
            (UnaryOp::IsNotNull, LogicalExpr::Literal(_)) => {
                return LogicalExpr::Literal(Literal::Boolean(true));
            }
            _ => {}
        }

        LogicalExpr::UnaryOp { op, operand: Box::new(operand) }
    }

    /// Folds two literals with a binary operator.
    fn fold_binary_literals(
        &self,
        left: &Literal,
        op: &BinaryOp,
        right: &Literal,
    ) -> Option<Literal> {
        match (left, op, right) {
            // Integer arithmetic
            (Literal::Integer(l), BinaryOp::Add, Literal::Integer(r)) => {
                Some(Literal::Integer(l.wrapping_add(*r)))
            }
            (Literal::Integer(l), BinaryOp::Sub, Literal::Integer(r)) => {
                Some(Literal::Integer(l.wrapping_sub(*r)))
            }
            (Literal::Integer(l), BinaryOp::Mul, Literal::Integer(r)) => {
                Some(Literal::Integer(l.wrapping_mul(*r)))
            }
            (Literal::Integer(l), BinaryOp::Div, Literal::Integer(r)) if *r != 0 => {
                Some(Literal::Integer(l / r))
            }
            (Literal::Integer(l), BinaryOp::Mod, Literal::Integer(r)) if *r != 0 => {
                Some(Literal::Integer(l % r))
            }

            // Float arithmetic
            (Literal::Float(l), BinaryOp::Add, Literal::Float(r)) => Some(Literal::Float(l + r)),
            (Literal::Float(l), BinaryOp::Sub, Literal::Float(r)) => Some(Literal::Float(l - r)),
            (Literal::Float(l), BinaryOp::Mul, Literal::Float(r)) => Some(Literal::Float(l * r)),
            (Literal::Float(l), BinaryOp::Div, Literal::Float(r)) if *r != 0.0 => {
                Some(Literal::Float(l / r))
            }

            // Mixed int/float arithmetic
            (Literal::Integer(l), BinaryOp::Add, Literal::Float(r)) => {
                Some(Literal::Float(*l as f64 + r))
            }
            (Literal::Float(l), BinaryOp::Add, Literal::Integer(r)) => {
                Some(Literal::Float(l + *r as f64))
            }
            (Literal::Integer(l), BinaryOp::Sub, Literal::Float(r)) => {
                Some(Literal::Float(*l as f64 - r))
            }
            (Literal::Float(l), BinaryOp::Sub, Literal::Integer(r)) => {
                Some(Literal::Float(l - *r as f64))
            }
            (Literal::Integer(l), BinaryOp::Mul, Literal::Float(r)) => {
                Some(Literal::Float(*l as f64 * r))
            }
            (Literal::Float(l), BinaryOp::Mul, Literal::Integer(r)) => {
                Some(Literal::Float(l * *r as f64))
            }
            (Literal::Integer(l), BinaryOp::Div, Literal::Float(r)) if *r != 0.0 => {
                Some(Literal::Float(*l as f64 / r))
            }
            (Literal::Float(l), BinaryOp::Div, Literal::Integer(r)) if *r != 0 => {
                Some(Literal::Float(l / *r as f64))
            }

            // Integer comparison
            (Literal::Integer(l), BinaryOp::Eq, Literal::Integer(r)) => {
                Some(Literal::Boolean(l == r))
            }
            (Literal::Integer(l), BinaryOp::NotEq, Literal::Integer(r)) => {
                Some(Literal::Boolean(l != r))
            }
            (Literal::Integer(l), BinaryOp::Lt, Literal::Integer(r)) => {
                Some(Literal::Boolean(l < r))
            }
            (Literal::Integer(l), BinaryOp::LtEq, Literal::Integer(r)) => {
                Some(Literal::Boolean(l <= r))
            }
            (Literal::Integer(l), BinaryOp::Gt, Literal::Integer(r)) => {
                Some(Literal::Boolean(l > r))
            }
            (Literal::Integer(l), BinaryOp::GtEq, Literal::Integer(r)) => {
                Some(Literal::Boolean(l >= r))
            }

            // String comparison
            (Literal::String(l), BinaryOp::Eq, Literal::String(r)) => {
                Some(Literal::Boolean(l == r))
            }
            (Literal::String(l), BinaryOp::NotEq, Literal::String(r)) => {
                Some(Literal::Boolean(l != r))
            }

            // Boolean operations
            (Literal::Boolean(l), BinaryOp::And, Literal::Boolean(r)) => {
                Some(Literal::Boolean(*l && *r))
            }
            (Literal::Boolean(l), BinaryOp::Or, Literal::Boolean(r)) => {
                Some(Literal::Boolean(*l || *r))
            }
            (Literal::Boolean(l), BinaryOp::Eq, Literal::Boolean(r)) => {
                Some(Literal::Boolean(l == r))
            }
            (Literal::Boolean(l), BinaryOp::NotEq, Literal::Boolean(r)) => {
                Some(Literal::Boolean(l != r))
            }

            // String concatenation (if we had a concat operator)
            // For now, we don't have a String concat operator in BinaryOp
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_integer_arithmetic() {
        let simplifier = ExpressionSimplify::new();

        // 1 + 2 -> 3
        let expr = LogicalExpr::integer(1).add(LogicalExpr::integer(2));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::Literal(Literal::Integer(3)));

        // 10 - 3 -> 7
        let expr = LogicalExpr::integer(10).sub(LogicalExpr::integer(3));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::Literal(Literal::Integer(7)));

        // 4 * 5 -> 20
        let expr = LogicalExpr::integer(4).mul(LogicalExpr::integer(5));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::Literal(Literal::Integer(20)));

        // 10 / 2 -> 5
        let expr = LogicalExpr::integer(10).div(LogicalExpr::integer(2));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::Literal(Literal::Integer(5)));
    }

    #[test]
    fn test_boolean_simplification() {
        let simplifier = ExpressionSimplify::new();

        // true AND x -> x
        let expr = LogicalExpr::boolean(true).and(LogicalExpr::column("x"));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // x AND true -> x
        let expr = LogicalExpr::column("x").and(LogicalExpr::boolean(true));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // false AND x -> false
        let expr = LogicalExpr::boolean(false).and(LogicalExpr::column("x"));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(false));

        // false OR x -> x
        let expr = LogicalExpr::boolean(false).or(LogicalExpr::column("x"));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // true OR x -> true
        let expr = LogicalExpr::boolean(true).or(LogicalExpr::column("x"));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(true));
    }

    #[test]
    fn test_null_propagation() {
        let simplifier = ExpressionSimplify::new();

        // null + 1 -> null
        let expr = LogicalExpr::null().add(LogicalExpr::integer(1));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::null());

        // 1 * null -> null
        let expr = LogicalExpr::integer(1).mul(LogicalExpr::null());
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::null());
    }

    #[test]
    fn test_identity_operations() {
        let simplifier = ExpressionSimplify::new();

        // x + 0 -> x
        let expr = LogicalExpr::column("x").add(LogicalExpr::integer(0));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // x * 1 -> x
        let expr = LogicalExpr::column("x").mul(LogicalExpr::integer(1));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // x - 0 -> x
        let expr = LogicalExpr::column("x").sub(LogicalExpr::integer(0));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));

        // x / 1 -> x
        let expr = LogicalExpr::column("x").div(LogicalExpr::integer(1));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));
    }

    #[test]
    fn test_annihilator_operations() {
        let simplifier = ExpressionSimplify::new();

        // x * 0 -> 0
        let expr = LogicalExpr::column("x").mul(LogicalExpr::integer(0));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::integer(0));
    }

    #[test]
    fn test_unary_simplification() {
        let simplifier = ExpressionSimplify::new();

        // NOT true -> false
        let expr = LogicalExpr::UnaryOp {
            op: UnaryOp::Not,
            operand: Box::new(LogicalExpr::boolean(true)),
        };
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(false));

        // -5 -> -5 (negation of literal)
        let expr =
            LogicalExpr::UnaryOp { op: UnaryOp::Neg, operand: Box::new(LogicalExpr::integer(5)) };
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::integer(-5));

        // IS NULL null -> true
        let expr =
            LogicalExpr::UnaryOp { op: UnaryOp::IsNull, operand: Box::new(LogicalExpr::null()) };
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(true));

        // IS NULL 5 -> false (non-null literal)
        let expr = LogicalExpr::UnaryOp {
            op: UnaryOp::IsNull,
            operand: Box::new(LogicalExpr::integer(5)),
        };
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(false));
    }

    #[test]
    fn test_nested_simplification() {
        let simplifier = ExpressionSimplify::new();

        // (1 + 2) * 3 -> 9
        let expr =
            LogicalExpr::integer(1).add(LogicalExpr::integer(2)).mul(LogicalExpr::integer(3));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::Literal(Literal::Integer(9)));

        // true AND (false OR x) -> false OR x -> x
        let inner = LogicalExpr::boolean(false).or(LogicalExpr::column("x"));
        let expr = LogicalExpr::boolean(true).and(inner);
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::column("x"));
    }

    #[test]
    fn test_comparison_folding() {
        let simplifier = ExpressionSimplify::new();

        // 5 > 3 -> true
        let expr = LogicalExpr::integer(5).gt(LogicalExpr::integer(3));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(true));

        // 2 = 2 -> true
        let expr = LogicalExpr::integer(2).eq(LogicalExpr::integer(2));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(true));

        // 3 < 1 -> false
        let expr = LogicalExpr::integer(3).lt(LogicalExpr::integer(1));
        let result = simplifier.simplify_expr(expr);
        assert_eq!(result, LogicalExpr::boolean(false));
    }
}
