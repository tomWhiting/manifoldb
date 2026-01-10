//! Expression type inference.
//!
//! This module provides type inference for logical expressions.
//! Given a type context (available schemas), it can determine the
//! output type of any expression.

use crate::ast::{BinaryOp, Literal, UnaryOp};

use super::expr::{AggregateFunction, LogicalExpr, ScalarFunction};
use super::types::{PlanType, TypeContext, TypedColumn};

/// Errors that can occur during type inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    /// Reference to an unknown column.
    UnknownColumn {
        /// The column name that was not found.
        name: String,
        /// Optional qualifier (table name).
        qualifier: Option<String>,
    },
    /// Type mismatch in an operation.
    TypeMismatch {
        /// Description of what was expected.
        expected: String,
        /// Description of what was found.
        found: String,
        /// The operation where the mismatch occurred.
        operation: String,
    },
    /// Incompatible types for an operation.
    IncompatibleTypes {
        /// Left operand type.
        left: PlanType,
        /// Right operand type.
        right: PlanType,
        /// The operation that failed.
        operation: String,
    },
    /// Unsupported operation.
    UnsupportedOperation(String),
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownColumn { name, qualifier } => {
                if let Some(q) = qualifier {
                    write!(f, "unknown column: {q}.{name}")
                } else {
                    write!(f, "unknown column: {name}")
                }
            }
            Self::TypeMismatch { expected, found, operation } => {
                write!(f, "type mismatch in {operation}: expected {expected}, found {found}")
            }
            Self::IncompatibleTypes { left, right, operation } => {
                write!(f, "incompatible types for {operation}: {left} and {right}")
            }
            Self::UnsupportedOperation(op) => {
                write!(f, "unsupported operation: {op}")
            }
        }
    }
}

impl std::error::Error for TypeError {}

/// Result type for type inference operations.
pub type TypeResult<T> = Result<T, TypeError>;

impl LogicalExpr {
    /// Infers the type of this expression given a type context.
    ///
    /// The type context provides information about available columns and their types.
    /// This method recursively infers types through the expression tree.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb_query::plan::logical::{LogicalExpr, Schema, TypeContext, TypedColumn, PlanType};
    ///
    /// let schema = Schema::new(vec![
    ///     TypedColumn::new("age", PlanType::Integer),
    ///     TypedColumn::new("name", PlanType::Text),
    /// ]);
    /// let ctx = TypeContext::with_schema(schema);
    ///
    /// let expr = LogicalExpr::column("age").gt(LogicalExpr::integer(21));
    /// let result_type = expr.infer_type(&ctx)?;
    /// assert_eq!(result_type, PlanType::Boolean);
    /// ```
    pub fn infer_type(&self, ctx: &TypeContext) -> TypeResult<PlanType> {
        match self {
            // Literals have known types
            Self::Literal(lit) => Ok(infer_literal_type(lit)),

            // Column references look up the type in the context
            Self::Column { qualifier, name } => ctx
                .lookup_column(qualifier.as_deref(), name)
                .map(|col| col.data_type.clone())
                .ok_or_else(|| TypeError::UnknownColumn {
                    name: name.clone(),
                    qualifier: qualifier.clone(),
                }),

            // Binary operations
            Self::BinaryOp { left, op, right } => {
                let left_type = left.infer_type(ctx)?;
                let right_type = right.infer_type(ctx)?;
                infer_binary_op_type(&left_type, op, &right_type)
            }

            // Unary operations
            Self::UnaryOp { op, operand } => {
                let operand_type = operand.infer_type(ctx)?;
                infer_unary_op_type(op, &operand_type)
            }

            // Scalar functions
            Self::ScalarFunction { func, args } => {
                let arg_types: Vec<_> =
                    args.iter().map(|a| a.infer_type(ctx)).collect::<TypeResult<_>>()?;
                infer_scalar_function_type(func, &arg_types)
            }

            // Aggregate functions
            Self::AggregateFunction { func, args, .. } => {
                let arg_types: Vec<_> =
                    args.iter().map(|a| a.infer_type(ctx)).collect::<TypeResult<_>>()?;
                infer_aggregate_function_type(func, &arg_types)
            }

            // CAST always returns the target type
            Self::Cast { data_type, .. } => Ok(PlanType::from(data_type)),

            // CASE returns the common type of all THEN/ELSE branches
            Self::Case { when_clauses, else_result, .. } => {
                let mut result_type = PlanType::Null;
                for (_, then_expr) in when_clauses {
                    let then_type = then_expr.infer_type(ctx)?;
                    result_type = result_type.common_type(&then_type).ok_or_else(|| {
                        TypeError::IncompatibleTypes {
                            left: result_type.clone(),
                            right: then_type,
                            operation: "CASE expression".to_string(),
                        }
                    })?;
                }
                if let Some(else_expr) = else_result {
                    let else_type = else_expr.infer_type(ctx)?;
                    result_type = result_type.common_type(&else_type).ok_or_else(|| {
                        TypeError::IncompatibleTypes {
                            left: result_type.clone(),
                            right: else_type,
                            operation: "CASE expression".to_string(),
                        }
                    })?;
                }
                Ok(result_type)
            }

            // IN list returns boolean
            Self::InList { .. } => Ok(PlanType::Boolean),

            // BETWEEN returns boolean
            Self::Between { .. } => Ok(PlanType::Boolean),

            // Scalar subquery - would need to look at subquery output schema
            // For now, return Any since we don't have the subquery schema
            Self::Subquery(_) => Ok(PlanType::Any),

            // EXISTS returns boolean
            Self::Exists { .. } => Ok(PlanType::Boolean),

            // IN subquery returns boolean
            Self::InSubquery { .. } => Ok(PlanType::Boolean),

            // Wildcard has undefined type (expanded during planning)
            Self::Wildcard | Self::QualifiedWildcard(_) => Ok(PlanType::Any),

            // Alias just wraps an expression
            Self::Alias { expr, .. } => expr.infer_type(ctx),

            // Parameter type is unknown without bind information
            Self::Parameter(_) => Ok(PlanType::Any),

            // Hybrid search returns a float (distance score)
            Self::HybridSearch { .. } => Ok(PlanType::DoublePrecision),

            // Window functions
            Self::WindowFunction { func, arg, .. } => {
                use crate::ast::WindowFunction;
                match func {
                    // Ranking functions return BigInt
                    WindowFunction::RowNumber
                    | WindowFunction::Rank
                    | WindowFunction::DenseRank
                    | WindowFunction::Ntile { .. } => Ok(PlanType::BigInt),

                    // Distribution functions return DoublePrecision
                    WindowFunction::PercentRank | WindowFunction::CumeDist => {
                        Ok(PlanType::DoublePrecision)
                    }

                    // Value functions return the type of the input expression
                    WindowFunction::Lag { .. }
                    | WindowFunction::Lead { .. }
                    | WindowFunction::FirstValue
                    | WindowFunction::LastValue
                    | WindowFunction::NthValue { .. } => {
                        if let Some(a) = arg {
                            a.infer_type(ctx)
                        } else {
                            Ok(PlanType::Any)
                        }
                    }

                    // Aggregate window functions delegate to aggregate type inference
                    WindowFunction::Aggregate(agg_func) => {
                        let agg = match agg_func {
                            crate::ast::AggregateWindowFunction::Sum => AggregateFunction::Sum,
                            crate::ast::AggregateWindowFunction::Avg => AggregateFunction::Avg,
                            crate::ast::AggregateWindowFunction::Count => AggregateFunction::Count,
                            crate::ast::AggregateWindowFunction::Min => AggregateFunction::Min,
                            crate::ast::AggregateWindowFunction::Max => AggregateFunction::Max,
                        };
                        let arg_types =
                            if let Some(a) = arg { vec![a.infer_type(ctx)?] } else { vec![] };
                        infer_aggregate_function_type(&agg, &arg_types)
                    }
                }
            }

            // List comprehension returns a list of the transform expression type
            Self::ListComprehension { transform_expr, list_expr, .. } => {
                let element_type = if let Some(transform) = transform_expr {
                    transform.infer_type(ctx)?
                } else {
                    // If no transform, return the list element type
                    let list_type = list_expr.infer_type(ctx)?;
                    list_type.element_type().cloned().unwrap_or(PlanType::Any)
                };
                Ok(PlanType::List(Box::new(element_type)))
            }

            // List literal returns list of the common element type
            Self::ListLiteral(exprs) => {
                if exprs.is_empty() {
                    return Ok(PlanType::List(Box::new(PlanType::Any)));
                }
                let first_type = exprs[0].infer_type(ctx)?;
                let mut common = first_type;
                for expr in exprs.iter().skip(1) {
                    let t = expr.infer_type(ctx)?;
                    common = common.common_type(&t).unwrap_or(PlanType::Any);
                }
                Ok(PlanType::List(Box::new(common)))
            }

            // List predicates return boolean
            Self::ListPredicateAll { .. }
            | Self::ListPredicateAny { .. }
            | Self::ListPredicateNone { .. }
            | Self::ListPredicateSingle { .. } => Ok(PlanType::Boolean),

            // List reduce returns the type of the accumulator expression
            Self::ListReduce { expression, .. } => expression.infer_type(ctx),

            // Map projection returns a map
            Self::MapProjection { .. } => {
                Ok(PlanType::Map { key: Box::new(PlanType::Text), value: Box::new(PlanType::Any) })
            }

            // Pattern comprehension returns a list
            Self::PatternComprehension { projection_expr, .. } => {
                let element_type = projection_expr.infer_type(ctx)?;
                Ok(PlanType::List(Box::new(element_type)))
            }

            // EXISTS subquery returns boolean
            Self::ExistsSubquery { .. } => Ok(PlanType::Boolean),

            // COUNT subquery returns bigint
            Self::CountSubquery { .. } => Ok(PlanType::BigInt),

            // CALL subquery - type depends on the inner plan's output
            Self::CallSubquery { .. } => Ok(PlanType::Any),
        }
    }

    /// Infers the output name (alias or derived name) for this expression.
    ///
    /// This is used when determining column names in projections.
    pub fn infer_name(&self) -> String {
        match self {
            Self::Column { name, .. } => name.clone(),
            Self::Alias { alias, .. } => alias.clone(),
            Self::Literal(Literal::Null) => "NULL".to_string(),
            Self::Literal(Literal::Boolean(_)) => "bool".to_string(),
            Self::Literal(Literal::Integer(_)) => "int".to_string(),
            Self::Literal(Literal::Float(_)) => "float".to_string(),
            Self::Literal(Literal::String(_)) => "string".to_string(),
            Self::Literal(Literal::Vector(_)) => "vector".to_string(),
            Self::Literal(Literal::MultiVector(_)) => "multivector".to_string(),
            Self::BinaryOp { .. } => "expr".to_string(),
            Self::UnaryOp { operand, .. } => operand.infer_name(),
            Self::ScalarFunction { func, .. } => format!("{func}").to_lowercase(),
            Self::AggregateFunction { func, .. } => format!("{func}").to_lowercase(),
            Self::Cast { expr, data_type } => {
                format!("cast_{}_as_{:?}", expr.infer_name(), data_type).to_lowercase()
            }
            Self::Case { .. } => "case".to_string(),
            Self::InList { expr, .. } => expr.infer_name(),
            Self::Between { expr, .. } => expr.infer_name(),
            Self::Subquery(_) => "subquery".to_string(),
            Self::Exists { .. } => "exists".to_string(),
            Self::InSubquery { expr, .. } => expr.infer_name(),
            Self::Wildcard => "*".to_string(),
            Self::QualifiedWildcard(q) => format!("{q}.*"),
            Self::Parameter(n) => format!("${n}"),
            Self::HybridSearch { .. } => "hybrid_score".to_string(),
            Self::WindowFunction { func, .. } => format!("{func:?}").to_lowercase(),
            Self::ListComprehension { .. } => "list".to_string(),
            Self::ListLiteral(_) => "list".to_string(),
            Self::ListPredicateAll { .. } => "all".to_string(),
            Self::ListPredicateAny { .. } => "any".to_string(),
            Self::ListPredicateNone { .. } => "none".to_string(),
            Self::ListPredicateSingle { .. } => "single".to_string(),
            Self::ListReduce { .. } => "reduce".to_string(),
            Self::MapProjection { .. } => "map".to_string(),
            Self::PatternComprehension { .. } => "pattern".to_string(),
            Self::ExistsSubquery { .. } => "exists".to_string(),
            Self::CountSubquery { .. } => "count".to_string(),
            Self::CallSubquery { .. } => "call".to_string(),
        }
    }

    /// Converts this expression to a typed column definition.
    ///
    /// Useful for building output schemas from projection expressions.
    pub fn to_typed_column(&self, ctx: &TypeContext) -> TypeResult<TypedColumn> {
        let name = self.infer_name();
        let data_type = self.infer_type(ctx)?;
        Ok(TypedColumn::new(name, data_type))
    }
}

/// Infers the type of a literal value.
fn infer_literal_type(lit: &Literal) -> PlanType {
    match lit {
        Literal::Null => PlanType::Null,
        Literal::Boolean(_) => PlanType::Boolean,
        Literal::Integer(_) => PlanType::BigInt,
        Literal::Float(_) => PlanType::DoublePrecision,
        Literal::String(_) => PlanType::Text,
        Literal::Vector(v) => PlanType::Vector(Some(v.len() as u32)),
        Literal::MultiVector(_) => PlanType::Any, // MultiVector is a specialized type
    }
}

/// Infers the result type of a binary operation.
fn infer_binary_op_type(left: &PlanType, op: &BinaryOp, right: &PlanType) -> TypeResult<PlanType> {
    match op {
        // Arithmetic operations return the common numeric type
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
            // String concatenation with +
            if matches!(op, BinaryOp::Add) && (left.is_string() || right.is_string()) {
                return Ok(PlanType::Text);
            }
            // Interval arithmetic
            if (matches!(left, PlanType::Interval) || matches!(right, PlanType::Interval))
                && (left.is_temporal() || right.is_temporal())
            {
                // Temporal + Interval = Temporal type
                let temporal = if left.is_temporal() { left.clone() } else { right.clone() };
                return Ok(temporal);
            }
            // Numeric arithmetic
            left.common_type(right).ok_or_else(|| TypeError::IncompatibleTypes {
                left: left.clone(),
                right: right.clone(),
                operation: format!("{op}"),
            })
        }

        // Comparison operations return boolean
        BinaryOp::Eq
        | BinaryOp::NotEq
        | BinaryOp::Lt
        | BinaryOp::LtEq
        | BinaryOp::Gt
        | BinaryOp::GtEq => {
            if !left.is_comparable_to(right) {
                return Err(TypeError::IncompatibleTypes {
                    left: left.clone(),
                    right: right.clone(),
                    operation: format!("{op}"),
                });
            }
            Ok(PlanType::Boolean)
        }

        // Logical operations return boolean (require boolean inputs)
        BinaryOp::And | BinaryOp::Or => Ok(PlanType::Boolean),

        // String pattern matching returns boolean
        BinaryOp::Like | BinaryOp::NotLike | BinaryOp::ILike | BinaryOp::NotILike => {
            Ok(PlanType::Boolean)
        }

        // Vector distance operations return double precision
        BinaryOp::EuclideanDistance
        | BinaryOp::CosineDistance
        | BinaryOp::InnerProduct
        | BinaryOp::MaxSim => Ok(PlanType::DoublePrecision),
    }
}

/// Infers the result type of a unary operation.
fn infer_unary_op_type(op: &UnaryOp, operand_type: &PlanType) -> TypeResult<PlanType> {
    match op {
        // NOT returns boolean
        UnaryOp::Not => Ok(PlanType::Boolean),
        // Negation preserves the type
        UnaryOp::Neg => Ok(operand_type.clone()),
        // IS NULL / IS NOT NULL return boolean
        UnaryOp::IsNull | UnaryOp::IsNotNull => Ok(PlanType::Boolean),
    }
}

/// Infers the result type of a scalar function.
#[allow(clippy::enum_glob_use)]
fn infer_scalar_function_type(
    func: &ScalarFunction,
    arg_types: &[PlanType],
) -> TypeResult<PlanType> {
    use ScalarFunction::*;

    match func {
        // String functions returning strings
        Upper | Lower | Trim | Ltrim | Rtrim | Replace | Concat | ConcatWs | Substring
        | SplitPart | Format | Lpad | Rpad | Left | Right | RegexpReplace => Ok(PlanType::Text),

        // String functions returning integers
        Length | Position => Ok(PlanType::BigInt),

        // RegexpMatch returns array of text
        RegexpMatch => Ok(PlanType::Array(Box::new(PlanType::Text))),

        // Coalesce returns the common type of all arguments
        Coalesce => {
            if arg_types.is_empty() {
                return Ok(PlanType::Null);
            }
            let mut result = arg_types[0].clone();
            for t in arg_types.iter().skip(1) {
                result = result.common_type(t).unwrap_or(PlanType::Any);
            }
            Ok(result)
        }

        // NullIf returns the type of the first argument
        NullIf => Ok(arg_types.first().cloned().unwrap_or(PlanType::Any)),

        // Numeric functions returning the input numeric type
        Abs | Ceil | Floor | Round | Trunc | Sign => {
            Ok(arg_types.first().cloned().unwrap_or(PlanType::DoublePrecision))
        }

        // Numeric functions always returning double
        Sqrt | Power | Exp | Ln | Log | Log10 | Sin | Cos | Tan | Asin | Acos | Atan | Atan2
        | Degrees | Radians | Pi | Random => Ok(PlanType::DoublePrecision),

        // Date/time functions
        Now | CypherDatetime | CypherLocalDatetime | MakeTimestamp | Timezone => {
            Ok(PlanType::Timestamp)
        }
        CurrentDate | CypherDate | MakeDate | ToDate => Ok(PlanType::Date),
        CurrentTime | CypherTime | CypherLocalTime | MakeTime => Ok(PlanType::Time),
        CypherDuration => Ok(PlanType::Interval),
        DateTrunc | CypherDatetimeTruncate | Age | DateAdd | DateSubtract => {
            // Usually returns the same temporal type as input
            Ok(arg_types.get(1).cloned().unwrap_or(PlanType::Timestamp))
        }
        Extract | DatePart => Ok(PlanType::DoublePrecision),
        ToTimestamp => Ok(PlanType::Timestamp),
        ToChar => Ok(PlanType::Text),

        // Vector functions
        VectorDimension => Ok(PlanType::Integer),
        VectorNorm => Ok(PlanType::DoublePrecision),

        // List/array functions
        Range => Ok(PlanType::List(Box::new(PlanType::BigInt))),
        Size => Ok(PlanType::BigInt),
        Head | Last => {
            // Returns the element type of the list
            arg_types.first().and_then(|t| t.element_type()).cloned().map_or(Ok(PlanType::Any), Ok)
        }
        Tail | Reverse => {
            // Returns a list of the same type
            Ok(arg_types
                .first()
                .cloned()
                .unwrap_or_else(|| PlanType::List(Box::new(PlanType::Any))))
        }
        ArrayLength | Cardinality | ArrayPosition => Ok(PlanType::BigInt),
        ArrayPositions => Ok(PlanType::Array(Box::new(PlanType::BigInt))),
        ArrayAppend | ArrayPrepend | ArrayCat | ArrayRemove | ArrayReplace => Ok(arg_types
            .first()
            .cloned()
            .unwrap_or_else(|| PlanType::Array(Box::new(PlanType::Any)))),
        Unnest => {
            // Returns the element type
            arg_types.first().and_then(|t| t.element_type()).cloned().map_or(Ok(PlanType::Any), Ok)
        }

        // JSON functions
        JsonExtractPath | JsonbExtractPath => Ok(PlanType::Json),
        JsonExtractPathText | JsonbExtractPathText => Ok(PlanType::Text),
        JsonBuildObject | JsonbBuildObject => Ok(PlanType::Jsonb),
        JsonBuildArray | JsonbBuildArray => Ok(PlanType::Jsonb),
        JsonbSet | JsonbInsert | JsonbStripNulls => Ok(PlanType::Jsonb),
        JsonExtractPathOp => Ok(PlanType::Json),
        JsonExtractPathTextOp => Ok(PlanType::Text),
        JsonContainsKey | JsonContainsAnyKey | JsonContainsAllKeys => Ok(PlanType::Boolean),
        JsonEach | JsonbEach | JsonEachText | JsonbEachText => {
            // Set-returning functions - type depends on context
            Ok(PlanType::Any)
        }
        JsonArrayElements | JsonbArrayElements => Ok(PlanType::Json),
        JsonArrayElementsText | JsonbArrayElementsText => Ok(PlanType::Text),
        JsonObjectKeys | JsonbObjectKeys => Ok(PlanType::Text),
        JsonbPathExists => Ok(PlanType::Boolean),
        JsonbPathQuery | JsonbPathQueryArray | JsonbPathQueryFirst => Ok(PlanType::Jsonb),

        // Cypher entity functions
        Type | CypherToString => Ok(PlanType::Text),
        Labels | Keys => Ok(PlanType::List(Box::new(PlanType::Text))),
        Id => Ok(PlanType::BigInt),
        Properties => {
            Ok(PlanType::Map { key: Box::new(PlanType::Text), value: Box::new(PlanType::Any) })
        }

        // Cypher path functions
        Nodes | Relationships => Ok(PlanType::List(Box::new(PlanType::Any))),
        StartNode | EndNode => Ok(PlanType::Node),
        PathLength => Ok(PlanType::BigInt),

        // Cypher type conversion
        ToBoolean => Ok(PlanType::Boolean),
        ToInteger => Ok(PlanType::BigInt),
        ToFloat => Ok(PlanType::DoublePrecision),

        // Spatial functions
        Point => Ok(PlanType::Custom("POINT".to_string())),
        PointDistance => Ok(PlanType::DoublePrecision),
        PointWithinBBox => Ok(PlanType::Boolean),

        // Custom functions - unknown type
        Custom(_) => Ok(PlanType::Any),
    }
}

/// Infers the result type of an aggregate function.
fn infer_aggregate_function_type(
    func: &AggregateFunction,
    arg_types: &[PlanType],
) -> TypeResult<PlanType> {
    match func {
        // COUNT always returns BigInt
        AggregateFunction::Count => Ok(PlanType::BigInt),

        // SUM returns the input numeric type (or double for floats)
        AggregateFunction::Sum => {
            let input = arg_types.first().cloned().unwrap_or(PlanType::BigInt);
            if matches!(input, PlanType::Real | PlanType::DoublePrecision) {
                Ok(PlanType::DoublePrecision)
            } else if input.is_numeric() {
                // Integer sums might overflow, so use BigInt
                Ok(PlanType::Numeric { precision: None, scale: None })
            } else {
                Ok(input)
            }
        }

        // AVG always returns double precision
        AggregateFunction::Avg => Ok(PlanType::DoublePrecision),

        // MIN/MAX preserve the input type
        AggregateFunction::Min | AggregateFunction::Max => {
            Ok(arg_types.first().cloned().unwrap_or(PlanType::Any))
        }

        // ARRAY_AGG returns an array of the input type
        AggregateFunction::ArrayAgg => {
            let element = arg_types.first().cloned().unwrap_or(PlanType::Any);
            Ok(PlanType::Array(Box::new(element)))
        }

        // STRING_AGG returns text
        AggregateFunction::StringAgg => Ok(PlanType::Text),

        // Statistical functions return double
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarianceSamp
        | AggregateFunction::VariancePop
        | AggregateFunction::PercentileCont => Ok(PlanType::DoublePrecision),

        // PERCENTILE_DISC returns the input type
        AggregateFunction::PercentileDisc => {
            // Second argument is the value expression
            Ok(arg_types.get(1).cloned().unwrap_or(PlanType::Any))
        }

        // JSON aggregates
        AggregateFunction::JsonAgg | AggregateFunction::JsonbAgg => Ok(PlanType::Jsonb),
        AggregateFunction::JsonObjectAgg | AggregateFunction::JsonbObjectAgg => Ok(PlanType::Jsonb),

        // Vector aggregates
        AggregateFunction::VectorAvg | AggregateFunction::VectorCentroid => {
            Ok(arg_types.first().cloned().unwrap_or(PlanType::Vector(None)))
        }

        // Boolean aggregates
        AggregateFunction::BoolAnd | AggregateFunction::BoolOr => Ok(PlanType::Boolean),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::types::Schema;

    fn test_context() -> TypeContext {
        let schema = Schema::new(vec![
            TypedColumn::new("id", PlanType::BigInt),
            TypedColumn::new("name", PlanType::Text),
            TypedColumn::new("age", PlanType::Integer),
            TypedColumn::new("salary", PlanType::DoublePrecision),
            TypedColumn::new("active", PlanType::Boolean),
            TypedColumn::new("embedding", PlanType::Vector(Some(384))),
        ]);
        TypeContext::with_schema(schema)
    }

    #[test]
    fn test_literal_types() {
        let ctx = TypeContext::new();

        assert_eq!(LogicalExpr::null().infer_type(&ctx).unwrap(), PlanType::Null);
        assert_eq!(LogicalExpr::boolean(true).infer_type(&ctx).unwrap(), PlanType::Boolean);
        assert_eq!(LogicalExpr::integer(42).infer_type(&ctx).unwrap(), PlanType::BigInt);
        assert_eq!(LogicalExpr::float(3.14).infer_type(&ctx).unwrap(), PlanType::DoublePrecision);
        assert_eq!(LogicalExpr::string("hello").infer_type(&ctx).unwrap(), PlanType::Text);
    }

    #[test]
    fn test_column_types() {
        let ctx = test_context();

        assert_eq!(LogicalExpr::column("id").infer_type(&ctx).unwrap(), PlanType::BigInt);
        assert_eq!(LogicalExpr::column("name").infer_type(&ctx).unwrap(), PlanType::Text);
        assert_eq!(LogicalExpr::column("age").infer_type(&ctx).unwrap(), PlanType::Integer);
    }

    #[test]
    fn test_unknown_column() {
        let ctx = test_context();

        let result = LogicalExpr::column("nonexistent").infer_type(&ctx);
        assert!(matches!(result, Err(TypeError::UnknownColumn { .. })));
    }

    #[test]
    fn test_comparison_types() {
        let ctx = test_context();

        let expr = LogicalExpr::column("age").gt(LogicalExpr::integer(21));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Boolean);

        let expr = LogicalExpr::column("name").eq(LogicalExpr::string("Alice"));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Boolean);
    }

    #[test]
    fn test_arithmetic_types() {
        let ctx = test_context();

        // Integer + Integer = Integer (promoted to BigInt for safety)
        let expr = LogicalExpr::column("age").add(LogicalExpr::integer(1));
        let result = expr.infer_type(&ctx).unwrap();
        assert!(result.is_numeric());

        // Double + Integer = Double
        let expr = LogicalExpr::column("salary").mul(LogicalExpr::float(1.1));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::DoublePrecision);
    }

    #[test]
    fn test_aggregate_types() {
        let ctx = test_context();

        let expr = LogicalExpr::count(LogicalExpr::wildcard(), false);
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::BigInt);

        let expr = LogicalExpr::avg(LogicalExpr::column("salary"), false);
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::DoublePrecision);

        let expr = LogicalExpr::max(LogicalExpr::column("age"));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Integer);
    }

    #[test]
    fn test_logical_types() {
        let ctx = test_context();

        let expr = LogicalExpr::column("active")
            .and(LogicalExpr::column("age").gt(LogicalExpr::integer(18)));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Boolean);
    }

    #[test]
    fn test_vector_distance_types() {
        let ctx = test_context();

        let expr =
            LogicalExpr::column("embedding").cosine_distance(LogicalExpr::vector(vec![0.1; 384]));
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::DoublePrecision);
    }

    #[test]
    fn test_cast_type() {
        let ctx = test_context();
        use crate::ast::DataType;

        let expr = LogicalExpr::column("age").cast(DataType::Text);
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Text);
    }

    #[test]
    fn test_alias_preserves_type() {
        let ctx = test_context();

        let expr = LogicalExpr::column("name").alias("user_name");
        assert_eq!(expr.infer_type(&ctx).unwrap(), PlanType::Text);
    }

    #[test]
    fn test_infer_name() {
        assert_eq!(LogicalExpr::column("id").infer_name(), "id");
        assert_eq!(LogicalExpr::column("name").alias("n").infer_name(), "n");
        assert_eq!(LogicalExpr::count(LogicalExpr::wildcard(), false).infer_name(), "count");
    }
}
