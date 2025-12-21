//! Expression AST types.
//!
//! This module defines the expression types that form the building blocks
//! of query predicates, projections, and computations.

use std::fmt;
use std::ops::Not;

/// A literal value in a query.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Null value.
    Null,
    /// Boolean value.
    Boolean(bool),
    /// 64-bit signed integer.
    Integer(i64),
    /// 64-bit floating point number.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// Vector literal (for embeddings).
    Vector(Vec<f32>),
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Boolean(b) => write!(f, "{b}"),
            Self::Integer(i) => write!(f, "{i}"),
            Self::Float(fl) => write!(f, "{fl}"),
            Self::String(s) => write!(f, "'{s}'"),
            Self::Vector(v) => {
                write!(f, "[")?;
                for (i, val) in v.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{val}")?;
                }
                write!(f, "]")
            }
        }
    }
}

/// An identifier (column name, table name, alias, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Identifier {
    /// The name of the identifier.
    pub name: String,
    /// Optional quote character used (for case-sensitive identifiers).
    pub quote_style: Option<char>,
}

impl Identifier {
    /// Creates a new unquoted identifier.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), quote_style: None }
    }

    /// Creates a new quoted identifier.
    #[must_use]
    pub fn quoted(name: impl Into<String>, quote: char) -> Self {
        Self { name: name.into(), quote_style: Some(quote) }
    }
}

impl fmt::Display for Identifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.quote_style {
            Some(q) => write!(f, "{q}{}{q}", self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

impl From<&str> for Identifier {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Identifier {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// A qualified identifier (e.g., `table.column` or `schema.table.column`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QualifiedName {
    /// The parts of the qualified name.
    pub parts: Vec<Identifier>,
}

impl QualifiedName {
    /// Creates a new qualified name from parts.
    #[must_use]
    pub const fn new(parts: Vec<Identifier>) -> Self {
        Self { parts }
    }

    /// Creates a simple (unqualified) name.
    #[must_use]
    pub fn simple(name: impl Into<Identifier>) -> Self {
        Self { parts: vec![name.into()] }
    }

    /// Creates a two-part qualified name (e.g., `table.column`).
    #[must_use]
    pub fn qualified(qualifier: impl Into<Identifier>, name: impl Into<Identifier>) -> Self {
        Self { parts: vec![qualifier.into(), name.into()] }
    }

    /// Returns the final (unqualified) name.
    #[must_use]
    pub fn name(&self) -> Option<&Identifier> {
        self.parts.last()
    }

    /// Returns the qualifier parts (everything except the final name).
    #[must_use]
    pub fn qualifiers(&self) -> &[Identifier] {
        if self.parts.is_empty() {
            &[]
        } else {
            &self.parts[..self.parts.len() - 1]
        }
    }
}

impl fmt::Display for QualifiedName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, part) in self.parts.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{part}")?;
        }
        Ok(())
    }
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    /// Addition (+).
    Add,
    /// Subtraction (-).
    Sub,
    /// Multiplication (*).
    Mul,
    /// Division (/).
    Div,
    /// Modulo (%).
    Mod,

    // Comparison
    /// Equal (=).
    Eq,
    /// Not equal (<> or !=).
    NotEq,
    /// Less than (<).
    Lt,
    /// Less than or equal (<=).
    LtEq,
    /// Greater than (>).
    Gt,
    /// Greater than or equal (>=).
    GtEq,

    // Logical
    /// Logical AND.
    And,
    /// Logical OR.
    Or,

    // String
    /// LIKE pattern matching.
    Like,
    /// NOT LIKE pattern matching.
    NotLike,
    /// ILIKE (case-insensitive LIKE).
    ILike,
    /// NOT ILIKE.
    NotILike,

    // Vector distance operators
    /// Euclidean distance (<->).
    EuclideanDistance,
    /// Cosine distance (<=>).
    CosineDistance,
    /// Inner product (<#>).
    InnerProduct,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Eq => "=",
            Self::NotEq => "<>",
            Self::Lt => "<",
            Self::LtEq => "<=",
            Self::Gt => ">",
            Self::GtEq => ">=",
            Self::And => "AND",
            Self::Or => "OR",
            Self::Like => "LIKE",
            Self::NotLike => "NOT LIKE",
            Self::ILike => "ILIKE",
            Self::NotILike => "NOT ILIKE",
            Self::EuclideanDistance => "<->",
            Self::CosineDistance => "<=>",
            Self::InnerProduct => "<#>",
        };
        write!(f, "{op}")
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Logical NOT.
    Not,
    /// Numeric negation (-).
    Neg,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            Self::Not => "NOT",
            Self::Neg => "-",
            Self::IsNull => "IS NULL",
            Self::IsNotNull => "IS NOT NULL",
        };
        write!(f, "{op}")
    }
}

/// A function call expression.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    /// The function name.
    pub name: QualifiedName,
    /// The function arguments.
    pub args: Vec<Expr>,
    /// Whether DISTINCT was specified (for aggregates).
    pub distinct: bool,
    /// Optional filter clause (for aggregates).
    pub filter: Option<Box<Expr>>,
    /// Optional OVER clause (for window functions).
    pub over: Option<WindowSpec>,
}

impl FunctionCall {
    /// Creates a new function call with the given name and arguments.
    #[must_use]
    pub fn new(name: impl Into<QualifiedName>, args: Vec<Expr>) -> Self {
        Self { name: name.into(), args, distinct: false, filter: None, over: None }
    }
}

impl From<QualifiedName> for FunctionCall {
    fn from(name: QualifiedName) -> Self {
        Self::new(name, vec![])
    }
}

/// Window specification for window functions.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    /// Partition by expressions.
    pub partition_by: Vec<Expr>,
    /// Order by expressions.
    pub order_by: Vec<OrderByExpr>,
    /// Window frame specification.
    pub frame: Option<WindowFrame>,
}

/// Window frame specification.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowFrame {
    /// Frame units (ROWS, RANGE, GROUPS).
    pub units: WindowFrameUnits,
    /// Frame start bound.
    pub start: WindowFrameBound,
    /// Frame end bound (if BETWEEN was used).
    pub end: Option<WindowFrameBound>,
}

/// Window frame units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameUnits {
    /// ROWS.
    Rows,
    /// RANGE.
    Range,
    /// GROUPS.
    Groups,
}

/// Window frame bound.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFrameBound {
    /// CURRENT ROW.
    CurrentRow,
    /// UNBOUNDED PRECEDING.
    UnboundedPreceding,
    /// UNBOUNDED FOLLOWING.
    UnboundedFollowing,
    /// N PRECEDING.
    Preceding(Box<Expr>),
    /// N FOLLOWING.
    Following(Box<Expr>),
}

/// Order by expression.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByExpr {
    /// The expression to order by.
    pub expr: Box<Expr>,
    /// Sort direction (true = ASC, false = DESC).
    pub asc: bool,
    /// NULLS FIRST or NULLS LAST.
    pub nulls_first: Option<bool>,
}

impl OrderByExpr {
    /// Creates a new ascending order by expression.
    #[must_use]
    pub fn asc(expr: Expr) -> Self {
        Self { expr: Box::new(expr), asc: true, nulls_first: None }
    }

    /// Creates a new descending order by expression.
    #[must_use]
    pub fn desc(expr: Expr) -> Self {
        Self { expr: Box::new(expr), asc: false, nulls_first: None }
    }
}

/// A CASE expression.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseExpr {
    /// The operand (for simple CASE).
    pub operand: Option<Box<Expr>>,
    /// WHEN...THEN branches.
    pub when_clauses: Vec<(Expr, Expr)>,
    /// ELSE expression.
    pub else_result: Option<Box<Expr>>,
}

/// A subquery expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Subquery {
    /// The subquery statement (must be a SELECT).
    pub query: Box<super::statement::SelectStatement>,
}

/// An expression in a query.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal value.
    Literal(Literal),

    /// A column reference.
    Column(QualifiedName),

    /// A parameter placeholder ($1, $name, ?).
    Parameter(ParameterRef),

    /// A binary operation.
    BinaryOp {
        /// Left operand.
        left: Box<Expr>,
        /// The operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expr>,
    },

    /// A unary operation.
    UnaryOp {
        /// The operator.
        op: UnaryOp,
        /// The operand.
        operand: Box<Expr>,
    },

    /// A function call.
    Function(FunctionCall),

    /// A CAST expression.
    Cast {
        /// The expression to cast.
        expr: Box<Expr>,
        /// The target type name.
        data_type: String,
    },

    /// A CASE expression.
    Case(CaseExpr),

    /// A subquery expression.
    Subquery(Subquery),

    /// EXISTS subquery.
    Exists(Subquery),

    /// IN list: expr IN (val1, val2, ...).
    InList {
        /// The expression to check.
        expr: Box<Expr>,
        /// The list of values.
        list: Vec<Expr>,
        /// Whether NOT IN.
        negated: bool,
    },

    /// IN subquery: expr IN (SELECT ...).
    InSubquery {
        /// The expression to check.
        expr: Box<Expr>,
        /// The subquery.
        subquery: Subquery,
        /// Whether NOT IN.
        negated: bool,
    },

    /// BETWEEN: expr BETWEEN low AND high.
    Between {
        /// The expression to check.
        expr: Box<Expr>,
        /// Lower bound.
        low: Box<Expr>,
        /// Upper bound.
        high: Box<Expr>,
        /// Whether NOT BETWEEN.
        negated: bool,
    },

    /// Array access: expr[index].
    ArrayIndex {
        /// The array expression.
        array: Box<Expr>,
        /// The index expression.
        index: Box<Expr>,
    },

    /// Tuple/row constructor: (expr1, expr2, ...).
    Tuple(Vec<Expr>),

    /// Wildcard (*) for SELECT *.
    Wildcard,

    /// Qualified wildcard (table.*).
    QualifiedWildcard(QualifiedName),
}

/// A parameter reference in a query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParameterRef {
    /// Positional parameter ($1, $2, ...).
    Positional(u32),
    /// Named parameter ($name).
    Named(String),
    /// Anonymous parameter (?).
    Anonymous,
}

impl fmt::Display for ParameterRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Positional(n) => write!(f, "${n}"),
            Self::Named(name) => write!(f, "${name}"),
            Self::Anonymous => write!(f, "?"),
        }
    }
}

impl Expr {
    /// Creates a literal null expression.
    #[must_use]
    pub const fn null() -> Self {
        Self::Literal(Literal::Null)
    }

    /// Creates a literal boolean expression.
    #[must_use]
    pub const fn boolean(value: bool) -> Self {
        Self::Literal(Literal::Boolean(value))
    }

    /// Creates a literal integer expression.
    #[must_use]
    pub const fn integer(value: i64) -> Self {
        Self::Literal(Literal::Integer(value))
    }

    /// Creates a literal float expression.
    #[must_use]
    pub const fn float(value: f64) -> Self {
        Self::Literal(Literal::Float(value))
    }

    /// Creates a literal string expression.
    #[must_use]
    pub fn string(value: impl Into<String>) -> Self {
        Self::Literal(Literal::String(value.into()))
    }

    /// Creates a column reference expression.
    #[must_use]
    pub fn column(name: impl Into<QualifiedName>) -> Self {
        Self::Column(name.into())
    }

    /// Creates a binary operation expression.
    #[must_use]
    pub fn binary(left: Self, op: BinaryOp, right: Self) -> Self {
        Self::BinaryOp { left: Box::new(left), op, right: Box::new(right) }
    }

    /// Creates a unary operation expression.
    #[must_use]
    pub fn unary(op: UnaryOp, operand: Self) -> Self {
        Self::UnaryOp { op, operand: Box::new(operand) }
    }

    /// Creates a function call expression.
    #[must_use]
    pub fn function(name: impl Into<QualifiedName>, args: Vec<Self>) -> Self {
        Self::Function(FunctionCall::new(name, args))
    }

    /// Creates an AND expression.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::And, other)
    }

    /// Creates an OR expression.
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::Or, other)
    }

    /// Creates a NOT expression.
    #[must_use]
    pub fn negate(self) -> Self {
        Self::unary(UnaryOp::Not, self)
    }

    /// Creates an equality expression.
    #[must_use]
    pub fn eq(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::Eq, other)
    }

    /// Creates a not-equal expression.
    #[must_use]
    pub fn not_eq(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::NotEq, other)
    }

    /// Creates a less-than expression.
    #[must_use]
    pub fn lt(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::Lt, other)
    }

    /// Creates a greater-than expression.
    #[must_use]
    pub fn gt(self, other: Self) -> Self {
        Self::binary(self, BinaryOp::Gt, other)
    }
}

impl From<i64> for Expr {
    fn from(value: i64) -> Self {
        Self::integer(value)
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Self::float(value)
    }
}

impl From<bool> for Expr {
    fn from(value: bool) -> Self {
        Self::boolean(value)
    }
}

impl From<&str> for Expr {
    fn from(value: &str) -> Self {
        Self::string(value)
    }
}

impl From<String> for Expr {
    fn from(value: String) -> Self {
        Self::string(value)
    }
}

// Allow QualifiedName to convert to Expr as a column reference
impl From<QualifiedName> for Expr {
    fn from(name: QualifiedName) -> Self {
        Self::Column(name)
    }
}

impl Not for Expr {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.negate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_display() {
        assert_eq!(Literal::Null.to_string(), "NULL");
        assert_eq!(Literal::Boolean(true).to_string(), "true");
        assert_eq!(Literal::Integer(42).to_string(), "42");
        assert_eq!(Literal::Float(3.14).to_string(), "3.14");
        assert_eq!(Literal::String("hello".into()).to_string(), "'hello'");
        assert_eq!(Literal::Vector(vec![1.0, 2.0, 3.0]).to_string(), "[1, 2, 3]");
    }

    #[test]
    fn identifier_display() {
        assert_eq!(Identifier::new("foo").to_string(), "foo");
        assert_eq!(Identifier::quoted("Foo", '"').to_string(), "\"Foo\"");
    }

    #[test]
    fn qualified_name() {
        let simple = QualifiedName::simple("column");
        assert_eq!(simple.to_string(), "column");

        let qualified = QualifiedName::qualified("table", "column");
        assert_eq!(qualified.to_string(), "table.column");

        assert_eq!(qualified.name().map(|i| i.name.as_str()), Some("column"));
        assert_eq!(qualified.qualifiers().len(), 1);
    }

    #[test]
    fn expr_builders() {
        let expr = Expr::column(QualifiedName::simple("id"))
            .eq(Expr::integer(42))
            .and(Expr::column(QualifiedName::simple("active")).eq(Expr::boolean(true)));

        match expr {
            Expr::BinaryOp { op: BinaryOp::And, .. } => (),
            _ => panic!("expected AND expression"),
        }
    }

    #[test]
    fn binary_op_display() {
        assert_eq!(BinaryOp::EuclideanDistance.to_string(), "<->");
        assert_eq!(BinaryOp::CosineDistance.to_string(), "<=>");
        assert_eq!(BinaryOp::InnerProduct.to_string(), "<#>");
    }

    #[test]
    fn parameter_display() {
        assert_eq!(ParameterRef::Positional(1).to_string(), "$1");
        assert_eq!(ParameterRef::Named("query".into()).to_string(), "$query");
        assert_eq!(ParameterRef::Anonymous.to_string(), "?");
    }
}
