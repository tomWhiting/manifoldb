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
    /// Multi-vector literal (for ColBERT-style token embeddings).
    ///
    /// This represents a collection of vectors, typically used with MaxSim (<##>) operator.
    /// Example: `[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]`
    MultiVector(Vec<Vec<f32>>),
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
            Self::MultiVector(vecs) => {
                write!(f, "[")?;
                for (i, vec) in vecs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for (j, val) in vec.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{val}")?;
                    }
                    write!(f, "]")?;
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
    /// MaxSim distance for ColBERT-style multi-vectors (<##>).
    MaxSim,
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
            Self::MaxSim => "<##>",
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

/// Window function types for ranking and value functions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WindowFunction {
    /// ROW_NUMBER() - assigns sequential numbers starting from 1.
    RowNumber,
    /// RANK() - assigns ranks with gaps for ties.
    Rank,
    /// DENSE_RANK() - assigns ranks without gaps.
    DenseRank,
    /// LAG(expr, offset, default) - access value from previous row.
    Lag {
        /// Offset from current row (default 1).
        offset: u64,
        /// Whether a default value was specified.
        has_default: bool,
    },
    /// LEAD(expr, offset, default) - access value from next row.
    Lead {
        /// Offset from current row (default 1).
        offset: u64,
        /// Whether a default value was specified.
        has_default: bool,
    },
    /// FIRST_VALUE(expr) - first value in window frame.
    FirstValue,
    /// LAST_VALUE(expr) - last value in window frame.
    LastValue,
    /// NTH_VALUE(expr, n) - nth value in window frame (1-indexed).
    NthValue {
        /// The 1-indexed position (n) in the frame.
        n: u64,
    },
    /// NTILE(n) - divides rows into n buckets (1 to n).
    ///
    /// The buckets are as equal in size as possible. If the number of rows
    /// doesn't divide evenly, earlier buckets get one extra row.
    Ntile {
        /// The number of buckets to divide rows into.
        n: u64,
    },
    /// PERCENT_RANK() - relative rank as percentage (0 to 1).
    ///
    /// Formula: (rank - 1) / (total_rows - 1)
    /// Returns 0 for the first row in each partition.
    PercentRank,
    /// CUME_DIST() - cumulative distribution (fraction of rows ≤ current).
    ///
    /// Formula: rows_up_to_current / total_rows
    /// Returns a value between 0 and 1 (exclusive of 0, inclusive of 1).
    CumeDist,
    /// Aggregate functions used as window functions.
    /// Example: `SUM(amount) OVER (ORDER BY date)` for running totals.
    Aggregate(AggregateWindowFunction),
}

/// Aggregate functions that can be used as window functions.
///
/// These functions compute aggregates over the window frame, enabling
/// common patterns like running totals, moving averages, and cumulative counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateWindowFunction {
    /// COUNT(*) or COUNT(expr) over window.
    Count,
    /// SUM(expr) over window - for running totals.
    Sum,
    /// AVG(expr) over window - for moving averages.
    Avg,
    /// MIN(expr) over window - cumulative minimum.
    Min,
    /// MAX(expr) over window - cumulative maximum.
    Max,
}

impl std::fmt::Display for WindowFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RowNumber => write!(f, "ROW_NUMBER"),
            Self::Rank => write!(f, "RANK"),
            Self::DenseRank => write!(f, "DENSE_RANK"),
            Self::Lag { offset, .. } => write!(f, "LAG({offset})"),
            Self::Lead { offset, .. } => write!(f, "LEAD({offset})"),
            Self::FirstValue => write!(f, "FIRST_VALUE"),
            Self::LastValue => write!(f, "LAST_VALUE"),
            Self::NthValue { n } => write!(f, "NTH_VALUE({n})"),
            Self::Ntile { n } => write!(f, "NTILE({n})"),
            Self::PercentRank => write!(f, "PERCENT_RANK"),
            Self::CumeDist => write!(f, "CUME_DIST"),
            Self::Aggregate(agg) => write!(f, "{agg}"),
        }
    }
}

impl std::fmt::Display for AggregateWindowFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Count => "COUNT",
            Self::Sum => "SUM",
            Self::Avg => "AVG",
            Self::Min => "MIN",
            Self::Max => "MAX",
        };
        write!(f, "{name}")
    }
}

/// Window specification for window functions.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowSpec {
    /// Reference to a named window (from WINDOW clause).
    /// When set, the window inherits definition from this named window.
    pub window_name: Option<Identifier>,
    /// Partition by expressions.
    pub partition_by: Vec<Expr>,
    /// Order by expressions.
    pub order_by: Vec<OrderByExpr>,
    /// Window frame specification.
    pub frame: Option<WindowFrame>,
}

/// A named window definition from the WINDOW clause.
///
/// Example: `WINDOW w AS (PARTITION BY department ORDER BY hire_date)`
#[derive(Debug, Clone, PartialEq)]
pub struct NamedWindowDefinition {
    /// The window name.
    pub name: Identifier,
    /// Reference to another named window (window inheritance).
    pub base_window: Option<Identifier>,
    /// The window specification.
    pub spec: WindowSpec,
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
    /// Frame exclusion (EXCLUDE CURRENT ROW, etc.).
    pub exclusion: Option<WindowFrameExclusion>,
}

/// Window frame exclusion clause.
///
/// Specifies which rows to exclude from the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFrameExclusion {
    /// EXCLUDE CURRENT ROW - exclude the current row from the frame.
    CurrentRow,
    /// EXCLUDE GROUP - exclude the current row and its ordering peers.
    Group,
    /// EXCLUDE TIES - exclude ordering peers but not the current row.
    Ties,
    /// EXCLUDE NO OTHERS - don't exclude any rows (default).
    NoOthers,
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

    /// EXISTS or NOT EXISTS subquery.
    Exists {
        /// The subquery to check for existence.
        subquery: Subquery,
        /// Whether NOT EXISTS (true) or EXISTS (false).
        negated: bool,
    },

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

    /// Array access: `expr[index]`.
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

    /// Hybrid vector search expression.
    ///
    /// Combines multiple vector distance operations with weights.
    /// Example: `HYBRID(dense <=> $q1, 0.7, sparse <#> $q2, 0.3)`
    HybridSearch {
        /// Vector search components (each has distance expr and weight).
        components: Vec<HybridSearchComponent>,
        /// Combination method (WeightedSum, RRF).
        method: HybridCombinationMethod,
    },

    /// Cypher list comprehension expression.
    ///
    /// Syntax: `[x IN list WHERE predicate | expression]`
    ///
    /// Examples:
    /// - Filter and transform: `[x IN range(1,10) WHERE x % 2 = 0 | x * x]`
    /// - Just filter: `[x IN names WHERE size(x) > 5]`
    /// - Just transform: `[x IN numbers | x * 2]`
    ListComprehension {
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Optional WHERE filter predicate.
        filter_predicate: Option<Box<Expr>>,
        /// Optional transform expression (after `|`). If None, returns the variable.
        transform_expr: Option<Box<Expr>>,
    },

    /// A list literal expression: `[expr1, expr2, ...]`.
    ListLiteral(Vec<Expr>),

    /// Cypher list predicate function: `all(variable IN list WHERE predicate)`.
    ///
    /// Returns true if ALL elements in the list satisfy the predicate.
    ///
    /// Examples:
    /// - `all(x IN [1, 2, 3] WHERE x > 0)` → true
    /// - `all(x IN [1, -2, 3] WHERE x > 0)` → false
    ListPredicateAll {
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<Expr>,
    },

    /// Cypher list predicate function: `any(variable IN list WHERE predicate)`.
    ///
    /// Returns true if ANY element in the list satisfies the predicate.
    ///
    /// Examples:
    /// - `any(x IN [1, 2, 3] WHERE x > 2)` → true
    /// - `any(x IN [1, 2, 3] WHERE x > 5)` → false
    ListPredicateAny {
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<Expr>,
    },

    /// Cypher list predicate function: `none(variable IN list WHERE predicate)`.
    ///
    /// Returns true if NO elements in the list satisfy the predicate.
    ///
    /// Examples:
    /// - `none(x IN [1, 2, 3] WHERE x < 0)` → true
    /// - `none(x IN [1, 2, 3] WHERE x > 2)` → false
    ListPredicateNone {
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<Expr>,
    },

    /// Cypher list predicate function: `single(variable IN list WHERE predicate)`.
    ///
    /// Returns true if EXACTLY ONE element in the list satisfies the predicate.
    ///
    /// Examples:
    /// - `single(x IN [1, 2, 3] WHERE x = 2)` → true
    /// - `single(x IN [1, 2, 2] WHERE x = 2)` → false (two matches)
    /// - `single(x IN [1, 3, 5] WHERE x = 2)` → false (no matches)
    ListPredicateSingle {
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<Expr>,
    },

    /// Cypher reduce function: `reduce(accumulator = initial, variable IN list | expression)`.
    ///
    /// Performs a fold/reduce operation over a list, accumulating a result.
    ///
    /// Examples:
    /// - `reduce(sum = 0, x IN [1, 2, 3] | sum + x)` → 6
    /// - `reduce(product = 1, x IN [2, 3, 4] | product * x)` → 24
    /// - `reduce(s = '', x IN ['a', 'b', 'c'] | s + x)` → 'abc'
    ListReduce {
        /// Accumulator variable name.
        accumulator: Identifier,
        /// Initial value for the accumulator.
        initial: Box<Expr>,
        /// Variable name for iteration.
        variable: Identifier,
        /// List expression to iterate over.
        list_expr: Box<Expr>,
        /// Expression to compute new accumulator value (can reference both accumulator and variable).
        expression: Box<Expr>,
    },

    /// Cypher map projection expression.
    ///
    /// Syntax: `node{.property1, .property2, key: expression, .*}`
    ///
    /// Examples:
    /// - Extract specific properties: `p{.name, .age}`
    /// - Add computed property: `p{.name, fullName: p.firstName + ' ' + p.lastName}`
    /// - All properties with override: `p{.*, age: p.birthYear - 2024}`
    MapProjection {
        /// Source expression (typically a node or relationship variable).
        source: Box<Expr>,
        /// List of projection items.
        items: Vec<MapProjectionItem>,
    },

    /// Cypher pattern comprehension expression.
    ///
    /// Pattern comprehensions allow inline pattern matching within expressions,
    /// producing a list of values for each match of the pattern.
    ///
    /// Syntax: `[(pattern) WHERE predicate | expression]`
    ///
    /// Examples:
    /// - Get names of friends: `[(p)-[:FRIEND]->(f) | f.name]`
    /// - With filter: `[(p)-[:KNOWS]->(other) WHERE other.age > 30 | other.name]`
    /// - Extract IDs: `[(n)-[:HAS]->(item) | id(item)]`
    PatternComprehension {
        /// The graph pattern to match (within parentheses).
        pattern: Box<super::pattern::PathPattern>,
        /// Optional WHERE filter predicate.
        filter_predicate: Option<Box<Expr>>,
        /// The projection expression (after `|`). This is evaluated for each pattern match.
        projection_expr: Box<Expr>,
    },

    /// Cypher EXISTS { } subquery expression.
    ///
    /// Returns a boolean indicating whether the pattern matches any results.
    /// This is a correlated subquery that is evaluated for each row in the outer query.
    ///
    /// Syntax: `EXISTS { pattern [WHERE predicate] }`
    ///
    /// Examples:
    /// - `EXISTS { (p)-[:FRIEND]->(:Person {name: 'Alice'}) }` - checks if pattern exists
    /// - `EXISTS { (p)-[:KNOWS]->(other) WHERE other.age > 30 }` - with filter
    /// - `EXISTS { MATCH (p)-[:FRIEND]->(f) WHERE f.name = 'Bob' }` - full MATCH syntax
    ExistsSubquery {
        /// The graph pattern to check for existence.
        pattern: Box<super::pattern::PathPattern>,
        /// Optional WHERE filter predicate.
        filter_predicate: Option<Box<Expr>>,
    },

    /// Cypher COUNT { } subquery expression.
    ///
    /// Returns the count of pattern matches. This is a correlated subquery
    /// that is evaluated for each row in the outer query.
    ///
    /// Syntax: `COUNT { pattern [WHERE predicate] }`
    ///
    /// Examples:
    /// - `COUNT { (p)-[:FRIEND]->() }` - count number of friends
    /// - `COUNT { (p)-[:KNOWS]->(other) WHERE other.age > 30 }` - count with filter
    CountSubquery {
        /// The graph pattern to count matches.
        pattern: Box<super::pattern::PathPattern>,
        /// Optional WHERE filter predicate.
        filter_predicate: Option<Box<Expr>>,
    },

    /// Cypher CALL { } inline subquery expression.
    ///
    /// Executes a subquery for each row with explicit variable import via WITH.
    /// The subquery can access variables from the outer query through the WITH clause.
    ///
    /// Syntax:
    /// ```cypher
    /// CALL {
    ///   WITH outer_var
    ///   MATCH (outer_var)-[:REL]->(other)
    ///   RETURN count(other) AS cnt
    /// }
    /// ```
    ///
    /// Examples:
    /// - Correlated subquery with aggregation:
    ///   ```cypher
    ///   MATCH (p:Person)
    ///   CALL {
    ///     WITH p
    ///     MATCH (p)-[:FRIEND]->(f)
    ///     RETURN count(f) AS friendCount
    ///   }
    ///   RETURN p.name, friendCount
    ///   ```
    CallSubquery {
        /// Variables imported from outer query (in WITH clause).
        imported_variables: Vec<Identifier>,
        /// The inner statements (typically MATCH/RETURN).
        inner_statements: Vec<super::statement::Statement>,
    },
}

/// An item in a map projection.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjectionItem {
    /// Property selector: `.propertyName` - copies property from source.
    Property(Identifier),
    /// Computed value: `key: expression` - adds a new key with computed value.
    Computed {
        /// The key name.
        key: Identifier,
        /// The value expression.
        value: Box<Expr>,
    },
    /// All properties: `.*` - includes all properties from the source.
    AllProperties,
}

/// A component of a hybrid vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridSearchComponent {
    /// The vector distance expression (e.g., `column <=> $query`).
    pub distance_expr: Box<Expr>,
    /// Weight for this component (0.0 to 1.0).
    pub weight: f64,
}

impl HybridSearchComponent {
    /// Creates a new hybrid search component.
    #[must_use]
    pub fn new(distance_expr: Expr, weight: f64) -> Self {
        Self { distance_expr: Box::new(distance_expr), weight }
    }
}

/// Combination method for hybrid vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridCombinationMethod {
    /// Weighted sum of distances: `w1*d1 + w2*d2`.
    #[default]
    WeightedSum,
    /// Reciprocal Rank Fusion with k parameter.
    RRF {
        /// The k parameter (typically 60).
        k: u32,
    },
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

    /// Converts the expression to a SQL string representation.
    ///
    /// This produces a string that can be re-parsed by the SQL parser.
    /// Used for storing CHECK constraints and other expressions in schema metadata.
    #[must_use]
    pub fn to_sql(&self) -> String {
        match self {
            Self::Literal(lit) => lit.to_string(),
            Self::Column(name) => name.to_string(),
            Self::Parameter(param) => param.to_string(),
            Self::BinaryOp { left, op, right } => {
                format!("({} {} {})", left.to_sql(), op, right.to_sql())
            }
            Self::UnaryOp { op, operand } => match op {
                UnaryOp::Not => format!("NOT ({})", operand.to_sql()),
                UnaryOp::Neg => format!("-({})", operand.to_sql()),
                UnaryOp::IsNull => format!("({}) IS NULL", operand.to_sql()),
                UnaryOp::IsNotNull => format!("({}) IS NOT NULL", operand.to_sql()),
            },
            Self::Function(func) => {
                let args = func.args.iter().map(Self::to_sql).collect::<Vec<_>>().join(", ");
                format!("{}({})", func.name, args)
            }
            Self::Cast { expr, data_type } => {
                format!("CAST({} AS {})", expr.to_sql(), data_type)
            }
            Self::Case(case) => {
                use std::fmt::Write;
                let mut sql = String::from("CASE");
                if let Some(operand) = &case.operand {
                    let _ = write!(sql, " {}", operand.to_sql());
                }
                for (when, then) in &case.when_clauses {
                    let _ = write!(sql, " WHEN {} THEN {}", when.to_sql(), then.to_sql());
                }
                if let Some(else_result) = &case.else_result {
                    let _ = write!(sql, " ELSE {}", else_result.to_sql());
                }
                sql.push_str(" END");
                sql
            }
            Self::InList { expr, list, negated } => {
                let list_sql = list.iter().map(Self::to_sql).collect::<Vec<_>>().join(", ");
                let not = if *negated { "NOT " } else { "" };
                format!("({}) {}IN ({})", expr.to_sql(), not, list_sql)
            }
            Self::Between { expr, low, high, negated } => {
                let not = if *negated { "NOT " } else { "" };
                format!("({}) {}BETWEEN {} AND {}", expr.to_sql(), not, low.to_sql(), high.to_sql())
            }
            Self::Wildcard => "*".to_string(),
            Self::QualifiedWildcard(name) => format!("{}.*", name),
            Self::Tuple(exprs) => {
                let inner = exprs.iter().map(Self::to_sql).collect::<Vec<_>>().join(", ");
                format!("({})", inner)
            }
            // For complex expressions not commonly used in CHECK constraints,
            // fall back to debug representation (these would be rare edge cases)
            _ => format!("{:?}", self),
        }
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
        assert_eq!(Literal::Float(1.5).to_string(), "1.5");
        assert_eq!(Literal::String("hello".into()).to_string(), "'hello'");
        assert_eq!(Literal::Vector(vec![1.0, 2.0, 3.0]).to_string(), "[1, 2, 3]");
        assert_eq!(
            Literal::MultiVector(vec![vec![0.1, 0.2], vec![0.3, 0.4]]).to_string(),
            "[[0.1, 0.2], [0.3, 0.4]]"
        );
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
        assert_eq!(BinaryOp::MaxSim.to_string(), "<##>");
    }

    #[test]
    fn parameter_display() {
        assert_eq!(ParameterRef::Positional(1).to_string(), "$1");
        assert_eq!(ParameterRef::Named("query".into()).to_string(), "$query");
        assert_eq!(ParameterRef::Anonymous.to_string(), "?");
    }
}
