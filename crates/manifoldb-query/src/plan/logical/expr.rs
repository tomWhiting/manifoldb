//! Logical plan expressions.
//!
//! This module defines the expression types used within logical plans.
//! These are similar to AST expressions but designed for plan manipulation.

// Allow arithmetic method names that match std traits - we intentionally
// don't implement the traits because these return new expressions, not Self
#![allow(clippy::should_implement_trait)]
// Allow the long Display impl - it's a big match but simple
#![allow(clippy::too_many_lines)]
// Allow expect - we use it after checking conditions
#![allow(clippy::expect_used)]
// Allow missing_const_for_fn - const fn with Vec isn't stable
#![allow(clippy::missing_const_for_fn)]

use std::fmt;

use crate::ast::{
    AggregateWindowFunction, BinaryOp, DataType, Literal, QualifiedName, UnaryOp, WindowFrame,
    WindowFrameBound, WindowFrameUnits, WindowFunction,
};

/// An expression in a logical plan.
///
/// Unlike AST expressions, logical expressions are designed for
/// query planning and optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalExpr {
    /// A literal value.
    Literal(Literal),

    /// A column reference with optional table qualifier.
    Column {
        /// Table/alias qualifier (e.g., "users" in "users.id").
        qualifier: Option<String>,
        /// Column name.
        name: String,
    },

    /// A binary operation.
    BinaryOp {
        /// Left operand.
        left: Box<LogicalExpr>,
        /// The operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<LogicalExpr>,
    },

    /// A unary operation.
    UnaryOp {
        /// The operator.
        op: UnaryOp,
        /// The operand.
        operand: Box<LogicalExpr>,
    },

    /// A scalar function call.
    ScalarFunction {
        /// Function name.
        func: ScalarFunction,
        /// Function arguments.
        args: Vec<LogicalExpr>,
    },

    /// An aggregate function call.
    AggregateFunction {
        /// Aggregate function.
        func: AggregateFunction,
        /// Arguments to the aggregate function.
        /// For most aggregates this is a single expression.
        /// For string_agg, this includes the expression and delimiter.
        args: Vec<LogicalExpr>,
        /// Whether DISTINCT is specified.
        distinct: bool,
    },

    /// A CAST expression.
    Cast {
        /// The expression to cast.
        expr: Box<LogicalExpr>,
        /// The target data type.
        data_type: DataType,
    },

    /// A CASE expression.
    Case {
        /// The operand for simple CASE (None for searched CASE).
        operand: Option<Box<LogicalExpr>>,
        /// WHEN...THEN branches.
        when_clauses: Vec<(LogicalExpr, LogicalExpr)>,
        /// ELSE result.
        else_result: Option<Box<LogicalExpr>>,
    },

    /// expr IN (val1, val2, ...).
    InList {
        /// The expression to check.
        expr: Box<LogicalExpr>,
        /// The list of values.
        list: Vec<LogicalExpr>,
        /// Whether NOT IN.
        negated: bool,
    },

    /// expr BETWEEN low AND high.
    Between {
        /// The expression to check.
        expr: Box<LogicalExpr>,
        /// Lower bound.
        low: Box<LogicalExpr>,
        /// Upper bound.
        high: Box<LogicalExpr>,
        /// Whether NOT BETWEEN.
        negated: bool,
    },

    /// A subquery expression.
    Subquery(Box<super::LogicalPlan>),

    /// EXISTS (subquery).
    Exists {
        /// The subquery.
        subquery: Box<super::LogicalPlan>,
        /// Whether NOT EXISTS.
        negated: bool,
    },

    /// expr IN (subquery).
    InSubquery {
        /// The expression to check.
        expr: Box<LogicalExpr>,
        /// The subquery.
        subquery: Box<super::LogicalPlan>,
        /// Whether NOT IN.
        negated: bool,
    },

    /// Wildcard (*) for SELECT *.
    Wildcard,

    /// Qualified wildcard (table.*).
    QualifiedWildcard(String),

    /// An aliased expression.
    Alias {
        /// The expression.
        expr: Box<LogicalExpr>,
        /// The alias name.
        alias: String,
    },

    /// A parameter placeholder.
    Parameter(u32),

    /// Hybrid vector search expression.
    ///
    /// Combines multiple vector distance operations with weights.
    /// Example: `HYBRID(dense <=> $q1, 0.7, sparse <#> $q2, 0.3)`
    HybridSearch {
        /// Vector search components (each has distance expr and weight).
        components: Vec<HybridExprComponent>,
        /// Combination method (WeightedSum, RRF).
        method: HybridCombinationMethod,
    },

    /// Window function expression.
    ///
    /// Used for ranking functions like ROW_NUMBER(), RANK(), DENSE_RANK()
    /// and value functions like LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE(), NTH_VALUE().
    /// Also supports aggregate functions as window functions with frame clauses.
    /// Example: `ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)`
    /// Example: `LAG(salary, 1, 0) OVER (PARTITION BY dept ORDER BY hire_date)`
    /// Example: `SUM(salary) OVER (ORDER BY hire_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`
    WindowFunction {
        /// The window function type.
        func: WindowFunction,
        /// The expression argument for value functions (e.g., the column to retrieve).
        /// For ranking functions (ROW_NUMBER, RANK, DENSE_RANK), this is None.
        arg: Option<Box<LogicalExpr>>,
        /// Default value for LAG/LEAD when the offset row doesn't exist.
        default_value: Option<Box<LogicalExpr>>,
        /// Partition by expressions (creates separate numbering per partition).
        partition_by: Vec<LogicalExpr>,
        /// Order by expressions (determines ranking order).
        order_by: Vec<SortOrder>,
        /// Window frame specification for controlling which rows are included in calculations.
        /// If None, uses the default frame (RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        /// when ORDER BY is present, or the entire partition otherwise).
        frame: Option<WindowFrame>,
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
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Optional WHERE filter predicate.
        filter_predicate: Option<Box<LogicalExpr>>,
        /// Optional transform expression (after `|`). If None, returns the variable.
        transform_expr: Option<Box<LogicalExpr>>,
    },

    /// A list literal expression: `[expr1, expr2, ...]`.
    ListLiteral(Vec<LogicalExpr>),

    /// Cypher list predicate function: `all(variable IN list WHERE predicate)`.
    ///
    /// Returns true if ALL elements in the list satisfy the predicate.
    ///
    /// Examples:
    /// - `all(x IN [1, 2, 3] WHERE x > 0)` → true
    /// - `all(x IN [1, -2, 3] WHERE x > 0)` → false
    ListPredicateAll {
        /// Variable name for iteration.
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<LogicalExpr>,
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
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<LogicalExpr>,
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
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<LogicalExpr>,
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
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Predicate expression to evaluate for each element.
        predicate: Box<LogicalExpr>,
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
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<LogicalExpr>,
        /// Variable name for iteration.
        variable: String,
        /// List expression to iterate over.
        list_expr: Box<LogicalExpr>,
        /// Expression to compute new accumulator value (can reference both accumulator and variable).
        expression: Box<LogicalExpr>,
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
        /// Source expression (typically a column reference to a node or relationship).
        source: Box<LogicalExpr>,
        /// List of projection items.
        items: Vec<LogicalMapProjectionItem>,
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
        /// The expand nodes representing the graph pattern steps.
        /// For each match of the pattern, we bind variables and evaluate the projection.
        expand_steps: Vec<super::graph::ExpandNode>,
        /// Optional WHERE filter predicate applied after pattern matching.
        filter_predicate: Option<Box<LogicalExpr>>,
        /// The projection expression evaluated for each pattern match.
        projection_expr: Box<LogicalExpr>,
    },

    /// Cypher EXISTS { } subquery expression.
    ///
    /// Returns a boolean indicating whether the pattern matches any results.
    /// Semantically similar to `size([(pattern) | 1]) > 0`.
    ///
    /// Syntax: `EXISTS { pattern [WHERE predicate] }`
    ///
    /// Examples:
    /// - `EXISTS { (p)-[:FRIEND]->(:Person {name: 'Alice'}) }`
    /// - `EXISTS { (p)-[:KNOWS]->(other) WHERE other.age > 30 }`
    ExistsSubquery {
        /// The expand nodes representing the graph pattern steps.
        expand_steps: Vec<super::graph::ExpandNode>,
        /// Optional WHERE filter predicate applied after pattern matching.
        filter_predicate: Option<Box<LogicalExpr>>,
    },

    /// Cypher COUNT { } subquery expression.
    ///
    /// Returns the count of pattern matches.
    /// Semantically similar to `size([(pattern) | 1])`.
    ///
    /// Syntax: `COUNT { pattern [WHERE predicate] }`
    ///
    /// Examples:
    /// - `COUNT { (p)-[:FRIEND]->() }` - count number of friends
    /// - `COUNT { (p)-[:KNOWS]->(other) WHERE other.age > 30 }` - count with filter
    CountSubquery {
        /// The expand nodes representing the graph pattern steps.
        expand_steps: Vec<super::graph::ExpandNode>,
        /// Optional WHERE filter predicate applied after pattern matching.
        filter_predicate: Option<Box<LogicalExpr>>,
    },

    /// Cypher CALL { } inline subquery expression.
    ///
    /// Executes a subquery for each row with explicit variable import via WITH.
    /// Returns the values produced by the subquery's RETURN clause.
    ///
    /// Syntax:
    /// ```cypher
    /// CALL {
    ///   WITH outer_var
    ///   MATCH (outer_var)-[:REL]->(other)
    ///   RETURN count(other) AS cnt
    /// }
    /// ```
    CallSubquery {
        /// Variables imported from outer query (in WITH clause).
        imported_variables: Vec<String>,
        /// The logical plan for the inner subquery.
        inner_plan: Box<super::LogicalPlan>,
    },
}

/// An item in a logical map projection.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalMapProjectionItem {
    /// Property selector: `.propertyName` - copies property from source.
    Property(String),
    /// Computed value: `key: expression` - adds a new key with computed value.
    Computed {
        /// The key name.
        key: String,
        /// The value expression.
        value: Box<LogicalExpr>,
    },
    /// All properties: `.*` - includes all properties from the source.
    AllProperties,
}

/// A component of a hybrid vector search expression.
///
/// This is used within `LogicalExpr::HybridSearch` to represent each
/// weighted distance computation in a hybrid search.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridExprComponent {
    /// The vector distance expression (e.g., `column <=> $query`).
    pub distance_expr: Box<LogicalExpr>,
    /// Weight for this component (0.0 to 1.0).
    pub weight: f64,
}

impl HybridExprComponent {
    /// Creates a new hybrid expression component.
    #[must_use]
    pub fn new(distance_expr: LogicalExpr, weight: f64) -> Self {
        Self { distance_expr: Box::new(distance_expr), weight }
    }
}

/// Combination method for hybrid vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridCombinationMethod {
    /// Weighted sum of distances: `w1*d1 + w2*d2`.
    WeightedSum,
    /// Reciprocal Rank Fusion with k parameter.
    RRF {
        /// The k parameter (typically 60).
        k: u32,
    },
}

impl LogicalExpr {
    // ========== Constructors ==========

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

    /// Creates a literal vector expression.
    #[must_use]
    pub fn vector(value: Vec<f32>) -> Self {
        Self::Literal(Literal::Vector(value))
    }

    /// Creates a column reference.
    #[must_use]
    pub fn column(name: impl Into<String>) -> Self {
        Self::Column { qualifier: None, name: name.into() }
    }

    /// Creates a qualified column reference.
    #[must_use]
    pub fn qualified_column(qualifier: impl Into<String>, name: impl Into<String>) -> Self {
        Self::Column { qualifier: Some(qualifier.into()), name: name.into() }
    }

    /// Creates a wildcard expression.
    #[must_use]
    pub const fn wildcard() -> Self {
        Self::Wildcard
    }

    /// Creates a qualified wildcard expression.
    #[must_use]
    pub fn qualified_wildcard(qualifier: impl Into<String>) -> Self {
        Self::QualifiedWildcard(qualifier.into())
    }

    /// Creates a parameter placeholder.
    #[must_use]
    pub const fn param(index: u32) -> Self {
        Self::Parameter(index)
    }

    // ========== Binary Operations ==========

    fn binary(self, op: BinaryOp, other: Self) -> Self {
        Self::BinaryOp { left: Box::new(self), op, right: Box::new(other) }
    }

    /// Creates an AND expression.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        self.binary(BinaryOp::And, other)
    }

    /// Creates an OR expression.
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        self.binary(BinaryOp::Or, other)
    }

    /// Creates an equality expression.
    #[must_use]
    pub fn eq(self, other: Self) -> Self {
        self.binary(BinaryOp::Eq, other)
    }

    /// Creates a not-equal expression.
    #[must_use]
    pub fn not_eq(self, other: Self) -> Self {
        self.binary(BinaryOp::NotEq, other)
    }

    /// Creates a less-than expression.
    #[must_use]
    pub fn lt(self, other: Self) -> Self {
        self.binary(BinaryOp::Lt, other)
    }

    /// Creates a less-than-or-equal expression.
    #[must_use]
    pub fn lt_eq(self, other: Self) -> Self {
        self.binary(BinaryOp::LtEq, other)
    }

    /// Creates a greater-than expression.
    #[must_use]
    pub fn gt(self, other: Self) -> Self {
        self.binary(BinaryOp::Gt, other)
    }

    /// Creates a greater-than-or-equal expression.
    #[must_use]
    pub fn gt_eq(self, other: Self) -> Self {
        self.binary(BinaryOp::GtEq, other)
    }

    /// Creates an addition expression.
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        self.binary(BinaryOp::Add, other)
    }

    /// Creates a subtraction expression.
    #[must_use]
    pub fn sub(self, other: Self) -> Self {
        self.binary(BinaryOp::Sub, other)
    }

    /// Creates a multiplication expression.
    #[must_use]
    pub fn mul(self, other: Self) -> Self {
        self.binary(BinaryOp::Mul, other)
    }

    /// Creates a division expression.
    #[must_use]
    pub fn div(self, other: Self) -> Self {
        self.binary(BinaryOp::Div, other)
    }

    /// Creates a modulo expression.
    #[must_use]
    pub fn modulo(self, other: Self) -> Self {
        self.binary(BinaryOp::Mod, other)
    }

    /// Creates a LIKE expression.
    #[must_use]
    pub fn like(self, pattern: Self) -> Self {
        self.binary(BinaryOp::Like, pattern)
    }

    /// Creates an ILIKE expression.
    #[must_use]
    pub fn ilike(self, pattern: Self) -> Self {
        self.binary(BinaryOp::ILike, pattern)
    }

    // ========== Vector Distance Operations ==========

    /// Creates an Euclidean distance expression.
    #[must_use]
    pub fn euclidean_distance(self, other: Self) -> Self {
        self.binary(BinaryOp::EuclideanDistance, other)
    }

    /// Creates a cosine distance expression.
    #[must_use]
    pub fn cosine_distance(self, other: Self) -> Self {
        self.binary(BinaryOp::CosineDistance, other)
    }

    /// Creates an inner product expression.
    #[must_use]
    pub fn inner_product(self, other: Self) -> Self {
        self.binary(BinaryOp::InnerProduct, other)
    }

    // ========== Unary Operations ==========

    /// Creates a NOT expression.
    #[must_use]
    pub fn not(self) -> Self {
        Self::UnaryOp { op: UnaryOp::Not, operand: Box::new(self) }
    }

    /// Creates a negation expression.
    #[must_use]
    pub fn neg(self) -> Self {
        Self::UnaryOp { op: UnaryOp::Neg, operand: Box::new(self) }
    }

    /// Creates an IS NULL expression.
    #[must_use]
    pub fn is_null(self) -> Self {
        Self::UnaryOp { op: UnaryOp::IsNull, operand: Box::new(self) }
    }

    /// Creates an IS NOT NULL expression.
    #[must_use]
    pub fn is_not_null(self) -> Self {
        Self::UnaryOp { op: UnaryOp::IsNotNull, operand: Box::new(self) }
    }

    // ========== Other Operations ==========

    /// Creates an IN list expression.
    #[must_use]
    pub fn in_list(self, list: Vec<Self>, negated: bool) -> Self {
        Self::InList { expr: Box::new(self), list, negated }
    }

    /// Creates a BETWEEN expression.
    #[must_use]
    pub fn between(self, low: Self, high: Self, negated: bool) -> Self {
        Self::Between { expr: Box::new(self), low: Box::new(low), high: Box::new(high), negated }
    }

    /// Creates a CAST expression.
    #[must_use]
    pub fn cast(self, data_type: DataType) -> Self {
        Self::Cast { expr: Box::new(self), data_type }
    }

    /// Creates an aliased expression.
    #[must_use]
    pub fn alias(self, name: impl Into<String>) -> Self {
        Self::Alias { expr: Box::new(self), alias: name.into() }
    }

    // ========== Aggregate Functions ==========

    /// Creates a COUNT aggregate.
    #[must_use]
    pub fn count(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::Count, args: vec![expr], distinct }
    }

    /// Creates a SUM aggregate.
    #[must_use]
    pub fn sum(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::Sum, args: vec![expr], distinct }
    }

    /// Creates an AVG aggregate.
    #[must_use]
    pub fn avg(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::Avg, args: vec![expr], distinct }
    }

    /// Creates a MIN aggregate.
    #[must_use]
    pub fn min(expr: Self) -> Self {
        Self::AggregateFunction { func: AggregateFunction::Min, args: vec![expr], distinct: false }
    }

    /// Creates a MAX aggregate.
    #[must_use]
    pub fn max(expr: Self) -> Self {
        Self::AggregateFunction { func: AggregateFunction::Max, args: vec![expr], distinct: false }
    }

    /// Creates an ARRAY_AGG aggregate.
    #[must_use]
    pub fn array_agg(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::ArrayAgg, args: vec![expr], distinct }
    }

    /// Creates a STRING_AGG aggregate.
    #[must_use]
    pub fn string_agg(expr: Self, delimiter: Self, distinct: bool) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::StringAgg,
            args: vec![expr, delimiter],
            distinct,
        }
    }

    /// Creates a sample standard deviation aggregate (STDDEV).
    #[must_use]
    pub fn stddev_samp(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::StddevSamp, args: vec![expr], distinct }
    }

    /// Creates a population standard deviation aggregate (STDDEV_POP).
    #[must_use]
    pub fn stddev_pop(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::StddevPop, args: vec![expr], distinct }
    }

    /// Creates a sample variance aggregate (VARIANCE).
    #[must_use]
    pub fn variance_samp(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::VarianceSamp,
            args: vec![expr],
            distinct,
        }
    }

    /// Creates a population variance aggregate (VAR_POP).
    #[must_use]
    pub fn variance_pop(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::VariancePop, args: vec![expr], distinct }
    }

    /// Creates a continuous percentile aggregate (percentileCont).
    /// The percentile argument should be between 0.0 and 1.0.
    #[must_use]
    pub fn percentile_cont(percentile: Self, expr: Self) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::PercentileCont,
            args: vec![percentile, expr],
            distinct: false,
        }
    }

    /// Creates a discrete percentile aggregate (percentileDisc).
    /// The percentile argument should be between 0.0 and 1.0.
    #[must_use]
    pub fn percentile_disc(percentile: Self, expr: Self) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::PercentileDisc,
            args: vec![percentile, expr],
            distinct: false,
        }
    }

    /// Creates a JSON_AGG aggregate.
    ///
    /// Aggregates values into a JSON array.
    /// Example: `SELECT json_agg(name) FROM users;` returns `["Alice", "Bob", "Charlie"]`
    #[must_use]
    pub fn json_agg(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::JsonAgg, args: vec![expr], distinct }
    }

    /// Creates a JSONB_AGG aggregate.
    ///
    /// Aggregates values into a JSONB array (same as JSON_AGG in our implementation).
    #[must_use]
    pub fn jsonb_agg(expr: Self, distinct: bool) -> Self {
        Self::AggregateFunction { func: AggregateFunction::JsonbAgg, args: vec![expr], distinct }
    }

    /// Creates a JSON_OBJECT_AGG aggregate.
    ///
    /// Aggregates key-value pairs into a JSON object.
    /// Example: `SELECT json_object_agg(key, value) FROM pairs;` returns `{"key1": "value1", "key2": "value2"}`
    #[must_use]
    pub fn json_object_agg(key: Self, value: Self, distinct: bool) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::JsonObjectAgg,
            args: vec![key, value],
            distinct,
        }
    }

    /// Creates a JSONB_OBJECT_AGG aggregate.
    ///
    /// Aggregates key-value pairs into a JSONB object (same as JSON_OBJECT_AGG in our implementation).
    #[must_use]
    pub fn jsonb_object_agg(key: Self, value: Self, distinct: bool) -> Self {
        Self::AggregateFunction {
            func: AggregateFunction::JsonbObjectAgg,
            args: vec![key, value],
            distinct,
        }
    }

    // ========== Window Functions ==========

    /// Creates a ROW_NUMBER window function.
    #[must_use]
    pub fn row_number(partition_by: Vec<Self>, order_by: Vec<SortOrder>) -> Self {
        Self::WindowFunction {
            func: WindowFunction::RowNumber,
            arg: None,
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a RANK window function.
    #[must_use]
    pub fn rank(partition_by: Vec<Self>, order_by: Vec<SortOrder>) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Rank,
            arg: None,
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a DENSE_RANK window function.
    #[must_use]
    pub fn dense_rank(partition_by: Vec<Self>, order_by: Vec<SortOrder>) -> Self {
        Self::WindowFunction {
            func: WindowFunction::DenseRank,
            arg: None,
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a LAG window function.
    ///
    /// LAG(expr, offset, default) accesses a value from a previous row.
    /// - `expr`: The value expression to retrieve.
    /// - `offset`: Number of rows back (default 1).
    /// - `default_value`: Value to return if offset row doesn't exist.
    #[must_use]
    pub fn lag(
        expr: Self,
        offset: u64,
        default_value: Option<Self>,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Lag { offset, has_default: default_value.is_some() },
            arg: Some(Box::new(expr)),
            default_value: default_value.map(Box::new),
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a LEAD window function.
    ///
    /// LEAD(expr, offset, default) accesses a value from a following row.
    /// - `expr`: The value expression to retrieve.
    /// - `offset`: Number of rows forward (default 1).
    /// - `default_value`: Value to return if offset row doesn't exist.
    #[must_use]
    pub fn lead(
        expr: Self,
        offset: u64,
        default_value: Option<Self>,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Lead { offset, has_default: default_value.is_some() },
            arg: Some(Box::new(expr)),
            default_value: default_value.map(Box::new),
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a FIRST_VALUE window function.
    ///
    /// FIRST_VALUE(expr) returns the first value in the window frame.
    #[must_use]
    pub fn first_value(expr: Self, partition_by: Vec<Self>, order_by: Vec<SortOrder>) -> Self {
        Self::WindowFunction {
            func: WindowFunction::FirstValue,
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a LAST_VALUE window function.
    ///
    /// LAST_VALUE(expr) returns the last value in the window frame.
    #[must_use]
    pub fn last_value(expr: Self, partition_by: Vec<Self>, order_by: Vec<SortOrder>) -> Self {
        Self::WindowFunction {
            func: WindowFunction::LastValue,
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates an NTH_VALUE window function.
    ///
    /// NTH_VALUE(expr, n) returns the nth value in the window frame (1-indexed).
    #[must_use]
    pub fn nth_value(
        expr: Self,
        n: u64,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::NthValue { n },
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame: None,
        }
    }

    /// Creates a SUM window function.
    ///
    /// SUM(expr) OVER (...) computes the running total over the window frame.
    /// Example: `SUM(amount) OVER (ORDER BY date)` for running totals.
    #[must_use]
    pub fn sum_window(
        expr: Self,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Aggregate(AggregateWindowFunction::Sum),
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame,
        }
    }

    /// Creates an AVG window function.
    ///
    /// AVG(expr) OVER (...) computes the moving average over the window frame.
    /// Example: `AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` for 7-day moving average.
    #[must_use]
    pub fn avg_window(
        expr: Self,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Aggregate(AggregateWindowFunction::Avg),
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame,
        }
    }

    /// Creates a COUNT window function.
    ///
    /// COUNT(expr) OVER (...) computes the cumulative count over the window frame.
    /// Use None for expr to count all rows (COUNT(*)).
    #[must_use]
    pub fn count_window(
        expr: Option<Self>,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Aggregate(AggregateWindowFunction::Count),
            arg: expr.map(Box::new),
            default_value: None,
            partition_by,
            order_by,
            frame,
        }
    }

    /// Creates a MIN window function.
    ///
    /// MIN(expr) OVER (...) computes the cumulative minimum over the window frame.
    #[must_use]
    pub fn min_window(
        expr: Self,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Aggregate(AggregateWindowFunction::Min),
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame,
        }
    }

    /// Creates a MAX window function.
    ///
    /// MAX(expr) OVER (...) computes the cumulative maximum over the window frame.
    #[must_use]
    pub fn max_window(
        expr: Self,
        partition_by: Vec<Self>,
        order_by: Vec<SortOrder>,
        frame: Option<WindowFrame>,
    ) -> Self {
        Self::WindowFunction {
            func: WindowFunction::Aggregate(AggregateWindowFunction::Max),
            arg: Some(Box::new(expr)),
            default_value: None,
            partition_by,
            order_by,
            frame,
        }
    }

    // ========== Utility Methods ==========

    /// Returns the alias if this is an aliased expression.
    #[must_use]
    pub fn get_alias(&self) -> Option<&str> {
        match self {
            Self::Alias { alias, .. } => Some(alias),
            _ => None,
        }
    }

    /// Returns the column name if this is a column reference.
    #[must_use]
    pub fn column_name(&self) -> Option<&str> {
        match self {
            Self::Column { name, .. } => Some(name),
            Self::Alias { alias, .. } => Some(alias),
            _ => None,
        }
    }

    /// Returns true if this expression contains an aggregate function.
    #[must_use]
    pub fn contains_aggregate(&self) -> bool {
        match self {
            Self::AggregateFunction { .. } => true,
            Self::BinaryOp { left, right, .. } => {
                left.contains_aggregate() || right.contains_aggregate()
            }
            Self::UnaryOp { operand, .. } => operand.contains_aggregate(),
            Self::ScalarFunction { args, .. } => args.iter().any(Self::contains_aggregate),
            Self::Cast { expr, .. } | Self::Alias { expr, .. } => expr.contains_aggregate(),
            Self::Case { operand, when_clauses, else_result } => {
                operand.as_ref().is_some_and(|e| e.contains_aggregate())
                    || when_clauses
                        .iter()
                        .any(|(w, t)| w.contains_aggregate() || t.contains_aggregate())
                    || else_result.as_ref().is_some_and(|e| e.contains_aggregate())
            }
            Self::InList { expr, list, .. } => {
                expr.contains_aggregate() || list.iter().any(Self::contains_aggregate)
            }
            Self::Between { expr, low, high, .. } => {
                expr.contains_aggregate() || low.contains_aggregate() || high.contains_aggregate()
            }
            Self::HybridSearch { components, .. } => {
                components.iter().any(|c| c.distance_expr.contains_aggregate())
            }
            Self::WindowFunction { partition_by, order_by, .. } => {
                partition_by.iter().any(Self::contains_aggregate)
                    || order_by.iter().any(|s| s.expr.contains_aggregate())
            }
            _ => false,
        }
    }

    /// Returns true if this expression contains a window function.
    #[must_use]
    pub fn contains_window_function(&self) -> bool {
        match self {
            Self::WindowFunction { .. } => true,
            Self::BinaryOp { left, right, .. } => {
                left.contains_window_function() || right.contains_window_function()
            }
            Self::UnaryOp { operand, .. } => operand.contains_window_function(),
            Self::ScalarFunction { args, .. } => args.iter().any(Self::contains_window_function),
            Self::Cast { expr, .. } | Self::Alias { expr, .. } => expr.contains_window_function(),
            Self::Case { operand, when_clauses, else_result } => {
                operand.as_ref().is_some_and(|e| e.contains_window_function())
                    || when_clauses
                        .iter()
                        .any(|(w, t)| w.contains_window_function() || t.contains_window_function())
                    || else_result.as_ref().is_some_and(|e| e.contains_window_function())
            }
            Self::InList { expr, list, .. } => {
                expr.contains_window_function() || list.iter().any(Self::contains_window_function)
            }
            Self::Between { expr, low, high, .. } => {
                expr.contains_window_function()
                    || low.contains_window_function()
                    || high.contains_window_function()
            }
            Self::HybridSearch { components, .. } => {
                components.iter().any(|c| c.distance_expr.contains_window_function())
            }
            _ => false,
        }
    }

    /// Converts from an AST `QualifiedName` to a column expression.
    #[must_use]
    pub fn from_qualified_name(name: &QualifiedName) -> Self {
        match name.parts.len() {
            0 => Self::column(""),
            1 => Self::column(&name.parts[0].name),
            _ => {
                let qualifier = name.parts[..name.parts.len() - 1]
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
                    .join(".");
                // Safe: we already checked len() > 1 so last() is Some
                let col_name = &name.parts.last().expect("checked len > 1").name;
                Self::qualified_column(qualifier, col_name)
            }
        }
    }
}

/// Helper function to format a window frame bound.
fn format_frame_bound(f: &mut fmt::Formatter<'_>, bound: &WindowFrameBound) -> fmt::Result {
    match bound {
        WindowFrameBound::CurrentRow => write!(f, "CURRENT ROW"),
        WindowFrameBound::UnboundedPreceding => write!(f, "UNBOUNDED PRECEDING"),
        WindowFrameBound::UnboundedFollowing => write!(f, "UNBOUNDED FOLLOWING"),
        WindowFrameBound::Preceding(expr) => write!(f, "{expr:?} PRECEDING"),
        WindowFrameBound::Following(expr) => write!(f, "{expr:?} FOLLOWING"),
    }
}

impl fmt::Display for LogicalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(lit) => write!(f, "{lit}"),
            Self::Column { qualifier, name } => {
                if let Some(q) = qualifier {
                    write!(f, "{q}.{name}")
                } else {
                    write!(f, "{name}")
                }
            }
            Self::BinaryOp { left, op, right } => write!(f, "({left} {op} {right})"),
            Self::UnaryOp { op, operand } => match op {
                UnaryOp::Not => write!(f, "NOT {operand}"),
                UnaryOp::Neg => write!(f, "-{operand}"),
                UnaryOp::IsNull => write!(f, "{operand} IS NULL"),
                UnaryOp::IsNotNull => write!(f, "{operand} IS NOT NULL"),
            },
            Self::ScalarFunction { func, args } => {
                write!(f, "{func}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::AggregateFunction { func, args, distinct } => {
                write!(f, "{func}(")?;
                if *distinct {
                    write!(f, "DISTINCT ")?;
                }
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            Self::Cast { expr, data_type } => write!(f, "CAST({expr} AS {data_type:?})"),
            Self::Case { operand, when_clauses, else_result } => {
                write!(f, "CASE")?;
                if let Some(op) = operand {
                    write!(f, " {op}")?;
                }
                for (when, then) in when_clauses {
                    write!(f, " WHEN {when} THEN {then}")?;
                }
                if let Some(else_res) = else_result {
                    write!(f, " ELSE {else_res}")?;
                }
                write!(f, " END")
            }
            Self::InList { expr, list, negated } => {
                write!(f, "{expr}")?;
                if *negated {
                    write!(f, " NOT")?;
                }
                write!(f, " IN (")?;
                for (i, item) in list.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
            Self::Between { expr, low, high, negated } => {
                write!(f, "{expr}")?;
                if *negated {
                    write!(f, " NOT")?;
                }
                write!(f, " BETWEEN {low} AND {high}")
            }
            Self::Subquery(_) => write!(f, "(subquery)"),
            Self::Exists { negated, .. } => {
                if *negated {
                    write!(f, "NOT EXISTS (subquery)")
                } else {
                    write!(f, "EXISTS (subquery)")
                }
            }
            Self::InSubquery { expr, negated, .. } => {
                write!(f, "{expr}")?;
                if *negated {
                    write!(f, " NOT")?;
                }
                write!(f, " IN (subquery)")
            }
            Self::Wildcard => write!(f, "*"),
            Self::QualifiedWildcard(qualifier) => write!(f, "{qualifier}.*"),
            Self::Alias { expr, alias } => write!(f, "{expr} AS {alias}"),
            Self::Parameter(idx) => write!(f, "${idx}"),
            Self::HybridSearch { components, method } => {
                let method_name = match method {
                    HybridCombinationMethod::WeightedSum => "HYBRID",
                    HybridCombinationMethod::RRF { .. } => "RRF",
                };
                write!(f, "{method_name}(")?;
                for (i, comp) in components.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}, {}", comp.distance_expr, comp.weight)?;
                }
                write!(f, ")")
            }
            Self::WindowFunction { func, arg, default_value, partition_by, order_by, frame } => {
                // Print function name and arguments
                match func {
                    WindowFunction::RowNumber
                    | WindowFunction::Rank
                    | WindowFunction::DenseRank => {
                        write!(f, "{func}()")?;
                    }
                    WindowFunction::Lag { offset, .. } | WindowFunction::Lead { offset, .. } => {
                        let name =
                            if matches!(func, WindowFunction::Lag { .. }) { "LAG" } else { "LEAD" };
                        write!(f, "{name}(")?;
                        if let Some(a) = arg {
                            write!(f, "{a}")?;
                        }
                        if *offset != 1 {
                            write!(f, ", {offset}")?;
                        }
                        if let Some(def) = default_value {
                            if *offset == 1 {
                                write!(f, ", 1")?;
                            }
                            write!(f, ", {def}")?;
                        }
                        write!(f, ")")?;
                    }
                    WindowFunction::FirstValue | WindowFunction::LastValue => {
                        let name = if matches!(func, WindowFunction::FirstValue) {
                            "FIRST_VALUE"
                        } else {
                            "LAST_VALUE"
                        };
                        write!(f, "{name}(")?;
                        if let Some(a) = arg {
                            write!(f, "{a}")?;
                        }
                        write!(f, ")")?;
                    }
                    WindowFunction::NthValue { n } => {
                        write!(f, "NTH_VALUE(")?;
                        if let Some(a) = arg {
                            write!(f, "{a}")?;
                        }
                        write!(f, ", {n})")?;
                    }
                    WindowFunction::Aggregate(agg_func) => {
                        write!(f, "{agg_func}(")?;
                        if let Some(a) = arg {
                            write!(f, "{a}")?;
                        }
                        write!(f, ")")?;
                    }
                }
                write!(f, " OVER (")?;
                if !partition_by.is_empty() {
                    write!(f, "PARTITION BY ")?;
                    for (i, expr) in partition_by.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{expr}")?;
                    }
                    if !order_by.is_empty() || frame.is_some() {
                        write!(f, " ")?;
                    }
                }
                if !order_by.is_empty() {
                    write!(f, "ORDER BY ")?;
                    for (i, sort) in order_by.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{sort}")?;
                    }
                    if frame.is_some() {
                        write!(f, " ")?;
                    }
                }
                // Display frame clause if present
                if let Some(window_frame) = frame {
                    let units = match window_frame.units {
                        WindowFrameUnits::Rows => "ROWS",
                        WindowFrameUnits::Range => "RANGE",
                        WindowFrameUnits::Groups => "GROUPS",
                    };
                    write!(f, "{units} ")?;
                    if let Some(ref end) = window_frame.end {
                        write!(f, "BETWEEN ")?;
                        format_frame_bound(f, &window_frame.start)?;
                        write!(f, " AND ")?;
                        format_frame_bound(f, end)?;
                    } else {
                        format_frame_bound(f, &window_frame.start)?;
                    }
                }
                write!(f, ")")
            }
            Self::ListComprehension { variable, list_expr, filter_predicate, transform_expr } => {
                write!(f, "[{variable} IN {list_expr}")?;
                if let Some(filter) = filter_predicate {
                    write!(f, " WHERE {filter}")?;
                }
                if let Some(transform) = transform_expr {
                    write!(f, " | {transform}")?;
                }
                write!(f, "]")
            }
            Self::ListLiteral(exprs) => {
                write!(f, "[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                write!(f, "]")
            }
            Self::ListPredicateAll { variable, list_expr, predicate } => {
                write!(f, "all({variable} IN {list_expr} WHERE {predicate})")
            }
            Self::ListPredicateAny { variable, list_expr, predicate } => {
                write!(f, "any({variable} IN {list_expr} WHERE {predicate})")
            }
            Self::ListPredicateNone { variable, list_expr, predicate } => {
                write!(f, "none({variable} IN {list_expr} WHERE {predicate})")
            }
            Self::ListPredicateSingle { variable, list_expr, predicate } => {
                write!(f, "single({variable} IN {list_expr} WHERE {predicate})")
            }
            Self::ListReduce { accumulator, initial, variable, list_expr, expression } => {
                write!(
                    f,
                    "reduce({accumulator} = {initial}, {variable} IN {list_expr} | {expression})"
                )
            }
            Self::MapProjection { source, items } => {
                write!(f, "{source}{{")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match item {
                        LogicalMapProjectionItem::Property(name) => write!(f, ".{name}")?,
                        LogicalMapProjectionItem::Computed { key, value } => {
                            write!(f, "{key}: {value}")?
                        }
                        LogicalMapProjectionItem::AllProperties => write!(f, ".*")?,
                    }
                }
                write!(f, "}}")
            }
            Self::PatternComprehension { expand_steps, filter_predicate, projection_expr } => {
                write!(f, "[(")?;
                format_expand_steps(f, expand_steps)?;
                write!(f, ")")?;
                if let Some(filter) = filter_predicate {
                    write!(f, " WHERE {filter}")?;
                }
                write!(f, " | {projection_expr}]")
            }
            Self::ExistsSubquery { expand_steps, filter_predicate } => {
                write!(f, "EXISTS {{ (")?;
                format_expand_steps(f, expand_steps)?;
                write!(f, ")")?;
                if let Some(filter) = filter_predicate {
                    write!(f, " WHERE {filter}")?;
                }
                write!(f, " }}")
            }
            Self::CountSubquery { expand_steps, filter_predicate } => {
                write!(f, "COUNT {{ (")?;
                format_expand_steps(f, expand_steps)?;
                write!(f, ")")?;
                if let Some(filter) = filter_predicate {
                    write!(f, " WHERE {filter}")?;
                }
                write!(f, " }}")
            }
            Self::CallSubquery { imported_variables, inner_plan: _ } => {
                write!(f, "CALL {{ WITH ")?;
                for (i, var) in imported_variables.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{var}")?;
                }
                write!(f, " ... }}")
            }
        }
    }
}

/// Helper function to format expand steps for pattern display.
fn format_expand_steps(
    f: &mut fmt::Formatter<'_>,
    expand_steps: &[super::graph::ExpandNode],
) -> fmt::Result {
    for (i, step) in expand_steps.iter().enumerate() {
        if i > 0 {
            write!(f, ")")?;
        }
        write!(f, "{}", step.src_var)?;
        write!(f, "-[")?;
        if let Some(ref edge_var) = &step.edge_var {
            write!(f, "{edge_var}")?;
        }
        for (j, et) in step.edge_types.iter().enumerate() {
            if j == 0 {
                write!(f, ":")?;
            } else {
                write!(f, "|")?;
            }
            write!(f, "{et}")?;
        }
        write!(f, "]{}", step.direction)?;
        write!(f, "({}", step.dst_var)?;
    }
    Ok(())
}

/// Scalar function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarFunction {
    // String functions
    /// UPPER(string).
    Upper,
    /// LOWER(string).
    Lower,
    /// LENGTH(string).
    Length,
    /// CONCAT(string, ...).
    Concat,
    /// SUBSTRING(string, start, length).
    Substring,
    /// TRIM(string).
    Trim,
    /// LTRIM(string).
    Ltrim,
    /// RTRIM(string).
    Rtrim,
    /// REPLACE(string, from, to).
    Replace,
    /// POSITION(substring IN string).
    Position,
    /// CONCAT_WS(separator, string, ...).
    ConcatWs,
    /// SPLIT_PART(string, delimiter, position).
    SplitPart,
    /// FORMAT(template, args...).
    Format,
    /// REGEXP_MATCH(string, pattern).
    RegexpMatch,
    /// REGEXP_REPLACE(string, pattern, replacement).
    RegexpReplace,
    /// COALESCE(expr, ...).
    Coalesce,
    /// NULLIF(expr1, expr2).
    NullIf,

    // Numeric functions
    /// ABS(number).
    Abs,
    /// CEIL(number).
    Ceil,
    /// FLOOR(number).
    Floor,
    /// ROUND(number, precision).
    Round,
    /// TRUNC(number, precision).
    Trunc,
    /// SQRT(number).
    Sqrt,
    /// POWER(base, exponent).
    Power,
    /// EXP(number) - e^x.
    Exp,
    /// LN(number) - natural logarithm.
    Ln,
    /// LOG(base, number) - logarithm with base.
    Log,
    /// LOG10(number) - base-10 logarithm.
    Log10,
    /// SIN(number).
    Sin,
    /// COS(number).
    Cos,
    /// TAN(number).
    Tan,
    /// ASIN(number).
    Asin,
    /// ACOS(number).
    Acos,
    /// ATAN(number).
    Atan,
    /// ATAN2(y, x).
    Atan2,
    /// DEGREES(radians).
    Degrees,
    /// RADIANS(degrees).
    Radians,
    /// SIGN(number).
    Sign,
    /// PI().
    Pi,
    /// RANDOM().
    Random,

    // Date/time functions
    /// `NOW()`.
    Now,
    /// `CURRENT_DATE`.
    CurrentDate,
    /// `CURRENT_TIME`.
    CurrentTime,
    /// `EXTRACT(field FROM datetime)`.
    Extract,
    /// `DATE_PART(field, datetime)`.
    DatePart,
    /// `DATE_TRUNC(field, datetime)`.
    DateTrunc,
    /// `TO_TIMESTAMP(string, format)`.
    ToTimestamp,
    /// `TO_DATE(string, format)`.
    ToDate,
    /// `TO_CHAR(datetime, format)`.
    ToChar,

    // Vector functions
    /// `VECTOR_DIMENSION(vector)`.
    VectorDimension,
    /// `VECTOR_NORM(vector)`.
    VectorNorm,

    // List/Collection functions
    /// `RANGE(start, end)` or `RANGE(start, end, step)`.
    /// Generates a list of integers from start to end (inclusive).
    Range,
    /// `SIZE(list)` or `SIZE(string)`.
    /// Returns the length of a list or string.
    Size,
    /// `HEAD(list)`.
    /// Returns the first element of a list.
    Head,
    /// `TAIL(list)`.
    /// Returns the list without its first element.
    Tail,
    /// `LAST(list)`.
    /// Returns the last element of a list.
    Last,
    /// `REVERSE(list)`.
    /// Returns the list in reverse order.
    Reverse,

    // Array functions (PostgreSQL-compatible)
    /// `ARRAY_LENGTH(array, dimension)`.
    /// Returns the length of the array in the specified dimension.
    ArrayLength,
    /// `CARDINALITY(array)`.
    /// Returns the total number of elements in the array.
    Cardinality,
    /// `ARRAY_APPEND(array, element)`.
    /// Appends an element to the end of an array.
    ArrayAppend,
    /// `ARRAY_PREPEND(element, array)`.
    /// Prepends an element to the beginning of an array.
    ArrayPrepend,
    /// `ARRAY_CAT(array1, array2)`.
    /// Concatenates two arrays.
    ArrayCat,
    /// `ARRAY_REMOVE(array, element)`.
    /// Removes all occurrences of an element from an array.
    ArrayRemove,
    /// `ARRAY_REPLACE(array, from, to)`.
    /// Replaces all occurrences of an element with another element.
    ArrayReplace,
    /// `ARRAY_POSITION(array, element)`.
    /// Returns the index of the first occurrence of an element (1-based).
    ArrayPosition,
    /// `ARRAY_POSITIONS(array, element)`.
    /// Returns an array of indices for all occurrences of an element (1-based).
    ArrayPositions,
    /// `UNNEST(array)`.
    /// Expands an array into a set of rows.
    /// Note: This is a set-returning function but can be used as a scalar.
    Unnest,

    // JSON functions
    /// `JSON_EXTRACT_PATH(json, VARIADIC path)`.
    /// Extracts a JSON value at the given path.
    JsonExtractPath,
    /// `JSONB_EXTRACT_PATH(jsonb, VARIADIC path)`.
    /// Extracts a JSONB value at the given path.
    JsonbExtractPath,
    /// `JSON_EXTRACT_PATH_TEXT(json, VARIADIC path)`.
    /// Extracts a JSON value at the given path as text.
    JsonExtractPathText,
    /// `JSONB_EXTRACT_PATH_TEXT(jsonb, VARIADIC path)`.
    /// Extracts a JSONB value at the given path as text.
    JsonbExtractPathText,
    /// `JSON_BUILD_OBJECT(key1, val1, ...)`.
    /// Builds a JSON object from key/value pairs.
    JsonBuildObject,
    /// `JSONB_BUILD_OBJECT(key1, val1, ...)`.
    /// Builds a JSONB object from key/value pairs.
    JsonbBuildObject,
    /// `JSON_BUILD_ARRAY(val1, val2, ...)`.
    /// Builds a JSON array from values.
    JsonBuildArray,
    /// `JSONB_BUILD_ARRAY(val1, val2, ...)`.
    /// Builds a JSONB array from values.
    JsonbBuildArray,
    /// `JSONB_SET(target, path, value, create_missing)`.
    /// Sets a value at a path within a JSONB document.
    JsonbSet,
    /// `JSONB_INSERT(target, path, value, before)`.
    /// Inserts a value at a path within a JSONB document.
    JsonbInsert,
    /// `JSONB_STRIP_NULLS(jsonb)`.
    /// Removes null values from a JSONB document.
    JsonbStripNulls,

    // Cypher entity functions
    /// `TYPE(relationship)`.
    /// Returns the type (string) of a relationship.
    /// Returns NULL if the argument is not a relationship or is NULL.
    Type,
    /// `LABELS(node)`.
    /// Returns a list of labels for a node.
    /// Returns NULL if the argument is not a node or is NULL.
    Labels,
    /// `ID(entity)`.
    /// Returns the internal ID of a node or relationship.
    /// Returns NULL if the argument is NULL.
    Id,
    /// `PROPERTIES(entity)`.
    /// Returns a map of all properties of a node or relationship.
    /// Returns NULL if the argument is NULL.
    Properties,
    /// `KEYS(map_or_entity)`.
    /// Returns a list of property keys from a map or entity.
    /// Returns NULL if the argument is NULL.
    Keys,

    // Other
    /// Custom/user-defined function.
    Custom(u32), // Index into function registry
}

impl fmt::Display for ScalarFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            // String functions
            Self::Upper => "UPPER",
            Self::Lower => "LOWER",
            Self::Length => "LENGTH",
            Self::Concat => "CONCAT",
            Self::Substring => "SUBSTRING",
            Self::Trim => "TRIM",
            Self::Ltrim => "LTRIM",
            Self::Rtrim => "RTRIM",
            Self::Replace => "REPLACE",
            Self::Position => "POSITION",
            Self::ConcatWs => "CONCAT_WS",
            Self::SplitPart => "SPLIT_PART",
            Self::Format => "FORMAT",
            Self::RegexpMatch => "REGEXP_MATCH",
            Self::RegexpReplace => "REGEXP_REPLACE",
            Self::Coalesce => "COALESCE",
            Self::NullIf => "NULLIF",
            // Numeric functions
            Self::Abs => "ABS",
            Self::Ceil => "CEIL",
            Self::Floor => "FLOOR",
            Self::Round => "ROUND",
            Self::Trunc => "TRUNC",
            Self::Sqrt => "SQRT",
            Self::Power => "POWER",
            Self::Exp => "EXP",
            Self::Ln => "LN",
            Self::Log => "LOG",
            Self::Log10 => "LOG10",
            Self::Sin => "SIN",
            Self::Cos => "COS",
            Self::Tan => "TAN",
            Self::Asin => "ASIN",
            Self::Acos => "ACOS",
            Self::Atan => "ATAN",
            Self::Atan2 => "ATAN2",
            Self::Degrees => "DEGREES",
            Self::Radians => "RADIANS",
            Self::Sign => "SIGN",
            Self::Pi => "PI",
            Self::Random => "RANDOM",
            // Date/time functions
            Self::Now => "NOW",
            Self::CurrentDate => "CURRENT_DATE",
            Self::CurrentTime => "CURRENT_TIME",
            Self::Extract => "EXTRACT",
            Self::DatePart => "DATE_PART",
            Self::DateTrunc => "DATE_TRUNC",
            Self::ToTimestamp => "TO_TIMESTAMP",
            Self::ToDate => "TO_DATE",
            Self::ToChar => "TO_CHAR",
            // Vector functions
            Self::VectorDimension => "VECTOR_DIMENSION",
            Self::VectorNorm => "VECTOR_NORM",
            // List/Collection functions
            Self::Range => "RANGE",
            Self::Size => "SIZE",
            Self::Head => "HEAD",
            Self::Tail => "TAIL",
            Self::Last => "LAST",
            Self::Reverse => "REVERSE",
            // Array functions
            Self::ArrayLength => "ARRAY_LENGTH",
            Self::Cardinality => "CARDINALITY",
            Self::ArrayAppend => "ARRAY_APPEND",
            Self::ArrayPrepend => "ARRAY_PREPEND",
            Self::ArrayCat => "ARRAY_CAT",
            Self::ArrayRemove => "ARRAY_REMOVE",
            Self::ArrayReplace => "ARRAY_REPLACE",
            Self::ArrayPosition => "ARRAY_POSITION",
            Self::ArrayPositions => "ARRAY_POSITIONS",
            Self::Unnest => "UNNEST",
            // JSON functions
            Self::JsonExtractPath => "JSON_EXTRACT_PATH",
            Self::JsonbExtractPath => "JSONB_EXTRACT_PATH",
            Self::JsonExtractPathText => "JSON_EXTRACT_PATH_TEXT",
            Self::JsonbExtractPathText => "JSONB_EXTRACT_PATH_TEXT",
            Self::JsonBuildObject => "JSON_BUILD_OBJECT",
            Self::JsonbBuildObject => "JSONB_BUILD_OBJECT",
            Self::JsonBuildArray => "JSON_BUILD_ARRAY",
            Self::JsonbBuildArray => "JSONB_BUILD_ARRAY",
            Self::JsonbSet => "JSONB_SET",
            Self::JsonbInsert => "JSONB_INSERT",
            Self::JsonbStripNulls => "JSONB_STRIP_NULLS",
            // Cypher entity functions
            Self::Type => "TYPE",
            Self::Labels => "LABELS",
            Self::Id => "ID",
            Self::Properties => "PROPERTIES",
            Self::Keys => "KEYS",
            // Other
            Self::Custom(id) => return write!(f, "CUSTOM_{id}"),
        };
        write!(f, "{name}")
    }
}

/// Aggregate function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateFunction {
    /// COUNT(*) or COUNT(expr).
    Count,
    /// SUM(expr).
    Sum,
    /// AVG(expr).
    Avg,
    /// MIN(expr).
    Min,
    /// MAX(expr).
    Max,
    /// `ARRAY_AGG(expr)`.
    ArrayAgg,
    /// `STRING_AGG(expr, separator)`.
    StringAgg,
    /// Sample standard deviation (n-1 denominator).
    /// SQL: STDDEV, STDDEV_SAMP; Cypher: stDev
    StddevSamp,
    /// Population standard deviation (n denominator).
    /// SQL: STDDEV_POP; Cypher: stDevP
    StddevPop,
    /// Sample variance (n-1 denominator).
    /// SQL: VARIANCE, VAR_SAMP
    VarianceSamp,
    /// Population variance (n denominator).
    /// SQL: VAR_POP
    VariancePop,
    /// Continuous percentile (interpolates between values).
    /// Cypher: percentileCont(percentile, expr)
    PercentileCont,
    /// Discrete percentile (returns exact value from set).
    /// Cypher: percentileDisc(percentile, expr)
    PercentileDisc,
    /// `JSON_AGG(expr)`.
    /// Aggregates values into a JSON array.
    JsonAgg,
    /// `JSONB_AGG(expr)`.
    /// Aggregates values into a JSONB array (same as JSON_AGG in our implementation).
    JsonbAgg,
    /// `JSON_OBJECT_AGG(key, value)`.
    /// Aggregates key-value pairs into a JSON object.
    JsonObjectAgg,
    /// `JSONB_OBJECT_AGG(key, value)`.
    /// Aggregates key-value pairs into a JSONB object (same as JSON_OBJECT_AGG in our implementation).
    JsonbObjectAgg,
    /// Vector average.
    VectorAvg,
    /// Vector centroid.
    VectorCentroid,
}

impl fmt::Display for AggregateFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Count => "COUNT",
            Self::Sum => "SUM",
            Self::Avg => "AVG",
            Self::Min => "MIN",
            Self::Max => "MAX",
            Self::ArrayAgg => "ARRAY_AGG",
            Self::StringAgg => "STRING_AGG",
            Self::StddevSamp => "STDDEV",
            Self::StddevPop => "STDDEV_POP",
            Self::VarianceSamp => "VARIANCE",
            Self::VariancePop => "VAR_POP",
            Self::PercentileCont => "PERCENTILE_CONT",
            Self::PercentileDisc => "PERCENTILE_DISC",
            Self::JsonAgg => "JSON_AGG",
            Self::JsonbAgg => "JSONB_AGG",
            Self::JsonObjectAgg => "JSON_OBJECT_AGG",
            Self::JsonbObjectAgg => "JSONB_OBJECT_AGG",
            Self::VectorAvg => "VECTOR_AVG",
            Self::VectorCentroid => "VECTOR_CENTROID",
        };
        write!(f, "{name}")
    }
}

/// Sort order for ORDER BY expressions.
#[derive(Debug, Clone, PartialEq)]
pub struct SortOrder {
    /// The expression to sort by.
    pub expr: LogicalExpr,
    /// Whether to sort ascending (true) or descending (false).
    pub ascending: bool,
    /// Whether nulls come first.
    pub nulls_first: Option<bool>,
}

impl SortOrder {
    /// Creates an ascending sort order.
    #[must_use]
    pub fn asc(expr: LogicalExpr) -> Self {
        Self { expr, ascending: true, nulls_first: None }
    }

    /// Creates a descending sort order.
    #[must_use]
    pub fn desc(expr: LogicalExpr) -> Self {
        Self { expr, ascending: false, nulls_first: None }
    }

    /// Sets nulls first ordering.
    #[must_use]
    pub const fn nulls_first(mut self) -> Self {
        self.nulls_first = Some(true);
        self
    }

    /// Sets nulls last ordering.
    #[must_use]
    pub const fn nulls_last(mut self) -> Self {
        self.nulls_first = Some(false);
        self
    }
}

impl fmt::Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)?;
        if self.ascending {
            write!(f, " ASC")?;
        } else {
            write!(f, " DESC")?;
        }
        match self.nulls_first {
            Some(true) => write!(f, " NULLS FIRST")?,
            Some(false) => write!(f, " NULLS LAST")?,
            None => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expr_builders() {
        let expr = LogicalExpr::column("age").gt(LogicalExpr::integer(21));
        assert_eq!(expr.to_string(), "(age > 21)");
    }

    #[test]
    fn compound_expressions() {
        let expr = LogicalExpr::column("age")
            .gt(LogicalExpr::integer(18))
            .and(LogicalExpr::column("status").eq(LogicalExpr::string("active")));
        assert_eq!(expr.to_string(), "((age > 18) AND (status = 'active'))");
    }

    #[test]
    fn qualified_column() {
        let expr = LogicalExpr::qualified_column("users", "id");
        assert_eq!(expr.to_string(), "users.id");
    }

    #[test]
    fn aggregate_functions() {
        let count = LogicalExpr::count(LogicalExpr::wildcard(), false);
        assert_eq!(count.to_string(), "COUNT(*)");

        let count_distinct = LogicalExpr::count(LogicalExpr::column("id"), true);
        assert_eq!(count_distinct.to_string(), "COUNT(DISTINCT id)");

        let sum = LogicalExpr::sum(LogicalExpr::column("amount"), false);
        assert_eq!(sum.to_string(), "SUM(amount)");
    }

    #[test]
    fn contains_aggregate() {
        let simple = LogicalExpr::column("id");
        assert!(!simple.contains_aggregate());

        let agg = LogicalExpr::count(LogicalExpr::wildcard(), false);
        assert!(agg.contains_aggregate());

        let nested =
            LogicalExpr::count(LogicalExpr::wildcard(), false).add(LogicalExpr::integer(1));
        assert!(nested.contains_aggregate());
    }

    #[test]
    fn sort_order_display() {
        let asc = SortOrder::asc(LogicalExpr::column("name"));
        assert_eq!(asc.to_string(), "name ASC");

        let desc_nulls_first = SortOrder::desc(LogicalExpr::column("date")).nulls_first();
        assert_eq!(desc_nulls_first.to_string(), "date DESC NULLS FIRST");
    }

    #[test]
    fn alias_expression() {
        let expr = LogicalExpr::count(LogicalExpr::wildcard(), false).alias("total");
        assert_eq!(expr.to_string(), "COUNT(*) AS total");
        assert_eq!(expr.column_name(), Some("total"));
    }

    #[test]
    fn in_list_expression() {
        let expr = LogicalExpr::column("status")
            .in_list(vec![LogicalExpr::string("active"), LogicalExpr::string("pending")], false);
        assert_eq!(expr.to_string(), "status IN ('active', 'pending')");
    }

    #[test]
    fn between_expression() {
        let expr = LogicalExpr::column("age").between(
            LogicalExpr::integer(18),
            LogicalExpr::integer(65),
            false,
        );
        assert_eq!(expr.to_string(), "age BETWEEN 18 AND 65");
    }
}
