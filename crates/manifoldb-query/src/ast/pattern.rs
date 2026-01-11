//! Graph pattern AST types.
//!
//! This module defines types for representing graph patterns used in MATCH clauses.
//! The syntax is inspired by Cypher/GQL patterns: `(a)-[r:TYPE]->(b)`.
//!
//! # Weighted Shortest Paths
//!
//! Path patterns can include weight specifications for shortest path algorithms:
//! - `SHORTEST PATH (a)-[*]->(b)` - unweighted BFS shortest path
//! - `SHORTEST PATH (a)-[*]->(b) WEIGHTED BY cost` - weighted Dijkstra shortest path
//! - `SHORTEST PATH (a)-[*]->(b) WEIGHTED BY distance + toll` - weighted with expression

use super::expr::{Expr, Identifier};
use std::fmt;

/// A label expression for advanced pattern matching.
///
/// Supports logical operators on labels in node patterns:
/// - `:Person|Company` - OR: matches nodes with either label
/// - `:Active&Premium` - AND: matches nodes with both labels
/// - `:!Deleted` - NOT: matches nodes without the label
/// - `:Person&!Bot` - combination: Person AND NOT Bot
///
/// # Precedence
/// NOT has highest precedence, then AND, then OR (like boolean logic).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LabelExpression {
    /// No label constraint.
    #[default]
    None,
    /// Single label: `:Person`.
    Single(Identifier),
    /// OR of labels: `:Person|Company`.
    Or(Vec<LabelExpression>),
    /// AND of labels: `:Active&Premium`.
    And(Vec<LabelExpression>),
    /// NOT label: `:!Deleted`.
    Not(Box<LabelExpression>),
}

impl LabelExpression {
    /// Creates an empty (no constraint) label expression.
    #[must_use]
    pub const fn none() -> Self {
        Self::None
    }

    /// Creates a single label expression.
    #[must_use]
    pub fn single(label: impl Into<Identifier>) -> Self {
        Self::Single(label.into())
    }

    /// Creates an OR expression from multiple label expressions.
    #[must_use]
    pub fn or(mut exprs: Vec<LabelExpression>) -> Self {
        match exprs.len() {
            0 => Self::None,
            1 => exprs.swap_remove(0),
            _ => Self::Or(exprs),
        }
    }

    /// Creates an AND expression from multiple label expressions.
    #[must_use]
    pub fn and(mut exprs: Vec<LabelExpression>) -> Self {
        match exprs.len() {
            0 => Self::None,
            1 => exprs.swap_remove(0),
            _ => Self::And(exprs),
        }
    }

    /// Creates a NOT expression.
    #[must_use]
    pub fn not(expr: LabelExpression) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Returns true if this is an empty constraint.
    #[must_use]
    pub const fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Converts a simple list of labels to an AND expression (legacy compatibility).
    /// This is the traditional Cypher behavior where `:Person:Employee` means both.
    #[must_use]
    pub fn from_labels(mut labels: Vec<Identifier>) -> Self {
        match labels.len() {
            0 => Self::None,
            1 => Self::Single(labels.swap_remove(0)),
            _ => Self::And(labels.into_iter().map(Self::Single).collect()),
        }
    }

    /// Extracts simple labels if this is a legacy-style expression (None, Single, or And of Singles).
    /// Returns None if the expression uses OR or NOT operators.
    #[must_use]
    pub fn as_simple_labels(&self) -> Option<Vec<&Identifier>> {
        match self {
            Self::None => Some(vec![]),
            Self::Single(id) => Some(vec![id]),
            Self::And(exprs) => {
                let mut labels = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    if let Self::Single(id) = expr {
                        labels.push(id);
                    } else {
                        return None;
                    }
                }
                Some(labels)
            }
            Self::Or(_) | Self::Not(_) => None,
        }
    }

    /// Extracts and clones simple labels. Returns an empty vec for complex expressions.
    /// Use this for backward compatibility with code expecting `Vec<Identifier>`.
    #[must_use]
    pub fn into_simple_labels(self) -> Vec<Identifier> {
        match self {
            Self::None => vec![],
            Self::Single(id) => vec![id],
            Self::And(exprs) => exprs
                .into_iter()
                .filter_map(|e| if let Self::Single(id) = e { Some(id) } else { None })
                .collect(),
            Self::Or(_) | Self::Not(_) => vec![],
        }
    }
}

impl fmt::Display for LabelExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => Ok(()),
            Self::Single(id) => write!(f, ":{id}"),
            Self::Or(exprs) => {
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "|")?;
                    }
                    // Don't write the leading colon for nested expressions after first
                    match expr {
                        Self::Single(id) if i > 0 => write!(f, "{id}")?,
                        Self::Single(id) => write!(f, ":{id}")?,
                        other => write!(f, "{other}")?,
                    }
                }
                Ok(())
            }
            Self::And(exprs) => {
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "&")?;
                    }
                    match expr {
                        Self::Single(id) if i > 0 => write!(f, "{id}")?,
                        Self::Single(id) => write!(f, ":{id}")?,
                        other => write!(f, "{other}")?,
                    }
                }
                Ok(())
            }
            Self::Not(expr) => write!(
                f,
                ":!{}",
                match expr.as_ref() {
                    Self::Single(id) => id.to_string(),
                    other => format!("({other})"),
                }
            ),
        }
    }
}

/// A complete graph pattern (used in MATCH clauses).
///
/// A graph pattern consists of one or more path patterns that may share nodes,
/// and optionally shortest path patterns.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPattern {
    /// The path patterns in this graph pattern.
    pub paths: Vec<PathPattern>,
    /// Shortest path patterns (from `shortestPath()` or `allShortestPaths()` functions).
    pub shortest_paths: Vec<ShortestPathPattern>,
}

impl GraphPattern {
    /// Creates a new graph pattern with a single path.
    #[must_use]
    pub fn single(path: PathPattern) -> Self {
        Self { paths: vec![path], shortest_paths: vec![] }
    }

    /// Creates a new graph pattern with multiple paths.
    #[must_use]
    pub fn new(paths: Vec<PathPattern>) -> Self {
        Self { paths, shortest_paths: vec![] }
    }

    /// Creates a graph pattern with a shortest path.
    #[must_use]
    pub fn shortest_path(sp: ShortestPathPattern) -> Self {
        Self { paths: vec![], shortest_paths: vec![sp] }
    }
}

impl fmt::Display for GraphPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, path) in self.paths.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{path}")?;
        }
        Ok(())
    }
}

/// A path pattern: a sequence of nodes connected by edges.
///
/// Example: `(a)-[r:FOLLOWS]->(b)-[:LIKES]->(c)` or `p = (a)-[r]->(b)`
#[derive(Debug, Clone, PartialEq)]
pub struct PathPattern {
    /// The starting node.
    pub start: NodePattern,
    /// Sequence of (edge, node) pairs forming the path.
    pub steps: Vec<(EdgePattern, NodePattern)>,
    /// Optional path variable name (e.g., `p` in `p = (a)-[r]->(b)`).
    pub variable: Option<Identifier>,
}

impl PathPattern {
    /// Creates a path pattern with just a starting node.
    #[must_use]
    pub fn node(node: NodePattern) -> Self {
        Self { start: node, steps: vec![], variable: None }
    }

    /// Creates a path pattern with a node, edge, and another node.
    #[must_use]
    pub fn chain(start: NodePattern, edge: EdgePattern, end: NodePattern) -> Self {
        Self { start, steps: vec![(edge, end)], variable: None }
    }

    /// Extends this path with another step.
    #[must_use]
    pub fn then(mut self, edge: EdgePattern, node: NodePattern) -> Self {
        self.steps.push((edge, node));
        self
    }

    /// Sets the path variable name.
    #[must_use]
    pub fn with_variable(mut self, name: impl Into<String>) -> Self {
        self.variable = Some(Identifier::new(name));
        self
    }
}

impl fmt::Display for PathPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.start)?;
        for (edge, node) in &self.steps {
            write!(f, "{edge}{node}")?;
        }
        Ok(())
    }
}

/// A node pattern in a graph pattern.
///
/// Example: `(p:Person {name: 'Alice'})` or `(n:Person|Company)` or `(n:Active&!Deleted)`
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    /// Optional variable binding for this node.
    pub variable: Option<Identifier>,
    /// Label expression for this node (supports OR, AND, NOT).
    pub label_expr: LabelExpression,
    /// Optional property conditions.
    pub properties: Vec<PropertyCondition>,
}

impl NodePattern {
    /// Creates an anonymous node pattern (no variable, no labels).
    #[must_use]
    pub const fn anonymous() -> Self {
        Self { variable: None, label_expr: LabelExpression::None, properties: vec![] }
    }

    /// Creates a node pattern with just a variable.
    #[must_use]
    pub fn var(name: impl Into<Identifier>) -> Self {
        Self { variable: Some(name.into()), label_expr: LabelExpression::None, properties: vec![] }
    }

    /// Creates a node pattern with a variable and label.
    #[must_use]
    pub fn with_label(name: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self {
            variable: Some(name.into()),
            label_expr: LabelExpression::single(label),
            properties: vec![],
        }
    }

    /// Creates a node pattern with a variable and label expression.
    #[must_use]
    pub fn with_label_expr(name: impl Into<Identifier>, label_expr: LabelExpression) -> Self {
        Self { variable: Some(name.into()), label_expr, properties: vec![] }
    }

    /// Adds a label to this node pattern (AND semantics, legacy compatibility).
    #[must_use]
    pub fn label(mut self, label: impl Into<Identifier>) -> Self {
        let new_label = LabelExpression::single(label);
        self.label_expr = match self.label_expr {
            LabelExpression::None => new_label,
            LabelExpression::Single(existing) => {
                LabelExpression::And(vec![LabelExpression::Single(existing), new_label])
            }
            LabelExpression::And(mut exprs) => {
                exprs.push(new_label);
                LabelExpression::And(exprs)
            }
            other => LabelExpression::And(vec![other, new_label]),
        };
        self
    }

    /// Adds a property condition to this node pattern.
    #[must_use]
    pub fn property(mut self, name: impl Into<Identifier>, value: Expr) -> Self {
        self.properties.push(PropertyCondition { name: name.into(), value });
        self
    }

    /// Returns the labels as a simple list if this is a legacy-style pattern.
    /// Returns None if the pattern uses OR or NOT operators.
    #[must_use]
    pub fn simple_labels(&self) -> Option<Vec<&Identifier>> {
        self.label_expr.as_simple_labels()
    }

    /// Returns true if this node has any label constraints.
    #[must_use]
    pub fn has_labels(&self) -> bool {
        !self.label_expr.is_none()
    }
}

impl fmt::Display for NodePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        if let Some(var) = &self.variable {
            write!(f, "{var}")?;
        }
        write!(f, "{}", self.label_expr)?;
        if !self.properties.is_empty() {
            write!(f, " {{")?;
            for (i, prop) in self.properties.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{prop}")?;
            }
            write!(f, "}}")?;
        }
        write!(f, ")")
    }
}

/// An edge pattern in a graph pattern.
///
/// Example: `-[r:FOLLOWS*1..3]->`
#[derive(Debug, Clone, PartialEq)]
pub struct EdgePattern {
    /// The direction of the edge.
    pub direction: EdgeDirection,
    /// Optional variable binding for this edge.
    pub variable: Option<Identifier>,
    /// Optional edge type(s).
    pub edge_types: Vec<Identifier>,
    /// Optional property conditions.
    pub properties: Vec<PropertyCondition>,
    /// Variable length pattern (for path traversal).
    pub length: EdgeLength,
}

impl EdgePattern {
    /// Creates an anonymous directed edge pattern (left to right).
    #[must_use]
    pub const fn directed() -> Self {
        Self {
            direction: EdgeDirection::Right,
            variable: None,
            edge_types: vec![],
            properties: vec![],
            length: EdgeLength::Single,
        }
    }

    /// Creates an anonymous undirected edge pattern.
    #[must_use]
    pub const fn undirected() -> Self {
        Self {
            direction: EdgeDirection::Undirected,
            variable: None,
            edge_types: vec![],
            properties: vec![],
            length: EdgeLength::Single,
        }
    }

    /// Creates an edge pattern with direction pointing left.
    #[must_use]
    pub const fn left() -> Self {
        Self {
            direction: EdgeDirection::Left,
            variable: None,
            edge_types: vec![],
            properties: vec![],
            length: EdgeLength::Single,
        }
    }

    /// Sets the variable for this edge pattern.
    #[must_use]
    pub fn var(mut self, name: impl Into<Identifier>) -> Self {
        self.variable = Some(name.into());
        self
    }

    /// Adds an edge type to this edge pattern.
    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<Identifier>) -> Self {
        self.edge_types.push(edge_type.into());
        self
    }

    /// Sets variable length for this edge (e.g., `*1..3`).
    #[must_use]
    pub const fn length(mut self, length: EdgeLength) -> Self {
        self.length = length;
        self
    }

    /// Adds a property condition to this edge pattern.
    #[must_use]
    pub fn property(mut self, name: impl Into<Identifier>, value: Expr) -> Self {
        self.properties.push(PropertyCondition { name: name.into(), value });
        self
    }
}

impl fmt::Display for EdgePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Opening
        match self.direction {
            EdgeDirection::Left => write!(f, "<-[")?,
            EdgeDirection::Right | EdgeDirection::Undirected => write!(f, "-[")?,
        }

        // Variable
        if let Some(var) = &self.variable {
            write!(f, "{var}")?;
        }

        // Types
        for (i, edge_type) in self.edge_types.iter().enumerate() {
            if i == 0 {
                write!(f, ":")?;
            } else {
                write!(f, "|")?;
            }
            write!(f, "{edge_type}")?;
        }

        // Length
        match &self.length {
            EdgeLength::Single => {}
            EdgeLength::Range { min, max } => {
                write!(f, "*")?;
                if let Some(min) = min {
                    write!(f, "{min}")?;
                }
                write!(f, "..")?;
                if let Some(max) = max {
                    write!(f, "{max}")?;
                }
            }
            EdgeLength::Exact(n) => write!(f, "*{n}")?,
            EdgeLength::Any => write!(f, "*")?,
        }

        // Properties
        if !self.properties.is_empty() {
            write!(f, " {{")?;
            for (i, prop) in self.properties.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{prop}")?;
            }
            write!(f, "}}")?;
        }

        // Closing
        match self.direction {
            EdgeDirection::Right => write!(f, "]->")?,
            EdgeDirection::Left | EdgeDirection::Undirected => write!(f, "]-")?,
        }

        Ok(())
    }
}

/// Direction of an edge in a pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Left-pointing edge: `<-[]-`.
    Left,
    /// Right-pointing edge: `-[]->`.
    Right,
    /// Undirected edge: `-[]-`.
    Undirected,
}

/// Variable length specification for edges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdgeLength {
    /// Single hop (no `*`).
    Single,
    /// Range of hops: `*min..max`.
    Range {
        /// Minimum number of hops (None means 0).
        min: Option<u32>,
        /// Maximum number of hops (None means unbounded).
        max: Option<u32>,
    },
    /// Exact number of hops: `*n`.
    Exact(u32),
    /// Any number of hops: `*`.
    Any,
}

impl EdgeLength {
    /// Creates a range length specification.
    #[must_use]
    pub const fn range(min: Option<u32>, max: Option<u32>) -> Self {
        Self::Range { min, max }
    }

    /// Creates a range with only a minimum.
    #[must_use]
    pub const fn at_least(min: u32) -> Self {
        Self::Range { min: Some(min), max: None }
    }

    /// Creates a range with only a maximum.
    #[must_use]
    pub const fn at_most(max: u32) -> Self {
        Self::Range { min: None, max: Some(max) }
    }

    /// Creates a bounded range.
    #[must_use]
    pub const fn between(min: u32, max: u32) -> Self {
        Self::Range { min: Some(min), max: Some(max) }
    }
}

/// A property condition in a pattern.
///
/// Example: `name: 'Alice'`
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyCondition {
    /// The property name.
    pub name: Identifier,
    /// The expected value (expression).
    pub value: Expr,
}

impl fmt::Display for PropertyCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // For now, just show the property name (value display would need Expr::Display)
        write!(f, "{}: ...", self.name)
    }
}

/// A shortest path pattern.
///
/// Represents a query for the shortest path between two nodes, optionally weighted.
///
/// # Examples
///
/// ```text
/// SHORTEST PATH (a)-[*]->(b)              -- unweighted BFS
/// SHORTEST PATH (a)-[*]->(b) WEIGHTED BY cost  -- weighted Dijkstra
/// ALL SHORTEST PATHS (a)-[*]->(b)         -- all equal-length shortest paths
/// p = shortestPath((a)-[*..10]->(b))      -- Cypher-style function syntax
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestPathPattern {
    /// The path pattern defining the traversal.
    pub path: PathPattern,
    /// Whether to find all shortest paths or just one.
    pub find_all: bool,
    /// Optional weight specification for weighted shortest path.
    pub weight: Option<WeightSpec>,
    /// Optional path variable name (e.g., `p` in `p = shortestPath(...)`).
    pub path_variable: Option<String>,
}

impl ShortestPathPattern {
    /// Creates a new shortest path pattern.
    #[must_use]
    pub fn new(path: PathPattern) -> Self {
        Self { path, find_all: false, weight: None, path_variable: None }
    }

    /// Creates a pattern that finds all shortest paths.
    #[must_use]
    pub fn all(path: PathPattern) -> Self {
        Self { path, find_all: true, weight: None, path_variable: None }
    }

    /// Sets the path variable name.
    #[must_use]
    pub fn with_path_variable(mut self, name: impl Into<String>) -> Self {
        self.path_variable = Some(name.into());
        self
    }

    /// Adds a weight specification (makes it use Dijkstra's algorithm).
    #[must_use]
    pub fn weighted_by(mut self, weight: WeightSpec) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Adds a simple property weight (e.g., `WEIGHTED BY cost`).
    #[must_use]
    pub fn weighted_by_property(self, property: impl Into<String>) -> Self {
        self.weighted_by(WeightSpec::property(property))
    }

    /// Returns true if this is a weighted shortest path query.
    #[must_use]
    pub const fn is_weighted(&self) -> bool {
        self.weight.is_some()
    }
}

impl fmt::Display for ShortestPathPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.find_all {
            write!(f, "ALL SHORTEST PATHS ")?;
        } else {
            write!(f, "SHORTEST PATH ")?;
        }
        write!(f, "{}", self.path)?;
        if let Some(ref weight) = self.weight {
            write!(f, " {weight}")?;
        }
        Ok(())
    }
}

/// Weight specification for weighted shortest path algorithms.
///
/// Specifies how edge weights should be calculated for Dijkstra's algorithm.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightSpec {
    /// Use a single edge property as the weight.
    ///
    /// Example: `WEIGHTED BY cost` uses the "cost" property on edges.
    Property {
        /// The name of the edge property containing the weight.
        name: String,
        /// Default weight to use if the property is missing.
        default: Option<f64>,
    },
    /// Use a constant weight for all edges.
    ///
    /// Example: `WEIGHTED BY 1.0` gives all edges weight 1.0.
    Constant(f64),
    /// Use an expression to calculate the weight.
    ///
    /// Example: `WEIGHTED BY distance + toll * 0.5`
    /// The expression can reference edge properties.
    Expression(Expr),
}

impl WeightSpec {
    /// Creates a weight spec using an edge property.
    #[must_use]
    pub fn property(name: impl Into<String>) -> Self {
        Self::Property { name: name.into(), default: None }
    }

    /// Creates a weight spec using an edge property with a default value.
    #[must_use]
    pub fn property_with_default(name: impl Into<String>, default: f64) -> Self {
        Self::Property { name: name.into(), default: Some(default) }
    }

    /// Creates a constant weight spec.
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Creates a weight spec from an expression.
    #[must_use]
    pub const fn expression(expr: Expr) -> Self {
        Self::Expression(expr)
    }
}

impl fmt::Display for WeightSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WEIGHTED BY ")?;
        match self {
            Self::Property { name, default } => {
                write!(f, "{name}")?;
                if let Some(d) = default {
                    write!(f, " DEFAULT {d}")?;
                }
            }
            Self::Constant(v) => write!(f, "{v}")?,
            Self::Expression(_) => write!(f, "<expr>")?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_pattern_display() {
        let node = NodePattern::anonymous();
        assert_eq!(node.to_string(), "()");

        let node = NodePattern::var("p");
        assert_eq!(node.to_string(), "(p)");

        let node = NodePattern::with_label("p", "Person");
        assert_eq!(node.to_string(), "(p:Person)");

        // Multiple labels with AND semantics (traditional Cypher behavior)
        let node = NodePattern::with_label("p", "Person").label("Employee");
        assert_eq!(node.to_string(), "(p:Person&Employee)");
    }

    #[test]
    fn label_expression_display() {
        // Single label
        let expr = LabelExpression::single("Person");
        assert_eq!(expr.to_string(), ":Person");

        // OR labels
        let expr = LabelExpression::or(vec![
            LabelExpression::single("Person"),
            LabelExpression::single("Company"),
        ]);
        assert_eq!(expr.to_string(), ":Person|Company");

        // AND labels
        let expr = LabelExpression::and(vec![
            LabelExpression::single("Active"),
            LabelExpression::single("Premium"),
        ]);
        assert_eq!(expr.to_string(), ":Active&Premium");

        // NOT label
        let expr = LabelExpression::not(LabelExpression::single("Deleted"));
        assert_eq!(expr.to_string(), ":!Deleted");

        // Combined: Person AND NOT Bot
        let expr = LabelExpression::and(vec![
            LabelExpression::single("Person"),
            LabelExpression::not(LabelExpression::single("Bot")),
        ]);
        assert_eq!(expr.to_string(), ":Person&:!Bot");
    }

    #[test]
    fn label_expression_from_labels() {
        // Empty labels
        let expr = LabelExpression::from_labels(vec![]);
        assert!(expr.is_none());

        // Single label
        let expr = LabelExpression::from_labels(vec![Identifier::new("Person")]);
        assert_eq!(expr, LabelExpression::single("Person"));

        // Multiple labels (AND semantics)
        let expr = LabelExpression::from_labels(vec![
            Identifier::new("Person"),
            Identifier::new("Employee"),
        ]);
        match expr {
            LabelExpression::And(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn label_expression_as_simple_labels() {
        // None
        let expr = LabelExpression::None;
        assert_eq!(expr.as_simple_labels(), Some(vec![]));

        // Single
        let id = Identifier::new("Person");
        let expr = LabelExpression::Single(id.clone());
        assert_eq!(expr.as_simple_labels(), Some(vec![&id]));

        // And of singles
        let id1 = Identifier::new("Person");
        let id2 = Identifier::new("Employee");
        let expr = LabelExpression::And(vec![
            LabelExpression::Single(id1.clone()),
            LabelExpression::Single(id2.clone()),
        ]);
        assert_eq!(expr.as_simple_labels(), Some(vec![&id1, &id2]));

        // Or - not simple
        let expr = LabelExpression::or(vec![
            LabelExpression::single("Person"),
            LabelExpression::single("Company"),
        ]);
        assert_eq!(expr.as_simple_labels(), None);

        // Not - not simple
        let expr = LabelExpression::not(LabelExpression::single("Deleted"));
        assert_eq!(expr.as_simple_labels(), None);
    }

    #[test]
    fn edge_pattern_display() {
        let edge = EdgePattern::directed();
        assert_eq!(edge.to_string(), "-[]->");

        let edge = EdgePattern::left();
        assert_eq!(edge.to_string(), "<-[]-");

        let edge = EdgePattern::undirected();
        assert_eq!(edge.to_string(), "-[]-");

        let edge = EdgePattern::directed().edge_type("FOLLOWS");
        assert_eq!(edge.to_string(), "-[:FOLLOWS]->");

        let edge =
            EdgePattern::directed().var("r").edge_type("FOLLOWS").length(EdgeLength::between(1, 3));
        assert_eq!(edge.to_string(), "-[r:FOLLOWS*1..3]->");
    }

    #[test]
    fn path_pattern_display() {
        let path = PathPattern::chain(
            NodePattern::with_label("a", "Person"),
            EdgePattern::directed().edge_type("FOLLOWS"),
            NodePattern::with_label("b", "Person"),
        );
        assert_eq!(path.to_string(), "(a:Person)-[:FOLLOWS]->(b:Person)");
    }

    #[test]
    fn path_pattern_chaining() {
        let path = PathPattern::chain(
            NodePattern::var("a"),
            EdgePattern::directed().edge_type("KNOWS"),
            NodePattern::var("b"),
        )
        .then(EdgePattern::directed().edge_type("LIKES"), NodePattern::var("c"));

        assert_eq!(path.steps.len(), 2);
        assert_eq!(path.to_string(), "(a)-[:KNOWS]->(b)-[:LIKES]->(c)");
    }

    #[test]
    fn edge_length_variants() {
        assert_eq!(EdgeLength::Single, EdgeLength::Single);
        assert_eq!(EdgeLength::Any, EdgeLength::Any);
        assert_eq!(EdgeLength::Exact(3), EdgeLength::Exact(3));
        assert_eq!(EdgeLength::at_least(2), EdgeLength::Range { min: Some(2), max: None });
    }

    #[test]
    fn shortest_path_pattern_display() {
        let path = PathPattern::chain(
            NodePattern::var("a"),
            EdgePattern::directed().length(EdgeLength::Any),
            NodePattern::var("b"),
        );

        let sp = ShortestPathPattern::new(path.clone());
        assert_eq!(sp.to_string(), "SHORTEST PATH (a)-[*]->(b)");
        assert!(!sp.is_weighted());

        let sp = ShortestPathPattern::all(path.clone());
        assert_eq!(sp.to_string(), "ALL SHORTEST PATHS (a)-[*]->(b)");

        let sp = ShortestPathPattern::new(path).weighted_by_property("cost");
        assert!(sp.is_weighted());
        assert!(sp.to_string().contains("WEIGHTED BY cost"));
    }

    #[test]
    fn weight_spec_variants() {
        let ws = WeightSpec::property("cost");
        assert_eq!(ws.to_string(), "WEIGHTED BY cost");

        let ws = WeightSpec::property_with_default("cost", 1.0);
        assert_eq!(ws.to_string(), "WEIGHTED BY cost DEFAULT 1");

        let ws = WeightSpec::constant(2.5);
        assert_eq!(ws.to_string(), "WEIGHTED BY 2.5");
    }
}
