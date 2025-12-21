//! Graph pattern AST types.
//!
//! This module defines types for representing graph patterns used in MATCH clauses.
//! The syntax is inspired by Cypher/GQL patterns: `(a)-[r:TYPE]->(b)`.

use super::expr::{Expr, Identifier};
use std::fmt;

/// A complete graph pattern (used in MATCH clauses).
///
/// A graph pattern consists of one or more path patterns that may share nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphPattern {
    /// The path patterns in this graph pattern.
    pub paths: Vec<PathPattern>,
}

impl GraphPattern {
    /// Creates a new graph pattern with a single path.
    #[must_use]
    pub fn single(path: PathPattern) -> Self {
        Self { paths: vec![path] }
    }

    /// Creates a new graph pattern with multiple paths.
    #[must_use]
    pub const fn new(paths: Vec<PathPattern>) -> Self {
        Self { paths }
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
/// Example: `(a)-[r:FOLLOWS]->(b)-[:LIKES]->(c)`
#[derive(Debug, Clone, PartialEq)]
pub struct PathPattern {
    /// The starting node.
    pub start: NodePattern,
    /// Sequence of (edge, node) pairs forming the path.
    pub steps: Vec<(EdgePattern, NodePattern)>,
}

impl PathPattern {
    /// Creates a path pattern with just a starting node.
    #[must_use]
    pub const fn node(node: NodePattern) -> Self {
        Self {
            start: node,
            steps: vec![],
        }
    }

    /// Creates a path pattern with a node, edge, and another node.
    #[must_use]
    pub fn chain(start: NodePattern, edge: EdgePattern, end: NodePattern) -> Self {
        Self {
            start,
            steps: vec![(edge, end)],
        }
    }

    /// Extends this path with another step.
    #[must_use]
    pub fn then(mut self, edge: EdgePattern, node: NodePattern) -> Self {
        self.steps.push((edge, node));
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
/// Example: `(p:Person {name: 'Alice'})`
#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    /// Optional variable binding for this node.
    pub variable: Option<Identifier>,
    /// Optional label(s) for this node.
    pub labels: Vec<Identifier>,
    /// Optional property conditions.
    pub properties: Vec<PropertyCondition>,
}

impl NodePattern {
    /// Creates an anonymous node pattern (no variable, no labels).
    #[must_use]
    pub const fn anonymous() -> Self {
        Self {
            variable: None,
            labels: vec![],
            properties: vec![],
        }
    }

    /// Creates a node pattern with just a variable.
    #[must_use]
    pub fn var(name: impl Into<Identifier>) -> Self {
        Self {
            variable: Some(name.into()),
            labels: vec![],
            properties: vec![],
        }
    }

    /// Creates a node pattern with a variable and label.
    #[must_use]
    pub fn with_label(name: impl Into<Identifier>, label: impl Into<Identifier>) -> Self {
        Self {
            variable: Some(name.into()),
            labels: vec![label.into()],
            properties: vec![],
        }
    }

    /// Adds a label to this node pattern.
    #[must_use]
    pub fn label(mut self, label: impl Into<Identifier>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Adds a property condition to this node pattern.
    #[must_use]
    pub fn property(mut self, name: impl Into<Identifier>, value: Expr) -> Self {
        self.properties.push(PropertyCondition {
            name: name.into(),
            value,
        });
        self
    }
}

impl fmt::Display for NodePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        if let Some(var) = &self.variable {
            write!(f, "{var}")?;
        }
        for label in &self.labels {
            write!(f, ":{label}")?;
        }
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
        self.properties.push(PropertyCondition {
            name: name.into(),
            value,
        });
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
        Self::Range {
            min: Some(min),
            max: None,
        }
    }

    /// Creates a range with only a maximum.
    #[must_use]
    pub const fn at_most(max: u32) -> Self {
        Self::Range {
            min: None,
            max: Some(max),
        }
    }

    /// Creates a bounded range.
    #[must_use]
    pub const fn between(min: u32, max: u32) -> Self {
        Self::Range {
            min: Some(min),
            max: Some(max),
        }
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

        let node = NodePattern::with_label("p", "Person").label("Employee");
        assert_eq!(node.to_string(), "(p:Person:Employee)");
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

        let edge = EdgePattern::directed()
            .var("r")
            .edge_type("FOLLOWS")
            .length(EdgeLength::between(1, 3));
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
        .then(
            EdgePattern::directed().edge_type("LIKES"),
            NodePattern::var("c"),
        );

        assert_eq!(path.steps.len(), 2);
        assert_eq!(path.to_string(), "(a)-[:KNOWS]->(b)-[:LIKES]->(c)");
    }

    #[test]
    fn edge_length_variants() {
        assert_eq!(EdgeLength::Single, EdgeLength::Single);
        assert_eq!(EdgeLength::Any, EdgeLength::Any);
        assert_eq!(EdgeLength::Exact(3), EdgeLength::Exact(3));
        assert_eq!(
            EdgeLength::at_least(2),
            EdgeLength::Range {
                min: Some(2),
                max: None
            }
        );
    }
}
