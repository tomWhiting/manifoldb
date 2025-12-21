//! Graph-specific plan nodes.
//!
//! This module defines plan nodes for graph traversal operations:
//! Expand (single-hop traversal) and `PathScan` (multi-hop patterns).

// Allow missing_const_for_fn - const fn with Vec isn't stable
#![allow(clippy::missing_const_for_fn)]

use super::expr::LogicalExpr;
use crate::ast::EdgeLength;

/// Direction of graph edge expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandDirection {
    /// Outgoing edges (->).
    Outgoing,
    /// Incoming edges (<-).
    Incoming,
    /// Both directions (-).
    Both,
}

impl std::fmt::Display for ExpandDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Outgoing => "->",
            Self::Incoming => "<-",
            Self::Both => "-",
        };
        write!(f, "{s}")
    }
}

/// An expand node for single-hop graph traversal.
///
/// Expands from input nodes along edges to find connected nodes.
///
/// # Example
///
/// For the pattern `(a)-[:FOLLOWS]->(b)`, this node:
/// - Takes node `a` as input
/// - Traverses outgoing FOLLOWS edges
/// - Outputs node `b` (and optionally edge `r`)
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandNode {
    /// The direction of expansion.
    pub direction: ExpandDirection,

    /// The source node variable (from input).
    pub src_var: String,

    /// The destination node variable (output).
    pub dst_var: String,

    /// Optional edge variable (for binding edge properties).
    pub edge_var: Option<String>,

    /// Edge type filter (e.g., "FOLLOWS", "LIKES").
    /// Empty means any edge type.
    pub edge_types: Vec<String>,

    /// Variable length specification.
    pub length: ExpandLength,

    /// Optional property filter on edges.
    pub edge_filter: Option<LogicalExpr>,

    /// Optional property filter on destination nodes.
    pub node_filter: Option<LogicalExpr>,

    /// Optional label filter on destination nodes.
    pub node_labels: Vec<String>,
}

impl ExpandNode {
    /// Creates a new single-hop expand node.
    #[must_use]
    pub fn new(
        src_var: impl Into<String>,
        dst_var: impl Into<String>,
        direction: ExpandDirection,
    ) -> Self {
        Self {
            direction,
            src_var: src_var.into(),
            dst_var: dst_var.into(),
            edge_var: None,
            edge_types: vec![],
            length: ExpandLength::Single,
            edge_filter: None,
            node_filter: None,
            node_labels: vec![],
        }
    }

    /// Creates an outgoing expand.
    #[must_use]
    pub fn outgoing(src_var: impl Into<String>, dst_var: impl Into<String>) -> Self {
        Self::new(src_var, dst_var, ExpandDirection::Outgoing)
    }

    /// Creates an incoming expand.
    #[must_use]
    pub fn incoming(src_var: impl Into<String>, dst_var: impl Into<String>) -> Self {
        Self::new(src_var, dst_var, ExpandDirection::Incoming)
    }

    /// Creates a bidirectional expand.
    #[must_use]
    pub fn both(src_var: impl Into<String>, dst_var: impl Into<String>) -> Self {
        Self::new(src_var, dst_var, ExpandDirection::Both)
    }

    /// Sets the edge variable.
    #[must_use]
    pub fn with_edge_var(mut self, var: impl Into<String>) -> Self {
        self.edge_var = Some(var.into());
        self
    }

    /// Adds edge type filters.
    #[must_use]
    pub fn with_edge_types(mut self, types: Vec<String>) -> Self {
        self.edge_types = types;
        self
    }

    /// Adds a single edge type filter.
    #[must_use]
    pub fn with_edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_types.push(edge_type.into());
        self
    }

    /// Sets variable length expansion.
    #[must_use]
    pub const fn with_length(mut self, length: ExpandLength) -> Self {
        self.length = length;
        self
    }

    /// Sets edge property filter.
    #[must_use]
    pub fn with_edge_filter(mut self, filter: LogicalExpr) -> Self {
        self.edge_filter = Some(filter);
        self
    }

    /// Sets node property filter.
    #[must_use]
    pub fn with_node_filter(mut self, filter: LogicalExpr) -> Self {
        self.node_filter = Some(filter);
        self
    }

    /// Sets node label filter.
    #[must_use]
    pub fn with_node_labels(mut self, labels: Vec<String>) -> Self {
        self.node_labels = labels;
        self
    }

    /// Returns true if this is a variable-length expansion.
    #[must_use]
    pub fn is_variable_length(&self) -> bool {
        !matches!(self.length, ExpandLength::Single)
    }
}

/// Length specification for graph expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpandLength {
    /// Single hop (no variable length).
    Single,

    /// Variable length with bounds.
    Range {
        /// Minimum number of hops.
        min: usize,
        /// Maximum number of hops (None means unbounded).
        max: Option<usize>,
    },

    /// Exact number of hops.
    Exact(usize),
}

impl ExpandLength {
    /// Creates a range length.
    #[must_use]
    pub const fn range(min: usize, max: Option<usize>) -> Self {
        Self::Range { min, max }
    }

    /// Creates an "at least N" length.
    #[must_use]
    pub const fn at_least(min: usize) -> Self {
        Self::Range { min, max: None }
    }

    /// Creates an "at most N" length.
    #[must_use]
    pub const fn at_most(max: usize) -> Self {
        Self::Range {
            min: 0,
            max: Some(max),
        }
    }

    /// Creates a bounded range length.
    #[must_use]
    pub const fn between(min: usize, max: usize) -> Self {
        Self::Range {
            min,
            max: Some(max),
        }
    }

    /// Creates from AST `EdgeLength`.
    #[must_use]
    pub fn from_ast(length: &EdgeLength) -> Self {
        match length {
            EdgeLength::Single => Self::Single,
            EdgeLength::Exact(n) => Self::Exact(*n as usize),
            EdgeLength::Any => Self::Range { min: 0, max: None },
            EdgeLength::Range { min, max } => Self::Range {
                min: min.unwrap_or(0) as usize,
                max: max.map(|m| m as usize),
            },
        }
    }
}

impl std::fmt::Display for ExpandLength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single => write!(f, ""),
            Self::Exact(n) => write!(f, "*{n}"),
            Self::Range { min, max: None } => write!(f, "*{min}.."),
            Self::Range {
                min,
                max: Some(max),
            } => write!(f, "*{min}..{max}"),
        }
    }
}

/// A path scan node for complex graph pattern matching.
///
/// Matches an entire path pattern from the MATCH clause.
///
/// # Example
///
/// For the pattern `(a:Person)-[:KNOWS*1..3]->(b:Person)-[:WORKS_AT]->(c:Company)`:
/// - Starts from nodes matching `a:Person`
/// - Traverses 1-3 `KNOWS` edges
/// - Finds intermediate Person nodes
/// - Traverses one `WORKS_AT` edge
/// - Ends at Company nodes
#[derive(Debug, Clone, PartialEq)]
pub struct PathScanNode {
    /// The steps in the path pattern.
    pub steps: Vec<PathStep>,

    /// Optional starting node filter.
    pub start_filter: Option<LogicalExpr>,

    /// Whether to return all paths or just distinct end nodes.
    pub all_paths: bool,

    /// Whether to track the path (for path expressions).
    pub track_path: bool,
}

impl PathScanNode {
    /// Creates a new path scan with the given steps.
    #[must_use]
    pub const fn new(steps: Vec<PathStep>) -> Self {
        Self {
            steps,
            start_filter: None,
            all_paths: false,
            track_path: false,
        }
    }

    /// Sets the starting node filter.
    #[must_use]
    pub fn with_start_filter(mut self, filter: LogicalExpr) -> Self {
        self.start_filter = Some(filter);
        self
    }

    /// Enables returning all paths.
    #[must_use]
    pub const fn all_paths(mut self) -> Self {
        self.all_paths = true;
        self
    }

    /// Enables path tracking.
    #[must_use]
    pub const fn track_path(mut self) -> Self {
        self.track_path = true;
        self
    }
}

/// A single step in a path pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct PathStep {
    /// The expand operation for this step.
    pub expand: ExpandNode,

    /// Whether this step is optional (for shortest path queries).
    pub optional: bool,
}

impl PathStep {
    /// Creates a new required path step.
    #[must_use]
    pub const fn required(expand: ExpandNode) -> Self {
        Self {
            expand,
            optional: false,
        }
    }

    /// Creates a new optional path step.
    #[must_use]
    pub const fn optional(expand: ExpandNode) -> Self {
        Self {
            expand,
            optional: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_node_basic() {
        let expand = ExpandNode::outgoing("a", "b")
            .with_edge_type("FOLLOWS")
            .with_edge_var("r");

        assert_eq!(expand.src_var, "a");
        assert_eq!(expand.dst_var, "b");
        assert_eq!(expand.edge_var.as_deref(), Some("r"));
        assert_eq!(expand.edge_types, vec!["FOLLOWS"]);
        assert_eq!(expand.direction, ExpandDirection::Outgoing);
    }

    #[test]
    fn expand_variable_length() {
        let expand = ExpandNode::outgoing("a", "b")
            .with_edge_type("KNOWS")
            .with_length(ExpandLength::between(1, 3));

        assert!(expand.is_variable_length());
        assert_eq!(expand.length, ExpandLength::Range { min: 1, max: Some(3) });
    }

    #[test]
    fn expand_length_display() {
        assert_eq!(ExpandLength::Single.to_string(), "");
        assert_eq!(ExpandLength::Exact(3).to_string(), "*3");
        assert_eq!(ExpandLength::at_least(2).to_string(), "*2..");
        assert_eq!(ExpandLength::between(1, 5).to_string(), "*1..5");
    }

    #[test]
    fn path_scan_basic() {
        let path = PathScanNode::new(vec![
            PathStep::required(
                ExpandNode::outgoing("a", "b")
                    .with_edge_type("KNOWS")
                    .with_length(ExpandLength::between(1, 3)),
            ),
            PathStep::required(
                ExpandNode::outgoing("b", "c").with_edge_type("WORKS_AT"),
            ),
        ])
        .track_path();

        assert_eq!(path.steps.len(), 2);
        assert!(path.track_path);
    }

    #[test]
    fn direction_display() {
        assert_eq!(ExpandDirection::Outgoing.to_string(), "->");
        assert_eq!(ExpandDirection::Incoming.to_string(), "<-");
        assert_eq!(ExpandDirection::Both.to_string(), "-");
    }
}
