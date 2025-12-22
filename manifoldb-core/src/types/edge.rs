//! Edge (relationship) types for the graph.
//!
//! This module provides the [`Edge`] type, which represents a directed relationship
//! between two entities in the graph.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::types::{Edge, EdgeId, EntityId};
//!
//! let alice = EntityId::new(1);
//! let bob = EntityId::new(2);
//!
//! // Create an edge from Alice to Bob
//! let follows = Edge::new(EdgeId::new(1), alice, bob, "FOLLOWS")
//!     .with_property("since", "2024-01-01")
//!     .with_property("close_friend", true);
//!
//! assert_eq!(follows.edge_type.as_str(), "FOLLOWS");
//! assert_eq!(follows.source, alice);
//! assert_eq!(follows.target, bob);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{EdgeId, EntityId, Value};

/// The type of an edge, describing the relationship.
///
/// Edge types categorize relationships, such as "FOLLOWS", "LIKES", "WORKS_AT",
/// or "PURCHASED". They are typically written in `SCREAMING_SNAKE_CASE` by convention.
///
/// # Example
///
/// ```
/// use manifoldb_core::EdgeType;
///
/// let edge_type = EdgeType::new("FOLLOWS");
/// assert_eq!(edge_type.as_str(), "FOLLOWS");
///
/// // Also works via From trait
/// let edge_type: EdgeType = "WORKS_AT".into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeType(String);

impl EdgeType {
    /// Create a new edge type.
    #[inline]
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the edge type name as a string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for EdgeType {
    #[inline]
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for EdgeType {
    #[inline]
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// An edge (relationship) between two entities in the graph.
///
/// Edges are directed relationships connecting a source entity to a target entity.
/// They have:
/// - A unique identifier ([`EdgeId`])
/// - A source and target entity ([`EntityId`])
/// - A type describing the relationship ([`EdgeType`])
/// - Properties as key-value pairs ([`Value`])
///
/// # Example
///
/// ```
/// use manifoldb_core::types::{Edge, EdgeId, EntityId, Value};
///
/// let user_id = EntityId::new(1);
/// let product_id = EntityId::new(100);
///
/// // Create a purchase relationship with properties
/// let purchased = Edge::new(EdgeId::new(1), user_id, product_id, "PURCHASED")
///     .with_property("quantity", 2i64)
///     .with_property("price", 29.99f64)
///     .with_property("timestamp", "2024-01-15T10:30:00Z");
///
/// assert_eq!(purchased.source, user_id);
/// assert_eq!(purchased.target, product_id);
/// assert_eq!(purchased.get_property("quantity"), Some(&Value::Int(2)));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier for this edge.
    pub id: EdgeId,
    /// The source entity ID.
    pub source: EntityId,
    /// The target entity ID.
    pub target: EntityId,
    /// The type of this edge/relationship.
    pub edge_type: EdgeType,
    /// Properties stored on this edge.
    pub properties: HashMap<String, Value>,
}

impl Edge {
    /// Create a new edge between two entities.
    #[must_use]
    pub fn new(
        id: EdgeId,
        source: EntityId,
        target: EntityId,
        edge_type: impl Into<EdgeType>,
    ) -> Self {
        Self { id, source, target, edge_type: edge_type.into(), properties: HashMap::new() }
    }

    /// Add a property to this edge.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Get a property value by key.
    #[inline]
    #[must_use]
    pub fn get_property(&self, key: &str) -> Option<&Value> {
        self.properties.get(key)
    }

    /// Set a property value.
    #[inline]
    pub fn set_property(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.properties.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_creation() {
        let edge = Edge::new(EdgeId::new(1), EntityId::new(10), EntityId::new(20), "FOLLOWS")
            .with_property("since", "2024-01-01");

        assert_eq!(edge.id.as_u64(), 1);
        assert_eq!(edge.source.as_u64(), 10);
        assert_eq!(edge.target.as_u64(), 20);
        assert_eq!(edge.edge_type.as_str(), "FOLLOWS");
        assert_eq!(edge.get_property("since"), Some(&Value::String("2024-01-01".to_owned())));
    }
}
