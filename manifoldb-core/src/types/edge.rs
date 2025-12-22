//! Edge (relationship) types for the graph.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{EdgeId, EntityId, Value};

/// The type of an edge, describing the relationship.
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
/// Edges connect entities and have:
/// - A unique identifier
/// - A source and target entity
/// - A type describing the relationship
/// - Properties as key-value pairs
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
