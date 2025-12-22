//! Entity (node) types for the graph.
//!
//! This module provides the [`Entity`] type, which represents a node in the graph
//! with optional labels and properties.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::types::{Entity, EntityId};
//!
//! let entity = Entity::new(EntityId::new(1))
//!     .with_label("Person")
//!     .with_label("Employee")
//!     .with_property("name", "Alice")
//!     .with_property("age", 30i64);
//!
//! assert!(entity.has_label("Person"));
//! assert_eq!(entity.get_property("name").and_then(|v| v.as_str()), Some("Alice"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{EntityId, Value};

/// A label that categorizes an entity.
///
/// Labels are used to group entities into categories like "Person", "Company",
/// or "Product". An entity can have multiple labels.
///
/// # Example
///
/// ```
/// use manifoldb_core::Label;
///
/// let label = Label::new("Person");
/// assert_eq!(label.as_str(), "Person");
///
/// // Also works via From trait
/// let label: Label = "Company".into();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(String);

impl Label {
    /// Create a new label.
    #[inline]
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the label name as a string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Label {
    #[inline]
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Label {
    #[inline]
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// A key-value property on an entity or edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Property {
    /// The property key.
    pub key: String,
    /// The property value.
    pub value: Value,
}

impl Property {
    /// Create a new property.
    #[inline]
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<Value>) -> Self {
        Self { key: key.into(), value: value.into() }
    }
}

/// An entity (node) in the graph.
///
/// Entities are the primary data objects in `ManifoldDB`. They can have:
/// - A unique identifier ([`EntityId`])
/// - One or more labels for categorization ([`Label`])
/// - Properties as key-value pairs ([`Value`])
///
/// # Example
///
/// ```
/// use manifoldb_core::types::{Entity, EntityId, Value};
///
/// // Create an entity with the builder pattern
/// let mut person = Entity::new(EntityId::new(1))
///     .with_label("Person")
///     .with_property("name", "Alice")
///     .with_property("email", "alice@example.com");
///
/// // Query properties
/// assert!(person.has_label("Person"));
/// assert_eq!(person.get_property("name"), Some(&Value::String("Alice".into())));
///
/// // Modify properties
/// person.set_property("verified", true);
/// assert_eq!(person.get_property("verified"), Some(&Value::Bool(true)));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity.
    pub id: EntityId,
    /// Labels that categorize this entity.
    pub labels: Vec<Label>,
    /// Properties stored on this entity.
    pub properties: HashMap<String, Value>,
}

impl Entity {
    /// Create a new entity with the given ID.
    #[must_use]
    pub fn new(id: EntityId) -> Self {
        Self { id, labels: Vec::new(), properties: HashMap::new() }
    }

    /// Add a label to this entity.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<Label>) -> Self {
        self.labels.push(label.into());
        self
    }

    /// Add a property to this entity.
    #[must_use]
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Check if this entity has a specific label.
    #[inline]
    #[must_use]
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l.as_str() == label)
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
    fn entity_builder() {
        let entity = Entity::new(EntityId::new(1))
            .with_label("Person")
            .with_property("name", "Alice")
            .with_property("age", 30i64);

        assert_eq!(entity.id.as_u64(), 1);
        assert!(entity.has_label("Person"));
        assert!(!entity.has_label("Company"));
        assert_eq!(entity.get_property("name"), Some(&Value::String("Alice".to_owned())));
        assert_eq!(entity.get_property("age"), Some(&Value::Int(30)));
    }

    #[test]
    fn entity_mutation() {
        let mut entity = Entity::new(EntityId::new(1));
        entity.set_property("key", "value");
        assert_eq!(entity.get_property("key"), Some(&Value::String("value".to_owned())));
    }
}
