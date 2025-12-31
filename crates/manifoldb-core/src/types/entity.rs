//! Entity (node) types for the graph.
//!
//! This module provides the [`Entity`] type, which represents a node in the graph
//! with optional labels, properties, and vector embeddings.
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
//!
//! # Entities with Vectors
//!
//! Entities can have vector embeddings attached for similarity search:
//!
//! ```
//! use manifoldb_core::types::{Entity, EntityId, VectorData};
//!
//! let entity = Entity::new(EntityId::new(1))
//!     .with_label("Symbol")
//!     .with_property("name", "parse_config")
//!     .with_vector("dense", vec![0.1, 0.2, 0.3, 0.4]);
//!
//! assert!(entity.has_vector("dense"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{EntityId, Value, VectorData};

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
/// - Named vector embeddings for similarity search ([`VectorData`])
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
///
/// # Entities with Vectors
///
/// ```
/// use manifoldb_core::types::{Entity, EntityId};
///
/// let symbol = Entity::new(EntityId::new(1))
///     .with_label("Symbol")
///     .with_property("name", "parse_config")
///     .with_property("language", "rust")
///     .with_vector("dense", vec![0.1; 768])
///     .with_vector("sparse", vec![(10, 0.5), (42, 0.8)]);
///
/// assert!(symbol.has_vector("dense"));
/// assert!(symbol.has_vector("sparse"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier for this entity.
    pub id: EntityId,
    /// Labels that categorize this entity.
    pub labels: Vec<Label>,
    /// Properties stored on this entity.
    pub properties: HashMap<String, Value>,
    /// Named vector embeddings for similarity search.
    ///
    /// Note: We always serialize this field (no skip_serializing_if) because
    /// bincode requires fields to be present during deserialization.
    #[serde(default)]
    pub vectors: HashMap<String, VectorData>,
}

impl Entity {
    /// Create a new entity with the given ID.
    #[must_use]
    pub fn new(id: EntityId) -> Self {
        Self { id, labels: Vec::new(), properties: HashMap::new(), vectors: HashMap::new() }
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

    /// Add a vector embedding to this entity.
    ///
    /// Entities can have multiple named vectors for different purposes
    /// (e.g., "dense" for semantic search, "sparse" for keyword matching).
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_core::types::{Entity, EntityId};
    ///
    /// let entity = Entity::new(EntityId::new(1))
    ///     .with_vector("dense", vec![0.1, 0.2, 0.3])
    ///     .with_vector("sparse", vec![(10, 0.5), (42, 0.8)]);
    /// ```
    #[must_use]
    pub fn with_vector(mut self, name: impl Into<String>, data: impl Into<VectorData>) -> Self {
        self.vectors.insert(name.into(), data.into());
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

    /// Check if this entity has a vector with the given name.
    #[inline]
    #[must_use]
    pub fn has_vector(&self, name: &str) -> bool {
        self.vectors.contains_key(name)
    }

    /// Get a vector by name.
    #[inline]
    #[must_use]
    pub fn get_vector(&self, name: &str) -> Option<&VectorData> {
        self.vectors.get(name)
    }

    /// Set a vector embedding.
    #[inline]
    pub fn set_vector(&mut self, name: impl Into<String>, data: impl Into<VectorData>) {
        self.vectors.insert(name.into(), data.into());
    }

    /// Remove a vector by name.
    ///
    /// Returns the removed vector if it existed.
    #[inline]
    pub fn remove_vector(&mut self, name: &str) -> Option<VectorData> {
        self.vectors.remove(name)
    }

    /// Returns an iterator over the entity's vectors.
    #[inline]
    pub fn vectors_iter(&self) -> impl Iterator<Item = (&str, &VectorData)> {
        self.vectors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns `true` if this entity has any vectors attached.
    #[inline]
    #[must_use]
    pub fn has_any_vectors(&self) -> bool {
        !self.vectors.is_empty()
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
        assert!(!entity.has_any_vectors());
    }

    #[test]
    fn entity_mutation() {
        let mut entity = Entity::new(EntityId::new(1));
        entity.set_property("key", "value");
        assert_eq!(entity.get_property("key"), Some(&Value::String("value".to_owned())));
    }

    #[test]
    fn entity_with_vectors() {
        let entity = Entity::new(EntityId::new(1))
            .with_label("Symbol")
            .with_property("name", "foo")
            .with_vector("dense", vec![0.1, 0.2, 0.3])
            .with_vector("sparse", vec![(10, 0.5), (42, 0.8)]);

        assert!(entity.has_vector("dense"));
        assert!(entity.has_vector("sparse"));
        assert!(!entity.has_vector("colbert"));
        assert!(entity.has_any_vectors());

        let dense = entity.get_vector("dense").unwrap();
        assert!(dense.is_dense());
        assert_eq!(dense.dimension(), Some(3));

        let sparse = entity.get_vector("sparse").unwrap();
        assert!(sparse.is_sparse());
    }

    #[test]
    fn entity_vector_mutation() {
        let mut entity = Entity::new(EntityId::new(1));
        assert!(!entity.has_any_vectors());

        entity.set_vector("embedding", vec![0.1, 0.2]);
        assert!(entity.has_vector("embedding"));

        let removed = entity.remove_vector("embedding");
        assert!(removed.is_some());
        assert!(!entity.has_vector("embedding"));
    }

    #[test]
    fn entity_vectors_iter() {
        let entity =
            Entity::new(EntityId::new(1)).with_vector("a", vec![0.1]).with_vector("b", vec![0.2]);

        let names: Vec<_> = entity.vectors_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }
}
