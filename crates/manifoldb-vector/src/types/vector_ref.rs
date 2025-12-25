//! Vector reference types for entity-to-vector relationships.
//!
//! This module provides types for referencing vectors attached to entities.

use manifoldb_core::EntityId;
use serde::{Deserialize, Serialize};

/// Reference to a vector attached to an entity.
///
/// This is the primary addressing scheme for vectors. Using `(EntityId, vector_name)`
/// instead of a separate `VectorId` provides simpler addressing that matches
/// query patterns ("find similar entities") and natural cascade delete semantics.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::VectorRef;
/// use manifoldb_core::EntityId;
///
/// let vector_ref = VectorRef::new(EntityId::new(42), "text_embedding");
/// assert_eq!(vector_ref.entity_id(), EntityId::new(42));
/// assert_eq!(vector_ref.vector_name(), "text_embedding");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorRef {
    /// The entity this vector belongs to.
    entity_id: EntityId,
    /// The named vector space (e.g., "text_embedding", "image_embedding").
    vector_name: String,
}

impl VectorRef {
    /// Create a new vector reference.
    #[must_use]
    pub fn new(entity_id: EntityId, vector_name: impl Into<String>) -> Self {
        Self { entity_id, vector_name: vector_name.into() }
    }

    /// Get the entity ID this vector belongs to.
    #[must_use]
    pub fn entity_id(&self) -> EntityId {
        self.entity_id
    }

    /// Get the vector name.
    #[must_use]
    pub fn vector_name(&self) -> &str {
        &self.vector_name
    }

    /// Consume self and return the entity ID and vector name.
    #[must_use]
    pub fn into_parts(self) -> (EntityId, String) {
        (self.entity_id, self.vector_name)
    }
}

impl std::fmt::Display for VectorRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.entity_id, self.vector_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_ref_creation() {
        let entity_id = EntityId::new(42);
        let vector_ref = VectorRef::new(entity_id, "text_embedding");

        assert_eq!(vector_ref.entity_id(), entity_id);
        assert_eq!(vector_ref.vector_name(), "text_embedding");
    }

    #[test]
    fn test_vector_ref_display() {
        let vector_ref = VectorRef::new(EntityId::new(42), "text");
        assert_eq!(format!("{}", vector_ref), "EntityId(42):text");
    }

    #[test]
    fn test_vector_ref_into_parts() {
        let vector_ref = VectorRef::new(EntityId::new(42), "text");
        let (entity_id, name) = vector_ref.into_parts();
        assert_eq!(entity_id, EntityId::new(42));
        assert_eq!(name, "text");
    }

    #[test]
    fn test_vector_ref_equality() {
        let ref1 = VectorRef::new(EntityId::new(1), "text");
        let ref2 = VectorRef::new(EntityId::new(1), "text");
        let ref3 = VectorRef::new(EntityId::new(2), "text");
        let ref4 = VectorRef::new(EntityId::new(1), "image");

        assert_eq!(ref1, ref2);
        assert_ne!(ref1, ref3);
        assert_ne!(ref1, ref4);
    }

    #[test]
    fn test_vector_ref_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(VectorRef::new(EntityId::new(1), "text"));
        set.insert(VectorRef::new(EntityId::new(1), "text")); // duplicate
        set.insert(VectorRef::new(EntityId::new(2), "text"));

        assert_eq!(set.len(), 2);
    }
}
