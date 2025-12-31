//! Scored entity types for search results.
//!
//! This module provides [`ScoredEntity`] which wraps an entity with its
//! similarity score from vector search operations.

use serde::{Deserialize, Serialize};

use super::{Entity, EntityId};

/// An entity with an associated similarity score from vector search.
///
/// This is the return type for vector similarity searches. The score
/// represents how similar the entity's vector is to the query vector,
/// with higher scores indicating greater similarity.
///
/// # Example
///
/// ```
/// use manifoldb_core::types::{Entity, EntityId, ScoredEntity};
///
/// let entity = Entity::new(EntityId::new(1))
///     .with_label("Document")
///     .with_property("title", "Hello World");
///
/// let scored = ScoredEntity::new(entity, 0.95);
/// assert_eq!(scored.score, 0.95);
/// assert!(scored.entity.has_label("Document"));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredEntity {
    /// The entity returned from the search.
    pub entity: Entity,
    /// The similarity score (higher = more similar).
    pub score: f32,
}

impl ScoredEntity {
    /// Create a new scored entity.
    #[inline]
    #[must_use]
    pub const fn new(entity: Entity, score: f32) -> Self {
        Self { entity, score }
    }

    /// Get the entity ID.
    #[inline]
    #[must_use]
    pub fn id(&self) -> EntityId {
        self.entity.id
    }

    /// Get a reference to the underlying entity.
    #[inline]
    #[must_use]
    pub const fn entity(&self) -> &Entity {
        &self.entity
    }

    /// Consume self and return the underlying entity.
    #[inline]
    #[must_use]
    pub fn into_entity(self) -> Entity {
        self.entity
    }
}

/// A lightweight scored reference containing just the ID and score.
///
/// This is useful for intermediate search results before fetching
/// full entity data. It avoids loading entity properties until needed.
///
/// # Example
///
/// ```
/// use manifoldb_core::types::{EntityId, ScoredId};
///
/// let result = ScoredId::new(EntityId::new(42), 0.87);
/// assert_eq!(result.id.as_u64(), 42);
/// assert_eq!(result.score, 0.87);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoredId {
    /// The entity ID.
    pub id: EntityId,
    /// The similarity score.
    pub score: f32,
}

impl ScoredId {
    /// Create a new scored ID.
    #[inline]
    #[must_use]
    pub const fn new(id: EntityId, score: f32) -> Self {
        Self { id, score }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_entity_basic() {
        let entity = Entity::new(EntityId::new(1)).with_label("Test").with_property("name", "foo");

        let scored = ScoredEntity::new(entity, 0.95);

        assert_eq!(scored.id().as_u64(), 1);
        assert!((scored.score - 0.95).abs() < 0.001);
        assert!(scored.entity().has_label("Test"));
    }

    #[test]
    fn scored_entity_into_entity() {
        let entity = Entity::new(EntityId::new(1)).with_label("Test");
        let scored = ScoredEntity::new(entity, 0.5);
        let recovered = scored.into_entity();
        assert!(recovered.has_label("Test"));
    }

    #[test]
    fn scored_id_basic() {
        let scored = ScoredId::new(EntityId::new(42), 0.87);
        assert_eq!(scored.id.as_u64(), 42);
        assert!((scored.score - 0.87).abs() < 0.001);
    }
}
