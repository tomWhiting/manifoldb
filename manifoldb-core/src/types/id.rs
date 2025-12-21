//! Unique identifiers for entities and edges.

use serde::{Deserialize, Serialize};

/// Unique identifier for an entity (node) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityId(u64);

impl EntityId {
    /// Create a new `EntityId` from a raw u64 value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for EntityId {
    fn from(id: u64) -> Self {
        Self::new(id)
    }
}

/// Unique identifier for an edge (relationship) in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EdgeId(u64);

impl EdgeId {
    /// Create a new `EdgeId` from a raw u64 value.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for EdgeId {
    fn from(id: u64) -> Self {
        Self::new(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_roundtrip() {
        let id = EntityId::new(42);
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn edge_id_roundtrip() {
        let id = EdgeId::new(123);
        assert_eq!(id.as_u64(), 123);
    }

    #[test]
    fn ids_are_ordered() {
        let a = EntityId::new(1);
        let b = EntityId::new(2);
        assert!(a < b);
    }
}
