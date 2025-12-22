//! Result types for transaction operations.

use crate::EdgeId;

/// Result of a cascade delete operation.
///
/// This struct contains information about what was deleted when performing
/// a cascade delete of an entity and its connected edges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeleteResult {
    /// Whether the entity existed and was deleted.
    pub entity_deleted: bool,

    /// The IDs of edges that were deleted.
    ///
    /// This includes both incoming and outgoing edges that were
    /// connected to the deleted entity.
    pub edges_deleted: Vec<EdgeId>,
}

impl DeleteResult {
    /// Creates a new `DeleteResult`.
    #[must_use]
    pub const fn new(entity_deleted: bool, edges_deleted: Vec<EdgeId>) -> Self {
        Self { entity_deleted, edges_deleted }
    }

    /// Returns `true` if the entity was deleted.
    #[must_use]
    pub const fn entity_deleted(&self) -> bool {
        self.entity_deleted
    }

    /// Returns the number of edges that were deleted.
    #[must_use]
    pub fn edges_deleted_count(&self) -> usize {
        self.edges_deleted.len()
    }

    /// Returns `true` if nothing was deleted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.entity_deleted && self.edges_deleted.is_empty()
    }
}
