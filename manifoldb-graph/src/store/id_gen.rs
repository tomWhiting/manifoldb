//! ID generation strategies for entities and edges.
//!
//! This module provides monotonically increasing ID generators that are
//! thread-safe and persistent across restarts.

use std::sync::atomic::{AtomicU64, Ordering};

use manifoldb_core::{EdgeId, EntityId};

/// A monotonic ID generator.
///
/// Generates unique, monotonically increasing IDs. The generator is thread-safe
/// and can be shared across threads. IDs start from 1 (0 is reserved for "no ID").
///
/// # Persistence
///
/// The generator can be initialized with the highest existing ID to resume
/// after a restart. Use [`IdGenerator::with_start`] to set the starting value.
///
/// # Example
///
/// ```
/// use manifoldb_graph::store::IdGenerator;
///
/// let gen = IdGenerator::new();
/// let id1 = gen.next_entity_id();
/// let id2 = gen.next_entity_id();
/// assert!(id1.as_u64() < id2.as_u64());
/// ```
#[derive(Debug)]
pub struct IdGenerator {
    /// The next entity ID to assign.
    next_entity_id: AtomicU64,
    /// The next edge ID to assign.
    next_edge_id: AtomicU64,
}

impl IdGenerator {
    /// Create a new ID generator starting from 1.
    #[must_use]
    pub const fn new() -> Self {
        Self { next_entity_id: AtomicU64::new(1), next_edge_id: AtomicU64::new(1) }
    }

    /// Create an ID generator starting from specific values.
    ///
    /// Use this to resume ID generation after loading existing data.
    /// The provided values should be one greater than the highest existing IDs.
    ///
    /// # Arguments
    ///
    /// * `entity_start` - The first entity ID to generate
    /// * `edge_start` - The first edge ID to generate
    #[must_use]
    pub const fn with_start(entity_start: u64, edge_start: u64) -> Self {
        Self {
            next_entity_id: AtomicU64::new(entity_start),
            next_edge_id: AtomicU64::new(edge_start),
        }
    }

    /// Generate the next entity ID.
    ///
    /// This operation is atomic and thread-safe.
    pub fn next_entity_id(&self) -> EntityId {
        let id = self.next_entity_id.fetch_add(1, Ordering::Relaxed);
        EntityId::new(id)
    }

    /// Generate the next edge ID.
    ///
    /// This operation is atomic and thread-safe.
    pub fn next_edge_id(&self) -> EdgeId {
        let id = self.next_edge_id.fetch_add(1, Ordering::Relaxed);
        EdgeId::new(id)
    }

    /// Get the current entity ID counter value (next ID to be assigned).
    #[must_use]
    pub fn current_entity_counter(&self) -> u64 {
        self.next_entity_id.load(Ordering::Relaxed)
    }

    /// Get the current edge ID counter value (next ID to be assigned).
    #[must_use]
    pub fn current_edge_counter(&self) -> u64 {
        self.next_edge_id.load(Ordering::Relaxed)
    }

    /// Reset the entity ID counter to a new value.
    ///
    /// This is primarily used for testing or when rebuilding indexes.
    pub fn reset_entity_counter(&self, value: u64) {
        self.next_entity_id.store(value, Ordering::Relaxed);
    }

    /// Reset the edge ID counter to a new value.
    ///
    /// This is primarily used for testing or when rebuilding indexes.
    pub fn reset_edge_counter(&self, value: u64) {
        self.next_edge_id.store(value, Ordering::Relaxed);
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_generator_starts_at_one() {
        let gen = IdGenerator::new();
        assert_eq!(gen.next_entity_id().as_u64(), 1);
        assert_eq!(gen.next_edge_id().as_u64(), 1);
    }

    #[test]
    fn ids_are_monotonically_increasing() {
        let gen = IdGenerator::new();
        let ids: Vec<_> = (0..100).map(|_| gen.next_entity_id().as_u64()).collect();
        for window in ids.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    #[test]
    fn with_start_sets_initial_values() {
        let gen = IdGenerator::with_start(100, 200);
        assert_eq!(gen.next_entity_id().as_u64(), 100);
        assert_eq!(gen.next_edge_id().as_u64(), 200);
        assert_eq!(gen.next_entity_id().as_u64(), 101);
        assert_eq!(gen.next_edge_id().as_u64(), 201);
    }

    #[test]
    fn current_counter_reflects_next_id() {
        let gen = IdGenerator::new();
        assert_eq!(gen.current_entity_counter(), 1);
        gen.next_entity_id();
        assert_eq!(gen.current_entity_counter(), 2);
    }

    #[test]
    fn reset_counter_works() {
        let gen = IdGenerator::new();
        gen.next_entity_id();
        gen.next_entity_id();
        gen.reset_entity_counter(1000);
        assert_eq!(gen.next_entity_id().as_u64(), 1000);
    }

    #[test]
    fn entity_and_edge_ids_are_independent() {
        let gen = IdGenerator::new();
        assert_eq!(gen.next_entity_id().as_u64(), 1);
        assert_eq!(gen.next_entity_id().as_u64(), 2);
        assert_eq!(gen.next_edge_id().as_u64(), 1);
        assert_eq!(gen.next_entity_id().as_u64(), 3);
        assert_eq!(gen.next_edge_id().as_u64(), 2);
    }
}
