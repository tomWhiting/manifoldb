//! Edge (relationship) storage operations.
//!
//! This module provides CRUD operations for edges in the graph.

use std::ops::Bound;

use manifoldb_core::encoding::keys::{
    decode_edge_key, encode_edge_key, encode_edge_type_index_prefix, PREFIX_EDGE,
};
use manifoldb_core::encoding::{Decoder, Encoder};
use manifoldb_core::{Edge, EdgeId, EdgeType, EntityId};
use manifoldb_storage::{Cursor, Transaction};

use super::error::{GraphError, GraphResult};
use super::node::NodeStore;
use super::IdGenerator;
use crate::index::{AdjacencyIndex, IndexMaintenance};

/// Table name for edge data.
pub const TABLE_EDGES: &str = "edges";

/// Table name for edges indexed by source entity.
pub const TABLE_EDGES_BY_SOURCE: &str = "edges_by_source";

/// Table name for edges indexed by target entity.
pub const TABLE_EDGES_BY_TARGET: &str = "edges_by_target";

/// Table name for edge type index.
pub const TABLE_EDGE_TYPES: &str = "edge_types";

/// Edge storage operations.
///
/// `EdgeStore` provides transactional CRUD operations for graph edges.
/// All operations work within a transaction context for ACID guarantees.
///
/// # Indexes
///
/// Edges are indexed by:
/// - Source entity (for outgoing edge queries)
/// - Target entity (for incoming edge queries)
/// - Edge type (for type-based filtering)
///
/// # Example
///
/// ```ignore
/// use manifoldb_graph::store::{EdgeStore, NodeStore, IdGenerator};
///
/// // Create an edge between two nodes
/// let gen = IdGenerator::new();
/// let edge = EdgeStore::create(&mut tx, &gen, source_id, target_id, "FOLLOWS", |id| {
///     Edge::new(id, source_id, target_id, "FOLLOWS")
///         .with_property("since", "2024-01-01")
/// })?;
///
/// // Find all outgoing edges from a node
/// let outgoing = EdgeStore::get_outgoing(&tx, source_id)?;
/// ```
pub struct EdgeStore;

impl EdgeStore {
    /// Create a new edge in the store.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id_gen` - The ID generator
    /// * `source` - The source entity ID
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type
    /// * `builder` - A function that builds the edge given an ID
    ///
    /// # Returns
    ///
    /// The created edge with its assigned ID.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidEntityReference`] if source or target doesn't exist.
    pub fn create<T: Transaction, F>(
        tx: &mut T,
        id_gen: &IdGenerator,
        source: EntityId,
        target: EntityId,
        edge_type: impl Into<EdgeType>,
        builder: F,
    ) -> GraphResult<Edge>
    where
        F: FnOnce(EdgeId) -> Edge,
    {
        // Verify source and target exist
        if !NodeStore::exists(tx, source)? {
            return Err(GraphError::InvalidEntityReference(source));
        }
        if !NodeStore::exists(tx, target)? {
            return Err(GraphError::InvalidEntityReference(target));
        }

        let id = id_gen.next_edge_id();
        let _edge_type = edge_type.into();
        let edge = builder(id);

        Self::store_edge(tx, &edge)?;
        Ok(edge)
    }

    /// Create an edge with a specific ID.
    ///
    /// This is useful when importing data or when you need to control IDs.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `edge` - The edge to store (must have a valid ID)
    /// * `validate_refs` - Whether to validate that source/target entities exist
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EdgeAlreadyExists`] if an edge with this ID exists.
    /// Returns [`GraphError::InvalidEntityReference`] if validation is enabled and entities don't exist.
    pub fn create_with_id<T: Transaction>(
        tx: &mut T,
        edge: &Edge,
        validate_refs: bool,
    ) -> GraphResult<()> {
        let key = encode_edge_key(edge.id);

        // Check if edge already exists
        if tx.get(TABLE_EDGES, &key)?.is_some() {
            return Err(GraphError::EdgeAlreadyExists(edge.id));
        }

        if validate_refs {
            if !NodeStore::exists(tx, edge.source)? {
                return Err(GraphError::InvalidEntityReference(edge.source));
            }
            if !NodeStore::exists(tx, edge.target)? {
                return Err(GraphError::InvalidEntityReference(edge.target));
            }
        }

        Self::store_edge(tx, edge)?;
        Ok(())
    }

    /// Internal helper to store an edge and its indexes.
    fn store_edge<T: Transaction>(tx: &mut T, edge: &Edge) -> GraphResult<()> {
        // Store the edge data
        let key = encode_edge_key(edge.id);
        let value = edge.encode()?;
        tx.put(TABLE_EDGES, &key, &value)?;

        // Add all indexes using IndexMaintenance
        IndexMaintenance::add_edge_indexes(tx, edge)?;

        Ok(())
    }

    /// Get an edge by ID.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The edge ID to look up
    ///
    /// # Returns
    ///
    /// The edge if found, or `None` if it doesn't exist.
    pub fn get<T: Transaction>(tx: &T, id: EdgeId) -> GraphResult<Option<Edge>> {
        let key = encode_edge_key(id);
        match tx.get(TABLE_EDGES, &key)? {
            Some(value) => {
                let edge = Edge::decode(&value)?;
                Ok(Some(edge))
            }
            None => Ok(None),
        }
    }

    /// Get an edge by ID, returning an error if not found.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The edge ID to look up
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EdgeNotFound`] if the edge doesn't exist.
    pub fn get_or_error<T: Transaction>(tx: &T, id: EdgeId) -> GraphResult<Edge> {
        Self::get(tx, id)?.ok_or(GraphError::EdgeNotFound(id))
    }

    /// Check if an edge exists.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The edge ID to check
    pub fn exists<T: Transaction>(tx: &T, id: EdgeId) -> GraphResult<bool> {
        let key = encode_edge_key(id);
        Ok(tx.get(TABLE_EDGES, &key)?.is_some())
    }

    /// Update an existing edge.
    ///
    /// Note: The source, target, and edge type cannot be changed.
    /// To change these, delete the edge and create a new one.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `edge` - The edge with updated data
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EdgeNotFound`] if the edge doesn't exist.
    pub fn update<T: Transaction>(tx: &mut T, edge: &Edge) -> GraphResult<()> {
        let key = encode_edge_key(edge.id);

        // Get the old edge to verify it exists and check if structure changed
        let old_value = tx.get(TABLE_EDGES, &key)?.ok_or(GraphError::EdgeNotFound(edge.id))?;
        let old_edge = Edge::decode(&old_value)?;

        // Update indexes if edge structure changed
        IndexMaintenance::update_edge_indexes(tx, &old_edge, edge)?;

        // Store updated edge
        let value = edge.encode()?;
        tx.put(TABLE_EDGES, &key, &value)?;

        Ok(())
    }

    /// Delete an edge by ID.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The edge ID to delete
    ///
    /// # Returns
    ///
    /// `true` if the edge was deleted, `false` if it didn't exist.
    pub fn delete<T: Transaction>(tx: &mut T, id: EdgeId) -> GraphResult<bool> {
        let key = encode_edge_key(id);

        // Get the edge to clean up indexes
        let Some(value) = tx.get(TABLE_EDGES, &key)? else {
            return Ok(false);
        };
        let edge = Edge::decode(&value)?;

        // Remove all indexes using IndexMaintenance
        IndexMaintenance::remove_edge_indexes(tx, &edge)?;

        // Delete the edge
        tx.delete(TABLE_EDGES, &key)?;
        Ok(true)
    }

    /// Get all outgoing edges from an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    pub fn get_outgoing<T: Transaction>(tx: &T, source: EntityId) -> GraphResult<Vec<Edge>> {
        let edge_ids = AdjacencyIndex::get_outgoing_edge_ids(tx, source)?;
        Self::get_edges_by_ids(tx, &edge_ids)
    }

    /// Get IDs of all outgoing edges from an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    pub fn get_outgoing_ids<T: Transaction>(tx: &T, source: EntityId) -> GraphResult<Vec<EdgeId>> {
        AdjacencyIndex::get_outgoing_edge_ids(tx, source)
    }

    /// Get outgoing edges of a specific type from an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `edge_type` - The edge type to filter by
    pub fn get_outgoing_by_type<T: Transaction>(
        tx: &T,
        source: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<Vec<Edge>> {
        let edge_ids = AdjacencyIndex::get_outgoing_by_type(tx, source, edge_type)?;
        Self::get_edges_by_ids(tx, &edge_ids)
    }

    /// Get all incoming edges to an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    pub fn get_incoming<T: Transaction>(tx: &T, target: EntityId) -> GraphResult<Vec<Edge>> {
        let edge_ids = AdjacencyIndex::get_incoming_edge_ids(tx, target)?;
        Self::get_edges_by_ids(tx, &edge_ids)
    }

    /// Get IDs of all incoming edges to an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    pub fn get_incoming_ids<T: Transaction>(tx: &T, target: EntityId) -> GraphResult<Vec<EdgeId>> {
        AdjacencyIndex::get_incoming_edge_ids(tx, target)
    }

    /// Get incoming edges of a specific type to an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type to filter by
    pub fn get_incoming_by_type<T: Transaction>(
        tx: &T,
        target: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<Vec<Edge>> {
        let edge_ids = AdjacencyIndex::get_incoming_by_type(tx, target, edge_type)?;
        Self::get_edges_by_ids(tx, &edge_ids)
    }

    /// Find all edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `edge_type` - The edge type to search for
    pub fn find_by_type<T: Transaction>(tx: &T, edge_type: &EdgeType) -> GraphResult<Vec<EdgeId>> {
        let prefix = encode_edge_type_index_prefix(edge_type);

        // Create end bound
        let mut end_prefix = prefix.clone();
        if let Some(last) = end_prefix.last_mut() {
            *last = last.saturating_add(1);
        }

        let mut cursor = tx.range(
            TABLE_EDGE_TYPES,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(end_prefix.as_slice()),
        )?;

        let mut ids = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            // Extract edge ID from the key (last 8 bytes after prefix + type hash)
            if key.len() >= 17 {
                let id_bytes: [u8; 8] = key[9..17].try_into().map_err(|_| {
                    GraphError::DataCorruption(format!(
                        "malformed edge type index key: expected 8 bytes for edge ID at offset 9, got {} total bytes",
                        key.len()
                    ))
                })?;
                ids.push(EdgeId::new(u64::from_be_bytes(id_bytes)));
            } else {
                return Err(GraphError::DataCorruption(format!(
                    "malformed edge type index key: expected at least 17 bytes, got {}",
                    key.len()
                )));
            }
        }

        Ok(ids)
    }

    /// Delete all edges connected to an entity.
    ///
    /// This removes all incoming and outgoing edges. Call this before
    /// deleting an entity to maintain graph consistency.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `entity_id` - The entity ID
    ///
    /// # Returns
    ///
    /// The number of edges deleted.
    pub fn delete_edges_for_entity<T: Transaction>(
        tx: &mut T,
        entity_id: EntityId,
    ) -> GraphResult<usize> {
        let mut deleted = 0;

        // Delete outgoing edges
        let outgoing_ids = Self::get_outgoing_ids(tx, entity_id)?;
        for edge_id in outgoing_ids {
            if Self::delete(tx, edge_id)? {
                deleted += 1;
            }
        }

        // Delete incoming edges
        let incoming_ids = Self::get_incoming_ids(tx, entity_id)?;
        for edge_id in incoming_ids {
            if Self::delete(tx, edge_id)? {
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    /// Helper to get full edges from a list of IDs.
    fn get_edges_by_ids<T: Transaction>(tx: &T, ids: &[EdgeId]) -> GraphResult<Vec<Edge>> {
        let mut edges = Vec::with_capacity(ids.len());
        for &id in ids {
            if let Some(edge) = Self::get(tx, id)? {
                edges.push(edge);
            }
        }
        Ok(edges)
    }

    /// Count all edges in the store.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    pub fn count<T: Transaction>(tx: &T) -> GraphResult<usize> {
        let start = [PREFIX_EDGE];
        let end = [PREFIX_EDGE + 1];

        let mut cursor = tx.range(
            TABLE_EDGES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        )?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Iterate over all edges.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `f` - A function to call for each edge. Return `false` to stop iteration.
    pub fn for_each<T: Transaction, F>(tx: &T, mut f: F) -> GraphResult<()>
    where
        F: FnMut(&Edge) -> bool,
    {
        let start = [PREFIX_EDGE];
        let end = [PREFIX_EDGE + 1];

        let mut cursor = tx.range(
            TABLE_EDGES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        )?;

        while let Some((_, value)) = cursor.next()? {
            let edge = Edge::decode(&value)?;
            if !f(&edge) {
                break;
            }
        }

        Ok(())
    }

    /// Get all edges as a vector.
    ///
    /// Use with caution on large datasets - prefer [`Self::for_each`] for
    /// processing edges without loading all into memory.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    pub fn all<T: Transaction>(tx: &T) -> GraphResult<Vec<Edge>> {
        let mut edges = Vec::new();
        Self::for_each(tx, |edge| {
            edges.push(edge.clone());
            true
        })?;
        Ok(edges)
    }

    /// Find the highest edge ID in the store.
    ///
    /// This is useful for initializing the ID generator after loading data.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    ///
    /// # Returns
    ///
    /// The highest edge ID, or `None` if there are no edges.
    pub fn max_id<T: Transaction>(tx: &T) -> GraphResult<Option<EdgeId>> {
        let start = [PREFIX_EDGE];
        let end = [PREFIX_EDGE + 1];

        let mut cursor = tx.range(
            TABLE_EDGES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        )?;

        // Seek to the last key in the range
        if cursor.seek_last()?.is_some() {
            if let Some((key, _)) = cursor.current().map(|(k, v)| (k.to_vec(), v.to_vec())) {
                return Ok(decode_edge_key(&key));
            }
        }

        Ok(None)
    }

    /// List all unique edge types with their counts.
    ///
    /// This scans all edges to collect edge type statistics.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    ///
    /// # Returns
    ///
    /// A vector of (edge_type, count) pairs sorted by type name.
    pub fn list_edge_types<T: Transaction>(tx: &T) -> GraphResult<Vec<(EdgeType, usize)>> {
        use std::collections::HashMap;

        let mut type_counts: HashMap<EdgeType, usize> = HashMap::new();

        Self::for_each(tx, |edge| {
            *type_counts.entry(edge.edge_type.clone()).or_insert(0) += 1;
            true
        })?;

        let mut result: Vec<_> = type_counts.into_iter().collect();
        result.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Integration tests with actual storage backend are in the tests/ directory

    #[test]
    fn table_names_are_valid() {
        assert!(!TABLE_EDGES.is_empty());
        assert!(!TABLE_EDGES_BY_SOURCE.is_empty());
        assert!(!TABLE_EDGES_BY_TARGET.is_empty());
        assert!(!TABLE_EDGE_TYPES.is_empty());
    }

    #[test]
    fn table_names_are_unique() {
        let tables = [TABLE_EDGES, TABLE_EDGES_BY_SOURCE, TABLE_EDGES_BY_TARGET, TABLE_EDGE_TYPES];
        for (i, a) in tables.iter().enumerate() {
            for (j, b) in tables.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
