//! Adjacency list index for graph traversal.
//!
//! This module provides efficient adjacency list queries for both outgoing
//! and incoming edges using composite key prefix scans.

use std::ops::Bound;

use manifoldb_core::encoding::keys::{
    decode_edge_by_source_edge_id, decode_edge_by_target_edge_id, encode_edge_by_source_prefix,
    encode_edge_by_source_type_prefix, encode_edge_by_target_prefix,
    encode_edge_by_target_type_prefix,
};
use manifoldb_core::{EdgeId, EdgeType, EntityId};
use manifoldb_storage::{Cursor, Transaction};

use crate::store::{GraphResult, TABLE_EDGES_BY_SOURCE, TABLE_EDGES_BY_TARGET};

/// Adjacency list index for efficient neighbor lookups.
///
/// `AdjacencyIndex` provides efficient traversal operations by using
/// composite keys that enable prefix-based range scans. The index supports:
///
/// - Getting all outgoing edges from an entity
/// - Getting all incoming edges to an entity
/// - Filtering edges by type
/// - Counting neighbors without loading edge data
/// - Memory-efficient iteration over neighbors
///
/// # Key Format
///
/// Outgoing index: `[prefix][source_id][edge_type_hash][edge_id]`
/// Incoming index: `[prefix][target_id][edge_type_hash][edge_id]`
///
/// This layout enables:
/// - O(k) scans for "all edges from/to entity X"
/// - O(k) scans for "edges of type Y from/to entity X"
///
/// where k is the number of matching edges.
pub struct AdjacencyIndex;

impl AdjacencyIndex {
    /// Get all outgoing edge IDs from an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    ///
    /// # Returns
    ///
    /// A vector of edge IDs for all outgoing edges.
    pub fn get_outgoing_edge_ids<T: Transaction>(
        tx: &T,
        source: EntityId,
    ) -> GraphResult<Vec<EdgeId>> {
        let prefix = encode_edge_by_source_prefix(source);
        Self::scan_edge_ids(tx, TABLE_EDGES_BY_SOURCE, &prefix, decode_edge_by_source_edge_id)
    }

    /// Get outgoing edge IDs filtered by edge type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `edge_type` - The edge type to filter by
    ///
    /// # Returns
    ///
    /// A vector of edge IDs for outgoing edges of the specified type.
    pub fn get_outgoing_by_type<T: Transaction>(
        tx: &T,
        source: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<Vec<EdgeId>> {
        let prefix = encode_edge_by_source_type_prefix(source, edge_type);
        Self::scan_edge_ids(tx, TABLE_EDGES_BY_SOURCE, &prefix, decode_edge_by_source_edge_id)
    }

    /// Get all incoming edge IDs to an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    ///
    /// # Returns
    ///
    /// A vector of edge IDs for all incoming edges.
    pub fn get_incoming_edge_ids<T: Transaction>(
        tx: &T,
        target: EntityId,
    ) -> GraphResult<Vec<EdgeId>> {
        let prefix = encode_edge_by_target_prefix(target);
        Self::scan_edge_ids(tx, TABLE_EDGES_BY_TARGET, &prefix, decode_edge_by_target_edge_id)
    }

    /// Get incoming edge IDs filtered by edge type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type to filter by
    ///
    /// # Returns
    ///
    /// A vector of edge IDs for incoming edges of the specified type.
    pub fn get_incoming_by_type<T: Transaction>(
        tx: &T,
        target: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<Vec<EdgeId>> {
        let prefix = encode_edge_by_target_type_prefix(target, edge_type);
        Self::scan_edge_ids(tx, TABLE_EDGES_BY_TARGET, &prefix, decode_edge_by_target_edge_id)
    }

    /// Count outgoing edges from an entity.
    ///
    /// This is more efficient than `get_outgoing_edge_ids().len()` as it
    /// doesn't need to allocate or decode edge IDs.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    pub fn count_outgoing<T: Transaction>(tx: &T, source: EntityId) -> GraphResult<usize> {
        let prefix = encode_edge_by_source_prefix(source);
        Self::count_with_prefix(tx, TABLE_EDGES_BY_SOURCE, &prefix)
    }

    /// Count outgoing edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `edge_type` - The edge type to filter by
    pub fn count_outgoing_by_type<T: Transaction>(
        tx: &T,
        source: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<usize> {
        let prefix = encode_edge_by_source_type_prefix(source, edge_type);
        Self::count_with_prefix(tx, TABLE_EDGES_BY_SOURCE, &prefix)
    }

    /// Count incoming edges to an entity.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    pub fn count_incoming<T: Transaction>(tx: &T, target: EntityId) -> GraphResult<usize> {
        let prefix = encode_edge_by_target_prefix(target);
        Self::count_with_prefix(tx, TABLE_EDGES_BY_TARGET, &prefix)
    }

    /// Count incoming edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type to filter by
    pub fn count_incoming_by_type<T: Transaction>(
        tx: &T,
        target: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<usize> {
        let prefix = encode_edge_by_target_type_prefix(target, edge_type);
        Self::count_with_prefix(tx, TABLE_EDGES_BY_TARGET, &prefix)
    }

    /// Iterate over outgoing edges without collecting.
    ///
    /// This is more memory-efficient for large result sets.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `f` - A function called for each edge ID. Return `Ok(false)` to stop iteration.
    ///
    /// # Returns
    ///
    /// `Ok(())` if iteration completed successfully.
    pub fn for_each_outgoing<T, F>(tx: &T, source: EntityId, f: F) -> GraphResult<()>
    where
        T: Transaction,
        F: FnMut(EdgeId) -> GraphResult<bool>,
    {
        let prefix = encode_edge_by_source_prefix(source);
        Self::for_each_with_prefix(
            tx,
            TABLE_EDGES_BY_SOURCE,
            &prefix,
            decode_edge_by_source_edge_id,
            f,
        )
    }

    /// Iterate over outgoing edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `edge_type` - The edge type to filter by
    /// * `f` - A function called for each edge ID. Return `Ok(false)` to stop iteration.
    pub fn for_each_outgoing_by_type<T, F>(
        tx: &T,
        source: EntityId,
        edge_type: &EdgeType,
        f: F,
    ) -> GraphResult<()>
    where
        T: Transaction,
        F: FnMut(EdgeId) -> GraphResult<bool>,
    {
        let prefix = encode_edge_by_source_type_prefix(source, edge_type);
        Self::for_each_with_prefix(
            tx,
            TABLE_EDGES_BY_SOURCE,
            &prefix,
            decode_edge_by_source_edge_id,
            f,
        )
    }

    /// Iterate over incoming edges without collecting.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `f` - A function called for each edge ID. Return `Ok(false)` to stop iteration.
    pub fn for_each_incoming<T, F>(tx: &T, target: EntityId, f: F) -> GraphResult<()>
    where
        T: Transaction,
        F: FnMut(EdgeId) -> GraphResult<bool>,
    {
        let prefix = encode_edge_by_target_prefix(target);
        Self::for_each_with_prefix(
            tx,
            TABLE_EDGES_BY_TARGET,
            &prefix,
            decode_edge_by_target_edge_id,
            f,
        )
    }

    /// Iterate over incoming edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type to filter by
    /// * `f` - A function called for each edge ID. Return `Ok(false)` to stop iteration.
    pub fn for_each_incoming_by_type<T, F>(
        tx: &T,
        target: EntityId,
        edge_type: &EdgeType,
        f: F,
    ) -> GraphResult<()>
    where
        T: Transaction,
        F: FnMut(EdgeId) -> GraphResult<bool>,
    {
        let prefix = encode_edge_by_target_type_prefix(target, edge_type);
        Self::for_each_with_prefix(
            tx,
            TABLE_EDGES_BY_TARGET,
            &prefix,
            decode_edge_by_target_edge_id,
            f,
        )
    }

    /// Check if an entity has any outgoing edges.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    pub fn has_outgoing<T: Transaction>(tx: &T, source: EntityId) -> GraphResult<bool> {
        let prefix = encode_edge_by_source_prefix(source);
        Self::has_any_with_prefix(tx, TABLE_EDGES_BY_SOURCE, &prefix)
    }

    /// Check if an entity has any outgoing edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `source` - The source entity ID
    /// * `edge_type` - The edge type to check
    pub fn has_outgoing_of_type<T: Transaction>(
        tx: &T,
        source: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<bool> {
        let prefix = encode_edge_by_source_type_prefix(source, edge_type);
        Self::has_any_with_prefix(tx, TABLE_EDGES_BY_SOURCE, &prefix)
    }

    /// Check if an entity has any incoming edges.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    pub fn has_incoming<T: Transaction>(tx: &T, target: EntityId) -> GraphResult<bool> {
        let prefix = encode_edge_by_target_prefix(target);
        Self::has_any_with_prefix(tx, TABLE_EDGES_BY_TARGET, &prefix)
    }

    /// Check if an entity has any incoming edges of a specific type.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `target` - The target entity ID
    /// * `edge_type` - The edge type to check
    pub fn has_incoming_of_type<T: Transaction>(
        tx: &T,
        target: EntityId,
        edge_type: &EdgeType,
    ) -> GraphResult<bool> {
        let prefix = encode_edge_by_target_type_prefix(target, edge_type);
        Self::has_any_with_prefix(tx, TABLE_EDGES_BY_TARGET, &prefix)
    }

    /// Get the out-degree of an entity (number of outgoing edges).
    ///
    /// Alias for [`Self::count_outgoing`].
    #[inline]
    pub fn out_degree<T: Transaction>(tx: &T, entity: EntityId) -> GraphResult<usize> {
        Self::count_outgoing(tx, entity)
    }

    /// Get the in-degree of an entity (number of incoming edges).
    ///
    /// Alias for [`Self::count_incoming`].
    #[inline]
    pub fn in_degree<T: Transaction>(tx: &T, entity: EntityId) -> GraphResult<usize> {
        Self::count_incoming(tx, entity)
    }

    /// Get the total degree of an entity (outgoing + incoming edges).
    ///
    /// Note: For undirected graphs, self-loops are counted twice.
    pub fn degree<T: Transaction>(tx: &T, entity: EntityId) -> GraphResult<usize> {
        let out = Self::count_outgoing(tx, entity)?;
        let inc = Self::count_incoming(tx, entity)?;
        Ok(out + inc)
    }

    // ========================================================================
    // Internal helper methods
    // ========================================================================

    /// Create an exclusive upper bound for a prefix scan.
    fn increment_prefix(prefix: &[u8]) -> Vec<u8> {
        let mut end_prefix = prefix.to_vec();
        let mut i = end_prefix.len();
        while i > 0 {
            i -= 1;
            if end_prefix[i] < 255 {
                end_prefix[i] += 1;
                return end_prefix;
            }
            end_prefix[i] = 0;
        }
        // All bytes were 255, append a byte
        end_prefix.push(0);
        end_prefix
    }

    /// Scan edge IDs from an index table with a prefix.
    fn scan_edge_ids<T: Transaction, F>(
        tx: &T,
        table: &str,
        prefix: &[u8],
        decoder: F,
    ) -> GraphResult<Vec<EdgeId>>
    where
        F: Fn(&[u8]) -> Option<EdgeId>,
    {
        let end_prefix = Self::increment_prefix(prefix);
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(end_prefix.as_slice()))?;

        let mut ids = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(id) = decoder(&key) {
                ids.push(id);
            }
        }

        Ok(ids)
    }

    /// Count entries with a prefix.
    fn count_with_prefix<T: Transaction>(tx: &T, table: &str, prefix: &[u8]) -> GraphResult<usize> {
        let end_prefix = Self::increment_prefix(prefix);
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(end_prefix.as_slice()))?;

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Check if any entries exist with a prefix.
    fn has_any_with_prefix<T: Transaction>(
        tx: &T,
        table: &str,
        prefix: &[u8],
    ) -> GraphResult<bool> {
        let end_prefix = Self::increment_prefix(prefix);
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(end_prefix.as_slice()))?;

        Ok(cursor.next()?.is_some())
    }

    /// Iterate over entries with a prefix.
    fn for_each_with_prefix<T, D, F>(
        tx: &T,
        table: &str,
        prefix: &[u8],
        decoder: D,
        mut f: F,
    ) -> GraphResult<()>
    where
        T: Transaction,
        D: Fn(&[u8]) -> Option<EdgeId>,
        F: FnMut(EdgeId) -> GraphResult<bool>,
    {
        let end_prefix = Self::increment_prefix(prefix);
        let mut cursor =
            tx.range(table, Bound::Included(prefix), Bound::Excluded(end_prefix.as_slice()))?;

        while let Some((key, _)) = cursor.next()? {
            if let Some(id) = decoder(&key) {
                if !f(id)? {
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increment_prefix_basic() {
        assert_eq!(AdjacencyIndex::increment_prefix(&[0x00]), vec![0x01]);
        assert_eq!(AdjacencyIndex::increment_prefix(&[0x01, 0x02]), vec![0x01, 0x03]);
        assert_eq!(AdjacencyIndex::increment_prefix(&[0x01, 0xFF]), vec![0x02, 0x00]);
    }

    #[test]
    fn increment_prefix_overflow() {
        assert_eq!(AdjacencyIndex::increment_prefix(&[0xFF]), vec![0x00, 0x00]);
        assert_eq!(AdjacencyIndex::increment_prefix(&[0xFF, 0xFF]), vec![0x00, 0x00, 0x00]);
    }
}
