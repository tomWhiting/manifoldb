//! Index maintenance operations.
//!
//! This module provides index update operations that are called when
//! edges are created, updated, or deleted to maintain index consistency.

use manifoldb_core::encoding::keys::{
    encode_edge_by_source_key, encode_edge_by_target_key, encode_edge_type_index_key,
};
use manifoldb_core::{Edge, EdgeId, EdgeType, EntityId};
use manifoldb_storage::Transaction;

use crate::store::GraphResult;
use crate::store::{TABLE_EDGES_BY_SOURCE, TABLE_EDGES_BY_TARGET, TABLE_EDGE_TYPES};

/// Index maintenance operations for edges.
///
/// `IndexMaintenance` provides low-level operations for maintaining
/// adjacency indexes when edges are mutated. These operations should
/// be called by the `EdgeStore` during create/update/delete operations.
///
/// # Index Tables
///
/// Three index tables are maintained:
/// - `edges_by_source` - Outgoing edge index
/// - `edges_by_target` - Incoming edge index
/// - `edge_types` - Edge type index
///
/// All indexes use empty values since the key contains all needed information.
pub struct IndexMaintenance;

impl IndexMaintenance {
    /// Add all indexes for an edge.
    ///
    /// This should be called after storing a new edge.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `edge` - The edge to index
    pub fn add_edge_indexes<T: Transaction>(tx: &mut T, edge: &Edge) -> GraphResult<()> {
        Self::add_outgoing_index(tx, edge.source, &edge.edge_type, edge.id)?;
        Self::add_incoming_index(tx, edge.target, &edge.edge_type, edge.id)?;
        Self::add_type_index(tx, &edge.edge_type, edge.id)?;
        Ok(())
    }

    /// Remove all indexes for an edge.
    ///
    /// This should be called before deleting an edge.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `edge` - The edge whose indexes should be removed
    pub fn remove_edge_indexes<T: Transaction>(tx: &mut T, edge: &Edge) -> GraphResult<()> {
        Self::remove_outgoing_index(tx, edge.source, &edge.edge_type, edge.id)?;
        Self::remove_incoming_index(tx, edge.target, &edge.edge_type, edge.id)?;
        Self::remove_type_index(tx, &edge.edge_type, edge.id)?;
        Ok(())
    }

    /// Update indexes when an edge's structure changes.
    ///
    /// This handles the case where source, target, or type changes.
    /// Only the changed indexes are updated.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `old_edge` - The edge before the update
    /// * `new_edge` - The edge after the update
    pub fn update_edge_indexes<T: Transaction>(
        tx: &mut T,
        old_edge: &Edge,
        new_edge: &Edge,
    ) -> GraphResult<()> {
        let source_changed = old_edge.source != new_edge.source;
        let target_changed = old_edge.target != new_edge.target;
        let type_changed = old_edge.edge_type.as_str() != new_edge.edge_type.as_str();

        // Update outgoing index if source or type changed
        if source_changed || type_changed {
            Self::remove_outgoing_index(tx, old_edge.source, &old_edge.edge_type, old_edge.id)?;
            Self::add_outgoing_index(tx, new_edge.source, &new_edge.edge_type, new_edge.id)?;
        }

        // Update incoming index if target or type changed
        if target_changed || type_changed {
            Self::remove_incoming_index(tx, old_edge.target, &old_edge.edge_type, old_edge.id)?;
            Self::add_incoming_index(tx, new_edge.target, &new_edge.edge_type, new_edge.id)?;
        }

        // Update type index if type changed
        if type_changed {
            Self::remove_type_index(tx, &old_edge.edge_type, old_edge.id)?;
            Self::add_type_index(tx, &new_edge.edge_type, new_edge.id)?;
        }

        Ok(())
    }

    // ========================================================================
    // Outgoing index operations
    // ========================================================================

    /// Add an entry to the outgoing edge index.
    pub fn add_outgoing_index<T: Transaction>(
        tx: &mut T,
        source: EntityId,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_by_source_key(source, edge_type, edge_id);
        tx.put(TABLE_EDGES_BY_SOURCE, &key, &[])?;
        Ok(())
    }

    /// Remove an entry from the outgoing edge index.
    pub fn remove_outgoing_index<T: Transaction>(
        tx: &mut T,
        source: EntityId,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_by_source_key(source, edge_type, edge_id);
        tx.delete(TABLE_EDGES_BY_SOURCE, &key)?;
        Ok(())
    }

    // ========================================================================
    // Incoming index operations
    // ========================================================================

    /// Add an entry to the incoming edge index.
    pub fn add_incoming_index<T: Transaction>(
        tx: &mut T,
        target: EntityId,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_by_target_key(target, edge_type, edge_id);
        tx.put(TABLE_EDGES_BY_TARGET, &key, &[])?;
        Ok(())
    }

    /// Remove an entry from the incoming edge index.
    pub fn remove_incoming_index<T: Transaction>(
        tx: &mut T,
        target: EntityId,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_by_target_key(target, edge_type, edge_id);
        tx.delete(TABLE_EDGES_BY_TARGET, &key)?;
        Ok(())
    }

    // ========================================================================
    // Type index operations
    // ========================================================================

    /// Add an entry to the edge type index.
    pub fn add_type_index<T: Transaction>(
        tx: &mut T,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_type_index_key(edge_type, edge_id);
        tx.put(TABLE_EDGE_TYPES, &key, &[])?;
        Ok(())
    }

    /// Remove an entry from the edge type index.
    pub fn remove_type_index<T: Transaction>(
        tx: &mut T,
        edge_type: &EdgeType,
        edge_id: EdgeId,
    ) -> GraphResult<()> {
        let key = encode_edge_type_index_key(edge_type, edge_id);
        tx.delete(TABLE_EDGE_TYPES, &key)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Integration tests are in tests/adjacency_tests.rs
    // Unit tests would require mocking the Transaction trait
}
