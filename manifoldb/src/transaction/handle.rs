//! Database transaction handle for user operations.

use manifoldb_core::{Edge, EdgeId, Entity, EntityId, TransactionError};
use manifoldb_graph::index::IndexMaintenance;
use manifoldb_storage::{Cursor, Transaction};

use super::VectorSyncStrategy;

/// Well-known table names for graph storage.
mod tables {
    pub const NODES: &str = "nodes";
    pub const EDGES: &str = "edges";
    pub const EDGES_OUT: &str = "edges_out";
    pub const EDGES_IN: &str = "edges_in";
    pub const METADATA: &str = "metadata";
}

/// Metadata keys for counters.
mod metadata {
    pub const NEXT_ENTITY_ID: &[u8] = b"next_entity_id";
    pub const NEXT_EDGE_ID: &[u8] = b"next_edge_id";
}

/// A database transaction handle for user operations.
///
/// `DatabaseTransaction` wraps a storage transaction and provides high-level
/// operations for graph entities and edges. It ensures that all operations
/// within the transaction are atomic.
///
/// # Read vs Write Transactions
///
/// - **Read transactions**: Can only read data. Attempting to write will return an error.
/// - **Write transactions**: Can both read and write data.
///
/// # Commit and Rollback
///
/// Transactions must be explicitly committed to persist changes. Dropping a
/// transaction without committing will roll back all changes.
///
/// # Example
///
/// ```ignore
/// // Write transaction
/// let mut tx = manager.begin_write()?;
/// let entity = tx.create_entity()?.with_label("Person");
/// tx.put_entity(&entity)?;
/// tx.commit()?;
///
/// // Read transaction
/// let tx = manager.begin_read()?;
/// let entity = tx.get_entity(entity_id)?;
/// ```
pub struct DatabaseTransaction<T: Transaction> {
    /// Unique transaction ID for debugging and logging.
    tx_id: u64,

    /// The underlying storage transaction.
    storage: Option<T>,

    /// Whether this is a read-only transaction.
    read_only: bool,

    /// Vector sync strategy (only relevant for write transactions).
    vector_sync_strategy: VectorSyncStrategy,
}

impl<T: Transaction> DatabaseTransaction<T> {
    /// Create a new read-only transaction.
    pub(crate) const fn new_read(tx_id: u64, storage: T) -> Self {
        Self {
            tx_id,
            storage: Some(storage),
            read_only: true,
            vector_sync_strategy: VectorSyncStrategy::Synchronous,
        }
    }

    /// Create a new read-write transaction.
    pub(crate) const fn new_write(
        tx_id: u64,
        storage: T,
        vector_sync_strategy: VectorSyncStrategy,
    ) -> Self {
        Self { tx_id, storage: Some(storage), read_only: false, vector_sync_strategy }
    }

    /// Get the transaction ID.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.tx_id
    }

    /// Check if this is a read-only transaction.
    #[must_use]
    pub const fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Get the vector sync strategy.
    #[must_use]
    pub const fn vector_sync_strategy(&self) -> VectorSyncStrategy {
        self.vector_sync_strategy
    }

    /// Get the storage transaction, returning an error if already consumed.
    fn storage(&self) -> Result<&T, TransactionError> {
        self.storage.as_ref().ok_or(TransactionError::AlreadyCompleted)
    }

    /// Get a reference to the underlying storage transaction for direct access.
    ///
    /// This is useful for graph traversal operations that need low-level access
    /// to the storage layer without going through the higher-level entity/edge APIs.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction has already been committed or rolled back.
    pub fn storage_ref(&self) -> Result<&T, TransactionError> {
        self.storage()
    }

    /// Get a mutable reference to the storage transaction.
    fn storage_mut(&mut self) -> Result<&mut T, TransactionError> {
        if self.read_only {
            return Err(TransactionError::ReadOnly);
        }
        self.storage.as_mut().ok_or(TransactionError::AlreadyCompleted)
    }

    // ========================================================================
    // Entity Operations
    // ========================================================================

    /// Get an entity by its ID.
    ///
    /// Returns `Ok(None)` if the entity does not exist.
    pub fn get_entity(&self, id: EntityId) -> Result<Option<Entity>, TransactionError> {
        let storage = self.storage()?;
        let key = id.as_u64().to_be_bytes();

        match storage.get(tables::NODES, &key) {
            Ok(Some(bytes)) => {
                let entity: Entity = bincode::deserialize(&bytes)
                    .map_err(|e| TransactionError::Serialization(e.to_string()))?;
                Ok(Some(entity))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(storage_error_to_tx_error(e)),
        }
    }

    /// Put an entity into the database.
    ///
    /// If an entity with the same ID already exists, it will be replaced.
    pub fn put_entity(&mut self, entity: &Entity) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;
        let key = entity.id.as_u64().to_be_bytes();
        let value = bincode::serialize(entity)
            .map_err(|e| TransactionError::Serialization(e.to_string()))?;

        storage.put(tables::NODES, &key, &value).map_err(storage_error_to_tx_error)
    }

    /// Delete an entity by its ID.
    ///
    /// Returns `true` if the entity existed and was deleted, `false` if it didn't exist.
    ///
    /// Note: This does not automatically delete edges connected to this entity.
    /// The caller is responsible for maintaining referential integrity.
    pub fn delete_entity(&mut self, id: EntityId) -> Result<bool, TransactionError> {
        let storage = self.storage_mut()?;
        let key = id.as_u64().to_be_bytes();

        storage.delete(tables::NODES, &key).map_err(storage_error_to_tx_error)
    }

    /// Create a new entity with an auto-generated ID.
    ///
    /// The entity is not persisted until [`put_entity`](Self::put_entity) is called.
    pub fn create_entity(&mut self) -> Result<Entity, TransactionError> {
        let id = self.next_entity_id()?;
        Ok(Entity::new(id))
    }

    /// Get the next entity ID and increment the counter.
    fn next_entity_id(&mut self) -> Result<EntityId, TransactionError> {
        let storage = self.storage_mut()?;

        // Read current counter
        let current = match storage.get(tables::METADATA, metadata::NEXT_ENTITY_ID) {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let arr: [u8; 8] = bytes
                    .try_into()
                    .map_err(|_| TransactionError::Internal("invalid counter".to_string()))?;
                u64::from_be_bytes(arr)
            }
            Ok(_) => 1, // Start from 1 if not set
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        // Increment and store
        let next = current + 1;
        storage
            .put(tables::METADATA, metadata::NEXT_ENTITY_ID, &next.to_be_bytes())
            .map_err(storage_error_to_tx_error)?;

        Ok(EntityId::new(current))
    }

    // ========================================================================
    // Edge Operations
    // ========================================================================

    /// Get an edge by its ID.
    ///
    /// Returns `Ok(None)` if the edge does not exist.
    pub fn get_edge(&self, id: EdgeId) -> Result<Option<Edge>, TransactionError> {
        let storage = self.storage()?;
        let key = id.as_u64().to_be_bytes();

        match storage.get(tables::EDGES, &key) {
            Ok(Some(bytes)) => {
                let edge: Edge = bincode::deserialize(&bytes)
                    .map_err(|e| TransactionError::Serialization(e.to_string()))?;
                Ok(Some(edge))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(storage_error_to_tx_error(e)),
        }
    }

    /// Put an edge into the database.
    ///
    /// If an edge with the same ID already exists, it will be replaced.
    /// This also updates the adjacency indexes (both the simple and graph-layer indexes).
    pub fn put_edge(&mut self, edge: &Edge) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;
        let key = edge.id.as_u64().to_be_bytes();
        let value =
            bincode::serialize(edge).map_err(|e| TransactionError::Serialization(e.to_string()))?;

        // Store the edge data
        storage.put(tables::EDGES, &key, &value).map_err(storage_error_to_tx_error)?;

        // Update simple outgoing edge index (source -> edge)
        let out_key = make_adjacency_key(edge.source, edge.id);
        storage.put(tables::EDGES_OUT, &out_key, &[]).map_err(storage_error_to_tx_error)?;

        // Update simple incoming edge index (target -> edge)
        let in_key = make_adjacency_key(edge.target, edge.id);
        storage.put(tables::EDGES_IN, &in_key, &[]).map_err(storage_error_to_tx_error)?;

        // Update graph layer indexes (edges_by_source, edges_by_target, edge_types)
        // These indexes include edge type and are used for graph traversal queries
        IndexMaintenance::add_edge_indexes(storage, edge)
            .map_err(|e| TransactionError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Iterate over all entities, optionally filtering by label.
    ///
    /// If `label` is `Some`, only entities with that label are returned.
    /// If `label` is `None`, all entities are returned.
    ///
    /// Returns an empty vector if no entities exist (including if the table hasn't been created).
    pub fn iter_entities(&self, label: Option<&str>) -> Result<Vec<Entity>, TransactionError> {
        use manifoldb_storage::StorageError;
        use std::ops::Bound;

        let storage = self.storage()?;

        // Create a cursor over all nodes
        // If the table doesn't exist yet, just return empty
        let cursor_result = storage.range(tables::NODES, Bound::Unbounded, Bound::Unbounded);

        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(StorageError::TableNotFound(_)) => {
                // Table doesn't exist, no entities yet
                return Ok(Vec::new());
            }
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        let mut entities = Vec::new();

        // Iterate through all nodes
        while let Some((_key, value)) = cursor.next().map_err(storage_error_to_tx_error)? {
            let entity: Entity = bincode::deserialize(&value)
                .map_err(|e| TransactionError::Serialization(e.to_string()))?;

            // Filter by label if specified
            if let Some(label_filter) = label {
                if entity.has_label(label_filter) {
                    entities.push(entity);
                }
            } else {
                entities.push(entity);
            }
        }

        Ok(entities)
    }

    /// Count entities, optionally filtering by label.
    pub fn count_entities(&self, label: Option<&str>) -> Result<usize, TransactionError> {
        // For now, use iter_entities and count
        // Could be optimized with a dedicated label index
        Ok(self.iter_entities(label)?.len())
    }

    /// Delete an edge by its ID.
    ///
    /// Returns `true` if the edge existed and was deleted, `false` if it didn't exist.
    pub fn delete_edge(&mut self, id: EdgeId) -> Result<bool, TransactionError> {
        // First get the edge to find its source/target for index cleanup
        let Some(edge) = self.get_edge(id)? else {
            return Ok(false);
        };

        let storage = self.storage_mut()?;
        let key = id.as_u64().to_be_bytes();

        // Delete the edge data
        let deleted = storage.delete(tables::EDGES, &key).map_err(storage_error_to_tx_error)?;

        if deleted {
            // Remove from simple outgoing edge index
            let out_key = make_adjacency_key(edge.source, edge.id);
            let _ = storage.delete(tables::EDGES_OUT, &out_key);

            // Remove from simple incoming edge index
            let in_key = make_adjacency_key(edge.target, edge.id);
            let _ = storage.delete(tables::EDGES_IN, &in_key);

            // Remove from graph layer indexes
            let _ = IndexMaintenance::remove_edge_indexes(storage, &edge);
        }

        Ok(deleted)
    }

    /// Create a new edge with an auto-generated ID.
    ///
    /// The edge is not persisted until [`put_edge`](Self::put_edge) is called.
    pub fn create_edge(
        &mut self,
        source: EntityId,
        target: EntityId,
        edge_type: impl Into<manifoldb_core::EdgeType>,
    ) -> Result<Edge, TransactionError> {
        let id = self.next_edge_id()?;
        Ok(Edge::new(id, source, target, edge_type))
    }

    /// Get the next edge ID and increment the counter.
    fn next_edge_id(&mut self) -> Result<EdgeId, TransactionError> {
        let storage = self.storage_mut()?;

        // Read current counter
        let current = match storage.get(tables::METADATA, metadata::NEXT_EDGE_ID) {
            Ok(Some(bytes)) if bytes.len() == 8 => {
                let arr: [u8; 8] = bytes
                    .try_into()
                    .map_err(|_| TransactionError::Internal("invalid counter".to_string()))?;
                u64::from_be_bytes(arr)
            }
            Ok(_) => 1, // Start from 1 if not set
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        // Increment and store
        let next = current + 1;
        storage
            .put(tables::METADATA, metadata::NEXT_EDGE_ID, &next.to_be_bytes())
            .map_err(storage_error_to_tx_error)?;

        Ok(EdgeId::new(current))
    }

    // ========================================================================
    // Traversal Operations
    // ========================================================================

    /// Get all outgoing edges from an entity.
    pub fn get_outgoing_edges(&self, entity_id: EntityId) -> Result<Vec<Edge>, TransactionError> {
        self.get_adjacent_edges(entity_id, tables::EDGES_OUT)
    }

    /// Get all incoming edges to an entity.
    pub fn get_incoming_edges(&self, entity_id: EntityId) -> Result<Vec<Edge>, TransactionError> {
        self.get_adjacent_edges(entity_id, tables::EDGES_IN)
    }

    /// Get adjacent edges from an index table.
    fn get_adjacent_edges(
        &self,
        entity_id: EntityId,
        index_table: &str,
    ) -> Result<Vec<Edge>, TransactionError> {
        use std::ops::Bound;

        let storage = self.storage()?;

        // Create range for this entity's edges
        // The index key format is: entity_id (8 bytes) + edge_id (8 bytes)
        // We want all keys that start with this entity_id
        let start_key = entity_id.as_u64().to_be_bytes().to_vec();
        // For the end key, use entity_id + 1 to capture all edge_ids for this entity
        let end_key = (entity_id.as_u64().wrapping_add(1)).to_be_bytes().to_vec();

        let mut cursor = storage
            .range(
                index_table,
                Bound::Included(start_key.as_slice()),
                Bound::Excluded(end_key.as_slice()),
            )
            .map_err(storage_error_to_tx_error)?;

        let mut edges = Vec::new();

        // Iterate through the range and collect edge IDs
        while let Some((key, _)) = cursor.next().map_err(storage_error_to_tx_error)? {
            if key.len() == 16 {
                // entity_id (8) + edge_id (8)
                let edge_id_bytes: [u8; 8] = key[8..16]
                    .try_into()
                    .map_err(|_| TransactionError::Internal("invalid key format".to_string()))?;
                let edge_id = EdgeId::new(u64::from_be_bytes(edge_id_bytes));

                if let Some(edge) = self.get_edge(edge_id)? {
                    edges.push(edge);
                }
            }
        }

        Ok(edges)
    }

    // ========================================================================
    // Metadata Operations
    // ========================================================================

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &[u8]) -> Result<Option<Vec<u8>>, TransactionError> {
        let storage = self.storage()?;
        storage.get(tables::METADATA, key).map_err(storage_error_to_tx_error)
    }

    /// Put a metadata value.
    pub fn put_metadata(&mut self, key: &[u8], value: &[u8]) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;
        storage.put(tables::METADATA, key, value).map_err(storage_error_to_tx_error)
    }

    /// Delete a metadata value.
    pub fn delete_metadata(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        let storage = self.storage_mut()?;
        storage.delete(tables::METADATA, key).map_err(storage_error_to_tx_error)
    }

    // ========================================================================
    // Transaction Lifecycle
    // ========================================================================

    /// Commit the transaction, making all changes durable.
    ///
    /// After commit, the transaction handle is consumed and cannot be used.
    pub fn commit(mut self) -> Result<(), TransactionError> {
        let storage = self.storage.take().ok_or(TransactionError::AlreadyCompleted)?;

        storage.commit().map_err(storage_error_to_tx_error)
    }

    /// Rollback the transaction, discarding all changes.
    ///
    /// This is typically implicit when a transaction is dropped without
    /// committing, but can be called explicitly for clarity.
    pub fn rollback(mut self) -> Result<(), TransactionError> {
        let storage = self.storage.take().ok_or(TransactionError::AlreadyCompleted)?;

        storage.rollback().map_err(storage_error_to_tx_error)
    }
}

impl<T: Transaction> Drop for DatabaseTransaction<T> {
    fn drop(&mut self) {
        // If storage is still Some, the transaction was not committed or rolled back.
        // We should attempt to roll back, but we can't return errors from drop.
        if let Some(storage) = self.storage.take() {
            // Best effort rollback - ignore errors since we can't propagate them
            let _ = storage.rollback();
        }
    }
}

/// Create a key for the adjacency index.
/// Format: `entity_id` (8 bytes big-endian) + `edge_id` (8 bytes big-endian)
fn make_adjacency_key(entity_id: EntityId, edge_id: EdgeId) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..8].copy_from_slice(&entity_id.as_u64().to_be_bytes());
    key[8..16].copy_from_slice(&edge_id.as_u64().to_be_bytes());
    key
}

/// Convert a storage error to a transaction error.
fn storage_error_to_tx_error(err: manifoldb_storage::StorageError) -> TransactionError {
    use manifoldb_storage::StorageError;

    match err {
        StorageError::ReadOnly => TransactionError::ReadOnly,
        StorageError::Conflict(msg) => TransactionError::Conflict(msg),
        StorageError::Serialization(msg) => TransactionError::Serialization(msg),
        StorageError::NotFound(msg) => TransactionError::EntityNotFound(msg),
        StorageError::KeyNotFound => TransactionError::EntityNotFound("key not found".to_string()),
        _ => TransactionError::Storage(err.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_key() {
        let entity_id = EntityId::new(42);
        let edge_id = EdgeId::new(123);

        let key = make_adjacency_key(entity_id, edge_id);

        // Verify the key contains both IDs in big-endian format
        assert_eq!(&key[0..8], &42u64.to_be_bytes());
        assert_eq!(&key[8..16], &123u64.to_be_bytes());
    }
}
