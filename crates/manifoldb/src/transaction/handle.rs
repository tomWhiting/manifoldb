//! Database transaction handle for user operations.

use manifoldb_core::encoding::keys::encode_edge_key;
use manifoldb_core::encoding::{Decoder, Encoder};
use manifoldb_core::{DeleteResult, Edge, EdgeId, Entity, EntityId, TransactionError};
use manifoldb_graph::index::IndexMaintenance;
use manifoldb_storage::{Cursor, Transaction};

use super::VectorSyncStrategy;

/// Well-known table names for graph storage.
mod tables {
    pub const NODES: &str = "entities";
    pub const EDGES: &str = "edges";
    pub const EDGES_OUT: &str = "edges_out";
    pub const EDGES_IN: &str = "edges_in";
    pub const METADATA: &str = "metadata";
    pub const PROPERTY_INDEX: &str = "property_index";
    /// Index mapping (label, entity_id) -> () for efficient label-based queries.
    pub const LABEL_INDEX: &str = "label_index";
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

    /// Get a mutable reference to the underlying storage transaction for direct access.
    ///
    /// This is useful for operations that need low-level access to the storage layer,
    /// such as storing vectors in dedicated tables separate from entity properties.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The transaction is read-only
    /// - The transaction has already been committed or rolled back
    pub fn storage_mut(&mut self) -> Result<&mut T, TransactionError> {
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
                let (entity, _): (Entity, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
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
    /// This also maintains the label index.
    pub fn put_entity(&mut self, entity: &Entity) -> Result<(), TransactionError> {
        // First, check if entity already exists to remove old label index entries
        let old_entity = self.get_entity(entity.id)?;

        let storage = self.storage_mut()?;
        let key = entity.id.as_u64().to_be_bytes();
        let value = bincode::serde::encode_to_vec(entity, bincode::config::standard())
            .map_err(|e| TransactionError::Serialization(e.to_string()))?;

        storage.put(tables::NODES, &key, &value).map_err(storage_error_to_tx_error)?;

        // Remove old label index entries if entity existed
        if let Some(old) = old_entity {
            for label in &old.labels {
                let label_key = make_label_key(label.as_str(), old.id);
                storage
                    .delete(tables::LABEL_INDEX, &label_key)
                    .map_err(storage_error_to_tx_error)?;
            }
        }

        // Add new label index entries
        for label in &entity.labels {
            let label_key = make_label_key(label.as_str(), entity.id);
            storage.put(tables::LABEL_INDEX, &label_key, &[]).map_err(storage_error_to_tx_error)?;
        }

        Ok(())
    }

    /// Delete an entity by its ID.
    ///
    /// Returns `true` if the entity existed and was deleted, `false` if it didn't exist.
    ///
    /// Note: This does not automatically delete edges connected to this entity.
    /// The caller is responsible for maintaining referential integrity.
    /// Consider using [`delete_entity_cascade`](Self::delete_entity_cascade) or
    /// [`delete_entity_checked`](Self::delete_entity_checked) for safer deletion.
    pub fn delete_entity(&mut self, id: EntityId) -> Result<bool, TransactionError> {
        // First get the entity to find its labels for index cleanup
        let entity = self.get_entity(id)?;

        let storage = self.storage_mut()?;
        let key = id.as_u64().to_be_bytes();

        let deleted = storage.delete(tables::NODES, &key).map_err(storage_error_to_tx_error)?;

        // Remove label index entries if entity existed
        if let Some(entity) = entity {
            for label in &entity.labels {
                let label_key = make_label_key(label.as_str(), entity.id);
                storage
                    .delete(tables::LABEL_INDEX, &label_key)
                    .map_err(storage_error_to_tx_error)?;
            }
        }

        Ok(deleted)
    }

    /// Delete an entity and all edges connected to it (cascade delete).
    ///
    /// This method ensures referential integrity by deleting all edges where
    /// the entity is either the source or target before deleting the entity itself.
    ///
    /// # Returns
    ///
    /// Returns a [`DeleteResult`] containing:
    /// - `entity_deleted`: `true` if the entity existed and was deleted
    /// - `edges_deleted`: Vector of edge IDs that were deleted
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = tx.delete_entity_cascade(entity_id)?;
    /// println!("Entity deleted: {}", result.entity_deleted);
    /// println!("Edges deleted: {}", result.edges_deleted_count());
    /// tx.commit()?;
    /// ```
    pub fn delete_entity_cascade(
        &mut self,
        id: EntityId,
    ) -> Result<DeleteResult, TransactionError> {
        // Collect all edges connected to this entity (both incoming and outgoing)
        let outgoing = self.get_outgoing_edges(id)?;
        let incoming = self.get_incoming_edges(id)?;

        // Collect edge IDs to delete, avoiding duplicates for self-loops
        let mut edges_to_delete: Vec<EdgeId> = Vec::with_capacity(outgoing.len() + incoming.len());
        for edge in &outgoing {
            edges_to_delete.push(edge.id);
        }
        for edge in &incoming {
            // Avoid duplicates (self-loops appear in both lists)
            if edge.source != edge.target {
                edges_to_delete.push(edge.id);
            }
        }

        // Delete all connected edges
        let mut deleted_edges = Vec::with_capacity(edges_to_delete.len());
        for edge_id in edges_to_delete {
            if self.delete_edge(edge_id)? {
                deleted_edges.push(edge_id);
            }
        }

        // Delete the entity itself
        let entity_deleted = self.delete_entity(id)?;

        Ok(DeleteResult::new(entity_deleted, deleted_edges))
    }

    /// Delete an entity only if it has no connected edges (checked delete).
    ///
    /// This method ensures referential integrity by refusing to delete an entity
    /// that still has edges connected to it. Use this when you want to enforce
    /// that edges are explicitly deleted before their endpoints.
    ///
    /// # Returns
    ///
    /// Returns `true` if the entity was deleted, `false` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns [`TransactionError::ReferentialIntegrity`] if the entity has
    /// connected edges (either incoming or outgoing).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // First delete all edges manually
    /// tx.delete_edge(edge_id)?;
    ///
    /// // Then delete the entity (will fail if edges remain)
    /// match tx.delete_entity_checked(entity_id) {
    ///     Ok(deleted) => println!("Deleted: {deleted}"),
    ///     Err(TransactionError::ReferentialIntegrity(msg)) => {
    ///         println!("Cannot delete: {msg}");
    ///     }
    ///     Err(e) => return Err(e),
    /// }
    /// ```
    pub fn delete_entity_checked(&mut self, id: EntityId) -> Result<bool, TransactionError> {
        // Check if the entity has any connected edges
        if self.has_edges(id)? {
            return Err(TransactionError::ReferentialIntegrity(format!(
                "cannot delete entity {}: has connected edges",
                id.as_u64()
            )));
        }

        self.delete_entity(id)
    }

    /// Check if an entity has any connected edges (incoming or outgoing).
    ///
    /// This is useful for checking referential integrity before deleting an entity.
    ///
    /// # Returns
    ///
    /// Returns `true` if the entity has at least one edge where it is the source or target.
    pub fn has_edges(&self, entity_id: EntityId) -> Result<bool, TransactionError> {
        // Check outgoing edges first
        if self.has_adjacent_edges(entity_id, tables::EDGES_OUT)? {
            return Ok(true);
        }

        // Check incoming edges
        self.has_adjacent_edges(entity_id, tables::EDGES_IN)
    }

    /// Check if an entity has any adjacent edges in the given index table.
    fn has_adjacent_edges(
        &self,
        entity_id: EntityId,
        index_table: &str,
    ) -> Result<bool, TransactionError> {
        use std::ops::Bound;

        let storage = self.storage()?;

        // Create range for this entity's edges
        let start_key = entity_id.as_u64().to_be_bytes().to_vec();
        let end_key = (entity_id.as_u64().wrapping_add(1)).to_be_bytes().to_vec();

        let mut cursor = match storage.range(
            index_table,
            Bound::Included(start_key.as_slice()),
            Bound::Excluded(end_key.as_slice()),
        ) {
            Ok(c) => c,
            Err(manifoldb_storage::StorageError::TableNotFound(_)) => {
                // Table doesn't exist, so no edges
                return Ok(false);
            }
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        // Just check if there's at least one entry
        Ok(cursor.next().map_err(storage_error_to_tx_error)?.is_some())
    }

    /// Put multiple entities into the database in a single batch operation.
    ///
    /// This is significantly more efficient than calling `put_entity` multiple times
    /// as it minimizes transaction overhead and enables bulk write optimizations.
    ///
    /// If any entity with the same ID already exists, it will be replaced.
    /// This also maintains the label index for all entities.
    ///
    /// # Performance
    ///
    /// For bulk loading, this can provide up to 10x throughput improvement over
    /// individual `put_entity` calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let entities = vec![
    ///     Entity::new(EntityId::new(1)).with_label("Person"),
    ///     Entity::new(EntityId::new(2)).with_label("Person"),
    ///     Entity::new(EntityId::new(3)).with_label("Company"),
    /// ];
    /// tx.put_entities_batch(&entities)?;
    /// ```
    pub fn put_entities_batch(&mut self, entities: &[Entity]) -> Result<(), TransactionError> {
        // First, collect old entities to clean up their label indexes
        let mut old_entities = Vec::with_capacity(entities.len());
        for entity in entities {
            old_entities.push(self.get_entity(entity.id)?);
        }

        let storage = self.storage_mut()?;

        for (entity, old_entity) in entities.iter().zip(old_entities.into_iter()) {
            let key = entity.id.as_u64().to_be_bytes();
            let value = bincode::serde::encode_to_vec(entity, bincode::config::standard())
                .map_err(|e| TransactionError::Serialization(e.to_string()))?;
            storage.put(tables::NODES, &key, &value).map_err(storage_error_to_tx_error)?;

            // Remove old label index entries if entity existed
            if let Some(old) = old_entity {
                for label in &old.labels {
                    let label_key = make_label_key(label.as_str(), old.id);
                    storage
                        .delete(tables::LABEL_INDEX, &label_key)
                        .map_err(storage_error_to_tx_error)?;
                }
            }

            // Add new label index entries
            for label in &entity.labels {
                let label_key = make_label_key(label.as_str(), entity.id);
                storage
                    .put(tables::LABEL_INDEX, &label_key, &[])
                    .map_err(storage_error_to_tx_error)?;
            }
        }

        Ok(())
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
        let key = encode_edge_key(id);

        match storage.get(tables::EDGES, &key) {
            Ok(Some(bytes)) => {
                let edge = Edge::decode(&bytes)
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
    ///
    /// The edge is stored with a format compatible with the graph traversal layer.
    pub fn put_edge(&mut self, edge: &Edge) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;

        // Use the same key encoding as EdgeStore for graph layer compatibility
        let key = encode_edge_key(edge.id);
        // Use the same value encoding as EdgeStore for graph layer compatibility
        let value = edge.encode().map_err(|e| TransactionError::Serialization(e.to_string()))?;

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

    /// Put multiple edges into the database in a single batch operation.
    ///
    /// This is significantly more efficient than calling `put_edge` multiple times
    /// as it minimizes transaction overhead and enables bulk write optimizations.
    /// All edges and their indexes are written within the same transaction.
    ///
    /// If any edge with the same ID already exists, it will be replaced.
    ///
    /// # Performance
    ///
    /// For bulk loading, this can provide up to 10x throughput improvement over
    /// individual `put_edge` calls.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let edges = vec![
    ///     Edge::new(EdgeId::new(1), source1, target1, "FOLLOWS"),
    ///     Edge::new(EdgeId::new(2), source2, target2, "FOLLOWS"),
    ///     Edge::new(EdgeId::new(3), source3, target3, "LIKES"),
    /// ];
    /// tx.put_edges_batch(&edges)?;
    /// ```
    pub fn put_edges_batch(&mut self, edges: &[Edge]) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;

        for edge in edges {
            // Use the same key encoding as EdgeStore for graph layer compatibility
            let key = encode_edge_key(edge.id);
            // Use the same value encoding as EdgeStore for graph layer compatibility
            let value =
                edge.encode().map_err(|e| TransactionError::Serialization(e.to_string()))?;

            // Store the edge data
            storage.put(tables::EDGES, &key, &value).map_err(storage_error_to_tx_error)?;

            // Update simple outgoing edge index (source -> edge)
            let out_key = make_adjacency_key(edge.source, edge.id);
            storage.put(tables::EDGES_OUT, &out_key, &[]).map_err(storage_error_to_tx_error)?;

            // Update simple incoming edge index (target -> edge)
            let in_key = make_adjacency_key(edge.target, edge.id);
            storage.put(tables::EDGES_IN, &in_key, &[]).map_err(storage_error_to_tx_error)?;

            // Update graph layer indexes (edges_by_source, edges_by_target, edge_types)
            IndexMaintenance::add_edge_indexes(storage, edge)
                .map_err(|e| TransactionError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Iterate over all entities, optionally filtering by label.
    ///
    /// If `label` is `Some`, only entities with that label are returned using
    /// the label index for efficient lookup (no full table scan).
    /// If `label` is `None`, all entities are returned (requires full table scan).
    ///
    /// Returns an empty vector if no entities exist (including if the table hasn't been created).
    pub fn iter_entities(&self, label: Option<&str>) -> Result<Vec<Entity>, TransactionError> {
        use manifoldb_storage::StorageError;
        use std::ops::Bound;

        let storage = self.storage()?;

        // If a label is specified, use the label index for efficient lookup
        if let Some(label_filter) = label {
            return self.iter_entities_by_label(label_filter);
        }

        // No label filter - scan all nodes
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
            let (entity, _): (Entity, _) =
                bincode::serde::decode_from_slice(&value, bincode::config::standard())
                    .map_err(|e| TransactionError::Serialization(e.to_string()))?;
            entities.push(entity);
        }

        Ok(entities)
    }

    /// Iterate over entities with a specific label using the label index.
    ///
    /// This is an efficient O(k) lookup where k is the number of entities with
    /// the given label, rather than O(n) for a full table scan.
    fn iter_entities_by_label(&self, label: &str) -> Result<Vec<Entity>, TransactionError> {
        use manifoldb_storage::StorageError;
        use std::ops::Bound;

        let storage = self.storage()?;

        // Create range bounds for the label index
        let start_key = make_label_scan_start(label);
        let end_key = make_label_scan_end(label);

        let cursor_result = storage.range(
            tables::LABEL_INDEX,
            Bound::Included(start_key.as_slice()),
            Bound::Excluded(end_key.as_slice()),
        );

        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(StorageError::TableNotFound(_)) => {
                // Label index table doesn't exist, no entities yet
                return Ok(Vec::new());
            }
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        let mut entities = Vec::new();

        // Iterate through the label index and fetch each entity
        while let Some((key, _)) = cursor.next().map_err(storage_error_to_tx_error)? {
            // Extract entity_id from the key
            // Key format: length (2 bytes) + label (variable) + entity_id (8 bytes)
            if key.len() >= 10 {
                // Minimum: 2 + 0 + 8 (empty label edge case)
                let entity_id_offset = key.len() - 8;
                let entity_id_bytes: [u8; 8] =
                    key[entity_id_offset..].try_into().map_err(|_| {
                        TransactionError::Internal("invalid label index key".to_string())
                    })?;
                let entity_id = EntityId::new(u64::from_be_bytes(entity_id_bytes));

                if let Some(entity) = self.get_entity(entity_id)? {
                    entities.push(entity);
                }
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
        let key = encode_edge_key(id);

        // Delete the edge data
        let deleted = storage.delete(tables::EDGES, &key).map_err(storage_error_to_tx_error)?;

        if deleted {
            // Remove from simple outgoing edge index
            let out_key = make_adjacency_key(edge.source, edge.id);
            storage.delete(tables::EDGES_OUT, &out_key).map_err(storage_error_to_tx_error)?;

            // Remove from simple incoming edge index
            let in_key = make_adjacency_key(edge.target, edge.id);
            storage.delete(tables::EDGES_IN, &in_key).map_err(storage_error_to_tx_error)?;

            // Remove from graph layer indexes
            IndexMaintenance::remove_edge_indexes(storage, &edge)
                .map_err(|e| TransactionError::Storage(e.to_string()))?;
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
    // Property Index Operations
    // ========================================================================

    /// Put a property index entry.
    ///
    /// The key should be encoded using `PropertyIndexEntry::encode_key()`.
    pub fn put_property_index(&mut self, key: &[u8]) -> Result<(), TransactionError> {
        let storage = self.storage_mut()?;
        // Property index entries have no value, just the key
        storage.put(tables::PROPERTY_INDEX, key, &[]).map_err(storage_error_to_tx_error)
    }

    /// Delete a property index entry.
    ///
    /// Returns `true` if the entry existed and was deleted.
    pub fn delete_property_index(&mut self, key: &[u8]) -> Result<bool, TransactionError> {
        let storage = self.storage_mut()?;
        storage.delete(tables::PROPERTY_INDEX, key).map_err(storage_error_to_tx_error)
    }

    /// Scan property index entries in a range.
    ///
    /// Returns all keys in the range [start, end).
    pub fn scan_property_index(
        &self,
        start: &[u8],
        end: &[u8],
    ) -> Result<Vec<Vec<u8>>, TransactionError> {
        use std::ops::Bound;

        let storage = self.storage()?;

        // Handle table not existing (no entries yet)
        let cursor_result =
            storage.range(tables::PROPERTY_INDEX, Bound::Included(start), Bound::Excluded(end));

        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(manifoldb_storage::StorageError::TableNotFound(_)) => {
                return Ok(Vec::new());
            }
            Err(e) => return Err(storage_error_to_tx_error(e)),
        };

        let mut keys = Vec::new();
        while let Some((key, _)) = cursor.next().map_err(storage_error_to_tx_error)? {
            keys.push(key);
        }

        Ok(keys)
    }

    /// Delete all property index entries in a range.
    ///
    /// Returns the number of entries deleted.
    pub fn delete_property_index_range(
        &mut self,
        start: &[u8],
        end: &[u8],
    ) -> Result<usize, TransactionError> {
        // First collect all keys in the range
        let keys = self.scan_property_index(start, end)?;
        let count = keys.len();

        // Then delete them
        let storage = self.storage_mut()?;
        for key in keys {
            storage.delete(tables::PROPERTY_INDEX, &key).map_err(storage_error_to_tx_error)?;
        }

        Ok(count)
    }

    // ========================================================================
    // Low-Level Storage Access
    // ========================================================================

    /// Get a mutable reference to the underlying storage transaction for direct access.
    ///
    /// This is useful for advanced operations like vector index maintenance that
    /// need low-level access to the storage layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is read-only or has already been
    /// committed/rolled back.
    pub fn storage_mut_ref(&mut self) -> Result<&mut T, TransactionError> {
        self.storage_mut()
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

/// Create a key for the label index.
/// Format: `label_length` (2 bytes big-endian) + `label` (variable) + `entity_id` (8 bytes big-endian)
///
/// The length prefix ensures labels don't collide (e.g., "ab" vs "a" followed by entity starting with 'b').
fn make_label_key(label: &str, entity_id: EntityId) -> Vec<u8> {
    let label_bytes = label.as_bytes();
    let len = label_bytes.len() as u16;
    let mut key = Vec::with_capacity(2 + label_bytes.len() + 8);
    key.extend_from_slice(&len.to_be_bytes());
    key.extend_from_slice(label_bytes);
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Create the start key for scanning all entities with a given label.
fn make_label_scan_start(label: &str) -> Vec<u8> {
    let label_bytes = label.as_bytes();
    let len = label_bytes.len() as u16;
    let mut key = Vec::with_capacity(2 + label_bytes.len());
    key.extend_from_slice(&len.to_be_bytes());
    key.extend_from_slice(label_bytes);
    key
}

/// Create the end key for scanning all entities with a given label.
/// Returns the start key for the "next" label (length prefix incremented or label bytes incremented).
fn make_label_scan_end(label: &str) -> Vec<u8> {
    let label_bytes = label.as_bytes();
    let len = label_bytes.len() as u16;
    let mut key = Vec::with_capacity(2 + label_bytes.len() + 8);
    key.extend_from_slice(&len.to_be_bytes());
    key.extend_from_slice(label_bytes);
    // Append maximum entity_id + 1 (effectively infinity for this label prefix)
    key.extend_from_slice(&u64::MAX.to_be_bytes());
    // Then add one more byte to ensure we're past all valid keys for this label
    key.push(0);
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
