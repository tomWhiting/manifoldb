//! Node (entity) storage operations.
//!
//! This module provides CRUD operations for nodes in the graph.

use std::ops::Bound;

use manifoldb_core::encoding::keys::{
    decode_entity_key, decode_label_index_entity_id, encode_entity_key, encode_label_index_key,
    encode_label_index_prefix, PREFIX_ENTITY,
};
use manifoldb_core::encoding::{Decoder, Encoder};
use manifoldb_core::{Entity, EntityId, Label};
use manifoldb_storage::{Cursor, Transaction};

use super::error::{GraphError, GraphResult};
use super::IdGenerator;

/// Table name for entity data.
pub const TABLE_ENTITIES: &str = "entities";

/// Table name for label index.
pub const TABLE_LABELS: &str = "labels";

/// Node storage operations.
///
/// `NodeStore` provides transactional CRUD operations for graph nodes (entities).
/// All operations work within a transaction context for ACID guarantees.
///
/// # Example
///
/// ```ignore
/// use manifoldb_graph::store::{NodeStore, IdGenerator};
///
/// // Create a node
/// let gen = IdGenerator::new();
/// let entity = NodeStore::create(&mut tx, &gen, |id| {
///     Entity::new(id)
///         .with_label("Person")
///         .with_property("name", "Alice")
/// })?;
///
/// // Read it back
/// let retrieved = NodeStore::get(&tx, entity.id)?;
/// ```
pub struct NodeStore;

impl NodeStore {
    /// Create a new entity in the store.
    ///
    /// The provided function receives a new unique ID and should return
    /// the entity to store. The entity's ID will be set to the generated ID.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id_gen` - The ID generator
    /// * `builder` - A function that builds the entity given an ID
    ///
    /// # Returns
    ///
    /// The created entity with its assigned ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the entity cannot be stored.
    pub fn create<T: Transaction, F>(
        tx: &mut T,
        id_gen: &IdGenerator,
        builder: F,
    ) -> GraphResult<Entity>
    where
        F: FnOnce(EntityId) -> Entity,
    {
        let id = id_gen.next_entity_id();
        let entity = builder(id);

        // Encode and store the entity
        let key = encode_entity_key(id);
        let value = entity.encode()?;
        tx.put(TABLE_ENTITIES, &key, &value)?;

        // Index all labels
        for label in &entity.labels {
            let label_key = encode_label_index_key(label, id);
            tx.put(TABLE_LABELS, &label_key, &[])?;
        }

        Ok(entity)
    }

    /// Create an entity with a specific ID.
    ///
    /// This is useful when importing data or when you need to control IDs.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `entity` - The entity to store (must have a valid ID)
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EntityAlreadyExists`] if an entity with this ID exists.
    pub fn create_with_id<T: Transaction>(tx: &mut T, entity: &Entity) -> GraphResult<()> {
        let key = encode_entity_key(entity.id);

        // Check if entity already exists
        if tx.get(TABLE_ENTITIES, &key)?.is_some() {
            return Err(GraphError::EntityAlreadyExists(entity.id));
        }

        // Store the entity
        let value = entity.encode()?;
        tx.put(TABLE_ENTITIES, &key, &value)?;

        // Index all labels
        for label in &entity.labels {
            let label_key = encode_label_index_key(label, entity.id);
            tx.put(TABLE_LABELS, &label_key, &[])?;
        }

        Ok(())
    }

    /// Get an entity by ID.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The entity ID to look up
    ///
    /// # Returns
    ///
    /// The entity if found, or `None` if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the entity cannot be decoded.
    pub fn get<T: Transaction>(tx: &T, id: EntityId) -> GraphResult<Option<Entity>> {
        let key = encode_entity_key(id);
        match tx.get(TABLE_ENTITIES, &key)? {
            Some(value) => {
                let entity = Entity::decode(&value)?;
                Ok(Some(entity))
            }
            None => Ok(None),
        }
    }

    /// Get an entity by ID, returning an error if not found.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The entity ID to look up
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EntityNotFound`] if the entity doesn't exist.
    pub fn get_or_error<T: Transaction>(tx: &T, id: EntityId) -> GraphResult<Entity> {
        Self::get(tx, id)?.ok_or(GraphError::EntityNotFound(id))
    }

    /// Check if an entity exists.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The entity ID to check
    pub fn exists<T: Transaction>(tx: &T, id: EntityId) -> GraphResult<bool> {
        let key = encode_entity_key(id);
        Ok(tx.get(TABLE_ENTITIES, &key)?.is_some())
    }

    /// Update an existing entity.
    ///
    /// This replaces the entire entity. To update specific fields,
    /// first get the entity, modify it, then update.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `entity` - The entity with updated data
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::EntityNotFound`] if the entity doesn't exist.
    pub fn update<T: Transaction>(tx: &mut T, entity: &Entity) -> GraphResult<()> {
        let key = encode_entity_key(entity.id);

        // Get the old entity to update label indexes
        let old_value =
            tx.get(TABLE_ENTITIES, &key)?.ok_or(GraphError::EntityNotFound(entity.id))?;
        let old_entity = Entity::decode(&old_value)?;

        // Remove old label indexes
        for label in &old_entity.labels {
            let label_key = encode_label_index_key(label, entity.id);
            tx.delete(TABLE_LABELS, &label_key)?;
        }

        // Store updated entity
        let value = entity.encode()?;
        tx.put(TABLE_ENTITIES, &key, &value)?;

        // Add new label indexes
        for label in &entity.labels {
            let label_key = encode_label_index_key(label, entity.id);
            tx.put(TABLE_LABELS, &label_key, &[])?;
        }

        Ok(())
    }

    /// Delete an entity by ID.
    ///
    /// This also removes all label index entries for the entity.
    /// Note: This does NOT delete edges connected to this entity.
    /// Use [`crate::store::EdgeStore::delete_edges_for_entity`] to clean up edges first.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `id` - The entity ID to delete
    ///
    /// # Returns
    ///
    /// `true` if the entity was deleted, `false` if it didn't exist.
    pub fn delete<T: Transaction>(tx: &mut T, id: EntityId) -> GraphResult<bool> {
        let key = encode_entity_key(id);

        // Get the entity to clean up label indexes
        let Some(value) = tx.get(TABLE_ENTITIES, &key)? else {
            return Ok(false);
        };
        let entity = Entity::decode(&value)?;

        // Remove label indexes
        for label in &entity.labels {
            let label_key = encode_label_index_key(label, id);
            tx.delete(TABLE_LABELS, &label_key)?;
        }

        // Delete the entity
        tx.delete(TABLE_ENTITIES, &key)?;
        Ok(true)
    }

    /// Find all entities with a specific label.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `label` - The label to search for
    ///
    /// # Returns
    ///
    /// A vector of entity IDs that have the label.
    pub fn find_by_label<T: Transaction>(tx: &T, label: &Label) -> GraphResult<Vec<EntityId>> {
        let prefix = encode_label_index_prefix(label);

        // Create the end bound by incrementing the last byte of the prefix
        let mut end_prefix = prefix.clone();
        if let Some(last) = end_prefix.last_mut() {
            *last = last.saturating_add(1);
        }

        let mut cursor = tx.range(
            TABLE_LABELS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(end_prefix.as_slice()),
        )?;

        let mut ids = Vec::new();
        while let Some((key, _)) = cursor.next()? {
            if let Some(id) = decode_label_index_entity_id(&key) {
                ids.push(id);
            }
        }

        Ok(ids)
    }

    /// Count all entities in the store.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    pub fn count<T: Transaction>(tx: &T) -> GraphResult<usize> {
        let start = [PREFIX_ENTITY];
        let end = [PREFIX_ENTITY + 1];

        let cursor_result = tx.range(
            TABLE_ENTITIES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        );

        // Handle table not existing (empty store)
        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(0),
            Err(e) => return Err(e.into()),
        };

        let mut count = 0;
        while cursor.next()?.is_some() {
            count += 1;
        }

        Ok(count)
    }

    /// Iterate over all entities.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `f` - A function to call for each entity. Return `false` to stop iteration.
    ///
    /// # Errors
    ///
    /// Returns an error if iteration fails or if any entity cannot be decoded.
    pub fn for_each<T: Transaction, F>(tx: &T, mut f: F) -> GraphResult<()>
    where
        F: FnMut(&Entity) -> bool,
    {
        let start = [PREFIX_ENTITY];
        let end = [PREFIX_ENTITY + 1];

        let cursor_result = tx.range(
            TABLE_ENTITIES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        );

        // Handle table not existing (empty store)
        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };

        while let Some((_, value)) = cursor.next()? {
            let entity = Entity::decode(&value)?;
            if !f(&entity) {
                break;
            }
        }

        Ok(())
    }

    /// Get all entities as a vector.
    ///
    /// Use with caution on large datasets - prefer [`Self::for_each`] for
    /// processing entities without loading all into memory.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    pub fn all<T: Transaction>(tx: &T) -> GraphResult<Vec<Entity>> {
        let mut entities = Vec::new();
        Self::for_each(tx, |entity| {
            entities.push(entity.clone());
            true
        })?;
        Ok(entities)
    }

    /// Find the highest entity ID in the store.
    ///
    /// This is useful for initializing the ID generator after loading data.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    ///
    /// # Returns
    ///
    /// The highest entity ID, or `None` if there are no entities.
    pub fn max_id<T: Transaction>(tx: &T) -> GraphResult<Option<EntityId>> {
        let start = [PREFIX_ENTITY];
        let end = [PREFIX_ENTITY + 1];

        let cursor_result = tx.range(
            TABLE_ENTITIES,
            Bound::Included(start.as_slice()),
            Bound::Excluded(end.as_slice()),
        );

        // Handle table not existing (empty store)
        let mut cursor = match cursor_result {
            Ok(c) => c,
            Err(manifoldb_storage::StorageError::TableNotFound(_)) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        // Seek to the last key in the range
        if cursor.seek_last()?.is_some() {
            if let Some((key, _)) = cursor.current().map(|(k, v)| (k.to_vec(), v.to_vec())) {
                return Ok(decode_entity_key(&key));
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Integration tests with actual storage backend are in the tests/ directory

    #[test]
    fn table_names_are_valid() {
        assert!(!TABLE_ENTITIES.is_empty());
        assert!(!TABLE_LABELS.is_empty());
        assert_ne!(TABLE_ENTITIES, TABLE_LABELS);
    }
}
