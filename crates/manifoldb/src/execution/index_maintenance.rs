//! Index maintenance for entity mutations.
//!
//! This module handles keeping secondary indexes in sync when entities are
//! inserted, updated, or deleted. It works within existing transactions to
//! ensure atomic updates.
//!
//! # Architecture
//!
//! Index maintenance is performed at mutation time to keep indexes consistent.
//! The approach:
//!
//! 1. After an entity mutation, look up all indexes for the entity's table/label
//! 2. For each index, determine which columns are affected
//! 3. Insert/update/delete index entries as needed
//!
//! This module uses [`PropertyIndexEntry`] for encoding index entries and
//! [`SchemaManager`] for looking up index definitions.

use manifoldb_core::index::{IndexId, PropertyIndexEntry};
use manifoldb_core::{Entity, TransactionError};
use manifoldb_storage::Transaction;

use crate::schema::{IndexSchema, SchemaManager};
use crate::transaction::DatabaseTransaction;

/// Error type for index maintenance operations.
#[derive(Debug, thiserror::Error)]
pub enum IndexMaintenanceError {
    /// Transaction error during index update.
    #[error("transaction error: {0}")]
    Transaction(#[from] TransactionError),

    /// Schema lookup error.
    #[error("schema error: {0}")]
    Schema(String),
}

/// Index maintenance operations for entity property indexes.
///
/// This struct provides static methods for maintaining secondary indexes
/// when entities are mutated. All operations work within the provided
/// transaction to ensure atomicity.
pub struct EntityIndexMaintenance;

impl EntityIndexMaintenance {
    /// Add index entries for a newly inserted entity.
    ///
    /// Looks up all indexes for the entity's labels and creates entries
    /// for properties that are indexed.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `entity` - The newly inserted entity
    pub fn on_insert<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        entity: &Entity,
    ) -> Result<(), IndexMaintenanceError> {
        // Get all labels from the entity
        for label in &entity.labels {
            let label_str = label.as_str();

            // Get all indexes for this label/table
            let indexes = Self::get_indexes_for_table(tx, label_str)?;

            for index in indexes {
                // Skip non-btree indexes (HNSW handled separately)
                if let Some(using) = &index.using {
                    let using_lower = using.to_lowercase();
                    if using_lower == "hnsw" || using_lower == "ivfflat" {
                        continue;
                    }
                }

                // For each indexed column, add an entry
                Self::add_index_entries(tx, entity, label_str, &index)?;
            }
        }

        Ok(())
    }

    /// Update index entries for a modified entity.
    ///
    /// Compares old and new property values for indexed columns,
    /// removing old entries and adding new ones as needed.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `old_entity` - The entity before modification
    /// * `new_entity` - The entity after modification
    pub fn on_update<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        old_entity: &Entity,
        new_entity: &Entity,
    ) -> Result<(), IndexMaintenanceError> {
        // Handle labels that were removed
        for old_label in &old_entity.labels {
            if !new_entity.labels.contains(old_label) {
                let old_label_str = old_label.as_str();
                // Label removed - remove all index entries for this label
                let indexes = Self::get_indexes_for_table(tx, old_label_str)?;
                for index in indexes {
                    if let Some(using) = &index.using {
                        let using_lower = using.to_lowercase();
                        if using_lower == "hnsw" || using_lower == "ivfflat" {
                            continue;
                        }
                    }
                    Self::remove_index_entries(tx, old_entity, old_label_str, &index)?;
                }
            }
        }

        // Handle labels that are still present or were added
        for label in &new_entity.labels {
            let label_str = label.as_str();
            let indexes = Self::get_indexes_for_table(tx, label_str)?;

            for index in indexes {
                if let Some(using) = &index.using {
                    let using_lower = using.to_lowercase();
                    if using_lower == "hnsw" || using_lower == "ivfflat" {
                        continue;
                    }
                }

                let is_new_label = !old_entity.labels.contains(label);

                if is_new_label {
                    // New label - add all index entries
                    Self::add_index_entries(tx, new_entity, label_str, &index)?;
                } else {
                    // Existing label - update changed columns
                    Self::update_index_entries(tx, old_entity, new_entity, label_str, &index)?;
                }
            }
        }

        Ok(())
    }

    /// Remove index entries for a deleted entity.
    ///
    /// Removes all index entries for properties that were indexed.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to use
    /// * `entity` - The entity being deleted
    pub fn on_delete<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        entity: &Entity,
    ) -> Result<(), IndexMaintenanceError> {
        // Get all labels from the entity
        for label in &entity.labels {
            let label_str = label.as_str();

            // Get all indexes for this label/table
            let indexes = Self::get_indexes_for_table(tx, label_str)?;

            for index in indexes {
                // Skip non-btree indexes (HNSW handled separately)
                if let Some(using) = &index.using {
                    let using_lower = using.to_lowercase();
                    if using_lower == "hnsw" || using_lower == "ivfflat" {
                        continue;
                    }
                }

                // For each indexed column, remove the entry
                Self::remove_index_entries(tx, entity, label_str, &index)?;
            }
        }

        Ok(())
    }

    /// Get all indexes for a table/label.
    fn get_indexes_for_table<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        table_name: &str,
    ) -> Result<Vec<IndexSchema>, IndexMaintenanceError> {
        let index_names = SchemaManager::list_indexes_for_table(tx, table_name)
            .map_err(|e| IndexMaintenanceError::Schema(e.to_string()))?;

        let mut indexes = Vec::new();
        for name in index_names {
            if let Some(index) = SchemaManager::get_index(tx, &name)
                .map_err(|e| IndexMaintenanceError::Schema(e.to_string()))?
            {
                indexes.push(index);
            }
        }

        Ok(indexes)
    }

    /// Add index entries for an entity.
    fn add_index_entries<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        entity: &Entity,
        label: &str,
        index: &IndexSchema,
    ) -> Result<(), IndexMaintenanceError> {
        // For now, only support single-column indexes
        if index.columns.len() != 1 {
            return Ok(());
        }

        let column_name = &index.columns[0].expr;

        // Get the property value
        if let Some(value) = entity.properties.get(column_name) {
            // Only index scalar values
            if PropertyIndexEntry::is_indexable(value) {
                let index_id = IndexId::from_label_property(label, column_name);
                let entry = PropertyIndexEntry::new(index_id, value.clone(), entity.id);

                if let Some(key) = entry.encode_key() {
                    tx.put_property_index(&key)?;
                }
            }
        }

        Ok(())
    }

    /// Remove index entries for an entity.
    fn remove_index_entries<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        entity: &Entity,
        label: &str,
        index: &IndexSchema,
    ) -> Result<(), IndexMaintenanceError> {
        // For now, only support single-column indexes
        if index.columns.len() != 1 {
            return Ok(());
        }

        let column_name = &index.columns[0].expr;

        // Get the property value
        if let Some(value) = entity.properties.get(column_name) {
            // Only index scalar values
            if PropertyIndexEntry::is_indexable(value) {
                let index_id = IndexId::from_label_property(label, column_name);
                let entry = PropertyIndexEntry::new(index_id, value.clone(), entity.id);

                if let Some(key) = entry.encode_key() {
                    tx.delete_property_index(&key)?;
                }
            }
        }

        Ok(())
    }

    /// Update index entries for changed columns.
    fn update_index_entries<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        old_entity: &Entity,
        new_entity: &Entity,
        label: &str,
        index: &IndexSchema,
    ) -> Result<(), IndexMaintenanceError> {
        // For now, only support single-column indexes
        if index.columns.len() != 1 {
            return Ok(());
        }

        let column_name = &index.columns[0].expr;
        let old_value = old_entity.properties.get(column_name);
        let new_value = new_entity.properties.get(column_name);

        // Check if value changed
        if old_value == new_value {
            return Ok(());
        }

        let index_id = IndexId::from_label_property(label, column_name);

        // Remove old entry if it existed and was indexable
        if let Some(old_val) = old_value {
            if PropertyIndexEntry::is_indexable(old_val) {
                let entry = PropertyIndexEntry::new(index_id, old_val.clone(), old_entity.id);
                if let Some(key) = entry.encode_key() {
                    tx.delete_property_index(&key)?;
                }
            }
        }

        // Add new entry if it exists and is indexable
        if let Some(new_val) = new_value {
            if PropertyIndexEntry::is_indexable(new_val) {
                let entry = PropertyIndexEntry::new(index_id, new_val.clone(), new_entity.id);
                if let Some(key) = entry.encode_key() {
                    tx.put_property_index(&key)?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Integration tests are in tests/index_maintenance_tests.rs
    // Unit tests would require mocking the Transaction trait
}
