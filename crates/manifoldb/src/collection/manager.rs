//! Collection metadata management.
//!
//! This module provides the `CollectionManager` for storing and retrieving
//! collection metadata in the database.

use manifoldb_core::{CollectionId, TransactionError};
use manifoldb_storage::Transaction;

use crate::transaction::DatabaseTransaction;

use super::metadata::{Collection, CollectionName, CollectionNameError};
use super::VectorConfig;

/// Prefix for collection metadata keys.
const COLLECTION_PREFIX: &[u8] = b"collection:";
/// Prefix for collection name to ID mapping.
const COLLECTION_NAME_PREFIX: &[u8] = b"collection_name:";
/// Key for the list of all collection names.
const COLLECTIONS_LIST_KEY: &[u8] = b"collections:list";
/// Key for the collection ID counter.
const COLLECTION_ID_COUNTER_KEY: &[u8] = b"collections:id_counter";

/// Manager for collection metadata operations.
///
/// The `CollectionManager` handles CRUD operations for collections,
/// including storage, retrieval, and updates. It stores metadata
/// in the database's metadata table using bincode serialization.
pub struct CollectionManager;

impl CollectionManager {
    /// Create a new collection.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A collection with the same name already exists
    /// - There's a storage error
    pub fn create<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
        vectors: impl IntoIterator<Item = (String, VectorConfig)>,
    ) -> Result<Collection, CollectionError> {
        // Check if collection already exists
        if Self::exists(tx, name)? {
            return Err(CollectionError::AlreadyExists(name.as_str().to_string()));
        }

        // Get next collection ID
        let id = Self::next_id(tx)?;

        // Create the collection
        let mut collection = Collection::new(id, name.clone());
        for (vector_name, config) in vectors {
            collection.add_vector(vector_name, config);
        }

        // Store the collection
        Self::store(tx, &collection)?;

        // Add to the collections list
        Self::add_to_list(tx, name)?;

        // Store the name -> ID mapping
        Self::store_name_mapping(tx, name, id)?;

        Ok(collection)
    }

    /// Get a collection by name.
    ///
    /// Returns `None` if the collection doesn't exist.
    pub fn get<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &CollectionName,
    ) -> Result<Option<Collection>, CollectionError> {
        // Look up the collection ID
        let id = match Self::get_id_by_name(tx, name)? {
            Some(id) => id,
            None => return Ok(None),
        };

        // Get the collection by ID
        Self::get_by_id(tx, id)
    }

    /// Get a collection by ID.
    ///
    /// Returns `None` if the collection doesn't exist.
    pub fn get_by_id<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        id: CollectionId,
    ) -> Result<Option<Collection>, CollectionError> {
        let key = Self::collection_key(id);
        match tx.get_metadata(&key)? {
            Some(bytes) => {
                let (collection, _): (Collection, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| CollectionError::Serialization(e.to_string()))?;
                Ok(Some(collection))
            }
            None => Ok(None),
        }
    }

    /// Check if a collection exists.
    pub fn exists<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &CollectionName,
    ) -> Result<bool, CollectionError> {
        let key = Self::name_key(name);
        Ok(tx.get_metadata(&key)?.is_some())
    }

    /// Update a collection.
    ///
    /// This replaces the entire collection metadata. Use the specific
    /// methods like `add_vector` or `remove_vector` for partial updates.
    pub fn update<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        collection: &Collection,
    ) -> Result<(), CollectionError> {
        // Verify the collection exists
        if Self::get_by_id(tx, collection.id())?.is_none() {
            return Err(CollectionError::NotFound(collection.name().as_str().to_string()));
        }

        Self::store(tx, collection)
    }

    /// Delete a collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist (unless `if_exists` is true).
    pub fn delete<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
        if_exists: bool,
    ) -> Result<(), CollectionError> {
        // Get the collection ID
        let id = match Self::get_id_by_name(tx, name)? {
            Some(id) => id,
            None => {
                if if_exists {
                    return Ok(());
                }
                return Err(CollectionError::NotFound(name.as_str().to_string()));
            }
        };

        // Delete the collection metadata
        let key = Self::collection_key(id);
        tx.delete_metadata(&key)?;

        // Delete the name -> ID mapping
        let name_key = Self::name_key(name);
        tx.delete_metadata(&name_key)?;

        // Remove from the collections list
        Self::remove_from_list(tx, name)?;

        Ok(())
    }

    /// List all collection names.
    pub fn list<T: Transaction>(
        tx: &DatabaseTransaction<T>,
    ) -> Result<Vec<CollectionName>, CollectionError> {
        match tx.get_metadata(COLLECTIONS_LIST_KEY)? {
            Some(bytes) => {
                let (names, _): (Vec<String>, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| CollectionError::Serialization(e.to_string()))?;
                names
                    .into_iter()
                    .map(|n| CollectionName::new(n).map_err(CollectionError::InvalidName))
                    .collect()
            }
            None => Ok(Vec::new()),
        }
    }

    /// Add a vector to an existing collection.
    pub fn add_vector<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
        vector_name: impl Into<String>,
        config: VectorConfig,
    ) -> Result<(), CollectionError> {
        let mut collection = Self::get(tx, name)?
            .ok_or_else(|| CollectionError::NotFound(name.as_str().to_string()))?;

        let vector_name = vector_name.into();
        if collection.has_vector(&vector_name) {
            return Err(CollectionError::VectorExists(vector_name));
        }

        collection.add_vector(vector_name, config);
        Self::store(tx, &collection)
    }

    /// Remove a vector from an existing collection.
    pub fn remove_vector<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
        vector_name: &str,
    ) -> Result<VectorConfig, CollectionError> {
        let mut collection = Self::get(tx, name)?
            .ok_or_else(|| CollectionError::NotFound(name.as_str().to_string()))?;

        let config = collection
            .remove_vector(vector_name)
            .ok_or_else(|| CollectionError::VectorNotFound(vector_name.to_string()))?;

        Self::store(tx, &collection)?;
        Ok(config)
    }

    // Internal helper methods

    fn next_id<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
    ) -> Result<CollectionId, CollectionError> {
        let current = match tx.get_metadata(COLLECTION_ID_COUNTER_KEY)? {
            Some(bytes) => {
                let (id, _): (u64, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| CollectionError::Serialization(e.to_string()))?;
                id
            }
            None => 0,
        };

        let next = current + 1;
        let value = bincode::serde::encode_to_vec(next, bincode::config::standard())
            .map_err(|e| CollectionError::Serialization(e.to_string()))?;
        tx.put_metadata(COLLECTION_ID_COUNTER_KEY, &value)?;

        Ok(CollectionId::new(next))
    }

    fn store<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        collection: &Collection,
    ) -> Result<(), CollectionError> {
        let key = Self::collection_key(collection.id());
        let value = bincode::serde::encode_to_vec(collection, bincode::config::standard())
            .map_err(|e| CollectionError::Serialization(e.to_string()))?;
        tx.put_metadata(&key, &value)?;
        Ok(())
    }

    fn get_id_by_name<T: Transaction>(
        tx: &DatabaseTransaction<T>,
        name: &CollectionName,
    ) -> Result<Option<CollectionId>, CollectionError> {
        let key = Self::name_key(name);
        match tx.get_metadata(&key)? {
            Some(bytes) => {
                let (id, _): (u64, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| CollectionError::Serialization(e.to_string()))?;
                Ok(Some(CollectionId::new(id)))
            }
            None => Ok(None),
        }
    }

    fn store_name_mapping<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
        id: CollectionId,
    ) -> Result<(), CollectionError> {
        let key = Self::name_key(name);
        let value = bincode::serde::encode_to_vec(id.as_u64(), bincode::config::standard())
            .map_err(|e| CollectionError::Serialization(e.to_string()))?;
        tx.put_metadata(&key, &value)?;
        Ok(())
    }

    fn collection_key(id: CollectionId) -> Vec<u8> {
        let mut key = COLLECTION_PREFIX.to_vec();
        key.extend_from_slice(&id.as_u64().to_be_bytes());
        key
    }

    fn name_key(name: &CollectionName) -> Vec<u8> {
        let mut key = COLLECTION_NAME_PREFIX.to_vec();
        key.extend_from_slice(name.as_str().as_bytes());
        key
    }

    fn add_to_list<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
    ) -> Result<(), CollectionError> {
        let mut list = Self::get_list(tx)?;
        if !list.contains(&name.as_str().to_string()) {
            list.push(name.as_str().to_string());
            let value = bincode::serde::encode_to_vec(&list, bincode::config::standard())
                .map_err(|e| CollectionError::Serialization(e.to_string()))?;
            tx.put_metadata(COLLECTIONS_LIST_KEY, &value)?;
        }
        Ok(())
    }

    fn remove_from_list<T: Transaction>(
        tx: &mut DatabaseTransaction<T>,
        name: &CollectionName,
    ) -> Result<(), CollectionError> {
        let mut list = Self::get_list(tx)?;
        list.retain(|n| n != name.as_str());
        let value = bincode::serde::encode_to_vec(&list, bincode::config::standard())
            .map_err(|e| CollectionError::Serialization(e.to_string()))?;
        tx.put_metadata(COLLECTIONS_LIST_KEY, &value)?;
        Ok(())
    }

    fn get_list<T: Transaction>(
        tx: &DatabaseTransaction<T>,
    ) -> Result<Vec<String>, CollectionError> {
        match tx.get_metadata(COLLECTIONS_LIST_KEY)? {
            Some(bytes) => {
                let (list, _): (Vec<String>, _) =
                    bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                        .map_err(|e| CollectionError::Serialization(e.to_string()))?;
                Ok(list)
            }
            None => Ok(Vec::new()),
        }
    }
}

/// Errors that can occur during collection operations.
#[derive(Debug, thiserror::Error)]
pub enum CollectionError {
    /// Collection already exists.
    #[error("collection already exists: {0}")]
    AlreadyExists(String),

    /// Collection not found.
    #[error("collection not found: {0}")]
    NotFound(String),

    /// Named vector already exists in the collection.
    #[error("named vector already exists: {0}")]
    VectorExists(String),

    /// Named vector not found in the collection.
    #[error("named vector not found: {0}")]
    VectorNotFound(String),

    /// Invalid collection name.
    #[error("invalid collection name: {0}")]
    InvalidName(#[from] CollectionNameError),

    /// Transaction error.
    #[error("transaction error: {0}")]
    Transaction(#[from] TransactionError),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(String),
}

#[cfg(test)]
mod tests {
    // Tests require database infrastructure which we'll test via integration tests
}
