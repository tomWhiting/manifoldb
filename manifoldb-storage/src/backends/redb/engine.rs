//! Redb storage engine implementation.
//!
//! This module provides the `RedbEngine` type which implements the
//! `StorageEngine` trait using the Redb embedded database.

use std::path::Path;

use redb::Database;

use crate::engine::{StorageEngine, StorageError};

use super::transaction::RedbTransaction;

/// Configuration options for the Redb storage engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RedbConfig {
    /// Maximum size of the database file in bytes.
    /// If not set, the database will grow as needed.
    pub max_size: Option<u64>,

    /// Cache size in bytes.
    /// If not set, uses Redb's default.
    pub cache_size: Option<usize>,
}

impl RedbConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum database size.
    #[must_use]
    pub const fn max_size(mut self, size: u64) -> Self {
        self.max_size = Some(size);
        self
    }

    /// Set the cache size.
    #[must_use]
    pub const fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = Some(size);
        self
    }
}

/// A storage engine backed by Redb.
///
/// Redb is a pure-Rust embedded database that provides ACID transactions
/// and excellent cross-platform support.
///
/// # Example
///
/// ```ignore
/// use manifoldb_storage::backends::RedbEngine;
///
/// let engine = RedbEngine::open("my_database.redb")?;
///
/// // Write some data
/// let mut tx = engine.begin_write()?;
/// tx.put("users", b"user:1", b"Alice")?;
/// tx.commit()?;
/// ```
pub struct RedbEngine {
    /// The underlying Redb database.
    db: Database,
}

impl RedbEngine {
    /// Open or create a database at the given path with default configuration.
    ///
    /// If the database file exists, it will be opened. Otherwise, a new
    /// database will be created.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Open`] if the database cannot be opened or created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        Self::open_with_config(path, RedbConfig::default())
    }

    /// Open or create a database at the given path with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file
    /// * `config` - Configuration options for the database
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Open`] if the database cannot be opened or created.
    pub fn open_with_config(
        path: impl AsRef<Path>,
        config: RedbConfig,
    ) -> Result<Self, StorageError> {
        let mut builder = Database::builder();

        if let Some(cache_size) = config.cache_size {
            builder.set_cache_size(cache_size);
        }

        let db = builder.create(path.as_ref()).map_err(|e| StorageError::Open(e.to_string()))?;

        Ok(Self { db })
    }

    /// Create an in-memory database for testing.
    ///
    /// The database will be lost when the engine is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Open`] if the database cannot be created.
    pub fn in_memory() -> Result<Self, StorageError> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(|e| StorageError::Open(e.to_string()))?;

        Ok(Self { db })
    }

    /// Get the underlying Redb database.
    ///
    /// This is primarily for advanced use cases and testing.
    pub const fn inner(&self) -> &Database {
        &self.db
    }
}

impl StorageEngine for RedbEngine {
    type Transaction<'a> = RedbTransaction;

    fn begin_read(&self) -> Result<Self::Transaction<'_>, StorageError> {
        let tx = self.db.begin_read().map_err(|e| StorageError::Transaction(e.to_string()))?;
        Ok(RedbTransaction::new_read(tx))
    }

    fn begin_write(&self) -> Result<Self::Transaction<'_>, StorageError> {
        let tx = self.db.begin_write().map_err(|e| StorageError::Transaction(e.to_string()))?;
        Ok(RedbTransaction::new_write(tx))
    }

    fn flush(&self) -> Result<(), StorageError> {
        // Redb handles durability on commit, but we can compact if needed
        // For now, this is a no-op as commits are durable
        Ok(())
    }
}

// Note: RedbEngine is Send + Sync because redb::Database is Send + Sync.
// This is automatically derived since all fields implement Send + Sync.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Transaction;

    #[test]
    fn test_in_memory_creation() {
        let engine = RedbEngine::in_memory().expect("failed to create in-memory db");

        // Verify we can begin transactions
        let tx = engine.begin_read().expect("failed to begin read");
        assert!(tx.is_read_only());
    }

    #[test]
    fn test_config_builder() {
        let config = RedbConfig::new().max_size(1024 * 1024 * 100).cache_size(1024 * 1024 * 10);

        assert_eq!(config.max_size, Some(100 * 1024 * 1024));
        assert_eq!(config.cache_size, Some(10 * 1024 * 1024));
    }

    #[test]
    fn test_write_and_read() {
        let engine = RedbEngine::in_memory().expect("failed to create in-memory db");

        // Write
        {
            let mut tx = engine.begin_write().expect("failed to begin write");
            tx.put("test", b"key", b"value").expect("failed to put");
            tx.commit().expect("failed to commit");
        }

        // Read
        {
            let tx = engine.begin_read().expect("failed to begin read");
            let value = tx.get("test", b"key").expect("failed to get");
            assert_eq!(value, Some(b"value".to_vec()));
        }
    }
}
