//! Database configuration and builder pattern.
//!
//! This module provides [`DatabaseBuilder`] for configuring and opening a database,
//! and [`Config`] which holds the final configuration values.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::DatabaseBuilder;
//!
//! let db = DatabaseBuilder::new()
//!     .path("mydb.manifold")
//!     .create_if_missing(true)
//!     .cache_size(64 * 1024 * 1024)  // 64MB cache
//!     .open()?;
//! ```

use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::transaction::{TransactionManagerConfig, VectorSyncStrategy};

/// Configuration for a database.
///
/// This struct holds the validated configuration for opening a database.
/// Use [`DatabaseBuilder`] to construct a `Config` with a fluent API.
#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the database file.
    pub path: PathBuf,

    /// Whether to create the database if it doesn't exist.
    pub create_if_missing: bool,

    /// Cache size in bytes. If `None`, uses the default.
    pub cache_size: Option<usize>,

    /// Maximum database size in bytes. If `None`, grows as needed.
    pub max_size: Option<u64>,

    /// Strategy for vector index synchronization.
    pub vector_sync_strategy: VectorSyncStrategy,

    /// Whether to use an in-memory database (for testing).
    pub in_memory: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            create_if_missing: true,
            cache_size: None,
            max_size: None,
            vector_sync_strategy: VectorSyncStrategy::Synchronous,
            in_memory: false,
        }
    }
}

impl Config {
    /// Create a new configuration with the given path.
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), ..Default::default() }
    }

    /// Create an in-memory configuration (for testing).
    #[must_use]
    pub fn in_memory() -> Self {
        Self { in_memory: true, ..Default::default() }
    }

    /// Set whether to create the database if it doesn't exist.
    #[must_use]
    pub const fn create_if_missing(mut self, create: bool) -> Self {
        self.create_if_missing = create;
        self
    }

    /// Get the transaction manager configuration.
    #[must_use]
    pub const fn transaction_config(&self) -> TransactionManagerConfig {
        TransactionManagerConfig { vector_sync_strategy: self.vector_sync_strategy }
    }
}

/// Builder for opening a database with custom configuration.
///
/// `DatabaseBuilder` provides a fluent API for configuring and opening a
/// `ManifoldDB` database.
///
/// # Examples
///
/// Open or create a database at a path:
///
/// ```ignore
/// use manifoldb::DatabaseBuilder;
///
/// let db = DatabaseBuilder::new()
///     .path("mydb.manifold")
///     .open()?;
/// ```
///
/// Configure cache size and vector sync strategy:
///
/// ```ignore
/// use manifoldb::{DatabaseBuilder, VectorSyncStrategy};
///
/// let db = DatabaseBuilder::new()
///     .path("mydb.manifold")
///     .cache_size(128 * 1024 * 1024)  // 128MB
///     .vector_sync_strategy(VectorSyncStrategy::Async)
///     .open()?;
/// ```
///
/// Create an in-memory database for testing:
///
/// ```ignore
/// use manifoldb::DatabaseBuilder;
///
/// let db = DatabaseBuilder::in_memory().open()?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct DatabaseBuilder {
    config: Config,
}

impl DatabaseBuilder {
    /// Create a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for an in-memory database.
    ///
    /// In-memory databases are useful for testing and temporary data.
    /// The data will be lost when the database is closed.
    #[must_use]
    pub fn in_memory() -> Self {
        Self { config: Config::in_memory() }
    }

    /// Set the path to the database file.
    ///
    /// The path should end with `.manifold` or `.redb` by convention.
    #[must_use]
    pub fn path(mut self, path: impl AsRef<Path>) -> Self {
        self.config.path = path.as_ref().to_path_buf();
        self.config.in_memory = false;
        self
    }

    /// Set whether to create the database if it doesn't exist.
    ///
    /// Defaults to `true`.
    #[must_use]
    pub const fn create_if_missing(mut self, create: bool) -> Self {
        self.config.create_if_missing = create;
        self
    }

    /// Set the cache size in bytes.
    ///
    /// A larger cache can improve read performance for frequently accessed data.
    /// If not set, the storage engine's default cache size is used.
    #[must_use]
    pub const fn cache_size(mut self, size: usize) -> Self {
        self.config.cache_size = Some(size);
        self
    }

    /// Set the maximum database size in bytes.
    ///
    /// If not set, the database will grow as needed.
    #[must_use]
    pub const fn max_size(mut self, size: u64) -> Self {
        self.config.max_size = Some(size);
        self
    }

    /// Set the vector synchronization strategy.
    ///
    /// This controls how vector index updates are synchronized with transactions:
    ///
    /// - [`VectorSyncStrategy::Synchronous`] - Strong consistency, slower writes
    /// - [`VectorSyncStrategy::Async`] - Eventual consistency, faster writes
    /// - [`VectorSyncStrategy::Hybrid`] - Adaptive based on batch size
    ///
    /// Defaults to [`VectorSyncStrategy::Synchronous`].
    #[must_use]
    pub const fn vector_sync_strategy(mut self, strategy: VectorSyncStrategy) -> Self {
        self.config.vector_sync_strategy = strategy;
        self
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Build and validate the configuration.
    ///
    /// This validates the configuration before opening the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (e.g., no path specified
    /// for a non-in-memory database).
    pub fn build(self) -> Result<Config, Error> {
        if !self.config.in_memory && self.config.path.as_os_str().is_empty() {
            return Err(Error::Config("database path is required".to_string()));
        }
        Ok(self.config)
    }

    /// Open the database with the configured options.
    ///
    /// This is a convenience method equivalent to calling `build()` followed
    /// by `Database::open_with_config()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or the database
    /// cannot be opened.
    pub fn open(self) -> Result<crate::database::Database, Error> {
        let config = self.build()?;
        crate::database::Database::open_with_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = Config::new("/tmp/test.manifold");
        assert_eq!(config.path, PathBuf::from("/tmp/test.manifold"));
        assert!(config.create_if_missing);
        assert!(!config.in_memory);
    }

    #[test]
    fn test_config_in_memory() {
        let config = Config::in_memory();
        assert!(config.in_memory);
        assert!(config.path.as_os_str().is_empty());
    }

    #[test]
    fn test_builder_path() {
        let builder = DatabaseBuilder::new().path("/tmp/test.manifold");
        assert_eq!(builder.config.path, PathBuf::from("/tmp/test.manifold"));
        assert!(!builder.config.in_memory);
    }

    #[test]
    fn test_builder_in_memory() {
        let builder = DatabaseBuilder::in_memory();
        assert!(builder.config.in_memory);
    }

    #[test]
    fn test_builder_cache_size() {
        let builder = DatabaseBuilder::new().cache_size(1024 * 1024);
        assert_eq!(builder.config.cache_size, Some(1024 * 1024));
    }

    #[test]
    fn test_builder_vector_sync_strategy() {
        let builder = DatabaseBuilder::new().vector_sync_strategy(VectorSyncStrategy::Async);
        assert_eq!(builder.config.vector_sync_strategy, VectorSyncStrategy::Async);
    }

    #[test]
    fn test_builder_build_requires_path() {
        let result = DatabaseBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_build_with_path() {
        let result = DatabaseBuilder::new().path("/tmp/test.manifold").build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_build_in_memory() {
        let result = DatabaseBuilder::in_memory().build();
        assert!(result.is_ok());
    }
}
