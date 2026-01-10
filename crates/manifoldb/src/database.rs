//! Main database interface.
//!
//! This module provides the [`Database`] struct, which is the primary entry point
//! for interacting with a `ManifoldDB` database.
//!
//! # Examples
//!
//! Open a database and perform basic operations:
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! // Open or create a database
//! let db = Database::open("mydb.manifold")?;
//!
//! // Execute a statement
//! db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")?;
//!
//! // Query data
//! let results = db.query("SELECT * FROM users WHERE age > 25")?;
//! for row in results {
//!     println!("{:?}", row);
//! }
//! ```
//!
//! Use transactions for atomic operations:
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::open("mydb.manifold")?;
//!
//! // Start a transaction
//! let mut txn = db.begin()?;
//!
//! // Perform operations
//! let entity = txn.create_entity()?.with_label("Person");
//! txn.put_entity(&entity)?;
//!
//! // Commit changes
//! txn.commit()?;
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use manifoldb_core::encoding::keys::encode_edge_key;
use manifoldb_core::encoding::Encoder;
use manifoldb_core::{Edge, EdgeId, Entity, EntityId};
use manifoldb_storage::backends::redb::{RedbConfig, RedbEngine};
use manifoldb_storage::Transaction;

use crate::cache::{extract_cache_hint, CacheHint, CacheMetrics, QueryCache, QueryCacheKey};
use crate::config::{Config, DatabaseBuilder};
use crate::error::{Error, Result};
use crate::execution::{execute_statement, extract_tables_from_sql};
use crate::index::{IndexInfo, IndexManager, IndexMetadata, IndexStats, IndexType};
use crate::metrics::{CacheMetricsSnapshot, DatabaseMetrics, MetricsSnapshot};
use crate::prepared::{PreparedStatement, PreparedStatementCache};
use crate::schema::SchemaManager;
use crate::transaction::{DatabaseTransaction, TransactionManager};

/// The main `ManifoldDB` database handle.
///
/// `Database` is the primary entry point for interacting with a `ManifoldDB` database.
/// It provides methods for:
///
/// - Opening and configuring databases
/// - Executing SQL statements
/// - Querying data with SQL and graph patterns
/// - Managing transactions
///
/// # Thread Safety
///
/// `Database` is `Send + Sync` and can be safely shared across threads.
/// Multiple concurrent read transactions are supported.
///
/// # Examples
///
/// ## Opening a Database
///
/// ```ignore
/// use manifoldb::Database;
///
/// // Simple open with default options
/// let db = Database::open("mydb.manifold")?;
///
/// // Or use the builder for more options
/// use manifoldb::DatabaseBuilder;
///
/// let db = DatabaseBuilder::new()
///     .path("mydb.manifold")
///     .cache_size(64 * 1024 * 1024)
///     .open()?;
/// ```
///
/// ## Executing Queries
///
/// ```ignore
/// // Execute a statement (INSERT, UPDATE, DELETE)
/// let affected = db.execute("INSERT INTO users (name) VALUES ('Alice')")?;
///
/// // Execute a query (SELECT)
/// let results = db.query("SELECT * FROM users WHERE name = 'Alice'")?;
/// ```
///
/// ## Using Transactions
///
/// ```ignore
/// let mut txn = db.begin()?;
///
/// // Create and store an entity
/// let entity = txn.create_entity()?.with_label("Person");
/// txn.put_entity(&entity)?;
///
/// // Create an edge
/// let edge = txn.create_edge(entity1.id, entity2.id, "FOLLOWS")?;
/// txn.put_edge(&edge)?;
///
/// txn.commit()?;
/// ```
///
/// # Cloning
///
/// `Database` is cheap to clone - it uses `Arc` internally, so cloning only
/// increments a reference count. This makes it easy to share a database handle
/// across threads or async tasks.
///
/// ```ignore
/// let db = Database::open("mydb.manifold")?;
/// let db2 = db.clone(); // Cheap clone, shares underlying data
///
/// // Use in multiple threads
/// std::thread::spawn(move || {
///     db2.query("SELECT * FROM users")?;
/// });
/// ```
#[derive(Clone)]
pub struct Database {
    /// The inner database state, shared via Arc for cheap cloning.
    inner: Arc<DatabaseInner>,
}

/// The internal state of a Database.
///
/// This is wrapped in an `Arc` by `Database` to enable cheap cloning.
struct DatabaseInner {
    /// The transaction manager coordinating storage and indexes.
    manager: TransactionManager<RedbEngine>,
    /// The configuration used to open this database.
    config: Config,
    /// Query result cache.
    query_cache: QueryCache,
    /// Prepared statement cache.
    prepared_cache: PreparedStatementCache,
    /// Database metrics.
    db_metrics: Arc<DatabaseMetrics>,
    /// Payload index manager.
    index_manager: IndexManager,
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// This is a convenience method that uses default configuration options.
    /// For more control, use [`DatabaseBuilder`].
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the database file
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or created.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::open("mydb.manifold")?;
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        DatabaseBuilder::new().path(path).open()
    }

    /// Open or create an in-memory database.
    ///
    /// In-memory databases are useful for testing and temporary data.
    /// All data is lost when the database is closed.
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory database cannot be created.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    /// ```
    pub fn in_memory() -> Result<Self> {
        DatabaseBuilder::in_memory().open()
    }

    /// Open a database with the given configuration.
    ///
    /// This is typically called through [`DatabaseBuilder::open()`].
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub fn open_with_config(config: Config) -> Result<Self> {
        let engine = if config.in_memory {
            RedbEngine::in_memory().map_err(|e| Error::Open(e.to_string()))?
        } else {
            let mut redb_config = RedbConfig::new();
            if let Some(cache_size) = config.cache_size {
                redb_config = redb_config.cache_size(cache_size);
            }
            if let Some(max_size) = config.max_size {
                redb_config = redb_config.max_size(max_size);
            }
            RedbEngine::open_with_config(&config.path, redb_config)
                .map_err(|e| Error::Open(e.to_string()))?
        };

        let manager = TransactionManager::with_config(engine, config.transaction_config());
        let query_cache = QueryCache::new(config.query_cache_config.clone());
        let prepared_cache = PreparedStatementCache::default();
        let db_metrics = Arc::new(DatabaseMetrics::new());
        let index_manager = IndexManager::new(manager.engine_arc());

        // Initialize prepared cache with current schema version
        let inner = DatabaseInner {
            manager,
            config,
            query_cache,
            prepared_cache,
            db_metrics,
            index_manager,
        };
        let db = Self { inner: Arc::new(inner) };

        // Load initial schema version
        if let Ok(tx) = db.begin_read() {
            if let Ok(version) = SchemaManager::get_version(&tx) {
                db.inner.prepared_cache.set_schema_version(version);
            }
        }

        Ok(db)
    }

    /// Returns a builder for creating a database with custom configuration.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::builder()
    ///     .path("mydb.manifold")
    ///     .cache_size(128 * 1024 * 1024)
    ///     .open()?;
    /// ```
    #[must_use]
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder::new()
    }

    /// Get the configuration used to open this database.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    /// Begin a new read-write transaction.
    ///
    /// Write transactions allow both reading and writing data. Only one
    /// write transaction can be active at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut txn = db.begin()?;
    /// let entity = txn.create_entity()?;
    /// txn.put_entity(&entity)?;
    /// txn.commit()?;
    /// ```
    pub fn begin(
        &self,
    ) -> Result<
        DatabaseTransaction<<RedbEngine as manifoldb_storage::StorageEngine>::Transaction<'_>>,
    > {
        self.inner.manager.begin_write().map_err(Error::Transaction)
    }

    /// Begin a new read-only transaction.
    ///
    /// Read transactions provide a consistent snapshot of the database.
    /// Multiple read transactions can run concurrently.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let txn = db.begin_read()?;
    /// let entity = txn.get_entity(entity_id)?;
    /// // Read transaction doesn't need commit
    /// ```
    pub fn begin_read(
        &self,
    ) -> Result<
        DatabaseTransaction<<RedbEngine as manifoldb_storage::StorageEngine>::Transaction<'_>>,
    > {
        self.inner.manager.begin_read().map_err(Error::Transaction)
    }

    /// Execute a SQL statement that doesn't return results.
    ///
    /// Use this for INSERT, UPDATE, DELETE, and DDL statements.
    /// For SELECT queries, use [`query()`](Self::query) instead.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement to execute
    ///
    /// # Returns
    ///
    /// The number of rows affected by the statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement cannot be parsed or executed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let affected = db.execute("INSERT INTO users (name) VALUES ('Alice')")?;
    /// println!("Inserted {} rows", affected);
    /// ```
    pub fn execute(&self, sql: &str) -> Result<u64> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL statement with bound parameters.
    ///
    /// Parameters are specified as `$1`, `$2`, etc. in the SQL string.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement with parameter placeholders
    /// * `params` - The parameter values to bind
    ///
    /// # Returns
    ///
    /// The number of rows affected by the statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement cannot be parsed, parameters are
    /// invalid, or execution fails.
    ///
    /// # Cache Invalidation
    ///
    /// This method automatically invalidates any cached query results that
    /// accessed the tables modified by this statement.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let affected = db.execute_with_params(
    ///     "INSERT INTO users (name, age) VALUES ($1, $2)",
    ///     &["Alice".into(), 30.into()],
    /// )?;
    /// ```
    pub fn execute_with_params(&self, sql: &str, params: &[manifoldb_core::Value]) -> Result<u64> {
        let start = Instant::now();

        // Extract tables that will be modified for cache invalidation
        let affected_tables = extract_tables_from_sql(sql);

        // Check if this is a DDL statement
        let is_ddl = Self::is_ddl_statement(sql);

        // Start a write transaction
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Execute the statement
        let result = execute_statement(&mut tx, sql, params);

        match result {
            Ok(count) => {
                // Get schema version before commit if DDL
                let new_schema_version =
                    if is_ddl { SchemaManager::get_version(&tx).ok() } else { None };

                // Commit the transaction
                let commit_start = Instant::now();
                tx.commit().map_err(Error::Transaction)?;
                self.inner.db_metrics.record_commit(commit_start.elapsed());

                // Record successful query
                self.inner.db_metrics.record_query(start.elapsed(), true);

                // Invalidate cache entries for affected tables
                self.inner.query_cache.invalidate_tables(&affected_tables);
                self.inner.prepared_cache.invalidate_tables(&affected_tables)?;

                // Update prepared statement cache schema version if DDL
                if let Some(version) = new_schema_version {
                    self.inner.prepared_cache.set_schema_version(version);
                }

                Ok(count)
            }
            Err(e) => {
                // Record failed query and rollback
                self.inner.db_metrics.record_query(start.elapsed(), false);
                self.inner.db_metrics.record_rollback();
                Err(e)
            }
        }
    }

    /// Check if a SQL statement is a DDL statement.
    fn is_ddl_statement(sql: &str) -> bool {
        let sql_upper = sql.trim().to_uppercase();
        sql_upper.starts_with("CREATE TABLE")
            || sql_upper.starts_with("DROP TABLE")
            || sql_upper.starts_with("CREATE INDEX")
            || sql_upper.starts_with("DROP INDEX")
            || sql_upper.starts_with("ALTER TABLE")
    }

    /// Execute a SQL query and return results.
    ///
    /// Use this for SELECT statements. For INSERT, UPDATE, and DELETE,
    /// use [`execute()`](Self::execute) instead.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL query to execute
    ///
    /// # Returns
    ///
    /// A [`QueryResult`] containing the query results.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be parsed or executed.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let results = db.query("SELECT * FROM users WHERE age > 25")?;
    /// for row in results {
    ///     println!("{:?}", row);
    /// }
    /// ```
    pub fn query(&self, sql: &str) -> Result<QueryResult> {
        self.query_with_params(sql, &[])
    }

    /// Execute a SQL query with bound parameters.
    ///
    /// Parameters are specified as `$1`, `$2`, etc. in the SQL string.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL query with parameter placeholders
    /// * `params` - The parameter values to bind
    ///
    /// # Returns
    ///
    /// A [`QueryResult`] containing the query results.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be parsed, parameters are
    /// invalid, or execution fails.
    ///
    /// # Cache Hints
    ///
    /// You can control caching behavior with hints:
    /// - `/*+ CACHE */` - Force caching of this query result
    /// - `/*+ NO_CACHE */` - Skip caching for this query
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let results = db.query_with_params(
    ///     "SELECT * FROM users WHERE name = $1",
    ///     &["Alice".into()],
    /// )?;
    ///
    /// // Force caching
    /// let results = db.query_with_params(
    ///     "/*+ CACHE */ SELECT * FROM users WHERE name = $1",
    ///     &["Alice".into()],
    /// )?;
    ///
    /// // Skip caching
    /// let results = db.query_with_params(
    ///     "/*+ NO_CACHE */ SELECT * FROM users WHERE name = $1",
    ///     &["Alice".into()],
    /// )?;
    /// ```
    pub fn query_with_params(
        &self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<QueryResult> {
        // Extract cache hint and clean SQL
        let (hint, clean_sql) = extract_cache_hint(sql);

        // Check if this is a Cypher DML statement (CREATE, MERGE, etc.)
        // These require a write transaction
        if crate::execution::is_cypher_dml(&clean_sql) {
            return self.execute_cypher_dml(&clean_sql, params);
        }

        // Determine if we should use caching
        let use_cache = match hint {
            CacheHint::Cache => true,
            CacheHint::NoCache => false,
            CacheHint::Default => self.inner.query_cache.is_enabled(),
        };

        // Try to get from cache if caching is enabled
        if use_cache {
            let cache_key = QueryCacheKey::new(&clean_sql, params);
            if let Some(cached_result) = self.inner.query_cache.get(&cache_key) {
                // Update LRU order
                self.inner.query_cache.touch(&cache_key);
                return Ok(cached_result);
            }
        }

        let start = Instant::now();

        // Start a read transaction
        let tx = self.begin_read()?;

        // Execute the query with the configured row limit
        let result = crate::execution::execute_query_with_limit(
            &tx,
            &clean_sql,
            params,
            self.inner.config.max_rows_in_memory,
        );

        match result {
            Ok(result_set) => {
                // Record successful query
                self.inner.db_metrics.record_query(start.elapsed(), true);

                // Convert the ResultSet to our QueryResult
                let result = QueryResult::from_result_set(result_set);

                // Cache the result if caching is enabled
                if use_cache {
                    let cache_key = QueryCacheKey::new(&clean_sql, params);
                    let accessed_tables = extract_tables_from_sql(&clean_sql);
                    self.inner.query_cache.insert(cache_key, result.clone(), accessed_tables);
                }

                Ok(result)
            }
            Err(e) => {
                // Record failed query
                self.inner.db_metrics.record_query(start.elapsed(), false);
                Err(e)
            }
        }
    }

    /// Execute a Cypher DML statement (CREATE, MERGE, etc.) that requires a write transaction.
    ///
    /// This is called internally when query() or query_with_params() detects a Cypher DML statement.
    fn execute_cypher_dml(
        &self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<QueryResult> {
        let start = Instant::now();

        // Start a write transaction
        let tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Execute the Cypher DML
        let result = crate::execution::execute_graph_dml(
            tx,
            sql,
            params,
            self.inner.config.max_rows_in_memory,
        );

        match result {
            Ok((result_set, tx)) => {
                // Commit the transaction
                let commit_start = Instant::now();
                tx.commit().map_err(Error::Transaction)?;
                self.inner.db_metrics.record_commit(commit_start.elapsed());

                // Record successful query
                self.inner.db_metrics.record_query(start.elapsed(), true);

                // Convert the ResultSet to our QueryResult
                Ok(QueryResult::from_result_set(result_set))
            }
            Err(e) => {
                // Record failed query and rollback
                self.inner.db_metrics.record_query(start.elapsed(), false);
                self.inner.db_metrics.record_rollback();
                Err(e)
            }
        }
    }

    /// Flush any buffered data to durable storage.
    ///
    /// This ensures all committed transactions are persisted to disk.
    /// It's typically called automatically on commit, but can be called
    /// explicitly for additional durability guarantees.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> Result<()> {
        self.inner.manager.flush().map_err(Error::Transaction)
    }

    /// Get the underlying transaction manager.
    ///
    /// This is useful for advanced use cases that require direct
    /// transaction management.
    #[must_use]
    pub fn transaction_manager(&self) -> &TransactionManager<RedbEngine> {
        &self.inner.manager
    }

    /// Get the query cache.
    ///
    /// Use this to access cache operations like clearing or checking metrics.
    #[must_use]
    pub fn query_cache(&self) -> &QueryCache {
        &self.inner.query_cache
    }

    /// Get the cache metrics.
    ///
    /// Returns metrics about cache hits, misses, and evictions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let metrics = db.cache_metrics();
    /// println!("Hit rate: {:?}", metrics.hit_rate());
    /// println!("Total lookups: {}", metrics.total_lookups());
    /// ```
    #[must_use]
    pub fn cache_metrics(&self) -> Arc<CacheMetrics> {
        self.inner.query_cache.metrics()
    }

    /// Clear the query cache.
    ///
    /// This removes all cached query results. Useful after bulk data
    /// modifications or when you want to ensure fresh data.
    pub fn clear_cache(&self) {
        self.inner.query_cache.clear();
    }

    /// Invalidate cache entries for specific tables.
    ///
    /// This is automatically called during write operations, but can
    /// be called manually if you modify data outside of the normal
    /// execute methods.
    pub fn invalidate_cache_for_tables(&self, tables: &[String]) {
        self.inner.query_cache.invalidate_tables(tables);
    }

    /// Get a snapshot of all database metrics.
    ///
    /// Returns a point-in-time snapshot of query, transaction, vector search,
    /// storage, and cache metrics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Perform some operations
    /// db.execute("INSERT INTO users (name) VALUES ('Alice')")?;
    /// db.query("SELECT * FROM users")?;
    ///
    /// // Get metrics snapshot
    /// let snapshot = db.metrics();
    /// println!("{}", snapshot);  // Pretty-printed summary
    ///
    /// // Access specific metrics
    /// println!("Queries executed: {}", snapshot.queries.total_queries);
    /// println!("Cache hit rate: {:?}", snapshot.cache.as_ref().and_then(|c| c.hit_rate()));
    /// println!("Transactions committed: {}", snapshot.transactions.commits);
    /// ```
    #[must_use]
    pub fn metrics(&self) -> MetricsSnapshot {
        let mut snapshot = self.inner.db_metrics.snapshot();

        // Include cache metrics from the query cache
        let cache_snapshot = self.inner.query_cache.metrics().snapshot();
        snapshot.cache = Some(CacheMetricsSnapshot::from_cache_snapshot(cache_snapshot));

        snapshot
    }

    /// Get access to the raw metrics instance.
    ///
    /// This is useful for custom metric collection or integration with
    /// external monitoring systems.
    #[must_use]
    pub fn raw_metrics(&self) -> Arc<DatabaseMetrics> {
        Arc::clone(&self.inner.db_metrics)
    }

    /// Reset all collected metrics.
    ///
    /// This is useful for benchmarking or when you want to collect
    /// metrics for a specific time window.
    ///
    /// Note: Storage size metrics are not reset as they represent
    /// current state rather than accumulated values.
    pub fn reset_metrics(&self) {
        self.inner.db_metrics.reset();
        self.inner.query_cache.metrics().reset();
    }

    // =========================================================================
    // Payload Indexing
    // =========================================================================

    /// Create an index on a property for entities with the given label.
    ///
    /// Indexes speed up filtered vector searches by allowing the query planner
    /// to narrow down candidates using B-tree lookups instead of scanning all entities.
    ///
    /// # Arguments
    ///
    /// * `label` - The entity label to index (e.g., "Symbol", "Document")
    /// * `property` - The property to index (e.g., "language", "category")
    ///
    /// # Errors
    ///
    /// Returns an error if the index already exists or creation fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create an index on the "language" property for "Symbol" entities
    /// db.create_index("Symbol", "language")?;
    ///
    /// // Now searches filtering by language will use the index
    /// ```
    pub fn create_index(&self, label: &str, property: &str) -> Result<()> {
        self.create_index_with_type(label, property, IndexType::Equality)
    }

    /// Create an index with a specific type.
    ///
    /// # Arguments
    ///
    /// * `label` - The entity label to index
    /// * `property` - The property to index
    /// * `index_type` - The type of index (Equality, Range, or Prefix)
    ///
    /// # Index Types
    ///
    /// - `Equality`: Best for enum-like fields. Supports `eq`, `ne`, `in` operators.
    /// - `Range`: Best for numeric fields. Supports `gt`, `gte`, `lt`, `lte`, `range`.
    /// - `Prefix`: Best for paths/names. Supports `starts_with`.
    pub fn create_index_with_type(
        &self,
        label: &str,
        property: &str,
        index_type: IndexType,
    ) -> Result<()> {
        self.inner.index_manager.create_index(label, property, index_type)
    }

    /// Drop an index.
    ///
    /// # Arguments
    ///
    /// * `label` - The entity label
    /// * `property` - The indexed property
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn drop_index(&self, label: &str, property: &str) -> Result<()> {
        self.inner.index_manager.drop_index(label, property)
    }

    /// List all indexes in the database.
    ///
    /// # Returns
    ///
    /// A vector of index information including label, property, type, and entry count.
    pub fn list_indexes(&self) -> Result<Vec<IndexInfo>> {
        self.inner.index_manager.list_indexes()
    }

    /// Get statistics for a specific index.
    ///
    /// # Arguments
    ///
    /// * `label` - The entity label
    /// * `property` - The indexed property
    ///
    /// # Returns
    ///
    /// Index statistics including entry count, distinct values, and selectivity.
    ///
    /// # Errors
    ///
    /// Returns an error if the index doesn't exist.
    pub fn index_stats(&self, label: &str, property: &str) -> Result<IndexStats> {
        self.inner.index_manager.index_stats(label, property)
    }

    /// Get metadata for an index, or None if it doesn't exist.
    ///
    /// This is useful for checking if an index exists without triggering an error.
    pub fn get_index_metadata(&self, label: &str, property: &str) -> Result<Option<IndexMetadata>> {
        self.inner.index_manager.get_index_metadata(label, property)
    }

    /// Look up entity IDs matching a filter value using an index.
    ///
    /// This is a low-level API primarily used by the query planner.
    /// Returns None if no index exists for the label/property combination.
    ///
    /// # Arguments
    ///
    /// * `label` - The entity label
    /// * `property` - The indexed property
    /// * `value` - The value to match
    pub fn index_lookup(
        &self,
        label: &str,
        property: &str,
        value: &manifoldb_core::Value,
    ) -> Result<Option<Vec<EntityId>>> {
        self.inner.index_manager.lookup_eq(label, property, value)
    }

    /// Build a planner catalog from the current state of the database.
    ///
    /// This creates a snapshot of index metadata that the query planner can use
    /// for index selection and cost estimation.
    ///
    /// # Returns
    ///
    /// A `PlannerCatalog` populated with:
    /// - Payload indexes (as B-tree indexes on label.property)
    /// - Index statistics for selectivity estimation
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    /// use manifoldb_query::{PhysicalPlanner, PlanBuilder};
    ///
    /// let db = Database::in_memory()?;
    /// db.create_index("Symbol", "language")?;
    ///
    /// // Get catalog with current index info
    /// let catalog = db.build_planner_catalog()?;
    ///
    /// // Use it for query planning
    /// let planner = PhysicalPlanner::new().with_catalog(catalog);
    /// ```
    pub fn build_planner_catalog(&self) -> Result<manifoldb_query::PlannerCatalog> {
        use manifoldb_query::{PlannerCatalog, PlannerIndexInfo, TableStats};

        let mut catalog = PlannerCatalog::new();

        // Add payload indexes
        for idx in self.list_indexes()? {
            // Create index name as "label_property_idx"
            let index_name = format!("{}_{}_idx", idx.label, idx.property);

            // All payload indexes are B-tree style
            let planner_idx = PlannerIndexInfo::btree(
                index_name,
                &idx.label, // Use label as "table" name
                vec![idx.property.clone()],
            );

            catalog = catalog.with_index(planner_idx);

            // Also add table stats based on index entry count
            // This gives the planner row count estimates
            catalog = catalog.with_table(TableStats::new(&idx.label, idx.entry_count as usize));
        }

        Ok(catalog)
    }

    /// Prepare a SQL statement for repeated execution.
    ///
    /// Prepared statements cache the parsed AST and query plan, amortizing
    /// parsing and planning costs over multiple executions.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement to prepare. Use `$1`, `$2`, etc. for parameters.
    ///
    /// # Returns
    ///
    /// A prepared statement that can be executed multiple times with different parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL cannot be parsed or planned.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, Value};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Prepare once
    /// let stmt = db.prepare("SELECT * FROM users WHERE age > $1")?;
    ///
    /// // Execute multiple times with different parameters
    /// let young = stmt.query(&db, &[Value::Int(18)])?;
    /// let old = stmt.query(&db, &[Value::Int(65)])?;
    /// ```
    pub fn prepare(&self, sql: &str) -> Result<Arc<PreparedStatement>> {
        self.inner.prepared_cache.prepare(sql)
    }

    /// Get or prepare a SQL statement (uses cache).
    ///
    /// This is like `prepare`, but uses the prepared statement cache.
    /// If the same SQL was previously prepared and the schema hasn't changed,
    /// the cached statement is returned.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement to prepare. Use `$1`, `$2`, etc. for parameters.
    ///
    /// # Returns
    ///
    /// A prepared statement that can be executed multiple times with different parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL cannot be parsed or planned.
    pub fn prepare_cached(&self, sql: &str) -> Result<Arc<PreparedStatement>> {
        self.inner.prepared_cache.get_or_prepare(sql)
    }

    /// Execute a prepared statement that returns results (SELECT).
    ///
    /// # Arguments
    ///
    /// * `stmt` - The prepared statement to execute
    /// * `params` - The parameter values to bind
    ///
    /// # Returns
    ///
    /// A [`QueryResult`] containing the query results.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement has been invalidated by schema changes,
    /// or if execution fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stmt = db.prepare("SELECT * FROM users WHERE age > $1")?;
    /// let results = db.query_prepared(&stmt, &[Value::Int(21)])?;
    /// ```
    pub fn query_prepared(
        &self,
        stmt: &PreparedStatement,
        params: &[manifoldb_core::Value],
    ) -> Result<QueryResult> {
        // Check if statement is still valid
        let current_version = self.inner.prepared_cache.schema_version();
        if !stmt.is_valid(current_version) {
            return Err(Error::Execution(
                "Prepared statement is invalid due to schema changes. Please re-prepare."
                    .to_string(),
            ));
        }

        let start = Instant::now();

        // Start a read transaction
        let tx = self.begin_read()?;

        // Execute using the cached plans
        let result = crate::execution::execute_prepared_query(&tx, stmt, params);

        match result {
            Ok(result_set) => {
                // Record successful query
                self.inner.db_metrics.record_query(start.elapsed(), true);

                // Convert the ResultSet to our QueryResult
                Ok(QueryResult::from_result_set(result_set))
            }
            Err(e) => {
                // Record failed query
                self.inner.db_metrics.record_query(start.elapsed(), false);
                Err(e)
            }
        }
    }

    /// Execute a prepared DML statement (INSERT, UPDATE, DELETE).
    ///
    /// # Arguments
    ///
    /// * `stmt` - The prepared statement to execute
    /// * `params` - The parameter values to bind
    ///
    /// # Returns
    ///
    /// The number of rows affected by the statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement has been invalidated by schema changes,
    /// or if execution fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stmt = db.prepare("INSERT INTO users (name, age) VALUES ($1, $2)")?;
    /// let count = db.execute_prepared(&stmt, &[Value::from("Alice"), Value::Int(30)])?;
    /// ```
    pub fn execute_prepared(
        &self,
        stmt: &PreparedStatement,
        params: &[manifoldb_core::Value],
    ) -> Result<u64> {
        // Check if statement is still valid
        let current_version = self.inner.prepared_cache.schema_version();
        if !stmt.is_valid(current_version) {
            return Err(Error::Execution(
                "Prepared statement is invalid due to schema changes. Please re-prepare."
                    .to_string(),
            ));
        }

        let start = Instant::now();

        // Start a write transaction
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Execute using the cached plans
        let result = crate::execution::execute_prepared_statement(&mut tx, stmt, params);

        match result {
            Ok(count) => {
                // Get schema version before commit if DDL
                let new_schema_version =
                    if stmt.is_ddl() { SchemaManager::get_version(&tx).ok() } else { None };

                // Commit the transaction
                let commit_start = Instant::now();
                tx.commit().map_err(Error::Transaction)?;
                self.inner.db_metrics.record_commit(commit_start.elapsed());

                // Record successful query
                self.inner.db_metrics.record_query(start.elapsed(), true);

                // Invalidate cache entries for affected tables
                let affected_tables: Vec<String> = stmt.accessed_tables().iter().cloned().collect();
                self.inner.query_cache.invalidate_tables(&affected_tables);
                self.inner.prepared_cache.invalidate_tables(&affected_tables)?;

                // Update prepared statement cache schema version if DDL
                if let Some(version) = new_schema_version {
                    self.inner.prepared_cache.set_schema_version(version);
                }

                Ok(count)
            }
            Err(e) => {
                // Record failed query and rollback
                self.inner.db_metrics.record_query(start.elapsed(), false);
                self.inner.db_metrics.record_rollback();
                Err(e)
            }
        }
    }

    /// Get the prepared statement cache.
    ///
    /// Use this to access cache operations like clearing or checking metrics.
    #[must_use]
    pub fn prepared_cache(&self) -> &PreparedStatementCache {
        &self.inner.prepared_cache
    }

    /// Clear the prepared statement cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal cache lock is poisoned.
    pub fn clear_prepared_cache(&self) -> Result<()> {
        self.inner.prepared_cache.clear()
    }

    /// Bulk insert entities with maximum throughput.
    ///
    /// All entities are inserted in a single transaction, with serialization
    /// parallelized across CPU cores using rayon. This is the most efficient
    /// way to insert many entities.
    ///
    /// # Performance
    ///
    /// This method achieves high throughput by:
    /// 1. **Parallel serialization**: Entities are serialized to binary format in parallel
    /// 2. **Single transaction**: All writes occur within one transaction (one fsync)
    /// 3. **Bulk ID allocation**: Entity IDs are allocated in a single atomic batch
    /// 4. **Index maintenance**: All indexes are updated within the same transaction
    ///
    /// # Arguments
    ///
    /// * `entities` - The entities to insert. All entities should have their labels
    ///   and properties set. The `id` field will be overwritten with auto-generated IDs.
    ///
    /// # Returns
    ///
    /// A vector of [`EntityId`] for the inserted entities, in the same order as the input.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any entity fails validation (the entire batch is aborted, no entities are inserted)
    /// - Serialization fails for any entity
    /// - The transaction cannot be committed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, Entity, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create many entities (note: IDs will be assigned by bulk_insert_entities)
    /// let entities: Vec<Entity> = (0..10_000)
    ///     .map(|i| {
    ///         Entity::new(EntityId::new(0)) // ID is placeholder, will be overwritten
    ///             .with_label("Document")
    ///             .with_property("index", i as i64)
    ///             .with_property("content", format!("Document {}", i))
    ///     })
    ///     .collect();
    ///
    /// // Insert all at once - much faster than individual inserts
    /// let ids = db.bulk_insert_entities(&entities)?;
    /// assert_eq!(ids.len(), 10_000);
    /// ```
    pub fn bulk_insert_entities(&self, entities: &[Entity]) -> Result<Vec<EntityId>> {
        use crate::execution::EntityIndexMaintenance;
        use rayon::prelude::*;

        if entities.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        // Phase 1: Parallel serialization
        // Serialize all entities in parallel, storing (original_index, serialized_bytes)
        // We serialize first (before IDs are assigned) to catch serialization errors early
        // The ID field will be updated in phase 3
        let serialized: std::result::Result<Vec<(usize, Vec<u8>)>, Error> = entities
            .par_iter()
            .enumerate()
            .map(|(idx, entity)| {
                // Create a temporary entity with the correct structure for serialization validation
                // Actual ID will be assigned in the write phase
                bincode::serde::encode_to_vec(entity, bincode::config::standard())
                    .map(|bytes| (idx, bytes))
                    .map_err(|e| {
                        Error::Execution(format!(
                            "Failed to serialize entity at index {}: {}",
                            idx, e
                        ))
                    })
            })
            .collect();

        let serialized = serialized?;

        // Phase 2: Begin transaction and allocate IDs
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Get the starting entity ID and reserve a range
        let entity_count = entities.len() as u64;
        let start_id = {
            // Read current counter
            let current = match tx.get_metadata(b"next_entity_id")? {
                Some(bytes) if bytes.len() == 8 => {
                    let arr: [u8; 8] = bytes
                        .try_into()
                        .map_err(|_| Error::Execution("invalid entity counter".to_string()))?;
                    u64::from_be_bytes(arr)
                }
                _ => 1, // Start from 1 if not set
            };

            // Update counter to reserve the range
            let next = current + entity_count;
            tx.put_metadata(b"next_entity_id", &next.to_be_bytes())?;

            current
        };

        // Phase 3: Re-serialize with correct IDs (in parallel) and write
        // Now we have the IDs, we need to serialize again with the correct IDs
        let entities_with_ids: std::result::Result<Vec<(EntityId, Entity, Vec<u8>)>, Error> =
            entities
                .par_iter()
                .enumerate()
                .map(|(idx, entity)| {
                    let id = EntityId::new(start_id + idx as u64);
                    let mut entity_with_id = entity.clone();
                    entity_with_id.id = id;

                    bincode::serde::encode_to_vec(&entity_with_id, bincode::config::standard())
                        .map(|bytes| (id, entity_with_id, bytes))
                        .map_err(|e| {
                            Error::Execution(format!(
                                "Failed to serialize entity at index {}: {}",
                                idx, e
                            ))
                        })
                })
                .collect();

        let entities_with_ids = entities_with_ids?;

        // Drop the initial serialized results - we only needed them for validation
        drop(serialized);

        // Phase 4: Sequential writes (fast - just memcpy to transaction buffer)
        let ids: Vec<EntityId> = entities_with_ids.iter().map(|(id, _, _)| *id).collect();

        for (id, entity, bytes) in &entities_with_ids {
            let key = id.as_u64().to_be_bytes();
            let storage = tx.storage_mut_ref().map_err(Error::Transaction)?;

            storage
                .put("entities", &key, bytes)
                .map_err(|e| Error::Execution(format!("Failed to write entity: {}", e)))?;

            // Phase 5a: Label index maintenance
            // Key format: <length:2 bytes><label:N bytes><entity_id:8 bytes>
            for label in &entity.labels {
                let label_bytes = label.as_str().as_bytes();
                let len = label_bytes.len() as u16;
                let mut label_key = Vec::with_capacity(2 + label_bytes.len() + 8);
                label_key.extend_from_slice(&len.to_be_bytes());
                label_key.extend_from_slice(label_bytes);
                label_key.extend_from_slice(&id.as_u64().to_be_bytes());
                storage
                    .put("label_index", &label_key, &[])
                    .map_err(|e| Error::Execution(format!("Failed to write label index: {}", e)))?;
            }

            // Phase 5b: Index maintenance (schema-based indexes)
            EntityIndexMaintenance::on_insert(&mut tx, entity)
                .map_err(|e| Error::Execution(format!("Index maintenance failed: {}", e)))?;

            // Phase 5c: Payload index maintenance
            self.inner.index_manager.on_entity_upsert_tx(
                tx.storage_mut_ref().map_err(Error::Transaction)?,
                entity,
                None, // New entity, no old version
            )?;
        }

        // Phase 6: Commit
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful bulk insert
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(ids)
    }

    /// Bulk insert edges with maximum throughput.
    ///
    /// All edges are inserted in a single transaction, with validation and
    /// serialization parallelized across CPU cores using rayon. This is the
    /// most efficient way to insert many edges.
    ///
    /// # Validation
    ///
    /// Before any edges are inserted, all source and target entity references
    /// are validated to ensure they exist. If any entity reference is invalid,
    /// the entire batch is rejected and no edges are inserted.
    ///
    /// # Performance
    ///
    /// This method achieves high throughput by:
    /// 1. **Parallel validation**: Entity existence checks are parallelized
    /// 2. **Parallel serialization**: Edges are serialized to binary format in parallel
    /// 3. **Single transaction**: All writes occur within one transaction (one fsync)
    /// 4. **Bulk ID allocation**: Edge IDs are allocated in a single atomic batch
    /// 5. **Index maintenance**: All edge indexes are updated within the same transaction
    ///
    /// # Arguments
    ///
    /// * `edges` - The edges to insert. All edges should have their source, target,
    ///   edge_type, and properties set. The `id` field will be overwritten with
    ///   auto-generated IDs.
    ///
    /// # Returns
    ///
    /// A vector of [`EdgeId`] for the inserted edges, in the same order as the input.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any source entity doesn't exist (the entire batch is aborted)
    /// - Any target entity doesn't exist (the entire batch is aborted)
    /// - Serialization fails for any edge
    /// - The transaction cannot be committed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, Edge, EdgeId, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // First create some entities
    /// let entity_ids = db.bulk_insert_entities(&entities)?;
    ///
    /// // Create edges between entities (IDs will be assigned by bulk_insert_edges)
    /// let edges: Vec<Edge> = entity_ids.windows(2)
    ///     .map(|pair| {
    ///         Edge::new(EdgeId::new(0), pair[0], pair[1], "FOLLOWS") // ID is placeholder
    ///             .with_property("weight", 1.0f64)
    ///     })
    ///     .collect();
    ///
    /// // Insert all at once - much faster than individual inserts
    /// let edge_ids = db.bulk_insert_edges(&edges)?;
    /// ```
    pub fn bulk_insert_edges(&self, edges: &[Edge]) -> Result<Vec<EdgeId>> {
        use manifoldb_graph::index::IndexMaintenance;
        use rayon::prelude::*;

        // Table names matching transaction handle
        const TABLE_EDGES: &str = "edges";
        const TABLE_EDGES_OUT: &str = "edges_out";
        const TABLE_EDGES_IN: &str = "edges_in";

        // Helper to create adjacency key (same as transaction handle)
        fn make_adjacency_key(entity_id: EntityId, edge_id: EdgeId) -> [u8; 16] {
            let mut key = [0u8; 16];
            key[0..8].copy_from_slice(&entity_id.as_u64().to_be_bytes());
            key[8..16].copy_from_slice(&edge_id.as_u64().to_be_bytes());
            key
        }

        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let start = std::time::Instant::now();

        // Phase 1: Validate all entity references exist
        // We need to check all source and target entities before any writes
        {
            let tx = self.begin_read()?;

            // Collect all unique entity IDs to check
            let mut entity_ids_to_check: Vec<EntityId> =
                edges.iter().flat_map(|e| [e.source, e.target]).collect();
            entity_ids_to_check.sort_unstable();
            entity_ids_to_check.dedup();

            // Check all entities exist
            for entity_id in &entity_ids_to_check {
                if tx.get_entity(*entity_id)?.is_none() {
                    return Err(Error::InvalidEntityReference(*entity_id));
                }
            }
        }

        // Phase 2: Parallel serialization (validation pass)
        // Serialize all edges in parallel to catch encoding errors early
        let serialized: std::result::Result<Vec<(usize, Vec<u8>)>, Error> = edges
            .par_iter()
            .enumerate()
            .map(|(idx, edge)| {
                bincode::serde::encode_to_vec(edge, bincode::config::standard())
                    .map(|bytes| (idx, bytes))
                    .map_err(|e| {
                        Error::Execution(format!(
                            "Failed to serialize edge at index {}: {}",
                            idx, e
                        ))
                    })
            })
            .collect();

        let serialized = serialized?;

        // Phase 3: Begin transaction and allocate IDs
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Get the starting edge ID and reserve a range
        let edge_count = edges.len() as u64;
        let start_id = {
            // Read current counter
            let current = match tx.get_metadata(b"next_edge_id")? {
                Some(bytes) if bytes.len() == 8 => {
                    let arr: [u8; 8] = bytes
                        .try_into()
                        .map_err(|_| Error::Execution("invalid edge counter".to_string()))?;
                    u64::from_be_bytes(arr)
                }
                _ => 1, // Start from 1 if not set
            };

            // Update counter to reserve the range
            let next = current + edge_count;
            tx.put_metadata(b"next_edge_id", &next.to_be_bytes())?;

            current
        };

        // Phase 4: Re-serialize with correct IDs (in parallel)
        // Now we have the IDs, we need to serialize again with the correct IDs
        // Use Edge::encode() for graph layer compatibility
        let edges_with_ids: std::result::Result<Vec<(EdgeId, Edge, Vec<u8>)>, Error> = edges
            .par_iter()
            .enumerate()
            .map(|(idx, edge)| {
                let id = EdgeId::new(start_id + idx as u64);
                let mut edge_with_id = edge.clone();
                edge_with_id.id = id;

                edge_with_id.encode().map(|bytes| (id, edge_with_id, bytes)).map_err(|e| {
                    Error::Execution(format!("Failed to serialize edge at index {}: {}", idx, e))
                })
            })
            .collect();

        let edges_with_ids = edges_with_ids?;

        // Drop the initial serialized results - we only needed them for validation
        drop(serialized);

        // Phase 5: Sequential writes (fast - just memcpy to transaction buffer)
        let ids: Vec<EdgeId> = edges_with_ids.iter().map(|(id, _, _)| *id).collect();

        for (id, edge, bytes) in &edges_with_ids {
            // Store edge data with key encoding for graph layer compatibility
            let key = encode_edge_key(*id);
            tx.storage_mut_ref()
                .map_err(Error::Transaction)?
                .put(TABLE_EDGES, &key, bytes)
                .map_err(|e| Error::Execution(format!("Failed to write edge: {}", e)))?;

            // Update simple outgoing edge index (source -> edge)
            let out_key = make_adjacency_key(edge.source, *id);
            tx.storage_mut_ref()
                .map_err(Error::Transaction)?
                .put(TABLE_EDGES_OUT, &out_key, &[])
                .map_err(|e| Error::Execution(format!("Failed to write outgoing index: {}", e)))?;

            // Update simple incoming edge index (target -> edge)
            let in_key = make_adjacency_key(edge.target, *id);
            tx.storage_mut_ref()
                .map_err(Error::Transaction)?
                .put(TABLE_EDGES_IN, &in_key, &[])
                .map_err(|e| Error::Execution(format!("Failed to write incoming index: {}", e)))?;

            // Phase 6: Index maintenance - update graph layer indexes
            // (edges_by_source, edges_by_target, edge_types)
            IndexMaintenance::add_edge_indexes(
                tx.storage_mut_ref().map_err(Error::Transaction)?,
                edge,
            )
            .map_err(|e| Error::Execution(format!("Edge index maintenance failed: {}", e)))?;
        }

        // Phase 7: Commit
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful bulk insert
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(ids)
    }

    // ========================================================================
    // Bulk Vector Operations
    // ========================================================================

    /// Bulk insert vectors for entities.
    ///
    /// This method efficiently inserts multiple vectors in a single batch operation.
    /// All vectors are stored atomically. HNSW indexes are updated if they exist
    /// for the specified vector names.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection
    /// * `vectors` - List of (entity_id, vector_name, vector_data) tuples
    ///
    /// # Returns
    ///
    /// The number of vectors successfully inserted.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any referenced entity doesn't exist
    /// - The storage operation fails
    ///
    /// The operation is all-or-nothing: if any vector fails validation,
    /// no vectors are inserted.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create entities first
    /// let mut tx = db.begin()?;
    /// let entity1 = tx.create_entity()?.with_label("documents");
    /// let entity2 = tx.create_entity()?.with_label("documents");
    /// tx.put_entity(&entity1)?;
    /// tx.put_entity(&entity2)?;
    /// tx.commit()?;
    ///
    /// // Bulk insert vectors
    /// let vectors = vec![
    ///     (entity1.id, "text_embedding".to_string(), vec![0.1f32; 384]),
    ///     (entity2.id, "text_embedding".to_string(), vec![0.2f32; 384]),
    /// ];
    ///
    /// let count = db.bulk_insert_vectors("documents", &vectors)?;
    /// assert_eq!(count, 2);
    /// ```
    ///
    /// # Performance
    ///
    /// This method is optimized for high throughput:
    /// - Single transaction for all storage operations
    /// - Batch HNSW index updates
    /// - Target: 100K+ vectors/second for typical workloads
    pub fn bulk_insert_vectors(
        &self,
        collection_name: &str,
        vectors: &[(manifoldb_core::EntityId, String, Vec<f32>)],
    ) -> Result<usize> {
        use crate::collection::{CollectionManager, CollectionName};
        use crate::vector::update_point_vector_in_index;
        use manifoldb_core::PointId;
        use manifoldb_vector::{
            encode_vector_value, encoding::encode_collection_vector_key, VectorData,
            TABLE_COLLECTION_VECTORS,
        };

        if vectors.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        let count = vectors.len();

        // Parse and validate collection name
        let coll_name =
            CollectionName::new(collection_name).map_err(|e| Error::InvalidInput(e.to_string()))?;

        // Phase 1: Validate all entities exist
        {
            let tx = self.begin_read()?;
            for (entity_id, _, _) in vectors {
                if tx.get_entity(*entity_id)?.is_none() {
                    return Err(Error::EntityNotFound(*entity_id));
                }
            }
        }

        // Phase 2: Store vectors in the CollectionVectorStore and update HNSW indexes
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Get or create collection ID
        let collection_id = match CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
        {
            Some(collection) => collection.id(),
            None => {
                // Create collection on first use with no vector configs
                // (vector configs can be added later if needed for schema validation)
                let collection = CollectionManager::create(&mut tx, &coll_name, std::iter::empty())
                    .map_err(|e| Error::Collection(e.to_string()))?;
                collection.id()
            }
        };

        // Store each vector in the collection_vectors table
        {
            let storage = tx.storage_mut().map_err(Error::Transaction)?;
            for (entity_id, vector_name, data) in vectors {
                // Convert Vec<f32> to VectorData::Dense
                let vector_data = VectorData::Dense(data.clone());

                // Encode key and value
                let key = encode_collection_vector_key(collection_id, *entity_id, vector_name);
                let value = encode_vector_value(&vector_data, vector_name);

                // Store in the collection_vectors table
                storage.put(TABLE_COLLECTION_VECTORS, &key, &value).map_err(Error::Storage)?;
            }
        }

        // Phase 3: Update HNSW indexes for each vector
        for (entity_id, vector_name, data) in vectors {
            let point_id = PointId::new(entity_id.as_u64());
            // update_point_vector_in_index checks if an index exists and only updates if it does
            update_point_vector_in_index(&mut tx, collection_name, vector_name, point_id, data)
                .map_err(|e| Error::Vector(e.to_string()))?;
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful operation
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(count)
    }

    /// Bulk insert vectors for a single named vector across multiple entities.
    ///
    /// This is a convenience method for the common case where all vectors
    /// have the same name (e.g., all are "text_embedding").
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection
    /// * `vector_name` - The name of the vector field
    /// * `vectors` - List of (entity_id, vector_data) tuples
    ///
    /// # Returns
    ///
    /// The number of vectors successfully inserted.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vectors = vec![
    ///     (entity1.id, vec![0.1f32; 384]),
    ///     (entity2.id, vec![0.2f32; 384]),
    /// ];
    ///
    /// let count = db.bulk_insert_named_vectors("documents", "text_embedding", &vectors)?;
    /// ```
    pub fn bulk_insert_named_vectors(
        &self,
        collection_name: &str,
        vector_name: &str,
        vectors: &[(manifoldb_core::EntityId, Vec<f32>)],
    ) -> Result<usize> {
        let expanded: Vec<(manifoldb_core::EntityId, String, Vec<f32>)> =
            vectors.iter().map(|(id, data)| (*id, vector_name.to_string(), data.clone())).collect();

        self.bulk_insert_vectors(collection_name, &expanded)
    }

    // ========================================================================
    // Bulk Delete Vector Operations
    // ========================================================================

    /// Delete specific vectors by entity ID and vector name.
    ///
    /// This method removes vector properties from entities. Each entry in the
    /// `vectors` slice is a tuple of `(entity_id, vector_name)` specifying which
    /// vector to delete from which entity.
    ///
    /// # Use Cases
    ///
    /// - Remove embeddings when re-embedding with a different model
    /// - Clean up vectors for entities that no longer need them
    /// - Selective vector removal (e.g., delete image embeddings, keep text embeddings)
    ///
    /// # Arguments
    ///
    /// * `vectors` - List of `(entity_id, vector_name)` tuples specifying which
    ///   vectors to delete
    ///
    /// # Returns
    ///
    /// The number of vectors that were actually deleted. This may be less than
    /// the input count if some vectors didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - An entity referenced in the input does not exist
    /// - The transaction cannot be committed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    /// use manifoldb_core::EntityId;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // First, insert some vectors
    /// let vectors = vec![
    ///     (entity1.id, "text_embedding".to_string(), vec![0.1f32; 384]),
    ///     (entity1.id, "image_embedding".to_string(), vec![0.2f32; 512]),
    ///     (entity2.id, "text_embedding".to_string(), vec![0.3f32; 384]),
    /// ];
    /// db.bulk_insert_vectors("documents", &vectors)?;
    ///
    /// // Now delete specific vectors
    /// let to_delete = vec![
    ///     (entity1.id, "image_embedding".to_string()),
    ///     (entity2.id, "text_embedding".to_string()),
    /// ];
    /// let deleted = db.bulk_delete_vectors(&to_delete)?;
    /// assert_eq!(deleted, 2);
    /// ```
    pub fn bulk_delete_vectors(
        &self,
        vectors: &[(manifoldb_core::EntityId, String)],
    ) -> Result<usize> {
        use crate::collection::{CollectionManager, CollectionName};
        use crate::vector::remove_point_vector_from_index;
        use manifoldb_core::PointId;
        use manifoldb_vector::{encoding::encode_collection_vector_key, TABLE_COLLECTION_VECTORS};

        if vectors.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();

        // Phase 1: Validate all entities exist
        {
            let tx = self.begin_read()?;
            for (entity_id, _) in vectors {
                if tx.get_entity(*entity_id)?.is_none() {
                    return Err(Error::EntityNotFound(*entity_id));
                }
            }
        }

        // Phase 2: Delete vectors from all collections
        // Since we don't know which collection the vector belongs to, we need to check all
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        let mut deleted_count = 0;

        // Get all collection names and IDs by listing collection names and looking them up
        let collections: Vec<(CollectionName, manifoldb_core::CollectionId)> = {
            let names =
                CollectionManager::list(&tx).map_err(|e| Error::Collection(e.to_string()))?;
            let mut result = Vec::new();
            for name in names {
                if let Some(collection) = CollectionManager::get(&tx, &name)
                    .map_err(|e| Error::Collection(e.to_string()))?
                {
                    result.push((name, collection.id()));
                }
            }
            result
        };

        // Track which vectors were deleted from which collections for HNSW updates
        let mut hnsw_updates: Vec<(&str, manifoldb_core::EntityId, &str)> = Vec::new();

        {
            let storage = tx.storage_mut().map_err(Error::Transaction)?;

            for (entity_id, vector_name) in vectors {
                // Try to delete from each collection
                for (collection_name, collection_id) in &collections {
                    let key = encode_collection_vector_key(*collection_id, *entity_id, vector_name);
                    // Check if key exists and delete it
                    if storage
                        .get(TABLE_COLLECTION_VECTORS, &key)
                        .map_err(Error::Storage)?
                        .is_some()
                    {
                        storage.delete(TABLE_COLLECTION_VECTORS, &key).map_err(Error::Storage)?;
                        deleted_count += 1;
                        // Track for HNSW update
                        hnsw_updates.push((collection_name.as_str(), *entity_id, vector_name));
                        break; // Found and deleted, no need to check other collections
                    }
                }
            }
        }

        // Phase 3: Update HNSW indexes - remove deleted vectors
        for (collection_name, entity_id, vector_name) in hnsw_updates {
            let point_id = PointId::new(entity_id.as_u64());
            remove_point_vector_from_index(&mut tx, collection_name, vector_name, point_id)
                .map_err(|e| Error::Vector(e.to_string()))?;
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful operation
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(deleted_count)
    }

    /// Delete all vectors with a given name across multiple entities.
    ///
    /// This is a convenience method for the common case where you want to
    /// delete the same named vector from multiple entities.
    ///
    /// # Use Cases
    ///
    /// - Re-embed all documents with a new model (delete old embeddings first)
    /// - Remove a deprecated embedding type across the dataset
    /// - Clean up after changing vector dimension requirements
    ///
    /// # Arguments
    ///
    /// * `vector_name` - The name of the vector field to delete
    /// * `entity_ids` - List of entity IDs from which to delete the vector
    ///
    /// # Returns
    ///
    /// The number of vectors that were actually deleted. This may be less than
    /// the input count if some vectors didn't exist.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    /// use manifoldb_core::EntityId;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Delete text embeddings from multiple entities
    /// let entity_ids = vec![entity1.id, entity2.id, entity3.id];
    /// let deleted = db.bulk_delete_vectors_by_name("text_embedding", &entity_ids)?;
    /// ```
    pub fn bulk_delete_vectors_by_name(
        &self,
        vector_name: &str,
        entity_ids: &[manifoldb_core::EntityId],
    ) -> Result<usize> {
        let expanded: Vec<(manifoldb_core::EntityId, String)> =
            entity_ids.iter().map(|id| (*id, vector_name.to_string())).collect();

        self.bulk_delete_vectors(&expanded)
    }

    // ========================================================================
    // Bulk Update Vector Operations
    // ========================================================================

    /// Bulk update (replace) vectors for entities.
    ///
    /// This method updates existing vectors with new data. It is optimized for
    /// re-embedding scenarios where you need to replace vectors with a new/better model.
    ///
    /// Unlike `bulk_insert_vectors`, this method:
    /// - Validates that entities already have vectors with the specified names
    /// - Returns an error if any entity is missing the vector to update
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection (used for HNSW index lookup)
    /// * `vectors` - List of (entity_id, vector_name, vector_data) tuples
    ///
    /// # Returns
    ///
    /// The number of vectors successfully updated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any entity does not exist
    /// - Any entity does not have a vector with the specified name
    /// - The transaction cannot be committed
    ///
    /// The operation is all-or-nothing: if any vector fails validation, no changes are made.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Re-embed documents with a new model
    /// let new_embeddings = vec![
    ///     (entity1.id, "text_embedding".to_string(), new_model.encode("doc1")),
    ///     (entity2.id, "text_embedding".to_string(), new_model.encode("doc2")),
    /// ];
    ///
    /// let count = db.bulk_update_vectors("documents", &new_embeddings)?;
    /// assert_eq!(count, 2);
    /// ```
    ///
    /// # Performance
    ///
    /// This method is optimized for high throughput:
    /// - Single transaction for all storage operations
    /// - HNSW index updates use efficient delete-then-insert strategy
    /// - Target: 100K+ vectors/second for typical workloads
    pub fn bulk_update_vectors(
        &self,
        collection_name: &str,
        vectors: &[(manifoldb_core::EntityId, String, Vec<f32>)],
    ) -> Result<usize> {
        use crate::collection::{CollectionManager, CollectionName};
        use crate::vector::update_point_vector_in_index;
        use manifoldb_core::PointId;
        use manifoldb_vector::{
            encode_vector_value, encoding::encode_collection_vector_key, VectorData,
            TABLE_COLLECTION_VECTORS,
        };

        if vectors.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        let count = vectors.len();

        // Parse and validate collection name
        let coll_name =
            CollectionName::new(collection_name).map_err(|e| Error::InvalidInput(e.to_string()))?;

        // Phase 1: Validate all entities exist AND have vectors with the specified names in the collection
        {
            let tx = self.begin_read()?;

            // Get collection - must exist for update
            let collection = CollectionManager::get(&tx, &coll_name)
                .map_err(|e| Error::Collection(e.to_string()))?
                .ok_or_else(|| {
                    Error::Collection(format!("Collection '{}' not found", collection_name))
                })?;
            let collection_id = collection.id();

            let storage = tx.storage_ref().map_err(Error::Transaction)?;

            for (entity_id, vector_name, _) in vectors {
                // Entity must exist
                if tx.get_entity(*entity_id)?.is_none() {
                    return Err(Error::EntityNotFound(*entity_id));
                }

                // Check that the vector exists in collection_vectors table
                let key = encode_collection_vector_key(collection_id, *entity_id, vector_name);
                if storage.get(TABLE_COLLECTION_VECTORS, &key).map_err(Error::Storage)?.is_none() {
                    return Err(Error::Vector(format!(
                        "Entity {} does not have vector '{}' to update",
                        entity_id, vector_name
                    )));
                }
            }
        }

        // Phase 2: Update vectors in the collection_vectors table and HNSW indexes
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Get collection ID again in write transaction
        let collection_id = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
            .map(|c| c.id())
            .ok_or_else(|| {
                Error::Collection(format!("Collection '{}' not found", collection_name))
            })?;

        {
            let storage = tx.storage_mut().map_err(Error::Transaction)?;

            for (entity_id, vector_name, data) in vectors {
                // Convert Vec<f32> to VectorData::Dense
                let vector_data = VectorData::Dense(data.clone());

                // Encode key and value
                let key = encode_collection_vector_key(collection_id, *entity_id, vector_name);
                let value = encode_vector_value(&vector_data, vector_name);

                // Store in the collection_vectors table (overwrites existing)
                storage.put(TABLE_COLLECTION_VECTORS, &key, &value).map_err(Error::Storage)?;
            }
        }

        // Phase 3: Update HNSW indexes for each vector
        // update_point_vector_in_index handles updating existing entries in HNSW
        for (entity_id, vector_name, data) in vectors {
            let point_id = PointId::new(entity_id.as_u64());
            update_point_vector_in_index(&mut tx, collection_name, vector_name, point_id, data)
                .map_err(|e| Error::Vector(e.to_string()))?;
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful operation
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(count)
    }

    /// Bulk replace vectors for a single named vector across multiple entities.
    ///
    /// This is a convenience method for the common re-embedding scenario where
    /// all vectors have the same name (e.g., re-embedding all "text_embedding" vectors).
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection
    /// * `vector_name` - The name of the vector field to replace
    /// * `vectors` - List of (entity_id, vector_data) tuples
    ///
    /// # Returns
    ///
    /// The number of vectors successfully updated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any entity does not exist
    /// - Any entity does not have a vector with the specified name
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Re-embed all documents with a new model
    /// let new_embeddings: Vec<(EntityId, Vec<f32>)> = documents
    ///     .iter()
    ///     .map(|doc| (doc.id, new_model.encode(&doc.text)))
    ///     .collect();
    ///
    /// let count = db.bulk_replace_named_vectors(
    ///     "documents",
    ///     "text_embedding",
    ///     &new_embeddings
    /// )?;
    /// ```
    pub fn bulk_replace_named_vectors(
        &self,
        collection_name: &str,
        vector_name: &str,
        vectors: &[(manifoldb_core::EntityId, Vec<f32>)],
    ) -> Result<usize> {
        let expanded: Vec<(manifoldb_core::EntityId, String, Vec<f32>)> =
            vectors.iter().map(|(id, data)| (*id, vector_name.to_string(), data.clone())).collect();

        self.bulk_update_vectors(collection_name, &expanded)
    }

    // ========================================================================
    // Vector Retrieval Operations
    // ========================================================================

    /// Get a specific named vector for an entity.
    ///
    /// Retrieves a single vector from the collection's vector storage. This is
    /// the primary method for accessing vector data that was stored using
    /// `bulk_insert_vectors` or `bulk_insert_named_vectors`.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The collection containing the entity
    /// * `entity_id` - The entity ID
    /// * `vector_name` - The name of the vector (e.g., "text_embedding")
    ///
    /// # Returns
    ///
    /// The vector data if it exists, `None` otherwise. The return type is
    /// [`VectorData`](manifoldb_vector::VectorData) which can be dense, sparse,
    /// multi-vector, or binary.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection name is invalid
    /// - The collection does not exist
    /// - The storage operation fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // After inserting vectors...
    /// if let Some(vector) = db.get_vector("documents", entity_id, "text_embedding")? {
    ///     println!("Vector dimension: {}", vector.dimension());
    ///     if let Some(dense) = vector.as_dense() {
    ///         println!("First element: {}", dense[0]);
    ///     }
    /// }
    /// ```
    pub fn get_vector(
        &self,
        collection_name: &str,
        entity_id: manifoldb_core::EntityId,
        vector_name: &str,
    ) -> Result<Option<manifoldb_vector::VectorData>> {
        use crate::collection::{CollectionManager, CollectionName};
        use manifoldb_vector::{encoding::encode_collection_vector_key, TABLE_COLLECTION_VECTORS};

        // Parse and validate collection name
        let coll_name =
            CollectionName::new(collection_name).map_err(|e| Error::InvalidInput(e.to_string()))?;

        // Get collection ID (read-only)
        let tx = self.begin_read()?;
        let collection = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
            .ok_or_else(|| {
                Error::Collection(format!("collection '{}' not found", collection_name))
            })?;
        let collection_id = collection.id();

        // Access storage directly for vector lookup
        let storage = tx.storage_ref().map_err(Error::Transaction)?;
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);

        match storage.get(TABLE_COLLECTION_VECTORS, &key).map_err(Error::Storage)? {
            Some(bytes) => {
                let (data, _name) =
                    manifoldb_vector::store::decode_vector_value(&bytes).map_err(|e| {
                        Error::Storage(manifoldb_storage::StorageError::Serialization(
                            e.to_string(),
                        ))
                    })?;
                Ok(Some(data))
            }
            None => Ok(None),
        }
    }

    /// Get all named vectors for an entity.
    ///
    /// Returns a map of vector names to their data. Useful when an entity
    /// has multiple embeddings (text, image, summary, etc.).
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The collection containing the entity
    /// * `entity_id` - The entity ID
    ///
    /// # Returns
    ///
    /// A `HashMap` of vector_name → vector_data. Returns an empty map if the
    /// entity has no vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection name is invalid
    /// - The collection does not exist
    /// - The storage operation fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // After inserting multiple named vectors...
    /// let vectors = db.get_all_vectors("documents", entity_id)?;
    /// for (name, vec) in vectors {
    ///     println!("{}: {} dimensions", name, vec.dimension());
    /// }
    /// ```
    pub fn get_all_vectors(
        &self,
        collection_name: &str,
        entity_id: manifoldb_core::EntityId,
    ) -> Result<std::collections::HashMap<String, manifoldb_vector::VectorData>> {
        use crate::collection::{CollectionManager, CollectionName};
        use manifoldb_storage::Cursor;
        use manifoldb_vector::{encoding::encode_entity_vector_prefix, TABLE_COLLECTION_VECTORS};
        use std::ops::Bound;

        // Parse and validate collection name
        let coll_name =
            CollectionName::new(collection_name).map_err(|e| Error::InvalidInput(e.to_string()))?;

        // Get collection ID (read-only)
        let tx = self.begin_read()?;
        let collection = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
            .ok_or_else(|| {
                Error::Collection(format!("collection '{}' not found", collection_name))
            })?;
        let collection_id = collection.id();

        // Access storage directly for vector scan
        let storage = tx.storage_ref().map_err(Error::Transaction)?;
        let prefix = encode_entity_vector_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = storage
            .range(
                TABLE_COLLECTION_VECTORS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )
            .map_err(Error::Storage)?;

        let mut vectors = std::collections::HashMap::new();
        while let Some((_key, value)) = cursor.next().map_err(Error::Storage)? {
            let (data, vector_name) = manifoldb_vector::store::decode_vector_value(&value)
                .map_err(|e| {
                    Error::Storage(manifoldb_storage::StorageError::Serialization(e.to_string()))
                })?;
            vectors.insert(vector_name, data);
        }

        Ok(vectors)
    }

    /// Check if an entity has a specific named vector.
    ///
    /// This is a lightweight existence check that doesn't load the vector data.
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The collection containing the entity
    /// * `entity_id` - The entity ID
    /// * `vector_name` - The name of the vector to check
    ///
    /// # Returns
    ///
    /// `true` if the vector exists, `false` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection name is invalid
    /// - The collection does not exist
    /// - The storage operation fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// if db.has_vector("documents", entity_id, "text_embedding")? {
    ///     println!("Entity has a text embedding");
    /// } else {
    ///     println!("Entity needs embedding generation");
    /// }
    /// ```
    pub fn has_vector(
        &self,
        collection_name: &str,
        entity_id: manifoldb_core::EntityId,
        vector_name: &str,
    ) -> Result<bool> {
        use crate::collection::{CollectionManager, CollectionName};
        use manifoldb_vector::{encoding::encode_collection_vector_key, TABLE_COLLECTION_VECTORS};

        // Parse and validate collection name
        let coll_name =
            CollectionName::new(collection_name).map_err(|e| Error::InvalidInput(e.to_string()))?;

        // Get collection ID (read-only)
        let tx = self.begin_read()?;
        let collection = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
            .ok_or_else(|| {
                Error::Collection(format!("collection '{}' not found", collection_name))
            })?;
        let collection_id = collection.id();

        // Access storage directly for existence check
        let storage = tx.storage_ref().map_err(Error::Transaction)?;
        let key = encode_collection_vector_key(collection_id, entity_id, vector_name);

        Ok(storage.get(TABLE_COLLECTION_VECTORS, &key).map_err(Error::Storage)?.is_some())
    }

    // ========================================================================
    // Bulk Upsert Operations
    // ========================================================================

    /// Bulk upsert (insert or update) entities.
    ///
    /// For each entity in the input:
    /// - If an entity with the same ID exists: update it with the new data
    /// - If no entity with that ID exists: insert it as a new entity
    ///
    /// All operations are performed in a single transaction for atomicity.
    ///
    /// # Performance
    ///
    /// This method achieves high throughput by:
    /// 1. **Parallel existence check**: Entity IDs are checked in parallel
    /// 2. **Parallel serialization**: Entities are serialized to binary format in parallel
    /// 3. **Single transaction**: All writes occur within one transaction (one fsync)
    /// 4. **Index maintenance**: All indexes are updated appropriately for inserts and updates
    ///
    /// # Arguments
    ///
    /// * `entities` - The entities to upsert. Each entity must have a valid ID set.
    ///   For inserts, use a placeholder ID (e.g., `EntityId::new(0)`) and the method
    ///   will assign new sequential IDs. For updates, use the existing entity's ID.
    ///
    /// # Returns
    ///
    /// A tuple of `(inserted_count, updated_count)` indicating how many entities
    /// were inserted vs updated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Serialization fails for any entity
    /// - The transaction cannot be committed
    /// - Index maintenance fails
    ///
    /// The operation is all-or-nothing: if any entity fails, no changes are made.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, Entity, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // First, insert some entities
    /// let entities: Vec<Entity> = (0..100)
    ///     .map(|i| Entity::new(EntityId::new(0))
    ///         .with_label("Document")
    ///         .with_property("version", 1i64))
    ///     .collect();
    /// let ids = db.bulk_insert_entities(&entities)?;
    ///
    /// // Now upsert: update existing + insert new
    /// let mut upsert_entities: Vec<Entity> = Vec::new();
    ///
    /// // Update first 50 with new version
    /// for id in ids.iter().take(50) {
    ///     upsert_entities.push(
    ///         Entity::new(*id)
    ///             .with_label("Document")
    ///             .with_property("version", 2i64)
    ///     );
    /// }
    ///
    /// // Add 50 new entities
    /// for _ in 0..50 {
    ///     upsert_entities.push(
    ///         Entity::new(EntityId::new(0))
    ///             .with_label("Document")
    ///             .with_property("version", 1i64)
    ///     );
    /// }
    ///
    /// let (inserted, updated) = db.bulk_upsert_entities(&upsert_entities)?;
    /// assert_eq!(inserted, 50);
    /// assert_eq!(updated, 50);
    /// ```
    pub fn bulk_upsert_entities(&self, entities: &[Entity]) -> Result<(usize, usize)> {
        use crate::execution::EntityIndexMaintenance;
        use rayon::prelude::*;

        if entities.is_empty() {
            return Ok((0, 0));
        }

        let start = std::time::Instant::now();

        // Phase 1: Determine which entities exist and which are new
        // Entities with id == 0 are always new inserts
        // Entities with id != 0 need to be checked
        let mut to_insert: Vec<(usize, &Entity)> = Vec::new();
        let mut to_update: Vec<(usize, &Entity, Entity)> = Vec::new(); // (index, new_entity, old_entity)

        {
            let tx = self.begin_read()?;
            for (idx, entity) in entities.iter().enumerate() {
                if entity.id.as_u64() == 0 {
                    // New entity to insert
                    to_insert.push((idx, entity));
                } else {
                    // Check if it exists
                    match tx.get_entity(entity.id)? {
                        Some(old_entity) => {
                            // Entity exists - update
                            to_update.push((idx, entity, old_entity));
                        }
                        None => {
                            // Entity doesn't exist - treat as insert with specified ID
                            // Note: We'll need to handle this carefully since we can't
                            // insert with a specific ID in the normal flow.
                            // For now, we treat non-existent IDs as inserts that will get new IDs.
                            to_insert.push((idx, entity));
                        }
                    }
                }
            }
        }

        let inserted_count = to_insert.len();
        let updated_count = to_update.len();

        // Phase 2: Parallel serialization validation for all entities
        // This catches errors before we start any writes
        let validation_result: std::result::Result<(), Error> =
            entities.par_iter().enumerate().try_for_each(|(idx, entity)| {
                bincode::serde::encode_to_vec(entity, bincode::config::standard())
                    .map(|_| ())
                    .map_err(|e| {
                        Error::Execution(format!(
                            "Failed to serialize entity at index {}: {}",
                            idx, e
                        ))
                    })
            });
        validation_result?;

        // Phase 3: Begin write transaction
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        // Phase 4: Handle inserts - allocate new IDs
        let new_entity_ids = if to_insert.is_empty() {
            Vec::new()
        } else {
            let entity_count = to_insert.len() as u64;
            let start_id = {
                // Read current counter
                let current = match tx.get_metadata(b"next_entity_id")? {
                    Some(bytes) if bytes.len() == 8 => {
                        let arr: [u8; 8] = bytes
                            .try_into()
                            .map_err(|_| Error::Execution("invalid entity counter".to_string()))?;
                        u64::from_be_bytes(arr)
                    }
                    _ => 1, // Start from 1 if not set
                };

                // Update counter to reserve the range
                let next = current + entity_count;
                tx.put_metadata(b"next_entity_id", &next.to_be_bytes())?;

                current
            };

            // Assign IDs to entities being inserted
            to_insert
                .iter()
                .enumerate()
                .map(|(i, _)| EntityId::new(start_id + i as u64))
                .collect::<Vec<_>>()
        };

        // Phase 5: Parallel serialization with correct IDs for inserts
        let serialized_inserts: std::result::Result<Vec<(EntityId, Entity, Vec<u8>)>, Error> =
            to_insert
                .par_iter()
                .zip(new_entity_ids.par_iter())
                .map(|((_, entity), &id)| {
                    let mut entity_with_id = (*entity).clone();
                    entity_with_id.id = id;

                    bincode::serde::encode_to_vec(&entity_with_id, bincode::config::standard())
                        .map(|bytes| (id, entity_with_id, bytes))
                        .map_err(|e| {
                            Error::Execution(format!(
                                "Failed to serialize entity for insert: {}",
                                e
                            ))
                        })
                })
                .collect();
        let serialized_inserts = serialized_inserts?;

        // Phase 6: Parallel serialization for updates (keeping original IDs)
        let serialized_updates: std::result::Result<
            Vec<(EntityId, Entity, Entity, Vec<u8>)>,
            Error,
        > = to_update
            .par_iter()
            .map(|(_, new_entity, old_entity)| {
                let entity_with_id = (*new_entity).clone();
                // Note: new_entity should already have the correct ID from the input

                bincode::serde::encode_to_vec(&entity_with_id, bincode::config::standard())
                    .map(|bytes| (entity_with_id.id, entity_with_id, old_entity.clone(), bytes))
                    .map_err(|e| {
                        Error::Execution(format!("Failed to serialize entity for update: {}", e))
                    })
            })
            .collect();
        let serialized_updates = serialized_updates?;

        // Phase 7: Sequential writes for inserts
        for (id, entity, bytes) in &serialized_inserts {
            let key = id.as_u64().to_be_bytes();
            let storage = tx.storage_mut_ref().map_err(Error::Transaction)?;

            storage
                .put("entities", &key, bytes)
                .map_err(|e| Error::Execution(format!("Failed to write entity: {}", e)))?;

            // Label index maintenance for insert
            for label in &entity.labels {
                let label_bytes = label.as_str().as_bytes();
                let len = label_bytes.len() as u16;
                let mut label_key = Vec::with_capacity(2 + label_bytes.len() + 8);
                label_key.extend_from_slice(&len.to_be_bytes());
                label_key.extend_from_slice(label_bytes);
                label_key.extend_from_slice(&id.as_u64().to_be_bytes());
                storage
                    .put("label_index", &label_key, &[])
                    .map_err(|e| Error::Execution(format!("Failed to write label index: {}", e)))?;
            }

            // Index maintenance for insert (schema-based indexes)
            EntityIndexMaintenance::on_insert(&mut tx, entity)
                .map_err(|e| Error::Execution(format!("Index maintenance failed: {}", e)))?;

            // Payload index maintenance for insert
            self.inner.index_manager.on_entity_upsert_tx(
                tx.storage_mut_ref().map_err(Error::Transaction)?,
                entity,
                None, // New entity, no old version
            )?;
        }

        // Phase 8: Sequential writes for updates
        for (id, new_entity, old_entity, bytes) in &serialized_updates {
            let key = id.as_u64().to_be_bytes();
            let storage = tx.storage_mut_ref().map_err(Error::Transaction)?;

            storage
                .put("entities", &key, bytes)
                .map_err(|e| Error::Execution(format!("Failed to write entity: {}", e)))?;

            // Label index maintenance for update - remove old labels not in new
            for old_label in &old_entity.labels {
                if !new_entity.labels.contains(old_label) {
                    let label_bytes = old_label.as_str().as_bytes();
                    let len = label_bytes.len() as u16;
                    let mut label_key = Vec::with_capacity(2 + label_bytes.len() + 8);
                    label_key.extend_from_slice(&len.to_be_bytes());
                    label_key.extend_from_slice(label_bytes);
                    label_key.extend_from_slice(&id.as_u64().to_be_bytes());
                    storage.delete("label_index", &label_key).map_err(|e| {
                        Error::Execution(format!("Failed to delete label index: {}", e))
                    })?;
                }
            }
            // Add new labels not in old
            for new_label in &new_entity.labels {
                if !old_entity.labels.contains(new_label) {
                    let label_bytes = new_label.as_str().as_bytes();
                    let len = label_bytes.len() as u16;
                    let mut label_key = Vec::with_capacity(2 + label_bytes.len() + 8);
                    label_key.extend_from_slice(&len.to_be_bytes());
                    label_key.extend_from_slice(label_bytes);
                    label_key.extend_from_slice(&id.as_u64().to_be_bytes());
                    storage.put("label_index", &label_key, &[]).map_err(|e| {
                        Error::Execution(format!("Failed to write label index: {}", e))
                    })?;
                }
            }

            // Index maintenance for update (schema-based indexes)
            EntityIndexMaintenance::on_update(&mut tx, old_entity, new_entity)
                .map_err(|e| Error::Execution(format!("Index maintenance failed: {}", e)))?;

            // Payload index maintenance for update
            self.inner.index_manager.on_entity_upsert_tx(
                tx.storage_mut_ref().map_err(Error::Transaction)?,
                new_entity,
                Some(old_entity),
            )?;
        }

        // Phase 9: Commit
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful bulk upsert
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok((inserted_count, updated_count))
    }

    // ========================================================================
    // Bulk Delete Operations
    // ========================================================================

    /// Bulk delete entities by ID.
    ///
    /// All deletions happen in a single transaction. This method properly
    /// cleans up all property indexes and vector indexes for deleted entities.
    ///
    /// By default, this method performs cascade deletion of connected edges.
    /// Use [`bulk_delete_entities_checked`](Self::bulk_delete_entities_checked) if you
    /// want to error when entities have connected edges.
    ///
    /// # Arguments
    ///
    /// * `entity_ids` - The IDs of entities to delete
    ///
    /// # Returns
    ///
    /// The number of entities that were actually deleted. This may be less
    /// than the input size if some entities didn't exist.
    ///
    /// # Performance
    ///
    /// This method achieves high throughput by:
    /// 1. **Single transaction**: All deletes occur within one transaction (one fsync)
    /// 2. **Batch index cleanup**: Property index entries removed efficiently
    /// 3. **Cascade edge deletion**: All connected edges deleted within same transaction
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create some entities
    /// let ids = db.bulk_insert_entities(&entities)?;
    ///
    /// // Delete half of them
    /// let to_delete: Vec<EntityId> = ids[0..ids.len()/2].to_vec();
    /// let deleted = db.bulk_delete_entities(&to_delete)?;
    /// assert_eq!(deleted, to_delete.len());
    /// ```
    pub fn bulk_delete_entities(&self, entity_ids: &[EntityId]) -> Result<usize> {
        self.bulk_delete_entities_impl(entity_ids, true)
    }

    /// Bulk delete entities by ID, erroring if any entity has connected edges.
    ///
    /// Similar to [`bulk_delete_entities`](Self::bulk_delete_entities), but
    /// returns an error if any entity has connected edges instead of
    /// automatically deleting them.
    ///
    /// # Arguments
    ///
    /// * `entity_ids` - The IDs of entities to delete
    ///
    /// # Returns
    ///
    /// The number of entities that were actually deleted.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BulkOperation`] if any entity has connected edges.
    /// In this case, no entities are deleted (the operation is atomic).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Delete entities that shouldn't have any edges
    /// match db.bulk_delete_entities_checked(&entity_ids) {
    ///     Ok(count) => println!("Deleted {} entities", count),
    ///     Err(e) => println!("Some entities had edges: {}", e),
    /// }
    /// ```
    pub fn bulk_delete_entities_checked(&self, entity_ids: &[EntityId]) -> Result<usize> {
        self.bulk_delete_entities_impl(entity_ids, false)
    }

    /// Internal implementation of bulk delete with cascade control.
    fn bulk_delete_entities_impl(
        &self,
        entity_ids: &[EntityId],
        cascade_edges: bool,
    ) -> Result<usize> {
        use crate::collection::CollectionManager;
        use crate::execution::EntityIndexMaintenance;
        use manifoldb_storage::Cursor;
        use manifoldb_vector::{encoding::encode_entity_vector_prefix, TABLE_COLLECTION_VECTORS};
        use std::collections::HashSet;
        use std::ops::Bound;

        if entity_ids.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();

        // Begin a write transaction
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        let mut deleted_count = 0;
        let mut affected_tables: HashSet<String> = HashSet::new();

        // Get all collection IDs for vector cascade deletion
        let collection_ids: Vec<_> = {
            let names =
                CollectionManager::list(&tx).map_err(|e| Error::Collection(e.to_string()))?;
            let mut ids = Vec::new();
            for name in names {
                if let Some(collection) = CollectionManager::get(&tx, &name)
                    .map_err(|e| Error::Collection(e.to_string()))?
                {
                    ids.push(collection.id());
                }
            }
            ids
        };

        for &entity_id in entity_ids {
            // Load the entity to get property values for index cleanup
            let entity = match tx.get_entity(entity_id)? {
                Some(e) => e,
                None => continue, // Entity doesn't exist, skip
            };

            // Track affected tables for cache invalidation
            for label in &entity.labels {
                affected_tables.insert(label.as_str().to_string());
            }

            // Handle edges
            if cascade_edges {
                // Cascade delete: delete all connected edges
                let outgoing = tx.get_outgoing_edges(entity_id)?;
                let incoming = tx.get_incoming_edges(entity_id)?;

                // Delete outgoing edges
                for edge in &outgoing {
                    tx.delete_edge(edge.id)?;
                }

                // Delete incoming edges (skip self-loops which appear in both lists)
                for edge in &incoming {
                    if edge.source != edge.target {
                        tx.delete_edge(edge.id)?;
                    }
                }
            } else {
                // Checked delete: error if entity has edges
                if tx.has_edges(entity_id)? {
                    return Err(Error::bulk_operation(format!(
                        "entity {} has connected edges; use bulk_delete_entities for cascade delete",
                        entity_id.as_u64()
                    )));
                }
            }

            // Cascade delete: delete all vectors for this entity from all collections
            {
                let storage = tx.storage_mut().map_err(Error::Transaction)?;
                for &collection_id in &collection_ids {
                    // Create prefix for all vectors of this entity in this collection
                    let prefix = encode_entity_vector_prefix(collection_id, entity_id);

                    // Calculate next prefix for range scan
                    let prefix_end = {
                        let mut result = prefix.clone();
                        for byte in result.iter_mut().rev() {
                            if *byte < 0xFF {
                                *byte += 1;
                                break;
                            }
                        }
                        result
                    };

                    // Collect keys to delete (can't delete while iterating)
                    let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
                    {
                        let mut cursor = storage
                            .range(
                                TABLE_COLLECTION_VECTORS,
                                Bound::Included(prefix.as_slice()),
                                Bound::Excluded(prefix_end.as_slice()),
                            )
                            .map_err(Error::Storage)?;

                        while let Some((key, _)) = cursor.next().map_err(Error::Storage)? {
                            keys_to_delete.push(key.clone());
                        }
                    }

                    // Delete collected keys
                    for key in keys_to_delete {
                        storage.delete(TABLE_COLLECTION_VECTORS, &key).map_err(Error::Storage)?;
                    }
                }
            }

            // Remove from property indexes before deleting (schema-based indexes)
            EntityIndexMaintenance::on_delete(&mut tx, &entity)
                .map_err(|e| Error::Execution(format!("property index removal failed: {e}")))?;

            // Remove from payload indexes before deleting
            self.inner
                .index_manager
                .on_entity_delete_tx(tx.storage_mut_ref().map_err(Error::Transaction)?, &entity)?;

            // Remove from HNSW/vector indexes
            crate::vector::remove_entity_from_indexes(&mut tx, &entity)
                .map_err(|e| Error::Execution(format!("vector index removal failed: {e}")))?;

            // Delete the entity itself
            if tx.delete_entity(entity_id)? {
                deleted_count += 1;
            }
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Invalidate cache entries for affected tables
        let table_list: Vec<String> = affected_tables.into_iter().collect();
        self.inner.query_cache.invalidate_tables(&table_list);
        if let Err(e) = self.inner.prepared_cache.invalidate_tables(&table_list) {
            // Log the error but don't fail the operation
            eprintln!("Warning: failed to invalidate prepared cache: {e}");
        }

        // Record successful operation
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(deleted_count)
    }

    /// Bulk delete edges by ID.
    ///
    /// All deletions happen in a single transaction with one fsync for
    /// maximum performance. Edge indexes are properly cleaned up.
    ///
    /// # Arguments
    ///
    /// * `edge_ids` - The IDs of edges to delete
    ///
    /// # Returns
    ///
    /// The number of edges that were actually deleted. This may be less
    /// than the input length if some edges didn't exist.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use manifoldb::{Database, EdgeId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create some edges...
    /// let edge_ids: Vec<EdgeId> = /* ... */;
    ///
    /// // Delete them all in one transaction
    /// let deleted = db.bulk_delete_edges(&edge_ids)?;
    /// assert_eq!(deleted, edge_ids.len());
    /// ```
    pub fn bulk_delete_edges(&self, edge_ids: &[EdgeId]) -> Result<usize> {
        if edge_ids.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();

        // Begin a write transaction
        let mut tx = self.begin()?;
        self.inner.db_metrics.transactions.record_start();

        let mut deleted_count = 0;

        for &edge_id in edge_ids {
            // delete_edge handles:
            // - Loading the edge to get source/target
            // - Deleting from main edges table
            // - Removing from EDGES_OUT and EDGES_IN adjacency indexes
            // - Removing from graph layer indexes (edges_by_source, edges_by_target, edge_types)
            if tx.delete_edge(edge_id)? {
                deleted_count += 1;
            }
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.inner.db_metrics.record_commit(commit_start.elapsed());

        // Record successful operation
        self.inner.db_metrics.record_query(start.elapsed(), true);

        Ok(deleted_count)
    }

    // ========================================================================
    // Collection API
    // ========================================================================

    /// Create a new collection with the given name.
    ///
    /// Returns a [`CollectionBuilder`] for configuring the collection's vectors
    /// and indexes. Call `.build()` to finalize the collection creation.
    ///
    /// # Arguments
    ///
    /// * `name` - The name for the collection (e.g., "documents", "products")
    ///
    /// # Errors
    ///
    /// Returns an error if the collection name is invalid.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::{Database, collection::DistanceMetric};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create a collection with a dense vector
    /// let collection = db.create_collection("documents")?
    ///     .with_dense_vector("text_embedding", 768, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// // Create a hybrid collection with dense and sparse vectors
    /// let collection = db.create_collection("articles")?
    ///     .with_dense_vector("semantic", 384, DistanceMetric::DotProduct)
    ///     .with_sparse_vector("keywords")
    ///     .build()?;
    /// ```
    pub fn create_collection(
        &self,
        name: &str,
    ) -> Result<
        crate::collection::CollectionBuilder<
            std::sync::Arc<manifoldb_storage::backends::RedbEngine>,
        >,
    > {
        let coll_name = crate::collection::CollectionName::new(name)
            .map_err(|e| Error::Collection(e.to_string()))?;

        Ok(crate::collection::CollectionBuilder::new(self.inner.manager.engine_arc(), coll_name))
    }

    /// Get a handle to an existing collection.
    ///
    /// Returns a [`CollectionHandle`] that can be used to perform point operations
    /// and vector searches on the collection.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the collection to open
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection name is invalid
    /// - The collection doesn't exist
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create a collection first
    /// db.create_collection("documents")?
    ///     .with_dense_vector("embedding", 384, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// // Later, get a handle to the collection
    /// let collection = db.collection("documents")?;
    ///
    /// // Use the handle for operations
    /// let results = collection.search("embedding")
    ///     .query(query_vector)
    ///     .limit(10)
    ///     .execute()?;
    /// ```
    pub fn collection(
        &self,
        name: &str,
    ) -> Result<
        crate::collection::CollectionHandle<
            std::sync::Arc<manifoldb_storage::backends::RedbEngine>,
        >,
    > {
        let coll_name = crate::collection::CollectionName::new(name)
            .map_err(|e| Error::Collection(e.to_string()))?;

        crate::collection::CollectionHandle::open(self.inner.manager.engine_arc(), coll_name)
            .map_err(|e| Error::Collection(e.to_string()))
    }

    /// Drop a collection and all its data.
    ///
    /// This permanently deletes:
    /// - The collection metadata
    /// - All vectors stored in the collection
    /// - All HNSW indexes for the collection
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the collection to drop
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection name is invalid
    /// - The collection doesn't exist
    /// - A storage error occurs
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create and then drop a collection
    /// db.create_collection("temp")?
    ///     .with_dense_vector("v", 128, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// db.drop_collection("temp")?;
    ///
    /// // Collection no longer exists
    /// assert!(db.collection("temp").is_err());
    /// ```
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        use crate::collection::{CollectionManager, CollectionName};
        use crate::vector::drop_indexes_for_collection;
        use manifoldb_storage::Cursor;
        use manifoldb_vector::{
            encoding::encode_collection_vector_prefix, TABLE_COLLECTION_VECTORS,
        };
        use std::ops::Bound;

        let coll_name = CollectionName::new(name).map_err(|e| Error::Collection(e.to_string()))?;

        let mut tx = self.begin()?;

        // Get the collection to verify it exists and get its ID
        let collection = CollectionManager::get(&tx, &coll_name)
            .map_err(|e| Error::Collection(e.to_string()))?
            .ok_or_else(|| Error::Collection(format!("Collection '{}' not found", name)))?;

        let collection_id = collection.id();

        // Drop all HNSW indexes for this collection
        drop_indexes_for_collection(&mut tx, name).map_err(|e| Error::Vector(e.to_string()))?;

        // Delete all vectors in the collection from collection_vectors table
        {
            let prefix = encode_collection_vector_prefix(collection_id);
            let next_prefix = next_prefix(&prefix);
            let storage = tx.storage_mut().map_err(Error::Transaction)?;

            // Collect all keys to delete using range cursor
            let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();
            let mut cursor = storage
                .range(
                    TABLE_COLLECTION_VECTORS,
                    Bound::Included(prefix.as_slice()),
                    Bound::Excluded(next_prefix.as_slice()),
                )
                .map_err(Error::Storage)?;

            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key.clone());
            }
            drop(cursor);

            // Delete each key
            for key in &keys_to_delete {
                storage.delete(TABLE_COLLECTION_VECTORS, key).map_err(Error::Storage)?;
            }
        }

        // Delete the collection metadata (if_exists = false to error if not found)
        CollectionManager::delete(&mut tx, &coll_name, false)
            .map_err(|e| Error::Collection(e.to_string()))?;

        tx.commit().map_err(Error::Transaction)?;

        Ok(())
    }

    /// List all collections in the database.
    ///
    /// Returns a vector of collection names.
    ///
    /// # Errors
    ///
    /// Returns an error if a storage error occurs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::Database;
    ///
    /// let db = Database::in_memory()?;
    ///
    /// db.create_collection("users")?
    ///     .with_dense_vector("embedding", 128, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// db.create_collection("products")?
    ///     .with_dense_vector("embedding", 256, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// let collections = db.list_collections()?;
    /// assert!(collections.contains(&"users".to_string()));
    /// assert!(collections.contains(&"products".to_string()));
    /// ```
    pub fn list_collections(&self) -> Result<Vec<String>> {
        use crate::collection::CollectionManager;

        let tx = self.begin_read()?;
        let collections =
            CollectionManager::list(&tx).map_err(|e| Error::Collection(e.to_string()))?;

        Ok(collections.into_iter().map(|c| c.as_str().to_string()).collect())
    }

    // ========================================================================
    // Unified Entity API
    // ========================================================================

    /// Create a search builder for vector similarity search.
    ///
    /// This is the unified search API that returns [`ScoredEntity`] results
    /// instead of collection-specific point types.
    ///
    /// # Arguments
    ///
    /// * `collection` - The name of the collection to search
    /// * `vector_name` - The name of the vector field to search
    ///
    /// # Returns
    ///
    /// A [`EntitySearchBuilder`] that can be configured with query vector,
    /// filters, and limits before executing the search.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::{Database, Filter, ScoredEntity};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create collection and insert data...
    ///
    /// // Search for similar entities
    /// let results: Vec<ScoredEntity> = db.search("documents", "embedding")
    ///     .query(query_vector)
    ///     .filter(Filter::eq("language", "rust"))
    ///     .limit(10)
    ///     .execute()?;
    ///
    /// for result in results {
    ///     println!("Entity {}: score {:.4}", result.entity.id.as_u64(), result.score);
    /// }
    /// ```
    pub fn search(
        &self,
        collection: &str,
        vector_name: &str,
    ) -> Result<crate::search::EntitySearchBuilder> {
        let handle = self.collection(collection)?;
        let engine = self.inner.manager.engine_arc();
        Ok(crate::search::EntitySearchBuilder::new(handle, engine, vector_name))
    }

    /// Upsert an entity into a collection.
    ///
    /// This is the unified upsert API that handles entities with optional vectors.
    /// The entity's properties are stored as payload, and any vectors attached
    /// to the entity are stored in the appropriate vector indexes.
    ///
    /// # Arguments
    ///
    /// * `collection` - The name of the collection to upsert into
    /// * `entity` - The entity to upsert (may include vectors via `with_vector()`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The collection doesn't exist
    /// - Vector dimensions don't match the collection schema
    /// - A storage error occurs
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::{Database, Entity, EntityId, VectorData, DistanceMetric};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// // Create collection with vector configuration
    /// db.create_collection("documents")?
    ///     .with_dense_vector("embedding", 768, DistanceMetric::Cosine)
    ///     .build()?;
    ///
    /// // Create entity with vector
    /// let entity = Entity::new(EntityId::new(1))
    ///     .with_label("Document")
    ///     .with_property("title", "Hello World")
    ///     .with_property("language", "rust")
    ///     .with_vector("embedding", vec![0.1f32; 768]);
    ///
    /// // Upsert entity (stores both properties and vectors)
    /// db.upsert("documents", &entity)?;
    /// ```
    pub fn upsert(&self, collection: &str, entity: &Entity) -> Result<()> {
        use crate::search::entity_to_point_struct;

        let handle = self.collection(collection)?;
        let point = entity_to_point_struct(entity, collection);

        handle.upsert_point(point).map_err(|e| Error::Collection(e.to_string()))
    }

    /// Upsert multiple entities into a collection.
    ///
    /// More efficient than calling `upsert` multiple times.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use manifoldb::{Database, Entity, EntityId};
    ///
    /// let db = Database::in_memory()?;
    ///
    /// let entities: Vec<Entity> = (0..100)
    ///     .map(|i| {
    ///         Entity::new(EntityId::new(i))
    ///             .with_label("Document")
    ///             .with_property("index", i as i64)
    ///             .with_vector("embedding", vec![0.1f32; 768])
    ///     })
    ///     .collect();
    ///
    /// db.upsert_batch("documents", &entities)?;
    /// ```
    pub fn upsert_batch(&self, collection: &str, entities: &[Entity]) -> Result<()> {
        use crate::search::entity_to_point_struct;

        let handle = self.collection(collection)?;

        for entity in entities {
            let point = entity_to_point_struct(entity, collection);
            handle.upsert_point(point).map_err(|e| Error::Collection(e.to_string()))?;
        }

        Ok(())
    }
}

// Note: Database automatically implements Send + Sync through its fields
// TransactionManager<RedbEngine> is Send + Sync, Config is Clone
// No unsafe impls needed - Rust derives these automatically

/// The result of a query execution.
///
/// `QueryResult` contains the rows returned by a SELECT query, along with
/// metadata about the result set.
///
/// # Examples
///
/// Iterate over results:
///
/// ```ignore
/// let results = db.query("SELECT * FROM users")?;
/// for row in &results {
///     let name: &str = row.get("name")?;
///     println!("User: {}", name);
/// }
/// ```
///
/// Access by index:
///
/// ```ignore
/// if let Some(row) = results.get(0) {
///     let name: &str = row.get(0)?;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The column names.
    columns: Vec<String>,
    /// The result rows.
    rows: Vec<QueryRow>,
}

impl QueryResult {
    /// Create an empty query result.
    #[must_use]
    pub fn empty() -> Self {
        Self { columns: Vec::new(), rows: Vec::new() }
    }

    /// Create a query result with the given columns and rows.
    #[must_use]
    pub fn new(columns: Vec<String>, rows: Vec<QueryRow>) -> Self {
        Self { columns, rows }
    }

    /// Create a query result from the query engine's result set.
    #[must_use]
    pub fn from_result_set(result_set: manifoldb_query::ResultSet) -> Self {
        let columns = result_set.columns().into_iter().map(|s| s.to_owned()).collect();
        let rows =
            result_set.into_iter().map(|row| QueryRow { values: row.values().to_vec() }).collect();
        Self { columns, rows }
    }

    /// Returns the column names.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Returns the rows.
    #[must_use]
    pub fn rows(&self) -> &[QueryRow] {
        &self.rows
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns `true` if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get a row by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&QueryRow> {
        self.rows.get(index)
    }

    /// Get the first row.
    #[must_use]
    pub fn first(&self) -> Option<&QueryRow> {
        self.rows.first()
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &QueryRow> {
        self.rows.iter()
    }

    /// Get the column index for a column name.
    #[must_use]
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }
}

impl IntoIterator for QueryResult {
    type Item = QueryRow;
    type IntoIter = std::vec::IntoIter<QueryRow>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.into_iter()
    }
}

impl<'a> IntoIterator for &'a QueryResult {
    type Item = &'a QueryRow;
    type IntoIter = std::slice::Iter<'a, QueryRow>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

/// A single row in a query result.
///
/// `QueryRow` provides access to the column values in a result row.
///
/// # Examples
///
/// Access values by column index:
///
/// ```ignore
/// let value = row.get(0)?;
/// ```
///
/// Get the raw values:
///
/// ```ignore
/// for value in row.values() {
///     println!("{:?}", value);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct QueryRow {
    values: Vec<manifoldb_core::Value>,
}

impl QueryRow {
    /// Create a new row with the given values.
    #[must_use]
    pub fn new(values: Vec<manifoldb_core::Value>) -> Self {
        Self { values }
    }

    /// Get a value by column index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&manifoldb_core::Value> {
        self.values.get(index)
    }

    /// Returns all values in the row.
    #[must_use]
    pub fn values(&self) -> &[manifoldb_core::Value] {
        &self.values
    }

    /// Returns the number of values in the row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if the row has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Try to get a value as a specific type.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let name: String = row.get_as(0)?;
    /// let age: i64 = row.get_as(1)?;
    /// ```
    pub fn get_as<T: FromValue>(&self, index: usize) -> Result<T> {
        self.values
            .get(index)
            .ok_or_else(|| Error::InvalidParameter(format!("column index {} out of bounds", index)))
            .and_then(T::from_value)
    }
}

/// Trait for converting from a database Value to a Rust type.
pub trait FromValue: Sized {
    /// Convert from a database Value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value cannot be converted to this type.
    fn from_value(value: &manifoldb_core::Value) -> Result<Self>;
}

impl FromValue for String {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::String(s) => Ok(s.clone()),
            _ => Err(Error::Type(format!("expected string, got {:?}", value))),
        }
    }
}

impl FromValue for i64 {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::Int(n) => Ok(*n),
            _ => Err(Error::Type(format!("expected integer, got {:?}", value))),
        }
    }
}

impl FromValue for f64 {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::Float(f) => Ok(*f),
            manifoldb_core::Value::Int(n) => Ok(*n as f64),
            _ => Err(Error::Type(format!("expected float, got {:?}", value))),
        }
    }
}

impl FromValue for bool {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::Bool(b) => Ok(*b),
            _ => Err(Error::Type(format!("expected boolean, got {:?}", value))),
        }
    }
}

impl FromValue for Vec<f32> {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::Vector(v) => Ok(v.clone()),
            _ => Err(Error::Type(format!("expected vector, got {:?}", value))),
        }
    }
}

impl<T: FromValue> FromValue for Option<T> {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        match value {
            manifoldb_core::Value::Null => Ok(None),
            _ => T::from_value(value).map(Some),
        }
    }
}

impl FromValue for manifoldb_core::Value {
    fn from_value(value: &manifoldb_core::Value) -> Result<Self> {
        Ok(value.clone())
    }
}

/// Query parameters for parameterized queries.
///
/// This struct holds parameter values that can be bound to a query.
/// Parameters are referenced by position (`$1`, `$2`, etc.) in the SQL string.
///
/// # Examples
///
/// ```ignore
/// use manifoldb::{Database, QueryParams};
///
/// let db = Database::in_memory()?;
///
/// let mut params = QueryParams::new();
/// params.add("Alice");
/// params.add(30i64);
///
/// let results = db.query_with_params(
///     "SELECT * FROM users WHERE name = $1 AND age > $2",
///     params.values(),
/// )?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct QueryParams {
    values: Vec<manifoldb_core::Value>,
}

impl QueryParams {
    /// Create a new empty parameter set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter value.
    pub fn add(&mut self, value: impl Into<manifoldb_core::Value>) -> &mut Self {
        self.values.push(value.into());
        self
    }

    /// Add a parameter value (builder pattern).
    #[must_use]
    pub fn with(mut self, value: impl Into<manifoldb_core::Value>) -> Self {
        self.values.push(value.into());
        self
    }

    /// Returns the parameter values.
    #[must_use]
    pub fn values(&self) -> &[manifoldb_core::Value] {
        &self.values
    }

    /// Returns the number of parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no parameters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all parameters.
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

/// Create query parameters from an array.
///
/// # Examples
///
/// ```ignore
/// use manifoldb::{params, Database};
///
/// let db = Database::in_memory()?;
///
/// let results = db.query_with_params(
///     "SELECT * FROM users WHERE name = $1",
///     &params!["Alice"],
/// )?;
/// ```
#[macro_export]
macro_rules! params {
    () => {
        []
    };
    ($($value:expr),+ $(,)?) => {
        [$($crate::Value::from($value)),+]
    };
}

/// Calculate the next prefix for range scanning.
///
/// Given a prefix byte slice, returns a new slice that serves as an exclusive upper bound
/// for a range scan. This is used to efficiently iterate over all keys with a given prefix.
fn next_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut result = prefix.to_vec();
    for byte in result.iter_mut().rev() {
        if *byte < 0xFF {
            *byte += 1;
            return result;
        }
    }
    // All bytes are 0xFF, append another 0xFF
    result.push(0xFF);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_in_memory() {
        let db = Database::in_memory().expect("failed to create in-memory db");
        assert!(db.config().in_memory);
    }

    #[test]
    fn test_database_begin_transaction() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let tx = db.begin().expect("failed to begin transaction");
        assert!(!tx.is_read_only());
        tx.rollback().expect("failed to rollback");
    }

    #[test]
    fn test_database_begin_read_transaction() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let tx = db.begin_read().expect("failed to begin read transaction");
        assert!(tx.is_read_only());
        tx.rollback().expect("failed to rollback");
    }

    #[test]
    fn test_query_result_empty() {
        let result = QueryResult::empty();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
        assert_eq!(result.num_columns(), 0);
    }

    #[test]
    fn test_query_result_with_data() {
        let columns = vec!["id".to_string(), "name".to_string()];
        let rows = vec![
            QueryRow::new(vec![
                manifoldb_core::Value::Int(1),
                manifoldb_core::Value::String("Alice".to_string()),
            ]),
            QueryRow::new(vec![
                manifoldb_core::Value::Int(2),
                manifoldb_core::Value::String("Bob".to_string()),
            ]),
        ];

        let result = QueryResult::new(columns, rows);

        assert_eq!(result.len(), 2);
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.columns(), &["id", "name"]);
        assert_eq!(result.column_index("name"), Some(1));
    }

    #[test]
    fn test_query_row_get_as() {
        let row = QueryRow::new(vec![
            manifoldb_core::Value::Int(42),
            manifoldb_core::Value::String("test".to_string()),
            manifoldb_core::Value::Bool(true),
            manifoldb_core::Value::Float(2.5),
        ]);

        assert_eq!(row.get_as::<i64>(0).unwrap(), 42);
        assert_eq!(row.get_as::<String>(1).unwrap(), "test");
        assert_eq!(row.get_as::<bool>(2).unwrap(), true);
        assert!((row.get_as::<f64>(3).unwrap() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_query_row_get_as_error() {
        let row = QueryRow::new(vec![manifoldb_core::Value::Int(42)]);

        assert!(row.get_as::<String>(0).is_err());
        assert!(row.get_as::<i64>(999).is_err());
    }

    #[test]
    fn test_query_params() {
        let params = QueryParams::new().with("Alice").with(30i64).with(true);

        assert_eq!(params.len(), 3);
        assert_eq!(params.values()[0], manifoldb_core::Value::String("Alice".to_string()));
        assert_eq!(params.values()[1], manifoldb_core::Value::Int(30));
        assert_eq!(params.values()[2], manifoldb_core::Value::Bool(true));
    }

    #[test]
    fn test_query_result_iterator() {
        let columns = vec!["n".to_string()];
        let rows = vec![
            QueryRow::new(vec![manifoldb_core::Value::Int(1)]),
            QueryRow::new(vec![manifoldb_core::Value::Int(2)]),
            QueryRow::new(vec![manifoldb_core::Value::Int(3)]),
        ];

        let result = QueryResult::new(columns, rows);

        let sum: i64 = result.iter().filter_map(|r| r.get_as::<i64>(0).ok()).sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_entity_crud_via_database() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create and store an entity
        let mut tx = db.begin().expect("failed to begin write");
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("Person")
            .with_property("name", "Alice");
        let entity_id = entity.id;
        tx.put_entity(&entity).expect("failed to put entity");
        tx.commit().expect("failed to commit");

        // Read it back
        let tx = db.begin_read().expect("failed to begin read");
        let retrieved =
            tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
        assert_eq!(retrieved.id, entity_id);
        assert!(retrieved.has_label("Person"));
    }

    #[test]
    fn test_parse_and_execute_query() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Just verify parsing works
        let result = db.query("SELECT * FROM users WHERE id = 1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_invalid_query() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let result = db.query("INVALID SQL SYNTAX !!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_query_cache_hit() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // First query - cache miss
        let _result1 = db.query("SELECT * FROM users").expect("query failed");

        // Check metrics
        let metrics = db.cache_metrics();
        assert_eq!(metrics.misses(), 1);
        assert_eq!(metrics.hits(), 0);

        // Second query - cache hit
        let _result2 = db.query("SELECT * FROM users").expect("query failed");

        assert_eq!(metrics.misses(), 1);
        assert_eq!(metrics.hits(), 1);
    }

    #[test]
    fn test_query_cache_invalidation_on_insert() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Execute initial query
        let _result1 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 1);

        // Second query - cache hit
        let _result2 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().hits(), 1);

        // Insert invalidates the cache
        db.execute("INSERT INTO users (name) VALUES ('Alice')").expect("insert failed");

        // Query again - should be a miss since cache was invalidated
        let _result3 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 2);
    }

    #[test]
    fn test_query_cache_no_cache_hint() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Query with NO_CACHE hint - should not cache
        let _result1 = db.query("/*+ NO_CACHE */ SELECT * FROM users").expect("query failed");

        // Metrics should show no cache activity (hint bypasses cache)
        // Since NO_CACHE skips caching entirely, we won't see a miss recorded

        // Regular query - cache miss
        let _result2 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 1);

        // Same query again - cache hit
        let _result3 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().hits(), 1);
    }

    #[test]
    fn test_query_cache_with_params() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let params1 = &[manifoldb_core::Value::Int(1)];
        let params2 = &[manifoldb_core::Value::Int(2)];

        // Query with param 1
        let _result1 = db
            .query_with_params("SELECT * FROM users WHERE id = $1", params1)
            .expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 1);

        // Same query, same params - cache hit
        let _result2 = db
            .query_with_params("SELECT * FROM users WHERE id = $1", params1)
            .expect("query failed");
        assert_eq!(db.cache_metrics().hits(), 1);

        // Same query, different params - cache miss
        let _result3 = db
            .query_with_params("SELECT * FROM users WHERE id = $1", params2)
            .expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 2);
    }

    #[test]
    fn test_query_cache_clear() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Populate cache
        let _result1 = db.query("SELECT * FROM users").expect("query failed");
        let _result2 = db.query("SELECT * FROM orders").expect("query failed");

        assert_eq!(db.query_cache().len(), 2);

        // Clear cache
        db.clear_cache();

        assert!(db.query_cache().is_empty());

        // Queries should miss again
        let _result3 = db.query("SELECT * FROM users").expect("query failed");
        assert_eq!(db.cache_metrics().misses(), 3);
    }

    #[test]
    fn test_query_cache_disabled() {
        use crate::cache::CacheConfig;

        let db = DatabaseBuilder::in_memory()
            .query_cache_config(CacheConfig::disabled())
            .open()
            .expect("failed to create db");

        // Queries should not be cached
        let _result1 = db.query("SELECT * FROM users").expect("query failed");
        let _result2 = db.query("SELECT * FROM users").expect("query failed");

        // No hits or misses should be recorded for disabled cache
        assert_eq!(db.cache_metrics().hits(), 0);
        // Misses are not recorded for disabled cache
    }

    #[test]
    fn test_query_cache_metrics() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Generate some cache activity
        let _result1 = db.query("SELECT * FROM users").expect("query failed");
        let _result2 = db.query("SELECT * FROM users").expect("query failed");
        let _result3 = db.query("SELECT * FROM orders").expect("query failed");
        let _result4 = db.query("SELECT * FROM users").expect("query failed");

        let metrics = db.cache_metrics();

        assert_eq!(metrics.total_lookups(), 4);
        assert_eq!(metrics.hits(), 2); // users hit twice (after first miss)
        assert_eq!(metrics.misses(), 2); // users first + orders first

        let hit_rate = metrics.hit_rate().expect("should have hit rate");
        assert!((hit_rate - 50.0).abs() < 0.1); // 50% hit rate
    }

    // ========================================================================
    // Bulk Insert Tests
    // ========================================================================

    #[test]
    fn test_bulk_insert_empty() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let entities: Vec<Entity> = Vec::new();
        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        assert!(ids.is_empty());
    }

    #[test]
    fn test_bulk_insert_single_entity() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let entities =
            vec![Entity::new(EntityId::new(0)).with_label("Person").with_property("name", "Alice")];

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        assert_eq!(ids.len(), 1);

        // Verify entity was persisted correctly
        let tx = db.begin_read().expect("failed to begin read");
        let retrieved = tx.get_entity(ids[0]).expect("get failed").expect("entity not found");
        assert!(retrieved.has_label("Person"));
        assert_eq!(
            retrieved.get_property("name"),
            Some(&manifoldb_core::Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_bulk_insert_multiple_entities() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let entities: Vec<Entity> = (0..100)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Document")
                    .with_property("index", i as i64)
            })
            .collect();

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        assert_eq!(ids.len(), 100);

        // Verify IDs are sequential
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(id.as_u64(), (i + 1) as u64);
        }

        // Verify all entities were persisted correctly
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in ids.iter().enumerate() {
            let entity = tx.get_entity(*id).expect("get failed").expect("entity not found");
            assert!(entity.has_label("Document"));
            assert_eq!(entity.get_property("index"), Some(&manifoldb_core::Value::Int(i as i64)));
        }
    }

    #[test]
    fn test_bulk_insert_preserves_existing_id_sequence() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create some entities first using regular inserts
        {
            let mut tx = db.begin().expect("failed to begin");
            for _ in 0..5 {
                let entity = tx.create_entity().expect("failed to create");
                tx.put_entity(&entity).expect("failed to put");
            }
            tx.commit().expect("failed to commit");
        }

        // Now do a bulk insert
        let entities: Vec<Entity> = (0..10)
            .map(|i| Entity::new(EntityId::new(0)).with_property("bulk_index", i as i64))
            .collect();

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        // IDs should start from 6 (after the 5 we created)
        assert_eq!(ids[0].as_u64(), 6);
        assert_eq!(ids[9].as_u64(), 15);
    }

    #[test]
    fn test_bulk_insert_with_multiple_labels() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let entities = vec![Entity::new(EntityId::new(0))
            .with_label("Person")
            .with_label("Employee")
            .with_label("Manager")
            .with_property("name", "Alice")];

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        let tx = db.begin_read().expect("failed to begin read");
        let retrieved = tx.get_entity(ids[0]).expect("get failed").expect("entity not found");

        assert!(retrieved.has_label("Person"));
        assert!(retrieved.has_label("Employee"));
        assert!(retrieved.has_label("Manager"));
    }

    #[test]
    fn test_bulk_insert_large_batch() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Insert 10,000 entities to test performance characteristics
        let entities: Vec<Entity> = (0..10_000)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Item")
                    .with_property("id", i as i64)
                    .with_property("data", format!("item_{}", i))
            })
            .collect();

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        assert_eq!(ids.len(), 10_000);

        // Spot check some entities
        let tx = db.begin_read().expect("failed to begin read");
        for check_idx in [0, 100, 5000, 9999] {
            let entity =
                tx.get_entity(ids[check_idx]).expect("get failed").expect("entity not found");
            assert!(entity.has_label("Item"));
            assert_eq!(
                entity.get_property("id"),
                Some(&manifoldb_core::Value::Int(check_idx as i64))
            );
        }
    }

    #[test]
    fn test_bulk_insert_returns_correct_order() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create entities with unique identifiable properties
        let entities: Vec<Entity> = (0..50)
            .map(|i| Entity::new(EntityId::new(0)).with_property("unique_marker", i * 1000 + 42))
            .collect();

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        // Verify each ID corresponds to the correct entity by checking the marker
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in ids.iter().enumerate() {
            let entity = tx.get_entity(*id).expect("get failed").expect("entity not found");
            let expected_marker = i as i64 * 1000 + 42;
            assert_eq!(
                entity.get_property("unique_marker"),
                Some(&manifoldb_core::Value::Int(expected_marker)),
                "Entity at position {} has wrong marker",
                i
            );
        }
    }

    // ========================================================================
    // Bulk Upsert Tests
    // ========================================================================

    #[test]
    fn test_bulk_upsert_empty() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let entities: Vec<Entity> = Vec::new();
        let (inserted, updated) = db.bulk_upsert_entities(&entities).expect("bulk upsert failed");

        assert_eq!(inserted, 0);
        assert_eq!(updated, 0);
    }

    #[test]
    fn test_bulk_upsert_all_inserts() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // All entities have ID 0, so they're all inserts
        let entities: Vec<Entity> = (0..10)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Document")
                    .with_property("index", i as i64)
            })
            .collect();

        let (inserted, updated) = db.bulk_upsert_entities(&entities).expect("bulk upsert failed");

        assert_eq!(inserted, 10);
        assert_eq!(updated, 0);

        // Verify entities were persisted
        let tx = db.begin_read().expect("failed to begin read");
        for i in 1..=10 {
            let entity =
                tx.get_entity(EntityId::new(i)).expect("get failed").expect("entity not found");
            assert!(entity.has_label("Document"));
            assert_eq!(
                entity.get_property("index"),
                Some(&manifoldb_core::Value::Int((i - 1) as i64))
            );
        }
    }

    #[test]
    fn test_bulk_upsert_all_updates() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // First, insert some entities
        let initial_entities: Vec<Entity> = (0..10)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Document")
                    .with_property("version", 1i64)
                    .with_property("index", i as i64)
            })
            .collect();

        let ids = db.bulk_insert_entities(&initial_entities).expect("bulk insert failed");

        // Now upsert all of them with updated version
        let update_entities: Vec<Entity> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                Entity::new(*id)
                    .with_label("Document")
                    .with_property("version", 2i64)
                    .with_property("index", i as i64)
            })
            .collect();

        let (inserted, updated) =
            db.bulk_upsert_entities(&update_entities).expect("bulk upsert failed");

        assert_eq!(inserted, 0);
        assert_eq!(updated, 10);

        // Verify entities were updated
        let tx = db.begin_read().expect("failed to begin read");
        for id in &ids {
            let entity = tx.get_entity(*id).expect("get failed").expect("entity not found");
            assert_eq!(entity.get_property("version"), Some(&manifoldb_core::Value::Int(2)));
        }
    }

    #[test]
    fn test_bulk_upsert_mixed_insert_and_update() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // First, insert some entities
        let initial_entities: Vec<Entity> = (0..50)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Document")
                    .with_property("version", 1i64)
                    .with_property("original_index", i as i64)
            })
            .collect();

        let ids = db.bulk_insert_entities(&initial_entities).expect("bulk insert failed");

        // Now upsert: update first 30, insert 20 new
        let mut upsert_entities: Vec<Entity> = Vec::new();

        // Update first 30 entities
        for (i, id) in ids.iter().take(30).enumerate() {
            upsert_entities.push(
                Entity::new(*id)
                    .with_label("Document")
                    .with_property("version", 2i64)
                    .with_property("original_index", i as i64),
            );
        }

        // Add 20 new entities (ID 0 = insert)
        for i in 0..20 {
            upsert_entities.push(
                Entity::new(EntityId::new(0))
                    .with_label("Document")
                    .with_property("version", 1i64)
                    .with_property("new_index", i as i64),
            );
        }

        let (inserted, updated) =
            db.bulk_upsert_entities(&upsert_entities).expect("bulk upsert failed");

        assert_eq!(inserted, 20);
        assert_eq!(updated, 30);

        // Verify updates
        let tx = db.begin_read().expect("failed to begin read");
        for id in ids.iter().take(30) {
            let entity = tx.get_entity(*id).expect("get failed").expect("entity not found");
            assert_eq!(entity.get_property("version"), Some(&manifoldb_core::Value::Int(2)));
        }

        // Verify inserts (new entities start after the last ID)
        let expected_start_id = ids.len() as u64 + 1;
        for i in 0..20u64 {
            let entity = tx
                .get_entity(EntityId::new(expected_start_id + i))
                .expect("get failed")
                .expect("entity not found");
            assert_eq!(entity.get_property("version"), Some(&manifoldb_core::Value::Int(1)));
            assert_eq!(
                entity.get_property("new_index"),
                Some(&manifoldb_core::Value::Int(i as i64))
            );
        }
    }

    #[test]
    fn test_bulk_upsert_preserves_labels_on_update() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Insert entity with multiple labels
        let entities = vec![Entity::new(EntityId::new(0))
            .with_label("Person")
            .with_label("Employee")
            .with_property("name", "Alice")];

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        // Upsert with different labels
        let update_entities = vec![Entity::new(ids[0])
            .with_label("Person")
            .with_label("Manager")  // Changed from Employee to Manager
            .with_property("name", "Alice")
            .with_property("promoted", true)];

        let (inserted, updated) =
            db.bulk_upsert_entities(&update_entities).expect("bulk upsert failed");

        assert_eq!(inserted, 0);
        assert_eq!(updated, 1);

        // Verify labels were updated
        let tx = db.begin_read().expect("failed to begin read");
        let entity = tx.get_entity(ids[0]).expect("get failed").expect("entity not found");
        assert!(entity.has_label("Person"));
        assert!(entity.has_label("Manager"));
        assert!(!entity.has_label("Employee")); // Should be removed
        assert_eq!(entity.get_property("promoted"), Some(&manifoldb_core::Value::Bool(true)));
    }

    #[test]
    fn test_bulk_upsert_nonexistent_id_becomes_insert() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Entity with a non-zero ID that doesn't exist in the database
        // should be treated as an insert and get a new ID
        let entities = vec![Entity::new(EntityId::new(999))
            .with_label("Ghost")
            .with_property("name", "Phantom")];

        let (inserted, updated) = db.bulk_upsert_entities(&entities).expect("bulk upsert failed");

        // Since ID 999 doesn't exist, it should be treated as an insert
        assert_eq!(inserted, 1);
        assert_eq!(updated, 0);

        // The entity should have gotten ID 1 (first available)
        let tx = db.begin_read().expect("failed to begin read");
        let entity =
            tx.get_entity(EntityId::new(1)).expect("get failed").expect("entity not found");
        assert!(entity.has_label("Ghost"));
        assert_eq!(
            entity.get_property("name"),
            Some(&manifoldb_core::Value::String("Phantom".to_string()))
        );
    }

    #[test]
    fn test_bulk_upsert_large_batch() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Insert 5000 entities
        let initial_entities: Vec<Entity> = (0..5000)
            .map(|i| {
                Entity::new(EntityId::new(0))
                    .with_label("Item")
                    .with_property("index", i as i64)
                    .with_property("version", 1i64)
            })
            .collect();

        let ids = db.bulk_insert_entities(&initial_entities).expect("bulk insert failed");

        // Upsert: update 2500, insert 2500 new
        let mut upsert_entities: Vec<Entity> = Vec::new();

        // Update first 2500
        for (i, id) in ids.iter().take(2500).enumerate() {
            upsert_entities.push(
                Entity::new(*id)
                    .with_label("Item")
                    .with_property("index", i as i64)
                    .with_property("version", 2i64),
            );
        }

        // Insert 2500 new
        for i in 0..2500 {
            upsert_entities.push(
                Entity::new(EntityId::new(0))
                    .with_label("Item")
                    .with_property("index", (5000 + i) as i64)
                    .with_property("version", 1i64),
            );
        }

        let (inserted, updated) =
            db.bulk_upsert_entities(&upsert_entities).expect("bulk upsert failed");

        assert_eq!(inserted, 2500);
        assert_eq!(updated, 2500);

        // Spot check some entities
        let tx = db.begin_read().expect("failed to begin read");

        // Check an updated entity
        let updated_entity =
            tx.get_entity(ids[100]).expect("get failed").expect("entity not found");
        assert_eq!(updated_entity.get_property("version"), Some(&manifoldb_core::Value::Int(2)));

        // Check an unchanged entity (not in the upsert batch)
        let unchanged_entity =
            tx.get_entity(ids[4000]).expect("get failed").expect("entity not found");
        assert_eq!(unchanged_entity.get_property("version"), Some(&manifoldb_core::Value::Int(1)));
    }

    #[test]
    fn test_bulk_upsert_update_removes_property() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Insert entity with multiple properties
        let entities = vec![Entity::new(EntityId::new(0))
            .with_label("Person")
            .with_property("name", "Alice")
            .with_property("age", 30i64)
            .with_property("email", "alice@example.com")];

        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

        // Upsert without the email property (should remove it)
        let update_entities = vec![Entity::new(ids[0])
            .with_label("Person")
            .with_property("name", "Alice")
            .with_property("age", 31i64)]; // Removed email, updated age

        let (inserted, updated) =
            db.bulk_upsert_entities(&update_entities).expect("bulk upsert failed");

        assert_eq!(inserted, 0);
        assert_eq!(updated, 1);

        // Verify properties
        let tx = db.begin_read().expect("failed to begin read");
        let entity = tx.get_entity(ids[0]).expect("get failed").expect("entity not found");
        assert_eq!(
            entity.get_property("name"),
            Some(&manifoldb_core::Value::String("Alice".to_string()))
        );
        assert_eq!(entity.get_property("age"), Some(&manifoldb_core::Value::Int(31)));
        assert_eq!(entity.get_property("email"), None); // Should be gone
    }

    // ========================================================================
    // Bulk Insert Edge Tests
    // ========================================================================

    #[test]
    fn test_bulk_insert_edges_empty() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        let edges: Vec<Edge> = Vec::new();
        let ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert!(ids.is_empty());
    }

    #[test]
    fn test_bulk_insert_edges_single() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // First create two entities
        let entities = vec![
            Entity::new(EntityId::new(0)).with_label("Person"),
            Entity::new(EntityId::new(0)).with_label("Person"),
        ];
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create one edge between them
        let edges = vec![Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "FOLLOWS")
            .with_property("since", "2024-01-01")];

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 1);

        // Verify edge was persisted correctly
        let tx = db.begin_read().expect("failed to begin read");
        let retrieved = tx.get_edge(edge_ids[0]).expect("get failed").expect("edge not found");
        assert_eq!(retrieved.source, entity_ids[0]);
        assert_eq!(retrieved.target, entity_ids[1]);
        assert_eq!(retrieved.edge_type.as_str(), "FOLLOWS");
        assert_eq!(
            retrieved.get_property("since"),
            Some(&manifoldb_core::Value::String("2024-01-01".to_string()))
        );
    }

    #[test]
    fn test_bulk_insert_edges_multiple() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create some entities
        let entities: Vec<Entity> = (0..10)
            .map(|i| {
                Entity::new(EntityId::new(0)).with_label("Node").with_property("idx", i as i64)
            })
            .collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create edges as a chain: 0->1->2->...->9
        let edges: Vec<Edge> = entity_ids
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                Edge::new(EdgeId::new(0), pair[0], pair[1], "NEXT").with_property("order", i as i64)
            })
            .collect();

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 9); // 9 edges for 10 nodes in a chain

        // Verify IDs are sequential
        for (i, id) in edge_ids.iter().enumerate() {
            assert_eq!(id.as_u64(), (i + 1) as u64);
        }

        // Verify all edges were persisted correctly
        let tx = db.begin_read().expect("failed to begin read");
        for (i, edge_id) in edge_ids.iter().enumerate() {
            let edge = tx.get_edge(*edge_id).expect("get failed").expect("edge not found");
            assert_eq!(edge.source, entity_ids[i]);
            assert_eq!(edge.target, entity_ids[i + 1]);
            assert_eq!(edge.edge_type.as_str(), "NEXT");
            assert_eq!(edge.get_property("order"), Some(&manifoldb_core::Value::Int(i as i64)));
        }
    }

    #[test]
    fn test_bulk_insert_edges_invalid_source() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create one entity
        let entities = vec![Entity::new(EntityId::new(0)).with_label("Person")];
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Try to create an edge with a non-existent source
        let edges = vec![Edge::new(
            EdgeId::new(0),
            EntityId::new(999), // Non-existent source
            entity_ids[0],
            "FOLLOWS",
        )];

        let result = db.bulk_insert_edges(&edges);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::InvalidEntityReference(_)));
    }

    #[test]
    fn test_bulk_insert_edges_invalid_target() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create one entity
        let entities = vec![Entity::new(EntityId::new(0)).with_label("Person")];
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Try to create an edge with a non-existent target
        let edges = vec![Edge::new(
            EdgeId::new(0),
            entity_ids[0],
            EntityId::new(999), // Non-existent target
            "FOLLOWS",
        )];

        let result = db.bulk_insert_edges(&edges);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::InvalidEntityReference(_)));
    }

    #[test]
    fn test_bulk_insert_edges_preserves_id_sequence() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create entities
        let entities: Vec<Entity> =
            (0..5).map(|_| Entity::new(EntityId::new(0)).with_label("Node")).collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create some edges individually first
        {
            let mut tx = db.begin().expect("failed to begin");
            for i in 0..3 {
                let edge = tx
                    .create_edge(entity_ids[i], entity_ids[i + 1], "LINK")
                    .expect("failed to create edge");
                tx.put_edge(&edge).expect("failed to put edge");
            }
            tx.commit().expect("failed to commit");
        }

        // Now do a bulk insert
        let edges: Vec<Edge> =
            vec![Edge::new(EdgeId::new(0), entity_ids[3], entity_ids[4], "LINK")];

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        // IDs should start from 4 (after the 3 we created)
        assert_eq!(edge_ids[0].as_u64(), 4);
    }

    #[test]
    fn test_bulk_insert_edges_with_properties() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create entities
        let entities: Vec<Entity> =
            (0..3).map(|_| Entity::new(EntityId::new(0)).with_label("User")).collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create edges with various properties
        let edges = vec![
            Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "FOLLOWS")
                .with_property("weight", 0.8f64)
                .with_property("since", "2024-01-01")
                .with_property("mutual", true),
            Edge::new(EdgeId::new(0), entity_ids[1], entity_ids[2], "KNOWS")
                .with_property("strength", 5i64)
                .with_property("context", "work"),
        ];

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 2);

        // Verify properties
        let tx = db.begin_read().expect("failed to begin read");

        let edge1 = tx.get_edge(edge_ids[0]).expect("get failed").expect("edge not found");
        assert_eq!(edge1.get_property("weight"), Some(&manifoldb_core::Value::Float(0.8)));
        assert_eq!(
            edge1.get_property("since"),
            Some(&manifoldb_core::Value::String("2024-01-01".to_string()))
        );
        assert_eq!(edge1.get_property("mutual"), Some(&manifoldb_core::Value::Bool(true)));

        let edge2 = tx.get_edge(edge_ids[1]).expect("get failed").expect("edge not found");
        assert_eq!(edge2.get_property("strength"), Some(&manifoldb_core::Value::Int(5)));
        assert_eq!(
            edge2.get_property("context"),
            Some(&manifoldb_core::Value::String("work".to_string()))
        );
    }

    #[test]
    fn test_bulk_insert_edges_self_referential() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create an entity
        let entities = vec![Entity::new(EntityId::new(0)).with_label("Node")];
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create a self-referential edge
        let edges = vec![Edge::new(
            EdgeId::new(0),
            entity_ids[0],
            entity_ids[0], // Self-reference
            "SELF_LINK",
        )];

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 1);

        // Verify the self-referential edge
        let tx = db.begin_read().expect("failed to begin read");
        let edge = tx.get_edge(edge_ids[0]).expect("get failed").expect("edge not found");
        assert_eq!(edge.source, entity_ids[0]);
        assert_eq!(edge.target, entity_ids[0]);
        assert_eq!(edge.edge_type.as_str(), "SELF_LINK");
    }

    #[test]
    fn test_bulk_insert_edges_large_batch() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create 100 entities
        let entities: Vec<Entity> = (0..100)
            .map(|i| {
                Entity::new(EntityId::new(0)).with_label("Node").with_property("idx", i as i64)
            })
            .collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create 1000 random edges
        let edges: Vec<Edge> = (0..1000)
            .map(|i| {
                let source_idx = i % 100;
                let target_idx = (i * 7 + 13) % 100;
                Edge::new(EdgeId::new(0), entity_ids[source_idx], entity_ids[target_idx], "LINK")
                    .with_property("edge_idx", i as i64)
            })
            .collect();

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 1000);

        // Spot check some edges
        let tx = db.begin_read().expect("failed to begin read");
        for check_idx in [0, 100, 500, 999] {
            let edge =
                tx.get_edge(edge_ids[check_idx]).expect("get failed").expect("edge not found");
            assert_eq!(
                edge.get_property("edge_idx"),
                Some(&manifoldb_core::Value::Int(check_idx as i64))
            );
        }
    }

    #[test]
    fn test_bulk_insert_edges_returns_correct_order() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create entities
        let entities: Vec<Entity> =
            (0..10).map(|_| Entity::new(EntityId::new(0)).with_label("Node")).collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create edges with unique markers
        let edges: Vec<Edge> = (0..20)
            .map(|i| {
                let source_idx = i % 10;
                let target_idx = (i + 1) % 10;
                Edge::new(EdgeId::new(0), entity_ids[source_idx], entity_ids[target_idx], "LINK")
                    .with_property("unique_marker", i as i64 * 1000 + 42)
            })
            .collect();

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        // Verify each ID corresponds to the correct edge by checking the marker
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in edge_ids.iter().enumerate() {
            let edge = tx.get_edge(*id).expect("get failed").expect("edge not found");
            let expected_marker = i as i64 * 1000 + 42;
            assert_eq!(
                edge.get_property("unique_marker"),
                Some(&manifoldb_core::Value::Int(expected_marker)),
                "Edge at position {} has wrong marker",
                i
            );
        }
    }

    #[test]
    fn test_bulk_insert_edges_multiple_types() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create entities
        let entities: Vec<Entity> =
            (0..4).map(|_| Entity::new(EntityId::new(0)).with_label("Node")).collect();
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Create edges with different types
        let edges = vec![
            Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "FOLLOWS"),
            Edge::new(EdgeId::new(0), entity_ids[1], entity_ids[2], "LIKES"),
            Edge::new(EdgeId::new(0), entity_ids[2], entity_ids[3], "KNOWS"),
            Edge::new(EdgeId::new(0), entity_ids[3], entity_ids[0], "WORKS_WITH"),
        ];

        let edge_ids = db.bulk_insert_edges(&edges).expect("bulk insert failed");

        assert_eq!(edge_ids.len(), 4);

        // Verify edge types
        let tx = db.begin_read().expect("failed to begin read");
        let edge_types = ["FOLLOWS", "LIKES", "KNOWS", "WORKS_WITH"];
        for (i, edge_id) in edge_ids.iter().enumerate() {
            let edge = tx.get_edge(*edge_id).expect("get failed").expect("edge not found");
            assert_eq!(edge.edge_type.as_str(), edge_types[i]);
        }
    }

    #[test]
    fn test_bulk_insert_edges_all_invalid_rejected() {
        let db = Database::in_memory().expect("failed to create in-memory db");

        // Create one entity
        let entities = vec![Entity::new(EntityId::new(0)).with_label("Node")];
        let entity_ids = db.bulk_insert_entities(&entities).expect("entity insert failed");

        // Try to create a batch where one edge has an invalid reference
        // The entire batch should be rejected
        let edges = vec![
            Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[0], "VALID"),
            Edge::new(
                EdgeId::new(0),
                entity_ids[0],
                EntityId::new(999), // Invalid target
                "INVALID",
            ),
        ];

        let result = db.bulk_insert_edges(&edges);
        assert!(result.is_err());

        // Verify no edges were created
        let tx = db.begin_read().expect("failed to begin read");
        // Try to get edge with ID 1 (which would be the first edge if any were created)
        let edge = tx.get_edge(EdgeId::new(1)).expect("get failed");
        assert!(edge.is_none(), "No edges should have been created");
    }
}
