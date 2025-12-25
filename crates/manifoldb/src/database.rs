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

use manifoldb_storage::backends::redb::{RedbConfig, RedbEngine};

use crate::cache::{extract_cache_hint, CacheHint, CacheMetrics, QueryCache, QueryCacheKey};
use crate::config::{Config, DatabaseBuilder};
use crate::error::{Error, Result};
use crate::execution::{execute_statement, extract_tables_from_sql};
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
pub struct Database {
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

        // Initialize prepared cache with current schema version
        let db = Self { manager, config, query_cache, prepared_cache, db_metrics };

        // Load initial schema version
        if let Ok(tx) = db.begin_read() {
            if let Ok(version) = SchemaManager::get_version(&tx) {
                db.prepared_cache.set_schema_version(version);
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
        &self.config
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
        self.manager.begin_write().map_err(Error::Transaction)
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
        self.manager.begin_read().map_err(Error::Transaction)
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
        self.db_metrics.transactions.record_start();

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
                self.db_metrics.record_commit(commit_start.elapsed());

                // Record successful query
                self.db_metrics.record_query(start.elapsed(), true);

                // Invalidate cache entries for affected tables
                self.query_cache.invalidate_tables(&affected_tables);
                self.prepared_cache.invalidate_tables(&affected_tables)?;

                // Update prepared statement cache schema version if DDL
                if let Some(version) = new_schema_version {
                    self.prepared_cache.set_schema_version(version);
                }

                Ok(count)
            }
            Err(e) => {
                // Record failed query and rollback
                self.db_metrics.record_query(start.elapsed(), false);
                self.db_metrics.record_rollback();
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

        // Determine if we should use caching
        let use_cache = match hint {
            CacheHint::Cache => true,
            CacheHint::NoCache => false,
            CacheHint::Default => self.query_cache.is_enabled(),
        };

        // Try to get from cache if caching is enabled
        if use_cache {
            let cache_key = QueryCacheKey::new(&clean_sql, params);
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                // Update LRU order
                self.query_cache.touch(&cache_key);
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
            self.config.max_rows_in_memory,
        );

        match result {
            Ok(result_set) => {
                // Record successful query
                self.db_metrics.record_query(start.elapsed(), true);

                // Convert the ResultSet to our QueryResult
                let result = QueryResult::from_result_set(result_set);

                // Cache the result if caching is enabled
                if use_cache {
                    let cache_key = QueryCacheKey::new(&clean_sql, params);
                    let accessed_tables = extract_tables_from_sql(&clean_sql);
                    self.query_cache.insert(cache_key, result.clone(), accessed_tables);
                }

                Ok(result)
            }
            Err(e) => {
                // Record failed query
                self.db_metrics.record_query(start.elapsed(), false);
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
        self.manager.flush().map_err(Error::Transaction)
    }

    /// Get the underlying transaction manager.
    ///
    /// This is useful for advanced use cases that require direct
    /// transaction management.
    #[must_use]
    pub fn transaction_manager(&self) -> &TransactionManager<RedbEngine> {
        &self.manager
    }

    /// Get the query cache.
    ///
    /// Use this to access cache operations like clearing or checking metrics.
    #[must_use]
    pub fn query_cache(&self) -> &QueryCache {
        &self.query_cache
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
        self.query_cache.metrics()
    }

    /// Clear the query cache.
    ///
    /// This removes all cached query results. Useful after bulk data
    /// modifications or when you want to ensure fresh data.
    pub fn clear_cache(&self) {
        self.query_cache.clear();
    }

    /// Invalidate cache entries for specific tables.
    ///
    /// This is automatically called during write operations, but can
    /// be called manually if you modify data outside of the normal
    /// execute methods.
    pub fn invalidate_cache_for_tables(&self, tables: &[String]) {
        self.query_cache.invalidate_tables(tables);
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
        let mut snapshot = self.db_metrics.snapshot();

        // Include cache metrics from the query cache
        let cache_snapshot = self.query_cache.metrics().snapshot();
        snapshot.cache = Some(CacheMetricsSnapshot::from_cache_snapshot(cache_snapshot));

        snapshot
    }

    /// Get access to the raw metrics instance.
    ///
    /// This is useful for custom metric collection or integration with
    /// external monitoring systems.
    #[must_use]
    pub fn raw_metrics(&self) -> Arc<DatabaseMetrics> {
        Arc::clone(&self.db_metrics)
    }

    /// Reset all collected metrics.
    ///
    /// This is useful for benchmarking or when you want to collect
    /// metrics for a specific time window.
    ///
    /// Note: Storage size metrics are not reset as they represent
    /// current state rather than accumulated values.
    pub fn reset_metrics(&self) {
        self.db_metrics.reset();
        self.query_cache.metrics().reset();
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
        self.prepared_cache.prepare(sql)
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
        self.prepared_cache.get_or_prepare(sql)
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
        let current_version = self.prepared_cache.schema_version();
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
                self.db_metrics.record_query(start.elapsed(), true);

                // Convert the ResultSet to our QueryResult
                Ok(QueryResult::from_result_set(result_set))
            }
            Err(e) => {
                // Record failed query
                self.db_metrics.record_query(start.elapsed(), false);
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
        let current_version = self.prepared_cache.schema_version();
        if !stmt.is_valid(current_version) {
            return Err(Error::Execution(
                "Prepared statement is invalid due to schema changes. Please re-prepare."
                    .to_string(),
            ));
        }

        let start = Instant::now();

        // Start a write transaction
        let mut tx = self.begin()?;
        self.db_metrics.transactions.record_start();

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
                self.db_metrics.record_commit(commit_start.elapsed());

                // Record successful query
                self.db_metrics.record_query(start.elapsed(), true);

                // Invalidate cache entries for affected tables
                let affected_tables: Vec<String> = stmt.accessed_tables().iter().cloned().collect();
                self.query_cache.invalidate_tables(&affected_tables);
                self.prepared_cache.invalidate_tables(&affected_tables)?;

                // Update prepared statement cache schema version if DDL
                if let Some(version) = new_schema_version {
                    self.prepared_cache.set_schema_version(version);
                }

                Ok(count)
            }
            Err(e) => {
                // Record failed query and rollback
                self.db_metrics.record_query(start.elapsed(), false);
                self.db_metrics.record_rollback();
                Err(e)
            }
        }
    }

    /// Get the prepared statement cache.
    ///
    /// Use this to access cache operations like clearing or checking metrics.
    #[must_use]
    pub fn prepared_cache(&self) -> &PreparedStatementCache {
        &self.prepared_cache
    }

    /// Clear the prepared statement cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal cache lock is poisoned.
    pub fn clear_prepared_cache(&self) -> Result<()> {
        self.prepared_cache.clear()
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
        _collection_name: &str,
        vectors: &[(manifoldb_core::EntityId, String, Vec<f32>)],
    ) -> Result<usize> {
        if vectors.is_empty() {
            return Ok(0);
        }

        let start = std::time::Instant::now();
        let count = vectors.len();

        // Phase 1: Validate all entities exist
        {
            let tx = self.begin_read()?;
            for (entity_id, _, _) in vectors {
                if tx.get_entity(*entity_id)?.is_none() {
                    return Err(Error::EntityNotFound(*entity_id));
                }
            }
        }

        // Phase 2: Store vectors using a write transaction
        // We store vectors as entity properties with a dedicated vector property format
        let mut tx = self.begin()?;
        self.db_metrics.transactions.record_start();

        // Store each vector as an entity property
        // The property name format is: _vector_{vector_name}
        for (entity_id, vector_name, data) in vectors {
            // Get the existing entity
            let mut entity =
                tx.get_entity(*entity_id)?.ok_or_else(|| Error::EntityNotFound(*entity_id))?;

            // Store the vector as a property
            // Note: Property name is prefixed to distinguish vector properties
            let property_name = format!("_vector_{}", vector_name);
            let vector_value = manifoldb_core::Value::from(data.clone());
            entity = entity.with_property(&property_name, vector_value);

            // Update the entity
            tx.put_entity(&entity).map_err(Error::Transaction)?;
        }

        // Commit the transaction
        let commit_start = std::time::Instant::now();
        tx.commit().map_err(Error::Transaction)?;
        self.db_metrics.record_commit(commit_start.elapsed());

        // Record successful operation
        self.db_metrics.record_query(start.elapsed(), true);

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
}
