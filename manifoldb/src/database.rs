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

use manifoldb_storage::backends::redb::{RedbConfig, RedbEngine};

use crate::config::{Config, DatabaseBuilder};
use crate::error::{Error, Result};
use crate::execution::{execute_query, execute_statement};
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

        Ok(Self { manager, config })
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
    /// # Examples
    ///
    /// ```ignore
    /// let affected = db.execute_with_params(
    ///     "INSERT INTO users (name, age) VALUES ($1, $2)",
    ///     &["Alice".into(), 30.into()],
    /// )?;
    /// ```
    pub fn execute_with_params(&self, sql: &str, params: &[manifoldb_core::Value]) -> Result<u64> {
        // Start a write transaction
        let mut tx = self.begin()?;

        // Execute the statement
        let count = execute_statement(&mut tx, sql, params)?;

        // Commit the transaction
        tx.commit().map_err(Error::Transaction)?;

        Ok(count)
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
    /// # Examples
    ///
    /// ```ignore
    /// let results = db.query_with_params(
    ///     "SELECT * FROM users WHERE name = $1",
    ///     &["Alice".into()],
    /// )?;
    /// ```
    pub fn query_with_params(
        &self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<QueryResult> {
        // Start a read transaction
        let tx = self.begin_read()?;

        // Execute the query
        let result_set = execute_query(&tx, sql, params)?;

        // Convert the ResultSet to our QueryResult
        Ok(QueryResult::from_result_set(result_set))
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
        let columns = result_set.columns().to_vec();
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
}
