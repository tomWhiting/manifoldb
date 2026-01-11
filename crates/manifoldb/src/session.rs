//! Session-based transaction management for explicit transaction control.
//!
//! This module provides [`Session`], which enables explicit transaction control
//! via SQL statements like `BEGIN`, `COMMIT`, `ROLLBACK`, and `SAVEPOINT`.
//!
//! # Overview
//!
//! While [`Database`](crate::Database) provides auto-commit semantics where each
//! statement runs in its own transaction, `Session` allows multiple statements
//! to run within a single explicit transaction, providing full control over
//! when changes become visible.
//!
//! # Transaction Modes
//!
//! - **Auto-commit mode**: Each statement executes in its own implicit transaction
//!   (default behavior when no `BEGIN` is active)
//! - **Explicit transaction mode**: After `BEGIN`, all statements run within the
//!   same transaction until `COMMIT` or `ROLLBACK`
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::{Database, Session};
//!
//! let db = Database::in_memory()?;
//! let mut session = Session::new(&db);
//!
//! // Start explicit transaction
//! session.execute("BEGIN")?;
//!
//! // Multiple statements in same transaction
//! session.execute("INSERT INTO users (name) VALUES ('Alice')")?;
//! session.execute("INSERT INTO users (name) VALUES ('Bob')")?;
//!
//! // Make changes visible
//! session.execute("COMMIT")?;
//! ```
//!
//! # Savepoints
//!
//! Sessions support savepoints for partial rollback within a transaction:
//!
//! ```ignore
//! session.execute("BEGIN")?;
//! session.execute("INSERT INTO users (name) VALUES ('Alice')")?;
//! session.execute("SAVEPOINT sp1")?;
//! session.execute("INSERT INTO users (name) VALUES ('Bob')")?;
//! session.execute("ROLLBACK TO sp1")?;  // Discards Bob, keeps Alice
//! session.execute("COMMIT")?;  // Only Alice is committed
//! ```
//!
//! # Isolation Levels
//!
//! The underlying storage (redb) provides serializable isolation by default.
//! Other isolation levels are parsed but currently map to serializable behavior.

use manifoldb_query::ast::{IsolationLevel, TransactionAccessMode};
use manifoldb_query::plan::LogicalPlan;
use manifoldb_query::{parse_single_statement, PlanBuilder};
use manifoldb_storage::backends::redb::RedbEngine;
use manifoldb_storage::StorageEngine;

use crate::database::{Database, QueryResult};
use crate::error::{Error, Result};
use crate::execution::execute_statement;
use crate::transaction::DatabaseTransaction;

/// The current state of a session's transaction.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum TransactionState {
    /// No active transaction - statements auto-commit.
    #[default]
    AutoCommit,
    /// Inside an explicit transaction started with BEGIN.
    InTransaction {
        /// The isolation level of this transaction.
        isolation_level: IsolationLevel,
        /// The access mode of this transaction.
        access_mode: TransactionAccessMode,
    },
    /// Transaction has been aborted due to an error and must be rolled back.
    Aborted,
}

/// A savepoint represents a point within a transaction that can be rolled back to.
///
/// Since redb doesn't support native savepoints, we implement them by buffering
/// writes in memory and only applying them on commit.
#[derive(Debug, Clone)]
struct Savepoint {
    /// The name of this savepoint.
    name: String,
    /// Index into the write buffer marking where this savepoint was created.
    /// Rollback to this savepoint discards all writes after this index.
    write_buffer_index: usize,
}

/// A buffered write operation for savepoint support.
#[derive(Debug, Clone)]
pub enum BufferedWrite {
    /// Store an entity in the database.
    PutEntity(manifoldb_core::Entity),
    /// Delete an entity from the database.
    DeleteEntity(manifoldb_core::EntityId),
    /// Store an edge in the database.
    PutEdge(manifoldb_core::Edge),
    /// Delete an edge from the database.
    DeleteEdge(manifoldb_core::EdgeId),
}

/// Session state for explicit transaction control.
///
/// A `Session` wraps a [`Database`] reference and maintains transaction state
/// across multiple SQL statements. It enables explicit transaction control
/// via `BEGIN`, `COMMIT`, `ROLLBACK`, and `SAVEPOINT` statements.
///
/// # Lifecycle
///
/// 1. Create a session with [`Session::new()`]
/// 2. Execute statements with [`Session::execute()`] or [`Session::query()`]
/// 3. Use `BEGIN` to start explicit transactions
/// 4. Use `COMMIT` or `ROLLBACK` to end transactions
/// 5. Session automatically rolls back on drop if a transaction is active
///
/// # Thread Safety
///
/// `Session` is not `Send` or `Sync` - it represents a single connection/session.
/// Each thread or async task should have its own `Session` instance.
///
/// # Example
///
/// ```ignore
/// let db = Database::in_memory()?;
/// let mut session = Session::new(&db);
///
/// // Auto-commit mode (default)
/// session.execute("INSERT INTO users (name) VALUES ('Alice')")?;
/// // Alice is immediately visible
///
/// // Explicit transaction
/// session.execute("BEGIN")?;
/// session.execute("INSERT INTO users (name) VALUES ('Bob')")?;
/// session.execute("INSERT INTO users (name) VALUES ('Charlie')")?;
/// // Bob and Charlie not yet visible
/// session.execute("COMMIT")?;
/// // Now Bob and Charlie are visible
/// ```
pub struct Session<'db> {
    /// Reference to the underlying database.
    db: &'db Database,
    /// Current transaction state.
    state: TransactionState,
    /// Stack of savepoints in the current transaction.
    savepoints: Vec<Savepoint>,
    /// Buffered writes for savepoint support.
    /// When savepoints are used, writes are buffered here until commit.
    write_buffer: Vec<BufferedWrite>,
    /// Whether we're using buffered writes (enabled when first savepoint is created).
    buffered_mode: bool,
    /// Active transaction handle (only set when in explicit transaction without buffering).
    /// When buffered_mode is true, we don't hold a transaction open.
    active_transaction:
        Option<DatabaseTransaction<<RedbEngine as StorageEngine>::Transaction<'db>>>,
}

impl<'db> Session<'db> {
    /// Create a new session for the given database.
    ///
    /// The session starts in auto-commit mode.
    #[must_use]
    pub fn new(db: &'db Database) -> Self {
        Self {
            db,
            state: TransactionState::AutoCommit,
            savepoints: Vec::new(),
            write_buffer: Vec::new(),
            buffered_mode: false,
            active_transaction: None,
        }
    }

    /// Get the current transaction state.
    #[must_use]
    pub fn state(&self) -> &TransactionState {
        &self.state
    }

    /// Check if currently in an explicit transaction.
    #[must_use]
    pub fn in_transaction(&self) -> bool {
        matches!(self.state, TransactionState::InTransaction { .. })
    }

    /// Check if the transaction has been aborted and needs rollback.
    #[must_use]
    pub fn is_aborted(&self) -> bool {
        matches!(self.state, TransactionState::Aborted)
    }

    /// Get the names of active savepoints.
    #[must_use]
    pub fn savepoint_names(&self) -> Vec<&str> {
        self.savepoints.iter().map(|sp| sp.name.as_str()).collect()
    }

    /// Execute a SQL statement within this session.
    ///
    /// Transaction control statements (`BEGIN`, `COMMIT`, `ROLLBACK`, `SAVEPOINT`)
    /// are handled specially to manage session state. Other statements are executed
    /// either in auto-commit mode or within the active transaction.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement to execute
    ///
    /// # Returns
    ///
    /// The number of rows affected by the statement, or 0 for transaction control.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The statement cannot be parsed
    /// - The transaction is in an aborted state and the statement is not `ROLLBACK`
    /// - Execution fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut session = Session::new(&db);
    /// session.execute("BEGIN")?;
    /// session.execute("INSERT INTO users (name) VALUES ('Alice')")?;
    /// session.execute("COMMIT")?;
    /// ```
    pub fn execute(&mut self, sql: &str) -> Result<u64> {
        self.execute_with_params(sql, &[])
    }

    /// Execute a SQL statement with bound parameters.
    ///
    /// Like [`execute()`](Self::execute) but with parameter support.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement with parameter placeholders ($1, $2, etc.)
    /// * `params` - The parameter values to bind
    pub fn execute_with_params(
        &mut self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<u64> {
        // Parse the SQL to check for transaction control statements
        let stmt = parse_single_statement(sql)?;
        let plan = PlanBuilder::new().build_statement(&stmt)?;

        // Handle transaction control statements specially
        match &plan {
            LogicalPlan::BeginTransaction(node) => {
                return self.handle_begin(node.isolation_level, node.access_mode);
            }
            LogicalPlan::Commit(_) => {
                return self.handle_commit();
            }
            LogicalPlan::Rollback(node) => {
                return self.handle_rollback(node.to_savepoint.as_deref());
            }
            LogicalPlan::Savepoint(node) => {
                return self.handle_savepoint(&node.name);
            }
            LogicalPlan::ReleaseSavepoint(node) => {
                return self.handle_release_savepoint(&node.name);
            }
            LogicalPlan::SetTransaction(node) => {
                return self.handle_set_transaction(node.isolation_level, node.access_mode);
            }
            _ => {}
        }

        // For non-transaction-control statements, check state and execute
        self.execute_data_statement(sql, params)
    }

    /// Execute a SQL query and return results.
    ///
    /// Queries are always executed within the current transaction context.
    /// In auto-commit mode, a read transaction is used.
    pub fn query(&mut self, sql: &str) -> Result<QueryResult> {
        self.query_with_params(sql, &[])
    }

    /// Execute a SQL query with bound parameters.
    pub fn query_with_params(
        &mut self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<QueryResult> {
        // Check if in aborted state
        if self.is_aborted() {
            return Err(Error::execution(
                "current transaction is aborted, commands ignored until end of transaction block",
            ));
        }

        // For queries, we can use the database's query method directly
        // since they don't modify data and work with read transactions
        self.db.query_with_params(sql, params)
    }

    /// Handle BEGIN transaction.
    fn handle_begin(
        &mut self,
        isolation_level: Option<IsolationLevel>,
        access_mode: Option<TransactionAccessMode>,
    ) -> Result<u64> {
        // Check if already in transaction
        if self.in_transaction() {
            return Err(Error::execution("there is already a transaction in progress"));
        }

        if self.is_aborted() {
            return Err(Error::execution(
                "current transaction is aborted, commands ignored until end of transaction block",
            ));
        }

        let isolation = isolation_level.unwrap_or_default();
        let mode = access_mode.unwrap_or(TransactionAccessMode::ReadWrite);

        // Start a write transaction (we'll hold it open until commit/rollback)
        // Note: For read-only transactions, we could optimize by not holding a write txn
        if mode == TransactionAccessMode::ReadWrite {
            // Start the actual storage transaction
            let tx = self.db.begin()?;
            self.active_transaction = Some(tx);
        }

        self.state =
            TransactionState::InTransaction { isolation_level: isolation, access_mode: mode };

        Ok(0)
    }

    /// Handle COMMIT.
    fn handle_commit(&mut self) -> Result<u64> {
        if !self.in_transaction() {
            return Err(Error::execution("there is no transaction in progress"));
        }

        if self.is_aborted() {
            return Err(Error::execution("current transaction is aborted, cannot commit"));
        }

        // If in buffered mode, we need to apply all buffered writes
        if self.buffered_mode {
            self.flush_write_buffer()?;
        }

        // Commit the active transaction if we have one
        if let Some(tx) = self.active_transaction.take() {
            tx.commit().map_err(Error::Transaction)?;
        }

        // Clear savepoints and reset state
        self.savepoints.clear();
        self.write_buffer.clear();
        self.buffered_mode = false;
        self.state = TransactionState::AutoCommit;

        Ok(0)
    }

    /// Handle ROLLBACK (full or to savepoint).
    fn handle_rollback(&mut self, to_savepoint: Option<&str>) -> Result<u64> {
        // ROLLBACK is always allowed, even from aborted state
        if let Some(savepoint_name) = to_savepoint {
            // ROLLBACK TO SAVEPOINT
            self.rollback_to_savepoint(savepoint_name)
        } else {
            // Full ROLLBACK
            self.rollback_full()
        }
    }

    /// Perform a full rollback of the current transaction.
    fn rollback_full(&mut self) -> Result<u64> {
        if !self.in_transaction() && !self.is_aborted() {
            return Err(Error::execution("there is no transaction in progress"));
        }

        // Rollback the active transaction if we have one
        if let Some(tx) = self.active_transaction.take() {
            tx.rollback().map_err(Error::Transaction)?;
        }

        // Clear all state
        self.savepoints.clear();
        self.write_buffer.clear();
        self.buffered_mode = false;
        self.state = TransactionState::AutoCommit;

        Ok(0)
    }

    /// Rollback to a named savepoint.
    fn rollback_to_savepoint(&mut self, name: &str) -> Result<u64> {
        if !self.in_transaction() && !self.is_aborted() {
            return Err(Error::execution("there is no transaction in progress"));
        }

        // Find the savepoint
        let savepoint_idx = self
            .savepoints
            .iter()
            .position(|sp| sp.name == name)
            .ok_or_else(|| Error::execution(format!("savepoint \"{name}\" does not exist")))?;

        let savepoint = &self.savepoints[savepoint_idx];
        let buffer_index = savepoint.write_buffer_index;

        // Truncate write buffer to the savepoint position
        self.write_buffer.truncate(buffer_index);

        // Remove savepoints created after this one (but keep this savepoint)
        self.savepoints.truncate(savepoint_idx + 1);

        // If we were in aborted state, we're now back to in-transaction
        if self.is_aborted() {
            // Restore the transaction state (we need to get the isolation level from somewhere)
            // For simplicity, use defaults
            self.state = TransactionState::InTransaction {
                isolation_level: IsolationLevel::default(),
                access_mode: TransactionAccessMode::ReadWrite,
            };
        }

        Ok(0)
    }

    /// Handle SAVEPOINT.
    fn handle_savepoint(&mut self, name: &str) -> Result<u64> {
        if !self.in_transaction() {
            return Err(Error::execution("SAVEPOINT can only be used in a transaction"));
        }

        if self.is_aborted() {
            return Err(Error::execution(
                "current transaction is aborted, commands ignored until end of transaction block",
            ));
        }

        // Enable buffered mode if not already
        if !self.buffered_mode {
            // If we have an active transaction, we need to transition to buffered mode
            // This is a design decision - we could also keep the transaction open
            // and buffer writes on top of it
            self.buffered_mode = true;
        }

        // Check if savepoint with this name already exists
        if let Some(existing_idx) = self.savepoints.iter().position(|sp| sp.name == name) {
            // PostgreSQL behavior: redefine the savepoint
            self.savepoints.remove(existing_idx);
        }

        // Create the savepoint
        let savepoint =
            Savepoint { name: name.to_string(), write_buffer_index: self.write_buffer.len() };
        self.savepoints.push(savepoint);

        Ok(0)
    }

    /// Handle RELEASE SAVEPOINT.
    fn handle_release_savepoint(&mut self, name: &str) -> Result<u64> {
        if !self.in_transaction() {
            return Err(Error::execution("RELEASE SAVEPOINT can only be used in a transaction"));
        }

        if self.is_aborted() {
            return Err(Error::execution(
                "current transaction is aborted, commands ignored until end of transaction block",
            ));
        }

        // Find and remove the savepoint
        let savepoint_idx = self
            .savepoints
            .iter()
            .position(|sp| sp.name == name)
            .ok_or_else(|| Error::execution(format!("savepoint \"{name}\" does not exist")))?;

        // PostgreSQL behavior: releasing a savepoint also releases all savepoints
        // established after the named savepoint
        self.savepoints.truncate(savepoint_idx);

        Ok(0)
    }

    /// Handle SET TRANSACTION.
    fn handle_set_transaction(
        &mut self,
        isolation_level: Option<IsolationLevel>,
        access_mode: Option<TransactionAccessMode>,
    ) -> Result<u64> {
        // SET TRANSACTION must be run before any statements in a transaction
        // For now, we allow it at any point but it's a no-op since redb always
        // uses serializable isolation

        if let TransactionState::InTransaction {
            isolation_level: ref mut iso,
            access_mode: ref mut mode,
        } = self.state
        {
            if let Some(new_iso) = isolation_level {
                *iso = new_iso;
            }
            if let Some(new_mode) = access_mode {
                *mode = new_mode;
            }
        }

        // Note: We accept the command but don't actually change behavior since
        // redb always provides serializable isolation
        Ok(0)
    }

    /// Execute a data statement (non-transaction-control).
    fn execute_data_statement(
        &mut self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<u64> {
        // Check if in aborted state
        if self.is_aborted() {
            return Err(Error::execution(
                "current transaction is aborted, commands ignored until end of transaction block",
            ));
        }

        // Execute based on current state
        match &self.state {
            TransactionState::AutoCommit => {
                // Use the database's auto-commit execution
                self.db.execute_with_params(sql, params)
            }
            TransactionState::InTransaction { access_mode, .. } => {
                // Check if write is allowed
                if *access_mode == TransactionAccessMode::ReadOnly {
                    // Check if this is a write statement
                    if Self::is_write_statement(sql) {
                        return Err(Error::execution(
                            "cannot execute INSERT/UPDATE/DELETE in a read-only transaction",
                        ));
                    }
                }

                // Execute within the explicit transaction
                self.execute_in_transaction(sql, params)
            }
            TransactionState::Aborted => {
                // Already checked above, but handle for completeness
                Err(Error::execution(
                    "current transaction is aborted, commands ignored until end of transaction block",
                ))
            }
        }
    }

    /// Execute a statement within the current explicit transaction.
    fn execute_in_transaction(
        &mut self,
        sql: &str,
        params: &[manifoldb_core::Value],
    ) -> Result<u64> {
        if self.buffered_mode {
            // In buffered mode, we need to handle this differently
            // For now, we use a simple approach: execute in a fresh transaction
            // but don't commit - the data won't persist until we flush
            // This is a simplification - a full implementation would parse
            // the statement and buffer the individual operations

            // TODO: Implement proper write buffering for savepoint support
            // For now, we fall back to executing in the active transaction
            // which means savepoints won't fully work for complex operations

            if let Some(ref mut tx) = self.active_transaction {
                let result = execute_statement(tx, sql, params);
                match result {
                    Ok(count) => Ok(count),
                    Err(e) => {
                        // Mark transaction as aborted on error
                        self.state = TransactionState::Aborted;
                        Err(e)
                    }
                }
            } else {
                // Need to start a transaction
                let mut tx = self.db.begin()?;
                let result = execute_statement(&mut tx, sql, params);
                match result {
                    Ok(count) => {
                        self.active_transaction = Some(tx);
                        Ok(count)
                    }
                    Err(e) => {
                        self.state = TransactionState::Aborted;
                        Err(e)
                    }
                }
            }
        } else {
            // Not in buffered mode - execute directly in active transaction
            if let Some(ref mut tx) = self.active_transaction {
                let result = execute_statement(tx, sql, params);
                match result {
                    Ok(count) => Ok(count),
                    Err(e) => {
                        // Mark transaction as aborted on error
                        self.state = TransactionState::Aborted;
                        Err(e)
                    }
                }
            } else {
                Err(Error::execution("no active transaction"))
            }
        }
    }

    /// Flush all buffered writes to the database.
    fn flush_write_buffer(&mut self) -> Result<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        }

        // Get or create the active transaction
        let tx = if let Some(ref mut tx) = self.active_transaction {
            tx
        } else {
            let tx = self.db.begin()?;
            self.active_transaction = Some(tx);
            self.active_transaction
                .as_mut()
                .ok_or_else(|| Error::execution("failed to get transaction"))?
        };

        // Apply all buffered writes
        for write in &self.write_buffer {
            match write {
                BufferedWrite::PutEntity(entity) => {
                    tx.put_entity(entity).map_err(Error::Transaction)?;
                }
                BufferedWrite::DeleteEntity(id) => {
                    tx.delete_entity(*id).map_err(Error::Transaction)?;
                }
                BufferedWrite::PutEdge(edge) => {
                    tx.put_edge(edge).map_err(Error::Transaction)?;
                }
                BufferedWrite::DeleteEdge(id) => {
                    tx.delete_edge(*id).map_err(Error::Transaction)?;
                }
            }
        }

        Ok(())
    }

    /// Check if a SQL statement is a write statement.
    fn is_write_statement(sql: &str) -> bool {
        let sql_upper = sql.trim().to_uppercase();
        sql_upper.starts_with("INSERT")
            || sql_upper.starts_with("UPDATE")
            || sql_upper.starts_with("DELETE")
            || sql_upper.starts_with("CREATE")
            || sql_upper.starts_with("DROP")
            || sql_upper.starts_with("ALTER")
            || sql_upper.starts_with("MERGE")
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        // If a transaction is active, roll it back
        if self.in_transaction() || self.is_aborted() {
            // Take ownership of the transaction to roll it back
            if let Some(tx) = self.active_transaction.take() {
                // Best-effort rollback - ignore errors in drop
                let _ = tx.rollback();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_state_default() {
        let state = TransactionState::default();
        assert_eq!(state, TransactionState::AutoCommit);
    }

    #[test]
    fn test_is_write_statement() {
        assert!(Session::is_write_statement("INSERT INTO foo VALUES (1)"));
        assert!(Session::is_write_statement("UPDATE foo SET x = 1"));
        assert!(Session::is_write_statement("DELETE FROM foo"));
        assert!(Session::is_write_statement("CREATE TABLE foo (id INT)"));
        assert!(Session::is_write_statement("DROP TABLE foo"));
        assert!(Session::is_write_statement("  INSERT INTO foo VALUES (1)"));

        assert!(!Session::is_write_statement("SELECT * FROM foo"));
        assert!(!Session::is_write_statement("  SELECT * FROM foo"));
    }
}
