//! Core storage engine traits.
//!
//! This module defines the fundamental traits for storage backends:
//!
//! - [`StorageEngine`] - The main entry point for storage operations
//! - [`Transaction`] - ACID transaction support with read/write operations
//! - [`Cursor`] - Ordered iteration over key-value pairs
//!
//! All traits are designed to be object-safe where possible and use associated
//! types for maximum flexibility in backend implementations.

use std::ops::Bound;
use std::sync::Arc;

use super::StorageError;

/// A key-value pair returned by cursor operations.
pub type KeyValue = (Vec<u8>, Vec<u8>);

/// Result type for cursor operations that return a key-value pair.
pub type CursorResult = Result<Option<KeyValue>, StorageError>;

/// A storage engine that provides transactional key-value operations.
///
/// Storage engines are the foundation of the database, providing durable
/// storage with ACID transaction support. Implementations must be thread-safe
/// (`Send + Sync`).
///
/// # Example
///
/// ```ignore
/// use manifoldb_storage::{StorageEngine, Transaction};
///
/// fn example<E: StorageEngine>(engine: &E) -> Result<(), StorageError> {
///     // Read transaction
///     let tx = engine.begin_read()?;
///     let value = tx.get("my_table", b"key")?;
///
///     // Write transaction
///     let mut tx = engine.begin_write()?;
///     tx.put("my_table", b"key", b"value")?;
///     tx.commit()?;
///     Ok(())
/// }
/// ```
pub trait StorageEngine: Send + Sync {
    /// The transaction type for this engine.
    type Transaction<'a>: Transaction
    where
        Self: 'a;

    /// Begin a read-only transaction.
    ///
    /// Read transactions provide a consistent snapshot of the database.
    /// Multiple read transactions can run concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Transaction`] if the transaction cannot be started.
    fn begin_read(&self) -> Result<Self::Transaction<'_>, StorageError>;

    /// Begin a read-write transaction.
    ///
    /// Write transactions allow modifying the database. Depending on the
    /// backend, write transactions may be serialized.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Transaction`] if the transaction cannot be started.
    fn begin_write(&self) -> Result<Self::Transaction<'_>, StorageError>;

    /// Flush any buffered data to durable storage.
    ///
    /// This is typically called after committing important transactions to
    /// ensure durability. The default implementation does nothing, as most
    /// backends handle durability on commit.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Io`] if the flush fails.
    fn flush(&self) -> Result<(), StorageError> {
        Ok(())
    }
}

/// A transaction that provides ACID key-value operations.
///
/// Transactions provide isolation from concurrent operations and atomicity
/// for batched changes. Write transactions must be explicitly committed;
/// dropping without committing will roll back changes.
///
/// # ACID Properties
///
/// - **Atomicity**: All operations in a transaction succeed or fail together
/// - **Consistency**: The database moves from one valid state to another
/// - **Isolation**: Transactions don't see uncommitted changes from others
/// - **Durability**: Committed changes survive system failures
pub trait Transaction {
    /// The cursor type for iteration.
    type Cursor<'a>: Cursor
    where
        Self: 'a;

    /// Get a value by key from a table.
    ///
    /// # Arguments
    ///
    /// * `table` - The table name to read from
    /// * `key` - The key to look up
    ///
    /// # Returns
    ///
    /// Returns `Ok(Some(value))` if the key exists, `Ok(None)` if it doesn't.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::TableNotFound`] if the table doesn't exist.
    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError>;

    /// Put a key-value pair into a table.
    ///
    /// If the key already exists, its value is replaced. If the table doesn't
    /// exist, it will be created (depending on backend configuration).
    ///
    /// # Arguments
    ///
    /// * `table` - The table name to write to
    /// * `key` - The key to insert or update
    /// * `value` - The value to store
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails or if this is a read-only transaction.
    fn put(&mut self, table: &str, key: &[u8], value: &[u8]) -> Result<(), StorageError>;

    /// Delete a key from a table.
    ///
    /// # Arguments
    ///
    /// * `table` - The table name to delete from
    /// * `key` - The key to delete
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the key was deleted, `Ok(false)` if it didn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails or if this is a read-only transaction.
    fn delete(&mut self, table: &str, key: &[u8]) -> Result<bool, StorageError>;

    /// Create a cursor for iterating over all key-value pairs in a table.
    ///
    /// The cursor starts before the first key and must be advanced with
    /// [`Cursor::next`] or positioned with [`Cursor::seek`].
    ///
    /// # Arguments
    ///
    /// * `table` - The table name to iterate over
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::TableNotFound`] if the table doesn't exist.
    fn cursor(&self, table: &str) -> Result<Self::Cursor<'_>, StorageError>;

    /// Create a cursor for iterating over a range of keys in a table.
    ///
    /// # Arguments
    ///
    /// * `table` - The table name to iterate over
    /// * `start` - The lower bound of the range
    /// * `end` - The upper bound of the range
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::ops::Bound;
    ///
    /// // Scan keys from "a" (inclusive) to "z" (exclusive)
    /// let cursor = tx.range(
    ///     "my_table",
    ///     Bound::Included(b"a".as_slice()),
    ///     Bound::Excluded(b"z".as_slice()),
    /// )?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::TableNotFound`] if the table doesn't exist.
    fn range(
        &self,
        table: &str,
        start: Bound<&[u8]>,
        end: Bound<&[u8]>,
    ) -> Result<Self::Cursor<'_>, StorageError>;

    /// Commit the transaction, making all changes durable.
    ///
    /// After commit, the transaction is consumed and cannot be used further.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Transaction`] if the commit fails.
    fn commit(self) -> Result<(), StorageError>;

    /// Rollback the transaction, discarding all changes.
    ///
    /// This is typically implicit when a transaction is dropped without
    /// committing, but can be called explicitly for clarity.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::Transaction`] if the rollback fails.
    fn rollback(self) -> Result<(), StorageError>;

    /// Check if this is a read-only transaction.
    ///
    /// Read-only transactions will return errors on write operations.
    fn is_read_only(&self) -> bool;
}

/// A cursor for ordered iteration over key-value pairs.
///
/// Cursors provide efficient sequential access to data in key order.
/// They can be positioned at a specific key and iterated forward.
///
/// # Iteration Pattern
///
/// ```ignore
/// let mut cursor = tx.cursor("my_table")?;
///
/// // Position at first key >= "prefix"
/// cursor.seek(b"prefix")?;
///
/// // Iterate through matching keys
/// while let Some((key, value)) = cursor.next()? {
///     // Process key-value pair
/// }
/// ```
pub trait Cursor {
    /// Seek to the first key greater than or equal to the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to seek to
    ///
    /// # Returns
    ///
    /// Returns the key-value pair at the position, or `None` if no key >= `key` exists.
    fn seek(&mut self, key: &[u8]) -> CursorResult;

    /// Seek to the first key-value pair.
    ///
    /// # Returns
    ///
    /// Returns the first key-value pair, or `None` if the table is empty.
    fn seek_first(&mut self) -> CursorResult;

    /// Seek to the last key-value pair.
    ///
    /// # Returns
    ///
    /// Returns the last key-value pair, or `None` if the table is empty.
    fn seek_last(&mut self) -> CursorResult;

    /// Move to the next key-value pair.
    ///
    /// # Returns
    ///
    /// Returns the next key-value pair, or `None` if at the end.
    fn next(&mut self) -> CursorResult;

    /// Move to the previous key-value pair.
    ///
    /// # Returns
    ///
    /// Returns the previous key-value pair, or `None` if at the beginning.
    fn prev(&mut self) -> CursorResult;

    /// Get the current key-value pair without advancing.
    ///
    /// Returns `None` if the cursor is not positioned at a valid entry
    /// (before first call to seek/next, or after iteration is exhausted).
    fn current(&self) -> Option<(&[u8], &[u8])>;
}

// ============================================================================
// Blanket Implementations
// ============================================================================

/// Implement `StorageEngine` for `Arc<E>` to allow shared ownership of engines.
///
/// This is useful when multiple components need access to the same engine,
/// such as when creating collection handles from a database.
impl<E: StorageEngine> StorageEngine for Arc<E> {
    type Transaction<'a>
        = E::Transaction<'a>
    where
        Self: 'a;

    fn begin_read(&self) -> Result<Self::Transaction<'_>, StorageError> {
        (**self).begin_read()
    }

    fn begin_write(&self) -> Result<Self::Transaction<'_>, StorageError> {
        (**self).begin_write()
    }

    fn flush(&self) -> Result<(), StorageError> {
        (**self).flush()
    }
}
