//! Core storage engine traits.

use super::StorageError;

/// A storage engine that provides key-value operations.
pub trait StorageEngine: Send + Sync {
    /// The transaction type for this engine.
    type Transaction<'a>: Transaction
    where
        Self: 'a;

    /// Begin a read-only transaction.
    fn begin_read(&self) -> Result<Self::Transaction<'_>, StorageError>;

    /// Begin a read-write transaction.
    fn begin_write(&self) -> Result<Self::Transaction<'_>, StorageError>;
}

/// A transaction that provides key-value operations.
pub trait Transaction {
    /// The cursor type for iteration.
    type Cursor<'a>: Cursor
    where
        Self: 'a;

    /// Get a value by key from a table.
    fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, StorageError>;

    /// Put a key-value pair into a table.
    fn put(&mut self, table: &str, key: &[u8], value: &[u8]) -> Result<(), StorageError>;

    /// Delete a key from a table.
    fn delete(&mut self, table: &str, key: &[u8]) -> Result<bool, StorageError>;

    /// Create a cursor for range iteration over a table.
    fn cursor(&self, table: &str) -> Result<Self::Cursor<'_>, StorageError>;

    /// Commit the transaction.
    fn commit(self) -> Result<(), StorageError>;

    /// Rollback the transaction (implicit on drop for uncommitted transactions).
    fn rollback(self) -> Result<(), StorageError>;
}

/// A cursor for iterating over key-value pairs.
pub trait Cursor {
    /// Seek to the first key >= the given key.
    fn seek(&mut self, key: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>, StorageError>;

    /// Move to the next key-value pair.
    fn next(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>, StorageError>;

    /// Get the current key-value pair without advancing.
    fn current(&self) -> Option<(&[u8], &[u8])>;
}
