//! Main database interface.

use std::path::Path;

use crate::error::Error;

/// The main `ManifoldDB` database handle.
///
/// This is the primary entry point for interacting with a `ManifoldDB` database.
pub struct Database {
    // TODO: Add internal state
    _private: (),
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open(_path: impl AsRef<Path>) -> Result<Self, Error> {
        // TODO: Implement
        Ok(Self { _private: () })
    }

    /// Execute a statement that doesn't return results.
    pub const fn execute(&self, _sql: &str) -> Result<(), Error> {
        // TODO: Implement
        Ok(())
    }

    /// Execute a query and return results.
    pub const fn query(&self, _sql: &str) -> Result<QueryResult, Error> {
        // TODO: Implement
        Ok(QueryResult { _private: () })
    }
}

/// The result of a query execution.
pub struct QueryResult {
    _private: (),
}
