//! Command implementations.

pub mod collections;
pub mod execute_command;
pub mod graph;
pub mod import_export;
pub mod indexes;
pub mod info;
pub mod open;
pub mod query;

use std::path::Path;

use manifoldb::Database;

use crate::error::{CliError, Result};

/// Open the database at the given path, or return an error if no path is provided.
pub fn open_database(path: Option<&Path>) -> Result<Database> {
    let path = path.ok_or(CliError::NoDatabaseSpecified)?;

    if !path.exists() {
        return Err(CliError::DatabaseNotFound(path.to_path_buf()));
    }

    Ok(Database::open(path)?)
}

/// Open or create the database at the given path.
pub fn open_or_create_database(path: Option<&Path>) -> Result<Database> {
    let path = path.ok_or(CliError::NoDatabaseSpecified)?;
    Ok(Database::open(path)?)
}
