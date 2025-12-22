//! Open command implementation.

use std::path::Path;

use manifoldb::Database;

use crate::error::Result;

/// Run the open command - validates that a database can be opened/created at the given path.
pub fn run(path: &Path) -> Result<()> {
    // Try to open or create the database
    let _db = Database::open(path)?;

    if path.exists() {
        println!("Opened database: {}", path.display());
    } else {
        println!("Created database: {}", path.display());
    }

    Ok(())
}
