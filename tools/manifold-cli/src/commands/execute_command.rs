//! Execute command implementation.

use std::path::Path;
use std::time::Instant;

use crate::commands::open_database;
use crate::error::Result;

/// Execute a SQL statement (INSERT, UPDATE, DELETE, DDL).
pub fn run(path: Option<&Path>, sql: &str) -> Result<()> {
    let db = open_database(path)?;

    let start = Instant::now();
    let affected = db.execute(sql)?;
    let elapsed = start.elapsed();

    println!("{affected} row(s) affected");
    println!("Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}
