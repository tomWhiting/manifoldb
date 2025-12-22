//! Query command implementation.

use std::path::Path;
use std::time::Instant;

use crate::commands::open_database;
use crate::error::Result;
use crate::output::format_query_result;
use crate::OutputFormat;

/// Run a SQL query and display results.
pub fn run(path: Option<&Path>, sql: &str, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    let start = Instant::now();
    let result = db.query(sql)?;
    let elapsed = start.elapsed();

    let output = format_query_result(&result, format)?;
    println!("{output}");

    // Show timing for table format
    if format == OutputFormat::Table {
        println!("Time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    }

    Ok(())
}
