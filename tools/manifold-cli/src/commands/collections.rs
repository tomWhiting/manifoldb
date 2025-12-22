//! Collections command implementations.

use std::path::Path;

use crate::commands::open_database;
use crate::error::Result;
use crate::output::{format_collections_table, format_info_table, CollectionRow, InfoRow};
use crate::{CollectionsCommands, OutputFormat};

/// Run a collections subcommand.
pub fn run(path: Option<&Path>, cmd: CollectionsCommands, format: OutputFormat) -> Result<()> {
    match cmd {
        CollectionsCommands::List => list(path, format),
        CollectionsCommands::Create { name, config } => create(path, &name, config.as_deref()),
        CollectionsCommands::Drop { name } => drop(path, &name),
        CollectionsCommands::Info { name } => info(path, &name, format),
    }
}

/// List all collections.
fn list(path: Option<&Path>, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    // Query the schema for tables
    // ManifoldDB stores schema info internally, we can query it
    let result = db.query("SELECT table_name FROM information_schema.tables")?;

    let rows: Vec<CollectionRow> = result
        .iter()
        .filter_map(|row| {
            row.get(0).and_then(|v| v.as_str()).map(|name| CollectionRow {
                name: name.to_string(),
                rows: "-".to_string(), // Row count would require a separate query
                columns: "-".to_string(), // Column count would require schema inspection
            })
        })
        .collect();

    match format {
        OutputFormat::Table => {
            println!("{}", format_collections_table(rows));
        }
        OutputFormat::Json => {
            let json: Vec<serde_json::Value> = rows
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "name": r.name,
                        "rows": r.rows,
                        "columns": r.columns
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OutputFormat::Csv | OutputFormat::Compact => {
            for row in rows {
                println!("{}\t{}\t{}", row.name, row.rows, row.columns);
            }
        }
    }

    Ok(())
}

/// Create a new collection.
fn create(path: Option<&Path>, name: &str, _config: Option<&Path>) -> Result<()> {
    let db = open_database(path)?;

    // Create table with a default schema if no config provided
    // In a real implementation, we'd parse the config file for schema definition
    let sql = format!("CREATE TABLE IF NOT EXISTS {name} (id INTEGER PRIMARY KEY)");
    db.execute(&sql)?;

    println!("Created collection: {name}");
    Ok(())
}

/// Drop a collection.
fn drop(path: Option<&Path>, name: &str) -> Result<()> {
    let db = open_database(path)?;

    let sql = format!("DROP TABLE IF EXISTS {name}");
    db.execute(&sql)?;

    println!("Dropped collection: {name}");
    Ok(())
}

/// Show collection info.
fn info(path: Option<&Path>, name: &str, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    // Get column information
    let result = db.query(&format!(
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{name}'"
    ))?;

    let columns: Vec<String> = result
        .iter()
        .filter_map(|row| {
            let col_name = row.get(0).and_then(|v| v.as_str())?;
            let col_type = row.get(1).and_then(|v| v.as_str())?;
            Some(format!("{col_name} ({col_type})"))
        })
        .collect();

    // Get row count
    let count_result = db.query(&format!("SELECT COUNT(*) FROM {name}"))?;
    let row_count =
        count_result.first().and_then(|r| r.get(0)).and_then(|v| v.as_int()).unwrap_or(0);

    let rows = vec![
        InfoRow { key: "Name".to_string(), value: name.to_string() },
        InfoRow { key: "Row Count".to_string(), value: row_count.to_string() },
        InfoRow { key: "Columns".to_string(), value: columns.join(", ") },
    ];

    match format {
        OutputFormat::Table => {
            println!("{}", format_info_table(rows));
        }
        OutputFormat::Json => {
            let map: serde_json::Map<String, serde_json::Value> =
                rows.into_iter().map(|r| (r.key, serde_json::Value::String(r.value))).collect();
            println!("{}", serde_json::to_string_pretty(&map)?);
        }
        OutputFormat::Csv | OutputFormat::Compact => {
            for row in rows {
                println!("{}\t{}", row.key, row.value);
            }
        }
    }

    Ok(())
}
