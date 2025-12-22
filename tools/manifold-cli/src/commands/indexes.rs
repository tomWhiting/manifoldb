//! Indexes command implementations.

use std::path::Path;

use crate::commands::open_database;
use crate::error::Result;
use crate::output::{format_indexes_table, IndexRow};
use crate::{IndexesCommands, OutputFormat};

/// Run an indexes subcommand.
pub fn run(path: Option<&Path>, cmd: IndexesCommands, format: OutputFormat) -> Result<()> {
    match cmd {
        IndexesCommands::List { collection } => list(path, collection.as_deref(), format),
        IndexesCommands::Rebuild { name } => rebuild(path, &name),
    }
}

/// List all indexes.
fn list(path: Option<&Path>, collection: Option<&str>, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    // Query for indexes - this uses the internal schema representation
    let sql = if let Some(table) = collection {
        format!(
            "SELECT index_name, table_name, column_name, index_type \
             FROM information_schema.indexes \
             WHERE table_name = '{table}'"
        )
    } else {
        "SELECT index_name, table_name, column_name, index_type \
         FROM information_schema.indexes"
            .to_string()
    };

    let result = db.query(&sql)?;

    let rows: Vec<IndexRow> = result
        .iter()
        .filter_map(|row| {
            Some(IndexRow {
                name: row.get(0).and_then(|v| v.as_str())?.to_string(),
                collection: row.get(1).and_then(|v| v.as_str())?.to_string(),
                columns: row.get(2).and_then(|v| v.as_str())?.to_string(),
                index_type: row.get(3).and_then(|v| v.as_str()).unwrap_or("btree").to_string(),
            })
        })
        .collect();

    match format {
        OutputFormat::Table => {
            println!("{}", format_indexes_table(rows));
        }
        OutputFormat::Json => {
            let json: Vec<serde_json::Value> = rows
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "name": r.name,
                        "collection": r.collection,
                        "columns": r.columns,
                        "type": r.index_type
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OutputFormat::Csv | OutputFormat::Compact => {
            for row in rows {
                println!("{}\t{}\t{}\t{}", row.name, row.collection, row.columns, row.index_type);
            }
        }
    }

    Ok(())
}

/// Rebuild an index.
fn rebuild(path: Option<&Path>, name: &str) -> Result<()> {
    let db = open_database(path)?;

    // REINDEX command to rebuild
    let sql = format!("REINDEX {name}");
    db.execute(&sql)?;

    println!("Rebuilt index: {name}");
    Ok(())
}
