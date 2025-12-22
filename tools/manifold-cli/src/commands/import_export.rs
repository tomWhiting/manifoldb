//! Import/Export command implementations.

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use manifoldb::Value;

use crate::commands::open_or_create_database;
use crate::error::{CliError, Result};
use crate::output::value_to_json;
use crate::{ExportFormat, ImportFormat};

/// Import data from a file into a collection.
pub fn import(
    db_path: Option<&Path>,
    file_path: &Path,
    collection: &str,
    format: ImportFormat,
) -> Result<()> {
    if !file_path.exists() {
        return Err(CliError::FileNotFound(file_path.to_path_buf()));
    }

    let db = open_or_create_database(db_path)?;

    match format {
        ImportFormat::Json => import_json(&db, file_path, collection),
        ImportFormat::Csv => import_csv(&db, file_path, collection),
    }
}

/// Import JSON data.
fn import_json(db: &manifoldb::Database, file_path: &Path, collection: &str) -> Result<()> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let data: Vec<serde_json::Value> = serde_json::from_reader(reader)?;

    if data.is_empty() {
        println!("No data to import");
        return Ok(());
    }

    // Get column names from the first object
    let columns: Vec<String> = if let Some(first) = data.first() {
        if let Some(obj) = first.as_object() {
            obj.keys().cloned().collect()
        } else {
            return Err(CliError::InvalidInput("Expected array of objects".to_string()));
        }
    } else {
        return Err(CliError::InvalidInput("Empty array".to_string()));
    };

    // Check if table exists, create if not
    let table_exists = db.query(&format!("SELECT 1 FROM {collection} LIMIT 1")).is_ok();
    if !table_exists {
        // Create table with columns (all as TEXT for simplicity)
        let col_defs = columns.iter().map(|c| format!("{c} TEXT")).collect::<Vec<_>>().join(", ");
        db.execute(&format!("CREATE TABLE {collection} ({col_defs})"))?;
    }

    // Import rows
    let mut imported = 0;
    for row in &data {
        if let Some(obj) = row.as_object() {
            let values: Vec<String> = columns
                .iter()
                .map(|col| {
                    obj.get(col)
                        .map(|v| match v {
                            serde_json::Value::Null => "NULL".to_string(),
                            serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => format!("'{}'", v.to_string().replace('\'', "''")),
                        })
                        .unwrap_or_else(|| "NULL".to_string())
                })
                .collect();

            let sql = format!(
                "INSERT INTO {collection} ({}) VALUES ({})",
                columns.join(", "),
                values.join(", ")
            );
            db.execute(&sql)?;
            imported += 1;
        }
    }

    println!("Imported {imported} row(s) into {collection}");
    Ok(())
}

/// Import CSV data.
fn import_csv(db: &manifoldb::Database, file_path: &Path, collection: &str) -> Result<()> {
    let file = File::open(file_path)?;
    let mut reader = csv::Reader::from_reader(file);

    let headers: Vec<String> = reader.headers()?.iter().map(String::from).collect();

    if headers.is_empty() {
        println!("No data to import");
        return Ok(());
    }

    // Check if table exists, create if not
    let table_exists = db.query(&format!("SELECT 1 FROM {collection} LIMIT 1")).is_ok();
    if !table_exists {
        let col_defs = headers.iter().map(|c| format!("{c} TEXT")).collect::<Vec<_>>().join(", ");
        db.execute(&format!("CREATE TABLE {collection} ({col_defs})"))?;
    }

    // Import rows
    let mut imported = 0;
    for result in reader.records() {
        let record = result?;
        let values: Vec<String> =
            record.iter().map(|v| format!("'{}'", v.replace('\'', "''"))).collect();

        let sql = format!(
            "INSERT INTO {collection} ({}) VALUES ({})",
            headers.join(", "),
            values.join(", ")
        );
        db.execute(&sql)?;
        imported += 1;
    }

    println!("Imported {imported} row(s) into {collection}");
    Ok(())
}

/// Export data from a collection to a file.
pub fn export(
    db_path: Option<&Path>,
    collection: &str,
    output_path: &Path,
    format: ExportFormat,
) -> Result<()> {
    let db = crate::commands::open_database(db_path)?;

    // Query all data from the collection
    let result = db.query(&format!("SELECT * FROM {collection}"))?;

    match format {
        ExportFormat::Json => export_json(&result, output_path),
        ExportFormat::Csv => export_csv(&result, output_path),
    }
}

/// Export to JSON.
fn export_json(result: &manifoldb::QueryResult, output_path: &Path) -> Result<()> {
    let rows: Vec<serde_json::Value> = result
        .iter()
        .map(|row| {
            let obj: serde_json::Map<String, serde_json::Value> = result
                .columns()
                .iter()
                .zip(row.values().iter())
                .map(|(col, val)| (col.clone(), value_to_json(val)))
                .collect();
            serde_json::Value::Object(obj)
        })
        .collect();

    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &rows)?;
    writer.flush()?;

    println!("Exported {} row(s) to {}", result.len(), output_path.display());
    Ok(())
}

/// Export to CSV.
fn export_csv(result: &manifoldb::QueryResult, output_path: &Path) -> Result<()> {
    let file = File::create(output_path)?;
    let mut writer = csv::Writer::from_writer(file);

    // Write header
    writer.write_record(result.columns())?;

    // Write data rows
    for row in result.iter() {
        let record: Vec<String> = row.values().iter().map(value_to_string).collect();
        writer.write_record(&record)?;
    }

    writer.flush()?;

    println!("Exported {} row(s) to {}", result.len(), output_path.display());
    Ok(())
}

/// Convert a Value to a string for CSV export.
fn value_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.clone(),
        Value::Bytes(b) => hex::encode(b),
        Value::Vector(v) => {
            format!("[{}]", v.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(","))
        }
        Value::SparseVector(sv) => {
            format!(
                "[{}]",
                sv.iter().map(|(i, v)| format!("{i}:{v}")).collect::<Vec<_>>().join(",")
            )
        }
        Value::MultiVector(mv) => format!("[{} vectors]", mv.len()),
        Value::Array(arr) => {
            format!("[{}]", arr.iter().map(value_to_string).collect::<Vec<_>>().join(","))
        }
    }
}
