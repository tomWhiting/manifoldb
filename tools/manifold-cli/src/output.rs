//! Output formatting utilities.

use manifoldb::{QueryResult, Value};
use tabled::settings::Style;
use tabled::{Table, Tabled};

use crate::error::Result;
use crate::OutputFormat;

/// Format a query result according to the specified format.
pub fn format_query_result(result: &QueryResult, format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Table => format_as_table(result),
        OutputFormat::Json => format_as_json(result),
        OutputFormat::Csv => format_as_csv(result),
        OutputFormat::Compact => format_as_compact(result),
    }
}

/// Format query result as a pretty table.
fn format_as_table(result: &QueryResult) -> Result<String> {
    if result.is_empty() {
        return Ok("(0 rows)".to_string());
    }

    // Build rows for the table
    let rows: Vec<Vec<String>> =
        result.iter().map(|row| row.values().iter().map(format_value).collect()).collect();

    // Create a dynamic table
    let mut builder = tabled::builder::Builder::new();

    // Add header
    builder.push_record(result.columns().iter().cloned());

    // Add data rows
    for row in rows {
        builder.push_record(row);
    }

    let mut table = builder.build();
    table.with(Style::rounded());

    Ok(format!("{table}\n({} rows)", result.len()))
}

/// Format query result as JSON.
fn format_as_json(result: &QueryResult) -> Result<String> {
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

    Ok(serde_json::to_string_pretty(&rows)?)
}

/// Format query result as CSV.
fn format_as_csv(result: &QueryResult) -> Result<String> {
    let mut wtr = csv::Writer::from_writer(Vec::new());

    // Write header
    wtr.write_record(result.columns())?;

    // Write data rows
    for row in result.iter() {
        let record: Vec<String> = row.values().iter().map(format_value).collect();
        wtr.write_record(&record)?;
    }

    let data = wtr.into_inner().map_err(|e| e.into_error())?;
    Ok(String::from_utf8_lossy(&data).to_string())
}

/// Format query result in a compact single-line format.
fn format_as_compact(result: &QueryResult) -> Result<String> {
    let mut output = String::new();

    for row in result.iter() {
        let values: Vec<String> = row.values().iter().map(format_value).collect();
        output.push_str(&values.join("\t"));
        output.push('\n');
    }

    Ok(output)
}

/// Format a single Value for display.
pub fn format_value(value: &Value) -> String {
    match value {
        Value::Null => "NULL".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => {
            // Use reasonable precision
            if f.fract() == 0.0 {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        Value::String(s) => s.clone(),
        Value::Bytes(b) => format!("\\x{}", hex::encode(b)),
        Value::Vector(v) => {
            if v.len() <= 5 {
                format!("[{}]", v.iter().map(|f| format!("{f:.4}")).collect::<Vec<_>>().join(", "))
            } else {
                format!(
                    "[{}, ... ({} elements)]",
                    v.iter().take(3).map(|f| format!("{f:.4}")).collect::<Vec<_>>().join(", "),
                    v.len()
                )
            }
        }
        Value::SparseVector(sv) => {
            if sv.len() <= 5 {
                format!(
                    "sparse[{}]",
                    sv.iter().map(|(i, v)| format!("{i}:{v:.4}")).collect::<Vec<_>>().join(", ")
                )
            } else {
                format!("sparse[{} non-zero elements]", sv.len())
            }
        }
        Value::MultiVector(mv) => {
            format!("multi[{} vectors]", mv.len())
        }
        Value::Array(arr) => {
            let elements: Vec<String> = arr.iter().map(format_value).collect();
            format!("[{}]", elements.join(", "))
        }
        Value::Point { x, y, z, srid } => match z {
            Some(z_val) => format!("point({{x: {x}, y: {y}, z: {z_val}, srid: {srid}}})"),
            None => format!("point({{x: {x}, y: {y}, srid: {srid}}})"),
        },
    }
}

/// Convert a Value to a JSON value.
pub fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(n) => serde_json::Value::Number((*n).into()),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Bytes(b) => serde_json::Value::String(format!("\\x{}", hex::encode(b))),
        Value::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
        Value::SparseVector(sv) => serde_json::json!(sv
            .iter()
            .map(|(i, v)| serde_json::json!({"index": i, "value": v}))
            .collect::<Vec<_>>()),
        Value::MultiVector(mv) => serde_json::Value::Array(
            mv.iter()
                .map(|v| {
                    serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
                })
                .collect(),
        ),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        Value::Point { x, y, z, srid } => {
            let mut map = serde_json::Map::new();
            map.insert("x".to_string(), serde_json::json!(*x));
            map.insert("y".to_string(), serde_json::json!(*y));
            if let Some(z_val) = z {
                map.insert("z".to_string(), serde_json::json!(*z_val));
            }
            map.insert("srid".to_string(), serde_json::json!(*srid));
            serde_json::Value::Object(map)
        }
    }
}

/// A simple key-value row for displaying info.
#[derive(Tabled)]
pub struct InfoRow {
    #[tabled(rename = "Property")]
    pub key: String,
    #[tabled(rename = "Value")]
    pub value: String,
}

/// Format an info table with key-value pairs.
pub fn format_info_table(rows: Vec<InfoRow>) -> String {
    let mut table = Table::new(rows);
    table.with(Style::rounded());
    table.to_string()
}

/// A collection info row.
#[derive(Tabled)]
pub struct CollectionRow {
    #[tabled(rename = "Name")]
    pub name: String,
    #[tabled(rename = "Rows")]
    pub rows: String,
    #[tabled(rename = "Columns")]
    pub columns: String,
}

/// Format a collections table.
pub fn format_collections_table(rows: Vec<CollectionRow>) -> String {
    if rows.is_empty() {
        return "(no collections)".to_string();
    }
    let mut table = Table::new(rows);
    table.with(Style::rounded());
    table.to_string()
}

/// An index info row.
#[derive(Tabled)]
pub struct IndexRow {
    #[tabled(rename = "Name")]
    pub name: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Columns")]
    pub columns: String,
    #[tabled(rename = "Type")]
    pub index_type: String,
}

/// Format an indexes table.
pub fn format_indexes_table(rows: Vec<IndexRow>) -> String {
    if rows.is_empty() {
        return "(no indexes)".to_string();
    }
    let mut table = Table::new(rows);
    table.with(Style::rounded());
    table.to_string()
}
