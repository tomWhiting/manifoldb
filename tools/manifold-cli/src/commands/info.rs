//! Info command implementation.

use std::path::Path;

use crate::commands::open_database;
use crate::error::Result;
use crate::output::{format_info_table, InfoRow};
use crate::OutputFormat;

/// Run the info command - displays database statistics.
pub fn run(path: Option<&Path>, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    // Get metrics
    let metrics = db.metrics();

    // Build info rows
    let mut rows = vec![
        InfoRow { key: "Path".to_string(), value: format!("{}", path.unwrap().display()) },
        InfoRow { key: "In Memory".to_string(), value: db.config().in_memory.to_string() },
    ];

    // Query metrics
    rows.push(InfoRow {
        key: "Total Queries".to_string(),
        value: metrics.queries.total_queries.to_string(),
    });
    let successful = metrics.queries.total_queries - metrics.queries.failed_queries;
    rows.push(InfoRow { key: "Successful Queries".to_string(), value: successful.to_string() });
    rows.push(InfoRow {
        key: "Failed Queries".to_string(),
        value: metrics.queries.failed_queries.to_string(),
    });

    // Transaction metrics
    let total_transactions = metrics.transactions.commits + metrics.transactions.rollbacks;
    rows.push(InfoRow {
        key: "Total Transactions".to_string(),
        value: total_transactions.to_string(),
    });
    rows.push(InfoRow {
        key: "Commits".to_string(),
        value: metrics.transactions.commits.to_string(),
    });
    rows.push(InfoRow {
        key: "Rollbacks".to_string(),
        value: metrics.transactions.rollbacks.to_string(),
    });

    // Cache metrics
    if let Some(cache) = &metrics.cache {
        rows.push(InfoRow { key: "Cache Hits".to_string(), value: cache.hits.to_string() });
        rows.push(InfoRow { key: "Cache Misses".to_string(), value: cache.misses.to_string() });
        if let Some(hit_rate) = cache.hit_rate() {
            rows.push(InfoRow {
                key: "Cache Hit Rate".to_string(),
                value: format!("{hit_rate:.1}%"),
            });
        }
    }

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
