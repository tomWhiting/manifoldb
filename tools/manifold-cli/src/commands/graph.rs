//! Graph command implementations.

use std::path::Path;

use manifoldb::EntityId;

use crate::commands::open_database;
use crate::error::Result;
use crate::output::{format_info_table, format_value, InfoRow};
use crate::{GraphCommands, OutputFormat, TraversalDirection};

/// Run a graph subcommand.
pub fn run(path: Option<&Path>, cmd: GraphCommands, format: OutputFormat) -> Result<()> {
    match cmd {
        GraphCommands::Stats => stats(path, format),
        GraphCommands::Traverse { from, depth, edge_type, direction } => {
            traverse(path, from, depth, edge_type.as_deref(), direction, format)
        }
    }
}

/// Show graph statistics.
fn stats(path: Option<&Path>, format: OutputFormat) -> Result<()> {
    let db = open_database(path)?;

    // Start a read transaction to count entities and edges
    let tx = db.begin_read()?;

    // Count entities
    let entity_count = tx.count_entities(None)?;

    // For edge count, we'd need to iterate
    // This is a simplified version - a real implementation might have a dedicated counter
    let mut edge_count = 0u64;
    let mut label_counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    for entity in tx.iter_entities(None)? {
        for label in &entity.labels {
            *label_counts.entry(label.as_str().to_string()).or_insert(0) += 1;
        }
        edge_count += tx.get_outgoing_edges(entity.id)?.len() as u64;
    }

    let rows = vec![
        InfoRow { key: "Total Entities".to_string(), value: entity_count.to_string() },
        InfoRow { key: "Total Edges".to_string(), value: edge_count.to_string() },
        InfoRow {
            key: "Labels".to_string(),
            value: if label_counts.is_empty() {
                "(none)".to_string()
            } else {
                label_counts
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            },
        },
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

/// Traverse the graph from a starting node.
fn traverse(
    path: Option<&Path>,
    from: u64,
    max_depth: u32,
    edge_type_filter: Option<&str>,
    direction: TraversalDirection,
    format: OutputFormat,
) -> Result<()> {
    let db = open_database(path)?;
    let tx = db.begin_read()?;

    let start_id = EntityId::new(from);

    // Check if start entity exists
    let start_entity = tx.get_entity(start_id)?.ok_or_else(|| {
        crate::error::CliError::InvalidInput(format!("Entity with ID {} not found", from))
    })?;

    println!("Starting from entity {from}:");
    print_entity_summary(&start_entity);

    // BFS traversal
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    visited.insert(start_id);
    queue.push_back((start_id, 0u32));

    let mut results: Vec<TraversalResult> = Vec::new();

    while let Some((current_id, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        // Get edges based on direction
        let edges = match direction {
            TraversalDirection::Outgoing => tx.get_outgoing_edges(current_id)?,
            TraversalDirection::Incoming => tx.get_incoming_edges(current_id)?,
            TraversalDirection::Both => {
                let mut all = tx.get_outgoing_edges(current_id)?;
                all.extend(tx.get_incoming_edges(current_id)?);
                all
            }
        };

        for edge in edges {
            // Filter by edge type if specified
            if let Some(filter) = edge_type_filter {
                if edge.edge_type.as_str() != filter {
                    continue;
                }
            }

            let target_id = if direction == TraversalDirection::Incoming {
                edge.source
            } else {
                edge.target
            };

            if visited.insert(target_id) {
                queue.push_back((target_id, depth + 1));

                if let Some(target) = tx.get_entity(target_id)? {
                    results.push(TraversalResult {
                        depth: depth + 1,
                        from_id: current_id.as_u64(),
                        edge_type: edge.edge_type.as_str().to_string(),
                        to_id: target_id.as_u64(),
                        labels: target.labels.iter().map(|l| l.as_str().to_string()).collect(),
                    });
                }
            }
        }
    }

    // Output results
    match format {
        OutputFormat::Table => {
            println!("\nTraversal results ({} nodes reached):", results.len());
            for result in &results {
                let indent = "  ".repeat(result.depth as usize);
                println!(
                    "{indent}[{}] --{}-> [{}] ({})",
                    result.from_id,
                    result.edge_type,
                    result.to_id,
                    result.labels.join(", ")
                );
            }
        }
        OutputFormat::Json => {
            let json: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| {
                    serde_json::json!({
                        "depth": r.depth,
                        "from": r.from_id,
                        "edge_type": r.edge_type,
                        "to": r.to_id,
                        "labels": r.labels
                    })
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        OutputFormat::Csv | OutputFormat::Compact => {
            for result in results {
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    result.depth,
                    result.from_id,
                    result.edge_type,
                    result.to_id,
                    result.labels.join(",")
                );
            }
        }
    }

    Ok(())
}

/// Helper struct for traversal results.
struct TraversalResult {
    depth: u32,
    from_id: u64,
    edge_type: String,
    to_id: u64,
    labels: Vec<String>,
}

/// Print a summary of an entity.
fn print_entity_summary(entity: &manifoldb::Entity) {
    let labels: Vec<&str> = entity.labels.iter().map(|l| l.as_str()).collect();
    let labels_str = labels.join(", ");
    println!("  Labels: {}", if labels_str.is_empty() { "(none)" } else { &labels_str });

    for (key, value) in &entity.properties {
        println!("  {key}: {}", format_value(value));
    }
}
