//! Type conversions from ManifoldDB to GraphQL types.
//!
//! This module handles the mapping between ManifoldDB's internal types
//! and the GraphQL schema types.

use async_graphql::{Json, Result, ID};
use manifoldb::{QueryResult as DbQueryResult, Value};
use serde_json::json;
use std::collections::HashMap;

use crate::schema::{Edge, GraphResult, Node, TableResult};

/// Convert a ManifoldDB Value to a serde_json Value.
pub fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => json!(null),
        Value::Bool(b) => json!(b),
        Value::Int(i) => json!(i),
        Value::Float(f) => json!(f),
        Value::String(s) => json!(s),
        Value::Bytes(b) => {
            // Encode bytes as base64
            use base64::Engine;
            json!(base64::engine::general_purpose::STANDARD.encode(b))
        }
        Value::Vector(v) => json!(v),
        Value::SparseVector(pairs) => {
            let indices: Vec<u32> = pairs.iter().map(|(i, _)| *i).collect();
            let values: Vec<f32> = pairs.iter().map(|(_, v)| *v).collect();
            json!({ "indices": indices, "values": values })
        }
        Value::MultiVector(vecs) => json!(vecs),
        Value::Array(items) => json!(items.iter().map(value_to_json).collect::<Vec<_>>()),
        Value::Point { x, y, z, srid } => {
            let mut obj = serde_json::Map::new();
            obj.insert("x".to_string(), json!(x));
            obj.insert("y".to_string(), json!(y));
            if let Some(z_val) = z {
                obj.insert("z".to_string(), json!(z_val));
            }
            obj.insert("srid".to_string(), json!(srid));
            serde_json::Value::Object(obj)
        }
        Value::Node { id, labels, properties } => {
            let mut obj = serde_json::Map::new();
            obj.insert("id".to_string(), json!(id));
            obj.insert("labels".to_string(), json!(labels));
            let props = hashmap_to_json(properties);
            obj.insert("properties".to_string(), props);
            serde_json::Value::Object(obj)
        }
        Value::Edge { id, edge_type, source, target, properties } => {
            let mut obj = serde_json::Map::new();
            obj.insert("id".to_string(), json!(id));
            obj.insert("type".to_string(), json!(edge_type));
            obj.insert("source".to_string(), json!(source));
            obj.insert("target".to_string(), json!(target));
            let props = hashmap_to_json(properties);
            obj.insert("properties".to_string(), props);
            serde_json::Value::Object(obj)
        }
    }
}

/// Convert a HashMap of properties to JSON.
fn hashmap_to_json(map: &HashMap<String, Value>) -> serde_json::Value {
    let obj: serde_json::Map<String, serde_json::Value> =
        map.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect();
    serde_json::Value::Object(obj)
}

/// Convert a Value::Node to a GraphQL Node.
fn value_node_to_graphql(id: i64, labels: &[String], properties: &HashMap<String, Value>) -> Node {
    Node {
        id: ID(id.to_string()),
        labels: labels.to_vec(),
        properties: Json(hashmap_to_json(properties)),
    }
}

/// Convert a Value::Edge to a GraphQL Edge.
fn value_edge_to_graphql(
    id: i64,
    edge_type: &str,
    source: i64,
    target: i64,
    properties: &HashMap<String, Value>,
) -> Edge {
    Edge {
        id: ID(id.to_string()),
        edge_type: edge_type.to_string(),
        source: ID(source.to_string()),
        target: ID(target.to_string()),
        properties: Json(hashmap_to_json(properties)),
    }
}

/// Convert a ManifoldDB QueryResult to a GraphQL TableResult.
pub fn query_result_to_table(result: &DbQueryResult) -> TableResult {
    let columns: Vec<String> = result.columns().to_vec();
    let rows: Vec<Json<Vec<serde_json::Value>>> =
        result.iter().map(|row| Json(row.values().iter().map(value_to_json).collect())).collect();
    let row_count = rows.len() as i32;

    TableResult { columns, rows, row_count }
}

/// Convert a ManifoldDB QueryResult to a GraphQL GraphResult.
///
/// This extracts Node and Edge values from the result columns.
pub fn query_result_to_graph(result: &DbQueryResult) -> Result<GraphResult> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut seen_node_ids = std::collections::HashSet::new();
    let mut seen_edge_ids = std::collections::HashSet::new();

    for row in result.iter() {
        for value in row.values() {
            extract_graph_elements(
                value,
                &mut nodes,
                &mut edges,
                &mut seen_node_ids,
                &mut seen_edge_ids,
            );
        }
    }

    Ok(GraphResult { nodes, edges })
}

/// Recursively extract nodes and edges from a Value.
fn extract_graph_elements(
    value: &Value,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    seen_node_ids: &mut std::collections::HashSet<i64>,
    seen_edge_ids: &mut std::collections::HashSet<i64>,
) {
    match value {
        Value::Node { id, labels, properties } => {
            if seen_node_ids.insert(*id) {
                nodes.push(value_node_to_graphql(*id, labels, properties));
            }
        }
        Value::Edge { id, edge_type, source, target, properties } => {
            if seen_edge_ids.insert(*id) {
                edges.push(value_edge_to_graphql(*id, edge_type, *source, *target, properties));
            }
        }
        Value::Array(items) => {
            for item in items {
                extract_graph_elements(item, nodes, edges, seen_node_ids, seen_edge_ids);
            }
        }
        _ => {}
    }
}
