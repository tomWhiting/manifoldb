//! Session Viewer HTTP Server
//!
//! Serves a REST API for the session viewer frontend to visualize
//! ManifoldDB session data as a network graph.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::get;
use axum::Router;
use clap::Parser;
use manifoldb::Database;
use manifoldb_core::Value;
use serde::Serialize;
use tower_http::cors::{Any, CorsLayer};

/// Session viewer HTTP server for ManifoldDB
#[derive(Parser, Debug)]
#[command(name = "session-server")]
#[command(about = "HTTP API server for ManifoldDB Session Viewer")]
struct Args {
    /// Path to the ManifoldDB database file
    #[arg(required = true)]
    database: PathBuf,

    /// Port to listen on
    #[arg(short, long, default_value = "3456")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

/// Node data for the graph API
#[derive(Debug, Serialize)]
struct NodeResponse {
    id: i64,
    labels: Vec<String>,
    properties: HashMap<String, serde_json::Value>,
}

/// Edge data for the graph API
#[derive(Debug, Serialize)]
struct EdgeResponse {
    id: i64,
    source: i64,
    target: i64,
    #[serde(rename = "type")]
    edge_type: String,
    properties: HashMap<String, serde_json::Value>,
}

/// Graph data combining nodes and edges
#[derive(Debug, Serialize)]
struct GraphResponse {
    nodes: Vec<NodeResponse>,
    edges: Vec<EdgeResponse>,
}

/// Statistics about the graph
#[derive(Debug, Serialize)]
struct StatsResponse {
    node_count: usize,
    edge_count: usize,
    labels: HashMap<String, usize>,
    edge_types: HashMap<String, usize>,
}

/// Application state shared across handlers
struct AppState {
    db: Database,
}

/// Convert a ManifoldDB Value to a JSON value
fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::json!(i),
        Value::Float(f) => serde_json::json!(f),
        Value::String(s) => serde_json::Value::String(s.clone()),
        Value::Bytes(b) => serde_json::json!(format!("<bytes:{}>", b.len())),
        Value::Vector(v) => serde_json::json!(v),
        Value::SparseVector(v) => serde_json::json!(v),
        Value::MultiVector(v) => serde_json::json!(v),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        Value::Point { x, y, z, srid } => {
            serde_json::json!({
                "_type": "point",
                "x": x,
                "y": y,
                "z": z,
                "srid": srid
            })
        }
        Value::Node { id, labels, properties } => {
            serde_json::json!({
                "_type": "node",
                "id": id,
                "labels": labels,
                "properties": properties.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect::<HashMap<_, _>>()
            })
        }
        Value::Edge { id, edge_type, source, target, properties } => {
            serde_json::json!({
                "_type": "edge",
                "id": id,
                "edge_type": edge_type,
                "source": source,
                "target": target,
                "properties": properties.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect::<HashMap<_, _>>()
            })
        }
    }
}

/// Convert properties HashMap to JSON
fn props_to_json(props: &HashMap<String, Value>) -> HashMap<String, serde_json::Value> {
    props.iter().map(|(k, v)| (k.clone(), value_to_json(v))).collect()
}

/// GET /api/graph - Returns all nodes and edges
async fn get_graph(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GraphResponse>, (StatusCode, String)> {
    // Query all nodes
    let nodes_result = state.db.query("MATCH (n) RETURN n").map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to query nodes: {}", e))
    })?;

    let mut nodes = Vec::new();
    for row in nodes_result.iter() {
        if let Some(Value::Node { id, labels, properties }) = row.get(0) {
            nodes.push(NodeResponse {
                id: *id,
                labels: labels.clone(),
                properties: props_to_json(properties),
            });
        }
    }

    // Query all edges
    let edges_result = state.db.query("MATCH ()-[r]->() RETURN r").map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to query edges: {}", e))
    })?;

    let mut edges = Vec::new();
    for row in edges_result.iter() {
        if let Some(Value::Edge { id, edge_type, source, target, properties }) = row.get(0) {
            edges.push(EdgeResponse {
                id: *id,
                source: *source,
                target: *target,
                edge_type: edge_type.clone(),
                properties: props_to_json(properties),
            });
        }
    }

    Ok(Json(GraphResponse { nodes, edges }))
}

/// GET /api/stats - Returns statistics about the graph
async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<StatsResponse>, (StatusCode, String)> {
    // Query all nodes to count labels
    let nodes_result = state.db.query("MATCH (n) RETURN n").map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to query nodes: {}", e))
    })?;

    let mut node_count = 0;
    let mut labels: HashMap<String, usize> = HashMap::new();

    for row in nodes_result.iter() {
        if let Some(Value::Node { labels: node_labels, .. }) = row.get(0) {
            node_count += 1;
            for label in node_labels {
                *labels.entry(label.clone()).or_insert(0) += 1;
            }
        }
    }

    // Query all edges to count types
    let edges_result = state.db.query("MATCH ()-[r]->() RETURN r").map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to query edges: {}", e))
    })?;

    let mut edge_count = 0;
    let mut edge_types: HashMap<String, usize> = HashMap::new();

    for row in edges_result.iter() {
        if let Some(Value::Edge { edge_type, .. }) = row.get(0) {
            edge_count += 1;
            *edge_types.entry(edge_type.clone()).or_insert(0) += 1;
        }
    }

    Ok(Json(StatsResponse { node_count, edge_count, labels, edge_types }))
}

/// GET /api/health - Health check endpoint
async fn health() -> &'static str {
    "OK"
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Open the database
    println!("Opening database: {}", args.database.display());
    let db = Database::open(&args.database)?;
    println!("Database opened successfully");

    // Create app state
    let state = Arc::new(AppState { db });

    // Configure CORS for the frontend
    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);

    // Build the router
    let app = Router::new()
        .route("/api/graph", get(get_graph))
        .route("/api/stats", get(get_stats))
        .route("/api/health", get(health))
        .layer(cors)
        .with_state(state);

    // Start the server
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    println!("Starting server on http://{}", addr);
    println!("API endpoints:");
    println!("  GET /api/graph  - Get all nodes and edges");
    println!("  GET /api/stats  - Get graph statistics");
    println!("  GET /api/health - Health check");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
