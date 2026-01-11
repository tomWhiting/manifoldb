//! HTTP server setup and routing.

use anyhow::Result;
use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse, GraphQLSubscription};
use axum::{
    response::{Html, IntoResponse},
    routing::get,
    Extension, Router,
};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use crate::{create_schema, AppSchema, EmbeddingService, PubSub};

async fn graphql_handler(
    Extension(schema): Extension<AppSchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(
        GraphQLPlaygroundConfig::new("/graphql").subscription_endpoint("/graphql/ws"),
    ))
}

/// Create a shutdown signal that listens for SIGTERM and SIGINT.
#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to register SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("Failed to register SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => info!("Received SIGTERM, shutting down..."),
        _ = sigint.recv() => info!("Received SIGINT, shutting down..."),
    }
}

/// Create a shutdown signal (Windows/other platforms).
#[cfg(not(unix))]
async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("Failed to register Ctrl+C handler");
    info!("Received Ctrl+C, shutting down...");
}

/// Run the GraphQL server.
pub async fn run(database_path: &str, host: &str, port: u16) -> Result<()> {
    // Open the database
    let db = manifoldb::Database::open(database_path)?;
    info!("Opened database: {}", database_path);

    // Create pub-sub hub for subscriptions
    let pubsub = PubSub::new();

    // Create embedding service (models are lazy-loaded on first use)
    let embedding = EmbeddingService::new();
    info!("Embedding service initialized (models load on first use)");

    // Create GraphQL schema
    let schema = create_schema(db, pubsub, embedding);

    // Build router
    let app = Router::new()
        .route("/graphql", get(graphql_playground).post(graphql_handler))
        .route_service("/graphql/ws", GraphQLSubscription::new(schema.clone()))
        .layer(Extension(schema))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any));

    // Start server
    let addr = format!("{}:{}", host, port);
    info!("GraphQL playground: http://{}/graphql", addr);
    info!("GraphQL WebSocket: ws://{}/graphql/ws", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Run with graceful shutdown
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    info!("Server shut down gracefully");
    Ok(())
}
