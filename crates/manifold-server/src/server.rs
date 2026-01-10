//! HTTP server setup and routing.

use anyhow::Result;
use async_graphql::http::{playground_source, GraphQLPlaygroundConfig};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{
    response::{Html, IntoResponse},
    routing::get,
    Extension, Router,
};
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use manifold_server::{create_schema, AppSchema};

async fn graphql_handler(
    Extension(schema): Extension<AppSchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

async fn graphql_playground() -> impl IntoResponse {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

/// Run the GraphQL server.
pub async fn run(database_path: &str, host: &str, port: u16) -> Result<()> {
    // Open the database
    let db = manifoldb::Database::open(database_path)?;
    info!("Opened database: {}", database_path);

    // Create GraphQL schema
    let schema = create_schema(db);

    // Build router
    let app = Router::new()
        .route("/graphql", get(graphql_playground).post(graphql_handler))
        .layer(Extension(schema))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Start server
    let addr = format!("{}:{}", host, port);
    info!("GraphQL playground: http://{}/graphql", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
