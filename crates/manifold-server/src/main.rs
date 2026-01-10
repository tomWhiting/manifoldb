//! ManifoldDB GraphQL Server
//!
//! A high-performance GraphQL API server for ManifoldDB.

use anyhow::Result;
use clap::Parser;

mod server;

#[derive(Parser)]
#[command(name = "manifold-server")]
#[command(about = "GraphQL server for ManifoldDB")]
struct Args {
    /// Path to the database file
    database: String,

    /// Port to listen on
    #[arg(short, long, default_value = "4000")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("manifold_server=info".parse()?),
        )
        .init();

    let args = Args::parse();
    server::run(&args.database, &args.host, args.port).await
}
