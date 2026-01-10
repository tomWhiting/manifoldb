//! GraphQL schema definition.
//!
//! This module contains the GraphQL schema, including:
//! - [`types`] - GraphQL type definitions (Node, Edge, TableResult, etc.)
//! - [`query`] - Query resolvers (sql, cypher, stats, etc.)
//! - [`mutation`] - Mutation resolvers (execute, createNode, etc.)

mod mutation;
mod query;
mod types;

use async_graphql::{EmptySubscription, Schema};
use manifoldb::Database;
use std::sync::Arc;

pub use mutation::MutationRoot;
pub use query::QueryRoot;
pub use types::*;

/// The GraphQL schema type for the ManifoldDB server.
pub type AppSchema = Schema<QueryRoot, MutationRoot, EmptySubscription>;

/// Create a new GraphQL schema with the given database.
pub fn create_schema(db: Database) -> AppSchema {
    Schema::build(QueryRoot, MutationRoot, EmptySubscription)
        .data(Arc::new(db))
        .finish()
}
