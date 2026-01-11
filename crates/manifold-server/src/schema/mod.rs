//! GraphQL schema definition.
//!
//! This module contains the GraphQL schema, including:
//! - [`types`] - GraphQL type definitions (Node, Edge, TableResult, etc.)
//! - [`query`] - Query resolvers (sql, cypher, stats, etc.)
//! - [`mutation`] - Mutation resolvers (execute, createNode, etc.)
//! - [`subscription`] - Subscription resolvers (nodeChanges, edgeChanges, etc.)

mod mutation;
mod query;
mod subscription;
pub mod types;

use async_graphql::Schema;
use manifoldb::Database;
use std::sync::Arc;

use crate::embedding::EmbeddingService;
use crate::pubsub::PubSub;

pub use mutation::MutationRoot;
pub use query::QueryRoot;
pub use subscription::SubscriptionRoot;
pub use types::*;

/// The GraphQL schema type for the ManifoldDB server.
pub type AppSchema = Schema<QueryRoot, MutationRoot, SubscriptionRoot>;

/// Create a new GraphQL schema with the given database, pub-sub hub, and embedding service.
pub fn create_schema(db: Database, pubsub: PubSub, embedding: EmbeddingService) -> AppSchema {
    Schema::build(QueryRoot, MutationRoot, SubscriptionRoot)
        .data(Arc::new(db))
        .data(pubsub)
        .data(Arc::new(embedding))
        .finish()
}
