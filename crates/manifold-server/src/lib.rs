//! GraphQL server for ManifoldDB.
//!
//! This crate provides a GraphQL API layer over ManifoldDB, exposing
//! SQL and Cypher query capabilities via HTTP and WebSocket.
//!
//! # Modules
//!
//! - [`schema`] - GraphQL schema definition (types, queries, mutations, subscriptions)
//! - [`convert`] - Type conversions from ManifoldDB to GraphQL types
//! - [`pubsub`] - Pub-sub infrastructure for subscriptions
//! - [`server`] - HTTP/WebSocket server implementation
//! - [`embedding`] - Text embedding service using tessera

#![deny(clippy::unwrap_used)]

pub mod convert;
pub mod embedding;
pub mod pubsub;
pub mod schema;
pub mod server;

pub use embedding::EmbeddingService;
pub use pubsub::PubSub;
pub use schema::{create_schema, AppSchema};
