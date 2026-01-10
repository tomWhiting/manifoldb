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

#![deny(clippy::unwrap_used)]

pub mod convert;
pub mod pubsub;
pub mod schema;

pub use pubsub::PubSub;
pub use schema::{create_schema, AppSchema};
