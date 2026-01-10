//! GraphQL server for ManifoldDB.
//!
//! This crate provides a GraphQL API layer over ManifoldDB, exposing
//! SQL and Cypher query capabilities via HTTP.
//!
//! # Modules
//!
//! - [`schema`] - GraphQL schema definition (types, queries, mutations)
//! - [`convert`] - Type conversions from ManifoldDB to GraphQL types

#![deny(clippy::unwrap_used)]

pub mod convert;
pub mod schema;

pub use schema::{create_schema, AppSchema};
