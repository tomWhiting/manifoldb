//! `ManifoldDB` Core
//!
//! This crate provides the fundamental types that unify graph, vector, and relational
//! data paradigms within `ManifoldDB`.
//!
//! # Modules
//!
//! - [`types`] - Core data types (Entity, Edge, Value, IDs)
//! - [`encoding`] - Serialization and key encoding
//! - [`error`] - Error types

pub mod encoding;
pub mod error;
pub mod types;

// Re-export commonly used types
pub use error::CoreError;
pub use types::{Edge, EdgeId, EdgeType, Entity, EntityId, Label, Property, Value};
