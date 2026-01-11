//! `ManifoldDB` Core
//!
//! This crate provides the fundamental types that unify graph, vector, and relational
//! data paradigms within `ManifoldDB`.
//!
//! # Overview
//!
//! The core crate defines the shared types used throughout ManifoldDB:
//!
//! - **Identifiers**: [`EntityId`], [`EdgeId`], and [`CollectionId`] for referencing graph elements
//! - **Graph primitives**: [`Entity`] (nodes) and [`Edge`] (relationships)
//! - **Values**: [`Value`] enum supporting strings, numbers, vectors, and more
//! - **Vectors**: [`VectorData`] for attaching embeddings to entities
//! - **Labels and types**: [`Label`] for entity categorization, [`EdgeType`] for relationships
//!
//! # Example
//!
//! ```
//! use manifoldb_core::{Entity, EntityId, Edge, EdgeId, Value, VectorData};
//!
//! // Create entities (graph nodes)
//! let alice = Entity::new(EntityId::new(1))
//!     .with_label("Person")
//!     .with_property("name", "Alice")
//!     .with_property("age", 30i64);
//!
//! let bob = Entity::new(EntityId::new(2))
//!     .with_label("Person")
//!     .with_property("name", "Bob");
//!
//! // Create edges (relationships)
//! let follows = Edge::new(EdgeId::new(1), alice.id, bob.id, "FOLLOWS")
//!     .with_property("since", "2024-01-01");
//!
//! // Query entity properties
//! assert!(alice.has_label("Person"));
//! assert_eq!(alice.get_property("name"), Some(&Value::String("Alice".into())));
//!
//! // Entity with vector embedding stored as a property
//! let doc = Entity::new(EntityId::new(3))
//!     .with_label("Document")
//!     .with_property("title", "Example")
//!     .with_property("embedding", Value::Vector(vec![0.1f32, 0.2, 0.3]));
//!
//! assert!(doc.get_property("embedding").is_some());
//! ```
//!
//! # Modules
//!
//! - [`types`] - Core data types ([`Entity`], [`Edge`], [`Value`], [`VectorData`], IDs)
//! - [`encoding`] - Serialization and key encoding utilities
//! - [`index`] - Property index types for secondary indexes
//! - [`error`] - Error types ([`CoreError`])
//! - [`transaction`] - Transaction error types ([`TransactionError`])

// Deny unwrap in library code to ensure proper error handling
#![deny(clippy::unwrap_used)]

pub mod encoding;
pub mod error;
pub mod index;
pub mod transaction;
pub mod types;

// Re-export commonly used types
pub use error::CoreError;
pub use transaction::{DeleteResult, TransactionError, TransactionResult};
pub use types::{
    CollectionId, Edge, EdgeId, EdgeType, Entity, EntityId, Label, PointId, Property, ScoredEntity,
    ScoredId, Value, VectorData,
};
