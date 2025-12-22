//! `ManifoldDB` Core
//!
//! This crate provides the fundamental types that unify graph, vector, and relational
//! data paradigms within `ManifoldDB`.
//!
//! # Overview
//!
//! The core crate defines the shared types used throughout ManifoldDB:
//!
//! - **Identifiers**: [`EntityId`] and [`EdgeId`] for referencing graph nodes and edges
//! - **Graph primitives**: [`Entity`] (nodes) and [`Edge`] (relationships)
//! - **Values**: [`Value`] enum supporting strings, numbers, vectors, and more
//! - **Labels and types**: [`Label`] for entity categorization, [`EdgeType`] for relationships
//!
//! # Example
//!
//! ```
//! use manifoldb_core::{Entity, EntityId, Edge, EdgeId, Value};
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
//! // Vector embeddings for similarity search
//! let doc = Entity::new(EntityId::new(3))
//!     .with_label("Document")
//!     .with_property("embedding", vec![0.1f32, 0.2, 0.3]);
//! ```
//!
//! # Modules
//!
//! - [`types`] - Core data types ([`Entity`], [`Edge`], [`Value`], IDs)
//! - [`encoding`] - Serialization and key encoding utilities
//! - [`error`] - Error types ([`CoreError`])
//! - [`transaction`] - Transaction error types ([`TransactionError`])

pub mod encoding;
pub mod error;
pub mod transaction;
pub mod types;

// Re-export commonly used types
pub use error::CoreError;
pub use transaction::{DeleteResult, TransactionError, TransactionResult};
pub use types::{Edge, EdgeId, EdgeType, Entity, EntityId, Label, PointId, Property, Value};
