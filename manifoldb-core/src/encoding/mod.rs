//! Serialization and key encoding for storage.
//!
//! This module provides traits and implementations for encoding `ManifoldDB` types
//! to bytes for storage and for generating ordered keys for range scans.
//!
//! # Encoding Traits
//!
//! - [`Encoder`] - Serialize types to bytes
//! - [`Decoder`] - Deserialize types from bytes
//!
//! # Type Encoding
//!
//! Implementations are provided for all core types:
//! - [`Value`](crate::types::Value) - Property values (primitives, vectors, arrays)
//! - [`Entity`](crate::types::Entity) - Graph nodes with labels and properties
//! - [`Edge`](crate::types::Edge) - Graph edges with source, target, type, and properties
//!
//! # Key Encoding
//!
//! The [`keys`] module provides functions for encoding ordered keys that support
//! efficient range scans in key-value storage backends. Keys use prefixes to
//! partition the keyspace and big-endian encoding to preserve sort order.
//!
//! # Example
//!
//! ```
//! use manifoldb_core::encoding::{Encoder, Decoder};
//! use manifoldb_core::types::{Entity, EntityId, Value};
//!
//! // Create an entity
//! let entity = Entity::new(EntityId::new(1))
//!     .with_label("Person")
//!     .with_property("name", "Alice");
//!
//! // Encode to bytes
//! let bytes = entity.encode().unwrap();
//!
//! // Decode back
//! let decoded = Entity::decode(&bytes).unwrap();
//! assert_eq!(decoded.id, entity.id);
//! ```

mod edge;
mod entity;
pub mod keys;
mod traits;
pub mod value;

#[cfg(test)]
mod proptest_tests;

pub use traits::{Decoder, Encoder, FORMAT_VERSION};
