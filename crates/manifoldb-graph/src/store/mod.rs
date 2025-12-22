//! Node and edge storage operations.
//!
//! This module provides CRUD operations for nodes (entities) and edges
//! in the graph. All operations work within a transaction context for
//! ACID guarantees.
//!
//! # Overview
//!
//! - [`NodeStore`] - Create, read, update, delete nodes
//! - [`EdgeStore`] - Create, read, update, delete edges
//! - [`IdGenerator`] - Monotonic ID generation for entities and edges
//!
//! # Tables
//!
//! The stores use the following tables in the storage backend:
//!
//! - `entities` - Entity data keyed by entity ID
//! - `labels` - Label index for entity lookups by label
//! - `edges` - Edge data keyed by edge ID
//! - `edges_by_source` - Index for outgoing edge lookups
//! - `edges_by_target` - Index for incoming edge lookups
//! - `edge_types` - Index for edge type lookups
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::store::{NodeStore, EdgeStore, IdGenerator};
//! use manifoldb_storage::backends::RedbEngine;
//!
//! let engine = RedbEngine::in_memory()?;
//! let id_gen = IdGenerator::new();
//!
//! // Create nodes
//! let mut tx = engine.begin_write()?;
//! let alice = NodeStore::create(&mut tx, &id_gen, |id| {
//!     Entity::new(id)
//!         .with_label("Person")
//!         .with_property("name", "Alice")
//! })?;
//! let bob = NodeStore::create(&mut tx, &id_gen, |id| {
//!     Entity::new(id)
//!         .with_label("Person")
//!         .with_property("name", "Bob")
//! })?;
//!
//! // Create an edge
//! let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
//!     Edge::new(id, alice.id, bob.id, "FOLLOWS")
//!         .with_property("since", "2024-01-01")
//! })?;
//!
//! tx.commit()?;
//!
//! // Query
//! let tx = engine.begin_read()?;
//! let outgoing = EdgeStore::get_outgoing(&tx, alice.id)?;
//! assert_eq!(outgoing.len(), 1);
//! ```

mod edge;
mod error;
mod id_gen;
mod node;

pub use edge::{
    EdgeStore, TABLE_EDGES, TABLE_EDGES_BY_SOURCE, TABLE_EDGES_BY_TARGET, TABLE_EDGE_TYPES,
};
pub use error::{GraphError, GraphResult};
pub use id_gen::IdGenerator;
pub use node::{NodeStore, TABLE_ENTITIES, TABLE_LABELS};
