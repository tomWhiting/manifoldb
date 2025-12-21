//! Core data types for `ManifoldDB`.
//!
//! This module defines the fundamental types that represent entities, edges,
//! and their properties in the unified data model.

mod edge;
mod entity;
mod id;
mod value;

pub use edge::{Edge, EdgeType};
pub use entity::{Entity, Label, Property};
pub use id::{EdgeId, EntityId};
pub use value::Value;
