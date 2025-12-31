//! Core data types for `ManifoldDB`.
//!
//! This module defines the fundamental types that represent entities, edges,
//! and their properties in the unified data model.

mod edge;
mod entity;
mod id;
mod scored;
mod value;
mod vector;

pub use edge::{Edge, EdgeType};
pub use entity::{Entity, Label, Property};
pub use id::{CollectionId, EdgeId, EntityId, PointId};
pub use scored::{ScoredEntity, ScoredId};
pub use value::Value;
pub use vector::VectorData;
