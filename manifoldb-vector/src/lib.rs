//! `ManifoldDB` Vector
//!
//! This crate provides vector embedding storage and similarity search
//! capabilities for `ManifoldDB`.
//!
//! # Modules
//!
//! - [`store`] - Vector embedding storage
//! - [`index`] - Vector indexes (HNSW)
//! - [`distance`] - Distance functions
//! - [`ops`] - Vector search operators

pub mod distance;
pub mod index;
pub mod ops;
pub mod store;
