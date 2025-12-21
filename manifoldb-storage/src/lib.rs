//! `ManifoldDB` Storage
//!
//! This crate provides the storage engine abstraction and backend implementations
//! for `ManifoldDB`.
//!
//! # Modules
//!
//! - [`engine`] - Storage engine traits and abstractions
//! - [`backends`] - Concrete storage backend implementations

pub mod backends;
pub mod engine;

pub use engine::{StorageEngine, Transaction};
