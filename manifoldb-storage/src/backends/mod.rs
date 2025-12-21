//! Storage backend implementations.
//!
//! This module contains concrete implementations of the storage engine traits.
//!
//! # Available Backends
//!
//! - [`redb`] - Pure-Rust embedded database with ACID transactions

pub mod redb;

pub use self::redb::{RedbConfig, RedbCursor, RedbEngine, RedbTransaction};
