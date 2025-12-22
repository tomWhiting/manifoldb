//! Storage backend implementations.
//!
//! This module contains concrete implementations of the storage engine traits.
//!
//! # Available Backends
//!
//! - [`redb`] - Pure-Rust embedded database with ACID transactions
//! - [`wal_engine`] - WAL-enabled storage engine wrapper for enhanced durability

pub mod redb;
pub mod wal_engine;

pub use self::redb::{RedbConfig, RedbCursor, RedbEngine, RedbTransaction};
pub use self::wal_engine::{WalEngine, WalEngineConfig, WalEngineOpenResult, WalTransaction};
