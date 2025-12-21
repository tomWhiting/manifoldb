//! Storage backend implementations.
//!
//! This module contains concrete implementations of the storage engine traits.

pub mod redb;

pub use self::redb::RedbEngine;
