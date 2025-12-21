//! Storage engine traits and abstractions.
//!
//! This module defines the core traits that storage backends must implement.

mod error;
mod traits;

pub use error::StorageError;
pub use traits::{Cursor, StorageEngine, Transaction};
