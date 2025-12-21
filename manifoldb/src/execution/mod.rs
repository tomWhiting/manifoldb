//! SQL execution bridge.
//!
//! This module bridges the query engine with actual storage access,
//! converting logical plans into operations on entities and edges.

mod executor;
mod scan;

pub use executor::{execute_query, execute_statement};
pub use scan::StorageScan;
