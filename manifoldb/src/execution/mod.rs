//! SQL execution bridge.
//!
//! This module bridges the query engine with actual storage access,
//! converting logical plans into operations on entities and edges.

mod executor;
mod graph_accessor;
mod scan;
mod table_extractor;

pub use executor::{execute_query, execute_statement};
pub use scan::StorageScan;
pub use table_extractor::extract_tables_from_sql;
