//! SQL execution bridge.
//!
//! This module bridges the query engine with actual storage access,
//! converting logical plans into operations on entities and edges.

mod executor;
mod graph_accessor;
mod scan;
mod table_extractor;

pub use executor::{
    execute_prepared_query, execute_prepared_statement, execute_query, execute_query_with_limit,
    execute_statement,
};
pub use scan::{CollectionContext, StorageScan};
pub use table_extractor::extract_tables_from_sql;
