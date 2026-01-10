//! SQL execution bridge.
//!
//! This module bridges the query engine with actual storage access,
//! converting logical plans into operations on entities and edges.

pub mod constraints;
mod executor;
mod graph_accessor;
mod index_maintenance;
mod index_scan;
mod scan;
mod table_extractor;

pub use executor::{
    execute_graph_dml, execute_prepared_query, execute_prepared_statement, execute_query,
    execute_query_with_catalog, execute_query_with_limit, execute_statement, is_cypher_dml,
};
pub use graph_accessor::{DatabaseGraphAccessor, DatabaseGraphMutator};
pub use index_maintenance::EntityIndexMaintenance;
pub use scan::{CollectionContext, StorageScan};
pub use table_extractor::extract_tables_from_sql;
