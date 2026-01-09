//! Procedure infrastructure for CALL/YIELD statements.
//!
//! This module provides the procedure registry and trait for defining callable
//! procedures that can be invoked via CALL statements. Procedures are used for:
//!
//! - Graph algorithms (PageRank, shortest path, community detection)
//! - Database introspection (list tables, describe schema)
//! - Administrative operations
//!
//! # Example
//!
//! ```ignore
//! // Calling a procedure from SQL
//! CALL algo.pageRank('nodes', 'edges', {damping: 0.85})
//! YIELD node, score
//! WHERE score > 0.1
//!
//! // Cypher-style
//! CALL algo.shortestPath(source, target) YIELD path, cost
//! ```
//!
//! # Architecture
//!
//! Procedures are registered in a [`ProcedureRegistry`] and implement the
//! [`Procedure`] trait. Each procedure declares its signature (name, parameters,
//! return columns) and provides an execution function that produces rows.

pub mod builtins;
mod registry;
mod signature;
mod traits;

pub use builtins::{
    execute_pagerank_with_tx, execute_shortest_path_with_tx, register_builtins, PageRankProcedure,
    ShortestPathProcedure,
};
pub use registry::ProcedureRegistry;
pub use signature::{ProcedureParameter, ProcedureSignature, ReturnColumn};
pub use traits::{make_row, Procedure, ProcedureArgs, ProcedureError, ProcedureResult};
