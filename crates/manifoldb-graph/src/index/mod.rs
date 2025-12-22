//! Graph indexes for efficient traversal.
//!
//! This module provides adjacency list indexes for fast neighbor lookups
//! and efficient graph traversal operations.
//!
//! # Overview
//!
//! - [`AdjacencyIndex`] - Adjacency list index for outgoing and incoming edges
//! - [`IndexMaintenance`] - Index update operations for edge mutations
//!
//! # Table Layout
//!
//! The adjacency indexes use composite keys for efficient prefix scans:
//!
//! - `outgoing`: `(EntityId, EdgeType, EdgeId) -> ()` - edges from a source
//! - `incoming`: `(EntityId, EdgeType, EdgeId) -> ()` - edges to a target
//!
//! This enables queries like:
//! - "All edges from entity X" - prefix scan on `(source_id)`
//! - "All edges of type Y from entity X" - prefix scan on `(source_id, edge_type)`
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::index::AdjacencyIndex;
//!
//! // Get all neighbors reachable from an entity
//! let outgoing_ids = AdjacencyIndex::get_outgoing_edge_ids(&tx, source)?;
//!
//! // Get neighbors filtered by edge type
//! let follows = AdjacencyIndex::get_outgoing_by_type(&tx, source, &"FOLLOWS".into())?;
//!
//! // Iterate over edges without collecting
//! AdjacencyIndex::for_each_outgoing(&tx, source, |edge_id| {
//!     // Process each edge
//!     Ok(true) // Continue iteration
//! })?;
//! ```

mod adjacency;
mod maintenance;

pub use adjacency::AdjacencyIndex;
pub use maintenance::IndexMaintenance;
