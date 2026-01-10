//! Integration tests for ManifoldDB.
//!
//! This module contains comprehensive integration tests that exercise
//! the database across multiple features and at various scales.

pub mod bulk_delete;
pub mod bulk_delete_edges;
pub mod bulk_vectors;
pub mod collection;
pub mod combined;
pub mod concurrency;
pub mod constraints;
pub mod correctness;
pub mod crud;
pub mod cypher_create;
pub mod cypher_delete;
pub mod cypher_foreach;
pub mod cypher_merge;
pub mod cypher_remove;
pub mod ddl;
pub mod e2e;
pub mod graph;
pub mod graph_vector_search;
pub mod index_maintenance;
pub mod index_query;
pub mod match_filter;
pub mod prepared;
pub mod procedure;
pub mod recovery;
pub mod scale;
pub mod session_transactions;
pub mod set_property;
pub mod sql;
pub mod transactions;
pub mod vector;
