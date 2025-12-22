//! `ManifoldDB` Graph
//!
//! This crate provides graph storage, indexing, and traversal capabilities
//! for `ManifoldDB`.
//!
//! # Modules
//!
//! - [`store`] - Node and edge storage operations
//! - [`index`] - Graph indexes (adjacency lists)
//! - [`traversal`] - Graph traversal algorithms
//! - [`analytics`] - Graph analytics algorithms (PageRank, centrality, community detection)

pub mod analytics;
pub mod index;
pub mod store;
pub mod traversal;
