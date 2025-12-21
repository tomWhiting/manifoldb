//! Vector indexes for similarity search.
//!
//! This module provides the HNSW (Hierarchical Navigable Small World) index
//! for approximate nearest neighbor (ANN) search.
//!
//! # Overview
//!
//! HNSW is a graph-based algorithm that builds a multi-layer graph structure
//! for efficient approximate nearest neighbor search. It provides:
//!
//! - **O(log N)** average search time complexity
//! - **High recall** (typically 95%+ with proper configuration)
//! - **Incremental updates** - supports insert and delete operations
//! - **Persistence** - indexes can be saved and loaded from storage
//!
//! # Algorithm Details
//!
//! The HNSW algorithm works by building a hierarchical graph where:
//! - Each layer contains a subset of nodes from the layer below
//! - The top layer has very few nodes for fast initial search
//! - The bottom layer (layer 0) contains all nodes
//! - Search starts from the top and greedily descends
//!
//! # Configuration Parameters
//!
//! - **M**: Maximum number of connections per node (typically 16-64)
//! - **`ef_construction`**: Beam width during index construction (higher = better quality, slower build)
//! - **`ef_search`**: Beam width during search (higher = better recall, slower search)
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::index::{HnswIndex, HnswConfig};
//! use manifoldb_vector::types::Embedding;
//! use manifoldb_vector::distance::DistanceMetric;
//! use manifoldb_storage::backends::RedbEngine;
//! use manifoldb_core::EntityId;
//!
//! // Create an HNSW index
//! let engine = RedbEngine::in_memory()?;
//! let config = HnswConfig::default();
//! let mut index = HnswIndex::new(engine, "embeddings", 384, DistanceMetric::Cosine, config)?;
//!
//! // Insert embeddings
//! let embedding = Embedding::new(vec![0.1; 384])?;
//! index.insert(EntityId::new(1), &embedding)?;
//!
//! // Search for similar vectors
//! let query = Embedding::new(vec![0.15; 384])?;
//! let results = index.search(&query, 10, None)?; // top 10 nearest neighbors
//! ```

mod config;
mod graph;
mod hnsw;
mod persistence;
mod traits;

pub use config::HnswConfig;
pub use graph::{
    search_layer, select_neighbors_heuristic, select_neighbors_simple, Candidate, HnswGraph,
    HnswNode,
};
pub use hnsw::HnswIndex;
pub use traits::{SearchResult, VectorIndex};
