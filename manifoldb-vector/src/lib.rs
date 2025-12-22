//! `ManifoldDB` Vector
//!
//! This crate provides vector embedding storage and similarity search
//! capabilities for `ManifoldDB`.
//!
//! # Overview
//!
//! The vector module provides:
//!
//! - **Embedding storage**: Store and retrieve vector embeddings associated with entities
//! - **Embedding spaces**: Named spaces with fixed dimensions and distance metrics
//! - **Distance functions**: Cosine, Euclidean, and dot product similarity
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::store::VectorStore;
//! use manifoldb_vector::types::{Embedding, EmbeddingName, EmbeddingSpace};
//! use manifoldb_vector::distance::DistanceMetric;
//! use manifoldb_core::EntityId;
//! use manifoldb_storage::backends::RedbEngine;
//!
//! // Create a store
//! let engine = RedbEngine::in_memory()?;
//! let store = VectorStore::new(engine);
//!
//! // Create an embedding space
//! let name = EmbeddingName::new("text_embedding")?;
//! let space = EmbeddingSpace::new(name.clone(), 384, DistanceMetric::Cosine);
//! store.create_space(&space)?;
//!
//! // Store embeddings
//! let embedding = Embedding::new(vec![0.1; 384])?;
//! store.put(EntityId::new(1), &name, &embedding)?;
//!
//! // Retrieve embeddings
//! let retrieved = store.get(EntityId::new(1), &name)?;
//! ```
//!
//! # Modules
//!
//! - [`store`] - Vector embedding storage
//! - [`types`] - Core types ([`Embedding`], [`EmbeddingSpace`], [`EmbeddingName`])
//! - [`distance`] - Distance functions
//! - [`encoding`] - Key encoding for storage
//! - [`error`] - Error types
//! - [`index`] - Vector indexes (HNSW)
//! - [`ops`] - Vector search operators ([`AnnScan`], [`ExactKnn`], [`VectorFilter`])
//!
//! # Search Operators
//!
//! The [`ops`] module provides operators for vector similarity search:
//!
//! - [`AnnScan`] - Approximate nearest neighbor search using HNSW
//! - [`ExactKnn`] - Brute force k-NN search for small sets or validation
//! - [`VectorFilter`] - Post-filter results by predicates
//!
//! ## Example
//!
//! ```ignore
//! use manifoldb_vector::ops::{AnnScan, VectorOperator, SearchConfig};
//!
//! // Search for 10 nearest neighbors
//! let mut scan = AnnScan::k_nearest(&index, &query, 10)?;
//!
//! while let Some(m) = scan.next()? {
//!     println!("Entity {:?} at distance {}", m.entity_id, m.distance);
//! }
//! ```

pub mod distance;
pub mod encoding;
pub mod error;
pub mod index;
pub mod ops;
pub mod store;
pub mod types;

// Re-export commonly used types
pub use distance::{CachedNorm, DistanceMetric};
pub use error::VectorError;
pub use index::{HnswConfig, HnswIndex, SearchResult, VectorIndex};
pub use ops::{AnnScan, ExactKnn, SearchConfig, VectorFilter, VectorMatch, VectorOperator};
pub use store::VectorStore;
pub use types::{Embedding, EmbeddingName, EmbeddingSpace};
