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
//! - [`index`] - Vector indexes (HNSW) - future
//! - [`ops`] - Vector search operators - future

pub mod distance;
pub mod encoding;
pub mod error;
pub mod index;
pub mod ops;
pub mod store;
pub mod types;

// Re-export commonly used types
pub use distance::DistanceMetric;
pub use error::VectorError;
pub use store::VectorStore;
pub use types::{Embedding, EmbeddingName, EmbeddingSpace};
