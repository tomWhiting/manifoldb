//! Vector embedding storage.
//!
//! This module provides storage for vector embeddings associated with entities.
//!
//! # Overview
//!
//! The [`VectorStore`] manages dense embeddings organized into named embedding spaces.
//! The [`SparseVectorStore`] manages sparse embeddings for high-dimensional vectors
//! with few non-zero elements (e.g., SPLADE embeddings).
//!
//! Each entity can have embeddings in multiple spaces (e.g., `text_embedding`,
//! `image_embedding`), and each space has a fixed dimension and distance metric.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::store::VectorStore;
//! use manifoldb_vector::types::{Embedding, EmbeddingName, EmbeddingSpace};
//! use manifoldb_vector::distance::DistanceMetric;
//! use manifoldb_core::EntityId;
//!
//! // Create a store with a storage backend
//! let store = VectorStore::new(engine);
//!
//! // Create an embedding space
//! let name = EmbeddingName::new("text_embedding")?;
//! let space = EmbeddingSpace::new(name.clone(), 384, DistanceMetric::Cosine);
//! store.create_space(&space)?;
//!
//! // Store an embedding
//! let embedding = Embedding::new(vec![0.1; 384])?;
//! store.put(EntityId::new(1), &name, &embedding)?;
//!
//! // Retrieve it
//! let retrieved = store.get(EntityId::new(1), &name)?;
//! ```

mod sparse_store;
mod vector_store;

pub use sparse_store::SparseVectorStore;
pub use vector_store::VectorStore;
