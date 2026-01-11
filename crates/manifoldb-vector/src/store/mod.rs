//! Vector embedding storage.
//!
//! This module provides storage for vector embeddings associated with entities.
//!
//! # Overview
//!
//! ## Entity-based embedding stores
//!
//! The [`VectorStore`] manages dense embeddings organized into named embedding spaces.
//! The [`SparseVectorStore`] manages sparse embeddings for high-dimensional vectors
//! with few non-zero elements (e.g., SPLADE embeddings).
//! The [`MultiVectorStore`] manages multi-vector embeddings for ColBERT-style
//! late interaction models.
//!
//! Each entity can have embeddings in multiple spaces (e.g., `text_embedding`,
//! `image_embedding`), and each space has a fixed dimension and distance metric.
//!
//! ## Collection vector store (entity-to-vector mapping)
//!
//! The [`CollectionVectorStore`] provides dedicated vector storage separate from
//! entity properties. This enables:
//! - Storage efficiency: Read entities without loading vector data
//! - Multiple embeddings per entity: Support text, image, summary embeddings
//! - Independent operations: Update vectors without touching entities
//! - Cascade deletion: Delete all vectors when entity is removed
//!
//! ## Inverted index for sparse vectors
//!
//! The [`InvertedIndex`] provides efficient top-k similarity search for sparse vectors
//! using posting lists and WAND/DAAT algorithms. This is optimized for SPLADE-style
//! sparse retrieval.
//!
//! # Example (Entity embeddings)
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

mod collection_vector_store;
mod inverted_index;
mod multi_vector_store;
mod sparse_store;
mod vector_store;

pub use collection_vector_store::{
    decode_vector_value, encode_vector_value, CollectionVectorStore, TABLE_COLLECTION_VECTORS,
};
pub use inverted_index::{
    InvertedIndex, InvertedIndexMeta, PostingEntry, PostingList, ScoringFunction, SearchResult,
};
pub use multi_vector_store::MultiVectorStore;
pub use sparse_store::SparseVectorStore;
pub use vector_store::VectorStore;
