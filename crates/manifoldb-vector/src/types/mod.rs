//! Core types for vector storage.
//!
//! This module provides the fundamental types for vector embedding storage:
//!
//! ## Entity-based embeddings
//! - [`Embedding`] - A validated dense vector embedding with dimension checks
//! - [`SparseEmbedding`] - A sparse vector embedding (index, value pairs)
//! - [`MultiVectorEmbedding`] - A multi-vector embedding (ColBERT-style token embeddings)
//! - [`BinaryEmbedding`] - A binary vector embedding (bit-packed for Hamming distance)
//! - [`EmbeddingName`] - A named embedding space identifier
//! - [`EmbeddingSpace`] - Metadata about an embedding space (dimension, distance metric)
//! - [`SparseEmbeddingSpace`] - Metadata about a sparse embedding space
//! - [`MultiVectorEmbeddingSpace`] - Metadata about a multi-vector embedding space
//! - [`BinaryEmbeddingSpace`] - Metadata about a binary embedding space
//!
//! ## Separated vector storage
//! - [`VectorData`] - Vector data supporting dense, sparse, multi-vector, and binary formats
//! - [`VectorRef`] - Reference to a vector attached to an entity
//!
//! ## Point collections (Qdrant-style)
//! - [`Collection`] - A named collection of points with a defined schema
//! - [`CollectionName`] - A validated collection name
//! - [`CollectionSchema`] - Schema defining the vectors in a collection
//! - [`VectorConfig`] - Configuration for a named vector (type, dimension)
//! - [`VectorName`] - A validated vector name
//! - [`Payload`] - JSON payload attached to a point
//! - [`NamedVector`] - A vector value (dense, sparse, or multi-vector)

mod binary;
mod embedding;
mod multi;
mod point;
mod space;
mod sparse;
mod vector_data;
mod vector_ref;

pub use binary::BinaryEmbedding;
pub use embedding::Embedding;
pub use multi::MultiVectorEmbedding;
pub use point::{
    Collection, CollectionName, CollectionSchema, NamedVector, Payload, VectorConfig, VectorName,
    VectorType,
};
pub use space::{
    BinaryEmbeddingSpace, EmbeddingName, EmbeddingSpace, MultiVectorEmbeddingSpace,
    SparseEmbeddingSpace,
};
pub use sparse::SparseEmbedding;
pub use vector_data::VectorData;
pub use vector_ref::VectorRef;
