//! Core types for vector storage.
//!
//! This module provides the fundamental types for vector embedding storage:
//!
//! - [`Embedding`] - A validated dense vector embedding with dimension checks
//! - [`SparseEmbedding`] - A sparse vector embedding (index, value pairs)
//! - [`EmbeddingName`] - A named embedding space identifier
//! - [`EmbeddingSpace`] - Metadata about an embedding space (dimension, distance metric)
//! - [`SparseEmbeddingSpace`] - Metadata about a sparse embedding space

mod embedding;
mod space;
mod sparse;

pub use embedding::Embedding;
pub use space::{EmbeddingName, EmbeddingSpace, SparseEmbeddingSpace};
pub use sparse::SparseEmbedding;
