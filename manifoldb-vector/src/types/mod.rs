//! Core types for vector storage.
//!
//! This module provides the fundamental types for vector embedding storage:
//!
//! - [`Embedding`] - A validated vector embedding with dimension checks
//! - [`EmbeddingName`] - A named embedding space identifier
//! - [`EmbeddingSpace`] - Metadata about an embedding space (dimension, distance metric)

mod embedding;
mod space;

pub use embedding::Embedding;
pub use space::{EmbeddingName, EmbeddingSpace};
