//! Key encoding for vector storage.
//!
//! This module provides key encoding for vector embeddings in storage backends.
//! Keys are designed to support efficient prefix-based range scans.
//!
//! # Key Prefixes
//!
//! - `0x10` - Embedding space metadata: `[0x10][space_name_hash]`
//! - `0x11` - Entity embedding: `[0x11][space_name_hash][entity_id]`
//! - `0x12` - Sparse embedding space metadata: `[0x12][space_name_hash]`
//! - `0x13` - Sparse entity embedding: `[0x13][space_name_hash][entity_id]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.

mod keys;

pub use keys::{
    // Dense embeddings
    decode_embedding_entity_id,
    decode_embedding_key,
    decode_embedding_space_key,
    // Sparse embeddings
    decode_sparse_embedding_entity_id,
    decode_sparse_embedding_key,
    decode_sparse_embedding_space_key,
    encode_embedding_key,
    encode_embedding_prefix,
    encode_embedding_space_key,
    encode_sparse_embedding_key,
    encode_sparse_embedding_prefix,
    encode_sparse_embedding_space_key,
    hash_name,
    EmbeddingKey,
    SparseEmbeddingKey,
    PREFIX_EMBEDDING,
    PREFIX_EMBEDDING_SPACE,
    PREFIX_SPARSE_EMBEDDING,
    PREFIX_SPARSE_EMBEDDING_SPACE,
};
