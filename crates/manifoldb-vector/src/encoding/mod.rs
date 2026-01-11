//! Key encoding for vector storage.
//!
//! This module provides key encoding for vector embeddings in storage backends.
//! Keys are designed to support efficient prefix-based range scans.
//!
//! # Key Prefixes
//!
//! ## Embedding spaces (entity-based)
//! - `0x10` - Embedding space metadata: `[0x10][space_name_hash]`
//! - `0x11` - Entity embedding: `[0x11][space_name_hash][entity_id]`
//! - `0x12` - Sparse embedding space metadata: `[0x12][space_name_hash]`
//! - `0x13` - Sparse entity embedding: `[0x13][space_name_hash][entity_id]`
//! - `0x14` - Multi-vector space metadata: `[0x14][space_name_hash]`
//! - `0x15` - Multi-vector embedding: `[0x15][space_name_hash][entity_id]`
//!
//! ## Inverted index (sparse vector index)
//! - `0x30` - Posting list: `[0x30][collection_hash][vector_name_hash][token_id]`
//! - `0x31` - Index metadata: `[0x31][collection_hash][vector_name_hash]`
//! - `0x32` - Point tokens: `[0x32][collection_hash][vector_name_hash][point_id]`
//!
//! ## Collection vectors (entity-to-vector mapping)
//! - `0x40` - Collection vector: `[0x40][collection_id][entity_id][vector_name_hash]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.

mod collection_vector_keys;
mod inverted_keys;
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
    PREFIX_MULTI_VECTOR,
    PREFIX_MULTI_VECTOR_SPACE,
    PREFIX_SPARSE_EMBEDDING,
    PREFIX_SPARSE_EMBEDDING_SPACE,
};

pub use inverted_keys::{
    decode_inverted_meta_key, decode_point_tokens_key, decode_posting_key,
    encode_inverted_meta_collection_prefix, encode_inverted_meta_key,
    encode_point_tokens_collection_prefix, encode_point_tokens_key, encode_point_tokens_prefix,
    encode_posting_collection_prefix, encode_posting_key, encode_posting_prefix, InvertedMetaKey,
    PointTokensKey, PostingKey, PREFIX_INVERTED_META, PREFIX_POINT_TOKENS, PREFIX_POSTING,
};

pub use collection_vector_keys::{
    decode_collection_vector_entity_id, decode_collection_vector_key, encode_collection_vector_key,
    encode_collection_vector_prefix, encode_entity_vector_prefix, CollectionVectorKey,
    PREFIX_COLLECTION_VECTOR,
};
