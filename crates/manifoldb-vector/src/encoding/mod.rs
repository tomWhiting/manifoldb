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
//! ## Point collections (Qdrant-style)
//! - `0x20` - Collection metadata: `[0x20][collection_name_hash]`
//! - `0x21` - Point payload: `[0x21][collection_name_hash][point_id]`
//! - `0x22` - Dense vector: `[0x22][collection_name_hash][point_id][vector_name_hash]`
//! - `0x23` - Sparse vector: `[0x23][collection_name_hash][point_id][vector_name_hash]`
//! - `0x24` - Multi-vector: `[0x24][collection_name_hash][point_id][vector_name_hash]`
//!
//! ## Inverted index (sparse vector index)
//! - `0x30` - Posting list: `[0x30][collection_hash][vector_name_hash][token_id]`
//! - `0x31` - Index metadata: `[0x31][collection_hash][vector_name_hash]`
//! - `0x32` - Point tokens: `[0x32][collection_hash][vector_name_hash][point_id]`
//!
//! All numeric values are encoded in big-endian format to preserve sort order.

mod inverted_keys;
mod keys;
mod point_keys;

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

pub use point_keys::{
    // Collection keys
    decode_collection_key,
    // Dense vector keys
    decode_dense_vector_key,
    // Multi-vector keys
    decode_multi_vector_key,
    // Point payload keys
    decode_point_payload_key,
    decode_point_payload_point_id,
    // Sparse vector keys
    decode_sparse_vector_key,
    encode_collection_key,
    encode_collection_prefix,
    encode_dense_vector_collection_prefix,
    encode_dense_vector_key,
    encode_dense_vector_point_prefix,
    encode_multi_vector_collection_prefix,
    encode_multi_vector_key,
    encode_multi_vector_point_prefix,
    encode_point_payload_key,
    encode_point_payload_prefix,
    encode_sparse_vector_collection_prefix,
    encode_sparse_vector_key,
    encode_sparse_vector_point_prefix,
    CollectionKey,
    DenseVectorKey,
    MultiVectorKey,
    PointPayloadKey,
    SparseVectorKey,
    PREFIX_COLLECTION,
    PREFIX_POINT_DENSE_VECTOR,
    PREFIX_POINT_MULTI_VECTOR,
    PREFIX_POINT_PAYLOAD,
    PREFIX_POINT_SPARSE_VECTOR,
};

pub use inverted_keys::{
    decode_inverted_meta_key, decode_point_tokens_key, decode_posting_key,
    encode_inverted_meta_collection_prefix, encode_inverted_meta_key,
    encode_point_tokens_collection_prefix, encode_point_tokens_key, encode_point_tokens_prefix,
    encode_posting_collection_prefix, encode_posting_key, encode_posting_prefix, InvertedMetaKey,
    PointTokensKey, PostingKey, PREFIX_INVERTED_META, PREFIX_POINT_TOKENS, PREFIX_POSTING,
};
