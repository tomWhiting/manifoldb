//! Collection metadata and configuration types.
//!
//! Collections are named containers for entities with associated vector
//! configurations. Each collection defines named vector spaces with their
//! dimensions, distance metrics, and index configurations.
//!
//! # Overview
//!
//! A collection in ManifoldDB is similar to a table in relational databases,
//! but with built-in support for multiple named vectors per entity. This allows
//! storing and querying different embedding types together:
//!
//! - Dense vectors (e.g., BERT, OpenAI embeddings)
//! - Sparse vectors (e.g., SPLADE, BM25)
//! - Multi-vectors (e.g., ColBERT token embeddings)
//! - Binary vectors (e.g., LSH, SimHash)
//!
//! # DDL Syntax
//!
//! Collections can be created using SQL-like DDL:
//!
//! ```sql
//! CREATE COLLECTION documents (
//!     dense VECTOR(768) USING hnsw WITH (distance = 'cosine'),
//!     sparse SPARSE_VECTOR USING inverted,
//!     colbert MULTI_VECTOR(128) USING hnsw WITH (aggregation = 'maxsim')
//! );
//! ```
//!
//! # Vector Search via GraphQL
//!
//! Vector search is performed through the GraphQL API using the `searchVectors`
//! and `upsertVector` mutations. See the GraphQL schema documentation for details.

mod config;
mod error;
mod filter;
mod manager;
mod metadata;
mod point;

// Configuration types
pub use config::{
    AggregationMethod, BinaryDistanceType, DistanceType, HnswParams, IndexConfig, IndexMethod,
    InvertedIndexParams, VectorConfig, VectorType,
};

// Manager and metadata types
pub use manager::{CollectionError, CollectionManager};
pub use metadata::{Collection, CollectionName, PayloadSchema};

// Error types
pub use error::{ApiError, ApiResult};

// Filter and point types (used by GraphQL API)
pub use filter::Filter;
pub use point::{PointStruct, ScoredPoint, Vector};

// Re-export distance metrics from manifoldb-vector for convenience
pub use manifoldb_vector::distance::sparse::SparseDistanceMetric;
pub use manifoldb_vector::distance::DistanceMetric;
