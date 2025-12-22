//! Collection metadata and named vector configuration.
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
//! # Example
//!
//! ```ignore
//! use manifoldb::collection::{Collection, VectorConfig, VectorType, IndexConfig};
//! use manifoldb_vector::distance::DistanceMetric;
//!
//! // Create a collection with multiple named vectors
//! let collection = Collection::new("documents")
//!     .with_vector("dense", VectorConfig::dense(768, DistanceMetric::Cosine))
//!     .with_vector("sparse", VectorConfig::sparse(30522))
//!     .with_vector("colbert", VectorConfig::multi_vector(128));
//! ```
//!
//! # DDL Syntax
//!
//! Collections can also be created using SQL-like DDL:
//!
//! ```sql
//! CREATE COLLECTION documents (
//!     dense VECTOR(768) USING hnsw WITH (distance = 'cosine'),
//!     sparse SPARSE_VECTOR USING inverted,
//!     colbert MULTI_VECTOR(128) USING hnsw WITH (aggregation = 'maxsim')
//! );
//! ```

mod config;
mod manager;
mod metadata;

pub use config::{
    AggregationMethod, HnswParams, IndexConfig, IndexMethod, InvertedIndexParams, VectorConfig,
    VectorType,
};
pub use manager::{CollectionError, CollectionManager};
pub use metadata::{Collection, CollectionName, PayloadSchema};
