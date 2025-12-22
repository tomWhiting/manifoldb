//! Collection metadata, named vector configuration, and programmatic API.
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
//! # Programmatic API
//!
//! The collection API provides a fluent interface for creating collections,
//! managing points, and performing vector search.
//!
//! ## Creating a Collection
//!
//! ```ignore
//! use manifoldb::collection::DistanceMetric;
//!
//! // Create a collection with multiple named vectors
//! let collection = db.create_collection("documents")
//!     .with_dense_vector("text", 768, DistanceMetric::Cosine)
//!     .with_sparse_vector("keywords")
//!     .build()?;
//! ```
//!
//! ## Point Operations
//!
//! ```ignore
//! use manifoldb::collection::{PointStruct, Vector};
//! use serde_json::json;
//!
//! // Upsert a point
//! collection.upsert_point(PointStruct::new(1)
//!     .with_payload(json!({"title": "Rust Book", "category": "programming"}))
//!     .with_vector("text", vec![0.1; 768]))?;
//!
//! // Get a point's payload
//! let payload = collection.get_payload(1.into())?;
//! ```
//!
//! ## Vector Search
//!
//! ```ignore
//! use manifoldb::collection::Filter;
//!
//! // Simple search
//! let results = collection.search("text")
//!     .query(query_vector)
//!     .limit(10)
//!     .execute()?;
//!
//! // Search with filter
//! let results = collection.search("text")
//!     .query(query_vector)
//!     .limit(10)
//!     .filter(Filter::eq("category", "programming"))
//!     .with_payload(true)
//!     .execute()?;
//!
//! // Hybrid search (multiple vectors)
//! let results = collection.hybrid_search()
//!     .query("text", dense_vector, 0.7)
//!     .query("keywords", sparse_vector, 0.3)
//!     .limit(10)
//!     .execute()?;
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

mod builder;
mod config;
mod error;
mod filter;
mod handle;
mod manager;
mod metadata;
mod point;
mod search;

// Configuration types
pub use config::{
    AggregationMethod, BinaryDistanceType, DistanceType, HnswParams, IndexConfig, IndexMethod,
    InvertedIndexParams, VectorConfig, VectorType,
};

// Manager and metadata types
pub use manager::{CollectionError, CollectionManager};
pub use metadata::{Collection, CollectionName, PayloadSchema};

// Programmatic API types
pub use builder::CollectionBuilder;
pub use error::{ApiError, ApiResult};
pub use filter::Filter;
pub use handle::CollectionHandle;
pub use point::{PointStruct, ScoredPoint, Vector};
pub use search::{FusionStrategy, HybridSearchBuilder, SearchBuilder};

// Re-export distance metrics from manifoldb-vector for convenience
pub use manifoldb_vector::distance::sparse::SparseDistanceMetric;
pub use manifoldb_vector::distance::DistanceMetric;
