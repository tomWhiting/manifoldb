//! Vector search operators.
//!
//! This module provides operators for vector similarity search that can be
//! composed into query pipelines. Operators implement an iterator-like interface
//! for streaming results.
//!
//! # Operators
//!
//! - [`AnnScan`] - Approximate nearest neighbor search using HNSW index
//! - [`ExactKnn`] - Brute force k-NN search for small sets or validation
//! - [`VectorFilter`] - Post-filter vector results by predicates
//!
//! # Iterator Design
//!
//! All operators implement the [`VectorOperator`] trait, which provides a
//! streaming interface for results:
//!
//! ```ignore
//! use manifoldb_vector::ops::{VectorOperator, AnnScan};
//!
//! let mut scan = AnnScan::new(index, query, k)?;
//! while let Some(result) = scan.next()? {
//!     println!("Entity: {:?}, Distance: {}", result.entity_id, result.distance);
//! }
//! ```
//!
//! # Search Modes
//!
//! The operators support two primary search modes:
//!
//! - **Find K nearest**: Return the K closest vectors to the query
//! - **Find within distance**: Return all vectors within distance D of the query
//!
//! # Combining with Graph Traversal
//!
//! Vector operators can be combined with graph traversal to find similar
//! neighbors or filter graph results by vector similarity.

mod ann_scan;
mod exact_knn;
mod filter;

pub use ann_scan::AnnScan;
pub use exact_knn::ExactKnn;
pub use filter::VectorFilter;

use manifoldb_core::EntityId;

use crate::error::VectorError;

/// A match from a vector search operation.
///
/// Contains the entity ID and its distance to the query vector.
#[derive(Debug, Clone, Copy)]
pub struct VectorMatch {
    /// The entity ID of the matching vector.
    pub entity_id: EntityId,
    /// The distance to the query vector (lower is more similar for most metrics).
    pub distance: f32,
}

impl VectorMatch {
    /// Create a new vector match.
    #[must_use]
    pub const fn new(entity_id: EntityId, distance: f32) -> Self {
        Self { entity_id, distance }
    }
}

impl From<crate::index::SearchResult> for VectorMatch {
    fn from(result: crate::index::SearchResult) -> Self {
        Self::new(result.entity_id, result.distance)
    }
}

/// Trait for vector search operators.
///
/// Operators implement an iterator-like interface for streaming vector search
/// results. Unlike standard iterators, the `next` method returns a `Result`
/// to handle storage or computation errors.
pub trait VectorOperator {
    /// Get the next match from the operator.
    ///
    /// Returns `Ok(Some(match))` if a match is available, `Ok(None)` if the
    /// operator is exhausted, or `Err` if an error occurred.
    fn next(&mut self) -> Result<Option<VectorMatch>, VectorError>;

    /// Collect all remaining matches into a vector.
    ///
    /// This consumes the operator and returns all matches.
    fn collect_all(&mut self) -> Result<Vec<VectorMatch>, VectorError> {
        let mut results = Vec::new();
        while let Some(m) = self.next()? {
            results.push(m);
        }
        Ok(results)
    }

    /// Get the dimension of vectors this operator works with.
    fn dimension(&self) -> usize;
}

/// Configuration for search operations.
#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    /// Maximum number of results to return.
    pub k: usize,
    /// Maximum distance threshold (only return results closer than this).
    pub max_distance: Option<f32>,
    /// Beam width for approximate search (HNSW `ef_search` parameter).
    pub ef_search: Option<usize>,
}

impl SearchConfig {
    /// Create a new search configuration for finding K nearest neighbors.
    #[must_use]
    pub const fn k_nearest(k: usize) -> Self {
        Self { k, max_distance: None, ef_search: None }
    }

    /// Create a search configuration for finding all vectors within a distance.
    #[must_use]
    pub const fn within_distance(max_distance: f32) -> Self {
        Self { k: usize::MAX, max_distance: Some(max_distance), ef_search: None }
    }

    /// Set the beam width for approximate search.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Set the maximum number of results.
    #[must_use]
    pub const fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the maximum distance threshold.
    #[must_use]
    pub const fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = Some(max_distance);
        self
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self::k_nearest(10)
    }
}
