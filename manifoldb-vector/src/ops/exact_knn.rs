//! Exact K-Nearest Neighbors operator.
//!
//! Performs brute force k-NN search by computing distances to all vectors.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use manifoldb_core::EntityId;

use super::{SearchConfig, VectorMatch, VectorOperator};
use crate::distance::{cosine_distance, dot_product, euclidean_distance, DistanceMetric};
use crate::error::VectorError;
use crate::types::Embedding;

/// Exact k-NN search operator using brute force.
///
/// Computes distances to all provided vectors and returns the K nearest.
/// This is useful for:
/// - Small datasets where HNSW overhead isn't justified
/// - Validating HNSW results
/// - Post-filtering a small candidate set from graph traversal
///
/// # Complexity
///
/// O(n * d) where n is the number of vectors and d is the dimension.
/// For large datasets, use [`AnnScan`](super::AnnScan) instead.
///
/// # Example
///
/// ```ignore
/// use manifoldb_vector::ops::{ExactKnn, VectorOperator, SearchConfig};
/// use manifoldb_vector::distance::DistanceMetric;
///
/// let vectors = vec![
///     (EntityId::new(1), embedding1),
///     (EntityId::new(2), embedding2),
/// ];
///
/// let config = SearchConfig::k_nearest(5);
/// let mut knn = ExactKnn::new(vectors, query, DistanceMetric::Cosine, config)?;
///
/// while let Some(m) = knn.next()? {
///     println!("Entity {:?} at distance {}", m.entity_id, m.distance);
/// }
/// ```
pub struct ExactKnn {
    /// Pre-computed and sorted results.
    results: Vec<VectorMatch>,
    /// Current position in results.
    position: usize,
    /// Dimension of the vectors.
    dim: usize,
}

/// Wrapper for max-heap comparison (we want smallest distances first).
#[derive(Debug)]
struct MaxHeapEntry {
    entity_id: EntityId,
    distance: f32,
}

impl PartialEq for MaxHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxHeapEntry {}

impl PartialOrd for MaxHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distances should come first (to be popped)
        // NaN values are treated as equal to maintain a total ordering for the heap.
        // In practice, NaN distances should not occur from valid distance calculations.
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl ExactKnn {
    /// Create a new exact k-NN search operator.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Iterator of (`entity_id`, embedding) pairs to search
    /// * `query` - The query embedding
    /// * `metric` - Distance metric to use
    /// * `config` - Search configuration
    ///
    /// # Errors
    ///
    /// Returns an error if any vector has a dimension mismatch with the query.
    pub fn new<I>(
        vectors: I,
        query: &Embedding,
        metric: DistanceMetric,
        config: SearchConfig,
    ) -> Result<Self, VectorError>
    where
        I: IntoIterator<Item = (EntityId, Embedding)>,
    {
        let dim = query.dimension();
        let query_slice = query.as_slice();
        let k = config.k;
        let max_distance = config.max_distance;

        // Use a max-heap to keep track of k smallest distances
        // Use saturating_add to avoid overflow when k is usize::MAX
        let mut heap: BinaryHeap<MaxHeapEntry> =
            BinaryHeap::with_capacity(k.saturating_add(1).min(1024));

        for (entity_id, embedding) in vectors {
            // Validate dimension
            if embedding.dimension() != dim {
                return Err(VectorError::DimensionMismatch {
                    expected: dim,
                    actual: embedding.dimension(),
                });
            }

            let distance = compute_distance(query_slice, embedding.as_slice(), metric);

            // Skip if exceeds max_distance
            if let Some(max_dist) = max_distance {
                if distance > max_dist {
                    continue;
                }
            }

            // Add to heap if within k or better than worst
            if heap.len() < k {
                heap.push(MaxHeapEntry { entity_id, distance });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(MaxHeapEntry { entity_id, distance });
                }
            }
        }

        // Convert heap to sorted vec
        let mut results: Vec<VectorMatch> =
            heap.into_iter().map(|e| VectorMatch::new(e.entity_id, e.distance)).collect();

        // Sort by distance (ascending). NaN distances are treated as equal to maintain
        // a stable sort order; in practice NaN should not occur from valid calculations.
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

        Ok(Self { results, position: 0, dim })
    }

    /// Create a k-nearest search from an iterator of vectors.
    ///
    /// Convenience method for common case.
    pub fn k_nearest<I>(
        vectors: I,
        query: &Embedding,
        metric: DistanceMetric,
        k: usize,
    ) -> Result<Self, VectorError>
    where
        I: IntoIterator<Item = (EntityId, Embedding)>,
    {
        Self::new(vectors, query, metric, SearchConfig::k_nearest(k))
    }

    /// Create a within-distance search from an iterator of vectors.
    ///
    /// Returns all vectors within the specified distance threshold.
    pub fn within_distance<I>(
        vectors: I,
        query: &Embedding,
        metric: DistanceMetric,
        max_distance: f32,
    ) -> Result<Self, VectorError>
    where
        I: IntoIterator<Item = (EntityId, Embedding)>,
    {
        Self::new(vectors, query, metric, SearchConfig::within_distance(max_distance))
    }

    /// Create from a slice of vectors (borrows and clones).
    ///
    /// Useful when you have a reference to existing data.
    pub fn from_slice(
        vectors: &[(EntityId, Embedding)],
        query: &Embedding,
        metric: DistanceMetric,
        config: SearchConfig,
    ) -> Result<Self, VectorError> {
        Self::new(vectors.iter().cloned(), query, metric, config)
    }

    /// Get the number of results found.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if no results were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Peek at the next result without consuming it.
    #[must_use]
    pub fn peek(&self) -> Option<&VectorMatch> {
        self.results.get(self.position)
    }

    /// Reset the iterator to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get all results as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[VectorMatch] {
        &self.results
    }
}

impl VectorOperator for ExactKnn {
    fn next(&mut self) -> Result<Option<VectorMatch>, VectorError> {
        if self.position < self.results.len() {
            let result = self.results[self.position];
            self.position += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Compute distance between two vectors using the specified metric.
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::DotProduct => -dot_product(a, b), // Negate for min-distance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding(dim: usize, value: f32) -> Embedding {
        Embedding::new(vec![value; dim]).unwrap()
    }

    fn create_test_vectors(count: usize) -> Vec<(EntityId, Embedding)> {
        (1..=count).map(|i| (EntityId::new(i as u64), create_test_embedding(4, i as f32))).collect()
    }

    #[test]
    fn test_exact_knn_empty() {
        let query = create_test_embedding(4, 1.0);
        let vectors: Vec<(EntityId, Embedding)> = vec![];

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5).unwrap();

        assert!(knn.is_empty());
        assert!(knn.next().unwrap().is_none());
    }

    #[test]
    fn test_exact_knn_single() {
        let query = create_test_embedding(4, 1.0);
        let vectors = vec![(EntityId::new(1), create_test_embedding(4, 1.0))];

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5).unwrap();

        assert_eq!(knn.len(), 1);
        let result = knn.next().unwrap().unwrap();
        assert_eq!(result.entity_id, EntityId::new(1));
        assert!(result.distance < 1e-6);
    }

    #[test]
    fn test_exact_knn_k_smaller_than_n() {
        let query = create_test_embedding(4, 5.0);
        let vectors = create_test_vectors(10);

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 3).unwrap();

        let results = knn.collect_all().unwrap();
        assert_eq!(results.len(), 3);

        // Results should be sorted by distance
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);

        // Closest should be entity 5 (same value as query)
        assert_eq!(results[0].entity_id, EntityId::new(5));
    }

    #[test]
    fn test_exact_knn_k_larger_than_n() {
        let query = create_test_embedding(4, 1.0);
        let vectors = create_test_vectors(3);

        let knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 10)
            .unwrap();

        assert_eq!(knn.len(), 3);
    }

    #[test]
    fn test_exact_knn_with_max_distance() {
        let query = create_test_embedding(4, 5.0);
        let vectors = create_test_vectors(10);

        let mut knn =
            ExactKnn::within_distance(vectors.into_iter(), &query, DistanceMetric::Euclidean, 2.5)
                .unwrap();

        let results = knn.collect_all().unwrap();
        for result in &results {
            assert!(result.distance <= 2.5);
        }
    }

    #[test]
    fn test_exact_knn_cosine_distance() {
        let query = Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let vectors = vec![
            (EntityId::new(1), Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap()), // Same direction
            (EntityId::new(2), Embedding::new(vec![0.0, 1.0, 0.0, 0.0]).unwrap()), // Orthogonal
            (EntityId::new(3), Embedding::new(vec![-1.0, 0.0, 0.0, 0.0]).unwrap()), // Opposite
        ];

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Cosine, 3).unwrap();

        let results = knn.collect_all().unwrap();

        // Entity 1 should be closest (cosine distance = 0)
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert!(results[0].distance < 1e-6);

        // Entity 2 should be next (cosine distance = 1)
        assert_eq!(results[1].entity_id, EntityId::new(2));
        assert!((results[1].distance - 1.0).abs() < 1e-6);

        // Entity 3 should be furthest (cosine distance = 2)
        assert_eq!(results[2].entity_id, EntityId::new(3));
        assert!((results[2].distance - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_exact_knn_dot_product() {
        let query = Embedding::new(vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let vectors = vec![
            (EntityId::new(1), Embedding::new(vec![2.0, 2.0, 0.0, 0.0]).unwrap()), // Dot = 4
            (EntityId::new(2), Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap()), // Dot = 1
            (EntityId::new(3), Embedding::new(vec![0.0, 0.0, 1.0, 1.0]).unwrap()), // Dot = 0
        ];

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::DotProduct, 3)
                .unwrap();

        let results = knn.collect_all().unwrap();

        // Entity 1 should be closest (highest dot product = -4 distance)
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert!((results[0].distance - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_exact_knn_dimension_mismatch() {
        let query = create_test_embedding(4, 1.0);
        let vectors = vec![(EntityId::new(1), create_test_embedding(8, 1.0))]; // Wrong dimension

        let result = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5);

        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_exact_knn_from_slice() {
        let query = create_test_embedding(4, 5.0);
        let vectors = create_test_vectors(10);

        let mut knn = ExactKnn::from_slice(
            &vectors,
            &query,
            DistanceMetric::Euclidean,
            SearchConfig::k_nearest(3),
        )
        .unwrap();

        assert_eq!(knn.len(), 3);
        assert_eq!(knn.collect_all().unwrap()[0].entity_id, EntityId::new(5));
    }

    #[test]
    fn test_exact_knn_peek_and_reset() {
        let query = create_test_embedding(4, 1.0);
        let vectors = create_test_vectors(3);

        let mut knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5).unwrap();

        let first_id = knn.peek().unwrap().entity_id;
        assert_eq!(knn.next().unwrap().unwrap().entity_id, first_id);

        // Exhaust
        while knn.next().unwrap().is_some() {}

        // Reset and check first again
        knn.reset();
        assert_eq!(knn.peek().unwrap().entity_id, first_id);
    }

    #[test]
    fn test_exact_knn_as_slice() {
        let query = create_test_embedding(4, 1.0);
        let vectors = create_test_vectors(5);

        let knn =
            ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5).unwrap();

        let slice = knn.as_slice();
        assert_eq!(slice.len(), 5);
        // First should be closest to query value 1.0
        assert_eq!(slice[0].entity_id, EntityId::new(1));
    }
}
