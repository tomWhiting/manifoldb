//! Approximate Nearest Neighbor scan operator.
//!
//! Uses the HNSW index for fast approximate k-NN search.

use manifoldb_storage::StorageEngine;

use super::{SearchConfig, VectorMatch, VectorOperator};
use crate::error::VectorError;
use crate::index::{HnswIndex, VectorIndex};
use crate::types::Embedding;

/// Approximate nearest neighbor search operator.
///
/// Uses an HNSW index to efficiently find the K nearest neighbors to a query
/// vector. The search is approximate, trading some accuracy for speed.
///
/// # Example
///
/// ```ignore
/// use manifoldb_vector::ops::{AnnScan, VectorOperator, SearchConfig};
///
/// let config = SearchConfig::k_nearest(10).with_ef_search(50);
/// let mut scan = AnnScan::new(&index, query, config)?;
///
/// while let Some(m) = scan.next()? {
///     println!("Entity {:?} at distance {}", m.entity_id, m.distance);
/// }
/// ```
pub struct AnnScan<'a, E: StorageEngine> {
    /// Reference to the HNSW index.
    index: &'a HnswIndex<E>,
    /// Pre-computed search results (buffered).
    results: Vec<VectorMatch>,
    /// Current position in the results.
    position: usize,
    /// The dimension of the index.
    dim: usize,
}

impl<'a, E: StorageEngine> AnnScan<'a, E> {
    /// Create a new ANN scan operator.
    ///
    /// Performs the HNSW search immediately and buffers the results for
    /// iteration. This is because HNSW search is most efficient when
    /// performed as a single operation.
    ///
    /// # Arguments
    ///
    /// * `index` - Reference to the HNSW index
    /// * `query` - The query embedding
    /// * `config` - Search configuration (k, `max_distance`, `ef_search`)
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index.
    pub fn new(
        index: &'a HnswIndex<E>,
        query: &Embedding,
        config: SearchConfig,
    ) -> Result<Self, VectorError> {
        let dim = index.dimension();

        // Validate dimension
        if query.dimension() != dim {
            return Err(VectorError::DimensionMismatch {
                expected: dim,
                actual: query.dimension(),
            });
        }

        // Perform the search
        let search_results = index.search(query, config.k, config.ef_search)?;

        // Convert and filter by max_distance if specified
        let results: Vec<VectorMatch> = search_results
            .into_iter()
            .filter(|r| match config.max_distance {
                Some(max_dist) => r.distance <= max_dist,
                None => true,
            })
            .map(VectorMatch::from)
            .collect();

        Ok(Self { index, results, position: 0, dim })
    }

    /// Create an ANN scan with simple k-nearest configuration.
    ///
    /// Convenience method for common case of finding K nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `index` - Reference to the HNSW index
    /// * `query` - The query embedding
    /// * `k` - Number of nearest neighbors to find
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index.
    pub fn k_nearest(
        index: &'a HnswIndex<E>,
        query: &Embedding,
        k: usize,
    ) -> Result<Self, VectorError> {
        Self::new(index, query, SearchConfig::k_nearest(k))
    }

    /// Create an ANN scan to find vectors within a distance threshold.
    ///
    /// Note: HNSW is optimized for k-NN search. For within-distance queries,
    /// this searches for a large k and filters by distance. Consider using
    /// `ExactKnn` for precise distance-based filtering on small sets.
    ///
    /// # Arguments
    ///
    /// * `index` - Reference to the HNSW index
    /// * `query` - The query embedding
    /// * `max_distance` - Maximum distance threshold
    /// * `max_results` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if the query dimension doesn't match the index.
    pub fn within_distance(
        index: &'a HnswIndex<E>,
        query: &Embedding,
        max_distance: f32,
        max_results: usize,
    ) -> Result<Self, VectorError> {
        Self::new(index, query, SearchConfig::within_distance(max_distance).with_k(max_results))
    }

    /// Get the underlying index.
    #[must_use]
    pub const fn index(&self) -> &'a HnswIndex<E> {
        self.index
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
}

impl<E: StorageEngine> VectorOperator for AnnScan<'_, E> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::DistanceMetric;
    use crate::index::HnswConfig;
    use manifoldb_core::EntityId;
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_embedding(dim: usize, value: f32) -> Embedding {
        Embedding::new(vec![value; dim]).unwrap()
    }

    fn create_test_index() -> HnswIndex<RedbEngine> {
        let engine = RedbEngine::in_memory().unwrap();
        HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, HnswConfig::new(4)).unwrap()
    }

    #[test]
    fn test_ann_scan_empty_index() {
        let index = create_test_index();
        let query = create_test_embedding(4, 1.0);

        let mut scan = AnnScan::k_nearest(&index, &query, 5).unwrap();
        assert!(scan.is_empty());
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_ann_scan_single_result() {
        let mut index = create_test_index();
        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        let query = create_test_embedding(4, 1.0);
        let mut scan = AnnScan::k_nearest(&index, &query, 5).unwrap();

        assert_eq!(scan.len(), 1);
        let result = scan.next().unwrap().unwrap();
        assert_eq!(result.entity_id, EntityId::new(1));
        assert!(result.distance < 1e-6);

        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_ann_scan_multiple_results() {
        let mut index = create_test_index();
        for i in 1..=10 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i), &embedding).unwrap();
        }

        let query = create_test_embedding(4, 5.0);
        let mut scan = AnnScan::k_nearest(&index, &query, 3).unwrap();

        assert_eq!(scan.len(), 3);

        let results = scan.collect_all().unwrap();
        assert_eq!(results.len(), 3);

        // Results should be sorted by distance
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn test_ann_scan_with_max_distance() {
        let mut index = create_test_index();
        for i in 1..=10 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i), &embedding).unwrap();
        }

        let query = create_test_embedding(4, 5.0);
        // Only get results within distance 2.0 (entities 4, 5, 6)
        let mut scan = AnnScan::within_distance(&index, &query, 2.5, 10).unwrap();

        let results = scan.collect_all().unwrap();
        for result in &results {
            assert!(result.distance <= 2.5);
        }
    }

    #[test]
    fn test_ann_scan_dimension_mismatch() {
        let index = create_test_index();
        let query = create_test_embedding(8, 1.0); // Wrong dimension

        let result = AnnScan::k_nearest(&index, &query, 5);
        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ann_scan_peek_and_reset() {
        let mut index = create_test_index();
        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        let query = create_test_embedding(4, 1.0);
        let mut scan = AnnScan::k_nearest(&index, &query, 5).unwrap();

        // Peek shouldn't consume
        let peeked = scan.peek().unwrap();
        assert_eq!(peeked.entity_id, EntityId::new(1));

        // Next should return the same
        let result = scan.next().unwrap().unwrap();
        assert_eq!(result.entity_id, EntityId::new(1));

        // Now exhausted
        assert!(scan.next().unwrap().is_none());

        // Reset and iterate again
        scan.reset();
        let result = scan.next().unwrap().unwrap();
        assert_eq!(result.entity_id, EntityId::new(1));
    }

    #[test]
    fn test_ann_scan_with_ef_search() {
        let mut index = create_test_index();
        for i in 1..=20 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i), &embedding).unwrap();
        }

        let query = create_test_embedding(4, 10.0);
        let config = SearchConfig::k_nearest(5).with_ef_search(100);
        let mut scan = AnnScan::new(&index, &query, config).unwrap();

        let results = scan.collect_all().unwrap();
        assert_eq!(results.len(), 5);
    }
}
