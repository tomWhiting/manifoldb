//! Vector result filtering operator.
//!
//! Post-filters vector search results by predicates.

use manifoldb_core::EntityId;

use super::{VectorMatch, VectorOperator};
use crate::error::VectorError;

/// Predicate function type for filtering vector matches.
///
/// Takes a reference to a `VectorMatch` and returns `true` if the match
/// should be included in the results.
pub type FilterPredicate = Box<dyn FnMut(&VectorMatch) -> bool + Send>;

/// Filter operator for post-filtering vector search results.
///
/// Wraps another `VectorOperator` and applies a predicate to filter results.
/// This is useful for:
/// - Filtering by distance threshold after approximate search
/// - Excluding specific entity IDs
/// - Combining vector search with external predicates
///
/// # Example
///
/// ```ignore
/// use manifoldb_vector::ops::{VectorFilter, AnnScan, VectorOperator};
///
/// let mut scan = AnnScan::k_nearest(&index, &query, 100)?;
///
/// // Filter to only include results with distance < 0.5
/// let mut filtered = VectorFilter::by_distance(scan, 0.5);
///
/// while let Some(m) = filtered.next()? {
///     println!("Filtered: {:?} at distance {}", m.entity_id, m.distance);
/// }
/// ```
pub struct VectorFilter<O: VectorOperator> {
    /// The source operator to filter.
    source: O,
    /// The filter predicate.
    predicate: FilterPredicate,
    /// Optional limit on number of results.
    limit: Option<usize>,
    /// Number of results returned so far.
    count: usize,
}

impl<O: VectorOperator> VectorFilter<O> {
    /// Create a new filter operator with a custom predicate.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `predicate` - A function that returns `true` for matches to include
    pub fn new<F>(source: O, predicate: F) -> Self
    where
        F: FnMut(&VectorMatch) -> bool + Send + 'static,
    {
        Self { source, predicate: Box::new(predicate), limit: None, count: 0 }
    }

    /// Create a filter that only includes results below a distance threshold.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `max_distance` - Maximum distance to include (exclusive for equality)
    pub fn by_distance(source: O, max_distance: f32) -> Self {
        Self::new(source, move |m| m.distance <= max_distance)
    }

    /// Create a filter that excludes specific entity IDs.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `exclude` - Set of entity IDs to exclude
    pub fn exclude_entities(source: O, exclude: Vec<EntityId>) -> Self {
        Self::new(source, move |m| !exclude.contains(&m.entity_id))
    }

    /// Create a filter that only includes specific entity IDs.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `include` - Set of entity IDs to include
    pub fn include_entities(source: O, include: Vec<EntityId>) -> Self {
        Self::new(source, move |m| include.contains(&m.entity_id))
    }

    /// Create a filter that limits the number of results.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `limit` - Maximum number of results to return
    pub fn take(source: O, limit: usize) -> Self {
        Self { source, predicate: Box::new(|_| true), limit: Some(limit), count: 0 }
    }

    /// Create a filter with a distance range.
    ///
    /// # Arguments
    ///
    /// * `source` - The source operator to filter
    /// * `min_distance` - Minimum distance to include (exclusive)
    /// * `max_distance` - Maximum distance to include (inclusive)
    pub fn by_distance_range(source: O, min_distance: f32, max_distance: f32) -> Self {
        Self::new(source, move |m| m.distance > min_distance && m.distance <= max_distance)
    }

    /// Add a limit to this filter.
    ///
    /// After the limit is reached, `next()` will return `None`.
    #[must_use]
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Chain another predicate with AND logic.
    ///
    /// Both the existing predicate and the new one must return `true`
    /// for a match to be included.
    #[must_use]
    pub fn and<F>(self, mut other_predicate: F) -> Self
    where
        F: FnMut(&VectorMatch) -> bool + Send + 'static,
    {
        let mut current_predicate = self.predicate;
        Self {
            source: self.source,
            predicate: Box::new(move |m| current_predicate(m) && other_predicate(m)),
            limit: self.limit,
            count: self.count,
        }
    }

    /// Get a reference to the source operator.
    #[must_use]
    pub const fn source(&self) -> &O {
        &self.source
    }

    /// Get the number of results returned so far.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

impl<O: VectorOperator> VectorOperator for VectorFilter<O> {
    fn next(&mut self) -> Result<Option<VectorMatch>, VectorError> {
        // Check limit
        if let Some(limit) = self.limit {
            if self.count >= limit {
                return Ok(None);
            }
        }

        // Keep getting from source until we find a match or exhaust
        while let Some(m) = self.source.next()? {
            if (self.predicate)(&m) {
                self.count += 1;
                return Ok(Some(m));
            }
        }

        Ok(None)
    }

    fn dimension(&self) -> usize {
        self.source.dimension()
    }
}

/// Builder for composing multiple filters.
///
/// Provides a fluent API for building complex filter chains.
#[allow(dead_code)]
pub struct FilterBuilder<O: VectorOperator> {
    filter: VectorFilter<O>,
}

#[allow(dead_code)]
impl<O: VectorOperator> FilterBuilder<O> {
    /// Start building a filter from a source operator.
    pub fn new(source: O) -> Self {
        Self { filter: VectorFilter::new(source, |_| true) }
    }

    /// Add a distance threshold filter.
    #[must_use]
    pub fn max_distance(self, max_distance: f32) -> Self {
        Self { filter: self.filter.and(move |m| m.distance <= max_distance) }
    }

    /// Add a minimum distance filter.
    #[must_use]
    pub fn min_distance(self, min_distance: f32) -> Self {
        Self { filter: self.filter.and(move |m| m.distance > min_distance) }
    }

    /// Exclude specific entities.
    #[must_use]
    pub fn exclude(self, entities: Vec<EntityId>) -> Self {
        Self { filter: self.filter.and(move |m| !entities.contains(&m.entity_id)) }
    }

    /// Include only specific entities.
    #[must_use]
    pub fn include_only(self, entities: Vec<EntityId>) -> Self {
        Self { filter: self.filter.and(move |m| entities.contains(&m.entity_id)) }
    }

    /// Limit the number of results.
    #[must_use]
    pub fn limit(self, limit: usize) -> Self {
        Self { filter: self.filter.with_limit(limit) }
    }

    /// Add a custom predicate.
    pub fn predicate<F>(self, predicate: F) -> Self
    where
        F: FnMut(&VectorMatch) -> bool + Send + 'static,
    {
        Self { filter: self.filter.and(predicate) }
    }

    /// Build the final filter operator.
    #[must_use]
    pub fn build(self) -> VectorFilter<O> {
        self.filter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple mock operator for testing.
    struct MockOperator {
        results: Vec<VectorMatch>,
        position: usize,
        dim: usize,
    }

    impl MockOperator {
        fn new(results: Vec<VectorMatch>) -> Self {
            Self { results, position: 0, dim: 4 }
        }
    }

    impl VectorOperator for MockOperator {
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

    fn create_test_results() -> Vec<VectorMatch> {
        vec![
            VectorMatch::new(EntityId::new(1), 0.1),
            VectorMatch::new(EntityId::new(2), 0.3),
            VectorMatch::new(EntityId::new(3), 0.5),
            VectorMatch::new(EntityId::new(4), 0.7),
            VectorMatch::new(EntityId::new(5), 0.9),
        ]
    }

    #[test]
    fn test_filter_by_distance() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::by_distance(source, 0.5);

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 3); // 0.1, 0.3, 0.5 are <= 0.5
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert_eq!(results[1].entity_id, EntityId::new(2));
        assert_eq!(results[2].entity_id, EntityId::new(3));
    }

    #[test]
    fn test_filter_exclude_entities() {
        let source = MockOperator::new(create_test_results());
        let mut filter =
            VectorFilter::exclude_entities(source, vec![EntityId::new(2), EntityId::new(4)]);

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 3);

        let ids: Vec<u64> = results.iter().map(|m| m.entity_id.as_u64()).collect();
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&4));
    }

    #[test]
    fn test_filter_include_entities() {
        let source = MockOperator::new(create_test_results());
        let mut filter =
            VectorFilter::include_entities(source, vec![EntityId::new(1), EntityId::new(3)]);

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert_eq!(results[1].entity_id, EntityId::new(3));
    }

    #[test]
    fn test_filter_take() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::take(source, 2);

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_distance_range() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::by_distance_range(source, 0.2, 0.7);

        let results = filter.collect_all().unwrap();
        // 0.3, 0.5, 0.7 are in range (0.2, 0.7]
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].entity_id, EntityId::new(2)); // 0.3
        assert_eq!(results[1].entity_id, EntityId::new(3)); // 0.5
        assert_eq!(results[2].entity_id, EntityId::new(4)); // 0.7
    }

    #[test]
    fn test_filter_with_limit() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::by_distance(source, 1.0).with_limit(2);

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_and_chain() {
        let source = MockOperator::new(create_test_results());
        let mut filter =
            VectorFilter::by_distance(source, 0.8).and(|m| m.entity_id != EntityId::new(2));

        let results = filter.collect_all().unwrap();
        // 0.1, 0.3, 0.5, 0.7 are <= 0.8, minus entity 2
        assert_eq!(results.len(), 3);

        let ids: Vec<u64> = results.iter().map(|m| m.entity_id.as_u64()).collect();
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_filter_empty_source() {
        let source = MockOperator::new(vec![]);
        let mut filter = VectorFilter::by_distance(source, 1.0);

        assert!(filter.next().unwrap().is_none());
    }

    #[test]
    fn test_filter_all_excluded() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::by_distance(source, 0.0); // Nothing at distance 0

        assert!(filter.next().unwrap().is_none());
    }

    #[test]
    fn test_filter_count() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::take(source, 3);

        assert_eq!(filter.count(), 0);
        let _ = filter.next().unwrap();
        assert_eq!(filter.count(), 1);
        let _ = filter.collect_all().unwrap();
        assert_eq!(filter.count(), 3);
    }

    #[test]
    fn test_filter_builder() {
        let source = MockOperator::new(create_test_results());
        let mut filter = FilterBuilder::new(source)
            .max_distance(0.8)
            .exclude(vec![EntityId::new(2)])
            .limit(2)
            .build();

        let results = filter.collect_all().unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert_eq!(results[1].entity_id, EntityId::new(3));
    }

    #[test]
    fn test_filter_builder_min_max_distance() {
        let source = MockOperator::new(create_test_results());
        let mut filter = FilterBuilder::new(source).min_distance(0.2).max_distance(0.6).build();

        let results = filter.collect_all().unwrap();
        // 0.3, 0.5 are in (0.2, 0.6]
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_dimension() {
        let source = MockOperator::new(create_test_results());
        let filter = VectorFilter::by_distance(source, 1.0);

        assert_eq!(filter.dimension(), 4);
    }

    #[test]
    fn test_filter_custom_predicate() {
        let source = MockOperator::new(create_test_results());
        let mut filter = VectorFilter::new(source, |m| m.entity_id.as_u64() % 2 == 1);

        let results = filter.collect_all().unwrap();
        // Entities 1, 3, 5 (odd)
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|m| m.entity_id.as_u64() % 2 == 1));
    }
}
