//! Hybrid dense+sparse vector search.
//!
//! This module provides support for combining dense and sparse vector similarity
//! scores using weighted combinations. This is useful for hybrid retrieval systems
//! that combine semantic (dense) and lexical (sparse) similarity.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_vector::ops::{HybridScore, HybridSearch};
//!
//! // Create a hybrid search with 0.7 weight for dense and 0.3 for sparse
//! let hybrid = HybridSearch::new(0.7, 0.3);
//!
//! // Combine scores (lower is better for distance-based scores)
//! let combined = hybrid.combine_scores(0.1, 0.2);
//! ```

use std::collections::HashMap;

use manifoldb_core::EntityId;

use super::VectorMatch;

/// Configuration for hybrid dense+sparse vector search.
///
/// Combines dense (semantic) and sparse (lexical) similarity scores
/// using a weighted combination.
#[derive(Debug, Clone, Copy)]
pub struct HybridConfig {
    /// Weight for dense vector scores (0.0 to 1.0).
    pub dense_weight: f32,
    /// Weight for sparse vector scores (0.0 to 1.0).
    pub sparse_weight: f32,
    /// Whether to normalize scores before combining (recommended).
    pub normalize: bool,
}

impl HybridConfig {
    /// Create a new hybrid configuration with the given weights.
    ///
    /// Weights should sum to 1.0 for proper score interpretation,
    /// but this is not enforced.
    ///
    /// # Example
    ///
    /// ```
    /// use manifoldb_vector::ops::hybrid::HybridConfig;
    ///
    /// // 70% dense, 30% sparse
    /// let config = HybridConfig::new(0.7, 0.3);
    /// assert!((config.dense_weight - 0.7).abs() < 1e-6);
    /// ```
    #[must_use]
    pub const fn new(dense_weight: f32, sparse_weight: f32) -> Self {
        Self { dense_weight, sparse_weight, normalize: true }
    }

    /// Create a dense-only configuration.
    #[must_use]
    pub const fn dense_only() -> Self {
        Self::new(1.0, 0.0)
    }

    /// Create a sparse-only configuration.
    #[must_use]
    pub const fn sparse_only() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Create an equal weighting configuration.
    #[must_use]
    pub const fn equal() -> Self {
        Self::new(0.5, 0.5)
    }

    /// Disable score normalization.
    #[must_use]
    pub const fn without_normalization(mut self) -> Self {
        self.normalize = false;
        self
    }

    /// Combine dense and sparse distance scores.
    ///
    /// For distance-based metrics (where lower is better), this computes
    /// a weighted sum: `dense_weight * dense_score + sparse_weight * sparse_score`
    ///
    /// # Arguments
    ///
    /// * `dense_score` - Distance score from dense vector search (lower is better)
    /// * `sparse_score` - Distance score from sparse vector search (lower is better)
    #[inline]
    #[must_use]
    pub fn combine_distances(&self, dense_score: f32, sparse_score: f32) -> f32 {
        self.dense_weight * dense_score + self.sparse_weight * sparse_score
    }

    /// Combine dense and sparse similarity scores.
    ///
    /// For similarity-based metrics (where higher is better), this computes
    /// a weighted sum: `dense_weight * dense_sim + sparse_weight * sparse_sim`
    ///
    /// The result is then converted to a distance (1 - similarity).
    #[inline]
    #[must_use]
    pub fn combine_similarities(&self, dense_sim: f32, sparse_sim: f32) -> f32 {
        let combined_sim = self.dense_weight * dense_sim + self.sparse_weight * sparse_sim;
        1.0 - combined_sim
    }
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self::equal()
    }
}

/// Result of a hybrid search operation.
#[derive(Debug, Clone, Copy)]
pub struct HybridMatch {
    /// The entity ID of the matching vector.
    pub entity_id: EntityId,
    /// The combined distance score.
    pub combined_distance: f32,
    /// The dense vector distance (if available).
    pub dense_distance: Option<f32>,
    /// The sparse vector distance (if available).
    pub sparse_distance: Option<f32>,
}

impl HybridMatch {
    /// Create a new hybrid match with both dense and sparse scores.
    #[must_use]
    pub const fn new(
        entity_id: EntityId,
        combined_distance: f32,
        dense_distance: Option<f32>,
        sparse_distance: Option<f32>,
    ) -> Self {
        Self { entity_id, combined_distance, dense_distance, sparse_distance }
    }

    /// Create a dense-only match.
    #[must_use]
    pub const fn dense_only(entity_id: EntityId, distance: f32) -> Self {
        Self::new(entity_id, distance, Some(distance), None)
    }

    /// Create a sparse-only match.
    #[must_use]
    pub const fn sparse_only(entity_id: EntityId, distance: f32) -> Self {
        Self::new(entity_id, distance, None, Some(distance))
    }
}

impl From<HybridMatch> for VectorMatch {
    fn from(m: HybridMatch) -> Self {
        VectorMatch::new(m.entity_id, m.combined_distance)
    }
}

/// Merge and re-rank results from dense and sparse searches.
///
/// This function takes results from separate dense and sparse searches
/// and combines them using the provided hybrid configuration.
///
/// # Algorithm
///
/// 1. Normalize scores if configured (min-max normalization)
/// 2. For entities with both dense and sparse scores: compute weighted combination
/// 3. For entities with only one score: use that score with its weight (other score = 1.0)
/// 4. Sort by combined score
/// 5. Return top K results
///
/// # Arguments
///
/// * `dense_results` - Results from dense vector search
/// * `sparse_results` - Results from sparse vector search
/// * `config` - Hybrid search configuration
/// * `k` - Maximum number of results to return
pub fn merge_results(
    dense_results: &[VectorMatch],
    sparse_results: &[VectorMatch],
    config: &HybridConfig,
    k: usize,
) -> Vec<HybridMatch> {
    // Collect scores by entity ID
    let mut dense_scores: HashMap<EntityId, f32> = HashMap::new();
    let mut sparse_scores: HashMap<EntityId, f32> = HashMap::new();

    for m in dense_results {
        dense_scores.insert(m.entity_id, m.distance);
    }

    for m in sparse_results {
        sparse_scores.insert(m.entity_id, m.distance);
    }

    // Normalize scores if configured
    let (dense_min, dense_max) = if config.normalize && !dense_results.is_empty() {
        let min = dense_results.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let max = dense_results.iter().map(|m| m.distance).fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    } else {
        (0.0, 1.0)
    };

    let (sparse_min, sparse_max) = if config.normalize && !sparse_results.is_empty() {
        let min = sparse_results.iter().map(|m| m.distance).fold(f32::INFINITY, f32::min);
        let max = sparse_results.iter().map(|m| m.distance).fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    } else {
        (0.0, 1.0)
    };

    // Collect all entity IDs
    let all_entities: Vec<EntityId> = dense_scores
        .keys()
        .chain(sparse_scores.keys())
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Compute combined scores
    let mut results: Vec<HybridMatch> = all_entities
        .into_iter()
        .map(|entity_id| {
            let dense_dist = dense_scores.get(&entity_id).copied();
            let sparse_dist = sparse_scores.get(&entity_id).copied();

            // Normalize scores to [0, 1] range
            let norm_dense = dense_dist.map(|d| {
                if dense_max - dense_min > 0.0 {
                    (d - dense_min) / (dense_max - dense_min)
                } else {
                    0.0
                }
            });

            let norm_sparse = sparse_dist.map(|d| {
                if sparse_max - sparse_min > 0.0 {
                    (d - sparse_min) / (sparse_max - sparse_min)
                } else {
                    0.0
                }
            });

            // Compute combined distance
            // If one score is missing, use 1.0 (worst normalized distance)
            let combined = match (norm_dense, norm_sparse) {
                (Some(d), Some(s)) => config.combine_distances(d, s),
                (Some(d), None) => config.combine_distances(d, 1.0),
                (None, Some(s)) => config.combine_distances(1.0, s),
                (None, None) => 1.0, // Should not happen
            };

            HybridMatch::new(entity_id, combined, dense_dist, sparse_dist)
        })
        .collect();

    // Sort by combined distance (lower is better)
    results.sort_by(|a, b| {
        a.combined_distance.partial_cmp(&b.combined_distance).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top K
    results.truncate(k);
    results
}

/// Reciprocal Rank Fusion (RRF) for combining ranked lists.
///
/// RRF is a simple but effective method for combining multiple ranked lists.
/// Each item's score is computed as the sum of 1/(k + rank) across all lists.
///
/// # Arguments
///
/// * `dense_results` - Results from dense vector search (in rank order)
/// * `sparse_results` - Results from sparse vector search (in rank order)
/// * `k_param` - RRF parameter (typically 60)
/// * `top_k` - Maximum number of results to return
pub fn reciprocal_rank_fusion(
    dense_results: &[VectorMatch],
    sparse_results: &[VectorMatch],
    k_param: u32,
    top_k: usize,
) -> Vec<HybridMatch> {
    let mut rrf_scores: HashMap<EntityId, f32> = HashMap::new();
    let mut dense_distances: HashMap<EntityId, f32> = HashMap::new();
    let mut sparse_distances: HashMap<EntityId, f32> = HashMap::new();

    // Add dense results
    for (rank, m) in dense_results.iter().enumerate() {
        let score = 1.0 / (k_param as f32 + rank as f32 + 1.0);
        *rrf_scores.entry(m.entity_id).or_insert(0.0) += score;
        dense_distances.insert(m.entity_id, m.distance);
    }

    // Add sparse results
    for (rank, m) in sparse_results.iter().enumerate() {
        let score = 1.0 / (k_param as f32 + rank as f32 + 1.0);
        *rrf_scores.entry(m.entity_id).or_insert(0.0) += score;
        sparse_distances.insert(m.entity_id, m.distance);
    }

    // Convert RRF scores to distances (higher RRF score = lower distance)
    let max_score = rrf_scores.values().fold(0.0f32, |a, &b| a.max(b));

    let mut results: Vec<HybridMatch> = rrf_scores
        .into_iter()
        .map(|(entity_id, score)| {
            // Convert score to distance: higher score = lower distance
            let combined_distance = if max_score > 0.0 { 1.0 - (score / max_score) } else { 1.0 };

            HybridMatch::new(
                entity_id,
                combined_distance,
                dense_distances.get(&entity_id).copied(),
                sparse_distances.get(&entity_id).copied(),
            )
        })
        .collect();

    // Sort by combined distance (lower is better)
    results.sort_by(|a, b| {
        a.combined_distance.partial_cmp(&b.combined_distance).unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(top_k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn assert_near(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "assertion failed: {} !~ {} (diff: {})",
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn hybrid_config_weights() {
        let config = HybridConfig::new(0.7, 0.3);
        assert_near(config.dense_weight, 0.7, EPSILON);
        assert_near(config.sparse_weight, 0.3, EPSILON);
    }

    #[test]
    fn hybrid_config_presets() {
        let dense_only = HybridConfig::dense_only();
        assert_near(dense_only.dense_weight, 1.0, EPSILON);
        assert_near(dense_only.sparse_weight, 0.0, EPSILON);

        let sparse_only = HybridConfig::sparse_only();
        assert_near(sparse_only.dense_weight, 0.0, EPSILON);
        assert_near(sparse_only.sparse_weight, 1.0, EPSILON);

        let equal = HybridConfig::equal();
        assert_near(equal.dense_weight, 0.5, EPSILON);
        assert_near(equal.sparse_weight, 0.5, EPSILON);
    }

    #[test]
    fn combine_distances() {
        let config = HybridConfig::new(0.7, 0.3);
        let combined = config.combine_distances(0.1, 0.2);
        // 0.7 * 0.1 + 0.3 * 0.2 = 0.07 + 0.06 = 0.13
        assert_near(combined, 0.13, EPSILON);
    }

    #[test]
    fn combine_similarities() {
        let config = HybridConfig::new(0.7, 0.3);
        let combined = config.combine_similarities(0.9, 0.8);
        // Combined sim = 0.7 * 0.9 + 0.3 * 0.8 = 0.63 + 0.24 = 0.87
        // Distance = 1 - 0.87 = 0.13
        assert_near(combined, 0.13, EPSILON);
    }

    #[test]
    fn merge_results_both_present() {
        let dense =
            vec![VectorMatch::new(EntityId::new(1), 0.1), VectorMatch::new(EntityId::new(2), 0.2)];
        let sparse =
            vec![VectorMatch::new(EntityId::new(1), 0.3), VectorMatch::new(EntityId::new(3), 0.1)];

        let config = HybridConfig::equal().without_normalization();
        let results = merge_results(&dense, &sparse, &config, 10);

        assert_eq!(results.len(), 3);

        // Entity 1 should have both scores
        let e1 = results.iter().find(|m| m.entity_id == EntityId::new(1)).unwrap();
        assert!(e1.dense_distance.is_some());
        assert!(e1.sparse_distance.is_some());
    }

    #[test]
    fn merge_results_respects_k() {
        let dense = vec![
            VectorMatch::new(EntityId::new(1), 0.1),
            VectorMatch::new(EntityId::new(2), 0.2),
            VectorMatch::new(EntityId::new(3), 0.3),
        ];
        let sparse =
            vec![VectorMatch::new(EntityId::new(4), 0.1), VectorMatch::new(EntityId::new(5), 0.2)];

        let config = HybridConfig::equal();
        let results = merge_results(&dense, &sparse, &config, 3);

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn reciprocal_rank_fusion_basic() {
        let dense = vec![
            VectorMatch::new(EntityId::new(1), 0.1),
            VectorMatch::new(EntityId::new(2), 0.2),
            VectorMatch::new(EntityId::new(3), 0.3),
        ];
        let sparse = vec![
            VectorMatch::new(EntityId::new(2), 0.1),
            VectorMatch::new(EntityId::new(1), 0.2),
            VectorMatch::new(EntityId::new(4), 0.3),
        ];

        let results = reciprocal_rank_fusion(&dense, &sparse, 60, 10);

        // Entity 1 and 2 should have higher RRF scores (lower distances)
        // as they appear in both lists
        assert!(results.len() >= 2);

        // First two results should be entities 1 and 2 (in some order)
        let top_two: Vec<EntityId> = results.iter().take(2).map(|m| m.entity_id).collect();
        assert!(top_two.contains(&EntityId::new(1)));
        assert!(top_two.contains(&EntityId::new(2)));
    }

    #[test]
    fn reciprocal_rank_fusion_respects_top_k() {
        let dense: Vec<VectorMatch> =
            (1..=10).map(|i| VectorMatch::new(EntityId::new(i), i as f32 * 0.1)).collect();
        let sparse: Vec<VectorMatch> =
            (5..=15).map(|i| VectorMatch::new(EntityId::new(i), i as f32 * 0.1)).collect();

        let results = reciprocal_rank_fusion(&dense, &sparse, 60, 5);

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn hybrid_match_conversions() {
        let hybrid = HybridMatch::dense_only(EntityId::new(1), 0.5);
        assert!(hybrid.dense_distance.is_some());
        assert!(hybrid.sparse_distance.is_none());

        let hybrid = HybridMatch::sparse_only(EntityId::new(2), 0.3);
        assert!(hybrid.dense_distance.is_none());
        assert!(hybrid.sparse_distance.is_some());

        let vector_match: VectorMatch = hybrid.into();
        assert_eq!(vector_match.entity_id, EntityId::new(2));
    }
}
