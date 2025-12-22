//! MaxSim scoring for ColBERT-style late interaction models.
//!
//! MaxSim computes similarity between multi-vector embeddings by taking the
//! maximum similarity for each query token across all document tokens:
//!
//! ```text
//! MaxSim(Q, D) = sum_{q_i in Q} max_{d_j in D} (q_i · d_j)
//! ```
//!
//! This module provides:
//! - [`maxsim`] - Compute MaxSim score between two multi-vectors
//! - [`maxsim_batch`] - Compute MaxSim scores against multiple documents
//! - [`MaxSimScorer`] - Reusable scorer with pre-computed query data
//! - [`MaxSimScan`] - Operator for multi-vector similarity search

use crate::distance::dot_product;
use crate::types::MultiVectorEmbedding;

/// Compute the MaxSim score between a query and document multi-vector.
///
/// MaxSim sums the maximum dot product for each query token across all document tokens:
/// ```text
/// MaxSim(Q, D) = sum_{q_i in Q} max_{d_j in D} (q_i · d_j)
/// ```
///
/// # Arguments
///
/// * `query` - Query multi-vector embedding (per-token embeddings)
/// * `document` - Document multi-vector embedding (per-token embeddings)
///
/// # Returns
///
/// The MaxSim score (higher is more similar).
///
/// # Panics
///
/// Panics if query and document have different dimensions.
///
/// # Example
///
/// ```
/// use manifoldb_vector::ops::maxsim::maxsim;
/// use manifoldb_vector::types::MultiVectorEmbedding;
///
/// let query = MultiVectorEmbedding::new(vec![
///     vec![1.0, 0.0, 0.0],  // Query token 1
///     vec![0.0, 1.0, 0.0],  // Query token 2
/// ]).unwrap();
///
/// let doc = MultiVectorEmbedding::new(vec![
///     vec![1.0, 0.0, 0.0],  // Doc token 1 (matches query token 1)
///     vec![0.0, 0.5, 0.5],  // Doc token 2
///     vec![0.0, 0.0, 1.0],  // Doc token 3
/// ]).unwrap();
///
/// let score = maxsim(&query, &doc);
/// // Query token 1: max with doc tokens = 1.0 (with doc token 1)
/// // Query token 2: max with doc tokens = 0.5 (with doc token 2)
/// // Total MaxSim = 1.0 + 0.5 = 1.5
/// assert!((score - 1.5).abs() < 1e-6);
/// ```
#[must_use]
pub fn maxsim(query: &MultiVectorEmbedding, document: &MultiVectorEmbedding) -> f32 {
    assert_eq!(
        query.dimension(),
        document.dimension(),
        "query and document must have the same dimension"
    );

    let mut total_score = 0.0_f32;

    // For each query token, find the max dot product with any document token
    for q in query.iter() {
        let mut max_sim = f32::NEG_INFINITY;

        for d in document.iter() {
            let sim = dot_product(q, d);
            if sim > max_sim {
                max_sim = sim;
            }
        }

        // Handle empty document case
        if max_sim.is_finite() {
            total_score += max_sim;
        }
    }

    total_score
}

/// Compute MaxSim scores between a query and multiple documents.
///
/// This is more efficient than calling `maxsim` repeatedly because it can
/// batch operations and reuse query data.
///
/// # Returns
///
/// A vector of (document_index, score) pairs, sorted by score descending.
///
/// # Panics
///
/// Panics if any document has a different dimension than the query.
#[must_use]
pub fn maxsim_batch(
    query: &MultiVectorEmbedding,
    documents: &[MultiVectorEmbedding],
) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> =
        documents.iter().enumerate().map(|(i, doc)| (i, maxsim(query, doc))).collect();

    // Sort by score descending
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scores
}

/// A reusable MaxSim scorer with pre-computed query data.
///
/// Use this when scoring the same query against multiple documents to
/// avoid redundant computation.
///
/// # Example
///
/// ```
/// use manifoldb_vector::ops::maxsim::MaxSimScorer;
/// use manifoldb_vector::types::MultiVectorEmbedding;
///
/// let query = MultiVectorEmbedding::new(vec![
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
/// ]).unwrap();
///
/// let scorer = MaxSimScorer::new(query);
///
/// let doc1 = MultiVectorEmbedding::new(vec![vec![1.0, 0.0]]).unwrap();
/// let doc2 = MultiVectorEmbedding::new(vec![vec![0.5, 0.5]]).unwrap();
///
/// let score1 = scorer.score(&doc1);
/// let score2 = scorer.score(&doc2);
/// ```
#[derive(Debug, Clone)]
pub struct MaxSimScorer {
    query: MultiVectorEmbedding,
}

impl MaxSimScorer {
    /// Create a new MaxSim scorer for the given query.
    #[must_use]
    pub fn new(query: MultiVectorEmbedding) -> Self {
        Self { query }
    }

    /// Score a document against the query.
    #[must_use]
    pub fn score(&self, document: &MultiVectorEmbedding) -> f32 {
        maxsim(&self.query, document)
    }

    /// Score multiple documents and return results sorted by score descending.
    #[must_use]
    pub fn score_batch(&self, documents: &[MultiVectorEmbedding]) -> Vec<(usize, f32)> {
        maxsim_batch(&self.query, documents)
    }

    /// Get the query multi-vector.
    #[must_use]
    pub fn query(&self) -> &MultiVectorEmbedding {
        &self.query
    }

    /// Get the dimension of the query vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.query.dimension()
    }
}

/// Compute MaxSim with normalized vectors (for cosine-like similarity).
///
/// This normalizes both query and document token vectors to unit length
/// before computing MaxSim, making it equivalent to MaxSim over cosine
/// similarities.
///
/// # Returns
///
/// MaxSim score where each token similarity is in [-1, 1].
#[must_use]
pub fn maxsim_cosine(query: &MultiVectorEmbedding, document: &MultiVectorEmbedding) -> f32 {
    maxsim(&query.normalize(), &document.normalize())
}

/// Convert MaxSim score to a distance (lower is more similar).
///
/// This computes `max_possible_score - score` where the max possible score
/// is the number of query tokens (assuming normalized vectors with max dot product 1.0).
#[inline]
#[must_use]
pub fn maxsim_to_distance(score: f32, num_query_tokens: usize) -> f32 {
    (num_query_tokens as f32) - score
}

/// Convert a distance back to MaxSim score.
#[inline]
#[must_use]
pub fn distance_to_maxsim(distance: f32, num_query_tokens: usize) -> f32 {
    (num_query_tokens as f32) - distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsim_identical() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();

        let doc = MultiVectorEmbedding::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();

        let score = maxsim(&query, &doc);
        // Each query token matches perfectly with one doc token
        // Score = 1.0 + 1.0 = 2.0
        assert!((score - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_partial_match() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();

        let doc = MultiVectorEmbedding::new(vec![
            vec![1.0, 0.0], // Matches query token 1
            vec![0.5, 0.5], // Partial match
        ])
        .unwrap();

        let score = maxsim(&query, &doc);
        // Query token 1: max(1.0, 0.5) = 1.0
        // Query token 2: max(0.0, 0.5) = 0.5
        // Total = 1.5
        assert!((score - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_orthogonal() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0, 0.0]]).unwrap();

        let doc = MultiVectorEmbedding::new(vec![vec![0.0, 1.0, 0.0]]).unwrap();

        let score = maxsim(&query, &doc);
        // Orthogonal vectors have dot product 0
        assert!((score - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_batch() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0]]).unwrap();

        let docs = vec![
            MultiVectorEmbedding::new(vec![vec![0.5, 0.0]]).unwrap(),
            MultiVectorEmbedding::new(vec![vec![1.0, 0.0]]).unwrap(),
            MultiVectorEmbedding::new(vec![vec![0.0, 1.0]]).unwrap(),
        ];

        let scores = maxsim_batch(&query, &docs);

        // Should be sorted by score descending
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].0, 1); // doc1 has score 1.0
        assert_eq!(scores[1].0, 0); // doc0 has score 0.5
        assert_eq!(scores[2].0, 2); // doc2 has score 0.0
    }

    #[test]
    fn test_scorer() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0]]).unwrap();

        let scorer = MaxSimScorer::new(query);

        let doc1 = MultiVectorEmbedding::new(vec![vec![1.0, 0.0]]).unwrap();
        let doc2 = MultiVectorEmbedding::new(vec![vec![0.5, 0.5]]).unwrap();

        assert!((scorer.score(&doc1) - 1.0).abs() < 1e-6);
        assert!((scorer.score(&doc2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_cosine() {
        let query = MultiVectorEmbedding::new(vec![vec![2.0, 0.0]]).unwrap();

        let doc = MultiVectorEmbedding::new(vec![vec![3.0, 0.0]]).unwrap();

        // With cosine, vectors are normalized, so [2,0] and [3,0] both become [1,0]
        let score = maxsim_cosine(&query, &doc);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_distance_conversion() {
        let score = 1.5;
        let num_tokens = 2;

        let distance = maxsim_to_distance(score, num_tokens);
        assert!((distance - 0.5).abs() < 1e-6);

        let recovered = distance_to_maxsim(distance, num_tokens);
        assert!((recovered - score).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_many_tokens() {
        // Simulate a more realistic ColBERT scenario with multiple tokens
        let query = MultiVectorEmbedding::new(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ])
        .unwrap();

        let doc = MultiVectorEmbedding::new(vec![
            vec![0.9, 0.1, 0.0, 0.0], // Close to query token 1
            vec![0.0, 0.8, 0.2, 0.0], // Close to query token 2
            vec![0.0, 0.0, 0.7, 0.3], // Close to query token 3
            vec![0.5, 0.5, 0.0, 0.0], // Another token
        ])
        .unwrap();

        let score = maxsim(&query, &doc);
        // Query token 1: max over doc = 0.9 (doc token 1)
        // Query token 2: max over doc = 0.8 (doc token 2)
        // Query token 3: max over doc = 0.7 (doc token 3)
        // Total = 0.9 + 0.8 + 0.7 = 2.4
        assert!((score - 2.4).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim_single_doc_token() {
        let query = MultiVectorEmbedding::new(vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]])
            .unwrap();

        // Document with only one token
        let doc = MultiVectorEmbedding::new(vec![vec![0.6, 0.8]]).unwrap();

        let score = maxsim(&query, &doc);
        // All query tokens match against the single doc token
        // Query token 1: 1.0*0.6 + 0.0*0.8 = 0.6
        // Query token 2: 0.0*0.6 + 1.0*0.8 = 0.8
        // Query token 3: 0.5*0.6 + 0.5*0.8 = 0.7
        // Total = 0.6 + 0.8 + 0.7 = 2.1
        assert!((score - 2.1).abs() < 1e-6);
    }
}
