//! Sparse vector distance functions.
//!
//! This module provides distance and similarity functions optimized for
//! sparse vectors, where only non-zero elements are stored.
//!
//! Sparse vectors are represented as sorted `&[(u32, f32)]` slices where
//! each pair is `(index, value)`.

/// Calculate the dot product between two sparse vectors.
///
/// Both vectors must be sorted by index in ascending order.
/// Time complexity: O(n + m) where n and m are the number of non-zero elements.
///
/// # Example
///
/// ```
/// use manifoldb_vector::distance::sparse::sparse_dot_product;
///
/// let a = [(0, 1.0), (2, 2.0), (5, 3.0)];
/// let b = [(1, 1.0), (2, 2.0), (5, 1.0)];
/// // Only indices 2 and 5 overlap: 2*2 + 3*1 = 7
/// assert!((sparse_dot_product(&a, &b) - 7.0).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn sparse_dot_product(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut result = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        let (idx_a, val_a) = a[i];
        let (idx_b, val_b) = b[j];

        match idx_a.cmp(&idx_b) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result += val_a * val_b;
                i += 1;
                j += 1;
            }
        }
    }

    result
}

/// Calculate the sum of squares (squared L2 norm) of a sparse vector.
#[inline]
#[must_use]
pub fn sparse_sum_of_squares(v: &[(u32, f32)]) -> f32 {
    v.iter().map(|&(_, val)| val * val).sum()
}

/// Calculate the L2 norm (magnitude) of a sparse vector.
#[inline]
#[must_use]
pub fn sparse_l2_norm(v: &[(u32, f32)]) -> f32 {
    sparse_sum_of_squares(v).sqrt()
}

/// Calculate the cosine similarity between two sparse vectors.
///
/// Returns a value in the range [-1, 1] where:
/// - 1 means identical direction
/// - 0 means orthogonal (no common indices or zero dot product)
/// - -1 means opposite direction
///
/// Returns 0.0 if either vector has zero magnitude.
#[inline]
#[must_use]
pub fn sparse_cosine_similarity(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let dot = sparse_dot_product(a, b);
    let norm_a = sparse_l2_norm(a);
    let norm_b = sparse_l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate the cosine similarity using pre-computed norms.
///
/// Returns `None` if either norm is zero.
#[inline]
#[must_use]
pub fn sparse_cosine_similarity_with_norms(
    a: &[(u32, f32)],
    b: &[(u32, f32)],
    norm_a: f32,
    norm_b: f32,
) -> Option<f32> {
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }

    let dot = sparse_dot_product(a, b);
    Some(dot / (norm_a * norm_b))
}

/// Calculate the cosine distance between two sparse vectors.
///
/// Cosine distance = 1 - cosine_similarity, returning a value in [0, 2].
#[inline]
#[must_use]
pub fn sparse_cosine_distance(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    1.0 - sparse_cosine_similarity(a, b)
}

/// Calculate the squared Euclidean (L2) distance between two sparse vectors.
///
/// For sparse vectors, this is computed as:
/// ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * (a . b)
///
/// This avoids the sqrt operation for cases where only relative distances matter.
#[inline]
#[must_use]
pub fn sparse_euclidean_distance_squared(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let norm_a_sq = sparse_sum_of_squares(a);
    let norm_b_sq = sparse_sum_of_squares(b);
    let dot = sparse_dot_product(a, b);

    // ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
    (norm_a_sq + norm_b_sq - 2.0 * dot).max(0.0)
}

/// Calculate the Euclidean (L2) distance between two sparse vectors.
#[inline]
#[must_use]
pub fn sparse_euclidean_distance(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    sparse_euclidean_distance_squared(a, b).sqrt()
}

/// Pre-computed L2 norm for efficient repeated sparse cosine similarity calculations.
#[derive(Debug, Clone, Copy)]
pub struct SparseCachedNorm {
    /// The squared L2 norm (sum of squares)
    norm_squared: f32,
    /// The L2 norm (square root of sum of squares)
    norm: f32,
}

impl SparseCachedNorm {
    /// Compute and cache the L2 norm of a sparse vector.
    #[must_use]
    pub fn new(v: &[(u32, f32)]) -> Self {
        let norm_squared = sparse_sum_of_squares(v);
        let norm = norm_squared.sqrt();
        Self { norm_squared, norm }
    }

    /// Get the cached L2 norm.
    #[inline]
    #[must_use]
    pub const fn norm(&self) -> f32 {
        self.norm
    }

    /// Get the cached squared L2 norm.
    #[inline]
    #[must_use]
    pub const fn norm_squared(&self) -> f32 {
        self.norm_squared
    }

    /// Check if the vector has zero magnitude.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.norm == 0.0
    }
}

/// Sparse distance metric for comparing sparse vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SparseDistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Dot product (negative, for max similarity).
    DotProduct,
}

impl SparseDistanceMetric {
    /// Calculate the distance between two sparse vectors using this metric.
    #[inline]
    #[must_use]
    pub fn calculate(&self, a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
        match self {
            Self::Euclidean => sparse_euclidean_distance(a, b),
            Self::Cosine => sparse_cosine_distance(a, b),
            Self::DotProduct => -sparse_dot_product(a, b),
        }
    }

    /// Calculate distance using cached norms (more efficient for repeated queries).
    #[inline]
    #[must_use]
    pub fn calculate_with_norms(
        &self,
        a: &[(u32, f32)],
        b: &[(u32, f32)],
        norm_a: f32,
        norm_b: f32,
    ) -> f32 {
        match self {
            Self::Euclidean => {
                // ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a . b)
                let norm_a_sq = norm_a * norm_a;
                let norm_b_sq = norm_b * norm_b;
                let dot = sparse_dot_product(a, b);
                (norm_a_sq + norm_b_sq - 2.0 * dot).max(0.0).sqrt()
            }
            Self::Cosine => {
                sparse_cosine_similarity_with_norms(a, b, norm_a, norm_b).map_or(1.0, |s| 1.0 - s)
            }
            Self::DotProduct => -sparse_dot_product(a, b),
        }
    }
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
    fn test_sparse_dot_product_no_overlap() {
        let a = [(0, 1.0), (2, 2.0)];
        let b = [(1, 1.0), (3, 1.0)];
        assert_near(sparse_dot_product(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_sparse_dot_product_full_overlap() {
        let a = [(0, 1.0), (2, 2.0), (5, 3.0)];
        let b = [(0, 1.0), (2, 2.0), (5, 3.0)];
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        assert_near(sparse_dot_product(&a, &b), 14.0, EPSILON);
    }

    #[test]
    fn test_sparse_dot_product_partial_overlap() {
        let a = [(0, 1.0), (2, 2.0), (5, 3.0)];
        let b = [(1, 1.0), (2, 2.0), (5, 1.0)];
        // Only indices 2 and 5 overlap: 2*2 + 3*1 = 7
        assert_near(sparse_dot_product(&a, &b), 7.0, EPSILON);
    }

    #[test]
    fn test_sparse_dot_product_empty() {
        let a: [(u32, f32); 0] = [];
        let b = [(0, 1.0)];
        assert_near(sparse_dot_product(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_sparse_l2_norm() {
        let v = [(0, 3.0), (5, 4.0)];
        assert_near(sparse_l2_norm(&v), 5.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_similarity_identical() {
        let a = [(0, 1.0), (1, 2.0)];
        assert_near(sparse_cosine_similarity(&a, &a), 1.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_similarity_orthogonal() {
        let a = [(0, 1.0)];
        let b = [(1, 1.0)];
        assert_near(sparse_cosine_similarity(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_similarity_opposite() {
        let a = [(0, 1.0)];
        let b = [(0, -1.0)];
        assert_near(sparse_cosine_similarity(&a, &b), -1.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_distance() {
        let a = [(0, 1.0), (1, 2.0)];
        assert_near(sparse_cosine_distance(&a, &a), 0.0, EPSILON);

        let b = [(2, 1.0)];
        assert_near(sparse_cosine_distance(&a, &b), 1.0, EPSILON);
    }

    #[test]
    fn test_sparse_euclidean_distance() {
        let a = [(0, 0.0)];
        let b = [(0, 3.0), (1, 4.0)];
        assert_near(sparse_euclidean_distance(&a, &b), 5.0, EPSILON);
    }

    #[test]
    fn test_sparse_euclidean_distance_no_overlap() {
        // a = [3, 0, 0], b = [0, 4, 0]
        let a = [(0, 3.0)];
        let b = [(1, 4.0)];
        // ||a - b||^2 = 9 + 16 = 25
        assert_near(sparse_euclidean_distance(&a, &b), 5.0, EPSILON);
    }

    #[test]
    fn test_sparse_cached_norm() {
        let v = [(0, 3.0), (5, 4.0)];
        let cached = SparseCachedNorm::new(&v);

        assert_near(cached.norm(), 5.0, EPSILON);
        assert_near(cached.norm_squared(), 25.0, EPSILON);
        assert!(!cached.is_zero());
    }

    #[test]
    fn test_sparse_cached_norm_empty() {
        let v: [(u32, f32); 0] = [];
        let cached = SparseCachedNorm::new(&v);
        assert!(cached.is_zero());
    }

    #[test]
    fn test_sparse_distance_metric() {
        let a = [(0, 3.0), (1, 4.0)];
        let b = [(0, 3.0), (1, 4.0)];

        assert_near(SparseDistanceMetric::Euclidean.calculate(&a, &b), 0.0, EPSILON);
        assert_near(SparseDistanceMetric::Cosine.calculate(&a, &b), 0.0, EPSILON);
        assert_near(SparseDistanceMetric::DotProduct.calculate(&a, &b), -25.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_with_norms() {
        let a = [(0, 3.0), (1, 4.0)];
        let b = [(0, 3.0), (1, 4.0)];
        let norm = sparse_l2_norm(&a);

        let sim = sparse_cosine_similarity_with_norms(&a, &b, norm, norm);
        assert!(sim.is_some());
        assert_near(sim.unwrap(), 1.0, EPSILON);
    }

    #[test]
    fn test_sparse_cosine_with_zero_norm() {
        let a: [(u32, f32); 0] = [];
        let b = [(0, 1.0)];

        let sim = sparse_cosine_similarity_with_norms(&a, &b, 0.0, 1.0);
        assert!(sim.is_none());
    }
}
