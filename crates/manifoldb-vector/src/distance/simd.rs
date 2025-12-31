//! SIMD-optimized distance functions using the `wide` crate.
//!
//! This module provides high-performance implementations of vector distance
//! calculations using SIMD (Single Instruction, Multiple Data) operations.
//!
//! The `wide` crate automatically selects the best available SIMD instruction set:
//! - x86/x86_64: SSE2, SSE4.1, AVX, AVX2
//! - ARM: NEON
//! - WebAssembly: SIMD128
//! - Fallback: Scalar operations
//!
//! All functions process 8 floats at a time using `f32x8` SIMD vectors.

#![allow(dead_code)] // l2_norm is available for external use but not used internally

use wide::f32x8;

/// Number of f32 elements processed per SIMD iteration.
const SIMD_WIDTH: usize = 8;

/// Convert a slice to a fixed-size array for SIMD.
/// Returns zero array if conversion fails (should never happen with correct loop bounds).
#[inline]
fn slice_to_simd_array(slice: &[f32]) -> [f32; SIMD_WIDTH] {
    slice.try_into().unwrap_or([0.0; SIMD_WIDTH])
}

/// Calculate the squared Euclidean (L2) distance between two vectors.
///
/// This avoids the sqrt operation for cases where only relative distances matter
/// (e.g., finding the k nearest neighbors).
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let len = a.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut sum = f32x8::ZERO;

    // Process 8 elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let va = f32x8::new(slice_to_simd_array(&a[i..i + SIMD_WIDTH]));
        let vb = f32x8::new(slice_to_simd_array(&b[i..i + SIMD_WIDTH]));
        let diff = va - vb;
        sum += diff * diff;
    }

    // Horizontal sum of SIMD register
    let mut result = horizontal_sum(sum);

    // Handle remaining elements
    for i in simd_len..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result
}

/// Calculate the Euclidean (L2) distance between two vectors.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Calculate the dot product between two vectors.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let len = a.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut sum = f32x8::ZERO;

    // Process 8 elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let va = f32x8::new(slice_to_simd_array(&a[i..i + SIMD_WIDTH]));
        let vb = f32x8::new(slice_to_simd_array(&b[i..i + SIMD_WIDTH]));
        sum += va * vb;
    }

    // Horizontal sum of SIMD register
    let mut result = horizontal_sum(sum);

    // Handle remaining elements
    for i in simd_len..len {
        result += a[i] * b[i];
    }

    result
}

/// Calculate the sum of squares (squared L2 norm) of a vector.
///
/// This is useful for precomputing norms for cosine similarity.
#[inline]
#[must_use]
pub fn sum_of_squares(v: &[f32]) -> f32 {
    let len = v.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut sum = f32x8::ZERO;

    // Process 8 elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let vv = f32x8::new(slice_to_simd_array(&v[i..i + SIMD_WIDTH]));
        sum += vv * vv;
    }

    // Horizontal sum of SIMD register
    let mut result = horizontal_sum(sum);

    // Handle remaining elements
    for i in simd_len..len {
        result += v[i] * v[i];
    }

    result
}

/// Calculate the L2 norm (magnitude) of a vector.
#[inline]
#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    sum_of_squares(v).sqrt()
}

/// Calculate the Manhattan (L1) distance between two vectors.
///
/// Manhattan distance is the sum of absolute differences between corresponding elements.
/// Also known as taxicab distance or city block distance.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let len = a.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut sum = f32x8::ZERO;

    // Process 8 elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let va = f32x8::new(slice_to_simd_array(&a[i..i + SIMD_WIDTH]));
        let vb = f32x8::new(slice_to_simd_array(&b[i..i + SIMD_WIDTH]));
        let diff = va - vb;
        sum += diff.abs();
    }

    // Horizontal sum of SIMD register
    let mut result = horizontal_sum(sum);

    // Handle remaining elements
    for i in simd_len..len {
        result += (a[i] - b[i]).abs();
    }

    result
}

/// Calculate the Chebyshev (L∞) distance between two vectors.
///
/// Chebyshev distance is the maximum absolute difference between corresponding elements.
/// Also known as chessboard distance or L-infinity norm.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn chebyshev_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let len = a.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut max_simd = f32x8::ZERO;

    // Process 8 elements at a time
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let va = f32x8::new(slice_to_simd_array(&a[i..i + SIMD_WIDTH]));
        let vb = f32x8::new(slice_to_simd_array(&b[i..i + SIMD_WIDTH]));
        let diff = (va - vb).abs();
        max_simd = max_simd.max(diff);
    }

    // Horizontal max of SIMD register
    let mut result = horizontal_max(max_simd);

    // Handle remaining elements
    for i in simd_len..len {
        result = result.max((a[i] - b[i]).abs());
    }

    result
}

/// Horizontal max of an f32x8 SIMD register.
///
/// This finds the maximum of all 8 f32 values in the register.
#[inline]
fn horizontal_max(v: f32x8) -> f32 {
    let arr: [f32; 8] = v.to_array();
    arr.iter().copied().fold(f32::MIN, f32::max)
}

/// Calculate the cosine similarity between two vectors.
///
/// Returns a value in the range [-1, 1] where:
/// - 1 means identical direction
/// - 0 means orthogonal
/// - -1 means opposite direction
///
/// Returns 0.0 if either vector has zero magnitude.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let len = a.len();
    let simd_len = len - (len % SIMD_WIDTH);

    let mut dot_sum = f32x8::ZERO;
    let mut norm_a_sum = f32x8::ZERO;
    let mut norm_b_sum = f32x8::ZERO;

    // Process 8 elements at a time, computing dot product and norms together
    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        let va = f32x8::new(slice_to_simd_array(&a[i..i + SIMD_WIDTH]));
        let vb = f32x8::new(slice_to_simd_array(&b[i..i + SIMD_WIDTH]));

        dot_sum += va * vb;
        norm_a_sum += va * va;
        norm_b_sum += vb * vb;
    }

    // Horizontal sums
    let mut dot = horizontal_sum(dot_sum);
    let mut norm_a_sq = horizontal_sum(norm_a_sum);
    let mut norm_b_sq = horizontal_sum(norm_b_sum);

    // Handle remaining elements
    for i in simd_len..len {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    let norm_product = (norm_a_sq * norm_b_sq).sqrt();

    if norm_product == 0.0 {
        return 0.0;
    }

    dot / norm_product
}

/// Calculate the cosine similarity using pre-computed norms.
///
/// This is more efficient when the same vector is compared against many others,
/// as the norm only needs to be computed once.
///
/// Returns `None` if either norm is zero.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn cosine_similarity_with_norms(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> Option<f32> {
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }

    let dot = dot_product(a, b);
    Some(dot / (norm_a * norm_b))
}

/// Calculate the cosine distance between two vectors.
///
/// Cosine distance = 1 - cosine_similarity, returning a value in [0, 2].
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Pre-computed L2 norm for efficient repeated cosine similarity calculations.
///
/// When comparing a query vector against many candidates, computing the query's
/// norm once and reusing it is more efficient than recomputing it for each comparison.
///
/// # Example
///
/// ```ignore
/// use manifoldb_vector::distance::CachedNorm;
///
/// let query = [1.0, 2.0, 3.0];
/// let cached_norm = CachedNorm::new(&query);
///
/// // Use cached norm for many comparisons
/// for candidate in candidates {
///     let candidate_norm = CachedNorm::new(&candidate);
///     let similarity = cached_norm.cosine_similarity_to(&query, &candidate, &candidate_norm);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CachedNorm {
    /// The squared L2 norm (sum of squares)
    norm_squared: f32,
    /// The L2 norm (square root of sum of squares)
    norm: f32,
}

impl CachedNorm {
    /// Compute and cache the L2 norm of a vector.
    #[must_use]
    pub fn new(v: &[f32]) -> Self {
        let norm_squared = sum_of_squares(v);
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

/// Horizontal sum of an f32x8 SIMD register.
///
/// This sums all 8 f32 values in the register into a single f32 result.
#[inline]
fn horizontal_sum(v: f32x8) -> f32 {
    // Extract the 8 values and sum them
    let arr: [f32; 8] = v.to_array();
    arr.iter().sum()
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
    fn test_dot_product_small() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_near(dot_product(&a, &b), 32.0, EPSILON);
    }

    #[test]
    fn test_dot_product_simd_aligned() {
        // 8 elements - exactly one SIMD iteration
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0; 8];
        // Sum of 1..8 = 36
        assert_near(dot_product(&a, &b), 36.0, EPSILON);
    }

    #[test]
    fn test_dot_product_mixed() {
        // 10 elements - one SIMD iteration + 2 remainder
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = [1.0; 10];
        // Sum of 1..10 = 55
        assert_near(dot_product(&a, &b), 55.0, EPSILON);
    }

    #[test]
    fn test_euclidean_distance_small() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(euclidean_distance(&a, &b), 5.0, EPSILON);
    }

    #[test]
    fn test_euclidean_large() {
        // 1536-dim vectors (OpenAI embedding size)
        let a: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i + 1) as f32 * 0.001).collect();

        let dist = euclidean_distance(&a, &b);
        // All differences are 0.001
        // Squared sum = 1536 * 0.001^2 = 0.001536
        // sqrt(0.001536) ≈ 0.0392
        assert!(dist > 0.039 && dist < 0.040, "Expected ~0.0392, got {}", dist);
    }

    #[test]
    fn test_sum_of_squares() {
        let v = [3.0, 4.0];
        assert_near(sum_of_squares(&v), 25.0, EPSILON);
    }

    #[test]
    fn test_l2_norm() {
        let v = [3.0, 4.0];
        assert_near(l2_norm(&v), 5.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 0.0];
        assert_near(cosine_similarity(&a, &a), 1.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert_near(cosine_similarity(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_large() {
        // 1536-dim vectors
        let a: Vec<f32> = (0..1536).map(|i| (i % 10) as f32).collect();
        let b = a.clone();
        assert_near(cosine_similarity(&a, &b), 1.0, EPSILON);
    }

    #[test]
    fn test_cosine_with_norms() {
        let a = [3.0, 4.0];
        let b = [3.0, 4.0];
        let norm_a = l2_norm(&a);
        let norm_b = l2_norm(&b);

        let sim = cosine_similarity_with_norms(&a, &b, norm_a, norm_b);
        assert!(sim.is_some());
        assert_near(sim.unwrap(), 1.0, EPSILON);
    }

    #[test]
    fn test_cached_norm() {
        let v = [3.0, 4.0];
        let cached = CachedNorm::new(&v);

        assert_near(cached.norm(), 5.0, EPSILON);
        assert_near(cached.norm_squared(), 25.0, EPSILON);
        assert!(!cached.is_zero());
    }

    #[test]
    fn test_cached_norm_zero() {
        let v = [0.0, 0.0, 0.0];
        let cached = CachedNorm::new(&v);

        assert!(cached.is_zero());
    }

    #[test]
    fn test_horizontal_sum() {
        let v = f32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_near(horizontal_sum(v), 36.0, EPSILON);
    }

    #[test]
    fn test_horizontal_max() {
        let v = f32x8::new([1.0, 8.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0]);
        assert_near(horizontal_max(v), 8.0, EPSILON);
    }

    #[test]
    fn test_manhattan_distance_small() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(manhattan_distance(&a, &b), 7.0, EPSILON);
    }

    #[test]
    fn test_manhattan_distance_simd_aligned() {
        // 8 elements - exactly one SIMD iteration
        let a = [0.0; 8];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Sum of 1..8 = 36
        assert_near(manhattan_distance(&a, &b), 36.0, EPSILON);
    }

    #[test]
    fn test_manhattan_distance_mixed() {
        // 10 elements - one SIMD iteration + 2 remainder
        let a = [0.0; 10];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // Sum of 1..10 = 55
        assert_near(manhattan_distance(&a, &b), 55.0, EPSILON);
    }

    #[test]
    fn test_manhattan_distance_large() {
        // 1536-dim vectors
        let a: Vec<f32> = (0..1536).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i + 1) as f32).collect();

        let dist = manhattan_distance(&a, &b);
        // All differences are 1, so sum = 1536
        assert_near(dist, 1536.0, EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_small() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(chebyshev_distance(&a, &b), 4.0, EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_simd_aligned() {
        // 8 elements - exactly one SIMD iteration
        let a = [0.0; 8];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Max of 1..8 = 8
        assert_near(chebyshev_distance(&a, &b), 8.0, EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_mixed() {
        // 10 elements - one SIMD iteration + 2 remainder
        let a = [0.0; 10];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // Max of 1..10 = 10
        assert_near(chebyshev_distance(&a, &b), 10.0, EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_max_in_remainder() {
        // 10 elements - max is in the remainder portion
        let a = [0.0; 10];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, 10.0];
        assert_near(chebyshev_distance(&a, &b), 100.0, EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_large() {
        // 1536-dim vectors
        let a: Vec<f32> = (0..1536).map(|_| 0.0).collect();
        let mut b: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        b[1000] = 999.0; // Set a large value in the middle

        let dist = chebyshev_distance(&a, &b);
        assert_near(dist, 999.0, EPSILON);
    }
}
