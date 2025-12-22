//! Scalar (non-SIMD) distance functions.
//!
//! This module provides fallback implementations that work on any platform
//! without requiring SIMD support. These are used when:
//! - The `scalar` feature is enabled
//! - For debugging and validation
//! - On platforms without SIMD support

#![allow(dead_code)] // Some functions may not be used internally but are exported

/// Calculate the squared Euclidean (L2) distance between two vectors.
///
/// This avoids the sqrt operation for cases where only relative distances matter.
///
/// # Panics
///
/// Debug-panics if vectors have different lengths.
#[inline]
#[must_use]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
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
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate the sum of squares (squared L2 norm) of a vector.
#[inline]
#[must_use]
pub fn sum_of_squares(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

/// Calculate the L2 norm (magnitude) of a vector.
#[inline]
#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    sum_of_squares(v).sqrt()
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

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate the cosine similarity using pre-computed norms.
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
/// Cosine distance = 1 - cosine_similarity.
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
#[derive(Debug, Clone, Copy)]
pub struct CachedNorm {
    norm_squared: f32,
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
    fn test_euclidean_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(euclidean_distance(&a, &b), 5.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert_near(cosine_similarity(&a, &b), 1.0, EPSILON);

        let c = [1.0, 0.0];
        let d = [0.0, 1.0];
        assert_near(cosine_similarity(&c, &d), 0.0, EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_near(dot_product(&a, &b), 32.0, EPSILON);
    }
}
