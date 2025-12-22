//! Distance functions for vector similarity.
//!
//! This module provides both SIMD-optimized and scalar implementations of
//! distance functions for vector similarity search.
//!
//! # SIMD Optimization
//!
//! When the `simd` feature is enabled (default), this module uses the `wide` crate
//! for portable SIMD operations that work across:
//! - x86/x86_64: SSE2, SSE4.1, AVX, AVX2
//! - ARM: NEON
//! - WebAssembly: SIMD128
//!
//! The SIMD implementations process 8 floats at a time using `f32x8` vectors,
//! with a scalar fallback for the remainder.
//!
//! # Performance
//!
//! For 1536-dimensional vectors (e.g., OpenAI embeddings), the SIMD implementations
//! typically achieve 5-10x speedup compared to scalar implementations.
//!
//! # Features
//!
//! - `simd` (default): Enable SIMD-optimized distance calculations
//! - `scalar`: Force scalar implementations (useful for debugging)

#[cfg(not(feature = "scalar"))]
mod simd;

#[cfg(feature = "scalar")]
mod scalar;

// Re-export the appropriate implementation
#[cfg(not(feature = "scalar"))]
pub use simd::{
    cosine_distance, cosine_similarity, cosine_similarity_with_norms, dot_product,
    euclidean_distance, euclidean_distance_squared, l2_norm, sum_of_squares, CachedNorm,
};

#[cfg(feature = "scalar")]
pub use scalar::{
    cosine_distance, cosine_similarity, cosine_similarity_with_norms, dot_product,
    euclidean_distance, euclidean_distance_squared, l2_norm, sum_of_squares, CachedNorm,
};

/// Distance metric for comparing vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
    /// Dot product (negative, for max similarity).
    DotProduct,
}

impl DistanceMetric {
    /// Calculate the distance between two vectors using this metric.
    #[inline]
    #[must_use]
    pub fn calculate(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Euclidean => euclidean_distance(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::DotProduct => -dot_product(a, b),
        }
    }

    /// Calculate distance using cached norms (more efficient for repeated queries).
    #[inline]
    #[must_use]
    pub fn calculate_with_norms(&self, a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
        match self {
            Self::Euclidean => euclidean_distance(a, b),
            Self::Cosine => {
                cosine_similarity_with_norms(a, b, norm_a, norm_b).map_or(1.0, |s| 1.0 - s)
            }
            Self::DotProduct => -dot_product(a, b),
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
    fn test_euclidean_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(euclidean_distance(&a, &b), 5.0, EPSILON);
    }

    #[test]
    fn test_euclidean_distance_squared() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_near(euclidean_distance_squared(&a, &b), 25.0, EPSILON);
    }

    #[test]
    fn test_euclidean_distance_large() {
        // Test with 1536-dim vectors (OpenAI embedding size)
        let a: Vec<f32> = (0..1536).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i + 1) as f32 * 0.001).collect();

        let dist = euclidean_distance(&a, &b);
        // All differences are 0.001, so squared sum = 1536 * 0.000001 = 0.001536
        // sqrt(0.001536) ≈ 0.0392
        assert!(dist > 0.039 && dist < 0.040, "Expected ~0.0392, got {}", dist);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert_near(cosine_similarity(&a, &b), 1.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let c = [1.0, 0.0];
        let d = [0.0, 1.0];
        assert_near(cosine_similarity(&c, &d), 0.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert_near(cosine_similarity(&a, &b), -1.0, EPSILON);
    }

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert_near(cosine_distance(&a, &b), 0.0, EPSILON);

        let c = [1.0, 0.0];
        let d = [0.0, 1.0];
        assert_near(cosine_distance(&c, &d), 1.0, EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_near(dot_product(&a, &b), 32.0, EPSILON);
    }

    #[test]
    fn test_dot_product_large() {
        // Test with 1536-dim vectors
        let a: Vec<f32> = (0..1536).map(|i| 1.0 / (i + 1) as f32).collect();
        let b: Vec<f32> = (0..1536).map(|i| (i + 1) as f32).collect();

        let dot = dot_product(&a, &b);
        // Sum of 1/(i+1) * (i+1) = sum of 1s = 1536
        assert_near(dot, 1536.0, EPSILON);
    }

    #[test]
    fn test_distance_metric_calculate() {
        let a = [3.0, 4.0];
        let b = [0.0, 0.0];

        assert_near(DistanceMetric::Euclidean.calculate(&a, &b), 5.0, EPSILON);

        // Cosine distance for [3,4] and [0,0] - zero vector handling
        let c = [1.0, 0.0];
        let d = [1.0, 0.0];
        assert_near(DistanceMetric::Cosine.calculate(&c, &d), 0.0, EPSILON);

        assert_near(DistanceMetric::DotProduct.calculate(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_cached_norm() {
        let v = [3.0, 4.0];
        let cached = CachedNorm::new(&v);

        assert_near(cached.norm(), 5.0, EPSILON);
        assert_near(cached.norm_squared(), 25.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_with_norms() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        let norm_a = 1.0;
        let norm_b = 1.0;

        let sim = cosine_similarity_with_norms(&a, &b, norm_a, norm_b);
        assert!(sim.is_some());
        assert_near(sim.unwrap(), 1.0, EPSILON);
    }

    #[test]
    fn test_cosine_similarity_with_zero_norm() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];

        let sim = cosine_similarity_with_norms(&a, &b, 0.0, 1.0);
        assert!(sim.is_none());
    }
}
