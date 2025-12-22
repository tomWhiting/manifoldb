//! Distance functions for binary vectors.
//!
//! This module provides efficient distance calculations for bit-packed binary vectors
//! using hardware popcount instructions.
//!
//! # Hamming Distance
//!
//! Hamming distance counts the number of positions where corresponding bits differ.
//! It's computed efficiently using XOR (to find differing bits) followed by popcount
//! (to count them).
//!
//! Modern CPUs provide hardware popcount instructions:
//! - x86/x86_64: POPCNT instruction (SSE4.2+)
//! - ARM: CNT instruction (NEON)
//!
//! Rust's `u64::count_ones()` automatically uses these intrinsics when available.

use crate::types::BinaryEmbedding;

/// Calculate the Hamming distance between two binary embeddings.
///
/// Hamming distance is the number of positions where corresponding bits differ.
/// It's computed as: popcount(a XOR b)
///
/// # Panics
///
/// Debug-panics if embeddings have different dimensions.
///
/// # Example
///
/// ```
/// use manifoldb_vector::types::BinaryEmbedding;
/// use manifoldb_vector::distance::binary::hamming_distance;
///
/// let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
/// let b = BinaryEmbedding::new(vec![0b1010_1010u64], 8).unwrap();
///
/// // Bits that differ: positions 1, 3, 4, 6 = 4 differences
/// assert_eq!(hamming_distance(&a, &b), 4);
/// ```
#[inline]
#[must_use]
pub fn hamming_distance(a: &BinaryEmbedding, b: &BinaryEmbedding) -> u32 {
    debug_assert_eq!(a.dimension(), b.dimension(), "binary embeddings must have same dimension");

    a.data().iter().zip(b.data().iter()).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// Calculate the Hamming distance between two raw u64 slices.
///
/// This is a lower-level function that operates directly on bit-packed data.
/// For most use cases, prefer [`hamming_distance`] which works with [`BinaryEmbedding`].
///
/// # Panics
///
/// Debug-panics if slices have different lengths.
#[inline]
#[must_use]
pub fn hamming_distance_raw(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "bit vectors must have same length");

    a.iter().zip(b.iter()).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// Calculate the normalized Hamming distance (Hamming distance / dimension).
///
/// Returns a value in [0.0, 1.0] where:
/// - 0.0 means identical
/// - 1.0 means completely opposite (all bits differ)
///
/// # Panics
///
/// Debug-panics if embeddings have different dimensions.
#[inline]
#[must_use]
pub fn hamming_distance_normalized(a: &BinaryEmbedding, b: &BinaryEmbedding) -> f32 {
    let distance = hamming_distance(a, b);
    distance as f32 / a.dimension() as f32
}

/// Calculate the Jaccard similarity between two binary embeddings.
///
/// Jaccard similarity = |A ∩ B| / |A ∪ B| = popcount(a AND b) / popcount(a OR b)
///
/// Returns a value in [0.0, 1.0] where:
/// - 1.0 means identical (same bits set)
/// - 0.0 means completely disjoint (no common bits)
///
/// Returns 1.0 if both embeddings are all zeros (by convention).
///
/// # Panics
///
/// Debug-panics if embeddings have different dimensions.
#[inline]
#[must_use]
pub fn jaccard_similarity(a: &BinaryEmbedding, b: &BinaryEmbedding) -> f32 {
    debug_assert_eq!(a.dimension(), b.dimension(), "binary embeddings must have same dimension");

    let mut intersection: u32 = 0;
    let mut union: u32 = 0;

    for (&x, &y) in a.data().iter().zip(b.data().iter()) {
        intersection += (x & y).count_ones();
        union += (x | y).count_ones();
    }

    if union == 0 {
        1.0 // Both are all zeros - consider them identical
    } else {
        intersection as f32 / union as f32
    }
}

/// Calculate the Jaccard distance between two binary embeddings.
///
/// Jaccard distance = 1 - Jaccard similarity
///
/// # Panics
///
/// Debug-panics if embeddings have different dimensions.
#[inline]
#[must_use]
pub fn jaccard_distance(a: &BinaryEmbedding, b: &BinaryEmbedding) -> f32 {
    1.0 - jaccard_similarity(a, b)
}

/// Distance metric for comparing binary vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BinaryDistanceMetric {
    /// Hamming distance (count of differing bits).
    Hamming,
    /// Normalized Hamming distance (Hamming / dimension), range [0, 1].
    HammingNormalized,
    /// Jaccard distance (1 - Jaccard similarity), range [0, 1].
    Jaccard,
}

impl BinaryDistanceMetric {
    /// Calculate the distance between two binary embeddings using this metric.
    ///
    /// # Panics
    ///
    /// Debug-panics if embeddings have different dimensions.
    #[inline]
    #[must_use]
    pub fn calculate(&self, a: &BinaryEmbedding, b: &BinaryEmbedding) -> f32 {
        match self {
            Self::Hamming => hamming_distance(a, b) as f32,
            Self::HammingNormalized => hamming_distance_normalized(a, b),
            Self::Jaccard => jaccard_distance(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

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
    fn test_hamming_distance_identical() {
        let a = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 64).unwrap();
        let b = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 64).unwrap();
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = BinaryEmbedding::new(vec![0x0000_0000_0000_0000u64], 64).unwrap();
        let b = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64], 64).unwrap();
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    #[test]
    fn test_hamming_distance_half() {
        let a = BinaryEmbedding::new(vec![0xFFFF_FFFF_0000_0000u64], 64).unwrap();
        let b = BinaryEmbedding::new(vec![0x0000_0000_FFFF_FFFFu64], 64).unwrap();
        assert_eq!(hamming_distance(&a, &b), 64);
    }

    #[test]
    fn test_hamming_distance_one_bit() {
        let a = BinaryEmbedding::new(vec![0b0000_0001u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b0000_0000u64], 8).unwrap();
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_multi_word() {
        let a = BinaryEmbedding::new(vec![0xFFFF_FFFF_FFFF_FFFFu64, 0x0000_0000_0000_0000u64], 128)
            .unwrap();
        let b = BinaryEmbedding::new(vec![0x0000_0000_0000_0000u64, 0xFFFF_FFFF_FFFF_FFFFu64], 128)
            .unwrap();
        assert_eq!(hamming_distance(&a, &b), 128);
    }

    #[test]
    fn test_hamming_distance_raw() {
        let a = [0xFFFF_0000u64, 0x0000_FFFFu64];
        let b = [0x0000_FFFFu64, 0xFFFF_0000u64];
        assert_eq!(hamming_distance_raw(&a, &b), 64);
    }

    #[test]
    fn test_hamming_distance_normalized() {
        let a = BinaryEmbedding::zeros(100).unwrap();
        let b = BinaryEmbedding::ones(100).unwrap();
        assert_near(hamming_distance_normalized(&a, &b), 1.0, EPSILON);

        let c = BinaryEmbedding::zeros(100).unwrap();
        assert_near(hamming_distance_normalized(&a, &c), 0.0, EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        assert_near(jaccard_similarity(&a, &b), 1.0, EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_disjoint() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b0000_1111u64], 8).unwrap();
        // No intersection, union = 8 bits
        assert_near(jaccard_similarity(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_partial_overlap() {
        let a = BinaryEmbedding::new(vec![0b1111_1100u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b0011_1111u64], 8).unwrap();
        // Intersection: 4 bits (middle 4)
        // Union: 8 bits
        assert_near(jaccard_similarity(&a, &b), 4.0 / 8.0, EPSILON);
    }

    #[test]
    fn test_jaccard_similarity_both_zero() {
        let a = BinaryEmbedding::zeros(64).unwrap();
        let b = BinaryEmbedding::zeros(64).unwrap();
        assert_near(jaccard_similarity(&a, &b), 1.0, EPSILON);
    }

    #[test]
    fn test_jaccard_distance() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        assert_near(jaccard_distance(&a, &b), 0.0, EPSILON);
    }

    #[test]
    fn test_binary_distance_metric_calculate() {
        let a = BinaryEmbedding::new(vec![0b1111_0000u64], 8).unwrap();
        let b = BinaryEmbedding::new(vec![0b1010_1010u64], 8).unwrap();

        // Hamming: 4 bits differ
        assert_near(BinaryDistanceMetric::Hamming.calculate(&a, &b), 4.0, EPSILON);

        // Normalized: 4/8 = 0.5
        assert_near(BinaryDistanceMetric::HammingNormalized.calculate(&a, &b), 0.5, EPSILON);
    }
}
