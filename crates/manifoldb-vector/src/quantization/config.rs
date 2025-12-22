//! Product Quantization configuration.

use crate::distance::DistanceMetric;
use crate::error::VectorError;

/// Configuration for Product Quantization.
///
/// # Parameters
///
/// - `dimension`: The dimension of vectors to quantize
/// - `num_segments`: Number of subspaces (M). Must divide dimension evenly.
/// - `num_centroids`: Number of centroids per subspace (K). Typically 256 (8 bits per code).
/// - `distance_metric`: Distance metric for codebook training and distance computation
///
/// # Memory Usage
///
/// - Codebooks: `M × K × (D/M) × 4` bytes
/// - Per-vector codes: `M × ceil(log2(K)/8)` bytes
///
/// For typical settings (D=128, M=8, K=256):
/// - Codebooks: 8 × 256 × 16 × 4 = 128KB
/// - Per-vector: 8 bytes (compression ratio: 64x)
#[derive(Debug, Clone)]
pub struct PQConfig {
    /// Dimension of input vectors.
    pub dimension: usize,
    /// Number of subspaces (segments).
    pub num_segments: usize,
    /// Number of centroids per subspace.
    pub num_centroids: usize,
    /// Distance metric for training and search.
    pub distance_metric: DistanceMetric,
    /// Number of training iterations for k-means.
    pub training_iterations: usize,
    /// Random seed for reproducible training.
    pub seed: Option<u64>,
}

impl PQConfig {
    /// Create a new PQ configuration.
    ///
    /// # Arguments
    ///
    /// - `dimension`: The dimension of vectors to quantize
    /// - `num_segments`: Number of subspaces. Must divide `dimension` evenly.
    ///
    /// # Defaults
    ///
    /// - `num_centroids`: 256 (8-bit codes)
    /// - `distance_metric`: Euclidean
    /// - `training_iterations`: 25
    /// - `seed`: None (non-deterministic)
    ///
    /// # Panics
    ///
    /// Panics if `num_segments` is 0 or doesn't divide `dimension` evenly.
    #[must_use]
    pub fn new(dimension: usize, num_segments: usize) -> Self {
        assert!(num_segments > 0, "num_segments must be > 0");
        assert!(
            dimension % num_segments == 0,
            "dimension ({}) must be divisible by num_segments ({})",
            dimension,
            num_segments
        );

        Self {
            dimension,
            num_segments,
            num_centroids: 256,
            distance_metric: DistanceMetric::Euclidean,
            training_iterations: 25,
            seed: None,
        }
    }

    /// Set the number of centroids per subspace.
    ///
    /// Common values:
    /// - 256 (8-bit codes, default)
    /// - 65536 (16-bit codes, more accurate but larger codebooks)
    #[must_use]
    pub const fn with_num_centroids(mut self, k: usize) -> Self {
        self.num_centroids = k;
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub const fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the number of training iterations.
    #[must_use]
    pub const fn with_training_iterations(mut self, iterations: usize) -> Self {
        self.training_iterations = iterations;
        self
    }

    /// Set the random seed for reproducible training.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the dimension of each subspace.
    #[must_use]
    pub const fn subspace_dimension(&self) -> usize {
        self.dimension / self.num_segments
    }

    /// Calculate the number of bits per code.
    ///
    /// Returns the number of bits needed to represent a centroid index.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn bits_per_code(&self) -> usize {
        // ceil(log2(num_centroids))
        if self.num_centroids <= 1 {
            1
        } else {
            (self.num_centroids as f64).log2().ceil() as usize
        }
    }

    /// Calculate bytes per encoded vector.
    #[must_use]
    pub fn bytes_per_code(&self) -> usize {
        // For 256 centroids, each code is 1 byte
        // For 65536 centroids, each code is 2 bytes
        let bits = self.bits_per_code();
        let total_bits = bits * self.num_segments;
        total_bits.div_ceil(8)
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` is 0
    /// - `num_segments` is 0 or doesn't divide `dimension`
    /// - `num_centroids` is 0
    pub fn validate(&self) -> Result<(), VectorError> {
        if self.dimension == 0 {
            return Err(VectorError::InvalidDimension { expected: 1, actual: 0 });
        }

        if self.num_segments == 0 {
            return Err(VectorError::Encoding("num_segments must be > 0".to_string()));
        }

        if self.dimension % self.num_segments != 0 {
            return Err(VectorError::Encoding(format!(
                "dimension ({}) must be divisible by num_segments ({})",
                self.dimension, self.num_segments
            )));
        }

        if self.num_centroids == 0 {
            return Err(VectorError::Encoding("num_centroids must be > 0".to_string()));
        }

        Ok(())
    }

    /// Calculate compression ratio compared to full f32 vectors.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dimension * 4; // 4 bytes per f32
        let compressed_bytes = self.bytes_per_code();
        original_bytes as f32 / compressed_bytes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_config() {
        let config = PQConfig::new(128, 8);
        assert_eq!(config.dimension, 128);
        assert_eq!(config.num_segments, 8);
        assert_eq!(config.num_centroids, 256);
        assert_eq!(config.subspace_dimension(), 16);
    }

    #[test]
    fn test_bits_per_code() {
        let config = PQConfig::new(128, 8).with_num_centroids(256);
        assert_eq!(config.bits_per_code(), 8);

        let config = PQConfig::new(128, 8).with_num_centroids(65536);
        assert_eq!(config.bits_per_code(), 16);

        let config = PQConfig::new(128, 8).with_num_centroids(16);
        assert_eq!(config.bits_per_code(), 4);
    }

    #[test]
    fn test_bytes_per_code() {
        // 8 segments × 8 bits = 64 bits = 8 bytes
        let config = PQConfig::new(128, 8).with_num_centroids(256);
        assert_eq!(config.bytes_per_code(), 8);

        // 8 segments × 16 bits = 128 bits = 16 bytes
        let config = PQConfig::new(128, 8).with_num_centroids(65536);
        assert_eq!(config.bytes_per_code(), 16);
    }

    #[test]
    fn test_compression_ratio() {
        // 128 × 4 = 512 bytes original, 8 bytes compressed = 64x
        let config = PQConfig::new(128, 8).with_num_centroids(256);
        assert!((config.compression_ratio() - 64.0).abs() < 0.01);
    }

    #[test]
    fn test_validation() {
        let config = PQConfig::new(128, 8);
        assert!(config.validate().is_ok());

        // Invalid: num_centroids = 0
        let mut config = PQConfig::new(128, 8);
        config.num_centroids = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    #[should_panic(expected = "num_segments must be > 0")]
    fn test_zero_segments_panics() {
        let _ = PQConfig::new(128, 0);
    }

    #[test]
    #[should_panic(expected = "must be divisible by")]
    fn test_indivisible_dimension_panics() {
        let _ = PQConfig::new(128, 7);
    }
}
