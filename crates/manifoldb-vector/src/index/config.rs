//! HNSW index configuration.

/// Configuration parameters for an HNSW index.
///
/// # Parameters
///
/// * `m` - Maximum number of connections per node in each layer.
///   Typical values: 16-64. Higher values give better recall but use more memory.
///
/// * `m_max0` - Maximum number of connections in layer 0 (the densest layer).
///   Typically set to `2 * m`.
///
/// * `ef_construction` - Beam width during index construction.
///   Higher values give better index quality but slower construction.
///   Typical values: 100-500.
///
/// * `ef_search` - Default beam width during search.
///   Higher values give better recall but slower search.
///   Can be overridden per-query. Typical values: 10-500.
///
/// * `ml` - Level multiplier for determining max level.
///   Typically `1 / ln(m)`. Affects the distribution of nodes across layers.
///
/// # Product Quantization (PQ) for Compression
///
/// When `pq_segments` is set (non-zero), vectors are compressed using Product Quantization:
///
/// * `pq_segments` - Number of subspaces to divide vectors into.
///   Must divide the vector dimension evenly. Typical values: 8, 16, 32.
///   Higher values = better accuracy but larger codes.
///
/// * `pq_centroids` - Number of centroids per subspace. Default: 256 (8-bit codes).
///
/// PQ reduces memory usage by 4-8x with minimal recall loss. During search:
/// - Full precision vectors are used for the first candidate selection
/// - Compressed vectors are stored in memory for efficient distance computation
/// - Original vectors can be retrieved from storage for final reranking
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node (M parameter).
    pub m: usize,
    /// Maximum connections in layer 0 (typically 2 * M).
    pub m_max0: usize,
    /// Beam width for construction.
    pub ef_construction: usize,
    /// Default beam width for search.
    pub ef_search: usize,
    /// Level multiplier (1 / ln(M)).
    pub ml: f64,
    /// Number of PQ segments (0 = disabled). Must divide vector dimension evenly.
    pub pq_segments: usize,
    /// Number of centroids per PQ segment. Default: 256.
    pub pq_centroids: usize,
    /// Minimum vectors required before PQ training. Default: 1000.
    pub pq_training_samples: usize,
}

impl HnswConfig {
    /// Create a new HNSW configuration with the specified M parameter.
    ///
    /// Other parameters are set to sensible defaults:
    /// - `m_max0` = 2 * m
    /// - `ef_construction` = 200
    /// - `ef_search` = 50
    /// - `ml` = 1 / ln(m)
    /// - `pq_segments` = 0 (disabled)
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // m is typically small (16-64), so no precision loss
    pub fn new(m: usize) -> Self {
        let m = m.max(2); // Ensure at least 2 connections
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            pq_segments: 0,
            pq_centroids: 256,
            pq_training_samples: 1000,
        }
    }

    /// Set the beam width for construction.
    #[must_use]
    pub const fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the default beam width for search.
    #[must_use]
    pub const fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set the maximum connections in layer 0.
    #[must_use]
    pub const fn with_m_max0(mut self, m_max0: usize) -> Self {
        self.m_max0 = m_max0;
        self
    }

    /// Enable Product Quantization with the specified number of segments.
    ///
    /// The vector dimension must be divisible by `segments`.
    /// This reduces memory usage by approximately `dimension * 4 / segments` bytes per vector.
    ///
    /// # Arguments
    ///
    /// - `segments`: Number of subspaces. Common values: 8, 16, 32.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For 128-dim vectors with 8 segments = 16x compression
    /// let config = HnswConfig::new(16).with_pq(8);
    /// ```
    #[must_use]
    pub const fn with_pq(mut self, segments: usize) -> Self {
        self.pq_segments = segments;
        self
    }

    /// Set the number of centroids per PQ segment.
    ///
    /// Default is 256 (8-bit codes). Higher values give better accuracy
    /// but require more memory for codebooks.
    #[must_use]
    pub const fn with_pq_centroids(mut self, centroids: usize) -> Self {
        self.pq_centroids = centroids;
        self
    }

    /// Set the minimum number of vectors required before training PQ.
    ///
    /// PQ training requires enough samples for k-means clustering.
    /// Default is 1000.
    #[must_use]
    pub const fn with_pq_training_samples(mut self, samples: usize) -> Self {
        self.pq_training_samples = samples;
        self
    }

    /// Check if Product Quantization is enabled.
    #[must_use]
    pub const fn pq_enabled(&self) -> bool {
        self.pq_segments > 0
    }
}

impl Default for HnswConfig {
    /// Create a default HNSW configuration.
    ///
    /// Uses M=16, which is a good balance between recall and speed.
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max0, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
        assert!((config.ml - 1.0 / 16_f64.ln()).abs() < 1e-10);
        assert_eq!(config.pq_segments, 0);
        assert!(!config.pq_enabled());
    }

    #[test]
    fn test_custom_config() {
        let config =
            HnswConfig::new(32).with_ef_construction(400).with_ef_search(100).with_m_max0(48);

        assert_eq!(config.m, 32);
        assert_eq!(config.m_max0, 48);
        assert_eq!(config.ef_construction, 400);
        assert_eq!(config.ef_search, 100);
    }

    #[test]
    fn test_minimum_m() {
        let config = HnswConfig::new(1);
        assert_eq!(config.m, 2); // Should be at least 2
    }

    #[test]
    fn test_pq_config() {
        let config =
            HnswConfig::new(16).with_pq(8).with_pq_centroids(512).with_pq_training_samples(2000);

        assert!(config.pq_enabled());
        assert_eq!(config.pq_segments, 8);
        assert_eq!(config.pq_centroids, 512);
        assert_eq!(config.pq_training_samples, 2000);
    }
}
