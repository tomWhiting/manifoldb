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
}

impl HnswConfig {
    /// Create a new HNSW configuration with the specified M parameter.
    ///
    /// Other parameters are set to sensible defaults:
    /// - `m_max0` = 2 * m
    /// - `ef_construction` = 200
    /// - `ef_search` = 50
    /// - `ml` = 1 / ln(m)
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // m is typically small (16-64), so no precision loss
    pub fn new(m: usize) -> Self {
        let m = m.max(2); // Ensure at least 2 connections
        Self { m, m_max0: m * 2, ef_construction: 200, ef_search: 50, ml: 1.0 / (m as f64).ln() }
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
}
