//! Cache metrics for monitoring cache performance.

use std::sync::atomic::{AtomicU64, Ordering};

/// Metrics for cache performance monitoring.
#[derive(Debug)]
pub struct CacheMetrics {
    /// Number of cache hits.
    hits: AtomicU64,
    /// Number of cache misses.
    misses: AtomicU64,
    /// Number of cache evictions due to LRU policy.
    evictions: AtomicU64,
    /// Number of entries invalidated due to writes.
    invalidations: AtomicU64,
}

impl CacheMetrics {
    /// Create a new metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    /// Record a cache hit.
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss.
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an LRU eviction.
    pub fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record invalidations due to writes.
    pub fn record_invalidations(&self, count: usize) {
        self.invalidations.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Get the number of cache hits.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get the number of cache misses.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get the number of LRU evictions.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Get the number of invalidations.
    #[must_use]
    pub fn invalidations(&self) -> u64 {
        self.invalidations.load(Ordering::Relaxed)
    }

    /// Get the total number of cache lookups (hits + misses).
    #[must_use]
    pub fn total_lookups(&self) -> u64 {
        self.hits() + self.misses()
    }

    /// Get the cache hit rate as a percentage (0.0 to 100.0).
    ///
    /// Returns `None` if there have been no lookups.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.total_lookups();
        if total == 0 {
            None
        } else {
            Some((self.hits() as f64 / total as f64) * 100.0)
        }
    }

    /// Get the cache miss rate as a percentage (0.0 to 100.0).
    ///
    /// Returns `None` if there have been no lookups.
    #[must_use]
    pub fn miss_rate(&self) -> Option<f64> {
        self.hit_rate().map(|hr| 100.0 - hr)
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
    }

    /// Get a snapshot of all metrics.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            hits: self.hits(),
            misses: self.misses(),
            evictions: self.evictions(),
            invalidations: self.invalidations(),
        }
    }
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of cache metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of LRU evictions.
    pub evictions: u64,
    /// Number of invalidations.
    pub invalidations: u64,
}

impl MetricsSnapshot {
    /// Get the total number of lookups.
    #[must_use]
    pub fn total_lookups(&self) -> u64 {
        self.hits + self.misses
    }

    /// Get the hit rate as a percentage.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.total_lookups();
        if total == 0 {
            None
        } else {
            Some((self.hits as f64 / total as f64) * 100.0)
        }
    }
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hit_rate =
            self.hit_rate().map(|r| format!("{r:.1}%")).unwrap_or_else(|| "N/A".to_string());
        write!(
            f,
            "Cache Stats: hits={}, misses={}, hit_rate={}, evictions={}, invalidations={}",
            self.hits, self.misses, hit_rate, self.evictions, self.invalidations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_basic() {
        let metrics = CacheMetrics::new();

        assert_eq!(metrics.hits(), 0);
        assert_eq!(metrics.misses(), 0);

        metrics.record_hit();
        metrics.record_hit();
        metrics.record_miss();

        assert_eq!(metrics.hits(), 2);
        assert_eq!(metrics.misses(), 1);
        assert_eq!(metrics.total_lookups(), 3);
    }

    #[test]
    fn test_hit_rate() {
        let metrics = CacheMetrics::new();

        // No lookups yet
        assert!(metrics.hit_rate().is_none());

        // 100% hit rate
        metrics.record_hit();
        assert!((metrics.hit_rate().unwrap() - 100.0).abs() < f64::EPSILON);

        // 50% hit rate
        metrics.record_miss();
        assert!((metrics.hit_rate().unwrap() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evictions_and_invalidations() {
        let metrics = CacheMetrics::new();

        metrics.record_eviction();
        metrics.record_eviction();
        metrics.record_invalidations(5);

        assert_eq!(metrics.evictions(), 2);
        assert_eq!(metrics.invalidations(), 5);
    }

    #[test]
    fn test_reset() {
        let metrics = CacheMetrics::new();

        metrics.record_hit();
        metrics.record_miss();
        metrics.record_eviction();

        metrics.reset();

        assert_eq!(metrics.hits(), 0);
        assert_eq!(metrics.misses(), 0);
        assert_eq!(metrics.evictions(), 0);
    }

    #[test]
    fn test_snapshot() {
        let metrics = CacheMetrics::new();

        metrics.record_hit();
        metrics.record_hit();
        metrics.record_miss();

        let snapshot = metrics.snapshot();

        assert_eq!(snapshot.hits, 2);
        assert_eq!(snapshot.misses, 1);
        assert_eq!(snapshot.total_lookups(), 3);
    }

    #[test]
    fn test_snapshot_display() {
        let snapshot = MetricsSnapshot { hits: 10, misses: 5, evictions: 2, invalidations: 3 };

        let display = format!("{snapshot}");
        assert!(display.contains("hits=10"));
        assert!(display.contains("misses=5"));
        assert!(display.contains("66.7%"));
    }
}
