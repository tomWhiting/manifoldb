//! Metrics and observability for `ManifoldDB`.
//!
//! This module provides comprehensive metrics collection for monitoring database
//! performance and health. Metrics are collected using atomic counters for thread-safe,
//! low-overhead instrumentation.
//!
//! # Key Metrics
//!
//! - **Query metrics**: Latency histograms, queries per second, error counts
//! - **Transaction metrics**: Commit/rollback counts, active transactions
//! - **Cache metrics**: Hit/miss ratios, evictions, invalidations
//! - **Vector search metrics**: HNSW search times, nodes visited
//! - **Storage metrics**: Database size, table counts
//!
//! # Integration with `metrics` crate
//!
//! This module integrates with the [`metrics`] crate for Prometheus-compatible
//! metric export. You can install a metrics recorder (like `metrics-exporter-prometheus`)
//! to export metrics via HTTP.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::Database;
//!
//! let db = Database::in_memory()?;
//!
//! // Perform some operations...
//! db.execute("INSERT INTO users (name) VALUES ('Alice')")?;
//! db.query("SELECT * FROM users")?;
//!
//! // Get a metrics snapshot
//! let snapshot = db.metrics();
//! println!("Queries executed: {}", snapshot.queries.total_queries);
//! println!("Cache hit rate: {:?}", snapshot.cache.hit_rate());
//! println!("Transactions committed: {}", snapshot.transactions.commits);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Re-export the metrics crate for integration
pub use ::metrics;

/// Global metrics instance for the database.
///
/// This is thread-safe and can be shared across all database operations.
#[derive(Debug)]
pub struct DatabaseMetrics {
    /// Query execution metrics.
    pub queries: QueryMetrics,
    /// Transaction metrics.
    pub transactions: TransactionMetrics,
    /// Vector search metrics.
    pub vector: VectorMetrics,
    /// Storage metrics.
    pub storage: StorageMetrics,
}

impl DatabaseMetrics {
    /// Create a new metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queries: QueryMetrics::new(),
            transactions: TransactionMetrics::new(),
            vector: VectorMetrics::new(),
            storage: StorageMetrics::new(),
        }
    }

    /// Record a query execution with its duration.
    pub fn record_query(&self, duration: Duration, success: bool) {
        self.queries.record_query(duration, success);

        // Also emit to the metrics crate for Prometheus integration
        let duration_secs = duration.as_secs_f64();
        ::metrics::histogram!("manifoldb_query_duration_seconds").record(duration_secs);
        ::metrics::counter!("manifoldb_queries_total").increment(1);
        if !success {
            ::metrics::counter!("manifoldb_query_errors_total").increment(1);
        }
    }

    /// Record a transaction commit.
    pub fn record_commit(&self, duration: Duration) {
        self.transactions.record_commit(duration);

        ::metrics::counter!("manifoldb_transactions_committed_total").increment(1);
        ::metrics::histogram!("manifoldb_transaction_commit_duration_seconds")
            .record(duration.as_secs_f64());
    }

    /// Record a transaction rollback.
    pub fn record_rollback(&self) {
        self.transactions.record_rollback();
        ::metrics::counter!("manifoldb_transactions_rolled_back_total").increment(1);
    }

    /// Record a vector search operation.
    pub fn record_vector_search(&self, duration: Duration, nodes_visited: u64) {
        self.vector.record_search(duration, nodes_visited);

        ::metrics::histogram!("manifoldb_vector_search_duration_seconds")
            .record(duration.as_secs_f64());
        ::metrics::histogram!("manifoldb_vector_search_nodes_visited").record(nodes_visited as f64);
    }

    /// Update storage size metric.
    pub fn update_storage_size(&self, size_bytes: u64) {
        self.storage.update_size(size_bytes);
        ::metrics::gauge!("manifoldb_storage_size_bytes").set(size_bytes as f64);
    }

    /// Get a point-in-time snapshot of all metrics.
    #[must_use]
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            queries: self.queries.snapshot(),
            transactions: self.transactions.snapshot(),
            vector: self.vector.snapshot(),
            storage: self.storage.snapshot(),
            cache: None, // Will be filled in by Database::metrics()
        }
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.queries.reset();
        self.transactions.reset();
        self.vector.reset();
        // Storage size is not reset, as it represents current state
    }
}

impl Default for DatabaseMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Query execution metrics.
#[derive(Debug)]
pub struct QueryMetrics {
    /// Total number of queries executed.
    total_queries: AtomicU64,
    /// Number of failed queries.
    failed_queries: AtomicU64,
    /// Total query execution time in nanoseconds.
    total_duration_ns: AtomicU64,
    /// Minimum query duration in nanoseconds.
    min_duration_ns: AtomicU64,
    /// Maximum query duration in nanoseconds.
    max_duration_ns: AtomicU64,
    /// Histogram buckets for query latency (in microseconds).
    /// Buckets: [<100us, <1ms, <10ms, <100ms, <1s, >=1s]
    histogram_buckets: [AtomicU64; 6],
}

impl QueryMetrics {
    const BUCKET_THRESHOLDS_US: [u64; 5] = [
        100,       // 100 microseconds
        1_000,     // 1 millisecond
        10_000,    // 10 milliseconds
        100_000,   // 100 milliseconds
        1_000_000, // 1 second
    ];

    /// Create a new query metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            failed_queries: AtomicU64::new(0),
            total_duration_ns: AtomicU64::new(0),
            min_duration_ns: AtomicU64::new(u64::MAX),
            max_duration_ns: AtomicU64::new(0),
            histogram_buckets: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    /// Record a query execution.
    pub fn record_query(&self, duration: Duration, success: bool) {
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.failed_queries.fetch_add(1, Ordering::Relaxed);
        }

        let duration_ns = duration.as_nanos() as u64;
        self.total_duration_ns.fetch_add(duration_ns, Ordering::Relaxed);

        // Update min (using compare-and-swap loop)
        let mut current_min = self.min_duration_ns.load(Ordering::Relaxed);
        while duration_ns < current_min {
            match self.min_duration_ns.compare_exchange_weak(
                current_min,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max
        let mut current_max = self.max_duration_ns.load(Ordering::Relaxed);
        while duration_ns > current_max {
            match self.max_duration_ns.compare_exchange_weak(
                current_max,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }

        // Update histogram bucket
        let duration_us = duration.as_micros() as u64;
        let bucket_idx = Self::BUCKET_THRESHOLDS_US
            .iter()
            .position(|&threshold| duration_us < threshold)
            .unwrap_or(5);
        self.histogram_buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of query metrics.
    #[must_use]
    pub fn snapshot(&self) -> QueryMetricsSnapshot {
        let total = self.total_queries.load(Ordering::Relaxed);
        let total_duration_ns = self.total_duration_ns.load(Ordering::Relaxed);
        let min_ns = self.min_duration_ns.load(Ordering::Relaxed);
        let max_ns = self.max_duration_ns.load(Ordering::Relaxed);

        QueryMetricsSnapshot {
            total_queries: total,
            failed_queries: self.failed_queries.load(Ordering::Relaxed),
            avg_duration: if total > 0 {
                Duration::from_nanos(total_duration_ns / total)
            } else {
                Duration::ZERO
            },
            min_duration: if min_ns == u64::MAX {
                None
            } else {
                Some(Duration::from_nanos(min_ns))
            },
            max_duration: if max_ns == 0 && total == 0 {
                None
            } else {
                Some(Duration::from_nanos(max_ns))
            },
            histogram: [
                self.histogram_buckets[0].load(Ordering::Relaxed),
                self.histogram_buckets[1].load(Ordering::Relaxed),
                self.histogram_buckets[2].load(Ordering::Relaxed),
                self.histogram_buckets[3].load(Ordering::Relaxed),
                self.histogram_buckets[4].load(Ordering::Relaxed),
                self.histogram_buckets[5].load(Ordering::Relaxed),
            ],
        }
    }

    /// Reset query metrics.
    pub fn reset(&self) {
        self.total_queries.store(0, Ordering::Relaxed);
        self.failed_queries.store(0, Ordering::Relaxed);
        self.total_duration_ns.store(0, Ordering::Relaxed);
        self.min_duration_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_duration_ns.store(0, Ordering::Relaxed);
        for bucket in &self.histogram_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction metrics.
#[derive(Debug)]
pub struct TransactionMetrics {
    /// Number of committed transactions.
    commits: AtomicU64,
    /// Number of rolled back transactions.
    rollbacks: AtomicU64,
    /// Total commit duration in nanoseconds.
    total_commit_duration_ns: AtomicU64,
    /// Number of currently active transactions.
    active_transactions: AtomicU64,
}

impl TransactionMetrics {
    /// Create a new transaction metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            commits: AtomicU64::new(0),
            rollbacks: AtomicU64::new(0),
            total_commit_duration_ns: AtomicU64::new(0),
            active_transactions: AtomicU64::new(0),
        }
    }

    /// Record a transaction start.
    pub fn record_start(&self) {
        self.active_transactions.fetch_add(1, Ordering::Relaxed);
        ::metrics::gauge!("manifoldb_active_transactions")
            .set(self.active_transactions.load(Ordering::Relaxed) as f64);
    }

    /// Record a transaction commit.
    pub fn record_commit(&self, duration: Duration) {
        self.commits.fetch_add(1, Ordering::Relaxed);
        self.total_commit_duration_ns.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.active_transactions.fetch_sub(1, Ordering::Relaxed);
        ::metrics::gauge!("manifoldb_active_transactions")
            .set(self.active_transactions.load(Ordering::Relaxed) as f64);
    }

    /// Record a transaction rollback.
    pub fn record_rollback(&self) {
        self.rollbacks.fetch_add(1, Ordering::Relaxed);
        self.active_transactions.fetch_sub(1, Ordering::Relaxed);
        ::metrics::gauge!("manifoldb_active_transactions")
            .set(self.active_transactions.load(Ordering::Relaxed) as f64);
    }

    /// Record a transaction end without explicit commit/rollback (e.g., drop).
    pub fn record_end(&self) {
        self.active_transactions.fetch_sub(1, Ordering::Relaxed);
        ::metrics::gauge!("manifoldb_active_transactions")
            .set(self.active_transactions.load(Ordering::Relaxed) as f64);
    }

    /// Get a snapshot of transaction metrics.
    #[must_use]
    pub fn snapshot(&self) -> TransactionMetricsSnapshot {
        let commits = self.commits.load(Ordering::Relaxed);
        let total_duration_ns = self.total_commit_duration_ns.load(Ordering::Relaxed);

        TransactionMetricsSnapshot {
            commits,
            rollbacks: self.rollbacks.load(Ordering::Relaxed),
            active: self.active_transactions.load(Ordering::Relaxed),
            avg_commit_duration: if commits > 0 {
                Duration::from_nanos(total_duration_ns / commits)
            } else {
                Duration::ZERO
            },
        }
    }

    /// Reset transaction metrics.
    pub fn reset(&self) {
        self.commits.store(0, Ordering::Relaxed);
        self.rollbacks.store(0, Ordering::Relaxed);
        self.total_commit_duration_ns.store(0, Ordering::Relaxed);
        // Don't reset active_transactions as it represents current state
    }
}

impl Default for TransactionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Vector search metrics.
#[derive(Debug)]
pub struct VectorMetrics {
    /// Total number of vector searches.
    total_searches: AtomicU64,
    /// Total search duration in nanoseconds.
    total_duration_ns: AtomicU64,
    /// Total nodes visited during searches.
    total_nodes_visited: AtomicU64,
    /// Minimum search duration in nanoseconds.
    min_duration_ns: AtomicU64,
    /// Maximum search duration in nanoseconds.
    max_duration_ns: AtomicU64,
}

impl VectorMetrics {
    /// Create a new vector metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            total_duration_ns: AtomicU64::new(0),
            total_nodes_visited: AtomicU64::new(0),
            min_duration_ns: AtomicU64::new(u64::MAX),
            max_duration_ns: AtomicU64::new(0),
        }
    }

    /// Record a vector search operation.
    pub fn record_search(&self, duration: Duration, nodes_visited: u64) {
        self.total_searches.fetch_add(1, Ordering::Relaxed);
        let duration_ns = duration.as_nanos() as u64;
        self.total_duration_ns.fetch_add(duration_ns, Ordering::Relaxed);
        self.total_nodes_visited.fetch_add(nodes_visited, Ordering::Relaxed);

        // Update min
        let mut current_min = self.min_duration_ns.load(Ordering::Relaxed);
        while duration_ns < current_min {
            match self.min_duration_ns.compare_exchange_weak(
                current_min,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_min = actual,
            }
        }

        // Update max
        let mut current_max = self.max_duration_ns.load(Ordering::Relaxed);
        while duration_ns > current_max {
            match self.max_duration_ns.compare_exchange_weak(
                current_max,
                duration_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }
    }

    /// Get a snapshot of vector metrics.
    #[must_use]
    pub fn snapshot(&self) -> VectorMetricsSnapshot {
        let total = self.total_searches.load(Ordering::Relaxed);
        let total_duration_ns = self.total_duration_ns.load(Ordering::Relaxed);
        let total_nodes = self.total_nodes_visited.load(Ordering::Relaxed);
        let min_ns = self.min_duration_ns.load(Ordering::Relaxed);
        let max_ns = self.max_duration_ns.load(Ordering::Relaxed);

        VectorMetricsSnapshot {
            total_searches: total,
            avg_duration: if total > 0 {
                Duration::from_nanos(total_duration_ns / total)
            } else {
                Duration::ZERO
            },
            min_duration: if min_ns == u64::MAX {
                None
            } else {
                Some(Duration::from_nanos(min_ns))
            },
            max_duration: if max_ns == 0 && total == 0 {
                None
            } else {
                Some(Duration::from_nanos(max_ns))
            },
            avg_nodes_visited: if total > 0 { total_nodes / total } else { 0 },
            total_nodes_visited: total_nodes,
        }
    }

    /// Reset vector metrics.
    pub fn reset(&self) {
        self.total_searches.store(0, Ordering::Relaxed);
        self.total_duration_ns.store(0, Ordering::Relaxed);
        self.total_nodes_visited.store(0, Ordering::Relaxed);
        self.min_duration_ns.store(u64::MAX, Ordering::Relaxed);
        self.max_duration_ns.store(0, Ordering::Relaxed);
    }
}

impl Default for VectorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Storage metrics.
#[derive(Debug)]
pub struct StorageMetrics {
    /// Current database size in bytes.
    size_bytes: AtomicU64,
    /// Number of tables.
    table_count: AtomicU64,
}

impl StorageMetrics {
    /// Create a new storage metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self { size_bytes: AtomicU64::new(0), table_count: AtomicU64::new(0) }
    }

    /// Update the storage size.
    pub fn update_size(&self, size_bytes: u64) {
        self.size_bytes.store(size_bytes, Ordering::Relaxed);
    }

    /// Update the table count.
    pub fn update_table_count(&self, count: u64) {
        self.table_count.store(count, Ordering::Relaxed);
        ::metrics::gauge!("manifoldb_table_count").set(count as f64);
    }

    /// Get a snapshot of storage metrics.
    #[must_use]
    pub fn snapshot(&self) -> StorageMetricsSnapshot {
        StorageMetricsSnapshot {
            size_bytes: self.size_bytes.load(Ordering::Relaxed),
            table_count: self.table_count.load(Ordering::Relaxed),
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time snapshot of all database metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Query metrics snapshot.
    pub queries: QueryMetricsSnapshot,
    /// Transaction metrics snapshot.
    pub transactions: TransactionMetricsSnapshot,
    /// Vector search metrics snapshot.
    pub vector: VectorMetricsSnapshot,
    /// Storage metrics snapshot.
    pub storage: StorageMetricsSnapshot,
    /// Cache metrics snapshot (optional, filled by Database).
    pub cache: Option<CacheMetricsSnapshot>,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ManifoldDB Metrics")?;
        writeln!(f, "==================")?;
        writeln!(f)?;
        writeln!(f, "Queries:")?;
        writeln!(f, "  Total: {}", self.queries.total_queries)?;
        writeln!(f, "  Failed: {}", self.queries.failed_queries)?;
        writeln!(f, "  Avg Duration: {:?}", self.queries.avg_duration)?;
        if let Some(min) = self.queries.min_duration {
            writeln!(f, "  Min Duration: {min:?}")?;
        }
        if let Some(max) = self.queries.max_duration {
            writeln!(f, "  Max Duration: {max:?}")?;
        }
        writeln!(f)?;
        writeln!(f, "Transactions:")?;
        writeln!(f, "  Commits: {}", self.transactions.commits)?;
        writeln!(f, "  Rollbacks: {}", self.transactions.rollbacks)?;
        writeln!(f, "  Active: {}", self.transactions.active)?;
        writeln!(f, "  Avg Commit Duration: {:?}", self.transactions.avg_commit_duration)?;
        writeln!(f)?;
        writeln!(f, "Vector Search:")?;
        writeln!(f, "  Total Searches: {}", self.vector.total_searches)?;
        writeln!(f, "  Avg Duration: {:?}", self.vector.avg_duration)?;
        writeln!(f, "  Avg Nodes Visited: {}", self.vector.avg_nodes_visited)?;
        writeln!(f)?;
        writeln!(f, "Storage:")?;
        writeln!(f, "  Size: {} bytes", self.storage.size_bytes)?;
        writeln!(f, "  Tables: {}", self.storage.table_count)?;
        if let Some(ref cache) = self.cache {
            writeln!(f)?;
            writeln!(f, "Cache:")?;
            writeln!(f, "  Hits: {}", cache.hits)?;
            writeln!(f, "  Misses: {}", cache.misses)?;
            if let Some(rate) = cache.hit_rate() {
                writeln!(f, "  Hit Rate: {rate:.1}%")?;
            }
            writeln!(f, "  Evictions: {}", cache.evictions)?;
            writeln!(f, "  Invalidations: {}", cache.invalidations)?;
        }
        Ok(())
    }
}

/// Query metrics snapshot.
#[derive(Debug, Clone)]
pub struct QueryMetricsSnapshot {
    /// Total queries executed.
    pub total_queries: u64,
    /// Failed queries.
    pub failed_queries: u64,
    /// Average query duration.
    pub avg_duration: Duration,
    /// Minimum query duration.
    pub min_duration: Option<Duration>,
    /// Maximum query duration.
    pub max_duration: Option<Duration>,
    /// Latency histogram buckets.
    /// Buckets: [<100us, <1ms, <10ms, <100ms, <1s, >=1s]
    pub histogram: [u64; 6],
}

impl QueryMetricsSnapshot {
    /// Get queries per second based on total queries and average duration.
    ///
    /// Note: This is an approximation. For accurate QPS, you need to track
    /// queries over a specific time window.
    #[must_use]
    pub fn success_rate(&self) -> Option<f64> {
        if self.total_queries == 0 {
            None
        } else {
            Some(
                ((self.total_queries - self.failed_queries) as f64 / self.total_queries as f64)
                    * 100.0,
            )
        }
    }

    /// Get the percentage of queries in each latency bucket.
    #[must_use]
    pub fn histogram_percentages(&self) -> [f64; 6] {
        if self.total_queries == 0 {
            return [0.0; 6];
        }
        let total = self.total_queries as f64;
        [
            self.histogram[0] as f64 / total * 100.0,
            self.histogram[1] as f64 / total * 100.0,
            self.histogram[2] as f64 / total * 100.0,
            self.histogram[3] as f64 / total * 100.0,
            self.histogram[4] as f64 / total * 100.0,
            self.histogram[5] as f64 / total * 100.0,
        ]
    }
}

/// Transaction metrics snapshot.
#[derive(Debug, Clone)]
pub struct TransactionMetricsSnapshot {
    /// Number of commits.
    pub commits: u64,
    /// Number of rollbacks.
    pub rollbacks: u64,
    /// Currently active transactions.
    pub active: u64,
    /// Average commit duration.
    pub avg_commit_duration: Duration,
}

impl TransactionMetricsSnapshot {
    /// Get the total number of completed transactions.
    #[must_use]
    pub fn total_completed(&self) -> u64 {
        self.commits + self.rollbacks
    }

    /// Get the commit success rate as a percentage.
    #[must_use]
    pub fn commit_rate(&self) -> Option<f64> {
        let total = self.total_completed();
        if total == 0 {
            None
        } else {
            Some((self.commits as f64 / total as f64) * 100.0)
        }
    }
}

/// Vector search metrics snapshot.
#[derive(Debug, Clone)]
pub struct VectorMetricsSnapshot {
    /// Total searches performed.
    pub total_searches: u64,
    /// Average search duration.
    pub avg_duration: Duration,
    /// Minimum search duration.
    pub min_duration: Option<Duration>,
    /// Maximum search duration.
    pub max_duration: Option<Duration>,
    /// Average nodes visited per search.
    pub avg_nodes_visited: u64,
    /// Total nodes visited across all searches.
    pub total_nodes_visited: u64,
}

/// Storage metrics snapshot.
#[derive(Debug, Clone)]
pub struct StorageMetricsSnapshot {
    /// Current database size in bytes.
    pub size_bytes: u64,
    /// Number of tables.
    pub table_count: u64,
}

impl StorageMetricsSnapshot {
    /// Get the size in a human-readable format.
    #[must_use]
    pub fn size_human_readable(&self) -> String {
        let size = self.size_bytes;
        if size < 1024 {
            format!("{size} B")
        } else if size < 1024 * 1024 {
            format!("{:.1} KB", size as f64 / 1024.0)
        } else if size < 1024 * 1024 * 1024 {
            format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
        } else {
            format!("{:.1} GB", size as f64 / (1024.0 * 1024.0 * 1024.0))
        }
    }
}

/// Cache metrics snapshot.
///
/// This mirrors the structure from `cache::MetricsSnapshot` but is exposed
/// in the metrics module for a unified API.
#[derive(Debug, Clone)]
pub struct CacheMetricsSnapshot {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of LRU evictions.
    pub evictions: u64,
    /// Number of invalidations.
    pub invalidations: u64,
}

impl CacheMetricsSnapshot {
    /// Create from the cache module's snapshot.
    #[must_use]
    pub fn from_cache_snapshot(s: crate::cache::MetricsSnapshot) -> Self {
        Self {
            hits: s.hits,
            misses: s.misses,
            evictions: s.evictions,
            invalidations: s.invalidations,
        }
    }

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

    /// Get the miss rate as a percentage.
    #[must_use]
    pub fn miss_rate(&self) -> Option<f64> {
        self.hit_rate().map(|hr| 100.0 - hr)
    }
}

/// A guard for timing operations.
///
/// Records the duration when dropped.
pub struct TimingGuard<F: FnOnce(Duration)> {
    start: Instant,
    callback: Option<F>,
}

impl<F: FnOnce(Duration)> TimingGuard<F> {
    /// Create a new timing guard.
    #[must_use]
    pub fn new(callback: F) -> Self {
        Self { start: Instant::now(), callback: Some(callback) }
    }

    /// Finish timing and return the duration.
    pub fn finish(mut self) -> Duration {
        let duration = self.start.elapsed();
        if let Some(cb) = self.callback.take() {
            cb(duration);
        }
        duration
    }
}

impl<F: FnOnce(Duration)> Drop for TimingGuard<F> {
    fn drop(&mut self) {
        if let Some(cb) = self.callback.take() {
            cb(self.start.elapsed());
        }
    }
}

/// Create a timing guard for query execution.
#[must_use]
pub fn time_query(metrics: Arc<DatabaseMetrics>) -> TimingGuard<impl FnOnce(Duration)> {
    TimingGuard::new(move |duration| {
        metrics.record_query(duration, true);
    })
}

/// Create a timing guard for transaction commits.
#[must_use]
pub fn time_commit(metrics: Arc<DatabaseMetrics>) -> TimingGuard<impl FnOnce(Duration)> {
    TimingGuard::new(move |duration| {
        metrics.record_commit(duration);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_metrics_basic() {
        let metrics = QueryMetrics::new();

        metrics.record_query(Duration::from_micros(50), true);
        metrics.record_query(Duration::from_millis(5), true);
        metrics.record_query(Duration::from_millis(50), false);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_queries, 3);
        assert_eq!(snapshot.failed_queries, 1);
        assert!(snapshot.min_duration.is_some());
        assert!(snapshot.max_duration.is_some());
    }

    #[test]
    fn test_query_metrics_histogram() {
        let metrics = QueryMetrics::new();

        // <100us bucket
        metrics.record_query(Duration::from_micros(50), true);
        // <1ms bucket
        metrics.record_query(Duration::from_micros(500), true);
        // <10ms bucket
        metrics.record_query(Duration::from_millis(5), true);
        // <100ms bucket
        metrics.record_query(Duration::from_millis(50), true);
        // <1s bucket
        metrics.record_query(Duration::from_millis(500), true);
        // >=1s bucket
        metrics.record_query(Duration::from_secs(2), true);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.histogram, [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_transaction_metrics() {
        let metrics = TransactionMetrics::new();

        metrics.record_start();
        metrics.record_start();
        assert_eq!(metrics.snapshot().active, 2);

        metrics.record_commit(Duration::from_millis(10));
        metrics.record_rollback();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.commits, 1);
        assert_eq!(snapshot.rollbacks, 1);
        assert_eq!(snapshot.active, 0);
    }

    #[test]
    fn test_vector_metrics() {
        let metrics = VectorMetrics::new();

        metrics.record_search(Duration::from_micros(100), 50);
        metrics.record_search(Duration::from_micros(200), 100);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_searches, 2);
        assert_eq!(snapshot.avg_nodes_visited, 75);
        assert_eq!(snapshot.total_nodes_visited, 150);
    }

    #[test]
    fn test_storage_metrics() {
        let metrics = StorageMetrics::new();

        metrics.update_size(1024 * 1024); // 1 MB
        metrics.update_table_count(5);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.size_bytes, 1024 * 1024);
        assert_eq!(snapshot.table_count, 5);
        assert_eq!(snapshot.size_human_readable(), "1.0 MB");
    }

    #[test]
    fn test_database_metrics_snapshot() {
        let metrics = DatabaseMetrics::new();

        metrics.record_query(Duration::from_millis(5), true);
        metrics.record_commit(Duration::from_millis(2));
        metrics.record_vector_search(Duration::from_micros(100), 25);
        metrics.update_storage_size(2048);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries.total_queries, 1);
        assert_eq!(snapshot.transactions.commits, 1);
        assert_eq!(snapshot.vector.total_searches, 1);
        assert_eq!(snapshot.storage.size_bytes, 2048);
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = DatabaseMetrics::new();

        metrics.record_query(Duration::from_millis(5), true);
        metrics.record_commit(Duration::from_millis(2));

        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.queries.total_queries, 0);
        assert_eq!(snapshot.transactions.commits, 0);
    }

    #[test]
    fn test_timing_guard() {
        let recorded = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let recorded_clone = Arc::clone(&recorded);

        {
            let _guard = TimingGuard::new(move |_duration| {
                recorded_clone.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(Duration::from_millis(1));
        }

        assert!(recorded.load(Ordering::SeqCst));
    }

    #[test]
    fn test_cache_metrics_snapshot() {
        let cache_snapshot =
            crate::cache::MetricsSnapshot { hits: 100, misses: 20, evictions: 5, invalidations: 3 };

        let snapshot = CacheMetricsSnapshot::from_cache_snapshot(cache_snapshot);
        assert_eq!(snapshot.hits, 100);
        assert_eq!(snapshot.misses, 20);
        assert_eq!(snapshot.total_lookups(), 120);

        let hit_rate = snapshot.hit_rate().unwrap();
        assert!((hit_rate - 83.33).abs() < 0.1);
    }

    #[test]
    fn test_metrics_display() {
        let metrics = DatabaseMetrics::new();
        metrics.record_query(Duration::from_millis(5), true);
        let snapshot = metrics.snapshot();

        let display = format!("{snapshot}");
        assert!(display.contains("ManifoldDB Metrics"));
        assert!(display.contains("Queries:"));
        assert!(display.contains("Total: 1"));
    }
}
