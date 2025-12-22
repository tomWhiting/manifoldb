//! Query result caching for ManifoldDB.
//!
//! This module provides an LRU cache for query results to avoid repeated
//! execution of identical queries. The cache is keyed by a hash of the
//! SQL query string and parameter values.
//!
//! # Features
//!
//! - **LRU eviction**: Least recently used entries are evicted when the cache is full
//! - **TTL support**: Optional time-to-live for cache entries
//! - **Cache hints**: Support for `/*+ CACHE */` and `/*+ NO_CACHE */` hints
//! - **Automatic invalidation**: Cache entries are invalidated when affected tables are modified
//! - **Metrics**: Track cache hit/miss rates for monitoring
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::cache::{QueryCache, CacheConfig};
//!
//! let cache = QueryCache::new(CacheConfig::default());
//!
//! // Cache will automatically be checked before query execution
//! // and populated with results after execution
//! ```

mod hints;
mod metrics;
mod query_cache;

pub use hints::{extract_cache_hint, CacheHint};
pub use metrics::{CacheMetrics, MetricsSnapshot};
pub use query_cache::{CacheConfig, CacheEntry, QueryCache, QueryCacheKey};
