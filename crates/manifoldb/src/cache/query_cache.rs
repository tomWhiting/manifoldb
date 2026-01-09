//! Query result cache implementation.
//!
//! Provides an LRU cache for query results with configurable size and TTL.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use manifoldb_core::Value;

use super::CacheMetrics;
use crate::database::QueryResult;

/// Configuration for the query cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache.
    /// Default: 1000
    pub max_entries: usize,

    /// Time-to-live for cache entries.
    /// If `None`, entries never expire based on time.
    /// Default: 5 minutes
    pub ttl: Option<Duration>,

    /// Whether to enable the cache.
    /// Default: true
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Some(Duration::from_secs(300)), // 5 minutes
            enabled: true,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of entries.
    #[must_use]
    pub const fn max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set the TTL for cache entries.
    #[must_use]
    pub const fn ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }

    /// Enable or disable the cache.
    #[must_use]
    pub const fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Create a configuration for a disabled cache.
    #[must_use]
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }
}

/// A unique key for cache entries based on query and parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueryCacheKey {
    /// Hash of the SQL query string.
    query_hash: u64,
    /// Hash of the parameter values.
    params_hash: u64,
}

impl QueryCacheKey {
    /// Create a new cache key from a SQL query and parameters.
    #[must_use]
    pub fn new(sql: &str, params: &[Value]) -> Self {
        let query_hash = hash_string(sql);
        let params_hash = hash_params(params);
        Self { query_hash, params_hash }
    }

    /// Get the combined hash for this key.
    #[must_use]
    pub fn combined_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.query_hash.hash(&mut hasher);
        self.params_hash.hash(&mut hasher);
        hasher.finish()
    }
}

/// A cached query result entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached query result.
    pub result: QueryResult,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// The tables this query accessed (for invalidation).
    pub accessed_tables: Vec<String>,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub fn new(result: QueryResult, accessed_tables: Vec<String>) -> Self {
        let now = Instant::now();
        Self { result, created_at: now, last_accessed: now, accessed_tables }
    }

    /// Check if this entry has expired based on the given TTL.
    #[must_use]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Touch this entry to update its last accessed time.
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }
}

/// Internal cache state for LRU tracking.
struct CacheState {
    /// Map from cache key to entry.
    entries: HashMap<QueryCacheKey, CacheEntry>,
    /// Order of keys for LRU eviction (oldest first).
    lru_order: Vec<QueryCacheKey>,
    /// Reverse index from table name to cache keys that accessed it.
    table_index: HashMap<String, Vec<QueryCacheKey>>,
}

impl CacheState {
    fn new() -> Self {
        Self { entries: HashMap::new(), lru_order: Vec::new(), table_index: HashMap::new() }
    }
}

/// Thread-safe query result cache with LRU eviction.
pub struct QueryCache {
    /// Cache configuration.
    config: CacheConfig,
    /// Cache state protected by a read-write lock.
    state: RwLock<CacheState>,
    /// Cache metrics.
    metrics: Arc<CacheMetrics>,
}

impl QueryCache {
    /// Create a new query cache with the given configuration.
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            state: RwLock::new(CacheState::new()),
            metrics: Arc::new(CacheMetrics::new()),
        }
    }

    /// Create a new disabled cache.
    #[must_use]
    pub fn disabled() -> Self {
        Self::new(CacheConfig::disabled())
    }

    /// Get the cache configuration.
    #[must_use]
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Get the cache metrics.
    #[must_use]
    pub fn metrics(&self) -> Arc<CacheMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Check if the cache is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get a cached result if it exists and hasn't expired.
    ///
    /// Returns `Some(result)` if found and valid, `None` otherwise.
    pub fn get(&self, key: &QueryCacheKey) -> Option<QueryResult> {
        if !self.config.enabled {
            return None;
        }

        // First try a read-only check
        let state = self.state.read().ok()?;
        let entry = match state.entries.get(key) {
            Some(e) => e,
            None => {
                self.metrics.record_miss();
                return None;
            }
        };

        // Check TTL
        if let Some(ttl) = self.config.ttl {
            if entry.is_expired(ttl) {
                // Entry expired, need to remove it
                let result_clone = None;
                drop(state);
                self.remove(key);
                self.metrics.record_miss();
                return result_clone;
            }
        }

        self.metrics.record_hit();
        Some(entry.result.clone())
    }

    /// Update LRU order after a cache hit.
    ///
    /// This is called separately from `get` to minimize write lock contention.
    pub fn touch(&self, key: &QueryCacheKey) {
        if !self.config.enabled {
            return;
        }

        if let Ok(mut state) = self.state.write() {
            if let Some(entry) = state.entries.get_mut(key) {
                entry.touch();

                // Move to end of LRU order
                if let Some(pos) = state.lru_order.iter().position(|k| k == key) {
                    state.lru_order.remove(pos);
                    state.lru_order.push(key.clone());
                }
            }
        }
    }

    /// Insert a query result into the cache.
    ///
    /// If the cache is full, the least recently used entry is evicted.
    pub fn insert(&self, key: QueryCacheKey, result: QueryResult, accessed_tables: Vec<String>) {
        if !self.config.enabled {
            return;
        }

        if let Ok(mut state) = self.state.write() {
            // Evict if necessary
            while state.entries.len() >= self.config.max_entries && !state.lru_order.is_empty() {
                let oldest_key = state.lru_order.remove(0);
                self.remove_from_state(&mut state, &oldest_key);
                self.metrics.record_eviction();
            }

            // Create the entry
            let entry = CacheEntry::new(result, accessed_tables.clone());

            // Update table index
            for table in &accessed_tables {
                state.table_index.entry(table.clone()).or_default().push(key.clone());
            }

            // Insert the entry
            state.entries.insert(key.clone(), entry);
            state.lru_order.push(key);
        }
    }

    /// Remove an entry from the cache.
    pub fn remove(&self, key: &QueryCacheKey) {
        if let Ok(mut state) = self.state.write() {
            self.remove_from_state(&mut state, key);
        }
    }

    /// Remove an entry from the cache state (internal).
    fn remove_from_state(&self, state: &mut CacheState, key: &QueryCacheKey) {
        if let Some(entry) = state.entries.remove(key) {
            // Remove from LRU order
            if let Some(pos) = state.lru_order.iter().position(|k| k == key) {
                state.lru_order.remove(pos);
            }

            // Remove from table index
            for table in &entry.accessed_tables {
                if let Some(keys) = state.table_index.get_mut(table) {
                    keys.retain(|k| k != key);
                    if keys.is_empty() {
                        state.table_index.remove(table);
                    }
                }
            }
        }
    }

    /// Invalidate all cache entries that accessed the given table.
    ///
    /// This should be called after any write operation to the table.
    pub fn invalidate_table(&self, table: &str) {
        if !self.config.enabled {
            return;
        }

        if let Ok(mut state) = self.state.write() {
            // Get all keys that accessed this table
            if let Some(keys) = state.table_index.remove(table) {
                let invalidated_count = keys.len();

                for key in keys {
                    // Remove the entry
                    if let Some(entry) = state.entries.remove(&key) {
                        // Remove from LRU order
                        if let Some(pos) = state.lru_order.iter().position(|k| *k == key) {
                            state.lru_order.remove(pos);
                        }

                        // Remove from other table indexes
                        for other_table in &entry.accessed_tables {
                            if other_table != table {
                                if let Some(other_keys) = state.table_index.get_mut(other_table) {
                                    other_keys.retain(|k| *k != key);
                                }
                            }
                        }
                    }
                }

                self.metrics.record_invalidations(invalidated_count);
            }
        }
    }

    /// Invalidate all cache entries that accessed any of the given tables.
    pub fn invalidate_tables(&self, tables: &[String]) {
        for table in tables {
            self.invalidate_table(table);
        }
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        if let Ok(mut state) = self.state.write() {
            let count = state.entries.len();
            state.entries.clear();
            state.lru_order.clear();
            state.table_index.clear();
            self.metrics.record_invalidations(count);
        }
    }

    /// Get the current number of entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state.read().map(|s| s.entries.len()).unwrap_or(0)
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

/// Hash a string using the default hasher.
fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Hash parameter values.
fn hash_params(params: &[Value]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for param in params {
        hash_value(param, &mut hasher);
    }
    hasher.finish()
}

/// Hash a single Value.
fn hash_value(value: &Value, hasher: &mut DefaultHasher) {
    // Hash a discriminant first to differentiate types
    std::mem::discriminant(value).hash(hasher);

    match value {
        Value::Null => {}
        Value::Bool(b) => b.hash(hasher),
        Value::Int(n) => n.hash(hasher),
        Value::Float(f) => f.to_bits().hash(hasher),
        Value::String(s) => s.hash(hasher),
        Value::Bytes(b) => b.hash(hasher),
        Value::Vector(v) => {
            v.len().hash(hasher);
            for f in v {
                f.to_bits().hash(hasher);
            }
        }
        Value::MultiVector(mv) => {
            mv.len().hash(hasher);
            for v in mv {
                v.len().hash(hasher);
                for f in v {
                    f.to_bits().hash(hasher);
                }
            }
        }
        Value::SparseVector(sv) => {
            sv.len().hash(hasher);
            for (idx, val) in sv {
                idx.hash(hasher);
                val.to_bits().hash(hasher);
            }
        }
        Value::Array(arr) => {
            arr.len().hash(hasher);
            for v in arr {
                hash_value(v, hasher);
            }
        }
        Value::Point { x, y, z, srid } => {
            x.to_bits().hash(hasher);
            y.to_bits().hash(hasher);
            if let Some(z_val) = z {
                z_val.to_bits().hash(hasher);
            }
            srid.hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_creation() {
        let key1 = QueryCacheKey::new("SELECT * FROM users", &[]);
        let key2 = QueryCacheKey::new("SELECT * FROM users", &[]);
        let key3 = QueryCacheKey::new("SELECT * FROM orders", &[]);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_key_with_params() {
        let params1 = vec![Value::Int(1), Value::String("test".to_string())];
        let params2 = vec![Value::Int(1), Value::String("test".to_string())];
        let params3 = vec![Value::Int(2), Value::String("test".to_string())];

        let key1 = QueryCacheKey::new("SELECT * FROM users WHERE id = $1", &params1);
        let key2 = QueryCacheKey::new("SELECT * FROM users WHERE id = $1", &params2);
        let key3 = QueryCacheKey::new("SELECT * FROM users WHERE id = $1", &params3);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_basic_operations() {
        let cache = QueryCache::new(CacheConfig::default());
        let key = QueryCacheKey::new("SELECT * FROM users", &[]);
        let result = QueryResult::empty();

        // Cache miss
        assert!(cache.get(&key).is_none());

        // Insert
        cache.insert(key.clone(), result.clone(), vec!["users".to_string()]);

        // Cache hit
        let cached = cache.get(&key);
        assert!(cached.is_some());

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = QueryCache::new(CacheConfig::default());

        let key1 = QueryCacheKey::new("SELECT * FROM users", &[]);
        let key2 = QueryCacheKey::new("SELECT * FROM orders", &[]);

        cache.insert(key1.clone(), QueryResult::empty(), vec!["users".to_string()]);
        cache.insert(key2.clone(), QueryResult::empty(), vec!["orders".to_string()]);

        assert_eq!(cache.len(), 2);

        // Invalidate users table
        cache.invalidate_table("users");

        assert!(cache.get(&key1).is_none());
        assert!(cache.get(&key2).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let config = CacheConfig::default().max_entries(2);
        let cache = QueryCache::new(config);

        let key1 = QueryCacheKey::new("query1", &[]);
        let key2 = QueryCacheKey::new("query2", &[]);
        let key3 = QueryCacheKey::new("query3", &[]);

        cache.insert(key1.clone(), QueryResult::empty(), vec![]);
        cache.insert(key2.clone(), QueryResult::empty(), vec![]);

        // Access key1 to make it more recently used
        cache.get(&key1);
        cache.touch(&key1);

        // Insert key3, should evict key2 (least recently used)
        cache.insert(key3.clone(), QueryResult::empty(), vec![]);

        assert!(cache.get(&key1).is_some());
        assert!(cache.get(&key2).is_none());
        assert!(cache.get(&key3).is_some());
    }

    #[test]
    fn test_cache_disabled() {
        let cache = QueryCache::disabled();
        let key = QueryCacheKey::new("SELECT * FROM users", &[]);

        cache.insert(key.clone(), QueryResult::empty(), vec![]);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let config = CacheConfig::default().ttl(Some(Duration::from_millis(10)));
        let cache = QueryCache::new(config);
        let key = QueryCacheKey::new("SELECT * FROM users", &[]);

        cache.insert(key.clone(), QueryResult::empty(), vec![]);
        assert!(cache.get(&key).is_some());

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(20));

        // Should be expired now
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let cache = QueryCache::new(CacheConfig::default());

        cache.insert(QueryCacheKey::new("query1", &[]), QueryResult::empty(), vec![]);
        cache.insert(QueryCacheKey::new("query2", &[]), QueryResult::empty(), vec![]);

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_hash_value() {
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        hash_value(&Value::Int(42), &mut hasher1);
        hash_value(&Value::Int(42), &mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_vector() {
        let v1 = Value::Vector(vec![1.0, 2.0, 3.0]);
        let v2 = Value::Vector(vec![1.0, 2.0, 3.0]);
        let v3 = Value::Vector(vec![1.0, 2.0, 4.0]);

        let params1 = vec![v1];
        let params2 = vec![v2];
        let params3 = vec![v3];

        assert_eq!(hash_params(&params1), hash_params(&params2));
        assert_ne!(hash_params(&params1), hash_params(&params3));
    }
}
