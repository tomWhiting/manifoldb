//! Execution context for query execution.
//!
//! The execution context provides access to transaction state,
//! query parameters, and runtime configuration.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use manifoldb_core::{CollectionId, EntityId, Value};
use manifoldb_vector::{Embedding, SearchResult, VectorData, VectorError};

/// A trait for providing access to vector indexes.
///
/// This trait allows type-erased access to vector indexes for query execution.
/// Implementations can wrap HNSW indexes or other vector index types.
pub trait VectorIndexProvider: Send + Sync {
    /// Search for nearest neighbors in the specified index.
    ///
    /// # Arguments
    ///
    /// * `index_name` - The name of the vector index to search
    /// * `query` - The query embedding
    /// * `k` - Number of nearest neighbors to return
    /// * `ef_search` - Optional HNSW ef_search parameter
    ///
    /// # Returns
    ///
    /// A vector of search results sorted by distance, or an error if the
    /// index is not found or the search fails.
    fn search(
        &self,
        index_name: &str,
        query: &Embedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError>;

    /// Check if a vector index exists.
    fn has_index(&self, index_name: &str) -> bool;

    /// Get the dimension of vectors in the specified index.
    fn dimension(&self, index_name: &str) -> Option<usize>;
}

/// A trait for providing collection-based vector operations.
///
/// This trait allows type-erased access to the separated vector storage
/// used by collections with named vectors. Unlike `VectorIndexProvider` which
/// works with entity-property-based vectors, this trait works with the new
/// collection-based vector storage architecture.
pub trait CollectionVectorProvider: Send + Sync {
    /// Store a vector for an entity in a collection.
    ///
    /// # Arguments
    ///
    /// * `collection_id` - The collection ID
    /// * `entity_id` - The entity ID
    /// * `collection_name` - The collection name (for index lookup)
    /// * `vector_name` - The named vector within the collection
    /// * `data` - The vector data to store
    fn upsert_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError>;

    /// Delete a vector from storage and any associated index.
    fn delete_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
        vector_name: &str,
    ) -> Result<bool, VectorError>;

    /// Delete all vectors for an entity in a collection.
    fn delete_entity_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        collection_name: &str,
    ) -> Result<usize, VectorError>;

    /// Get a vector from storage.
    fn get_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<Option<VectorData>, VectorError>;

    /// Get all vectors for an entity.
    fn get_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<std::collections::HashMap<String, VectorData>, VectorError>;

    /// Search for similar vectors using HNSW (if index exists).
    fn search(
        &self,
        collection_name: &str,
        vector_name: &str,
        query: &Embedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError>;
}

use super::graph_accessor::{GraphAccessor, NullGraphAccessor};

/// Execution context for a query.
///
/// The context provides access to:
/// - Query parameters (bound values for placeholders)
/// - Cancellation support
/// - Execution statistics
/// - Runtime configuration
/// - Graph storage access (for graph traversal queries)
/// - Vector index access (optional)
/// - Collection vector storage access (for named vectors)
pub struct ExecutionContext {
    /// Query parameters (1-indexed).
    parameters: HashMap<u32, Value>,
    /// Whether the query has been cancelled.
    cancelled: AtomicBool,
    /// Execution statistics.
    stats: ExecutionStats,
    /// Configuration options.
    config: ExecutionConfig,
    /// Graph accessor for graph traversal operations.
    graph: Arc<dyn GraphAccessor>,
    /// Optional vector index provider for HNSW searches (entity-property based).
    vector_index_provider: Option<Arc<dyn VectorIndexProvider>>,
    /// Optional collection vector provider for named vector storage.
    collection_vector_provider: Option<Arc<dyn CollectionVectorProvider>>,
}

impl ExecutionContext {
    /// Creates a new execution context without graph storage.
    ///
    /// Use [`with_graph`](Self::with_graph) to add graph storage access.
    #[must_use]
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            cancelled: AtomicBool::new(false),
            stats: ExecutionStats::new(),
            config: ExecutionConfig::default(),
            graph: Arc::new(NullGraphAccessor),
            vector_index_provider: None,
            collection_vector_provider: None,
        }
    }

    /// Creates a context with parameters.
    #[must_use]
    pub fn with_parameters(parameters: HashMap<u32, Value>) -> Self {
        Self {
            parameters,
            cancelled: AtomicBool::new(false),
            stats: ExecutionStats::new(),
            config: ExecutionConfig::default(),
            graph: Arc::new(NullGraphAccessor),
            vector_index_provider: None,
            collection_vector_provider: None,
        }
    }

    /// Sets the graph accessor for graph traversal operations.
    ///
    /// This enables the query executor to perform actual graph traversals
    /// using the underlying storage.
    #[must_use]
    pub fn with_graph(mut self, graph: Arc<dyn GraphAccessor>) -> Self {
        self.graph = graph;
        self
    }

    /// Returns a reference to the graph accessor.
    #[inline]
    #[must_use]
    pub fn graph(&self) -> &dyn GraphAccessor {
        self.graph.as_ref()
    }

    /// Returns the graph accessor as an Arc.
    #[inline]
    #[must_use]
    pub fn graph_arc(&self) -> Arc<dyn GraphAccessor> {
        Arc::clone(&self.graph)
    }

    /// Creates a context with a vector index provider.
    #[must_use]
    pub fn with_vector_index_provider(mut self, provider: Arc<dyn VectorIndexProvider>) -> Self {
        self.vector_index_provider = Some(provider);
        self
    }

    /// Sets the vector index provider.
    pub fn set_vector_index_provider(&mut self, provider: Arc<dyn VectorIndexProvider>) {
        self.vector_index_provider = Some(provider);
    }

    /// Returns a reference to the vector index provider if one is set.
    #[must_use]
    pub fn vector_index_provider(&self) -> Option<&dyn VectorIndexProvider> {
        self.vector_index_provider.as_deref()
    }

    /// Returns a clone of the vector index provider Arc if one is set.
    ///
    /// This is useful for operators that need to hold onto the provider
    /// for the duration of their execution.
    #[must_use]
    pub fn vector_index_provider_arc(&self) -> Option<Arc<dyn VectorIndexProvider>> {
        self.vector_index_provider.clone()
    }

    /// Sets the collection vector provider for named vector storage.
    #[must_use]
    pub fn with_collection_vector_provider(
        mut self,
        provider: Arc<dyn CollectionVectorProvider>,
    ) -> Self {
        self.collection_vector_provider = Some(provider);
        self
    }

    /// Sets the collection vector provider.
    pub fn set_collection_vector_provider(&mut self, provider: Arc<dyn CollectionVectorProvider>) {
        self.collection_vector_provider = Some(provider);
    }

    /// Returns a reference to the collection vector provider if one is set.
    #[must_use]
    pub fn collection_vector_provider(&self) -> Option<&dyn CollectionVectorProvider> {
        self.collection_vector_provider.as_deref()
    }

    /// Returns a clone of the collection vector provider Arc if one is set.
    #[must_use]
    pub fn collection_vector_provider_arc(&self) -> Option<Arc<dyn CollectionVectorProvider>> {
        self.collection_vector_provider.clone()
    }

    /// Adds a parameter value.
    pub fn set_parameter(&mut self, index: u32, value: Value) {
        self.parameters.insert(index, value);
    }

    /// Gets a parameter value.
    #[inline]
    #[must_use]
    pub fn get_parameter(&self, index: u32) -> Option<&Value> {
        self.parameters.get(&index)
    }

    /// Returns all parameters.
    #[inline]
    #[must_use]
    pub fn parameters(&self) -> &HashMap<u32, Value> {
        &self.parameters
    }

    /// Cancels the query execution.
    #[inline]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Checks if the query has been cancelled.
    #[inline]
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Returns the execution statistics.
    #[inline]
    #[must_use]
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Returns mutable configuration.
    pub fn config_mut(&mut self) -> &mut ExecutionConfig {
        &mut self.config
    }

    /// Records that rows were read.
    #[inline]
    pub fn record_rows_read(&self, count: u64) {
        self.stats.rows_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Records that rows were produced.
    #[inline]
    pub fn record_rows_produced(&self, count: u64) {
        self.stats.rows_produced.fetch_add(count, Ordering::Relaxed);
    }

    /// Records that rows were filtered.
    #[inline]
    pub fn record_rows_filtered(&self, count: u64) {
        self.stats.rows_filtered.fetch_add(count, Ordering::Relaxed);
    }

    /// Sets the execution configuration.
    #[must_use]
    pub fn with_config(mut self, config: ExecutionConfig) -> Self {
        self.config = config;
        self
    }

    /// Returns the maximum rows in memory limit.
    ///
    /// Returns 0 if the limit is disabled.
    #[inline]
    #[must_use]
    pub fn max_rows_in_memory(&self) -> usize {
        self.config.max_rows_in_memory
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("parameters", &self.parameters)
            .field("cancelled", &self.cancelled)
            .field("stats", &self.stats)
            .field("config", &self.config)
            .field("graph", &"<GraphAccessor>")
            .field("vector_index_provider", &self.vector_index_provider.is_some())
            .finish_non_exhaustive()
    }
}

/// Execution statistics collected during query execution.
#[derive(Debug)]
pub struct ExecutionStats {
    /// When execution started.
    start_time: Instant,
    /// Number of rows read from storage.
    rows_read: AtomicU64,
    /// Number of rows produced by the query.
    rows_produced: AtomicU64,
    /// Number of rows filtered out.
    rows_filtered: AtomicU64,
}

impl ExecutionStats {
    /// Creates new execution statistics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            rows_read: AtomicU64::new(0),
            rows_produced: AtomicU64::new(0),
            rows_filtered: AtomicU64::new(0),
        }
    }

    /// Returns the number of rows read.
    #[inline]
    #[must_use]
    pub fn rows_read(&self) -> u64 {
        self.rows_read.load(Ordering::Relaxed)
    }

    /// Returns the number of rows produced.
    #[inline]
    #[must_use]
    pub fn rows_produced(&self) -> u64 {
        self.rows_produced.load(Ordering::Relaxed)
    }

    /// Returns the number of rows filtered.
    #[inline]
    #[must_use]
    pub fn rows_filtered(&self) -> u64 {
        self.rows_filtered.load(Ordering::Relaxed)
    }

    /// Returns the elapsed execution time.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Default maximum rows in memory (1 million rows).
pub const DEFAULT_MAX_ROWS_IN_MEMORY: usize = 1_000_000;

/// Configuration options for query execution.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum number of rows to buffer in memory.
    pub max_batch_size: usize,
    /// Whether to collect detailed statistics.
    pub collect_stats: bool,
    /// Memory limit in bytes (0 for no limit).
    pub memory_limit: usize,
    /// Maximum number of rows that operators can materialize in memory.
    ///
    /// This limit applies to blocking operators like sort, join, and aggregate
    /// that need to collect rows before producing output. When an operator
    /// exceeds this limit, it returns a `QueryTooLarge` error.
    ///
    /// Set to 0 to disable the limit (not recommended for production).
    /// Default: 1,000,000 rows.
    pub max_rows_in_memory: usize,
}

impl ExecutionConfig {
    /// Creates a new configuration with defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_batch_size: 1024,
            collect_stats: false,
            memory_limit: 0,
            max_rows_in_memory: DEFAULT_MAX_ROWS_IN_MEMORY,
        }
    }

    /// Sets the maximum batch size.
    #[must_use]
    pub const fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Enables statistics collection.
    #[must_use]
    pub const fn with_stats(mut self) -> Self {
        self.collect_stats = true;
        self
    }

    /// Sets the memory limit.
    #[must_use]
    pub const fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Sets the maximum rows that can be materialized in memory.
    ///
    /// This limit applies to blocking operators like sort, join, and aggregate.
    /// Set to 0 to disable the limit (not recommended for production).
    #[must_use]
    pub const fn with_max_rows_in_memory(mut self, limit: usize) -> Self {
        self.max_rows_in_memory = limit;
        self
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle for cancelling query execution.
///
/// Can be shared between threads to allow cancellation from outside
/// the query execution thread.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Creates a new cancellation token.
    #[must_use]
    pub fn new() -> Self {
        Self { cancelled: Arc::new(AtomicBool::new(false)) }
    }

    /// Cancels the associated query.
    #[inline]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Checks if cancellation was requested.
    #[inline]
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_parameters() {
        let mut ctx = ExecutionContext::new();
        ctx.set_parameter(1, Value::Int(42));
        ctx.set_parameter(2, Value::from("hello"));

        assert_eq!(ctx.get_parameter(1), Some(&Value::Int(42)));
        assert_eq!(ctx.get_parameter(2), Some(&Value::from("hello")));
        assert_eq!(ctx.get_parameter(3), None);
    }

    #[test]
    fn context_cancellation() {
        let ctx = ExecutionContext::new();
        assert!(!ctx.is_cancelled());
        ctx.cancel();
        assert!(ctx.is_cancelled());
    }

    #[test]
    fn context_stats() {
        let ctx = ExecutionContext::new();
        ctx.record_rows_read(100);
        ctx.record_rows_produced(50);
        ctx.record_rows_filtered(50);

        assert_eq!(ctx.stats().rows_read(), 100);
        assert_eq!(ctx.stats().rows_produced(), 50);
        assert_eq!(ctx.stats().rows_filtered(), 50);
    }

    #[test]
    fn cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        let token2 = token.clone();
        token.cancel();

        assert!(token.is_cancelled());
        assert!(token2.is_cancelled());
    }
}
