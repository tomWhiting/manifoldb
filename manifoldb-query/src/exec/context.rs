//! Execution context for query execution.
//!
//! The execution context provides access to transaction state,
//! query parameters, and runtime configuration.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use manifoldb_core::Value;

use super::graph_accessor::{GraphAccessor, NullGraphAccessor};

/// Execution context for a query.
///
/// The context provides access to:
/// - Query parameters (bound values for placeholders)
/// - Cancellation support
/// - Execution statistics
/// - Runtime configuration
/// - Graph storage access (for graph traversal queries)
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
    #[must_use]
    pub fn graph(&self) -> &dyn GraphAccessor {
        self.graph.as_ref()
    }

    /// Returns the graph accessor as an Arc.
    #[must_use]
    pub fn graph_arc(&self) -> Arc<dyn GraphAccessor> {
        Arc::clone(&self.graph)
    }

    /// Adds a parameter value.
    pub fn set_parameter(&mut self, index: u32, value: Value) {
        self.parameters.insert(index, value);
    }

    /// Gets a parameter value.
    #[must_use]
    pub fn get_parameter(&self, index: u32) -> Option<&Value> {
        self.parameters.get(&index)
    }

    /// Returns all parameters.
    #[must_use]
    pub fn parameters(&self) -> &HashMap<u32, Value> {
        &self.parameters
    }

    /// Cancels the query execution.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Checks if the query has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Returns the execution statistics.
    #[must_use]
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Returns mutable configuration.
    pub fn config_mut(&mut self) -> &mut ExecutionConfig {
        &mut self.config
    }

    /// Records that rows were read.
    pub fn record_rows_read(&self, count: u64) {
        self.stats.rows_read.fetch_add(count, Ordering::Relaxed);
    }

    /// Records that rows were produced.
    pub fn record_rows_produced(&self, count: u64) {
        self.stats.rows_produced.fetch_add(count, Ordering::Relaxed);
    }

    /// Records that rows were filtered.
    pub fn record_rows_filtered(&self, count: u64) {
        self.stats.rows_filtered.fetch_add(count, Ordering::Relaxed);
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
            .finish()
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
    #[must_use]
    pub fn rows_read(&self) -> u64 {
        self.rows_read.load(Ordering::Relaxed)
    }

    /// Returns the number of rows produced.
    #[must_use]
    pub fn rows_produced(&self) -> u64 {
        self.rows_produced.load(Ordering::Relaxed)
    }

    /// Returns the number of rows filtered.
    #[must_use]
    pub fn rows_filtered(&self) -> u64 {
        self.rows_filtered.load(Ordering::Relaxed)
    }

    /// Returns the elapsed execution time.
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

/// Configuration options for query execution.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum number of rows to buffer in memory.
    pub max_batch_size: usize,
    /// Whether to collect detailed statistics.
    pub collect_stats: bool,
    /// Memory limit in bytes (0 for no limit).
    pub memory_limit: usize,
}

impl ExecutionConfig {
    /// Creates a new configuration with defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_batch_size: 1024, collect_stats: false, memory_limit: 0 }
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
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Checks if cancellation was requested.
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
