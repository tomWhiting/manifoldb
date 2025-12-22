//! Prepared statements with query plan caching.
//!
//! This module provides [`PreparedStatement`] which caches the parsed AST and
//! logical plan for repeated query execution. Plans are automatically invalidated
//! when the schema changes (e.g., after DDL operations like CREATE/DROP TABLE/INDEX).
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::{Database, Value};
//!
//! let db = Database::in_memory()?;
//!
//! // Prepare a statement once
//! let stmt = db.prepare("SELECT * FROM users WHERE age > $1")?;
//!
//! // Execute multiple times with different parameters
//! let young = stmt.query(&db, &[Value::Int(18)])?;
//! let old = stmt.query(&db, &[Value::Int(65)])?;
//! ```

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use manifoldb_query::ast::Statement;
use manifoldb_query::plan::{LogicalPlan, PhysicalPlan, PhysicalPlanner, PlanBuilder};
use manifoldb_query::ExtendedParser;

use crate::error::{Error, Result};

/// A prepared statement with cached AST and query plan.
///
/// Prepared statements amortize the cost of parsing and planning by caching
/// these results. The cached plan is automatically invalidated when the
/// schema version changes.
///
/// # Thread Safety
///
/// `PreparedStatement` is `Send + Sync` and can be safely shared across threads.
/// Multiple threads can execute the same prepared statement concurrently.
#[derive(Debug)]
pub struct PreparedStatement {
    /// The original SQL text.
    sql: String,
    /// The cached parsed AST.
    ast: Statement,
    /// The cached logical plan.
    logical_plan: LogicalPlan,
    /// The cached physical plan.
    physical_plan: PhysicalPlan,
    /// The schema version when this statement was prepared.
    schema_version: u64,
    /// Tables accessed by this query (for cache invalidation).
    accessed_tables: HashSet<String>,
    /// Whether this is a DML statement (INSERT/UPDATE/DELETE).
    is_dml: bool,
    /// Whether this is a DDL statement (CREATE/DROP).
    is_ddl: bool,
}

impl PreparedStatement {
    /// Parse and plan a SQL statement.
    ///
    /// # Arguments
    ///
    /// * `sql` - The SQL statement to prepare
    /// * `schema_version` - The current schema version for invalidation tracking
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL cannot be parsed or planned.
    pub fn new(sql: &str, schema_version: u64) -> Result<Self> {
        // Parse the SQL
        let ast = ExtendedParser::parse_single(sql)?;

        // Build logical plan
        let mut builder = PlanBuilder::new();
        let logical_plan =
            builder.build_statement(&ast).map_err(|e| Error::Parse(e.to_string()))?;

        // Build physical plan
        let planner = PhysicalPlanner::new();
        let physical_plan = planner.plan(&logical_plan);

        // Extract accessed tables from the plan
        let accessed_tables = Self::extract_tables(&logical_plan);

        // Determine statement type
        let (is_dml, is_ddl) = Self::classify_statement(&logical_plan);

        Ok(Self {
            sql: sql.to_string(),
            ast,
            logical_plan,
            physical_plan,
            schema_version,
            accessed_tables,
            is_dml,
            is_ddl,
        })
    }

    /// Returns the original SQL text.
    #[must_use]
    pub fn sql(&self) -> &str {
        &self.sql
    }

    /// Returns the schema version when this statement was prepared.
    #[must_use]
    pub fn schema_version(&self) -> u64 {
        self.schema_version
    }

    /// Returns the parsed AST.
    #[must_use]
    pub fn ast(&self) -> &Statement {
        &self.ast
    }

    /// Returns the logical plan.
    #[must_use]
    pub fn logical_plan(&self) -> &LogicalPlan {
        &self.logical_plan
    }

    /// Returns the physical plan.
    #[must_use]
    pub fn physical_plan(&self) -> &PhysicalPlan {
        &self.physical_plan
    }

    /// Returns the tables accessed by this statement.
    #[must_use]
    pub fn accessed_tables(&self) -> &HashSet<String> {
        &self.accessed_tables
    }

    /// Returns true if this is a DML statement (INSERT/UPDATE/DELETE).
    #[must_use]
    pub fn is_dml(&self) -> bool {
        self.is_dml
    }

    /// Returns true if this is a DDL statement (CREATE/DROP TABLE/INDEX).
    #[must_use]
    pub fn is_ddl(&self) -> bool {
        self.is_ddl
    }

    /// Returns true if this is a query (SELECT).
    #[must_use]
    pub fn is_query(&self) -> bool {
        !self.is_dml && !self.is_ddl
    }

    /// Check if this prepared statement is still valid for the given schema version.
    ///
    /// A prepared statement becomes invalid when the schema changes after it
    /// was prepared (e.g., due to DDL operations).
    #[must_use]
    pub fn is_valid(&self, current_schema_version: u64) -> bool {
        self.schema_version == current_schema_version
    }

    /// Extract table names from a logical plan.
    fn extract_tables(plan: &LogicalPlan) -> HashSet<String> {
        let mut tables = HashSet::new();
        Self::extract_tables_recursive(plan, &mut tables);
        tables
    }

    fn extract_tables_recursive(plan: &LogicalPlan, tables: &mut HashSet<String>) {
        match plan {
            LogicalPlan::Scan(scan_node) => {
                tables.insert(scan_node.table_name.clone());
            }
            LogicalPlan::Insert { table, input, .. } => {
                tables.insert(table.clone());
                Self::extract_tables_recursive(input, tables);
            }
            LogicalPlan::Update { table, .. } => {
                tables.insert(table.clone());
            }
            LogicalPlan::Delete { table, .. } => {
                tables.insert(table.clone());
            }
            LogicalPlan::CreateTable(node) => {
                tables.insert(node.name.clone());
            }
            LogicalPlan::DropTable(node) => {
                for name in &node.names {
                    tables.insert(name.clone());
                }
            }
            LogicalPlan::CreateIndex(node) => {
                tables.insert(node.table.clone());
            }
            LogicalPlan::DropIndex(_) => {
                // DROP INDEX doesn't directly reference a table in the node
            }
            _ => {
                // Recurse into children
                for child in plan.children() {
                    Self::extract_tables_recursive(child, tables);
                }
            }
        }
    }

    /// Classify the statement type.
    fn classify_statement(plan: &LogicalPlan) -> (bool, bool) {
        match plan {
            LogicalPlan::Insert { .. }
            | LogicalPlan::Update { .. }
            | LogicalPlan::Delete { .. } => (true, false),
            LogicalPlan::CreateTable(_)
            | LogicalPlan::DropTable(_)
            | LogicalPlan::CreateIndex(_)
            | LogicalPlan::DropIndex(_) => (false, true),
            _ => (false, false),
        }
    }
}

/// A cache of prepared statements.
///
/// This cache stores prepared statements by their SQL text and automatically
/// invalidates them when the schema changes.
#[derive(Debug)]
pub struct PreparedStatementCache {
    /// The cached prepared statements.
    statements: RwLock<std::collections::HashMap<String, Arc<PreparedStatement>>>,
    /// Maximum number of cached statements.
    max_size: usize,
    /// Current schema version.
    schema_version: AtomicU64,
    /// Cache metrics.
    hits: AtomicU64,
    misses: AtomicU64,
    invalidations: AtomicU64,
}

impl PreparedStatementCache {
    /// Create a new prepared statement cache with the given maximum size.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            statements: RwLock::new(std::collections::HashMap::new()),
            max_size,
            schema_version: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            invalidations: AtomicU64::new(0),
        }
    }

    /// Create a new cache with default settings.
    #[must_use]
    pub fn default_cache() -> Self {
        Self::new(1000)
    }

    /// Get or create a prepared statement for the given SQL.
    ///
    /// If the statement is already cached and the schema hasn't changed,
    /// returns the cached statement. Otherwise, prepares a new statement.
    pub fn get_or_prepare(&self, sql: &str) -> Result<Arc<PreparedStatement>> {
        let schema_version = self.schema_version.load(Ordering::Acquire);

        // Try to get from cache
        {
            let cache = self.statements.read().unwrap();
            if let Some(stmt) = cache.get(sql) {
                if stmt.is_valid(schema_version) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(Arc::clone(stmt));
                }
            }
        }

        // Cache miss or invalid - prepare a new statement
        self.misses.fetch_add(1, Ordering::Relaxed);
        let stmt = Arc::new(PreparedStatement::new(sql, schema_version)?);

        // Insert into cache
        {
            let mut cache = self.statements.write().unwrap();

            // If cache is full, remove oldest entries (simple LRU-ish eviction)
            if cache.len() >= self.max_size {
                // Remove 10% of entries (minimum 1)
                let to_remove = (self.max_size / 10).max(1);
                let keys_to_remove: Vec<String> = cache.keys().take(to_remove).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }

            cache.insert(sql.to_string(), Arc::clone(&stmt));
        }

        Ok(stmt)
    }

    /// Prepare a statement (always creates a new one, bypassing cache).
    pub fn prepare(&self, sql: &str) -> Result<Arc<PreparedStatement>> {
        let schema_version = self.schema_version.load(Ordering::Acquire);
        Ok(Arc::new(PreparedStatement::new(sql, schema_version)?))
    }

    /// Update the schema version, invalidating all cached statements.
    pub fn set_schema_version(&self, version: u64) {
        let old_version = self.schema_version.swap(version, Ordering::Release);
        if old_version != version {
            self.invalidations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get the current schema version.
    #[must_use]
    pub fn schema_version(&self) -> u64 {
        self.schema_version.load(Ordering::Acquire)
    }

    /// Clear all cached statements.
    pub fn clear(&self) {
        let mut cache = self.statements.write().unwrap();
        cache.clear();
    }

    /// Invalidate cached statements that access specific tables.
    pub fn invalidate_tables(&self, tables: &[String]) {
        if tables.is_empty() {
            return;
        }

        let tables_set: HashSet<&String> = tables.iter().collect();
        let mut cache = self.statements.write().unwrap();

        cache.retain(|_, stmt| !stmt.accessed_tables().iter().any(|t| tables_set.contains(t)));
    }

    /// Returns the number of cached statements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.statements.read().unwrap().len()
    }

    /// Returns true if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.statements.read().unwrap().is_empty()
    }

    /// Returns the cache hit count.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Returns the cache miss count.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Returns the number of invalidations.
    #[must_use]
    pub fn invalidations(&self) -> u64 {
        self.invalidations.load(Ordering::Relaxed)
    }

    /// Returns the cache hit rate as a percentage.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            None
        } else {
            Some((hits as f64 / total as f64) * 100.0)
        }
    }

    /// Reset cache metrics.
    pub fn reset_metrics(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.invalidations.store(0, Ordering::Relaxed);
    }
}

impl Default for PreparedStatementCache {
    fn default() -> Self {
        Self::default_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_select() {
        let stmt = PreparedStatement::new("SELECT * FROM users WHERE id = $1", 0).unwrap();
        assert!(!stmt.is_dml());
        assert!(!stmt.is_ddl());
        assert!(stmt.is_query());
        assert!(stmt.accessed_tables().contains("users"));
    }

    #[test]
    fn test_prepare_insert() {
        let stmt = PreparedStatement::new("INSERT INTO users (name) VALUES ($1)", 0).unwrap();
        assert!(stmt.is_dml());
        assert!(!stmt.is_ddl());
        assert!(!stmt.is_query());
        assert!(stmt.accessed_tables().contains("users"));
    }

    #[test]
    fn test_prepare_update() {
        let stmt = PreparedStatement::new("UPDATE users SET name = $1 WHERE id = $2", 0).unwrap();
        assert!(stmt.is_dml());
        assert!(!stmt.is_ddl());
        assert!(stmt.accessed_tables().contains("users"));
    }

    #[test]
    fn test_prepare_delete() {
        let stmt = PreparedStatement::new("DELETE FROM users WHERE id = $1", 0).unwrap();
        assert!(stmt.is_dml());
        assert!(!stmt.is_ddl());
        assert!(stmt.accessed_tables().contains("users"));
    }

    #[test]
    fn test_schema_version_validity() {
        let stmt = PreparedStatement::new("SELECT * FROM users", 5).unwrap();
        assert!(stmt.is_valid(5));
        assert!(!stmt.is_valid(6));
        assert!(!stmt.is_valid(4));
    }

    #[test]
    fn test_cache_basic() {
        let cache = PreparedStatementCache::new(100);

        // First access - cache miss
        let stmt1 = cache.get_or_prepare("SELECT * FROM users").unwrap();
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);

        // Second access - cache hit
        let stmt2 = cache.get_or_prepare("SELECT * FROM users").unwrap();
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        // Same statement
        assert!(Arc::ptr_eq(&stmt1, &stmt2));
    }

    #[test]
    fn test_cache_invalidation_on_schema_change() {
        let cache = PreparedStatementCache::new(100);

        // Prepare at version 0
        let stmt1 = cache.get_or_prepare("SELECT * FROM users").unwrap();
        assert_eq!(stmt1.schema_version(), 0);

        // Change schema version
        cache.set_schema_version(1);

        // Access again - should re-prepare due to schema change
        let stmt2 = cache.get_or_prepare("SELECT * FROM users").unwrap();
        assert_eq!(stmt2.schema_version(), 1);

        // Different statement instance
        assert!(!Arc::ptr_eq(&stmt1, &stmt2));
    }

    #[test]
    fn test_cache_table_invalidation() {
        let cache = PreparedStatementCache::new(100);

        // Prepare statements for different tables
        let _ = cache.get_or_prepare("SELECT * FROM users").unwrap();
        let _ = cache.get_or_prepare("SELECT * FROM orders").unwrap();
        assert_eq!(cache.len(), 2);

        // Invalidate users table
        cache.invalidate_tables(&["users".to_string()]);
        assert_eq!(cache.len(), 1);

        // Orders should still be cached
        let _ = cache.get_or_prepare("SELECT * FROM orders").unwrap();
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_clear() {
        let cache = PreparedStatementCache::new(100);

        let _ = cache.get_or_prepare("SELECT * FROM users").unwrap();
        let _ = cache.get_or_prepare("SELECT * FROM orders").unwrap();
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = PreparedStatementCache::new(5);

        // Fill the cache
        for i in 0..10 {
            let _ = cache.get_or_prepare(&format!("SELECT * FROM table{}", i)).unwrap();
        }

        // Cache should not exceed max size (though it may temporarily during eviction)
        assert!(cache.len() <= 5);
    }
}
