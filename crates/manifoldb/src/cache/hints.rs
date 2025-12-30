//! Cache hint parsing for SQL queries.
//!
//! Supports Oracle-style hint syntax:
//! - `/*+ CACHE */` - Force caching of the query result
//! - `/*+ NO_CACHE */` - Skip caching for this query

/// Cache hint extracted from a SQL query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheHint {
    /// Use default caching behavior.
    Default,
    /// Force caching of the query result.
    Cache,
    /// Skip caching for this query.
    NoCache,
}

impl Default for CacheHint {
    fn default() -> Self {
        Self::Default
    }
}

/// Extract a cache hint from a SQL query string.
///
/// Returns `(hint, cleaned_sql)` where `cleaned_sql` has the hint removed.
///
/// # Examples
///
/// ```ignore
/// use manifoldb::cache::extract_cache_hint;
///
/// let (hint, sql) = extract_cache_hint("/*+ CACHE */ SELECT * FROM users");
/// assert_eq!(hint, CacheHint::Cache);
/// assert_eq!(sql, "SELECT * FROM users");
///
/// let (hint, sql) = extract_cache_hint("/*+ NO_CACHE */ SELECT * FROM users");
/// assert_eq!(hint, CacheHint::NoCache);
/// assert_eq!(sql, "SELECT * FROM users");
/// ```
#[must_use]
pub fn extract_cache_hint(sql: &str) -> (CacheHint, String) {
    let trimmed = sql.trim();

    // Look for hint at the start of the query
    if let Some(rest) = trimmed.strip_prefix("/*+") {
        if let Some(end_pos) = rest.find("*/") {
            let hint_content = rest[..end_pos].trim().to_uppercase();
            let remaining_sql = rest[end_pos + 2..].trim().to_string();

            let hint = match hint_content.as_str() {
                "CACHE" => CacheHint::Cache,
                "NO_CACHE" | "NOCACHE" => CacheHint::NoCache,
                _ => CacheHint::Default,
            };

            return (hint, remaining_sql);
        }
    }

    // No hint found
    (CacheHint::Default, sql.to_string())
}

/// Check if a SQL statement should be cached based on its type.
///
/// Only SELECT queries are cacheable. DML and DDL statements are not.
///
/// This function is provided for external use when you want to check
/// if a statement is cacheable before executing it.
///
/// TODO(v0.2): Expose this function via the public API for cache control.
#[must_use]
#[allow(dead_code)]
pub fn is_cacheable_statement(sql: &str) -> bool {
    let normalized = sql.trim().to_uppercase();

    // Only cache SELECT statements
    if normalized.starts_with("SELECT") || normalized.starts_with("/*+") {
        // Check if it's actually a SELECT after the hint
        if normalized.starts_with("/*+") {
            if let Some(end) = normalized.find("*/") {
                let after_hint = normalized[end + 2..].trim();
                return after_hint.starts_with("SELECT");
            }
        }
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_cache_hint() {
        let (hint, sql) = extract_cache_hint("/*+ CACHE */ SELECT * FROM users");
        assert_eq!(hint, CacheHint::Cache);
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn test_extract_no_cache_hint() {
        let (hint, sql) = extract_cache_hint("/*+ NO_CACHE */ SELECT * FROM orders");
        assert_eq!(hint, CacheHint::NoCache);
        assert_eq!(sql, "SELECT * FROM orders");
    }

    #[test]
    fn test_extract_nocache_hint() {
        let (hint, sql) = extract_cache_hint("/*+ NOCACHE */ SELECT * FROM products");
        assert_eq!(hint, CacheHint::NoCache);
        assert_eq!(sql, "SELECT * FROM products");
    }

    #[test]
    fn test_no_hint() {
        let (hint, sql) = extract_cache_hint("SELECT * FROM users");
        assert_eq!(hint, CacheHint::Default);
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn test_unknown_hint() {
        let (hint, sql) = extract_cache_hint("/*+ UNKNOWN */ SELECT * FROM users");
        assert_eq!(hint, CacheHint::Default);
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn test_case_insensitive_hint() {
        let (hint, _) = extract_cache_hint("/*+ cache */ SELECT * FROM users");
        assert_eq!(hint, CacheHint::Cache);

        let (hint, _) = extract_cache_hint("/*+ Cache */ SELECT * FROM users");
        assert_eq!(hint, CacheHint::Cache);
    }

    #[test]
    fn test_hint_with_whitespace() {
        let (hint, sql) = extract_cache_hint("  /*+  CACHE  */   SELECT * FROM users  ");
        assert_eq!(hint, CacheHint::Cache);
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn test_is_cacheable_select() {
        assert!(is_cacheable_statement("SELECT * FROM users"));
        assert!(is_cacheable_statement("  SELECT * FROM users  "));
        assert!(is_cacheable_statement("/*+ CACHE */ SELECT * FROM users"));
    }

    #[test]
    fn test_is_not_cacheable() {
        assert!(!is_cacheable_statement("INSERT INTO users VALUES (1)"));
        assert!(!is_cacheable_statement("UPDATE users SET name = 'test'"));
        assert!(!is_cacheable_statement("DELETE FROM users"));
        assert!(!is_cacheable_statement("CREATE TABLE users (id INT)"));
        assert!(!is_cacheable_statement("DROP TABLE users"));
    }
}
