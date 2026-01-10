//! Extract table names from SQL statements.
//!
//! This module provides functionality to extract table names from SQL statements
//! for cache invalidation purposes.

use manifoldb_query::plan::{LogicalPlan, PlanBuilder};
use manifoldb_query::ExtendedParser;

/// Extract table names from a SQL statement.
///
/// This parses the SQL and walks the logical plan to find all referenced tables.
/// Used for cache invalidation: when a table is modified, all cached queries
/// that reference that table need to be invalidated.
///
/// # Arguments
///
/// * `sql` - The SQL statement to analyze
///
/// # Returns
///
/// A vector of table names referenced in the statement.
///
/// # Examples
///
/// ```ignore
/// let tables = extract_tables_from_sql("SELECT * FROM users");
/// assert_eq!(tables, vec!["users"]);
///
/// let tables = extract_tables_from_sql("INSERT INTO orders VALUES (1)");
/// assert_eq!(tables, vec!["orders"]);
/// ```
#[must_use]
pub fn extract_tables_from_sql(sql: &str) -> Vec<String> {
    // Try to parse the SQL and extract tables from the logical plan
    if let Ok(stmt) = ExtendedParser::parse_single(sql) {
        let mut builder = PlanBuilder::new();
        if let Ok(plan) = builder.build_statement(&stmt) {
            return extract_tables_from_plan(&plan);
        }
    }

    // Fall back to simple regex-based extraction if parsing fails
    extract_tables_simple(sql)
}

/// Extract table names from a logical plan.
fn extract_tables_from_plan(plan: &LogicalPlan) -> Vec<String> {
    let mut tables = Vec::new();
    collect_tables_from_plan(plan, &mut tables);
    // Remove duplicates
    tables.sort();
    tables.dedup();
    tables
}

/// Recursively collect table names from a logical plan.
fn collect_tables_from_plan(plan: &LogicalPlan, tables: &mut Vec<String>) {
    match plan {
        LogicalPlan::Scan(node) => {
            tables.push(node.table_name.clone());
        }

        LogicalPlan::Insert { table, input, .. } => {
            tables.push(table.clone());
            collect_tables_from_plan(input, tables);
        }

        LogicalPlan::Update { table, .. } => {
            tables.push(table.clone());
        }

        LogicalPlan::Delete { table, .. } => {
            tables.push(table.clone());
        }

        LogicalPlan::CreateTable(node) => {
            tables.push(node.name.clone());
        }

        LogicalPlan::DropTable(node) => {
            for name in &node.names {
                tables.push(name.clone());
            }
        }

        LogicalPlan::TruncateTable(node) => {
            for name in &node.names {
                tables.push(name.clone());
            }
        }

        LogicalPlan::AlterTable(node) => {
            tables.push(node.name.clone());
        }

        LogicalPlan::CreateIndex(node) => {
            tables.push(node.table.clone());
        }

        LogicalPlan::DropIndex(_) | LogicalPlan::AlterIndex(_) => {
            // Index operations don't directly reference tables
        }

        LogicalPlan::Project { input, .. }
        | LogicalPlan::Filter { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Alias { input, .. }
        | LogicalPlan::Unwind { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Expand { input, .. }
        | LogicalPlan::PathScan { input, .. }
        | LogicalPlan::ShortestPath { input, .. }
        | LogicalPlan::AnnSearch { input, .. }
        | LogicalPlan::VectorDistance { input, .. }
        | LogicalPlan::HybridSearch { input, .. }
        | LogicalPlan::Window { input, .. } => {
            collect_tables_from_plan(input, tables);
        }

        LogicalPlan::Join { left, right, .. } => {
            collect_tables_from_plan(left, tables);
            collect_tables_from_plan(right, tables);
        }

        LogicalPlan::SetOp { left, right, .. } => {
            collect_tables_from_plan(left, tables);
            collect_tables_from_plan(right, tables);
        }

        LogicalPlan::RecursiveCTE { initial, recursive, .. } => {
            collect_tables_from_plan(initial, tables);
            collect_tables_from_plan(recursive, tables);
        }

        LogicalPlan::CallSubquery { input, subquery, .. } => {
            collect_tables_from_plan(input, tables);
            collect_tables_from_plan(subquery, tables);
        }

        LogicalPlan::Union { inputs, .. } => {
            for input in inputs {
                collect_tables_from_plan(input, tables);
            }
        }

        LogicalPlan::Values(_)
        | LogicalPlan::Empty { .. }
        | LogicalPlan::CreateCollection(_)
        | LogicalPlan::DropCollection(_)
        | LogicalPlan::CreateView(_)
        | LogicalPlan::DropView(_)
        | LogicalPlan::CreateSchema(_)
        | LogicalPlan::AlterSchema(_)
        | LogicalPlan::DropSchema(_)
        | LogicalPlan::CreateFunction(_)
        | LogicalPlan::DropFunction(_)
        | LogicalPlan::DropTrigger(_)
        | LogicalPlan::ProcedureCall(_) => {
            // These don't reference tables
        }

        LogicalPlan::CreateTrigger(node) => {
            // Triggers reference a table
            tables.push(node.table.clone());
        }

        // Graph DML operations - may have an optional input plan
        LogicalPlan::GraphCreate { input, .. } | LogicalPlan::GraphMerge { input, .. } => {
            if let Some(input) = input {
                collect_tables_from_plan(input, tables);
            }
        }

        // Graph SET/DELETE/REMOVE/FOREACH operations - have a required input plan
        LogicalPlan::GraphSet { input, .. }
        | LogicalPlan::GraphDelete { input, .. }
        | LogicalPlan::GraphRemove { input, .. }
        | LogicalPlan::GraphForeach { input, .. } => {
            collect_tables_from_plan(input, tables);
        }

        // Transaction control statements don't reference tables
        LogicalPlan::BeginTransaction(_)
        | LogicalPlan::Commit(_)
        | LogicalPlan::Rollback(_)
        | LogicalPlan::Savepoint(_)
        | LogicalPlan::ReleaseSavepoint(_)
        | LogicalPlan::SetTransaction(_) => {
            // Transaction statements don't reference tables
        }

        // Utility statements - some may reference tables
        LogicalPlan::ExplainAnalyze(node) => {
            // Analyze the inner plan for table references
            collect_tables_from_plan(&node.input, tables);
        }
        LogicalPlan::Vacuum(node) => {
            // VACUUM may target a specific table
            if let Some(table) = &node.table {
                tables.push(table.to_string());
            }
        }
        LogicalPlan::Analyze(node) => {
            // ANALYZE may target a specific table
            if let Some(table) = &node.table {
                tables.push(table.to_string());
            }
        }
        LogicalPlan::Copy(node) => {
            // COPY may target a table or have a query plan
            use manifoldb_query::ast::CopyTarget;
            if let CopyTarget::Table { name, .. } = &node.target {
                tables.push(name.to_string());
            }
            if let Some(query_plan) = &node.query_plan {
                collect_tables_from_plan(query_plan, tables);
            }
        }
        LogicalPlan::SetSession(_)
        | LogicalPlan::Show(_)
        | LogicalPlan::Reset(_)
        | LogicalPlan::ShowProcedures(_) => {
            // Session variable and utility statements don't reference tables
        }
    }
}

/// Simple fallback table extraction using basic parsing.
///
/// This is used when the SQL parser fails. It looks for common patterns
/// like `FROM table`, `INTO table`, `UPDATE table`, etc.
fn extract_tables_simple(sql: &str) -> Vec<String> {
    let mut tables = Vec::new();
    let normalized = sql.to_uppercase();
    let words: Vec<&str> = sql.split_whitespace().collect();
    let upper_words: Vec<String> = normalized.split_whitespace().map(String::from).collect();

    // Look for patterns like FROM table, INTO table, UPDATE table, etc.
    for (i, word) in upper_words.iter().enumerate() {
        let next_idx = i + 1;
        if next_idx < words.len() {
            let is_table_keyword =
                matches!(word.as_str(), "FROM" | "INTO" | "UPDATE" | "TABLE" | "JOIN");

            if is_table_keyword {
                let table_name = words[next_idx]
                    .trim_matches(|c: char| c == '(' || c == ')' || c == ',')
                    .to_string();

                // Skip SQL keywords that might follow
                if !is_sql_keyword(&table_name) && !table_name.is_empty() {
                    tables.push(table_name);
                }
            }
        }
    }

    tables.sort();
    tables.dedup();
    tables
}

/// Check if a word is a SQL keyword (to avoid false positives).
fn is_sql_keyword(word: &str) -> bool {
    let upper = word.to_uppercase();
    matches!(
        upper.as_str(),
        "SELECT"
            | "FROM"
            | "WHERE"
            | "AND"
            | "OR"
            | "NOT"
            | "IN"
            | "LIKE"
            | "BETWEEN"
            | "IS"
            | "NULL"
            | "TRUE"
            | "FALSE"
            | "ORDER"
            | "BY"
            | "GROUP"
            | "HAVING"
            | "LIMIT"
            | "OFFSET"
            | "AS"
            | "ON"
            | "JOIN"
            | "LEFT"
            | "RIGHT"
            | "INNER"
            | "OUTER"
            | "FULL"
            | "CROSS"
            | "NATURAL"
            | "USING"
            | "UNION"
            | "INTERSECT"
            | "EXCEPT"
            | "ALL"
            | "DISTINCT"
            | "SET"
            | "VALUES"
            | "INSERT"
            | "UPDATE"
            | "DELETE"
            | "CREATE"
            | "DROP"
            | "ALTER"
            | "INDEX"
            | "TABLE"
            | "IF"
            | "EXISTS"
            | "CASCADE"
            | "RESTRICT"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_from_select() {
        let tables = extract_tables_from_sql("SELECT * FROM users");
        assert_eq!(tables, vec!["users"]);
    }

    #[test]
    fn test_extract_from_select_with_alias() {
        let tables = extract_tables_from_sql("SELECT * FROM users u WHERE u.id = 1");
        assert_eq!(tables, vec!["users"]);
    }

    #[test]
    fn test_extract_from_join() {
        let tables =
            extract_tables_from_sql("SELECT * FROM users u JOIN orders o ON u.id = o.user_id");
        assert!(tables.contains(&"users".to_string()));
        assert!(tables.contains(&"orders".to_string()));
    }

    #[test]
    fn test_extract_from_insert() {
        let tables = extract_tables_from_sql("INSERT INTO users (name, age) VALUES ('Alice', 30)");
        assert_eq!(tables, vec!["users"]);
    }

    #[test]
    fn test_extract_from_update() {
        let tables = extract_tables_from_sql("UPDATE users SET name = 'Bob' WHERE id = 1");
        assert_eq!(tables, vec!["users"]);
    }

    #[test]
    fn test_extract_from_delete() {
        let tables = extract_tables_from_sql("DELETE FROM users WHERE id = 1");
        assert_eq!(tables, vec!["users"]);
    }

    #[test]
    fn test_extract_empty_for_invalid_sql() {
        // Invalid SQL might still extract something via simple parsing
        let tables = extract_tables_from_sql("INVALID SQL !!!");
        assert!(tables.is_empty());
    }

    #[test]
    fn test_is_sql_keyword() {
        assert!(is_sql_keyword("SELECT"));
        assert!(is_sql_keyword("from"));
        assert!(is_sql_keyword("WHERE"));
        assert!(!is_sql_keyword("users"));
        assert!(!is_sql_keyword("orders"));
    }
}
