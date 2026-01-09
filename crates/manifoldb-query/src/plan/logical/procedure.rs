//! Procedure call logical plan node.
//!
//! This module defines the logical plan node for CALL/YIELD statements.

use super::expr::LogicalExpr;

/// Logical plan node for procedure calls.
///
/// Represents a call to a procedure with arguments and optional YIELD clause.
///
/// # Example
///
/// ```text
/// CALL algo.pageRank('nodes', 'edges', 0.85)
/// YIELD node, score
/// WHERE score > 0.1
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProcedureCallNode {
    /// The fully-qualified procedure name (e.g., "algo.pageRank").
    pub procedure_name: String,
    /// Arguments to pass to the procedure.
    pub arguments: Vec<LogicalExpr>,
    /// Columns to yield from the procedure result.
    /// Empty means all columns (YIELD *).
    pub yield_columns: Vec<YieldColumn>,
    /// Optional filter to apply to yielded results.
    pub filter: Option<LogicalExpr>,
}

/// A column yielded from a procedure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YieldColumn {
    /// The original column name from the procedure.
    pub name: String,
    /// Optional alias for the column.
    pub alias: Option<String>,
}

impl YieldColumn {
    /// Creates a new yield column.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), alias: None }
    }

    /// Creates a new yield column with alias.
    #[must_use]
    pub fn with_alias(name: impl Into<String>, alias: impl Into<String>) -> Self {
        Self { name: name.into(), alias: Some(alias.into()) }
    }

    /// Returns the output name (alias if present, otherwise name).
    #[must_use]
    pub fn output_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

impl ProcedureCallNode {
    /// Creates a new procedure call node.
    #[must_use]
    pub fn new(procedure_name: impl Into<String>, arguments: Vec<LogicalExpr>) -> Self {
        Self {
            procedure_name: procedure_name.into(),
            arguments,
            yield_columns: vec![],
            filter: None,
        }
    }

    /// Sets the yield columns.
    #[must_use]
    pub fn with_yields(mut self, columns: Vec<YieldColumn>) -> Self {
        self.yield_columns = columns;
        self
    }

    /// Adds a yield column.
    #[must_use]
    pub fn yield_column(mut self, column: YieldColumn) -> Self {
        self.yield_columns.push(column);
        self
    }

    /// Sets the filter predicate.
    #[must_use]
    pub fn with_filter(mut self, predicate: LogicalExpr) -> Self {
        self.filter = Some(predicate);
        self
    }

    /// Returns true if this is a "yield all" call (no explicit columns).
    #[must_use]
    pub fn is_yield_all(&self) -> bool {
        self.yield_columns.is_empty()
    }

    /// Returns the output column names.
    #[must_use]
    pub fn output_columns(&self) -> Vec<&str> {
        self.yield_columns.iter().map(|c| c.output_name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedure_call_node_basic() {
        let node = ProcedureCallNode::new(
            "algo.pageRank",
            vec![LogicalExpr::string("nodes"), LogicalExpr::string("edges")],
        )
        .with_yields(vec![YieldColumn::new("node"), YieldColumn::with_alias("score", "rank")]);

        assert_eq!(node.procedure_name, "algo.pageRank");
        assert_eq!(node.arguments.len(), 2);
        assert_eq!(node.yield_columns.len(), 2);
        assert_eq!(node.output_columns(), vec!["node", "rank"]);
    }

    #[test]
    fn yield_all() {
        let node = ProcedureCallNode::new("db.labels", vec![]);

        assert!(node.is_yield_all());
        assert!(node.output_columns().is_empty());
    }
}
