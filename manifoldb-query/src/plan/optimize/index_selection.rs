//! Index selection optimization.
//!
//! Analyzes predicates and available indexes to choose optimal
//! access methods for table scans.

// Allow unwrap - partial_cmp on f64 values that are guaranteed to be valid
#![allow(clippy::unwrap_used)]

use crate::ast::BinaryOp;
use crate::plan::logical::LogicalExpr;

/// Index selector.
///
/// Analyzes predicates and matches them against available indexes
/// to determine optimal access methods.
#[derive(Debug, Clone, Default)]
pub struct IndexSelector {}

impl IndexSelector {
    /// Creates a new index selector.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }

    /// Analyzes a predicate to find index candidates.
    ///
    /// Returns a list of column-value pairs that could use an index.
    #[must_use]
    pub fn find_index_candidates(&self, predicate: &LogicalExpr) -> Vec<IndexCandidate> {
        let mut candidates = Vec::new();
        self.analyze_predicate(predicate, &mut candidates);
        candidates
    }

    /// Recursively analyzes a predicate for index opportunities.
    fn analyze_predicate(&self, expr: &LogicalExpr, candidates: &mut Vec<IndexCandidate>) {
        match expr {
            // AND: analyze both sides
            LogicalExpr::BinaryOp { left, op: BinaryOp::And, right } => {
                self.analyze_predicate(left, candidates);
                self.analyze_predicate(right, candidates);
            }

            // OR: can only use index if all branches use same column
            LogicalExpr::BinaryOp { left, op: BinaryOp::Or, right } => {
                let mut left_candidates = Vec::new();
                let mut right_candidates = Vec::new();

                self.analyze_predicate(left, &mut left_candidates);
                self.analyze_predicate(right, &mut right_candidates);

                // Check if both sides reference the same column
                for lc in &left_candidates {
                    for rc in &right_candidates {
                        if lc.column == rc.column && lc.table == rc.table {
                            // Can potentially use IN-list or range merge
                            candidates.push(IndexCandidate {
                                table: lc.table.clone(),
                                column: lc.column.clone(),
                                access_type: AccessType::InList,
                                selectivity: (lc.selectivity + rc.selectivity).min(1.0),
                            });
                        }
                    }
                }
            }

            // Equality: column = value
            LogicalExpr::BinaryOp { left, op: BinaryOp::Eq, right } => {
                if let Some(candidate) = self.check_equality(left, right) {
                    candidates.push(candidate);
                }
            }

            // Comparison: column > value, column < value, etc.
            LogicalExpr::BinaryOp {
                left,
                op: op @ (BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq),
                right,
            } => {
                if let Some(candidate) = self.check_comparison(left, right, op) {
                    candidates.push(candidate);
                }
            }

            // BETWEEN: column BETWEEN low AND high
            LogicalExpr::Between { expr, low, high, negated } => {
                if !negated {
                    if let Some(candidate) = self.check_between(expr, low, high) {
                        candidates.push(candidate);
                    }
                }
            }

            // IN list: column IN (v1, v2, v3)
            LogicalExpr::InList { expr, list, negated } => {
                if !negated {
                    if let Some(candidate) = self.check_in_list(expr, list) {
                        candidates.push(candidate);
                    }
                }
            }

            // LIKE with prefix: column LIKE 'abc%'
            LogicalExpr::BinaryOp { left, op: BinaryOp::Like, right } => {
                if let Some(candidate) = self.check_like_prefix(left, right) {
                    candidates.push(candidate);
                }
            }

            _ => {}
        }
    }

    /// Checks for column = value pattern.
    fn check_equality(&self, left: &LogicalExpr, right: &LogicalExpr) -> Option<IndexCandidate> {
        // column = literal
        if let LogicalExpr::Column { qualifier, name } = left {
            if self.is_literal(right) {
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: AccessType::PointLookup,
                    selectivity: 0.01, // Assume high selectivity for equality
                });
            }
        }

        // literal = column (swap)
        if let LogicalExpr::Column { qualifier, name } = right {
            if self.is_literal(left) {
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: AccessType::PointLookup,
                    selectivity: 0.01,
                });
            }
        }

        None
    }

    /// Checks for comparison patterns.
    fn check_comparison(
        &self,
        left: &LogicalExpr,
        right: &LogicalExpr,
        op: &BinaryOp,
    ) -> Option<IndexCandidate> {
        // column <op> literal
        if let LogicalExpr::Column { qualifier, name } = left {
            if self.is_literal(right) {
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: self.op_to_access_type(op),
                    selectivity: self.estimate_range_selectivity(op),
                });
            }
        }

        // literal <op> column (need to swap the operator)
        if let LogicalExpr::Column { qualifier, name } = right {
            if self.is_literal(left) {
                let swapped_op = self.swap_comparison_op(op);
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: self.op_to_access_type(&swapped_op),
                    selectivity: self.estimate_range_selectivity(op),
                });
            }
        }

        None
    }

    /// Checks for BETWEEN pattern.
    fn check_between(
        &self,
        expr: &LogicalExpr,
        low: &LogicalExpr,
        high: &LogicalExpr,
    ) -> Option<IndexCandidate> {
        if let LogicalExpr::Column { qualifier, name } = expr {
            if self.is_literal(low) && self.is_literal(high) {
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: AccessType::RangeScan,
                    selectivity: 0.1, // Assume 10% selectivity for BETWEEN
                });
            }
        }
        None
    }

    /// Checks for IN list pattern.
    fn check_in_list(&self, expr: &LogicalExpr, list: &[LogicalExpr]) -> Option<IndexCandidate> {
        if let LogicalExpr::Column { qualifier, name } = expr {
            if list.iter().all(|e| self.is_literal(e)) {
                let selectivity = (list.len() as f64 * 0.01).min(0.5);
                return Some(IndexCandidate {
                    table: qualifier.clone(),
                    column: name.clone(),
                    access_type: AccessType::InList,
                    selectivity,
                });
            }
        }
        None
    }

    /// Checks for LIKE with prefix pattern.
    fn check_like_prefix(&self, left: &LogicalExpr, right: &LogicalExpr) -> Option<IndexCandidate> {
        if let LogicalExpr::Column { qualifier, name } = left {
            if let LogicalExpr::Literal(crate::ast::Literal::String(pattern)) = right {
                // Check if pattern has a constant prefix (no leading %)
                if !pattern.starts_with('%') && !pattern.starts_with('_') {
                    // Find where the prefix ends
                    let prefix_len = pattern.chars().take_while(|&c| c != '%' && c != '_').count();

                    if prefix_len > 0 {
                        return Some(IndexCandidate {
                            table: qualifier.clone(),
                            column: name.clone(),
                            access_type: AccessType::PrefixScan,
                            selectivity: 0.1,
                        });
                    }
                }
            }
        }
        None
    }

    /// Checks if an expression is a literal value.
    fn is_literal(&self, expr: &LogicalExpr) -> bool {
        matches!(expr, LogicalExpr::Literal(_) | LogicalExpr::Parameter(_))
    }

    /// Converts a comparison operator to access type.
    fn op_to_access_type(&self, op: &BinaryOp) -> AccessType {
        match op {
            BinaryOp::Lt | BinaryOp::LtEq => AccessType::RangeScanLt,
            BinaryOp::Gt | BinaryOp::GtEq => AccessType::RangeScanGt,
            _ => AccessType::RangeScan,
        }
    }

    /// Swaps a comparison operator for reversed operands.
    fn swap_comparison_op(&self, op: &BinaryOp) -> BinaryOp {
        match op {
            BinaryOp::Lt => BinaryOp::Gt,
            BinaryOp::LtEq => BinaryOp::GtEq,
            BinaryOp::Gt => BinaryOp::Lt,
            BinaryOp::GtEq => BinaryOp::LtEq,
            other => *other,
        }
    }

    /// Estimates selectivity for a range comparison.
    fn estimate_range_selectivity(&self, op: &BinaryOp) -> f64 {
        match op {
            BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => 0.3,
            _ => 0.5,
        }
    }

    /// Scores index candidates by selectivity.
    ///
    /// Lower score = better (fewer rows to read).
    #[must_use]
    pub fn score_candidates(&self, candidates: &[IndexCandidate]) -> Vec<(usize, f64)> {
        candidates.iter().enumerate().map(|(i, c)| (i, c.selectivity)).collect()
    }

    /// Selects the best index candidate for a column.
    #[must_use]
    pub fn best_candidate<'a>(
        &self,
        candidates: &'a [IndexCandidate],
        column: &str,
    ) -> Option<&'a IndexCandidate> {
        candidates
            .iter()
            .filter(|c| c.column == column)
            .min_by(|a, b| a.selectivity.partial_cmp(&b.selectivity).unwrap())
    }
}

/// An index candidate identified from a predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexCandidate {
    /// Table name (if qualified).
    pub table: Option<String>,
    /// Column name.
    pub column: String,
    /// Type of access this predicate enables.
    pub access_type: AccessType,
    /// Estimated selectivity (0.0 to 1.0).
    pub selectivity: f64,
}

/// Type of index access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Exact key lookup.
    PointLookup,
    /// Range scan (bounded on both ends).
    RangeScan,
    /// Range scan with upper bound only.
    RangeScanLt,
    /// Range scan with lower bound only.
    RangeScanGt,
    /// Multiple point lookups (IN list).
    InList,
    /// Prefix scan (for LIKE 'abc%').
    PrefixScan,
}

impl IndexCandidate {
    /// Creates a point lookup candidate.
    #[must_use]
    pub fn point_lookup(column: impl Into<String>) -> Self {
        Self {
            table: None,
            column: column.into(),
            access_type: AccessType::PointLookup,
            selectivity: 0.01,
        }
    }

    /// Creates a range scan candidate.
    #[must_use]
    pub fn range_scan(column: impl Into<String>, selectivity: f64) -> Self {
        Self { table: None, column: column.into(), access_type: AccessType::RangeScan, selectivity }
    }

    /// Sets the table qualifier.
    #[must_use]
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table = Some(table.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical::LogicalExpr;

    #[test]
    fn find_equality_candidate() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::column("id").eq(LogicalExpr::integer(42));

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].column, "id");
        assert_eq!(candidates[0].access_type, AccessType::PointLookup);
    }

    #[test]
    fn find_range_candidate() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::column("age").gt(LogicalExpr::integer(21));

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].column, "age");
        assert_eq!(candidates[0].access_type, AccessType::RangeScanGt);
    }

    #[test]
    fn find_between_candidate() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::column("price").between(
            LogicalExpr::integer(10),
            LogicalExpr::integer(100),
            false,
        );

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].column, "price");
        assert_eq!(candidates[0].access_type, AccessType::RangeScan);
    }

    #[test]
    fn find_in_list_candidate() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::column("status")
            .in_list(vec![LogicalExpr::string("active"), LogicalExpr::string("pending")], false);

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].column, "status");
        assert_eq!(candidates[0].access_type, AccessType::InList);
    }

    #[test]
    fn find_multiple_candidates_and() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::column("id")
            .eq(LogicalExpr::integer(42))
            .and(LogicalExpr::column("age").gt(LogicalExpr::integer(21)));

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn no_candidate_for_expression() {
        let selector = IndexSelector::new();
        // column + 1 = 10 cannot use index
        let pred =
            LogicalExpr::column("id").add(LogicalExpr::integer(1)).eq(LogicalExpr::integer(10));

        let candidates = selector.find_index_candidates(&pred);

        assert!(candidates.is_empty());
    }

    #[test]
    fn reversed_operands() {
        let selector = IndexSelector::new();
        // 42 = id (reversed)
        let pred = LogicalExpr::integer(42).eq(LogicalExpr::column("id"));

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].column, "id");
        assert_eq!(candidates[0].access_type, AccessType::PointLookup);
    }

    #[test]
    fn qualified_column() {
        let selector = IndexSelector::new();
        let pred = LogicalExpr::qualified_column("users", "id").eq(LogicalExpr::integer(42));

        let candidates = selector.find_index_candidates(&pred);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].table.as_deref(), Some("users"));
        assert_eq!(candidates[0].column, "id");
    }

    #[test]
    fn best_candidate_selection() {
        let selector = IndexSelector::new();
        let candidates = vec![
            IndexCandidate::point_lookup("id"),     // 0.01
            IndexCandidate::range_scan("age", 0.3), // 0.30
            IndexCandidate::range_scan("age", 0.1), // 0.10
        ];

        let best = selector.best_candidate(&candidates, "age");
        assert!(best.is_some());
        assert_eq!(best.unwrap().selectivity, 0.1);
    }
}
