//! Scan operators for reading data from tables.
//!
//! This module provides operators for:
//! - Full table scans
//! - Index point lookups
//! - Index range scans

use std::sync::Arc;

use manifoldb_core::Value;

use crate::exec::context::ExecutionContext;
use crate::exec::operator::{Operator, OperatorBase, OperatorResult, OperatorState};
use crate::exec::row::{Row, Schema};
use crate::plan::physical::{FullScanNode, IndexRangeScanNode, IndexScanNode};

/// Full table scan operator.
///
/// Reads all rows from a table, optionally filtering them.
pub struct FullScanOp {
    /// Base operator state.
    base: OperatorBase,
    /// The node configuration.
    node: FullScanNode,
    /// Current row index (for simulation).
    current_row: usize,
    /// Simulated data (in a real implementation, this would read from storage).
    data: Vec<Vec<Value>>,
}

impl FullScanOp {
    /// Creates a new full scan operator.
    #[must_use]
    pub fn new(node: FullScanNode) -> Self {
        // Build schema from projection or use default columns
        let columns =
            node.projection.clone().unwrap_or_else(|| vec!["id".to_string(), "data".to_string()]);
        let schema = Arc::new(Schema::new(columns));

        Self { base: OperatorBase::new(schema), node, current_row: 0, data: Vec::new() }
    }

    /// Sets simulated data for testing.
    pub fn with_data(mut self, data: Vec<Vec<Value>>) -> Self {
        self.data = data;
        self
    }

    /// Returns the table name being scanned.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.node.table_name
    }
}

impl Operator for FullScanOp {
    fn open(&mut self, ctx: &ExecutionContext) -> OperatorResult<()> {
        self.current_row = 0;

        // If we have no data yet, try to load data from graph storage
        // This handles MATCH (n:Label) patterns where the scan table name is a label
        if self.data.is_empty() {
            // Try to use the graph accessor to scan nodes by label
            let graph = ctx.graph();
            match graph.scan_nodes(Some(&self.node.table_name)) {
                Ok(nodes) => {
                    // Get the alias from the node configuration, or use table name
                    let alias = self.node.alias.as_deref().unwrap_or(&self.node.table_name);

                    // Build rows from the scanned nodes
                    // For MATCH patterns, we typically need the entity ID under the alias
                    self.data = nodes
                        .into_iter()
                        .map(|node| vec![Value::Int(node.id.as_u64() as i64)])
                        .collect();

                    // Update schema to use the alias
                    self.base = OperatorBase::new(Arc::new(Schema::new(vec![alias.to_string()])));
                }
                Err(_) => {
                    // If scan_nodes returns an error (e.g., NoStorage), we keep the empty data
                    // This is expected for contexts without graph storage
                }
            }
        }

        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.current_row >= self.data.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let values = self.data[self.current_row].clone();
        self.current_row += 1;
        self.base.inc_rows_produced();

        let row = Row::new(self.base.schema(), values);
        Ok(Some(row))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "FullScan"
    }
}

/// Index scan operator for point lookups.
///
/// Uses an index to look up rows matching exact key values.
pub struct IndexScanOp {
    /// Base operator state.
    base: OperatorBase,
    /// The node configuration.
    node: IndexScanNode,
    /// Whether we've returned the result.
    returned: bool,
    /// Simulated data.
    data: Vec<Vec<Value>>,
}

impl IndexScanOp {
    /// Creates a new index scan operator.
    #[must_use]
    pub fn new(node: IndexScanNode) -> Self {
        let columns = node.projection.clone().unwrap_or_else(|| node.key_columns.clone());
        let schema = Arc::new(Schema::new(columns));

        Self { base: OperatorBase::new(schema), node, returned: false, data: Vec::new() }
    }

    /// Sets simulated data for testing.
    pub fn with_data(mut self, data: Vec<Vec<Value>>) -> Self {
        self.data = data;
        self
    }

    /// Returns the index name being scanned.
    #[must_use]
    pub fn index_name(&self) -> &str {
        &self.node.index_name
    }

    /// Returns the table name being scanned.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.node.table_name
    }

    /// Returns the key columns for this index scan.
    #[must_use]
    pub fn key_columns(&self) -> &[String] {
        &self.node.key_columns
    }
}

impl Operator for IndexScanOp {
    fn open(&mut self, _ctx: &ExecutionContext) -> OperatorResult<()> {
        self.returned = false;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.returned || self.data.is_empty() {
            self.base.set_finished();
            return Ok(None);
        }

        // In a real implementation, we would look up by key
        // For now, just return the first matching row
        self.returned = true;
        self.base.inc_rows_produced();

        let values = self.data[0].clone();
        let row = Row::new(self.base.schema(), values);
        Ok(Some(row))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "IndexScan"
    }
}

/// Index range scan operator.
///
/// Scans an index within a key range.
pub struct IndexRangeScanOp {
    /// Base operator state.
    base: OperatorBase,
    /// The node configuration.
    node: IndexRangeScanNode,
    /// Current row index.
    current_row: usize,
    /// Simulated data within range.
    data: Vec<Vec<Value>>,
}

impl IndexRangeScanOp {
    /// Creates a new index range scan operator.
    #[must_use]
    pub fn new(node: IndexRangeScanNode) -> Self {
        let columns = node.projection.clone().unwrap_or_else(|| vec![node.key_column.clone()]);
        let schema = Arc::new(Schema::new(columns));

        Self { base: OperatorBase::new(schema), node, current_row: 0, data: Vec::new() }
    }

    /// Sets simulated data for testing.
    pub fn with_data(mut self, data: Vec<Vec<Value>>) -> Self {
        self.data = data;
        self
    }

    /// Returns the index name being scanned.
    #[must_use]
    pub fn index_name(&self) -> &str {
        &self.node.index_name
    }

    /// Returns the table name being scanned.
    #[must_use]
    pub fn table_name(&self) -> &str {
        &self.node.table_name
    }

    /// Returns the key column being scanned.
    #[must_use]
    pub fn key_column(&self) -> &str {
        &self.node.key_column
    }
}

impl Operator for IndexRangeScanOp {
    fn open(&mut self, _ctx: &ExecutionContext) -> OperatorResult<()> {
        self.current_row = 0;
        self.base.set_open();
        Ok(())
    }

    fn next(&mut self) -> OperatorResult<Option<Row>> {
        if self.current_row >= self.data.len() {
            self.base.set_finished();
            return Ok(None);
        }

        let values = self.data[self.current_row].clone();
        self.current_row += 1;
        self.base.inc_rows_produced();

        let row = Row::new(self.base.schema(), values);
        Ok(Some(row))
    }

    fn close(&mut self) -> OperatorResult<()> {
        self.base.set_closed();
        Ok(())
    }

    fn schema(&self) -> Arc<Schema> {
        self.base.schema()
    }

    fn state(&self) -> OperatorState {
        self.base.state()
    }

    fn name(&self) -> &'static str {
        "IndexRangeScan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_scan_basic() {
        let node =
            FullScanNode::new("users").with_projection(vec!["id".to_string(), "name".to_string()]);

        let mut op = FullScanOp::new(node).with_data(vec![
            vec![Value::Int(1), Value::from("Alice")],
            vec![Value::Int(2), Value::from("Bob")],
        ]);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert_eq!(op.state(), OperatorState::Open);

        let row1 = op.next().unwrap().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));

        let row2 = op.next().unwrap().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(2)));

        let row3 = op.next().unwrap();
        assert!(row3.is_none());
        assert_eq!(op.state(), OperatorState::Finished);

        op.close().unwrap();
        assert_eq!(op.state(), OperatorState::Closed);
    }

    #[test]
    fn index_scan_basic() {
        let node = IndexScanNode::new(
            "users",
            "users_pk",
            vec!["id".to_string()],
            vec![crate::plan::logical::LogicalExpr::integer(1)],
        );

        let mut op = IndexScanOp::new(node).with_data(vec![vec![Value::Int(1)]]);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        let row = op.next().unwrap().unwrap();
        assert_eq!(row.get(0), Some(&Value::Int(1)));

        let row2 = op.next().unwrap();
        assert!(row2.is_none());

        op.close().unwrap();
    }

    #[test]
    fn full_scan_empty() {
        let node = FullScanNode::new("empty_table");
        let mut op = FullScanOp::new(node);

        let ctx = ExecutionContext::new();
        op.open(&ctx).unwrap();

        assert!(op.next().unwrap().is_none());
        op.close().unwrap();
    }
}
