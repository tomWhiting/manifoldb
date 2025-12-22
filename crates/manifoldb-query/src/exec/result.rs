//! Query result types.
//!
//! This module defines the types used to represent query results.

use std::sync::Arc;

use manifoldb_core::Value;

use super::row::{Row, Schema};

/// The result of a query execution.
#[derive(Debug)]
pub enum QueryResult {
    /// A result set with rows.
    Select(ResultSet),
    /// The number of rows affected by an INSERT/UPDATE/DELETE.
    Affected(u64),
    /// An empty result (e.g., from DDL statements).
    Empty,
}

impl QueryResult {
    /// Creates a new select result.
    #[must_use]
    pub fn select(result_set: ResultSet) -> Self {
        Self::Select(result_set)
    }

    /// Creates an affected rows result.
    #[must_use]
    pub const fn affected(count: u64) -> Self {
        Self::Affected(count)
    }

    /// Creates an empty result.
    #[must_use]
    pub const fn empty() -> Self {
        Self::Empty
    }

    /// Returns true if this is a select result.
    #[must_use]
    pub const fn is_select(&self) -> bool {
        matches!(self, Self::Select(_))
    }

    /// Returns the result set if this is a select result.
    #[must_use]
    pub fn as_select(&self) -> Option<&ResultSet> {
        match self {
            Self::Select(rs) => Some(rs),
            _ => None,
        }
    }

    /// Consumes and returns the result set if this is a select result.
    #[must_use]
    pub fn into_select(self) -> Option<ResultSet> {
        match self {
            Self::Select(rs) => Some(rs),
            _ => None,
        }
    }

    /// Returns the affected row count if this is an affected result.
    #[must_use]
    pub const fn affected_rows(&self) -> Option<u64> {
        match self {
            Self::Affected(n) => Some(*n),
            _ => None,
        }
    }
}

/// A set of result rows from a SELECT query.
#[derive(Debug, Clone)]
pub struct ResultSet {
    /// The schema of the result set.
    schema: Arc<Schema>,
    /// The rows in the result set.
    rows: Vec<Row>,
}

impl ResultSet {
    /// Creates a new result set.
    #[must_use]
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema, rows: Vec::new() }
    }

    /// Creates a result set with the given rows.
    #[must_use]
    pub fn with_rows(schema: Arc<Schema>, rows: Vec<Row>) -> Self {
        Self { schema, rows }
    }

    /// Returns the schema.
    #[must_use]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Returns the shared schema reference.
    #[must_use]
    pub fn schema_arc(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Returns the column names.
    #[must_use]
    pub fn columns(&self) -> Vec<&str> {
        self.schema.columns()
    }

    /// Returns the rows.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns true if there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Adds a row to the result set.
    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// Gets a row by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Row> {
        self.rows.get(index)
    }

    /// Consumes the result set and returns the rows.
    #[must_use]
    pub fn into_rows(self) -> Vec<Row> {
        self.rows
    }

    /// Returns an iterator over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &Row> {
        self.rows.iter()
    }

    /// Converts to a vector of value arrays.
    #[must_use]
    pub fn to_values(&self) -> Vec<Vec<Value>> {
        self.rows.iter().map(|r| r.values().to_vec()).collect()
    }
}

impl IntoIterator for ResultSet {
    type Item = Row;
    type IntoIter = std::vec::IntoIter<Row>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.into_iter()
    }
}

impl<'a> IntoIterator for &'a ResultSet {
    type Item = &'a Row;
    type IntoIter = std::slice::Iter<'a, Row>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

/// Builder for constructing result sets incrementally.
///
/// Provides a convenient way to build a `ResultSet` row by row.
pub struct ResultSetBuilder {
    schema: Arc<Schema>,
    rows: Vec<Row>,
}

impl ResultSetBuilder {
    /// Creates a new builder with the given schema.
    #[must_use]
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema, rows: Vec::new() }
    }

    /// Creates a builder with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(schema: Arc<Schema>, capacity: usize) -> Self {
        Self { schema, rows: Vec::with_capacity(capacity) }
    }

    /// Adds a row to the result set.
    pub fn push(&mut self, row: Row) {
        self.rows.push(row);
    }

    /// Adds a row from values.
    pub fn push_values(&mut self, values: Vec<Value>) {
        let row = Row::new(Arc::clone(&self.schema), values);
        self.rows.push(row);
    }

    /// Returns the current number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns true if no rows have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Builds the result set.
    #[must_use]
    pub fn build(self) -> ResultSet {
        ResultSet::with_rows(self.schema, self.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_result_types() {
        let result = QueryResult::affected(5);
        assert!(!result.is_select());
        assert_eq!(result.affected_rows(), Some(5));

        let schema = Arc::new(Schema::new(vec!["id".to_string()]));
        let result = QueryResult::select(ResultSet::new(schema));
        assert!(result.is_select());
        assert_eq!(result.affected_rows(), None);
    }

    #[test]
    fn result_set_basic() {
        let schema = Arc::new(Schema::new(vec!["id".to_string(), "name".to_string()]));
        let mut rs = ResultSet::new(Arc::clone(&schema));

        rs.push(Row::new(Arc::clone(&schema), vec![Value::Int(1), Value::from("Alice")]));
        rs.push(Row::new(Arc::clone(&schema), vec![Value::Int(2), Value::from("Bob")]));

        assert_eq!(rs.len(), 2);
        assert_eq!(rs.columns(), &["id", "name"]);
        assert_eq!(rs.get(0).and_then(|r| r.get(0)), Some(&Value::Int(1)));
    }

    #[test]
    fn result_set_builder() {
        let schema = Arc::new(Schema::new(vec!["x".to_string()]));
        let mut builder = ResultSetBuilder::new(Arc::clone(&schema));

        builder.push_values(vec![Value::Int(1)]);
        builder.push_values(vec![Value::Int(2)]);

        let rs = builder.build();
        assert_eq!(rs.len(), 2);
    }

    #[test]
    fn result_set_iterator() {
        let schema = Arc::new(Schema::new(vec!["n".to_string()]));
        let rs = ResultSet::with_rows(
            Arc::clone(&schema),
            vec![
                Row::new(Arc::clone(&schema), vec![Value::Int(1)]),
                Row::new(Arc::clone(&schema), vec![Value::Int(2)]),
            ],
        );

        let sum: i64 = rs
            .iter()
            .filter_map(|r| match r.get(0) {
                Some(Value::Int(n)) => Some(*n),
                _ => None,
            })
            .sum();

        assert_eq!(sum, 3);
    }
}
