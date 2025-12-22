//! Row types for query execution.
//!
//! This module defines the [`Row`] type used as the unit of data
//! flowing through the execution operators.

use std::collections::HashMap;
use std::sync::Arc;

use manifoldb_core::Value;

/// A schema defines the column names and their order in a row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Column names in order (using Arc<str> to avoid cloning).
    columns: Vec<Arc<str>>,
    /// Map from column name to index for fast lookup.
    name_to_index: HashMap<Arc<str>, usize>,
}

impl Schema {
    /// Creates a new schema from column names.
    #[must_use]
    pub fn new(columns: Vec<String>) -> Self {
        let arc_columns: Vec<Arc<str>> =
            columns.into_iter().map(|s| Arc::from(s.as_str())).collect();
        let name_to_index =
            arc_columns.iter().enumerate().map(|(i, name)| (Arc::clone(name), i)).collect();
        Self { columns: arc_columns, name_to_index }
    }

    /// Creates a new schema from Arc<str> column names (avoids allocation).
    #[must_use]
    pub fn from_arcs(columns: Vec<Arc<str>>) -> Self {
        let name_to_index =
            columns.iter().enumerate().map(|(i, name)| (Arc::clone(name), i)).collect();
        Self { columns, name_to_index }
    }

    /// Creates an empty schema.
    #[must_use]
    pub fn empty() -> Self {
        Self { columns: Vec::new(), name_to_index: HashMap::new() }
    }

    /// Returns the column names as string slices.
    #[must_use]
    pub fn columns(&self) -> Vec<&str> {
        self.columns.iter().map(|s| s.as_ref()).collect()
    }

    /// Returns the Arc<str> column names (for efficient cloning).
    #[must_use]
    pub fn columns_arc(&self) -> &[Arc<str>] {
        &self.columns
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Returns true if the schema has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Gets the index for a column name.
    #[must_use]
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_index.get(name).copied()
    }

    /// Gets the column name at an index.
    #[must_use]
    pub fn column_at(&self, index: usize) -> Option<&str> {
        self.columns.get(index).map(|s| s.as_ref())
    }

    /// Creates a new schema with an additional column.
    #[must_use]
    pub fn with_column(&self, name: impl Into<String>) -> Self {
        let mut columns: Vec<Arc<str>> = self.columns.iter().map(Arc::clone).collect();
        columns.push(Arc::from(name.into().as_str()));
        Self::from_arcs(columns)
    }

    /// Creates a new schema by merging with another (efficiently clones Arc<str>).
    #[must_use]
    pub fn merge(&self, other: &Schema) -> Self {
        let mut columns: Vec<Arc<str>> = self.columns.iter().map(Arc::clone).collect();
        columns.extend(other.columns.iter().map(Arc::clone));
        Self::from_arcs(columns)
    }

    /// Creates a projection of this schema with only the given columns.
    #[must_use]
    pub fn project(&self, indices: &[usize]) -> Self {
        let columns: Vec<Arc<str>> =
            indices.iter().filter_map(|&i| self.columns.get(i).map(Arc::clone)).collect();
        Self::from_arcs(columns)
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Vec<String>> for Schema {
    fn from(columns: Vec<String>) -> Self {
        Self::new(columns)
    }
}

impl From<Vec<&str>> for Schema {
    fn from(columns: Vec<&str>) -> Self {
        Self::new(columns.into_iter().map(String::from).collect())
    }
}

/// A row of values.
///
/// Rows are the unit of data flowing through execution operators.
/// Each row contains values that correspond to the schema columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The schema describing the columns.
    schema: Arc<Schema>,
    /// The values in this row.
    values: Vec<Value>,
}

impl Row {
    /// Creates a new row with the given schema and values.
    ///
    /// # Panics
    ///
    /// Panics if the number of values doesn't match the schema.
    #[must_use]
    pub fn new(schema: Arc<Schema>, values: Vec<Value>) -> Self {
        debug_assert_eq!(
            schema.len(),
            values.len(),
            "Row values count must match schema column count"
        );
        Self { schema, values }
    }

    /// Creates a row with a single value.
    #[must_use]
    pub fn single(name: impl Into<String>, value: Value) -> Self {
        let schema = Arc::new(Schema::new(vec![name.into()]));
        Self { schema, values: vec![value] }
    }

    /// Creates an empty row with the given schema.
    #[must_use]
    pub fn empty(schema: Arc<Schema>) -> Self {
        let values = vec![Value::Null; schema.len()];
        Self { schema, values }
    }

    /// Returns the schema of this row.
    #[must_use]
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Returns the shared schema reference.
    #[must_use]
    pub fn schema_arc(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Returns the values in this row.
    #[must_use]
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the row has no columns.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Gets a value by column index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Value> {
        self.values.get(index)
    }

    /// Gets a value by column name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&Value> {
        self.schema.index_of(name).and_then(|i| self.values.get(i))
    }

    /// Gets a mutable value by column index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.values.get_mut(index)
    }

    /// Sets a value by column index.
    ///
    /// Returns the old value if the index was valid.
    pub fn set(&mut self, index: usize, value: Value) -> Option<Value> {
        if index < self.values.len() {
            Some(std::mem::replace(&mut self.values[index], value))
        } else {
            None
        }
    }

    /// Creates a new row by projecting to specific column indices.
    #[must_use]
    pub fn project(&self, indices: &[usize]) -> Self {
        let schema = Arc::new(self.schema.project(indices));
        let values: Vec<Value> =
            indices.iter().filter_map(|&i| self.values.get(i).cloned()).collect();
        Self { schema, values }
    }

    /// Creates a new row by merging with another row.
    #[must_use]
    pub fn merge(&self, other: &Row) -> Self {
        let schema = Arc::new(self.schema.merge(&other.schema));
        let mut values = self.values.clone();
        values.extend(other.values.iter().cloned());
        Self { schema, values }
    }

    /// Consumes self and merges with another row's values (borrowed).
    /// More efficient than `merge` when the left row can be consumed.
    #[must_use]
    pub fn merge_consume_left(mut self, other: &Row) -> Self {
        let schema = Arc::new(self.schema.merge(&other.schema));
        self.values.extend(other.values.iter().cloned());
        Self { schema, values: self.values }
    }

    /// Consumes both rows and merges them.
    /// Most efficient merge operation when both rows can be consumed.
    #[must_use]
    pub fn merge_consume_both(mut self, mut other: Row) -> Self {
        let schema = Arc::new(self.schema.merge(&other.schema));
        self.values.append(&mut other.values);
        Self { schema, values: self.values }
    }

    /// Consumes the row and returns the values.
    #[must_use]
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }

    /// Converts the row to a map of column names to values.
    #[must_use]
    pub fn to_map(&self) -> HashMap<String, Value> {
        self.schema
            .columns_arc()
            .iter()
            .zip(self.values.iter())
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    }
}

impl IntoIterator for Row {
    type Item = (Arc<str>, Value);
    type IntoIter = std::iter::Zip<std::vec::IntoIter<Arc<str>>, std::vec::IntoIter<Value>>;

    fn into_iter(self) -> Self::IntoIter {
        self.schema.columns.iter().map(Arc::clone).collect::<Vec<_>>().into_iter().zip(self.values)
    }
}

/// A batch of rows for efficient processing.
///
/// Row batches allow vectorized operations on multiple rows at once.
#[derive(Debug, Clone)]
pub struct RowBatch {
    /// The schema shared by all rows.
    schema: Arc<Schema>,
    /// The rows in this batch.
    rows: Vec<Row>,
}

impl RowBatch {
    /// Creates a new row batch.
    #[must_use]
    pub fn new(schema: Arc<Schema>) -> Self {
        Self { schema, rows: Vec::new() }
    }

    /// Creates a row batch with the given rows.
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

    /// Returns the rows in this batch.
    #[must_use]
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns true if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Adds a row to the batch.
    pub fn push(&mut self, row: Row) {
        debug_assert_eq!(
            row.schema().columns(),
            self.schema.columns(),
            "Row schema must match batch schema"
        );
        self.rows.push(row);
    }

    /// Removes and returns the last row.
    pub fn pop(&mut self) -> Option<Row> {
        self.rows.pop()
    }

    /// Clears the batch.
    pub fn clear(&mut self) {
        self.rows.clear();
    }

    /// Consumes the batch and returns the rows.
    #[must_use]
    pub fn into_rows(self) -> Vec<Row> {
        self.rows
    }
}

impl IntoIterator for RowBatch {
    type Item = Row;
    type IntoIter = std::vec::IntoIter<Row>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.into_iter()
    }
}

impl<'a> IntoIterator for &'a RowBatch {
    type Item = &'a Row;
    type IntoIter = std::slice::Iter<'a, Row>;

    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_basic() {
        let schema = Schema::new(vec!["id".to_string(), "name".to_string()]);
        assert_eq!(schema.len(), 2);
        assert_eq!(schema.index_of("id"), Some(0));
        assert_eq!(schema.index_of("name"), Some(1));
        assert_eq!(schema.index_of("unknown"), None);
    }

    #[test]
    fn schema_merge() {
        let s1 = Schema::new(vec!["a".to_string()]);
        let s2 = Schema::new(vec!["b".to_string()]);
        let merged = s1.merge(&s2);
        assert_eq!(merged.columns(), &["a", "b"]);
    }

    #[test]
    fn row_basic() {
        let schema = Arc::new(Schema::new(vec!["id".to_string(), "name".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1), Value::from("Alice")]);

        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::Int(1)));
        assert_eq!(row.get_by_name("name"), Some(&Value::from("Alice")));
    }

    #[test]
    fn row_project() {
        let schema = Arc::new(Schema::new(vec!["a".to_string(), "b".to_string(), "c".to_string()]));
        let row = Row::new(Arc::clone(&schema), vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

        let projected = row.project(&[0, 2]);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected.schema().columns(), &["a", "c"]);
        assert_eq!(projected.get(0), Some(&Value::Int(1)));
        assert_eq!(projected.get(1), Some(&Value::Int(3)));
    }

    #[test]
    fn row_merge() {
        let s1 = Arc::new(Schema::new(vec!["a".to_string()]));
        let s2 = Arc::new(Schema::new(vec!["b".to_string()]));
        let r1 = Row::new(s1, vec![Value::Int(1)]);
        let r2 = Row::new(s2, vec![Value::Int(2)]);

        let merged = r1.merge(&r2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.get(0), Some(&Value::Int(1)));
        assert_eq!(merged.get(1), Some(&Value::Int(2)));
    }

    #[test]
    fn row_batch_basic() {
        let schema = Arc::new(Schema::new(vec!["id".to_string()]));
        let mut batch = RowBatch::new(Arc::clone(&schema));

        batch.push(Row::new(Arc::clone(&schema), vec![Value::Int(1)]));
        batch.push(Row::new(Arc::clone(&schema), vec![Value::Int(2)]));

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.rows()[0].get(0), Some(&Value::Int(1)));
    }
}
