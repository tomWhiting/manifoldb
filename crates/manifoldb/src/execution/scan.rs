//! Storage-aware scan that reads entities from the database.

use std::sync::Arc;

use manifoldb_core::{CollectionId, Entity, Value};
use manifoldb_query::exec::row::{Row, Schema};
use manifoldb_query::exec::CollectionVectorProvider;

/// Collection context for scanning entities with named vectors.
pub struct CollectionContext {
    /// The collection ID.
    pub collection_id: CollectionId,
    /// Named vectors defined in the collection.
    pub vector_names: Vec<String>,
    /// Provider for fetching vectors.
    pub provider: Arc<dyn CollectionVectorProvider>,
}

/// A scan result that contains entities loaded from storage.
pub struct StorageScan {
    /// The entities to scan.
    entities: Vec<Entity>,
    /// Column names for the result schema.
    columns: Vec<String>,
    /// The schema for result rows.
    schema: Arc<Schema>,
    /// Current position in the scan.
    position: usize,
    /// Optional collection context for vector lookup.
    collection_context: Option<CollectionContext>,
}

impl StorageScan {
    /// Create a new storage scan from entities.
    ///
    /// The `table_name` is the label used to scan entities.
    /// Columns are derived from the projection or all properties if none specified.
    #[must_use]
    pub fn new(entities: Vec<Entity>, columns: Vec<String>) -> Self {
        let schema = Arc::new(Schema::new(columns.clone()));
        Self { entities, columns, schema, position: 0, collection_context: None }
    }

    /// Set the collection context for looking up named vectors.
    #[must_use]
    pub fn with_collection_context(mut self, context: CollectionContext) -> Self {
        self.collection_context = Some(context);
        self
    }

    /// Returns the schema for this scan.
    #[must_use]
    pub fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }

    /// Returns the column names.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Returns the number of entities in the scan.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Returns true if the scan is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Converts an entity to a row based on the column projection.
    fn entity_to_row(&self, entity: &Entity) -> Row {
        let values: Vec<Value> = self
            .columns
            .iter()
            .map(|col| {
                // Handle special columns
                if col == "_rowid" {
                    Value::Int(entity.id.as_u64() as i64)
                } else if col == "_labels" {
                    // Return labels as a string (comma-separated)
                    Value::String(
                        entity.labels.iter().map(|l| l.as_str()).collect::<Vec<_>>().join(","),
                    )
                } else if let Some(ref ctx) = self.collection_context {
                    // Check if this column is a named vector
                    if ctx.vector_names.contains(col) {
                        // Fetch vector from collection storage
                        match ctx.provider.get_vector(ctx.collection_id, entity.id, col) {
                            Ok(Some(data)) => vector_data_to_value(&data),
                            Ok(None) => Value::Null,
                            Err(_) => Value::Null,
                        }
                    } else {
                        // Regular property
                        entity.get_property(col).cloned().unwrap_or(Value::Null)
                    }
                } else {
                    // Regular property
                    entity.get_property(col).cloned().unwrap_or(Value::Null)
                }
            })
            .collect();

        Row::new(Arc::clone(&self.schema), values)
    }

    /// Get the next row from the scan.
    pub fn next_row(&mut self) -> Option<Row> {
        if self.position >= self.entities.len() {
            return None;
        }

        let row = self.entity_to_row(&self.entities[self.position]);
        self.position += 1;
        Some(row)
    }

    /// Collect all rows from the scan.
    #[must_use]
    pub fn collect_rows(&self) -> Vec<Row> {
        self.entities.iter().map(|e| self.entity_to_row(e)).collect()
    }

    /// Collect all values from the scan.
    #[must_use]
    pub fn collect_values(&self) -> Vec<Vec<Value>> {
        self.collect_rows().into_iter().map(|r| r.values().to_vec()).collect()
    }
}

/// Convert VectorData to Value.
fn vector_data_to_value(data: &manifoldb_vector::types::VectorData) -> Value {
    use manifoldb_vector::types::VectorData;

    match data {
        VectorData::Dense(v) => Value::Vector(v.clone()),
        VectorData::Sparse(v) => Value::SparseVector(v.clone()),
        VectorData::Multi(v) => Value::MultiVector(v.clone()),
        VectorData::Binary(v) => Value::Bytes(v.clone()), // Binary vectors stored as raw bytes
    }
}

impl Iterator for StorageScan {
    type Item = Row;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_row()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_core::EntityId;

    #[test]
    fn test_storage_scan_basic() {
        let entities = vec![
            Entity::new(EntityId::new(1))
                .with_label("Person")
                .with_property("name", "Alice")
                .with_property("age", 30i64),
            Entity::new(EntityId::new(2))
                .with_label("Person")
                .with_property("name", "Bob")
                .with_property("age", 25i64),
        ];

        let columns = vec!["_rowid".to_string(), "name".to_string(), "age".to_string()];
        let mut scan = StorageScan::new(entities, columns);

        assert_eq!(scan.len(), 2);

        let row1 = scan.next_row().unwrap();
        assert_eq!(row1.get(0), Some(&Value::Int(1)));
        assert_eq!(row1.get(1), Some(&Value::String("Alice".to_string())));
        assert_eq!(row1.get(2), Some(&Value::Int(30)));

        let row2 = scan.next_row().unwrap();
        assert_eq!(row2.get(0), Some(&Value::Int(2)));
        assert_eq!(row2.get(1), Some(&Value::String("Bob".to_string())));

        assert!(scan.next_row().is_none());
    }

    #[test]
    fn test_storage_scan_missing_property() {
        let entities =
            vec![Entity::new(EntityId::new(1)).with_label("Person").with_property("name", "Alice")];

        let columns = vec!["name".to_string(), "missing".to_string()];
        let mut scan = StorageScan::new(entities, columns);

        let row = scan.next_row().unwrap();
        assert_eq!(row.get(0), Some(&Value::String("Alice".to_string())));
        assert_eq!(row.get(1), Some(&Value::Null));
    }
}
