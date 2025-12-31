//! Integration tests for payload indexing.
//!
//! These tests verify that payload indexes are correctly:
//! - Created and tracked in the index catalog
//! - Maintained during entity insert/update/delete
//! - Used for lookups

use manifoldb::{Database, Entity, EntityId, IndexType, Value};

/// Create a test database with some entities.
fn setup_test_db() -> Database {
    let db = Database::in_memory().expect("Failed to create in-memory database");

    // Insert some test entities
    let entities: Vec<Entity> = (0..100)
        .map(|i| {
            let language = match i % 4 {
                0 => "rust",
                1 => "python",
                2 => "javascript",
                _ => "go",
            };
            let visibility = if i % 2 == 0 { "public" } else { "private" };

            Entity::new(EntityId::new(0)) // ID assigned during insert
                .with_label("Symbol")
                .with_property("language", language)
                .with_property("visibility", visibility)
                .with_property("index", i as i64)
        })
        .collect();

    db.bulk_insert_entities(&entities).expect("Failed to insert entities");
    db
}

#[test]
fn test_create_index() {
    let db = setup_test_db();

    // Create an index
    db.create_index("Symbol", "language").expect("Failed to create index");

    // Verify index exists
    let indexes = db.list_indexes().expect("Failed to list indexes");
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].label, "Symbol");
    assert_eq!(indexes[0].property, "language");
    assert_eq!(indexes[0].index_type, IndexType::Equality);
    assert_eq!(indexes[0].entry_count, 100); // All 100 entities indexed
}

#[test]
fn test_create_index_with_type() {
    let db = setup_test_db();

    // Create a range index
    db.create_index_with_type("Symbol", "index", IndexType::Range)
        .expect("Failed to create range index");

    // Verify index type
    let indexes = db.list_indexes().expect("Failed to list indexes");
    assert_eq!(indexes.len(), 1);
    assert_eq!(indexes[0].index_type, IndexType::Range);
}

#[test]
fn test_create_duplicate_index_fails() {
    let db = setup_test_db();

    db.create_index("Symbol", "language").expect("Failed to create index");

    // Try to create the same index again - should fail
    let result = db.create_index("Symbol", "language");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn test_drop_index() {
    let db = setup_test_db();

    db.create_index("Symbol", "language").expect("Failed to create index");
    assert_eq!(db.list_indexes().unwrap().len(), 1);

    // Drop the index
    db.drop_index("Symbol", "language").expect("Failed to drop index");

    // Verify it's gone
    assert_eq!(db.list_indexes().unwrap().len(), 0);
}

#[test]
fn test_drop_nonexistent_index_fails() {
    let db = setup_test_db();

    let result = db.drop_index("Symbol", "nonexistent");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No index exists"));
}

#[test]
fn test_index_stats() {
    let db = setup_test_db();

    db.create_index("Symbol", "language").expect("Failed to create index");

    let stats = db.index_stats("Symbol", "language").expect("Failed to get stats");
    assert_eq!(stats.entry_count, 100);
    assert_eq!(stats.distinct_values, 4); // rust, python, javascript, go
    assert!((stats.selectivity - 0.25).abs() < 0.01); // 1/4 = 0.25
}

#[test]
fn test_index_lookup_eq() {
    let db = setup_test_db();

    db.create_index("Symbol", "language").expect("Failed to create index");

    // Lookup entities with language = "rust"
    let result = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup");

    assert!(result.is_some());
    let ids = result.unwrap();
    assert_eq!(ids.len(), 25); // 100/4 = 25 entities per language
}

#[test]
fn test_index_lookup_without_index() {
    let db = setup_test_db();

    // Don't create an index - lookup should return None
    let result = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup");

    assert!(result.is_none());
}

#[test]
fn test_index_metadata() {
    let db = setup_test_db();

    // No index yet
    let meta = db.get_index_metadata("Symbol", "language").expect("Failed to get metadata");
    assert!(meta.is_none());

    // Create index
    db.create_index("Symbol", "language").expect("Failed to create index");

    // Now metadata should exist
    let meta = db.get_index_metadata("Symbol", "language").expect("Failed to get metadata");
    assert!(meta.is_some());
    let meta = meta.unwrap();
    assert_eq!(meta.label, "Symbol");
    assert_eq!(meta.property, "language");
    assert!(meta.created_at > 0);
}

#[test]
fn test_index_maintenance_on_insert() {
    let db = Database::in_memory().expect("Failed to create database");

    // Create index first (on empty database)
    db.create_index("Symbol", "language").expect("Failed to create index");

    // Now insert entities - index should be maintained automatically
    let entities: Vec<Entity> = (0..10)
        .map(|i| {
            Entity::new(EntityId::new(0))
                .with_label("Symbol")
                .with_property("language", "rust")
                .with_property("index", i as i64)
        })
        .collect();

    db.bulk_insert_entities(&entities).expect("Failed to insert entities");

    // Verify lookup works - this tests actual index entries
    // Note: index_stats.entry_count is computed at creation time only,
    // so we verify the actual index entries via lookup
    let result = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup");
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 10);
}

#[test]
fn test_index_maintenance_on_update() {
    let db = Database::in_memory().expect("Failed to create database");

    // Insert entities
    let entities: Vec<Entity> = (0..10)
        .map(|i| {
            Entity::new(EntityId::new(0))
                .with_label("Symbol")
                .with_property("language", "rust")
                .with_property("index", i as i64)
        })
        .collect();

    let ids = db.bulk_insert_entities(&entities).expect("Failed to insert entities");

    // Create index after insert
    db.create_index("Symbol", "language").expect("Failed to create index");

    // Verify initial state
    let rust_ids = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(rust_ids.len(), 10);

    // Update some entities to change language
    let updated_entities: Vec<Entity> = ids
        .iter()
        .take(5)
        .map(|&id| {
            Entity::new(id)
                .with_label("Symbol")
                .with_property("language", "python") // Changed!
                .with_property("index", id.as_u64() as i64)
        })
        .collect();

    db.bulk_upsert_entities(&updated_entities).expect("Failed to update entities");

    // Verify index was updated
    let rust_ids = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(rust_ids.len(), 5); // 5 remaining

    let python_ids = db
        .index_lookup("Symbol", "language", &Value::String("python".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(python_ids.len(), 5); // 5 new
}

#[test]
fn test_index_maintenance_on_delete() {
    let db = setup_test_db();

    db.create_index("Symbol", "language").expect("Failed to create index");

    // Get initial count
    let initial_rust_ids = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    let initial_count = initial_rust_ids.len();

    // Delete half of the rust entities
    let to_delete: Vec<EntityId> =
        initial_rust_ids.iter().take(initial_count / 2).copied().collect();
    db.bulk_delete_entities(&to_delete).expect("Failed to delete entities");

    // Verify index was updated
    let remaining_rust_ids = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(remaining_rust_ids.len(), initial_count - to_delete.len());
}

#[test]
fn test_multiple_indexes() {
    let db = setup_test_db();

    // Create two indexes on different properties
    db.create_index("Symbol", "language").expect("Failed to create language index");
    db.create_index("Symbol", "visibility").expect("Failed to create visibility index");

    let indexes = db.list_indexes().expect("Failed to list indexes");
    assert_eq!(indexes.len(), 2);

    // Both should work for lookups
    let rust_ids = db
        .index_lookup("Symbol", "language", &Value::String("rust".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(rust_ids.len(), 25);

    let public_ids = db
        .index_lookup("Symbol", "visibility", &Value::String("public".to_string()))
        .expect("Failed to lookup")
        .unwrap();
    assert_eq!(public_ids.len(), 50); // Every other entity is public
}

#[test]
fn test_index_with_null_values() {
    let db = Database::in_memory().expect("Failed to create database");

    // Insert entities, some without the indexed property
    let entities = vec![
        Entity::new(EntityId::new(0)).with_label("Symbol").with_property("language", "rust"),
        Entity::new(EntityId::new(0)).with_label("Symbol").with_property("language", "python"),
        Entity::new(EntityId::new(0)).with_label("Symbol"), // No language property
    ];

    db.bulk_insert_entities(&entities).expect("Failed to insert entities");

    // Create index
    db.create_index("Symbol", "language").expect("Failed to create index");

    // Only 2 entities should be indexed (the ones with the property)
    let stats = db.index_stats("Symbol", "language").expect("Failed to get stats");
    assert_eq!(stats.entry_count, 2);
}

#[test]
fn test_index_key_ordering() {
    let db = Database::in_memory().expect("Failed to create database");

    // Insert entities with numeric values
    let entities: Vec<Entity> = (0..100)
        .map(|i| {
            Entity::new(EntityId::new(0)).with_label("Counter").with_property("value", i as i64)
        })
        .collect();

    db.bulk_insert_entities(&entities).expect("Failed to insert entities");

    // Create range index
    db.create_index_with_type("Counter", "value", IndexType::Range)
        .expect("Failed to create index");

    // Lookup for value = 50 should return exactly 1 entity
    let ids =
        db.index_lookup("Counter", "value", &Value::Int(50)).expect("Failed to lookup").unwrap();
    assert_eq!(ids.len(), 1);
}
