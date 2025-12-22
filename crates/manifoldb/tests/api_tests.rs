//! Integration tests for the ManifoldDB public API.

use manifoldb::{
    Config, Database, DatabaseBuilder, Error, QueryParams, QueryResult, QueryRow, Value,
    VectorSyncStrategy,
};

// ============================================================================
// Database Opening Tests
// ============================================================================

#[test]
fn test_database_in_memory() {
    let db = Database::in_memory().expect("failed to create in-memory db");
    assert!(db.config().in_memory);
}

#[test]
fn test_database_builder_in_memory() {
    let db = DatabaseBuilder::in_memory().open().expect("failed to create in-memory db");
    assert!(db.config().in_memory);
}

#[test]
fn test_database_builder_with_cache_size() {
    let db = DatabaseBuilder::in_memory()
        .cache_size(64 * 1024 * 1024)
        .open()
        .expect("failed to create db");

    assert!(db.config().in_memory);
    assert_eq!(db.config().cache_size, Some(64 * 1024 * 1024));
}

#[test]
fn test_database_builder_vector_sync_strategy() {
    let db = DatabaseBuilder::in_memory()
        .vector_sync_strategy(VectorSyncStrategy::Async)
        .open()
        .expect("failed to create db");

    assert_eq!(db.config().vector_sync_strategy, VectorSyncStrategy::Async);
}

#[test]
fn test_database_builder_requires_path() {
    let result = DatabaseBuilder::new().build();
    assert!(result.is_err());
}

#[test]
fn test_database_static_builder() {
    let db = Database::builder()
        .path("/tmp/test.manifold")
        .create_if_missing(false)
        .build()
        .expect("failed to build config");

    assert!(!db.create_if_missing);
}

// ============================================================================
// Transaction Tests
// ============================================================================

#[test]
fn test_begin_write_transaction() {
    let db = Database::in_memory().expect("failed to create db");

    let tx = db.begin().expect("failed to begin transaction");
    assert!(!tx.is_read_only());
    tx.rollback().expect("failed to rollback");
}

#[test]
fn test_begin_read_transaction() {
    let db = Database::in_memory().expect("failed to create db");

    let tx = db.begin_read().expect("failed to begin read transaction");
    assert!(tx.is_read_only());
    tx.rollback().expect("failed to rollback");
}

#[test]
fn test_transaction_commit() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id;
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        let entity = tx.create_entity().expect("failed to create entity");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed to put entity");
        tx.commit().expect("failed to commit");
    }

    // Verify entity is visible after commit
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get entity");
    assert!(entity.is_some());
}

#[test]
fn test_transaction_rollback() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id;
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        let entity = tx.create_entity().expect("failed to create entity");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed to put entity");
        tx.rollback().expect("failed to rollback");
    }

    // Verify entity is NOT visible after rollback
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get entity");
    assert!(entity.is_none());
}

#[test]
fn test_transaction_drop_rollback() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id;
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        let entity = tx.create_entity().expect("failed to create entity");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed to put entity");
        // Drop without commit or rollback
    }

    // Verify entity is NOT visible (should be rolled back)
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get entity");
    assert!(entity.is_none());
}

// ============================================================================
// Entity CRUD Tests
// ============================================================================

#[test]
fn test_entity_create_and_retrieve() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");
    let entity = tx
        .create_entity()
        .expect("failed to create entity")
        .with_label("Person")
        .with_property("name", "Alice")
        .with_property("age", 30i64);

    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Retrieve in new transaction
    let tx = db.begin_read().expect("failed to begin read");
    let retrieved = tx.get_entity(entity_id).expect("failed to get").expect("entity not found");

    assert_eq!(retrieved.id, entity_id);
    assert!(retrieved.has_label("Person"));
    assert_eq!(retrieved.get_property("name"), Some(&Value::String("Alice".to_string())));
    assert_eq!(retrieved.get_property("age"), Some(&Value::Int(30)));
}

#[test]
fn test_entity_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Create
    let mut tx = db.begin().expect("failed to begin");
    let entity = tx.create_entity().expect("failed to create").with_property("count", 1i64);
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Update
    let mut tx = db.begin().expect("failed to begin");
    let mut entity = tx.get_entity(entity_id).expect("failed to get").expect("not found");
    entity.set_property("count", 2i64);
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Verify
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get").expect("not found");
    assert_eq!(entity.get_property("count"), Some(&Value::Int(2)));
}

#[test]
fn test_entity_delete() {
    let db = Database::in_memory().expect("failed to create db");

    // Create
    let mut tx = db.begin().expect("failed to begin");
    let entity = tx.create_entity().expect("failed to create");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Delete
    let mut tx = db.begin().expect("failed to begin");
    let deleted = tx.delete_entity(entity_id).expect("failed to delete");
    assert!(deleted);
    tx.commit().expect("failed to commit");

    // Verify deletion
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get");
    assert!(entity.is_none());
}

#[test]
fn test_entity_multiple_labels() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");
    let entity = tx
        .create_entity()
        .expect("failed to create")
        .with_label("Person")
        .with_label("Employee")
        .with_label("Manager");

    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get").expect("not found");

    assert!(entity.has_label("Person"));
    assert!(entity.has_label("Employee"));
    assert!(entity.has_label("Manager"));
    assert!(!entity.has_label("Customer"));
}

// ============================================================================
// Edge Tests
// ============================================================================

#[test]
fn test_edge_create_and_retrieve() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    let alice = tx.create_entity().expect("failed to create").with_label("Person");
    let bob = tx.create_entity().expect("failed to create").with_label("Person");
    let alice_id = alice.id;
    let bob_id = bob.id;
    tx.put_entity(&alice).expect("failed to put");
    tx.put_entity(&bob).expect("failed to put");

    let edge = tx
        .create_edge(alice_id, bob_id, "FOLLOWS")
        .expect("failed to create edge")
        .with_property("since", "2024");
    let edge_id = edge.id;
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Retrieve
    let tx = db.begin_read().expect("failed to begin read");
    let edge = tx.get_edge(edge_id).expect("failed to get").expect("not found");

    assert_eq!(edge.id, edge_id);
    assert_eq!(edge.source, alice_id);
    assert_eq!(edge.target, bob_id);
    assert_eq!(edge.edge_type.as_str(), "FOLLOWS");
    assert_eq!(edge.get_property("since"), Some(&Value::String("2024".to_string())));
}

#[test]
fn test_edge_traversal_outgoing() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    let a = tx.create_entity().expect("failed to create");
    let b = tx.create_entity().expect("failed to create");
    let c = tx.create_entity().expect("failed to create");

    tx.put_entity(&a).expect("failed to put");
    tx.put_entity(&b).expect("failed to put");
    tx.put_entity(&c).expect("failed to put");

    // A -> B, A -> C
    let e1 = tx.create_edge(a.id, b.id, "LINKS").expect("failed to create");
    let e2 = tx.create_edge(a.id, c.id, "LINKS").expect("failed to create");
    tx.put_edge(&e1).expect("failed to put");
    tx.put_edge(&e2).expect("failed to put");

    tx.commit().expect("failed to commit");

    // Traverse from A
    let tx = db.begin_read().expect("failed to begin read");
    let edges = tx.get_outgoing_edges(a.id).expect("failed to get");

    assert_eq!(edges.len(), 2);
    let targets: Vec<_> = edges.iter().map(|e| e.target).collect();
    assert!(targets.contains(&b.id));
    assert!(targets.contains(&c.id));
}

#[test]
fn test_edge_traversal_incoming() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");

    let a = tx.create_entity().expect("failed to create");
    let b = tx.create_entity().expect("failed to create");
    let c = tx.create_entity().expect("failed to create");

    tx.put_entity(&a).expect("failed to put");
    tx.put_entity(&b).expect("failed to put");
    tx.put_entity(&c).expect("failed to put");

    // A -> C, B -> C
    let e1 = tx.create_edge(a.id, c.id, "POINTS").expect("failed to create");
    let e2 = tx.create_edge(b.id, c.id, "POINTS").expect("failed to create");
    tx.put_edge(&e1).expect("failed to put");
    tx.put_edge(&e2).expect("failed to put");

    tx.commit().expect("failed to commit");

    // Traverse to C
    let tx = db.begin_read().expect("failed to begin read");
    let edges = tx.get_incoming_edges(c.id).expect("failed to get");

    assert_eq!(edges.len(), 2);
    let sources: Vec<_> = edges.iter().map(|e| e.source).collect();
    assert!(sources.contains(&a.id));
    assert!(sources.contains(&b.id));
}

// ============================================================================
// Query Result Tests
// ============================================================================

#[test]
fn test_query_result_empty() {
    let result = QueryResult::empty();
    assert!(result.is_empty());
    assert_eq!(result.len(), 0);
    assert_eq!(result.num_columns(), 0);
}

#[test]
fn test_query_result_with_rows() {
    let columns = vec!["id".to_string(), "name".to_string()];
    let rows = vec![
        QueryRow::new(vec![Value::Int(1), Value::String("Alice".to_string())]),
        QueryRow::new(vec![Value::Int(2), Value::String("Bob".to_string())]),
    ];

    let result = QueryResult::new(columns, rows);

    assert_eq!(result.len(), 2);
    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.columns(), &["id", "name"]);
    assert_eq!(result.column_index("name"), Some(1));
    assert_eq!(result.column_index("missing"), None);
}

#[test]
fn test_query_row_get_as() {
    let row = QueryRow::new(vec![
        Value::Int(42),
        Value::String("hello".to_string()),
        Value::Bool(true),
        Value::Float(2.5),
        Value::Null,
    ]);

    assert_eq!(row.get_as::<i64>(0).unwrap(), 42);
    assert_eq!(row.get_as::<String>(1).unwrap(), "hello");
    assert_eq!(row.get_as::<bool>(2).unwrap(), true);
    assert!((row.get_as::<f64>(3).unwrap() - 2.5).abs() < f64::EPSILON);
    assert_eq!(row.get_as::<Option<String>>(4).unwrap(), None);
}

#[test]
fn test_query_row_type_error() {
    let row = QueryRow::new(vec![Value::Int(42)]);

    let result = row.get_as::<String>(0);
    assert!(result.is_err());
}

#[test]
fn test_query_result_iterator() {
    let columns = vec!["n".to_string()];
    let rows = vec![
        QueryRow::new(vec![Value::Int(1)]),
        QueryRow::new(vec![Value::Int(2)]),
        QueryRow::new(vec![Value::Int(3)]),
    ];

    let result = QueryResult::new(columns, rows);

    let sum: i64 = result.iter().filter_map(|r| r.get_as::<i64>(0).ok()).sum();
    assert_eq!(sum, 6);
}

// ============================================================================
// Query Parameters Tests
// ============================================================================

#[test]
fn test_query_params() {
    let params =
        QueryParams::new().with("Alice").with(30i64).with(true).with(vec![1.0f32, 2.0, 3.0]);

    assert_eq!(params.len(), 4);
    assert!(!params.is_empty());
}

#[test]
fn test_query_params_add() {
    let mut params = QueryParams::new();
    params.add("Alice");
    params.add(30i64);

    assert_eq!(params.len(), 2);
    assert_eq!(params.values()[0], Value::String("Alice".to_string()));
}

#[test]
fn test_query_params_clear() {
    let mut params = QueryParams::new().with("test");
    assert_eq!(params.len(), 1);

    params.clear();
    assert!(params.is_empty());
}

// ============================================================================
// SQL Parsing Tests
// ============================================================================

#[test]
fn test_query_parsing_valid() {
    let db = Database::in_memory().expect("failed to create db");

    // These should parse successfully
    assert!(db.query("SELECT * FROM users").is_ok());
    assert!(db.query("SELECT id, name FROM users WHERE age > 25").is_ok());
    assert!(db.query("SELECT * FROM users ORDER BY name LIMIT 10").is_ok());
}

#[test]
fn test_query_parsing_invalid() {
    let db = Database::in_memory().expect("failed to create db");

    // These should fail to parse
    assert!(db.query("INVALID SYNTAX !!!").is_err());
    assert!(db.query("SELECTT * FROM users").is_err());
}

#[test]
fn test_execute_parsing_valid() {
    let db = Database::in_memory().expect("failed to create db");

    // These should parse successfully
    assert!(db.execute("INSERT INTO users (name) VALUES ('Alice')").is_ok());
    assert!(db.execute("UPDATE users SET name = 'Bob' WHERE id = 1").is_ok());
    assert!(db.execute("DELETE FROM users WHERE id = 1").is_ok());
}

// ============================================================================
// Error Tests
// ============================================================================

#[test]
fn test_error_is_recoverable() {
    assert!(Error::Parse("test".to_string()).is_recoverable());
    assert!(Error::InvalidParameter("test".to_string()).is_recoverable());
    assert!(Error::Type("test".to_string()).is_recoverable());
    assert!(Error::Execution("test".to_string()).is_recoverable());

    assert!(!Error::Open("test".to_string()).is_recoverable());
    assert!(!Error::Config("test".to_string()).is_recoverable());
    assert!(!Error::Closed.is_recoverable());
}

#[test]
fn test_error_helpers() {
    let err = Error::parse("unexpected token");
    assert_eq!(err.to_string(), "parse error: unexpected token");

    let err = Error::execution("query timeout");
    assert_eq!(err.to_string(), "execution error: query timeout");

    let err = Error::config("invalid path");
    assert_eq!(err.to_string(), "configuration error: invalid path");
}

// ============================================================================
// Config Tests
// ============================================================================

#[test]
fn test_config_new() {
    let config = Config::new("/tmp/test.manifold");
    assert!(!config.in_memory);
    assert!(config.create_if_missing);
}

#[test]
fn test_config_in_memory() {
    let config = Config::in_memory();
    assert!(config.in_memory);
}

#[test]
fn test_config_vector_sync_strategy() {
    let config = Config::new("/tmp/test.manifold");
    assert_eq!(config.vector_sync_strategy, VectorSyncStrategy::Synchronous);
}

// ============================================================================
// Vector Property Tests
// ============================================================================

#[test]
fn test_entity_with_vector_property() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");
    let embedding = vec![0.1f32, 0.2, 0.3, 0.4];
    let entity = tx
        .create_entity()
        .expect("failed to create")
        .with_label("Document")
        .with_property("embedding", embedding.clone());

    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Retrieve
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get").expect("not found");

    match entity.get_property("embedding") {
        Some(Value::Vector(v)) => {
            assert_eq!(v.len(), 4);
            assert!((v[0] - 0.1f32).abs() < f32::EPSILON);
        }
        _ => panic!("Expected vector property"),
    }
}

// ============================================================================
// Flush Tests
// ============================================================================

#[test]
fn test_database_flush() {
    let db = Database::in_memory().expect("failed to create db");

    // Create some data
    let mut tx = db.begin().expect("failed to begin");
    let entity = tx.create_entity().expect("failed to create");
    tx.put_entity(&entity).expect("failed to put");
    tx.commit().expect("failed to commit");

    // Flush should succeed
    assert!(db.flush().is_ok());
}
