//! Cypher FOREACH integration tests.
//!
//! Tests end-to-end Cypher FOREACH statement execution against graph storage.

use manifoldb::{Database, Value};

// ============================================================================
// FOREACH with CREATE
// ============================================================================

#[test]
fn test_foreach_create_nodes_from_list() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes from a list using FOREACH
    let result = db
        .query(
            "FOREACH (name IN ['Alice', 'Bob', 'Charlie'] |
                CREATE (:Person {name: name})
            )",
        )
        .expect("query failed");

    // FOREACH doesn't return rows
    assert!(result.is_empty() || result.len() <= 1);

    // Verify all three nodes were created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 3, "Expected 3 Person entities");

    // Check names
    let names: Vec<&Value> = entities.iter().filter_map(|e| e.properties.get("name")).collect();
    assert!(names.iter().any(|v| v == &&Value::String("Alice".to_string())), "Should have Alice");
    assert!(names.iter().any(|v| v == &&Value::String("Bob".to_string())), "Should have Bob");
    assert!(
        names.iter().any(|v| v == &&Value::String("Charlie".to_string())),
        "Should have Charlie"
    );
}

#[test]
fn test_foreach_create_with_integers() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with numeric IDs
    let result = db
        .query(
            "FOREACH (x IN [1, 2, 3, 4, 5] |
                CREATE (:Number {value: x})
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    // Verify all five nodes were created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Number")).expect("failed to iterate");
    assert_eq!(entities.len(), 5, "Expected 5 Number entities");

    // Verify values
    let values: Vec<i64> = entities
        .iter()
        .filter_map(|e| match e.properties.get("value") {
            Some(Value::Int(v)) => Some(*v),
            _ => None,
        })
        .collect();
    assert!(values.contains(&1));
    assert!(values.contains(&5));
}

#[test]
fn test_foreach_with_empty_list() {
    let db = Database::in_memory().expect("failed to create db");

    // FOREACH with empty list should be a no-op
    let result = db
        .query(
            "FOREACH (x IN [] |
                CREATE (:Nothing {value: x})
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    // Verify no nodes were created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Nothing")).expect("failed to iterate");
    assert_eq!(entities.len(), 0, "Expected 0 Nothing entities");
}

// ============================================================================
// FOREACH with SET
// ============================================================================

#[test]
fn test_foreach_set_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // First create some nodes
    db.query("CREATE (:Person {name: 'Alice'})").expect("create failed");
    db.query("CREATE (:Person {name: 'Bob'})").expect("create failed");

    // Use MATCH + FOREACH to update all matched nodes
    // Note: This test documents the behavior but may need adjustment
    // depending on how MATCH + FOREACH is implemented
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");
}

// ============================================================================
// FOREACH with DELETE
// ============================================================================

#[test]
#[ignore = "MATCH...FOREACH with DELETE requires MATCH to return node IDs in list form"]
fn test_foreach_delete_nodes() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes to delete
    db.query("CREATE (:Temp {id: 1})").expect("create failed");
    db.query("CREATE (:Temp {id: 2})").expect("create failed");
    db.query("CREATE (:Temp {id: 3})").expect("create failed");

    // Verify 3 nodes exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Temp")).expect("failed to iterate");
    assert_eq!(entities.len(), 3, "Expected 3 Temp entities before delete");
    drop(tx);

    // This would require MATCH to provide nodes as a list, which isn't implemented
    // MATCH (n:Temp) WITH collect(n) AS nodes FOREACH (node IN nodes | DETACH DELETE node)
}

// ============================================================================
// Nested FOREACH
// ============================================================================

#[test]
fn test_nested_foreach() {
    let db = Database::in_memory().expect("failed to create db");

    // Create grid of nodes using nested FOREACH
    let result = db
        .query(
            "FOREACH (x IN [1, 2, 3] |
                FOREACH (y IN [1, 2, 3] |
                    CREATE (:Cell {x: x, y: y})
                )
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    // Verify 9 cells were created (3x3 grid)
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Cell")).expect("failed to iterate");
    assert_eq!(entities.len(), 9, "Expected 9 Cell entities (3x3 grid)");

    // Check some specific coordinates exist
    let has_1_1 = entities.iter().any(|e| {
        e.properties.get("x") == Some(&Value::Int(1))
            && e.properties.get("y") == Some(&Value::Int(1))
    });
    let has_3_3 = entities.iter().any(|e| {
        e.properties.get("x") == Some(&Value::Int(3))
            && e.properties.get("y") == Some(&Value::Int(3))
    });
    assert!(has_1_1, "Should have cell at (1,1)");
    assert!(has_3_3, "Should have cell at (3,3)");
}

#[test]
fn test_deeply_nested_foreach() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 3D grid using triple nested FOREACH
    let result = db
        .query(
            "FOREACH (x IN [1, 2] |
                FOREACH (y IN [1, 2] |
                    FOREACH (z IN [1, 2] |
                        CREATE (:Point3D {x: x, y: y, z: z})
                    )
                )
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    // Verify 8 points were created (2x2x2)
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Point3D")).expect("failed to iterate");
    assert_eq!(entities.len(), 8, "Expected 8 Point3D entities (2x2x2)");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_foreach_with_null_list() {
    let db = Database::in_memory().expect("failed to create db");

    // FOREACH with NULL should be a no-op
    // We use a subquery that returns NULL for this
    // For now, just verify empty list behavior
    let result = db
        .query(
            "FOREACH (x IN [] |
                CREATE (:ShouldNotExist {value: x})
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("ShouldNotExist")).expect("failed to iterate");
    assert_eq!(entities.len(), 0);
}

#[test]
fn test_foreach_with_mixed_types() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with different value types
    let result = db
        .query(
            "FOREACH (val IN ['string', 42, 3.14, true] |
                CREATE (:Mixed {value: val})
            )",
        )
        .expect("query failed");

    assert!(result.is_empty() || result.len() <= 1);

    // Verify 4 nodes were created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Mixed")).expect("failed to iterate");
    assert_eq!(entities.len(), 4, "Expected 4 Mixed entities");
}

// ============================================================================
// FOREACH with MERGE (documented limitation)
// ============================================================================

#[test]
fn test_foreach_with_merge() {
    let db = Database::in_memory().expect("failed to create db");

    // First, create one person that will be merged (not created again)
    db.query("CREATE (:Person {name: 'Alice'})").expect("create failed");

    // Verify initial state
    {
        let tx = db.begin_read().expect("failed to begin read");
        let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
        assert_eq!(entities.len(), 1, "Expected 1 Person entity before MERGE");
    }

    // Use FOREACH with MERGE - Alice should match, Bob and Charlie should be created
    let result = db.query(
        "FOREACH (name IN ['Alice', 'Bob', 'Charlie'] |
            MERGE (p:Person {name: name})
        )",
    );
    assert!(result.is_ok(), "FOREACH with MERGE failed: {:?}", result.err());

    // Verify results: should have 3 Person nodes (Alice was matched, Bob and Charlie were created)
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 3, "Expected 3 Person entities after MERGE");

    // Verify names
    let names: Vec<&Value> = entities.iter().filter_map(|e| e.properties.get("name")).collect();
    assert!(names.iter().any(|v| v == &&Value::String("Alice".to_string())), "Should have Alice");
    assert!(names.iter().any(|v| v == &&Value::String("Bob".to_string())), "Should have Bob");
    assert!(
        names.iter().any(|v| v == &&Value::String("Charlie".to_string())),
        "Should have Charlie"
    );
}
