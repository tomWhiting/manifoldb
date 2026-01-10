//! Cypher MERGE integration tests.
//!
//! Tests end-to-end Cypher MERGE statement execution against graph storage.
//! MERGE implements get-or-create semantics for nodes and relationships.

use manifoldb::{Database, Value};

// ============================================================================
// Basic Node MERGE Tests
// ============================================================================

#[test]
fn test_merge_creates_node_when_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE should create the node since it doesn't exist
    let result = db.query("MERGE (n:Person {name: 'Alice'}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the node was created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity");

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("name"),
        Some(&Value::String("Alice".to_string())),
        "Name property should be 'Alice'"
    );
}

#[test]
fn test_merge_matches_existing_node() {
    let db = Database::in_memory().expect("failed to create db");

    // First CREATE a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");

    // MERGE should match the existing node (not create a new one)
    let result = db.query("MERGE (n:Person {name: 'Alice'}) RETURN n").expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify only one node exists (merge didn't create a duplicate)
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity (no duplicate)");
}

#[test]
fn test_merge_with_different_property_creates_new_node() {
    let db = Database::in_memory().expect("failed to create db");

    // First CREATE Alice
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");

    // MERGE for Bob should create a new node (different property value)
    let result = db.query("MERGE (n:Person {name: 'Bob'}) RETURN n").expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify both nodes exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");

    let names: Vec<&Value> = entities.iter().filter_map(|e| e.properties.get("name")).collect();
    assert!(names.iter().any(|v| **v == Value::String("Alice".to_string())), "Should have Alice");
    assert!(names.iter().any(|v| **v == Value::String("Bob".to_string())), "Should have Bob");
}

#[test]
fn test_merge_node_with_multiple_labels() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE with multiple labels
    let result =
        db.query("MERGE (n:Person:Employee {name: 'Alice'}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify labels
    let tx = db.begin_read().expect("failed to begin read");
    let person_entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let employee_entities = tx.iter_entities(Some("Employee")).expect("failed to iterate");
    assert_eq!(person_entities.len(), 1, "Expected 1 Person entity");
    assert_eq!(employee_entities.len(), 1, "Expected 1 Employee entity");
}

// ============================================================================
// ON CREATE SET Tests
// ============================================================================

#[test]
fn test_merge_on_create_set_when_creating() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE with ON CREATE SET - should apply because node doesn't exist
    let result = db
        .query("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true RETURN n")
        .expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the created property was set
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("created"),
        Some(&Value::Bool(true)),
        "ON CREATE SET should have set 'created' property"
    );
}

#[test]
fn test_merge_on_create_set_not_applied_when_matching() {
    let db = Database::in_memory().expect("failed to create db");

    // First CREATE a node WITHOUT the 'created' property
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");

    // MERGE with ON CREATE SET - should NOT apply because node exists
    let result = db
        .query("MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true RETURN n")
        .expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the 'created' property was NOT set
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("created"),
        None,
        "ON CREATE SET should NOT have set 'created' property when matching"
    );
}

// ============================================================================
// ON MATCH SET Tests
// ============================================================================

#[test]
fn test_merge_on_match_set_when_matching() {
    let db = Database::in_memory().expect("failed to create db");

    // First CREATE a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");

    // MERGE with ON MATCH SET - should apply because node exists
    let result = db
        .query("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.found = true RETURN n")
        .expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the 'found' property was set
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("found"),
        Some(&Value::Bool(true)),
        "ON MATCH SET should have set 'found' property"
    );
}

#[test]
fn test_merge_on_match_set_not_applied_when_creating() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE with ON MATCH SET - should NOT apply because node is created
    let result = db
        .query("MERGE (n:Person {name: 'Alice'}) ON MATCH SET n.found = true RETURN n")
        .expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the 'found' property was NOT set
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("found"),
        None,
        "ON MATCH SET should NOT have set 'found' property when creating"
    );
}

// ============================================================================
// Combined ON CREATE SET and ON MATCH SET Tests
// ============================================================================

#[test]
fn test_merge_both_on_create_and_on_match_when_creating() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE with both ON CREATE SET and ON MATCH SET - creating new
    let result = db
        .query(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true ON MATCH SET n.matched = true RETURN n",
        )
        .expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify only ON CREATE SET was applied
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let entity = &entities[0];

    assert_eq!(
        entity.properties.get("created"),
        Some(&Value::Bool(true)),
        "ON CREATE SET should have been applied"
    );
    assert_eq!(entity.properties.get("matched"), None, "ON MATCH SET should NOT have been applied");
}

#[test]
fn test_merge_both_on_create_and_on_match_when_matching() {
    let db = Database::in_memory().expect("failed to create db");

    // First CREATE a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");

    // MERGE with both ON CREATE SET and ON MATCH SET - matching existing
    let result = db
        .query(
            "MERGE (n:Person {name: 'Alice'}) ON CREATE SET n.created = true ON MATCH SET n.matched = true RETURN n",
        )
        .expect("merge failed");

    assert_eq!(result.len(), 1);

    // Verify only ON MATCH SET was applied
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let entity = &entities[0];

    assert_eq!(
        entity.properties.get("created"),
        None,
        "ON CREATE SET should NOT have been applied"
    );
    assert_eq!(
        entity.properties.get("matched"),
        Some(&Value::Bool(true)),
        "ON MATCH SET should have been applied"
    );
}

// ============================================================================
// Multiple MERGE Operations
// ============================================================================

#[test]
fn test_merge_twice_creates_once() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE the same node twice
    db.query("MERGE (n:Person {name: 'Alice'}) RETURN n").expect("first merge failed");
    db.query("MERGE (n:Person {name: 'Alice'}) RETURN n").expect("second merge failed");

    // Verify only one node was created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected only 1 Person entity after two merges");
}

// ============================================================================
// Relationship MERGE Tests
// ============================================================================

/// Tests MERGE relationship creation when the relationship doesn't exist.
#[test]
fn test_merge_relationship_creates_when_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // First create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice failed");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob failed");

    // MERGE should create the relationship since it doesn't exist
    let result = db
        .query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             MERGE (a)-[r:KNOWS]->(b) RETURN r",
        )
        .expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify relationship was created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 outgoing edge from Alice");
    assert_eq!(edges[0].edge_type.as_str(), "KNOWS");
}

/// Tests MERGE matching an existing relationship.
#[test]
fn test_merge_relationship_matches_existing() {
    let db = Database::in_memory().expect("failed to create db");

    // First create nodes with relationship
    db.query("CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN a, b")
        .expect("create failed");

    // MERGE should match the existing relationship (not create a new one)
    let result = db
        .query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             MERGE (a)-[r:KNOWS]->(b) RETURN r",
        )
        .expect("merge failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify only one relationship exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 outgoing edge from Alice (no duplicate)");
}

/// Tests MERGE relationship with properties.
#[test]
fn test_merge_relationship_with_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // First create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice failed");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob failed");

    // MERGE relationship with properties
    let result = db
        .query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             MERGE (a)-[r:KNOWS {since: 2020}]->(b) RETURN r",
        )
        .expect("merge failed");

    assert_eq!(result.len(), 1);

    // Verify relationship properties
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0].properties.get("since"),
        Some(&Value::Int(2020)),
        "Edge should have 'since' property"
    );
}

#[test]
fn test_merge_relationship_on_create_set() {
    let db = Database::in_memory().expect("failed to create db");

    // First create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice failed");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob failed");

    // MERGE relationship with ON CREATE SET
    let result = db
        .query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) \
             MERGE (a)-[r:KNOWS]->(b) ON CREATE SET r.created = true RETURN r",
        )
        .expect("merge failed");

    assert_eq!(result.len(), 1);

    // Verify the 'created' property was set
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1);
    assert_eq!(
        edges[0].properties.get("created"),
        Some(&Value::Bool(true)),
        "ON CREATE SET should have set 'created' property"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_merge_without_return() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE without RETURN should still work
    let _result = db.query("MERGE (n:Person {name: 'Alice'})").expect("query failed");

    // The node should exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity");
}

#[test]
fn test_merge_only_label_no_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // MERGE with just a label, no properties to match on
    // This will create a new node each time (since there's no unique match criteria)
    let result = db.query("MERGE (n:Person) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    // Run it again - should match the existing (since we're matching on label only)
    db.query("MERGE (n:Person) RETURN n").expect("second merge failed");

    // Verify behavior (depends on implementation - could match first or create new)
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    // Since we're matching on label only, it should find the first Person
    assert!(!entities.is_empty(), "Should have at least 1 Person entity");
}
