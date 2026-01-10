//! Cypher CREATE integration tests.
//!
//! Tests end-to-end Cypher CREATE statement execution against graph storage.

use manifoldb::{Database, Value};

// ============================================================================
// Basic Node Creation Tests
// ============================================================================

#[test]
fn test_create_node_without_label() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node without any labels
    let result = db.query("CREATE (n) RETURN n").expect("query failed");

    // Should return one row with the created node ID
    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}

#[test]
fn test_create_node_with_single_label() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a Person node
    let result = db.query("CREATE (n:Person) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the node was created with the correct label
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity");
}

#[test]
fn test_create_node_with_multiple_labels() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with multiple labels
    let result = db.query("CREATE (n:Person:Employee) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify both labels are present
    let tx = db.begin_read().expect("failed to begin read");
    let person_entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let employee_entities = tx.iter_entities(Some("Employee")).expect("failed to iterate");
    assert_eq!(person_entities.len(), 1, "Expected 1 Person entity");
    assert_eq!(employee_entities.len(), 1, "Expected 1 Employee entity");
}

#[test]
fn test_create_node_with_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with properties
    let result =
        db.query("CREATE (n:Person {name: 'Alice', age: 30}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify the node has correct properties
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);

    let entity = &entities[0];
    assert_eq!(
        entity.properties.get("name"),
        Some(&Value::String("Alice".to_string())),
        "Name property should be 'Alice'"
    );
    assert_eq!(entity.properties.get("age"), Some(&Value::Int(30)), "Age property should be 30");
}

// ============================================================================
// Multiple Node Creation Tests
// ============================================================================

#[test]
fn test_create_multiple_nodes() {
    let db = Database::in_memory().expect("failed to create db");

    // Create multiple nodes in one statement
    let result = db
        .query("CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) RETURN a, b")
        .expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row with 2 columns");

    // Verify both nodes were created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");

    // Check names
    let names: Vec<&Value> = entities.iter().filter_map(|e| e.properties.get("name")).collect();
    assert!(names.iter().any(|v| v == &&Value::String("Alice".to_string())), "Should have Alice");
    assert!(names.iter().any(|v| v == &&Value::String("Bob".to_string())), "Should have Bob");
}

// ============================================================================
// Relationship Creation Tests
// ============================================================================

#[test]
fn test_create_relationship_between_new_nodes() {
    let db = Database::in_memory().expect("failed to create db");

    // Create two nodes and a relationship between them in one statement
    let result = db
        .query(
            "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'}) RETURN a, r, b",
        )
        .expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row");

    // Verify nodes
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");

    // Find Alice and Bob's IDs
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");
    let bob = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Bob".to_string())))
        .expect("Bob not found");

    // Verify relationship
    let outgoing = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(outgoing.len(), 1, "Expected 1 outgoing edge from Alice");
    assert_eq!(outgoing[0].target, bob.id, "Edge should point to Bob");
    assert_eq!(outgoing[0].edge_type.as_str(), "KNOWS", "Edge type should be KNOWS");
}

#[test]
fn test_create_relationship_with_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // Create relationship with properties
    let result = db
        .query(
            "CREATE (a:Person {name: 'Alice'})-[r:KNOWS {since: 2020}]->(b:Person {name: 'Bob'}) RETURN a, r, b",
        )
        .expect("query failed");

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

// ============================================================================
// MATCH + CREATE Tests
// ============================================================================

/// Tests MATCH ... CREATE with property-based matching.
///
/// NOTE: This test is marked as `#[ignore]` because MATCH clause execution
/// with property filters is not yet fully implemented. The CREATE execution
/// itself works correctly - the limitation is in the MATCH pattern matching
/// for standalone node patterns with property predicates.
#[test]
#[ignore = "MATCH with property filters not yet implemented - see task #5150"]
fn test_match_then_create_relationship() {
    let db = Database::in_memory().expect("failed to create db");

    // First create the nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create failed");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create failed");

    // Now match them and create a relationship
    let result = db
        .query(
            "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS]->(b) RETURN r",
        )
        .expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row for relationship creation");

    // Verify the relationship was created
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 outgoing edge from Alice");
}

// ============================================================================
// Anonymous Node/Relationship Tests
// ============================================================================

#[test]
fn test_create_anonymous_node() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node without binding to a variable
    let result = db.query("CREATE (:Person {name: 'Alice'})").expect("query failed");

    // No rows returned since no RETURN clause
    assert!(result.is_empty() || result.len() == 1);

    // But the node should exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity");
}

#[test]
fn test_create_anonymous_relationship() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes and relationship without binding rel to variable
    let result = db
        .query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'}) RETURN a, b")
        .expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify relationship exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1);
}

// ============================================================================
// Property Type Tests
// ============================================================================

#[test]
fn test_create_node_with_string_property() {
    let db = Database::in_memory().expect("failed to create db");

    let result = db.query("CREATE (n:Test {value: 'hello world'}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Test")).expect("failed to iterate");
    assert_eq!(
        entities[0].properties.get("value"),
        Some(&Value::String("hello world".to_string()))
    );
}

#[test]
fn test_create_node_with_integer_property() {
    let db = Database::in_memory().expect("failed to create db");

    let result = db.query("CREATE (n:Test {value: 42}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Test")).expect("failed to iterate");
    assert_eq!(entities[0].properties.get("value"), Some(&Value::Int(42)));
}

#[test]
fn test_create_node_with_float_property() {
    let db = Database::in_memory().expect("failed to create db");

    let result = db.query("CREATE (n:Test {value: 3.14}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Test")).expect("failed to iterate");

    if let Some(Value::Float(f)) = entities[0].properties.get("value") {
        assert!((f - 3.14).abs() < 0.001);
    } else {
        panic!("Expected float property");
    }
}

#[test]
fn test_create_node_with_boolean_property() {
    let db = Database::in_memory().expect("failed to create db");

    let result = db.query("CREATE (n:Test {active: true}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Test")).expect("failed to iterate");
    assert_eq!(entities[0].properties.get("active"), Some(&Value::Bool(true)));
}

// ============================================================================
// Complex Pattern Tests
// ============================================================================

#[test]
fn test_create_path_with_multiple_relationships() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a path: Alice -KNOWS-> Bob -WORKS_WITH-> Charlie
    let result = db
        .query(
            "CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})-[:WORKS_WITH]->(c:Person {name: 'Charlie'}) RETURN a, b, c",
        )
        .expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify nodes
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 3, "Expected 3 Person entities");

    // Verify relationships
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");
    let bob = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Bob".to_string())))
        .expect("Bob not found");

    let alice_edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(alice_edges.len(), 1, "Alice should have 1 outgoing edge");
    assert_eq!(alice_edges[0].edge_type.as_str(), "KNOWS");

    let bob_edges = tx.get_outgoing_edges(bob.id).expect("failed to get edges");
    assert_eq!(bob_edges.len(), 1, "Bob should have 1 outgoing edge");
    assert_eq!(bob_edges[0].edge_type.as_str(), "WORKS_WITH");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_create_without_return() {
    let db = Database::in_memory().expect("failed to create db");

    // CREATE without RETURN should still work
    let _result = db.query("CREATE (n:Person {name: 'Alice'})").expect("query failed");

    // The result may be empty or have metadata
    // But the node should exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);
}

#[test]
fn test_create_empty_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // CREATE with empty properties block
    let result = db.query("CREATE (n:Person {}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1);

    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1);
    // Properties should be empty (just id added by system)
    assert!(
        entities[0].properties.is_empty() || entities[0].properties.len() <= 1,
        "Expected no custom properties"
    );
}

// ============================================================================
// RETURN Full Node Data Tests
// ============================================================================

#[test]
fn test_return_node_includes_full_data() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with properties
    let result =
        db.query("CREATE (n:Person {name: 'Alice', age: 30}) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row");

    // Get the returned node value
    let row = result.iter().next().expect("Expected at least one row");
    let node_value = row.values().first().expect("Expected at least one value");

    // Verify we get a full Node value with id, labels, and properties
    match node_value {
        Value::Node { id, labels, properties } => {
            // ID should be a positive integer
            assert!(*id > 0, "Node ID should be positive, got {id}");

            // Should have the Person label
            assert!(
                labels.contains(&"Person".to_string()),
                "Expected Person label, got {:?}",
                labels
            );

            // Should have properties
            assert_eq!(
                properties.get("name"),
                Some(&Value::String("Alice".to_string())),
                "Expected name property"
            );
            assert_eq!(properties.get("age"), Some(&Value::Int(30)), "Expected age property");
        }
        other => {
            panic!("Expected Value::Node, got {:?}", other);
        }
    }
}

#[test]
fn test_match_return_node_includes_full_data() {
    let db = Database::in_memory().expect("failed to create db");

    // First create a node
    db.query("CREATE (n:Person {name: 'Bob', score: 100})").expect("create failed");

    // Now MATCH and RETURN it
    let result = db.query("MATCH (n:Person) RETURN n").expect("query failed");

    assert_eq!(result.len(), 1, "Expected 1 row");

    // Get the returned node value
    let row = result.iter().next().expect("Expected at least one row");
    let node_value = row.values().first().expect("Expected at least one value");

    // Verify we get a full Node value
    match node_value {
        Value::Node { id, labels, properties } => {
            assert!(*id > 0, "Node ID should be positive");
            assert!(labels.contains(&"Person".to_string()), "Expected Person label");
            assert_eq!(
                properties.get("name"),
                Some(&Value::String("Bob".to_string())),
                "Expected name property"
            );
            assert_eq!(properties.get("score"), Some(&Value::Int(100)), "Expected score property");
        }
        other => {
            panic!("Expected Value::Node from MATCH RETURN, got {:?}", other);
        }
    }
}
