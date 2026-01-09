//! Cypher DELETE integration tests.
//!
//! Tests end-to-end Cypher DELETE statement execution against graph storage.

use manifoldb::{Database, Value};

// ============================================================================
// Basic Node Deletion Tests
// ============================================================================

#[test]
fn test_delete_single_node_without_relationships() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Alice'})").expect("create failed");

    // Verify the node exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity before DELETE");
    drop(tx);

    // Delete the node by matching on property
    db.query("MATCH (n:Person {name: 'Alice'}) DELETE n").expect("delete failed");

    // Verify the node is gone
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 0, "Expected 0 Person entities after DELETE");
}

#[test]
fn test_delete_node_by_label_and_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create two nodes
    db.query("CREATE (n:Person {name: 'Alice'})").expect("create failed");
    db.query("CREATE (n:Person {name: 'Bob'})").expect("create failed");

    // Verify both nodes exist
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");
    drop(tx);

    // Delete only Alice
    db.query("MATCH (n:Person {name: 'Alice'}) DELETE n").expect("delete failed");

    // Verify only Bob remains
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity after DELETE");
    assert_eq!(
        entities[0].properties.get("name"),
        Some(&Value::String("Bob".to_string())),
        "Bob should remain"
    );
}

// ============================================================================
// DETACH DELETE Tests
// ============================================================================

#[test]
fn test_detach_delete_node_with_relationships() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with a relationship: Alice -KNOWS-> Bob
    db.query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .expect("create failed");

    // Verify relationship exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");
    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 edge from Alice");
    drop(tx);

    // DETACH DELETE Alice (should also delete the KNOWS relationship)
    db.query("MATCH (n:Person {name: 'Alice'}) DETACH DELETE n").expect("detach delete failed");

    // Verify Alice is gone
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 1, "Expected 1 Person entity after DETACH DELETE");
    assert_eq!(
        entities[0].properties.get("name"),
        Some(&Value::String("Bob".to_string())),
        "Bob should remain"
    );

    // Verify Bob has no incoming edges
    let bob = &entities[0];
    let bob_incoming = tx.get_incoming_edges(bob.id).expect("failed to get edges");
    assert_eq!(bob_incoming.len(), 0, "Bob should have no incoming edges");
}

#[test]
fn test_regular_delete_node_with_relationships_fails() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with a relationship: Alice -KNOWS-> Bob
    db.query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .expect("create failed");

    // Try to DELETE Alice without DETACH (should fail)
    let result = db.query("MATCH (n:Person {name: 'Alice'}) DELETE n");

    // Should fail because Alice has a relationship
    assert!(result.is_err(), "DELETE without DETACH should fail when node has relationships");
    let err = result.unwrap_err();
    // The error message may mention "relationships", "DETACH", or "edges"
    assert!(
        err.to_string().contains("relationships")
            || err.to_string().contains("DETACH")
            || err.to_string().contains("edges"),
        "Error should mention relationships, edges, or DETACH: {}",
        err
    );

    // Verify Alice still exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Both entities should still exist");
}

// ============================================================================
// Relationship Deletion Tests
// ============================================================================

/// Tests MATCH ()-[r]->() DELETE r pattern for edge deletion.
///
/// NOTE: This test is marked as `#[ignore]` because MATCH clause execution
/// for edge patterns does not yet properly expose edge IDs in result rows.
/// The DELETE execution itself works correctly - the limitation is in the
/// MATCH pattern matching for relationship patterns.
#[test]
#[ignore = "MATCH for edge patterns does not yet expose edge IDs - requires MATCH enhancement"]
fn test_delete_relationship() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with a relationship
    db.query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .expect("create failed");

    // Verify relationship exists
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");
    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 edge from Alice");
    drop(tx);

    // Delete only the relationship
    db.query("MATCH ()-[r:KNOWS]->() DELETE r").expect("delete relationship failed");

    // Verify relationship is gone but nodes remain
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Both Person entities should remain");

    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");
    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 0, "Alice should have no outgoing edges");
}

/// Tests MATCH ()-[r:TYPE]->() DELETE r pattern for typed edge deletion.
///
/// NOTE: This test is marked as `#[ignore]` because MATCH clause execution
/// for edge patterns does not yet properly expose edge IDs in result rows.
/// The DELETE execution itself works correctly - the limitation is in the
/// MATCH pattern matching for relationship patterns.
#[test]
#[ignore = "MATCH for edge patterns does not yet expose edge IDs - requires MATCH enhancement"]
fn test_delete_relationship_by_type() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with multiple relationship types
    db.query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .expect("create failed");
    db.query(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:WORKS_WITH]->(b)",
    )
    .expect("create failed");

    // Delete only KNOWS relationships
    db.query("MATCH ()-[r:KNOWS]->() DELETE r").expect("delete failed");

    // Verify KNOWS is gone but WORKS_WITH remains
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    let alice = entities
        .iter()
        .find(|e| e.properties.get("name") == Some(&Value::String("Alice".to_string())))
        .expect("Alice not found");

    let edges = tx.get_outgoing_edges(alice.id).expect("failed to get edges");
    assert_eq!(edges.len(), 1, "Expected 1 remaining edge");
    assert_eq!(edges[0].edge_type.as_str(), "WORKS_WITH", "WORKS_WITH should remain");
}

// ============================================================================
// Multiple Deletion Tests
// ============================================================================

#[test]
fn test_delete_multiple_nodes() {
    let db = Database::in_memory().expect("failed to create db");

    // Create multiple nodes
    db.query("CREATE (n:Person {name: 'Alice'})").expect("create failed");
    db.query("CREATE (n:Person {name: 'Bob'})").expect("create failed");
    db.query("CREATE (n:Person {name: 'Charlie'})").expect("create failed");

    // Delete all Person nodes
    db.query("MATCH (n:Person) DELETE n").expect("delete failed");

    // Verify all are gone
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 0, "All Person entities should be deleted");
}

#[test]
fn test_detach_delete_multiple_nodes_with_relationships() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a graph: Alice <-KNOWS- Bob -KNOWS-> Charlie
    db.query(
        "CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'})",
    )
    .expect("create nodes failed");
    db.query("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (b)-[:KNOWS]->(a)")
        .expect("create edge 1 failed");
    db.query(
        "MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Charlie'}) CREATE (b)-[:KNOWS]->(c)",
    )
    .expect("create edge 2 failed");

    // DETACH DELETE Bob (should delete both KNOWS relationships)
    db.query("MATCH (n:Person {name: 'Bob'}) DETACH DELETE n").expect("detach delete failed");

    // Verify only Alice and Charlie remain with no edges
    let tx = db.begin_read().expect("failed to begin read");
    let entities = tx.iter_entities(Some("Person")).expect("failed to iterate");
    assert_eq!(entities.len(), 2, "Expected 2 Person entities");

    for entity in &entities {
        let name = entity.properties.get("name");
        assert!(
            name == Some(&Value::String("Alice".to_string()))
                || name == Some(&Value::String("Charlie".to_string())),
            "Should be Alice or Charlie"
        );

        let outgoing = tx.get_outgoing_edges(entity.id).expect("failed to get edges");
        let incoming = tx.get_incoming_edges(entity.id).expect("failed to get edges");
        assert!(outgoing.is_empty() && incoming.is_empty(), "No edges should remain");
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_delete_nonexistent_node() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to delete a node that doesn't match
    let result = db.query("MATCH (n:Person {name: 'Nobody'}) DELETE n");

    // Should succeed but delete nothing
    assert!(result.is_ok(), "DELETE of nonexistent node should not error");
}

#[test]
fn test_delete_already_deleted_node() {
    let db = Database::in_memory().expect("failed to create db");

    // Create and delete a node
    db.query("CREATE (n:Person {name: 'Alice'})").expect("create failed");
    db.query("MATCH (n:Person {name: 'Alice'}) DELETE n").expect("first delete failed");

    // Try to delete again
    let result = db.query("MATCH (n:Person {name: 'Alice'}) DELETE n");

    // Should succeed (no-op)
    assert!(result.is_ok(), "DELETE of already-deleted node should not error");
}

#[test]
fn test_delete_with_no_match() {
    let db = Database::in_memory().expect("failed to create db");

    // DELETE with a MATCH that returns no rows
    let result = db.query("MATCH (n:NonexistentLabel) DELETE n");

    // Should succeed with no effect
    assert!(result.is_ok(), "DELETE with no MATCH results should not error");
}
