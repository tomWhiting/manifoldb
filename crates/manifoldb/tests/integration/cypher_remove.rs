//! Tests for Cypher REMOVE execution.
//!
//! Tests cover:
//! - Removing a single property from a node
//! - Removing multiple properties from a node
//! - Removing a label from a node
//! - Combined property and label removal

use manifoldb::Database;

/// Test removing a single property from a node.
#[test]
fn test_remove_single_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with properties
    db.query("CREATE (n:Person {name: 'Alice', age: 30, city: 'NYC'}) RETURN n")
        .expect("create alice");

    // Remove a single property
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) REMOVE n.age RETURN n")
        .expect("remove age failed");

    assert_eq!(result.len(), 1, "Expected 1 row for REMOVE operation");

    // Verify the property was removed by querying
    // The age property should no longer exist
    let check = db
        .query("MATCH (n:Person {name: 'Alice'}) RETURN n.name, n.city")
        .expect("check query failed");
    assert_eq!(check.len(), 1, "Expected 1 Person named Alice");
}

/// Test removing multiple properties from a node.
#[test]
fn test_remove_multiple_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with properties
    db.query(
        "CREATE (n:Person {name: 'Bob', age: 25, city: 'LA', email: 'bob@example.com'}) RETURN n",
    )
    .expect("create bob");

    // Remove multiple properties
    let result = db
        .query("MATCH (n:Person {name: 'Bob'}) REMOVE n.age, n.city RETURN n")
        .expect("remove multiple failed");

    assert_eq!(result.len(), 1, "Expected 1 row for REMOVE operation");

    // Verify the properties were removed
    let check = db
        .query("MATCH (n:Person {name: 'Bob'}) RETURN n.name, n.email")
        .expect("check query failed");
    assert_eq!(check.len(), 1, "Expected 1 Person named Bob");
}

/// Test removing a label from a node.
#[test]
fn test_remove_label() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with multiple labels
    db.query("CREATE (n:Person:Employee {name: 'Carol'}) RETURN n").expect("create carol");

    // Remove a label
    let result = db
        .query("MATCH (n:Person {name: 'Carol'}) REMOVE n:Employee RETURN n")
        .expect("remove label failed");

    assert_eq!(result.len(), 1, "Expected 1 row for REMOVE operation");

    // Verify the label was removed - the node should still match :Person but not :Employee
    let as_person =
        db.query("MATCH (n:Person {name: 'Carol'}) RETURN n").expect("check Person query failed");
    assert_eq!(as_person.len(), 1, "Expected node to still be a Person");

    // Querying for :Employee should return 0 results
    let as_employee = db
        .query("MATCH (n:Employee {name: 'Carol'}) RETURN n")
        .expect("check Employee query failed");
    assert_eq!(as_employee.len(), 0, "Expected node to no longer be an Employee");
}

/// Test removing property from non-matching node (should be a no-op).
#[test]
fn test_remove_property_no_match() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Dave'}) RETURN n").expect("create dave");

    // Try to remove from a non-matching node - should return 0 rows (no error)
    let result = db
        .query("MATCH (n:Person {name: 'NonExistent'}) REMOVE n.age RETURN n")
        .expect("remove should not fail");

    assert_eq!(result.len(), 0, "Expected 0 rows when no nodes match");

    // Verify original node is unchanged
    let check =
        db.query("MATCH (n:Person {name: 'Dave'}) RETURN n.name").expect("check query failed");
    assert_eq!(check.len(), 1, "Expected Dave to still exist");
}

/// Test removing a property that doesn't exist (should be a no-op).
#[test]
fn test_remove_nonexistent_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Eve'}) RETURN n").expect("create eve");

    // Remove a property that doesn't exist - should not error
    let result = db
        .query("MATCH (n:Person {name: 'Eve'}) REMOVE n.age RETURN n")
        .expect("remove nonexistent property should not fail");

    assert_eq!(result.len(), 1, "Expected 1 row for REMOVE operation");

    // Verify node is unchanged
    let check =
        db.query("MATCH (n:Person {name: 'Eve'}) RETURN n.name").expect("check query failed");
    assert_eq!(check.len(), 1, "Expected Eve to still exist");
}
