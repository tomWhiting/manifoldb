//! Tests for Cypher SET operations.
//!
//! Tests property updates and label additions on nodes and relationships.

use manifoldb::Database;

/// Test SET single property on a node.
#[test]
fn test_set_single_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with initial properties
    db.query("CREATE (n:Person {name: 'Alice', age: 30}) RETURN n").expect("create alice");

    // Update the age property
    let result =
        db.query("MATCH (n:Person {name: 'Alice'}) SET n.age = 31 RETURN n").expect("set failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the property was updated
    let verify = db.query("MATCH (n:Person {name: 'Alice'}) RETURN n.age").expect("verify failed");

    assert_eq!(verify.len(), 1, "Expected 1 row in verify, got {}", verify.len());
}

/// Test SET multiple properties on a node.
#[test]
fn test_set_multiple_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");

    // Set multiple properties
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) SET n.age = 31, n.city = 'Seattle' RETURN n")
        .expect("set failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}

/// Test SET adding a label.
#[test]
fn test_set_add_label() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");

    // Add a label
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) SET n:Employee RETURN n")
        .expect("set label failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());

    // Verify the label was added by matching on both labels
    let verify = db.query("MATCH (n:Employee {name: 'Alice'}) RETURN n").expect("verify failed");

    assert_eq!(verify.len(), 1, "Expected 1 Employee named Alice, got {}", verify.len());
}

/// Test SET on relationship properties.
///
/// Sets a property on a relationship using MATCH pattern matching.
#[test]
fn test_set_relationship_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes and relationship
    db.query("CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})").expect("create nodes");
    db.query(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS]->(b) RETURN r",
    )
    .expect("create edge");

    // Set relationship property
    let result = db
        .query("MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->() SET r.since = 2020 RETURN r")
        .expect("set rel property failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}

/// Test SET property to null (effectively removing it).
#[test]
fn test_set_property_to_null() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node with a property
    db.query("CREATE (n:Person {name: 'Alice', temporary: 'value'}) RETURN n")
        .expect("create alice");

    // Set property to null
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) SET n.temporary = null RETURN n")
        .expect("set null failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}

/// Test SET updates multiple nodes.
#[test]
fn test_set_multiple_nodes() {
    let db = Database::in_memory().expect("failed to create db");

    // Create multiple nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");
    db.query("CREATE (n:Person {name: 'Charlie'}) RETURN n").expect("create charlie");

    // Set property on all Person nodes
    let result = db.query("MATCH (n:Person) SET n.active = true RETURN n").expect("set all failed");

    assert_eq!(result.len(), 3, "Expected 3 rows, got {}", result.len());
}

/// Test SET with expression value.
#[test]
fn test_set_property_with_expression() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Alice', age: 30}) RETURN n").expect("create alice");

    // Set property using an expression
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) SET n.age = n.age + 1 RETURN n")
        .expect("set expr failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}

/// Test SET property and label together.
#[test]
fn test_set_property_and_label() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a node
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");

    // Set property and add label
    let result = db
        .query("MATCH (n:Person {name: 'Alice'}) SET n.employed = true, n:Employee RETURN n")
        .expect("set both failed");

    assert_eq!(result.len(), 1, "Expected 1 row, got {}", result.len());
}
