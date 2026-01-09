//! Tests for MATCH clause with property filtering.

use manifoldb::Database;

/// Test simple MATCH with single node pattern and label.
#[test]
fn test_match_single_node_label_only() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");

    // Test simple MATCH - should return 2
    let result = db.query("MATCH (n:Person) RETURN n").expect("match failed");
    assert_eq!(result.len(), 2, "Expected 2 Person nodes, got {}", result.len());
}

/// Test MATCH with single node pattern and property filter.
#[test]
fn test_match_single_node_with_property() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");

    // Test MATCH with property filter - should return 1
    let result = db.query("MATCH (n:Person {name: 'Alice'}) RETURN n").expect("match failed");
    assert_eq!(result.len(), 1, "Expected 1 Person named Alice, got {}", result.len());
}

/// Test MATCH with two standalone node patterns.
#[test]
fn test_match_two_node_patterns() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");

    // Test MATCH with two patterns - should return cross product (2x2=4)
    let result = db.query("MATCH (a:Person), (b:Person) RETURN a, b").expect("match failed");
    assert_eq!(result.len(), 4, "Expected 4 rows (cross product), got {}", result.len());
}

/// Test MATCH with two node patterns and property filters.
#[test]
fn test_match_two_node_patterns_with_properties() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");

    // Test MATCH with two patterns and property filters - should return 1
    let result = db
        .query("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) RETURN a, b")
        .expect("match failed");
    assert_eq!(result.len(), 1, "Expected 1 row (Alice + Bob), got {}", result.len());
}

/// Test MATCH + CREATE - creating a relationship between matched nodes.
#[test]
fn test_match_then_create_edge() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes
    db.query("CREATE (n:Person {name: 'Alice'}) RETURN n").expect("create alice");
    db.query("CREATE (n:Person {name: 'Bob'}) RETURN n").expect("create bob");

    // Match and create relationship
    let result = db
        .query("MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS]->(b) RETURN r")
        .expect("match+create failed");

    assert_eq!(result.len(), 1, "Expected 1 row for relationship creation, got {}", result.len());
}
