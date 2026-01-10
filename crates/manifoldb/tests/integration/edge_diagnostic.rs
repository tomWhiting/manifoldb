//! Tests for edge variable binding in MATCH patterns.
//!
//! These tests verify that edge variables (e.g., `r` in `(a)-[r:KNOWS]->(b)`)
//! are properly bound in result rows and can be used in SET and DELETE operations.

use manifoldb::Database;

#[test]
fn test_edge_variable_diagnostic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes and relationship
    println!("\n=== Creating nodes and relationship ===");
    db.query("CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})").expect("create nodes");

    // Test 0: Can we even match nodes?
    println!("\n=== Test 0: Basic node matching ===");
    let result = db.query("MATCH (a:Person) RETURN a").expect("match nodes failed");
    println!("MATCH (a:Person) returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    let result = db.query(
        "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[r:KNOWS]->(b) RETURN r",
    ).expect("create edge");
    println!("\nCREATE returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Test 0.5: Does inline CREATE with edge work?
    println!("\n=== Test 0.5: Inline CREATE with edge ===");
    db.query("CREATE (c:Person {name: 'Charlie'})-[:FRIENDS]->(d:Person {name: 'Diana'})")
        .expect("inline create");
    let result = db.query("MATCH (c:Person {name: 'Charlie'}) RETURN c").expect("match charlie");
    println!("MATCH Charlie returned {} rows", result.len());

    // Test 1: Basic MATCH with edge variable - does it return rows?
    println!("\n=== Test 1: MATCH with edge variable ===");
    let result =
        db.query("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b").expect("match failed");
    println!("MATCH returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Test 2: MATCH without edge variable
    println!("\n=== Test 2: MATCH without edge variable ===");
    let result =
        db.query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b").expect("match failed");
    println!("MATCH returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Test 3: Just source node with filter
    println!("\n=== Test 3: Match Alice specifically then expand ===");
    let result = db
        .query("MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b) RETURN a, r, b")
        .expect("match failed");
    println!("MATCH returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Test 4: Simple match without filters (needs at least one label to work)
    println!("\n=== Test 4: Simple MATCH without filters but with label ===");
    let result = db.query("MATCH (a:Person)-[r]->(b) RETURN a, r, b").expect("match failed");
    println!("MATCH returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }

    // Assert we got rows with the edge variable
    assert!(!result.is_empty(), "Should have at least one row");

    // Verify edge variable is properly bound (not null)
    let first_row = result.iter().next().unwrap();
    println!("First row values: {:?}", first_row);
    // r should be the second value (index 1) and should be an Int
    let edge_id = first_row.get(1).expect("should have edge column");
    println!("Edge ID: {:?}", edge_id);
    assert!(matches!(edge_id, manifoldb::Value::Int(_)), "Edge should be an Int ID");

    // Test 5: DELETE pattern simulation - check if edge is bound correctly
    println!("\n=== Test 5: DELETE pattern simulation ===");
    let result =
        db.query("MATCH (a:Person)-[r:KNOWS]->() RETURN a, r").expect("match for delete failed");
    println!("MATCH for DELETE returned {} rows", result.len());
    for (i, row) in result.iter().enumerate() {
        println!("  Row {}: {:?}", i, row);
    }
    assert!(!result.is_empty(), "Should have at least one row for DELETE");
}

#[test]
fn test_delete_relationship_diagnostic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create nodes with a relationship
    db.query("CREATE (a:Person {name: 'Alice'})-[:KNOWS]->(b:Person {name: 'Bob'})")
        .expect("create failed");

    // Try DELETE
    println!("\n=== Running DELETE ===");
    let result = db.query("MATCH (a:Person)-[r:KNOWS]->() DELETE r");
    match result {
        Ok(r) => println!("DELETE succeeded, {} rows returned", r.len()),
        Err(e) => {
            println!("DELETE failed with error: {}", e);
            panic!("DELETE should succeed: {}", e);
        }
    }
}
