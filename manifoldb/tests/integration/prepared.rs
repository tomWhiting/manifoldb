//! Integration tests for prepared statements.

use manifoldb::{Database, Value};

/// Test basic prepared statement functionality for SELECT queries.
#[test]
fn test_prepared_select() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert some test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Prepare a statement
    let stmt = db.prepare("SELECT * FROM users WHERE age > $1").expect("prepare failed");
    assert!(stmt.is_query());
    assert!(!stmt.is_dml());
    assert!(!stmt.is_ddl());
    assert!(stmt.accessed_tables().contains("users"));

    // Execute with different parameters
    let result1 = db.query_prepared(&stmt, &[Value::Int(28)]).expect("query failed");
    assert_eq!(result1.len(), 2); // Alice (30) and Charlie (35)

    let result2 = db.query_prepared(&stmt, &[Value::Int(32)]).expect("query failed");
    assert_eq!(result2.len(), 1); // Only Charlie (35)

    let result3 = db.query_prepared(&stmt, &[Value::Int(40)]).expect("query failed");
    assert_eq!(result3.len(), 0); // No one
}

/// Test prepared INSERT statement.
#[test]
fn test_prepared_insert() {
    let db = Database::in_memory().expect("failed to create db");

    // Prepare an INSERT statement
    let stmt =
        db.prepare("INSERT INTO products (name, price) VALUES ($1, $2)").expect("prepare failed");
    assert!(stmt.is_dml());
    assert!(!stmt.is_query());
    assert!(stmt.accessed_tables().contains("products"));

    // Execute multiple times with different values
    db.execute_prepared(&stmt, &[Value::String("Apple".to_string()), Value::Float(1.99)])
        .expect("insert failed");
    db.execute_prepared(&stmt, &[Value::String("Banana".to_string()), Value::Float(0.99)])
        .expect("insert failed");
    db.execute_prepared(&stmt, &[Value::String("Cherry".to_string()), Value::Float(3.99)])
        .expect("insert failed");

    // Verify the inserts
    let result = db.query("SELECT * FROM products").expect("query failed");
    assert_eq!(result.len(), 3);
}

/// Test prepared UPDATE statement.
#[test]
fn test_prepared_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO items (name, quantity) VALUES ('Item1', 10)").expect("insert failed");
    db.execute("INSERT INTO items (name, quantity) VALUES ('Item2', 20)").expect("insert failed");

    // Prepare an UPDATE statement
    let stmt =
        db.prepare("UPDATE items SET quantity = $1 WHERE name = $2").expect("prepare failed");
    assert!(stmt.is_dml());

    // Execute the update
    let count = db
        .execute_prepared(&stmt, &[Value::Int(50), Value::String("Item1".to_string())])
        .expect("update failed");
    assert_eq!(count, 1);

    // Verify the update
    let result = db
        .query_with_params(
            "SELECT quantity FROM items WHERE name = $1",
            &[Value::String("Item1".to_string())],
        )
        .expect("query failed");
    assert_eq!(result.len(), 1);
    assert_eq!(result.first().unwrap().get(0), Some(&Value::Int(50)));
}

/// Test prepared DELETE statement.
#[test]
fn test_prepared_delete() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO records (id, status) VALUES (1, 'active')").expect("insert failed");
    db.execute("INSERT INTO records (id, status) VALUES (2, 'inactive')").expect("insert failed");
    db.execute("INSERT INTO records (id, status) VALUES (3, 'active')").expect("insert failed");

    // Prepare a DELETE statement
    let stmt = db.prepare("DELETE FROM records WHERE status = $1").expect("prepare failed");
    assert!(stmt.is_dml());

    // Execute the delete
    let count = db
        .execute_prepared(&stmt, &[Value::String("inactive".to_string())])
        .expect("delete failed");
    assert_eq!(count, 1);

    // Verify the delete
    let result = db.query("SELECT * FROM records").expect("query failed");
    assert_eq!(result.len(), 2);
}

/// Test that prepared statements are invalidated after schema changes.
#[test]
fn test_prepared_statement_invalidation() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert some data
    db.execute("INSERT INTO test_table (value) VALUES (1)").expect("insert failed");

    // Prepare a statement
    let stmt = db.prepare("SELECT * FROM test_table").expect("prepare failed");

    // Execute once - should work
    let result1 = db.query_prepared(&stmt, &[]).expect("query failed");
    assert_eq!(result1.len(), 1);

    // Execute DDL to change schema
    db.execute("CREATE TABLE another_table (id INTEGER)").expect("create failed");

    // The statement should now be invalid
    let result2 = db.query_prepared(&stmt, &[]);
    assert!(result2.is_err());

    // Re-prepare should work
    let stmt2 = db.prepare("SELECT * FROM test_table").expect("re-prepare failed");
    let result3 = db.query_prepared(&stmt2, &[]).expect("query failed");
    assert_eq!(result3.len(), 1);
}

/// Test prepared statement cache.
#[test]
fn test_prepared_statement_cache() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert some data
    db.execute("INSERT INTO cache_test (x) VALUES (1)").expect("insert failed");

    // Use prepare_cached multiple times for the same query
    let stmt1 = db.prepare_cached("SELECT * FROM cache_test WHERE x = $1").expect("prepare failed");
    let stmt2 = db.prepare_cached("SELECT * FROM cache_test WHERE x = $1").expect("prepare failed");

    // Should be the same Arc (cache hit)
    assert!(std::sync::Arc::ptr_eq(&stmt1, &stmt2));

    // Cache metrics should reflect this
    let cache = db.prepared_cache();
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.misses(), 1);
}

/// Test that DDL statements update schema version properly.
#[test]
fn test_schema_version_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Get initial schema version
    let initial_version = db.prepared_cache().schema_version();

    // Execute DDL
    db.execute("CREATE TABLE version_test (id INTEGER)").expect("create failed");

    // Schema version should have increased
    let new_version = db.prepared_cache().schema_version();
    assert!(new_version > initial_version);

    // Execute another DDL
    db.execute("DROP TABLE version_test").expect("drop failed");

    // Schema version should have increased again
    let final_version = db.prepared_cache().schema_version();
    assert!(final_version > new_version);
}

/// Test prepared statement with multiple parameters.
#[test]
fn test_prepared_multiple_params() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO orders (customer, product, quantity) VALUES ('Alice', 'Widget', 5)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (customer, product, quantity) VALUES ('Alice', 'Gadget', 3)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (customer, product, quantity) VALUES ('Bob', 'Widget', 10)")
        .expect("insert failed");

    // Prepare with multiple parameters
    let stmt = db
        .prepare("SELECT * FROM orders WHERE customer = $1 AND quantity >= $2")
        .expect("prepare failed");

    // Execute with different parameter combinations
    let result1 = db
        .query_prepared(&stmt, &[Value::String("Alice".to_string()), Value::Int(4)])
        .expect("query failed");
    assert_eq!(result1.len(), 1); // Only Widget order (quantity 5)

    let result2 = db
        .query_prepared(&stmt, &[Value::String("Bob".to_string()), Value::Int(5)])
        .expect("query failed");
    assert_eq!(result2.len(), 1); // Bob's Widget order (quantity 10)
}

/// Test clearing the prepared statement cache.
#[test]
fn test_clear_prepared_cache() {
    let db = Database::in_memory().expect("failed to create db");

    // Prepare some statements
    let _ = db.prepare_cached("SELECT 1").expect("prepare failed");
    let _ = db.prepare_cached("SELECT 2").expect("prepare failed");

    assert_eq!(db.prepared_cache().len(), 2);

    // Clear the cache
    db.clear_prepared_cache();

    assert!(db.prepared_cache().is_empty());
}

/// Test table-specific invalidation in prepared cache.
#[test]
fn test_table_invalidation() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert data into multiple tables
    db.execute("INSERT INTO table_a (x) VALUES (1)").expect("insert failed");
    db.execute("INSERT INTO table_b (y) VALUES (2)").expect("insert failed");

    // Prepare statements for different tables
    let _ = db.prepare_cached("SELECT * FROM table_a").expect("prepare failed");
    let _ = db.prepare_cached("SELECT * FROM table_b").expect("prepare failed");

    assert_eq!(db.prepared_cache().len(), 2);

    // Invalidate only table_a
    db.prepared_cache().invalidate_tables(&["table_a".to_string()]);

    // Should only have table_b left
    assert_eq!(db.prepared_cache().len(), 1);

    // table_b query should still hit cache
    let _ = db.prepare_cached("SELECT * FROM table_b").expect("prepare failed");
    assert_eq!(db.prepared_cache().hits(), 1);
}
