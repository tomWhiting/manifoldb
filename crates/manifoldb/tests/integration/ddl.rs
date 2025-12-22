//! DDL (Data Definition Language) integration tests.
//!
//! Tests for CREATE TABLE, DROP TABLE, CREATE INDEX, and DROP INDEX.

#![allow(dead_code, unused_variables)]

use manifoldb::Database;

// ============================================================================
// CREATE TABLE Tests
// ============================================================================

#[test]
fn test_create_table_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a simple table
    let affected = db
        .execute("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT, email TEXT)")
        .expect("create table failed");

    // DDL returns 0 affected rows
    assert_eq!(affected, 0);

    // Insert some data to verify the table works
    let inserted = db
        .execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
        .expect("insert failed");
    assert_eq!(inserted, 1);

    // Query the data
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_create_table_if_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Create the table
    db.execute("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create table failed");

    // Creating again without IF NOT EXISTS would be an error,
    // but with IF NOT EXISTS it should succeed
    let affected = db
        .execute("CREATE TABLE IF NOT EXISTS users (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create table if not exists failed");

    assert_eq!(affected, 0);
}

#[test]
fn test_create_table_with_constraints() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a table with various column constraints
    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            sku TEXT UNIQUE,
            price NUMERIC(10, 2),
            created_at TIMESTAMP
        )",
    )
    .expect("create table with constraints failed");

    // Insert data
    db.execute(
        "INSERT INTO products (id, name, sku, price) VALUES (1, 'Widget', 'WDG-001', 29.99)",
    )
    .expect("insert failed");

    let result = db.query("SELECT * FROM products").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_create_table_with_table_constraints() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a table with table-level constraints
    db.execute(
        "CREATE TABLE order_items (
            order_id BIGINT,
            product_id BIGINT,
            quantity INTEGER,
            PRIMARY KEY (order_id, product_id)
        )",
    )
    .expect("create table with table constraints failed");

    // Insert data
    db.execute("INSERT INTO order_items (order_id, product_id, quantity) VALUES (1, 100, 5)")
        .expect("insert failed");

    let result = db.query("SELECT * FROM order_items").expect("query failed");
    assert_eq!(result.len(), 1);
}

// ============================================================================
// DROP TABLE Tests
// ============================================================================

#[test]
fn test_drop_table_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and insert data
    db.execute("CREATE TABLE temp_users (id BIGINT, name TEXT)").expect("create table failed");
    db.execute("INSERT INTO temp_users (id, name) VALUES (1, 'Alice')").expect("insert failed");

    // Verify data exists
    let result = db.query("SELECT * FROM temp_users").expect("query failed");
    assert_eq!(result.len(), 1);

    // Drop the table
    let affected = db.execute("DROP TABLE temp_users").expect("drop table failed");
    assert_eq!(affected, 0);

    // Query should now return empty (no data in that table)
    let result = db.query("SELECT * FROM temp_users").expect("query failed");
    assert_eq!(result.len(), 0);
}

#[test]
fn test_drop_table_if_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Drop a table that doesn't exist - should fail without IF EXISTS
    // With IF EXISTS, it should succeed silently
    let affected =
        db.execute("DROP TABLE IF EXISTS nonexistent_table").expect("drop if exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_drop_table_cascade() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("CREATE TABLE cascade_test (id BIGINT, value TEXT)").expect("create table failed");
    db.execute("INSERT INTO cascade_test (id, value) VALUES (1, 'test')").expect("insert failed");

    // Drop with CASCADE
    let affected = db.execute("DROP TABLE cascade_test CASCADE").expect("drop cascade failed");
    assert_eq!(affected, 0);
}

// ============================================================================
// CREATE INDEX Tests
// ============================================================================

#[test]
fn test_create_index_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table first
    db.execute("CREATE TABLE indexed_users (id BIGINT, email TEXT, name TEXT)")
        .expect("create table failed");

    // Create an index
    let affected = db
        .execute("CREATE INDEX idx_users_email ON indexed_users (email)")
        .expect("create index failed");

    assert_eq!(affected, 0);
}

#[test]
fn test_create_unique_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table first
    db.execute("CREATE TABLE unique_test (id BIGINT, sku TEXT)").expect("create table failed");

    // Create a unique index
    let affected = db
        .execute("CREATE UNIQUE INDEX idx_unique_sku ON unique_test (sku)")
        .expect("create unique index failed");

    assert_eq!(affected, 0);
}

#[test]
fn test_create_index_if_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE idx_test (id BIGINT, value TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_value ON idx_test (value)").expect("create index failed");

    // Creating again with IF NOT EXISTS should succeed
    let affected = db
        .execute("CREATE INDEX IF NOT EXISTS idx_value ON idx_test (value)")
        .expect("create index if not exists failed");

    assert_eq!(affected, 0);
}

#[test]
fn test_create_index_with_method() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table first
    db.execute("CREATE TABLE method_test (id BIGINT, embedding VECTOR(128))")
        .expect("create table failed");

    // Create index with USING clause (for vector search)
    let affected = db
        .execute("CREATE INDEX idx_embedding ON method_test USING hnsw (embedding)")
        .expect("create index with method failed");

    assert_eq!(affected, 0);
}

// ============================================================================
// DROP INDEX Tests
// ============================================================================

#[test]
fn test_drop_index_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE drop_idx_test (id BIGINT, value TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_drop_test ON drop_idx_test (value)").expect("create index failed");

    // Drop the index
    let affected = db.execute("DROP INDEX idx_drop_test").expect("drop index failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_drop_index_if_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Drop an index that doesn't exist - should succeed with IF EXISTS
    let affected =
        db.execute("DROP INDEX IF EXISTS nonexistent_index").expect("drop index if exists failed");
    assert_eq!(affected, 0);
}

// ============================================================================
// Combined DDL/DML Tests
// ============================================================================

#[test]
fn test_ddl_then_dml_workflow() {
    let db = Database::in_memory().expect("failed to create db");

    // 1. Create table
    db.execute(
        "CREATE TABLE orders (
            id BIGINT PRIMARY KEY,
            customer_name TEXT,
            total NUMERIC(10, 2),
            status TEXT
        )",
    )
    .expect("create table failed");

    // 2. Create indexes
    db.execute("CREATE INDEX idx_orders_customer ON orders (customer_name)")
        .expect("create index 1 failed");
    db.execute("CREATE INDEX idx_orders_status ON orders (status)").expect("create index 2 failed");

    // 3. Insert data
    db.execute("INSERT INTO orders (id, customer_name, total, status) VALUES (1, 'Alice', 100.50, 'pending')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO orders (id, customer_name, total, status) VALUES (2, 'Bob', 75.00, 'completed')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO orders (id, customer_name, total, status) VALUES (3, 'Alice', 50.25, 'pending')")
        .expect("insert 3 failed");

    // 4. Query data
    let result =
        db.query("SELECT * FROM orders WHERE customer_name = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 2);

    // 5. Update data
    let updated =
        db.execute("UPDATE orders SET status = 'completed' WHERE id = 1").expect("update failed");
    assert_eq!(updated, 1);

    // 6. Drop an index
    db.execute("DROP INDEX idx_orders_status").expect("drop index failed");

    // 7. Delete some data
    let deleted = db.execute("DELETE FROM orders WHERE id = 3").expect("delete failed");
    assert_eq!(deleted, 1);

    // 8. Final query
    let result = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(result.len(), 2);

    // 9. Drop the table
    db.execute("DROP TABLE orders CASCADE").expect("drop table failed");
}

#[test]
fn test_table_schema_persists() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with schema
    db.execute("CREATE TABLE schema_test (id BIGINT, name TEXT, active BOOLEAN)")
        .expect("create table failed");

    // Insert and query to verify schema works
    db.execute("INSERT INTO schema_test (id, name, active) VALUES (1, 'Test', true)")
        .expect("insert failed");

    let result = db.query("SELECT id, name, active FROM schema_test").expect("query failed");
    assert_eq!(result.len(), 1);
    assert_eq!(result.columns().len(), 3);
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_create_table_already_exists_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE duplicate_table (id BIGINT)").expect("create table failed");

    // Try to create again without IF NOT EXISTS - should fail
    let result = db.execute("CREATE TABLE duplicate_table (id BIGINT)");
    assert!(result.is_err());
}

#[test]
fn test_drop_nonexistent_table_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to drop a table that doesn't exist without IF EXISTS
    let result = db.execute("DROP TABLE nonexistent_table");
    assert!(result.is_err());
}

#[test]
fn test_create_index_nonexistent_table_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to create index on a table that doesn't exist
    let result = db.execute("CREATE INDEX idx_test ON nonexistent_table (col)");
    assert!(result.is_err());
}

#[test]
fn test_create_index_already_exists_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE idx_dup_test (id BIGINT, val TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_dup ON idx_dup_test (val)").expect("create index failed");

    // Try to create again without IF NOT EXISTS
    let result = db.execute("CREATE INDEX idx_dup ON idx_dup_test (val)");
    assert!(result.is_err());
}

#[test]
fn test_drop_nonexistent_index_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to drop an index that doesn't exist without IF EXISTS
    let result = db.execute("DROP INDEX nonexistent_index");
    assert!(result.is_err());
}
