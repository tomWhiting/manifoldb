//! Index-based query execution tests.
//!
//! Tests that verify queries use indexes when available and return correct results.

#![allow(dead_code, unused_variables)]

use manifoldb::Database;

// ============================================================================
// Point Lookup Tests (Equality Predicates)
// ============================================================================

#[test]
fn test_index_point_lookup_string() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE users (id BIGINT, name TEXT, email TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_users_name ON users (name)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO users (id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO users (id, name, email) VALUES (3, 'Alice', 'alice2@example.com')")
        .expect("insert 3 failed");
    db.execute("INSERT INTO users (id, name, email) VALUES (4, 'Charlie', 'charlie@example.com')")
        .expect("insert 4 failed");

    // Query using the indexed column (should use index)
    let result = db.query("SELECT * FROM users WHERE name = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 Alices");

    let result = db.query("SELECT * FROM users WHERE name = 'Bob'").expect("query failed");
    assert_eq!(result.len(), 1, "Expected 1 Bob");

    let result = db.query("SELECT * FROM users WHERE name = 'NotExists'").expect("query failed");
    assert_eq!(result.len(), 0, "Expected no results for non-existent name");
}

#[test]
fn test_index_point_lookup_integer() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index on integer column
    db.execute("CREATE TABLE products (id BIGINT, category_id INTEGER, name TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_products_category ON products (category_id)")
        .expect("create index failed");

    // Insert data
    db.execute("INSERT INTO products (id, category_id, name) VALUES (1, 100, 'Product A')")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, category_id, name) VALUES (2, 200, 'Product B')")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, category_id, name) VALUES (3, 100, 'Product C')")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, category_id, name) VALUES (4, 300, 'Product D')")
        .expect("insert failed");

    // Query using the indexed column
    let result = db.query("SELECT * FROM products WHERE category_id = 100").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 products in category 100");

    let result = db.query("SELECT * FROM products WHERE category_id = 200").expect("query failed");
    assert_eq!(result.len(), 1, "Expected 1 product in category 200");
}

// ============================================================================
// Range Scan Tests
// ============================================================================

#[test]
fn test_index_range_scan_greater_than() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE orders (id BIGINT, amount INTEGER, status TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_orders_amount ON orders (amount)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO orders (id, amount, status) VALUES (1, 100, 'pending')")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, amount, status) VALUES (2, 250, 'shipped')")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, amount, status) VALUES (3, 500, 'pending')")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, amount, status) VALUES (4, 150, 'delivered')")
        .expect("insert failed");

    // Query with > predicate (should use index range scan)
    let result = db.query("SELECT * FROM orders WHERE amount > 200").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 orders > 200");

    let result = db.query("SELECT * FROM orders WHERE amount >= 250").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 orders >= 250");
}

#[test]
fn test_index_range_scan_less_than() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE scores (id BIGINT, score INTEGER, player TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_scores_score ON scores (score)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO scores (id, score, player) VALUES (1, 50, 'Player A')")
        .expect("insert failed");
    db.execute("INSERT INTO scores (id, score, player) VALUES (2, 100, 'Player B')")
        .expect("insert failed");
    db.execute("INSERT INTO scores (id, score, player) VALUES (3, 75, 'Player C')")
        .expect("insert failed");
    db.execute("INSERT INTO scores (id, score, player) VALUES (4, 25, 'Player D')")
        .expect("insert failed");

    // Query with < predicate (should use index range scan)
    let result = db.query("SELECT * FROM scores WHERE score < 75").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 scores < 75");

    let result = db.query("SELECT * FROM scores WHERE score <= 50").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 scores <= 50");
}

#[test]
fn test_index_range_scan_between() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE inventory (id BIGINT, quantity INTEGER, product TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_inventory_qty ON inventory (quantity)")
        .expect("create index failed");

    // Insert data
    db.execute("INSERT INTO inventory (id, quantity, product) VALUES (1, 5, 'Item A')")
        .expect("insert failed");
    db.execute("INSERT INTO inventory (id, quantity, product) VALUES (2, 15, 'Item B')")
        .expect("insert failed");
    db.execute("INSERT INTO inventory (id, quantity, product) VALUES (3, 25, 'Item C')")
        .expect("insert failed");
    db.execute("INSERT INTO inventory (id, quantity, product) VALUES (4, 35, 'Item D')")
        .expect("insert failed");
    db.execute("INSERT INTO inventory (id, quantity, product) VALUES (5, 10, 'Item E')")
        .expect("insert failed");

    // Query with BETWEEN predicate (should use index range scan)
    let result =
        db.query("SELECT * FROM inventory WHERE quantity BETWEEN 10 AND 30").expect("query failed");
    assert_eq!(result.len(), 3, "Expected 3 items with quantity between 10 and 30");
}

// ============================================================================
// Combined Predicate Tests
// ============================================================================

#[test]
fn test_index_with_residual_filter() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE employees (id BIGINT, department TEXT, status TEXT, salary INTEGER)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_emp_dept ON employees (department)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO employees (id, department, status, salary) VALUES (1, 'Engineering', 'active', 100000)")
        .expect("insert failed");
    db.execute("INSERT INTO employees (id, department, status, salary) VALUES (2, 'Engineering', 'inactive', 90000)")
        .expect("insert failed");
    db.execute("INSERT INTO employees (id, department, status, salary) VALUES (3, 'Sales', 'active', 80000)")
        .expect("insert failed");
    db.execute("INSERT INTO employees (id, department, status, salary) VALUES (4, 'Engineering', 'active', 120000)")
        .expect("insert failed");

    // Query with indexed column + residual filter
    // Should use index for department, then filter on status
    let result = db
        .query("SELECT * FROM employees WHERE department = 'Engineering' AND status = 'active'")
        .expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 active Engineering employees");
}

#[test]
fn test_index_query_with_projection() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE customers (id BIGINT, name TEXT, country TEXT, tier TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_customers_country ON customers (country)")
        .expect("create index failed");

    // Insert data
    db.execute(
        "INSERT INTO customers (id, name, country, tier) VALUES (1, 'Customer A', 'USA', 'gold')",
    )
    .expect("insert failed");
    db.execute(
        "INSERT INTO customers (id, name, country, tier) VALUES (2, 'Customer B', 'UK', 'silver')",
    )
    .expect("insert failed");
    db.execute(
        "INSERT INTO customers (id, name, country, tier) VALUES (3, 'Customer C', 'USA', 'bronze')",
    )
    .expect("insert failed");

    // Query with projection
    let result =
        db.query("SELECT name, tier FROM customers WHERE country = 'USA'").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 USA customers");
    assert_eq!(result.num_columns(), 2, "Expected 2 columns in projection");
}

// ============================================================================
// Index Selection Tests
// ============================================================================

#[test]
fn test_query_without_index_works() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table WITHOUT an index
    db.execute("CREATE TABLE logs (id BIGINT, level TEXT, message TEXT)")
        .expect("create table failed");

    // Insert data
    db.execute("INSERT INTO logs (id, level, message) VALUES (1, 'INFO', 'Message 1')")
        .expect("insert failed");
    db.execute("INSERT INTO logs (id, level, message) VALUES (2, 'ERROR', 'Message 2')")
        .expect("insert failed");

    // Query should work (full scan)
    let result = db.query("SELECT * FROM logs WHERE level = 'ERROR'").expect("query failed");
    assert_eq!(result.len(), 1, "Expected 1 ERROR log");
}

#[test]
fn test_large_table_with_index() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE large_table (id BIGINT, key_col TEXT, value_col INTEGER)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_large_key ON large_table (key_col)").expect("create index failed");

    // Insert many rows
    for i in 0..100 {
        db.execute(&format!(
            "INSERT INTO large_table (id, key_col, value_col) VALUES ({}, 'key_{:03}', {})",
            i,
            i % 10,
            i * 10
        ))
        .expect("insert failed");
    }

    // Query using index - should be efficient
    let result =
        db.query("SELECT * FROM large_table WHERE key_col = 'key_005'").expect("query failed");
    assert_eq!(result.len(), 10, "Expected 10 rows matching key_005");
}

// ============================================================================
// Empty Table Tests
// ============================================================================

#[test]
fn test_index_query_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE empty_table (id BIGINT, name TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_empty_name ON empty_table (name)").expect("create index failed");

    // Query on empty table should return empty result
    let result = db.query("SELECT * FROM empty_table WHERE name = 'test'").expect("query failed");
    assert_eq!(result.len(), 0, "Expected no results from empty table");
}

// ============================================================================
// String Range Scan Tests
// ============================================================================

#[test]
fn test_index_string_range_scan() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE items (id BIGINT, code TEXT, description TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_items_code ON items (code)").expect("create index failed");

    // Insert data with sorted codes
    db.execute("INSERT INTO items (id, code, description) VALUES (1, 'A001', 'Item A')")
        .expect("insert failed");
    db.execute("INSERT INTO items (id, code, description) VALUES (2, 'B001', 'Item B')")
        .expect("insert failed");
    db.execute("INSERT INTO items (id, code, description) VALUES (3, 'C001', 'Item C')")
        .expect("insert failed");
    db.execute("INSERT INTO items (id, code, description) VALUES (4, 'A002', 'Item A2')")
        .expect("insert failed");

    // Range query on strings
    let result = db.query("SELECT * FROM items WHERE code > 'A001'").expect("query failed");
    assert_eq!(result.len(), 3, "Expected 3 items with code > 'A001'");

    let result = db.query("SELECT * FROM items WHERE code < 'B001'").expect("query failed");
    assert_eq!(result.len(), 2, "Expected 2 items with code < 'B001'");
}
