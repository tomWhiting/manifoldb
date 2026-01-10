//! Constraint enforcement integration tests.
//!
//! Tests for CHECK and FOREIGN KEY constraint enforcement during
//! INSERT, UPDATE, and DELETE operations.

use manifoldb::Database;

// ============================================================================
// CHECK Constraint Tests
// ============================================================================

#[test]
fn test_check_constraint_on_insert_success() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with CHECK constraint
    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            price NUMERIC CHECK (price >= 0)
        )",
    )
    .expect("create table failed");

    // Insert with valid price
    let result = db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 29.99)");
    assert!(result.is_ok(), "insert with valid price should succeed");

    // Verify the data was inserted
    let rows = db.query("SELECT * FROM products").expect("query failed");
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_check_constraint_on_insert_failure() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with CHECK constraint requiring price > 100
    // Note: Using > 100 instead of >= 0 to avoid negative literal parsing issues
    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            price NUMERIC CHECK (price > 100)
        )",
    )
    .expect("create table failed");

    // Insert with invalid price (below 100)
    let result = db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 50.00)");
    assert!(result.is_err(), "insert with price < 100 should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("CHECK constraint violation"),
        "error should mention CHECK constraint: {}",
        err
    );
}

#[test]
fn test_check_constraint_complex_expression() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with complex CHECK constraint
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER CHECK (age >= 18 AND age <= 120),
            salary NUMERIC
        )",
    )
    .expect("create table failed");

    // Valid age
    let result =
        db.execute("INSERT INTO employees (id, name, age, salary) VALUES (1, 'Alice', 30, 50000)");
    assert!(result.is_ok(), "insert with valid age should succeed");

    // Age too low
    let result =
        db.execute("INSERT INTO employees (id, name, age, salary) VALUES (2, 'Minor', 15, 0)");
    assert!(result.is_err(), "insert with age < 18 should fail");

    // Age too high
    let result =
        db.execute("INSERT INTO employees (id, name, age, salary) VALUES (3, 'Ancient', 150, 0)");
    assert!(result.is_err(), "insert with age > 120 should fail");
}

#[test]
fn test_check_constraint_on_update_success() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            price NUMERIC CHECK (price >= 0)
        )",
    )
    .expect("create table failed");

    // Insert valid data
    db.execute("INSERT INTO products (id, price) VALUES (1, 100.00)").expect("insert failed");

    // Update to valid price
    let result = db.execute("UPDATE products SET price = 200.00 WHERE id = 1");
    assert!(result.is_ok(), "update to valid price should succeed");
}

#[test]
fn test_check_constraint_on_update_failure() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            price NUMERIC CHECK (price >= 0)
        )",
    )
    .expect("create table failed");

    // Insert valid data
    db.execute("INSERT INTO products (id, price) VALUES (1, 100.00)").expect("insert failed");

    // Update to invalid price
    let result = db.execute("UPDATE products SET price = -50.00 WHERE id = 1");
    assert!(result.is_err(), "update to negative price should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("CHECK constraint violation"),
        "error should mention CHECK constraint: {}",
        err
    );
}

// ============================================================================
// FOREIGN KEY Constraint Tests - INSERT
// ============================================================================

#[test]
fn test_foreign_key_on_insert_success() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE departments (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create departments failed");

    // Create child table with FK
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            department_id BIGINT REFERENCES departments(id)
        )",
    )
    .expect("create employees failed");

    // Insert into parent
    db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
        .expect("insert department failed");

    // Insert into child with valid FK
    let result =
        db.execute("INSERT INTO employees (id, name, department_id) VALUES (1, 'Alice', 1)");
    assert!(result.is_ok(), "insert with valid FK should succeed");
}

#[test]
fn test_foreign_key_on_insert_failure() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE departments (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create departments failed");

    // Create child table with FK
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            department_id BIGINT REFERENCES departments(id)
        )",
    )
    .expect("create employees failed");

    // Insert into child with invalid FK (no matching department)
    let result =
        db.execute("INSERT INTO employees (id, name, department_id) VALUES (1, 'Alice', 999)");
    assert!(result.is_err(), "insert with invalid FK should fail");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("FOREIGN KEY violation"), "error should mention FOREIGN KEY: {}", err);
}

#[test]
fn test_foreign_key_null_allowed() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE departments (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create departments failed");

    // Create child table with nullable FK
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            department_id BIGINT REFERENCES departments(id)
        )",
    )
    .expect("create employees failed");

    // Insert with NULL FK (should be allowed per SQL standard)
    let result = db
        .execute("INSERT INTO employees (id, name, department_id) VALUES (1, 'Freelancer', NULL)");
    assert!(result.is_ok(), "insert with NULL FK should succeed");
}

// ============================================================================
// FOREIGN KEY Constraint Tests - DELETE (RESTRICT behavior)
// ============================================================================

#[test]
fn test_foreign_key_on_delete_restrict() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE departments (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create departments failed");

    // Create child table with FK
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            department_id BIGINT REFERENCES departments(id)
        )",
    )
    .expect("create employees failed");

    // Insert data
    db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
        .expect("insert department failed");
    db.execute("INSERT INTO employees (id, name, department_id) VALUES (1, 'Alice', 1)")
        .expect("insert employee failed");

    // Try to delete parent row that is referenced
    let result = db.execute("DELETE FROM departments WHERE id = 1");
    assert!(result.is_err(), "delete of referenced row should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("FOREIGN KEY violation") || err.contains("referenced"),
        "error should mention FK violation: {}",
        err
    );
}

#[test]
fn test_foreign_key_on_delete_no_references() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE departments (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create departments failed");

    // Create child table with FK
    db.execute(
        "CREATE TABLE employees (
            id BIGINT PRIMARY KEY,
            name TEXT,
            department_id BIGINT REFERENCES departments(id)
        )",
    )
    .expect("create employees failed");

    // Insert data (only in departments)
    db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
        .expect("insert department failed");
    db.execute("INSERT INTO departments (id, name) VALUES (2, 'Marketing')")
        .expect("insert department failed");

    // Delete non-referenced department (should succeed)
    let result = db.execute("DELETE FROM departments WHERE id = 2");
    assert!(result.is_ok(), "delete of non-referenced row should succeed");
}

// ============================================================================
// Table-level CHECK Constraints
// ============================================================================

#[test]
fn test_table_level_check_constraint() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with table-level CHECK constraint
    db.execute(
        "CREATE TABLE orders (
            id BIGINT PRIMARY KEY,
            quantity INTEGER,
            unit_price NUMERIC,
            total NUMERIC,
            CONSTRAINT valid_total CHECK (total = quantity * unit_price)
        )",
    )
    .expect("create table failed");

    // Insert with valid total
    let result = db.execute(
        "INSERT INTO orders (id, quantity, unit_price, total) VALUES (1, 5, 10.00, 50.00)",
    );
    assert!(result.is_ok(), "insert with valid total should succeed");

    // Insert with invalid total
    let result = db.execute(
        "INSERT INTO orders (id, quantity, unit_price, total) VALUES (2, 5, 10.00, 100.00)",
    );
    assert!(result.is_err(), "insert with mismatched total should fail");
}

// ============================================================================
// Table-level FOREIGN KEY Constraints
// ============================================================================

#[test]
fn test_table_level_foreign_key() {
    let db = Database::in_memory().expect("failed to create db");

    // Create parent table
    db.execute("CREATE TABLE categories (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create categories failed");

    // Create child table with table-level FK
    db.execute(
        "CREATE TABLE products (
            id BIGINT PRIMARY KEY,
            name TEXT,
            category_id BIGINT,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )",
    )
    .expect("create products failed");

    // Insert parent
    db.execute("INSERT INTO categories (id, name) VALUES (1, 'Electronics')")
        .expect("insert category failed");

    // Insert with valid FK
    let result = db.execute("INSERT INTO products (id, name, category_id) VALUES (1, 'Phone', 1)");
    assert!(result.is_ok(), "insert with valid FK should succeed");

    // Insert with invalid FK
    let result =
        db.execute("INSERT INTO products (id, name, category_id) VALUES (2, 'Invalid', 999)");
    assert!(result.is_err(), "insert with invalid FK should fail");
}
