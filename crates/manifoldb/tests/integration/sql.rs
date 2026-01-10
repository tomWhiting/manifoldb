//! SQL execution integration tests.
//!
//! Tests end-to-end SQL query and statement execution against storage.

#![allow(dead_code, unused_variables)]

use manifoldb::{Database, Value};

/// Helper to get a value by column name from a query result.
fn get_by_name<'a>(
    result: &'a manifoldb::QueryResult,
    row_idx: usize,
    col_name: &str,
) -> Option<&'a Value> {
    result
        .column_index(col_name)
        .and_then(|_col_idx| result.rows().get(row_idx))
        .and_then(|row| result.column_index(col_name).and_then(|idx| row.get(idx)))
}

// ============================================================================
// Basic Query Tests
// ============================================================================

#[test]
fn test_select_all_from_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    // Query a table with no data
    let result = db.query("SELECT * FROM users").expect("query failed");

    assert!(result.is_empty());
}

#[test]
fn test_insert_and_select() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert data using SQL
    let affected =
        db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    assert_eq!(affected, 1);

    // Query the data back
    let result = db.query("SELECT * FROM users").expect("query failed");

    assert_eq!(result.len(), 1);

    // Verify the data
    let name_idx = result.column_index("name").expect("name column not found");
    let age_idx = result.column_index("age").expect("age column not found");

    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(30)));
}

#[test]
fn test_insert_multiple_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert multiple rows
    let affected = db.execute(
        "INSERT INTO products (name, price) VALUES ('Widget', 10), ('Gadget', 25), ('Gizmo', 15)"
    ).expect("insert failed");
    assert_eq!(affected, 3);

    // Query all rows
    let result = db.query("SELECT * FROM products").expect("query failed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_select_with_where_clause() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Query with filter
    let result = db.query("SELECT * FROM users WHERE age > 28").expect("query failed");

    assert_eq!(result.len(), 2);
    // Should have Alice (30) and Charlie (35)
}

#[test]
fn test_select_with_projection() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age, email) VALUES ('Alice', 30, 'alice@example.com')")
        .expect("insert failed");

    // Query with projection
    let result = db.query("SELECT name, age FROM users").expect("query failed");

    assert_eq!(result.len(), 1);
    assert_eq!(result.columns().len(), 2);
    assert!(result.columns().contains(&"name".to_string()));
    assert!(result.columns().contains(&"age".to_string()));
}

// ============================================================================
// Update Tests
// ============================================================================

#[test]
fn test_update_single_row() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");

    // Update the row
    let affected =
        db.execute("UPDATE users SET age = 31 WHERE name = 'Alice'").expect("update failed");
    assert_eq!(affected, 1);

    // Verify the update
    let result = db.query("SELECT * FROM users WHERE name = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 1);

    let age_idx = result.column_index("age").expect("age column not found");
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(31)));
}

#[test]
fn test_update_multiple_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age, active) VALUES ('Alice', 30, false)")
        .expect("insert failed");
    db.execute("INSERT INTO users (name, age, active) VALUES ('Bob', 25, false)")
        .expect("insert failed");
    db.execute("INSERT INTO users (name, age, active) VALUES ('Charlie', 35, false)")
        .expect("insert failed");

    // Update multiple rows
    let affected =
        db.execute("UPDATE users SET active = true WHERE age >= 30").expect("update failed");
    assert_eq!(affected, 2);

    // Verify
    let result = db.query("SELECT * FROM users WHERE active = true").expect("query failed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_update_no_matching_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");

    // Update with non-matching filter
    let affected =
        db.execute("UPDATE users SET age = 100 WHERE name = 'Nobody'").expect("update failed");
    assert_eq!(affected, 0);
}

// ============================================================================
// Delete Tests
// ============================================================================

#[test]
fn test_delete_single_row() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");

    // Delete one row
    let affected = db.execute("DELETE FROM users WHERE name = 'Alice'").expect("delete failed");
    assert_eq!(affected, 1);

    // Verify
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Bob".to_string())));
}

#[test]
fn test_delete_all_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");

    // Delete all rows
    let affected = db.execute("DELETE FROM users").expect("delete failed");
    assert_eq!(affected, 2);

    // Verify
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert!(result.is_empty());
}

// ============================================================================
// Parameterized Query Tests
// ============================================================================

#[test]
fn test_parameterized_insert() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert with parameters
    let affected = db
        .execute_with_params(
            "INSERT INTO users (name, age) VALUES ($1, $2)",
            &[Value::String("Alice".to_string()), Value::Int(30)],
        )
        .expect("insert failed");
    assert_eq!(affected, 1);

    // Verify
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));
}

#[test]
fn test_parameterized_select() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");

    // Query with parameter
    let result = db
        .query_with_params("SELECT * FROM users WHERE age > $1", &[Value::Int(28)])
        .expect("query failed");

    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));
}

#[test]
fn test_parameterized_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");

    // Update with parameters
    let affected = db
        .execute_with_params(
            "UPDATE users SET age = $1 WHERE name = $2",
            &[Value::Int(31), Value::String("Alice".to_string())],
        )
        .expect("update failed");
    assert_eq!(affected, 1);

    // Verify
    let result = db.query("SELECT * FROM users WHERE name = 'Alice'").expect("query failed");

    let age_idx = result.column_index("age").expect("age column not found");
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(31)));
}

// ============================================================================
// Complex Query Tests
// ============================================================================

#[test]
fn test_select_with_order_by() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Query with order by
    let result = db.query("SELECT * FROM users ORDER BY age").expect("query failed");

    assert_eq!(result.len(), 3);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Bob".to_string())));
    assert_eq!(result.rows()[1].get(name_idx), Some(&Value::String("Alice".to_string())));
    assert_eq!(result.rows()[2].get(name_idx), Some(&Value::String("Charlie".to_string())));
}

#[test]
fn test_select_with_order_by_multiple_columns() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data with duplicate ages
    db.execute("INSERT INTO users (id, age, name) VALUES (1, 25, 'Bob')").expect("insert failed");
    db.execute("INSERT INTO users (id, age, name) VALUES (2, 25, 'Alice')").expect("insert failed");
    db.execute("INSERT INTO users (id, age, name) VALUES (3, 30, 'Carol')").expect("insert failed");

    // Query with ORDER BY age, name - should sort by age first, then by name within same age
    let result = db.query("SELECT * FROM users ORDER BY age, name").expect("query failed");

    assert_eq!(result.len(), 3);

    let name_idx = result.column_index("name").expect("name column not found");
    let age_idx = result.column_index("age").expect("age column not found");

    // First row: age 25, name Alice (alphabetically first among age 25)
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(25)));
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));

    // Second row: age 25, name Bob (alphabetically second among age 25)
    assert_eq!(result.rows()[1].get(age_idx), Some(&Value::Int(25)));
    assert_eq!(result.rows()[1].get(name_idx), Some(&Value::String("Bob".to_string())));

    // Third row: age 30, name Carol
    assert_eq!(result.rows()[2].get(age_idx), Some(&Value::Int(30)));
    assert_eq!(result.rows()[2].get(name_idx), Some(&Value::String("Carol".to_string())));
}

#[test]
fn test_select_with_limit() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Query with limit
    let result = db.query("SELECT * FROM users LIMIT 2").expect("query failed");

    assert_eq!(result.len(), 2);
}

#[test]
fn test_select_with_limit_and_offset() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Query with limit and offset
    let result = db.query("SELECT * FROM users LIMIT 1 OFFSET 1").expect("query failed");

    assert_eq!(result.len(), 1);
}

// ============================================================================
// Mixed Transaction and SQL Tests
// ============================================================================

#[test]
fn test_transaction_api_and_sql() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert data using transaction API
    {
        let mut tx = db.begin().expect("failed to begin");
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("users")
            .with_property("name", "Alice")
            .with_property("age", 30i64);
        tx.put_entity(&entity).expect("failed to put entity");
        tx.commit().expect("failed to commit");
    }

    // Query using SQL
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));

    // Insert using SQL
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");

    // Query to verify both
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 2);

    // Verify using transaction API
    {
        let tx = db.begin_read().expect("failed to begin read");
        let entities = tx.iter_entities(Some("users")).expect("failed to iterate");
        assert_eq!(entities.len(), 2);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_null_values() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert with NULL value
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', NULL)").expect("insert failed");

    // Query and verify
    let result = db.query("SELECT * FROM users").expect("query failed");
    assert_eq!(result.len(), 1);

    let age_idx = result.column_index("age").expect("age column not found");
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Null));
}

#[test]
fn test_string_values_with_special_chars() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert with special characters
    db.execute("INSERT INTO notes (content) VALUES ('Hello, World!')").expect("insert failed");

    // Query and verify
    let result = db.query("SELECT * FROM notes").expect("query failed");
    assert_eq!(result.len(), 1);

    let content_idx = result.column_index("content").expect("content column not found");
    assert_eq!(
        result.rows()[0].get(content_idx),
        Some(&Value::String("Hello, World!".to_string()))
    );
}

#[test]
fn test_numeric_types() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert integer and float
    db.execute("INSERT INTO data (int_val, float_val) VALUES (42, 3.14)").expect("insert failed");

    // Query and verify
    let result = db.query("SELECT * FROM data").expect("query failed");
    assert_eq!(result.len(), 1);

    let int_idx = result.column_index("int_val").expect("int_val column not found");
    assert_eq!(result.rows()[0].get(int_idx), Some(&Value::Int(42)));

    // Float comparison
    let float_idx = result.column_index("float_val").expect("float_val column not found");
    if let Some(Value::Float(f)) = result.rows()[0].get(float_idx) {
        let expected = 3.14;
        assert!((f - expected).abs() < 0.001);
    } else {
        panic!("Expected float value");
    }
}

#[test]
fn test_boolean_values() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert with boolean values
    db.execute("INSERT INTO flags (active) VALUES (true)").expect("insert failed");
    db.execute("INSERT INTO flags (active) VALUES (false)").expect("insert failed");

    // Query and verify
    let result = db.query("SELECT * FROM flags WHERE active = true").expect("query failed");
    assert_eq!(result.len(), 1);
}

// ============================================================================
// JOIN Tests
// ============================================================================

#[test]
fn test_inner_join() {
    let db = Database::in_memory().expect("failed to create db");

    // Create users table with data
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')").expect("insert failed");
    db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')").expect("insert failed");
    db.execute("INSERT INTO users (id, name) VALUES (3, 'Carol')").expect("insert failed");

    // Create orders table with data
    db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 100)").expect("insert failed");
    db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 150)").expect("insert failed");
    db.execute("INSERT INTO orders (user_id, amount) VALUES (2, 200)").expect("insert failed");

    // Test inner join - should match users with orders (Alice has 2, Bob has 1, Carol has 0)
    let result = db
        .query("SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id")
        .expect("join query failed");

    // Alice (2 orders) + Bob (1 order) = 3 rows
    assert_eq!(result.len(), 3, "Expected 3 rows from inner join, got {}", result.len());
}

#[test]
fn test_left_join() {
    let db = Database::in_memory().expect("failed to create db");

    // Create users table with data
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')").expect("insert failed");
    db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')").expect("insert failed");
    db.execute("INSERT INTO users (id, name) VALUES (3, 'Carol')").expect("insert failed");

    // Create orders table with data - Carol has no orders
    db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 100)").expect("insert failed");
    db.execute("INSERT INTO orders (user_id, amount) VALUES (2, 200)").expect("insert failed");

    // Test left join - should include Carol even though she has no orders
    let result = db
        .query("SELECT * FROM users u LEFT JOIN orders o ON u.id = o.user_id")
        .expect("join query failed");

    // Alice (1 order) + Bob (1 order) + Carol (no orders but still included) = 3 rows
    assert_eq!(result.len(), 3, "Expected 3 rows from left join, got {}", result.len());
}

#[test]
fn test_cross_join() {
    let db = Database::in_memory().expect("failed to create db");

    // Create small tables for cross join
    db.execute("INSERT INTO colors (name) VALUES ('red')").expect("insert failed");
    db.execute("INSERT INTO colors (name) VALUES ('blue')").expect("insert failed");

    db.execute("INSERT INTO sizes (name) VALUES ('small')").expect("insert failed");
    db.execute("INSERT INTO sizes (name) VALUES ('large')").expect("insert failed");

    // Test cross join - should produce 2*2=4 rows
    let result =
        db.query("SELECT * FROM colors CROSS JOIN sizes").expect("cross join query failed");

    assert_eq!(result.len(), 4, "Expected 4 rows from cross join, got {}", result.len());
}

#[test]
fn test_join_with_filter() {
    let db = Database::in_memory().expect("failed to create db");

    // Create users table with data
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')").expect("insert failed");
    db.execute("INSERT INTO users (id, name) VALUES (2, 'Bob')").expect("insert failed");

    // Create orders table with data
    db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 100)").expect("insert failed");
    db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 150)").expect("insert failed");
    db.execute("INSERT INTO orders (user_id, amount) VALUES (2, 200)").expect("insert failed");

    // Test join with WHERE filter on order amount
    let result = db
        .query("SELECT * FROM users u INNER JOIN orders o ON u.id = o.user_id WHERE o.amount > 120")
        .expect("join query failed");

    // Alice's 150 order + Bob's 200 order = 2 rows
    assert_eq!(result.len(), 2, "Expected 2 rows from filtered join, got {}", result.len());
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_invalid_sql_syntax() {
    let db = Database::in_memory().expect("failed to create db");

    // This should fail to parse
    let result = db.query("SELEKT * FORM users");
    assert!(result.is_err());
}

// ============================================================================
// Aggregate Query Tests
// ============================================================================

#[test]
fn test_count_aggregate() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Count all rows
    let result = db.query("SELECT COUNT(*) FROM users").expect("query failed");

    assert_eq!(result.len(), 1);
    // Check the count value
    let count_val = result.rows()[0].get(0);
    assert_eq!(count_val, Some(&Value::Int(3)));
}

#[test]
fn test_sum_aggregate() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO products (name, price) VALUES ('Widget', 10)").expect("insert failed");
    db.execute("INSERT INTO products (name, price) VALUES ('Gadget', 25)").expect("insert failed");
    db.execute("INSERT INTO products (name, price) VALUES ('Gizmo', 15)").expect("insert failed");

    // Sum prices
    let result = db.query("SELECT SUM(price) FROM products").expect("query failed");

    assert_eq!(result.len(), 1);
    // Check the sum value - it returns as float
    if let Some(Value::Float(sum)) = result.rows()[0].get(0) {
        assert!((sum - 50.0).abs() < 0.001);
    } else {
        panic!("Expected float sum value, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_group_by_count() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data - two departments with different counts
    db.execute("INSERT INTO employees (name, dept) VALUES ('Alice', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Bob', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Charlie', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Diana', 'Sales')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Eve', 'Sales')")
        .expect("insert failed");

    // Group by department and count
    let result =
        db.query("SELECT dept, COUNT(*) FROM employees GROUP BY dept").expect("query failed");

    assert_eq!(result.len(), 2);

    // Find Engineering and Sales rows
    let mut eng_count = 0;
    let mut sales_count = 0;

    for row in result.rows() {
        if let Some(Value::String(dept)) = row.get(0) {
            if let Some(Value::Int(count)) = row.get(1) {
                if dept == "Engineering" {
                    eng_count = *count;
                } else if dept == "Sales" {
                    sales_count = *count;
                }
            }
        }
    }

    assert_eq!(eng_count, 3, "Engineering should have 3 employees");
    assert_eq!(sales_count, 2, "Sales should have 2 employees");
}

#[test]
fn test_group_by_sum() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data - products by category
    db.execute("INSERT INTO products (category, price) VALUES ('Electronics', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Electronics', 150)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Clothing', 50)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Clothing', 75)")
        .expect("insert failed");

    // Group by category and sum prices
    let result = db
        .query("SELECT category, SUM(price) FROM products GROUP BY category")
        .expect("query failed");

    assert_eq!(result.len(), 2);

    // Find the sums for each category
    let mut electronics_sum = 0.0;
    let mut clothing_sum = 0.0;

    for row in result.rows() {
        if let Some(Value::String(category)) = row.get(0) {
            if let Some(Value::Float(sum)) = row.get(1) {
                if category == "Electronics" {
                    electronics_sum = *sum;
                } else if category == "Clothing" {
                    clothing_sum = *sum;
                }
            }
        }
    }

    assert!((electronics_sum - 250.0).abs() < 0.001, "Electronics sum should be 250");
    assert!((clothing_sum - 125.0).abs() < 0.001, "Clothing sum should be 125");
}

#[test]
fn test_min_max_aggregate() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Get min and max age
    let result = db.query("SELECT MIN(age), MAX(age) FROM users").expect("query failed");

    assert_eq!(result.len(), 1);

    let min_val = result.rows()[0].get(0);
    let max_val = result.rows()[0].get(1);

    assert_eq!(min_val, Some(&Value::Int(25)), "Min age should be 25");
    assert_eq!(max_val, Some(&Value::Int(35)), "Max age should be 35");
}

#[test]
fn test_aggregate_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    // Count on empty table should return 0
    let result = db.query("SELECT COUNT(*) FROM empty_table").expect("query failed");

    assert_eq!(result.len(), 1);
    let count_val = result.rows()[0].get(0);
    assert_eq!(count_val, Some(&Value::Int(0)));
}

#[test]
fn test_multiple_aggregates() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO products (name, price) VALUES ('Widget', 10)").expect("insert failed");
    db.execute("INSERT INTO products (name, price) VALUES ('Gadget', 25)").expect("insert failed");
    db.execute("INSERT INTO products (name, price) VALUES ('Gizmo', 15)").expect("insert failed");

    // Multiple aggregates in one query
    let result = db
        .query("SELECT COUNT(*), SUM(price), MIN(price), MAX(price) FROM products")
        .expect("query failed");

    assert_eq!(result.len(), 1);
    let row = &result.rows()[0];

    // COUNT(*)
    assert_eq!(row.get(0), Some(&Value::Int(3)));

    // SUM(price)
    if let Some(Value::Float(sum)) = row.get(1) {
        assert!((sum - 50.0).abs() < 0.001);
    } else {
        panic!("Expected float sum value");
    }

    // MIN(price)
    assert_eq!(row.get(2), Some(&Value::Int(10)));

    // MAX(price)
    assert_eq!(row.get(3), Some(&Value::Int(25)));
}

// ============================================================================
// DISTINCT Query Tests
// ============================================================================

#[test]
fn test_select_distinct_single_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data with duplicate categories
    db.execute("INSERT INTO products (name, category) VALUES ('Widget', 'Electronics')")
        .expect("insert failed");
    db.execute("INSERT INTO products (name, category) VALUES ('Gadget', 'Electronics')")
        .expect("insert failed");
    db.execute("INSERT INTO products (name, category) VALUES ('Shirt', 'Clothing')")
        .expect("insert failed");
    db.execute("INSERT INTO products (name, category) VALUES ('Pants', 'Clothing')")
        .expect("insert failed");
    db.execute("INSERT INTO products (name, category) VALUES ('Hat', 'Clothing')")
        .expect("insert failed");

    // Query distinct categories
    let result = db.query("SELECT DISTINCT category FROM products").expect("query failed");

    // Should only have 2 unique categories
    assert_eq!(result.len(), 2, "Expected 2 distinct categories, got {}", result.len());

    // Verify both categories are present
    let mut categories: Vec<String> =
        result
            .rows()
            .iter()
            .filter_map(|row| {
                if let Some(Value::String(s)) = row.get(0) {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
    categories.sort();
    assert_eq!(categories, vec!["Clothing", "Electronics"]);
}

#[test]
fn test_select_distinct_multiple_columns() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data with duplicate name/city combinations
    db.execute("INSERT INTO users (name, city) VALUES ('Alice', 'NYC')").expect("insert failed");
    db.execute("INSERT INTO users (name, city) VALUES ('Alice', 'NYC')").expect("insert failed");
    db.execute("INSERT INTO users (name, city) VALUES ('Alice', 'LA')").expect("insert failed");
    db.execute("INSERT INTO users (name, city) VALUES ('Bob', 'NYC')").expect("insert failed");
    db.execute("INSERT INTO users (name, city) VALUES ('Bob', 'NYC')").expect("insert failed");

    // Query distinct name/city combinations
    let result = db.query("SELECT DISTINCT name, city FROM users").expect("query failed");

    // Should have 3 unique combinations: (Alice, NYC), (Alice, LA), (Bob, NYC)
    assert_eq!(result.len(), 3, "Expected 3 distinct name/city combinations, got {}", result.len());
}

#[test]
fn test_select_distinct_all_unique() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert data where all values are already unique
    db.execute("INSERT INTO users (name) VALUES ('Alice')").expect("insert failed");
    db.execute("INSERT INTO users (name) VALUES ('Bob')").expect("insert failed");
    db.execute("INSERT INTO users (name) VALUES ('Charlie')").expect("insert failed");

    // Query distinct
    let result = db.query("SELECT DISTINCT name FROM users").expect("query failed");

    // Should still have all 3 rows
    assert_eq!(result.len(), 3, "Expected 3 distinct names, got {}", result.len());
}

#[test]
fn test_select_distinct_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    // Query distinct on empty table
    let result = db.query("SELECT DISTINCT category FROM products").expect("query failed");

    // Should be empty
    assert!(result.is_empty());
}

#[test]
fn test_select_distinct_with_where() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO products (category, price) VALUES ('Electronics', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Electronics', 200)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Clothing', 50)")
        .expect("insert failed");
    db.execute("INSERT INTO products (category, price) VALUES ('Clothing', 75)")
        .expect("insert failed");

    // Query distinct categories with price filter
    let result =
        db.query("SELECT DISTINCT category FROM products WHERE price > 60").expect("query failed");

    // Should have 2 categories (Electronics: 100, 200 > 60; Clothing: 75 > 60)
    assert_eq!(result.len(), 2, "Expected 2 distinct categories, got {}", result.len());
}

#[test]
fn test_select_distinct_star() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert data with some completely duplicate rows
    db.execute("INSERT INTO simple (val) VALUES (1)").expect("insert failed");
    db.execute("INSERT INTO simple (val) VALUES (1)").expect("insert failed");
    db.execute("INSERT INTO simple (val) VALUES (2)").expect("insert failed");

    // Note: SELECT DISTINCT * will include 'id' column which is unique per row
    // So all 3 rows will be distinct because they have different ids
    let result = db.query("SELECT DISTINCT * FROM simple").expect("query failed");

    // All rows are distinct because 'id' column is unique
    assert_eq!(result.len(), 3);
}

// ============================================================================
// Vector Distance Query Tests
// ============================================================================

#[test]
fn test_vector_distance_euclidean() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert documents with vector embeddings
    // Using simple vectors where distance can be calculated manually
    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc1', $1)",
        &[Value::Vector(vec![1.0, 0.0, 0.0, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc2', $1)",
        &[Value::Vector(vec![0.0, 1.0, 0.0, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc3', $1)",
        &[Value::Vector(vec![0.0, 0.0, 1.0, 0.0])],
    )
    .expect("insert failed");

    // Query vector distance using Euclidean distance operator <->
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <-> $1 as distance FROM documents",
            &[query_vector],
        )
        .expect("query failed");

    assert_eq!(result.len(), 3);

    // Find the distance column index
    let dist_idx = result.column_index("distance").expect("distance column not found");

    // doc1 should have distance 0 (same vector)
    // doc2 and doc3 should have distance sqrt(2) ≈ 1.414
    let mut has_zero_distance = false;
    for row in result.rows() {
        if let Some(Value::Float(dist)) = row.get(dist_idx) {
            if *dist < 0.001 {
                has_zero_distance = true;
            }
        }
    }
    assert!(has_zero_distance, "Should have one document with zero distance");
}

#[test]
fn test_vector_distance_order_by() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert documents at varying distances from query
    // Query vector: [1, 0, 0]
    // doc1: [1, 0, 0] -> distance 0
    // doc2: [0.8, 0.2, 0] -> distance sqrt((0.2)^2 + (0.2)^2) ≈ 0.28
    // doc3: [0.5, 0.5, 0] -> distance sqrt((0.5)^2 + (0.5)^2) ≈ 0.71
    // doc4: [0, 1, 0] -> distance sqrt(2) ≈ 1.41

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc1', $1)",
        &[Value::Vector(vec![1.0, 0.0, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc2', $1)",
        &[Value::Vector(vec![0.8, 0.2, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc3', $1)",
        &[Value::Vector(vec![0.5, 0.5, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('doc4', $1)",
        &[Value::Vector(vec![0.0, 1.0, 0.0])],
    )
    .expect("insert failed");

    // Query with ORDER BY distance
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <-> $1 as distance FROM documents ORDER BY embedding <-> $1",
            &[query_vector],
        )
        .expect("query failed");

    assert_eq!(result.len(), 4);

    // Verify order: doc1, doc2, doc3, doc4
    let title_idx = result.column_index("title").expect("title column not found");

    // First result should be doc1 (closest)
    assert_eq!(
        result.rows()[0].get(title_idx),
        Some(&Value::String("doc1".to_string())),
        "First result should be doc1 (closest)"
    );

    // Last result should be doc4 (farthest)
    assert_eq!(
        result.rows()[3].get(title_idx),
        Some(&Value::String("doc4".to_string())),
        "Last result should be doc4 (farthest)"
    );
}

#[test]
fn test_vector_distance_with_limit() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert 5 documents
    for i in 1..=5 {
        let v = i as f32 / 5.0;
        db.execute_with_params(
            &format!("INSERT INTO documents (title, embedding) VALUES ('doc{}', $1)", i),
            &[Value::Vector(vec![v, 1.0 - v, 0.0])],
        )
        .expect("insert failed");
    }

    // Query top 3 closest to [1, 0, 0]
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <-> $1 as distance FROM documents ORDER BY embedding <-> $1 LIMIT 3",
            &[query_vector],
        )
        .expect("query failed");

    assert_eq!(result.len(), 3, "Expected 3 results with LIMIT 3");
}

#[test]
fn test_vector_distance_cosine() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert documents with different cosine similarities
    // doc1: same direction as query
    // doc2: opposite direction
    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('same_dir', $1)",
        &[Value::Vector(vec![2.0, 0.0, 0.0])], // Same direction, different magnitude
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('opposite', $1)",
        &[Value::Vector(vec![-1.0, 0.0, 0.0])], // Opposite direction
    )
    .expect("insert failed");

    // Query using cosine distance operator <=>
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <=> $1 as distance FROM documents",
            &[query_vector],
        )
        .expect("query failed");

    assert_eq!(result.len(), 2);

    let title_idx = result.column_index("title").expect("title column not found");
    let dist_idx = result.column_index("distance").expect("distance column not found");

    // Find the row with same_dir - should have distance ~0 (cosine distance)
    for row in result.rows() {
        if let Some(Value::String(title)) = row.get(title_idx) {
            if let Some(Value::Float(dist)) = row.get(dist_idx) {
                if title == "same_dir" {
                    assert!(
                        *dist < 0.001,
                        "Same direction should have cosine distance ~0, got {}",
                        dist
                    );
                } else if title == "opposite" {
                    assert!(
                        (*dist - 2.0).abs() < 0.001,
                        "Opposite direction should have cosine distance ~2, got {}",
                        dist
                    );
                }
            }
        }
    }
}

#[test]
fn test_vector_distance_inner_product() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert documents with known inner products
    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('high_dot', $1)",
        &[Value::Vector(vec![2.0, 2.0])], // Dot with [1,1] = 4
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('low_dot', $1)",
        &[Value::Vector(vec![0.5, 0.5])], // Dot with [1,1] = 1
    )
    .expect("insert failed");

    // Query using inner product operator <#>
    // Note: inner product distance is negated for ordering
    let query_vector = Value::Vector(vec![1.0, 1.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <#> $1 as distance FROM documents ORDER BY embedding <#> $1",
            &[query_vector],
        )
        .expect("query failed");

    assert_eq!(result.len(), 2);

    let title_idx = result.column_index("title").expect("title column not found");

    // high_dot should be first (lower distance because it's negated inner product)
    assert_eq!(
        result.rows()[0].get(title_idx),
        Some(&Value::String("high_dot".to_string())),
        "Higher inner product should come first"
    );
}

#[test]
fn test_vector_distance_with_filter() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert documents with category and embedding
    db.execute_with_params(
        "INSERT INTO documents (title, category, embedding) VALUES ('tech1', 'Technology', $1)",
        &[Value::Vector(vec![1.0, 0.0, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, category, embedding) VALUES ('tech2', 'Technology', $1)",
        &[Value::Vector(vec![0.8, 0.2, 0.0])],
    )
    .expect("insert failed");

    db.execute_with_params(
        "INSERT INTO documents (title, category, embedding) VALUES ('sports1', 'Sports', $1)",
        &[Value::Vector(vec![0.5, 0.5, 0.0])],
    )
    .expect("insert failed");

    // Query with WHERE filter and ORDER BY distance
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <-> $1 as distance FROM documents WHERE category = 'Technology' ORDER BY embedding <-> $1",
            &[query_vector],
        )
        .expect("query failed");

    // Should only have 2 Technology documents
    assert_eq!(result.len(), 2);

    let title_idx = result.column_index("title").expect("title column not found");

    // First should be tech1 (closest to query)
    assert_eq!(result.rows()[0].get(title_idx), Some(&Value::String("tech1".to_string())));
}

#[test]
fn test_vector_distance_null_handling() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert document with embedding
    db.execute_with_params(
        "INSERT INTO documents (title, embedding) VALUES ('with_embedding', $1)",
        &[Value::Vector(vec![1.0, 0.0, 0.0])],
    )
    .expect("insert failed");

    // Insert document without embedding (NULL)
    db.execute("INSERT INTO documents (title) VALUES ('no_embedding')").expect("insert failed");

    // Query all documents
    let query_vector = Value::Vector(vec![1.0, 0.0, 0.0]);
    let result = db
        .query_with_params(
            "SELECT title, embedding <-> $1 as distance FROM documents",
            &[query_vector],
        )
        .expect("query failed");

    // Should return both documents
    assert_eq!(result.len(), 2);

    // The document without embedding should have NULL distance
    let title_idx = result.column_index("title").expect("title column not found");
    let dist_idx = result.column_index("distance").expect("distance column not found");

    for row in result.rows() {
        if let Some(Value::String(title)) = row.get(title_idx) {
            if title == "no_embedding" {
                assert_eq!(
                    row.get(dist_idx),
                    Some(&Value::Null),
                    "Document without embedding should have NULL distance"
                );
            }
        }
    }
}

// ============================================================================
// EXPLAIN Tests
// ============================================================================

#[test]
fn test_explain_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Run EXPLAIN on a simple query
    let result = db.query("EXPLAIN SELECT * FROM users WHERE age > 21").expect("explain failed");

    // Should have at least some rows (plan output)
    assert!(!result.is_empty(), "EXPLAIN should return plan output");

    // Check that we have a 'plan' column
    let plan_idx = result.column_index("plan").expect("plan column not found");

    // First row should be the logical plan header
    let first_row = result.rows().first().expect("no rows");
    if let Some(Value::String(line)) = first_row.get(plan_idx) {
        assert!(line.contains("Logical Plan"), "Expected Logical Plan header, got: {}", line);
    }

    // Should also have physical plan output
    let has_physical = result.rows().iter().any(|row| {
        if let Some(Value::String(line)) = row.get(plan_idx) {
            line.contains("Physical Plan")
        } else {
            false
        }
    });
    assert!(has_physical, "EXPLAIN should include Physical Plan");
}

#[test]
fn test_explain_with_join() {
    let db = Database::in_memory().expect("failed to create db");

    // Run EXPLAIN on a join query
    let result = db
        .query("EXPLAIN SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
        .expect("explain failed");

    assert!(!result.is_empty(), "EXPLAIN should return plan output");

    // Should mention Join somewhere in the output
    let plan_idx = result.column_index("plan").expect("plan column not found");
    let has_join = result.rows().iter().any(|row| {
        if let Some(Value::String(line)) = row.get(plan_idx) {
            line.contains("Join")
        } else {
            false
        }
    });
    assert!(has_join, "EXPLAIN of JOIN query should mention Join in plan");
}

// ============================================================================
// JSON Aggregate Function Tests
// ============================================================================

#[test]
fn test_json_agg_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Bob', 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // Aggregate names into JSON array
    let result = db.query("SELECT JSON_AGG(name) FROM users").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        // Parse the JSON to verify structure
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        // Check that all names are present (order may vary)
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Bob"));
        assert!(names.contains(&"Charlie"));
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_jsonb_agg_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO products (name, price) VALUES ('Widget', 10)").expect("insert failed");
    db.execute("INSERT INTO products (name, price) VALUES ('Gadget', 25)").expect("insert failed");

    // Aggregate prices into JSONB array
    let result = db.query("SELECT JSONB_AGG(price) FROM products").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // Prices should be numbers
        let prices: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
        assert!(prices.contains(&10));
        assert!(prices.contains(&25));
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_json_agg_with_group_by() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO employees (name, dept) VALUES ('Alice', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Bob', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (name, dept) VALUES ('Charlie', 'Sales')")
        .expect("insert failed");

    // Aggregate names by department
    let result =
        db.query("SELECT dept, JSON_AGG(name) FROM employees GROUP BY dept").expect("query failed");

    assert_eq!(result.len(), 2);

    for row in result.rows() {
        if let Some(Value::String(dept)) = row.get(0) {
            if let Some(Value::String(json_str)) = row.get(1) {
                let parsed: serde_json::Value =
                    serde_json::from_str(json_str).expect("invalid JSON");
                let arr = parsed.as_array().unwrap();
                if dept == "Engineering" {
                    assert_eq!(arr.len(), 2);
                } else if dept == "Sales" {
                    assert_eq!(arr.len(), 1);
                }
            }
        }
    }
}

#[test]
fn test_json_object_agg_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO config (key, value) VALUES ('host', 'localhost')")
        .expect("insert failed");
    db.execute("INSERT INTO config (key, value) VALUES ('port', '8080')").expect("insert failed");
    db.execute("INSERT INTO config (key, value) VALUES ('debug', 'true')").expect("insert failed");

    // Aggregate into JSON object
    let result = db.query("SELECT JSON_OBJECT_AGG(key, value) FROM config").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        assert!(parsed.is_object());
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("host"), Some(&serde_json::Value::String("localhost".to_string())));
        assert_eq!(obj.get("port"), Some(&serde_json::Value::String("8080".to_string())));
        assert_eq!(obj.get("debug"), Some(&serde_json::Value::String("true".to_string())));
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_jsonb_object_agg_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data with numeric values
    db.execute("INSERT INTO stats (metric, value) VALUES ('users', 100)").expect("insert failed");
    db.execute("INSERT INTO stats (metric, value) VALUES ('orders', 50)").expect("insert failed");

    // Aggregate into JSONB object
    let result =
        db.query("SELECT JSONB_OBJECT_AGG(metric, value) FROM stats").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        assert!(parsed.is_object());
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj.get("users"), Some(&serde_json::Value::Number(100.into())));
        assert_eq!(obj.get("orders"), Some(&serde_json::Value::Number(50.into())));
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_json_object_agg_with_group_by() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data
    db.execute("INSERT INTO settings (category, key, value) VALUES ('app', 'name', 'MyApp')")
        .expect("insert failed");
    db.execute("INSERT INTO settings (category, key, value) VALUES ('app', 'version', '1.0')")
        .expect("insert failed");
    db.execute("INSERT INTO settings (category, key, value) VALUES ('db', 'host', 'localhost')")
        .expect("insert failed");
    db.execute("INSERT INTO settings (category, key, value) VALUES ('db', 'port', '5432')")
        .expect("insert failed");

    // Aggregate by category
    let result = db
        .query("SELECT category, JSON_OBJECT_AGG(key, value) FROM settings GROUP BY category")
        .expect("query failed");

    assert_eq!(result.len(), 2);

    for row in result.rows() {
        if let Some(Value::String(category)) = row.get(0) {
            if let Some(Value::String(json_str)) = row.get(1) {
                let parsed: serde_json::Value =
                    serde_json::from_str(json_str).expect("invalid JSON");
                let obj = parsed.as_object().unwrap();

                if category == "app" {
                    assert_eq!(
                        obj.get("name"),
                        Some(&serde_json::Value::String("MyApp".to_string()))
                    );
                    assert_eq!(
                        obj.get("version"),
                        Some(&serde_json::Value::String("1.0".to_string()))
                    );
                } else if category == "db" {
                    assert_eq!(
                        obj.get("host"),
                        Some(&serde_json::Value::String("localhost".to_string()))
                    );
                    assert_eq!(
                        obj.get("port"),
                        Some(&serde_json::Value::String("5432".to_string()))
                    );
                }
            }
        }
    }
}

#[test]
fn test_json_agg_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    // JSON_AGG on empty table should return NULL
    let result = db.query("SELECT JSON_AGG(name) FROM empty_table").expect("query failed");

    assert_eq!(result.len(), 1);
    // Should be NULL for empty aggregation
    let val = result.rows()[0].get(0);
    assert!(
        val == Some(&Value::Null) || val.is_none(),
        "Expected NULL for empty JSON_AGG, got {:?}",
        val
    );
}

#[test]
fn test_json_object_agg_empty_table() {
    let db = Database::in_memory().expect("failed to create db");

    // JSON_OBJECT_AGG on empty table should return NULL
    let result =
        db.query("SELECT JSON_OBJECT_AGG(key, value) FROM empty_table").expect("query failed");

    assert_eq!(result.len(), 1);
    let val = result.rows()[0].get(0);
    assert!(
        val == Some(&Value::Null) || val.is_none(),
        "Expected NULL for empty JSON_OBJECT_AGG, got {:?}",
        val
    );
}

#[test]
fn test_json_agg_with_nulls() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert test data with NULL values
    db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES (NULL, 25)").expect("insert failed");
    db.execute("INSERT INTO users (name, age) VALUES ('Charlie', 35)").expect("insert failed");

    // JSON_AGG should skip NULL values
    let result = db.query("SELECT JSON_AGG(name) FROM users").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        let arr = parsed.as_array().unwrap();
        // Should only have 2 elements (NULL was skipped)
        assert_eq!(arr.len(), 2);
        let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"Alice"));
        assert!(names.contains(&"Charlie"));
        assert!(!names.iter().any(|&n| n == "null" || n.is_empty()));
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

#[test]
fn test_json_agg_mixed_types() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert various value types
    db.execute("INSERT INTO mixed (val) VALUES ('hello')").expect("insert failed");
    db.execute("INSERT INTO mixed (val) VALUES (42)").expect("insert failed");
    db.execute("INSERT INTO mixed (val) VALUES (3.14)").expect("insert failed");
    db.execute("INSERT INTO mixed (val) VALUES (true)").expect("insert failed");

    // JSON_AGG should handle mixed types
    let result = db.query("SELECT JSON_AGG(val) FROM mixed").expect("query failed");

    assert_eq!(result.len(), 1);
    if let Some(Value::String(json_str)) = result.rows()[0].get(0) {
        let parsed: serde_json::Value = serde_json::from_str(json_str).expect("invalid JSON");
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 4);
    } else {
        panic!("Expected JSON string result, got {:?}", result.rows()[0].get(0));
    }
}

// ============================================================================
// View Expansion Tests
// ============================================================================

#[test]
fn test_view_expansion_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a table with data
    db.execute("INSERT INTO employees (id, name, dept) VALUES (1, 'Alice', 'Engineering')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (id, name, dept) VALUES (2, 'Bob', 'Sales')")
        .expect("insert failed");
    db.execute("INSERT INTO employees (id, name, dept) VALUES (3, 'Charlie', 'Engineering')")
        .expect("insert failed");

    // Create a view
    db.execute(
        "CREATE VIEW engineering_staff AS SELECT * FROM employees WHERE dept = 'Engineering'",
    )
    .expect("create view failed");

    // Query the view
    let result = db.query("SELECT * FROM engineering_staff").expect("query failed");

    // Should have 2 rows (Alice and Charlie)
    assert_eq!(result.len(), 2);
}

#[test]
fn test_view_expansion_with_alias() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a table with data
    db.execute("INSERT INTO items (id, name) VALUES (1, 'Item A')").expect("insert failed");
    db.execute("INSERT INTO items (id, name) VALUES (2, 'Item B')").expect("insert failed");

    // Create a view
    db.execute("CREATE VIEW all_items AS SELECT * FROM items").expect("create view failed");

    // Query with alias
    let result = db.query("SELECT i.id, i.name FROM all_items AS i").expect("query failed");

    assert_eq!(result.len(), 2);
}

#[test]
fn test_view_in_join() {
    let db = Database::in_memory().expect("failed to create db");

    // Create tables with data
    db.execute("INSERT INTO orders (id, customer_id, total) VALUES (1, 100, 50)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, customer_id, total) VALUES (2, 100, 75)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, customer_id, total) VALUES (3, 200, 30)")
        .expect("insert failed");

    db.execute("INSERT INTO customers (id, name) VALUES (100, 'Alice')").expect("insert failed");
    db.execute("INSERT INTO customers (id, name) VALUES (200, 'Bob')").expect("insert failed");

    // Create a view for high-value orders
    db.execute("CREATE VIEW high_value_orders AS SELECT * FROM orders WHERE total > 40")
        .expect("create view failed");

    // Join with the view
    let result = db
        .query(
            "SELECT c.name, o.total
             FROM customers c
             JOIN high_value_orders o ON c.id = o.customer_id",
        )
        .expect("query failed");

    // Should have 2 rows (Alice's orders at 50 and 75)
    assert_eq!(result.len(), 2);
}

// ============================================================================
// Uncorrelated Subquery Tests
// ============================================================================

#[test]
fn test_exists_uncorrelated_subquery() {
    let db = Database::in_memory().expect("failed to create db");

    // Create tables with data
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')").expect("insert failed");
    db.execute("INSERT INTO products (id, name) VALUES (2, 'Gadget')").expect("insert failed");

    db.execute("INSERT INTO orders (id, product_id) VALUES (1, 1)").expect("insert failed");

    // Uncorrelated EXISTS - returns all products if any order exists
    let result = db
        .query("SELECT * FROM products WHERE EXISTS (SELECT 1 FROM orders)")
        .expect("query failed");

    // Both products returned because orders table has data
    assert_eq!(result.len(), 2);
}

#[test]
fn test_exists_uncorrelated_empty() {
    let db = Database::in_memory().expect("failed to create db");

    // Create products table with data
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')").expect("insert failed");

    // Create an empty orders table explicitly (don't insert any data)
    db.execute("CREATE TABLE orders (id BIGINT, product_id BIGINT)").expect("create table failed");

    // First verify the empty orders table returns 0 rows
    let empty_check = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(empty_check.len(), 0, "orders table should be empty");

    // Empty orders table - EXISTS should return false
    let result = db
        .query("SELECT * FROM products WHERE EXISTS (SELECT 1 FROM orders)")
        .expect("query failed");

    // No products returned because orders table is empty
    assert_eq!(result.len(), 0);
}

#[test]
fn test_not_exists_uncorrelated() {
    let db = Database::in_memory().expect("failed to create db");

    // Create products table with data
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')").expect("insert failed");

    // Create an empty orders table explicitly (don't insert any data)
    db.execute("CREATE TABLE orders (id BIGINT, product_id BIGINT)").expect("create table failed");

    // Empty orders table - NOT EXISTS should return true
    let result = db
        .query("SELECT * FROM products WHERE NOT EXISTS (SELECT 1 FROM orders)")
        .expect("query failed");

    // All products returned because orders table is empty
    assert_eq!(result.len(), 1);
}

#[test]
fn test_in_subquery_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create tables with data
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')").expect("insert failed");
    db.execute("INSERT INTO products (id, name) VALUES (2, 'Gadget')").expect("insert failed");
    db.execute("INSERT INTO products (id, name) VALUES (3, 'Gizmo')").expect("insert failed");

    db.execute("INSERT INTO orders (id, product_id) VALUES (1, 1)").expect("insert failed");
    db.execute("INSERT INTO orders (id, product_id) VALUES (2, 3)").expect("insert failed");

    // IN subquery - get products that have orders
    let result = db
        .query("SELECT * FROM products WHERE id IN (SELECT product_id FROM orders)")
        .expect("query failed");

    // Should return Widget and Gizmo (ids 1 and 3)
    assert_eq!(result.len(), 2);
}

#[test]
fn test_not_in_subquery() {
    let db = Database::in_memory().expect("failed to create db");

    // Create tables with data
    db.execute("INSERT INTO products (id, name) VALUES (1, 'Widget')").expect("insert failed");
    db.execute("INSERT INTO products (id, name) VALUES (2, 'Gadget')").expect("insert failed");
    db.execute("INSERT INTO products (id, name) VALUES (3, 'Gizmo')").expect("insert failed");

    db.execute("INSERT INTO orders (id, product_id) VALUES (1, 1)").expect("insert failed");
    db.execute("INSERT INTO orders (id, product_id) VALUES (2, 3)").expect("insert failed");

    // NOT IN subquery - get products that have NO orders
    let result = db
        .query("SELECT * FROM products WHERE id NOT IN (SELECT product_id FROM orders)")
        .expect("query failed");

    // Should return only Gadget (id 2)
    assert_eq!(result.len(), 1);
}

#[test]
fn test_scalar_subquery_in_select() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 10)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 25)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (3, 'Gizmo', 15)")
        .expect("insert failed");

    // Scalar subquery in SELECT to get max price
    let result = db
        .query("SELECT name, (SELECT MAX(price) FROM products) AS max_price FROM products")
        .expect("query failed");

    assert_eq!(result.len(), 3);
    // Each row should have max_price = 25
    if let Some(max_price_idx) = result.column_index("max_price") {
        for row in result.rows() {
            assert_eq!(row.get(max_price_idx), Some(&Value::Int(25)));
        }
    }
}

#[test]
fn test_scalar_subquery_in_where() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 10)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 25)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (3, 'Gizmo', 15)")
        .expect("insert failed");

    // Scalar subquery in WHERE to find products priced above average
    let result = db
        .query("SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)")
        .expect("query failed");

    // Average is ~16.67, so only Gadget (25) should be returned
    assert_eq!(result.len(), 1);
}

// ============================================================================
// CTE Scoping Tests
// ============================================================================

#[test]
fn test_cte_basic_execution() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 10)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 25)")
        .expect("insert failed");

    // Use CTE
    let result = db
        .query(
            "WITH expensive AS (SELECT * FROM products WHERE price > 15)
             SELECT * FROM expensive",
        )
        .expect("query failed");

    // Should return only Gadget
    assert_eq!(result.len(), 1);
}

#[test]
fn test_cte_multiple_references() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO numbers (n) VALUES (1)").expect("insert failed");
    db.execute("INSERT INTO numbers (n) VALUES (2)").expect("insert failed");
    db.execute("INSERT INTO numbers (n) VALUES (3)").expect("insert failed");

    // CTE referenced twice
    let result = db
        .query(
            "WITH nums AS (SELECT * FROM numbers)
             SELECT a.n AS a, b.n AS b FROM nums a, nums b WHERE a.n < b.n",
        )
        .expect("query failed");

    // Should return pairs: (1,2), (1,3), (2,3) = 3 rows
    assert_eq!(result.len(), 3);
}

#[test]
fn test_cte_chained() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO data (val) VALUES (5)").expect("insert failed");
    db.execute("INSERT INTO data (val) VALUES (10)").expect("insert failed");

    // Chained CTEs where second references first
    let result = db
        .query(
            "WITH
                step1 AS (SELECT val * 2 AS doubled FROM data),
                step2 AS (SELECT doubled + 1 AS result FROM step1)
             SELECT * FROM step2",
        )
        .expect("query failed");

    assert_eq!(result.len(), 2);
    // Values should be (5*2)+1=11 and (10*2)+1=21
}

#[test]
fn test_cte_shadows_table() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with data
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Real Alice')").expect("insert failed");

    // First, EXPLAIN to see the plan
    let explain = db
        .query(
            "EXPLAIN WITH users AS (SELECT 999 AS id, 'CTE User' AS name)
             SELECT * FROM users",
        )
        .expect("explain failed");

    println!("=== EXPLAIN ===");
    for row in explain.rows() {
        if let Some(val) = row.get(0) {
            println!("{:?}", val);
        }
    }

    // CTE with same name as table should shadow it
    let result = db
        .query(
            "WITH users AS (SELECT 999 AS id, 'CTE User' AS name)
             SELECT * FROM users",
        )
        .expect("query failed");

    println!("=== RESULT ===");
    println!("Row count: {}", result.len());
    println!("Columns: {:?}", result.columns());
    for (i, row) in result.rows().iter().enumerate() {
        println!("Row {}: {:?}", i, row);
    }

    assert_eq!(result.len(), 1);
    // Should get CTE data, not table data
    let id_idx = result.column_index("id").expect("id column not found");
    assert_eq!(result.rows()[0].get(id_idx), Some(&Value::Int(999)));
}

// ============================================================================
// INSERT ON CONFLICT (Upsert) Tests
// ============================================================================

#[test]
fn test_insert_on_conflict_do_nothing() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data
    db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .expect("initial insert failed");

    // Insert with ON CONFLICT DO NOTHING - should skip the conflicting row
    let affected = db
        .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice Updated', 35) ON CONFLICT (id) DO NOTHING")
        .expect("upsert failed");
    assert_eq!(affected, 0, "Should not insert any rows when conflict exists");

    // Verify original data is unchanged
    let result = db.query("SELECT * FROM users WHERE id = 1").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    let age_idx = result.column_index("age").expect("age column not found");

    // Original values should be preserved
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(30)));
}

#[test]
fn test_insert_on_conflict_do_nothing_no_conflict() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data
    db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .expect("initial insert failed");

    // Insert with ON CONFLICT DO NOTHING but with no conflict - should insert
    let affected = db
        .execute(
            "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25) ON CONFLICT (id) DO NOTHING",
        )
        .expect("upsert failed");
    assert_eq!(affected, 1, "Should insert one row when no conflict");

    // Verify both rows exist
    let result = db.query("SELECT * FROM users ORDER BY id").expect("query failed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_insert_on_conflict_do_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data
    db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .expect("initial insert failed");

    // Insert with ON CONFLICT DO UPDATE - should update the existing row
    let affected = db
        .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 35) ON CONFLICT (id) DO UPDATE SET age = 35")
        .expect("upsert failed");
    assert_eq!(affected, 1, "Should update one row when conflict exists");

    // Verify data was updated
    let result = db.query("SELECT * FROM users WHERE id = 1").expect("query failed");
    assert_eq!(result.len(), 1);

    let age_idx = result.column_index("age").expect("age column not found");
    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(35)));
}

#[test]
fn test_insert_on_conflict_do_update_multiple_columns() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data
    db.execute(
        "INSERT INTO users (id, name, age, email) VALUES (1, 'Alice', 30, 'old@example.com')",
    )
    .expect("initial insert failed");

    // Update multiple columns on conflict
    let affected = db
        .execute("INSERT INTO users (id, name, age, email) VALUES (1, 'Alice', 35, 'new@example.com') ON CONFLICT (id) DO UPDATE SET age = 35, email = 'new@example.com'")
        .expect("upsert failed");
    assert_eq!(affected, 1);

    // Verify both columns were updated
    let result = db.query("SELECT * FROM users WHERE id = 1").expect("query failed");
    assert_eq!(result.len(), 1);

    let age_idx = result.column_index("age").expect("age column not found");
    let email_idx = result.column_index("email").expect("email column not found");

    assert_eq!(result.rows()[0].get(age_idx), Some(&Value::Int(35)));
    assert_eq!(
        result.rows()[0].get(email_idx),
        Some(&Value::String("new@example.com".to_string()))
    );
}

#[test]
fn test_insert_on_conflict_multiple_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data
    db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .expect("initial insert failed");
    db.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
        .expect("initial insert failed");

    // Insert multiple rows, some conflicting, some new
    // Row id=1 conflicts (skip), id=3 is new (insert)
    let affected = db
        .execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice Updated', 35), (3, 'Charlie', 40) ON CONFLICT (id) DO NOTHING")
        .expect("upsert failed");
    assert_eq!(affected, 1, "Should insert one new row and skip the conflicting one");

    // Verify total count
    let result = db.query("SELECT * FROM users ORDER BY id").expect("query failed");
    assert_eq!(result.len(), 3);
}

#[test]
fn test_insert_on_conflict_multi_column_key() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert initial data with composite key
    db.execute("INSERT INTO orders (user_id, product_id, quantity) VALUES (1, 100, 5)")
        .expect("initial insert failed");

    // Try to insert with same composite key - should skip
    let affected = db
        .execute("INSERT INTO orders (user_id, product_id, quantity) VALUES (1, 100, 10) ON CONFLICT (user_id, product_id) DO NOTHING")
        .expect("upsert failed");
    assert_eq!(affected, 0, "Should skip when composite key conflicts");

    // Insert with different composite key - should insert
    let affected = db
        .execute("INSERT INTO orders (user_id, product_id, quantity) VALUES (1, 101, 3) ON CONFLICT (user_id, product_id) DO NOTHING")
        .expect("upsert failed");
    assert_eq!(affected, 1, "Should insert when composite key doesn't conflict");

    // Verify total count
    let result = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(result.len(), 2);
}

// ============================================================================
// SQL MERGE Statement Tests
// ============================================================================

#[test]
fn test_merge_insert_when_not_matched() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with initial data
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, value) VALUES (2, 'Bob', 200)")
        .expect("insert failed");

    // Create source table with new data
    db.execute("INSERT INTO source (id, name, value) VALUES (3, 'Charlie', 300)")
        .expect("insert failed");

    // MERGE: insert when not matched
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN NOT MATCHED THEN INSERT (id, name, value) VALUES (source.id, source.name, source.value)",
        )
        .expect("merge failed");
    assert_eq!(affected, 1, "Should insert one new row");

    // Verify target has 3 rows now
    let result = db.query("SELECT * FROM target ORDER BY id").expect("query failed");
    assert_eq!(result.len(), 3);

    // Verify Charlie was inserted
    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[2].get(name_idx), Some(&Value::String("Charlie".to_string())));
}

#[test]
fn test_merge_update_when_matched() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with initial data
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, value) VALUES (2, 'Bob', 200)")
        .expect("insert failed");

    // Create source table with updated data
    db.execute("INSERT INTO source (id, name, value) VALUES (1, 'Alice', 150)")
        .expect("insert failed");

    // MERGE: update when matched
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED THEN UPDATE SET value = source.value",
        )
        .expect("merge failed");
    assert_eq!(affected, 1, "Should update one row");

    // Verify Alice's value was updated
    let result = db.query("SELECT * FROM target WHERE id = 1").expect("query failed");
    assert_eq!(result.len(), 1);

    let value_idx = result.column_index("value").expect("value column not found");
    assert_eq!(result.rows()[0].get(value_idx), Some(&Value::Int(150)));
}

#[test]
fn test_merge_delete_when_matched() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with initial data
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, value) VALUES (2, 'Bob', 200)")
        .expect("insert failed");

    // Create source table with matching id (marking for deletion)
    db.execute("INSERT INTO source (id) VALUES (1)").expect("insert failed");

    // MERGE: delete when matched
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED THEN DELETE",
        )
        .expect("merge failed");
    assert_eq!(affected, 1, "Should delete one row");

    // Verify only Bob remains
    let result = db.query("SELECT * FROM target").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Bob".to_string())));
}

#[test]
fn test_merge_combined_update_and_insert() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with initial data
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, value) VALUES (2, 'Bob', 200)")
        .expect("insert failed");

    // Create source table with:
    // - id=1 exists in target (should update)
    // - id=3 doesn't exist in target (should insert)
    db.execute("INSERT INTO source (id, name, value) VALUES (1, 'Alice', 150)")
        .expect("insert failed");
    db.execute("INSERT INTO source (id, name, value) VALUES (3, 'Charlie', 300)")
        .expect("insert failed");

    // MERGE: update when matched, insert when not matched
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED THEN UPDATE SET value = source.value
             WHEN NOT MATCHED THEN INSERT (id, name, value) VALUES (source.id, source.name, source.value)",
        )
        .expect("merge failed");
    assert_eq!(affected, 2, "Should affect 2 rows (1 update + 1 insert)");

    // Verify target has 3 rows
    let result = db.query("SELECT * FROM target ORDER BY id").expect("query failed");
    assert_eq!(result.len(), 3);

    // Verify Alice's value was updated
    let value_idx = result.column_index("value").expect("value column not found");
    assert_eq!(result.rows()[0].get(value_idx), Some(&Value::Int(150)));

    // Verify Charlie was inserted
    let name_idx = result.column_index("name").expect("name column not found");
    assert_eq!(result.rows()[2].get(name_idx), Some(&Value::String("Charlie".to_string())));
    assert_eq!(result.rows()[2].get(value_idx), Some(&Value::Int(300)));
}

#[test]
fn test_merge_with_conditional_when_matched() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, value) VALUES (2, 'Bob', 200)")
        .expect("insert failed");

    // Create source table with values to conditionally update
    db.execute("INSERT INTO source (id, value) VALUES (1, 50)").expect("insert failed");
    db.execute("INSERT INTO source (id, value) VALUES (2, 300)").expect("insert failed");

    // MERGE: update only when source value > 100
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED AND source.value > 100 THEN UPDATE SET value = source.value",
        )
        .expect("merge failed");
    assert_eq!(affected, 1, "Should update only one row (where source.value > 100)");

    // Verify Alice's value unchanged (source.value = 50 <= 100)
    let result = db.query("SELECT * FROM target WHERE id = 1").expect("query failed");
    let value_idx = result.column_index("value").expect("value column not found");
    assert_eq!(result.rows()[0].get(value_idx), Some(&Value::Int(100)));

    // Verify Bob's value was updated (source.value = 300 > 100)
    let result = db.query("SELECT * FROM target WHERE id = 2").expect("query failed");
    let value_idx = result.column_index("value").expect("value column not found");
    assert_eq!(result.rows()[0].get(value_idx), Some(&Value::Int(300)));
}

#[test]
fn test_merge_with_subquery_source() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO target (id, name, value) VALUES (1, 'Alice', 100)")
        .expect("insert failed");

    // Create staging table
    db.execute("INSERT INTO staging (id, name, value, active) VALUES (1, 'Alice', 150, true)")
        .expect("insert failed");
    db.execute("INSERT INTO staging (id, name, value, active) VALUES (2, 'Bob', 200, true)")
        .expect("insert failed");
    db.execute("INSERT INTO staging (id, name, value, active) VALUES (3, 'Charlie', 300, false)")
        .expect("insert failed");

    // MERGE using a subquery as source (only active rows)
    let affected = db
        .execute(
            "MERGE INTO target
             USING (SELECT * FROM staging WHERE active = true) AS source ON target.id = source.id
             WHEN MATCHED THEN UPDATE SET value = source.value
             WHEN NOT MATCHED THEN INSERT (id, name, value) VALUES (source.id, source.name, source.value)",
        )
        .expect("merge failed");
    assert_eq!(
        affected, 2,
        "Should affect 2 rows (1 update + 1 insert, excluding inactive Charlie)"
    );

    // Verify target has 2 rows (Alice updated, Bob inserted, Charlie not included)
    let result = db.query("SELECT * FROM target ORDER BY id").expect("query failed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_merge_no_matching_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO target (id, name) VALUES (1, 'Alice')").expect("insert failed");

    // Create source table with no matching ids
    db.execute("INSERT INTO source (id, name) VALUES (2, 'Bob')").expect("insert failed");

    // MERGE: only update clause, no matching rows
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED THEN UPDATE SET name = source.name",
        )
        .expect("merge failed");
    assert_eq!(affected, 0, "Should not affect any rows when no matches");

    // Target unchanged
    let result = db.query("SELECT * FROM target").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_merge_empty_source() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO target (id, name) VALUES (1, 'Alice')").expect("insert failed");

    // Create empty source table
    db.execute("CREATE TABLE source (id BIGINT, name TEXT)").expect("create table failed");

    // MERGE with empty source
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED THEN UPDATE SET name = source.name
             WHEN NOT MATCHED THEN INSERT (id, name) VALUES (source.id, source.name)",
        )
        .expect("merge failed");
    assert_eq!(affected, 0, "Should not affect any rows with empty source");

    // Target unchanged
    let result = db.query("SELECT * FROM target").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_merge_multiple_when_matched_clauses() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with status
    db.execute("INSERT INTO target (id, name, status, value) VALUES (1, 'Alice', 'active', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO target (id, name, status, value) VALUES (2, 'Bob', 'inactive', 200)")
        .expect("insert failed");

    // Create source table
    db.execute("INSERT INTO source (id, value) VALUES (1, 150)").expect("insert failed");
    db.execute("INSERT INTO source (id, value) VALUES (2, 250)").expect("insert failed");

    // MERGE with multiple WHEN MATCHED clauses (first matching wins)
    // Only update active rows, delete inactive rows
    let affected = db
        .execute(
            "MERGE INTO target
             USING source ON target.id = source.id
             WHEN MATCHED AND status = 'active' THEN UPDATE SET value = source.value
             WHEN MATCHED AND status = 'inactive' THEN DELETE",
        )
        .expect("merge failed");
    assert_eq!(affected, 2, "Should affect 2 rows (1 update + 1 delete)");

    // Verify only Alice remains with updated value
    let result = db.query("SELECT * FROM target").expect("query failed");
    assert_eq!(result.len(), 1);

    let name_idx = result.column_index("name").expect("name column not found");
    let value_idx = result.column_index("value").expect("value column not found");
    assert_eq!(result.rows()[0].get(name_idx), Some(&Value::String("Alice".to_string())));
    assert_eq!(result.rows()[0].get(value_idx), Some(&Value::Int(150)));
}

// ============================================================================
// Multi-table DML Tests (UPDATE FROM, DELETE USING)
// ============================================================================

#[test]
fn test_update_from_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with data
    db.execute("INSERT INTO orders (order_id, customer_id, status) VALUES (1, 100, 'pending')")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id, status) VALUES (2, 101, 'pending')")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id, status) VALUES (3, 102, 'pending')")
        .expect("insert failed");

    // Create source table with update values
    db.execute("INSERT INTO customers (customer_id, tier) VALUES (100, 'gold')")
        .expect("insert failed");
    db.execute("INSERT INTO customers (customer_id, tier) VALUES (101, 'silver')")
        .expect("insert failed");

    // UPDATE ... FROM to update orders based on customer tier
    let affected = db
        .execute(
            "UPDATE orders SET status = 'prioritized'
             FROM customers
             WHERE orders.customer_id = customers.customer_id AND customers.tier = 'gold'",
        )
        .expect("update from failed");
    assert_eq!(affected, 1);

    // Verify results
    let result =
        db.query("SELECT * FROM orders WHERE status = 'prioritized'").expect("query failed");
    assert_eq!(result.len(), 1);

    let order_id_idx = result.column_index("order_id").expect("order_id column not found");
    assert_eq!(result.rows()[0].get(order_id_idx), Some(&Value::Int(1)));
}

#[test]
fn test_update_from_with_source_column_in_assignment() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target and source tables
    db.execute("INSERT INTO products (product_id, price) VALUES (1, 100)").expect("insert failed");
    db.execute("INSERT INTO products (product_id, price) VALUES (2, 200)").expect("insert failed");

    db.execute("INSERT INTO price_updates (product_id, new_price) VALUES (1, 120)")
        .expect("insert failed");
    db.execute("INSERT INTO price_updates (product_id, new_price) VALUES (2, 180)")
        .expect("insert failed");

    // UPDATE ... FROM with value from source table
    let affected = db
        .execute(
            "UPDATE products SET price = price_updates.new_price
             FROM price_updates
             WHERE products.product_id = price_updates.product_id",
        )
        .expect("update from failed");
    assert_eq!(affected, 2);

    // Verify results
    let result = db.query("SELECT * FROM products ORDER BY product_id").expect("query failed");
    assert_eq!(result.len(), 2);

    let price_idx = result.column_index("price").expect("price column not found");
    assert_eq!(result.rows()[0].get(price_idx), Some(&Value::Int(120)));
    assert_eq!(result.rows()[1].get(price_idx), Some(&Value::Int(180)));
}

#[test]
fn test_update_from_no_match() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO orders (order_id, status) VALUES (1, 'pending')")
        .expect("insert failed");

    // Create source table with no matching rows
    db.execute("INSERT INTO updates (order_id, new_status) VALUES (999, 'shipped')")
        .expect("insert failed");

    // UPDATE ... FROM with no matching rows
    let affected = db
        .execute(
            "UPDATE orders SET status = updates.new_status
             FROM updates
             WHERE orders.order_id = updates.order_id",
        )
        .expect("update from failed");
    assert_eq!(affected, 0);

    // Verify original data unchanged
    let result = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(result.len(), 1);

    let status_idx = result.column_index("status").expect("status column not found");
    assert_eq!(result.rows()[0].get(status_idx), Some(&Value::String("pending".to_string())));
}

#[test]
fn test_delete_using_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table with data
    db.execute("INSERT INTO orders (order_id, customer_id) VALUES (1, 100)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id) VALUES (2, 101)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id) VALUES (3, 102)")
        .expect("insert failed");

    // Create source table identifying rows to delete
    db.execute("INSERT INTO cancelled_customers (customer_id) VALUES (100)")
        .expect("insert failed");
    db.execute("INSERT INTO cancelled_customers (customer_id) VALUES (102)")
        .expect("insert failed");

    // DELETE ... USING
    let affected = db
        .execute(
            "DELETE FROM orders
             USING cancelled_customers
             WHERE orders.customer_id = cancelled_customers.customer_id",
        )
        .expect("delete using failed");
    assert_eq!(affected, 2);

    // Verify results
    let result = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(result.len(), 1);

    let customer_id_idx = result.column_index("customer_id").expect("customer_id column not found");
    assert_eq!(result.rows()[0].get(customer_id_idx), Some(&Value::Int(101)));
}

#[test]
fn test_delete_using_no_match() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target table
    db.execute("INSERT INTO orders (order_id, customer_id) VALUES (1, 100)")
        .expect("insert failed");

    // Create source table with no matching rows
    db.execute("INSERT INTO cancelled (customer_id) VALUES (999)").expect("insert failed");

    // DELETE ... USING with no matching rows
    let affected = db
        .execute(
            "DELETE FROM orders
             USING cancelled
             WHERE orders.customer_id = cancelled.customer_id",
        )
        .expect("delete using failed");
    assert_eq!(affected, 0);

    // Verify original data unchanged
    let result = db.query("SELECT * FROM orders").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_update_from_multiple_source_tables() {
    let db = Database::in_memory().expect("failed to create db");

    // Create target and source tables
    db.execute("INSERT INTO items (item_id, category_id, price) VALUES (1, 10, 100)")
        .expect("insert failed");
    db.execute("INSERT INTO items (item_id, category_id, price) VALUES (2, 20, 200)")
        .expect("insert failed");

    db.execute("INSERT INTO categories (category_id, discount) VALUES (10, 0.1)")
        .expect("insert failed");

    // UPDATE ... FROM with multiple source tables (cross join)
    // Note: This is a basic test - in practice you'd want to join them properly
    let affected = db
        .execute(
            "UPDATE items SET price = 90
             FROM categories
             WHERE items.category_id = categories.category_id AND categories.discount > 0",
        )
        .expect("update from failed");
    assert_eq!(affected, 1);

    // Verify results
    let result = db.query("SELECT * FROM items WHERE item_id = 1").expect("query failed");
    let price_idx = result.column_index("price").expect("price column not found");
    assert_eq!(result.rows()[0].get(price_idx), Some(&Value::Int(90)));
}

#[test]
#[ignore = "Multiple source tables in DELETE USING require physical plan execution for cross join"]
fn test_delete_using_multiple_source_tables() {
    let db = Database::in_memory().expect("failed to create db");

    // Create tables
    db.execute("INSERT INTO orders (order_id, customer_id, product_id) VALUES (1, 100, 1)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id, product_id) VALUES (2, 100, 2)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (order_id, customer_id, product_id) VALUES (3, 101, 1)")
        .expect("insert failed");

    db.execute("INSERT INTO bad_customers (customer_id) VALUES (100)").expect("insert failed");
    db.execute("INSERT INTO bad_products (product_id) VALUES (1)").expect("insert failed");

    // DELETE ... USING with multiple tables
    // This should delete orders where customer is bad AND product is bad
    let affected = db
        .execute(
            "DELETE FROM orders
             USING bad_customers, bad_products
             WHERE orders.customer_id = bad_customers.customer_id
               AND orders.product_id = bad_products.product_id",
        )
        .expect("delete using failed");
    assert_eq!(affected, 1); // Only order 1 matches both conditions

    // Verify results
    let result = db.query("SELECT * FROM orders ORDER BY order_id").expect("query failed");
    assert_eq!(result.len(), 2);
}
