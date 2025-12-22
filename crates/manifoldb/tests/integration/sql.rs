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
