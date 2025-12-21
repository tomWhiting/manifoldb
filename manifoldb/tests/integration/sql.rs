//! SQL execution integration tests.
//!
//! Tests end-to-end SQL query and statement execution against storage.

use manifoldb::{Database, Value};

/// Helper to get a value by column name from a query result.
fn get_by_name<'a>(
    result: &'a manifoldb::QueryResult,
    row_idx: usize,
    col_name: &str,
) -> Option<&'a Value> {
    result
        .column_index(col_name)
        .and_then(|col_idx| result.rows().get(row_idx))
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
// Error Cases
// ============================================================================

#[test]
fn test_invalid_sql_syntax() {
    let db = Database::in_memory().expect("failed to create db");

    // This should fail to parse
    let result = db.query("SELEKT * FORM users");
    assert!(result.is_err());
}
