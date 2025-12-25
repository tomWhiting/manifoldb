//! Index maintenance integration tests.
//!
//! Tests that verify indexes stay in sync during INSERT, UPDATE, and DELETE operations.

#![allow(dead_code, unused_variables)]

use manifoldb::Database;

// ============================================================================
// INSERT Index Maintenance Tests
// ============================================================================

#[test]
fn test_insert_updates_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE products (id BIGINT, name TEXT, price INTEGER)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_products_name ON products (name)").expect("create index failed");

    // Insert data - index should be updated automatically
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 100)")
        .expect("insert 1 failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 200)")
        .expect("insert 2 failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (3, 'Widget', 150)")
        .expect("insert 3 failed");

    // Query using the indexed column - should find matching rows
    let result = db.query("SELECT * FROM products WHERE name = 'Widget'").expect("query failed");
    assert_eq!(result.len(), 2);

    let result = db.query("SELECT * FROM products WHERE name = 'Gadget'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_insert_without_indexed_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE nullable_test (id BIGINT, optional_field TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_nullable_optional ON nullable_test (optional_field)")
        .expect("create index failed");

    // Insert rows without the indexed column - should not create index entries
    db.execute("INSERT INTO nullable_test (id) VALUES (1)").expect("insert 1 failed");
    db.execute("INSERT INTO nullable_test (id, optional_field) VALUES (2, 'has_value')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO nullable_test (id) VALUES (3)").expect("insert 3 failed");

    // Query by the indexed column - should find only the row with the value
    let result = db
        .query("SELECT * FROM nullable_test WHERE optional_field = 'has_value'")
        .expect("query failed");
    assert_eq!(result.len(), 1);

    // Total rows should be 3
    let all = db.query("SELECT * FROM nullable_test").expect("query failed");
    assert_eq!(all.len(), 3);
}

#[test]
fn test_insert_multiple_indexed_columns() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with multiple indexes
    db.execute("CREATE TABLE multi_idx (id BIGINT, category TEXT, status TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_multi_category ON multi_idx (category)")
        .expect("create category index failed");
    db.execute("CREATE INDEX idx_multi_status ON multi_idx (status)")
        .expect("create status index failed");

    // Insert data
    db.execute("INSERT INTO multi_idx (id, category, status) VALUES (1, 'electronics', 'active')")
        .expect("insert 1 failed");
    db.execute(
        "INSERT INTO multi_idx (id, category, status) VALUES (2, 'electronics', 'inactive')",
    )
    .expect("insert 2 failed");
    db.execute("INSERT INTO multi_idx (id, category, status) VALUES (3, 'clothing', 'active')")
        .expect("insert 3 failed");

    // Query using first index
    let result =
        db.query("SELECT * FROM multi_idx WHERE category = 'electronics'").expect("query failed");
    assert_eq!(result.len(), 2);

    // Query using second index
    let result = db.query("SELECT * FROM multi_idx WHERE status = 'active'").expect("query failed");
    assert_eq!(result.len(), 2);
}

// ============================================================================
// UPDATE Index Maintenance Tests
// ============================================================================

#[test]
fn test_update_updates_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE users (id BIGINT, email TEXT, status TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_users_email ON users (email)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO users (id, email, status) VALUES (1, 'old@example.com', 'active')")
        .expect("insert failed");

    // Verify initial state
    let result =
        db.query("SELECT * FROM users WHERE email = 'old@example.com'").expect("query failed");
    assert_eq!(result.len(), 1);

    // Update the indexed column
    db.execute("UPDATE users SET email = 'new@example.com' WHERE id = 1").expect("update failed");

    // Old value should not find anything
    let result =
        db.query("SELECT * FROM users WHERE email = 'old@example.com'").expect("query failed");
    assert_eq!(result.len(), 0);

    // New value should find the row
    let result =
        db.query("SELECT * FROM users WHERE email = 'new@example.com'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_update_non_indexed_column_preserves_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE orders (id BIGINT, customer TEXT, total INTEGER)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_orders_customer ON orders (customer)")
        .expect("create index failed");

    // Insert data
    db.execute("INSERT INTO orders (id, customer, total) VALUES (1, 'Alice', 100)")
        .expect("insert failed");

    // Update a non-indexed column
    db.execute("UPDATE orders SET total = 200 WHERE id = 1").expect("update failed");

    // Index should still work correctly
    let result = db.query("SELECT * FROM orders WHERE customer = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_update_multiple_rows() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE tasks (id BIGINT, priority TEXT, status TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_tasks_status ON tasks (status)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO tasks (id, priority, status) VALUES (1, 'high', 'pending')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO tasks (id, priority, status) VALUES (2, 'low', 'pending')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO tasks (id, priority, status) VALUES (3, 'high', 'completed')")
        .expect("insert 3 failed");

    // Verify initial state
    let result = db.query("SELECT * FROM tasks WHERE status = 'pending'").expect("query failed");
    assert_eq!(result.len(), 2);

    // Update multiple rows
    db.execute("UPDATE tasks SET status = 'in_progress' WHERE priority = 'high'")
        .expect("update failed");

    // Check the indexes are correct
    let result = db.query("SELECT * FROM tasks WHERE status = 'pending'").expect("query failed");
    assert_eq!(result.len(), 1); // Only low priority task is pending now

    let result =
        db.query("SELECT * FROM tasks WHERE status = 'in_progress'").expect("query failed");
    assert_eq!(result.len(), 2); // Both high priority tasks are in progress now
}

// ============================================================================
// DELETE Index Maintenance Tests
// ============================================================================

#[test]
fn test_delete_removes_from_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE articles (id BIGINT, title TEXT, author TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_articles_author ON articles (author)")
        .expect("create index failed");

    // Insert data
    db.execute("INSERT INTO articles (id, title, author) VALUES (1, 'Post 1', 'Alice')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO articles (id, title, author) VALUES (2, 'Post 2', 'Bob')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO articles (id, title, author) VALUES (3, 'Post 3', 'Alice')")
        .expect("insert 3 failed");

    // Verify initial state
    let result = db.query("SELECT * FROM articles WHERE author = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 2);

    // Delete one of Alice's articles
    db.execute("DELETE FROM articles WHERE id = 1").expect("delete failed");

    // Index should be updated
    let result = db.query("SELECT * FROM articles WHERE author = 'Alice'").expect("query failed");
    assert_eq!(result.len(), 1);

    // Bob's article should still be indexed
    let result = db.query("SELECT * FROM articles WHERE author = 'Bob'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_delete_all_rows_clears_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE temp_items (id BIGINT, name TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_temp_name ON temp_items (name)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO temp_items (id, name) VALUES (1, 'item_1')").expect("insert 1 failed");
    db.execute("INSERT INTO temp_items (id, name) VALUES (2, 'item_2')").expect("insert 2 failed");

    // Verify data exists
    let result = db.query("SELECT * FROM temp_items WHERE name = 'item_1'").expect("query failed");
    assert_eq!(result.len(), 1);

    // Delete all rows
    db.execute("DELETE FROM temp_items").expect("delete failed");

    // Index should be empty
    let result = db.query("SELECT * FROM temp_items WHERE name = 'item_1'").expect("query failed");
    assert_eq!(result.len(), 0);

    let result = db.query("SELECT * FROM temp_items WHERE name = 'item_2'").expect("query failed");
    assert_eq!(result.len(), 0);
}

#[test]
fn test_delete_with_filter() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE logs (id BIGINT, level TEXT, message TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_logs_level ON logs (level)").expect("create index failed");

    // Insert data
    db.execute("INSERT INTO logs (id, level, message) VALUES (1, 'ERROR', 'Error 1')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO logs (id, level, message) VALUES (2, 'INFO', 'Info 1')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO logs (id, level, message) VALUES (3, 'ERROR', 'Error 2')")
        .expect("insert 3 failed");
    db.execute("INSERT INTO logs (id, level, message) VALUES (4, 'WARN', 'Warn 1')")
        .expect("insert 4 failed");

    // Delete only ERROR logs
    db.execute("DELETE FROM logs WHERE level = 'ERROR'").expect("delete failed");

    // Error logs should be gone
    let result = db.query("SELECT * FROM logs WHERE level = 'ERROR'").expect("query failed");
    assert_eq!(result.len(), 0);

    // Other logs should remain
    let result = db.query("SELECT * FROM logs WHERE level = 'INFO'").expect("query failed");
    assert_eq!(result.len(), 1);

    let result = db.query("SELECT * FROM logs WHERE level = 'WARN'").expect("query failed");
    assert_eq!(result.len(), 1);
}

// ============================================================================
// Combined Mutation Tests
// ============================================================================

#[test]
fn test_insert_update_delete_sequence() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE inventory (id BIGINT, sku TEXT, quantity INTEGER)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_inventory_sku ON inventory (sku)").expect("create index failed");

    // 1. Insert initial data
    db.execute("INSERT INTO inventory (id, sku, quantity) VALUES (1, 'SKU-001', 10)")
        .expect("insert 1 failed");
    db.execute("INSERT INTO inventory (id, sku, quantity) VALUES (2, 'SKU-002', 20)")
        .expect("insert 2 failed");

    let result = db.query("SELECT * FROM inventory WHERE sku = 'SKU-001'").expect("query failed");
    assert_eq!(result.len(), 1);

    // 2. Update the SKU
    db.execute("UPDATE inventory SET sku = 'SKU-001-A' WHERE id = 1").expect("update failed");

    let result = db.query("SELECT * FROM inventory WHERE sku = 'SKU-001'").expect("query failed");
    assert_eq!(result.len(), 0);

    let result = db.query("SELECT * FROM inventory WHERE sku = 'SKU-001-A'").expect("query failed");
    assert_eq!(result.len(), 1);

    // 3. Delete the updated row
    db.execute("DELETE FROM inventory WHERE id = 1").expect("delete failed");

    let result = db.query("SELECT * FROM inventory WHERE sku = 'SKU-001-A'").expect("query failed");
    assert_eq!(result.len(), 0);

    // 4. Original SKU-002 should still be indexed
    let result = db.query("SELECT * FROM inventory WHERE sku = 'SKU-002'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_index_consistency_after_many_operations() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table and index
    db.execute("CREATE TABLE counters (id BIGINT, name TEXT, status TEXT)")
        .expect("create table failed");
    db.execute("CREATE INDEX idx_counters_name ON counters (name)").expect("create index failed");

    // Insert counters with different statuses
    db.execute("INSERT INTO counters (id, name, status) VALUES (1, 'counter_1', 'pending')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (2, 'counter_2', 'pending')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (3, 'counter_3', 'pending')")
        .expect("insert 3 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (4, 'counter_4', 'active')")
        .expect("insert 4 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (5, 'counter_5', 'active')")
        .expect("insert 5 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (6, 'counter_6', 'complete')")
        .expect("insert 6 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (7, 'counter_7', 'complete')")
        .expect("insert 7 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (8, 'counter_8', 'archived')")
        .expect("insert 8 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (9, 'counter_9', 'archived')")
        .expect("insert 9 failed");
    db.execute("INSERT INTO counters (id, name, status) VALUES (10, 'counter_10', 'archived')")
        .expect("insert 10 failed");

    // Verify all 10 were inserted
    let all = db.query("SELECT * FROM counters").expect("query failed");
    assert_eq!(all.len(), 10);

    // Update pending counters to 'renamed'
    let updated = db
        .execute("UPDATE counters SET name = 'renamed' WHERE status = 'pending'")
        .expect("update failed");
    assert_eq!(updated, 3, "should update 3 pending rows");

    // Delete archived counters
    let deleted =
        db.execute("DELETE FROM counters WHERE status = 'archived'").expect("delete failed");
    assert_eq!(deleted, 3, "should delete 3 archived rows");

    // Verify index consistency
    let result = db.query("SELECT * FROM counters WHERE name = 'renamed'").expect("query failed");
    assert_eq!(result.len(), 3, "should find 3 renamed rows");

    let result = db.query("SELECT * FROM counters WHERE name = 'counter_5'").expect("query failed");
    assert_eq!(result.len(), 1);

    let result = db.query("SELECT * FROM counters WHERE name = 'counter_9'").expect("query failed");
    assert_eq!(result.len(), 0); // was deleted

    // Total count should be 7 (10 - 3 deleted)
    let all = db.query("SELECT * FROM counters").expect("query failed");
    assert_eq!(all.len(), 7);
}
