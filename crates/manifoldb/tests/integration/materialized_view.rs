//! Materialized view integration tests.
//!
//! Tests for CREATE MATERIALIZED VIEW, DROP MATERIALIZED VIEW, and REFRESH MATERIALIZED VIEW.

use manifoldb::Database;

// ============================================================================
// CREATE MATERIALIZED VIEW Tests
// ============================================================================

#[test]
fn test_create_materialized_view_basic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create some test data first
    db.execute("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT, active BOOLEAN)")
        .expect("create table failed");
    db.execute("INSERT INTO users (id, name, active) VALUES (1, 'Alice', true)")
        .expect("insert failed");
    db.execute("INSERT INTO users (id, name, active) VALUES (2, 'Bob', false)")
        .expect("insert failed");
    db.execute("INSERT INTO users (id, name, active) VALUES (3, 'Carol', true)")
        .expect("insert failed");

    // Create materialized view
    let affected = db
        .execute("CREATE MATERIALIZED VIEW active_users AS SELECT id, name FROM users WHERE active = true")
        .expect("create materialized view failed");
    assert_eq!(affected, 0);

    // REFRESH the view to populate it
    let refreshed = db.execute("REFRESH MATERIALIZED VIEW active_users").expect("refresh failed");
    assert_eq!(refreshed, 2); // 2 active users
}

#[test]
fn test_create_materialized_view_if_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE users (id BIGINT PRIMARY KEY, name TEXT)")
        .expect("create table failed");
    db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')").expect("insert failed");

    // Create the materialized view
    db.execute("CREATE MATERIALIZED VIEW mv AS SELECT * FROM users").expect("create mv failed");

    // Creating again with IF NOT EXISTS should succeed
    let affected = db
        .execute("CREATE MATERIALIZED VIEW IF NOT EXISTS mv AS SELECT * FROM users")
        .expect("create mv if not exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_query_materialized_view_returns_cached_data() {
    let db = Database::in_memory().expect("failed to create db");

    // Create test data
    db.execute("CREATE TABLE products (id BIGINT PRIMARY KEY, name TEXT, price BIGINT)")
        .expect("create table failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 100)")
        .expect("insert failed");
    db.execute("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 200)")
        .expect("insert failed");

    // Create and refresh materialized view
    db.execute("CREATE MATERIALIZED VIEW product_list AS SELECT id, name, price FROM products")
        .expect("create mv failed");
    db.execute("REFRESH MATERIALIZED VIEW product_list").expect("refresh failed");

    // Query the materialized view
    let result = db.query("SELECT * FROM product_list").expect("query mv failed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_materialized_view_caches_data_not_live() {
    let db = Database::in_memory().expect("failed to create db");

    // Create test data
    db.execute("CREATE TABLE counters (id BIGINT PRIMARY KEY, value BIGINT)")
        .expect("create table failed");
    db.execute("INSERT INTO counters (id, value) VALUES (1, 10)").expect("insert failed");

    // Create and refresh materialized view
    db.execute("CREATE MATERIALIZED VIEW counter_snapshot AS SELECT id, value FROM counters")
        .expect("create mv failed");
    db.execute("REFRESH MATERIALIZED VIEW counter_snapshot").expect("refresh failed");

    // Verify initial state
    let result = db.query("SELECT * FROM counter_snapshot").expect("query mv failed");
    assert_eq!(result.len(), 1);

    // Modify the underlying data
    db.execute("UPDATE counters SET value = 20 WHERE id = 1").expect("update failed");
    db.execute("INSERT INTO counters (id, value) VALUES (2, 30)").expect("insert failed");

    // Verify the underlying table has 2 rows now
    let underlying_result = db.query("SELECT * FROM counters").expect("query counters failed");
    assert_eq!(underlying_result.len(), 2, "Underlying table should have 2 rows");

    // Query materialized view - should still show OLD cached data
    let result_after_update =
        db.query("SELECT * FROM counter_snapshot").expect("query mv after update failed");
    assert_eq!(result_after_update.len(), 1); // Still 1 row, not 2

    // REFRESH the view
    let refresh_count = db
        .execute("REFRESH MATERIALIZED VIEW counter_snapshot")
        .expect("refresh after update failed");
    assert_eq!(refresh_count, 2, "REFRESH should return row count of 2");

    // Now it should show the NEW data
    let result_after_refresh =
        db.query("SELECT * FROM counter_snapshot").expect("query mv after refresh failed");
    assert_eq!(result_after_refresh.len(), 2); // Now 2 rows
}

#[test]
fn test_drop_materialized_view_basic() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE data (id BIGINT PRIMARY KEY, value TEXT)")
        .expect("create table failed");
    db.execute("INSERT INTO data (id, value) VALUES (1, 'test')").expect("insert failed");

    db.execute("CREATE MATERIALIZED VIEW mv AS SELECT * FROM data").expect("create mv failed");
    db.execute("REFRESH MATERIALIZED VIEW mv").expect("refresh failed");

    // Verify it exists
    let result = db.query("SELECT * FROM mv").expect("query mv failed");
    assert_eq!(result.len(), 1);

    // Drop the materialized view
    let affected = db.execute("DROP MATERIALIZED VIEW mv").expect("drop mv failed");
    assert_eq!(affected, 0);

    // After drop, querying returns empty (as there are no entities with label 'mv')
    // The materialized view no longer exists, so it falls through to entity scan
    let result_after_drop = db.query("SELECT * FROM mv").expect("query after drop");
    assert_eq!(result_after_drop.len(), 0);

    // Refreshing should fail since the view no longer exists
    let refresh_result = db.execute("REFRESH MATERIALIZED VIEW mv");
    assert!(refresh_result.is_err());
}

#[test]
fn test_drop_materialized_view_if_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Drop non-existent view with IF EXISTS should succeed
    let affected =
        db.execute("DROP MATERIALIZED VIEW IF EXISTS nonexistent").expect("drop if exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_materialized_view_with_simple_filter() {
    let db = Database::in_memory().expect("failed to create db");

    // Create orders table
    db.execute("CREATE TABLE orders (id BIGINT PRIMARY KEY, customer_id BIGINT, amount BIGINT)")
        .expect("create table failed");
    db.execute("INSERT INTO orders (id, customer_id, amount) VALUES (1, 100, 50)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, customer_id, amount) VALUES (2, 100, 75)")
        .expect("insert failed");
    db.execute("INSERT INTO orders (id, customer_id, amount) VALUES (3, 200, 100)")
        .expect("insert failed");

    // Create materialized view with simple filter
    db.execute(
        "CREATE MATERIALIZED VIEW big_orders AS SELECT id, customer_id, amount FROM orders WHERE amount >= 75",
    )
    .expect("create mv failed");
    db.execute("REFRESH MATERIALIZED VIEW big_orders").expect("refresh failed");

    // Query the filtered view
    let result = db.query("SELECT * FROM big_orders").expect("query mv failed");
    assert_eq!(result.len(), 2); // 2 orders with amount >= 75
}

#[test]
fn test_refresh_materialized_view_nonexistent_fails() {
    let db = Database::in_memory().expect("failed to create db");

    // Refreshing non-existent view should fail
    let result = db.execute("REFRESH MATERIALIZED VIEW nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_query_unrefreshed_materialized_view_fails() {
    let db = Database::in_memory().expect("failed to create db");

    db.execute("CREATE TABLE data (id BIGINT PRIMARY KEY)").expect("create table failed");
    db.execute("INSERT INTO data (id) VALUES (1)").expect("insert failed");

    // Create but don't refresh
    db.execute("CREATE MATERIALIZED VIEW mv AS SELECT * FROM data").expect("create mv failed");

    // Querying unrefreshed view should fail
    let result = db.query("SELECT * FROM mv");
    assert!(result.is_err());
}
