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
// ALTER TABLE Tests
// ============================================================================

#[test]
fn test_alter_table_add_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE alter_test (id BIGINT, name TEXT)").expect("create table failed");

    // Insert some data
    db.execute("INSERT INTO alter_test (id, name) VALUES (1, 'Alice')").expect("insert failed");

    // Add a new column
    let affected =
        db.execute("ALTER TABLE alter_test ADD COLUMN email TEXT").expect("alter table failed");
    assert_eq!(affected, 0);

    // Insert data with the new column
    db.execute("INSERT INTO alter_test (id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        .expect("insert with new column failed");

    // Query should work
    let result = db.query("SELECT * FROM alter_test").expect("query failed");
    assert_eq!(result.len(), 2);
}

#[test]
fn test_alter_table_add_column_if_not_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with existing email column
    db.execute("CREATE TABLE alter_exists_test (id BIGINT, email TEXT)")
        .expect("create table failed");

    // Add column that already exists with IF NOT EXISTS - should succeed silently
    let affected = db
        .execute("ALTER TABLE alter_exists_test ADD COLUMN IF NOT EXISTS email TEXT")
        .expect("alter table if not exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_drop_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with multiple columns
    db.execute("CREATE TABLE drop_col_test (id BIGINT, name TEXT, temp_col TEXT)")
        .expect("create table failed");

    // Drop a column
    let affected =
        db.execute("ALTER TABLE drop_col_test DROP COLUMN temp_col").expect("alter table failed");
    assert_eq!(affected, 0);

    // Insert data - should work with remaining columns
    db.execute("INSERT INTO drop_col_test (id, name) VALUES (1, 'Test')").expect("insert failed");

    let result = db.query("SELECT * FROM drop_col_test").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_alter_table_drop_column_if_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE drop_if_exists_test (id BIGINT, name TEXT)")
        .expect("create table failed");

    // Drop column that doesn't exist with IF EXISTS - should succeed silently
    let affected = db
        .execute("ALTER TABLE drop_if_exists_test DROP COLUMN IF EXISTS nonexistent")
        .expect("alter table drop if exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_rename_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE rename_col_test (id BIGINT, old_name TEXT)")
        .expect("create table failed");

    // Rename column
    let affected = db
        .execute("ALTER TABLE rename_col_test RENAME COLUMN old_name TO new_name")
        .expect("rename column failed");
    assert_eq!(affected, 0);

    // Insert data using new column name
    db.execute("INSERT INTO rename_col_test (id, new_name) VALUES (1, 'Test')")
        .expect("insert failed");

    let result = db.query("SELECT * FROM rename_col_test").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_alter_table_rename_table() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE old_table_name (id BIGINT, value TEXT)").expect("create table failed");

    // Rename table (schema-only operation - entities use labels, not schema names)
    let affected = db
        .execute("ALTER TABLE old_table_name RENAME TO new_table_name")
        .expect("rename table failed");
    assert_eq!(affected, 0);

    // The schema is now renamed - new inserts should work
    db.execute("INSERT INTO new_table_name (id, value) VALUES (1, 'test')")
        .expect("insert with new name failed");

    // Query with new table name
    let result = db.query("SELECT * FROM new_table_name").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_alter_table_if_exists() {
    let db = Database::in_memory().expect("failed to create db");

    // Alter a table that doesn't exist with IF EXISTS - should succeed silently
    let affected = db
        .execute("ALTER TABLE IF EXISTS nonexistent_table ADD COLUMN new_col TEXT")
        .expect("alter table if exists failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_alter_column_set_not_null() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE set_not_null_test (id BIGINT, name TEXT)")
        .expect("create table failed");

    // Alter column to SET NOT NULL
    let affected = db
        .execute("ALTER TABLE set_not_null_test ALTER COLUMN name SET NOT NULL")
        .expect("set not null failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_alter_column_drop_not_null() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with NOT NULL constraint
    db.execute("CREATE TABLE drop_not_null_test (id BIGINT, name TEXT NOT NULL)")
        .expect("create table failed");

    // Alter column to DROP NOT NULL
    let affected = db
        .execute("ALTER TABLE drop_not_null_test ALTER COLUMN name DROP NOT NULL")
        .expect("drop not null failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_alter_column_set_default() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE set_default_test (id BIGINT, status TEXT)")
        .expect("create table failed");

    // Alter column to SET DEFAULT
    let affected = db
        .execute("ALTER TABLE set_default_test ALTER COLUMN status SET DEFAULT 'pending'")
        .expect("set default failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_alter_column_drop_default() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with DEFAULT constraint
    db.execute("CREATE TABLE drop_default_test (id BIGINT, status TEXT DEFAULT 'active')")
        .expect("create table failed");

    // Alter column to DROP DEFAULT
    let affected = db
        .execute("ALTER TABLE drop_default_test ALTER COLUMN status DROP DEFAULT")
        .expect("drop default failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_alter_column_set_data_type() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE type_change_test (id BIGINT, score INTEGER)")
        .expect("create table failed");

    // Alter column type
    let affected = db
        .execute("ALTER TABLE type_change_test ALTER COLUMN score SET DATA TYPE BIGINT")
        .expect("set data type failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_multiple_actions() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE multi_alter_test (id BIGINT, col1 TEXT, col2 TEXT)")
        .expect("create table failed");

    // Multiple ALTER actions in one statement
    let affected = db
        .execute(
            "ALTER TABLE multi_alter_test
             ADD COLUMN col3 INTEGER,
             DROP COLUMN col2",
        )
        .expect("multiple alter failed");
    assert_eq!(affected, 0);
}

#[test]
fn test_alter_table_nonexistent_table_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Alter a table that doesn't exist without IF EXISTS - should fail
    let result = db.execute("ALTER TABLE nonexistent_table ADD COLUMN new_col TEXT");
    assert!(result.is_err());
}

#[test]
fn test_alter_table_add_column_duplicate_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with existing column
    db.execute("CREATE TABLE dup_col_test (id BIGINT, existing_col TEXT)")
        .expect("create table failed");

    // Try to add a column that already exists without IF NOT EXISTS - should fail
    let result = db.execute("ALTER TABLE dup_col_test ADD COLUMN existing_col TEXT");
    assert!(result.is_err());
}

#[test]
fn test_alter_table_drop_nonexistent_column_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE drop_missing_test (id BIGINT)").expect("create table failed");

    // Try to drop a column that doesn't exist without IF EXISTS - should fail
    let result = db.execute("ALTER TABLE drop_missing_test DROP COLUMN nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_alter_table_rename_to_existing_table_error() {
    let db = Database::in_memory().expect("failed to create db");

    // Create two tables
    db.execute("CREATE TABLE source_table (id BIGINT)").expect("create source failed");
    db.execute("CREATE TABLE target_table (id BIGINT)").expect("create target failed");

    // Try to rename to an existing table name - should fail
    let result = db.execute("ALTER TABLE source_table RENAME TO target_table");
    assert!(result.is_err());
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

// ============================================================================
// Index Backfill Tests
// ============================================================================

#[test]
fn test_create_index_backfills_existing_data() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table first
    db.execute("CREATE TABLE backfill_users (id BIGINT, name TEXT, email TEXT)")
        .expect("create table failed");

    // Insert some data
    db.execute(
        "INSERT INTO backfill_users (id, name, email) VALUES (1, 'Alice', 'alice@example.com')",
    )
    .expect("insert 1 failed");
    db.execute("INSERT INTO backfill_users (id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        .expect("insert 2 failed");
    db.execute(
        "INSERT INTO backfill_users (id, name, email) VALUES (3, 'Charlie', 'charlie@example.com')",
    )
    .expect("insert 3 failed");

    // Now create an index on the email column
    // This should backfill the existing 3 rows into the index
    let affected = db
        .execute("CREATE INDEX idx_backfill_users_email ON backfill_users (email)")
        .expect("create index failed");

    // The executor returns the number of entries backfilled
    assert_eq!(affected, 3);

    // Verify the data can still be queried
    let result = db
        .query("SELECT * FROM backfill_users WHERE email = 'alice@example.com'")
        .expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_create_index_on_empty_table_returns_zero() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table but don't insert any data
    db.execute("CREATE TABLE empty_test (id BIGINT, value TEXT)").expect("create table failed");

    // Create an index - should return 0 since no data to backfill
    let affected = db
        .execute("CREATE INDEX idx_empty_value ON empty_test (value)")
        .expect("create index failed");

    assert_eq!(affected, 0);
}

#[test]
fn test_create_index_skips_missing_column() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE nulltest (id BIGINT, name TEXT)").expect("create table failed");

    // Insert some data - some rows have name, some don't
    db.execute("INSERT INTO nulltest (id, name) VALUES (1, 'Alice')").expect("insert 1 failed");
    db.execute("INSERT INTO nulltest (id) VALUES (2)")  // name is missing (not stored)
        .expect("insert 2 failed");
    db.execute("INSERT INTO nulltest (id, name) VALUES (3, 'Charlie')").expect("insert 3 failed");

    // Create an index on name column
    // Only rows with the property set get indexed
    let affected = db
        .execute("CREATE INDEX idx_nulltest_name ON nulltest (name)")
        .expect("create index failed");

    // We should get 2 entries (only rows with name property)
    assert_eq!(affected, 2);
}

#[test]
fn test_drop_index_and_recreate() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE cleanup_test (id BIGINT, value TEXT)").expect("create table failed");

    // Insert data
    db.execute("INSERT INTO cleanup_test (id, value) VALUES (1, 'one')").expect("insert 1 failed");
    db.execute("INSERT INTO cleanup_test (id, value) VALUES (2, 'two')").expect("insert 2 failed");

    // Create index (backfills 2 entries)
    let created = db
        .execute("CREATE INDEX idx_cleanup ON cleanup_test (value)")
        .expect("create index failed");
    assert_eq!(created, 2);

    // Drop the index
    db.execute("DROP INDEX idx_cleanup").expect("drop index failed");

    // Verify we can create the index again and it backfills
    // This proves the drop succeeded and cleaned up properly
    let recreated = db
        .execute("CREATE INDEX idx_cleanup ON cleanup_test (value)")
        .expect("recreate index failed");
    assert_eq!(recreated, 2);

    // Query should still work
    let result = db.query("SELECT * FROM cleanup_test WHERE value = 'one'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_create_index_with_numeric_types() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table
    db.execute("CREATE TABLE mixed (id BIGINT, score INTEGER)").expect("create table failed");

    // Insert data with numeric values
    db.execute("INSERT INTO mixed (id, score) VALUES (1, 100)").expect("insert 1 failed");
    db.execute("INSERT INTO mixed (id, score) VALUES (2, 200)").expect("insert 2 failed");
    db.execute("INSERT INTO mixed (id, score) VALUES (3, 150)").expect("insert 3 failed");

    // Create index on numeric column
    let affected =
        db.execute("CREATE INDEX idx_mixed_score ON mixed (score)").expect("create index failed");

    assert_eq!(affected, 3);

    // Query should still work
    let result = db.query("SELECT * FROM mixed WHERE score > 120").expect("query failed");
    assert_eq!(result.len(), 2); // 200 and 150
}
