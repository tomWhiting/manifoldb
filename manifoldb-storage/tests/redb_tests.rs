//! Tests for the Redb storage backend.
//!
//! This module runs the standard storage engine compliance tests against
//! the Redb backend, plus Redb-specific tests.

mod engine_tests;

use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{Cursor, StorageEngine, StorageResult, Transaction};

use engine_tests::{run_test_suite, TestHarness};

/// Test harness for the Redb in-memory backend.
struct RedbHarness;

impl TestHarness for RedbHarness {
    type Engine = RedbEngine;

    fn create_engine() -> StorageResult<Self::Engine> {
        RedbEngine::in_memory()
    }
}

/// Run the full compliance test suite for Redb.
#[test]
fn test_redb_compliance() {
    run_test_suite::<RedbHarness>();
}

/// Test Redb-specific: multiple tables in same transaction.
#[test]
fn test_multiple_tables() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // Write to multiple tables in one transaction
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("users", b"user:1", b"Alice").expect("failed to put user");
        tx.put("orders", b"order:1", b"Order for Alice").expect("failed to put order");
        tx.put("users", b"user:2", b"Bob").expect("failed to put user");
        tx.commit().expect("failed to commit");
    }

    // Read from both tables
    {
        let tx = engine.begin_read().expect("failed to begin read");

        let user1 = tx.get("users", b"user:1").expect("failed to get user");
        assert_eq!(user1, Some(b"Alice".to_vec()));

        let user2 = tx.get("users", b"user:2").expect("failed to get user");
        assert_eq!(user2, Some(b"Bob".to_vec()));

        let order = tx.get("orders", b"order:1").expect("failed to get order");
        assert_eq!(order, Some(b"Order for Alice".to_vec()));

        // Non-existent table key
        let missing = tx.get("users", b"user:999").expect("failed to get");
        assert_eq!(missing, None);
    }
}

/// Test Redb-specific: table isolation (keys don't collide across tables).
#[test]
fn test_table_isolation() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // Write same key to different tables
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("table_a", b"key", b"value_a").expect("failed to put");
        tx.put("table_b", b"key", b"value_b").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Verify they're isolated
    {
        let tx = engine.begin_read().expect("failed to begin read");

        let a = tx.get("table_a", b"key").expect("failed to get");
        assert_eq!(a, Some(b"value_a".to_vec()));

        let b = tx.get("table_b", b"key").expect("failed to get");
        assert_eq!(b, Some(b"value_b".to_vec()));
    }
}

/// Test Redb-specific: rollback discards changes.
#[test]
fn test_rollback_discards_changes() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // First, write some initial data
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test", b"key", b"initial").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Start a write transaction but rollback
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test", b"key", b"modified").expect("failed to put");
        tx.put("test", b"new_key", b"new_value").expect("failed to put");
        tx.rollback().expect("failed to rollback");
    }

    // Verify original data is unchanged
    {
        let tx = engine.begin_read().expect("failed to begin read");

        let value = tx.get("test", b"key").expect("failed to get");
        assert_eq!(value, Some(b"initial".to_vec()));

        let new_value = tx.get("test", b"new_key").expect("failed to get");
        assert_eq!(new_value, None);
    }
}

/// Test Redb-specific: concurrent read transactions.
#[test]
fn test_concurrent_read_transactions() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // Write some data
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test", b"key1", b"value1").expect("failed to put");
        tx.put("test", b"key2", b"value2").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Open multiple read transactions simultaneously
    let tx1 = engine.begin_read().expect("failed to begin read 1");
    let tx2 = engine.begin_read().expect("failed to begin read 2");

    // Both should see the same data
    let v1_tx1 = tx1.get("test", b"key1").expect("failed to get");
    let v1_tx2 = tx2.get("test", b"key1").expect("failed to get");
    assert_eq!(v1_tx1, v1_tx2);

    let v2_tx1 = tx1.get("test", b"key2").expect("failed to get");
    let v2_tx2 = tx2.get("test", b"key2").expect("failed to get");
    assert_eq!(v2_tx1, v2_tx2);
}

/// Test Redb-specific: large values.
#[test]
fn test_large_values() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // Create a large value (1 MB)
    let large_value = vec![0xAB_u8; 1024 * 1024];

    // Write large value
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test", b"large", &large_value).expect("failed to put large value");
        tx.commit().expect("failed to commit");
    }

    // Read it back
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let value = tx.get("test", b"large").expect("failed to get");
        assert_eq!(value, Some(large_value));
    }
}

/// Test Redb-specific: many keys in a single table.
#[test]
fn test_many_keys() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    const NUM_KEYS: usize = 1000;

    // Write many keys
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        for i in 0..NUM_KEYS {
            let key = format!("key:{i:05}");
            let value = format!("value:{i:05}");
            tx.put("test", key.as_bytes(), value.as_bytes()).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Read them back and verify
    {
        let tx = engine.begin_read().expect("failed to begin read");
        for i in 0..NUM_KEYS {
            let key = format!("key:{i:05}");
            let expected = format!("value:{i:05}");
            let value = tx.get("test", key.as_bytes()).expect("failed to get");
            assert_eq!(value, Some(expected.into_bytes()));
        }
    }

    // Test cursor iteration
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test").expect("failed to create cursor");

        let mut count = 0;
        cursor.seek_first().expect("failed to seek_first");
        while cursor.current().is_some() {
            count += 1;
            if cursor.next().expect("failed to next").is_none() {
                break;
            }
        }
        assert_eq!(count, NUM_KEYS);
    }
}
