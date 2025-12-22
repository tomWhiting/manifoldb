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

/// Test that the streaming cursor handles large datasets (> batch size) correctly.
/// This verifies that batch boundaries are handled properly.
#[test]
fn test_streaming_cursor_large_dataset() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // Use a number larger than the default batch size (1000) to test batching
    const NUM_KEYS: usize = 3500;

    // Write many keys with predictable order
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        for i in 0..NUM_KEYS {
            let key = format!("key:{i:06}");
            let value = format!("value:{i:06}");
            tx.put("test", key.as_bytes(), value.as_bytes()).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Test forward iteration across batches
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test").expect("failed to create cursor");

        let mut count = 0;
        let mut last_key: Option<Vec<u8>> = None;

        cursor.seek_first().expect("failed to seek_first");
        while let Some((k, _)) = cursor.current() {
            // Verify keys are in ascending order
            if let Some(prev) = &last_key {
                assert!(k > prev.as_slice(), "keys should be in ascending order");
            }
            last_key = Some(k.to_vec());
            count += 1;

            if cursor.next().expect("failed to next").is_none() {
                break;
            }
        }
        assert_eq!(count, NUM_KEYS);
    }

    // Test backward iteration across batches
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test").expect("failed to create cursor");

        let mut count = 0;
        let mut last_key: Option<Vec<u8>> = None;

        cursor.seek_last().expect("failed to seek_last");
        while let Some((k, _)) = cursor.current() {
            // Verify keys are in descending order
            if let Some(prev) = &last_key {
                assert!(k < prev.as_slice(), "keys should be in descending order");
            }
            last_key = Some(k.to_vec());
            count += 1;

            if cursor.prev().expect("failed to prev").is_none() {
                break;
            }
        }
        assert_eq!(count, NUM_KEYS);
    }

    // Test seeking across batches
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test").expect("failed to create cursor");

        // Seek to middle of dataset
        let seek_key = format!("key:{:06}", NUM_KEYS / 2);
        let result = cursor.seek(seek_key.as_bytes()).expect("failed to seek");
        assert!(result.is_some());
        let (k, _) = result.unwrap();
        assert_eq!(k, seek_key.as_bytes());

        // Continue forward
        let next = cursor.next().expect("failed to next");
        assert!(next.is_some());
        let expected_next = format!("key:{:06}", NUM_KEYS / 2 + 1);
        assert_eq!(next.unwrap().0, expected_next.as_bytes());
    }

    // Test range queries across batches
    {
        let tx = engine.begin_read().expect("failed to begin read");

        // Range from key 1000 to key 2500 (1500 keys, spanning multiple batches)
        let start_key = format!("key:{:06}", 1000);
        let end_key = format!("key:{:06}", 2500);
        let mut cursor = tx
            .range(
                "test",
                std::ops::Bound::Included(start_key.as_bytes()),
                std::ops::Bound::Excluded(end_key.as_bytes()),
            )
            .expect("failed to create range cursor");

        let mut count = 0;
        while cursor.next().expect("failed to next").is_some() {
            count += 1;
        }
        assert_eq!(count, 1500);
    }
}

/// Test cursor bidirectional navigation across batch boundaries.
#[test]
fn test_cursor_bidirectional_across_batches() {
    let engine = RedbEngine::in_memory().expect("failed to create engine");

    // 2100 keys = 3 batches with default batch size of 1000
    const NUM_KEYS: usize = 2100;

    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        for i in 0..NUM_KEYS {
            let key = format!("key:{i:05}");
            let value = format!("value:{i:05}");
            tx.put("test", key.as_bytes(), value.as_bytes()).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Navigate forward past first batch, then backward
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test").expect("failed to create cursor");

        // Go forward 1050 items (past first batch)
        cursor.seek_first().expect("failed to seek_first");
        for _ in 0..1050 {
            cursor.next().expect("failed to next");
        }

        // Current should be key:01050
        let (k, _) = cursor.current().expect("should have current");
        assert_eq!(k, b"key:01050");

        // Go backward
        cursor.prev().expect("failed to prev");
        let (k, _) = cursor.current().expect("should have current");
        assert_eq!(k, b"key:01049");

        // Go backward to cross batch boundary
        for _ in 0..100 {
            cursor.prev().expect("failed to prev");
        }
        let (k, _) = cursor.current().expect("should have current");
        assert_eq!(k, b"key:00949");
    }
}
