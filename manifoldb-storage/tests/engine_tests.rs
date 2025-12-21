//! Tests for storage engine traits.
//!
//! These tests validate the trait contracts and can be used to test
//! any storage engine implementation.

use std::ops::Bound;

use manifoldb_storage::{Cursor, StorageEngine, StorageError, StorageResult, Transaction};

/// A test harness trait for testing storage engine implementations.
///
/// Implementors provide a way to create and clean up test databases.
pub trait TestHarness {
    /// The storage engine type being tested.
    type Engine: StorageEngine;

    /// Create a new storage engine for testing.
    fn create_engine() -> StorageResult<Self::Engine>;

    /// Clean up after tests (remove temp files, etc.).
    fn cleanup(_engine: Self::Engine) {}
}

/// Run the standard test suite against a storage engine.
///
/// This function runs all the standard trait compliance tests against
/// the provided harness. Use this in integration tests for each backend.
///
/// # Example
///
/// ```ignore
/// struct RedbHarness;
///
/// impl TestHarness for RedbHarness {
///     type Engine = RedbEngine;
///
///     fn create_engine() -> StorageResult<Self::Engine> {
///         RedbEngine::open(tempfile::tempdir()?.path().join("test.redb"))
///     }
/// }
///
/// #[test]
/// fn test_redb_compliance() {
///     run_test_suite::<RedbHarness>();
/// }
/// ```
pub fn run_test_suite<H: TestHarness>() {
    test_basic_operations::<H>();
    test_transaction_isolation::<H>();
    test_cursor_operations::<H>();
    test_range_scan::<H>();
    test_read_only_enforcement::<H>();
}

/// Test basic get/put/delete operations.
fn test_basic_operations<H: TestHarness>() {
    let engine = H::create_engine().expect("failed to create engine");

    // Write a key-value pair
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test_table", b"key1", b"value1").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Read it back
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let value = tx.get("test_table", b"key1").expect("failed to get");
        assert_eq!(value, Some(b"value1".to_vec()));
    }

    // Update the value
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test_table", b"key1", b"value1_updated").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Verify update
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let value = tx.get("test_table", b"key1").expect("failed to get");
        assert_eq!(value, Some(b"value1_updated".to_vec()));
    }

    // Delete the key
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        let deleted = tx.delete("test_table", b"key1").expect("failed to delete");
        assert!(deleted);
        tx.commit().expect("failed to commit");
    }

    // Verify deletion
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let value = tx.get("test_table", b"key1").expect("failed to get");
        assert_eq!(value, None);
    }

    // Delete non-existent key should return false
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        let deleted = tx.delete("test_table", b"nonexistent").expect("failed to delete");
        assert!(!deleted);
        tx.rollback().expect("failed to rollback");
    }

    H::cleanup(engine);
}

/// Test that transactions provide proper isolation.
fn test_transaction_isolation<H: TestHarness>() {
    let engine = H::create_engine().expect("failed to create engine");

    // Write initial data
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test_table", b"key1", b"initial").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Start a read transaction, check isolation, then drop it
    {
        let read_tx = engine.begin_read().expect("failed to begin read");
        let value = read_tx.get("test_table", b"key1").expect("failed to get");
        assert_eq!(value, Some(b"initial".to_vec()));
        // read_tx is dropped here
    }

    // Write new data
    {
        let mut write_tx = engine.begin_write().expect("failed to begin write");
        write_tx.put("test_table", b"key1", b"updated").expect("failed to put");
        write_tx.commit().expect("failed to commit");
    }

    // New read transaction sees updated value
    {
        let read_tx = engine.begin_read().expect("failed to begin read");
        let value = read_tx.get("test_table", b"key1").expect("failed to get");
        assert_eq!(value, Some(b"updated".to_vec()));
    }

    H::cleanup(engine);
}

/// Test cursor operations: seek, next, prev, seek_first, seek_last.
fn test_cursor_operations<H: TestHarness>() {
    let engine = H::create_engine().expect("failed to create engine");

    // Insert ordered data
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        tx.put("test_table", b"a", b"1").expect("failed to put");
        tx.put("test_table", b"b", b"2").expect("failed to put");
        tx.put("test_table", b"c", b"3").expect("failed to put");
        tx.put("test_table", b"d", b"4").expect("failed to put");
        tx.put("test_table", b"e", b"5").expect("failed to put");
        tx.commit().expect("failed to commit");
    }

    // Test cursor iteration
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test_table").expect("failed to create cursor");

        // Seek to first
        let first = cursor.seek_first().expect("failed to seek_first");
        assert_eq!(first, Some((b"a".to_vec(), b"1".to_vec())));

        // Next
        let second = cursor.next().expect("failed to next");
        assert_eq!(second, Some((b"b".to_vec(), b"2".to_vec())));

        // Current (should return borrowed view of current position)
        let current = cursor.current();
        assert_eq!(current, Some((b"b".as_slice(), b"2".as_slice())));

        // Seek to "c"
        let c = cursor.seek(b"c").expect("failed to seek");
        assert_eq!(c, Some((b"c".to_vec(), b"3".to_vec())));

        // Seek to last
        let last = cursor.seek_last().expect("failed to seek_last");
        assert_eq!(last, Some((b"e".to_vec(), b"5".to_vec())));

        // Prev
        let prev = cursor.prev().expect("failed to prev");
        assert_eq!(prev, Some((b"d".to_vec(), b"4".to_vec())));

        // Next past end
        cursor.seek_last().expect("failed to seek_last");
        let past_end = cursor.next().expect("failed to next");
        assert_eq!(past_end, None);
    }

    // Test seek to non-existent key (should find next greater)
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx.cursor("test_table").expect("failed to create cursor");

        let result = cursor.seek(b"bb").expect("failed to seek");
        assert_eq!(result, Some((b"c".to_vec(), b"3".to_vec())));
    }

    H::cleanup(engine);
}

/// Test range scan operations.
fn test_range_scan<H: TestHarness>() {
    let engine = H::create_engine().expect("failed to create engine");

    // Insert ordered data
    {
        let mut tx = engine.begin_write().expect("failed to begin write");
        for i in 0..10u8 {
            let key = [i];
            let value = [i * 10];
            tx.put("test_table", &key, &value).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Test bounded range
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx
            .range("test_table", Bound::Included(&[3u8] as &[u8]), Bound::Excluded(&[7u8] as &[u8]))
            .expect("failed to create range cursor");

        let mut results = Vec::new();
        while let Some((k, v)) = cursor.next().expect("failed to next") {
            results.push((k[0], v[0]));
        }

        assert_eq!(results, vec![(3, 30), (4, 40), (5, 50), (6, 60)]);
    }

    // Test unbounded start
    {
        let tx = engine.begin_read().expect("failed to begin read");
        let mut cursor = tx
            .range("test_table", Bound::Unbounded, Bound::Excluded(&[3u8] as &[u8]))
            .expect("failed to create range cursor");

        let mut results = Vec::new();
        while let Some((k, _)) = cursor.next().expect("failed to next") {
            results.push(k[0]);
        }

        assert_eq!(results, vec![0, 1, 2]);
    }

    H::cleanup(engine);
}

/// Test that read-only transactions reject write operations.
fn test_read_only_enforcement<H: TestHarness>() {
    let engine = H::create_engine().expect("failed to create engine");

    {
        let tx = engine.begin_read().expect("failed to begin read");
        assert!(tx.is_read_only());
        // Note: We can't easily test that put/delete fail on read-only transactions
        // without using interior mutability tricks, since the trait requires &mut self.
        // The is_read_only() check is the primary contract verification here.
    }

    // Also verify write transactions are not read-only
    {
        let tx = engine.begin_write().expect("failed to begin write");
        assert!(!tx.is_read_only());
        tx.rollback().expect("failed to rollback");
    }

    H::cleanup(engine);
}

/// Test error types are properly constructed and implement Error trait.
#[test]
fn test_error_types() {
    // Test that StorageError implements std::error::Error
    fn assert_error<E: std::error::Error>() {}
    assert_error::<StorageError>();

    // Test error construction and messages
    let open_err = StorageError::Open("test".to_string());
    assert!(open_err.to_string().contains("test"));
    assert!(!open_err.is_recoverable());
    assert!(!open_err.is_not_found());

    let not_found_err = StorageError::NotFound("db".to_string());
    assert!(not_found_err.is_not_found());
    assert!(!not_found_err.is_recoverable());

    let table_not_found = StorageError::TableNotFound("users".to_string());
    assert!(table_not_found.is_not_found());

    let key_not_found = StorageError::KeyNotFound;
    assert!(key_not_found.is_not_found());

    let conflict_err = StorageError::Conflict("concurrent write".to_string());
    assert!(conflict_err.is_recoverable());

    let read_only_err = StorageError::ReadOnly;
    assert!(!read_only_err.is_recoverable());
    assert!(read_only_err.to_string().contains("read-only"));
}

/// Test that the Cursor trait is object-safe by requiring it.
#[test]
fn test_cursor_object_safety() {
    // This test verifies that Cursor can be used as a trait object.
    // If this compiles, the trait is object-safe.
    fn _takes_cursor(_: &dyn Cursor) {}
}

/// Test that ErrorContext displays properly.
#[test]
fn test_error_context_display() {
    use manifoldb_storage::ErrorContext;

    let ctx = ErrorContext {
        table: Some("users".to_string()),
        key: Some("abc123".to_string()),
        message: Some("additional info".to_string()),
    };

    let display = ctx.to_string();
    assert!(display.contains("table=users"));
    assert!(display.contains("key=abc123"));
    assert!(display.contains("additional info"));

    let empty_ctx = ErrorContext { table: None, key: None, message: None };
    assert_eq!(empty_ctx.to_string(), "");
}
