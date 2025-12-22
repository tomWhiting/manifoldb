//! Concurrency integration tests.
//!
//! Tests for concurrent access patterns including:
//! - Multiple readers + writers
//! - Deadlock detection
//! - Transaction isolation under concurrency
//! - Stress testing with many concurrent operations

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Multiple Readers Tests
// ============================================================================

/// Multiple read transactions can run concurrently
#[test]
fn test_concurrent_reads() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));

    // Create test data
    let entity_ids: Vec<EntityId>;
    {
        let mut tx = db.begin().expect("failed to begin");
        let mut ids = Vec::new();
        for i in 0..100 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("TestEntity")
                .with_property("index", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
        entity_ids = ids;
    }

    let entity_ids = Arc::new(entity_ids);
    let num_readers = 10;
    let barrier = Arc::new(Barrier::new(num_readers));

    let handles: Vec<_> = (0..num_readers)
        .map(|reader_id| {
            let db = Arc::clone(&db);
            let entity_ids = Arc::clone(&entity_ids);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                // All threads start reading at the same time
                barrier.wait();

                let tx = db.begin_read().expect("failed to begin read");

                // Each reader reads all entities
                for (i, &id) in entity_ids.iter().enumerate() {
                    let entity = tx.get_entity(id).expect("failed").expect("entity should exist");
                    assert_eq!(
                        entity.get_property("index"),
                        Some(&Value::Int(i as i64)),
                        "reader {reader_id} got wrong value"
                    );
                }

                tx.rollback().expect("failed");
                reader_id
            })
        })
        .collect();

    // All readers should complete successfully
    for handle in handles {
        handle.join().expect("reader thread panicked");
    }
}

/// Readers see consistent snapshot while writer is active
#[test]
fn test_reader_isolation_during_write() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));

    // Create initial data
    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("value", 1i64);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    let reader_started = Arc::new(AtomicBool::new(false));
    let writer_started = Arc::new(AtomicBool::new(false));
    let writer_committed = Arc::new(AtomicBool::new(false));

    let reader_started_clone = Arc::clone(&reader_started);
    let writer_started_clone = Arc::clone(&writer_started);
    let writer_committed_clone = Arc::clone(&writer_committed);
    let db_clone = Arc::clone(&db);

    // Reader thread: starts a read transaction and holds it
    let reader_handle = thread::spawn(move || {
        let tx = db_clone.begin_read().expect("failed to begin read");
        reader_started_clone.store(true, Ordering::SeqCst);

        // Wait for writer to start
        while !writer_started_clone.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }

        // Read should still see original value
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        let value_before = entity.get_property("value").cloned();

        // Wait for writer to commit
        while !writer_committed_clone.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }

        // Same transaction should still see original value (snapshot isolation)
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        let value_after = entity.get_property("value").cloned();

        tx.rollback().expect("failed");

        (value_before, value_after)
    });

    let db_writer = Arc::clone(&db);

    // Writer thread: updates the value while reader is active
    let writer_handle = thread::spawn(move || {
        // Wait for reader to start
        while !reader_started.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }

        let mut tx = db_writer.begin().expect("failed to begin write");
        writer_started.store(true, Ordering::SeqCst);

        let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        entity.set_property("value", 999i64);
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");

        writer_committed.store(true, Ordering::SeqCst);
    });

    writer_handle.join().expect("writer panicked");
    let (value_before, value_after) = reader_handle.join().expect("reader panicked");

    // Reader should see consistent snapshot (original value) throughout
    assert_eq!(value_before, Some(Value::Int(1)));
    assert_eq!(value_after, Some(Value::Int(1)));

    // New read should see updated value
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(entity.get_property("value"), Some(&Value::Int(999)));
}

// ============================================================================
// Writer Serialization Tests
// ============================================================================

/// Write transactions are serialized (one at a time)
#[test]
fn test_write_serialization() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let counter = Arc::new(AtomicU64::new(0));
    let num_writers = 5;
    let ops_per_writer = 20;
    let barrier = Arc::new(Barrier::new(num_writers));

    let handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let db = Arc::clone(&db);
            let counter = Arc::clone(&counter);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..ops_per_writer {
                    let mut tx = db.begin().expect("failed to begin write");

                    // Create entity with unique counter value
                    let value = counter.fetch_add(1, Ordering::SeqCst);
                    let entity =
                        tx.create_entity().expect("failed").with_property("value", value as i64);
                    tx.put_entity(&entity).expect("failed");
                    tx.commit().expect("failed");
                }

                writer_id
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("writer panicked");
    }

    // Verify all entities were created
    let tx = db.begin_read().expect("failed");
    let expected_count = (num_writers * ops_per_writer) as u64;

    // Count entities by checking sequential IDs
    let mut count = 0u64;
    for id in 1..=expected_count {
        if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
            count += 1;
        }
    }

    assert_eq!(count, expected_count, "all entities should be created");
}

/// Concurrent writers don't lose data
#[test]
fn test_no_lost_updates() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let num_writers = 10;
    let updates_per_writer = 50;
    let barrier = Arc::new(Barrier::new(num_writers));

    // Create entity to update
    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("counter", 0i64);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    let handles: Vec<_> = (0..num_writers)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..updates_per_writer {
                    let mut tx = db.begin().expect("failed");

                    // Read current value
                    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
                    let current = match entity.get_property("counter") {
                        Some(Value::Int(n)) => *n,
                        _ => panic!("counter should be int"),
                    };

                    // Increment
                    let mut updated = entity;
                    updated.set_property("counter", current + 1);
                    tx.put_entity(&updated).expect("failed");
                    tx.commit().expect("failed");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("writer panicked");
    }

    // Verify final counter value
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
    let final_value = match entity.get_property("counter") {
        Some(Value::Int(n)) => *n,
        _ => panic!("counter should be int"),
    };

    let expected = (num_writers * updates_per_writer) as i64;
    assert_eq!(final_value, expected, "no updates should be lost");
}

// ============================================================================
// Mixed Read/Write Tests
// ============================================================================

/// Concurrent readers and writers work correctly
#[test]
fn test_concurrent_readers_and_writers() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let running = Arc::new(AtomicBool::new(true));

    let num_readers = 4;
    let num_writers = 2;
    let test_duration = Duration::from_millis(500);

    // Counter to track operations
    let read_ops = Arc::new(AtomicU64::new(0));
    let write_ops = Arc::new(AtomicU64::new(0));

    // Start reader threads
    let reader_handles: Vec<_> = (0..num_readers)
        .map(|reader_id| {
            let db = Arc::clone(&db);
            let running = Arc::clone(&running);
            let read_ops = Arc::clone(&read_ops);

            thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let tx = db.begin_read().expect("failed to begin read");

                    // Read some entities
                    for id in 1..=10 {
                        let _ = tx.get_entity(EntityId::new(id));
                    }

                    tx.rollback().expect("failed");
                    read_ops.fetch_add(1, Ordering::Relaxed);
                }
                reader_id
            })
        })
        .collect();

    // Start writer threads
    let writer_handles: Vec<_> = (0..num_writers)
        .map(|writer_id| {
            let db = Arc::clone(&db);
            let running = Arc::clone(&running);
            let write_ops = Arc::clone(&write_ops);

            thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let mut tx = db.begin().expect("failed to begin write");

                    let entity = tx
                        .create_entity()
                        .expect("failed")
                        .with_property("writer", writer_id as i64);
                    tx.put_entity(&entity).expect("failed");
                    tx.commit().expect("failed");

                    write_ops.fetch_add(1, Ordering::Relaxed);
                }
                writer_id
            })
        })
        .collect();

    // Let them run
    thread::sleep(test_duration);
    running.store(false, Ordering::Relaxed);

    // Join all threads
    for handle in reader_handles {
        handle.join().expect("reader panicked");
    }
    for handle in writer_handles {
        handle.join().expect("writer panicked");
    }

    let total_reads = read_ops.load(Ordering::Relaxed);
    let total_writes = write_ops.load(Ordering::Relaxed);

    // Verify we actually did work
    assert!(total_reads > 0, "should have done some reads");
    assert!(total_writes > 0, "should have done some writes");

    // Verify database is consistent
    db.flush().expect("flush should succeed");
}

// ============================================================================
// Transaction Timeout and Deadlock Detection Tests
// ============================================================================

/// Long-running transactions don't block indefinitely
#[test]
fn test_long_running_transaction_handling() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let first_writer_started = Arc::new(AtomicBool::new(false));
    let first_writer_done = Arc::new(AtomicBool::new(false));

    let db_clone = Arc::clone(&db);
    let started = Arc::clone(&first_writer_started);
    let done = Arc::clone(&first_writer_done);

    // First writer: holds transaction for a while
    let first_handle = thread::spawn(move || {
        let mut tx = db_clone.begin().expect("failed to begin");
        started.store(true, Ordering::SeqCst);

        let entity = tx.create_entity().expect("failed").with_property("first", true);
        tx.put_entity(&entity).expect("failed");

        // Hold the transaction
        thread::sleep(Duration::from_millis(100));

        tx.commit().expect("failed");
        done.store(true, Ordering::SeqCst);
    });

    // Wait for first writer to start
    while !first_writer_started.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(1));
    }

    // Second writer: should wait and eventually succeed
    let start = std::time::Instant::now();
    {
        let mut tx = db.begin().expect("failed to begin");
        let entity = tx.create_entity().expect("failed").with_property("second", true);
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }
    let elapsed = start.elapsed();

    first_handle.join().expect("first writer panicked");

    // Second writer should have waited
    assert!(elapsed > Duration::from_millis(50), "second writer should have waited");
}

// ============================================================================
// Edge Operations Under Concurrency
// ============================================================================

/// Concurrent edge creation doesn't lose edges
#[test]
fn test_concurrent_edge_creation() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));

    // Create nodes
    let node_ids: Vec<EntityId>;
    {
        let mut tx = db.begin().expect("failed");
        let mut ids = Vec::new();
        for i in 0..10 {
            let entity =
                tx.create_entity().expect("failed").with_label("Node").with_property("id", i);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
        node_ids = ids;
    }

    let node_ids = Arc::new(node_ids);
    let num_workers = 4;
    let edges_per_worker = 25;
    let barrier = Arc::new(Barrier::new(num_workers));

    let handles: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let db = Arc::clone(&db);
            let node_ids = Arc::clone(&node_ids);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for i in 0..edges_per_worker {
                    let mut tx = db.begin().expect("failed");

                    let src_idx = (worker_id * edges_per_worker + i) % node_ids.len();
                    let dst_idx = (src_idx + 1) % node_ids.len();

                    let edge = tx
                        .create_edge(node_ids[src_idx], node_ids[dst_idx], "CONNECTS")
                        .expect("failed");
                    tx.put_edge(&edge).expect("failed");
                    tx.commit().expect("failed");
                }

                worker_id
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("worker panicked");
    }

    // Verify all edges were created
    let tx = db.begin_read().expect("failed");
    let mut total_edges = 0;

    for &node_id in node_ids.iter() {
        let edges = tx.get_outgoing_edges(node_id).expect("failed");
        total_edges += edges.len();
    }

    let expected_edges = num_workers * edges_per_worker;
    assert_eq!(total_edges, expected_edges, "all edges should be created");
}

// ============================================================================
// Stress Tests
// ============================================================================

/// Heavy concurrent load stress test
#[test]
fn test_concurrent_stress() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let num_threads = 8;
    let ops_per_thread = 100;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for op_num in 0..ops_per_thread {
                    // Alternate between read and write operations
                    if op_num % 3 == 0 {
                        // Read operation
                        let tx = db.begin_read().expect("failed");
                        for id in 1..=10 {
                            let _ = tx.get_entity(EntityId::new(id));
                        }
                        tx.rollback().expect("failed");
                    } else {
                        // Write operation
                        let mut tx = db.begin().expect("failed");
                        let entity = tx
                            .create_entity()
                            .expect("failed")
                            .with_property("thread", thread_id as i64);
                        tx.put_entity(&entity).expect("failed");

                        // Sometimes create edges too
                        if op_num % 5 == 0 && op_num > 0 {
                            // Try to create edge to previous entity if possible
                            let prev_id = EntityId::new(
                                ((thread_id * ops_per_thread + op_num - 1) + 1) as u64,
                            );
                            if tx.get_entity(prev_id).expect("failed").is_some() {
                                let edge =
                                    tx.create_edge(entity.id, prev_id, "FOLLOWS").expect("failed");
                                tx.put_edge(&edge).expect("failed");
                            }
                        }

                        tx.commit().expect("failed");
                    }
                }

                thread_id
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Verify database is still consistent
    let tx = db.begin_read().expect("failed");

    // Count entities
    let mut entity_count = 0u64;
    for id in 1..=((num_threads * ops_per_thread * 2) as u64) {
        if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
            entity_count += 1;
        }
    }

    assert!(entity_count > 0, "should have created some entities");
}

/// Rapid transaction creation and abandonment
#[test]
fn test_transaction_churn() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let num_threads = 4;
    let iterations = 100;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for i in 0..iterations {
                    // Mix of commits and rollbacks
                    if i % 2 == 0 {
                        // Commit
                        let mut tx = db.begin().expect("failed");
                        let entity = tx.create_entity().expect("failed");
                        tx.put_entity(&entity).expect("failed");
                        tx.commit().expect("failed");
                    } else {
                        // Rollback
                        let mut tx = db.begin().expect("failed");
                        let entity = tx.create_entity().expect("failed");
                        tx.put_entity(&entity).expect("failed");
                        tx.rollback().expect("failed");
                    }

                    // Also do some read transactions
                    let tx = db.begin_read().expect("failed");
                    let _ = tx.get_entity(EntityId::new(1));
                    drop(tx); // Implicit rollback
                }

                thread_id
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // Verify database is still functional
    db.flush().expect("flush should succeed");
}

// ============================================================================
// Read-Your-Own-Writes Tests
// ============================================================================

/// Writer sees their own uncommitted changes
#[test]
fn test_read_own_writes() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed");

    // Create entity
    let entity = tx.create_entity().expect("failed").with_property("test", "value");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed");

    // Should be able to read it back in same transaction
    let retrieved = tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(retrieved.get_property("test"), Some(&Value::String("value".to_string())));

    // Update it
    let mut updated = retrieved;
    updated.set_property("test", "updated");
    tx.put_entity(&updated).expect("failed");

    // Should see updated value
    let retrieved = tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(retrieved.get_property("test"), Some(&Value::String("updated".to_string())));

    tx.commit().expect("failed");
}

// ============================================================================
// Transaction ID Uniqueness Under Concurrency
// ============================================================================

/// Transaction IDs are unique even under heavy concurrency
#[test]
fn test_transaction_id_uniqueness() {
    let db = Arc::new(Database::in_memory().expect("failed to create db"));
    let num_threads = 8;
    let txns_per_thread = 50;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                let mut tx_ids = Vec::with_capacity(txns_per_thread);

                for _ in 0..txns_per_thread {
                    let tx = db.begin_read().expect("failed");
                    tx_ids.push(tx.id());
                    tx.rollback().expect("failed");
                }

                tx_ids
            })
        })
        .collect();

    let mut all_ids: Vec<u64> = Vec::new();
    for handle in handles {
        all_ids.extend(handle.join().expect("thread panicked"));
    }

    // All IDs should be unique
    let unique_count = all_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, all_ids.len(), "all transaction IDs should be unique");
}
