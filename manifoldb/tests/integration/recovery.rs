//! Crash recovery integration tests.
//!
//! Tests for durability and recovery including:
//! - Data persistence across restarts
//! - WAL recovery after simulated crashes
//! - Uncommitted transaction rollback
//! - Checkpoint and recovery

use manifoldb::{Database, DatabaseBuilder, EntityId, Value};
use tempfile::tempdir;

// ============================================================================
// Basic Persistence Tests
// ============================================================================

/// Committed data survives database close and reopen
#[test]
fn test_basic_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("persist.manifold");

    let entity_ids: Vec<EntityId>;

    // Phase 1: Create and commit data
    {
        let db = Database::open(&db_path).expect("failed to open db");

        let mut tx = db.begin().expect("failed to begin");
        let mut ids = Vec::new();

        for i in 0..10 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("TestEntity")
                .with_property("index", i as i64)
                .with_property("data", format!("item_{i}"));
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed to commit");
        db.flush().expect("failed to flush");
        entity_ids = ids;
    }

    // Phase 2: Reopen and verify
    {
        let db = Database::open(&db_path).expect("failed to reopen db");
        let tx = db.begin_read().expect("failed to begin read");

        for (i, &id) in entity_ids.iter().enumerate() {
            let entity = tx.get_entity(id).expect("failed").expect("entity should exist");
            assert_eq!(entity.get_property("index"), Some(&Value::Int(i as i64)));
            assert_eq!(entity.get_property("data"), Some(&Value::String(format!("item_{i}"))));
        }
    }
}

/// Edges persist across database restarts
#[test]
fn test_edge_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("edges.manifold");

    let (src_id, dst_id): (EntityId, EntityId);

    // Phase 1: Create entities and edge
    {
        let db = Database::open(&db_path).expect("failed to open db");
        let mut tx = db.begin().expect("failed");

        let src = tx.create_entity().expect("failed").with_label("Source");
        let dst = tx.create_entity().expect("failed").with_label("Destination");
        src_id = src.id;
        dst_id = dst.id;

        tx.put_entity(&src).expect("failed");
        tx.put_entity(&dst).expect("failed");

        let edge = tx
            .create_edge(src_id, dst_id, "CONNECTS")
            .expect("failed")
            .with_property("weight", 42i64);
        tx.put_edge(&edge).expect("failed");

        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Verify edges persist
    {
        let db = Database::open(&db_path).expect("failed to reopen");
        let tx = db.begin_read().expect("failed");

        let edges = tx.get_outgoing_edges(src_id).expect("failed");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, src_id);
        assert_eq!(edges[0].target, dst_id);
        assert_eq!(edges[0].edge_type.as_str(), "CONNECTS");
        assert_eq!(edges[0].get_property("weight"), Some(&Value::Int(42)));
    }
}

/// Multiple transactions persist correctly
#[test]
fn test_multiple_transaction_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("multi.manifold");

    // Phase 1: Multiple separate transactions
    {
        let db = Database::open(&db_path).expect("failed");

        for batch in 0..5 {
            let mut tx = db.begin().expect("failed");

            for i in 0..10 {
                let entity = tx
                    .create_entity()
                    .expect("failed")
                    .with_property("batch", batch as i64)
                    .with_property("index", i as i64);
                tx.put_entity(&entity).expect("failed");
            }

            tx.commit().expect("failed");
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify all batches
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let mut found_count = 0;
        for id in 1..=100 {
            if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
                found_count += 1;
            }
        }

        assert_eq!(found_count, 50, "all entities should persist");
    }
}

// ============================================================================
// Uncommitted Transaction Tests
// ============================================================================

/// Uncommitted transactions are not visible after restart
#[test]
fn test_uncommitted_not_persisted() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("uncommitted.manifold");

    let committed_id: EntityId;
    let uncommitted_id: EntityId;

    // Phase 1: Create committed and uncommitted data
    {
        let db = Database::open(&db_path).expect("failed");

        // Committed transaction
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("status", "committed");
            committed_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        // Uncommitted transaction - drop without commit
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("status", "uncommitted");
            uncommitted_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            // No commit!
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify only committed data exists
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        // Committed entity should exist
        let committed = tx.get_entity(committed_id).expect("failed");
        assert!(committed.is_some(), "committed entity should exist");

        // Uncommitted entity should not exist
        let uncommitted = tx.get_entity(uncommitted_id).expect("failed");
        assert!(uncommitted.is_none(), "uncommitted entity should not exist");
    }
}

/// Rolled back transactions are not persisted
#[test]
fn test_rollback_not_persisted() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("rollback.manifold");

    let entity_id: EntityId;

    // Phase 1: Create and rollback
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("data", "test");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.rollback().expect("failed");

        db.flush().expect("failed");
    }

    // Phase 2: Verify nothing was persisted
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let result = tx.get_entity(entity_id).expect("failed");
        assert!(result.is_none(), "rolled back entity should not exist");
    }
}

// ============================================================================
// Update and Delete Persistence Tests
// ============================================================================

/// Entity updates persist correctly
#[test]
fn test_update_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("update.manifold");

    let entity_id: EntityId;

    // Phase 1: Create and update
    {
        let db = Database::open(&db_path).expect("failed");

        // Create
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_property("version", 1i64)
                .with_property("data", "original");
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        // Update multiple times
        for version in 2..=5 {
            let mut tx = db.begin().expect("failed");
            let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
            entity.set_property("version", version as i64);
            entity.set_property("data", format!("version_{version}"));
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify final state
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("version"), Some(&Value::Int(5)));
        assert_eq!(entity.get_property("data"), Some(&Value::String("version_5".to_string())));
    }
}

/// Entity deletes persist correctly
#[test]
fn test_delete_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("delete.manifold");

    let kept_id: EntityId;
    let deleted_id: EntityId;

    // Phase 1: Create and delete
    {
        let db = Database::open(&db_path).expect("failed");

        // Create two entities
        {
            let mut tx = db.begin().expect("failed");

            let kept = tx.create_entity().expect("failed").with_property("status", "kept");
            let deleted = tx.create_entity().expect("failed").with_property("status", "deleted");

            kept_id = kept.id;
            deleted_id = deleted.id;

            tx.put_entity(&kept).expect("failed");
            tx.put_entity(&deleted).expect("failed");
            tx.commit().expect("failed");
        }

        // Delete one
        {
            let mut tx = db.begin().expect("failed");
            tx.delete_entity(deleted_id).expect("failed");
            tx.commit().expect("failed");
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify delete persisted
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        assert!(tx.get_entity(kept_id).expect("failed").is_some());
        assert!(tx.get_entity(deleted_id).expect("failed").is_none());
    }
}

// ============================================================================
// Crash Simulation Tests
// ============================================================================

/// Simulate crash by not flushing - verify recovery
#[test]
fn test_recovery_without_flush() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("noflush.manifold");

    let entity_id: EntityId;

    // Phase 1: Create data and commit (redb auto-flushes on commit)
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("important", true);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
        // Intentionally not calling flush()
    }

    // Phase 2: Reopen and verify (redb should have written on commit)
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let entity = tx.get_entity(entity_id).expect("failed");
        // Committed data should survive (redb durability)
        assert!(entity.is_some(), "committed data should survive");
    }
}

/// Large transaction that commits should persist
#[test]
fn test_large_transaction_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("large.manifold");

    let entity_count = 1000;
    let first_id: EntityId;
    let last_id: EntityId;

    // Phase 1: Large write transaction
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");

        let first = tx.create_entity().expect("failed").with_property("index", 0i64);
        first_id = first.id;
        tx.put_entity(&first).expect("failed");

        for i in 1..entity_count - 1 {
            let entity = tx.create_entity().expect("failed").with_property("index", i as i64);
            tx.put_entity(&entity).expect("failed");
        }

        let last =
            tx.create_entity().expect("failed").with_property("index", (entity_count - 1) as i64);
        last_id = last.id;
        tx.put_entity(&last).expect("failed");

        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Verify all data
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let first = tx.get_entity(first_id).expect("failed").expect("should exist");
        assert_eq!(first.get_property("index"), Some(&Value::Int(0)));

        let last = tx.get_entity(last_id).expect("failed").expect("should exist");
        assert_eq!(last.get_property("index"), Some(&Value::Int((entity_count - 1) as i64)));

        // Count all entities
        let mut count = 0;
        for id in first_id.as_u64()..=last_id.as_u64() {
            if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
                count += 1;
            }
        }
        assert_eq!(count, entity_count);
    }
}

// ============================================================================
// Interleaved Commit/Rollback Persistence
// ============================================================================

/// Interleaved commits and rollbacks persist correctly
#[test]
fn test_interleaved_commit_rollback_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("interleaved.manifold");

    let mut committed_count = 0;
    let mut committed_indices = Vec::new();

    // Phase 1: Interleaved commits and rollbacks
    {
        let db = Database::open(&db_path).expect("failed");

        for i in 0..20 {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("index", i);
            tx.put_entity(&entity).expect("failed");

            if i % 2 == 0 {
                tx.commit().expect("failed");
                committed_count += 1;
                committed_indices.push(i);
            } else {
                tx.rollback().expect("failed");
            }
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify correct persistence
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        // Count total entities - should be exactly committed_count
        let mut found_count = 0;
        let mut found_indices = Vec::new();

        // Scan all possible IDs
        for id_val in 1..=100 {
            if let Some(entity) = tx.get_entity(EntityId::new(id_val)).expect("failed") {
                found_count += 1;
                if let Some(manifoldb::Value::Int(index)) = entity.get_property("index") {
                    found_indices.push(*index);
                }
            }
        }

        assert_eq!(
            found_count, committed_count,
            "should have exactly {} committed entities, found {}",
            committed_count, found_count
        );

        // Verify the indices match (committed were even indices)
        for idx in &found_indices {
            assert!(
                idx % 2 == 0,
                "found entity with index {} which should have been rolled back",
                idx
            );
        }
    }
}

// ============================================================================
// Database Configuration Persistence
// ============================================================================

/// Database configuration doesn't affect stored data
#[test]
fn test_config_independence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("config.manifold");

    let entity_id: EntityId;

    // Phase 1: Create with one configuration
    {
        let db = DatabaseBuilder::new()
            .path(&db_path)
            .cache_size(1024 * 1024) // 1MB cache
            .open()
            .expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("test", "data");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Reopen with different configuration
    {
        let db = DatabaseBuilder::new()
            .path(&db_path)
            .cache_size(10 * 1024 * 1024) // 10MB cache
            .open()
            .expect("failed");

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("test"), Some(&Value::String("data".to_string())));
    }
}

// ============================================================================
// Vector Data Persistence
// ============================================================================

/// Vector properties persist correctly
#[test]
fn test_vector_property_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("vectors.manifold");

    let entity_id: EntityId;
    let expected_vector = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];

    // Phase 1: Create with vector property
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity = tx
            .create_entity()
            .expect("failed")
            .with_label("Document")
            .with_property("embedding", expected_vector.clone());
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Verify vector persists
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        let vector = entity.get_property("embedding");

        match vector {
            Some(Value::Vector(v)) => {
                assert_eq!(v.len(), expected_vector.len());
                for (a, b) in v.iter().zip(expected_vector.iter()) {
                    assert!((a - b).abs() < f32::EPSILON, "vector values should match");
                }
            }
            _ => panic!("expected vector property"),
        }
    }
}

// ============================================================================
// Sequential Operations Persistence
// ============================================================================

/// Many sequential small transactions persist correctly
#[test]
fn test_many_small_transactions() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("manysmall.manifold");

    let transaction_count = 100;

    // Phase 1: Many small transactions
    {
        let db = Database::open(&db_path).expect("failed");

        for i in 0..transaction_count {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("txn", i);
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        db.flush().expect("failed");
    }

    // Phase 2: Verify all
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let mut count = 0;
        for id in 1..=(transaction_count as u64) {
            if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
                count += 1;
            }
        }

        assert_eq!(count, transaction_count);
    }
}

// ============================================================================
// Partial Write Simulation
// ============================================================================

/// Database can be reopened after normal close
#[test]
fn test_multiple_open_close_cycles() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("cycles.manifold");

    for cycle in 0..5 {
        // Open, write, close
        {
            let db = Database::open(&db_path).expect("failed");

            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("cycle", cycle);
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
            db.flush().expect("failed");
        }

        // Verify current state
        {
            let db = Database::open(&db_path).expect("failed");
            let tx = db.begin_read().expect("failed");

            // Should have cycle+1 entities
            let mut count = 0;
            for id in 1..=100 {
                if tx.get_entity(EntityId::new(id)).expect("failed").is_some() {
                    count += 1;
                }
            }
            assert_eq!(count, cycle + 1, "cycle {cycle} should have correct count");
        }
    }
}

// ============================================================================
// Property Types Persistence
// ============================================================================

/// All property types persist correctly
#[test]
fn test_all_property_types_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("types.manifold");

    let entity_id: EntityId;

    // Phase 1: Create with all property types
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity = tx
            .create_entity()
            .expect("failed")
            .with_property("string", "hello")
            .with_property("int", 42i64)
            .with_property("float", 3.14159f64)
            .with_property("bool_true", true)
            .with_property("bool_false", false)
            .with_property("vector", vec![1.0f32, 2.0, 3.0]);

        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Verify all types
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

        assert_eq!(entity.get_property("string"), Some(&Value::String("hello".to_string())));
        assert_eq!(entity.get_property("int"), Some(&Value::Int(42)));

        if let Some(Value::Float(f)) = entity.get_property("float") {
            assert!((f - 3.14159).abs() < 0.00001);
        } else {
            panic!("expected float property");
        }

        assert_eq!(entity.get_property("bool_true"), Some(&Value::Bool(true)));
        assert_eq!(entity.get_property("bool_false"), Some(&Value::Bool(false)));

        if let Some(Value::Vector(v)) = entity.get_property("vector") {
            assert_eq!(v.len(), 3);
        } else {
            panic!("expected vector property");
        }
    }
}

// ============================================================================
// Label Persistence
// ============================================================================

/// Entity labels persist correctly
#[test]
fn test_label_persistence() {
    let dir = tempdir().expect("failed to create temp dir");
    let db_path = dir.path().join("labels.manifold");

    let entity_id: EntityId;

    // Phase 1: Create with labels
    {
        let db = Database::open(&db_path).expect("failed");

        let mut tx = db.begin().expect("failed");
        let entity =
            tx.create_entity().expect("failed").with_label("Person").with_label("Employee");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
        db.flush().expect("failed");
    }

    // Phase 2: Verify labels
    {
        let db = Database::open(&db_path).expect("failed");
        let tx = db.begin_read().expect("failed");

        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        assert!(entity.has_label("Person"));
        assert!(entity.has_label("Employee"));
    }
}
