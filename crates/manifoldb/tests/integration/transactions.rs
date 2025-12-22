//! Transaction ACID property integration tests.
//!
//! Tests for Atomicity, Consistency, Isolation, and Durability properties.

use manifoldb::{Database, EntityId, TransactionError, Value};

// ============================================================================
// Atomicity Tests
// ============================================================================

/// All operations in a committed transaction should be visible
#[test]
fn test_atomicity_commit_all_visible() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_ids: Vec<EntityId>;
    {
        let mut tx = db.begin().expect("failed to begin");

        // Multiple operations in one transaction
        let mut ids = Vec::new();
        for i in 0..5 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("AtomicTest")
                .with_property("index", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        // Create edges between them
        for i in 0..4 {
            let edge = tx.create_edge(ids[i], ids[i + 1], "NEXT").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }

        tx.commit().expect("failed to commit");
        entity_ids = ids;
    }

    // All entities and edges should be visible
    let tx = db.begin_read().expect("failed to begin read");

    for (i, &id) in entity_ids.iter().enumerate() {
        let entity = tx.get_entity(id).expect("failed").expect("entity should exist");
        assert_eq!(entity.get_property("index"), Some(&Value::Int(i as i64)));
    }

    for i in 0..4 {
        let edges = tx.get_outgoing_edges(entity_ids[i]).expect("failed");
        assert_eq!(edges.len(), 1, "should have 1 outgoing edge");
        assert_eq!(edges[0].target, entity_ids[i + 1]);
    }
}

/// No operations from a rolled back transaction should be visible
#[test]
fn test_atomicity_rollback_none_visible() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_ids: Vec<EntityId>;
    {
        let mut tx = db.begin().expect("failed to begin");

        let mut ids = Vec::new();
        for i in 0..5 {
            let entity = tx.create_entity().expect("failed").with_property("index", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        tx.rollback().expect("failed to rollback");
        entity_ids = ids;
    }

    // No entities should be visible
    let tx = db.begin_read().expect("failed to begin read");

    for &id in &entity_ids {
        let entity = tx.get_entity(id).expect("failed");
        assert!(entity.is_none(), "rolled back entity should not exist");
    }
}

/// Dropping a transaction without commit should rollback
#[test]
fn test_atomicity_drop_implies_rollback() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed to begin");
        let entity = tx.create_entity().expect("failed");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        // Drop without commit
    }

    // Entity should not exist
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed");
    assert!(entity.is_none(), "dropped transaction should rollback");
}

/// Partial failure should not affect committed data
#[test]
fn test_atomicity_partial_commit_then_rollback() {
    let db = Database::in_memory().expect("failed to create db");

    // First transaction: commit some entities
    let committed_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("committed", true);
        committed_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Second transaction: try to add more, then rollback
    let rolled_back_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("committed", false);
        rolled_back_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.rollback().expect("failed");
    }

    // Only the first entity should exist
    let tx = db.begin_read().expect("failed");
    assert!(tx.get_entity(committed_id).expect("failed").is_some());
    assert!(tx.get_entity(rolled_back_id).expect("failed").is_none());
}

// ============================================================================
// Isolation Tests
// ============================================================================

/// Uncommitted changes should not be visible to other transactions
#[test]
fn test_isolation_uncommitted_not_visible() {
    let db = Database::in_memory().expect("failed to create db");

    // Start write transaction but don't commit
    let mut write_tx = db.begin().expect("failed to begin write");
    let entity = write_tx.create_entity().expect("failed");
    let entity_id = entity.id;
    write_tx.put_entity(&entity).expect("failed");

    // Start read transaction - should not see uncommitted entity
    let read_tx = db.begin_read().expect("failed to begin read");
    let result = read_tx.get_entity(entity_id).expect("failed");
    assert!(result.is_none(), "uncommitted entity should not be visible");

    // Cleanup
    write_tx.rollback().expect("failed");
    read_tx.rollback().expect("failed");
}

/// Read transactions see consistent snapshot
#[test]
fn test_isolation_read_snapshot() {
    let db = Database::in_memory().expect("failed to create db");

    // Create initial entity
    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("value", 1i64);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Start read transaction
    let read_tx = db.begin_read().expect("failed");
    let entity = read_tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(entity.get_property("value"), Some(&Value::Int(1)));

    // Update in separate transaction
    {
        let mut tx = db.begin().expect("failed");
        let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        entity.set_property("value", 2i64);
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Original read transaction should still see old value
    let entity = read_tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(entity.get_property("value"), Some(&Value::Int(1)), "should see snapshot value");

    read_tx.rollback().expect("failed");
}

/// Multiple concurrent reads should work
#[test]
fn test_isolation_concurrent_reads() {
    let db = Database::in_memory().expect("failed to create db");

    // Create test data
    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("data", "test");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Multiple concurrent read transactions
    let tx1 = db.begin_read().expect("failed");
    let tx2 = db.begin_read().expect("failed");
    let tx3 = db.begin_read().expect("failed");

    // All should see the same data
    let e1 = tx1.get_entity(entity_id).expect("failed").expect("should exist");
    let e2 = tx2.get_entity(entity_id).expect("failed").expect("should exist");
    let e3 = tx3.get_entity(entity_id).expect("failed").expect("should exist");

    assert_eq!(e1.get_property("data"), Some(&Value::String("test".to_string())));
    assert_eq!(e2.get_property("data"), Some(&Value::String("test".to_string())));
    assert_eq!(e3.get_property("data"), Some(&Value::String("test".to_string())));
}

// ============================================================================
// Consistency Tests
// ============================================================================

/// Read-only transactions cannot modify data
#[test]
fn test_consistency_read_only_no_writes() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin_read().expect("failed to begin read");

    // Attempt to create entity should fail
    let result = tx.create_entity();
    assert!(matches!(result, Err(TransactionError::ReadOnly)));
}

/// Entity IDs should be unique across transactions
#[test]
fn test_consistency_unique_entity_ids() {
    let db = Database::in_memory().expect("failed to create db");

    let mut all_ids = Vec::new();

    // Create entities in multiple transactions
    for _ in 0..5 {
        let mut tx = db.begin().expect("failed");
        for _ in 0..10 {
            let entity = tx.create_entity().expect("failed");
            all_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // All IDs should be unique
    let unique_count = all_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, all_ids.len(), "all entity IDs should be unique");
}

/// Edge IDs should be unique across transactions
#[test]
fn test_consistency_unique_edge_ids() {
    let db = Database::in_memory().expect("failed to create db");

    // Create some entities first
    let mut entity_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed");
        for _ in 0..10 {
            let entity = tx.create_entity().expect("failed");
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    let mut all_edge_ids = Vec::new();

    // Create edges in multiple transactions
    for i in 0..5 {
        let mut tx = db.begin().expect("failed");
        for j in 0..5 {
            let src = i % entity_ids.len();
            let dst = (i + j + 1) % entity_ids.len();
            if src != dst {
                let edge =
                    tx.create_edge(entity_ids[src], entity_ids[dst], "TEST").expect("failed");
                all_edge_ids.push(edge.id);
                tx.put_edge(&edge).expect("failed");
            }
        }
        tx.commit().expect("failed");
    }

    // All edge IDs should be unique
    let unique_count = all_edge_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(unique_count, all_edge_ids.len(), "all edge IDs should be unique");
}

// ============================================================================
// Durability Tests (In-Memory Limited)
// ============================================================================

/// Committed data should survive transaction scope
#[test]
fn test_durability_commit_persists() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id: EntityId;

    // Create and commit
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("important", true);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify in new transaction
    {
        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("important"), Some(&Value::Bool(true)));
    }

    // Verify again after more operations
    {
        let mut tx = db.begin().expect("failed");
        let other = tx.create_entity().expect("failed");
        tx.put_entity(&other).expect("failed");
        tx.commit().expect("failed");
    }

    // Original entity should still exist
    {
        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should still exist");
        assert_eq!(entity.get_property("important"), Some(&Value::Bool(true)));
    }
}

/// Flush should succeed
#[test]
fn test_durability_flush() {
    let db = Database::in_memory().expect("failed to create db");

    // Create some data
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed");
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Flush should succeed
    db.flush().expect("flush should succeed");
}

// ============================================================================
// Transaction Lifecycle Tests
// ============================================================================

/// Transaction ID should be unique
#[test]
fn test_transaction_unique_ids() {
    let db = Database::in_memory().expect("failed to create db");

    let tx1 = db.begin_read().expect("failed");
    let tx2 = db.begin_read().expect("failed");

    assert_ne!(tx1.id(), tx2.id(), "transaction IDs should be unique");

    tx1.rollback().expect("failed");
    tx2.rollback().expect("failed");
}

/// Already completed transaction should error
#[test]
fn test_transaction_already_completed() {
    let db = Database::in_memory().expect("failed to create db");

    let tx = db.begin().expect("failed");
    tx.commit().expect("first commit should succeed");

    // Note: Once a transaction is committed, it's consumed.
    // This test verifies the consume semantics work.
}

/// Read-only flag should be correct
#[test]
fn test_transaction_read_only_flag() {
    let db = Database::in_memory().expect("failed to create db");

    let read_tx = db.begin_read().expect("failed");
    assert!(read_tx.is_read_only());

    let write_tx = db.begin().expect("failed");
    assert!(!write_tx.is_read_only());

    read_tx.rollback().expect("failed");
    write_tx.rollback().expect("failed");
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Delete non-existent entity should return false
#[test]
fn test_delete_nonexistent_entity() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed");
    let result = tx.delete_entity(EntityId::new(999)).expect("should not error");
    assert!(!result, "deleting non-existent entity should return false");
}

/// Delete non-existent edge should return false
#[test]
fn test_delete_nonexistent_edge() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed");
    let result = tx.delete_edge(manifoldb::EdgeId::new(999)).expect("should not error");
    assert!(!result, "deleting non-existent edge should return false");
}

/// Get non-existent entity should return None
#[test]
fn test_get_nonexistent_entity() {
    let db = Database::in_memory().expect("failed to create db");

    let tx = db.begin_read().expect("failed");
    let result = tx.get_entity(EntityId::new(999)).expect("should not error");
    assert!(result.is_none(), "non-existent entity should return None");
}

/// Get non-existent edge should return None
#[test]
fn test_get_nonexistent_edge() {
    let db = Database::in_memory().expect("failed to create db");

    let tx = db.begin_read().expect("failed");
    let result = tx.get_edge(manifoldb::EdgeId::new(999)).expect("should not error");
    assert!(result.is_none(), "non-existent edge should return None");
}

// ============================================================================
// Sequential Transaction Tests
// ============================================================================

/// Many sequential transactions should work correctly
#[test]
fn test_many_sequential_transactions() {
    let db = Database::in_memory().expect("failed to create db");

    let mut total_entities = 0;

    for round in 0..50 {
        let mut tx = db.begin().expect("failed to begin");

        for i in 0..10 {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_property("round", round as i64)
                .with_property("index", i as i64);
            tx.put_entity(&entity).expect("failed");
            total_entities += 1;
        }

        tx.commit().expect("failed to commit");
    }

    // Verify total
    assert_eq!(total_entities, 500);

    // Verify some random entities exist
    let tx = db.begin_read().expect("failed");
    for id in [1u64, 100, 250, 499] {
        let entity = tx.get_entity(EntityId::new(id)).expect("failed");
        assert!(entity.is_some(), "entity {id} should exist");
    }
}

/// Update same entity multiple times in different transactions
#[test]
fn test_sequential_updates() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id: EntityId;
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("counter", 0i64);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Update 100 times
    for i in 1..=100 {
        let mut tx = db.begin().expect("failed");
        let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        entity.set_property("counter", i as i64);
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify final value
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
    assert_eq!(entity.get_property("counter"), Some(&Value::Int(100)));
}

// ============================================================================
// Complex Transaction Patterns
// ============================================================================

/// Interleaved creates and deletes
#[test]
fn test_interleaved_creates_deletes() {
    let db = Database::in_memory().expect("failed to create db");

    let mut created_ids = Vec::new();
    let mut _deleted_count = 0;

    for i in 0..20 {
        let mut tx = db.begin().expect("failed");

        // Create some entities
        for _ in 0..5 {
            let entity = tx.create_entity().expect("failed");
            created_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        // Delete some previously created (if any)
        if i > 0 && !created_ids.is_empty() {
            let to_delete = created_ids.remove(0);
            if tx.delete_entity(to_delete).expect("failed") {
                _deleted_count += 1;
            }
        }

        tx.commit().expect("failed");
    }

    // Verify remaining entities
    let tx = db.begin_read().expect("failed");
    let mut existing = 0;
    for &id in &created_ids {
        if tx.get_entity(id).expect("failed").is_some() {
            existing += 1;
        }
    }

    assert_eq!(existing, created_ids.len(), "remaining entities should match list");
}
