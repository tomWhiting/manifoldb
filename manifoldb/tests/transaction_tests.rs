//! Integration tests for the transaction manager.

use manifoldb::transaction::{TransactionManager, VectorSyncStrategy};
use manifoldb::{EntityId, TransactionError};
use manifoldb_storage::backends::RedbEngine;

/// Create an in-memory engine for testing.
fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("failed to create in-memory engine")
}

// ============================================================================
// Basic Transaction Tests
// ============================================================================

#[test]
fn test_create_transaction_manager() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    assert_eq!(manager.vector_sync_strategy(), VectorSyncStrategy::Synchronous);
}

#[test]
fn test_begin_read_transaction() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let tx = manager.begin_read().expect("failed to begin read transaction");
    assert!(tx.is_read_only());
    tx.rollback().expect("failed to rollback");
}

#[test]
fn test_begin_write_transaction() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let tx = manager.begin_write().expect("failed to begin write transaction");
    assert!(!tx.is_read_only());
    tx.rollback().expect("failed to rollback");
}

#[test]
fn test_transaction_ids_are_unique() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let tx1 = manager.begin_read().expect("failed to begin tx1");
    let tx2 = manager.begin_read().expect("failed to begin tx2");

    assert_ne!(tx1.id(), tx2.id());

    tx1.rollback().expect("failed to rollback tx1");
    tx2.rollback().expect("failed to rollback tx2");
}

// ============================================================================
// Entity CRUD Tests
// ============================================================================

#[test]
fn test_create_and_get_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create and store entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx
        .create_entity()
        .expect("failed to create entity")
        .with_label("Person")
        .with_property("name", "Alice");

    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Read entity back
    let tx = manager.begin_read().expect("failed to begin read");
    let retrieved =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert_eq!(retrieved.id, entity_id);
    assert!(retrieved.has_label("Person"));
    assert_eq!(
        retrieved.get_property("name"),
        Some(&manifoldb::Value::String("Alice".to_string()))
    );
}

#[test]
fn test_update_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity").with_property("count", 1i64);
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Update entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let mut entity =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    entity.set_property("count", 2i64);
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Verify update
    let tx = manager.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert_eq!(entity.get_property("count"), Some(&manifoldb::Value::Int(2)));
}

#[test]
fn test_delete_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Delete entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_entity(entity_id).expect("failed to delete entity");
    assert!(deleted);
    tx.commit().expect("failed to commit");

    // Verify deletion
    let tx = manager.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed to get entity");
    assert!(entity.is_none());
}

#[test]
fn test_delete_nonexistent_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_entity(EntityId::new(999)).expect("failed to delete");
    assert!(!deleted);
}

#[test]
fn test_get_nonexistent_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let tx = manager.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(EntityId::new(999)).expect("failed to get entity");
    assert!(entity.is_none());
}

// ============================================================================
// Edge CRUD Tests
// ============================================================================

#[test]
fn test_create_and_get_edge() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities first
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity1 = tx.create_entity().expect("failed to create entity1");
    let entity2 = tx.create_entity().expect("failed to create entity2");
    tx.put_entity(&entity1).expect("failed to put entity1");
    tx.put_entity(&entity2).expect("failed to put entity2");

    // Create edge
    let edge = tx
        .create_edge(entity1.id, entity2.id, "FOLLOWS")
        .expect("failed to create edge")
        .with_property("since", "2024");
    let edge_id = edge.id;
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Read edge back
    let tx = manager.begin_read().expect("failed to begin read");
    let retrieved = tx.get_edge(edge_id).expect("failed to get edge").expect("edge not found");

    assert_eq!(retrieved.id, edge_id);
    assert_eq!(retrieved.source, entity1.id);
    assert_eq!(retrieved.target, entity2.id);
    assert_eq!(retrieved.edge_type.as_str(), "FOLLOWS");
    assert_eq!(
        retrieved.get_property("since"),
        Some(&manifoldb::Value::String("2024".to_string()))
    );
}

#[test]
fn test_delete_edge() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities and edge
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity1 = tx.create_entity().expect("failed to create entity1");
    let entity2 = tx.create_entity().expect("failed to create entity2");
    tx.put_entity(&entity1).expect("failed to put entity1");
    tx.put_entity(&entity2).expect("failed to put entity2");
    let edge = tx.create_edge(entity1.id, entity2.id, "KNOWS").expect("failed to create edge");
    let edge_id = edge.id;
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Delete edge
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_edge(edge_id).expect("failed to delete edge");
    assert!(deleted);
    tx.commit().expect("failed to commit");

    // Verify deletion
    let tx = manager.begin_read().expect("failed to begin read");
    let edge = tx.get_edge(edge_id).expect("failed to get edge");
    assert!(edge.is_none());
}

// ============================================================================
// Traversal Tests
// ============================================================================

#[test]
fn test_get_outgoing_edges() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> B, A -> C
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    let c = tx.create_entity().expect("failed to create C");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    tx.put_entity(&c).expect("failed to put C");

    let edge_ab = tx.create_edge(a.id, b.id, "LINKS").expect("failed to create edge A->B");
    let edge_ac = tx.create_edge(a.id, c.id, "LINKS").expect("failed to create edge A->C");
    tx.put_edge(&edge_ab).expect("failed to put edge A->B");
    tx.put_edge(&edge_ac).expect("failed to put edge A->C");
    tx.commit().expect("failed to commit");

    // Get outgoing edges from A
    let tx = manager.begin_read().expect("failed to begin read");
    let edges = tx.get_outgoing_edges(a.id).expect("failed to get outgoing edges");
    assert_eq!(edges.len(), 2);

    // Verify edge targets
    let targets: std::collections::HashSet<_> = edges.iter().map(|e| e.target).collect();
    assert!(targets.contains(&b.id));
    assert!(targets.contains(&c.id));

    // B should have no outgoing edges
    let edges = tx.get_outgoing_edges(b.id).expect("failed to get outgoing edges from B");
    assert!(edges.is_empty());
}

#[test]
fn test_get_incoming_edges() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> C, B -> C
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    let c = tx.create_entity().expect("failed to create C");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    tx.put_entity(&c).expect("failed to put C");

    let edge_ac = tx.create_edge(a.id, c.id, "POINTS_TO").expect("failed to create edge A->C");
    let edge_bc = tx.create_edge(b.id, c.id, "POINTS_TO").expect("failed to create edge B->C");
    tx.put_edge(&edge_ac).expect("failed to put edge A->C");
    tx.put_edge(&edge_bc).expect("failed to put edge B->C");
    tx.commit().expect("failed to commit");

    // Get incoming edges to C
    let tx = manager.begin_read().expect("failed to begin read");
    let edges = tx.get_incoming_edges(c.id).expect("failed to get incoming edges");
    assert_eq!(edges.len(), 2);

    // Verify edge sources
    let sources: std::collections::HashSet<_> = edges.iter().map(|e| e.source).collect();
    assert!(sources.contains(&a.id));
    assert!(sources.contains(&b.id));

    // A should have no incoming edges
    let edges = tx.get_incoming_edges(a.id).expect("failed to get incoming edges to A");
    assert!(edges.is_empty());
}

// ============================================================================
// ACID Property Tests
// ============================================================================

#[test]
fn test_atomicity_commit() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Multiple operations in one transaction
    let mut tx = manager.begin_write().expect("failed to begin write");
    let e1 = tx.create_entity().expect("failed to create e1");
    let e2 = tx.create_entity().expect("failed to create e2");
    tx.put_entity(&e1).expect("failed to put e1");
    tx.put_entity(&e2).expect("failed to put e2");
    tx.commit().expect("failed to commit");

    // Both should be visible
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(e1.id).expect("failed to get e1").is_some());
    assert!(tx.get_entity(e2.id).expect("failed to get e2").is_some());
}

#[test]
fn test_atomicity_rollback() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create one entity and commit
    let mut tx = manager.begin_write().expect("failed to begin write");
    let committed_entity = tx.create_entity().expect("failed to create committed entity");
    tx.put_entity(&committed_entity).expect("failed to put committed entity");
    tx.commit().expect("failed to commit");

    // Start new transaction, add entity, then rollback
    let mut tx = manager.begin_write().expect("failed to begin write");
    let rolled_back_entity = tx.create_entity().expect("failed to create rolled back entity");
    tx.put_entity(&rolled_back_entity).expect("failed to put rolled back entity");
    tx.rollback().expect("failed to rollback");

    // Only committed entity should be visible
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(committed_entity.id).expect("failed to get committed entity").is_some());
    assert!(tx
        .get_entity(rolled_back_entity.id)
        .expect("failed to get rolled back entity")
        .is_none());
}

#[test]
fn test_isolation_uncommitted_not_visible() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Start a write transaction (don't commit yet)
    let mut write_tx = manager.begin_write().expect("failed to begin write");
    let entity = write_tx.create_entity().expect("failed to create entity");
    write_tx.put_entity(&entity).expect("failed to put entity");

    // Start a read transaction - should not see uncommitted entity
    let read_tx = manager.begin_read().expect("failed to begin read");
    let result = read_tx.get_entity(entity.id).expect("failed to get entity");
    assert!(result.is_none(), "uncommitted entity should not be visible");

    // Cleanup
    write_tx.rollback().expect("failed to rollback");
    read_tx.rollback().expect("failed to rollback");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_read_only_transaction_cannot_write() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let mut tx = manager.begin_read().expect("failed to begin read");

    // Try to create entity - should fail
    let result = tx.create_entity();
    assert!(matches!(result, Err(TransactionError::ReadOnly)));
}

#[test]
fn test_transaction_drop_triggers_rollback() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let entity_id;
    {
        // Create entity but don't commit - let it drop
        let mut tx = manager.begin_write().expect("failed to begin write");
        let entity = tx.create_entity().expect("failed to create entity");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed to put entity");
        // tx drops here without commit
    }

    // Entity should not be visible (rolled back)
    let tx = manager.begin_read().expect("failed to begin read");
    let result = tx.get_entity(entity_id).expect("failed to get entity");
    assert!(result.is_none(), "dropped transaction should rollback");
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_vector_sync_strategy_configuration() {
    use manifoldb::transaction::TransactionManagerConfig;

    let engine = create_test_engine();
    let config = TransactionManagerConfig { vector_sync_strategy: VectorSyncStrategy::Async };
    let manager = TransactionManager::with_config(engine, config);

    assert_eq!(manager.vector_sync_strategy(), VectorSyncStrategy::Async);
}

#[test]
fn test_hybrid_vector_sync_strategy() {
    use manifoldb::transaction::TransactionManagerConfig;

    let engine = create_test_engine();
    let config = TransactionManagerConfig {
        vector_sync_strategy: VectorSyncStrategy::Hybrid { async_threshold: 100 },
    };
    let manager = TransactionManager::with_config(engine, config);

    assert_eq!(manager.vector_sync_strategy(), VectorSyncStrategy::Hybrid { async_threshold: 100 });
}

// ============================================================================
// Concurrent Access Tests
// ============================================================================

#[test]
fn test_multiple_read_transactions() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create some data first
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity");
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Multiple concurrent read transactions should work
    let tx1 = manager.begin_read().expect("failed to begin read 1");
    let tx2 = manager.begin_read().expect("failed to begin read 2");
    let tx3 = manager.begin_read().expect("failed to begin read 3");

    // All should see the same data
    assert!(tx1.get_entity(entity.id).expect("failed to get from tx1").is_some());
    assert!(tx2.get_entity(entity.id).expect("failed to get from tx2").is_some());
    assert!(tx3.get_entity(entity.id).expect("failed to get from tx3").is_some());

    tx1.rollback().expect("failed to rollback tx1");
    tx2.rollback().expect("failed to rollback tx2");
    tx3.rollback().expect("failed to rollback tx3");
}

#[test]
fn test_entity_id_generation_across_transactions() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities in separate transactions
    let mut tx = manager.begin_write().expect("failed to begin write 1");
    let e1 = tx.create_entity().expect("failed to create e1");
    tx.put_entity(&e1).expect("failed to put e1");
    tx.commit().expect("failed to commit 1");

    let mut tx = manager.begin_write().expect("failed to begin write 2");
    let e2 = tx.create_entity().expect("failed to create e2");
    tx.put_entity(&e2).expect("failed to put e2");
    tx.commit().expect("failed to commit 2");

    // IDs should be unique
    assert_ne!(e1.id, e2.id);

    // IDs should be sequential (1, 2, ...)
    assert_eq!(e1.id.as_u64(), 1);
    assert_eq!(e2.id.as_u64(), 2);
}

#[test]
fn test_edge_id_generation_across_transactions() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities
    let mut tx = manager.begin_write().expect("failed to begin write");
    let e1 = tx.create_entity().expect("failed to create e1");
    let e2 = tx.create_entity().expect("failed to create e2");
    let e3 = tx.create_entity().expect("failed to create e3");
    tx.put_entity(&e1).expect("failed to put e1");
    tx.put_entity(&e2).expect("failed to put e2");
    tx.put_entity(&e3).expect("failed to put e3");

    let edge1 = tx.create_edge(e1.id, e2.id, "A").expect("failed to create edge1");
    tx.put_edge(&edge1).expect("failed to put edge1");
    tx.commit().expect("failed to commit 1");

    let mut tx = manager.begin_write().expect("failed to begin write 2");
    let edge2 = tx.create_edge(e2.id, e3.id, "B").expect("failed to create edge2");
    tx.put_edge(&edge2).expect("failed to put edge2");
    tx.commit().expect("failed to commit 2");

    // Edge IDs should be unique and sequential
    assert_ne!(edge1.id, edge2.id);
    assert_eq!(edge1.id.as_u64(), 1);
    assert_eq!(edge2.id.as_u64(), 2);
}
