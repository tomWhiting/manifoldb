//! Integration tests for the transaction manager.

use manifoldb::transaction::{TransactionManager, VectorSyncStrategy};
use manifoldb::{DeleteResult, EntityId, TransactionError};
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

#[test]
fn test_delete_edge_cleans_up_indexes() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> B -> C with multiple edges from A
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    let c = tx.create_entity().expect("failed to create C");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    tx.put_entity(&c).expect("failed to put C");

    // Create edges: A -> B, A -> C, B -> C
    let edge_ab = tx.create_edge(a.id, b.id, "CONNECTS").expect("failed to create edge A->B");
    let edge_ac = tx.create_edge(a.id, c.id, "CONNECTS").expect("failed to create edge A->C");
    let edge_bc = tx.create_edge(b.id, c.id, "CONNECTS").expect("failed to create edge B->C");
    let edge_ab_id = edge_ab.id;
    let edge_ac_id = edge_ac.id;
    let edge_bc_id = edge_bc.id;
    tx.put_edge(&edge_ab).expect("failed to put edge A->B");
    tx.put_edge(&edge_ac).expect("failed to put edge A->C");
    tx.put_edge(&edge_bc).expect("failed to put edge B->C");
    tx.commit().expect("failed to commit");

    // Verify initial state: A has 2 outgoing edges, B has 1 incoming and 1 outgoing
    let tx = manager.begin_read().expect("failed to begin read");
    let outgoing_a = tx.get_outgoing_edges(a.id).expect("failed to get outgoing from A");
    assert_eq!(outgoing_a.len(), 2);
    let incoming_b = tx.get_incoming_edges(b.id).expect("failed to get incoming to B");
    assert_eq!(incoming_b.len(), 1);
    let outgoing_b = tx.get_outgoing_edges(b.id).expect("failed to get outgoing from B");
    assert_eq!(outgoing_b.len(), 1);
    let incoming_c = tx.get_incoming_edges(c.id).expect("failed to get incoming to C");
    assert_eq!(incoming_c.len(), 2);
    tx.rollback().expect("failed to rollback");

    // Delete edge A->B
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_edge(edge_ab_id).expect("failed to delete edge A->B");
    assert!(deleted);
    tx.commit().expect("failed to commit");

    // Verify indexes are cleaned up:
    // - A should now have 1 outgoing edge (only A->C remains)
    // - B should have 0 incoming edges (A->B was deleted)
    // - The deleted edge should not appear in any traversals
    let tx = manager.begin_read().expect("failed to begin read");

    // Verify edge is gone
    assert!(tx.get_edge(edge_ab_id).expect("failed to get edge").is_none());

    // Verify outgoing index for A is correct
    let outgoing_a = tx.get_outgoing_edges(a.id).expect("failed to get outgoing from A");
    assert_eq!(outgoing_a.len(), 1);
    assert_eq!(outgoing_a[0].id, edge_ac_id);

    // Verify incoming index for B is correct (should be empty now)
    let incoming_b = tx.get_incoming_edges(b.id).expect("failed to get incoming to B");
    assert!(incoming_b.is_empty());

    // Verify B's outgoing edges are unaffected
    let outgoing_b = tx.get_outgoing_edges(b.id).expect("failed to get outgoing from B");
    assert_eq!(outgoing_b.len(), 1);
    assert_eq!(outgoing_b[0].id, edge_bc_id);

    // Verify C's incoming edges are correct (only A->C and B->C)
    let incoming_c = tx.get_incoming_edges(c.id).expect("failed to get incoming to C");
    assert_eq!(incoming_c.len(), 2);
    let incoming_ids: std::collections::HashSet<_> = incoming_c.iter().map(|e| e.id).collect();
    assert!(incoming_ids.contains(&edge_ac_id));
    assert!(incoming_ids.contains(&edge_bc_id));
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
    use manifoldb::transaction::{BatchWriterConfig, TransactionManagerConfig};

    let engine = create_test_engine();
    let config = TransactionManagerConfig {
        vector_sync_strategy: VectorSyncStrategy::Async,
        batch_writer_config: BatchWriterConfig::default(),
    };
    let manager = TransactionManager::with_config(engine, config);

    assert_eq!(manager.vector_sync_strategy(), VectorSyncStrategy::Async);
}

#[test]
fn test_hybrid_vector_sync_strategy() {
    use manifoldb::transaction::{BatchWriterConfig, TransactionManagerConfig};

    let engine = create_test_engine();
    let config = TransactionManagerConfig {
        vector_sync_strategy: VectorSyncStrategy::Hybrid { async_threshold: 100 },
        batch_writer_config: BatchWriterConfig::default(),
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

// ============================================================================
// Batch Operation Tests
// ============================================================================

#[test]
fn test_put_entities_batch() {
    use manifoldb::Entity;

    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities with specific IDs for batch insert
    let entities: Vec<Entity> = (1..=100u64)
        .map(|i| {
            Entity::new(EntityId::new(i)).with_label("Person").with_property("index", i as i64)
        })
        .collect();

    // Batch insert all entities
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_entities_batch(&entities).expect("failed to batch insert entities");
    tx.commit().expect("failed to commit");

    // Verify all entities were inserted
    let tx = manager.begin_read().expect("failed to begin read");
    for entity in &entities {
        let retrieved =
            tx.get_entity(entity.id).expect("failed to get entity").expect("entity not found");
        assert!(retrieved.has_label("Person"));
        assert_eq!(
            retrieved.get_property("index"),
            Some(&manifoldb::Value::Int(entity.id.as_u64() as i64))
        );
    }
}

#[test]
fn test_put_entities_batch_empty() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Empty batch should succeed
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_entities_batch(&[]).expect("failed to batch insert empty slice");
    tx.commit().expect("failed to commit");
}

#[test]
fn test_put_entities_batch_replaces_existing() {
    use manifoldb::Entity;

    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    let id = EntityId::new(1);

    // Create initial entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = Entity::new(id).with_property("version", 1i64);
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Batch update with new version
    let mut tx = manager.begin_write().expect("failed to begin write");
    let updated_entity = Entity::new(id).with_property("version", 2i64);
    tx.put_entities_batch(&[updated_entity]).expect("failed to batch insert");
    tx.commit().expect("failed to commit");

    // Verify update
    let tx = manager.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(id).expect("failed to get entity").expect("entity not found");
    assert_eq!(entity.get_property("version"), Some(&manifoldb::Value::Int(2)));
}

#[test]
fn test_put_edges_batch() {
    use manifoldb::{Edge, EdgeId, Entity};

    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities first
    let entity_count = 10;
    let entities: Vec<Entity> = (1..=entity_count).map(|i| Entity::new(EntityId::new(i))).collect();

    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_entities_batch(&entities).expect("failed to batch insert entities");
    tx.commit().expect("failed to commit");

    // Create edges connecting entities in a chain: 1->2, 2->3, 3->4, ...
    let edges: Vec<Edge> = (1..entity_count)
        .map(|i| {
            Edge::new(EdgeId::new(i), EntityId::new(i), EntityId::new(i + 1), "NEXT")
                .with_property("order", i as i64)
        })
        .collect();

    // Batch insert edges
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_edges_batch(&edges).expect("failed to batch insert edges");
    tx.commit().expect("failed to commit");

    // Verify all edges and their indexes
    let tx = manager.begin_read().expect("failed to begin read");

    for edge in &edges {
        // Verify edge exists
        let retrieved = tx.get_edge(edge.id).expect("failed to get edge").expect("edge not found");
        assert_eq!(retrieved.source, edge.source);
        assert_eq!(retrieved.target, edge.target);
        assert_eq!(retrieved.edge_type.as_str(), "NEXT");

        // Verify outgoing index
        let outgoing = tx.get_outgoing_edges(edge.source).expect("failed to get outgoing");
        assert!(outgoing.iter().any(|e| e.id == edge.id));

        // Verify incoming index
        let incoming = tx.get_incoming_edges(edge.target).expect("failed to get incoming");
        assert!(incoming.iter().any(|e| e.id == edge.id));
    }
}

#[test]
fn test_put_edges_batch_empty() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Empty batch should succeed
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_edges_batch(&[]).expect("failed to batch insert empty edge slice");
    tx.commit().expect("failed to commit");
}

#[test]
fn test_batch_operations_atomic() {
    use manifoldb::{Edge, EdgeId, Entity};

    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities and edges in one transaction
    let entities: Vec<Entity> = (1..=5u64).map(|i| Entity::new(EntityId::new(i))).collect();

    let edges: Vec<Edge> = vec![
        Edge::new(EdgeId::new(1), EntityId::new(1), EntityId::new(2), "LINKS"),
        Edge::new(EdgeId::new(2), EntityId::new(2), EntityId::new(3), "LINKS"),
        Edge::new(EdgeId::new(3), EntityId::new(3), EntityId::new(4), "LINKS"),
    ];

    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_entities_batch(&entities).expect("failed to batch insert entities");
    tx.put_edges_batch(&edges).expect("failed to batch insert edges");

    // Rollback - nothing should be persisted
    tx.rollback().expect("failed to rollback");

    // Verify nothing was persisted
    let tx = manager.begin_read().expect("failed to begin read");
    for entity in &entities {
        assert!(tx.get_entity(entity.id).expect("failed to get entity").is_none());
    }
    for edge in &edges {
        assert!(tx.get_edge(edge.id).expect("failed to get edge").is_none());
    }
}

#[test]
fn test_batch_entities_large() {
    use manifoldb::Entity;

    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create 1000 entities
    let count = 1000u64;
    let entities: Vec<Entity> = (1..=count)
        .map(|i| {
            Entity::new(EntityId::new(i))
                .with_label("TestEntity")
                .with_property("data", format!("Entity {}", i))
        })
        .collect();

    // Batch insert
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.put_entities_batch(&entities).expect("failed to batch insert 1000 entities");
    tx.commit().expect("failed to commit");

    // Verify count
    let tx = manager.begin_read().expect("failed to begin read");
    let all_entities = tx.iter_entities(Some("TestEntity")).expect("failed to iter entities");
    assert_eq!(all_entities.len(), count as usize);
}

// ============================================================================
// Cascade Delete Tests
// ============================================================================

#[test]
fn test_has_edges_returns_false_for_isolated_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create an isolated entity with no edges
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity");
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Check has_edges
    let tx = manager.begin_read().expect("failed to begin read");
    let has_edges = tx.has_edges(entity.id).expect("failed to check has_edges");
    assert!(!has_edges);
}

#[test]
fn test_has_edges_returns_true_for_outgoing_edge() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities with an outgoing edge
    let mut tx = manager.begin_write().expect("failed to begin write");
    let source = tx.create_entity().expect("failed to create source");
    let target = tx.create_entity().expect("failed to create target");
    tx.put_entity(&source).expect("failed to put source");
    tx.put_entity(&target).expect("failed to put target");
    let edge = tx.create_edge(source.id, target.id, "LINKS").expect("failed to create edge");
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Source should have edges
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.has_edges(source.id).expect("failed to check has_edges"));
}

#[test]
fn test_has_edges_returns_true_for_incoming_edge() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entities with an edge
    let mut tx = manager.begin_write().expect("failed to begin write");
    let source = tx.create_entity().expect("failed to create source");
    let target = tx.create_entity().expect("failed to create target");
    tx.put_entity(&source).expect("failed to put source");
    tx.put_entity(&target).expect("failed to put target");
    let edge = tx.create_edge(source.id, target.id, "LINKS").expect("failed to create edge");
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Target should have edges (incoming)
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.has_edges(target.id).expect("failed to check has_edges"));
}

#[test]
fn test_delete_entity_cascade_removes_outgoing_edges() {
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

    // Cascade delete A
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(a.id).expect("failed to cascade delete");
    tx.commit().expect("failed to commit");

    // Verify result
    assert!(result.entity_deleted);
    assert_eq!(result.edges_deleted_count(), 2);
    assert!(result.edges_deleted.contains(&edge_ab.id));
    assert!(result.edges_deleted.contains(&edge_ac.id));

    // Verify entity and edges are gone
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(a.id).expect("failed to get A").is_none());
    assert!(tx.get_edge(edge_ab.id).expect("failed to get edge A->B").is_none());
    assert!(tx.get_edge(edge_ac.id).expect("failed to get edge A->C").is_none());

    // B and C should still exist
    assert!(tx.get_entity(b.id).expect("failed to get B").is_some());
    assert!(tx.get_entity(c.id).expect("failed to get C").is_some());
}

#[test]
fn test_delete_entity_cascade_removes_incoming_edges() {
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

    let edge_ac = tx.create_edge(a.id, c.id, "POINTS").expect("failed to create edge A->C");
    let edge_bc = tx.create_edge(b.id, c.id, "POINTS").expect("failed to create edge B->C");
    tx.put_edge(&edge_ac).expect("failed to put edge A->C");
    tx.put_edge(&edge_bc).expect("failed to put edge B->C");
    tx.commit().expect("failed to commit");

    // Cascade delete C
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(c.id).expect("failed to cascade delete");
    tx.commit().expect("failed to commit");

    // Verify result
    assert!(result.entity_deleted);
    assert_eq!(result.edges_deleted_count(), 2);

    // Verify entity and edges are gone
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(c.id).expect("failed to get C").is_none());
    assert!(tx.get_edge(edge_ac.id).expect("failed to get edge A->C").is_none());
    assert!(tx.get_edge(edge_bc.id).expect("failed to get edge B->C").is_none());

    // A and B should still exist with no outgoing edges
    assert!(tx.get_entity(a.id).expect("failed to get A").is_some());
    assert!(tx.get_entity(b.id).expect("failed to get B").is_some());
    assert!(tx.get_outgoing_edges(a.id).expect("failed to get outgoing from A").is_empty());
    assert!(tx.get_outgoing_edges(b.id).expect("failed to get outgoing from B").is_empty());
}

#[test]
fn test_delete_entity_cascade_handles_self_loop() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create entity with self-loop: A -> A
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    tx.put_entity(&a).expect("failed to put A");
    let self_edge = tx.create_edge(a.id, a.id, "SELF_REF").expect("failed to create self edge");
    tx.put_edge(&self_edge).expect("failed to put self edge");
    tx.commit().expect("failed to commit");

    // Cascade delete A
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(a.id).expect("failed to cascade delete");
    tx.commit().expect("failed to commit");

    // Verify result - self-loop should only be counted once
    assert!(result.entity_deleted);
    assert_eq!(result.edges_deleted_count(), 1);
    assert!(result.edges_deleted.contains(&self_edge.id));

    // Verify entity and edge are gone
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(a.id).expect("failed to get A").is_none());
    assert!(tx.get_edge(self_edge.id).expect("failed to get self edge").is_none());
}

#[test]
fn test_delete_entity_cascade_nonexistent_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Try to cascade delete non-existent entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(EntityId::new(999)).expect("failed to cascade delete");
    tx.commit().expect("failed to commit");

    // Verify result
    assert!(!result.entity_deleted);
    assert!(result.edges_deleted.is_empty());
    assert!(result.is_empty());
}

#[test]
fn test_delete_entity_cascade_isolated_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create isolated entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity");
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Cascade delete (should work like regular delete)
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(entity.id).expect("failed to cascade delete");
    tx.commit().expect("failed to commit");

    // Verify result
    assert!(result.entity_deleted);
    assert!(result.edges_deleted.is_empty());
    assert_eq!(result.edges_deleted_count(), 0);
}

// ============================================================================
// Checked Delete Tests
// ============================================================================

#[test]
fn test_delete_entity_checked_succeeds_for_isolated_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create isolated entity
    let mut tx = manager.begin_write().expect("failed to begin write");
    let entity = tx.create_entity().expect("failed to create entity");
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Checked delete should succeed
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_entity_checked(entity.id).expect("failed to checked delete");
    tx.commit().expect("failed to commit");

    assert!(deleted);

    // Verify entity is gone
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_entity(entity.id).expect("failed to get entity").is_none());
}

#[test]
fn test_delete_entity_checked_fails_with_outgoing_edges() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> B
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    let edge = tx.create_edge(a.id, b.id, "LINKS").expect("failed to create edge");
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Checked delete of A should fail
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_checked(a.id);

    assert!(matches!(result, Err(TransactionError::ReferentialIntegrity(_))));
    if let Err(TransactionError::ReferentialIntegrity(msg)) = result {
        assert!(msg.contains("has connected edges"));
    }
}

#[test]
fn test_delete_entity_checked_fails_with_incoming_edges() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> B
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    let edge = tx.create_edge(a.id, b.id, "LINKS").expect("failed to create edge");
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // Checked delete of B should fail (has incoming edge)
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_checked(b.id);

    assert!(matches!(result, Err(TransactionError::ReferentialIntegrity(_))));
}

#[test]
fn test_delete_entity_checked_succeeds_after_edge_removal() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create graph: A -> B
    let mut tx = manager.begin_write().expect("failed to begin write");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    let edge = tx.create_edge(a.id, b.id, "LINKS").expect("failed to create edge");
    tx.put_edge(&edge).expect("failed to put edge");
    tx.commit().expect("failed to commit");

    // First delete the edge
    let mut tx = manager.begin_write().expect("failed to begin write");
    tx.delete_edge(edge.id).expect("failed to delete edge");
    tx.commit().expect("failed to commit");

    // Now checked delete should succeed for both entities
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted_a = tx.delete_entity_checked(a.id).expect("failed to checked delete A");
    let deleted_b = tx.delete_entity_checked(b.id).expect("failed to checked delete B");
    tx.commit().expect("failed to commit");

    assert!(deleted_a);
    assert!(deleted_b);
}

#[test]
fn test_delete_entity_checked_nonexistent_entity() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Checked delete of non-existent entity should return false (not error)
    let mut tx = manager.begin_write().expect("failed to begin write");
    let deleted = tx.delete_entity_checked(EntityId::new(999)).expect("failed to checked delete");

    assert!(!deleted);
}

#[test]
fn test_delete_result_default() {
    let result = DeleteResult::default();
    assert!(!result.entity_deleted);
    assert!(result.edges_deleted.is_empty());
    assert!(result.is_empty());
    assert_eq!(result.edges_deleted_count(), 0);
}

#[test]
fn test_delete_entity_cascade_complex_graph() {
    let engine = create_test_engine();
    let manager = TransactionManager::new(engine);

    // Create a hub-and-spoke graph:
    //     B
    //    /|\
    //   A-H-C  (H is the hub with edges to/from all others)
    //    \|/
    //     D
    let mut tx = manager.begin_write().expect("failed to begin write");
    let h = tx.create_entity().expect("failed to create H");
    let a = tx.create_entity().expect("failed to create A");
    let b = tx.create_entity().expect("failed to create B");
    let c = tx.create_entity().expect("failed to create C");
    let d = tx.create_entity().expect("failed to create D");
    tx.put_entity(&h).expect("failed to put H");
    tx.put_entity(&a).expect("failed to put A");
    tx.put_entity(&b).expect("failed to put B");
    tx.put_entity(&c).expect("failed to put C");
    tx.put_entity(&d).expect("failed to put D");

    // Edges from hub
    let e_ha = tx.create_edge(h.id, a.id, "OUT").expect("failed to create H->A");
    let e_hb = tx.create_edge(h.id, b.id, "OUT").expect("failed to create H->B");
    tx.put_edge(&e_ha).expect("failed to put H->A");
    tx.put_edge(&e_hb).expect("failed to put H->B");

    // Edges to hub
    let e_ch = tx.create_edge(c.id, h.id, "IN").expect("failed to create C->H");
    let e_dh = tx.create_edge(d.id, h.id, "IN").expect("failed to create D->H");
    tx.put_edge(&e_ch).expect("failed to put C->H");
    tx.put_edge(&e_dh).expect("failed to put D->H");
    tx.commit().expect("failed to commit");

    // Cascade delete the hub
    let mut tx = manager.begin_write().expect("failed to begin write");
    let result = tx.delete_entity_cascade(h.id).expect("failed to cascade delete hub");
    tx.commit().expect("failed to commit");

    // Verify result - should have deleted 4 edges (2 outgoing, 2 incoming)
    assert!(result.entity_deleted);
    assert_eq!(result.edges_deleted_count(), 4);

    // Verify all edges are gone
    let tx = manager.begin_read().expect("failed to begin read");
    assert!(tx.get_edge(e_ha.id).expect("failed to get H->A").is_none());
    assert!(tx.get_edge(e_hb.id).expect("failed to get H->B").is_none());
    assert!(tx.get_edge(e_ch.id).expect("failed to get C->H").is_none());
    assert!(tx.get_edge(e_dh.id).expect("failed to get D->H").is_none());

    // Verify hub is gone but other entities remain
    assert!(tx.get_entity(h.id).expect("failed to get H").is_none());
    assert!(tx.get_entity(a.id).expect("failed to get A").is_some());
    assert!(tx.get_entity(b.id).expect("failed to get B").is_some());
    assert!(tx.get_entity(c.id).expect("failed to get C").is_some());
    assert!(tx.get_entity(d.id).expect("failed to get D").is_some());

    // Verify remaining entities have no edges
    assert!(!tx.has_edges(a.id).expect("failed to check A edges"));
    assert!(!tx.has_edges(b.id).expect("failed to check B edges"));
    assert!(!tx.has_edges(c.id).expect("failed to check C edges"));
    assert!(!tx.has_edges(d.id).expect("failed to check D edges"));
}
