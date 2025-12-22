//! CRUD operation integration tests.
//!
//! Tests basic Create, Read, Update, Delete operations at various scales.

use manifoldb::{Database, Value};

// ============================================================================
// Small Scale Tests (100 entities)
// ============================================================================

#[test]
fn test_crud_100_entities() {
    let db = Database::in_memory().expect("failed to create db");
    let count = 100;

    // Create
    let mut entity_ids = Vec::with_capacity(count);
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..count {
            let entity = tx
                .create_entity()
                .expect("failed to create entity")
                .with_label("TestEntity")
                .with_property("index", i as i64)
                .with_property("name", format!("entity_{i}"));
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed to put entity");
        }
        tx.commit().expect("failed to commit");
    }

    // Read all
    {
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in entity_ids.iter().enumerate() {
            let entity = tx.get_entity(*id).expect("failed to get").expect("entity not found");
            assert!(entity.has_label("TestEntity"));
            assert_eq!(entity.get_property("index"), Some(&Value::Int(i as i64)));
        }
    }

    // Update every other entity
    {
        let mut tx = db.begin().expect("failed to begin");
        for (i, id) in entity_ids.iter().enumerate() {
            if i % 2 == 0 {
                let mut entity = tx.get_entity(*id).expect("failed to get").expect("not found");
                entity.set_property("updated", true);
                tx.put_entity(&entity).expect("failed to put");
            }
        }
        tx.commit().expect("failed to commit");
    }

    // Verify updates
    {
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in entity_ids.iter().enumerate() {
            let entity = tx.get_entity(*id).expect("failed to get").expect("not found");
            if i % 2 == 0 {
                assert_eq!(entity.get_property("updated"), Some(&Value::Bool(true)));
            } else {
                assert!(entity.get_property("updated").is_none());
            }
        }
    }

    // Delete half
    {
        let mut tx = db.begin().expect("failed to begin");
        for id in entity_ids.iter().take(count / 2) {
            assert!(tx.delete_entity(*id).expect("failed to delete"));
        }
        tx.commit().expect("failed to commit");
    }

    // Verify deletions
    {
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in entity_ids.iter().enumerate() {
            let result = tx.get_entity(*id).expect("failed to get");
            if i < count / 2 {
                assert!(result.is_none(), "entity {} should be deleted", i);
            } else {
                assert!(result.is_some(), "entity {} should exist", i);
            }
        }
    }
}

// ============================================================================
// Medium Scale Tests (10,000 entities)
// ============================================================================

#[test]
fn test_crud_10k_entities() {
    let db = Database::in_memory().expect("failed to create db");
    let count = 10_000;

    // Create in batches of 1000
    let batch_size = 1000;
    let mut entity_ids = Vec::with_capacity(count);

    for batch in 0..(count / batch_size) {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..batch_size {
            let global_idx = batch * batch_size + i;
            let entity = tx
                .create_entity()
                .expect("failed to create entity")
                .with_label("Batch")
                .with_property("batch", batch as i64)
                .with_property("idx", i as i64)
                .with_property("global", global_idx as i64);
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed to put entity");
        }
        tx.commit().expect("failed to commit");
    }

    assert_eq!(entity_ids.len(), count);

    // Spot check reads
    {
        let tx = db.begin_read().expect("failed to begin read");

        // Check first, middle, and last
        let check_indices = [0, count / 2, count - 1];
        for &idx in &check_indices {
            let entity =
                tx.get_entity(entity_ids[idx]).expect("failed to get").expect("entity not found");
            assert!(entity.has_label("Batch"));
            assert_eq!(entity.get_property("global"), Some(&Value::Int(idx as i64)));
        }
    }

    // Update 10% of entities
    let update_count = count / 10;
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..update_count {
            let mut entity =
                tx.get_entity(entity_ids[i]).expect("failed to get").expect("not found");
            entity.set_property("updated", true);
            entity.set_property("update_order", i as i64);
            tx.put_entity(&entity).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Delete 5% of entities
    let delete_count = count / 20;
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..delete_count {
            assert!(tx.delete_entity(entity_ids[i]).expect("failed to delete"));
        }
        tx.commit().expect("failed to commit");
    }

    // Verify final state
    {
        let tx = db.begin_read().expect("failed to begin read");

        // Deleted entities should be gone
        for i in 0..delete_count {
            let result = tx.get_entity(entity_ids[i]).expect("failed to get");
            assert!(result.is_none(), "entity {} should be deleted", i);
        }

        // Updated (but not deleted) entities should have updates
        for i in delete_count..update_count {
            let entity = tx.get_entity(entity_ids[i]).expect("failed to get").expect("not found");
            assert_eq!(entity.get_property("updated"), Some(&Value::Bool(true)));
        }

        // Remaining entities should be unchanged
        let entity =
            tx.get_entity(entity_ids[count - 1]).expect("failed to get").expect("not found");
        assert!(entity.get_property("updated").is_none());
    }
}

// ============================================================================
// Edge CRUD Tests
// ============================================================================

#[test]
fn test_edge_crud_100_edges() {
    let db = Database::in_memory().expect("failed to create db");
    let node_count = 50;
    let edge_count = 100;

    // Create nodes
    let mut node_ids = Vec::with_capacity(node_count);
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..node_count {
            let entity =
                tx.create_entity().expect("failed to create").with_property("index", i as i64);
            node_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Create edges (random-ish pattern)
    let mut edge_ids = Vec::with_capacity(edge_count);
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..edge_count {
            let source_idx = i % node_count;
            let target_idx = (i * 7 + 3) % node_count;
            let edge = tx
                .create_edge(node_ids[source_idx], node_ids[target_idx], "CONNECTS")
                .expect("failed to create edge")
                .with_property("weight", (i as f64) / 10.0);
            edge_ids.push(edge.id);
            tx.put_edge(&edge).expect("failed to put edge");
        }
        tx.commit().expect("failed to commit");
    }

    // Read all edges
    {
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in edge_ids.iter().enumerate() {
            let edge = tx.get_edge(*id).expect("failed to get").expect("edge not found");
            assert_eq!(edge.edge_type.as_str(), "CONNECTS");

            // Verify weight is approximately correct
            if let Some(Value::Float(w)) = edge.get_property("weight") {
                let expected = (i as f64) / 10.0;
                assert!((w - expected).abs() < 0.001, "weight mismatch for edge {i}");
            } else {
                panic!("missing weight for edge {i}");
            }
        }
    }

    // Update edge types
    {
        let mut tx = db.begin().expect("failed to begin");
        for id in edge_ids.iter().take(edge_count / 2) {
            let mut edge = tx.get_edge(*id).expect("failed to get").expect("not found");
            edge.set_property("modified", true);
            tx.put_edge(&edge).expect("failed to put");
        }
        tx.commit().expect("failed to commit");
    }

    // Delete some edges
    {
        let mut tx = db.begin().expect("failed to begin");
        for id in edge_ids.iter().take(10) {
            assert!(tx.delete_edge(*id).expect("failed to delete"));
        }
        tx.commit().expect("failed to commit");
    }

    // Verify final state
    {
        let tx = db.begin_read().expect("failed to begin read");

        // First 10 should be deleted
        for id in edge_ids.iter().take(10) {
            assert!(tx.get_edge(*id).expect("failed to get").is_none());
        }

        // Next 40 should be modified
        for id in edge_ids.iter().skip(10).take(40) {
            let edge = tx.get_edge(*id).expect("failed to get").expect("not found");
            assert_eq!(edge.get_property("modified"), Some(&Value::Bool(true)));
        }

        // Rest should be unmodified
        for id in edge_ids.iter().skip(50) {
            let edge = tx.get_edge(*id).expect("failed to get").expect("not found");
            assert!(edge.get_property("modified").is_none());
        }
    }
}

// ============================================================================
// Mixed Entity/Edge CRUD
// ============================================================================

#[test]
fn test_mixed_entity_edge_operations() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a small graph and perform interleaved operations
    let mut tx = db.begin().expect("failed to begin");

    // Create nodes
    let alice =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Alice");
    let bob = tx.create_entity().expect("failed").with_label("Person").with_property("name", "Bob");
    let charlie =
        tx.create_entity().expect("failed").with_label("Person").with_property("name", "Charlie");

    tx.put_entity(&alice).expect("failed");
    tx.put_entity(&bob).expect("failed");
    tx.put_entity(&charlie).expect("failed");

    // Create edges
    let e1 = tx.create_edge(alice.id, bob.id, "KNOWS").expect("failed");
    let e2 = tx.create_edge(bob.id, charlie.id, "KNOWS").expect("failed");
    let e3 = tx.create_edge(alice.id, charlie.id, "KNOWS").expect("failed");

    tx.put_edge(&e1).expect("failed");
    tx.put_edge(&e2).expect("failed");
    tx.put_edge(&e3).expect("failed");

    tx.commit().expect("failed to commit");

    // Read and verify graph structure
    let tx = db.begin_read().expect("failed to begin read");

    // Alice should have 2 outgoing edges
    let alice_out = tx.get_outgoing_edges(alice.id).expect("failed");
    assert_eq!(alice_out.len(), 2);

    // Charlie should have 2 incoming edges
    let charlie_in = tx.get_incoming_edges(charlie.id).expect("failed");
    assert_eq!(charlie_in.len(), 2);

    // Bob should have 1 outgoing and 1 incoming
    let bob_out = tx.get_outgoing_edges(bob.id).expect("failed");
    let bob_in = tx.get_incoming_edges(bob.id).expect("failed");
    assert_eq!(bob_out.len(), 1);
    assert_eq!(bob_in.len(), 1);
}

// ============================================================================
// Property Type Tests
// ============================================================================

#[test]
fn test_various_property_types() {
    let db = Database::in_memory().expect("failed to create db");

    let mut tx = db.begin().expect("failed to begin");
    let entity = tx
        .create_entity()
        .expect("failed")
        .with_property("int_val", 42i64)
        .with_property("float_val", 3.14159f64)
        .with_property("string_val", "hello world")
        .with_property("bool_val", true)
        .with_property("vector_val", vec![1.0f32, 2.0, 3.0, 4.0]);

    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed");
    tx.commit().expect("failed to commit");

    // Verify all types
    let tx = db.begin_read().expect("failed to begin read");
    let entity = tx.get_entity(entity_id).expect("failed").expect("not found");

    assert_eq!(entity.get_property("int_val"), Some(&Value::Int(42)));
    assert_eq!(entity.get_property("bool_val"), Some(&Value::Bool(true)));
    assert_eq!(entity.get_property("string_val"), Some(&Value::String("hello world".to_string())));

    if let Some(Value::Float(f)) = entity.get_property("float_val") {
        assert!((f - 3.14159).abs() < 0.00001);
    } else {
        panic!("float property missing or wrong type");
    }

    if let Some(Value::Vector(v)) = entity.get_property("vector_val") {
        assert_eq!(v.len(), 4);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
        assert!((v[3] - 4.0).abs() < f32::EPSILON);
    } else {
        panic!("vector property missing or wrong type");
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

#[test]
fn test_batch_insert_performance() {
    let db = Database::in_memory().expect("failed to create db");

    // Insert 1000 entities in a single transaction
    let count = 1000;
    let mut ids = Vec::with_capacity(count);

    let start = std::time::Instant::now();
    {
        let mut tx = db.begin().expect("failed to begin");
        for i in 0..count {
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("Bulk")
                .with_property("idx", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed to commit");
    }
    let elapsed = start.elapsed();

    // Sanity check - 1000 inserts should complete in reasonable time
    assert!(elapsed.as_secs() < 10, "batch insert took too long: {elapsed:?}");

    // Verify all entities exist
    let tx = db.begin_read().expect("failed");
    for (i, id) in ids.iter().enumerate() {
        let entity = tx.get_entity(*id).expect("failed").expect("not found");
        assert_eq!(entity.get_property("idx"), Some(&Value::Int(i as i64)));
    }
}
