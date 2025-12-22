//! Property-based tests for ManifoldDB invariants.
//!
//! These tests verify that certain properties always hold regardless
//! of the input data or operation sequence.

use proptest::prelude::*;
use std::collections::HashSet;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Entity Invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Entity IDs are unique across transactions
    #[test]
    fn prop_entity_ids_unique(count in 1usize..100) {
        let db = Database::in_memory().expect("failed to create db");
        let mut ids = HashSet::new();

        for _ in 0..count {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed");
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");

            prop_assert!(ids.insert(entity.id), "ID should be unique");
        }
    }

    /// Entity IDs are unique within a single transaction
    #[test]
    fn prop_entity_ids_unique_in_transaction(count in 1usize..100) {
        let db = Database::in_memory().expect("failed to create db");
        let mut ids = HashSet::new();

        let mut tx = db.begin().expect("failed");
        for _ in 0..count {
            let entity = tx.create_entity().expect("failed");
            tx.put_entity(&entity).expect("failed");
            prop_assert!(ids.insert(entity.id), "ID should be unique");
        }
        tx.commit().expect("failed");
    }

    /// Created entities can always be read back
    #[test]
    fn prop_entity_readable_after_create(
        label in "[A-Z][a-z]{0,10}",
        value in any::<i64>()
    ) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity()
                .expect("failed")
                .with_label(label.as_str())
                .with_property("value", value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed");

        prop_assert!(entity.is_some(), "entity should exist");
        let entity = entity.unwrap();
        prop_assert!(entity.has_label(label.as_str()));
        prop_assert_eq!(entity.get_property("value"), Some(&Value::Int(value)));
    }

    /// Deleted entities cannot be read
    #[test]
    fn prop_entity_not_readable_after_delete(_x in any::<u8>()) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed");
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        {
            let mut tx = db.begin().expect("failed");
            tx.delete_entity(entity_id).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed");
        prop_assert!(entity.is_none(), "deleted entity should not exist");
    }
}

// ============================================================================
// Edge Invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Edge IDs are unique
    #[test]
    fn prop_edge_ids_unique(count in 1usize..50) {
        let db = Database::in_memory().expect("failed to create db");

        // Create source and target entities
        let (src_id, dst_id): (EntityId, EntityId);
        {
            let mut tx = db.begin().expect("failed");
            let src = tx.create_entity().expect("failed");
            let dst = tx.create_entity().expect("failed");
            src_id = src.id;
            dst_id = dst.id;
            tx.put_entity(&src).expect("failed");
            tx.put_entity(&dst).expect("failed");
            tx.commit().expect("failed");
        }

        let mut edge_ids = HashSet::new();
        for _ in 0..count {
            let mut tx = db.begin().expect("failed");
            let edge = tx.create_edge(src_id, dst_id, "TEST").expect("failed");
            tx.put_edge(&edge).expect("failed");
            tx.commit().expect("failed");

            prop_assert!(edge_ids.insert(edge.id), "edge ID should be unique");
        }
    }

    /// Outgoing edges contain the edge we created
    #[test]
    fn prop_outgoing_edges_contain_created(edge_type in "[A-Z_]{1,10}") {
        let db = Database::in_memory().expect("failed to create db");

        let (src_id, dst_id, edge_id): (EntityId, EntityId, manifoldb::EdgeId);
        {
            let mut tx = db.begin().expect("failed");
            let src = tx.create_entity().expect("failed");
            let dst = tx.create_entity().expect("failed");
            src_id = src.id;
            dst_id = dst.id;
            tx.put_entity(&src).expect("failed");
            tx.put_entity(&dst).expect("failed");

            let edge = tx.create_edge(src_id, dst_id, edge_type.as_str()).expect("failed");
            edge_id = edge.id;
            tx.put_edge(&edge).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let edges = tx.get_outgoing_edges(src_id).expect("failed");

        let found = edges.iter().any(|e| e.id == edge_id);
        prop_assert!(found, "should find created edge in outgoing edges");
    }

    /// Incoming edges contain the edge we created
    #[test]
    fn prop_incoming_edges_contain_created(edge_type in "[A-Z_]{1,10}") {
        let db = Database::in_memory().expect("failed to create db");

        let (src_id, dst_id, edge_id): (EntityId, EntityId, manifoldb::EdgeId);
        {
            let mut tx = db.begin().expect("failed");
            let src = tx.create_entity().expect("failed");
            let dst = tx.create_entity().expect("failed");
            src_id = src.id;
            dst_id = dst.id;
            tx.put_entity(&src).expect("failed");
            tx.put_entity(&dst).expect("failed");

            let edge = tx.create_edge(src_id, dst_id, edge_type.as_str()).expect("failed");
            edge_id = edge.id;
            tx.put_edge(&edge).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let edges = tx.get_incoming_edges(dst_id).expect("failed");

        let found = edges.iter().any(|e| e.id == edge_id);
        prop_assert!(found, "should find created edge in incoming edges");
    }

    /// Edge source and target are correct
    #[test]
    fn prop_edge_endpoints_correct(edge_type in "[A-Z_]{1,10}") {
        let db = Database::in_memory().expect("failed to create db");

        let (src_id, dst_id): (EntityId, EntityId);
        {
            let mut tx = db.begin().expect("failed");
            let src = tx.create_entity().expect("failed");
            let dst = tx.create_entity().expect("failed");
            src_id = src.id;
            dst_id = dst.id;
            tx.put_entity(&src).expect("failed");
            tx.put_entity(&dst).expect("failed");

            let edge = tx.create_edge(src_id, dst_id, edge_type.as_str()).expect("failed");
            tx.put_edge(&edge).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let edges = tx.get_outgoing_edges(src_id).expect("failed");

        prop_assert!(!edges.is_empty());
        for edge in &edges {
            prop_assert_eq!(edge.source, src_id);
            prop_assert_eq!(edge.target, dst_id);
            prop_assert_eq!(edge.edge_type.as_str(), edge_type.as_str());
        }
    }
}

// ============================================================================
// Transaction Invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Rolled back transactions don't affect database state
    #[test]
    fn prop_rollback_no_effect(value in any::<i64>()) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("value", value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.rollback().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed");
        prop_assert!(entity.is_none(), "rolled back entity should not exist");
    }

    /// Transaction IDs are unique
    #[test]
    fn prop_transaction_ids_unique(count in 1usize..50) {
        let db = Database::in_memory().expect("failed to create db");
        let mut tx_ids = HashSet::new();

        for _ in 0..count {
            let tx = db.begin_read().expect("failed");
            let id = tx.id();
            tx.rollback().expect("failed");

            prop_assert!(tx_ids.insert(id), "transaction ID should be unique");
        }
    }

    /// Read transactions see consistent snapshot
    #[test]
    fn prop_read_snapshot_consistent(initial_value in any::<i64>(), new_value in any::<i64>()) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("value", initial_value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        // Start read transaction
        let read_tx = db.begin_read().expect("failed");

        // Read initial value
        let entity = read_tx.get_entity(entity_id).expect("failed").expect("should exist");
        let before = entity.get_property("value").cloned();

        // Update in separate transaction
        {
            let mut tx = db.begin().expect("failed");
            let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
            entity.set_property("value", new_value);
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        // Read again in original transaction
        let entity = read_tx.get_entity(entity_id).expect("failed").expect("should exist");
        let after = entity.get_property("value").cloned();

        // Should be same (snapshot isolation)
        prop_assert_eq!(before.clone(), after, "snapshot should be consistent");
        prop_assert_eq!(before, Some(Value::Int(initial_value)));
    }
}

// ============================================================================
// Property Type Invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// String properties roundtrip correctly
    #[test]
    fn prop_string_roundtrip(value in ".*") {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("data", value.clone());
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        prop_assert_eq!(entity.get_property("data"), Some(&Value::String(value)));
    }

    /// Integer properties roundtrip correctly
    #[test]
    fn prop_int_roundtrip(value in any::<i64>()) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("data", value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        prop_assert_eq!(entity.get_property("data"), Some(&Value::Int(value)));
    }

    /// Float properties roundtrip correctly (for finite values)
    #[test]
    fn prop_float_roundtrip(value in any::<f64>().prop_filter("finite", |f| f.is_finite())) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("data", value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

        if let Some(Value::Float(retrieved)) = entity.get_property("data") {
            prop_assert!((retrieved - value).abs() < f64::EPSILON);
        } else {
            prop_assert!(false, "expected float property");
        }
    }

    /// Boolean properties roundtrip correctly
    #[test]
    fn prop_bool_roundtrip(value in any::<bool>()) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("data", value);
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");
        prop_assert_eq!(entity.get_property("data"), Some(&Value::Bool(value)));
    }

    /// Vector properties roundtrip correctly
    #[test]
    fn prop_vector_roundtrip(
        value in proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            0..100
        )
    ) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let entity = tx.create_entity().expect("failed").with_property("data", value.clone());
            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

        if let Some(Value::Vector(retrieved)) = entity.get_property("data") {
            prop_assert_eq!(retrieved.len(), value.len());
            for (a, b) in retrieved.iter().zip(value.iter()) {
                prop_assert!((a - b).abs() < f32::EPSILON);
            }
        } else {
            prop_assert!(false, "expected vector property");
        }
    }
}

// ============================================================================
// Label Invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Labels are preserved after entity creation
    #[test]
    fn prop_labels_preserved(labels in proptest::collection::vec("[A-Z][a-z]{0,10}", 0..5)) {
        let db = Database::in_memory().expect("failed to create db");

        let entity_id: EntityId;
        {
            let mut tx = db.begin().expect("failed");
            let mut entity = tx.create_entity().expect("failed");

            for label in &labels {
                entity = entity.with_label(label.as_str());
            }

            entity_id = entity.id;
            tx.put_entity(&entity).expect("failed");
            tx.commit().expect("failed");
        }

        let tx = db.begin_read().expect("failed");
        let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

        for label in &labels {
            prop_assert!(entity.has_label(label.as_str()), "should have label {}", label);
        }
    }
}
