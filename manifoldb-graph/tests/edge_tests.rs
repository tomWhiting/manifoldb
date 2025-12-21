//! Integration tests for EdgeStore.

use manifoldb_core::{Edge, EdgeId, EdgeType, Entity, EntityId, Value};
use manifoldb_graph::store::{EdgeStore, GraphError, IdGenerator, NodeStore};
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{StorageEngine, Transaction};

fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("Failed to create in-memory engine")
}

/// Helper to create two nodes for edge tests
fn create_two_nodes(engine: &RedbEngine, id_gen: &IdGenerator) -> (Entity, Entity) {
    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, id_gen, |id| {
        Entity::new(id).with_label("Person").with_property("name", "Alice")
    })
    .unwrap();
    let bob = NodeStore::create(&mut tx, id_gen, |id| {
        Entity::new(id).with_label("Person").with_property("name", "Bob")
    })
    .unwrap();
    tx.commit().unwrap();
    (alice, bob)
}

#[test]
fn create_and_get_edge() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS").with_property("since", "2024-01-01")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = EdgeStore::get(&tx, edge.id).unwrap();
    assert!(retrieved.is_some());
    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, edge.id);
    assert_eq!(retrieved.source, alice.id);
    assert_eq!(retrieved.target, bob.id);
    assert_eq!(retrieved.edge_type.as_str(), "FOLLOWS");
    assert_eq!(retrieved.get_property("since"), Some(&Value::String("2024-01-01".to_owned())));
}

#[test]
fn create_edge_invalid_source_fails() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Only create target node
    let mut tx = engine.begin_write().unwrap();
    let target = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let result = EdgeStore::create(&mut tx, &id_gen, EntityId::new(999), target.id, "TEST", |id| {
        Edge::new(id, EntityId::new(999), target.id, "TEST")
    });
    assert!(matches!(result, Err(GraphError::InvalidEntityReference(_))));
}

#[test]
fn create_edge_invalid_target_fails() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Only create source node
    let mut tx = engine.begin_write().unwrap();
    let source = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let result = EdgeStore::create(&mut tx, &id_gen, source.id, EntityId::new(999), "TEST", |id| {
        Edge::new(id, source.id, EntityId::new(999), "TEST")
    });
    assert!(matches!(result, Err(GraphError::InvalidEntityReference(_))));
}

#[test]
fn create_with_id() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let edge = Edge::new(EdgeId::new(42), alice.id, bob.id, "KNOWS");

    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create_with_id(&mut tx, &edge, true).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = EdgeStore::get(&tx, EdgeId::new(42)).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id.as_u64(), 42);
}

#[test]
fn create_with_id_duplicate_fails() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let edge = Edge::new(EdgeId::new(1), alice.id, bob.id, "KNOWS");

    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create_with_id(&mut tx, &edge, true).unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let result = EdgeStore::create_with_id(&mut tx, &edge, true);
    assert!(matches!(result, Err(GraphError::EdgeAlreadyExists(_))));
}

#[test]
fn get_nonexistent_returns_none() {
    let engine = create_test_engine();

    let tx = engine.begin_read().unwrap();
    let result = EdgeStore::get(&tx, EdgeId::new(999)).unwrap();
    assert!(result.is_none());
}

#[test]
fn get_or_error_nonexistent_returns_error() {
    let engine = create_test_engine();

    let tx = engine.begin_read().unwrap();
    let result = EdgeStore::get_or_error(&tx, EdgeId::new(999));
    assert!(matches!(result, Err(GraphError::EdgeNotFound(_))));
}

#[test]
fn exists_check() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(EdgeStore::exists(&tx, edge.id).unwrap());
    assert!(!EdgeStore::exists(&tx, EdgeId::new(999)).unwrap());
}

#[test]
fn update_edge() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS").with_property("weight", 1.0f64)
    })
    .unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let mut updated = edge.clone();
    updated.set_property("weight", 2.0f64);
    EdgeStore::update(&mut tx, &updated).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let retrieved = EdgeStore::get(&tx, edge.id).unwrap().unwrap();
    assert_eq!(retrieved.get_property("weight"), Some(&Value::Float(2.0)));
}

#[test]
fn update_nonexistent_fails() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let edge = Edge::new(EdgeId::new(999), alice.id, bob.id, "KNOWS");

    let mut tx = engine.begin_write().unwrap();
    let result = EdgeStore::update(&mut tx, &edge);
    assert!(matches!(result, Err(GraphError::EdgeNotFound(_))));
}

#[test]
fn delete_edge() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    assert!(EdgeStore::delete(&mut tx, edge.id).unwrap());
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(!EdgeStore::exists(&tx, edge.id).unwrap());
}

#[test]
fn delete_nonexistent_returns_false() {
    let engine = create_test_engine();

    let mut tx = engine.begin_write().unwrap();
    assert!(!EdgeStore::delete(&mut tx, EdgeId::new(999)).unwrap());
}

#[test]
fn get_outgoing_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create nodes: Alice, Bob, Carol
    let mut tx = engine.begin_write().unwrap();
    let alice =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_property("name", "Alice"))
            .unwrap();
    let bob =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_property("name", "Bob"))
            .unwrap();
    let carol =
        NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id).with_property("name", "Carol"))
            .unwrap();

    // Alice -> Bob (FOLLOWS)
    // Alice -> Carol (FOLLOWS)
    // Bob -> Carol (FOLLOWS)
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, carol.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, carol.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, bob.id, carol.id, "FOLLOWS", |id| {
        Edge::new(id, bob.id, carol.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Alice has 2 outgoing edges
    let alice_outgoing = EdgeStore::get_outgoing(&tx, alice.id).unwrap();
    assert_eq!(alice_outgoing.len(), 2);

    // Bob has 1 outgoing edge
    let bob_outgoing = EdgeStore::get_outgoing(&tx, bob.id).unwrap();
    assert_eq!(bob_outgoing.len(), 1);

    // Carol has 0 outgoing edges
    let carol_outgoing = EdgeStore::get_outgoing(&tx, carol.id).unwrap();
    assert_eq!(carol_outgoing.len(), 0);
}

#[test]
fn get_incoming_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let carol = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Alice -> Carol, Bob -> Carol
    EdgeStore::create(&mut tx, &id_gen, alice.id, carol.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, carol.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, bob.id, carol.id, "FOLLOWS", |id| {
        Edge::new(id, bob.id, carol.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Carol has 2 incoming edges
    let carol_incoming = EdgeStore::get_incoming(&tx, carol.id).unwrap();
    assert_eq!(carol_incoming.len(), 2);

    // Alice has 0 incoming edges
    let alice_incoming = EdgeStore::get_incoming(&tx, alice.id).unwrap();
    assert_eq!(alice_incoming.len(), 0);
}

#[test]
fn get_outgoing_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "KNOWS", |id| {
        Edge::new(id, alice.id, bob.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let follows =
        EdgeStore::get_outgoing_by_type(&tx, alice.id, &EdgeType::new("FOLLOWS")).unwrap();
    assert_eq!(follows.len(), 2);

    let knows = EdgeStore::get_outgoing_by_type(&tx, alice.id, &EdgeType::new("KNOWS")).unwrap();
    assert_eq!(knows.len(), 1);

    let none = EdgeStore::get_outgoing_by_type(&tx, alice.id, &EdgeType::new("LIKES")).unwrap();
    assert!(none.is_empty());
}

#[test]
fn get_incoming_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "KNOWS", |id| {
        Edge::new(id, alice.id, bob.id, "KNOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let follows = EdgeStore::get_incoming_by_type(&tx, bob.id, &EdgeType::new("FOLLOWS")).unwrap();
    assert_eq!(follows.len(), 1);

    let knows = EdgeStore::get_incoming_by_type(&tx, bob.id, &EdgeType::new("KNOWS")).unwrap();
    assert_eq!(knows.len(), 1);
}

#[test]
fn find_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge1 = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    let _edge2 = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "KNOWS", |id| {
        Edge::new(id, alice.id, bob.id, "KNOWS")
    })
    .unwrap();
    let edge3 = EdgeStore::create(&mut tx, &id_gen, bob.id, alice.id, "FOLLOWS", |id| {
        Edge::new(id, bob.id, alice.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let follows = EdgeStore::find_by_type(&tx, &EdgeType::new("FOLLOWS")).unwrap();
    assert_eq!(follows.len(), 2);
    assert!(follows.contains(&edge1.id));
    assert!(follows.contains(&edge3.id));

    let knows = EdgeStore::find_by_type(&tx, &EdgeType::new("KNOWS")).unwrap();
    assert_eq!(knows.len(), 1);
}

#[test]
fn delete_edges_for_entity() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let carol = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Alice -> Bob, Carol -> Alice
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, carol.id, alice.id, "FOLLOWS", |id| {
        Edge::new(id, carol.id, alice.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let mut tx = engine.begin_write().unwrap();
    let deleted = EdgeStore::delete_edges_for_entity(&mut tx, alice.id).unwrap();
    assert_eq!(deleted, 2); // One outgoing, one incoming
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(EdgeStore::count(&tx).unwrap(), 0);
}

#[test]
fn count_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let tx = engine.begin_read().unwrap();
    assert_eq!(EdgeStore::count(&tx).unwrap(), 0);
    drop(tx);

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..5 {
        EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
            Edge::new(id, alice.id, bob.id, "FOLLOWS")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(EdgeStore::count(&tx).unwrap(), 5);
}

#[test]
fn for_each_iteration() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..10 {
        EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
            Edge::new(id, alice.id, bob.id, "FOLLOWS")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    EdgeStore::for_each(&tx, |_edge| {
        count += 1;
        true
    })
    .unwrap();
    assert_eq!(count, 10);
}

#[test]
fn all_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    for _ in 0..5 {
        EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
            Edge::new(id, alice.id, bob.id, "FOLLOWS")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let all = EdgeStore::all(&tx).unwrap();
    assert_eq!(all.len(), 5);
}

#[test]
fn max_id_returns_highest() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    // Empty store returns None
    let tx = engine.begin_read().unwrap();
    assert!(EdgeStore::max_id(&tx).unwrap().is_none());
    drop(tx);

    // Create edges with specific IDs
    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create_with_id(&mut tx, &Edge::new(EdgeId::new(10), alice.id, bob.id, "A"), false)
        .unwrap();
    EdgeStore::create_with_id(&mut tx, &Edge::new(EdgeId::new(5), alice.id, bob.id, "B"), false)
        .unwrap();
    EdgeStore::create_with_id(&mut tx, &Edge::new(EdgeId::new(100), alice.id, bob.id, "C"), false)
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let max = EdgeStore::max_id(&tx).unwrap();
    assert_eq!(max, Some(EdgeId::new(100)));
}

#[test]
fn indexes_cleaned_up_on_delete() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    // Verify indexes before delete
    let tx = engine.begin_read().unwrap();
    assert_eq!(EdgeStore::get_outgoing(&tx, alice.id).unwrap().len(), 1);
    assert_eq!(EdgeStore::get_incoming(&tx, bob.id).unwrap().len(), 1);
    assert_eq!(EdgeStore::find_by_type(&tx, &EdgeType::new("FOLLOWS")).unwrap().len(), 1);
    drop(tx);

    // Delete edge
    let mut tx = engine.begin_write().unwrap();
    EdgeStore::delete(&mut tx, edge.id).unwrap();
    tx.commit().unwrap();

    // Verify indexes are cleaned up
    let tx = engine.begin_read().unwrap();
    assert!(EdgeStore::get_outgoing(&tx, alice.id).unwrap().is_empty());
    assert!(EdgeStore::get_incoming(&tx, bob.id).unwrap().is_empty());
    assert!(EdgeStore::find_by_type(&tx, &EdgeType::new("FOLLOWS")).unwrap().is_empty());
}

#[test]
fn self_referential_edge() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, node.id, node.id, "SELF", |id| {
        Edge::new(id, node.id, node.id, "SELF")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Should appear in both outgoing and incoming
    let outgoing = EdgeStore::get_outgoing(&tx, node.id).unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].id, edge.id);

    let incoming = EdgeStore::get_incoming(&tx, node.id).unwrap();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0].id, edge.id);
}

#[test]
fn multiple_edges_same_nodes() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();
    let (alice, bob) = create_two_nodes(&engine, &id_gen);

    let mut tx = engine.begin_write().unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "KNOWS", |id| {
        Edge::new(id, alice.id, bob.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "LIKES", |id| {
        Edge::new(id, alice.id, bob.id, "LIKES")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let outgoing = EdgeStore::get_outgoing(&tx, alice.id).unwrap();
    assert_eq!(outgoing.len(), 3);
}
