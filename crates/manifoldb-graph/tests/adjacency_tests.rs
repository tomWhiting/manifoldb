//! Integration tests for AdjacencyIndex.
//!
//! These tests verify bidirectional traversal, filtered traversal by edge type,
//! and index consistency after mutations.

use manifoldb_core::{Edge, EdgeType, Entity};
use manifoldb_graph::index::AdjacencyIndex;
use manifoldb_graph::store::{EdgeStore, IdGenerator, NodeStore};
use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{StorageEngine, Transaction};

fn create_test_engine() -> RedbEngine {
    RedbEngine::in_memory().expect("Failed to create in-memory engine")
}

// ============================================================================
// Basic traversal tests
// ============================================================================

#[test]
fn get_outgoing_edge_ids_empty() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let outgoing = AdjacencyIndex::get_outgoing_edge_ids(&tx, node.id).unwrap();
    assert!(outgoing.is_empty());
}

#[test]
fn get_incoming_edge_ids_empty() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let node = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let incoming = AdjacencyIndex::get_incoming_edge_ids(&tx, node.id).unwrap();
    assert!(incoming.is_empty());
}

#[test]
fn get_outgoing_edge_ids_single() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let outgoing = AdjacencyIndex::get_outgoing_edge_ids(&tx, alice.id).unwrap();
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0], edge.id);
}

#[test]
fn get_incoming_edge_ids_single() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let incoming = AdjacencyIndex::get_incoming_edge_ids(&tx, bob.id).unwrap();
    assert_eq!(incoming.len(), 1);
    assert_eq!(incoming[0], edge.id);
}

// ============================================================================
// Bidirectional traversal tests
// ============================================================================

#[test]
fn bidirectional_traversal() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create a simple graph: A -> B -> C
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    let edge_ab = EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "EDGE", |id| {
        Edge::new(id, a.id, b.id, "EDGE")
    })
    .unwrap();
    let edge_bc = EdgeStore::create(&mut tx, &id_gen, b.id, c.id, "EDGE", |id| {
        Edge::new(id, b.id, c.id, "EDGE")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // A has 1 outgoing, 0 incoming
    assert_eq!(AdjacencyIndex::get_outgoing_edge_ids(&tx, a.id).unwrap().len(), 1);
    assert_eq!(AdjacencyIndex::get_incoming_edge_ids(&tx, a.id).unwrap().len(), 0);

    // B has 1 outgoing, 1 incoming
    let b_outgoing = AdjacencyIndex::get_outgoing_edge_ids(&tx, b.id).unwrap();
    let b_incoming = AdjacencyIndex::get_incoming_edge_ids(&tx, b.id).unwrap();
    assert_eq!(b_outgoing.len(), 1);
    assert_eq!(b_incoming.len(), 1);
    assert_eq!(b_outgoing[0], edge_bc.id);
    assert_eq!(b_incoming[0], edge_ab.id);

    // C has 0 outgoing, 1 incoming
    assert_eq!(AdjacencyIndex::get_outgoing_edge_ids(&tx, c.id).unwrap().len(), 0);
    assert_eq!(AdjacencyIndex::get_incoming_edge_ids(&tx, c.id).unwrap().len(), 1);
}

#[test]
fn multiple_outgoing_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create: A -> B, A -> C, A -> D
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let d = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "KNOWS", |id| {
        Edge::new(id, a.id, b.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "KNOWS", |id| {
        Edge::new(id, a.id, c.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, d.id, "KNOWS", |id| {
        Edge::new(id, a.id, d.id, "KNOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let outgoing = AdjacencyIndex::get_outgoing_edge_ids(&tx, a.id).unwrap();
    assert_eq!(outgoing.len(), 3);
}

#[test]
fn multiple_incoming_edges() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create: B -> A, C -> A, D -> A
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let d = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "KNOWS", |id| {
        Edge::new(id, b.id, a.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "KNOWS", |id| {
        Edge::new(id, c.id, a.id, "KNOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, d.id, a.id, "KNOWS", |id| {
        Edge::new(id, d.id, a.id, "KNOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let incoming = AdjacencyIndex::get_incoming_edge_ids(&tx, a.id).unwrap();
    assert_eq!(incoming.len(), 3);
}

// ============================================================================
// Filtered traversal by edge type
// ============================================================================

#[test]
fn get_outgoing_by_type_filters_correctly() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let carol = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Alice FOLLOWS Bob, Alice BLOCKS Carol
    let follows_edge = EdgeStore::create(&mut tx, &id_gen, alice.id, bob.id, "FOLLOWS", |id| {
        Edge::new(id, alice.id, bob.id, "FOLLOWS")
    })
    .unwrap();
    let _blocks_edge = EdgeStore::create(&mut tx, &id_gen, alice.id, carol.id, "BLOCKS", |id| {
        Edge::new(id, alice.id, carol.id, "BLOCKS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    // Get only FOLLOWS edges
    let follows_type = EdgeType::new("FOLLOWS");
    let follows = AdjacencyIndex::get_outgoing_by_type(&tx, alice.id, &follows_type).unwrap();
    assert_eq!(follows.len(), 1);
    assert_eq!(follows[0], follows_edge.id);

    // Get only BLOCKS edges
    let blocks_type = EdgeType::new("BLOCKS");
    let blocks = AdjacencyIndex::get_outgoing_by_type(&tx, alice.id, &blocks_type).unwrap();
    assert_eq!(blocks.len(), 1);

    // Get non-existent type
    let likes_type = EdgeType::new("LIKES");
    let likes = AdjacencyIndex::get_outgoing_by_type(&tx, alice.id, &likes_type).unwrap();
    assert!(likes.is_empty());
}

#[test]
fn get_incoming_by_type_filters_correctly() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let alice = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let bob = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let carol = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Bob FOLLOWS Alice, Carol BLOCKS Alice
    let follows_edge = EdgeStore::create(&mut tx, &id_gen, bob.id, alice.id, "FOLLOWS", |id| {
        Edge::new(id, bob.id, alice.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, carol.id, alice.id, "BLOCKS", |id| {
        Edge::new(id, carol.id, alice.id, "BLOCKS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();

    let follows_type = EdgeType::new("FOLLOWS");
    let follows = AdjacencyIndex::get_incoming_by_type(&tx, alice.id, &follows_type).unwrap();
    assert_eq!(follows.len(), 1);
    assert_eq!(follows[0], follows_edge.id);
}

// ============================================================================
// Counting operations
// ============================================================================

#[test]
fn count_outgoing() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E1", |id| Edge::new(id, a.id, b.id, "E1"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "E2", |id| Edge::new(id, a.id, c.id, "E2"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, a.id).unwrap(), 2);
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, b.id).unwrap(), 0);
}

#[test]
fn count_incoming() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "E1", |id| Edge::new(id, b.id, a.id, "E1"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "E2", |id| Edge::new(id, c.id, a.id, "E2"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_incoming(&tx, a.id).unwrap(), 2);
    assert_eq!(AdjacencyIndex::count_incoming(&tx, b.id).unwrap(), 0);
}

#[test]
fn count_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, b.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, c.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "BLOCKS", |id| {
        Edge::new(id, a.id, b.id, "BLOCKS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let follows = EdgeType::new("FOLLOWS");
    let blocks = EdgeType::new("BLOCKS");

    assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, a.id, &follows).unwrap(), 2);
    assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, a.id, &blocks).unwrap(), 1);
}

// ============================================================================
// Degree operations
// ============================================================================

#[test]
fn out_degree_alias() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::out_degree(&tx, a.id).unwrap(), 1);
}

#[test]
fn in_degree_alias() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::in_degree(&tx, b.id).unwrap(), 1);
}

#[test]
fn total_degree() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create: A <-> B (bidirectional)
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "E", |id| Edge::new(id, b.id, a.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    // A has 1 outgoing + 1 incoming = 2
    assert_eq!(AdjacencyIndex::degree(&tx, a.id).unwrap(), 2);
    assert_eq!(AdjacencyIndex::degree(&tx, b.id).unwrap(), 2);
}

// ============================================================================
// Has edge existence checks
// ============================================================================

#[test]
fn has_outgoing_true() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(AdjacencyIndex::has_outgoing(&tx, a.id).unwrap());
    assert!(!AdjacencyIndex::has_outgoing(&tx, b.id).unwrap());
}

#[test]
fn has_incoming_true() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert!(!AdjacencyIndex::has_incoming(&tx, a.id).unwrap());
    assert!(AdjacencyIndex::has_incoming(&tx, b.id).unwrap());
}

#[test]
fn has_outgoing_of_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, b.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let follows = EdgeType::new("FOLLOWS");
    let blocks = EdgeType::new("BLOCKS");

    assert!(AdjacencyIndex::has_outgoing_of_type(&tx, a.id, &follows).unwrap());
    assert!(!AdjacencyIndex::has_outgoing_of_type(&tx, a.id, &blocks).unwrap());
}

#[test]
fn has_incoming_of_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, b.id, "FOLLOWS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let follows = EdgeType::new("FOLLOWS");
    let blocks = EdgeType::new("BLOCKS");

    assert!(AdjacencyIndex::has_incoming_of_type(&tx, b.id, &follows).unwrap());
    assert!(!AdjacencyIndex::has_incoming_of_type(&tx, b.id, &blocks).unwrap());
}

// ============================================================================
// Iteration (for_each) tests
// ============================================================================

#[test]
fn for_each_outgoing() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "E", |id| Edge::new(id, a.id, c.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    AdjacencyIndex::for_each_outgoing(&tx, a.id, |_edge_id| {
        count += 1;
        Ok(true) // Continue
    })
    .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn for_each_outgoing_early_stop() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "E", |id| Edge::new(id, a.id, c.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    AdjacencyIndex::for_each_outgoing(&tx, a.id, |_edge_id| {
        count += 1;
        Ok(false) // Stop after first
    })
    .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn for_each_incoming() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "E", |id| Edge::new(id, b.id, a.id, "E"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "E", |id| Edge::new(id, c.id, a.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let mut count = 0;
    AdjacencyIndex::for_each_incoming(&tx, a.id, |_edge_id| {
        count += 1;
        Ok(true)
    })
    .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn for_each_outgoing_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "FOLLOWS", |id| {
        Edge::new(id, a.id, b.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, a.id, c.id, "BLOCKS", |id| {
        Edge::new(id, a.id, c.id, "BLOCKS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let follows = EdgeType::new("FOLLOWS");
    let mut count = 0;
    AdjacencyIndex::for_each_outgoing_by_type(&tx, a.id, &follows, |_edge_id| {
        count += 1;
        Ok(true)
    })
    .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn for_each_incoming_by_type() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "FOLLOWS", |id| {
        Edge::new(id, b.id, a.id, "FOLLOWS")
    })
    .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "BLOCKS", |id| {
        Edge::new(id, c.id, a.id, "BLOCKS")
    })
    .unwrap();
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    let blocks = EdgeType::new("BLOCKS");
    let mut count = 0;
    AdjacencyIndex::for_each_incoming_by_type(&tx, a.id, &blocks, |_edge_id| {
        count += 1;
        Ok(true)
    })
    .unwrap();
    assert_eq!(count, 1);
}

// ============================================================================
// Index consistency after mutations
// ============================================================================

#[test]
fn index_consistent_after_edge_delete() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let edge =
        EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "E", |id| Edge::new(id, a.id, b.id, "E"))
            .unwrap();
    tx.commit().unwrap();

    // Verify edge exists in indexes
    {
        let tx = engine.begin_read().unwrap();
        assert_eq!(AdjacencyIndex::count_outgoing(&tx, a.id).unwrap(), 1);
        assert_eq!(AdjacencyIndex::count_incoming(&tx, b.id).unwrap(), 1);
    }

    // Delete edge
    let mut tx = engine.begin_write().unwrap();
    EdgeStore::delete(&mut tx, edge.id).unwrap();
    tx.commit().unwrap();

    // Verify indexes are cleaned up
    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, a.id).unwrap(), 0);
    assert_eq!(AdjacencyIndex::count_incoming(&tx, b.id).unwrap(), 0);
    assert!(AdjacencyIndex::get_outgoing_edge_ids(&tx, a.id).unwrap().is_empty());
    assert!(AdjacencyIndex::get_incoming_edge_ids(&tx, b.id).unwrap().is_empty());
}

#[test]
fn index_consistent_after_delete_edges_for_entity() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create: B -> A <- C (A has 2 incoming edges)
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let c = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    EdgeStore::create(&mut tx, &id_gen, b.id, a.id, "E", |id| Edge::new(id, b.id, a.id, "E"))
        .unwrap();
    EdgeStore::create(&mut tx, &id_gen, c.id, a.id, "E", |id| Edge::new(id, c.id, a.id, "E"))
        .unwrap();
    tx.commit().unwrap();

    // Delete all edges for A
    let mut tx = engine.begin_write().unwrap();
    let deleted = EdgeStore::delete_edges_for_entity(&mut tx, a.id).unwrap();
    assert_eq!(deleted, 2);
    tx.commit().unwrap();

    // Verify all indexes cleaned up
    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_incoming(&tx, a.id).unwrap(), 0);
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, b.id).unwrap(), 0);
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, c.id).unwrap(), 0);
}

#[test]
fn index_consistent_after_edge_update() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    // Create A -> B with type "OLD"
    let mut tx = engine.begin_write().unwrap();
    let a = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let b = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
    let edge = EdgeStore::create(&mut tx, &id_gen, a.id, b.id, "OLD", |id| {
        Edge::new(id, a.id, b.id, "OLD")
    })
    .unwrap();
    tx.commit().unwrap();

    // Verify initial state
    {
        let tx = engine.begin_read().unwrap();
        let old_type = EdgeType::new("OLD");
        assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, a.id, &old_type).unwrap(), 1);
    }

    // Update edge type to "NEW"
    let mut tx = engine.begin_write().unwrap();
    let mut updated_edge = edge.clone();
    updated_edge.edge_type = EdgeType::new("NEW");
    EdgeStore::update(&mut tx, &updated_edge).unwrap();
    tx.commit().unwrap();

    // Verify indexes updated
    let tx = engine.begin_read().unwrap();
    let old_type = EdgeType::new("OLD");
    let new_type = EdgeType::new("NEW");
    assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, a.id, &old_type).unwrap(), 0);
    assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, a.id, &new_type).unwrap(), 1);
}

// ============================================================================
// Large graph tests for performance characteristics
// ============================================================================

#[test]
fn large_fan_out() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let hub = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Create 100 outgoing edges from hub
    for _ in 0..100 {
        let target = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        EdgeStore::create(&mut tx, &id_gen, hub.id, target.id, "CONNECTED", |id| {
            Edge::new(id, hub.id, target.id, "CONNECTED")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, hub.id).unwrap(), 100);
    assert_eq!(AdjacencyIndex::get_outgoing_edge_ids(&tx, hub.id).unwrap().len(), 100);
}

#[test]
fn large_fan_in() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let target = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Create 100 incoming edges to target
    for _ in 0..100 {
        let source = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
        EdgeStore::create(&mut tx, &id_gen, source.id, target.id, "CONNECTED", |id| {
            Edge::new(id, source.id, target.id, "CONNECTED")
        })
        .unwrap();
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_incoming(&tx, target.id).unwrap(), 100);
    assert_eq!(AdjacencyIndex::get_incoming_edge_ids(&tx, target.id).unwrap().len(), 100);
}

#[test]
fn mixed_edge_types_large() {
    let engine = create_test_engine();
    let id_gen = IdGenerator::new();

    let mut tx = engine.begin_write().unwrap();
    let hub = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();

    // Create edges with different types
    let types = ["FOLLOWS", "BLOCKS", "LIKES", "COMMENTS", "SHARES"];
    for edge_type in &types {
        for _ in 0..10 {
            let target = NodeStore::create(&mut tx, &id_gen, |id| Entity::new(id)).unwrap();
            EdgeStore::create(&mut tx, &id_gen, hub.id, target.id, *edge_type, |id| {
                Edge::new(id, hub.id, target.id, *edge_type)
            })
            .unwrap();
        }
    }
    tx.commit().unwrap();

    let tx = engine.begin_read().unwrap();
    assert_eq!(AdjacencyIndex::count_outgoing(&tx, hub.id).unwrap(), 50);

    for edge_type in &types {
        let et = EdgeType::new(*edge_type);
        assert_eq!(AdjacencyIndex::count_outgoing_by_type(&tx, hub.id, &et).unwrap(), 10);
    }
}
