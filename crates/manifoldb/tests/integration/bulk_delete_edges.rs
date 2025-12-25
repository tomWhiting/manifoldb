//! Bulk delete edges integration tests.
//!
//! Tests for bulk_delete_edges method.

use manifoldb::{Database, Edge, EdgeId, Entity, EntityId};

// ============================================================================
// Basic Bulk Delete Tests
// ============================================================================

#[test]
fn test_bulk_delete_edges_empty() {
    let db = Database::in_memory().expect("failed to create db");

    let edge_ids: Vec<EdgeId> = Vec::new();
    let deleted = db.bulk_delete_edges(&edge_ids).expect("bulk delete failed");

    assert_eq!(deleted, 0);
}

#[test]
fn test_bulk_delete_single_edge() {
    let db = Database::in_memory().expect("failed to create db");

    // Create two entities and an edge
    let mut tx = db.begin().expect("failed to begin");

    let entity1 = tx.create_entity().expect("create failed").with_label("Node");
    let entity2 = tx.create_entity().expect("create failed").with_label("Node");

    tx.put_entity(&entity1).expect("put failed");
    tx.put_entity(&entity2).expect("put failed");

    let edge = tx.create_edge(entity1.id, entity2.id, "CONNECTS").expect("create edge");
    tx.put_edge(&edge).expect("put edge");
    tx.commit().expect("commit failed");

    // Verify edge exists
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_edge(edge.id).expect("get failed").is_some());
    }

    // Delete the edge
    let deleted = db.bulk_delete_edges(&[edge.id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify edge is gone but entities remain
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_edge(edge.id).expect("get failed").is_none());
        assert!(tx.get_entity(entity1.id).expect("get").is_some());
        assert!(tx.get_entity(entity2.id).expect("get").is_some());
    }
}

#[test]
fn test_bulk_delete_multiple_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a chain of entities with edges
    let mut tx = db.begin().expect("failed to begin");

    let mut entities = Vec::new();
    for _ in 0..11 {
        let entity = tx.create_entity().expect("create failed").with_label("Node");
        tx.put_entity(&entity).expect("put failed");
        entities.push(entity);
    }

    // Create 10 edges: 0->1, 1->2, ..., 9->10
    let mut edge_ids = Vec::new();
    for i in 0..10 {
        let edge = tx.create_edge(entities[i].id, entities[i + 1].id, "NEXT").expect("create edge");
        tx.put_edge(&edge).expect("put edge");
        edge_ids.push(edge.id);
    }
    tx.commit().expect("commit failed");

    // Delete all edges
    let deleted = db.bulk_delete_edges(&edge_ids).expect("bulk delete failed");
    assert_eq!(deleted, 10);

    // Verify all edges are gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        for edge_id in &edge_ids {
            assert!(tx.get_edge(*edge_id).expect("get failed").is_none());
        }
        // Entities should still exist
        for entity in &entities {
            assert!(tx.get_entity(entity.id).expect("get").is_some());
        }
    }
}

#[test]
fn test_bulk_delete_partial_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities and edges
    let mut tx = db.begin().expect("failed to begin");

    let entity1 = tx.create_entity().expect("create failed");
    let entity2 = tx.create_entity().expect("create failed");
    let entity3 = tx.create_entity().expect("create failed");

    tx.put_entity(&entity1).expect("put");
    tx.put_entity(&entity2).expect("put");
    tx.put_entity(&entity3).expect("put");

    let edge1 = tx.create_edge(entity1.id, entity2.id, "A").expect("create");
    let edge2 = tx.create_edge(entity2.id, entity3.id, "B").expect("create");
    let edge3 = tx.create_edge(entity1.id, entity3.id, "C").expect("create");

    tx.put_edge(&edge1).expect("put");
    tx.put_edge(&edge2).expect("put");
    tx.put_edge(&edge3).expect("put");
    tx.commit().expect("commit");

    // Delete only edge1 and edge3
    let deleted = db.bulk_delete_edges(&[edge1.id, edge3.id]).expect("bulk delete failed");
    assert_eq!(deleted, 2);

    // Verify edge1 and edge3 gone, edge2 remains
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_edge(edge1.id).expect("get").is_none());
        assert!(tx.get_edge(edge2.id).expect("get").is_some());
        assert!(tx.get_edge(edge3.id).expect("get").is_none());
    }
}

#[test]
fn test_bulk_delete_nonexistent_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to delete edges that don't exist
    let ids = vec![EdgeId::new(999), EdgeId::new(1000), EdgeId::new(1001)];
    let deleted = db.bulk_delete_edges(&ids).expect("bulk delete failed");

    // Should return 0 since none existed
    assert_eq!(deleted, 0);
}

#[test]
fn test_bulk_delete_mixed_existing_nonexisting_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create some edges
    let mut tx = db.begin().expect("failed to begin");

    let entity1 = tx.create_entity().expect("create");
    let entity2 = tx.create_entity().expect("create");
    let entity3 = tx.create_entity().expect("create");

    tx.put_entity(&entity1).expect("put");
    tx.put_entity(&entity2).expect("put");
    tx.put_entity(&entity3).expect("put");

    let edge1 = tx.create_edge(entity1.id, entity2.id, "A").expect("create");
    let edge2 = tx.create_edge(entity2.id, entity3.id, "B").expect("create");

    tx.put_edge(&edge1).expect("put");
    tx.put_edge(&edge2).expect("put");
    tx.commit().expect("commit");

    // Mix existing IDs with nonexistent ones
    let to_delete = vec![
        edge1.id,
        EdgeId::new(9999), // doesn't exist
        edge2.id,
        EdgeId::new(9998), // doesn't exist
    ];

    let deleted = db.bulk_delete_edges(&to_delete).expect("bulk delete failed");
    assert_eq!(deleted, 2); // Only 2 edges existed

    // Verify both edges are deleted
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_edge(edge1.id).expect("get").is_none());
        assert!(tx.get_edge(edge2.id).expect("get").is_none());
    }
}

// ============================================================================
// Index Cleanup Tests
// ============================================================================

#[test]
fn test_bulk_delete_edges_cleans_outgoing_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities and edges
    let mut tx = db.begin().expect("failed to begin");

    let source = tx.create_entity().expect("create").with_label("Source");
    let target1 = tx.create_entity().expect("create").with_label("Target");
    let target2 = tx.create_entity().expect("create").with_label("Target");

    tx.put_entity(&source).expect("put");
    tx.put_entity(&target1).expect("put");
    tx.put_entity(&target2).expect("put");

    let edge1 = tx.create_edge(source.id, target1.id, "LINKS").expect("create");
    let edge2 = tx.create_edge(source.id, target2.id, "LINKS").expect("create");

    tx.put_edge(&edge1).expect("put");
    tx.put_edge(&edge2).expect("put");
    tx.commit().expect("commit");

    // Verify source has 2 outgoing edges
    {
        let tx = db.begin_read().expect("begin");
        let outgoing = tx.get_outgoing_edges(source.id).expect("get outgoing");
        assert_eq!(outgoing.len(), 2);
    }

    // Delete one edge
    let deleted = db.bulk_delete_edges(&[edge1.id]).expect("bulk delete");
    assert_eq!(deleted, 1);

    // Verify only 1 outgoing edge remains
    {
        let tx = db.begin_read().expect("begin");
        let outgoing = tx.get_outgoing_edges(source.id).expect("get outgoing");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].id, edge2.id);
    }
}

#[test]
fn test_bulk_delete_edges_cleans_incoming_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities and edges
    let mut tx = db.begin().expect("failed to begin");

    let source1 = tx.create_entity().expect("create");
    let source2 = tx.create_entity().expect("create");
    let target = tx.create_entity().expect("create");

    tx.put_entity(&source1).expect("put");
    tx.put_entity(&source2).expect("put");
    tx.put_entity(&target).expect("put");

    let edge1 = tx.create_edge(source1.id, target.id, "POINTS").expect("create");
    let edge2 = tx.create_edge(source2.id, target.id, "POINTS").expect("create");

    tx.put_edge(&edge1).expect("put");
    tx.put_edge(&edge2).expect("put");
    tx.commit().expect("commit");

    // Verify target has 2 incoming edges
    {
        let tx = db.begin_read().expect("begin");
        let incoming = tx.get_incoming_edges(target.id).expect("get incoming");
        assert_eq!(incoming.len(), 2);
    }

    // Delete one edge
    let deleted = db.bulk_delete_edges(&[edge1.id]).expect("bulk delete");
    assert_eq!(deleted, 1);

    // Verify only 1 incoming edge remains
    {
        let tx = db.begin_read().expect("begin");
        let incoming = tx.get_incoming_edges(target.id).expect("get incoming");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].id, edge2.id);
    }
}

#[test]
fn test_bulk_delete_edges_cleans_type_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities and edges with different types
    let mut tx = db.begin().expect("failed to begin");

    let entity1 = tx.create_entity().expect("create");
    let entity2 = tx.create_entity().expect("create");
    let entity3 = tx.create_entity().expect("create");

    tx.put_entity(&entity1).expect("put");
    tx.put_entity(&entity2).expect("put");
    tx.put_entity(&entity3).expect("put");

    let edge_follows = tx.create_edge(entity1.id, entity2.id, "FOLLOWS").expect("create");
    let edge_likes = tx.create_edge(entity1.id, entity3.id, "LIKES").expect("create");

    tx.put_edge(&edge_follows).expect("put");
    tx.put_edge(&edge_likes).expect("put");
    tx.commit().expect("commit");

    // Delete the FOLLOWS edge
    let deleted = db.bulk_delete_edges(&[edge_follows.id]).expect("bulk delete");
    assert_eq!(deleted, 1);

    // LIKES edge should still work
    {
        let tx = db.begin_read().expect("begin");
        let likes = tx.get_edge(edge_likes.id).expect("get");
        assert!(likes.is_some());
        assert_eq!(likes.unwrap().edge_type.as_str(), "LIKES");
    }
}

// ============================================================================
// Self-Loop Edge Tests
// ============================================================================

#[test]
fn test_bulk_delete_self_loop_edge() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entity with self-loop
    let mut tx = db.begin().expect("failed to begin");

    let entity = tx.create_entity().expect("create").with_label("Node");
    tx.put_entity(&entity).expect("put");

    let self_loop = tx.create_edge(entity.id, entity.id, "SELF").expect("create");
    tx.put_edge(&self_loop).expect("put");
    tx.commit().expect("commit");

    // Verify self-loop exists
    {
        let tx = db.begin_read().expect("begin");
        let outgoing = tx.get_outgoing_edges(entity.id).expect("get");
        let incoming = tx.get_incoming_edges(entity.id).expect("get");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(incoming.len(), 1);
        assert_eq!(outgoing[0].id, self_loop.id);
        assert_eq!(incoming[0].id, self_loop.id);
    }

    // Delete the self-loop
    let deleted = db.bulk_delete_edges(&[self_loop.id]).expect("bulk delete");
    assert_eq!(deleted, 1);

    // Verify self-loop is gone
    {
        let tx = db.begin_read().expect("begin");
        assert!(tx.get_edge(self_loop.id).expect("get").is_none());
        let outgoing = tx.get_outgoing_edges(entity.id).expect("get");
        let incoming = tx.get_incoming_edges(entity.id).expect("get");
        assert!(outgoing.is_empty());
        assert!(incoming.is_empty());
    }
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_bulk_delete_large_batch_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    let entities: Vec<Entity> =
        (0..101).map(|_| Entity::new(EntityId::new(0)).with_label("Node")).collect();
    let entity_ids = db.bulk_insert_entities(&entities).expect("insert entities");

    // Create 100 edges
    let edges: Vec<Edge> = (0..100)
        .map(|i| Edge::new(EdgeId::new(0), entity_ids[i], entity_ids[i + 1], "NEXT"))
        .collect();
    let edge_ids = db.bulk_insert_edges(&edges).expect("insert edges");
    assert_eq!(edge_ids.len(), 100);

    // Delete all edges
    let deleted = db.bulk_delete_edges(&edge_ids).expect("bulk delete");
    assert_eq!(deleted, 100);

    // Verify all edges are gone
    {
        let tx = db.begin_read().expect("begin");
        for edge_id in &edge_ids {
            assert!(tx.get_edge(*edge_id).expect("get").is_none());
        }
    }
}

#[test]
fn test_bulk_delete_star_graph_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create center and leaf entities
    let mut entities: Vec<Entity> = vec![Entity::new(EntityId::new(0)).with_label("Center")];
    for _ in 0..50 {
        entities.push(Entity::new(EntityId::new(0)).with_label("Leaf"));
    }
    let entity_ids = db.bulk_insert_entities(&entities).expect("insert entities");
    let center_id = entity_ids[0];

    // Create 50 edges from center to each leaf
    let edges: Vec<Edge> =
        (1..51).map(|i| Edge::new(EdgeId::new(0), center_id, entity_ids[i], "RADIATES")).collect();
    let edge_ids = db.bulk_insert_edges(&edges).expect("insert edges");

    // Verify center has 50 outgoing edges
    {
        let tx = db.begin_read().expect("begin");
        let outgoing = tx.get_outgoing_edges(center_id).expect("get");
        assert_eq!(outgoing.len(), 50);
    }

    // Delete all edges
    let deleted = db.bulk_delete_edges(&edge_ids).expect("bulk delete");
    assert_eq!(deleted, 50);

    // Verify center has no outgoing edges
    {
        let tx = db.begin_read().expect("begin");
        let outgoing = tx.get_outgoing_edges(center_id).expect("get");
        assert!(outgoing.is_empty());
    }
}

// ============================================================================
// ID Sequence Tests
// ============================================================================

#[test]
fn test_bulk_delete_edges_preserves_id_sequence() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    let entities: Vec<Entity> =
        (0..3).map(|_| Entity::new(EntityId::new(0)).with_label("Node")).collect();
    let entity_ids = db.bulk_insert_entities(&entities).expect("insert");

    // Create 3 edges
    let edges: Vec<Edge> = vec![
        Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "A"),
        Edge::new(EdgeId::new(0), entity_ids[1], entity_ids[2], "B"),
        Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[2], "C"),
    ];
    let edge_ids = db.bulk_insert_edges(&edges).expect("insert edges");

    // Edge IDs should be 1, 2, 3
    assert_eq!(edge_ids[0].as_u64(), 1);
    assert_eq!(edge_ids[2].as_u64(), 3);

    // Delete all edges
    db.bulk_delete_edges(&edge_ids).expect("bulk delete");

    // Create 2 more edges
    let new_edges: Vec<Edge> = vec![
        Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "D"),
        Edge::new(EdgeId::new(0), entity_ids[1], entity_ids[2], "E"),
    ];
    let new_edge_ids = db.bulk_insert_edges(&new_edges).expect("insert new edges");

    // IDs should continue from 4 (IDs are not reused)
    assert_eq!(new_edge_ids[0].as_u64(), 4);
    assert_eq!(new_edge_ids[1].as_u64(), 5);
}

// ============================================================================
// Transaction Atomicity Tests
// ============================================================================

#[test]
fn test_bulk_delete_edges_is_atomic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities and edges
    let entities: Vec<Entity> = (0..3).map(|_| Entity::new(EntityId::new(0))).collect();
    let entity_ids = db.bulk_insert_entities(&entities).expect("insert");

    let edges: Vec<Edge> = vec![
        Edge::new(EdgeId::new(0), entity_ids[0], entity_ids[1], "A"),
        Edge::new(EdgeId::new(0), entity_ids[1], entity_ids[2], "B"),
    ];
    let edge_ids = db.bulk_insert_edges(&edges).expect("insert edges");

    // Delete both edges in one transaction
    let deleted = db.bulk_delete_edges(&edge_ids).expect("bulk delete");
    assert_eq!(deleted, 2);

    // Both should be gone
    {
        let tx = db.begin_read().expect("begin");
        assert!(tx.get_edge(edge_ids[0]).expect("get").is_none());
        assert!(tx.get_edge(edge_ids[1]).expect("get").is_none());
    }
}
