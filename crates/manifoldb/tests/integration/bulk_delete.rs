//! Bulk delete integration tests.
//!
//! Tests for bulk_delete_entities and bulk_delete_entities_checked methods.

use manifoldb::{Database, Entity, EntityId, Value};

// ============================================================================
// Basic Bulk Delete Tests
// ============================================================================

#[test]
fn test_bulk_delete_empty() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_ids: Vec<EntityId> = Vec::new();
    let deleted = db.bulk_delete_entities(&entity_ids).expect("bulk delete failed");

    assert_eq!(deleted, 0);
}

#[test]
fn test_bulk_delete_single_entity() {
    let db = Database::in_memory().expect("failed to create db");

    // Create an entity
    let entities =
        vec![Entity::new(EntityId::new(0)).with_label("Person").with_property("name", "Alice")];
    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // Verify it exists
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(ids[0]).expect("get failed").is_some());
    }

    // Delete it
    let deleted = db.bulk_delete_entities(&ids).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify it's gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(ids[0]).expect("get failed").is_none());
    }
}

#[test]
fn test_bulk_delete_multiple_entities() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 100 entities
    let entities: Vec<Entity> = (0..100)
        .map(|i| {
            Entity::new(EntityId::new(0)).with_label("Document").with_property("index", i as i64)
        })
        .collect();

    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");
    assert_eq!(ids.len(), 100);

    // Delete all of them
    let deleted = db.bulk_delete_entities(&ids).expect("bulk delete failed");
    assert_eq!(deleted, 100);

    // Verify all are gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        for id in &ids {
            assert!(tx.get_entity(*id).expect("get failed").is_none());
        }
    }
}

#[test]
fn test_bulk_delete_partial() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 100 entities
    let entities: Vec<Entity> = (0..100)
        .map(|i| Entity::new(EntityId::new(0)).with_label("Item").with_property("index", i as i64))
        .collect();

    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // Delete only the first half
    let to_delete: Vec<EntityId> = ids[0..50].to_vec();
    let deleted = db.bulk_delete_entities(&to_delete).expect("bulk delete failed");
    assert_eq!(deleted, 50);

    // Verify first half is gone, second half remains
    {
        let tx = db.begin_read().expect("failed to begin read");
        for (i, id) in ids.iter().enumerate() {
            let exists = tx.get_entity(*id).expect("get failed").is_some();
            if i < 50 {
                assert!(!exists, "entity at index {} should be deleted", i);
            } else {
                assert!(exists, "entity at index {} should exist", i);
            }
        }
    }
}

#[test]
fn test_bulk_delete_nonexistent() {
    let db = Database::in_memory().expect("failed to create db");

    // Try to delete entities that don't exist
    let ids = vec![EntityId::new(999), EntityId::new(1000), EntityId::new(1001)];
    let deleted = db.bulk_delete_entities(&ids).expect("bulk delete failed");

    // Should return 0 since none existed
    assert_eq!(deleted, 0);
}

#[test]
fn test_bulk_delete_mixed_existing_nonexisting() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 5 entities
    let entities: Vec<Entity> =
        (0..5).map(|i| Entity::new(EntityId::new(0)).with_property("index", i as i64)).collect();

    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // Mix existing IDs with nonexistent ones
    let to_delete = vec![
        ids[0],
        EntityId::new(9999), // doesn't exist
        ids[2],
        EntityId::new(9998), // doesn't exist
        ids[4],
    ];

    let deleted = db.bulk_delete_entities(&to_delete).expect("bulk delete failed");
    assert_eq!(deleted, 3); // Only 3 entities existed

    // Verify correct ones deleted
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(ids[0]).expect("get").is_none());
        assert!(tx.get_entity(ids[1]).expect("get").is_some());
        assert!(tx.get_entity(ids[2]).expect("get").is_none());
        assert!(tx.get_entity(ids[3]).expect("get").is_some());
        assert!(tx.get_entity(ids[4]).expect("get").is_none());
    }
}

// ============================================================================
// Index Cleanup Tests
// ============================================================================

#[test]
fn test_bulk_delete_cleans_property_indexes() {
    let db = Database::in_memory().expect("failed to create db");

    // This test verifies that index entries are properly cleaned up.
    // We use SQL inserts to ensure both entity and index are properly populated.
    db.execute("CREATE TABLE products (sku TEXT, name TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_products_sku ON products (sku)").expect("create index failed");

    // Insert via SQL
    db.execute("INSERT INTO products (sku, name) VALUES ('SKU-001', 'Widget')")
        .expect("insert 1 failed");
    db.execute("INSERT INTO products (sku, name) VALUES ('SKU-002', 'Gadget')")
        .expect("insert 2 failed");
    db.execute("INSERT INTO products (sku, name) VALUES ('SKU-003', 'Gizmo')")
        .expect("insert 3 failed");

    // Verify index works
    let result = db.query("SELECT * FROM products WHERE sku = 'SKU-001'").expect("query failed");
    assert_eq!(result.len(), 1);

    // Get the entity ID for the first product by reading from storage
    let entity_id = {
        let tx = db.begin_read().expect("begin read");
        // Scan all entities and find the one with sku = SKU-001
        let mut found_id = None;
        for i in 1..=10 {
            if let Some(entity) = tx.get_entity(EntityId::new(i)).expect("get entity") {
                eprintln!("Entity {}: {:?}", i, entity.properties);
                if entity.get_property("sku") == Some(&Value::String("SKU-001".to_string())) {
                    found_id = Some(entity.id);
                    eprintln!("Found SKU-001 at entity ID {}", i);
                }
            }
        }
        found_id.expect("entity with SKU-001 not found")
    };

    // Bulk delete the entity
    let deleted = db.bulk_delete_entities(&[entity_id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify entity is actually gone from storage
    {
        let tx = db.begin_read().expect("begin read");
        eprintln!("After delete, checking entity {}:", entity_id.as_u64());
        let entity_check = tx.get_entity(entity_id).expect("get entity");
        eprintln!("Result: {:?}", entity_check);
        assert!(entity_check.is_none(), "entity should be deleted");
    }

    // Verify index is updated - SKU-001 should no longer be found
    // First check total count
    let all_result = db.query("SELECT * FROM products").expect("query all failed");
    assert_eq!(all_result.len(), 2, "should have 2 products after deleting 1");

    let result = db.query("SELECT * FROM products WHERE sku = 'SKU-001'").expect("query failed");
    // Debug: print what was found
    for row in &result {
        eprintln!("Found row: {:?}", row);
    }
    assert_eq!(result.len(), 0);

    // Other SKUs should still work
    let result = db.query("SELECT * FROM products WHERE sku = 'SKU-002'").expect("query failed");
    assert_eq!(result.len(), 1);

    let result = db.query("SELECT * FROM products WHERE sku = 'SKU-003'").expect("query failed");
    assert_eq!(result.len(), 1);
}

#[test]
fn test_bulk_delete_all_cleans_index() {
    let db = Database::in_memory().expect("failed to create db");

    // Create table with index
    db.execute("CREATE TABLE items (category TEXT)").expect("create table failed");
    db.execute("CREATE INDEX idx_items_category ON items (category)").expect("create index failed");

    // Insert via SQL
    db.execute("INSERT INTO items (category) VALUES ('electronics')").expect("insert 1 failed");
    db.execute("INSERT INTO items (category) VALUES ('electronics')").expect("insert 2 failed");
    db.execute("INSERT INTO items (category) VALUES ('clothing')").expect("insert 3 failed");

    // Verify initial state
    let result =
        db.query("SELECT * FROM items WHERE category = 'electronics'").expect("query failed");
    assert_eq!(result.len(), 2);

    // Get the entity IDs by scanning storage
    let entity_ids: Vec<EntityId> = {
        let tx = db.begin_read().expect("begin read");
        let mut ids = Vec::new();
        for i in 1..=10 {
            if tx.get_entity(EntityId::new(i)).expect("get entity").is_some() {
                ids.push(EntityId::new(i));
            }
        }
        ids
    };
    assert_eq!(entity_ids.len(), 3);

    // Bulk delete all
    let deleted = db.bulk_delete_entities(&entity_ids).expect("bulk delete failed");
    assert_eq!(deleted, 3);

    // Index should be empty
    let result =
        db.query("SELECT * FROM items WHERE category = 'electronics'").expect("query failed");
    assert_eq!(result.len(), 0);

    let result = db.query("SELECT * FROM items WHERE category = 'clothing'").expect("query failed");
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Edge Cascade Tests
// ============================================================================

#[test]
fn test_bulk_delete_cascades_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities with edges
    let mut tx = db.begin().expect("failed to begin");

    let person1 = tx.create_entity().expect("failed to create").with_label("Person");
    let person2 = tx.create_entity().expect("failed to create").with_label("Person");
    let person3 = tx.create_entity().expect("failed to create").with_label("Person");

    tx.put_entity(&person1).expect("put failed");
    tx.put_entity(&person2).expect("put failed");
    tx.put_entity(&person3).expect("put failed");

    // Create edges: 1 -> 2, 2 -> 3, 1 -> 3
    let edge1 = tx.create_edge(person1.id, person2.id, "FOLLOWS").expect("create edge");
    let edge2 = tx.create_edge(person2.id, person3.id, "FOLLOWS").expect("create edge");
    let edge3 = tx.create_edge(person1.id, person3.id, "FOLLOWS").expect("create edge");

    tx.put_edge(&edge1).expect("put edge");
    tx.put_edge(&edge2).expect("put edge");
    tx.put_edge(&edge3).expect("put edge");
    tx.commit().expect("commit failed");

    // Delete person1 (has 2 outgoing edges)
    let deleted = db.bulk_delete_entities(&[person1.id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify person1 is gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(person1.id).expect("get").is_none());

        // Person2 and person3 should still exist
        assert!(tx.get_entity(person2.id).expect("get").is_some());
        assert!(tx.get_entity(person3.id).expect("get").is_some());

        // Edges from person1 should be gone
        assert!(tx.get_edge(edge1.id).expect("get").is_none());
        assert!(tx.get_edge(edge3.id).expect("get").is_none());

        // Edge from person2 -> person3 should still exist
        assert!(tx.get_edge(edge2.id).expect("get").is_some());
    }
}

#[test]
fn test_bulk_delete_cascades_incoming_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities with edges
    let mut tx = db.begin().expect("failed to begin");

    let person1 = tx.create_entity().expect("failed to create").with_label("Person");
    let person2 = tx.create_entity().expect("failed to create").with_label("Person");
    let person3 = tx.create_entity().expect("failed to create").with_label("Person");

    tx.put_entity(&person1).expect("put failed");
    tx.put_entity(&person2).expect("put failed");
    tx.put_entity(&person3).expect("put failed");

    // Create edges pointing TO person2: 1 -> 2, 3 -> 2
    let edge1 = tx.create_edge(person1.id, person2.id, "FOLLOWS").expect("create edge");
    let edge2 = tx.create_edge(person3.id, person2.id, "FOLLOWS").expect("create edge");

    tx.put_edge(&edge1).expect("put edge");
    tx.put_edge(&edge2).expect("put edge");
    tx.commit().expect("commit failed");

    // Delete person2 (has 2 incoming edges)
    let deleted = db.bulk_delete_entities(&[person2.id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify person2 is gone and incoming edges are deleted
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(person2.id).expect("get").is_none());

        // Person1 and person3 should still exist
        assert!(tx.get_entity(person1.id).expect("get").is_some());
        assert!(tx.get_entity(person3.id).expect("get").is_some());

        // Both edges should be gone
        assert!(tx.get_edge(edge1.id).expect("get").is_none());
        assert!(tx.get_edge(edge2.id).expect("get").is_none());
    }
}

#[test]
fn test_bulk_delete_self_loop_edge() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entity with self-loop edge
    let mut tx = db.begin().expect("failed to begin");

    let entity = tx.create_entity().expect("failed to create").with_label("Node");
    tx.put_entity(&entity).expect("put failed");

    // Self-loop: entity -> entity
    let edge = tx.create_edge(entity.id, entity.id, "LINKS").expect("create edge");
    tx.put_edge(&edge).expect("put edge");
    tx.commit().expect("commit failed");

    // Delete the entity
    let deleted = db.bulk_delete_entities(&[entity.id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify entity and edge are gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(entity.id).expect("get").is_none());
        assert!(tx.get_edge(edge.id).expect("get").is_none());
    }
}

// ============================================================================
// Checked Delete Tests
// ============================================================================

#[test]
fn test_bulk_delete_checked_no_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities without edges
    let entities: Vec<Entity> =
        (0..3).map(|i| Entity::new(EntityId::new(0)).with_property("index", i as i64)).collect();

    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // Checked delete should succeed
    let deleted = db.bulk_delete_entities_checked(&ids).expect("bulk delete failed");
    assert_eq!(deleted, 3);

    // Verify all deleted
    {
        let tx = db.begin_read().expect("failed to begin read");
        for id in &ids {
            assert!(tx.get_entity(*id).expect("get").is_none());
        }
    }
}

#[test]
fn test_bulk_delete_checked_with_edges_fails() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities with an edge
    let mut tx = db.begin().expect("failed to begin");

    let entity1 = tx.create_entity().expect("failed to create");
    let entity2 = tx.create_entity().expect("failed to create");

    tx.put_entity(&entity1).expect("put failed");
    tx.put_entity(&entity2).expect("put failed");

    let edge = tx.create_edge(entity1.id, entity2.id, "LINKS").expect("create edge");
    tx.put_edge(&edge).expect("put edge");
    tx.commit().expect("commit failed");

    // Checked delete should fail
    let result = db.bulk_delete_entities_checked(&[entity1.id]);
    assert!(result.is_err());

    // Entity should still exist
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(entity1.id).expect("get").is_some());
        assert!(tx.get_entity(entity2.id).expect("get").is_some());
        assert!(tx.get_edge(edge.id).expect("get").is_some());
    }
}

#[test]
fn test_bulk_delete_checked_is_atomic() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 5 entities, put an edge only on entity3
    let mut tx = db.begin().expect("failed to begin");

    let mut entities = Vec::with_capacity(5);
    for _ in 0..5 {
        let entity = tx.create_entity().expect("failed to create");
        entities.push(entity.clone());
        tx.put_entity(&entity).expect("put failed");
    }

    // Create edge: entity2 -> entity3 (so entity3 has incoming edge)
    let edge = tx.create_edge(entities[2].id, entities[3].id, "LINKS").expect("create edge");
    tx.put_edge(&edge).expect("put edge");
    tx.commit().expect("commit failed");

    // Get IDs
    let ids: Vec<EntityId> = entities.iter().map(|e| e.id).collect();

    // Attempt to delete all - should fail because entity3 has incoming edge
    let result = db.bulk_delete_entities_checked(&ids);
    assert!(result.is_err());

    // ALL entities should still exist (atomic failure)
    {
        let tx = db.begin_read().expect("failed to begin read");
        for id in &ids {
            assert!(
                tx.get_entity(*id).expect("get").is_some(),
                "entity {} should still exist after atomic failure",
                id.as_u64()
            );
        }
    }
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_bulk_delete_large_batch() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 10,000 entities
    let entities: Vec<Entity> = (0..10_000)
        .map(|i| Entity::new(EntityId::new(0)).with_label("Item").with_property("index", i as i64))
        .collect();

    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // Delete all at once
    let deleted = db.bulk_delete_entities(&ids).expect("bulk delete failed");
    assert_eq!(deleted, 10_000);

    // Spot check some entities are gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(ids[0]).expect("get").is_none());
        assert!(tx.get_entity(ids[5000]).expect("get").is_none());
        assert!(tx.get_entity(ids[9999]).expect("get").is_none());
    }
}

#[test]
fn test_bulk_delete_with_many_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a star graph: center with 100 connections
    let mut tx = db.begin().expect("failed to begin");

    let center = tx.create_entity().expect("failed").with_label("Center");
    tx.put_entity(&center).expect("put failed");

    for _ in 0..100 {
        let leaf = tx.create_entity().expect("failed").with_label("Leaf");
        tx.put_entity(&leaf).expect("put failed");

        let edge = tx.create_edge(center.id, leaf.id, "CONNECTS").expect("create edge");
        tx.put_edge(&edge).expect("put edge");
    }

    tx.commit().expect("commit failed");

    // Delete the center node - should cascade delete all 100 edges
    let deleted = db.bulk_delete_entities(&[center.id]).expect("bulk delete failed");
    assert_eq!(deleted, 1);

    // Verify center is gone
    {
        let tx = db.begin_read().expect("failed to begin read");
        assert!(tx.get_entity(center.id).expect("get").is_none());

        // All edges should be gone
        let edges = tx.get_outgoing_edges(center.id).expect("get edges");
        assert!(edges.is_empty());
    }
}

// ============================================================================
// Integration with ID sequence
// ============================================================================

#[test]
fn test_bulk_delete_preserves_id_sequence() {
    let db = Database::in_memory().expect("failed to create db");

    // Create 5 entities
    let entities: Vec<Entity> =
        (0..5).map(|_| Entity::new(EntityId::new(0)).with_label("Item")).collect();
    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");

    // IDs should be 1-5
    assert_eq!(ids[0].as_u64(), 1);
    assert_eq!(ids[4].as_u64(), 5);

    // Delete all entities
    db.bulk_delete_entities(&ids).expect("bulk delete failed");

    // Create 3 more entities
    let more_entities: Vec<Entity> =
        (0..3).map(|_| Entity::new(EntityId::new(0)).with_label("NewItem")).collect();
    let new_ids = db.bulk_insert_entities(&more_entities).expect("bulk insert failed");

    // IDs should continue from 6 (IDs are not reused)
    assert_eq!(new_ids[0].as_u64(), 6);
    assert_eq!(new_ids[2].as_u64(), 8);
}
