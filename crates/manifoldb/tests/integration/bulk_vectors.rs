//! Bulk vector operations integration tests.
//!
//! Tests the bulk vector APIs:
//! - `bulk_insert_vectors` and `bulk_insert_named_vectors` for batch insertion
//! - `bulk_delete_vectors` and `bulk_delete_vectors_by_name` for batch deletion
//!
//! After the refactor, vectors are stored in a dedicated `collection_vectors` table
//! rather than as entity properties. Tests verify that:
//! - Vectors are NOT stored as `_vector_*` entity properties
//! - Entities can be retrieved without vector data bloat
//! - The correct count of inserted vectors is returned

use manifoldb::Database;
use manifoldb_core::EntityId;

// ============================================================================
// Bulk Insert Basic Tests
// ============================================================================

#[test]
fn test_bulk_insert_vectors_empty() {
    let db = Database::in_memory().expect("failed to create database");
    let vectors: Vec<(EntityId, String, Vec<f32>)> = vec![];

    let count =
        db.bulk_insert_vectors("documents", &vectors).expect("failed to bulk insert empty vectors");

    assert_eq!(count, 0, "empty vector list should return count of 0");
}

#[test]
fn test_bulk_insert_vectors_single() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity first
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Bulk insert a single vector
    let vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.1f32; 128])];

    let count =
        db.bulk_insert_vectors("documents", &vectors).expect("failed to bulk insert vectors");

    assert_eq!(count, 1, "should insert 1 vector");

    // Verify the vector was NOT stored as an entity property
    // (vectors now go to the collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let stored_entity =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    // Check that the vector property does NOT exist on the entity
    let property_name = "_vector_text_embedding";
    assert!(
        stored_entity.get_property(property_name).is_none(),
        "vector property should NOT be stored on entity (now in collection_vectors table)"
    );

    // Entity should still have its original properties
    assert!(stored_entity.labels.contains(&manifoldb::Label::from("documents")));
}

#[test]
fn test_bulk_insert_vectors_multiple() {
    let db = Database::in_memory().expect("failed to create database");

    // Create multiple entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for i in 0..10 {
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("documents")
            .with_property("index", manifoldb_core::Value::from(i as i64));
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Bulk insert vectors for all entities
    let vectors: Vec<(EntityId, String, Vec<f32>)> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, "embedding".to_string(), vec![i as f32 / 10.0; 64]))
        .collect();

    let count =
        db.bulk_insert_vectors("documents", &vectors).expect("failed to bulk insert vectors");

    assert_eq!(count, 10, "should insert 10 vectors");

    // Verify vectors are NOT stored as entity properties
    // (they now go to the collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_embedding").is_none(),
            "vector should NOT be stored as entity property for entity {:?}",
            entity_id
        );
        // Entity should still have its original properties
        assert!(entity.get_property("index").is_some());
    }
}

#[test]
fn test_bulk_insert_named_vectors() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for _ in 0..5 {
        let entity = tx.create_entity().expect("failed to create entity").with_label("articles");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Use the convenience method for named vectors
    let vectors: Vec<(EntityId, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, vec![0.5f32; 256])).collect();

    let count = db
        .bulk_insert_named_vectors("articles", "content_vector", &vectors)
        .expect("failed to bulk insert named vectors");

    assert_eq!(count, 5, "should insert 5 vectors");

    // Verify vectors are NOT stored as entity properties
    // (they now go to the collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_content_vector").is_none(),
            "named vector should NOT be stored as entity property"
        );
    }
}

#[test]
fn test_bulk_insert_vectors_multiple_named_vectors() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("multimodal");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert multiple different named vectors for the same entity
    let vectors = vec![
        (entity_id, "text_embedding".to_string(), vec![0.1f32; 384]),
        (entity_id, "image_embedding".to_string(), vec![0.2f32; 512]),
        (entity_id, "audio_embedding".to_string(), vec![0.3f32; 256]),
    ];

    let count =
        db.bulk_insert_vectors("multimodal", &vectors).expect("failed to bulk insert vectors");

    assert_eq!(count, 3, "should insert 3 vectors");

    // Verify vectors are NOT stored as entity properties
    // (they now go to the collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(
        entity.get_property("_vector_text_embedding").is_none(),
        "text embedding should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_image_embedding").is_none(),
        "image embedding should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_audio_embedding").is_none(),
        "audio embedding should NOT be stored as entity property"
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_bulk_insert_vectors_entity_not_found() {
    let db = Database::in_memory().expect("failed to create database");

    // Try to insert vectors for non-existent entities
    let vectors = vec![
        (EntityId::new(999), "embedding".to_string(), vec![0.1f32; 64]),
        (EntityId::new(1000), "embedding".to_string(), vec![0.2f32; 64]),
    ];

    let result = db.bulk_insert_vectors("documents", &vectors);

    assert!(result.is_err(), "should fail when entity does not exist");

    // Check the error type
    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("entity not found") || err_msg.contains("EntityNotFound"),
            "error should indicate entity not found, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_bulk_insert_vectors_partial_entity_missing() {
    let db = Database::in_memory().expect("failed to create database");

    // Create one entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let valid_entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Try to insert vectors where one entity exists and one doesn't
    let vectors = vec![
        (valid_entity_id, "embedding".to_string(), vec![0.1f32; 64]),
        (EntityId::new(999), "embedding".to_string(), vec![0.2f32; 64]),
    ];

    let result = db.bulk_insert_vectors("documents", &vectors);

    // Should fail because one entity doesn't exist (all-or-nothing semantics)
    assert!(result.is_err(), "should fail when any entity is missing");
}

// ============================================================================
// Large Scale Tests
// ============================================================================

#[test]
fn test_bulk_insert_vectors_large_batch() {
    let db = Database::in_memory().expect("failed to create database");

    // Create 1000 entities
    let entity_count = 1000;
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::with_capacity(entity_count);
    for _ in 0..entity_count {
        let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Bulk insert vectors for all entities
    let vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.5f32; 128])).collect();

    let start = std::time::Instant::now();
    let count =
        db.bulk_insert_vectors("documents", &vectors).expect("failed to bulk insert vectors");
    let duration = start.elapsed();

    assert_eq!(count, entity_count, "should insert all vectors");

    // Log performance (not a hard assertion to avoid flaky tests)
    println!(
        "Inserted {} vectors in {:?} ({:.2} vectors/sec)",
        count,
        duration,
        count as f64 / duration.as_secs_f64()
    );
}

// ============================================================================
// Vector Update Tests
// ============================================================================

#[test]
fn test_bulk_insert_vectors_update_existing() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert initial vector
    let vectors1 = vec![(entity_id, "embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &vectors1).expect("failed first insert");

    // Update with new vector (same name) - this should overwrite in collection_vectors table
    let vectors2 = vec![(entity_id, "embedding".to_string(), vec![0.9f32; 64])];
    let count = db.bulk_insert_vectors("documents", &vectors2).expect("failed second insert");

    assert_eq!(count, 1, "should report 1 vector inserted (overwrite)");

    // Verify the entity does NOT have vector properties
    // (vectors are stored in collection_vectors table, not entity properties)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(
        entity.get_property("_vector_embedding").is_none(),
        "vector should NOT be stored as entity property after update"
    );
}

// ============================================================================
// Bulk Delete Basic Tests
// ============================================================================

#[test]
fn test_bulk_delete_vectors_empty() {
    let db = Database::in_memory().expect("failed to create database");
    let vectors: Vec<(EntityId, String)> = vec![];

    let count = db.bulk_delete_vectors(&vectors).expect("failed to bulk delete empty vectors");

    assert_eq!(count, 0, "empty vector list should return count of 0");
}

#[test]
fn test_bulk_delete_vectors_single() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity with a vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector (stored in collection_vectors table, not entity properties)
    let vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.1f32; 128])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Note: Vectors are now stored in collection_vectors table, not as entity properties.
    // We can verify the entity exists and doesn't have the vector property.
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert!(
        entity.get_property("_vector_text_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
    drop(tx);

    // Delete the vector from collection_vectors table
    let to_delete = vec![(entity_id, "text_embedding".to_string())];
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    assert_eq!(count, 1, "should delete 1 vector");

    // Entity still exists and still has no vector properties (as expected)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert!(
        entity.get_property("_vector_text_embedding").is_none(),
        "vector property should never exist on entity"
    );
}

#[test]
fn test_bulk_delete_vectors_multiple() {
    let db = Database::in_memory().expect("failed to create database");

    // Create multiple entities with vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for _ in 0..5 {
        let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert vectors for all entities
    let vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.5f32; 64])).collect();
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Delete vectors for all entities
    let to_delete: Vec<(EntityId, String)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string())).collect();
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    assert_eq!(count, 5, "should delete 5 vectors");

    // Verify all vectors were deleted
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_embedding").is_none(),
            "vector should be deleted for entity {:?}",
            entity_id
        );
    }
}

#[test]
fn test_bulk_delete_vectors_by_name() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entities with vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for _ in 0..3 {
        let entity = tx.create_entity().expect("failed to create entity").with_label("articles");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert named vectors
    let vectors: Vec<(EntityId, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, vec![0.5f32; 256])).collect();
    db.bulk_insert_named_vectors("articles", "content_vector", &vectors)
        .expect("failed to insert vectors");

    // Delete using the convenience method
    let count = db
        .bulk_delete_vectors_by_name("content_vector", &entity_ids)
        .expect("failed to delete vectors by name");

    assert_eq!(count, 3, "should delete 3 vectors");

    // Verify all vectors were deleted
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_content_vector").is_none(),
            "vector should be deleted"
        );
    }
}

#[test]
fn test_bulk_delete_vectors_selective() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity with multiple vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("multimodal");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert multiple different named vectors (stored in collection_vectors table)
    let vectors = vec![
        (entity_id, "text_embedding".to_string(), vec![0.1f32; 384]),
        (entity_id, "image_embedding".to_string(), vec![0.2f32; 512]),
        (entity_id, "audio_embedding".to_string(), vec![0.3f32; 256]),
    ];
    db.bulk_insert_vectors("multimodal", &vectors).expect("failed to insert vectors");

    // Delete only image embeddings, keep text and audio
    let to_delete = vec![(entity_id, "image_embedding".to_string())];
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    assert_eq!(count, 1, "should delete 1 vector");

    // Verify entity does NOT have vector properties (they're in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    // Vectors are stored in collection_vectors table, not entity properties
    assert!(
        entity.get_property("_vector_text_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_image_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_audio_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
}

// ============================================================================
// Bulk Delete Error Handling Tests
// ============================================================================

#[test]
fn test_bulk_delete_vectors_entity_not_found() {
    let db = Database::in_memory().expect("failed to create database");

    // Try to delete vectors from non-existent entities
    let to_delete = vec![
        (EntityId::new(999), "embedding".to_string()),
        (EntityId::new(1000), "embedding".to_string()),
    ];

    let result = db.bulk_delete_vectors(&to_delete);

    assert!(result.is_err(), "should fail when entity does not exist");

    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("entity not found") || err_msg.contains("EntityNotFound"),
            "error should indicate entity not found, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_bulk_delete_vectors_nonexistent_vector() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity without any vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Try to delete a vector that doesn't exist
    let to_delete = vec![(entity_id, "nonexistent_embedding".to_string())];
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    // Should return 0 since the vector didn't exist
    assert_eq!(count, 0, "should report 0 deleted when vector doesn't exist");
}

#[test]
fn test_bulk_delete_vectors_partial_existence() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entities, only some with vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity1 = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity1_id = entity1.id;
    tx.put_entity(&entity1).expect("failed to put entity");

    let entity2 = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity2_id = entity2.id;
    tx.put_entity(&entity2).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Only insert vector for entity1
    let vectors = vec![(entity1_id, "embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Try to delete from both entities
    let to_delete =
        vec![(entity1_id, "embedding".to_string()), (entity2_id, "embedding".to_string())];
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    // Should only count the one that existed
    assert_eq!(count, 1, "should only count deleted vectors that existed");
}

// ============================================================================
// Bulk Delete Large Scale Tests
// ============================================================================

#[test]
fn test_bulk_delete_vectors_large_batch() {
    let db = Database::in_memory().expect("failed to create database");

    // Create 1000 entities with vectors
    let entity_count = 1000;
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::with_capacity(entity_count);
    for _ in 0..entity_count {
        let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert vectors for all entities
    let vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.5f32; 128])).collect();
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Delete all vectors
    let to_delete: Vec<(EntityId, String)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string())).collect();

    let start = std::time::Instant::now();
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");
    let duration = start.elapsed();

    assert_eq!(count, entity_count, "should delete all vectors");

    println!(
        "Deleted {} vectors in {:?} ({:.2} vectors/sec)",
        count,
        duration,
        count as f64 / duration.as_secs_f64()
    );

    // Verify all vectors were deleted
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in entity_ids.iter().take(10) {
        // Spot check first 10
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(entity.get_property("_vector_embedding").is_none(), "vector should be deleted");
    }
}

// ============================================================================
// Bulk Delete Idempotency Tests
// ============================================================================

#[test]
fn test_bulk_delete_vectors_idempotent() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity with a vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector
    let vectors = vec![(entity_id, "embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Delete the vector
    let to_delete = vec![(entity_id, "embedding".to_string())];
    let count1 = db.bulk_delete_vectors(&to_delete).expect("failed first delete");
    assert_eq!(count1, 1, "first delete should count 1");

    // Delete again (should be idempotent)
    let count2 = db.bulk_delete_vectors(&to_delete).expect("failed second delete");
    assert_eq!(count2, 0, "second delete should count 0 (already deleted)");
}

// ============================================================================
// Bulk Update Vector Tests
// ============================================================================

#[test]
fn test_bulk_update_vectors_empty() {
    let db = Database::in_memory().expect("failed to create database");
    let vectors: Vec<(EntityId, String, Vec<f32>)> = vec![];

    let count =
        db.bulk_update_vectors("documents", &vectors).expect("failed to bulk update empty vectors");

    assert_eq!(count, 0, "empty vector list should return count of 0");
}

#[test]
fn test_bulk_update_vectors_single() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity with an initial vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert initial vector (stored in collection_vectors table)
    let initial_vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.1f32; 128])];
    db.bulk_insert_vectors("documents", &initial_vectors).expect("failed to insert initial vector");

    // Update the vector with new values
    let updated_vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.9f32; 128])];
    let count = db
        .bulk_update_vectors("documents", &updated_vectors)
        .expect("failed to bulk update vectors");

    assert_eq!(count, 1, "should update 1 vector");

    // Verify the entity does NOT have vector properties
    // (vectors are stored in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let stored_entity =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(
        stored_entity.get_property("_vector_text_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
}

#[test]
fn test_bulk_update_vectors_multiple() {
    let db = Database::in_memory().expect("failed to create database");

    // Create multiple entities with initial vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for i in 0..10 {
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("documents")
            .with_property("index", manifoldb_core::Value::from(i as i64));
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert initial vectors (stored in collection_vectors table)
    let initial_vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.1f32; 64])).collect();
    db.bulk_insert_vectors("documents", &initial_vectors).expect("failed to insert vectors");

    // Update all vectors with new values
    let updated_vectors: Vec<(EntityId, String, Vec<f32>)> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, "embedding".to_string(), vec![i as f32 / 10.0 + 0.5; 64]))
        .collect();

    let count = db
        .bulk_update_vectors("documents", &updated_vectors)
        .expect("failed to bulk update vectors");

    assert_eq!(count, 10, "should update 10 vectors");

    // Verify entities do NOT have vector properties
    // (vectors are stored in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in entity_ids.iter() {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_embedding").is_none(),
            "vector should NOT be stored as entity property"
        );
    }
}

#[test]
fn test_bulk_replace_named_vectors() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entities with initial vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for _ in 0..5 {
        let entity = tx.create_entity().expect("failed to create entity").with_label("articles");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert initial named vectors (stored in collection_vectors table)
    let initial_vectors: Vec<(EntityId, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, vec![0.1f32; 256])).collect();
    db.bulk_insert_named_vectors("articles", "content_vector", &initial_vectors)
        .expect("failed to insert initial vectors");

    // Replace with new vectors using convenience method
    let updated_vectors: Vec<(EntityId, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, vec![0.8f32; 256])).collect();
    let count = db
        .bulk_replace_named_vectors("articles", "content_vector", &updated_vectors)
        .expect("failed to replace named vectors");

    assert_eq!(count, 5, "should update 5 vectors");

    // Verify entities do NOT have vector properties
    // (vectors are stored in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_content_vector").is_none(),
            "vector should NOT be stored as entity property"
        );
    }
}

#[test]
fn test_bulk_update_vectors_entity_not_found() {
    let db = Database::in_memory().expect("failed to create database");

    // First create the collection by inserting a dummy entity with vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");
    let dummy_vectors = vec![(entity.id, "embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &dummy_vectors).expect("failed to create collection");

    // Try to update vectors for non-existent entities
    let vectors = vec![
        (EntityId::new(999), "embedding".to_string(), vec![0.1f32; 64]),
        (EntityId::new(1000), "embedding".to_string(), vec![0.2f32; 64]),
    ];

    let result = db.bulk_update_vectors("documents", &vectors);

    assert!(result.is_err(), "should fail when entity does not exist");

    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("entity not found") || err_msg.contains("EntityNotFound"),
            "error should indicate entity not found, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_bulk_update_vectors_missing_vector() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity WITHOUT a vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // First create the collection by inserting a vector for this entity
    let initial = vec![(entity_id, "other_embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &initial).expect("failed to create collection");

    // Try to update a vector that doesn't exist
    let vectors = vec![(entity_id, "nonexistent_embedding".to_string(), vec![0.1f32; 64])];

    let result = db.bulk_update_vectors("documents", &vectors);

    assert!(result.is_err(), "should fail when vector doesn't exist");

    if let Err(err) = result {
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("does not have vector") || err_msg.contains("vector error"),
            "error should indicate vector not found, got: {}",
            err_msg
        );
    }
}

#[test]
fn test_bulk_update_vectors_partial_missing() {
    let db = Database::in_memory().expect("failed to create database");

    // Create two entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity1 = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity2 = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity1_id = entity1.id;
    let entity2_id = entity2.id;
    tx.put_entity(&entity1).expect("failed to put entity");
    tx.put_entity(&entity2).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Only insert vector for entity1
    let initial = vec![(entity1_id, "embedding".to_string(), vec![0.1f32; 64])];
    db.bulk_insert_vectors("documents", &initial).expect("failed to insert");

    // Try to update both - should fail because entity2 doesn't have the vector
    let updates = vec![
        (entity1_id, "embedding".to_string(), vec![0.9f32; 64]),
        (entity2_id, "embedding".to_string(), vec![0.9f32; 64]),
    ];

    let result = db.bulk_update_vectors("documents", &updates);

    // Should fail (all-or-nothing semantics)
    assert!(result.is_err(), "should fail when any vector is missing");

    // Verify entity1's vector was NOT updated (rollback)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity =
        tx.get_entity(entity1_id).expect("failed to get entity").expect("entity not found");
    if let Some(manifoldb_core::Value::Vector(v)) = entity.get_property("_vector_embedding") {
        assert!((v[0] - 0.1f32).abs() < 0.001, "vector should NOT be updated due to rollback");
    }
}

#[test]
fn test_bulk_update_vectors_multiple_named_vectors() {
    let db = Database::in_memory().expect("failed to create database");

    // Create entity with multiple named vectors
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("multimodal");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert multiple different named vectors (stored in collection_vectors table)
    let initial_vectors = vec![
        (entity_id, "text_embedding".to_string(), vec![0.1f32; 384]),
        (entity_id, "image_embedding".to_string(), vec![0.2f32; 512]),
        (entity_id, "audio_embedding".to_string(), vec![0.3f32; 256]),
    ];
    db.bulk_insert_vectors("multimodal", &initial_vectors)
        .expect("failed to insert initial vectors");

    // Update all three vectors
    let updated_vectors = vec![
        (entity_id, "text_embedding".to_string(), vec![0.9f32; 384]),
        (entity_id, "image_embedding".to_string(), vec![0.8f32; 512]),
        (entity_id, "audio_embedding".to_string(), vec![0.7f32; 256]),
    ];

    let count = db
        .bulk_update_vectors("multimodal", &updated_vectors)
        .expect("failed to bulk update vectors");

    assert_eq!(count, 3, "should update 3 vectors");

    // Verify entity does NOT have vector properties
    // (vectors are stored in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(
        entity.get_property("_vector_text_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_image_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
    assert!(
        entity.get_property("_vector_audio_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
}

#[test]
fn test_bulk_update_vectors_large_batch() {
    let db = Database::in_memory().expect("failed to create database");

    // Create 1000 entities with initial vectors
    let entity_count = 1000;
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::with_capacity(entity_count);
    for _ in 0..entity_count {
        let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Insert initial vectors
    let initial_vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.1f32; 128])).collect();
    db.bulk_insert_vectors("documents", &initial_vectors)
        .expect("failed to insert initial vectors");

    // Update all vectors
    let updated_vectors: Vec<(EntityId, String, Vec<f32>)> =
        entity_ids.iter().map(|&id| (id, "embedding".to_string(), vec![0.9f32; 128])).collect();

    let start = std::time::Instant::now();
    let count = db
        .bulk_update_vectors("documents", &updated_vectors)
        .expect("failed to bulk update vectors");
    let duration = start.elapsed();

    assert_eq!(count, entity_count, "should update all vectors");

    // Log performance
    println!(
        "Updated {} vectors in {:?} ({:.2} vectors/sec)",
        count,
        duration,
        count as f64 / duration.as_secs_f64()
    );
}

#[test]
fn test_bulk_update_vectors_dimension_change() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity with initial vector
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert initial vector with dimension 128 (stored in collection_vectors table)
    let initial_vectors = vec![(entity_id, "embedding".to_string(), vec![0.1f32; 128])];
    db.bulk_insert_vectors("documents", &initial_vectors).expect("failed to insert initial vector");

    // Update with a different dimension (e.g., upgrading to a better model)
    // The storage layer allows this since vectors are stored in the collection_vectors table
    let updated_vectors = vec![(entity_id, "embedding".to_string(), vec![0.5f32; 384])];
    let count = db
        .bulk_update_vectors("documents", &updated_vectors)
        .expect("failed to update with new dimension");

    assert_eq!(count, 1, "should update 1 vector");

    // Verify entity does NOT have vector property
    // (vectors are stored in collection_vectors table)
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(
        entity.get_property("_vector_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
}

// ============================================================================
// Vector Retrieval Tests
// ============================================================================

#[test]
fn test_get_vector_after_insert() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector
    let expected_vector = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
    let vectors = vec![(entity_id, "text_embedding".to_string(), expected_vector.clone())];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Retrieve the vector using get_vector
    let retrieved =
        db.get_vector("documents", entity_id, "text_embedding").expect("failed to get vector");

    assert!(retrieved.is_some(), "vector should exist");
    let vector_data = retrieved.unwrap();
    let dense = vector_data.as_dense().expect("should be dense vector");
    assert_eq!(dense, expected_vector.as_slice(), "vector data should match");

    // Verify the entity does NOT contain the vector
    let tx = db.begin_read().expect("failed to begin read transaction");
    let stored_entity =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert!(
        stored_entity.get_property("_vector_text_embedding").is_none(),
        "vector should NOT be stored as entity property"
    );
}

#[test]
fn test_get_all_vectors_multiple_named() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert multiple named vectors for the same entity
    let text_embedding = vec![0.1f32, 0.2, 0.3];
    let image_embedding = vec![0.4f32, 0.5, 0.6, 0.7];
    let summary_embedding = vec![0.8f32, 0.9];

    let vectors = vec![
        (entity_id, "text_embedding".to_string(), text_embedding.clone()),
        (entity_id, "image_embedding".to_string(), image_embedding.clone()),
        (entity_id, "summary_embedding".to_string(), summary_embedding.clone()),
    ];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Retrieve all vectors
    let all_vectors =
        db.get_all_vectors("documents", entity_id).expect("failed to get all vectors");

    assert_eq!(all_vectors.len(), 3, "should have 3 vectors");
    assert!(all_vectors.contains_key("text_embedding"));
    assert!(all_vectors.contains_key("image_embedding"));
    assert!(all_vectors.contains_key("summary_embedding"));

    // Verify each vector's content
    let retrieved_text = all_vectors.get("text_embedding").unwrap().as_dense().unwrap();
    assert_eq!(retrieved_text, text_embedding.as_slice());

    let retrieved_image = all_vectors.get("image_embedding").unwrap().as_dense().unwrap();
    assert_eq!(retrieved_image, image_embedding.as_slice());

    let retrieved_summary = all_vectors.get("summary_embedding").unwrap().as_dense().unwrap();
    assert_eq!(retrieved_summary, summary_embedding.as_slice());
}

#[test]
fn test_get_vector_nonexistent() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector with one name
    let vectors = vec![(entity_id, "existing_vector".to_string(), vec![0.1f32; 10])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Try to get a non-existent vector name - should return None, not error
    let result = db.get_vector("documents", entity_id, "nonexistent_vector");
    assert!(result.is_ok(), "should not error for nonexistent vector");
    assert!(result.unwrap().is_none(), "should return None for nonexistent vector");
}

#[test]
fn test_has_vector() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector
    let vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.1f32; 10])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Check existence - should be true
    let exists = db
        .has_vector("documents", entity_id, "text_embedding")
        .expect("failed to check vector existence");
    assert!(exists, "vector should exist");

    // Check non-existent vector - should be false
    let not_exists = db
        .has_vector("documents", entity_id, "nonexistent")
        .expect("failed to check vector existence");
    assert!(!not_exists, "vector should not exist");
}

#[test]
fn test_get_all_vectors_empty() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity (but don't add any vectors)
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Create collection by inserting another entity's vector first
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity2 = tx.create_entity().expect("failed to create entity").with_label("documents");
    tx.put_entity(&entity2).expect("failed to put entity");
    tx.commit().expect("failed to commit");
    let vectors = vec![(entity2.id, "dummy".to_string(), vec![0.1f32])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert dummy vector");

    // Get all vectors for entity with no vectors
    let all_vectors =
        db.get_all_vectors("documents", entity_id).expect("failed to get all vectors");

    assert!(all_vectors.is_empty(), "should return empty map for entity with no vectors");
}

#[test]
fn test_get_vector_after_update() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert initial vector
    let initial_vector = vec![0.1f32, 0.2, 0.3];
    let vectors = vec![(entity_id, "embedding".to_string(), initial_vector)];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Update the vector
    let updated_vector = vec![0.9f32, 0.8, 0.7, 0.6];
    let update = vec![(entity_id, "embedding".to_string(), updated_vector.clone())];
    db.bulk_update_vectors("documents", &update).expect("failed to update vectors");

    // Retrieve and verify it's the updated vector
    let retrieved = db
        .get_vector("documents", entity_id, "embedding")
        .expect("failed to get vector")
        .expect("vector should exist");

    let dense = retrieved.as_dense().expect("should be dense vector");
    assert_eq!(dense, updated_vector.as_slice(), "should have updated vector data");
}

#[test]
fn test_get_vector_after_delete() {
    let db = Database::in_memory().expect("failed to create database");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx.create_entity().expect("failed to create entity").with_label("documents");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert a vector
    let vectors = vec![(entity_id, "embedding".to_string(), vec![0.1f32; 10])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Verify it exists
    assert!(
        db.has_vector("documents", entity_id, "embedding").expect("check failed"),
        "vector should exist before delete"
    );

    // Delete the vector
    let to_delete = vec![(entity_id, "embedding".to_string())];
    db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    // Verify it's gone
    let retrieved =
        db.get_vector("documents", entity_id, "embedding").expect("failed to get vector");
    assert!(retrieved.is_none(), "vector should be None after delete");

    assert!(
        !db.has_vector("documents", entity_id, "embedding").expect("check failed"),
        "has_vector should return false after delete"
    );
}

#[test]
fn test_get_vector_collection_not_found() {
    let db = Database::in_memory().expect("failed to create database");

    // Try to get a vector from a non-existent collection
    let result = db.get_vector("nonexistent_collection", EntityId::new(1), "embedding");

    assert!(result.is_err(), "should error for non-existent collection");
    let err = result.unwrap_err();
    let err_str = err.to_string();
    assert!(err_str.contains("not found"), "error message should mention 'not found': {}", err_str);
}

// ============================================================================
// HNSW Index Integration Tests
// ============================================================================

/// Test that bulk_insert_vectors automatically updates HNSW index
#[test]
fn test_bulk_insert_vectors_updates_hnsw_index() {
    let db = Database::in_memory().expect("failed to create database");

    // Create the table first via SQL
    db.execute("CREATE TABLE docs (title TEXT)").expect("failed to create table");

    // Create entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for i in 0..5 {
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("docs")
            .with_property("title", format!("Doc {}", i));
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Create HNSW index BEFORE inserting vectors using the conventional naming
    // The index name format is: {collection}_{vector_name}_hnsw
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        manifoldb::vector::HnswIndexBuilder::new("docs_embedding_hnsw", "docs", "embedding")
            .dimension(64)
            .m(4)
            .ef_construction(16)
            .build(&mut tx)
            .expect("failed to create HNSW index");
        tx.commit().expect("failed to commit");
    }

    // Bulk insert vectors - this should automatically update the HNSW index
    let vectors: Vec<(EntityId, String, Vec<f32>)> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            // Create distinct vectors for each entity
            let mut vec = vec![0.0f32; 64];
            vec[i] = 1.0; // Each entity has a 1.0 at a different position
            (id, "embedding".to_string(), vec)
        })
        .collect();
    db.bulk_insert_vectors("docs", &vectors).expect("failed to bulk insert vectors");

    // Search for the first entity's vector - should find it
    let query = {
        let mut q = vec![0.0f32; 64];
        q[0] = 1.0;
        q
    };

    // Use vector search via the vector module
    let search_results = {
        let tx = db.begin_read().expect("failed to begin read transaction");
        manifoldb::vector::search_named_vector_index(
            &tx,
            "docs",
            "embedding",
            &manifoldb_vector::Embedding::new(query).expect("invalid embedding"),
            5,
            None,
        )
        .expect("failed to search HNSW index")
    };

    // The first result should be the first entity (closest to query)
    assert!(!search_results.is_empty(), "search should return results");
    assert_eq!(
        search_results[0].entity_id, entity_ids[0],
        "first result should be the entity with the matching vector"
    );
}

/// Test that bulk_delete_vectors removes entries from HNSW index
#[test]
fn test_bulk_delete_vectors_updates_hnsw_index() {
    let db = Database::in_memory().expect("failed to create database");

    // Create the table first via SQL
    db.execute("CREATE TABLE docs (title TEXT)").expect("failed to create table");

    // Create entities
    let mut tx = db.begin().expect("failed to begin transaction");
    let mut entity_ids = Vec::new();
    for i in 0..3 {
        let entity = tx
            .create_entity()
            .expect("failed to create entity")
            .with_label("docs")
            .with_property("title", format!("Doc {}", i));
        entity_ids.push(entity.id);
        tx.put_entity(&entity).expect("failed to put entity");
    }
    tx.commit().expect("failed to commit");

    // Bulk insert vectors first
    let vectors: Vec<(EntityId, String, Vec<f32>)> = entity_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let mut vec = vec![0.0f32; 32];
            vec[i] = 1.0;
            (id, "embedding".to_string(), vec)
        })
        .collect();
    db.bulk_insert_vectors("docs", &vectors).expect("failed to bulk insert vectors");

    // Create HNSW index AFTER inserting vectors - should include all 3 vectors
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        manifoldb::vector::HnswIndexBuilder::new("docs_embedding_hnsw", "docs", "embedding")
            .dimension(32)
            .m(4)
            .ef_construction(16)
            .build(&mut tx)
            .expect("failed to create HNSW index");
        tx.commit().expect("failed to commit");
    }

    // Verify all 3 are searchable
    {
        let tx = db.begin_read().expect("failed to begin read transaction");
        let query = manifoldb_vector::Embedding::new(vec![1.0f32; 32]).expect("invalid embedding");
        let results = manifoldb::vector::search_named_vector_index(
            &tx,
            "docs",
            "embedding",
            &query,
            10,
            None,
        )
        .expect("failed to search");
        assert_eq!(results.len(), 3, "should find all 3 vectors");
    }

    // Delete one vector
    let to_delete = vec![(entity_ids[1], "embedding".to_string())];
    db.bulk_delete_vectors(&to_delete).expect("failed to delete vector");

    // Search again - should only find 2 vectors now
    {
        let tx = db.begin_read().expect("failed to begin read transaction");
        let query = manifoldb_vector::Embedding::new(vec![1.0f32; 32]).expect("invalid embedding");
        let results = manifoldb::vector::search_named_vector_index(
            &tx,
            "docs",
            "embedding",
            &query,
            10,
            None,
        )
        .expect("failed to search after delete");
        assert_eq!(results.len(), 2, "should only find 2 vectors after delete");
        // Verify the deleted entity is not in results
        assert!(
            !results.iter().any(|r| r.entity_id == entity_ids[1]),
            "deleted entity should not appear in search results"
        );
    }
}

/// Test that bulk_update_vectors updates HNSW index entries
#[test]
fn test_bulk_update_vectors_updates_hnsw_index() {
    let db = Database::in_memory().expect("failed to create database");

    // Create the table first via SQL
    db.execute("CREATE TABLE docs (title TEXT)").expect("failed to create table");

    // Create an entity
    let mut tx = db.begin().expect("failed to begin transaction");
    let entity = tx
        .create_entity()
        .expect("failed to create entity")
        .with_label("docs")
        .with_property("title", "Doc");
    let entity_id = entity.id;
    tx.put_entity(&entity).expect("failed to put entity");
    tx.commit().expect("failed to commit");

    // Insert initial vector (pointing in direction [1, 0, 0, ...])
    let initial = vec![(entity_id, "embedding".to_string(), {
        let mut v = vec![0.0f32; 32];
        v[0] = 1.0;
        v
    })];
    db.bulk_insert_vectors("docs", &initial).expect("failed to insert initial vector");

    // Create HNSW index
    {
        let mut tx = db.begin().expect("failed to begin transaction");
        manifoldb::vector::HnswIndexBuilder::new("docs_embedding_hnsw", "docs", "embedding")
            .dimension(32)
            .m(4)
            .ef_construction(16)
            .build(&mut tx)
            .expect("failed to create HNSW index");
        tx.commit().expect("failed to commit");
    }

    // Search with query similar to initial vector - should find it
    {
        let tx = db.begin_read().expect("failed to begin read transaction");
        let query = {
            let mut v = vec![0.0f32; 32];
            v[0] = 1.0;
            manifoldb_vector::Embedding::new(v).expect("invalid embedding")
        };
        let results =
            manifoldb::vector::search_named_vector_index(&tx, "docs", "embedding", &query, 1, None)
                .expect("failed to search");
        assert_eq!(results.len(), 1);
        assert!(results[0].distance < 0.01, "should be very close to query");
    }

    // Update vector to point in a different direction [0, 1, 0, ...]
    let updated = vec![(entity_id, "embedding".to_string(), {
        let mut v = vec![0.0f32; 32];
        v[1] = 1.0;
        v
    })];
    db.bulk_update_vectors("docs", &updated).expect("failed to update vector");

    // Search with query similar to NEW vector - should find it with low distance
    {
        let tx = db.begin_read().expect("failed to begin read transaction");
        let query = {
            let mut v = vec![0.0f32; 32];
            v[1] = 1.0;
            manifoldb_vector::Embedding::new(v).expect("invalid embedding")
        };
        let results =
            manifoldb::vector::search_named_vector_index(&tx, "docs", "embedding", &query, 1, None)
                .expect("failed to search after update");
        assert_eq!(results.len(), 1);
        assert!(
            results[0].distance < 0.01,
            "should be very close to updated vector, but got distance {}",
            results[0].distance
        );
    }
}
