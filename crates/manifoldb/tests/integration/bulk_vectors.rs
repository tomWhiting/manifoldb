//! Bulk vector operations integration tests.
//!
//! Tests the bulk vector APIs:
//! - `bulk_insert_vectors` and `bulk_insert_named_vectors` for batch insertion
//! - `bulk_delete_vectors` and `bulk_delete_vectors_by_name` for batch deletion

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

    // Verify the vector was stored as a property
    let tx = db.begin_read().expect("failed to begin read transaction");
    let stored_entity =
        tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    // Check that the vector property exists
    let property_name = "_vector_text_embedding";
    assert!(
        stored_entity.get_property(property_name).is_some(),
        "vector property should be stored"
    );
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

    // Verify all vectors were stored
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_embedding").is_some(),
            "vector should be stored for entity {:?}",
            entity_id
        );
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

    // Verify vectors are stored with correct property name
    let tx = db.begin_read().expect("failed to begin read transaction");
    for entity_id in &entity_ids {
        let entity =
            tx.get_entity(*entity_id).expect("failed to get entity").expect("entity not found");
        assert!(
            entity.get_property("_vector_content_vector").is_some(),
            "named vector should be stored"
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

    // Verify all three vectors are stored
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(entity.get_property("_vector_text_embedding").is_some(), "text embedding should exist");
    assert!(
        entity.get_property("_vector_image_embedding").is_some(),
        "image embedding should exist"
    );
    assert!(
        entity.get_property("_vector_audio_embedding").is_some(),
        "audio embedding should exist"
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

    // Update with new vector (same name)
    let vectors2 = vec![(entity_id, "embedding".to_string(), vec![0.9f32; 64])];
    db.bulk_insert_vectors("documents", &vectors2).expect("failed second insert");

    // Verify the vector was updated
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    if let Some(value) = entity.get_property("_vector_embedding") {
        // The vector should contain the new values
        if let manifoldb_core::Value::Vector(v) = value {
            assert!((v[0] - 0.9f32).abs() < 0.001, "vector should be updated to new value");
        } else {
            panic!("expected Vector value");
        }
    } else {
        panic!("vector property should exist");
    }
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

    // Insert a vector
    let vectors = vec![(entity_id, "text_embedding".to_string(), vec![0.1f32; 128])];
    db.bulk_insert_vectors("documents", &vectors).expect("failed to insert vectors");

    // Verify vector exists
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert!(
        entity.get_property("_vector_text_embedding").is_some(),
        "vector should exist before deletion"
    );
    drop(tx);

    // Delete the vector
    let to_delete = vec![(entity_id, "text_embedding".to_string())];
    let count = db.bulk_delete_vectors(&to_delete).expect("failed to delete vectors");

    assert_eq!(count, 1, "should delete 1 vector");

    // Verify vector was deleted
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");
    assert!(entity.get_property("_vector_text_embedding").is_none(), "vector should be deleted");
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

    // Insert multiple different named vectors
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

    // Verify selective deletion
    let tx = db.begin_read().expect("failed to begin read transaction");
    let entity = tx.get_entity(entity_id).expect("failed to get entity").expect("entity not found");

    assert!(entity.get_property("_vector_text_embedding").is_some(), "text embedding should exist");
    assert!(
        entity.get_property("_vector_image_embedding").is_none(),
        "image embedding should be deleted"
    );
    assert!(
        entity.get_property("_vector_audio_embedding").is_some(),
        "audio embedding should exist"
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
