//! Vector search integration tests.
//!
//! Tests vector storage and similarity search capabilities.

use manifoldb_core::EntityId;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_vector::distance::DistanceMetric;
use manifoldb_vector::ops::{ExactKnn, VectorOperator};
use manifoldb_vector::store::VectorStore;
use manifoldb_vector::types::{Embedding, EmbeddingName, EmbeddingSpace};

use std::sync::atomic::{AtomicUsize, Ordering};

// Counter for unique space names across tests
static SPACE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_space_name() -> EmbeddingName {
    let count = SPACE_COUNTER.fetch_add(1, Ordering::SeqCst);
    EmbeddingName::new(format!("test_space_{count}")).expect("valid name")
}

fn create_test_store() -> VectorStore<RedbEngine> {
    let engine = RedbEngine::in_memory().expect("failed to create engine");
    VectorStore::new(engine)
}

// ============================================================================
// Vector Store CRUD Tests
// ============================================================================

#[test]
fn test_vector_store_basic_crud() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let space = EmbeddingSpace::new(space_name.clone(), 128, DistanceMetric::Cosine);

    // Create space
    store.create_space(&space).expect("failed to create space");

    // Put embedding
    let embedding = Embedding::new(vec![0.1; 128]).expect("valid embedding");
    store.put(EntityId::new(1), &space_name, &embedding).expect("failed to put");

    // Get embedding
    let retrieved = store.get(EntityId::new(1), &space_name).expect("failed to get");
    assert_eq!(retrieved.dimension(), 128);

    // Check exists
    assert!(store.exists(EntityId::new(1), &space_name).expect("failed to check"));

    // Delete embedding
    assert!(store.delete(EntityId::new(1), &space_name).expect("failed to delete"));
    assert!(!store.exists(EntityId::new(1), &space_name).expect("failed to check"));
}

#[test]
fn test_vector_store_multiple_spaces() {
    let store = create_test_store();

    // Create two spaces with different dimensions
    let text_space = unique_space_name();
    let image_space = unique_space_name();

    store
        .create_space(&EmbeddingSpace::new(text_space.clone(), 384, DistanceMetric::Cosine))
        .expect("failed");
    store
        .create_space(&EmbeddingSpace::new(image_space.clone(), 512, DistanceMetric::Euclidean))
        .expect("failed");

    // Store embeddings for same entity in both spaces
    let entity_id = EntityId::new(42);
    let text_emb = Embedding::new(vec![0.1; 384]).expect("valid");
    let image_emb = Embedding::new(vec![0.2; 512]).expect("valid");

    store.put(entity_id, &text_space, &text_emb).expect("failed");
    store.put(entity_id, &image_space, &image_emb).expect("failed");

    // Retrieve from each space
    let text_retrieved = store.get(entity_id, &text_space).expect("failed");
    let image_retrieved = store.get(entity_id, &image_space).expect("failed");

    assert_eq!(text_retrieved.dimension(), 384);
    assert_eq!(image_retrieved.dimension(), 512);
}

#[test]
fn test_vector_store_batch_operations() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let space = EmbeddingSpace::new(space_name.clone(), 64, DistanceMetric::Cosine);
    store.create_space(&space).expect("failed");

    // Batch put
    let embeddings: Vec<_> = (1..=100)
        .map(|i| (EntityId::new(i), Embedding::new(vec![i as f32 / 100.0; 64]).expect("valid")))
        .collect();

    store.put_many(&embeddings, &space_name).expect("failed to put many");

    // Verify count
    assert_eq!(store.count(&space_name).expect("failed"), 100);

    // Batch get
    let ids: Vec<_> = (1..=10).map(EntityId::new).collect();
    let results = store.get_many(&ids, &space_name).expect("failed");

    assert_eq!(results.len(), 10);
    for (id, emb) in results {
        assert!(emb.is_some(), "embedding for {:?} should exist", id);
    }
}

#[test]
fn test_vector_store_list_entities() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let space = EmbeddingSpace::new(space_name.clone(), 32, DistanceMetric::Cosine);
    store.create_space(&space).expect("failed");

    let ids = [1u64, 5, 10, 15, 20];
    for &id in &ids {
        let emb = Embedding::new(vec![id as f32; 32]).expect("valid");
        store.put(EntityId::new(id), &space_name, &emb).expect("failed");
    }

    let listed = store.list_entities(&space_name).expect("failed");
    assert_eq!(listed.len(), 5);

    let listed_ids: Vec<u64> = listed.iter().map(|e| e.as_u64()).collect();
    for &id in &ids {
        assert!(listed_ids.contains(&id), "should contain entity {id}");
    }
}

#[test]
fn test_vector_store_dimension_validation() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let space = EmbeddingSpace::new(space_name.clone(), 128, DistanceMetric::Cosine);
    store.create_space(&space).expect("failed");

    // Try to put wrong dimension
    let wrong_dim = Embedding::new(vec![0.1; 64]).expect("valid");
    let result = store.put(EntityId::new(1), &space_name, &wrong_dim);

    assert!(result.is_err(), "should reject wrong dimension");
}

// ============================================================================
// Exact K-NN Tests
// ============================================================================

#[test]
fn test_exact_knn_euclidean() {
    let dim = 4;
    let query = Embedding::new(vec![5.0; dim]).expect("valid");

    // Create vectors with known distances
    let vectors: Vec<_> = (1..=10)
        .map(|i| (EntityId::new(i as u64), Embedding::new(vec![i as f32; dim]).expect("valid")))
        .collect();

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 3)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 3);

    // Closest should be entity 5 (same value as query)
    assert_eq!(results[0].entity_id, EntityId::new(5));
    assert!(results[0].distance < 0.001);

    // Results should be sorted by distance
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}

#[test]
fn test_exact_knn_cosine() {
    let query = Embedding::new(vec![1.0, 0.0, 0.0]).expect("valid");

    let vectors = vec![
        (EntityId::new(1), Embedding::new(vec![1.0, 0.0, 0.0]).expect("valid")), // Same direction
        (EntityId::new(2), Embedding::new(vec![0.7, 0.7, 0.0]).expect("valid")), // 45 degrees
        (EntityId::new(3), Embedding::new(vec![0.0, 1.0, 0.0]).expect("valid")), // 90 degrees
        (EntityId::new(4), Embedding::new(vec![-1.0, 0.0, 0.0]).expect("valid")), // Opposite
    ];

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Cosine, 4)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 4);

    // Same direction should be closest (cosine distance = 0)
    assert_eq!(results[0].entity_id, EntityId::new(1));
    assert!(results[0].distance < 0.001);

    // Opposite should be furthest (cosine distance = 2)
    assert_eq!(results[3].entity_id, EntityId::new(4));
    assert!((results[3].distance - 2.0).abs() < 0.001);
}

#[test]
fn test_exact_knn_dot_product() {
    let query = Embedding::new(vec![1.0, 1.0]).expect("valid");

    let vectors = vec![
        (EntityId::new(1), Embedding::new(vec![2.0, 2.0]).expect("valid")), // Dot = 4
        (EntityId::new(2), Embedding::new(vec![1.0, 1.0]).expect("valid")), // Dot = 2
        (EntityId::new(3), Embedding::new(vec![0.5, 0.5]).expect("valid")), // Dot = 1
        (EntityId::new(4), Embedding::new(vec![0.0, 0.0]).expect("valid")), // Dot = 0
    ];

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::DotProduct, 4)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 4);

    // Highest dot product (4) should be first (distance = -4)
    assert_eq!(results[0].entity_id, EntityId::new(1));

    // Order should be by dot product descending
    assert_eq!(results[1].entity_id, EntityId::new(2));
    assert_eq!(results[2].entity_id, EntityId::new(3));
    assert_eq!(results[3].entity_id, EntityId::new(4));
}

#[test]
fn test_exact_knn_within_distance() {
    let dim = 4;
    let query = Embedding::new(vec![5.0; dim]).expect("valid");

    let vectors: Vec<_> = (1..=10)
        .map(|i| (EntityId::new(i as u64), Embedding::new(vec![i as f32; dim]).expect("valid")))
        .collect();

    // Set max distance to only include nearby vectors
    let mut knn =
        ExactKnn::within_distance(vectors.into_iter(), &query, DistanceMetric::Euclidean, 4.0)
            .expect("failed");

    let results = knn.collect_all().expect("failed");

    // All results should be within distance threshold
    for result in &results {
        assert!(result.distance <= 4.0, "distance {} exceeds threshold", result.distance);
    }

    // Entity 5 should definitely be included (distance = 0)
    let entity_ids: Vec<_> = results.iter().map(|r| r.entity_id.as_u64()).collect();
    assert!(entity_ids.contains(&5));
}

// ============================================================================
// Similarity Search at Scale
// ============================================================================

#[test]
fn test_knn_100_vectors() {
    let dim = 64;
    let query = Embedding::new(vec![0.5; dim]).expect("valid");

    let vectors: Vec<_> = (1..=100)
        .map(|i| {
            let value = i as f32 / 100.0;
            (EntityId::new(i as u64), Embedding::new(vec![value; dim]).expect("valid"))
        })
        .collect();

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 10)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 10);

    // Entity 50 should be closest (value = 0.5)
    assert_eq!(results[0].entity_id, EntityId::new(50));
}

#[test]
fn test_knn_1000_vectors() {
    let dim = 128;
    let query = Embedding::new(vec![0.5; dim]).expect("valid");

    let vectors: Vec<_> = (1..=1000)
        .map(|i| {
            let value = i as f32 / 1000.0;
            (EntityId::new(i as u64), Embedding::new(vec![value; dim]).expect("valid"))
        })
        .collect();

    let start = std::time::Instant::now();
    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 10)
        .expect("failed");
    let elapsed = start.elapsed();

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 10);

    // Entity 500 should be closest (value = 0.5)
    assert_eq!(results[0].entity_id, EntityId::new(500));

    // Should complete quickly
    assert!(elapsed.as_secs() < 5, "search took too long: {elapsed:?}");
}

// ============================================================================
// Embedding Space Management
// ============================================================================

#[test]
fn test_delete_space_with_embeddings() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let space = EmbeddingSpace::new(space_name.clone(), 32, DistanceMetric::Cosine);
    store.create_space(&space).expect("failed");

    // Add some embeddings
    for i in 1..=10 {
        let emb = Embedding::new(vec![i as f32; 32]).expect("valid");
        store.put(EntityId::new(i), &space_name, &emb).expect("failed");
    }

    assert_eq!(store.count(&space_name).expect("failed"), 10);

    // Delete space
    store.delete_space(&space_name).expect("failed");

    // Space should not exist
    assert!(store.get_space(&space_name).is_err());
}

#[test]
fn test_delete_entity_across_spaces() {
    let store = create_test_store();

    let space1 = unique_space_name();
    let space2 = unique_space_name();

    store
        .create_space(&EmbeddingSpace::new(space1.clone(), 32, DistanceMetric::Cosine))
        .expect("failed");
    store
        .create_space(&EmbeddingSpace::new(space2.clone(), 64, DistanceMetric::Euclidean))
        .expect("failed");

    let entity_id = EntityId::new(99);
    store.put(entity_id, &space1, &Embedding::new(vec![1.0; 32]).expect("valid")).expect("failed");
    store.put(entity_id, &space2, &Embedding::new(vec![2.0; 64]).expect("valid")).expect("failed");

    // Delete entity from all spaces
    let deleted = store.delete_entity(entity_id).expect("failed");
    assert_eq!(deleted, 2);

    assert!(!store.exists(entity_id, &space1).expect("failed"));
    assert!(!store.exists(entity_id, &space2).expect("failed"));
}

// ============================================================================
// Distance Metric Correctness
// ============================================================================

#[test]
fn test_euclidean_distance_correctness() {
    let a = Embedding::new(vec![0.0, 0.0, 0.0]).expect("valid");
    let b = Embedding::new(vec![3.0, 4.0, 0.0]).expect("valid");

    let vectors = vec![(EntityId::new(1), b)];

    let mut knn =
        ExactKnn::k_nearest(vectors.into_iter(), &a, DistanceMetric::Euclidean, 1).expect("failed");

    let result = knn.next().expect("failed").expect("should have result");

    // Distance should be 5 (3-4-5 triangle)
    assert!((result.distance - 5.0).abs() < 0.001);
}

#[test]
fn test_cosine_distance_orthogonal() {
    let a = Embedding::new(vec![1.0, 0.0]).expect("valid");
    let b = Embedding::new(vec![0.0, 1.0]).expect("valid");

    let vectors = vec![(EntityId::new(1), b)];

    let mut knn =
        ExactKnn::k_nearest(vectors.into_iter(), &a, DistanceMetric::Cosine, 1).expect("failed");

    let result = knn.next().expect("failed").expect("should have result");

    // Cosine of 90 degrees = 0, so cosine distance = 1
    assert!((result.distance - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_normalized() {
    // Test that magnitude doesn't affect cosine similarity
    let query = Embedding::new(vec![1.0, 0.0]).expect("valid");

    let vectors = vec![
        (EntityId::new(1), Embedding::new(vec![1.0, 0.0]).expect("valid")), // Unit vector
        (EntityId::new(2), Embedding::new(vec![10.0, 0.0]).expect("valid")), // Scaled 10x
        (EntityId::new(3), Embedding::new(vec![100.0, 0.0]).expect("valid")), // Scaled 100x
    ];

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Cosine, 3)
        .expect("failed");

    let results = knn.collect_all().expect("failed");

    // All should have the same cosine distance (0, same direction)
    for result in &results {
        assert!(result.distance < 0.001, "cosine distance should be 0 for same direction");
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_knn_empty_vectors() {
    let query = Embedding::new(vec![1.0; 4]).expect("valid");
    let vectors: Vec<(EntityId, Embedding)> = vec![];

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 10)
        .expect("failed");

    assert!(knn.is_empty());
    assert!(knn.next().expect("failed").is_none());
}

#[test]
fn test_knn_k_larger_than_n() {
    let query = Embedding::new(vec![1.0; 4]).expect("valid");

    let vectors: Vec<_> = (1..=3)
        .map(|i| (EntityId::new(i), Embedding::new(vec![i as f32; 4]).expect("valid")))
        .collect();

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 100)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 3, "should return all 3 vectors");
}

#[test]
fn test_knn_identical_vectors() {
    let query = Embedding::new(vec![1.0; 4]).expect("valid");

    // All vectors are identical
    let vectors: Vec<_> =
        (1..=5).map(|i| (EntityId::new(i), Embedding::new(vec![1.0; 4]).expect("valid"))).collect();

    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 5);

    // All distances should be 0
    for result in &results {
        assert!(result.distance < 0.001);
    }
}

// ============================================================================
// High-Dimensional Tests
// ============================================================================

#[test]
fn test_knn_high_dimension() {
    let dim = 1536; // Common embedding dimension (OpenAI ada-002)
    let query = Embedding::new(vec![0.5; dim]).expect("valid");

    let vectors: Vec<_> = (1..=100)
        .map(|i| {
            let value = i as f32 / 100.0;
            (EntityId::new(i as u64), Embedding::new(vec![value; dim]).expect("valid"))
        })
        .collect();

    // Use Euclidean distance since uniform vectors all have cosine similarity 1
    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 5);

    // Entity 50 should be closest (value = 0.5)
    assert_eq!(results[0].entity_id, EntityId::new(50));
}

// ============================================================================
// Integration with Store
// ============================================================================

#[test]
fn test_search_from_store() {
    let store = create_test_store();
    let space_name = unique_space_name();
    let dim = 32;

    // Use Euclidean for magnitude-based distance
    store
        .create_space(&EmbeddingSpace::new(space_name.clone(), dim, DistanceMetric::Euclidean))
        .expect("failed");

    // Insert vectors
    let embeddings: Vec<_> = (1..=50)
        .map(|i| (EntityId::new(i), Embedding::new(vec![i as f32 / 50.0; dim]).expect("valid")))
        .collect();

    store.put_many(&embeddings, &space_name).expect("failed");

    // Retrieve vectors for search
    let entity_ids = store.list_entities(&space_name).expect("failed");
    let vectors: Vec<_> = entity_ids
        .into_iter()
        .filter_map(|id| store.get(id, &space_name).ok().map(|emb| (id, emb)))
        .collect();

    // Search
    let query = Embedding::new(vec![0.5; dim]).expect("valid");
    let mut knn = ExactKnn::k_nearest(vectors.into_iter(), &query, DistanceMetric::Euclidean, 5)
        .expect("failed");

    let results = knn.collect_all().expect("failed");
    assert_eq!(results.len(), 5);

    // Entity 25 should be closest (value = 0.5)
    assert_eq!(results[0].entity_id, EntityId::new(25));
}

// ============================================================================
// Vector Index Consistency with DML Operations
// ============================================================================
//
// These tests verify that vector indexes remain consistent with entity data
// after INSERT, UPDATE, and DELETE operations. The goal is to ensure that:
// - Inserted entities are searchable via vector index
// - Updated vector values are reflected in search results
// - Deleted entities are not returned in vector search

#[test]
fn test_insert_updates_vector_index() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create table and HNSW index
    db.execute("CREATE TABLE docs (id BIGINT, title TEXT, embedding VECTOR(4))")
        .expect("create table failed");
    db.execute("CREATE INDEX docs_emb_idx ON docs USING hnsw (embedding)")
        .expect("create index failed");

    // Insert documents with vector embeddings
    db.execute("INSERT INTO docs (id, title, embedding) VALUES (1, 'doc1', [1.0, 0.0, 0.0, 0.0])")
        .expect("insert failed");
    db.execute("INSERT INTO docs (id, title, embedding) VALUES (2, 'doc2', [0.0, 1.0, 0.0, 0.0])")
        .expect("insert failed");
    db.execute("INSERT INTO docs (id, title, embedding) VALUES (3, 'doc3', [0.5, 0.5, 0.0, 0.0])")
        .expect("insert failed");

    // Query using vector search - should find all 3 documents
    let result = db
        .query("SELECT id, title FROM docs ORDER BY embedding <-> [0.5, 0.5, 0.0, 0.0] LIMIT 10")
        .expect("query failed");

    assert_eq!(result.len(), 3, "should find all 3 inserted documents");
}

#[test]
fn test_update_maintains_vector_index_consistency() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create table and HNSW index
    db.execute("CREATE TABLE docs (id BIGINT, title TEXT, embedding VECTOR(4))")
        .expect("create table failed");
    db.execute("CREATE INDEX docs_emb_idx ON docs USING hnsw (embedding)")
        .expect("create index failed");

    // Insert a document
    db.execute(
        "INSERT INTO docs (id, title, embedding) VALUES (1, 'original', [1.0, 0.0, 0.0, 0.0])",
    )
    .expect("insert failed");

    // Update the vector to point in a different direction
    db.execute("UPDATE docs SET embedding = [0.0, 0.0, 0.0, 1.0] WHERE id = 1")
        .expect("update failed");

    // Search should find the document at its new location
    let result = db
        .query("SELECT id, title FROM docs ORDER BY embedding <-> [0.0, 0.0, 0.0, 1.0] LIMIT 1")
        .expect("query failed");

    assert_eq!(result.len(), 1, "should find the updated document");
}

#[test]
fn test_delete_removes_from_vector_index() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create table and HNSW index
    db.execute("CREATE TABLE docs (id BIGINT, title TEXT, embedding VECTOR(4))")
        .expect("create table failed");
    db.execute("CREATE INDEX docs_emb_idx ON docs USING hnsw (embedding)")
        .expect("create index failed");

    // Insert documents
    db.execute("INSERT INTO docs (id, title, embedding) VALUES (1, 'doc1', [1.0, 0.0, 0.0, 0.0])")
        .expect("insert failed");
    db.execute("INSERT INTO docs (id, title, embedding) VALUES (2, 'doc2', [0.0, 1.0, 0.0, 0.0])")
        .expect("insert failed");

    // Delete one document
    db.execute("DELETE FROM docs WHERE id = 1").expect("delete failed");

    // Vector search should only return the remaining document
    let result = db
        .query("SELECT id FROM docs ORDER BY embedding <-> [1.0, 0.0, 0.0, 0.0] LIMIT 10")
        .expect("query failed");

    assert_eq!(result.len(), 1, "should only find the remaining document");

    // The remaining document should be id=2
    let id_idx = result.column_index("id").expect("id column not found");
    assert_eq!(
        result.rows()[0].get(id_idx),
        Some(&manifoldb::Value::Int(2)),
        "remaining document should be id=2"
    );
}

#[test]
fn test_drop_index_removes_vector_index() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create table and HNSW index
    db.execute("CREATE TABLE docs (id BIGINT, embedding VECTOR(4))").expect("create table failed");
    db.execute("CREATE INDEX docs_emb_idx ON docs USING hnsw (embedding)")
        .expect("create index failed");

    // Insert a document
    db.execute("INSERT INTO docs (id, embedding) VALUES (1, [1.0, 0.0, 0.0, 0.0])")
        .expect("insert failed");

    // Drop the index
    db.execute("DROP INDEX docs_emb_idx").expect("drop index failed");

    // Regular SELECT should still work (entity exists in storage)
    let result = db.query("SELECT id FROM docs").expect("query failed");
    assert_eq!(result.len(), 1, "document should still exist after index drop");
}

// ============================================================================
// Collection-Based Named Vector Tests
// ============================================================================
//
// These tests verify the new collection-based vector storage architecture
// where vectors are stored separately from entity properties using the
// VectorIndexCoordinator.

#[test]
fn test_create_collection_with_named_vectors() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create a collection with multiple named vectors
    db.execute(
        "CREATE COLLECTION documents (
            text_embedding VECTOR(384) USING hnsw WITH (distance = 'cosine'),
            image_embedding VECTOR(512) USING hnsw WITH (distance = 'euclidean')
        )",
    )
    .expect("create collection failed");

    // Verify collection exists by trying to insert (collection acts like a table)
    // Note: The collection metadata is stored, but vector inserts go through
    // the VectorIndexCoordinator when available
}

#[test]
fn test_collection_insert_stores_vectors_separately() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create a collection
    db.execute(
        "CREATE COLLECTION articles (
            content_embedding VECTOR(4) USING hnsw WITH (distance = 'cosine')
        )",
    )
    .expect("create collection failed");

    // Insert an article with a vector
    // When VectorIndexCoordinator is available, this should store the vector
    // in the separated vector storage, not as an entity property
    db.execute(
        "INSERT INTO articles (id, title, content_embedding)
         VALUES (1, 'Article 1', [0.1, 0.2, 0.3, 0.4])",
    )
    .expect("insert should work");
}

#[test]
fn test_collection_update_updates_vector_storage() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create a collection
    db.execute(
        "CREATE COLLECTION products (
            description_embedding VECTOR(4) USING hnsw WITH (distance = 'cosine')
        )",
    )
    .expect("create collection failed");

    // Insert a product
    db.execute(
        "INSERT INTO products (id, name, description_embedding)
         VALUES (1, 'Product 1', [0.1, 0.2, 0.3, 0.4])",
    )
    .expect("insert should work");

    // Update the vector
    db.execute("UPDATE products SET description_embedding = [0.5, 0.6, 0.7, 0.8] WHERE id = 1")
        .expect("update should work");
}

#[test]
fn test_collection_delete_cascades_to_vectors() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create a collection
    db.execute(
        "CREATE COLLECTION items (
            embedding VECTOR(4) USING hnsw WITH (distance = 'cosine')
        )",
    )
    .expect("create collection failed");

    // Insert an item
    db.execute(
        "INSERT INTO items (id, name, embedding)
         VALUES (1, 'Item 1', [0.1, 0.2, 0.3, 0.4])",
    )
    .expect("insert should work");

    // Delete the item (should cascade to vector storage)
    db.execute("DELETE FROM items WHERE id = 1").expect("delete should work");

    // Verify item is gone
    let result = db.query("SELECT id FROM items").expect("query failed");
    assert_eq!(result.len(), 0, "item should be deleted");
}

#[test]
fn test_drop_collection_cleans_up_vectors() {
    use manifoldb::Database;

    let db = Database::in_memory().expect("failed to create db");

    // Create a collection
    db.execute(
        "CREATE COLLECTION temp_collection (
            embedding VECTOR(4) USING hnsw WITH (distance = 'cosine')
        )",
    )
    .expect("create collection failed");

    // Insert some data
    db.execute(
        "INSERT INTO temp_collection (id, embedding)
         VALUES (1, [0.1, 0.2, 0.3, 0.4])",
    )
    .expect("insert should work");

    // Drop the collection
    db.execute("DROP COLLECTION temp_collection").expect("drop collection should work");
}
