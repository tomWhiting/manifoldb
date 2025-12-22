//! Scale integration tests.
//!
//! Tests for large-scale data handling including:
//! - Large entity counts (up to 1M in full tests)
//! - Large edge counts (up to 10M in full tests)
//! - Large vector counts (up to 100K in full tests)
//!
//! Note: Full scale tests are marked `#[ignore]` by default as they take
//! significant time and resources. Run with `--ignored` flag to execute.

use std::collections::HashSet;
use std::time::Instant;

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Constants for test sizes
// ============================================================================

/// Small scale for quick CI tests
const SMALL_ENTITY_COUNT: usize = 1_000;
const SMALL_EDGE_COUNT: usize = 5_000;
const SMALL_VECTOR_DIM: usize = 64;
const SMALL_VECTOR_COUNT: usize = 500;

/// Medium scale for integration tests
const MEDIUM_ENTITY_COUNT: usize = 10_000;
const MEDIUM_EDGE_COUNT: usize = 50_000;
const MEDIUM_VECTOR_COUNT: usize = 5_000;

/// Full scale for comprehensive testing (run with --ignored)
const FULL_ENTITY_COUNT: usize = 1_000_000;
const FULL_EDGE_COUNT: usize = 10_000_000;
const FULL_VECTOR_COUNT: usize = 100_000;

/// Batch size for bulk operations
const BATCH_SIZE: usize = 10_000;

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple pseudo-random number generator for reproducible tests
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x853c_49e6_748f_ea9b } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_range(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        self.next_u64() % max
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

/// Create entities in batches
fn create_entities_batched(db: &Database, count: usize, batch_size: usize) -> Vec<EntityId> {
    let mut all_ids = Vec::with_capacity(count);
    let mut rng = Rng::new(42);

    let labels = ["Person", "Document", "Product", "Event", "Location"];

    for batch_start in (0..count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(count);

        let mut tx = db.begin().expect("failed to begin");

        for i in batch_start..batch_end {
            let label_idx = rng.next_range(labels.len() as u64) as usize;
            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label(labels[label_idx])
                .with_property("index", i as i64)
                .with_property("batch", (i / batch_size) as i64);

            all_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed to commit batch");
    }

    all_ids
}

/// Create edges in batches
fn create_edges_batched(
    db: &Database,
    entity_ids: &[EntityId],
    count: usize,
    batch_size: usize,
) -> usize {
    let mut rng = Rng::new(12345);
    let edge_types = ["CONNECTS", "FOLLOWS", "RELATED", "LINKS"];
    let mut created = 0;

    for batch_start in (0..count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(count);

        let mut tx = db.begin().expect("failed to begin");

        for _ in batch_start..batch_end {
            let src_idx = rng.next_range(entity_ids.len() as u64) as usize;
            let mut dst_idx = rng.next_range(entity_ids.len() as u64) as usize;

            // Avoid self-loops
            while dst_idx == src_idx {
                dst_idx = rng.next_range(entity_ids.len() as u64) as usize;
            }

            let type_idx = rng.next_range(edge_types.len() as u64) as usize;

            let edge = tx
                .create_edge(entity_ids[src_idx], entity_ids[dst_idx], edge_types[type_idx])
                .expect("failed");
            tx.put_edge(&edge).expect("failed");
            created += 1;
        }

        tx.commit().expect("failed to commit batch");
    }

    created
}

/// Create entities with vector properties
fn create_vectors_batched(
    db: &Database,
    count: usize,
    dimension: usize,
    batch_size: usize,
) -> Vec<EntityId> {
    let mut all_ids = Vec::with_capacity(count);
    let mut rng = Rng::new(54321);

    for batch_start in (0..count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(count);

        let mut tx = db.begin().expect("failed to begin");

        for i in batch_start..batch_end {
            // Generate random vector
            let vector: Vec<f32> = (0..dimension).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

            let entity = tx
                .create_entity()
                .expect("failed")
                .with_label("VectorEntity")
                .with_property("index", i as i64)
                .with_property("embedding", vector);

            all_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed to commit batch");
    }

    all_ids
}

// ============================================================================
// Small Scale Tests (Default - Run in CI)
// ============================================================================

#[test]
fn test_small_scale_entities() {
    let db = Database::in_memory().expect("failed to create db");

    let start = Instant::now();
    let ids = create_entities_batched(&db, SMALL_ENTITY_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), SMALL_ENTITY_COUNT);

    // Verify random samples
    let tx = db.begin_read().expect("failed");
    let mut rng = Rng::new(999);

    for _ in 0..100 {
        let idx = rng.next_range(ids.len() as u64) as usize;
        let entity = tx.get_entity(ids[idx]).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("index"), Some(&Value::Int(idx as i64)));
    }

    println!("Small scale ({SMALL_ENTITY_COUNT} entities): {create_time:?}");
}

#[test]
fn test_small_scale_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities first
    let entity_ids = create_entities_batched(&db, SMALL_ENTITY_COUNT, BATCH_SIZE);

    let start = Instant::now();
    let edge_count = create_edges_batched(&db, &entity_ids, SMALL_EDGE_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(edge_count, SMALL_EDGE_COUNT);

    // Verify edges exist
    let tx = db.begin_read().expect("failed");
    let mut total_edges = 0;

    for &id in &entity_ids[..100] {
        let outgoing = tx.get_outgoing_edges(id).expect("failed");
        let incoming = tx.get_incoming_edges(id).expect("failed");
        total_edges += outgoing.len() + incoming.len();
    }

    assert!(total_edges > 0, "should have created edges");
    println!("Small scale ({SMALL_EDGE_COUNT} edges): {create_time:?}");
}

#[test]
fn test_small_scale_vectors() {
    let db = Database::in_memory().expect("failed to create db");

    let start = Instant::now();
    let ids = create_vectors_batched(&db, SMALL_VECTOR_COUNT, SMALL_VECTOR_DIM, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), SMALL_VECTOR_COUNT);

    // Verify vectors
    let tx = db.begin_read().expect("failed");

    for &id in &ids[..100] {
        let entity = tx.get_entity(id).expect("failed").expect("should exist");
        if let Some(Value::Vector(v)) = entity.get_property("embedding") {
            assert_eq!(v.len(), SMALL_VECTOR_DIM);
        } else {
            panic!("expected vector property");
        }
    }

    println!("Small scale ({SMALL_VECTOR_COUNT} vectors, dim={SMALL_VECTOR_DIM}): {create_time:?}");
}

// ============================================================================
// Medium Scale Tests (Default - Run in CI)
// ============================================================================

#[test]
fn test_medium_scale_entities() {
    let db = Database::in_memory().expect("failed to create db");

    let start = Instant::now();
    let ids = create_entities_batched(&db, MEDIUM_ENTITY_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), MEDIUM_ENTITY_COUNT);

    // Sequential scan
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");
    let mut count = 0;

    for &id in &ids {
        if tx.get_entity(id).expect("failed").is_some() {
            count += 1;
        }
    }
    let scan_time = start.elapsed();

    assert_eq!(count, MEDIUM_ENTITY_COUNT);
    println!(
        "Medium scale ({MEDIUM_ENTITY_COUNT} entities): create={create_time:?}, scan={scan_time:?}"
    );
}

#[test]
fn test_medium_scale_edges() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_ids = create_entities_batched(&db, MEDIUM_ENTITY_COUNT, BATCH_SIZE);

    let start = Instant::now();
    let edge_count = create_edges_batched(&db, &entity_ids, MEDIUM_EDGE_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(edge_count, MEDIUM_EDGE_COUNT);

    // Graph traversal test
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");

    // BFS from first node
    let mut visited = HashSet::new();
    let mut queue = vec![entity_ids[0]];
    visited.insert(entity_ids[0]);

    while let Some(node) = queue.pop() {
        if visited.len() >= 1000 {
            break;
        }

        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                queue.push(edge.target);
            }
        }
    }
    let traversal_time = start.elapsed();

    println!("Medium scale ({MEDIUM_EDGE_COUNT} edges): create={create_time:?}, traversal={traversal_time:?}");
}

#[test]
fn test_medium_scale_vectors() {
    let db = Database::in_memory().expect("failed to create db");

    let start = Instant::now();
    let ids = create_vectors_batched(&db, MEDIUM_VECTOR_COUNT, SMALL_VECTOR_DIM, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), MEDIUM_VECTOR_COUNT);

    // Vector scan and distance computation
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");

    let query: Vec<f32> =
        (0..SMALL_VECTOR_DIM).map(|i| (i as f32) / (SMALL_VECTOR_DIM as f32)).collect();

    let mut distances = Vec::new();
    for &id in &ids {
        let entity = tx.get_entity(id).expect("failed").expect("should exist");
        if let Some(Value::Vector(v)) = entity.get_property("embedding") {
            // Euclidean distance
            let dist: f32 =
                query.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            distances.push((id, dist));
        }
    }

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let scan_time = start.elapsed();

    assert_eq!(distances.len(), MEDIUM_VECTOR_COUNT);
    println!("Medium scale ({MEDIUM_VECTOR_COUNT} vectors): create={create_time:?}, scan+sort={scan_time:?}");
}

// ============================================================================
// Large Scale Tests (Ignored by default - Run with --ignored)
// ============================================================================

#[test]
#[ignore = "Large scale test - run with --ignored"]
fn test_large_scale_entities() {
    let db = Database::in_memory().expect("failed to create db");

    println!("Creating {} entities...", FULL_ENTITY_COUNT);
    let start = Instant::now();
    let ids = create_entities_batched(&db, FULL_ENTITY_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), FULL_ENTITY_COUNT);
    println!("Created {} entities in {:?}", FULL_ENTITY_COUNT, create_time);

    // Random access test
    println!("Running random access test...");
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");
    let mut rng = Rng::new(12345);

    for _ in 0..10_000 {
        let idx = rng.next_range(ids.len() as u64) as usize;
        let entity = tx.get_entity(ids[idx]).expect("failed").expect("should exist");
        assert!(entity.get_property("index").is_some());
    }
    let access_time = start.elapsed();
    println!("10,000 random accesses in {:?}", access_time);

    // Sequential scan
    println!("Running sequential scan...");
    let start = Instant::now();
    let mut count = 0;
    for &id in &ids {
        if tx.get_entity(id).expect("failed").is_some() {
            count += 1;
        }
    }
    let scan_time = start.elapsed();

    assert_eq!(count, FULL_ENTITY_COUNT);
    println!("Sequential scan of {} entities in {:?}", FULL_ENTITY_COUNT, scan_time);
}

#[test]
#[ignore = "Large scale test - run with --ignored"]
fn test_large_scale_edges() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    println!("Creating {} entities...", FULL_ENTITY_COUNT);
    let entity_ids = create_entities_batched(&db, FULL_ENTITY_COUNT, BATCH_SIZE);
    println!("Entities created.");

    // Create edges
    println!("Creating {} edges...", FULL_EDGE_COUNT);
    let start = Instant::now();
    let edge_count = create_edges_batched(&db, &entity_ids, FULL_EDGE_COUNT, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(edge_count, FULL_EDGE_COUNT);
    println!("Created {} edges in {:?}", FULL_EDGE_COUNT, create_time);

    // Edge lookup test
    println!("Running edge lookup test...");
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");
    let mut total_edges = 0;

    for &id in &entity_ids[..1000] {
        let edges = tx.get_outgoing_edges(id).expect("failed");
        total_edges += edges.len();
    }
    let lookup_time = start.elapsed();
    println!("1000 edge lookups ({} edges found) in {:?}", total_edges, lookup_time);

    // Graph traversal
    println!("Running graph traversal...");
    let start = Instant::now();
    let mut visited = HashSet::new();
    let mut queue = vec![entity_ids[0]];
    visited.insert(entity_ids[0]);

    while let Some(node) = queue.pop() {
        if visited.len() >= 100_000 {
            break;
        }

        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target);
                queue.push(edge.target);
            }
        }
    }
    let traversal_time = start.elapsed();
    println!("Traversed {} nodes in {:?}", visited.len(), traversal_time);
}

#[test]
#[ignore = "Large scale test - run with --ignored"]
fn test_large_scale_vectors() {
    let db = Database::in_memory().expect("failed to create db");
    let dim = 128; // Common embedding dimension

    println!("Creating {} vectors of dimension {}...", FULL_VECTOR_COUNT, dim);
    let start = Instant::now();
    let ids = create_vectors_batched(&db, FULL_VECTOR_COUNT, dim, BATCH_SIZE);
    let create_time = start.elapsed();

    assert_eq!(ids.len(), FULL_VECTOR_COUNT);
    println!("Created {} vectors in {:?}", FULL_VECTOR_COUNT, create_time);

    // k-NN search simulation
    println!("Running k-NN search...");
    let start = Instant::now();
    let tx = db.begin_read().expect("failed");

    let query: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();

    let mut distances: Vec<(EntityId, f32)> = Vec::with_capacity(FULL_VECTOR_COUNT);
    for &id in &ids {
        let entity = tx.get_entity(id).expect("failed").expect("should exist");
        if let Some(Value::Vector(v)) = entity.get_property("embedding") {
            let dist: f32 =
                query.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            distances.push((id, dist));
        }
    }

    // Partial sort for top-k
    distances.select_nth_unstable_by(10, |a, b| a.1.partial_cmp(&b.1).unwrap());
    let search_time = start.elapsed();

    println!("k-NN search (k=10) over {} vectors in {:?}", FULL_VECTOR_COUNT, search_time);
    println!("Top 10 distances: {:?}", &distances[..10].iter().map(|(_, d)| d).collect::<Vec<_>>());
}

// ============================================================================
// Mixed Workload Tests
// ============================================================================

#[test]
fn test_mixed_workload_small() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    let entity_ids = create_entities_batched(&db, 1000, 500);

    // Create edges
    let _ = create_edges_batched(&db, &entity_ids, 5000, 1000);

    // Create vectors
    let vector_ids = create_vectors_batched(&db, 500, 32, 250);

    // Mixed operations
    let tx = db.begin_read().expect("failed");
    let mut rng = Rng::new(42);

    for _ in 0..100 {
        let op = rng.next_range(3);

        match op {
            0 => {
                // Entity lookup
                let idx = rng.next_range(entity_ids.len() as u64) as usize;
                let _ = tx.get_entity(entity_ids[idx]).expect("failed");
            }
            1 => {
                // Edge traversal
                let idx = rng.next_range(entity_ids.len() as u64) as usize;
                let _ = tx.get_outgoing_edges(entity_ids[idx]).expect("failed");
            }
            2 => {
                // Vector lookup
                let idx = rng.next_range(vector_ids.len() as u64) as usize;
                let _ = tx.get_entity(vector_ids[idx]).expect("failed");
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Delete at Scale Tests
// ============================================================================

#[test]
fn test_bulk_delete() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    let entity_ids = create_entities_batched(&db, 5000, 1000);

    // Delete half
    let to_delete: Vec<_> = entity_ids.iter().step_by(2).cloned().collect();
    let to_keep: Vec<_> = entity_ids.iter().skip(1).step_by(2).cloned().collect();

    for batch in to_delete.chunks(500) {
        let mut tx = db.begin().expect("failed");
        for &id in batch {
            tx.delete_entity(id).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Verify
    let tx = db.begin_read().expect("failed");

    for &id in &to_delete {
        assert!(tx.get_entity(id).expect("failed").is_none(), "deleted entity should be gone");
    }

    for &id in &to_keep {
        assert!(tx.get_entity(id).expect("failed").is_some(), "kept entity should exist");
    }
}

// ============================================================================
// Update at Scale Tests
// ============================================================================

#[test]
fn test_bulk_update() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities
    let entity_ids = create_entities_batched(&db, 2000, 500);

    // Update all entities
    for batch in entity_ids.chunks(500) {
        let mut tx = db.begin().expect("failed");

        for &id in batch {
            let mut entity = tx.get_entity(id).expect("failed").expect("should exist");
            entity.set_property("updated", true);
            entity.set_property("version", 2i64);
            tx.put_entity(&entity).expect("failed");
        }

        tx.commit().expect("failed");
    }

    // Verify updates
    let tx = db.begin_read().expect("failed");

    for &id in &entity_ids {
        let entity = tx.get_entity(id).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("updated"), Some(&Value::Bool(true)));
        assert_eq!(entity.get_property("version"), Some(&Value::Int(2)));
    }
}

// ============================================================================
// Memory Efficiency Tests
// ============================================================================

#[test]
fn test_repeated_transactions() {
    let db = Database::in_memory().expect("failed to create db");

    // Many small transactions to test for memory leaks
    for _ in 0..1000 {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_property("temp", true);
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Many read transactions
    for _ in 0..1000 {
        let tx = db.begin_read().expect("failed");
        let _ = tx.get_entity(EntityId::new(1));
        tx.rollback().expect("failed");
    }

    // Should still be functional
    let tx = db.begin_read().expect("failed");
    assert!(tx.get_entity(EntityId::new(1)).expect("failed").is_some());
}

// ============================================================================
// Query Performance at Scale
// ============================================================================

#[test]
fn test_label_filtering_at_scale() {
    let db = Database::in_memory().expect("failed to create db");

    // Create entities with different labels
    let mut tx = db.begin().expect("failed");

    for i in 0..1000 {
        let label = match i % 4 {
            0 => "Person",
            1 => "Document",
            2 => "Product",
            _ => "Event",
        };

        let entity = tx.create_entity().expect("failed").with_label(label).with_property("idx", i);
        tx.put_entity(&entity).expect("failed");
    }

    tx.commit().expect("failed");

    // Count entities by label (manual scan since no label index)
    let tx = db.begin_read().expect("failed");
    let mut person_count = 0;

    for id in 1..=1000 {
        if let Some(entity) = tx.get_entity(EntityId::new(id)).expect("failed") {
            if entity.has_label("Person") {
                person_count += 1;
            }
        }
    }

    assert_eq!(person_count, 250); // 1000 / 4
}
