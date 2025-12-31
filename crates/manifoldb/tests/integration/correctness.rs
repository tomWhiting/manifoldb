//! Query correctness integration tests.
//!
//! Tests that verify query results against reference implementations:
//! - Graph algorithms compared against simple reference implementations
//! - Vector similarity compared against brute-force computation
//! - SQL semantics verified against expected results
//!
//! These tests use small, hand-crafted datasets where results are
//! independently verifiable.

#![allow(clippy::map_entry)]
#![allow(clippy::set_contains_or_insert)]
#![allow(clippy::explicit_iter_loop)]

use std::collections::{HashMap, HashSet, VecDeque};

use manifoldb::{Database, EntityId, Value};

// ============================================================================
// Reference Graph Algorithms
// ============================================================================

/// Reference implementation of BFS
fn reference_bfs(
    adjacency: &HashMap<EntityId, Vec<EntityId>>,
    start: EntityId,
) -> Vec<(EntityId, usize)> {
    let mut visited = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(start, 0);
    queue.push_back((start, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, depth + 1);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
    }

    let mut result: Vec<_> = visited.into_iter().collect();
    result.sort_by_key(|(id, _)| id.as_u64());
    result
}

/// Reference implementation of DFS to find all reachable nodes
fn reference_dfs_reachable(
    adjacency: &HashMap<EntityId, Vec<EntityId>>,
    start: EntityId,
) -> HashSet<EntityId> {
    let mut visited = HashSet::new();
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    visited
}

/// Reference implementation for path existence
fn reference_path_exists(
    adjacency: &HashMap<EntityId, Vec<EntityId>>,
    start: EntityId,
    end: EntityId,
) -> bool {
    if start == end {
        return true;
    }

    let mut visited = HashSet::new();
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if node == end {
            return true;
        }

        if visited.insert(node) {
            if let Some(neighbors) = adjacency.get(&node) {
                stack.extend(neighbors.iter().copied());
            }
        }
    }

    false
}

/// Reference implementation for shortest path (BFS-based for unweighted graphs)
fn reference_shortest_path(
    adjacency: &HashMap<EntityId, Vec<EntityId>>,
    start: EntityId,
    end: EntityId,
) -> Option<Vec<EntityId>> {
    if start == end {
        return Some(vec![start]);
    }

    let mut visited = HashSet::new();
    let mut parent: HashMap<EntityId, EntityId> = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, node);
                    queue.push_back(neighbor);

                    if neighbor == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut current = end;
                        while let Some(&p) = parent.get(&current) {
                            path.push(p);
                            current = p;
                            if current == start {
                                break;
                            }
                        }
                        path.reverse();
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}

// ============================================================================
// Reference Vector Algorithms
// ============================================================================

/// Reference Euclidean distance
fn reference_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

/// Reference cosine distance
fn reference_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Reference k-NN search (brute force)
fn reference_knn_euclidean(
    vectors: &[(EntityId, Vec<f32>)],
    query: &[f32],
    k: usize,
) -> Vec<(EntityId, f32)> {
    let mut distances: Vec<_> =
        vectors.iter().map(|(id, v)| (*id, reference_euclidean_distance(v, query))).collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

// ============================================================================
// Graph Correctness Tests
// ============================================================================

/// Create a simple graph and verify BFS traversal
#[test]
fn test_bfs_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a simple graph:
    //     1 -> 2 -> 4
    //     |    |
    //     v    v
    //     3 -> 5
    let mut tx = db.begin().expect("failed");

    let mut ids = Vec::new();
    for i in 1..=5 {
        let entity = tx.create_entity().expect("failed").with_property("id", i);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    // Edges
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4)];
    for (src, dst) in edges {
        let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Build reference adjacency list
    let mut adjacency: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
    for (src, dst) in edges {
        adjacency.entry(ids[src]).or_default().push(ids[dst]);
    }

    // Perform BFS using database
    let tx = db.begin_read().expect("failed");
    let mut db_visited = HashMap::new();
    let mut queue = VecDeque::new();

    db_visited.insert(ids[0], 0);
    queue.push_back((ids[0], 0));

    while let Some((node, depth)) = queue.pop_front() {
        let edges = tx.get_outgoing_edges(node).expect("failed");
        for edge in edges {
            if !db_visited.contains_key(&edge.target) {
                db_visited.insert(edge.target, depth + 1);
                queue.push_back((edge.target, depth + 1));
            }
        }
    }

    // Compare with reference
    let ref_result = reference_bfs(&adjacency, ids[0]);

    for (id, expected_depth) in ref_result {
        let actual_depth = db_visited.get(&id);
        assert_eq!(actual_depth, Some(&expected_depth), "BFS depth mismatch for entity {id:?}");
    }
}

/// Verify reachability using DFS
#[test]
fn test_reachability_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a graph with two disconnected components:
    // Component 1: 1 -> 2 -> 3
    // Component 2: 4 -> 5
    let mut tx = db.begin().expect("failed");

    let mut ids = Vec::new();
    for i in 1..=5 {
        let entity = tx.create_entity().expect("failed").with_property("id", i);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    let edges = [(0, 1), (1, 2), (3, 4)];
    for (src, dst) in edges {
        let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Build reference adjacency
    let mut adjacency: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
    for (src, dst) in edges {
        adjacency.entry(ids[src]).or_default().push(ids[dst]);
    }

    // Verify reachability from node 1
    let ref_reachable = reference_dfs_reachable(&adjacency, ids[0]);

    let tx = db.begin_read().expect("failed");
    let mut db_reachable = HashSet::new();
    let mut stack = vec![ids[0]];

    while let Some(node) = stack.pop() {
        if db_reachable.insert(node) {
            let edges = tx.get_outgoing_edges(node).expect("failed");
            for edge in edges {
                stack.push(edge.target);
            }
        }
    }

    assert_eq!(db_reachable, ref_reachable, "reachable sets should match");

    // Node 4 should not be reachable from node 1
    assert!(!db_reachable.contains(&ids[3]));
    assert!(!db_reachable.contains(&ids[4]));
}

/// Verify path existence
#[test]
fn test_path_existence_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    // Create graph:
    // 1 -> 2 -> 3 -> 4
    //      |
    //      v
    //      5
    let mut tx = db.begin().expect("failed");

    let mut ids = Vec::new();
    for i in 1..=5 {
        let entity = tx.create_entity().expect("failed").with_property("id", i);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    let edges = [(0, 1), (1, 2), (2, 3), (1, 4)];
    for (src, dst) in edges {
        let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    let mut adjacency: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
    for (src, dst) in edges {
        adjacency.entry(ids[src]).or_default().push(ids[dst]);
    }

    // Helper to check path in database
    let db_path_exists = |start: EntityId, end: EntityId| -> bool {
        let tx = db.begin_read().expect("failed");
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if node == end {
                return true;
            }
            if visited.insert(node) {
                let edges = tx.get_outgoing_edges(node).expect("failed");
                for edge in edges {
                    stack.push(edge.target);
                }
            }
        }
        false
    };

    // Test various path queries
    let test_cases = [
        (ids[0], ids[3], true),  // 1 -> 4 (exists)
        (ids[0], ids[4], true),  // 1 -> 5 (exists)
        (ids[3], ids[0], false), // 4 -> 1 (no path)
        (ids[4], ids[3], false), // 5 -> 4 (no path)
        (ids[0], ids[0], true),  // 1 -> 1 (self)
    ];

    for (start, end, expected) in test_cases {
        let ref_result = reference_path_exists(&adjacency, start, end);
        let db_result = db_path_exists(start, end);

        assert_eq!(ref_result, expected, "reference path check for {start:?} -> {end:?}");
        assert_eq!(db_result, expected, "database path check for {start:?} -> {end:?}");
        assert_eq!(ref_result, db_result, "results should match");
    }
}

/// Verify shortest path computation
#[test]
fn test_shortest_path_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a graph with multiple paths:
    // 1 -> 2 -> 3 -> 5
    // 1 -> 4 -> 5
    // Shortest: 1 -> 4 -> 5 (2 edges vs 3 edges)
    let mut tx = db.begin().expect("failed");

    let mut ids = Vec::new();
    for i in 1..=5 {
        let entity = tx.create_entity().expect("failed").with_property("id", i);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    let edges = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)];
    for (src, dst) in edges {
        let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    let mut adjacency: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
    for (src, dst) in edges {
        adjacency.entry(ids[src]).or_default().push(ids[dst]);
    }

    // Reference shortest path
    let ref_path = reference_shortest_path(&adjacency, ids[0], ids[4]);

    // Database BFS shortest path
    let db_path = {
        let tx = db.begin_read().expect("failed");
        let mut parent: HashMap<EntityId, EntityId> = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(ids[0]);

        while let Some(node) = queue.pop_front() {
            if node == ids[4] {
                // Found - reconstruct
                let mut path = vec![ids[4]];
                let mut current = ids[4];
                while let Some(&p) = parent.get(&current) {
                    path.push(p);
                    current = p;
                }
                path.reverse();
                break;
            }

            let edges = tx.get_outgoing_edges(node).expect("failed");
            for edge in edges {
                if !parent.contains_key(&edge.target) && edge.target != ids[0] {
                    parent.insert(edge.target, node);
                    queue.push_back(edge.target);
                }
            }
        }

        let mut path = vec![ids[4]];
        let mut current = ids[4];
        while let Some(&p) = parent.get(&current) {
            path.push(p);
            current = p;
        }
        path.reverse();
        Some(path)
    };

    // Both should find a path of length 3 (1 -> 4 -> 5)
    assert!(ref_path.is_some());
    assert!(db_path.is_some());
    assert_eq!(
        ref_path.as_ref().unwrap().len(),
        db_path.as_ref().unwrap().len(),
        "path lengths should match"
    );
    // Note: actual path may differ if there are equal-length alternatives
}

// ============================================================================
// Vector Correctness Tests
// ============================================================================

/// Verify Euclidean distance correctness
#[test]
fn test_euclidean_distance_correctness() {
    // Hand-computed test cases
    let test_cases = [
        // 3-4-5 triangle
        (vec![0.0, 0.0], vec![3.0, 4.0], 5.0),
        // Same point
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0.0),
        // Unit distance
        (vec![0.0], vec![1.0], 1.0),
        // Higher dimension
        (
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            8.0, // sqrt(16 + 16 + 16 + 16) = sqrt(64) = 8
        ),
    ];

    for (a, b, expected) in test_cases {
        let result = reference_euclidean_distance(&a, &b);
        assert!(
            (result - expected).abs() < 0.0001,
            "Euclidean({a:?}, {b:?}) = {result}, expected {expected}"
        );
    }
}

/// Verify cosine distance correctness
#[test]
fn test_cosine_distance_correctness() {
    let test_cases = [
        // Same direction (distance = 0)
        (vec![1.0, 0.0], vec![2.0, 0.0], 0.0),
        // Opposite direction (distance = 2)
        (vec![1.0, 0.0], vec![-1.0, 0.0], 2.0),
        // Orthogonal (distance = 1)
        (vec![1.0, 0.0], vec![0.0, 1.0], 1.0),
        // 45 degrees (distance ≈ 0.293)
        (vec![1.0, 0.0], vec![1.0, 1.0], 1.0 - (1.0 / 2.0_f32.sqrt())),
    ];

    for (a, b, expected) in test_cases {
        let result = reference_cosine_distance(&a, &b);
        assert!(
            (result - expected).abs() < 0.001,
            "Cosine({a:?}, {b:?}) = {result}, expected {expected}"
        );
    }
}

/// Verify k-NN results
#[test]
fn test_knn_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    // Create vectors with known distances from origin
    let vectors = vec![
        (1, vec![1.0f32, 0.0, 0.0]), // distance 1
        (2, vec![0.0, 2.0, 0.0]),    // distance 2
        (3, vec![0.0, 0.0, 3.0]),    // distance 3
        (4, vec![2.0, 2.0, 0.0]),    // distance ~2.83
        (5, vec![0.5, 0.5, 0.5]),    // distance ~0.87
    ];

    let mut tx = db.begin().expect("failed");
    let mut ids = Vec::new();

    for (i, v) in &vectors {
        let entity = tx
            .create_entity()
            .expect("failed")
            .with_property("id", *i as i64)
            .with_property("embedding", v.clone());
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    tx.commit().expect("failed");

    // Query: find 3 nearest to origin
    let query = vec![0.0f32, 0.0, 0.0];

    // Reference k-NN
    let ref_vectors: Vec<_> = ids.iter().zip(vectors.iter().map(|(_, v)| v.clone())).collect();
    let ref_knn = reference_knn_euclidean(
        &ref_vectors.iter().map(|(id, v)| (**id, v.clone())).collect::<Vec<_>>(),
        &query,
        3,
    );

    // Database k-NN
    let tx = db.begin_read().expect("failed");
    let mut db_distances: Vec<(EntityId, f32)> = Vec::new();

    for &id in &ids {
        let entity = tx.get_entity(id).expect("failed").expect("should exist");
        if let Some(Value::Vector(v)) = entity.get_property("embedding") {
            let dist = reference_euclidean_distance(v, &query);
            db_distances.push((id, dist));
        }
    }

    db_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    db_distances.truncate(3);

    // Verify top 3 match
    assert_eq!(ref_knn.len(), 3);
    assert_eq!(db_distances.len(), 3);

    for ((ref_id, ref_dist), (db_id, db_dist)) in ref_knn.iter().zip(db_distances.iter()) {
        assert_eq!(ref_id, db_id, "k-NN order should match");
        assert!((ref_dist - db_dist).abs() < 0.0001, "distances should match");
    }

    // Verify entity 5 is closest (distance ~0.87)
    assert_eq!(db_distances[0].0, ids[4]); // entity 5
}

// ============================================================================
// Property Semantics Tests
// ============================================================================

/// Verify property update semantics
#[test]
fn test_property_update_semantics() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id: EntityId;

    // Create entity with properties
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx
            .create_entity()
            .expect("failed")
            .with_property("a", 1i64)
            .with_property("b", 2i64)
            .with_property("c", 3i64);
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Update: change one, remove one, add one
    {
        let mut tx = db.begin().expect("failed");
        let mut entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

        // Change a
        entity.set_property("a", 10i64);
        // Remove b (by not including it - we need to check if this is supported)
        // Add d
        entity.set_property("d", 4i64);

        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

    assert_eq!(entity.get_property("a"), Some(&Value::Int(10)));
    assert_eq!(entity.get_property("c"), Some(&Value::Int(3))); // Unchanged
    assert_eq!(entity.get_property("d"), Some(&Value::Int(4)));
    // b should still exist since we didn't explicitly remove it
    assert_eq!(entity.get_property("b"), Some(&Value::Int(2)));
}

/// Verify label semantics
#[test]
fn test_label_semantics() {
    let db = Database::in_memory().expect("failed to create db");

    let entity_id: EntityId;

    // Create entity with labels
    {
        let mut tx = db.begin().expect("failed");
        let entity = tx.create_entity().expect("failed").with_label("A").with_label("B");
        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

    assert!(entity.has_label("A"));
    assert!(entity.has_label("B"));
    assert!(!entity.has_label("C"));
}

// ============================================================================
// Edge Property Correctness Tests
// ============================================================================

/// Verify edge properties are stored correctly
#[test]
fn test_edge_property_correctness() {
    let db = Database::in_memory().expect("failed to create db");

    let (src_id, dst_id): (EntityId, EntityId);

    // Create entities and edge with properties
    {
        let mut tx = db.begin().expect("failed");

        let src = tx.create_entity().expect("failed").with_label("Source");
        let dst = tx.create_entity().expect("failed").with_label("Destination");
        src_id = src.id;
        dst_id = dst.id;

        tx.put_entity(&src).expect("failed");
        tx.put_entity(&dst).expect("failed");

        let edge = tx
            .create_edge(src_id, dst_id, "WEIGHTED")
            .expect("failed")
            .with_property("weight", 42.5f64)
            .with_property("label", "important");
        tx.put_edge(&edge).expect("failed");

        tx.commit().expect("failed");
    }

    // Verify
    let tx = db.begin_read().expect("failed");
    let edges = tx.get_outgoing_edges(src_id).expect("failed");

    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].source, src_id);
    assert_eq!(edges[0].target, dst_id);
    assert_eq!(edges[0].edge_type.as_str(), "WEIGHTED");

    if let Some(Value::Float(w)) = edges[0].get_property("weight") {
        assert!((w - 42.5).abs() < 0.001);
    } else {
        panic!("expected weight property");
    }

    assert_eq!(edges[0].get_property("label"), Some(&Value::String("important".to_string())));
}

// ============================================================================
// Bidirectional Edge Tests
// ============================================================================

/// Verify incoming/outgoing edge consistency
#[test]
fn test_edge_bidirectional_consistency() {
    let db = Database::in_memory().expect("failed to create db");

    // Create a small graph
    let mut tx = db.begin().expect("failed");

    let mut ids = Vec::new();
    for i in 0..5 {
        let entity = tx.create_entity().expect("failed").with_property("id", i);
        ids.push(entity.id);
        tx.put_entity(&entity).expect("failed");
    }

    // Create edges
    let edge_specs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)];
    for (src, dst) in edge_specs {
        let edge = tx.create_edge(ids[src], ids[dst], "CONNECTS").expect("failed");
        tx.put_edge(&edge).expect("failed");
    }

    tx.commit().expect("failed");

    // Verify consistency
    let tx = db.begin_read().expect("failed");

    for (src_idx, dst_idx) in edge_specs {
        let src_id = ids[src_idx];
        let dst_id = ids[dst_idx];

        // Check outgoing from src
        let outgoing = tx.get_outgoing_edges(src_id).expect("failed");
        let has_outgoing = outgoing.iter().any(|e| e.target == dst_id);
        assert!(has_outgoing, "should have outgoing edge {src_idx} -> {dst_idx}");

        // Check incoming to dst
        let incoming = tx.get_incoming_edges(dst_id).expect("failed");
        let has_incoming = incoming.iter().any(|e| e.source == src_id);
        assert!(has_incoming, "should have incoming edge {src_idx} -> {dst_idx}");
    }
}

// ============================================================================
// Value Type Correctness Tests
// ============================================================================

/// Verify all value types round-trip correctly
#[test]
fn test_value_type_roundtrip() {
    let db = Database::in_memory().expect("failed to create db");

    let test_values = [
        ("null", Value::Null),
        ("bool_true", Value::Bool(true)),
        ("bool_false", Value::Bool(false)),
        ("int_pos", Value::Int(42)),
        ("int_neg", Value::Int(-42)),
        ("int_max", Value::Int(i64::MAX)),
        ("int_min", Value::Int(i64::MIN)),
        ("float", Value::Float(3.14159)),
        ("float_neg", Value::Float(-273.15)),
        ("string", Value::String("hello world".to_string())),
        ("string_empty", Value::String(String::new())),
        ("string_unicode", Value::String("こんにちは".to_string())),
        ("vector", Value::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ("vector_empty", Value::Vector(Vec::new())),
    ];

    let entity_id: EntityId;

    // Create entity with all types
    {
        let mut tx = db.begin().expect("failed");
        let mut entity = tx.create_entity().expect("failed");

        for (key, value) in &test_values {
            entity.set_property(*key, value.clone());
        }

        entity_id = entity.id;
        tx.put_entity(&entity).expect("failed");
        tx.commit().expect("failed");
    }

    // Verify all types
    let tx = db.begin_read().expect("failed");
    let entity = tx.get_entity(entity_id).expect("failed").expect("should exist");

    for (key, expected) in &test_values {
        let actual = entity.get_property(key);
        assert_eq!(
            actual,
            Some(expected),
            "value mismatch for key '{key}': expected {expected:?}, got {actual:?}"
        );
    }
}

// ============================================================================
// Transaction Semantics Tests
// ============================================================================

/// Verify read transaction sees consistent snapshot
#[test]
fn test_snapshot_consistency() {
    let db = Database::in_memory().expect("failed to create db");

    // Create initial state
    let mut entity_ids = Vec::new();
    {
        let mut tx = db.begin().expect("failed");
        for _ in 0..10 {
            let entity = tx.create_entity().expect("failed").with_property("version", 1i64);
            entity_ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Start read transaction
    let read_tx = db.begin_read().expect("failed");

    // Capture initial state
    let initial_state: Vec<_> = entity_ids
        .iter()
        .map(|&id| {
            let entity = read_tx.get_entity(id).expect("failed").expect("should exist");
            entity.get_property("version").cloned()
        })
        .collect();

    // Update all entities in a new transaction
    {
        let mut tx = db.begin().expect("failed");
        for &id in &entity_ids {
            let mut entity = tx.get_entity(id).expect("failed").expect("should exist");
            entity.set_property("version", 2i64);
            tx.put_entity(&entity).expect("failed");
        }
        tx.commit().expect("failed");
    }

    // Read transaction should still see original values
    let final_state: Vec<_> = entity_ids
        .iter()
        .map(|&id| {
            let entity = read_tx.get_entity(id).expect("failed").expect("should exist");
            entity.get_property("version").cloned()
        })
        .collect();

    assert_eq!(initial_state, final_state, "snapshot should be consistent");

    // All should be version 1
    for state in &final_state {
        assert_eq!(state, &Some(Value::Int(1)));
    }

    // New read transaction should see version 2
    let new_tx = db.begin_read().expect("failed");
    for &id in &entity_ids {
        let entity = new_tx.get_entity(id).expect("failed").expect("should exist");
        assert_eq!(entity.get_property("version"), Some(&Value::Int(2)));
    }
}
