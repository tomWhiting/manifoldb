//! ManifoldDB benchmarks.
//!
//! Comprehensive benchmarks covering:
//! - Storage layer operations (entity/edge CRUD)
//! - Graph traversal operations
//! - Vector search operations
//! - Query execution

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use manifoldb::{Database, EntityId};
use manifoldb_core::EntityId as CoreEntityId;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_vector::distance::DistanceMetric;
use manifoldb_vector::ops::{ExactKnn, VectorOperator};
use manifoldb_vector::store::VectorStore;
use manifoldb_vector::types::{Embedding, EmbeddingName, EmbeddingSpace};

use std::sync::atomic::{AtomicUsize, Ordering};

// ============================================================================
// Helper: Simple RNG for reproducible benchmarks
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    const fn new(seed: u64) -> Self {
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

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

static SPACE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn unique_space_name() -> EmbeddingName {
    let count = SPACE_COUNTER.fetch_add(1, Ordering::SeqCst);
    EmbeddingName::new(format!("bench_space_{count}")).expect("valid name")
}

// ============================================================================
// Storage Layer Benchmarks
// ============================================================================

fn storage_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage");

    // Entity insert throughput
    for count in [100, 1000, 10_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::new("entity_insert", count), &count, |b, &count| {
            b.iter_with_setup(
                || Database::in_memory().expect("failed to create db"),
                |db| {
                    let mut tx = db.begin().expect("failed");
                    for i in 0..count {
                        let entity =
                            tx.create_entity().expect("failed").with_property("index", i as i64);
                        tx.put_entity(&entity).expect("failed");
                    }
                    tx.commit().expect("failed");
                    black_box(db)
                },
            );
        });
    }

    // Point lookup throughput
    for count in [100, 1000, 10_000] {
        group.bench_with_input(BenchmarkId::new("entity_lookup", count), &count, |b, &count| {
            // Setup: create database with entities
            let db = Database::in_memory().expect("failed");
            let mut ids = Vec::with_capacity(count as usize);
            {
                let mut tx = db.begin().expect("failed");
                for i in 0..count {
                    let entity =
                        tx.create_entity().expect("failed").with_property("index", i as i64);
                    ids.push(entity.id);
                    tx.put_entity(&entity).expect("failed");
                }
                tx.commit().expect("failed");
            }

            let mut rng = Rng::new(42);

            b.iter(|| {
                let tx = db.begin_read().expect("failed");
                // Look up 100 random entities
                for _ in 0..100 {
                    let idx = (rng.next_u64() % count) as usize;
                    let entity = tx.get_entity(ids[idx]).expect("failed");
                    black_box(entity);
                }
            });
        });
    }

    // Entity update throughput
    group.bench_function("entity_update_100", |b| {
        let db = Database::in_memory().expect("failed");
        let mut ids = Vec::new();
        {
            let mut tx = db.begin().expect("failed");
            for _ in 0..100 {
                let entity = tx.create_entity().expect("failed").with_property("count", 0i64);
                ids.push(entity.id);
                tx.put_entity(&entity).expect("failed");
            }
            tx.commit().expect("failed");
        }

        let mut counter = 0i64;

        b.iter(|| {
            let mut tx = db.begin().expect("failed");
            for &id in &ids {
                let mut entity = tx.get_entity(id).expect("failed").expect("not found");
                entity.set_property("count", counter);
                tx.put_entity(&entity).expect("failed");
            }
            tx.commit().expect("failed");
            counter += 1;
        });
    });

    // Edge insert throughput
    for count in [100, 1000, 10_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::new("edge_insert", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let db = Database::in_memory().expect("failed");
                    // Pre-create nodes
                    let mut ids = Vec::new();
                    {
                        let mut tx = db.begin().expect("failed");
                        for _ in 0..100 {
                            let entity = tx.create_entity().expect("failed");
                            ids.push(entity.id);
                            tx.put_entity(&entity).expect("failed");
                        }
                        tx.commit().expect("failed");
                    }
                    (db, ids)
                },
                |(db, ids)| {
                    let mut rng = Rng::new(42);
                    let mut tx = db.begin().expect("failed");
                    for _ in 0..count {
                        let src = (rng.next_u64() % ids.len() as u64) as usize;
                        let mut dst = (rng.next_u64() % ids.len() as u64) as usize;
                        while dst == src {
                            dst = (rng.next_u64() % ids.len() as u64) as usize;
                        }
                        let edge = tx.create_edge(ids[src], ids[dst], "LINKS").expect("failed");
                        tx.put_edge(&edge).expect("failed");
                    }
                    tx.commit().expect("failed");
                    black_box(db)
                },
            );
        });
    }

    group.finish();
}

// ============================================================================
// Graph Traversal Benchmarks
// ============================================================================

fn graph_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph");

    // Setup linear chain
    let chain_db = Database::in_memory().expect("failed");
    let chain_ids: Vec<EntityId> = {
        let mut tx = chain_db.begin().expect("failed");
        let mut ids = Vec::new();
        for i in 0..1000 {
            let entity = tx.create_entity().expect("failed").with_property("pos", i as i64);
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }
        for i in 0..999 {
            let edge = tx.create_edge(ids[i], ids[i + 1], "NEXT").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }
        tx.commit().expect("failed");
        ids
    };

    // Traverse linear chain
    for hops in [10, 100, 500] {
        group.bench_with_input(BenchmarkId::new("chain_traverse", hops), &hops, |b, &hops| {
            b.iter(|| {
                let tx = chain_db.begin_read().expect("failed");
                let mut current = chain_ids[0];
                for _ in 0..hops {
                    let edges = tx.get_outgoing_edges(current).expect("failed");
                    if edges.is_empty() {
                        break;
                    }
                    current = edges[0].target;
                }
                black_box(current)
            });
        });
    }

    // Setup star graph
    let star_db = Database::in_memory().expect("failed");
    let star_ids: Vec<EntityId> = {
        let mut tx = star_db.begin().expect("failed");
        let center = tx.create_entity().expect("failed").with_label("Center");
        tx.put_entity(&center).expect("failed");

        let mut ids = vec![center.id];
        for i in 0..1000 {
            let leaf = tx.create_entity().expect("failed").with_property("index", i as i64);
            ids.push(leaf.id);
            tx.put_entity(&leaf).expect("failed");

            let edge = tx.create_edge(center.id, leaf.id, "CONNECTS").expect("failed");
            tx.put_edge(&edge).expect("failed");
        }
        tx.commit().expect("failed");
        ids
    };

    // Fan-out query (get all neighbors)
    group.bench_function("star_fan_out_1000", |b| {
        b.iter(|| {
            let tx = star_db.begin_read().expect("failed");
            let edges = tx.get_outgoing_edges(star_ids[0]).expect("failed");
            black_box(edges.len())
        });
    });

    // Fan-in query
    group.bench_function("star_fan_in", |b| {
        b.iter(|| {
            let tx = star_db.begin_read().expect("failed");
            // Pick a random leaf and get its incoming edges
            let edges = tx.get_incoming_edges(star_ids[500]).expect("failed");
            black_box(edges.len())
        });
    });

    // Dense graph - setup
    let dense_db = Database::in_memory().expect("failed");
    let dense_ids: Vec<EntityId> = {
        let mut tx = dense_db.begin().expect("failed");
        let mut ids = Vec::new();
        let n = 100;

        for _ in 0..n {
            let entity = tx.create_entity().expect("failed");
            ids.push(entity.id);
            tx.put_entity(&entity).expect("failed");
        }

        // Each node connects to 10 others
        let mut rng = Rng::new(42);
        for i in 0..n {
            for _ in 0..10 {
                let mut dst = (rng.next_u64() % n as u64) as usize;
                while dst == i {
                    dst = (rng.next_u64() % n as u64) as usize;
                }
                let edge = tx.create_edge(ids[i], ids[dst], "LINKS").expect("failed");
                tx.put_edge(&edge).expect("failed");
            }
        }
        tx.commit().expect("failed");
        ids
    };

    // BFS traversal
    group.bench_function("dense_bfs_depth_3", |b| {
        use std::collections::HashSet;

        b.iter(|| {
            let tx = dense_db.begin_read().expect("failed");
            let mut visited = HashSet::new();
            let mut queue = vec![dense_ids[0]];
            let mut depth = 0;

            while depth < 3 && !queue.is_empty() {
                let mut next_level = Vec::new();
                for node in queue {
                    if visited.insert(node) {
                        let edges = tx.get_outgoing_edges(node).expect("failed");
                        for edge in edges {
                            if !visited.contains(&edge.target) {
                                next_level.push(edge.target);
                            }
                        }
                    }
                }
                queue = next_level;
                depth += 1;
            }
            black_box(visited.len())
        });
    });

    group.finish();
}

// ============================================================================
// Vector Search Benchmarks
// ============================================================================

fn vector_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector");

    // Generate test vectors
    fn generate_vectors(rng: &mut Rng, count: usize, dim: usize) -> Vec<(CoreEntityId, Embedding)> {
        (0..count)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
                (CoreEntityId::new((i + 1) as u64), Embedding::new(v).expect("valid"))
            })
            .collect()
    }

    // k-NN search benchmarks
    for n in [100, 1000, 10_000] {
        for dim in [64, 128, 384] {
            let id = format!("n{}_d{}", n, dim);
            group.bench_with_input(
                BenchmarkId::new("exact_knn", &id),
                &(n, dim),
                |b, &(n, dim)| {
                    let mut rng = Rng::new(42);
                    let vectors = generate_vectors(&mut rng, n, dim);
                    let query =
                        Embedding::new((0..dim).map(|_| rng.next_f32_range(-1.0, 1.0)).collect())
                            .expect("valid");

                    b.iter(|| {
                        let mut knn = ExactKnn::k_nearest(
                            vectors.iter().cloned(),
                            &query,
                            DistanceMetric::Cosine,
                            10,
                        )
                        .expect("failed");
                        let results = knn.collect_all().expect("failed");
                        black_box(results)
                    });
                },
            );
        }
    }

    // Different k values
    for k in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::new("knn_k", k), &k, |b, &k| {
            let mut rng = Rng::new(42);
            let vectors = generate_vectors(&mut rng, 1000, 128);
            let query = Embedding::new((0..128).map(|_| rng.next_f32_range(-1.0, 1.0)).collect())
                .expect("valid");

            b.iter(|| {
                let mut knn = ExactKnn::k_nearest(
                    vectors.iter().cloned(),
                    &query,
                    DistanceMetric::Euclidean,
                    k,
                )
                .expect("failed");
                let results = knn.collect_all().expect("failed");
                black_box(results)
            });
        });
    }

    // Distance metric comparison
    for metric in [DistanceMetric::Cosine, DistanceMetric::Euclidean, DistanceMetric::DotProduct] {
        let name = match metric {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::DotProduct => "dot_product",
        };
        group.bench_with_input(BenchmarkId::new("metric", name), &metric, |b, &metric| {
            let mut rng = Rng::new(42);
            let vectors = generate_vectors(&mut rng, 1000, 128);
            let query = Embedding::new((0..128).map(|_| rng.next_f32_range(-1.0, 1.0)).collect())
                .expect("valid");

            b.iter(|| {
                let mut knn = ExactKnn::k_nearest(vectors.iter().cloned(), &query, metric, 10)
                    .expect("failed");
                let results = knn.collect_all().expect("failed");
                black_box(results)
            });
        });
    }

    // Vector store operations
    group.bench_function("store_put_100", |b| {
        b.iter_with_setup(
            || {
                let engine = RedbEngine::in_memory().expect("failed");
                let store = VectorStore::new(engine);
                let space_name = unique_space_name();
                store
                    .create_space(&EmbeddingSpace::new(
                        space_name.clone(),
                        128,
                        DistanceMetric::Cosine,
                    ))
                    .expect("failed");

                let mut rng = Rng::new(42);
                let embeddings: Vec<_> = (1..=100)
                    .map(|i| {
                        let v: Vec<f32> = (0..128).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
                        (CoreEntityId::new(i), Embedding::new(v).expect("valid"))
                    })
                    .collect();

                (store, space_name, embeddings)
            },
            |(store, space_name, embeddings)| {
                for (id, emb) in embeddings {
                    store.put(id, &space_name, &emb).expect("failed");
                }
                black_box(store)
            },
        );
    });

    group.bench_function("store_get_100", |b| {
        // Setup: create store with embeddings
        let engine = RedbEngine::in_memory().expect("failed");
        let store = VectorStore::new(engine);
        let space_name = unique_space_name();
        store
            .create_space(&EmbeddingSpace::new(space_name.clone(), 128, DistanceMetric::Cosine))
            .expect("failed");

        let mut rng = Rng::new(42);
        for i in 1..=100 {
            let v: Vec<f32> = (0..128).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
            store
                .put(CoreEntityId::new(i), &space_name, &Embedding::new(v).expect("valid"))
                .expect("failed");
        }

        b.iter(|| {
            for i in 1..=100 {
                let emb = store.get(CoreEntityId::new(i), &space_name).expect("failed");
                black_box(emb);
            }
        });
    });

    group.finish();
}

// ============================================================================
// Query Execution Benchmarks
// ============================================================================

fn query_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");

    // SQL parsing
    group.bench_function("parse_simple_select", |b| {
        let db = Database::in_memory().expect("failed");
        b.iter(|| {
            let result = db.query("SELECT * FROM users WHERE id = 1");
            black_box(result)
        });
    });

    group.bench_function("parse_complex_select", |b| {
        let db = Database::in_memory().expect("failed");
        b.iter(|| {
            let result = db.query(
                "SELECT u.name, u.age, COUNT(o.id) as order_count \
                 FROM users u \
                 LEFT JOIN orders o ON u.id = o.user_id \
                 WHERE u.active = true AND u.age > 18 \
                 GROUP BY u.id, u.name, u.age \
                 HAVING COUNT(o.id) > 0 \
                 ORDER BY u.name ASC \
                 LIMIT 100 OFFSET 10",
            );
            black_box(result)
        });
    });

    // Full query cycle (parse to result)
    group.bench_function("query_cycle_simple", |b| {
        let db = Database::in_memory().expect("failed");
        b.iter(|| {
            let result = db.query("SELECT id, name FROM users WHERE id = 1").expect("parse ok");
            black_box(result)
        });
    });

    // Transaction begin/commit overhead
    group.bench_function("tx_begin_commit_empty", |b| {
        let db = Database::in_memory().expect("failed");
        b.iter(|| {
            let tx = db.begin().expect("failed");
            tx.commit().expect("failed");
        });
    });

    group.bench_function("tx_begin_read_rollback", |b| {
        let db = Database::in_memory().expect("failed");
        b.iter(|| {
            let tx = db.begin_read().expect("failed");
            tx.rollback().expect("failed");
        });
    });

    // Mixed read/write workload
    group.bench_function("mixed_workload", |b| {
        let db = Database::in_memory().expect("failed");

        // Pre-populate with some data
        {
            let mut tx = db.begin().expect("failed");
            for i in 0..100 {
                let entity = tx.create_entity().expect("failed").with_property("id", i as i64);
                tx.put_entity(&entity).expect("failed");
            }
            tx.commit().expect("failed");
        }

        let mut rng = Rng::new(42);
        let mut write_counter = 100i64;

        b.iter(|| {
            // 80% reads, 20% writes
            if rng.next_f32() < 0.8 {
                // Read
                let tx = db.begin_read().expect("failed");
                let id = EntityId::new((rng.next_u64() % 100) + 1);
                let entity = tx.get_entity(id).expect("failed");
                black_box(entity);
            } else {
                // Write
                let mut tx = db.begin().expect("failed");
                let entity = tx.create_entity().expect("failed").with_property("id", write_counter);
                tx.put_entity(&entity).expect("failed");
                tx.commit().expect("failed");
                write_counter += 1;
            }
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Main
// ============================================================================

criterion_group!(
    benches,
    storage_benchmarks,
    graph_benchmarks,
    vector_benchmarks,
    query_benchmarks
);
criterion_main!(benches);
