//! ManifoldDB benchmarks.
//!
//! Comprehensive benchmarks covering:
//! - Storage layer operations (entity/edge CRUD)
//! - Graph traversal operations
//! - Vector search operations
//! - Query execution

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use manifoldb::{BatchWriter, BatchWriterConfig, Database, EntityId};
use manifoldb_core::EntityId as CoreEntityId;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_vector::distance::DistanceMetric;
use manifoldb_vector::ops::{ExactKnn, VectorOperator};
use manifoldb_vector::store::VectorStore;
use manifoldb_vector::types::{Embedding, EmbeddingName, EmbeddingSpace};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

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
// Wide Graph Pattern Matching Benchmarks
// ============================================================================

fn wide_graph_benchmarks(c: &mut Criterion) {
    use manifoldb_graph::store::{EdgeStore, IdGenerator, NodeStore};
    use manifoldb_graph::traversal::{AllShortestPaths, Direction, PathPattern, PathStep};
    use manifoldb_storage::backends::RedbEngine;
    use manifoldb_storage::{StorageEngine, Transaction as StorageTransaction};

    let mut group = c.benchmark_group("wide_graph");

    // Setup wide graph with high fanout (100+ neighbors per node)
    // Structure: center -> 100 level1 nodes -> 10 level2 nodes each
    let wide_engine = RedbEngine::in_memory().expect("failed");
    let (center_id, _level1_ids, level2_ids) = {
        let id_gen = IdGenerator::new();
        let mut tx = wide_engine.begin_write().expect("failed");

        // Create center node
        let center = NodeStore::create(&mut tx, &id_gen, |id| {
            manifoldb_core::Entity::new(id).with_label("Center")
        })
        .expect("failed");

        // Create 100 level1 nodes connected to center
        let mut level1 = Vec::with_capacity(100);
        for _ in 0..100 {
            let node = NodeStore::create(&mut tx, &id_gen, |id| {
                manifoldb_core::Entity::new(id).with_property("level", 1i64)
            })
            .expect("failed");
            level1.push(node.id);

            EdgeStore::create(&mut tx, &id_gen, center.id, node.id, "CONNECTS", |id| {
                manifoldb_core::Edge::new(id, center.id, node.id, "CONNECTS")
            })
            .expect("failed");
        }

        // Create 10 level2 nodes per level1 node (1000 total)
        let mut level2 = Vec::with_capacity(1000);
        for &l1_id in &level1 {
            for _ in 0..10 {
                let node = NodeStore::create(&mut tx, &id_gen, |id| {
                    manifoldb_core::Entity::new(id).with_property("level", 2i64)
                })
                .expect("failed");
                level2.push(node.id);

                EdgeStore::create(&mut tx, &id_gen, l1_id, node.id, "CONNECTS", |id| {
                    manifoldb_core::Edge::new(id, l1_id, node.id, "CONNECTS")
                })
                .expect("failed");
            }
        }

        tx.commit().expect("failed");
        (center.id, level1, level2)
    };

    // Variable-length pattern matching on wide graph: (center)-[:CONNECTS*1..2]->(target)
    group.bench_function("pattern_variable_length_fanout_100", |b| {
        b.iter(|| {
            let tx = wide_engine.begin_read().expect("failed");
            let pattern = PathPattern::new()
                .add_step(PathStep::outgoing("CONNECTS").variable_length(1, 2))
                .with_limit(100); // Limit to avoid measuring result collection

            let matches = pattern.find_from(&tx, center_id).expect("failed");
            black_box(matches.len())
        });
    });

    // Multi-step pattern: (center)-[:CONNECTS]->(l1)-[:CONNECTS]->(l2)
    group.bench_function("pattern_multi_step_fanout_100", |b| {
        b.iter(|| {
            let tx = wide_engine.begin_read().expect("failed");
            let pattern = PathPattern::new()
                .add_step(PathStep::outgoing("CONNECTS"))
                .add_step(PathStep::outgoing("CONNECTS"))
                .with_limit(100);

            let matches = pattern.find_from(&tx, center_id).expect("failed");
            black_box(matches.len())
        });
    });

    // All shortest paths in wide graph
    group.bench_function("all_shortest_paths_wide", |b| {
        // Find paths from center to a level2 node
        let target = level2_ids[500]; // Pick a node in the middle

        b.iter(|| {
            let tx = wide_engine.begin_read().expect("failed");
            let paths = AllShortestPaths::new(center_id, target, Direction::Outgoing)
                .find(&tx)
                .expect("failed");
            black_box(paths.len())
        });
    });

    // Setup very wide graph (fanout 500+)
    let very_wide_engine = RedbEngine::in_memory().expect("failed");
    let very_wide_center = {
        let id_gen = IdGenerator::new();
        let mut tx = very_wide_engine.begin_write().expect("failed");

        let center = NodeStore::create(&mut tx, &id_gen, |id| {
            manifoldb_core::Entity::new(id).with_label("Center")
        })
        .expect("failed");

        // Create 500 directly connected nodes
        for i in 0..500 {
            let node = NodeStore::create(&mut tx, &id_gen, |id| {
                manifoldb_core::Entity::new(id).with_property("index", i as i64)
            })
            .expect("failed");

            EdgeStore::create(&mut tx, &id_gen, center.id, node.id, "LINKS", |id| {
                manifoldb_core::Edge::new(id, center.id, node.id, "LINKS")
            })
            .expect("failed");
        }

        tx.commit().expect("failed");
        center.id
    };

    // Pattern matching with 500 fanout
    group.bench_function("pattern_fanout_500", |b| {
        b.iter(|| {
            let tx = very_wide_engine.begin_read().expect("failed");
            let pattern = PathPattern::new().add_step(PathStep::outgoing("LINKS"));

            let matches = pattern.find_from(&tx, very_wide_center).expect("failed");
            black_box(matches.len())
        });
    });

    // Variable length with 500 fanout
    group.bench_function("pattern_variable_fanout_500", |b| {
        b.iter(|| {
            let tx = very_wide_engine.begin_read().expect("failed");
            let pattern = PathPattern::new()
                .add_step(PathStep::outgoing("LINKS").variable_length(0, 1))
                .with_limit(100);

            let matches = pattern.find_from(&tx, very_wide_center).expect("failed");
            black_box(matches.len())
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
            DistanceMetric::Manhattan => "manhattan",
            DistanceMetric::Chebyshev => "chebyshev",
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
// Write Batching Benchmarks
// ============================================================================

fn write_batching_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_batching");

    // Compare batched vs immediate writes (sequential)
    for count in [100, 500, 1000] {
        // Immediate commit (no batching)
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("immediate_sequential", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(engine, BatchWriterConfig::disabled())
                    },
                    |writer| {
                        for i in 0..count {
                            let mut tx = writer.begin();
                            let key = format!("key_{i}");
                            let value = format!("value_{i}");
                            tx.put("test", key.as_bytes(), value.as_bytes()).expect("put failed");
                            tx.commit().expect("commit failed");
                        }
                        black_box(writer)
                    },
                );
            },
        );

        // Batched writes
        group.bench_with_input(
            BenchmarkId::new("batched_sequential", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(
                            engine,
                            BatchWriterConfig::new()
                                .max_batch_size(50)
                                .flush_interval(Duration::from_millis(5)),
                        )
                    },
                    |writer| {
                        for i in 0..count {
                            let mut tx = writer.begin();
                            let key = format!("key_{i}");
                            let value = format!("value_{i}");
                            tx.put("test", key.as_bytes(), value.as_bytes()).expect("put failed");
                            tx.commit().expect("commit failed");
                        }
                        writer.flush().expect("flush failed");
                        black_box(writer)
                    },
                );
            },
        );
    }

    // Concurrent write benchmark
    for num_threads in [2, 4, 8] {
        let writes_per_thread = 100;

        // Immediate commit with concurrent writers
        group.throughput(Throughput::Elements(num_threads * writes_per_thread));
        group.bench_with_input(
            BenchmarkId::new("immediate_concurrent", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(engine, BatchWriterConfig::disabled())
                    },
                    |writer| {
                        let handles: Vec<_> = (0..num_threads)
                            .map(|t| {
                                let writer = writer.clone();
                                thread::spawn(move || {
                                    for i in 0..writes_per_thread {
                                        let mut tx = writer.begin();
                                        let key = format!("t{t}_key_{i}");
                                        let value = format!("value_{i}");
                                        tx.put("test", key.as_bytes(), value.as_bytes())
                                            .expect("put failed");
                                        tx.commit().expect("commit failed");
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().expect("thread panicked");
                        }
                        black_box(writer)
                    },
                );
            },
        );

        // Batched concurrent writes
        group.bench_with_input(
            BenchmarkId::new("batched_concurrent", num_threads),
            &num_threads,
            |b, &num_threads| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(
                            engine,
                            BatchWriterConfig::new()
                                .max_batch_size(25)
                                .flush_interval(Duration::from_millis(2)),
                        )
                    },
                    |writer| {
                        let handles: Vec<_> = (0..num_threads)
                            .map(|t| {
                                let writer = writer.clone();
                                thread::spawn(move || {
                                    for i in 0..writes_per_thread {
                                        let mut tx = writer.begin();
                                        let key = format!("t{t}_key_{i}");
                                        let value = format!("value_{i}");
                                        tx.put("test", key.as_bytes(), value.as_bytes())
                                            .expect("put failed");
                                        tx.commit().expect("commit failed");
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().expect("thread panicked");
                        }
                        writer.flush().expect("flush failed");
                        black_box(writer)
                    },
                );
            },
        );
    }

    // TRUE BATCH: Multiple writes in single transaction, one commit
    // This is the correct way to batch writes for maximum throughput
    for count in [100, 500, 1000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("true_batch_single_tx", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(engine, BatchWriterConfig::disabled())
                    },
                    |writer| {
                        // ONE transaction with ALL writes
                        let mut tx = writer.begin();
                        for i in 0..count {
                            let key = format!("key_{i}");
                            let value = format!("value_{i}");
                            tx.put("test", key.as_bytes(), value.as_bytes()).expect("put failed");
                        }
                        tx.commit().expect("commit failed"); // ONE commit!
                        black_box(writer)
                    },
                );
            },
        );
    }

    // Batch size tuning benchmark
    for batch_size in [10, 25, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("batch_size_tuning", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter_with_setup(
                    || {
                        let engine = Arc::new(RedbEngine::in_memory().expect("failed"));
                        BatchWriter::new(
                            engine,
                            BatchWriterConfig::new()
                                .max_batch_size(batch_size)
                                .flush_interval(Duration::from_millis(50)),
                        )
                    },
                    |writer| {
                        let num_threads = 4;
                        let writes_per_thread = 100;

                        let handles: Vec<_> = (0..num_threads)
                            .map(|t| {
                                let writer = writer.clone();
                                thread::spawn(move || {
                                    for i in 0..writes_per_thread {
                                        let mut tx = writer.begin();
                                        let key = format!("t{t}_key_{i}");
                                        let value = format!("value_{i}");
                                        tx.put("test", key.as_bytes(), value.as_bytes())
                                            .expect("put failed");
                                        tx.commit().expect("commit failed");
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().expect("thread panicked");
                        }
                        writer.flush().expect("flush failed");
                        black_box(writer)
                    },
                );
            },
        );
    }

    group.finish();
}

// ============================================================================
// Bulk Insert Benchmarks
// ============================================================================

fn bulk_insert_benchmarks(c: &mut Criterion) {
    use manifoldb_core::Entity;

    let mut group = c.benchmark_group("bulk_insert");

    // Compare individual inserts vs bulk insert
    for count in [100, 1000, 10_000] {
        group.throughput(Throughput::Elements(count));

        // Individual inserts (one transaction per entity)
        group.bench_with_input(
            BenchmarkId::new("individual_inserts", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || Database::in_memory().expect("failed to create db"),
                    |db| {
                        for i in 0..count {
                            let mut tx = db.begin().expect("failed");
                            let entity = tx
                                .create_entity()
                                .expect("failed")
                                .with_label("Item")
                                .with_property("index", i as i64);
                            tx.put_entity(&entity).expect("failed");
                            tx.commit().expect("failed");
                        }
                        black_box(db)
                    },
                );
            },
        );

        // Individual inserts in single transaction (baseline for bulk)
        group.bench_with_input(
            BenchmarkId::new("single_tx_inserts", count),
            &count,
            |b, &count| {
                b.iter_with_setup(
                    || Database::in_memory().expect("failed to create db"),
                    |db| {
                        let mut tx = db.begin().expect("failed");
                        for i in 0..count {
                            let entity = tx
                                .create_entity()
                                .expect("failed")
                                .with_label("Item")
                                .with_property("index", i as i64);
                            tx.put_entity(&entity).expect("failed");
                        }
                        tx.commit().expect("failed");
                        black_box(db)
                    },
                );
            },
        );

        // Bulk insert (parallel serialization + single transaction)
        group.bench_with_input(BenchmarkId::new("bulk_insert", count), &count, |b, &count| {
            b.iter_with_setup(
                || {
                    let db = Database::in_memory().expect("failed to create db");
                    let entities: Vec<Entity> = (0..count)
                        .map(|i| {
                            Entity::new(EntityId::new(0))
                                .with_label("Item")
                                .with_property("index", i as i64)
                        })
                        .collect();
                    (db, entities)
                },
                |(db, entities)| {
                    let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");
                    black_box((db, ids))
                },
            );
        });
    }

    // Bulk insert with varying entity sizes (more properties)
    for num_props in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("bulk_insert_props", num_props),
            &num_props,
            |b, &num_props| {
                let count = 1000;
                b.iter_with_setup(
                    || {
                        let db = Database::in_memory().expect("failed to create db");
                        let entities: Vec<Entity> = (0..count)
                            .map(|i| {
                                let mut entity = Entity::new(EntityId::new(0)).with_label("Item");
                                for p in 0..num_props {
                                    entity = entity.with_property(
                                        format!("prop_{}", p),
                                        format!("value_{}_{}", i, p),
                                    );
                                }
                                entity
                            })
                            .collect();
                        (db, entities)
                    },
                    |(db, entities)| {
                        let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");
                        black_box((db, ids))
                    },
                );
            },
        );
    }

    // Very large bulk insert (stress test)
    group.sample_size(10); // Fewer samples for large tests
    group.throughput(Throughput::Elements(50_000));
    group.bench_function("bulk_insert_50k", |b| {
        b.iter_with_setup(
            || {
                let db = Database::in_memory().expect("failed to create db");
                let entities: Vec<Entity> = (0..50_000)
                    .map(|i| {
                        Entity::new(EntityId::new(0))
                            .with_label("Item")
                            .with_property("index", i as i64)
                            .with_property("data", format!("item_{}", i))
                    })
                    .collect();
                (db, entities)
            },
            |(db, entities)| {
                let ids = db.bulk_insert_entities(&entities).expect("bulk insert failed");
                black_box((db, ids))
            },
        );
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
    wide_graph_benchmarks,
    vector_benchmarks,
    query_benchmarks,
    write_batching_benchmarks,
    bulk_insert_benchmarks
);
criterion_main!(benches);
