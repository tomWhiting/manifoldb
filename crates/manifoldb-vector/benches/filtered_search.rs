//! Benchmarks comparing post-filter vs in-traversal filter performance.
//!
//! This benchmark measures the performance difference between:
//! 1. Post-filtering: Search for N * (1/selectivity) results, then filter
//! 2. In-traversal filtering: Apply filter during HNSW graph traversal
//!
//! The in-traversal approach should be significantly faster for selective filters.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use manifoldb_core::EntityId;
use manifoldb_storage::backends::RedbEngine;
use manifoldb_vector::distance::DistanceMetric;
use manifoldb_vector::index::{HnswConfig, HnswIndex, SearchResult, VectorIndex};
use manifoldb_vector::types::Embedding;
use rand::Rng;

/// Create a random embedding of the given dimension.
fn random_embedding(dim: usize) -> Embedding {
    let mut rng = rand::thread_rng();
    let values: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    Embedding::new(values).unwrap()
}

/// Build an index with N embeddings.
fn build_index(n: usize, dim: usize) -> HnswIndex<RedbEngine> {
    let engine = RedbEngine::in_memory().unwrap();
    let config = HnswConfig::default();
    let mut index =
        HnswIndex::new(engine, "bench", dim, DistanceMetric::Euclidean, config).unwrap();

    for i in 0..n {
        let embedding = random_embedding(dim);
        index.insert(EntityId::new(i as u64), &embedding).unwrap();
    }

    index
}

/// Simulate post-filter approach: search for more results, then filter.
fn post_filter_search(
    index: &HnswIndex<RedbEngine>,
    query: &Embedding,
    k: usize,
    predicate: impl Fn(EntityId) -> bool,
    selectivity: f32,
) -> Vec<SearchResult> {
    // Search for more results to account for filtering
    let oversample = ((k as f32) / selectivity).ceil() as usize;
    let oversample = oversample.min(1000).max(k); // Cap at 1000

    let results = index.search(query, oversample, None).unwrap();

    // Post-filter the results
    results.into_iter().filter(|r| predicate(r.entity_id)).take(k).collect()
}

/// In-traversal filter: apply filter during graph traversal.
fn intraversal_filter_search(
    index: &HnswIndex<RedbEngine>,
    query: &Embedding,
    k: usize,
    predicate: impl Fn(EntityId) -> bool,
) -> Vec<SearchResult> {
    index.search_with_filter(query, k, predicate, None, None).unwrap()
}

fn bench_filtered_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_search");

    // Test parameters
    let n = 10_000; // Number of vectors
    let dim = 128; // Vector dimension
    let k = 10; // Number of results to return

    // Selectivity levels to test
    let selectivities = [0.5, 0.1, 0.01];

    // Build the index once
    let index = build_index(n, dim);
    let query = random_embedding(dim);

    for selectivity in selectivities {
        // Create a predicate that passes approximately `selectivity` fraction of entities
        let threshold = (n as f32 * selectivity) as u64;
        let predicate = move |id: EntityId| id.as_u64() < threshold;

        group.bench_with_input(
            BenchmarkId::new("post_filter", format!("{:.0}%", selectivity * 100.0)),
            &selectivity,
            |b, &sel| {
                b.iter(|| {
                    black_box(post_filter_search(&index, &query, k, predicate, sel));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("intraversal_filter", format!("{:.0}%", selectivity * 100.0)),
            &selectivity,
            |b, _| {
                b.iter(|| {
                    black_box(intraversal_filter_search(&index, &query, k, predicate));
                });
            },
        );
    }

    group.finish();
}

fn bench_filter_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_quality");

    // Test parameters
    let n = 10_000;
    let dim = 128;
    let k = 10;

    // Build the index
    let index = build_index(n, dim);
    let query = random_embedding(dim);

    // Test with very selective filter (1%)
    let selectivity = 0.01;
    let threshold = (n as f32 * selectivity) as u64;
    let predicate = move |id: EntityId| id.as_u64() < threshold;

    // Compare the number of results returned
    group.bench_function("post_filter_1pct", |b| {
        b.iter(|| {
            let results = black_box(post_filter_search(&index, &query, k, predicate, selectivity));
            // May return fewer than k if not enough matching in oversample
            black_box(results.len())
        });
    });

    group.bench_function("intraversal_1pct", |b| {
        b.iter(|| {
            let results = black_box(intraversal_filter_search(&index, &query, k, predicate));
            // Should always return k results if k matching exist
            black_box(results.len())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_filtered_search, bench_filter_quality);
criterion_main!(benches);
