//! Benchmarks for vector distance calculations.
//!
//! Run with: `cargo bench -p manifoldb-vector`
//!
//! Compare SIMD vs scalar: `cargo bench -p manifoldb-vector --features scalar`

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use manifoldb_vector::distance::{
    cosine_distance, cosine_similarity, cosine_similarity_with_norms, dot_product,
    euclidean_distance, euclidean_distance_squared, CachedNorm,
};
use rand::Rng;

/// Generate a random vector of the specified dimension.
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Benchmark Euclidean distance across different dimensions.
fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");

    // Test common embedding dimensions
    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

/// Benchmark squared Euclidean distance (avoids sqrt).
fn bench_euclidean_squared(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance_squared");

    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance_squared(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

/// Benchmark dot product across different dimensions.
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| dot_product(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

/// Benchmark cosine similarity (computes norms each time).
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cosine_similarity(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

/// Benchmark cosine similarity with cached norms.
fn bench_cosine_with_cached_norms(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_cached_norms");

    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);
        let norm_a = CachedNorm::new(&a);
        let norm_b = CachedNorm::new(&b);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| {
                cosine_similarity_with_norms(
                    black_box(&a),
                    black_box(&b),
                    norm_a.norm(),
                    norm_b.norm(),
                )
            });
        });
    }

    group.finish();
}

/// Benchmark cosine distance.
fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dim in [128, 384, 768, 1536, 3072] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

/// Benchmark CachedNorm creation.
fn bench_cached_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_norm_creation");

    for dim in [128, 384, 768, 1536, 3072] {
        let v = random_vector(dim);

        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| CachedNorm::new(black_box(&v)));
        });
    }

    group.finish();
}

/// Benchmark a realistic search scenario: one query vs many candidates.
fn bench_search_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_scenario");

    let dim = 1536; // OpenAI embedding dimension
    let num_candidates = 1000;

    let query = random_vector(dim);
    let query_norm = CachedNorm::new(&query);
    let candidates: Vec<Vec<f32>> = (0..num_candidates).map(|_| random_vector(dim)).collect();
    let candidate_norms: Vec<CachedNorm> = candidates.iter().map(|c| CachedNorm::new(c)).collect();

    // Benchmark: compute cosine similarity to all candidates
    group.throughput(Throughput::Elements(num_candidates as u64));

    group.bench_function("cosine_1000_candidates", |bench| {
        bench.iter(|| {
            for (candidate, norm) in candidates.iter().zip(&candidate_norms) {
                black_box(cosine_similarity_with_norms(
                    &query,
                    candidate,
                    query_norm.norm(),
                    norm.norm(),
                ));
            }
        });
    });

    group.bench_function("euclidean_1000_candidates", |bench| {
        bench.iter(|| {
            for candidate in &candidates {
                black_box(euclidean_distance(&query, candidate));
            }
        });
    });

    group.bench_function("dot_product_1000_candidates", |bench| {
        bench.iter(|| {
            for candidate in &candidates {
                black_box(dot_product(&query, candidate));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean_distance,
    bench_euclidean_squared,
    bench_dot_product,
    bench_cosine_similarity,
    bench_cosine_with_cached_norms,
    bench_cosine_distance,
    bench_cached_norm,
    bench_search_scenario,
);

criterion_main!(benches);
