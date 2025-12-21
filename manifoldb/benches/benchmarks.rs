//! ManifoldDB benchmarks.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder_benchmark(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // TODO: Add actual benchmarks
            std::hint::black_box(1 + 1)
        });
    });
}

criterion_group!(benches, placeholder_benchmark);
criterion_main!(benches);
