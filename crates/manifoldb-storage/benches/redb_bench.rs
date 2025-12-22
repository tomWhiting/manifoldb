//! Benchmarks for the Redb storage backend.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use manifoldb_storage::backends::RedbEngine;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

/// Benchmark single key-value writes.
fn bench_put_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("redb_put_single");
    group.throughput(Throughput::Elements(1));

    group.bench_function("put_single", |b| {
        b.iter_batched(
            || RedbEngine::in_memory().unwrap(),
            |engine| {
                let mut tx = engine.begin_write().unwrap();
                tx.put("bench", b"key", b"value").unwrap();
                tx.commit().unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark batch writes.
fn bench_put_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("redb_put_batch");

    for size in [10, 100, 1000] {
        group.throughput(Throughput::Elements(size));
        group.bench_function(format!("put_batch_{size}"), |b| {
            b.iter_batched(
                || RedbEngine::in_memory().unwrap(),
                |engine| {
                    let mut tx = engine.begin_write().unwrap();
                    for i in 0..size {
                        let key = format!("key:{i:05}");
                        let value = format!("value:{i:05}");
                        tx.put("bench", key.as_bytes(), value.as_bytes()).unwrap();
                    }
                    tx.commit().unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark single key reads.
fn bench_get_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("redb_get_single");
    group.throughput(Throughput::Elements(1));

    group.bench_function("get_single", |b| {
        b.iter_batched(
            || {
                let engine = RedbEngine::in_memory().unwrap();
                {
                    let mut tx = engine.begin_write().unwrap();
                    tx.put("bench", b"key", b"value").unwrap();
                    tx.commit().unwrap();
                }
                engine
            },
            |engine| {
                let tx = engine.begin_read().unwrap();
                let _ = black_box(tx.get("bench", b"key").unwrap());
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark random reads from a populated database.
fn bench_get_random(c: &mut Criterion) {
    const NUM_KEYS: u64 = 10000;
    let mut group = c.benchmark_group("redb_get_random");
    group.throughput(Throughput::Elements(100));

    group.bench_function("get_random_100", |b| {
        b.iter_batched(
            || {
                let engine = RedbEngine::in_memory().unwrap();
                {
                    let mut tx = engine.begin_write().unwrap();
                    for i in 0..NUM_KEYS {
                        let key = format!("key:{i:05}");
                        let value = format!("value:{i:05}");
                        tx.put("bench", key.as_bytes(), value.as_bytes()).unwrap();
                    }
                    tx.commit().unwrap();
                }
                engine
            },
            |engine| {
                let tx = engine.begin_read().unwrap();
                for i in (0..100).map(|x| x * 97 % NUM_KEYS) {
                    let key = format!("key:{i:05}");
                    let _ = black_box(tx.get("bench", key.as_bytes()).unwrap());
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark cursor iteration.
fn bench_cursor_iterate(c: &mut Criterion) {
    let mut group = c.benchmark_group("redb_cursor_iterate");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size));
        group.bench_function(format!("cursor_iterate_{size}"), |b| {
            b.iter_batched(
                || {
                    let engine = RedbEngine::in_memory().unwrap();
                    {
                        let mut tx = engine.begin_write().unwrap();
                        for i in 0..size {
                            let key = format!("key:{i:05}");
                            let value = format!("value:{i:05}");
                            tx.put("bench", key.as_bytes(), value.as_bytes()).unwrap();
                        }
                        tx.commit().unwrap();
                    }
                    engine
                },
                |engine| {
                    let tx = engine.begin_read().unwrap();
                    let mut cursor = tx.cursor("bench").unwrap();
                    let mut count = 0u64;
                    cursor.seek_first().unwrap();
                    while cursor.current().is_some() {
                        count += 1;
                        if cursor.next().unwrap().is_none() {
                            break;
                        }
                    }
                    black_box(count);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark transaction overhead.
fn bench_transaction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("redb_transaction");

    group.bench_function("begin_read", |b| {
        b.iter_batched(
            || RedbEngine::in_memory().unwrap(),
            |engine| {
                let _tx = black_box(engine.begin_read().unwrap());
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("begin_write_commit_empty", |b| {
        b.iter_batched(
            || RedbEngine::in_memory().unwrap(),
            |engine| {
                let tx = engine.begin_write().unwrap();
                tx.commit().unwrap();
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_put_single,
    bench_put_batch,
    bench_get_single,
    bench_get_random,
    bench_cursor_iterate,
    bench_transaction_overhead,
);

criterion_main!(benches);
