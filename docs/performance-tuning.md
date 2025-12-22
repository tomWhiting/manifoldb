# Performance Tuning Guide

This guide covers configuration and optimization strategies for ManifoldDB.

## Table of Contents

- [HNSW Vector Index](#hnsw-vector-index)
- [Storage Configuration](#storage-configuration)
- [Query Caching](#query-caching)
- [Transaction Strategies](#transaction-strategies)
- [Batch Operations](#batch-operations)
- [Memory Management](#memory-management)
- [Benchmarking](#benchmarking)

## HNSW Vector Index

The HNSW (Hierarchical Navigable Small World) index is the core of vector similarity search. Understanding its parameters is key to balancing recall, speed, and memory usage.

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| M | 16 | 4-64 | Connections per node. Higher = better recall, more memory |
| ef_construction | 200 | 50-500 | Build-time beam width. Higher = better index quality |
| ef_search | 50 | 10-500 | Search-time beam width. Higher = better recall, slower |

### Configuration Presets

```rust
use manifoldb_vector::HnswConfig;

// High Speed (lower recall ~90%)
let config = HnswConfig::new(8)
    .with_ef_construction(100)
    .with_ef_search(20);

// Balanced (good recall ~95%)
let config = HnswConfig::new(16)
    .with_ef_construction(200)
    .with_ef_search(50);

// High Recall (~99%)
let config = HnswConfig::new(32)
    .with_ef_construction(400)
    .with_ef_search(200);

// Maximum Recall (~99.5%)
let config = HnswConfig::new(48)
    .with_ef_construction(500)
    .with_ef_search(400);
```

### M Parameter Guidelines

| Dataset Size | Recommended M | Memory per Vector |
|--------------|---------------|-------------------|
| < 10K | 8-12 | ~256 bytes |
| 10K - 100K | 12-16 | ~512 bytes |
| 100K - 1M | 16-24 | ~768 bytes |
| > 1M | 24-32 | ~1KB |

Memory formula: `~(M * 2 + M_max0) * 8 bytes` per vector for connection data.

### ef_search Tuning

ef_search can be adjusted per query for dynamic quality/speed tradeoff:

```rust
// Quick search for browsing
let results = index.search(&query, 10, Some(SearchConfig { ef: 20 }))?;

// Thorough search for final results
let results = index.search(&query, 10, Some(SearchConfig { ef: 200 }))?;
```

Typical ef_search values:
- **10-30**: Fast browsing, 85-90% recall
- **50-100**: Standard queries, 92-97% recall
- **200+**: High-precision queries, 98%+ recall

### Product Quantization

For memory-constrained deployments, enable PQ compression:

```rust
// Enable PQ with 8 segments
let config = HnswConfig::new(16)
    .with_pq(8)
    .with_pq_centroids(256)
    .with_pq_training_samples(10000);
```

PQ compression ratio depends on segments:

| Original Dimension | Segments | Code Size | Compression Ratio |
|-------------------|----------|-----------|-------------------|
| 128 | 8 | 8 bytes | 64x |
| 256 | 16 | 16 bytes | 64x |
| 384 | 32 | 32 bytes | 48x |
| 768 | 64 | 64 bytes | 48x |

**Trade-off**: PQ reduces recall by 1-5% but dramatically reduces memory usage.

### Distance Metric Selection

| Metric | When to Use | Normalized Vectors? |
|--------|-------------|---------------------|
| Cosine | Text embeddings (BERT, OpenAI) | No (auto-normalizes) |
| Dot Product | Pre-normalized vectors, speed | Yes (required) |
| Euclidean | General purpose, spatial | No |
| Manhattan | Sparse features | No |

```rust
use manifoldb_vector::distance::DistanceMetric;

// For text embeddings (most common)
let index = HnswIndex::new(engine, "embeddings", 384, DistanceMetric::Cosine, config)?;

// For pre-normalized vectors (faster)
let index = HnswIndex::new(engine, "embeddings", 384, DistanceMetric::DotProduct, config)?;
```

## Storage Configuration

### Cache Size

The storage cache holds frequently accessed data in memory:

```rust
let db = DatabaseBuilder::new()
    .path("data.manifold")
    .cache_size(256 * 1024 * 1024)  // 256MB
    .open()?;
```

**Guidelines**:
- **Small datasets (< 100MB)**: 64-128MB cache
- **Medium datasets (100MB - 1GB)**: 256-512MB cache
- **Large datasets (> 1GB)**: 1GB+ or match working set size

Ideally, cache should be large enough to hold frequently accessed data (entities, hot indexes).

### Maximum Database Size

Limit database growth if needed:

```rust
let db = DatabaseBuilder::new()
    .path("data.manifold")
    .max_size(10 * 1024 * 1024 * 1024)  // 10GB limit
    .open()?;
```

## Query Caching

### Cache Configuration

```rust
use manifoldb::cache::CacheConfig;
use std::time::Duration;

let config = CacheConfig::new()
    .max_entries(10000)           // Max cached queries
    .max_size(100 * 1024 * 1024)  // 100MB max cache size
    .ttl(Some(Duration::from_secs(300)));  // 5 minute TTL

let db = DatabaseBuilder::new()
    .path("data.manifold")
    .query_cache_config(config)
    .open()?;
```

### When to Use Cache Hints

```rust
// CACHE hint: Force caching for complex aggregations
let results = db.query("/*+ CACHE */ SELECT category, COUNT(*) FROM products GROUP BY category")?;

// NO_CACHE hint: Skip cache for real-time data
let results = db.query("/*+ NO_CACHE */ SELECT * FROM live_events ORDER BY timestamp DESC")?;
```

### Cache Monitoring

```rust
let metrics = db.cache_metrics();
let hit_rate = metrics.hit_rate().unwrap_or(0.0);

if hit_rate < 50.0 {
    // Consider:
    // 1. Increasing cache size
    // 2. Adjusting TTL
    // 3. Using CACHE hints on hot queries
    println!("Warning: Low cache hit rate: {:.1}%", hit_rate);
}
```

## Transaction Strategies

### Vector Sync Strategies

```rust
use manifoldb::{DatabaseBuilder, VectorSyncStrategy};

// Synchronous (default): Strong consistency
let db = DatabaseBuilder::new()
    .path("data.manifold")
    .vector_sync_strategy(VectorSyncStrategy::Synchronous)
    .open()?;

// Async: Faster writes, eventual consistency
let db = DatabaseBuilder::new()
    .path("data.manifold")
    .vector_sync_strategy(VectorSyncStrategy::Async)
    .open()?;

// Hybrid: Adaptive based on batch size
let db = DatabaseBuilder::new()
    .path("data.manifold")
    .vector_sync_strategy(VectorSyncStrategy::Hybrid)
    .open()?;
```

**When to use each**:
- **Synchronous**: When vector search must immediately see new data
- **Async**: High write throughput, can tolerate brief delay before vectors are searchable
- **Hybrid**: Automatic selection based on operation size

### Transaction Sizing

Keep transactions focused:

```rust
// Good: Small, focused transactions
for batch in data.chunks(1000) {
    let mut tx = db.begin()?;
    tx.put_entities_batch(batch)?;
    tx.commit()?;
}

// Avoid: Very large transactions (high memory, long locks)
let mut tx = db.begin()?;
for item in all_million_items {  // Too large!
    tx.put_entity(&item)?;
}
tx.commit()?;
```

### Read-Heavy Workloads

For read-heavy workloads, prefer read transactions:

```rust
// Multiple concurrent reads
let handles: Vec<_> = (0..num_threads)
    .map(|_| {
        let db = db.clone();  // Database is Clone + Send + Sync
        std::thread::spawn(move || {
            let tx = db.begin_read().unwrap();
            // Read operations...
        })
    })
    .collect();
```

## Batch Operations

### Entity Batch Loading

```rust
// Efficient: Batch insert
let entities: Vec<Entity> = generate_entities(10000);
let mut tx = db.begin()?;
tx.put_entities_batch(&entities)?;  // Single batch operation
tx.commit()?;

// Inefficient: Individual inserts
let mut tx = db.begin()?;
for entity in &entities {
    tx.put_entity(entity)?;  // Many small operations
}
tx.commit()?;
```

**Expected speedup**: 5-10x for batch vs individual operations.

### Edge Batch Loading

```rust
let edges: Vec<Edge> = generate_edges(100000);
let mut tx = db.begin()?;
tx.put_edges_batch(&edges)?;
tx.commit()?;
```

### Optimal Batch Sizes

| Data Type | Recommended Batch Size | Commit Frequency |
|-----------|----------------------|------------------|
| Entities | 1,000 - 10,000 | Per batch |
| Edges | 5,000 - 50,000 | Per batch |
| Vectors (HNSW) | 100 - 1,000 | Per batch |

## Memory Management

### Estimating Memory Requirements

```
Total Memory = Storage Cache + Query Cache + HNSW Index + Working Set

Storage Cache: Configurable (64MB - 1GB typical)
Query Cache: Configurable (10MB - 100MB typical)
HNSW Index: (num_vectors * dimension * 4) + (num_vectors * M * 16) bytes
Working Set: Depends on query patterns
```

**Example for 1M 384-dim vectors with M=16**:
- Vector data: 1M * 384 * 4 = 1.5GB
- Graph structure: 1M * 16 * 16 = 256MB
- With PQ (8 segments): Vectors reduced to 1M * 8 = 8MB
- Total without PQ: ~1.8GB
- Total with PQ: ~300MB

### Reducing Memory Usage

1. **Enable Product Quantization**:
   ```rust
   let config = HnswConfig::new(16).with_pq(8);
   ```

2. **Lower M parameter**:
   ```rust
   let config = HnswConfig::new(12);  // vs default 16
   ```

3. **Reduce cache sizes**:
   ```rust
   let db = DatabaseBuilder::new()
       .cache_size(32 * 1024 * 1024)  // 32MB instead of 64MB
       .query_cache_config(CacheConfig::new().max_entries(1000))
       .open()?;
   ```

4. **Use on-demand embedding loading** instead of keeping all in memory

## Benchmarking

### Built-in Metrics

```rust
// Run workload
for _ in 0..1000 {
    db.query("SELECT * FROM users WHERE id = 1")?;
}

// Check metrics
let metrics = db.metrics();
println!("Total queries: {}", metrics.queries.total_queries);
println!("Avg query time: {:?}", metrics.queries.avg_duration);
println!("Cache hit rate: {:?}", metrics.cache.as_ref().and_then(|c| c.hit_rate()));
```

### Benchmarking Vector Search

```rust
use std::time::Instant;

fn benchmark_search(index: &HnswIndex, queries: &[Embedding], k: usize) {
    let start = Instant::now();
    let mut total_results = 0;

    for query in queries {
        let results = index.search(query, k, None).unwrap();
        total_results += results.len();
    }

    let elapsed = start.elapsed();
    let qps = queries.len() as f64 / elapsed.as_secs_f64();

    println!("Queries: {}", queries.len());
    println!("Total time: {:?}", elapsed);
    println!("QPS: {:.1}", qps);
    println!("Avg results: {:.1}", total_results as f64 / queries.len() as f64);
}
```

### Measuring Recall

```rust
fn measure_recall(
    index: &HnswIndex,
    queries: &[Embedding],
    ground_truth: &[Vec<EntityId>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;

    for (query, truth) in queries.iter().zip(ground_truth) {
        let results = index.search(query, k, None).unwrap();
        let result_ids: HashSet<_> = results.iter().map(|r| r.entity_id).collect();
        let truth_set: HashSet<_> = truth.iter().take(k).copied().collect();

        let intersection = result_ids.intersection(&truth_set).count();
        total_recall += intersection as f64 / k as f64;
    }

    total_recall / queries.len() as f64
}
```

### Performance Checklist

1. **Index Configuration**
   - [ ] M parameter appropriate for dataset size
   - [ ] ef_construction high enough for quality
   - [ ] ef_search tuned for recall/speed tradeoff
   - [ ] PQ enabled if memory constrained

2. **Storage**
   - [ ] Cache size matches working set
   - [ ] Database on SSD if possible
   - [ ] Query cache enabled for repeat queries

3. **Transactions**
   - [ ] Using batch operations for bulk loads
   - [ ] Transaction size reasonable (not too large)
   - [ ] Vector sync strategy matches consistency needs

4. **Queries**
   - [ ] Cache hints on hot queries
   - [ ] Parameterized queries for cache efficiency
   - [ ] Appropriate LIMIT clauses

5. **Monitoring**
   - [ ] Tracking cache hit rates
   - [ ] Monitoring query latencies
   - [ ] Watching memory usage
