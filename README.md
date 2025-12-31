# ManifoldDB

A multi-paradigm embedded database for Rust that unifies graph, vector, and relational data in a single transactional system.

## Features

- **Graph Database**: Store entities and relationships with property graphs. Traverse with BFS, Dijkstra, A*, and pattern matching.
- **Vector Database**: HNSW-based similarity search with multiple distance metrics (Cosine, Euclidean, Dot Product, Manhattan, Chebyshev, Hamming).
- **SQL Support**: Query using familiar SQL syntax with graph pattern extensions.
- **ACID Transactions**: Full transactional support across all operations with automatic rollback.
- **Graph Analytics**: Built-in PageRank, betweenness centrality, and community detection.
- **Product Quantization**: Optional vector compression for memory-efficient large-scale deployments.
- **Query Caching**: Automatic caching of query results with smart invalidation.

## Installation

Add ManifoldDB to your `Cargo.toml`:

```toml
[dependencies]
manifoldb = "0.1"
```

## Quick Start

### Opening a Database

```rust
use manifoldb::Database;

// Open or create a database file
let db = Database::open("mydb.manifold")?;

// Or create an in-memory database for testing
let db = Database::in_memory()?;
```

### Multi-threaded Usage

`Database` is cheaply cloneable and can be shared across threads:

```rust
use manifoldb::Database;
use std::thread;

let db = Database::open("mydb.manifold")?;

// Clone for another thread (shares underlying engine)
let db_clone = db.clone();

thread::spawn(move || {
    let tx = db_clone.begin_read().unwrap();
    // Read operations...
});
```

### Working with Entities and Edges

```rust
use manifoldb::Database;

let db = Database::in_memory()?;

// Write transaction
let mut tx = db.begin()?;

// Create entities (nodes)
let alice = tx.create_entity()?
    .with_label("Person")
    .with_property("name", "Alice")
    .with_property("age", 30);
let bob = tx.create_entity()?
    .with_label("Person")
    .with_property("name", "Bob");
tx.put_entity(&alice)?;
tx.put_entity(&bob)?;

// Create an edge (relationship)
let follows = tx.create_edge(alice.id, bob.id, "FOLLOWS")?;
tx.put_edge(&follows)?;

tx.commit()?;

// Read transaction
let tx = db.begin_read()?;
if let Some(entity) = tx.get_entity(alice.id)? {
    println!("Found: {:?}", entity.get_property("name"));
}
```

### SQL Queries

```rust
use manifoldb::Database;

let db = Database::open("mydb.manifold")?;

// Execute statements
db.execute("INSERT INTO users (name, age) VALUES ('Alice', 30)")?;

// Query data
let results = db.query("SELECT * FROM users WHERE age > 25")?;
for row in results {
    let name: String = row.get_as(0)?;
    println!("User: {}", name);
}

// Parameterized queries
let results = db.query_with_params(
    "SELECT * FROM users WHERE name = $1",
    &["Alice".into()],
)?;
```

### Vector Similarity Search

```rust
use manifoldb::{Database, DatabaseBuilder, Value};

let db = Database::in_memory()?;

// Store entities with vector embeddings
let mut tx = db.begin()?;
let doc = tx.create_entity()?
    .with_label("Document")
    .with_property("title", "Introduction to ML")
    .with_property("embedding", vec![0.1f32, 0.2, 0.3, 0.4]);
tx.put_entity(&doc)?;
tx.commit()?;

// Vector similarity search using SQL
let query_vector = vec![0.1f32, 0.2, 0.3, 0.4];
let similar = db.query_with_params("
    SELECT * FROM documents
    ORDER BY embedding <-> $1
    LIMIT 10
", &[Value::Vector(query_vector)])?;
```

### Graph Traversal

```rust
use manifoldb_graph::traversal::{
    Expand, ExpandAll, ShortestPath, Dijkstra, Direction
};

// Single-hop: find all friends
let friends = Expand::neighbors(&tx, user_id, Direction::Outgoing)?
    .with_edge_type("FRIEND")
    .collect()?;

// Multi-hop: find users within 3 degrees
let nearby = ExpandAll::new(&tx, user_id, Direction::Outgoing)
    .with_max_depth(3)
    .collect_nodes()?;

// Shortest path (unweighted)
let path = ShortestPath::find(&tx, user_a, user_b, Direction::Both)?;

// Weighted shortest path
let path = Dijkstra::new(city_a, city_b, Direction::Outgoing)
    .with_weight_property("distance")
    .find(&tx)?;
```

### Graph Analytics

```rust
use manifoldb_graph::analytics::{PageRank, PageRankConfig};

// Compute PageRank scores
let config = PageRankConfig::default();
let scores = PageRank::compute(&tx, &config)?;

for (node, score) in scores.iter().take(10) {
    println!("Node {:?}: {:.4}", node, score);
}
```

## Configuration

Use `DatabaseBuilder` for advanced configuration:

```rust
use manifoldb::{DatabaseBuilder, VectorSyncStrategy};
use manifoldb::cache::CacheConfig;
use std::time::Duration;

let db = DatabaseBuilder::new()
    .path("mydb.manifold")
    .cache_size(128 * 1024 * 1024)  // 128MB storage cache
    .vector_sync_strategy(VectorSyncStrategy::Async)
    .query_cache_config(
        CacheConfig::new()
            .max_entries(5000)
            .ttl(Some(Duration::from_secs(600)))
    )
    .open()?;
```

### Vector Sync Strategies

- `Synchronous` - Strong consistency, updates visible immediately (default)
- `Async` - Eventual consistency, faster writes
- `Hybrid` - Adaptive based on batch size

## Architecture

ManifoldDB is organized as a workspace with multiple crates:

```
manifoldb/           # Main database crate and public API
manifoldb-core/      # Core types (EntityId, Value, Entity, Edge)
manifoldb-storage/   # Storage engine traits and Redb backend
manifoldb-graph/     # Graph storage, traversal, and analytics
manifoldb-vector/    # Vector storage, HNSW index, similarity search
manifoldb-query/     # SQL parser and query execution
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## Performance

### HNSW Index Configuration

The HNSW vector index can be tuned for your workload:

```rust
use manifoldb_vector::HnswConfig;

// High recall configuration
let config = HnswConfig::new(32)  // M parameter
    .with_ef_construction(400)    // Build quality
    .with_ef_search(200);         // Search quality

// Memory-efficient with Product Quantization
let config = HnswConfig::new(16)
    .with_pq(8)  // 8 segments for ~8x compression
    .with_pq_centroids(256);
```

Key parameters:
- **M** (16-64): Connections per node. Higher = better recall, more memory.
- **ef_construction** (100-500): Build-time beam width. Higher = better index quality.
- **ef_search** (10-500): Search-time beam width. Higher = better recall, slower search.

See [docs/performance-tuning.md](docs/performance-tuning.md) for detailed tuning guidance.

## Distance Metrics

ManifoldDB supports multiple distance metrics for vector search:

| Metric | Use Case |
|--------|----------|
| Cosine | Text embeddings, normalized vectors |
| Euclidean | General purpose, spatial data |
| Dot Product | Pre-normalized vectors, recommendation |
| Manhattan | Sparse features, grid-based data |
| Chebyshev | Game AI, uniform-cost movement |
| Hamming | Binary embeddings, fingerprints |

## Metrics and Monitoring

```rust
let metrics = db.metrics();
println!("{}", metrics);  // Pretty-printed summary

// Access specific metrics
println!("Queries: {}", metrics.queries.total_queries);
println!("Cache hit rate: {:?}", metrics.cache.as_ref().and_then(|c| c.hit_rate()));
println!("Commits: {}", metrics.transactions.commits);
```

## Documentation

- [Architecture Guide](docs/architecture.md) - How the layers fit together
- [Cookbook](docs/cookbook.md) - Common patterns and recipes
- [Performance Tuning](docs/performance-tuning.md) - HNSW parameters, cache sizing

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
