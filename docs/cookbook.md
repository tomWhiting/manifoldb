# ManifoldDB Cookbook

Common patterns and recipes for working with ManifoldDB.

## Table of Contents

- [Getting Started](#getting-started)
- [Entity and Edge Operations](#entity-and-edge-operations)
- [Graph Traversal](#graph-traversal)
- [Vector Search](#vector-search)
- [Hybrid Queries](#hybrid-queries)
- [Batch Operations](#batch-operations)
- [Query Caching](#query-caching)
- [Error Handling](#error-handling)
- [Testing](#testing)

## Getting Started

### Opening a Database

```rust
use manifoldb::{Database, DatabaseBuilder};

// Simple: open or create at path
let db = Database::open("data/myapp.manifold")?;

// Configured: use builder for options
let db = DatabaseBuilder::new()
    .path("data/myapp.manifold")
    .cache_size(256 * 1024 * 1024)  // 256MB
    .create_if_missing(true)
    .open()?;

// Testing: in-memory database
let db = Database::in_memory()?;
```

### Basic Transaction Pattern

```rust
use manifoldb::Database;

let db = Database::in_memory()?;

// Write transaction - must be committed
let mut tx = db.begin()?;
// ... make changes ...
tx.commit()?;  // Changes are durable

// Read transaction - no commit needed
let tx = db.begin_read()?;
// ... read data ...
// Transaction auto-closes on drop

// Error handling pattern
let mut tx = db.begin()?;
match do_work(&mut tx) {
    Ok(_) => tx.commit()?,
    Err(e) => {
        tx.rollback()?;  // Explicit rollback (also happens on drop)
        return Err(e);
    }
}
```

## Entity and Edge Operations

### Creating Entities with Properties

```rust
let mut tx = db.begin()?;

// Create with builder pattern
let user = tx.create_entity()?
    .with_label("User")
    .with_label("Premium")  // Multiple labels
    .with_property("name", "Alice")
    .with_property("email", "alice@example.com")
    .with_property("age", 30)
    .with_property("verified", true);

tx.put_entity(&user)?;
tx.commit()?;
```

### Creating Edges with Properties

```rust
let mut tx = db.begin()?;

// Create basic edge
let follows = tx.create_edge(alice.id, bob.id, "FOLLOWS")?;
tx.put_edge(&follows)?;

// Create edge with properties
let rated = tx.create_edge(user.id, movie.id, "RATED")?
    .with_property("score", 4.5)
    .with_property("timestamp", "2024-01-15");
tx.put_edge(&rated)?;

tx.commit()?;
```

### Querying Entities

```rust
let tx = db.begin_read()?;

// Get by ID
if let Some(entity) = tx.get_entity(user_id)? {
    println!("Name: {:?}", entity.get_property("name"));
    println!("Has User label: {}", entity.has_label("User"));
}

// Iterate all entities with a label
let users = tx.iter_entities(Some("User"))?;
for user in users {
    println!("User: {:?}", user.id);
}

// Count entities
let user_count = tx.count_entities(Some("User"))?;
```

### Updating Entities

```rust
let mut tx = db.begin()?;

// Fetch, modify, save
if let Some(mut user) = tx.get_entity(user_id)? {
    // Update property
    user.set_property("verified", true);

    // Add new property
    user.set_property("last_login", "2024-01-15");

    // Remove property
    user.remove_property("temp_token");

    // Save changes
    tx.put_entity(&user)?;
}

tx.commit()?;
```

### Deleting Data

```rust
let mut tx = db.begin()?;

// Delete entity (note: doesn't auto-delete connected edges)
let deleted = tx.delete_entity(user_id)?;

// Delete edge
let deleted = tx.delete_edge(follows_id)?;

// Clean up: delete all edges for an entity first
for edge in tx.get_outgoing_edges(user_id)? {
    tx.delete_edge(edge.id)?;
}
for edge in tx.get_incoming_edges(user_id)? {
    tx.delete_edge(edge.id)?;
}
tx.delete_entity(user_id)?;

tx.commit()?;
```

## Graph Traversal

### Finding Neighbors

```rust
use manifoldb_graph::traversal::{Expand, Direction};

let tx = db.begin_read()?;

// All outgoing neighbors
let following = Expand::neighbors(tx.storage_ref()?, user_id, Direction::Outgoing)?
    .collect()?;

// Filter by edge type
let friends = Expand::neighbors(tx.storage_ref()?, user_id, Direction::Both)?
    .with_edge_type("FRIEND")
    .collect()?;

// Get edges instead of nodes
let friendships = Expand::edges(tx.storage_ref()?, user_id, Direction::Outgoing)?
    .with_edge_type("FRIEND")
    .collect()?;
```

### Multi-Hop Traversal

```rust
use manifoldb_graph::traversal::{ExpandAll, Direction};

let tx = db.begin_read()?;

// Find all nodes within 3 hops
let nodes = ExpandAll::new(tx.storage_ref()?, start_id, Direction::Outgoing)
    .with_max_depth(3)
    .collect_nodes()?;

// Find nodes with depth info
let nodes_with_depth = ExpandAll::new(tx.storage_ref()?, start_id, Direction::Both)
    .with_max_depth(2)
    .collect()?;

for node in nodes_with_depth {
    println!("Node {:?} at depth {}", node.id, node.depth);
}

// Limit results
let top_10 = ExpandAll::new(tx.storage_ref()?, start_id, Direction::Outgoing)
    .with_max_depth(5)
    .with_limit(10)
    .collect_nodes()?;
```

### Shortest Path

```rust
use manifoldb_graph::traversal::{ShortestPath, Dijkstra, AStar, Direction};

let tx = db.begin_read()?;

// Unweighted shortest path (BFS)
if let Some(path) = ShortestPath::find(tx.storage_ref()?, from, to, Direction::Both)? {
    println!("Path length: {}", path.nodes.len());
    for node in &path.nodes {
        println!("  -> {:?}", node);
    }
}

// Weighted shortest path
if let Some(result) = Dijkstra::new(from, to, Direction::Outgoing)
    .with_weight_property("distance")
    .find(tx.storage_ref()?)? {
    println!("Total distance: {}", result.total_weight);
}

// A* with heuristic (for spatial graphs)
use manifoldb_graph::traversal::EuclideanHeuristic;

if let Some(result) = AStar::new(from, to, Direction::Outgoing)
    .with_heuristic(EuclideanHeuristic::xy())  // Uses x, y properties
    .with_weight_property("cost")
    .find(tx.storage_ref()?)? {
    println!("Path cost: {}", result.total_weight);
}
```

### Graph Analytics

```rust
use manifoldb_graph::analytics::{
    PageRank, PageRankConfig,
    BetweennessCentrality, BetweennessCentralityConfig,
    CommunityDetection, CommunityDetectionConfig,
};

let tx = db.begin_read()?;

// PageRank
let config = PageRankConfig::default()
    .damping_factor(0.85)
    .max_iterations(100)
    .tolerance(1e-6);
let scores = PageRank::compute(tx.storage_ref()?, &config)?;

// Top 10 by PageRank
let mut sorted: Vec<_> = scores.iter().collect();
sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
for (node, score) in sorted.iter().take(10) {
    println!("{:?}: {:.4}", node, score);
}

// Betweenness Centrality
let config = BetweennessCentralityConfig::default();
let centrality = BetweennessCentrality::compute(tx.storage_ref()?, &config)?;

// Community Detection
let config = CommunityDetectionConfig::default();
let communities = CommunityDetection::compute(tx.storage_ref()?, &config)?;
println!("Found {} communities", communities.num_communities());
```

## Vector Search

### Storing Vectors

```rust
let mut tx = db.begin()?;

// Store embedding as entity property
let document = tx.create_entity()?
    .with_label("Document")
    .with_property("title", "Machine Learning Guide")
    .with_property("embedding", vec![0.1f32, 0.2, 0.3, /* ... */]);
tx.put_entity(&document)?;

tx.commit()?;
```

### HNSW Index Setup

```rust
use manifoldb_vector::{HnswIndex, HnswConfig};
use manifoldb_vector::distance::DistanceMetric;

// Create index with custom config
let config = HnswConfig::new(16)
    .with_ef_construction(200)
    .with_ef_search(100);

let mut index = HnswIndex::new(
    engine,
    "embeddings",
    384,  // dimension
    DistanceMetric::Cosine,
    config,
)?;

// Insert vectors
index.insert(entity_id, &embedding)?;

// Search
let results = index.search(&query_embedding, 10, None)?;
for result in results {
    println!("ID: {:?}, Distance: {}", result.entity_id, result.distance);
}
```

### Vector Search via SQL

```rust
use manifoldb::{Database, Value};

let db = Database::in_memory()?;

// Store documents with embeddings
db.execute("INSERT INTO docs (title, embedding) VALUES ('Doc 1', [0.1, 0.2, 0.3])")?;

// Similarity search with ORDER BY distance
let query = vec![0.1f32, 0.2, 0.3];
let results = db.query_with_params(
    "SELECT title FROM docs ORDER BY embedding <-> $1 LIMIT 10",
    &[Value::Vector(query)],
)?;
```

### Multiple Embedding Spaces

```rust
use manifoldb_vector::types::{EmbeddingName, EmbeddingSpace};

// Create different embedding spaces
let text_space = EmbeddingSpace::new(
    EmbeddingName::new("text_embedding")?,
    768,  // BERT dimension
    DistanceMetric::Cosine,
);

let image_space = EmbeddingSpace::new(
    EmbeddingName::new("image_embedding")?,
    512,  // Image model dimension
    DistanceMetric::Euclidean,
);

store.create_space(&text_space)?;
store.create_space(&image_space)?;

// Store embeddings in different spaces
store.put(doc_id, &EmbeddingName::new("text_embedding")?, &text_vec)?;
store.put(doc_id, &EmbeddingName::new("image_embedding")?, &image_vec)?;
```

## Hybrid Queries

### Graph + Vector: Semantic Search with Context

```rust
// Find similar documents, then expand to related entities
let similar_docs = index.search(&query_embedding, 20, None)?;

let mut tx = db.begin_read()?;
let mut results = Vec::new();

for doc in similar_docs {
    // Get the document
    if let Some(entity) = tx.get_entity(doc.entity_id)? {
        // Find connected entities
        let related = Expand::neighbors(tx.storage_ref()?, doc.entity_id, Direction::Outgoing)?
            .with_edge_type("MENTIONS")
            .collect()?;

        results.push((entity, related, doc.distance));
    }
}
```

### Graph + Vector: Personalized Recommendations

```rust
// Get user's preferences from graph
let liked_items = Expand::neighbors(tx.storage_ref()?, user_id, Direction::Outgoing)?
    .with_edge_type("LIKED")
    .collect()?;

// Compute average embedding of liked items
let mut avg_embedding = vec![0.0f32; 384];
let mut count = 0;

for item_id in liked_items {
    if let Some(item) = tx.get_entity(item_id)? {
        if let Some(Value::Vector(emb)) = item.get_property("embedding") {
            for (i, v) in emb.iter().enumerate() {
                avg_embedding[i] += v;
            }
            count += 1;
        }
    }
}

if count > 0 {
    for v in &mut avg_embedding {
        *v /= count as f32;
    }
}

// Find similar items using average embedding
let recommendations = index.search(&avg_embedding.into(), 10, None)?;

// Filter out already-liked items
let new_recommendations: Vec<_> = recommendations
    .into_iter()
    .filter(|r| !liked_items.contains(&r.entity_id))
    .collect();
```

## Batch Operations

### Bulk Entity Loading

```rust
let mut tx = db.begin()?;

// Prepare batch
let entities: Vec<Entity> = (0..10000)
    .map(|i| {
        Entity::new(EntityId::new(i))
            .with_label("Item")
            .with_property("index", i as i64)
    })
    .collect();

// Batch insert (much faster than individual puts)
tx.put_entities_batch(&entities)?;

tx.commit()?;
```

### Bulk Edge Loading

```rust
let mut tx = db.begin()?;

// Prepare edges
let edges: Vec<Edge> = relationships
    .iter()
    .enumerate()
    .map(|(i, (src, tgt))| {
        Edge::new(EdgeId::new(i as u64), *src, *tgt, "CONNECTED")
    })
    .collect();

// Batch insert
tx.put_edges_batch(&edges)?;

tx.commit()?;
```

### Import from CSV

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn import_nodes(db: &Database, path: &str) -> Result<(), Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut tx = db.begin()?;
    let mut batch = Vec::with_capacity(1000);

    for line in reader.lines().skip(1) {  // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();

        let entity = tx.create_entity()?
            .with_label("Record")
            .with_property("id", parts[0])
            .with_property("name", parts[1]);

        batch.push(entity);

        // Commit in batches of 1000
        if batch.len() >= 1000 {
            tx.put_entities_batch(&batch)?;
            batch.clear();
        }
    }

    // Final batch
    if !batch.is_empty() {
        tx.put_entities_batch(&batch)?;
    }

    tx.commit()?;
    Ok(())
}
```

## Query Caching

### Using Cache Hints

```rust
// Force caching for expensive queries
let results = db.query("/*+ CACHE */ SELECT * FROM large_table WHERE complex_condition")?;

// Skip cache for real-time data
let results = db.query("/*+ NO_CACHE */ SELECT * FROM live_metrics")?;
```

### Cache Configuration

```rust
use manifoldb::cache::CacheConfig;
use std::time::Duration;

let db = DatabaseBuilder::new()
    .path("data.manifold")
    .query_cache_config(
        CacheConfig::new()
            .max_entries(10000)
            .max_size(100 * 1024 * 1024)  // 100MB
            .ttl(Some(Duration::from_secs(3600)))  // 1 hour
    )
    .open()?;
```

### Manual Cache Control

```rust
// Clear entire cache
db.clear_cache();

// Check cache metrics
let metrics = db.cache_metrics();
println!("Hit rate: {:?}", metrics.hit_rate());
println!("Total lookups: {}", metrics.total_lookups());

// Invalidate specific tables
db.invalidate_cache_for_tables(&["users".to_string()]);
```

## Error Handling

### Pattern for Transactional Operations

```rust
use manifoldb::{Database, Error};

fn transfer_funds(db: &Database, from: EntityId, to: EntityId, amount: f64) -> Result<(), Error> {
    let mut tx = db.begin()?;

    // Get accounts
    let mut from_account = tx.get_entity(from)?
        .ok_or(Error::Type("Source account not found".into()))?;
    let mut to_account = tx.get_entity(to)?
        .ok_or(Error::Type("Destination account not found".into()))?;

    // Check balance
    let from_balance: f64 = from_account
        .get_property("balance")
        .and_then(|v| v.as_float())
        .ok_or(Error::Type("Invalid balance".into()))?;

    if from_balance < amount {
        return Err(Error::Type("Insufficient funds".into()));
    }

    // Update balances
    let to_balance: f64 = to_account
        .get_property("balance")
        .and_then(|v| v.as_float())
        .unwrap_or(0.0);

    from_account.set_property("balance", from_balance - amount);
    to_account.set_property("balance", to_balance + amount);

    // Save
    tx.put_entity(&from_account)?;
    tx.put_entity(&to_account)?;

    // Commit
    tx.commit()?;

    Ok(())
}
```

### Retry Pattern

```rust
fn with_retry<T, F>(mut f: F, max_attempts: usize) -> Result<T, Error>
where
    F: FnMut() -> Result<T, Error>,
{
    let mut last_error = None;

    for attempt in 1..=max_attempts {
        match f() {
            Ok(result) => return Ok(result),
            Err(Error::Transaction(manifoldb_core::TransactionError::Conflict(_))) => {
                last_error = Some(Error::Transaction(
                    manifoldb_core::TransactionError::Conflict("retry".into())
                ));
                // Optional: add backoff
                std::thread::sleep(std::time::Duration::from_millis(10 * attempt as u64));
            }
            Err(e) => return Err(e),
        }
    }

    Err(last_error.unwrap())
}
```

## Testing

### Test Setup

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb::Database;

    fn setup_test_db() -> Database {
        Database::in_memory().expect("failed to create test db")
    }

    fn seed_test_data(db: &Database) -> Result<Vec<EntityId>, Error> {
        let mut tx = db.begin()?;
        let mut ids = Vec::new();

        for i in 0..10 {
            let entity = tx.create_entity()?
                .with_label("TestEntity")
                .with_property("index", i);
            ids.push(entity.id);
            tx.put_entity(&entity)?;
        }

        tx.commit()?;
        Ok(ids)
    }

    #[test]
    fn test_basic_crud() {
        let db = setup_test_db();
        let ids = seed_test_data(&db).unwrap();

        // Test read
        let tx = db.begin_read().unwrap();
        let entity = tx.get_entity(ids[0]).unwrap().unwrap();
        assert!(entity.has_label("TestEntity"));
    }
}
```

### Property-Based Testing

```rust
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;
    use manifoldb::Database;

    proptest! {
        #[test]
        fn entity_roundtrip(
            name in "[a-z]+",
            age in 0i64..150,
        ) {
            let db = Database::in_memory().unwrap();

            let mut tx = db.begin().unwrap();
            let entity = tx.create_entity().unwrap()
                .with_property("name", name.clone())
                .with_property("age", age);
            let id = entity.id;
            tx.put_entity(&entity).unwrap();
            tx.commit().unwrap();

            let tx = db.begin_read().unwrap();
            let loaded = tx.get_entity(id).unwrap().unwrap();

            assert_eq!(
                loaded.get_property("name").and_then(|v| v.as_string()),
                Some(&name)
            );
            assert_eq!(
                loaded.get_property("age").and_then(|v| v.as_int()),
                Some(age)
            );
        }
    }
}
```
