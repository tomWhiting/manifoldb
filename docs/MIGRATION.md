# ManifoldDB Migration Guide

## Overview

ManifoldDB has moved from a dual API (Entity + Collection) to a unified Entity-first API. This document explains what has changed and how to update your code.

**Status:** Phase 1 and Phase 2 complete (December 2024)

## Previous State (0.1.3)

ManifoldDB previously had two parallel systems:

### Graph API (Entity-based)
```rust
use manifoldb::{Entity, EntityId, Edge};

let entity = Entity::new("Symbol")
    .with_property("name", "parse_config")
    .with_property("language", "rust");

db.upsert_entity(entity)?;
db.add_edge(file_id, symbol_id, "CONTAINS")?;
```

### Collection API (Point-based)
```rust
use manifoldb::collection::{CollectionHandle, PointStruct, Filter};

let collection = db.collection("symbols")?;
collection.upsert_point(PointStruct::new(1)
    .with_payload(json!({"name": "parse_config", "language": "rust"}))
    .with_vector("dense", embedding)
)?;

let results = collection.search("dense")
    .query(query_vector)
    .filter(Filter::eq("language", "rust"))
    .limit(10)
    .execute()?;
```

### The Problem

- `PointId` and `EntityId` share the same ID space (both wrap `u64`)
- HNSW indexes internally convert `PointId` → `EntityId`
- But there's no enforced relationship between Points and Entities
- You can have a Point without an Entity, or vice versa
- Two APIs to learn, easy to use the wrong one

---

## What Has Changed

### Unified Entity API

Everything is an Entity. Vectors are attached to entities, not stored separately as "points."

```rust
use manifoldb::{Database, Entity, EntityId, Filter, ScoredEntity};

// Create entity with vectors
let symbol = Entity::new(EntityId::new(1))
    .with_label("Symbol")
    .with_property("name", "parse_config")
    .with_property("language", "rust")
    .with_property("kind", "function")
    .with_vector("dense", embedding)
    .with_vector("sparse", sparse_embedding);

// Upsert to a collection (still needed for vector storage)
db.upsert("symbols", &symbol)?;

// Graph edges work the same
db.add_edge(file_id, symbol_id, "CONTAINS")?;

// Search returns entities
let results: Vec<ScoredEntity> = db.search("symbols", "dense")?
    .query(query_vector)
    .filter(Filter::eq("language", "rust"))
    .limit(10)
    .execute()?;

for result in results {
    let entity = &result.entity;
    let name = entity.get_property("name");
    println!("{:?}: {}", name, result.score);
}
```

### Hidden Types

The collection module is now hidden (`#[doc(hidden)]`). The following types are replaced:

| Hidden | Replacement |
|--------|-------------|
| `PointId` | `EntityId` |
| `PointStruct` | `Entity` with `.with_vector()` |
| `Payload` | Entity properties |
| `CollectionHandle` | Use `db.search()` and `db.upsert()` directly |
| `ScoredPoint` | `ScoredEntity` |

### Moved Types

| Before | After |
|--------|-------|
| `manifoldb::collection::Filter` | `manifoldb::Filter` |

### New Types

| Type | Purpose |
|------|---------|
| `VectorData` | Enum for Dense, Sparse, and Multi vectors |
| `ScoredEntity` | Entity with similarity score from search |
| `ScoredId` | Lightweight ID + score (when you don't need full entity) |
| `EntitySearchBuilder` | Fluent builder for vector search |

---

## Breaking Changes Summary

### 1. Collection Module Hidden

```rust
// BEFORE
use manifoldb::collection::{CollectionHandle, PointStruct, Filter};
let collection = db.collection("symbols")?;

// AFTER
use manifoldb::{Filter, Entity, EntityId};
// Work directly with db methods
```

### 2. Point Operations → Entity Operations

```rust
// BEFORE
collection.upsert_point(PointStruct::new(id)
    .with_payload(json!({"name": "foo"}))
    .with_vector("dense", vec))?;

// AFTER
let entity = Entity::new(EntityId::new(id))
    .with_label("MyLabel")
    .with_property("name", "foo")
    .with_vector("dense", vec);
db.upsert("collection_name", &entity)?;
```

### 3. Search API

```rust
// BEFORE
let results: Vec<ScoredPoint> = collection.search("dense")
    .query(vec)
    .filter(Filter::eq("name", "foo"))
    .execute()?;
let name = results[0].payload.get("name");

// AFTER
let results: Vec<ScoredEntity> = db.search("collection_name", "dense")?
    .query(vec)
    .filter(Filter::eq("name", "foo"))
    .execute()?;
let name = results[0].entity.get_property("name");
```

### 4. ID Type

```rust
// BEFORE
let id = PointId::new(42);

// AFTER
let id = EntityId::new(42);
```

---

## New Capabilities

### Available Now

#### Unified Search API

Search with the fluent `EntitySearchBuilder`:

```rust
let results = db.search("symbols", "dense")?
    .query(query_vector)
    .filter(Filter::eq("language", "rust"))
    .limit(10)
    .offset(0)
    .score_threshold(0.5)
    .execute()?;
```

#### Entity Vectors

Attach multiple named vectors to entities:

```rust
let entity = Entity::new(EntityId::new(1))
    .with_vector("dense", vec![0.1; 768])        // Dense vector
    .with_vector("sparse", vec![(10, 0.5)])      // Sparse vector
    .with_vector("multi", vec![vec![0.1; 128]]); // Multi-vector (ColBERT-style)
```

### Planned (Phase 3+)

#### Payload Indexing

Filtered searches will be faster with B-tree indexes:

```rust
db.create_index("symbols", "language")?;
db.create_index("symbols", "kind")?;
```

#### Graph-Constrained Vector Search

Search within graph traversal results:

```rust
let results = db.search("symbols", "dense")?
    .query(query_vector)
    .within_graph(|t| t
        .from(repo_id)
        .traverse("CONTAINS")
        .variable_length(1..)
    )
    .limit(10)
    .execute()?;
```

#### Full Cypher Support

Query with Cypher syntax, including vector operations:

```cypher
MATCH (r:Repository {name: "gimbal"})-[:CONTAINS*]->(s:Symbol)
WHERE s.language = "rust" AND s.visibility = "public"
RETURN s
ORDER BY s.embedding <-> $query
LIMIT 10
```

---

## Migration Checklist

- [ ] Replace `PointId` with `EntityId`
- [ ] Replace `PointStruct` with `Entity` + `.with_vector()`
- [ ] Replace `collection.upsert_point()` with `db.upsert("collection", &entity)`
- [ ] Replace `collection.search("vector")` with `db.search("collection", "vector")?`
- [ ] Update search result handling: `ScoredPoint` → `ScoredEntity`
- [ ] Access entity properties via `result.entity.get_property("key")` instead of `result.payload.get("key")`
- [ ] Move `Filter` import from `collection::Filter` to `manifoldb::Filter`
- [ ] Import new types: `VectorData`, `ScoredEntity`, `EntitySearchBuilder`

---

## Questions?

The unified API eliminates the conceptual split between "graph entities" and "vector points." An entity is an entity—it can have properties, edges, and vectors. One type, one API, one mental model.

---

## See Also

- [Unified Entity API Design](design/unified-entity-api.md) - Full architecture plan
- [Architecture](architecture.md) - System overview
