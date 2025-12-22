# ManifoldDB Architecture

This document describes the internal architecture of ManifoldDB and how its components interact.

## Overview

ManifoldDB is a multi-paradigm embedded database that unifies three data models:
- **Graph**: Property graphs with nodes (entities) and relationships (edges)
- **Vector**: High-dimensional embeddings with similarity search
- **Relational**: SQL queries over structured data

The system is designed as a layered architecture where each layer builds on the one below it.

## Crate Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         manifoldb                               │
│            (Main API, Database, Transactions)                   │
├─────────────────────────────────────────────────────────────────┤
│  manifoldb-query  │  manifoldb-graph   │  manifoldb-vector     │
│  (SQL Parser,     │  (Traversal,       │  (HNSW Index,         │
│   Execution)      │   Analytics)       │   Similarity Search)  │
├─────────────────────────────────────────────────────────────────┤
│                       manifoldb-storage                         │
│              (Storage Engine, Redb Backend, WAL)                │
├─────────────────────────────────────────────────────────────────┤
│                        manifoldb-core                           │
│            (Entity, Edge, Value, IDs, Encoding)                 │
└─────────────────────────────────────────────────────────────────┘
```

### manifoldb-core

The foundation layer containing types shared across all other crates:

- **EntityId / EdgeId**: Type-safe identifiers for graph nodes and edges
- **Entity**: A node with labels and properties
- **Edge**: A directed relationship between entities with a type and properties
- **Value**: The value type supporting multiple data types (String, Int, Float, Bool, Vector, Null)
- **Encoding**: Binary serialization for storage

### manifoldb-storage

Abstracts storage operations with a key-value interface:

- **StorageEngine trait**: Main entry point for creating transactions
- **Transaction trait**: ACID operations (get, put, delete, range scans)
- **Cursor trait**: Ordered iteration over key-value pairs
- **RedbEngine**: Concrete implementation using the [redb](https://github.com/cberner/redb) embedded database
- **WAL (Write-Ahead Log)**: Durability and crash recovery

The storage layer uses a multi-table design:
- `nodes`: Entity data keyed by EntityId
- `edges`: Edge data keyed by EdgeId
- `edges_out`: Outgoing adjacency index (source_id -> edge_id)
- `edges_in`: Incoming adjacency index (target_id -> edge_id)
- `metadata`: Counters and configuration

### manifoldb-graph

Graph-specific operations built on top of storage:

**Traversal Algorithms:**
- `Expand`: Single-hop neighbor expansion
- `ExpandAll`: Multi-hop traversal with depth control
- `ShortestPath`: BFS for unweighted shortest paths
- `Dijkstra`: Weighted shortest path
- `AStar`: Goal-directed search with heuristics
- `PathPattern`: Pattern matching for paths

**Analytics Algorithms:**
- `PageRank`: Node importance via iterative power method
- `BetweennessCentrality`: Node centrality via Brandes algorithm
- `CommunityDetection`: Community finding via label propagation

**Indexing:**
- Adjacency list indexes for efficient neighbor lookups
- Edge type indexes for filtered traversal

### manifoldb-vector

Vector embedding storage and similarity search:

**Distance Metrics:**
- Cosine similarity
- Euclidean distance
- Dot product
- Manhattan distance (L1)
- Chebyshev distance (L-infinity)
- Hamming distance (for binary embeddings)

**HNSW Index:**
- Hierarchical graph structure for O(log N) approximate nearest neighbor search
- Configurable parameters (M, ef_construction, ef_search)
- Incremental updates (insert/delete)
- Persistence to storage

**Embedding Types:**
- Dense embeddings (f32 vectors)
- Sparse embeddings (for high-dimensional sparse data)
- Binary embeddings (for Hamming distance)

**Compression:**
- Product Quantization (PQ) for memory-efficient storage
- Configurable segments and centroids
- 4-8x memory reduction with minimal recall loss

### manifoldb-query

SQL parsing and query execution:

**Parser:**
- Standard SQL syntax via sqlparser
- Graph pattern extensions (`MATCH (a)-[:TYPE]->(b)`)
- Vector operators (`embedding <-> vector`)

**AST (Abstract Syntax Tree):**
- Rich representation of parsed queries
- Programmatic query building

**Logical Plan:**
- High-level query representation
- Plan optimization

**Execution:**
- Operator-based execution model
- Streaming results via iterators

### manifoldb (Main Crate)

The user-facing API integrating all components:

**Database:**
- Opening/creating databases
- Configuration via `DatabaseBuilder`
- SQL execution (`execute`, `query`)
- Transaction management (`begin`, `begin_read`)

**Transactions:**
- `DatabaseTransaction`: High-level entity/edge operations
- `TransactionManager`: Coordinates storage, graph, and vector layers
- Vector sync strategies (Synchronous, Async, Hybrid)

**Caching:**
- Query result caching with TTL
- Smart invalidation on writes
- Cache hints (`/*+ CACHE */`, `/*+ NO_CACHE */`)

**Metrics:**
- Query statistics
- Transaction counters
- Cache hit rates
- Vector search metrics

## Data Flow

### Write Path

```
1. User calls db.begin() -> TransactionManager creates DatabaseTransaction
2. User modifies entities/edges via tx.put_entity(), tx.put_edge()
3. Transaction buffers writes in storage layer
4. User calls tx.commit()
5. Storage layer persists to disk
6. Graph indexes updated (adjacency lists)
7. Vector indexes updated (based on sync strategy)
8. Query cache invalidated for affected tables
```

### Read Path (Direct)

```
1. User calls db.begin_read() -> Read transaction created
2. User fetches data via tx.get_entity(), tx.get_edge()
3. Storage layer reads from disk/cache
4. Data deserialized and returned
```

### Query Path

```
1. User calls db.query(sql)
2. Check query cache for cached result
3. If cache miss:
   a. Parse SQL -> AST
   b. Build logical plan
   c. Create physical operators
   d. Execute operators, stream results
   e. Cache result (if caching enabled)
4. Return QueryResult
```

### Vector Search Path

```
1. Query includes vector similarity operator
2. HNSW index performs approximate nearest neighbor search
3. Returns candidate set with distances
4. Optional: Re-rank candidates with exact distance
5. Apply any additional filters
6. Return sorted results
```

## Threading Model

- `Database` is `Send + Sync` - safe to share across threads
- Multiple concurrent read transactions allowed
- Single write transaction at a time (serialized)
- Background vector index updates with Async sync strategy

## Persistence

ManifoldDB uses redb for storage, which provides:
- ACID transactions
- Copy-on-write B+ tree
- Memory-mapped I/O
- Automatic crash recovery

HNSW indexes are persisted to storage using a custom format:
- Index metadata (dimensions, distance metric, config)
- Node data (embedding + layer information)
- Connection data (neighbor lists per layer)

## Memory Management

- **Storage cache**: Configurable via `cache_size` in builder
- **Query cache**: Configurable via `CacheConfig`
- **HNSW graph**: In-memory graph structure with optional PQ compression
- **Embeddings**: Can be loaded on-demand or cached

## Extensibility Points

- **Storage backends**: Implement `StorageEngine` trait for new backends
- **Distance metrics**: Add new metrics in `manifoldb-vector::distance`
- **Graph algorithms**: Add new traversal/analytics in `manifoldb-graph`
- **Query operators**: Extend parser and add physical operators
