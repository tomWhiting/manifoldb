# ManifoldDB Vision & Design Principles

## What is ManifoldDB?

ManifoldDB is a **unified graph-vector database** that combines property graph storage with vector similarity search. It provides a single database that handles:

- **Graph data** - Entities with labels, properties, and typed edges
- **Vector embeddings** - Dense, sparse, and multi-vector storage with ANN search
- **Hybrid queries** - SQL and Cypher syntax with vector distance operators

## Core Purpose

**Enable applications to store, query, and search structured knowledge with semantic understanding.**

ManifoldDB answers questions like:
- "Find similar documents" (vector similarity)
- "Find all users who follow Alice" (graph traversal)
- "Find similar code within this repository" (graph-constrained vector search)
- "What are the dependencies of this package?" (structured queries)

## Design Principles

### 1. Unified Entity Model

Everything is an **Entity**. No separate Point/Node/Document types. An entity can have:
- Labels (categories/types)
- Properties (key-value pairs)
- Vectors (embeddings for similarity search)
- Edges (relationships to other entities)

```rust
let entity = Entity::new(id)
    .with_label("Document")
    .with_property("title", "Hello World")
    .with_vector("embedding", dense_vec)
    .with_vector("sparse", sparse_vec);
```

### 2. Query Language Flexibility

Support multiple query syntaxes that compile to the same execution engine:
- **SQL** for tabular queries, joins, aggregations
- **Cypher** for graph pattern matching
- **Programmatic API** for type-safe Rust access

Users choose the syntax that fits their mental model.

### 3. Graph-Aware Vector Search

Vector search should understand graph structure. Key capability:
- Search within traversal results (graph-constrained vector search)
- Filter by relationships, not just properties
- Combine semantic similarity with structural relevance

### 4. Production Quality

Code must be:
- **Robust** - Proper error handling, no panics in library code
- **Efficient** - Minimal allocations, smart indexing, streaming where possible
- **Maintainable** - Clear module boundaries, comprehensive tests
- **Documented** - Public APIs have doc comments with examples

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Public API (manifoldb)                   │
│         Database, Entity, Filter, Query Execution           │
├─────────────────────────────────────────────────────────────┤
│                   Query Engine (manifoldb-query)            │
│         Parser → AST → LogicalPlan → PhysicalPlan           │
│                      → Execution Operators                   │
├─────────────────────────────────────────────────────────────┤
│  Graph Layer          │  Vector Layer        │  Core Types  │
│  (manifoldb-graph)    │  (manifoldb-vector)  │  (manifoldb- │
│  - Edge storage       │  - ANN indexes       │   core)      │
│  - Traversal          │  - Hybrid search     │  - Entity    │
│  - Path patterns      │  - Distance metrics  │  - Value     │
├─────────────────────────────────────────────────────────────┤
│               Storage Engine (manifoldb-storage)            │
│                    Redb backend, Transactions               │
└─────────────────────────────────────────────────────────────┘
```

## Key Capabilities

| Capability | Status | Description |
|------------|--------|-------------|
| Entity CRUD | Complete | Create, read, update, delete entities |
| Edge CRUD | Complete | Typed edges with properties |
| Vector Search | Complete | Dense, sparse, multi-vector ANN |
| Hybrid Search | Complete | Combine dense + sparse with RRF |
| Payload Indexing | Complete | B-tree indexes on properties |
| SQL Queries | ~90% | SELECT, JOIN, GROUP BY, subqueries |
| Graph Patterns | ~60% | MATCH clause, variable-length paths |
| Cypher Write | ~40% | CREATE, MERGE in progress |
| Graph-Constrained Search | Not Started | Vector search within traversal |

## Success Criteria

### Functional Requirements

1. **Unified API** - Single Entity type for all data, no separate collection types
2. **Multi-model queries** - SQL, Cypher, and programmatic access to same data
3. **Graph + Vector** - Seamlessly combine graph traversal with vector search
4. **Transactional** - ACID transactions for data integrity

### Quality Requirements

1. **No runtime panics** in library code
2. **Comprehensive tests** - Unit tests for logic, integration tests for workflows
3. **Clean clippy** with strict lints
4. **Sub-second queries** for typical workloads

## What ManifoldDB is NOT

- **Not a distributed database** - Single-node, embedded database
- **Not a full SQL database** - Focused subset for knowledge graphs
- **Not schema-enforced** - Flexible property graphs, schema is convention
- **Not an ML platform** - Stores embeddings, doesn't generate them

## Integration Points

### Primary Consumer: Gimbal

Gimbal is a code/documentation knowledge graph that uses ManifoldDB for:
- Storing code symbols, documentation sections
- Graph relationships (CONTAINS, FOLLOWED_BY, DEPENDS_ON)
- Vector similarity search with language/visibility filters
- **Graph-constrained search** (search within a repository) - CRITICAL GAP

### Embedding Generation: Tessera

ManifoldDB stores vectors but doesn't generate them. Tessera handles:
- Dense embeddings (BGE, sentence-transformers)
- Sparse embeddings (SPLADE)
- Multi-vector embeddings (ColBERT)

## Guiding Questions for Review

When reviewing code, ask:

1. **Does it fit the unified model?** - Uses Entity, not custom point types?
2. **Is query execution correct?** - Logical plan → physical plan → results?
3. **Is it production-ready?** - Error handling, performance, tests?
4. **Does it follow conventions?** - Module structure, coding standards?
5. **Does it serve Gimbal's needs?** - Graph-constrained search, write operations?
