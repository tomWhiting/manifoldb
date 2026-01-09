# ManifoldDB Query Implementation Roadmap

This document provides a comprehensive roadmap for completing full SQL and Cypher support in ManifoldDB. It outlines the architectural approach, implementation phases, and specific work items needed to achieve production-ready query capabilities.

> **Note:** This document covers **query language features** (SQL/Cypher completion). For **infrastructure improvements** (storage backends, connection handling, performance), see [docs/improvement-roadmap.md](./docs/improvement-roadmap.md).

## Executive Summary

ManifoldDB currently has solid SQL fundamentals and graph pattern matching. This roadmap describes the work needed to achieve:

1. **Complete PostgreSQL-compatible SQL** - Full DML/DDL, window functions, CTEs, JSON/array support
2. **Complete openCypher support** - Full read/write clauses, path operations, procedures
3. **Graph algorithms via CALL/YIELD** - PageRank, shortest path, centrality, community detection
4. **Unified query architecture** - Single IR that both languages compile to

### Recently Completed

- ✅ **CTEs (WITH clause)** - Non-recursive CTEs with multiple CTE support
- ✅ **OPTIONAL MATCH** - LEFT OUTER JOIN semantics for graph patterns
- ✅ **Graph-Constrained Vector Search** - `.within_traversal()` API
- ✅ **EXPLAIN command** - Shows logical and physical plan trees
- ✅ **Payload Indexing** - B-tree indexes on entity properties

---

## Architecture Overview

### Current State

```
┌─────────────┐     ┌─────────────┐
│  SQL Parser │     │Cypher Parser│   (separate, limited)
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│           AST Layer             │   (partially unified)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│    Logical Plan (incomplete)    │   (missing nodes)
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Physical Operators (partial)  │   (some stubs)
└─────────────────────────────────┘
```

### Target State

```
┌─────────────┐     ┌─────────────┐
│  SQL Parser │     │Cypher Parser│   (complete, validated)
│ (PostgreSQL)│     │ (openCypher)│
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│        Unified AST Layer        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Unified Logical Plan        │   (complete IR)
│   (see LOGICAL_PLAN_SPEC.md)    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      Physical Plan Layer        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│    Execution Operators          │   (all implemented)
└─────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation & Architecture (Priority: Critical)

#### 1.1 Unified Logical Plan IR
**Goal:** Define a complete intermediate representation that both SQL and Cypher compile to.

- [ ] Design complete logical plan node taxonomy (see LOGICAL_PLAN_SPECIFICATION.md)
- [ ] Implement missing logical plan nodes:
  - `Window` - Window function evaluation
  - `RecursiveCTE` - Recursive CTE evaluation
  - `Procedure` - CALL ... YIELD procedure invocation
  - `ShortestPath` - Graph shortest path computation
  - `AllShortestPaths` - All shortest paths computation
  - `GraphCreate` - CREATE for nodes/edges
  - `GraphMerge` - MERGE with ON CREATE/ON MATCH
  - `GraphSet` - SET property updates
  - `GraphRemove` - REMOVE property/label
  - `GraphDelete` - DELETE nodes/edges
  - `DetachDelete` - DETACH DELETE
  - `Foreach` - Cypher FOREACH iteration
- [ ] Add type system for plan nodes (input/output schemas)
- [ ] Implement plan validation and sanity checks

#### 1.2 Expression System Completion
**Goal:** Complete expression evaluation for both languages.

- [ ] Extend `Expr` enum for missing expression types:
  - Window function expressions with frame specs
  - List comprehensions `[x IN list WHERE ... | expr]`
  - Map projections `node{.prop1, .prop2}`
  - Pattern expressions (for EXISTS subqueries)
  - Temporal literals and operations
  - Spatial point literals
- [ ] Implement expression type inference
- [ ] Add expression simplification/optimization

---

### Phase 2: SQL Completion (Priority: High)

#### 2.1 Parser Extensions
**Goal:** Parse complete PostgreSQL-compatible SQL.

- [x] **CTEs** (basic support complete)
  - [x] Non-recursive CTEs with multiple CTE support
  - [ ] WITH RECURSIVE support (currently rejected)
  - [ ] SEARCH DEPTH/BREADTH FIRST
  - [ ] CYCLE detection clause
  - [ ] MATERIALIZED/NOT MATERIALIZED hints

- [ ] **Window Functions**
  - [ ] Named window definitions (WINDOW w AS ...)
  - [ ] Full frame clause support (ROWS/RANGE/GROUPS)
  - [ ] Frame exclusion (EXCLUDE CURRENT ROW, etc.)
  - [ ] FILTER clause on window functions

- [ ] **Advanced SELECT**
  - [ ] DISTINCT ON (expression, ...)
  - [ ] FETCH FIRST ... WITH TIES
  - [ ] TABLESAMPLE clause
  - [ ] LATERAL subqueries

- [ ] **DDL Extensions**
  - [ ] ALTER TABLE (ADD/DROP/ALTER COLUMN)
  - [ ] ALTER INDEX
  - [ ] CREATE/ALTER/DROP VIEW
  - [ ] CREATE/ALTER/DROP SCHEMA
  - [ ] CREATE/ALTER/DROP FUNCTION
  - [ ] CREATE/ALTER/DROP TRIGGER
  - [ ] TRUNCATE TABLE
  - [ ] Partitioned tables (PARTITION BY RANGE/LIST/HASH)

- [ ] **Transactions**
  - [ ] BEGIN/START TRANSACTION
  - [ ] COMMIT/ROLLBACK
  - [ ] SAVEPOINT/RELEASE SAVEPOINT/ROLLBACK TO
  - [ ] SET TRANSACTION ISOLATION LEVEL

- [ ] **Utility Statements**
  - [ ] EXPLAIN ANALYZE with options
  - [ ] VACUUM/ANALYZE
  - [ ] COPY (import/export)
  - [ ] SET/SHOW/RESET session variables

#### 2.2 Function Library
**Goal:** Implement PostgreSQL-compatible function library.

- [ ] **String Functions** (Tier 1)
  - [ ] length, char_length, octet_length
  - [ ] substring, substr, position, strpos
  - [ ] upper, lower, initcap
  - [ ] ltrim, rtrim, btrim, trim
  - [ ] lpad, rpad
  - [ ] concat, concat_ws
  - [ ] replace, translate
  - [ ] split_part, string_to_array
  - [ ] format
  - [ ] regexp_match, regexp_replace, regexp_split_to_array

- [ ] **Numeric Functions** (Tier 1)
  - [ ] abs, sign, ceil, floor, round, trunc
  - [ ] sqrt, cbrt, power, exp, ln, log, log10
  - [ ] mod, div
  - [ ] sin, cos, tan, asin, acos, atan, atan2
  - [ ] degrees, radians
  - [ ] random, setseed

- [ ] **Date/Time Functions** (Tier 1)
  - [ ] now, current_timestamp, current_date, current_time
  - [ ] date_part, extract, date_trunc
  - [ ] to_timestamp, to_date, to_char
  - [ ] age, date_add, date_subtract
  - [ ] make_timestamp, make_date, make_time
  - [ ] timezone, at time zone

- [ ] **Aggregate Functions** (Tier 1)
  - [ ] count, sum, avg, min, max (already have)
  - [ ] array_agg, string_agg
  - [ ] json_agg, jsonb_agg, json_object_agg
  - [ ] bool_and, bool_or, every
  - [ ] FILTER clause on aggregates

- [ ] **Window Functions** (Tier 2)
  - [ ] row_number, rank, dense_rank, ntile
  - [ ] lag, lead
  - [ ] first_value, last_value, nth_value
  - [ ] percent_rank, cume_dist
  - [ ] Any aggregate as window function

- [ ] **JSON Functions** (Tier 2)
  - [ ] Operators: ->, ->>, #>, #>>, @>, <@, ?, ?|, ?&
  - [ ] json_extract_path, jsonb_extract_path
  - [ ] json_build_object, json_build_array
  - [ ] json_each, json_each_text
  - [ ] json_array_elements
  - [ ] jsonb_set, jsonb_insert
  - [ ] jsonb_path_query, jsonb_path_exists

- [ ] **Array Functions** (Tier 2)
  - [ ] array_length, cardinality
  - [ ] array_append, array_prepend, array_cat
  - [ ] array_position, array_positions
  - [ ] array_remove, array_replace
  - [ ] unnest
  - [ ] Subscript access array[n]

- [ ] **Type Conversion** (Tier 1)
  - [ ] CAST(expr AS type)
  - [ ] :: operator
  - [ ] to_text, to_number, to_boolean

#### 2.3 Logical Plan Generation
**Goal:** Generate correct logical plans from SQL AST.

- [ ] Implement SQL → Logical Plan translation for all new constructs
- [ ] Handle correlated subqueries correctly
- [ ] Implement proper scoping for CTEs and subqueries
- [ ] Add LATERAL join support in planner

#### 2.4 Physical Operators
**Goal:** Implement missing physical operators.

- [ ] **WindowOp** - Window function evaluation
  - [ ] Partition management
  - [ ] Frame calculation
  - [ ] Support all ranking functions

- [ ] **RecursiveCTEOp** - Recursive CTE execution
  - [ ] Working table management
  - [ ] Cycle detection
  - [ ] Depth/breadth-first ordering

- [ ] **IndexNestedLoopJoinOp** - Index-accelerated joins
- [ ] **SortMergeJoinOp** - Sort-merge join implementation
- [ ] **HashAggregateOp** - Complete HAVING support (already partial)

---

### Phase 3: Cypher Completion (Priority: High)

#### 3.1 Parser Extensions
**Goal:** Parse complete openCypher queries.

- [ ] **Writing Clauses**
  - [ ] CREATE (nodes and relationships)
  - [ ] MERGE with ON CREATE SET and ON MATCH SET
  - [ ] SET (properties and labels)
  - [ ] REMOVE (properties and labels)
  - [ ] DELETE and DETACH DELETE
  - [ ] FOREACH

- [ ] **Reading Clauses**
  - [ ] MANDATORY MATCH (optional, Neo4j extension)
  - [ ] Full label expressions (:Label1|Label2, :Label1&Label2)

- [ ] **Path Functions**
  - [ ] shortestPath() pattern function
  - [ ] allShortestPaths() pattern function
  - [ ] Variable-length path improvements

- [ ] **Subqueries**
  - [ ] EXISTS { } subquery
  - [ ] COUNT { } subquery
  - [ ] CALL { } subquery (inline)

- [ ] **Advanced Patterns**
  - [ ] Quantified path patterns (GQL)
  - [ ] Path pattern assignment
  - [ ] Multiple relationship types in pattern

#### 3.2 Expression Extensions
**Goal:** Support all Cypher expression forms.

- [ ] List comprehensions: `[x IN list WHERE pred | expr]`
- [ ] Map projections: `node{.prop1, .prop2, key: expr}`
- [ ] Pattern comprehensions: `[(n)-[:REL]->(m) | m.name]`
- [ ] CASE expressions (simple and searched)
- [ ] Parameter syntax ($param)

#### 3.3 Function Library
**Goal:** Implement openCypher function library.

- [ ] **Scalar Functions**
  - [ ] coalesce, head, tail, last
  - [ ] size, length (for lists, strings, paths)
  - [ ] type, labels, id, properties
  - [ ] keys, nodes, relationships
  - [ ] startNode, endNode
  - [ ] toBoolean, toInteger, toFloat, toString

- [ ] **Aggregating Functions**
  - [ ] collect, count, sum, avg, min, max
  - [ ] percentileCont, percentileDisc
  - [ ] stDev, stDevP

- [ ] **List Functions**
  - [ ] range, reverse
  - [ ] reduce (fold operation)
  - [ ] all, any, none, single (predicate tests)

- [ ] **String Functions**
  - [ ] Same as SQL tier + specific Cypher names
  - [ ] left, right

- [ ] **Mathematical Functions**
  - [ ] Same as SQL tier

- [ ] **Temporal Functions** (Tier 2)
  - [ ] datetime, localdatetime, date, time, localtime
  - [ ] duration
  - [ ] datetime.truncate
  - [ ] Temporal arithmetic

- [ ] **Spatial Functions** (Tier 3)
  - [ ] point construction
  - [ ] point.distance
  - [ ] point.withinBBox

#### 3.4 Logical Plan Generation
**Goal:** Generate correct logical plans from Cypher AST.

- [ ] Implement Cypher → Logical Plan translation for:
  - [ ] MATCH with complex patterns
  - [ ] CREATE patterns
  - [ ] MERGE with conditions
  - [ ] SET/REMOVE operations
  - [ ] DELETE/DETACH DELETE
  - [ ] WITH clause chaining
  - [ ] UNION/UNION ALL
  - [ ] CALL ... YIELD

#### 3.5 Physical Operators
**Goal:** Implement Cypher-specific operators.

- [ ] **GraphCreateOp** - Create nodes and edges
- [ ] **GraphMergeOp** - Merge with create/match logic
- [ ] **GraphSetOp** - Set properties/labels
- [ ] **GraphRemoveOp** - Remove properties/labels
- [ ] **GraphDeleteOp** - Delete with referential checks
- [ ] **ShortestPathOp** - BFS-based shortest path
- [ ] **AllShortestPathsOp** - All shortest paths
- [ ] **VariableLengthExpandOp** - Multi-hop expansion

---

### Phase 4: Graph Algorithms (Priority: Medium)

#### 4.1 CALL/YIELD Infrastructure
**Goal:** Implement procedure call framework.

- [ ] Design procedure registry interface
- [ ] Implement CALL ... YIELD parsing (both languages)
- [ ] Create `ProcedureCall` logical plan node
- [ ] Implement procedure dispatcher
- [ ] Add built-in procedure discovery (SHOW PROCEDURES)

#### 4.2 Path Algorithms
**Goal:** Expose path algorithms as procedures.

- [ ] `algo.shortestPath(start, end, config) YIELD path, cost`
- [ ] `algo.allShortestPaths(start, end, config) YIELD path, cost`
- [ ] `algo.dijkstra(start, end, weightProperty) YIELD path, cost`
- [ ] `algo.astar(start, end, heuristic, weightProperty) YIELD path, cost`
- [ ] `algo.bfs(start, config) YIELD node, depth`
- [ ] `algo.dfs(start, config) YIELD node, depth`

#### 4.3 Centrality Algorithms
**Goal:** Expose centrality algorithms as procedures.

- [ ] `algo.pageRank(config) YIELD node, score`
- [ ] `algo.betweennessCentrality(config) YIELD node, score`
- [ ] `algo.closenessCentrality(config) YIELD node, score`
- [ ] `algo.degreeCentrality(config) YIELD node, score`
- [ ] `algo.eigenvectorCentrality(config) YIELD node, score`

#### 4.4 Community Detection
**Goal:** Expose community algorithms as procedures.

- [ ] `algo.louvain(config) YIELD node, community`
- [ ] `algo.labelPropagation(config) YIELD node, community`
- [ ] `algo.connectedComponents(config) YIELD node, component`
- [ ] `algo.stronglyConnectedComponents(config) YIELD node, component`
- [ ] `algo.triangleCount(config) YIELD node, triangles`
- [ ] `algo.localClusteringCoefficient(config) YIELD node, coefficient`

#### 4.5 Similarity Algorithms
**Goal:** Expose similarity algorithms as procedures.

- [ ] `algo.nodeSimilarity(config) YIELD node1, node2, similarity`
- [ ] `algo.jaccard(node1, node2, relationshipType) YIELD similarity`
- [ ] `algo.overlap(node1, node2, relationshipType) YIELD similarity`
- [ ] `algo.cosine(node1, node2, property) YIELD similarity`

---

### Phase 5: Query Optimization (Priority: Medium)

#### 5.1 Rule-Based Optimization
**Goal:** Implement comprehensive rewrite rules.

- [ ] **Predicate Pushdown**
  - [ ] Push through joins
  - [ ] Push through aggregations where safe
  - [ ] Push through graph expansions

- [ ] **Projection Pushdown**
  - [ ] Remove unused columns early
  - [ ] Push through joins and unions

- [ ] **Join Reordering**
  - [ ] Basic heuristic ordering
  - [ ] Cost-based join ordering (Phase 6)

- [ ] **Subquery Decorrelation**
  - [ ] Convert correlated EXISTS to semi-joins
  - [ ] Convert correlated IN to joins

- [ ] **Common Subexpression Elimination**
  - [ ] Identify shared CTE references
  - [ ] Materialize when beneficial

#### 5.2 Index Selection
**Goal:** Choose optimal indexes for queries.

- [ ] Implement index capability descriptions
- [ ] Match predicates to index capabilities
- [ ] Choose between index scan and full scan
- [ ] Support composite index selection

#### 5.3 Statistics & Cost Model
**Goal:** Build cost-based optimization foundation.

- [ ] Implement table statistics collection
- [ ] Cardinality estimation for predicates
- [ ] Join cardinality estimation
- [ ] Cost model for physical operators

---

### Phase 6: Testing & Validation (Priority: Critical, Ongoing)

#### 6.1 Parser Tests
- [ ] SQL syntax coverage tests (one per construct)
- [ ] Cypher syntax coverage tests (one per construct)
- [ ] Error message quality tests
- [ ] Negative tests (invalid syntax rejection)

#### 6.2 Planner Tests
- [ ] Logical plan structure tests
- [ ] Optimization rule tests
- [ ] Plan equivalence tests

#### 6.3 Execution Tests
- [ ] Operator unit tests
- [ ] Integration tests (end-to-end queries)
- [ ] Correctness tests (compare to reference DB)
- [ ] Performance regression tests

#### 6.4 Compliance Tests
- [ ] PostgreSQL compatibility tests
- [ ] openCypher TCK (Technology Compatibility Kit)

---

## Appendix A: File Locations

### Parser
- `crates/manifoldb-query/src/parser/sql.rs` - SQL parser
- `crates/manifoldb-query/src/parser/extensions.rs` - Graph/vector extensions
- `crates/manifoldb-query/src/parser/cypher.rs` - Cypher parser (to be created)

### AST
- `crates/manifoldb-query/src/ast/statement.rs` - Statement types
- `crates/manifoldb-query/src/ast/expr.rs` - Expression types
- `crates/manifoldb-query/src/ast/pattern.rs` - Graph pattern types

### Logical Plan
- `crates/manifoldb-query/src/plan/logical/node.rs` - Plan nodes
- `crates/manifoldb-query/src/plan/logical/mod.rs` - Module organization

### Physical Plan / Execution
- `crates/manifoldb-query/src/exec/operators/` - Physical operators
- `crates/manifoldb-query/src/exec/engine.rs` - Execution engine

### Optimization
- `crates/manifoldb-query/src/plan/optimize/` - Optimizer passes

### Tests
- `crates/manifoldb/tests/integration/` - Integration tests
- `crates/manifoldb-query/tests/` - Unit tests

---

## Appendix B: Priority Matrix

| Feature | Ablative Need | Complexity | Priority |
|---------|--------------|------------|----------|
| Basic SQL CRUD | High | Low | P0 |
| Joins (all types) | High | Medium | P0 |
| Aggregations | High | Medium | P0 |
| CTEs (non-recursive) | Medium | Low | P1 |
| Window Functions | Medium | High | P2 |
| Recursive CTEs | Low | High | P3 |
| MATCH patterns | High | Medium | P0 |
| CREATE/MERGE nodes | High | Medium | P1 |
| Graph algorithms | High | Medium | P1 |
| CALL ... YIELD | High | Medium | P1 |
| Vector search | High | Done | P0 |
| Full-text search | Low | High | P3 |

---

## Appendix C: Estimated Effort

### Phase 1: Foundation (2-3 weeks)
- Logical plan completion
- Expression system
- Type system foundation

### Phase 2: SQL Completion (4-6 weeks)
- Parser extensions
- Function library (iterative)
- Physical operators

### Phase 3: Cypher Completion (3-4 weeks)
- Parser for writing clauses
- Graph mutation operators
- Path algorithms

### Phase 4: Graph Algorithms (2-3 weeks)
- CALL/YIELD infrastructure
- Algorithm implementations (many exist in manifoldb-graph)

### Phase 5: Optimization (3-4 weeks)
- Rule-based optimizer
- Cost model foundation
- Index selection

### Phase 6: Testing (Ongoing)
- Continuous throughout development
- TCK compliance testing at end

**Total Estimated: 14-20 weeks for full implementation**

---

## References

- [LOGICAL_PLAN_SPECIFICATION.md](./LOGICAL_PLAN_SPECIFICATION.md) - Unified IR specification
- [COVERAGE_MATRICES.md](./COVERAGE_MATRICES.md) - Complete feature coverage tracking
- [PostgreSQL Documentation](https://www.postgresql.org/docs/current/)
- [openCypher Specification](https://opencypher.org/)
- [Memgraph MAGE](https://memgraph.com/docs/mage) - Graph algorithm reference
