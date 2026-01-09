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

### Completed January 2026

- ✅ **WITH RECURSIVE** - Recursive CTEs with working table management
- ✅ **Basic Window Functions** - ROW_NUMBER, RANK, DENSE_RANK with PARTITION BY/ORDER BY
- ✅ **SQL Core Functions** - String (position, concat_ws, ltrim, rtrim, replace, split_part, regexp_match, regexp_replace, format), Numeric (exp, ln, log, log10, trig functions, degrees, radians, sign, pi), Date/Time (date_part, extract, date_trunc, to_timestamp, to_date, to_char)
- ✅ **Variable-Length Paths** - Full execution with BFS traversal and cycle detection
- ✅ **UNWIND Clause** - List expansion to rows
- ✅ **Cypher CREATE/MERGE** - Parsing and logical/physical planning (execution TBD)
- ✅ **Cypher SET/DELETE/REMOVE** - Parsing and logical/physical planning (execution TBD)
- ✅ **CALL/YIELD Infrastructure** - Procedure registry, ProcedureCall plan node, algo.pageRank, algo.shortestPath
- ✅ **Window Value Functions** - LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE
- ✅ **Cypher List Comprehensions** - `[x IN list WHERE pred | expr]` with list functions (range, size, head, tail, last, reverse)
- ✅ **Centrality Procedures** - algo.betweennessCentrality, algo.closenessCentrality, algo.degreeCentrality, algo.eigenvectorCentrality
- ✅ **Community Detection Procedures** - algo.labelPropagation, algo.connectedComponents, algo.stronglyConnectedComponents
- ✅ **Path Algorithm Procedures** - algo.dijkstra, algo.astar, algo.allShortestPaths, algo.sssp
- ✅ **Window Frame Clause** - ROWS/RANGE BETWEEN with all bound types (UNBOUNDED PRECEDING/FOLLOWING, n PRECEDING/FOLLOWING, CURRENT ROW)
- ✅ **JSON Functions†** - json_extract_path, jsonb_set, json_build_object, json_build_array, jsonb_strip_nulls (11 functions)
- ✅ **Cypher Map Projections†** - `node{.property, key: expr, .*}` syntax for property extraction and transformation
- ✅ **Aggregate Window Functions†** - SUM/AVG/COUNT/MIN/MAX OVER with frame awareness

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
- [x] Implement missing logical plan nodes:
  - [x] `Window` - Window function evaluation ✅ Jan 2026
  - [x] `RecursiveCTE` - Recursive CTE evaluation ✅ Jan 2026
  - [x] `Procedure` - CALL ... YIELD procedure invocation ✅ Jan 2026
  - [ ] `ShortestPath` - Graph shortest path computation
  - [ ] `AllShortestPaths` - All shortest paths computation
  - [x] `GraphCreate` - CREATE for nodes/edges ✅ Jan 2026
  - [x] `GraphMerge` - MERGE with ON CREATE/ON MATCH ✅ Jan 2026
  - [x] `GraphSet` - SET property updates ✅ Jan 2026
  - [x] `GraphRemove` - REMOVE property/label ✅ Jan 2026
  - [x] `GraphDelete` - DELETE nodes/edges ✅ Jan 2026
  - [x] `DetachDelete` - DETACH DELETE ✅ Jan 2026 (part of GraphDelete)
  - [x] `Foreach` - Cypher FOREACH iteration ✅ Jan 2026
- [ ] Add type system for plan nodes (input/output schemas)
- [ ] Implement plan validation and sanity checks

#### 1.2 Expression System Completion
**Goal:** Complete expression evaluation for both languages.

- [x] Extend `Expr` enum for missing expression types:
  - [x] Window function expressions with frame specs ✅ Jan 2026
  - [x] List comprehensions `[x IN list WHERE ... | expr]` ✅ Jan 2026
  - [x] Map projections `node{.prop1, .prop2}` ✅ Jan 2026
  - [ ] Pattern expressions (for EXISTS subqueries)
  - [ ] Temporal literals and operations
  - [ ] Spatial point literals
- [ ] Implement expression type inference
- [ ] Add expression simplification/optimization

---

### Phase 2: SQL Completion (Priority: High)

#### 2.1 Parser Extensions
**Goal:** Parse complete PostgreSQL-compatible SQL.

- [x] **CTEs** (basic support complete)
  - [x] Non-recursive CTEs with multiple CTE support
  - [x] WITH RECURSIVE support ✅ Jan 2026
  - [ ] SEARCH DEPTH/BREADTH FIRST
  - [ ] CYCLE detection clause
  - [ ] MATERIALIZED/NOT MATERIALIZED hints

- [ ] **Window Functions**
  - [ ] Named window definitions (WINDOW w AS ...)
  - [x] Frame clause support (ROWS/RANGE BETWEEN) ✅ Jan 2026
  - [ ] GROUPS frame type
  - [ ] Frame exclusion (EXCLUDE CURRENT ROW, etc.)
  - [ ] FILTER clause on window functions

- [ ] **Advanced SELECT**
  - [ ] DISTINCT ON (expression, ...)
  - [ ] FETCH FIRST ... WITH TIES
  - [ ] TABLESAMPLE clause
  - [ ] LATERAL subqueries

- [x] **DDL Extensions** (partial)
  - [x] ALTER TABLE (ADD/DROP/ALTER COLUMN) ✅ Jan 2026
  - [ ] ALTER INDEX
  - [x] CREATE VIEW / DROP VIEW ✅ Jan 2026 (ALTER VIEW not implemented)
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

- [x] **String Functions** (Tier 1) - Mostly complete ✅ Jan 2026
  - [x] length, char_length, octet_length
  - [x] substring, substr, position, strpos
  - [x] upper, lower, initcap
  - [x] ltrim, rtrim, btrim, trim
  - [ ] lpad, rpad
  - [x] concat, concat_ws
  - [x] replace, translate
  - [x] split_part, string_to_array
  - [x] format
  - [x] regexp_match, regexp_replace, regexp_split_to_array

- [x] **Numeric Functions** (Tier 1) - Complete ✅ Jan 2026
  - [x] abs, sign, ceil, floor, round, trunc
  - [x] sqrt, cbrt, power, exp, ln, log, log10
  - [x] mod, div
  - [x] sin, cos, tan, asin, acos, atan, atan2
  - [x] degrees, radians
  - [x] random, setseed

- [x] **Date/Time Functions** (Tier 1 & 2) - Complete ✅ Jan 2026
  - [x] now, current_timestamp, current_date, current_time
  - [x] date_part, extract, date_trunc
  - [x] to_timestamp, to_date, to_char
  - [x] age, date_add, date_subtract ✅ Complete (Jan 2026)
  - [x] make_timestamp, make_date, make_time ✅ Complete (Jan 2026)
  - [x] timezone ✅ Complete (Jan 2026)

- [x] **Aggregate Functions** (Tier 1 & 2) - Complete ✅ Jan 2026
  - [x] count, sum, avg, min, max ✅ Complete
  - [x] array_agg, string_agg ✅ Complete (Jan 2026)
  - [x] json_agg, jsonb_agg, json_object_agg ✅ Complete (Jan 2026)
  - [x] stddev, stddev_pop, variance, var_pop ✅ Complete (Jan 2026)
  - [x] percentileCont, percentileDisc ✅ Complete (Jan 2026)
  - [ ] bool_and, bool_or, every
  - [ ] FILTER clause on aggregates

- [x] **Window Functions** (Tier 2) - Complete ✅ Jan 2026
  - [x] row_number, rank, dense_rank ✅ Jan 2026
  - [ ] ntile
  - [x] lag, lead ✅ Jan 2026
  - [x] first_value, last_value, nth_value ✅ Jan 2026
  - [ ] percent_rank, cume_dist
  - [x] Any aggregate as window function ✅ Jan 2026

- [x] **JSON Functions** (Tier 2) - Core functions complete ✅ Jan 2026
  - [ ] Operators: ->, ->>, #>, #>>, @>, <@, ?, ?|, ?&
  - [x] json_extract_path, jsonb_extract_path ✅ Jan 2026
  - [x] json_build_object, json_build_array ✅ Jan 2026
  - [ ] json_each, json_each_text
  - [ ] json_array_elements
  - [x] jsonb_set, jsonb_insert, jsonb_strip_nulls ✅ Jan 2026
  - [ ] jsonb_path_query, jsonb_path_exists

- [x] **Array Functions** (Tier 2) - Core functions complete ✅ Jan 2026
  - [x] array_length, cardinality ✅ Jan 2026
  - [x] array_append, array_prepend, array_cat ✅ Jan 2026
  - [x] array_position, array_positions ✅ Jan 2026
  - [x] array_remove, array_replace ✅ Jan 2026
  - [x] unnest (scalar mode) ✅ Jan 2026
  - [ ] Subscript access array[n]

- [x] **Type Conversion** (Tier 1) - Core complete
  - [x] CAST(expr AS type) ✅ Complete
  - [x] :: operator ✅ Complete
  - [ ] to_text, to_number, to_boolean

#### 2.3 Logical Plan Generation
**Goal:** Generate correct logical plans from SQL AST.

- [ ] Implement SQL → Logical Plan translation for all new constructs
- [ ] Handle correlated subqueries correctly
- [ ] Implement proper scoping for CTEs and subqueries
- [ ] Add LATERAL join support in planner

#### 2.4 Physical Operators
**Goal:** Implement missing physical operators.

- [x] **WindowOp** - Window function evaluation ✅ Jan 2026
  - [x] Partition management
  - [x] Frame calculation (ROWS/RANGE BETWEEN) ✅ Jan 2026
  - [x] Support ranking functions (row_number, rank, dense_rank)

- [x] **RecursiveCTEOp** - Recursive CTE execution ✅ Jan 2026
  - [x] Working table management
  - [x] Cycle detection
  - [ ] Depth/breadth-first ordering

- [ ] **IndexNestedLoopJoinOp** - Index-accelerated joins
- [ ] **SortMergeJoinOp** - Sort-merge join implementation
- [ ] **HashAggregateOp** - Complete HAVING support (already partial)

---

### Phase 3: Cypher Completion (Priority: High)

#### 3.1 Parser Extensions
**Goal:** Parse complete openCypher queries.

- [x] **Writing Clauses** - Parsing + planning complete ✅ Jan 2026
  - [x] CREATE (nodes and relationships) ✅ Jan 2026
  - [x] MERGE with ON CREATE SET and ON MATCH SET ✅ Jan 2026
  - [x] SET (properties and labels) ✅ Jan 2026
  - [x] REMOVE (properties and labels) ✅ Jan 2026
  - [x] DELETE and DETACH DELETE ✅ Jan 2026
  - [x] FOREACH ✅ Jan 2026

- [ ] **Reading Clauses**
  - [ ] MANDATORY MATCH (optional, Neo4j extension)
  - [ ] Full label expressions (:Label1|Label2, :Label1&Label2)

- [x] **Path Functions** - Variable-length paths complete ✅ Jan 2026
  - [ ] shortestPath() pattern function
  - [ ] allShortestPaths() pattern function
  - [x] Variable-length path execution ✅ Jan 2026

- [x] **Subqueries** (parsing + planning complete) ✅ Jan 2026
  - [x] EXISTS { } subquery ✅ Jan 2026
  - [x] COUNT { } subquery ✅ Jan 2026
  - [x] CALL { } subquery (inline) ✅ Jan 2026

- [ ] **Advanced Patterns**
  - [ ] Quantified path patterns (GQL)
  - [ ] Path pattern assignment
  - [ ] Multiple relationship types in pattern

#### 3.2 Expression Extensions
**Goal:** Support all Cypher expression forms.

- [x] List comprehensions: `[x IN list WHERE pred | expr]` ✅ Jan 2026
- [x] Map projections: `node{.prop1, .prop2, key: expr}` ✅ Jan 2026
- [x] Pattern comprehensions: `[(n)-[:REL]->(m) | m.name]` ✅ Jan 2026
- [x] CASE expressions (simple and searched) ✅ Complete
- [x] Parameter syntax ($param) ✅ Complete

#### 3.3 Function Library
**Goal:** Implement openCypher function library.

- [x] **Scalar Functions** - Mostly complete ✅ Jan 2026
  - [x] head, tail, last ✅ Jan 2026
  - [x] coalesce ✅ Complete
  - [x] size (for lists, strings) ✅ Jan 2026
  - [ ] length (for paths)
  - [x] type, labels, id, properties, keys ✅ Jan 2026
  - [ ] nodes, relationships
  - [ ] startNode, endNode
  - [x] toBoolean, toInteger, toFloat, toString ✅ Jan 2026

- [x] **Aggregating Functions** - Complete ✅ Jan 2026
  - [x] collect, count, sum, avg, min, max ✅ Complete
  - [x] percentileCont, percentileDisc ✅ Jan 2026
  - [x] stDev, stDevP ✅ Jan 2026

- [x] **List Functions** ✅ Jan 2026
  - [x] range, reverse ✅ Jan 2026
  - [x] reduce (fold operation) ✅ Jan 2026
  - [x] all, any, none, single (predicate tests) ✅ Jan 2026

- [ ] **String Functions**
  - [ ] Same as SQL tier + specific Cypher names
  - [ ] left, right

- [x] **Mathematical Functions** - Shares SQL implementation
  - [x] Same as SQL tier ✅ Jan 2026

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

- [x] Implement Cypher → Logical Plan translation for:
  - [x] MATCH with complex patterns
  - [x] CREATE patterns ✅ Jan 2026
  - [x] MERGE with conditions ✅ Jan 2026
  - [x] SET/REMOVE operations ✅ Jan 2026
  - [x] DELETE/DETACH DELETE ✅ Jan 2026
  - [x] WITH clause chaining
  - [x] UNION/UNION ALL
  - [x] CALL ... YIELD ✅ Jan 2026

#### 3.5 Physical Operators
**Goal:** Implement Cypher-specific operators.

- [x] **GraphCreateOp** - Create nodes and edges ✅ Jan 2026 (planning only)
- [x] **GraphMergeOp** - Merge with create/match logic ✅ Jan 2026 (planning only)
- [x] **GraphSetOp** - Set properties/labels ✅ Jan 2026 (planning only)
- [x] **GraphRemoveOp** - Remove properties/labels ✅ Jan 2026 (planning only)
- [x] **GraphDeleteOp** - Delete with referential checks ✅ Jan 2026 (planning only)
- [ ] **ShortestPathOp** - BFS-based shortest path
- [ ] **AllShortestPathsOp** - All shortest paths
- [x] **VariableLengthExpandOp** - Multi-hop expansion ✅ Jan 2026

---

### Phase 4: Graph Algorithms (Priority: Medium)

#### 4.1 CALL/YIELD Infrastructure
**Goal:** Implement procedure call framework.

- [x] Design procedure registry interface ✅ Jan 2026
- [x] Implement CALL ... YIELD parsing (both languages) ✅ Jan 2026
- [x] Create `ProcedureCall` logical plan node ✅ Jan 2026
- [x] Implement procedure dispatcher ✅ Jan 2026
- [ ] Add built-in procedure discovery (SHOW PROCEDURES)

#### 4.2 Path Algorithms
**Goal:** Expose path algorithms as procedures.

- [x] `algo.shortestPath(start, end, config) YIELD path, cost` ✅ Jan 2026
- [x] `algo.allShortestPaths(start, end, config) YIELD path, cost` ✅ Jan 2026
- [x] `algo.dijkstra(start, end, weightProperty) YIELD path, cost` ✅ Jan 2026
- [x] `algo.astar(start, end, heuristic, weightProperty) YIELD path, cost` ✅ Jan 2026
- [x] `algo.sssp(start, weightProperty) YIELD nodeId, distance` ✅ Jan 2026
- [ ] `algo.bfs(start, config) YIELD node, depth`
- [ ] `algo.dfs(start, config) YIELD node, depth`

#### 4.3 Centrality Algorithms
**Goal:** Expose centrality algorithms as procedures.

- [x] `algo.pageRank(config) YIELD node, score` ✅ Jan 2026
- [x] `algo.betweennessCentrality(config) YIELD node, score` ✅ Jan 2026
- [x] `algo.closenessCentrality(config) YIELD node, score` ✅ Jan 2026
- [x] `algo.degreeCentrality(config) YIELD node, score` ✅ Jan 2026
- [x] `algo.eigenvectorCentrality(config) YIELD node, score` ✅ Jan 2026

#### 4.4 Community Detection
**Goal:** Expose community algorithms as procedures.

- [ ] `algo.louvain(config) YIELD node, community`
- [x] `algo.labelPropagation(config) YIELD node, community` ✅ Jan 2026
- [x] `algo.connectedComponents(config) YIELD node, component` ✅ Jan 2026
- [x] `algo.stronglyConnectedComponents(config) YIELD node, component` ✅ Jan 2026
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
