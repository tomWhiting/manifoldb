# ManifoldDB Unified Architecture Plan

**Status:** In Progress
**Started:** December 2025
**Last Updated:** January 1, 2026

---

## Overview

This document covers the complete architectural evolution of ManifoldDB:

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Unified Entity API | ✅ Complete |
| 2 | Payload Indexing | ✅ Complete |
| 3 | Query Planner Integration | ✅ Complete |
| 4 | Complete SQL | ⚠️ ~95% Complete |
| 5 | Complete Cypher | ⚠️ ~40% Complete |
| 6 | Graph-Constrained Vector Search | ❌ Not Started |

---

## Completed Work

### Phase 1: Unified Entity API ✅

- `VectorData` enum (Dense, Sparse, Multi)
- `Entity.vectors` field with builder methods
- `ScoredEntity` and `ScoredId` structs
- `EntitySearchBuilder` fluent API
- 151+ tests passing

### Phase 2: Payload Indexing ✅

- `IndexType` enum (Equality, Range, Prefix)
- `IndexManager` with create/drop/lookup
- Automatic index maintenance in bulk operations
- 15 tests in `payload_index_tests.rs`

### Phase 3: Query Planner Integration ✅

- `Database::build_planner_catalog()` - exports indexes to planner
- `PlannerCatalog::merge()` - combines schema + payload indexes
- `execute_query_with_catalog()` - accepts external catalog
- `EXPLAIN` command - shows logical + physical plan trees
- 2 EXPLAIN tests added

---

## What's Left: Complete SQL

### Current SQL Support

| Feature | Status | Notes |
|---------|--------|-------|
| SELECT/FROM/WHERE | ✅ | Full support |
| JOINs (all types) | ✅ | Hash, Merge, NestedLoop algorithms |
| GROUP BY / HAVING | ✅ | Full support |
| Aggregates | ✅ | COUNT, SUM, AVG, MIN, MAX |
| ORDER BY / LIMIT / OFFSET | ✅ | Full support |
| DISTINCT | ✅ | Full support |
| UNION / INTERSECT / EXCEPT | ✅ | Including ALL variants |
| Subqueries (scalar, IN, EXISTS) | ✅ | Full support |
| CASE expressions | ✅ | Full support |
| MATCH clause (graph) | ✅ | Graph pattern matching |
| Vector distance (<->) | ✅ | Full support |
| EXPLAIN | ✅ | Shows logical + physical plans |
| CTEs (WITH clause) | ✅ | Non-recursive CTEs, multiple CTEs, CTE shadowing |

### Recently Completed SQL Features

#### CTEs (WITH clause) - ✅ Complete

```sql
WITH active_users AS (
    SELECT * FROM users WHERE status = 'active'
)
SELECT * FROM active_users WHERE age > 21
```

**Implementation (Done):**
- [x] Add `WithClause` to AST (`ast/statement.rs`)
- [x] Parse WITH in `parser/sql.rs`
- [x] Add CTE resolution in `PlanBuilder`
- [x] Inline CTE plans when referenced
- [x] Support multiple CTEs in one WITH clause
- [x] CTE names shadow table names (standard SQL)
- [x] Later CTEs can reference earlier CTEs
- [ ] Recursive CTEs (out of scope, separate task)

**Files modified:**
- `manifoldb-query/src/ast/statement.rs` - Added `WithClause` struct
- `manifoldb-query/src/ast/mod.rs` - Re-exported `WithClause`
- `manifoldb-query/src/parser/sql.rs` - Added `convert_with_clause()`
- `manifoldb-query/src/plan/logical/builder.rs` - Added `cte_plans` HashMap

**Tests:** 17 tests (8 parser + 9 plan builder)
**Review:** `docs/reviews/sql-cte-review.md`

---

### Missing SQL Features

#### 1. Window Functions - Large Effort

```sql
SELECT name, salary,
       ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rank,
       SUM(salary) OVER (PARTITION BY dept) as dept_total
FROM employees
```

**Implementation:**
- [ ] Add `WindowExpr` to `LogicalExpr`
- [ ] Add window function types (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, NTILE)
- [ ] Add aggregate window functions (SUM, AVG, COUNT over windows)
- [ ] Parse OVER clause with PARTITION BY / ORDER BY / frame spec
- [ ] Add `WindowOp` execution operator
- [ ] Handle frame specifications (ROWS BETWEEN, RANGE BETWEEN)

**Files to modify:**
- `manifoldb-query/src/ast/expr.rs` - WindowSpec already exists, needs expansion
- `manifoldb-query/src/plan/logical/expr.rs` - Add WindowExpr variant
- `manifoldb-query/src/plan/logical/builder.rs` - Build window expressions
- `manifoldb-query/src/exec/operators/` - New `window.rs` operator

**Window functions to support:**
| Function | Type | Complexity |
|----------|------|------------|
| ROW_NUMBER() | Ranking | Simple |
| RANK() | Ranking | Simple |
| DENSE_RANK() | Ranking | Simple |
| NTILE(n) | Ranking | Simple |
| LAG(col, n) | Value | Medium |
| LEAD(col, n) | Value | Medium |
| FIRST_VALUE(col) | Value | Medium |
| LAST_VALUE(col) | Value | Medium |
| SUM() OVER | Aggregate | Medium |
| AVG() OVER | Aggregate | Medium |
| COUNT() OVER | Aggregate | Medium |

#### 2. Correlated Subqueries - Medium Effort

```sql
SELECT * FROM orders o
WHERE amount > (SELECT AVG(amount) FROM orders WHERE user_id = o.user_id)
```

**Current state:** IR supports correlation, execution may not handle it properly.

**Implementation:**
- [ ] Detect correlated references in subquery planning
- [ ] Implement nested loop execution for correlated subqueries
- [ ] Consider decorrelation optimization (convert to JOIN)

---

## What's Left: Complete Cypher

### Current Cypher Support

| Feature | Status | Notes |
|---------|--------|-------|
| MATCH patterns | ✅ | Node and relationship patterns |
| Variable-length paths | ✅ | `[:REL*1..3]` syntax |
| WHERE filtering | ✅ | Property predicates |
| RETURN clause | ✅ | With expressions |
| Pattern parsing | ✅ | Via SQL MATCH extension |

### Missing Cypher Features

#### 1. Standalone Cypher Parser - Large Effort

Currently Cypher is parsed as a SQL extension (`SELECT ... MATCH ...`). Full Cypher needs its own parser.

```cypher
MATCH (u:User)-[:FOLLOWS]->(f:User)
WHERE u.name = 'Alice'
RETURN f.name, f.age
```

**Implementation:**
- [ ] Create `parser/cypher.rs` with dedicated lexer/parser
- [ ] Parse standalone Cypher (not embedded in SQL)
- [ ] Convert to `LogicalPlan` (reuse existing plan nodes)

**Cypher grammar elements:**
- MATCH clause with patterns
- OPTIONAL MATCH
- WHERE clause
- WITH clause (projection/aggregation mid-query)
- RETURN clause
- ORDER BY, SKIP, LIMIT
- UNWIND for list expansion
- CALL for procedures

#### 2. WITH Clause - Medium Effort

```cypher
MATCH (u:User)
WITH u, count(*) as cnt
WHERE cnt > 5
RETURN u.name
```

**Implementation:**
- [ ] Add WITH to Cypher AST
- [ ] WITH acts as a pipeline stage (project + optional aggregate)
- [ ] Chain multiple WITH clauses

#### 3. OPTIONAL MATCH - Medium Effort

```cypher
MATCH (u:User)
OPTIONAL MATCH (u)-[:LIKES]->(p:Post)
RETURN u.name, p.title
```

**Implementation:**
- [ ] Parse OPTIONAL MATCH
- [ ] Convert to LEFT JOIN in logical plan
- [ ] Handle NULL for unmatched patterns

#### 4. Write Operations - Large Effort

```cypher
CREATE (u:User {name: 'Alice', age: 30})
MERGE (u:User {email: 'alice@example.com'})
SET u.verified = true
DELETE u
```

**Implementation:**
- [ ] CREATE - Insert new nodes/relationships
- [ ] MERGE - Upsert (create if not exists)
- [ ] SET - Update properties
- [ ] DELETE/DETACH DELETE - Remove nodes/relationships
- [ ] REMOVE - Remove properties/labels

#### 5. Path Expressions - Medium Effort

```cypher
MATCH p = (a)-[*]->(b)
RETURN length(p), nodes(p), relationships(p)
```

**Implementation:**
- [ ] Path variable binding
- [ ] Path functions: `length()`, `nodes()`, `relationships()`
- [ ] `shortestPath()` and `allShortestPaths()`

#### 6. List Operations - Medium Effort

```cypher
UNWIND [1, 2, 3] AS x RETURN x
MATCH (u) WHERE u.age IN [25, 30, 35] RETURN u
WITH [1, 2, 3] AS nums RETURN [x IN nums WHERE x > 1] AS filtered
```

**Implementation:**
- [ ] UNWIND clause
- [ ] List comprehensions `[x IN list WHERE pred | expr]`
- [ ] List functions: `size()`, `head()`, `tail()`, `range()`

---

## Priority Order for Implementation

### Tier 1: High Value, Delegatable to Sub-Agents

| Task | Effort | Why |
|------|--------|-----|
| CTEs (WITH clause) | Medium | Well-defined, clear scope, useful |
| OPTIONAL MATCH | Medium | Clear mapping to LEFT JOIN |
| UNWIND | Small | Simple list expansion |

### Tier 2: High Value, Needs Design

| Task | Effort | Why |
|------|--------|-----|
| Cypher WITH clause | Medium | Query chaining, powerful feature |
| Window functions (basic) | Medium | Start with ROW_NUMBER, RANK |
| Cypher write ops | Large | CREATE/MERGE/SET/DELETE |

### Tier 3: Larger Undertakings

| Task | Effort | Why |
|------|--------|-----|
| Standalone Cypher parser | Large | Unlocks full Cypher |
| Window functions (full) | Large | Frame specs, all functions |
| Graph-constrained vector search | Large | Novel feature |
| Cost-based optimizer | Medium | Performance optimization |

---

## Sub-Agent Task Breakdown

### CTE Implementation (Good for sub-agent)

**Context needed:**
- `ast/statement.rs` - Where to add WithClause
- `parser/sql.rs` - How existing parsing works
- `plan/logical/builder.rs` - How build_select works

**Acceptance criteria:**
- Parse `WITH name AS (SELECT ...) SELECT ... FROM name`
- Build correct LogicalPlan
- Execute and return correct results
- Support multiple CTEs
- Tests for basic, multiple, and nested CTEs

### OPTIONAL MATCH (Good for sub-agent)

**Context needed:**
- Current MATCH parsing in `parser/extensions.rs`
- How MATCH converts to LogicalPlan
- LEFT JOIN implementation

**Acceptance criteria:**
- Parse `OPTIONAL MATCH` in Cypher/SQL
- Convert to LEFT JOIN in plan
- NULLs for unmatched patterns
- Tests for optional match scenarios

### Window Functions - ROW_NUMBER (Good for sub-agent)

**Context needed:**
- `ast/expr.rs` - WindowSpec already exists
- `plan/logical/expr.rs` - Where to add WindowExpr
- `exec/operators/sort.rs` - For ordering within partitions

**Acceptance criteria:**
- Parse `ROW_NUMBER() OVER (PARTITION BY x ORDER BY y)`
- Build WindowExpr in logical plan
- Implement WindowOp operator
- Tests for basic partitioning and ordering

---

## Architecture Notes

### Query Pipeline

```
SQL/Cypher Text
      ↓
   Parser (sql.rs / cypher.rs)
      ↓
    AST (Statement, Expr)
      ↓
  PlanBuilder
      ↓
  LogicalPlan (Scan, Filter, Project, Join, Aggregate, etc.)
      ↓
  Optimizer (predicate pushdown, projection pushdown, index selection)
      ↓
  PhysicalPlanner
      ↓
  PhysicalPlan (FullScan, HashJoin, HashAggregate, etc.)
      ↓
  Executor (operator tree)
      ↓
  ResultSet
```

### Key Files

| Purpose | File |
|---------|------|
| SQL AST | `manifoldb-query/src/ast/statement.rs`, `expr.rs` |
| SQL Parser | `manifoldb-query/src/parser/sql.rs`, `extensions.rs` |
| Logical Plan | `manifoldb-query/src/plan/logical/node.rs`, `expr.rs` |
| Plan Builder | `manifoldb-query/src/plan/logical/builder.rs` |
| Physical Plan | `manifoldb-query/src/plan/physical/node.rs` |
| Optimizer | `manifoldb-query/src/plan/optimize/` |
| Execution | `manifoldb-query/src/exec/operators/` |

---

## Testing Strategy

Each new feature should have:
1. **Parser tests** - Verify parsing produces correct AST
2. **Plan builder tests** - Verify AST converts to correct LogicalPlan
3. **Integration tests** - End-to-end SQL execution
4. **Edge case tests** - NULLs, empty results, large datasets

Test file locations:
- `manifoldb-query/tests/parser_tests.rs`
- `manifoldb-query/src/plan/logical/builder.rs` (inline tests)
- `manifoldb/tests/integration/sql.rs`
- `manifoldb/tests/integration/graph.rs`

---

*Last updated: January 1, 2026 - Completed payload index integration and EXPLAIN command*
