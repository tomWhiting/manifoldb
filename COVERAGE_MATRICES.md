# ManifoldDB Query Coverage Matrices

This document tracks implementation status for all SQL and Cypher language features. Each feature is tracked across the query pipeline stages.

## Status Legend

| Status | Symbol | Meaning |
|--------|--------|---------|
| Not Started | ` ` | No implementation |
| Parsed | `P` | Parser recognizes syntax |
| AST Node | `A` | AST type defined |
| Logical Plan | `L` | Logical plan node exists |
| Physical Op | `O` | Physical operator implemented |
| Execution | `E` | Fully executable |
| Tested | `T` | Has integration tests |

**Full Implementation:** `PALOE T` (all stages complete with tests)

### Verification Notes

Items marked with **†** were implemented by automated agents and have unit tests, but have not yet been manually verified with end-to-end integration testing. The agent's automated review confirmed:
- All workspace tests pass
- No clippy warnings
- Proper NULL handling
- No `unwrap()`/`expect()` in library code

See `docs/reviews/` for detailed review documents.

---

## Implementation Dependencies

This section helps agents understand what order to implement features. Features listed first must be completed before their dependents.

### Core Infrastructure (Already Exists)

These components are in place and can be extended:

| Component | Location | Status |
|-----------|----------|--------|
| SQL Parser | `parser/sql.rs` | ✅ Extensible |
| Cypher Parser | `parser/extensions.rs` | ✅ Partial, embedded in SQL |
| AST Types | `ast/statement.rs`, `ast/expr.rs` | ✅ Extensible |
| LogicalPlan enum | `plan/logical/node.rs` | ✅ Add new variants |
| PlanBuilder | `plan/logical/builder.rs` | ✅ Add new build methods |
| PhysicalPlan enum | `plan/physical/node.rs` | ✅ Add new variants |
| Execution operators | `exec/operators/*.rs` | ✅ Add new operators |

### Feature Dependency Graph

```
Window Functions
├── Depends on: Aggregate (exists)
├── Depends on: Sort (exists)
├── WindowOp physical operator: ✅ Complete
├── Ranking functions (ROW_NUMBER, RANK, DENSE_RANK): ✅ Complete
├── Value functions (LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE): ✅ Complete
├── Frame clause (ROWS/RANGE/GROUPS BETWEEN): ✅ Complete (Jan 2026)
├── Frame exclusion (EXCLUDE CURRENT ROW/GROUP/TIES): ✅ Complete (Jan 2026)
├── Named window definitions (WINDOW w AS): ✅ Complete (Jan 2026)
├── FILTER clause on window functions: ✅ Complete (Jan 2026)
├── FILTER clause on aggregates: ✅ Complete (Jan 2026)
├── Aggregate as window (SUM, AVG, COUNT, MIN, MAX OVER): ✅ Complete (Jan 2026)
└── Status: ✅ All SQL-2003 window function features complete (Jan 2026)

Recursive CTEs
├── Depends on: Non-recursive CTEs (exists)
├── RecursiveCTE logical plan node: ✅ Complete
├── RecursiveCTEOp physical operator: ✅ Complete
└── Status: ✅ Complete with tests (Jan 2026)

Cypher Writing Clauses
├── CREATE: ✅ GraphCreate logical node + operator + EXECUTION complete (Jan 2026)
├── MERGE: ✅ GraphMerge node + operator + EXECUTION complete (Jan 2026)
├── SET: ✅ GraphSet logical node + physical plan + EXECUTION complete (Jan 2026)
├── DELETE: ✅ GraphDelete logical + physical + EXECUTION complete (Jan 2026)
├── REMOVE: ✅ GraphRemove logical node + operator + EXECUTION complete (Jan 2026)
├── FOREACH: ✅ GraphForeach logical + physical + EXECUTION complete (Jan 2026)
└── Status: ✅ CREATE, MERGE, SET, DELETE, REMOVE, FOREACH fully executable

CALL/YIELD Procedures
├── ProcedureCall logical node: ✅ Complete
├── Procedure registry infrastructure: ✅ Complete
├── PageRank/ShortestPath: ⚠️ Registered but execution not wired (returns EmptyOp)
├── Traversal (bfs, dfs): ⚠️ Registered but execution not wired
├── Centrality (betweenness, closeness, degree, eigenvector): ⚠️ Registered but execution not wired
├── Community (labelPropagation, connectedComponents, stronglyConnected): ⚠️ Registered but execution not wired
├── Path (dijkstra, astar, allShortestPaths, sssp): ⚠️ Registered but execution not wired
├── Similarity (nodeSimilarity, jaccard, overlap, cosine): ⚠️ Registered but execution not wired
└── Status: ⚠️ 20 procedures registered; helpers exist but executor integration missing (Jan 2026)

Variable-Length Paths (Execution)
├── Parser: ✅ Exists
├── Logical Plan: ✅ PathScan node exists
├── Physical Operator: ✅ GraphExpandOp with ExpandLength
├── Execution: ✅ BFS traversal with cycle detection
└── Status: ✅ Complete with integration tests

List Comprehensions
├── Parser: ✅ Complete (parse_list_or_comprehension)
├── AST: ✅ ListComprehension, ListLiteral variants
├── Logical Plan: ✅ LogicalExpr::ListComprehension/ListLiteral
├── Execution: ✅ evaluate_expr handles comprehensions
├── List Functions: ✅ range, size, head, tail, last, reverse
└── Status: ✅ Complete with tests (Jan 2026)

Physical Join Operators
├── NestedLoopJoinOp: ✅ Exists (CROSS, INNER, LEFT joins)
├── HashJoinOp: ✅ Exists (INNER joins with hash table)
├── MergeJoinOp: ✅ Exists (INNER joins on sorted inputs)
├── IndexNestedLoopJoinOp: ✅ Complete (Jan 2026) - Index-accelerated nested loop
├── SortMergeJoinOp: ✅ Complete (Jan 2026) - Full outer join support (LEFT, RIGHT, FULL)
└── Status: ✅ 5 join operator strategies available

HAVING Clause Enhancement
├── Basic HAVING: ✅ Exists
├── Complex expressions (AND, OR, comparisons): ✅ Complete (Jan 2026)
├── Aggregate resolution in HAVING: ✅ Complete (Jan 2026)
└── Status: ✅ Full HAVING support with complex expressions

Query Optimization
├── Predicate Pushdown: ✅ Exists
├── Expression Simplification: ✅ Complete (Jan 2026) - constant folding, boolean algebra, null propagation
├── Column Pruning: ✅ Exists
└── Status: ✅ Core optimization passes available

LATERAL Subqueries
├── Parser: ✅ Complete (Jan 2026) - LATERAL keyword recognized in FROM clause
├── AST: ✅ Complete (Jan 2026) - LateralSubquery variant in TableRef
├── Logical Plan: ✅ Complete (Jan 2026) - Reuses CallSubqueryNode
├── Execution: ✅ Complete (Jan 2026) - Reuses CallSubqueryOp
└── Status: ✅ Complete with tests - correlated inline table expressions
```

### Parallel Work Streams

These feature groups can be implemented independently:

1. **SQL Functions** - String, numeric, date functions can be added without conflicts
2. **Cypher Writing** - CREATE/MERGE/SET/DELETE are isolated from SQL features
3. **Window Functions** - ✅ Complete (Jan 2026) - all ranking/distribution functions implemented
4. **Graph Algorithms** - ⚠️ 20 procedures registered (Jan 2026); helpers exist but executor wiring needed for execution
5. **Variable-Length Paths** - ✅ Complete (Jan 2026)

---

# Part 1: SQL Coverage Matrix

## 1.1 SELECT Statement

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **Basic SELECT** |
| SELECT columns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| SELECT * | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| SELECT table.* | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| SELECT DISTINCT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| SELECT DISTINCT ON | | | | | | | PostgreSQL extension |
| Column aliases (AS) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| **FROM Clause** |
| Single table | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Table alias | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Subquery in FROM | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| Table function | ✓ | ✓ | | | | | Parsed only |
| LATERAL subquery | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Uses CallSubqueryOp |
| VALUES clause | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **JOIN Types** |
| INNER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| LEFT OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| RIGHT OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - SortMergeJoinOp |
| FULL OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - SortMergeJoinOp |
| CROSS JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| NATURAL JOIN | ✓ | ✓ | ✓ | | | | Needs physical op |
| JOIN ... USING | ✓ | ✓ | ✓ | | | | Needs physical op |
| JOIN ... ON | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Self join | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **WHERE Clause** |
| Basic predicates | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| AND/OR/NOT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Comparison operators | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| BETWEEN | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| IN (list) | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| IN (subquery) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - NULL semantics |
| EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| NOT EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| Scalar subquery in WHERE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - e.g. `WHERE price > (SELECT AVG(price) FROM products)` |
| Scalar subquery in SELECT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - e.g. `SELECT name, (SELECT MAX(price) FROM products) AS max_price` |
| LIKE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| ILIKE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| SIMILAR TO | ✓ | ✓ | | | | | Parsed only |
| Regex match (~ ~*) | ✓ | ✓ | | | | | Parsed only |
| IS NULL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| IS NOT NULL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| IS TRUE/FALSE | ✓ | ✓ | | | | | Parsed only |
| **GROUP BY Clause** |
| Basic grouping | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple columns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Expressions | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| ROLLUP | | | | | | | Not implemented |
| CUBE | | | | | | | Not implemented |
| GROUPING SETS | | | | | | | Not implemented |
| GROUP BY ALL | | | | | | | Not implemented |
| **HAVING Clause** |
| Basic HAVING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Complex expressions |
| **ORDER BY Clause** |
| Single column | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple columns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| ASC/DESC | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| NULLS FIRST/LAST | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| Ordinal reference | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| Expression | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **LIMIT/OFFSET** |
| LIMIT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| OFFSET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| FETCH FIRST n ROWS | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| WITH TIES | | | | | | | Not implemented |
| **Set Operations** |
| UNION | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| UNION ALL | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| INTERSECT | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| INTERSECT ALL | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| EXCEPT | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| EXCEPT ALL | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |

## 1.2 Common Table Expressions (WITH)

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| Basic WITH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - basic CTE execution |
| Multiple CTEs | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - chained CTEs |
| Column aliases | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| CTE reference in main | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - multiple refs supported |
| CTE shadowing tables | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - CTE takes precedence over base table |
| WITH RECURSIVE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete† |
| SEARCH DEPTH FIRST | | | | | | | Not implemented |
| SEARCH BREADTH FIRST | | | | | | | Not implemented |
| CYCLE detection | | | | | | | Not implemented |
| MATERIALIZED hint | | | | | | | Not implemented |
| NOT MATERIALIZED hint | | | | | | | Not implemented |

## 1.3 Window Functions

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **Window Specification** |
| OVER () | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| PARTITION BY | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| ORDER BY in OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| Named windows (WINDOW w AS) | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 - window inheritance |
| **Frame Clause** |
| ROWS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| RANGE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| GROUPS | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 - peer group counting |
| UNBOUNDED PRECEDING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| n PRECEDING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| CURRENT ROW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| n FOLLOWING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| UNBOUNDED FOLLOWING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| EXCLUDE CURRENT ROW | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| EXCLUDE GROUP | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| EXCLUDE TIES | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| EXCLUDE NO OTHERS | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 (default) |
| **FILTER Clause** |
| FILTER on window functions | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| FILTER on aggregates | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| **Ranking Functions** |
| row_number() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| dense_rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| ntile(n) | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| percent_rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| cume_dist() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| **Value Functions** |
| lag() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| lead() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| first_value() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| last_value() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| nth_value() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| **Aggregate as Window** |
| count() OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| sum() OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| avg() OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| min() OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| max() OVER | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |

## 1.4 INSERT Statement

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| INSERT INTO ... VALUES | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple rows | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Column list | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| INSERT ... SELECT | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| DEFAULT VALUES | ✓ | ✓ | ✓ | | | | Needs physical |
| ON CONFLICT DO NOTHING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| ON CONFLICT DO UPDATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| RETURNING | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |

## 1.5 UPDATE Statement

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| Basic UPDATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple columns | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| WHERE clause | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| FROM clause | ✓ | ✓ | ✓ | | | | Needs physical |
| Subquery in SET | ✓ | ✓ | ✓ | | | | Needs physical |
| RETURNING | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |

## 1.6 DELETE Statement

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| Basic DELETE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| WHERE clause | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| USING clause | ✓ | ✓ | ✓ | | | | Needs physical |
| RETURNING | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |

## 1.7 MERGE Statement

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| MERGE INTO | | | | | | | Not implemented |
| WHEN MATCHED | | | | | | | Not implemented |
| WHEN NOT MATCHED | | | | | | | Not implemented |

## 1.8 DDL Statements

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **CREATE TABLE** |
| Basic CREATE TABLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Column constraints | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| NOT NULL | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| PRIMARY KEY | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| UNIQUE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| FOREIGN KEY | ✓ | ✓ | ✓ | | | | Needs physical |
| CHECK | ✓ | ✓ | ✓ | | | | Needs physical |
| DEFAULT | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| GENERATED AS | | | | | | | Not implemented |
| IF NOT EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| PARTITION BY | ✓ | ✓ | ✓ | ✓ | | | **Parsing complete (Jan 2026)** - AST support, storage TBD |
| INHERITS | | | | | | | Not implemented |
| **ALTER TABLE** |
| ADD COLUMN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ADD COLUMN IF NOT EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| DROP COLUMN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| DROP COLUMN IF EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ALTER COLUMN SET NOT NULL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ALTER COLUMN DROP NOT NULL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ALTER COLUMN SET DEFAULT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ALTER COLUMN DROP DEFAULT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ALTER COLUMN SET DATA TYPE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| RENAME COLUMN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| RENAME TABLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| ADD CONSTRAINT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| DROP CONSTRAINT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| IF EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| SET SCHEMA | | | | | | | Not implemented |
| **DROP TABLE** |
| Basic DROP TABLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| IF EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| CASCADE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| RESTRICT | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **TRUNCATE** |
| TRUNCATE TABLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - deletes all rows from table |
| RESTART IDENTITY | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - resets identity columns |
| CASCADE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - cascades to dependent tables |
| **CREATE INDEX** |
| Basic CREATE INDEX | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| UNIQUE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| IF NOT EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| USING method | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| WHERE predicate | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| INCLUDE columns | | | | | | | Not implemented |
| Concurrently | | | | | | | Not implemented |
| **DROP INDEX** |
| Basic DROP INDEX | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| IF EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| CASCADE | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **ALTER INDEX** |
| RENAME TO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - renames index |
| SET options | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - sets index options |
| RESET options | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - resets options to defaults |
| **VIEW** |
| CREATE VIEW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - creates view, stores SQL, expansion works |
| CREATE OR REPLACE VIEW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - replaces existing view |
| DROP VIEW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - removes view from schema |
| DROP VIEW IF EXISTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - silent if not exists |
| DROP VIEW CASCADE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete - CASCADE same as normal drop |
| SELECT FROM VIEW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Complete (Jan 2026)** - view expansion in query planning |
| MATERIALIZED VIEW | | | | | | | Not implemented |
| **SCHEMA** |
| CREATE SCHEMA | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| CREATE SCHEMA IF NOT EXISTS | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP SCHEMA | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP SCHEMA CASCADE | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| ALTER SCHEMA | | A | L | O | | | AST node and logical plan only (Jan 2026) |
| SET search_path | | | | | | | Not implemented |
| **FUNCTION** |
| CREATE FUNCTION | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| CREATE OR REPLACE FUNCTION | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP FUNCTION | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP FUNCTION IF EXISTS | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| **TRIGGER** |
| CREATE TRIGGER | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| CREATE OR REPLACE TRIGGER | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP TRIGGER | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |
| DROP TRIGGER IF EXISTS | P | A | L | O | | T | Parser, AST, LogicalPlan, PhysicalPlan with unit tests (Jan 2026) |

## 1.9 Transaction Statements

Transaction execution is implemented via the `Session` API which provides explicit transaction control.
Use `Session::new(&db)` to create a session, then execute transaction control statements.

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| BEGIN | P | A | L | O | E | T | Session API manages transaction state (Jan 2026) |
| START TRANSACTION | P | A | L | O | E | T | Alias for BEGIN, with isolation levels (Jan 2026) |
| COMMIT | P | A | L | O | E | T | Commits active transaction (Jan 2026) |
| ROLLBACK | P | A | L | O | E | T | Rolls back active transaction (Jan 2026) |
| SAVEPOINT | P | A | L | O | E | T | Creates named savepoint in transaction (Jan 2026) |
| RELEASE SAVEPOINT | P | A | L | O | E | T | Releases savepoint (Jan 2026) |
| ROLLBACK TO SAVEPOINT | P | A | L | O | E | T | Partial rollback to savepoint (Jan 2026) |
| SET TRANSACTION | P | A | L | O | E | T | Sets transaction characteristics (Jan 2026) |

**Session API Limitations:**
- Isolation levels are parsed but redb provides serializable isolation only
- Queries within explicit transactions use separate read transactions (no read-your-own-writes)
- Savepoints are tracked but full buffered write rollback requires simple operations

## 1.10 Utility Statements

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| EXPLAIN | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| EXPLAIN ANALYZE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - executes plan with stats |
| ANALYZE TABLE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - placeholder execution |
| VACUUM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - parsing with sqlparser 0.60 |
| COPY TO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - export with CSV/TEXT |
| COPY FROM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - import with CSV/TEXT |
| SET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - session variables |
| SET LOCAL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - transaction-scoped |
| SHOW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - view variable values |
| SHOW ALL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| RESET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - parsing with sqlparser 0.60 |

**Utility Statement Notes:**
- EXPLAIN ANALYZE supports FORMAT TEXT/JSON/XML/YAML, BUFFERS, TIMING, VERBOSE, COSTS options
- VACUUM supports FULL option; parsed with sqlparser 0.60
- COPY supports CSV/TEXT/BINARY formats with HEADER, DELIMITER, QUOTE, ESCAPE options
- SET/SHOW/RESET handle session configuration variables; RESET supports ALL and specific variables

## 1.11 Data Types

| Type | P | A | L | O | E | T | Notes |
|------|---|---|---|---|---|---|-------|
| **Numeric** |
| SMALLINT | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| INTEGER | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| BIGINT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| REAL | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| DOUBLE PRECISION | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| NUMERIC/DECIMAL | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| SERIAL | | | | | | | Not implemented |
| BIGSERIAL | | | | | | | Not implemented |
| **Character** |
| VARCHAR(n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| CHAR(n) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| TEXT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **Binary** |
| BYTEA | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Boolean** |
| BOOLEAN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| **Temporal** |
| DATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - DATE 'x' literals |
| TIME | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - TIME 'x' literals |
| TIMESTAMP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - TIMESTAMP 'x' literals |
| TIMESTAMPTZ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| INTERVAL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - INTERVAL 'x' to ISO 8601 |
| **JSON** |
| JSON | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| JSONB | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Other** |
| UUID | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ARRAY | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Vector** |
| VECTOR(n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| SPARSE_VECTOR | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| MULTI_VECTOR | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| BINARY_VECTOR | ✓ | ✓ | ✓ | ✓ | ✓ | | |

## 1.12 Expressions and Operators

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **Arithmetic** |
| + - * / | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| % (modulo) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ^ (power) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| @ (absolute) | ✓ | ✓ | | | | | |
| **Comparison** |
| = <> != | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| < <= > >= | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| **String** |
| \|\| (concat) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **JSON Operators** |
| -> | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ->> | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| #> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - JSON path extraction |
| #>> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - JSON path extraction as text |
| @> | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| <@ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ? | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Key existence check |
| ?\| | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Any key exists |
| ?& | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - All keys exist |
| **Array Operators** |
| [n] subscript | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - proper ArrayIndex impl |
| [n:m] slice | ✓ | ✓ | | | | | |
| @> (contains) | ✓ | ✓ | | | | | |
| <@ (contained) | ✓ | ✓ | | | | | |
| && (overlap) | ✓ | ✓ | | | | | |
| **Vector Operators** |
| <-> (Euclidean) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| <=> (Cosine) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| <#> (Inner product) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| <##> (Multi-vector) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Special** |
| CASE expression | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| CAST (::) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| COALESCE | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| NULLIF | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| GREATEST | ✓ | ✓ | | | | | |
| LEAST | ✓ | ✓ | | | | | |

## 1.13 Aggregate Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| COUNT(*) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| COUNT(expr) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| COUNT(DISTINCT) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| SUM | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| AVG | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| MIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| MAX | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| FILTER clause | ✓ | ✓ | ✓ | ✓ | ⚠️ | | **Partial** - HashAggregateOp works, SortMergeAggregateOp ignores filter (bug) |
| array_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| string_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| json_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| jsonb_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| json_object_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| jsonb_object_agg | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| bool_and | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| bool_or | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| every | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† SQL-standard synonym for bool_and |
| stddev | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| stddev_pop | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| variance | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| var_pop | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |

## 1.14 Scalar Functions

### String Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| length | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| upper | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| lower | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| substring | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| position | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| concat | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| concat_ws | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| trim | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| ltrim | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| rtrim | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| replace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| split_part | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| regexp_match | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| regexp_replace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| format | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| lpad | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| rpad | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |

### Numeric Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| abs | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ceil | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| floor | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| round | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Precision arg added † |
| trunc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Precision arg added † |
| sqrt | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| power | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| exp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| ln | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| log | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| log10 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| sin/cos/tan | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| asin/acos/atan | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| atan2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| degrees | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| radians | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| sign | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| pi | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| random | ✓ | ✓ | ✓ | ✓ | ✓ | | |

### Date/Time Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| now | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| current_timestamp | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| current_date | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| date_part/extract | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| date_trunc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| to_timestamp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| to_date | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| to_char | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| to_number | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - SQL format string parsing |
| to_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - SQL format string output |
| age | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| date_add | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| date_subtract | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| make_timestamp | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| make_date | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| make_time | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| timezone | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |

### JSON Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| json_extract_path | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_extract_path | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| json_extract_path_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_extract_path_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| json_build_object | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_build_object | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| json_build_array | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_build_array | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_set | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_insert | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| jsonb_strip_nulls | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| json_each | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand object to key/value pairs |
| jsonb_each | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand object to key/value pairs |
| json_each_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand object, values as text |
| jsonb_each_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand object, values as text |
| json_array_elements | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand array to elements |
| jsonb_array_elements | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand array to elements |
| json_array_elements_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand array, elements as text |
| jsonb_array_elements_text | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Expand array, elements as text |
| json_object_keys | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Return keys of JSON object |
| jsonb_object_keys | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Return keys of JSONB object |
| jsonb_path_exists | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Check if JSON path returns items |
| jsonb_path_query | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Get all items matching path |
| jsonb_path_query_array | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Get matching items as array |
| jsonb_path_query_first | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † - Get first item matching path |

### Array Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| array_length | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| cardinality | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_append | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_prepend | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_cat | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_remove | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_replace | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_position | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| array_positions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † |
| unnest | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Added 2026-01 † (scalar mode) |

---

# Part 2: Cypher Coverage Matrix

## 2.1 Reading Clauses

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **MATCH** |
| Basic MATCH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Node pattern (n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Node with label (n:Label) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple labels | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Label expressions (:A\|B, :A&B, :!A) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - GQL/Cypher label operators |
| Node properties | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern -[]-> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern <-[]- | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern -[]- | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Edge type [:TYPE] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple edge types [:A\|B] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Match multiple relationship types |
| Edge properties | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Path assignment (p = ...) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Named path patterns |
| **Variable-Length Paths** |
| [*] (any length) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*n] (exact) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*m..n] (range) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*..n] (up to) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*n..] (at least) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| **GQL Quantified Paths** |
| {n} (exact, GQL) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - GQL-style exact quantifier |
| {n,m} (range, GQL) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - GQL-style range quantifier |
| + (one or more, GQL) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - GQL-style one-or-more |
| ? (zero or one, GQL) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - GQL-style optional |
| **OPTIONAL MATCH** |
| Basic OPTIONAL MATCH | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **MANDATORY MATCH** |
| MANDATORY MATCH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Neo4j extension for strict matching |
| **WHERE** |
| Basic predicates | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Pattern predicates | ✓ | ✓ | | | | | |
| EXISTS { } subquery | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| COUNT { } subquery | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| CALL { } subquery | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **Partial (Jan 2026)** - uncorrelated complete, correlated variable binding infrastructure added (variable_bindings in ExecutionContext) |
| **Path Functions** |
| shortestPath() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Pattern function complete with BFS execution (Jan 2026)† |
| allShortestPaths() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Pattern function complete with BFS execution (Jan 2026)† |
| nodes(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| relationships(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| length(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |

## 2.2 Writing Clauses

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **CREATE** |
| CREATE node | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| CREATE relationship | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| CREATE with properties | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| MATCH + CREATE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) - Match nodes then create relationships |
| **MERGE** |
| MERGE node | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| MERGE relationship | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| ON CREATE SET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| ON MATCH SET | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| **SET** |
| SET property | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Single and multiple properties |
| SET properties (+=) | | | | | | | Not implemented |
| SET label | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - Add labels to nodes |
| **REMOVE** |
| REMOVE property | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| REMOVE label | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| **DELETE** |
| DELETE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| DETACH DELETE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| **FOREACH** |
| FOREACH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |

## 2.3 Projecting Clauses

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **RETURN** |
| Basic RETURN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| RETURN DISTINCT | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Aliases (AS) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| RETURN * | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **WITH** |
| Basic WITH | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| WITH WHERE | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| WITH ORDER BY | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| WITH aggregation | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **UNWIND** |
| Basic UNWIND | ✓ | ✓ | ✓ | ✓† | ✓† | ✓† | Agent impl |
| **ORDER BY** |
| Single column | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple columns | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ASC/DESC | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ASCENDING/DESCENDING | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **SKIP/LIMIT** |
| SKIP | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| LIMIT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |

## 2.4 Set Operations

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| UNION | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| UNION ALL | ✓ | ✓ | ✓ | ✓ | ✓ | | |

## 2.5 Procedure Calls

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| CALL procedure() | ✓ | ✓ | ✓† | ✓† | | | Infrastructure complete; execution returns EmptyOp (needs wiring) |
| YIELD columns | ✓ | ✓ | ✓† | ✓† | | | Infrastructure complete; execution returns EmptyOp (needs wiring) |
| YIELD * | ✓ | ✓ | ✓† | ✓† | | | Infrastructure complete; execution returns EmptyOp (needs wiring) |
| YIELD with WHERE | ✓ | ✓ | ✓† | ✓† | | | Infrastructure complete; execution returns EmptyOp (needs wiring) |
| SHOW PROCEDURES | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - lists registered procedures |

## 2.6 Operators

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **Comparison** |
| = <> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| < <= > >= | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| **Logical** |
| AND | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| OR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| NOT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| XOR | ✓ | ✓ | ✓ | | | | |
| **String** |
| + (concat) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| STARTS WITH | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ENDS WITH | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| CONTAINS | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| =~ (regex) | ✓ | ✓ | | | | | |
| **List** |
| [n] subscript | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† - proper ArrayIndex LogicalExpr |
| [m..n] slice | ✓ | ✓ | | | | | |
| IN | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| + (concat) | ✓ | ✓ | | | | | |
| **Null** |
| IS NULL | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| IS NOT NULL | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Arithmetic** |
| + - * / % ^ | ✓ | ✓ | ✓ | ✓ | ✓ | | |

## 2.7 Expressions

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| CASE (searched) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| CASE (simple) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| List comprehension | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| List literal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| Map projection | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| Pattern comprehension | ✓ | ✓ | ✓ | | | | Parsed + logical plan; execution returns empty (placeholder) |
| Parameters ($param) | ✓ | ✓ | ✓ | ✓ | ✓ | | |

## 2.8 Scalar Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| **Type/Property** |
| type(r) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| labels(n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| id(n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| properties(n) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| keys(map) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| **List Functions** |
| range() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| size() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| head() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| tail() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| last() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| reverse() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| **Null Handling** |
| coalesce() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Type Conversion** |
| toBoolean() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| toInteger() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| toFloat() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| toString() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| **Path Functions** |
| startNode(r) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| endNode(r) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| length(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| nodes(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| relationships(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |

## 2.9 Aggregating Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| count(*) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| count(expr) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| collect() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| sum() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| avg() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| min() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| max() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| percentileCont() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| percentileDisc() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| stDev() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| stDevP() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |

## 2.10 List Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| size() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| head() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| tail() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| last() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| reverse() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| range() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| reduce() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| all() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| any() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| none() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| single() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |

## 2.11 String Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| toUpper() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| toLower() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| trim() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ltrim() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| rtrim() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| substring() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| replace() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| split() | ✓ | ✓ | | | | | |
| left() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |
| right() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Agent impl Jan 2026 † |

## 2.12 Mathematical Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| abs() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ceil() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| floor() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| round() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| sign() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| sqrt() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| rand() | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| log() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Shares SQL impl † |
| log10() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Shares SQL impl † |
| exp() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Shares SQL impl † |
| e() | ✓ | ✓ | | | | | |
| sin/cos/tan/etc | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Shares SQL impl † |
| pi() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Shares SQL impl † |

## 2.13 Temporal Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| datetime() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ISO 8601 string & map construction (Jan 2026)† |
| localdatetime() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | No timezone variant (Jan 2026)† |
| date() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ISO 8601 date (Jan 2026)† |
| time() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ISO 8601 time with timezone (Jan 2026)† |
| localtime() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | No timezone variant (Jan 2026)† |
| duration() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ISO 8601 duration & map construction (Jan 2026)† |
| datetime.truncate() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Truncate to unit (Jan 2026)† |
| datetime + duration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Temporal arithmetic (Jan 2026)† |
| datetime - duration | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Temporal arithmetic (Jan 2026)† |
| datetime - datetime | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Returns duration (Jan 2026)† |
| timestamp() | | | | | | | Not impl - use datetime() |

## 2.14 Spatial Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| point() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Geographic (lat/lon) and cartesian (x/y) coordinates (Jan 2026)† |
| point.distance() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Haversine (geo) / Euclidean (cart) distance (Jan 2026)† |
| point.withinBBox() | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Bounding box containment test (Jan 2026)† |

---

# Part 3: Graph Algorithms Coverage

## 3.0 Traversal Algorithms

| Algorithm | Via CALL | Tested | Notes |
|-----------|----------|--------|-------|
| BFS (Breadth-First Search) | ✓ | ✓ | `algo.bfs()` - level-by-level traversal (Jan 2026)† |
| DFS (Depth-First Search) | ✓ | ✓ | `algo.dfs()` - branch exploration (Jan 2026)† |

## 3.1 Path Algorithms

| Algorithm | Via CALL | Via Query | Tested | Notes |
|-----------|----------|-----------|--------|-------|
| Shortest Path (BFS) | ✓ | ✓ | ✓ | `algo.shortestPath()` + `shortestPath()` pattern with execution (Jan 2026)† |
| Weighted Shortest Path (Dijkstra) | ✓ | | ✓ | `algo.dijkstra()` |
| A* Search | ✓ | | ✓ | `algo.astar()` |
| All Shortest Paths | ✓ | ✓ | ✓ | `algo.allShortestPaths()` + `allShortestPaths()` pattern with execution (Jan 2026)† |
| Single-Source Shortest Paths | ✓ | | ✓ | `algo.sssp()` |
| All-Pairs Shortest Paths | | | | Needs impl |

## 3.2 Centrality Algorithms

| Algorithm | Via CALL | Operator | Tested | Notes |
|-----------|----------|----------|--------|-------|
| PageRank | ✓ | ✓ | ✓ | `algo.pageRank()` |
| Betweenness Centrality | ✓ | ✓ | ✓ | `algo.betweennessCentrality()` |
| Closeness Centrality | ✓ | | ✓ | `algo.closenessCentrality()` |
| Degree Centrality | ✓ | | ✓ | `algo.degreeCentrality()` |
| Eigenvector Centrality | ✓ | | ✓ | `algo.eigenvectorCentrality()` |

## 3.3 Community Detection

| Algorithm | Via CALL | Operator | Tested | Notes |
|-----------|----------|----------|--------|-------|
| Louvain | ✓ | ✓ | ✓ | `algo.louvain()` |
| Label Propagation | ✓ | | ✓ | `algo.labelPropagation()` |
| Connected Components | ✓ | | ✓ | `algo.connectedComponents()` |
| Strongly Connected Components | ✓ | | ✓ | `algo.stronglyConnectedComponents()` |
| Triangle Count | ✓ | ✓ | ✓ | `algo.triangleCount()` |
| Local Clustering Coefficient | ✓ | ✓ | ✓ | `algo.localClusteringCoefficient()` |

## 3.4 Similarity Algorithms

| Algorithm | Via CALL | Tested | Notes |
|-----------|----------|--------|-------|
| Node Similarity | ✓ | ✓ | `algo.nodeSimilarity()` - Jaccard-based bulk similarity (Jan 2026)† |
| Jaccard | ✓ | ✓ | `algo.jaccard()` - Pairwise Jaccard coefficient (Jan 2026)† |
| Overlap | ✓ | ✓ | `algo.overlap()` - Pairwise Overlap coefficient (Jan 2026)† |
| Cosine | ✓ | ✓ | `algo.cosine()` - Property-based cosine similarity (Jan 2026)† |

---

# Part 4: Vector Operations Coverage

## 4.1 Vector Search

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **Index Types** |
| HNSW index | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| IVF index | ✓ | ✓ | ✓ | | | | Needs exec |
| Brute force | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Distance Metrics** |
| Euclidean (L2) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Cosine | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Inner Product | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Manhattan (L1) | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Hamming | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Chebyshev | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Vector Types** |
| Dense vector | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Sparse vector | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Multi-vector | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Binary vector | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Search Features** |
| k-NN search | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Filtered search | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Hybrid search | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ef_search parameter | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| n_probe parameter | ✓ | ✓ | ✓ | | | | |

## 4.2 Vector Collections

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| CREATE COLLECTION | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| DROP COLLECTION | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Vector dimensions | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Payload fields | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Index parameters | ✓ | ✓ | ✓ | ✓ | ✓ | | |

---

# Summary Statistics

## SQL Coverage

| Category | Total Features | Fully Implemented | Parsed Only | Not Started |
|----------|----------------|-------------------|-------------|-------------|
| SELECT Statement | 52 | 28 | 15 | 9 |
| CTEs | 10 | 6 | 0 | 4 |
| Window Functions | 32 | 30 | 0 | 2 |
| DML (INSERT/UPDATE/DELETE) | 20 | 12 | 5 | 3 |
| DDL | 44 | 32 | 3 | 9 |
| Transactions | 8 | 8 | 0 | 0 |
| Utility | 10 | 1 | 0 | 9 |
| Data Types | 25 | 22 | 0 | 3 |
| Expressions | 40 | 30 | 8 | 2 |
| Functions | 80 | 65 | 5 | 10 |

## Cypher Coverage

| Category | Total Features | Fully Implemented | Parsed Only | Not Started |
|----------|----------------|-------------------|-------------|-------------|
| Reading Clauses | 25 | 19 | 3 | 3 |
| Writing Clauses | 16 | 14 | 1 | 1 |
| Projecting Clauses | 15 | 10 | 3 | 2 |
| Operators | 25 | 20 | 5 | 0 |
| Expressions | 10 | 7 | 0 | 3 |
| Scalar Functions | 20 | 18 | 0 | 2 |
| Aggregating Functions | 12 | 10 | 0 | 2 |
| List Functions | 12 | 11 | 0 | 1 |
| String Functions | 10 | 10 | 0 | 0 |
| Math Functions | 15 | 13 | 0 | 2 |
| Temporal Functions | 10 | 7 | 0 | 3 |
| Spatial Functions | 5 | 3 | 0 | 2 |

## Vector Coverage

| Category | Total Features | Fully Implemented | Partial | Not Started |
|----------|----------------|-------------------|---------|-------------|
| Index Types | 3 | 2 | 1 | 0 |
| Distance Metrics | 6 | 6 | 0 | 0 |
| Vector Types | 4 | 2 | 2 | 0 |
| Search Features | 5 | 4 | 1 | 0 |
| Collections | 5 | 4 | 1 | 0 |

---

## References

- [QUERY_IMPLEMENTATION_ROADMAP.md](./QUERY_IMPLEMENTATION_ROADMAP.md) - Implementation phases
- [LOGICAL_PLAN_SPECIFICATION.md](./LOGICAL_PLAN_SPECIFICATION.md) - IR specification
- [PostgreSQL 17 Documentation](https://www.postgresql.org/docs/17/)
- [openCypher Specification v9](https://opencypher.org/)
