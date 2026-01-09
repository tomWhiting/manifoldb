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
├── Frame clause (ROWS/RANGE BETWEEN): ✅ Complete
├── Aggregate as window (SUM, AVG, COUNT, MIN, MAX OVER): ✅ Complete (Jan 2026)
└── Status: ✅ All core window functions complete (Jan 2026)

Recursive CTEs
├── Depends on: Non-recursive CTEs (exists)
├── RecursiveCTE logical plan node: ✅ Complete
├── RecursiveCTEOp physical operator: ✅ Complete
└── Status: ✅ Complete with tests (Jan 2026)

Cypher Writing Clauses
├── CREATE: ✅ GraphCreate logical node + operator complete
├── MERGE: ✅ GraphMerge node + operator complete
├── SET: ✅ GraphSet logical node + physical plan complete (Jan 2026)
├── DELETE: ✅ GraphDelete logical node + physical plan complete (Jan 2026)
├── REMOVE: ✅ GraphRemove logical node + physical plan complete (Jan 2026)
├── FOREACH: ✅ GraphForeach logical + physical plan complete (Jan 2026)
└── Status: ✅ Parsing + planning complete, execution TBD

CALL/YIELD Procedures
├── ProcedureCall logical node: ✅ Complete
├── Procedure registry infrastructure: ✅ Complete
├── PageRank/ShortestPath: ✅ Complete
├── Centrality (betweenness, closeness, degree, eigenvector): ✅ Complete
├── Community (labelPropagation, connectedComponents, stronglyConnected): ✅ Complete
├── Path (dijkstra, astar, allShortestPaths, sssp): ✅ Complete
└── Status: ✅ 13 procedures registered (Jan 2026)

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
```

### Parallel Work Streams

These feature groups can be implemented independently:

1. **SQL Functions** - String, numeric, date functions can be added without conflicts
2. **Cypher Writing** - CREATE/MERGE/SET/DELETE are isolated from SQL features
3. **Window Functions** - ✅ Complete (Jan 2026) - only NTILE remaining
4. **Graph Algorithms** - ✅ 13 procedures via CALL/YIELD (Jan 2026)
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
| LATERAL subquery | | | | | | | Not implemented |
| VALUES clause | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| **JOIN Types** |
| INNER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| LEFT OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| RIGHT OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| FULL OUTER JOIN | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
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
| IN (subquery) | ✓ | ✓ | ✓ | | | | Needs exec |
| EXISTS | ✓ | ✓ | ✓ | | | | Needs exec |
| NOT EXISTS | ✓ | ✓ | ✓ | | | | Needs exec |
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
| Basic HAVING | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
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
| Basic WITH | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| Multiple CTEs | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| Column aliases | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| CTE reference in main | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
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
| Named windows | | | | | | | Not implemented |
| **Frame Clause** |
| ROWS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| RANGE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| GROUPS | | | | | | | Not implemented |
| UNBOUNDED PRECEDING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| n PRECEDING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| CURRENT ROW | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| n FOLLOWING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| UNBOUNDED FOLLOWING | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
| EXCLUDE CURRENT ROW | | | | | | | Not implemented |
| EXCLUDE GROUP | | | | | | | Not implemented |
| EXCLUDE TIES | | | | | | | Not implemented |
| **Ranking Functions** |
| row_number() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| dense_rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl |
| ntile(n) | ✓ | ✓ | | | | | Needs impl |
| percent_rank() | | | | | | | Not implemented |
| cume_dist() | | | | | | | Not implemented |
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
| ON CONFLICT DO NOTHING | ✓ | ✓ | ✓ | | | | Needs physical |
| ON CONFLICT DO UPDATE | ✓ | ✓ | ✓ | | | | Needs physical |
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
| PARTITION BY | | | | | | | Not implemented |
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
| TRUNCATE TABLE | | | | | | | Not implemented |
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
| **VIEW** |
| CREATE VIEW | ✓ | ✓ | ✓ | ✓ | | ✓ | Parsing, planning, schema storage |
| CREATE OR REPLACE VIEW | ✓ | ✓ | ✓ | ✓ | | ✓ | Parsing, planning, schema storage |
| DROP VIEW | ✓ | ✓ | ✓ | ✓ | | ✓ | Parsing, planning, schema storage |
| DROP VIEW IF EXISTS | ✓ | ✓ | ✓ | ✓ | | ✓ | Parsing, planning, schema storage |
| DROP VIEW CASCADE | ✓ | ✓ | ✓ | ✓ | | | Parsing, planning (cascade not exec) |
| MATERIALIZED VIEW | | | | | | | Not implemented |
| **SCHEMA** |
| CREATE SCHEMA | | | | | | | Not implemented |
| DROP SCHEMA | | | | | | | Not implemented |
| SET search_path | | | | | | | Not implemented |

## 1.9 Transaction Statements

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| BEGIN | | | | | | | Not implemented |
| START TRANSACTION | | | | | | | Not implemented |
| COMMIT | | | | | | | Not implemented |
| ROLLBACK | | | | | | | Not implemented |
| SAVEPOINT | | | | | | | Not implemented |
| RELEASE SAVEPOINT | | | | | | | Not implemented |
| ROLLBACK TO SAVEPOINT | | | | | | | Not implemented |
| SET TRANSACTION | | | | | | | Not implemented |

## 1.10 Utility Statements

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| EXPLAIN | ✓ | ✓ | ✓ | ✓ | ✓ | | Needs tests |
| EXPLAIN ANALYZE | | | | | | | Not implemented |
| ANALYZE | | | | | | | Not implemented |
| VACUUM | | | | | | | Not implemented |
| COPY | | | | | | | Not implemented |
| SET | | | | | | | Not implemented |
| SHOW | | | | | | | Not implemented |
| RESET | | | | | | | Not implemented |

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
| DATE | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| TIME | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| TIMESTAMP | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| TIMESTAMPTZ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| INTERVAL | ✓ | ✓ | ✓ | | | | Needs exec |
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
| #> | ✓ | ✓ | | | | | |
| #>> | ✓ | ✓ | | | | | |
| @> | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| <@ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| ? | ✓ | ✓ | | | | | |
| ?\| | ✓ | ✓ | | | | | |
| ?& | ✓ | ✓ | | | | | |
| **Array Operators** |
| [n] subscript | ✓ | ✓ | ✓ | ✓ | ✓ | | |
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
| FILTER clause | ✓ | ✓ | ✓ | | | | Needs exec |
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
| json_each | | | | | | | Not impl |
| jsonb_each | | | | | | | Not impl |

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
| Node properties | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern -[]-> | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern <-[]- | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Edge pattern -[]- | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Edge type [:TYPE] | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Multiple edge types | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Edge properties | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **Variable-Length Paths** |
| [*] (any length) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*n] (exact) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*m..n] (range) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*..n] (up to) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| [*n..] (at least) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| **OPTIONAL MATCH** |
| Basic OPTIONAL MATCH | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| **WHERE** |
| Basic predicates | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete |
| Pattern predicates | ✓ | ✓ | | | | | |
| EXISTS { } subquery | ✓ | ✓ | ✓ | ✓ | | | P, A, L, T |
| COUNT { } subquery | ✓ | ✓ | ✓ | ✓ | | | P, A, L, T |
| CALL { } subquery | ✓ | ✓ | ✓ | ✓ | | | P, A, L, T |
| **Path Functions** |
| shortestPath() | ✓ | ✓ | ✓ | ✓ | | ✓ | Pattern function with L/O/T (Jan 2026) |
| allShortestPaths() | ✓ | ✓ | ✓ | ✓ | | ✓ | Pattern function with L/O/T (Jan 2026) |
| nodes(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| relationships(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |
| length(path) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026)† |

## 2.2 Writing Clauses

| Feature | P | A | L | O | E | T | Notes |
|---------|---|---|---|---|---|---|-------|
| **CREATE** |
| CREATE node | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| CREATE relationship | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| CREATE with properties | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| **MERGE** |
| MERGE node | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| MERGE relationship | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| ON CREATE SET | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| ON MATCH SET | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| **SET** |
| SET property | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| SET properties (+=) | | | | | | | Not implemented |
| SET label | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| **REMOVE** |
| REMOVE property | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| REMOVE label | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| **DELETE** |
| DELETE | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| DETACH DELETE | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning |
| **FOREACH** |
| FOREACH | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl, parsing+planning (Jan 2026) |

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
| CALL procedure() | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl |
| YIELD columns | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl |
| YIELD * | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl |
| YIELD with WHERE | ✓ | ✓ | ✓† | ✓† | | ✓† | Agent impl |

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
| [n] subscript | ✓ | ✓ | ✓ | ✓ | ✓ | | |
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
| Pattern comprehension | ✓ | ✓ | ✓ | | | ✓ | Parsed, logical plan complete (Jan 2026)† |
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
| left() | | | | | | | Not impl |
| right() | | | | | | | Not impl |

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
| datetime() | | | | | | | Not impl |
| localdatetime() | | | | | | | Not impl |
| date() | | | | | | | Not impl |
| time() | | | | | | | Not impl |
| localtime() | | | | | | | Not impl |
| duration() | | | | | | | Not impl |
| datetime.truncate() | | | | | | | Not impl |
| timestamp() | | | | | | | Not impl |

## 2.14 Spatial Functions

| Function | P | A | L | O | E | T | Notes |
|----------|---|---|---|---|---|---|-------|
| point() | | | | | | | Not impl |
| point.distance() | | | | | | | Not impl |
| point.withinBBox() | | | | | | | Not impl |

---

# Part 3: Graph Algorithms Coverage

## 3.1 Path Algorithms

| Algorithm | Via CALL | Via Query | Tested | Notes |
|-----------|----------|-----------|--------|-------|
| Shortest Path (BFS) | ✓ | ✓ | ✓ | `algo.shortestPath()` + `shortestPath()` pattern (Jan 2026) |
| Weighted Shortest Path (Dijkstra) | ✓ | | ✓ | `algo.dijkstra()` |
| A* Search | ✓ | | ✓ | `algo.astar()` |
| All Shortest Paths | ✓ | ✓ | ✓ | `algo.allShortestPaths()` + `allShortestPaths()` pattern (Jan 2026) |
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
| Louvain | | ✓ | | Has operator |
| Label Propagation | ✓ | | ✓ | `algo.labelPropagation()` |
| Connected Components | ✓ | | ✓ | `algo.connectedComponents()` |
| Strongly Connected Components | ✓ | | ✓ | `algo.stronglyConnectedComponents()` |
| Triangle Count | | | | Needs impl |
| Local Clustering Coefficient | | | | Needs impl |

## 3.4 Similarity Algorithms

| Algorithm | Via CALL | Tested | Notes |
|-----------|----------|--------|-------|
| Node Similarity | | | Needs impl |
| Jaccard | | | Needs impl |
| Overlap | | | Needs impl |
| Cosine | | | Needs impl |

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
| CTEs | 10 | 5 | 0 | 5 |
| Window Functions | 25 | 20 | 3 | 2 |
| DML (INSERT/UPDATE/DELETE) | 20 | 12 | 5 | 3 |
| DDL | 44 | 26 | 3 | 15 |
| Transactions | 8 | 0 | 0 | 8 |
| Utility | 10 | 1 | 0 | 9 |
| Data Types | 25 | 22 | 0 | 3 |
| Expressions | 40 | 30 | 8 | 2 |
| Functions | 80 | 30 | 35 | 15 |

## Cypher Coverage

| Category | Total Features | Fully Implemented | Parsed Only | Not Started |
|----------|----------------|-------------------|-------------|-------------|
| Reading Clauses | 25 | 17 | 3 | 5 |
| Writing Clauses | 15 | 6 | 0 | 9 |
| Projecting Clauses | 15 | 10 | 3 | 2 |
| Operators | 25 | 20 | 5 | 0 |
| Expressions | 10 | 5 | 0 | 5 |
| Scalar Functions | 20 | 10 | 5 | 5 |
| Aggregating Functions | 12 | 6 | 2 | 4 |
| List Functions | 12 | 8 | 0 | 4 |
| String Functions | 10 | 6 | 2 | 2 |
| Math Functions | 15 | 13 | 0 | 2 |
| Temporal Functions | 10 | 0 | 0 | 10 |
| Spatial Functions | 5 | 0 | 0 | 5 |

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
