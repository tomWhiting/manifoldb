# Materialized View Implementation Review

**Reviewed:** 2026-01-10
**Reviewer:** Code Reviewer Agent
**Task:** Implement MATERIALIZED VIEW
**Branch:** vk/b511-implement-materi

---

## 1. Summary

This review covers the complete implementation of `CREATE MATERIALIZED VIEW`, `DROP MATERIALIZED VIEW`, and `REFRESH MATERIALIZED VIEW` SQL statements for ManifoldDB. The implementation includes parsing, AST nodes, logical plan nodes, physical plan nodes, schema storage, cached data storage, and query execution with proper cache invalidation.

---

## 2. Files Changed

### Parser Layer (manifoldb-query)
| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/parser/sql.rs` | Added parsing for all three statements, custom REFRESH parser |
| `crates/manifoldb-query/src/ast/statement.rs` | Added `CreateMaterializedViewStatement`, `DropMaterializedViewStatement`, `RefreshMaterializedViewStatement` |
| `crates/manifoldb-query/src/ast/mod.rs` | Re-exported new types |

### Logical Plan Layer (manifoldb-query)
| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/plan/logical/ddl.rs` | Added `CreateMaterializedViewNode`, `DropMaterializedViewNode`, `RefreshMaterializedViewNode` |
| `crates/manifoldb-query/src/plan/logical/node.rs` | Added `LogicalPlan` variants for all three operations |
| `crates/manifoldb-query/src/plan/logical/mod.rs` | Re-exported new types |
| `crates/manifoldb-query/src/plan/logical/builder.rs` | Added `build_create_materialized_view`, `build_drop_materialized_view`, `build_refresh_materialized_view`, materialized view registration |
| `crates/manifoldb-query/src/plan/logical/schema.rs` | Updated output_schema handling for new plan types |
| `crates/manifoldb-query/src/plan/logical/validate.rs` | Added validation for new plan types |

### Physical Plan Layer (manifoldb-query)
| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/plan/physical/node.rs` | Added `PhysicalPlan` variants for all three operations |
| `crates/manifoldb-query/src/plan/physical/builder.rs` | Added physical plan conversion for materialized view operations |

### Optimizer (manifoldb-query)
| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/plan/optimize/predicate_pushdown.rs` | Updated to handle new plan types |

### Execution Layer (manifoldb-query)
| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/exec/executor.rs` | Added execution handling in `execute_plan` match |

### Schema Storage (manifoldb)
| File | Changes |
|------|---------|
| `crates/manifoldb/src/schema/mod.rs` | Added `MaterializedViewSchema`, `MaterializedViewData`, `MaterializedViewRows`, SchemaManager methods for CRUD operations |

### Database Execution (manifoldb)
| File | Changes |
|------|---------|
| `crates/manifoldb/src/execution/executor.rs` | Added `execute_create_materialized_view`, `execute_drop_materialized_view`, `execute_refresh_materialized_view`, `try_execute_materialized_view_scan`, `try_execute_materialized_view_as_entities` |
| `crates/manifoldb/src/execution/table_extractor.rs` | Added table extraction for `RefreshMaterializedView` and `DropMaterializedView` for cache invalidation |

### Tests
| File | Changes |
|------|---------|
| `crates/manifoldb/tests/integration/materialized_view.rs` | 9 integration tests |
| `crates/manifoldb/tests/integration/mod.rs` | Module declaration |

---

## 3. Implementation Review

### 3.1 Parsing

**Status:** Complete

The parser correctly handles:
- `CREATE MATERIALIZED VIEW [IF NOT EXISTS] name [(columns)] AS SELECT ...`
- `DROP MATERIALIZED VIEW [IF EXISTS] name [, ...] [CASCADE]`
- `REFRESH MATERIALIZED VIEW [CONCURRENTLY] name`

The REFRESH statement uses a custom pre-parser (`try_parse_refresh_materialized_view`) since sqlparser-rs doesn't natively support this syntax. This is a pragmatic solution.

**Tests:** 12 parser tests covering all syntax variations (sql.rs lines 3066-3228).

### 3.2 AST Nodes

**Status:** Complete

Three AST statement types with proper builder patterns:
- `CreateMaterializedViewStatement` - stores name, columns, query, if_not_exists
- `DropMaterializedViewStatement` - stores names, if_exists, cascade
- `RefreshMaterializedViewStatement` - stores name, concurrently flag

All types implement required traits (Debug, Clone, PartialEq) and use `#[must_use]`.

### 3.3 Logical Plan Nodes

**Status:** Complete

DDL nodes in `ddl.rs`:
- `CreateMaterializedViewNode` - with builder methods and `#[must_use]`
- `DropMaterializedViewNode` - with builder methods and `#[must_use]`
- `RefreshMaterializedViewNode` - with builder methods and `#[must_use]`

All nodes properly integrate with `LogicalPlan` enum and implement `Display` for EXPLAIN output.

### 3.4 Physical Plan

**Status:** Complete

Physical plan variants are direct pass-throughs from logical plan (appropriate for DDL operations). All three operations are handled in `PhysicalPlanner::plan()`.

### 3.5 Schema Storage

**Status:** Complete

Storage prefixes in `schema/mod.rs`:
- `schema:matview:` for view definitions
- `schema:matview_data:` for refresh metadata (last_refreshed, row_count, result_columns)
- `schema:matview_rows:` for cached row data
- `schema:matviews_list` for listing all materialized views

SchemaManager methods:
- `create_materialized_view()` - stores definition and initializes empty data
- `drop_materialized_view()` - removes definition, data, and rows
- `update_materialized_view_data()` - updates refresh metadata
- `store_materialized_view_rows()` - stores cached row data
- `get_materialized_view_rows()` - retrieves cached rows
- `materialized_view_exists()` / `get_materialized_view()` / `list_materialized_views()`

### 3.6 Execution

**Status:** Complete

`execute_create_materialized_view` (executor.rs:4208):
- Extracts SELECT query from SQL using `extract_materialized_view_query_sql()`
- Stores view definition via SchemaManager
- Initializes empty data entry (view must be REFRESHed to populate)

`execute_drop_materialized_view` (executor.rs:4250):
- Removes view definition and cached data via SchemaManager

`execute_refresh_materialized_view` (executor.rs:4266):
- Retrieves view schema to get stored query
- Executes query using `execute_query()` to get fresh results
- Stores cached rows via `store_materialized_view_rows()`
- Updates metadata (last_refreshed, row_count, result_columns)
- Returns row count

### 3.7 Query Resolution

**Status:** Complete

Materialized views are queried via:

`try_execute_materialized_view_scan()` (executor.rs:471):
- Checks if table name is a materialized view
- Retrieves cached data from `get_materialized_view_data()`
- Retrieves cached rows from `get_materialized_view_rows()`
- Returns error if view has not been refreshed yet
- Builds ResultSet from cached data with proper schema

`try_execute_materialized_view_as_entities()` (executor.rs:517):
- Converts cached rows to synthetic Entity objects for graph queries

These functions are called from scan execution paths (lines 1868, 2076, 3418).

### 3.8 Cache Invalidation

**Status:** Complete

The `table_extractor.rs` file correctly extracts view names for cache invalidation:

```rust
LogicalPlan::RefreshMaterializedView(node) => {
    tables.push(node.name.clone());
}

LogicalPlan::DropMaterializedView(node) => {
    for name in &node.names {
        tables.push(name.clone());
    }
}
```

This ensures that when a REFRESH or DROP is executed, any cached query results referencing the materialized view are properly invalidated.

---

## 4. Code Quality Assessment

### Error Handling
- No `unwrap()` or `expect()` in library code (except test assertion)
- Proper use of `.map_err()` with context messages
- All error paths return meaningful messages

### Memory & Performance
- No unnecessary `.clone()` calls
- Appropriate use of references
- Cached rows are stored efficiently using bincode serialization

### Module Structure
- DDL nodes in named file (`ddl.rs`), not mod.rs
- Proper re-exports in mod.rs files
- Schema types in dedicated schema module

### Documentation
- Doc comments on public types and functions
- Clear explanations of view refresh semantics
- Module-level documentation present

### Type Design
- Builder patterns with `#[must_use]` on all builder methods
- Proper derive macros (Debug, Clone, PartialEq)
- Serializable types use serde

### Testing
- 12 parser tests
- 9 integration tests covering:
  - Basic create/drop/refresh
  - IF NOT EXISTS / IF EXISTS
  - Query cached data
  - Cache isolation (data not live)
  - Error cases (nonexistent view, unrefreshed view)

---

## 5. Issues Found

No issues found. The implementation is complete and correct.

**Previous issue (fixed):** The coding agent summary notes that `extract_tables_from_sql` in `table_extractor.rs` was updated to extract view names from REFRESH and DROP statements for proper cache invalidation. This fix has been verified as correctly implemented.

---

## 6. Changes Made

No changes required. The implementation passed all quality checks.

---

## 7. Test Results

```
$ cargo fmt --all --check
(no output - all files formatted correctly)

$ cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 23.68s

$ cargo test --workspace -- materialized_view
running 9 tests (integration)
test integration::materialized_view::test_create_materialized_view_basic ... ok
test integration::materialized_view::test_create_materialized_view_if_not_exists ... ok
test integration::materialized_view::test_query_materialized_view_returns_cached_data ... ok
test integration::materialized_view::test_materialized_view_caches_data_not_live ... ok
test integration::materialized_view::test_drop_materialized_view_basic ... ok
test integration::materialized_view::test_drop_materialized_view_if_exists ... ok
test integration::materialized_view::test_materialized_view_with_simple_filter ... ok
test integration::materialized_view::test_refresh_materialized_view_nonexistent_fails ... ok
test integration::materialized_view::test_query_unrefreshed_materialized_view_fails ... ok
test result: ok. 9 passed

running 12 tests (parser)
test parser::sql::tests::parse_create_materialized_view_basic ... ok
test parser::sql::tests::parse_create_materialized_view_if_not_exists ... ok
test parser::sql::tests::parse_create_materialized_view_with_columns ... ok
test parser::sql::tests::parse_create_materialized_view_with_join ... ok
test parser::sql::tests::parse_drop_materialized_view_basic ... ok
test parser::sql::tests::parse_drop_materialized_view_if_exists ... ok
test parser::sql::tests::parse_drop_materialized_view_cascade ... ok
test parser::sql::tests::parse_drop_materialized_view_multiple ... ok
test parser::sql::tests::parse_refresh_materialized_view_basic ... ok
test parser::sql::tests::parse_refresh_materialized_view_concurrently ... ok
test parser::sql::tests::parse_refresh_materialized_view_qualified_name ... ok
test parser::sql::tests::parse_refresh_materialized_view_with_semicolon ... ok
test result: ok. 12 passed
```

---

## 8. Verdict

**Approved**

The implementation is complete, well-structured, and follows project coding standards. It properly implements:

1. Parsing for CREATE/DROP/REFRESH MATERIALIZED VIEW
2. Schema storage for view definitions
3. Cached data storage for query results
4. REFRESH executes query and updates stored data
5. Queries against materialized views return cached data
6. Proper cache invalidation on REFRESH/DROP
7. Comprehensive test coverage

---

## 9. Architecture Notes

The implementation follows the ManifoldDB query pipeline:

```
SQL Text -> Parser -> AST -> PlanBuilder -> LogicalPlan -> PhysicalPlanner -> PhysicalPlan -> Executor
```

Materialized views are handled at three levels:

1. **DDL execution** - CREATE/DROP/REFRESH are executed as side effects that modify schema storage
2. **Data caching** - REFRESH stores query results; DROP removes cached data
3. **Query resolution** - SELECT queries check if table is a materialized view and return cached data

Storage layout:
```
schema:matview:{name}       -> MaterializedViewSchema (definition)
schema:matview_data:{name}  -> MaterializedViewData (refresh metadata)
schema:matview_rows:{name}  -> MaterializedViewRows (cached row data)
schema:matviews_list        -> Vec<String> (list of all views)
```

This design enables efficient caching while maintaining consistency through proper cache invalidation.

---

## 10. Future Enhancements

The task description mentions considering "incremental refresh for future enhancement." The current implementation provides a foundation for this:

- View definitions are stored separately from cached data
- Metadata tracks last refresh time
- Row storage is modular and could support partial updates

Potential future work:
- `REFRESH MATERIALIZED VIEW CONCURRENTLY` (already parsed, needs implementation)
- Incremental refresh based on source table changes
- Automatic refresh triggers
