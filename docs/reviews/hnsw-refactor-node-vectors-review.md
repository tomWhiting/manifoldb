# Review: Refactor HNSW to not store vectors + implement nodeVectors GraphQL query

**Reviewer:** Automated Code Review
**Date:** 2026-01-11
**Task Branch:** vk/337c-refactor-hnsw-to
**Target Branch:** main

---

## 1. Summary

This review covers the implementation of two related changes:

1. **HNSW Refactoring**: Removed vector storage from `HnswNode` structs so that vectors live exclusively in `CollectionVectorStore`. The HNSW index now stores only the navigation graph structure.

2. **nodeVectors GraphQL Query**: Implemented a new GraphQL query to retrieve vector data for specific node IDs from the `CollectionVectorStore`.

### Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| HnswNode has NO embedding field | ✅ Verified |
| HNSW persistence does NOT write vectors | ✅ Verified |
| Vectors fetched from CollectionVectorStore when needed | ✅ Verified |
| All existing HNSW tests pass | ✅ Verified |
| nodeVectors GraphQL query implemented | ✅ Verified |
| All tests pass | ✅ Verified |
| No new clippy warnings | ✅ Verified (after fixes) |

---

## 2. Files Changed

### Core HNSW Refactoring
- `crates/manifoldb-vector/src/index/graph.rs` - `HnswNode` struct, search layer functions (vector fetcher pattern)
- `crates/manifoldb-vector/src/index/hnsw.rs` - `HnswIndex` methods, embeddings cache
- `crates/manifoldb-vector/src/index/persistence.rs` - `NodeData` v2 format (no embedding), backward compatibility for v1

### Public API
- `crates/manifoldb/src/vector/mod.rs` - `load_index_with_embeddings()`, embeddings cache integration

### GraphQL API
- `crates/manifold-server/src/schema/types.rs` - Added `NodeVector` type
- `crates/manifold-server/src/schema/query.rs` - Added `node_vectors` query resolver

---

## 3. Issues Found

### Code Quality Issues (Fixed)

The implementation was functionally correct but had several clippy warnings that needed to be addressed to meet the project's coding standards:

1. **Derivable Default impls** - Multiple `impl Default` blocks that could use `#[derive(Default)]` with `#[default]` attribute:
   - `manifoldb-graph/src/traversal/pattern.rs:33` - `StepFilter`
   - `manifoldb-vector/src/store/inverted_index.rs:288` - `ScoringFunction`
   - `manifoldb-query/src/ast/expr.rs:908` - `HybridCombinationMethod`
   - `manifoldb-query/src/ast/pattern.rs:134` - `LabelExpression`
   - `manifoldb-query/src/plan/logical/graph.rs:746` - `ShortestPathWeight`
   - `manifoldb-query/src/plan/logical/vector.rs:273` - `ScoreCombinationMethod`
   - `manifoldb-query/src/plan/physical/node.rs:1696` - `PhysicalScoreCombinationMethod`
   - `manifoldb/src/backup/types.rs:18` - `BackupFormat`
   - `manifoldb/src/cache/hints.rs:18` - `CacheHint`
   - `manifoldb/src/index/mod.rs:61` - `IndexType`
   - `manifoldb/src/session.rs:84` - `TransactionState`

2. **Self-only-used-in-recursion** - Several recursive methods used `&self` but only called themselves:
   - `manifoldb-query/src/plan/optimize/predicate_pushdown.rs:586` - `collect_columns_recursive`
   - `manifoldb-query/src/plan/optimize/predicate_pushdown.rs:648` - `collect_tables_recursive`
   - `manifoldb-query/src/plan/optimize/predicate_pushdown.rs:672` - `collect_expr_tables`
   - `manifoldb-query/src/plan/physical/builder.rs:1811` - `get_table_name`

3. **Useless let-if-seq** - Variable mutation pattern that could be expressed as a single `if let`:
   - `manifoldb-graph/src/analytics/similarity.rs:233`
   - `manifoldb/src/execution/executor.rs:4549`

4. **Redundant closures** - Error mapping closures that could use tuple variant constructors directly:
   - `manifold-server/src/schema/query.rs:452,461,467`

5. **Unnecessary map_or** - Could use `is_none_or` instead:
   - `manifold-server/src/schema/query.rs:220`

6. **Needless lifetimes** - Explicit lifetimes that could be elided:
   - `manifoldb-query/src/plan/physical/builder.rs:1811`

7. **Formatting issues** - Code not formatted according to `cargo fmt` standards in multiple files.

---

## 4. Changes Made

All issues listed above were fixed. The changes fall into these categories:

### Clippy Fixes (Code Quality)

1. Replaced manual `Default` implementations with `#[derive(Default)]` + `#[default]` on 11 enum types
2. Converted recursive methods to use `Self::method()` instead of `self.method()` (4 methods)
3. Refactored `useless_let_if_seq` patterns to idiomatic `if let` expressions (2 locations)
4. Simplified error mapping closures to use tuple variant constructors (3 locations)
5. Changed `map_or(true, ...)` to `is_none_or(...)` (1 location)
6. Removed needless lifetime annotations (1 method)

### Formatting Fixes

- Ran `cargo fmt --all` to fix all formatting issues across the workspace

---

## 5. Test Results

### Full Test Suite

```
running 596 tests
...
test result: ok. 596 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

All 596 unit and integration tests pass.

### HNSW-Specific Tests

The following HNSW-related tests verify the refactoring works correctly:

- `test_hnsw_index_builder` - Verifies index creation with CollectionVectorStore
- `test_drop_index` - Verifies index deletion
- `test_update_single_entity_hnsw` - Verifies update operations work after removing embedding from HnswNode

### Clippy

```
cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s)
```

Zero warnings with strict `-D warnings` flag.

### Format Check

```
cargo fmt --all --check
```

Passes with no differences.

---

## 6. Architecture Notes

### Vector Fetcher Pattern

The implementation correctly uses a closure-based vector fetcher pattern for search operations:

```rust
pub fn search_layer<F>(
    graph: &HnswGraph,
    query: &Embedding,
    entry_points: &[EntityId],
    ef: usize,
    layer: usize,
    get_vector: &F,  // Vector fetcher
) -> Vec<Candidate>
where
    F: Fn(EntityId) -> Option<Embedding>,
```

This allows the search algorithm to remain decoupled from storage, enabling:
- In-memory caching during search operations
- Lazy loading from disk
- Fallback paths for legacy data

### Backward Compatibility

The persistence layer includes version handling:
- **Version 2 (current)**: Node data without embedding
- **Version 1 (legacy)**: Node data with embedding (read support only)

This ensures existing databases can still be read while new writes use the v2 format.

### nodeVectors Query

The GraphQL query implementation correctly:
- Accepts optional collection and vector_name filters
- Scans CollectionVectorStore for matching vectors
- Returns empty array for nodes without vectors (not errors)
- Handles both filtered (single collection) and unfiltered (all collections) cases

---

## 7. Verdict

### ✅ **Approved with Fixes**

The implementation correctly fulfills all task requirements. The HNSW refactoring successfully removes vector storage from nodes while maintaining search functionality through the vector fetcher pattern. The nodeVectors GraphQL query is properly implemented.

Code quality issues (clippy warnings, formatting) were identified and fixed as part of this review. These were pre-existing issues in the codebase that were exposed by the strict clippy configuration, not problems introduced by this change.

### Recommendations

1. The vector fetcher pattern is well-designed but may benefit from additional caching strategies for high-throughput scenarios.

2. Consider adding integration tests specifically for the nodeVectors GraphQL query once a test harness for the GraphQL layer is available.

3. The backward compatibility for v1 persistence format should be documented with a timeline for eventual removal.

---

*Review completed: 2026-01-11*
