# Graph-Constrained Vector Search API Review

**Date:** January 1, 2026
**Reviewer:** Claude (Opus 4.5)
**Task:** Implement Graph-Constrained Vector Search API

---

## Summary

This review covers the implementation of the graph-constrained vector search API, which enables vector similarity search to be constrained to entities reachable via graph traversal patterns. This is the #1 feature needed for Gimbal integration, enabling queries like "find similar code symbols within this repository."

The implementation adds a `.within_traversal()` method to `EntitySearchBuilder` that accepts a starting entity ID and a closure-based pattern builder.

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/manifoldb/src/search.rs` | Modified | Added `TraversalConstraint`, `TraversalPatternBuilder`, and `within_traversal()` method |
| `crates/manifoldb/src/database.rs` | Modified | Updated search builder to pass storage engine for traversal; fixed edge serialization |
| `crates/manifoldb/src/transaction/handle.rs` | Modified | Fixed edge key/value encoding to be compatible with graph layer |
| `crates/manifoldb/src/backup/export.rs` | Modified | Updated edge deserialization to use `Edge::decode()` |
| `crates/manifoldb/src/lib.rs` | Modified | Re-exported `TraversalConstraint` and `TraversalPatternBuilder` |
| `crates/manifoldb/tests/integration/graph_vector_search.rs` | Added | 6 comprehensive integration tests |
| `crates/manifoldb/tests/integration/mod.rs` | Modified | Added `graph_vector_search` module |

---

## Implementation Analysis

### New API

The implementation provides an ergonomic closure-based API:

```rust
let results = db.search("symbols", "embedding")
    .query(query_vector)
    .within_traversal(repo_id, |p| p
        .edge_out("CONTAINS")
        .variable_length(1, 10)
    )
    .filter(Filter::eq("visibility", "public"))
    .limit(10)
    .execute()?;
```

### Key Components

1. **`TraversalConstraint`** (`search.rs:68-93`)
   - Holds start entity ID and `PathPattern`
   - Implements `Clone` for reusability
   - Proper `#[must_use]` on accessor methods

2. **`TraversalPatternBuilder`** (`search.rs:95-213`)
   - Fluent builder API with methods: `edge_out()`, `edge_in()`, `edge_both()`, `any_out()`, `any_in()`, `any_both()`, `variable_length()`, `step()`
   - All builder methods are `#[must_use]`
   - Implements `Debug`, `Clone`, `Default`

3. **`EntitySearchBuilder::within_traversal()`** (`search.rs:322-367`)
   - Accepts closure for pattern building
   - Also provides `with_traversal_constraint()` for pre-built constraints

4. **Execution Strategy** (`search.rs:388-472`)
   - Executes traversal first to get reachable entity IDs
   - Uses heuristic fetch limit (10x requested limit, minimum 100)
   - Post-filters vector results against traversal set
   - Maintains similarity ordering

### Bug Fixes Included

The original agent discovered and fixed edge encoding mismatches:

1. **Edge Key Encoding** (`transaction/handle.rs`)
   - Changed from `id.as_u64().to_be_bytes()` to `encode_edge_key(id)`
   - Now uses `PREFIX_EDGE` constant for graph layer compatibility

2. **Edge Value Encoding** (`transaction/handle.rs`, `database.rs`)
   - Changed from `bincode::serde::encode_to_vec()` to `Edge::encode()`
   - Ensures consistency with `EdgeStore` in graph layer

3. **Edge Decoding** (`transaction/handle.rs`, `backup/export.rs`)
   - Changed from `bincode::serde::decode_from_slice()` to `Edge::decode()`

---

## Code Quality Assessment

### Error Handling

| Criterion | Status | Notes |
|-----------|--------|-------|
| No `unwrap()` in library code | PASS | Only in test code |
| No `expect()` in library code | PASS | Only in test code |
| No `panic!()` macro | PASS | |
| Errors have context | PASS | `Error::Execution(format!(...))` provides context |

### Memory & Performance

| Criterion | Status | Notes |
|-----------|--------|-------|
| No unnecessary `.clone()` | PASS | Clones only where ownership needed |
| Proper use of references | PASS | |
| No `unsafe` blocks | PASS | |

### Module Organization

| Criterion | Status | Notes |
|-----------|--------|-------|
| `mod.rs` for declarations only | PASS | search.rs is a named file |
| Clear separation of concerns | PASS | |

### Documentation

| Criterion | Status | Notes |
|-----------|--------|-------|
| Module-level docs | PASS | `//!` docs at top of search.rs |
| Public item docs | PASS | All public types and methods documented |
| Examples in docs | PASS | Both module-level and method-level examples |

### Testing

| Criterion | Status | Notes |
|-----------|--------|-------|
| Unit tests present | PASS | In search.rs |
| Integration tests | PASS | 6 tests in graph_vector_search.rs |
| Edge cases covered | PASS | Empty traversal, limits, filters |

---

## Issues Found

### Issue 1: Code Formatting (FIXED)

**Severity:** Minor
**Location:** Multiple files

The code was not formatted according to `cargo fmt` standards.

**Resolution:** Ran `cargo fmt --all` - formatting now passes.

---

## Test Results

### Unit Tests
All search.rs unit tests pass:
- `test_json_to_value_primitives`
- `test_value_to_json_roundtrip`
- `test_entity_properties_to_json`

### Integration Tests
All 6 graph-constrained vector search tests pass:

```
test integration::graph_vector_search::test_graph_constrained_search_basic ... ok
test integration::graph_vector_search::test_graph_constrained_search_variable_length ... ok
test integration::graph_vector_search::test_graph_constrained_search_with_filter ... ok
test integration::graph_vector_search::test_search_without_traversal_constraint ... ok
test integration::graph_vector_search::test_graph_constrained_search_empty_traversal ... ok
test integration::graph_vector_search::test_graph_constrained_search_respects_limit ... ok
```

### Full Test Suite
All 227 workspace tests pass.

### Clippy
No warnings with `-D warnings` flag.

---

## Verification Commands

```bash
cargo fmt --all --check     # PASS
cargo clippy --workspace --all-targets -- -D warnings  # PASS
cargo test --workspace      # PASS (227 tests)
```

---

## Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| `EntitySearchBuilder` has `.within_traversal()` method | PASS |
| Traversal accepts starting entity ID and pattern builder | PASS |
| Vector search results constrained to traversal results | PASS |
| Works with variable-length paths (`*1..10`) | PASS |
| Works with property filters | PASS |
| Integration test: repo -> files -> symbols graph | PASS |
| Integration test: symbols outside traversal excluded | PASS |
| All existing tests pass | PASS |
| No clippy warnings | PASS |

---

## Changes Made During Review

1. **Code Formatting** - Ran `cargo fmt --all` to fix formatting issues in:
   - `crates/manifoldb/src/database.rs`
   - `crates/manifoldb/src/search.rs`
   - `crates/manifoldb/src/transaction/handle.rs`
   - `crates/manifoldb/tests/integration/graph_vector_search.rs`

---

## Verdict

**APPROVED WITH FIXES**

The implementation is complete, well-documented, and follows all project coding standards. The only issue found was minor formatting discrepancies, which have been resolved.

The API design is ergonomic and follows the established builder pattern. The implementation correctly:
- Executes graph traversal before vector search
- Filters results to only include reachable entities
- Maintains similarity ordering
- Combines properly with property filters

The bug fixes for edge encoding ensure consistency between the transaction layer and graph layer, which is critical for this feature to work correctly.

---

*Review completed: January 1, 2026*
