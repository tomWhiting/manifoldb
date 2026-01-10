# Fix MATCH...MERGE Parser Review

**Date:** January 10, 2026
**Task:** Fix MATCH...MERGE parser without ON CREATE/MATCH SET
**Branch:** vk/6200-fix-match-merge

---

## 1. Summary

This review covers a bug fix for the Cypher parser that was failing to parse MATCH...MERGE statements that don't have ON CREATE SET or ON MATCH SET clauses.

**Problem:** Queries like this were failing:
```cypher
MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
MERGE (a)-[r:KNOWS]->(b) RETURN r
```

Error: `Parse("invalid graph pattern: expected '(' at start of node pattern, found: M")`

**Root Cause:** The `is_standalone_match` function in `extensions.rs` was missing a check for the `MERGE` keyword, causing MATCH...MERGE queries to be incorrectly routed to `parse_standalone_match` instead of `parse_cypher_merge`.

---

## 2. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/manifoldb-query/src/parser/extensions.rs` | Modified | Added MERGE keyword check in `is_standalone_match` |
| `crates/manifoldb/tests/integration/cypher_merge.rs` | Modified | Removed `#[ignore]` from 3 tests |

---

## 3. Issues Found

### 3.1 Original Bug (Fixed)

**Location:** `crates/manifoldb-query/src/parser/extensions.rs:184-217`

The `is_standalone_match` function checks if a query should be parsed as a standalone MATCH (read-only Cypher query). It was checking for mutation keywords like CREATE, SET, DELETE, and REMOVE, but **was missing a check for MERGE**.

This caused queries like `MATCH (a), (b) MERGE (a)-[r:KNOWS]->(b) RETURN r` to:
1. Pass all the negative checks (no CREATE, SET, DELETE, REMOVE)
2. Return `true` from `is_standalone_match`
3. Be routed to `parse_standalone_match` instead of `parse_cypher_merge`
4. Fail with a parsing error

### 3.2 Code Quality Review

Verified against `docs/CODING_STANDARDS.md`:

| Criterion | Status |
|-----------|--------|
| No `unwrap()` in library code | ✅ Pass |
| No `expect()` in library code | ✅ Pass |
| No `panic!()` in library code | ✅ Pass |
| Proper error handling with context | ✅ Pass |
| No unnecessary `.clone()` | ✅ Pass |
| No `unsafe` blocks | ✅ Pass |
| Module structure (mod.rs for declarations) | ✅ Pass |
| Tests for new functionality | ✅ Pass (tests reactivated) |

---

## 4. Changes Made

### 4.1 Parser Fix

**File:** `crates/manifoldb-query/src/parser/extensions.rs`

Added MERGE keyword check at line 212-214:
```rust
if upper.contains("MERGE") {
    return false; // This is a MERGE statement
}
```

Also updated the comment at line 199 to document the added keyword:
```diff
- // Check for CREATE, SET, DELETE, REMOVE which indicate mutation operations
+ // Check for CREATE, SET, DELETE, REMOVE, MERGE which indicate mutation operations
```

### 4.2 Test Reactivation

**File:** `crates/manifoldb/tests/integration/cypher_merge.rs`

Removed `#[ignore]` attributes and associated documentation comments from three tests:
- `test_merge_relationship_creates_when_not_exists` (line 289)
- `test_merge_relationship_matches_existing` (line 321)
- `test_merge_relationship_with_properties` (line 352)

---

## 5. Test Results

### 5.1 MERGE Tests

```
cargo test --package manifoldb merge

running 27 tests
test integration::cypher_merge::test_merge_creates_node_when_not_exists ... ok
test integration::cypher_merge::test_merge_matches_existing_node ... ok
test integration::cypher_merge::test_merge_with_different_property_creates_new_node ... ok
test integration::cypher_merge::test_merge_node_with_multiple_labels ... ok
test integration::cypher_merge::test_merge_on_create_set_when_creating ... ok
test integration::cypher_merge::test_merge_on_create_set_not_applied_when_matching ... ok
test integration::cypher_merge::test_merge_on_match_set_when_matching ... ok
test integration::cypher_merge::test_merge_on_match_set_not_applied_when_creating ... ok
test integration::cypher_merge::test_merge_both_on_create_and_on_match_when_creating ... ok
test integration::cypher_merge::test_merge_both_on_create_and_on_match_when_matching ... ok
test integration::cypher_merge::test_merge_twice_creates_once ... ok
test integration::cypher_merge::test_merge_relationship_creates_when_not_exists ... ok  ← Previously ignored
test integration::cypher_merge::test_merge_relationship_matches_existing ... ok         ← Previously ignored
test integration::cypher_merge::test_merge_relationship_with_properties ... ok          ← Previously ignored
test integration::cypher_merge::test_merge_relationship_on_create_set ... ok
test integration::cypher_merge::test_merge_without_return ... ok
test integration::cypher_merge::test_merge_only_label_no_properties ... ok
...
test result: ok. 26 passed; 0 failed; 1 ignored
```

All 17 Cypher MERGE tests pass, plus 9 SQL MERGE tests. The only ignored test is `test_foreach_with_merge` which is unrelated (requires full FOREACH with MERGE semantics).

### 5.2 Full Test Suite

```
cargo test --workspace
test result: ok. (all tests pass)
```

### 5.3 Clippy

```
cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo]
```

No warnings.

### 5.4 Formatting

```
cargo fmt --all -- --check
(no output - formatting is correct)
```

---

## 6. Verdict

**✅ Approved**

The fix is correct and minimal:
- The root cause was correctly identified (missing MERGE check in `is_standalone_match`)
- The fix is surgical - only 4 lines changed in the parser
- The three previously-ignored tests now pass
- All 26 MERGE-related tests pass
- Full test suite passes
- No clippy warnings
- Formatting is correct
- The change follows existing code patterns and conventions

The implementation aligns with ManifoldDB's design principles and coding standards.
