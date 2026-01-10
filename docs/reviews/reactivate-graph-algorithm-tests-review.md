# Review: Reactivate Graph Algorithm Tests

**Task:** Reactivate graph algorithm tests (entities fix)
**Branch:** vk/ee15-reactivate-graph
**Reviewer:** Claude Code
**Date:** 2026-01-10

---

## 1. Summary

This task reactivated 5 graph algorithm tests that were previously ignored due to a mismatch between how the high-level Database API and the low-level graph NodeStore stored and retrieved entities. The tests were:

1. `test_pagerank_execution`
2. `test_pagerank_with_parameters`
3. `test_connected_components_execution`
4. `test_louvain_execution`
5. `test_degree_centrality_execution`

The root cause was twofold:
- **Key encoding mismatch**: The Database API stored entities with just the 8-byte ID as the key, while NodeStore expected keys prefixed with `PREFIX_ENTITY` (0x01).
- **Entity serialization mismatch**: The Database API used bincode for serialization, while NodeStore expected the custom `Encoder` trait format from `manifoldb-core`.

The fix unified entity key encoding and serialization across all layers.

---

## 2. Files Changed

| File | Changes |
|------|---------|
| `crates/manifoldb/tests/integration/procedure.rs` | Removed `#[ignore]` from 5 tests, updated comments, fixed `degree` → `totalDegree` field name |
| `crates/manifoldb/src/transaction/handle.rs` | Use `encode_entity_key()` and `Entity::encode()`/`Entity::decode()` for all entity operations |
| `crates/manifoldb/src/database.rs` | Use `encode_entity_key()` and `Entity::encode()` in bulk insert/upsert operations |
| `crates/manifoldb/src/index/mod.rs` | Use `encode_entity_key()` and `Entity::decode()` for index building |
| `crates/manifoldb/src/backup/export.rs` | Use `Entity::decode()` for backup export |

---

## 3. Issues Found

### Issue 1: Inconsistent entity decoding in `iter_entities`

**Location:** `crates/manifoldb/src/transaction/handle.rs:605-609`

**Problem:** The `iter_entities` method was still using `bincode::serde::decode_from_slice` to decode entities instead of `Entity::decode()`. This was inconsistent with the other fixed methods (`get_entity`, `put_entity`, etc.) and would fail when reading entities stored in the new format.

**Severity:** High - would cause deserialization failures when iterating entities.

---

## 4. Changes Made

### Fix Applied

Updated `iter_entities` in `handle.rs` to use `Entity::decode()`:

```rust
// Before (line 605-609):
while let Some((_key, value)) = cursor.next().map_err(storage_error_to_tx_error)? {
    let (entity, _): (Entity, _) =
        bincode::serde::decode_from_slice(&value, bincode::config::standard())
            .map_err(|e| TransactionError::Serialization(e.to_string()))?;
    entities.push(entity);
}

// After:
while let Some((_key, value)) = cursor.next().map_err(storage_error_to_tx_error)? {
    let entity = Entity::decode(&value)
        .map_err(|e| TransactionError::Serialization(e.to_string()))?;
    entities.push(entity);
}
```

---

## 5. Test Results

### Procedure Tests (11 total)
```
running 11 tests
test integration::procedure::test_pagerank_execution ... ok
test integration::procedure::test_pagerank_with_parameters ... ok
test integration::procedure::test_connected_components_execution ... ok
test integration::procedure::test_louvain_execution ... ok
test integration::procedure::test_degree_centrality_execution ... ok
test integration::procedure::test_bfs_execution ... ok
test integration::procedure::test_dfs_execution ... ok
test integration::procedure::test_shortest_path_execution ... ok
test integration::procedure::test_shortest_path_no_path ... ok
test integration::procedure::test_jaccard_similarity_execution ... ok
test integration::procedure::test_unknown_procedure_error ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 610 filtered out
```

### Full Workspace Tests
All tests pass.

### Code Quality
- `cargo fmt --all -- --check`: ✅ Pass
- `cargo clippy --workspace --all-targets -- -D warnings`: ✅ Pass

---

## 6. Verdict

✅ **Approved with Fixes**

The implementation correctly unifies entity key encoding and serialization across the Database API and graph layer. One issue was found and fixed: the `iter_entities` method was still using bincode instead of `Entity::decode()`.

### Key Observations

1. **Unified Key Encoding**: All entity storage now uses `encode_entity_key()` which adds the `PREFIX_ENTITY` (0x01) byte, matching the graph layer's expectations.

2. **Unified Serialization**: All entity serialization now uses `Entity::encode()`/`Entity::decode()` from the `Encoder`/`Decoder` traits, ensuring compatibility with the graph layer.

3. **Test Corrections**: The `test_degree_centrality_execution` test was updated to use `totalDegree` instead of `degree`, matching the actual procedure output schema.

4. **Comments Cleaned Up**: Outdated comments about table name mismatches were removed from the procedure test file.

### Remaining Considerations

- Other uses of `bincode` in the codebase (for `IndexMetadata`, `TableSchema`, etc.) are correct - those are not entities and don't interact with the graph layer.
