# Review: Reactivate Collection Drop Tests

**Date:** January 10, 2026
**Reviewer:** Claude Code
**Branch:** `vk/2abb-reactivate-colle`
**Task:** Reactivate collection drop tests

---

## 1. Summary

This review covers the reactivation of two previously ignored integration tests (`test_drop_collection` and `test_drop_collection_with_data`) and the integration fixes required to make them pass.

The original task was to simply remove `#[ignore]` attributes, but the tests failed because collections created via `CollectionHandle::create()` were not registered with `CollectionManager`, causing `drop_collection()` to fail when looking up the collection metadata.

The coding agent correctly identified and fixed this integration gap between the Qdrant-style Collection API (`CollectionHandle`/`PointStore`) and the internal metadata management (`CollectionManager`).

---

## 2. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/manifoldb/tests/integration/collection.rs` | Modified | Removed `#[ignore]` from two tests |
| `crates/manifoldb/src/collection/handle.rs` | Modified | Added `CollectionManager` registration in `create()` |
| `crates/manifoldb/src/collection/error.rs` | Modified | Added `From` impls for `StorageError` and `TransactionError` |
| `crates/manifoldb/src/database.rs` | Modified | Added deletion from `point_collections` table in `drop_collection()` |
| `crates/manifoldb-vector/src/store/point_store.rs` | Modified | Exported `TABLE_POINT_COLLECTIONS` constant |
| `crates/manifoldb-vector/src/store/mod.rs` | Modified | Re-exported `TABLE_POINT_COLLECTIONS` |
| `crates/manifoldb-vector/src/lib.rs` | Modified | Re-exported `TABLE_POINT_COLLECTIONS` from crate root |

---

## 3. Issues Found

### 3.1 No Code Quality Issues

All changes follow the coding standards:

- **Error handling:** Uses `?` operator with proper error propagation
- **No `unwrap()` or `expect()`:** All error paths are handled via `?` and `From` implementations
- **Proper module structure:** No implementation in `mod.rs`, only re-exports
- **Crate boundaries respected:** Changes in `manifoldb-vector` only expose an existing constant; integration logic remains in `manifoldb`

### 3.2 Design Consideration (Not a Bug)

The implementation creates a secondary transaction within `CollectionHandle::create()` to register with `CollectionManager`:

```rust
// crates/manifoldb/src/collection/handle.rs:114-120
{
    let storage_tx = point_store.engine().begin_write()?;
    let mut db_tx =
        DatabaseTransaction::new_write(0, storage_tx, VectorSyncStrategy::Synchronous);
    CollectionManager::create(&mut db_tx, &name, vectors.iter().cloned())?;
    db_tx.commit()?;
}
```

This is acceptable because:
1. `PointStore::create_collection()` already committed its transaction internally
2. The `CollectionManager` registration is a separate metadata operation
3. Both operations are idempotent, so partial failure leaves consistent state

A future improvement could unify both operations into a single transaction, but this is outside the scope of the current task.

---

## 4. Changes Made

No fixes were required. The implementation is correct and follows project standards.

---

## 5. Test Results

### Collection Integration Tests

```
running 38 tests
test integration::collection::test_drop_collection ... ok
test integration::collection::test_drop_collection_with_data ... ok
...
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 583 filtered out
```

### Full Test Suite

```
test result: ok. 689 passed; 0 failed; 17 ignored; 0 measured
```

### Code Quality

| Check | Result |
|-------|--------|
| `cargo fmt --all --check` | Pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | Pass |
| `cargo test --workspace` | Pass |

---

## 6. Verdict

**Approved**

The implementation correctly integrates `CollectionHandle` creation with `CollectionManager` metadata tracking, and properly cleans up both `point_collections` and `collection_manager` tables on drop. All tests pass and code quality checks are clean.

The two previously ignored tests are now active and verifying that:
1. `drop_collection()` removes the collection completely
2. `drop_collection()` removes collections that contain data
3. Dropped collections cannot be reopened
4. `list_collections()` no longer includes dropped collections

---

*Reviewed on vk/2abb-reactivate-colle branch*
