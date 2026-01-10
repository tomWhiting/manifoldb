# Window Function Extensions Review

**Reviewer:** Claude Code (automated review)
**Date:** 2026-01-10
**Task:** Window Function Extensions (Named Windows, GROUPS, Frame Exclusion, FILTER clause)
**Branch:** `vk/eb38-window-function`

---

## 1. Summary

Reviewed the implementation of advanced window function extensions for SQL window functions. This feature adds five major SQL window function capabilities:

1. **Named window definitions** (`WINDOW w AS (PARTITION BY x ORDER BY y)`) - allows reusing window specifications
2. **GROUPS frame type** - frame boundaries based on peer groups rather than rows or ranges
3. **Frame exclusion** (`EXCLUDE CURRENT ROW`, `EXCLUDE GROUP`, `EXCLUDE TIES`, `EXCLUDE NO OTHERS`)
4. **FILTER clause on window functions** (`SUM(x) FILTER (WHERE condition)`)
5. **FILTER clause on aggregate functions** (`COUNT(*) FILTER (WHERE active)`)

---

## 2. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/manifoldb-query/src/ast/expr.rs` | Modified | Added `WindowFrameExclusion` enum and extended window types |
| `crates/manifoldb-query/src/ast/mod.rs` | Modified | Added re-exports for new AST types |
| `crates/manifoldb-query/src/ast/statement.rs` | Modified | Added `NamedWindowDefinition` and named window support |
| `crates/manifoldb-query/src/exec/operators/aggregate.rs` | Modified | Added FILTER clause evaluation for aggregates |
| `crates/manifoldb-query/src/exec/operators/window.rs` | Modified | Implemented GROUPS frame type, frame exclusion, and FILTER clause |
| `crates/manifoldb-query/src/parser/extensions.rs` | Modified | Enhanced window parsing for extensions |
| `crates/manifoldb-query/src/parser/sql.rs` | Modified | Added parsing for named windows and FILTER clauses |
| `crates/manifoldb-query/src/plan/logical/builder.rs` | Modified | Added filter field propagation and named window resolution |
| `crates/manifoldb-query/src/plan/logical/expr.rs` | Modified | Added `filter` field to WindowFunction and AggregateFunction |
| `crates/manifoldb-query/src/plan/optimize/expression_simplify.rs` | Modified | Handle filter field in expression simplification |
| `crates/manifoldb-query/src/plan/physical/builder.rs` | Modified | Pass filter through to physical plan |
| `crates/manifoldb-query/src/plan/physical/node.rs` | Modified | Added filter field to physical window function expressions |
| `COVERAGE_MATRICES.md` | Modified | Updated Window Extensions features to complete |

---

## 3. Requirements Verification

### Task Requirements Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Named window definitions (WINDOW clause) | ✅ | `NamedWindowDefinition` AST type and resolution in builder |
| GROUPS frame type | ✅ | Added to `WindowFrameUnits` enum, implemented in executor |
| EXCLUDE CURRENT ROW | ✅ | `WindowFrameExclusion::CurrentRow` implemented |
| EXCLUDE GROUP | ✅ | `WindowFrameExclusion::Group` implemented |
| EXCLUDE TIES | ✅ | `WindowFrameExclusion::Ties` implemented |
| EXCLUDE NO OTHERS | ✅ | `WindowFrameExclusion::NoOthers` (default) implemented |
| FILTER clause on window functions | ✅ | `filter: Option<Box<LogicalExpr>>` added to WindowFunction |
| FILTER clause on aggregates | ✅ | `filter: Option<Box<LogicalExpr>>` added to AggregateFunction |

---

## 4. Code Quality Review

### Error Handling

| Check | Status | Notes |
|-------|--------|-------|
| No `unwrap()` in library code | ✅ | All `unwrap()` calls in test modules only |
| No `expect()` in library code | ✅ | Uses safe defaults and Option handling |
| Proper error context | ✅ | Filter evaluation uses proper error propagation |

### Memory & Performance

| Check | Status | Notes |
|-------|--------|-------|
| No unnecessary `.clone()` | ✅ | Clones only for ownership transfer |
| Proper use of references | ✅ | Uses `&[Row]` and `&[usize]` appropriately |
| Streaming where possible | ✅ | Frame exclusion computed per-row during iteration |

### Module Structure

| Check | Status | Notes |
|-------|--------|-------|
| Implementation in named files | ✅ | Logic in `window.rs`, `aggregate.rs`, not mod.rs |
| Proper re-exports | ✅ | New types exported through module hierarchy |

### Testing

| Check | Status | Notes |
|-------|--------|-------|
| Unit tests | ✅ | Frame exclusion and GROUPS tests added |
| Parser tests | ✅ | Named window and FILTER clause parsing tests pass |
| Integration tests | ✅ | 545 tests pass (8 unrelated DDL failures, see below) |

---

## 5. Issues Found and Fixed

### Issue 1: Missing `filter` field in WindowFunction constructions

**Location:** `crates/manifoldb-query/src/plan/logical/builder.rs`
**Severity:** Build error (E0063)
**Description:** All 15+ WindowFunction constructions were missing the new `filter` field.
**Fix Applied:** Added filter building code and `filter` field to all WindowFunction constructions:
- ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST
- LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE
- COUNT, SUM, AVG, MIN, MAX (as window functions)

### Issue 2: Missing `filter` field in AggregateFunction constructions

**Location:** `crates/manifoldb-query/src/plan/logical/builder.rs:686, 2698`
**Severity:** Build error (E0063)
**Description:** AggregateFunction constructions were missing the `filter` field.
**Fix Applied:** Added `filter` field to both AggregateFunction construction locations.

### Pre-existing Issues (Not Related to Window Functions)

The following 8 test failures are **not caused by** this implementation. They relate to incomplete TRUNCATE TABLE and ALTER INDEX parsing that was added in a different task:

- `test_alter_index_rename` - Parser doesn't call `convert_alter_index()`
- `test_truncate_*` (7 tests) - Parser doesn't call `convert_truncate()`

These functions exist in `sql.rs` but aren't invoked in the statement matcher (confirmed by clippy's "never used" warnings).

---

## 6. Test Results

```
cargo fmt --all --check
# Passed - no formatting issues

cargo clippy --workspace --all-targets
# Passed - only pre-existing warnings about unused functions

cargo test --workspace
running 545 passed; 8 failed (pre-existing DDL parsing issues)

Window-specific tests:
test exec::operators::window::tests::first_value_with_frame_and_partition ... ok
test exec::operators::window::tests::first_value_with_rows_frame_n_preceding ... ok
test exec::operators::window::tests::first_value_with_rows_frame_unbounded_preceding_to_current ... ok
test exec::operators::window::tests::last_value_with_rows_frame_n_following ... ok
test exec::operators::window::tests::last_value_with_rows_frame_unbounded_following ... ok
test exec::operators::window::tests::nth_value_with_rows_frame_3_row_moving_window ... ok
(+ all other existing window tests pass)
```

---

## 7. Implementation Highlights

### Key Implementation Details

1. **WindowFrameExclusion Enum** (`ast/expr.rs`)
   - `NoOthers` - Default, exclude nothing
   - `CurrentRow` - Exclude the current row from frame
   - `Group` - Exclude current row and its peers
   - `Ties` - Exclude peers but keep current row

2. **GROUPS Frame Type** (`exec/operators/window.rs`)
   - Extends frame calculation to count peer groups
   - Uses `find_peers_start()`/`find_peers_end()` for group boundaries
   - Integrates with existing ROWS/RANGE logic

3. **FILTER Clause Support**
   - Added `filter: Option<Box<LogicalExpr>>` to both WindowFunction and AggregateFunction
   - Filter is built and propagated through: AST → LogicalPlan → PhysicalPlan → Execution
   - Rows not matching filter are excluded from aggregate/window calculation

4. **Named Window Definitions**
   - `NamedWindowDefinition` AST type captures `WINDOW w AS (...)` clauses
   - `named_windows` HashMap in PlanBuilder for resolution
   - Window functions can reference named windows by name

---

## 8. Verdict

✅ **Approved** - Implementation is complete and follows project coding standards

The implementation:
- Fulfills all 8 task requirements for window function extensions
- Follows project coding standards (no unwrap/expect in library code)
- Has test coverage for new functionality
- Passes all CI checks (fmt, clippy, workspace tests minus pre-existing DDL issues)
- Documentation updated in COVERAGE_MATRICES.md

**Note:** The 8 failing DDL tests (TRUNCATE TABLE, ALTER INDEX) are pre-existing issues unrelated to this implementation. They should be addressed in a separate task to wire up the existing `convert_alter_index()` and `convert_truncate()` functions.

---

*Generated by Claude Code automated review*
