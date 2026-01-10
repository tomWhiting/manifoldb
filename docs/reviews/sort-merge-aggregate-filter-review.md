# Review: Fix SortMergeAggregateOp FILTER clause bug

**Task:** Fix SortMergeAggregateOp FILTER clause bug
**Branch:** vk/90a3-fix-sortmergeagg
**Reviewer:** Claude Code
**Date:** 2026-01-10

## Summary

Reviewed the fix for the `SortMergeAggregateOp` operator which was ignoring FILTER clauses on aggregate functions. The fix correctly mirrors the filter handling logic from `HashAggregateOp`.

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `crates/manifoldb-query/src/exec/operators/aggregate.rs` | Modified | Added FILTER clause handling in `update_accumulators()`, fixed double-accumulation bug, added 2 tests |

## Implementation Review

### 1. FILTER clause handling (lines 341-358)

The fix adds proper FILTER clause evaluation to `SortMergeAggregateOp::update_accumulators()`:

```rust
if let LogicalExpr::AggregateFunction { func, args, distinct: _, filter } = agg_expr {
    // Check FILTER clause before updating accumulator
    let passes_filter = if let Some(filter_expr) = filter {
        match evaluate_expr(filter_expr, row) {
            Ok(Value::Bool(true)) => true,
            _ => false, // Filter out if not true (including NULL)
        }
    } else {
        true
    };

    if passes_filter {
        // ... accumulate the row
    }
}
```

This correctly matches the pattern used in `HashAggregateOp` (lines 138-147), ensuring consistency between the two operators.

### 2. Double-accumulation bug fix (lines 436-446)

The coding agent identified and fixed a subtle bug in the "new group" branch:

**Before:** When encountering a new group, the code both stored the row as `pending_row` AND processed it immediately, causing double-accumulation.

**After:** The row is processed in place without setting `pending_row`, with a clear comment explaining why.

### 3. New tests added

Two new unit tests were added:
- `sort_merge_aggregate_filter_clause` - Tests `COUNT(*) FILTER (WHERE salary > 120)`
- `sort_merge_aggregate_sum_with_filter` - Tests `SUM(salary) FILTER (WHERE salary > 120)`

Both tests verify that filtered aggregates return different values than unfiltered aggregates, correctly validating the fix.

## Code Quality Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| No `unwrap()` in library code | N/A | Changes are in test code only use unwrap; library code uses `?` |
| No `expect()` in library code | PASS | None added |
| Error context with `.context()` | N/A | No new error paths added |
| No unnecessary `.clone()` | PASS | No new clones added |
| `mod.rs` structure | N/A | No module changes |
| Unit tests added | PASS | 2 new tests |
| `cargo fmt` | PASS | No formatting issues |
| `cargo clippy` | PASS* | Pre-existing warnings only (see below) |

### Pre-existing Issues (not caused by this fix)

1. **Large enum variant warnings** in `ast/statement.rs` - These are existing technical debt in the AST module
2. **Dead code warning** for `named_windows` field in `builder.rs` - Part of the Window Function Extensions work, will be resolved when that feature is completed

## Test Results

```
running 2 tests
test exec::operators::aggregate::tests::sort_merge_aggregate_sum_with_filter ... ok
test exec::operators::aggregate::tests::sort_merge_aggregate_filter_clause ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1060 filtered out
```

All 67 aggregate-related tests pass:
```
test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 995 filtered out
```

## Issues Found

None. The implementation is correct and follows project conventions.

## Changes Made

None required. The coding agent's implementation is correct.

## Verdict

**APPROVED** - The fix correctly addresses the FILTER clause bug in `SortMergeAggregateOp`:
1. The FILTER handling matches `HashAggregateOp` exactly
2. A secondary double-accumulation bug was identified and fixed
3. Comprehensive tests were added to verify both COUNT and SUM with FILTER
4. All tests pass
5. Code follows project conventions

Ready to merge.
