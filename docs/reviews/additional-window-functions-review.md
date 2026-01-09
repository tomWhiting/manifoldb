# Review: Additional Window Functions (NTILE, PERCENT_RANK, CUME_DIST)

**Reviewer:** Claude Code Review Agent
**Date:** January 10, 2026
**Task:** Implement Additional Window Functions (ntile, percent_rank, cume_dist)
**Branch:** `vk/705d-implement-additi`

---

## 1. Summary

This review covers the implementation of three additional SQL window ranking/distribution functions:

1. **NTILE(n)** - Divides rows into n buckets numbered 1 to n
2. **PERCENT_RANK()** - Relative rank as percentage (0 to 1), formula: (rank - 1) / (total_rows - 1)
3. **CUME_DIST()** - Cumulative distribution, formula: rows_up_to_current / total_rows

All three functions are fully implemented with proper handling of:
- Partition boundaries
- Peer groups (ties) for PERCENT_RANK and CUME_DIST
- Edge cases (single row, empty partitions, more buckets than rows)

---

## 2. Files Changed

### Modified Files

| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/ast/expr.rs:340-357` | Added `Ntile { n }`, `PercentRank`, `CumeDist` variants to `WindowFunction` enum |
| `crates/manifoldb-query/src/ast/expr.rs:392-394` | Added Display impl for new variants |
| `crates/manifoldb-query/src/plan/logical/builder.rs:692-741` | Added parsing for NTILE, PERCENT_RANK, CUME_DIST; added `get_window_ntile_arg()` helper |
| `crates/manifoldb-query/src/plan/logical/expr.rs:1333-1341` | Added Display impl for logical expression formatting |
| `crates/manifoldb-query/src/exec/operators/window.rs:155-163` | Added match arms for dispatch to compute functions |
| `crates/manifoldb-query/src/exec/operators/window.rs:584-763` | Added `compute_ntile()`, `compute_percent_rank()`, `compute_cume_dist()` implementations |
| `crates/manifoldb-query/src/exec/operators/window.rs:2816-3347` | Added 12 unit tests for the new functions |
| `COVERAGE_MATRICES.md:230-232` | Updated status for ntile, percent_rank, cume_dist to complete |

---

## 3. Issues Found

**No issues found.** The implementation is complete and follows project standards.

---

## 4. Changes Made

No changes were necessary. The original implementation passes all quality checks.

---

## 5. Code Quality Verification

### 5.1 Error Handling

- ✅ No `unwrap()` or `expect()` in library code (only in test code starting line 1198)
- ✅ Proper null handling for edge cases (NTILE(0) returns NULL, empty partitions return appropriate values)
- ✅ Argument validation in `get_window_ntile_arg()` returns `PlanError::Unsupported` for invalid inputs

### 5.2 Code Quality

- ✅ No unnecessary `.clone()` calls
- ✅ No `unsafe` blocks
- ✅ Proper use of `#[must_use]` on the `WindowOp::new()` constructor
- ✅ Documentation comments on all public APIs

### 5.3 Module Structure

- ✅ Implementation in proper file (`window.rs`), not in `mod.rs`
- ✅ Functions organized logically with ranking functions grouped together

### 5.4 Algorithms

The implementations follow PostgreSQL semantics:

**NTILE(n):**
- Correctly handles uneven distribution (earlier buckets get extra rows)
- Handles edge case when n > partition_size (assigns buckets 1 to partition_size)
- Returns NULL for NTILE(0)

**PERCENT_RANK():**
- Formula: (rank - 1) / (total_rows - 1)
- Returns 0.0 for single-row partitions (avoids division by zero)
- Correctly handles ties (peers get same percent_rank)

**CUME_DIST():**
- Formula: rows_up_to_current / total_rows
- Correctly includes peer rows in the count
- Uses `find_peers_end()` helper to locate end of peer group

---

## 6. Test Results

### 6.1 Formatting Check

```bash
$ cargo fmt --all -- --check
# No output (all files formatted correctly)
```

### 6.2 Clippy Check

```bash
$ cargo clippy --workspace --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 20.34s
```

### 6.3 Test Results

```bash
$ cargo test --workspace -p manifoldb-query -- window
test result: ok. 59 passed; 0 failed; 0 ignored; 0 measured; 657 filtered out
```

New tests added (12 total):

**NTILE (4 tests):**
- `ntile_basic` - NTILE(4) over 4 rows, each gets different bucket
- `ntile_uneven_distribution` - NTILE(3) over 4 rows, uneven distribution
- `ntile_with_partition` - NTILE with PARTITION BY
- `ntile_more_buckets_than_rows` - NTILE(10) over 4 rows

**PERCENT_RANK (4 tests):**
- `percent_rank_basic` - Values 0, 1/3, 2/3, 1 for 4 rows
- `percent_rank_with_ties` - Tied rows get same percent_rank
- `percent_rank_single_row` - Single row returns 0.0
- `percent_rank_with_partition` - PERCENT_RANK with PARTITION BY

**CUME_DIST (4 tests):**
- `cume_dist_basic` - Values 0.25, 0.5, 0.75, 1.0 for 4 rows
- `cume_dist_with_ties` - Tied rows get same cume_dist (end of peer group)
- `cume_dist_single_row` - Single row returns 1.0
- `cume_dist_with_partition` - CUME_DIST with PARTITION BY

### 6.4 Full Test Suite

```bash
$ cargo test --workspace
# All tests pass
```

---

## 7. Coverage Matrix Update

The `COVERAGE_MATRICES.md` file was correctly updated:

```markdown
| ntile(n) | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| percent_rank() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
| cume_dist() | ✓ | ✓ | ✓† | ✓† | ✓† | ✓† | Agent impl Jan 2026 |
```

---

## 8. Verdict

✅ **Approved** - No issues found, ready to merge.

The implementation:
- Follows PostgreSQL semantics for all three functions
- Handles edge cases correctly (single row, ties, more buckets than rows)
- Has comprehensive test coverage (12 new tests)
- Passes all quality checks (fmt, clippy, tests)
- Properly integrates with existing window function infrastructure
- Updates documentation (COVERAGE_MATRICES.md)

---

## 9. References

- PostgreSQL NTILE: https://www.postgresql.org/docs/current/functions-window.html
- PostgreSQL PERCENT_RANK: https://www.postgresql.org/docs/current/functions-window.html
- PostgreSQL CUME_DIST: https://www.postgresql.org/docs/current/functions-window.html
