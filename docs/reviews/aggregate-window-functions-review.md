# Aggregate Window Functions Review

**Task:** Implement Aggregates as Window Functions (SUM/AVG/COUNT/MIN/MAX OVER)
**Reviewed:** January 2026
**Reviewer:** Claude Code

---

## Summary

This implementation adds support for using standard aggregate functions (SUM, AVG, COUNT, MIN, MAX) as window functions with the OVER clause. This enables common patterns like running totals, moving averages, and cumulative counts.

Example queries enabled:
```sql
-- Running total
SELECT date, amount, SUM(amount) OVER (ORDER BY date) AS running_total FROM sales;

-- Moving average
SELECT date, value, AVG(value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS weekly_avg FROM metrics;

-- Cumulative count per department
SELECT dept, date, COUNT(*) OVER (PARTITION BY dept ORDER BY date) AS dept_count FROM events;
```

---

## Files Changed

### New Types

| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/ast/expr.rs:349-390` | Added `AggregateWindowFunction` enum and `WindowFunction::Aggregate` variant |

### Logical Plan

| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/plan/logical/expr.rs:662-763` | Added constructor methods: `sum_window`, `avg_window`, `count_window`, `min_window`, `max_window` |
| `crates/manifoldb-query/src/plan/logical/expr.rs:1033-1040` | Updated `Display` impl for aggregate window functions |
| `crates/manifoldb-query/src/plan/logical/builder.rs:603-662` | Added parsing for SUM, AVG, COUNT, MIN, MAX with OVER clauses |

### Execution

| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/exec/operators/window.rs:155-157` | Added dispatch for `WindowFunction::Aggregate` |
| `crates/manifoldb-query/src/exec/operators/window.rs:358-411` | Implemented `compute_aggregate_window()` method |
| `crates/manifoldb-query/src/exec/operators/window.rs:414-573` | Implemented helper methods: `compute_count`, `compute_sum`, `compute_avg`, `compute_min`, `compute_max` |

### Documentation

| File | Changes |
|------|---------|
| `COVERAGE_MATRICES.md` | Updated status to mark aggregate window functions as complete |

---

## Issues Found

None. The implementation is complete and follows all coding standards.

---

## Code Quality Checklist

### Error Handling
- [x] No `unwrap()` calls in library code (only in tests)
- [x] No `expect()` calls in library code (only in tests)
- [x] Proper error handling with `?` operator

### Memory & Performance
- [x] No unnecessary `.clone()` calls
- [x] Values collected once per frame, then aggregated
- [x] Frame boundaries computed efficiently

### Safety
- [x] No `unsafe` blocks
- [x] Input validation at boundaries
- [x] NULL handling throughout aggregate computations

### Module Structure
- [x] Types defined in appropriate files (ast/expr.rs, plan/logical/expr.rs)
- [x] Re-exports in mod.rs files
- [x] Clear separation between AST, plan, and execution

### Testing
- [x] 19 new tests for aggregate window functions
- [x] Tests cover all aggregate functions (SUM, AVG, COUNT, MIN, MAX)
- [x] Tests cover running totals, moving averages, cumulative aggregates
- [x] Tests cover PARTITION BY with aggregates
- [x] Tests cover NULL value handling
- [x] Tests cover COUNT(*) vs COUNT(expr) distinction
- [x] Tests cover frame clauses with aggregates

### Tooling
- [x] `cargo fmt --all` passes
- [x] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [x] `cargo test --workspace` passes (666+ tests)

---

## Test Results

```
running 47 tests (window-related)

test exec::operators::window::tests::sum_running_total ... ok
test exec::operators::window::tests::sum_with_partition ... ok
test exec::operators::window::tests::sum_with_null_values ... ok
test exec::operators::window::tests::avg_moving_average ... ok
test exec::operators::window::tests::avg_with_entire_partition_frame ... ok
test exec::operators::window::tests::count_cumulative ... ok
test exec::operators::window::tests::count_with_expression ... ok
test exec::operators::window::tests::min_cumulative ... ok
test exec::operators::window::tests::max_cumulative ... ok
test plan::logical::builder::tests::window_sum_running_total ... ok
test plan::logical::builder::tests::window_avg_moving_average ... ok
test plan::logical::builder::tests::window_count_cumulative ... ok
test plan::logical::builder::tests::window_count_with_expression ... ok
test plan::logical::builder::tests::window_min_cumulative ... ok
test plan::logical::builder::tests::window_max_cumulative ... ok
test plan::logical::builder::tests::window_aggregate_with_partition ... ok
test plan::logical::builder::tests::window_multiple_aggregates ... ok
test plan::logical::builder::tests::window_mixed_ranking_and_aggregate ... ok
... (all ranking and value function tests also pass)

test result: ok. 47 passed; 0 failed; 0 ignored
```

Full test suite: 666+ tests pass.

---

## Implementation Details

### Aggregate Window Function Types

The `AggregateWindowFunction` enum was added to `ast/expr.rs`:

```rust
pub enum AggregateWindowFunction {
    Count,  // COUNT(*) or COUNT(expr) over window
    Sum,    // SUM(expr) over window
    Avg,    // AVG(expr) over window
    Min,    // MIN(expr) over window
    Max,    // MAX(expr) over window
}
```

### Frame Awareness

All aggregate window functions correctly respect the window frame clause:

1. **Default frame** (with ORDER BY): `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` - cumulative aggregates
2. **Default frame** (no ORDER BY): Entire partition - total aggregates
3. **Custom frame**: e.g., `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` - moving averages

### NULL Handling

- `SUM`, `AVG`, `MIN`, `MAX` - Skip NULL values in calculations
- `COUNT(*)` - Counts all rows including NULLs
- `COUNT(expr)` - Counts only non-NULL values
- `AVG` - Returns NULL if no non-NULL values in frame

### Type Handling

- Integer-only frames return `Value::Int` for SUM, MIN, MAX
- Mixed int/float frames return `Value::Float` for SUM, MIN, MAX
- `AVG` always returns `Value::Float`
- String comparisons supported for MIN/MAX

---

## Verdict

**✅ Approved**

The implementation is complete, well-tested, and follows all project coding standards. The aggregate window functions:

1. Work correctly with PARTITION BY
2. Respect window frame clauses
3. Handle NULL values properly
4. Are properly documented in COVERAGE_MATRICES.md

No changes were required during review.

---

*Review completed: January 2026*
