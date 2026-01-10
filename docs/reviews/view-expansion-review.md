# View Expansion & Correlated Subqueries Review

**Task:** View Expansion & Correlated Subqueries
**Reviewed:** January 10, 2026
**Reviewer:** Code Review Agent

---

## 1. Summary

This review covers the implementation of view expansion in query planning and the infrastructure for correlated subquery variable binding. The implementation adds:

1. **View Execution** - CREATE VIEW and DROP VIEW statement handlers that store views with their raw SQL
2. **View Expansion** - Views are expanded inline during query planning when referenced as table sources
3. **Correlated CALL {} Infrastructure** - Variable bindings added to ExecutionContext for passing outer scope values to subqueries
4. **Comprehensive Tests** - 10 new integration tests for view operations

---

## 2. Files Changed

| File | Changes |
|------|---------|
| `crates/manifoldb-query/src/exec/context.rs` | Added `variable_bindings` field and related methods for correlated subqueries |
| `crates/manifoldb-query/src/exec/operators/call_subquery.rs` | Updated `open_subquery()` to create contexts with variable bindings from outer rows |
| `crates/manifoldb-query/src/plan/logical/builder.rs` | Added `ViewDefinition` struct and view expansion in `build_table_ref()` |
| `crates/manifoldb-query/src/plan/logical/mod.rs` | Re-exported `ViewDefinition` |
| `crates/manifoldb/src/execution/executor.rs` | Added `execute_create_view()`, `execute_drop_view()`, `extract_view_query_sql()`, and `load_views_into_builder()` |
| `crates/manifoldb/tests/integration/ddl.rs` | Added 10 new view tests |
| `COVERAGE_MATRICES.md` | Updated VIEW and CALL {} status |

---

## 3. Issues Found

**No critical issues found.** The implementation is well-structured and follows project conventions.

### Minor Observations (Not Issues)

1. **Clone Usage** - Several `.clone()` calls in the new code were reviewed:
   - `view.name.clone()` in `register_view()` - Required because name is used as HashMap key
   - `bindings.insert(var_name.clone(), value.clone())` - Required for ownership transfer
   - `.cloned()` on view_def lookup - Required to avoid borrow issues

   All clones are necessary and appropriate.

2. **Incomplete Features** (Documented in task as out of scope):
   - Correlated subqueries in WHERE clause (EXISTS, IN, scalar) - Infrastructure added but full execution not implemented
   - Advanced CTE/subquery scoping rules - Basic shadowing works

---

## 4. Changes Made

**None.** The implementation passed all quality checks.

---

## 5. Code Quality Verification

### Error Handling ✅
- No `unwrap()` or `expect()` in library code
- Errors use `.map_err()` with context (e.g., `Error::Execution(format!(...))`)
- `extract_view_query_sql()` returns proper `Result<String>` with descriptive errors

### Code Quality ✅
- No unnecessary clones (all reviewed and justified)
- No `unsafe` blocks
- `#[must_use]` applied to builder methods (`ViewDefinition::new()`, `with_columns()`)
- `#[inline]` used on hot-path accessors (`get_variable()`, `variable_bindings()`)

### Module Structure ✅
- `mod.rs` contains only declarations and re-exports
- `ViewDefinition` properly exported from `plan::logical::mod.rs`
- Implementation in appropriate files (not in mod.rs)

### Documentation ✅
- `ViewDefinition` has doc comments with examples
- `variable_bindings` field documented with purpose
- `open_subquery()` has doc comment explaining its role

---

## 6. Test Results

### All Tests Pass ✅

```
cargo fmt --all --check       # ✓ No formatting issues
cargo clippy --workspace      # ✓ No warnings
cargo test --workspace        # ✓ All tests pass
```

### View-Specific Tests (10 tests)

```
test integration::ddl::test_create_view_basic ... ok
test integration::ddl::test_create_view_with_projection ... ok
test integration::ddl::test_create_view_or_replace ... ok
test integration::ddl::test_create_view_with_columns ... ok
test integration::ddl::test_view_with_where_filter ... ok
test integration::ddl::test_drop_view_basic ... ok
test integration::ddl::test_drop_view_if_exists ... ok
test integration::ddl::test_drop_multiple_views ... ok
test integration::ddl::test_create_view_already_exists_error ... ok
test integration::ddl::test_drop_view_nonexistent_error ... ok
```

### CallSubquery Tests (4 tests)

```
test exec::operators::call_subquery::tests::call_subquery_uncorrelated_basic ... ok
test exec::operators::call_subquery::tests::call_subquery_empty_subquery ... ok
test exec::operators::call_subquery::tests::call_subquery_schema ... ok
test exec::operators::call_subquery::tests::call_subquery_is_uncorrelated ... ok
```

---

## 7. Implementation Analysis

### View Expansion Flow

1. `execute_create_view()` stores view with raw SQL in schema
2. `load_views_into_builder()` loads views at query plan time
3. `PlanBuilder.register_view()` adds view to `view_definitions` HashMap
4. `build_table_ref()` checks: CTE (highest priority) → View → Table
5. View's SELECT statement is recursively built as the plan

### Correlated Subquery Infrastructure

1. `ExecutionContext.variable_bindings` - HashMap<String, Value> for outer scope variables
2. `with_variable_bindings()`, `bind_variable()`, `get_variable()` - Methods to manage bindings
3. `CallSubqueryOp.open_subquery()` - Creates new context with bindings extracted from outer row
4. Variables imported via `WITH p` in CALL {} are now bound when opening subquery

### Shadowing Semantics

The implementation correctly implements SQL shadowing rules:
- CTEs shadow both views and tables (highest priority)
- Views shadow tables (second priority)
- Tables are the fallback

---

## 8. Verdict

✅ **Approved**

The implementation is complete, well-tested, and follows all project coding standards. No issues require fixing. The task requirements are met:

- [x] View expansion in queries - Complete with tests
- [x] Correlated CALL {} variable binding - Infrastructure complete
- [ ] Correlated subqueries in WHERE - Infrastructure added (documented as out of scope)
- [x] CTE/view shadowing semantics - Complete

The implementation is ready to merge.

---

*Review completed: January 10, 2026*
