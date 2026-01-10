# Constraint Enforcement Review

**Task:** Implement constraint enforcement (FOREIGN KEY, CHECK)
**Reviewed:** 2026-01-10
**Verdict:** **Approved**

---

## Summary

This review covers the implementation of runtime constraint enforcement for CHECK and FOREIGN KEY constraints during DML operations (INSERT, UPDATE, DELETE). The implementation correctly enforces constraints that were previously only parsed and stored in schema metadata.

## Files Changed

### New Files

1. **`crates/manifoldb/src/execution/constraints.rs`** (516 lines)
   - `ConstraintError` enum for constraint violation errors
   - `ConstraintValidator` struct with validation methods:
     - `validate_insert()` - CHECK and FK validation before INSERT
     - `validate_update()` - CHECK and FK validation before UPDATE
     - `validate_delete()` - FK reference checking before DELETE (RESTRICT behavior)
   - Helper methods for parsing expressions, checking references, and comparing values

2. **`crates/manifoldb/tests/integration/constraints.rs`** (357 lines)
   - 12 integration tests covering:
     - CHECK constraint validation on INSERT/UPDATE (success/failure cases)
     - Complex CHECK expressions (e.g., `age >= 18 AND age <= 120`)
     - FOREIGN KEY validation on INSERT (success/failure)
     - FOREIGN KEY NULL handling (NULLs allowed per SQL standard)
     - FOREIGN KEY RESTRICT behavior on DELETE
     - Table-level CHECK and FOREIGN KEY constraints

### Modified Files

1. **`crates/manifoldb/src/execution/mod.rs`**
   - Added `pub mod constraints;` declaration

2. **`crates/manifoldb/src/execution/executor.rs`**
   - Added `evaluate_predicate_for_constraint()` public function (line 4457)
   - Integrated constraint validation in `execute_insert()` (line 2344)
   - Integrated constraint validation in `execute_update()` (line 2511)
   - Integrated constraint validation in `execute_delete()` (line 2587)

3. **`crates/manifoldb-query/src/ast/expr.rs`**
   - Added `to_sql()` method (line 1037) for serializing expressions to SQL strings

4. **`crates/manifoldb-query/src/parser/sql.rs`**
   - Added `parse_check_expression()` function (line 79) for parsing CHECK expressions

5. **`crates/manifoldb-query/src/parser/mod.rs`**
   - Re-exported `parse_check_expression` (line 74)

6. **`crates/manifoldb/src/schema/mod.rs`**
   - Updated `StoredColumnConstraint::from_constraint()` (line 265) to use `expr.to_sql()`
   - Updated `StoredTableConstraint::from_table_constraint()` (line 297) to use `expr.to_sql()`

7. **`crates/manifoldb/tests/integration/mod.rs`**
   - Added `pub mod constraints;` declaration (line 12)

## Code Quality Analysis

### Error Handling

| Check | Status |
|-------|--------|
| No `unwrap()` in library code | **PASS** |
| No `expect()` in library code | **PASS** |
| No `panic!()` in library code | **PASS** |
| Errors have context | **PASS** - Uses `map_err()` with descriptive messages |

The implementation uses proper Result types and error enums throughout. The `ConstraintError` enum provides specific error variants for different violation types with meaningful messages.

### Memory & Performance

| Check | Status |
|-------|--------|
| No unnecessary `.clone()` | **PASS** |
| Uses references appropriately | **PASS** |
| Iterators over collect where possible | **PASS** |

The code avoids unnecessary allocations. The `values_equal()` function takes references. The validator methods take references to avoid ownership issues.

### Module Organization

| Check | Status |
|-------|--------|
| `mod.rs` for declarations only | **PASS** - execution/mod.rs only has module declarations |
| Implementation in named files | **PASS** - constraints.rs contains implementation |
| Consistent naming | **PASS** |

### Testing

| Check | Status |
|-------|--------|
| Unit tests for new functionality | **PASS** - `test_values_equal` in constraints.rs |
| Integration tests for workflows | **PASS** - 12 tests in constraints.rs |
| Edge cases covered | **PASS** - NULL handling, complex expressions |

### Tooling

| Check | Status |
|-------|--------|
| `cargo fmt --all` passes | **PASS** |
| `cargo clippy --workspace --all-targets -- -D warnings` passes | **PASS** |
| `cargo test --workspace` passes | **PASS** - All 1900+ tests pass |

## Issues Found

**None.** The implementation follows all coding standards and project conventions.

## Architecture Notes

### Design Decisions

1. **Expression Storage**: CHECK constraints are stored as SQL strings and re-parsed at validation time. This approach:
   - Simplifies serialization (no complex AST serialization needed)
   - Allows schema to be human-readable
   - Has minor runtime overhead for parsing, acceptable for constraint validation

2. **FK Validation Strategy**: The implementation scans entities to check references rather than using indexes. This is correct for the current state but could be optimized with payload indexes in the future.

3. **DELETE Behavior**: Currently only RESTRICT is implemented (fails if referenced). CASCADE and SET NULL are documented as future work.

4. **NULL Handling**: NULLs in FK columns are allowed per SQL standard, correctly implemented.

### Crate Boundaries

The implementation respects crate boundaries:
- `manifoldb-query/src/ast/expr.rs` - Added `to_sql()` for AST serialization
- `manifoldb-query/src/parser/sql.rs` - Added expression parsing
- `manifoldb/src/execution/constraints.rs` - Constraint validation logic
- `manifoldb/src/schema/mod.rs` - Uses `to_sql()` for storage

## Test Results

```
running 14 tests
test integration::constraints::test_foreign_key_on_delete_restrict ... ok
test integration::constraints::test_foreign_key_on_insert_failure ... ok
test integration::constraints::test_foreign_key_on_insert_success ... ok
test integration::constraints::test_check_constraint_on_update_failure ... ok
test integration::constraints::test_check_constraint_on_update_success ... ok
test integration::constraints::test_check_constraint_on_insert_success ... ok
test integration::constraints::test_check_constraint_complex_expression ... ok
test integration::constraints::test_check_constraint_on_insert_failure ... ok
test integration::constraints::test_foreign_key_on_delete_no_references ... ok
test integration::constraints::test_foreign_key_null_allowed ... ok
test integration::constraints::test_table_level_check_constraint ... ok
test integration::ddl::test_create_table_with_constraints ... ok
test integration::ddl::test_create_table_with_table_constraints ... ok
test integration::constraints::test_table_level_foreign_key ... ok

test result: ok. 14 passed; 0 failed; 0 ignored; 0 measured; 582 filtered out
```

## Verdict

**Approved** - The implementation is complete, follows all coding standards, passes all tests, and correctly implements the required constraint enforcement behavior.

### Future Considerations

1. **CASCADE/SET NULL on DELETE**: Currently only RESTRICT is implemented. These could be added as future enhancements.
2. **Index-based FK lookups**: The current implementation scans entities. Payload indexes could improve performance for large tables.
3. **Deferred constraints**: PostgreSQL supports deferred constraint checking. Not currently needed but worth considering for complex transactions.
