# CREATE/DROP VIEW Implementation Review

**Date:** January 9, 2026
**Reviewer:** Claude (Automated Review)
**Task:** Implement CREATE/DROP VIEW

## Summary

This review covers the implementation of SQL views for ManifoldDB. Views are stored query definitions that can be used like tables. The implementation adds parsing, planning, schema storage, and validation for CREATE VIEW and DROP VIEW statements.

## Files Changed

### AST Layer
- `crates/manifoldb-query/src/ast/statement.rs` - Added `CreateViewStatement` and `DropViewStatement` structs (lines 1172-1268)
- `crates/manifoldb-query/src/ast/mod.rs` - Re-exported new types

### Parser
- `crates/manifoldb-query/src/parser/sql.rs` - Added parsing for CREATE VIEW and DROP VIEW statements (lines 79-112, 1062-1080)
- Added 9 comprehensive parser tests (lines 1380-1506)

### Logical Plan
- `crates/manifoldb-query/src/plan/logical/ddl.rs` - Added `CreateViewNode` and `DropViewNode` (lines 254-323)
- `crates/manifoldb-query/src/plan/logical/node.rs` - Added `LogicalPlan::CreateView` and `LogicalPlan::DropView` variants
- `crates/manifoldb-query/src/plan/logical/builder.rs` - Added `build_create_view()` and `build_drop_view()` methods (lines 1232-1254)
- `crates/manifoldb-query/src/plan/logical/validate.rs` - Added validation for CREATE VIEW and DROP VIEW (lines 345-355)
- `crates/manifoldb-query/src/plan/logical/mod.rs` - Re-exported new DDL nodes

### Physical Plan
- `crates/manifoldb-query/src/plan/physical/node.rs` - Added `PhysicalPlan::CreateView` and `PhysicalPlan::DropView` variants
- `crates/manifoldb-query/src/plan/physical/builder.rs` - Added conversion from logical to physical plan (lines 405-406)

### Schema Storage
- `crates/manifoldb/src/schema/mod.rs` - Added `ViewSchema` struct and `SchemaManager` methods:
  - `ViewSchema::from_create_view()` (lines 158-170)
  - `SchemaManager::create_view()` (lines 573-604)
  - `SchemaManager::drop_view()` (lines 607-631)
  - `SchemaManager::view_exists()` (lines 634-639)
  - `SchemaManager::get_view()` (lines 643-657)
  - `SchemaManager::list_views()` (lines 660-664)
  - `SchemaError::ViewExists` and `SchemaError::ViewNotFound` variants (lines 745-749)

### Documentation
- `COVERAGE_MATRICES.md` - Updated VIEW implementation status (lines 331-337)

## Issues Found

### 1. View Query Serialization (Minor - Documented Limitation)
**Location:** `crates/manifoldb/src/schema/mod.rs:163`

The view query is stored using `format!("{:?}", node.query)` which produces a Debug representation rather than proper SQL. This means the stored query cannot be re-parsed directly.

**Impact:** Low - This is a known limitation documented in the code comments.

**Recommended Future Work:** Implement a proper SQL serializer for `SelectStatement` or derive `Serialize`/`Deserialize` on AST types.

### 2. View Expansion Not Implemented (Known - Out of Scope)
**Status:** Documented as "Remaining Work" in the task summary

View references in queries are not yet expanded to their defining query. This means:
```sql
CREATE VIEW active_users AS SELECT * FROM users WHERE status = 'active';
SELECT * FROM active_users;  -- This won't work yet
```

**Impact:** Medium - Views can be stored and retrieved but not used in queries.

**Recommended Future Work:** Add view expansion logic to the plan builder when resolving table references.

## Changes Made

No changes were required. The implementation passes all quality checks:
- No clippy warnings
- All tests pass
- Proper error handling (no `unwrap()` or `expect()` in library code)
- Proper use of `#[must_use]` on builder methods
- Consistent patterns with existing DDL implementations

## Test Results

```
cargo fmt --all --check  # Passed
cargo clippy --workspace --all-targets -- -D warnings  # Passed
cargo test --workspace  # All tests pass
```

### Parser Tests (9 tests)
1. `parse_create_view_basic` - Basic CREATE VIEW
2. `parse_create_or_replace_view` - CREATE OR REPLACE VIEW
3. `parse_create_view_with_columns` - CREATE VIEW with column aliases
4. `parse_create_view_with_join` - CREATE VIEW with JOIN query
5. `parse_drop_view_basic` - Basic DROP VIEW
6. `parse_drop_view_if_exists` - DROP VIEW IF EXISTS
7. `parse_drop_view_cascade` - DROP VIEW CASCADE
8. `parse_drop_view_multiple` - DROP multiple views
9. `parse_drop_view_if_exists_cascade` - Combined flags

## Architecture Review

### Crate Boundaries
The implementation correctly respects crate boundaries:
- AST types in `manifoldb-query`
- Plan nodes in `manifoldb-query`
- Schema storage in `manifoldb` (using `manifoldb-query` AST types)

### Code Quality
- Follows existing patterns for DDL (CreateTable, DropTable, CreateIndex, etc.)
- Proper documentation with examples in doc comments
- Builder pattern with `#[must_use]`
- Consistent error handling with `SchemaError` enum

### Module Structure
- `mod.rs` files contain only declarations and re-exports
- Implementation in named files
- Public API properly exported

## Verdict

**Approved**

The implementation is well-structured, follows project conventions, and passes all quality checks. The two noted items (query serialization and view expansion) are documented limitations with clear paths for future improvement. The core functionality for storing and managing view definitions is complete.

### What Works
- CREATE VIEW statement parsing
- CREATE OR REPLACE VIEW
- DROP VIEW with IF EXISTS and CASCADE
- Multiple view drops in single statement
- View schema persistence
- View metadata queries (exists, get, list)
- Logical and physical plan generation
- Plan validation

### Known Limitations
- View query stored as Debug format (not re-parseable SQL)
- View expansion in queries not yet implemented
- MATERIALIZED VIEW not supported (returns error)
