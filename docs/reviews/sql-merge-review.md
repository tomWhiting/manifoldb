# SQL MERGE Statement Implementation Review

**Reviewer:** Claude Code Review Agent
**Date:** January 10, 2026
**Task:** Implement SQL MERGE statement

---

## 1. Summary

This review covers the implementation of the SQL MERGE statement for ManifoldDB. The MERGE statement enables conditional INSERT/UPDATE/DELETE operations based on whether source rows match target rows.

**Syntax Supported:**
```sql
MERGE INTO target_table t
USING source_table s
ON t.id = s.id
WHEN MATCHED AND condition THEN UPDATE SET t.col = s.col
WHEN MATCHED THEN DELETE
WHEN NOT MATCHED THEN INSERT (col1, col2) VALUES (val1, val2)
WHEN NOT MATCHED BY SOURCE THEN DELETE
```

---

## 2. Files Changed

### AST Layer (`manifoldb-query/src/ast/`)

| File | Change |
|------|--------|
| `statement.rs` | Added `MergeSqlStatement`, `MergeClause`, `MergeMatchType`, `MergeAction` types |
| `mod.rs` | Re-exported new MERGE types |

### Parser Layer (`manifoldb-query/src/parser/`)

| File | Change |
|------|--------|
| `sql.rs` | Added `convert_merge()`, `convert_merge_clause()`, `convert_merge_action()` functions |

### Logical Plan Layer (`manifoldb-query/src/plan/logical/`)

| File | Change |
|------|--------|
| `node.rs` | Added `MergeSql` variant, `LogicalMergeClause`, `LogicalMergeMatchType`, `LogicalMergeAction` types |
| `builder.rs` | Added `build_merge_sql()`, `build_merge_clause()` functions |
| `mod.rs` | Re-exported new types |

### Physical Plan Layer (`manifoldb-query/src/plan/physical/`)

| File | Change |
|------|--------|
| `node.rs` | Added `MergeSql` variant to `PhysicalPlan` |
| `builder.rs` | Added `plan_merge_sql()` function |
| `mod.rs` | No changes needed |

### Execution Layer

| File | Change |
|------|--------|
| `manifoldb-query/src/exec/executor.rs` | Added `MergeSql` case returning EmptyOp |
| `manifoldb/src/execution/executor.rs` | Added `MergeSql` case returning empty Vec |
| `manifoldb/src/execution/table_extractor.rs` | Added `MergeSql` case to extract tables |

### Tests

| File | Change |
|------|--------|
| `manifoldb-query/tests/parser_tests.rs` | Added 6 MERGE parsing tests |

### Documentation

| File | Change |
|------|--------|
| `COVERAGE_MATRICES.md` | Updated to show MERGE as complete |

---

## 3. Issues Found

### 3.1 CRITICAL: Execution Not Implemented

**Location:** `manifoldb/src/execution/executor.rs:1932-1933`

```rust
// MERGE SQL is handled separately by execute_merge_sql
LogicalPlan::MergeSql { .. } => Ok(Vec::new()),
```

The comment claims MERGE is "handled separately by execute_merge_sql" but **no such function exists**. The MERGE statement currently:
1. Parses correctly ✅
2. Builds logical plan correctly ✅
3. Builds physical plan correctly ✅
4. **Does nothing at execution time** ❌

The physical plan is constructed with all the correct information (target table, source plan, ON condition, WHEN clauses), but when it reaches execution, it simply returns an empty result.

**Impact:** Any user attempting to use MERGE will get no error, but also no data modification will occur. This is a silent failure.

### 3.2 Documentation Incorrectly Claims Execution Complete

**Location:** `COVERAGE_MATRICES.md:323-328`

The coverage matrix shows:
```
| MERGE INTO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
```

The "E" column (Execution) shows ✓ but execution is **not implemented**.

### 3.3 Missing Integration Tests

While there are 6 parser tests, there are **no integration tests** that verify MERGE actually modifies data. Given the execution gap, this is expected - tests would fail.

---

## 4. Changes Made

### 4.1 Fixed Documentation

Updated `COVERAGE_MATRICES.md` to accurately reflect the implementation status:

**Before:**
```
| MERGE INTO | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | Complete (Jan 2026) |
```

**After:**
```
| MERGE INTO | ✓ | ✓ | ✓ | ✓ | | | Execution not implemented |
```

---

## 5. Test Results

### Parser Tests (6/6 passing)
```
test merge_sql::parse_simple_merge_update ... ok
test merge_sql::parse_merge_insert ... ok
test merge_sql::parse_merge_delete ... ok
test merge_sql::parse_merge_multiple_clauses ... ok
test merge_sql::parse_merge_with_condition ... ok
test merge_sql::parse_merge_with_subquery_source ... ok
```

### Clippy
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 27.72s
```
No warnings.

### Format Check
No formatting issues.

### Full Test Suite
All workspace tests pass.

---

## 6. Code Quality Assessment

### Error Handling ✅
- No `unwrap()` or `expect()` in library code
- Proper error context via `PlanError::Unsupported`

### Code Style ✅
- Follows existing patterns in codebase
- Uses builder methods with `#[must_use]`
- Proper documentation comments

### Module Structure ✅
- `mod.rs` contains only declarations and re-exports
- Implementation in named files

### Testing ⚠️
- Parser tests present
- **No integration tests** (would fail due to missing execution)

---

## 7. Verdict

### ⚠️ **Needs Manual Review**

**Why This Cannot Be Auto-Approved:**

The implementation is **incomplete**. While the parser, AST, logical plan, and physical plan are correctly implemented, the actual execution is missing. This means:

1. **MERGE statements will silently do nothing** - No error, no data modification
2. **The documentation was incorrectly updated** - Claims execution is complete when it isn't

**What Works:**
- Parsing MERGE statements ✅
- Building logical plans ✅
- Building physical plans ✅
- 6 parser tests ✅

**What Doesn't Work:**
- Actual execution of MERGE operations ❌
- Integration tests ❌

**Recommendation:**

The task should be considered **partially complete**. The remaining work is:

1. Implement `execute_merge_sql()` function in `manifoldb/src/execution/executor.rs`
2. Add integration tests that verify MERGE modifies data correctly
3. Update documentation accurately once execution is implemented

The current implementation provides the foundation (parsing, planning) but the execution logic that actually performs INSERT/UPDATE/DELETE operations based on the WHEN clauses is not present.

---

## Appendix: Implementation Guidance for Execution

The `execute_merge_sql` function would need to:

1. Execute the source query to get source rows
2. For each source row:
   - Check if it matches any target row (using ON condition)
   - If matched: apply first matching WHEN MATCHED clause
   - If not matched: apply WHEN NOT MATCHED clause
3. For each target row not matched by source:
   - Apply WHEN NOT MATCHED BY SOURCE clause (if present)

Reference implementation patterns:
- `execute_insert()` in the same file for INSERT logic
- `execute_update()` for UPDATE logic
- `execute_delete()` for DELETE logic

The physical plan already contains all necessary information:
- `target_table: String`
- `source: Box<PhysicalPlan>` (can be executed to get rows)
- `on_condition: LogicalExpr`
- `clauses: Vec<LogicalMergeClause>`
