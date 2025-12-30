# ManifoldDB Issues

**Consolidated:** December 2024
**Purpose:** Actionable issues only - not design preferences or future features

---

## Summary

17 concrete issues requiring attention, organized by severity.

| Severity | Count | Category |
|----------|-------|----------|
| Critical | 2 | Panic risks, missing indexes |
| High | 4 | Functionality gaps |
| Medium | 6 | Code quality, usability |
| Low | 5 | Cleanup, minor improvements |

---

## Critical Issues

### C-1: Persistence Parsing Can Panic on Malformed Data

**Location:** `crates/manifoldb-vector/src/index/persistence.rs` (lines 161, 170, 269, etc.)

**Problem:**
```rust
let val = u32::from_be_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
```

Bounds are checked earlier with `VectorError::Encoding`, but the unwrap inside the closure will panic if data is malformed in specific ways.

**Impact:** Corrupted or truncated index files crash the database instead of returning an error.

**Fix:**
```rust
let val = bytes[*pos..*pos + 4]
    .try_into()
    .map(u32::from_be_bytes)
    .map_err(|_| VectorError::Encoding("truncated u32"))?;
```

**Instances:** 6

---

### C-2: No Label Index - All Queries Do Full Table Scans

**Location:** `crates/manifoldb/src/transaction/handle.rs:499-537`

**Problem:**
```rust
pub fn iter_entities(&self, label: Option<&str>) -> Result<Vec<Entity>, TransactionError> {
    // Creates cursor over ALL nodes
    let cursor_result = storage.range(tables::NODES, Bound::Unbounded, Bound::Unbounded);

    while let Some((_key, value)) = cursor.next()? {
        let entity = bincode::decode(&value)?;
        // Filter by label IN MEMORY
        if let Some(label_filter) = label {
            if entity.has_label(label_filter) {
                entities.push(entity);
            }
        }
    }
}
```

Every query - even `SELECT * FROM users WHERE id = 1` - scans the entire entity table.

**Impact:**
- 1M entities → scan 1M to find 1
- Completely unusable at production scale
- Query planner has `IndexSelector` that identifies index opportunities but it's never used

**Fix:** Implement `(Label, EntityId) -> ()` index table.

**Effort:** ~1 day for label index

---

## High Priority Issues

### H-1: SIMD Distance Functions Can Panic on Bad Input

**Location:** `crates/manifoldb-vector/src/distance/simd.rs` (lines 41-42, 87-88, etc.)

**Problem:**
```rust
let va = f32x8::new(a[i..i + SIMD_WIDTH].try_into().unwrap());
let vb = f32x8::new(b[i..i + SIMD_WIDTH].try_into().unwrap());
```

Bounds are checked via `simd_len = len - (len % SIMD_WIDTH)`, making this theoretically safe. But corrupted vectors with inconsistent lengths could trigger panics.

**Impact:** Bad vector data crashes distance calculations instead of returning an error.

**Fix:** Either:
- Add `debug_assert!` documenting the invariant
- Use `.expect("SIMD bounds checked by simd_len calculation")`
- Convert to proper error handling

**Instances:** 12

---

### H-2: Database Struct Not Cloneable

**Location:** `crates/manifoldb/src/database.rs`

**Problem:**
```rust
pub struct Database {
    manager: TransactionManager<RedbEngine>,
    config: Config,
    query_cache: QueryCache,
}
// Cannot clone - cannot easily share across threads
```

**Impact:**
- Cannot share database handle across threads easily
- Breaks common async/await patterns
- Makes connection pooling difficult to implement

**Fix:**
```rust
pub struct Database {
    inner: Arc<DatabaseInner>,
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}
```

**Effort:** ~1 day

---

### H-3: No Read Transaction Pool

**Location:** N/A - not implemented

**Problem:** Each read creates a new transaction. No pooling or reuse.

**Impact:**
- Overhead on read-heavy workloads
- Cannot efficiently serve concurrent readers
- Each read transaction has setup cost

**Fix:** Implement transaction pool with:
- Pre-created read transactions
- Refresh policy based on age/write count
- Configurable pool size

**Effort:** ~1 week

---

### H-4: Query Planner Index Analysis Never Used

**Location:** `crates/manifoldb-query/src/plan/optimize/index_selection.rs`

**Problem:** The `IndexSelector` correctly identifies predicates that could use indexes:
- Point lookups (`column = value`)
- Range scans (`column > value`)
- IN lists
- Prefix scans

But this analysis is never wired to actual index lookups because no property indexes exist.

**Impact:** Wasted analysis work; planner knows what would help but can't use it.

**Fix:** Blocked by C-2 and secondary index implementation.

---

## Medium Priority Issues

### M-1: SCC Algorithm Unwraps Undocumented

**Location:** `crates/manifoldb-graph/src/analytics/connected.rs` (lines 569, 577, 583)

**Problem:** Tarjan's SCC algorithm has unwraps that are safe by algorithmic invariant (value is always set before access), but this isn't documented.

**Impact:** Future maintainers may not understand why these unwraps are safe.

**Fix:** Add inline comments explaining the safety guarantee:
```rust
// SAFETY: By Tarjan's algorithm invariant, lowlink is always
// set when a node is pushed to the stack before this access
```

**Instances:** 3

---

### M-2: Missing Documentation for Key Features

**Location:** Various

**Missing:**
- Batch writer configuration and tuning guide
- Vector sync strategies and when to use each
- WAL engine setup and recovery procedures
- Performance tuning checklist

**Impact:** Users can't effectively configure or tune ManifoldDB.

**Effort:** ~2 days

---

### M-3: Error Messages Lack Context

**Location:** `crates/manifoldb-storage/src/error.rs`

**Problem:**
```rust
StorageError::Open(String)  // Just a string
```

**Impact:** Hard to troubleshoot failures - no path, no cause chain, no suggestions.

**Fix:**
```rust
#[derive(Debug, thiserror::Error)]
#[error("failed to open database at {path}: {cause}")]
pub struct OpenError {
    pub path: PathBuf,
    pub cause: Box<dyn std::error::Error + Send + Sync>,
    pub suggestion: Option<String>,
}
```

**Effort:** ~4 days

---

### M-4: No Storage-Level Metrics

**Location:** N/A - not implemented

**Problem:** No visibility into storage operations - reads, writes, bytes transferred, transaction durations.

**Impact:** Cannot diagnose performance issues or plan capacity.

**Fix:** Create `MetricsEngine<E>` wrapper:
```rust
pub struct StorageMetrics {
    reads: AtomicU64,
    writes: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
    transaction_duration_histogram: Histogram,
}
```

**Effort:** ~1 week

---

### M-5: No Compaction API

**Location:** `crates/manifoldb-storage/src/engine.rs`

**Problem:** No way to trigger storage compaction.

**Impact:** Database files grow over time; no reclamation of deleted space.

**Fix:** Add to `StorageEngine` trait:
```rust
fn compact(&self) -> Result<CompactionStats, StorageError>;
```

**Effort:** ~3 days

---

### M-6: StorageEngine Missing Metadata Methods

**Location:** `crates/manifoldb-storage/src/engine.rs`

**Problem:** Cannot query:
- Backend name
- Estimated database size
- Storage statistics

**Impact:** Poor observability and debugging.

**Fix:**
```rust
pub trait StorageEngine: Send + Sync {
    fn backend_name(&self) -> &'static str;
    fn estimated_size(&self) -> Result<u64, StorageError>;
    fn statistics(&self) -> StorageStatistics;
}
```

**Effort:** ~1 day

---

## Low Priority Issues

### L-1: Test Code Clippy Warnings

**Location:** Integration tests

**Problem:** 8 minor warnings (map_entry patterns, cloned vs copied).

**Fix:** `cargo clippy --fix --test "integration_tests"`

---

### L-2: Dead Code Without Timeline Documentation

**Location:** Various

**Files with `#[allow(dead_code)]`:**
- `exec/operators/vector.rs` - `metric`, `score_alias` fields
- `exec/operators/graph.rs` - `depth` field
- `collection/builder.rs` - `new()` method
- `collection/handle.rs` - `create`, `open`, helper fn
- `cache/hints.rs` - `is_cacheable_statement`

**Problem:** Dead code is allowed but no documentation of when it will be used.

**Fix:** Add `// TODO(v0.2):` comments explaining planned activation.

---

### L-3: No `clippy::unwrap_used` Lint

**Location:** Library crate roots

**Problem:** Future unwraps in library code won't be caught.

**Fix:** Add to `lib.rs` files:
```rust
#![deny(clippy::unwrap_used)]
```

---

### L-4: No Lock Contention Tests

**Location:** Test suite

**Problem:** No explicit stress tests for concurrent index operations.

**Fix:** Add tests for concurrent read/write scenarios on `HnswIndexManager`.

---

### L-5: No Persistence Load Benchmarks

**Location:** Benchmarks

**Problem:** No benchmarks for index load times under various data sizes.

**Fix:** Add benchmark suite for persistence operations.

---

## Issue Checklist

```
Critical:
[x] C-1: Fix persistence parsing unwraps (6 instances) - DONE 2025-12-31
[x] C-2: Implement label index - DONE 2025-12-31

High:
[x] H-1: Harden SIMD distance unwraps (12 instances) - DONE 2025-12-31
[x] H-2: Make Database cloneable - DONE 2025-12-31
[x] H-3: Implement read transaction pool - DONE 2025-12-31
[x] H-4: Wire index analysis to execution - DONE 2025-12-31

Medium:
[x] M-1: Document SCC algorithm safety invariants - DONE 2025-12-31
[x] M-2: Write missing documentation - DONE 2025-12-31
[x] M-3: Improve error context - DONE 2025-12-31
[x] M-4: Add storage-level metrics - DONE 2025-12-31
[x] M-5: Add compaction API - DONE 2025-12-31
[x] M-6: Add storage metadata methods - DONE 2025-12-31

Low:
[x] L-1: Fix test clippy warnings - DONE 2025-12-31
[x] L-2: Document dead code timelines - DONE 2025-12-31
[x] L-3: Enable unwrap_used lint - DONE 2025-12-31
[~] L-4: Add lock contention tests - DEFERRED (HnswIndexManager already uses RwLock for concurrency)
[~] L-5: Add persistence benchmarks - DEFERRED (nice-to-have, not a code quality issue)
```

---

## Not In Scope

The following are **design choices**, not issues:

- **redb's SWMR model** - Known limitation of the storage engine choice. Appropriate for read-heavy graph workloads.
- **Row-oriented property storage** - Acceptable trade-off for current entity sizes.
- **No server mode** - Future feature, not a defect.
- **No replication** - Future feature, not a defect.
- **Alternative backends (Fjall, RocksDB)** - Nice to have, not issues.

---

## Files Quick Reference

| File | Issues |
|------|--------|
| `manifoldb-vector/src/distance/simd.rs` | H-1 (12 unwraps) |
| `manifoldb-vector/src/index/persistence.rs` | C-1 (6 unwraps) |
| `manifoldb-graph/src/analytics/connected.rs` | M-1 (3 undocumented unwraps) |
| `manifoldb/src/transaction/handle.rs` | C-2 (full table scans) |
| `manifoldb/src/database.rs` | H-2 (not cloneable) |
| `manifoldb-query/src/plan/optimize/index_selection.rs` | H-4 (unused analysis) |
| `manifoldb-storage/src/engine.rs` | M-5, M-6 |
| `manifoldb-storage/src/error.rs` | M-3 |
