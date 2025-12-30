# ManifoldDB Improvement Roadmap

**Analysis Date:** December 2024
**Author:** Garak Fork (bc4ef5b7-b364-49f9-8bb7-0bcc538399d2)

---

## Overview

This roadmap prioritizes improvements to ManifoldDB based on:
1. **Impact**: Benefit to users and performance
2. **Effort**: Development time and complexity
3. **Risk**: Potential for regressions or instability
4. **Dependencies**: Prerequisites and blockers

Each item is categorized by priority tier and includes concrete recommendations.

---

## Priority Tiers

### Tier 1: Quick Wins (1-3 days each)

Low-effort changes with immediate benefit.

#### 1.1 Make Database Cloneable

**Status:** Not implemented
**Effort:** ~1 day
**Impact:** High (enables multi-threaded usage patterns)

**Current State:**
```rust
pub struct Database {
    manager: TransactionManager<RedbEngine>,
    config: Config,
    query_cache: QueryCache,
    // ...
}
```

**Recommended Change:**
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

**Benefits:**
- Easy sharing across threads
- Enables async/await patterns
- Foundation for connection pooling

---

#### 1.2 Storage Engine Metadata Methods

**Status:** Partially implemented
**Effort:** ~1 day
**Impact:** Medium (better diagnostics)

**Add to `StorageEngine` trait:**
```rust
pub trait StorageEngine: Send + Sync {
    // Existing methods...

    fn backend_name(&self) -> &'static str;
    fn estimated_size(&self) -> Result<u64, StorageError>;
    fn statistics(&self) -> StorageStatistics;
}
```

**Benefits:**
- Better observability
- Debugging support
- Capacity planning

---

#### 1.3 Document Existing Features

**Status:** Partially documented
**Effort:** ~2 days
**Impact:** High (user adoption, contributor onboarding)

**Missing Documentation:**
- Batch writer configuration and tuning guide
- Vector sync strategies and when to use each
- WAL engine setup and recovery procedures
- Performance tuning checklist

---

### Tier 2: Short-Term Improvements (1-2 weeks each)

Medium-effort changes with significant benefit.

#### 2.1 Read Transaction Pool

**Status:** Not implemented
**Effort:** ~1 week
**Impact:** High for read-heavy workloads

**Implementation:**
- Pool of pre-created read transactions
- Refresh policy based on age and write count
- Configurable pool size

See `connection-pooling.md` for detailed design.

---

#### 2.2 Storage-Level Metrics Engine

**Status:** Not implemented
**Effort:** ~1 week
**Impact:** Medium (observability)

**Create `MetricsEngine<E>` wrapper:**
```rust
pub struct MetricsEngine<E: StorageEngine> {
    inner: E,
    metrics: StorageMetrics,
}

pub struct StorageMetrics {
    reads: AtomicU64,
    writes: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
    transaction_duration_histogram: Histogram,
}
```

**Benefits:**
- Detailed storage-level insights
- Performance regression detection
- Capacity planning data

---

#### 2.3 Compact Trait Method

**Status:** Not implemented
**Effort:** ~3 days
**Impact:** Medium (storage efficiency)

**Add to StorageEngine:**
```rust
fn compact(&self) -> Result<CompactionStats, StorageError> {
    Ok(CompactionStats::default()) // Default no-op
}
```

**Implementations:**
- RedbEngine: Trigger redb compaction
- Future RocksDB: Trigger LSM compaction
- Future Fjall: Trigger blob garbage collection

---

#### 2.4 Improved Error Context

**Status:** Basic
**Effort:** ~4 days
**Impact:** Medium (debugging experience)

**Current:**
```rust
StorageError::Open(String)
```

**Proposed:**
```rust
#[derive(Debug, thiserror::Error)]
#[error("failed to open database at {path}: {cause}")]
pub struct OpenError {
    pub path: PathBuf,
    pub cause: Box<dyn std::error::Error + Send + Sync>,
    pub suggestion: Option<String>,
}
```

**Benefits:**
- Actionable error messages
- Better troubleshooting
- Structured error data

---

### Tier 3: Medium-Term Projects (1-2 months each)

Significant undertakings with transformative potential.

#### 3.1 Alternative Storage Backend: Fjall

**Status:** Not implemented
**Effort:** ~3 weeks
**Impact:** High (compression, flexibility)

**Why Fjall:**
- Pure Rust (no FFI)
- Built-in compression (LZ4, Zlib)
- Lower disk usage
- Column families for table separation

**Implementation Plan:**
1. Add `fjall` feature flag
2. Implement `StorageEngine` for Fjall
3. Map ManifoldDB tables to Fjall partitions
4. Add configuration options
5. Benchmark against Redb

See `storage-backends.md` for detailed comparison.

---

#### 3.2 Label Index Implementation

**Status:** Not implemented (identified in ARCHITECTURE_INVESTIGATION_REPORT.md)
**Effort:** ~1 week
**Impact:** Very High (eliminates full table scans)

**Current Problem:**
```sql
SELECT * FROM users WHERE email = 'alice@example.com'
-- Currently scans ALL entities
```

**Solution:**
```rust
// Label index: (Label, EntityId) -> ()
const LABEL_INDEX: TableDefinition<(&str, u64), ()> = ...;

// When querying by label, use index scan instead of full scan
```

**This is flagged as "Critical for Production" in the architecture report.**

---

#### 3.3 Secondary Property Indexes

**Status:** Not implemented (identified in ARCHITECTURE_INVESTIGATION_REPORT.md)
**Effort:** ~3-5 weeks
**Impact:** Very High (query performance)

**Implement:**
1. `CREATE INDEX` DDL support
2. `(table, column, value, EntityId) -> ()` index structure
3. Index maintenance on insert/update/delete
4. Query planner integration

**This is flagged as "High Priority" in the architecture report.**

---

#### 3.4 Alternative Storage Backend: RocksDB

**Status:** Not implemented
**Effort:** ~4 weeks
**Impact:** High (production deployments)

**Why RocksDB:**
- Battle-tested at massive scale
- Excellent compression
- Tunable for workloads
- Column families

**Implementation Plan:**
1. Add `rocksdb` feature flag
2. Implement `StorageEngine` for RocksDB
3. Map tables to column families
4. Add tuning configuration
5. Document trade-offs

---

### Tier 4: Long-Term Vision (3-6 months)

Major architectural initiatives.

#### 4.1 Server Mode Architecture

**Status:** Not designed
**Effort:** ~3 months
**Impact:** Transformative (multi-process support)

**Components:**
1. Protocol design (gRPC recommended)
2. Connection manager
3. Query dispatcher
4. Transaction coordinator
5. Client library

**Benefits:**
- Multi-process access
- True connection pooling
- Language-agnostic clients
- Horizontal read scaling (replicas)

---

#### 4.2 Replication via WAL Streaming

**Status:** WAL exists, streaming not implemented
**Effort:** ~2 months
**Impact:** High (reliability, read scaling)

**Build on existing WAL:**
1. WAL segment management
2. WAL streaming protocol
3. Replica consumption
4. Catch-up mechanism
5. Failover procedures

---

#### 4.3 Distributed Query Execution

**Status:** Not designed
**Effort:** ~4-6 months
**Impact:** High (scalability)

**For when single-node is insufficient:**
1. Data partitioning strategy
2. Query routing
3. Distributed transactions
4. Consensus (Raft?)

---

## Recommended Priority Order

Based on impact-to-effort ratio:

### Phase 1: Foundation (Weeks 1-4)

1. **Make Database Cloneable** (Tier 1.1)
2. **Document Existing Features** (Tier 1.3)
3. **Label Index Implementation** (Tier 3.2) - Critical for production
4. **Storage-Level Metrics** (Tier 2.2)

### Phase 2: Performance (Weeks 5-10)

5. **Read Transaction Pool** (Tier 2.1)
6. **Secondary Property Indexes** (Tier 3.3)
7. **Alternative Backend: Fjall** (Tier 3.1)

### Phase 3: Production Readiness (Weeks 11-16)

8. **Improved Error Context** (Tier 2.4)
9. **Alternative Backend: RocksDB** (Tier 3.4)
10. **Compact Trait Method** (Tier 2.3)

### Phase 4: Scale (Beyond Week 16)

11. **Server Mode Design**
12. **Replication**
13. **Distributed Execution** (if needed)

---

## Risk Assessment

| Improvement | Risk Level | Mitigation |
|-------------|------------|------------|
| Database Cloneable | Low | API addition, no breaking changes |
| Label Index | Low | New code path, existing queries unchanged |
| Read Pool | Medium | Careful lifetime management required |
| Fjall Backend | Medium | Feature-flagged, optional |
| Secondary Indexes | Medium | Thorough testing, incremental rollout |
| RocksDB Backend | Medium | FFI complexity, well-documented library |
| Server Mode | High | Major architecture change, phased implementation |

---

## Success Metrics

Track progress with these metrics:

1. **Query Performance**
   - P50/P99 latency for common queries
   - Full scan frequency
   - Index utilization rate

2. **Write Performance**
   - Writes per second
   - Batch coalescing ratio
   - WAL throughput

3. **Resource Usage**
   - Database size on disk
   - Memory usage
   - Transaction handle count

4. **Reliability**
   - Recovery time from crash
   - WAL replay success rate
   - Error rate by type

---

## Conclusion

ManifoldDB has a solid architectural foundation. The most impactful improvements are:

1. **Label Index** - Eliminates full table scans (critical)
2. **Database Cloneable** - Enables modern usage patterns (easy win)
3. **Secondary Indexes** - Query performance (high impact)
4. **Alternative Backend** - Flexibility and compression (medium effort)

The long-term vision of server mode would enable true connection pooling and multi-process access, but the embedded model with the above improvements will serve most use cases well.

---

*This roadmap synthesizes findings from the architecture analysis, storage backend comparison, and connection pooling strategy documents.*
