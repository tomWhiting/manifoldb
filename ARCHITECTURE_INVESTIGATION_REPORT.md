# ManifoldDB Architecture Investigation Report

**Date:** December 2024
**Focus:** High Performance and Production Readiness
**Scope:** Edge indexing, Secondary indexes, Property storage, Write concurrency

---

## Executive Summary

ManifoldDB has a solid foundation for graph operations with proper edge indexing. However, there are significant gaps in property indexing that will impact production SQL query performance at scale. The storage model and concurrency approach are reasonable for current use cases but have known limitations.

**Key Findings:**
- **Edge Indexing**: Well-designed with proper indexes for traversal operations
- **Secondary Property Indexes**: **Not implemented** - all queries do full table scans
- **Property Storage**: Row-oriented (all properties stored with entity) - acceptable
- **Write Concurrency**: Single-writer, multiple-reader (SWMR) via redb - reasonable

---

## 1. Edge Indexing Strategy

### Current Implementation

**Status: Well-Implemented**

Edge storage uses three index tables for efficient graph traversal:

| Table | Key Structure | Purpose |
|-------|---------------|---------|
| `edges` | `EdgeId` | Primary edge data storage |
| `edges_by_source` | `(EntityId, EdgeType, EdgeId)` | Outgoing edge lookups |
| `edges_by_target` | `(EntityId, EdgeType, EdgeId)` | Incoming edge lookups |
| `edge_types` | `(EdgeType, EdgeId)` | Find edges by type |

**Location:** `crates/manifoldb-graph/src/store/edge.rs`, `crates/manifoldb-graph/src/index/`

### Supported Operations (O(1) or O(k) where k = result size)

- Get all outgoing edges from entity X
- Get all incoming edges to entity X
- Get outgoing edges of type Y from entity X
- Get all edges of type Y
- Bidirectional traversal

### Assessment

The edge indexing is production-ready. The composite key design `(EntityId, EdgeType, EdgeId)` enables efficient prefix scans for all common graph query patterns. Index maintenance is properly synchronized with edge mutations.

**Recommendation:** No changes needed.

---

## 2. Secondary Indexes on Properties

### Current Implementation

**Status: NOT IMPLEMENTED - Critical Gap**

When executing a SQL query like:
```sql
SELECT * FROM users WHERE email = 'alice@example.com'
```

The execution path is:

1. `execute_logical_plan()` matches on `LogicalPlan::Scan`
2. Calls `tx.iter_entities(Some("users"))`
3. **Iterates through ALL entities with a full table scan**
4. Deserializes every entity and filters in memory

**Location:** `crates/manifoldb/src/transaction/handle.rs:499-537`

```rust
// Current implementation - FULL TABLE SCAN
pub fn iter_entities(&self, label: Option<&str>) -> Result<Vec<Entity>, TransactionError> {
    // Creates cursor over ALL nodes
    let cursor_result = storage.range(tables::NODES, Bound::Unbounded, Bound::Unbounded);

    // Iterates through all nodes
    while let Some((_key, value)) = cursor.next()? {
        let entity = bincode::decode(&value)?;
        // Filter by label in memory
        if let Some(label_filter) = label {
            if entity.has_label(label_filter) {
                entities.push(entity);
            }
        }
    }
}
```

### Query Planner Gap

The query planner has an `IndexSelector` (`crates/manifoldb-query/src/plan/optimize/index_selection.rs`) that *identifies* which predicates could benefit from indexes:

- Point lookups (`column = value`)
- Range scans (`column > value`, `BETWEEN`)
- IN lists (`column IN (...)`)
- Prefix scans (`LIKE 'abc%'`)

But this analysis is **never used** - there are no actual property indexes to utilize.

### Impact

| Scenario | Current Performance | With Indexes |
|----------|---------------------|--------------|
| 1M entities, find 1 by email | Scan 1M entities | 1 index lookup |
| WHERE age > 21 | Scan all, filter in memory | Range scan on age index |
| COUNT(*) WHERE status = 'active' | Full scan + count | Index count |

### Recommendation: Implement Secondary Property Indexes

**Priority: HIGH for production readiness**

Options (in order of implementation complexity):

1. **Label Index** (minimal, high impact)
   - Maintain `(Label, EntityId) -> ()` index
   - Eliminates full scans for label filtering
   - ~1 day of work

2. **User-Defined Secondary Indexes** (standard approach)
   - `CREATE INDEX idx_email ON users(email)`
   - Maintain `(table, column, value, EntityId) -> ()`
   - Requires DDL support and index maintenance on write
   - ~3-5 days of work

3. **Automatic Indexing** (advanced)
   - Like Fauna - auto-index all queryable fields
   - Higher write overhead, simpler user experience
   - ~1-2 weeks of work

---

## 3. Property Storage Format

### Current Implementation

**Status: Acceptable (row-oriented)**

Entities are stored as complete units using bincode serialization:

```rust
pub struct Entity {
    pub id: EntityId,
    pub labels: Vec<Label>,
    pub properties: HashMap<String, Value>,  // All properties together
}
```

When reading an entity, all properties are loaded and deserialized together.

**Location:** `crates/manifoldb-core/src/types/entity.rs`

### Trade-offs

| Aspect | Row-Oriented (Current) | Column-Oriented |
|--------|------------------------|-----------------|
| Read single property | Load all, extract one | Load just that column |
| Read all properties | Single read | Multiple column reads |
| Write one property | Rewrite entire entity | Write to one column |
| Implementation | Simple | Complex |
| Compression | Limited | Better (similar values) |

### Assessment

Row-oriented storage is appropriate for ManifoldDB's current use case (graph database with SQL interface). The overhead of loading all properties is minimal for typical entity sizes (< 1KB). Column-oriented storage would add significant complexity without proportional benefit.

**Recommendation:** Keep current approach. Consider revisiting only if:
- Average entity size exceeds 10KB
- Queries frequently need single properties from large entities
- Analytical workloads become primary use case

---

## 4. Write Concurrency Model

### Current Implementation

**Status: Reasonable (SWMR via redb)**

ManifoldDB uses [redb](https://github.com/cberner/redb) as its storage engine, which provides:

- **Single Writer, Multiple Readers (SWMR)**
- **MVCC** for snapshot isolation
- Readers never block writers
- Writers block on each other
- Serializable isolation level

**Location:** `crates/manifoldb-storage/src/backends/redb/engine.rs`

### Implications

| Scenario | Behavior |
|----------|----------|
| Concurrent reads | No blocking, each sees consistent snapshot |
| Concurrent writes | Serialized - one at a time |
| Read during write | Read sees pre-write state |
| Write throughput | Limited by single-writer bottleneck |

### Batch Writer Optimization

The `BatchWriter` (`crates/manifoldb/src/transaction/batch_writer.rs`) provides group commit for concurrent transactions, reducing fsync overhead. This is already implemented and working correctly.

### Assessment

SWMR is a reasonable trade-off for ManifoldDB's target use case:
- Graph databases typically have read-heavy workloads
- Simpler than MVCC with multiple writers
- No deadlock concerns
- Proven approach (SQLite, LMDB use similar models)

### When This Becomes a Problem

- High-concurrency write workloads (> 1000 writes/sec from multiple threads)
- Real-time systems with strict write latency requirements
- Large-scale ETL with parallel loading

### Recommendations

**Short-term:** Keep current model. It's simple and correct.

**Medium-term considerations:**
- Monitor write latency under concurrent load
- Consider `parking_lot` for faster uncontended locks
- Ensure batch writer is properly utilized for bulk operations

**Long-term (if needed):**
- MVCC with optimistic concurrency control
- Partition writes by entity type/label
- Shard by entity ID range

---

## Summary of Recommendations

### Critical (Before Production)

1. **Implement Label Index**
   - Eliminate full table scans for basic queries
   - `(Label, EntityId) -> ()` index structure
   - Estimated effort: 1 day

### High Priority

2. **Implement User-Defined Secondary Indexes**
   - `CREATE INDEX` DDL support
   - Property value to EntityId mapping
   - Index maintenance on insert/update/delete
   - Estimated effort: 3-5 days

### Nice to Have

3. **Query Plan Caching**
   - Already have `PreparedStatement` infrastructure
   - Cache compiled query plans for parameterized queries

4. **Index Usage in Query Executor**
   - Wire `IndexSelector` analysis to actual index lookups
   - Replace full scans with index scans when available

### Not Needed Currently

- Column-oriented property storage
- Multi-writer concurrency (MVCC)
- Partitioned/sharded storage

---

## Files Referenced

| File | Purpose |
|------|---------|
| `crates/manifoldb-graph/src/store/edge.rs` | Edge storage with indexes |
| `crates/manifoldb-graph/src/index/maintenance.rs` | Edge index maintenance |
| `crates/manifoldb/src/transaction/handle.rs` | Entity iteration (full scan) |
| `crates/manifoldb-query/src/plan/optimize/index_selection.rs` | Index candidate analysis |
| `crates/manifoldb-query/src/exec/operators/scan.rs` | Scan operators (stub) |
| `crates/manifoldb/src/execution/executor.rs` | Query execution |
| `crates/manifoldb-core/src/types/entity.rs` | Entity/property storage format |
| `crates/manifoldb-storage/src/backends/redb/engine.rs` | Storage engine concurrency |

---

*Report generated by architecture investigation of ManifoldDB codebase.*
