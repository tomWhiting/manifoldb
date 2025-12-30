# Storage Backend Options for ManifoldDB

**Analysis Date:** December 2024
**Author:** Garak Fork (bc4ef5b7-b364-49f9-8bb7-0bcc538399d2)

---

## Current Backend: Redb

ManifoldDB currently uses [redb](https://github.com/cberner/redb) as its storage engine. This is an excellent choice for several reasons:

### Redb Characteristics

| Property | Value |
|----------|-------|
| Language | Pure Rust |
| Data Structure | Copy-on-write B+ tree |
| Concurrency | Single Writer, Multiple Readers (SWMR) |
| Transactions | ACID with snapshot isolation |
| Memory Mapping | Yes |
| Dependencies | Zero non-Rust dependencies |
| Maintenance | Active, 1.0 stable release (June 2023) |

### Redb Strengths

1. **Pure Rust**: No C dependencies means easier cross-compilation, no FFI overhead, and memory safety guarantees throughout the stack.

2. **Solid Performance**: Redb's benchmarks show competitive performance against RocksDB and LMDB for common operations, with particular strength in single-write scenarios.

3. **Simple API**: The API is straightforward and aligns well with ManifoldDB's `StorageEngine` trait design.

4. **In-Memory Backend**: Redb includes an `InMemoryBackend` for testing, which ManifoldDB already uses.

### Redb Limitations

1. **Single-Process Only**: File locking prevents multi-process access to the same database.

2. **Newer Project**: While stable, redb has less production mileage than RocksDB or LMDB.

3. **No Compression**: Unlike RocksDB, redb does not include built-in data compression.

---

## Alternative Backend Options

### 1. RocksDB

[RocksDB](https://rocksdb.org/) is Facebook's fork of LevelDB, battle-tested at massive scale.

**Rust Bindings:** [rust-rocksdb](https://github.com/rust-rocksdb/rust-rocksdb)

| Property | Value |
|----------|-------|
| Language | C++ with Rust bindings |
| Data Structure | LSM-tree (Log-Structured Merge) |
| Concurrency | Multi-writer with OCC |
| Compression | Yes (LZ4, Snappy, Zstd, etc.) |
| Column Families | Yes (multiple keyspaces) |
| Maturity | Extremely mature, used in production at Meta, TiKV, CockroachDB |

**Strengths:**
- Best-in-class storage efficiency with compression
- Tunable for various workloads (write-heavy, read-heavy, mixed)
- Extensive configuration options
- Column families for logical separation with shared commit

**Considerations:**
- C++ dependency increases build complexity
- Larger binary size due to FFI
- More complex tuning required for optimal performance

**ManifoldDB Fit:** Excellent for deployments where storage efficiency is critical or for very large datasets. The column family feature could map well to ManifoldDB's table concept.

### 2. LMDB (Lightning Memory-Mapped Database)

[LMDB](http://www.lmdb.tech/doc/) is the gold standard for read-heavy, multi-process embedded databases.

**Rust Bindings:** [lmdb-rs](https://github.com/danburkert/lmdb-rs) or [heed](https://github.com/meilisearch/heed)

| Property | Value |
|----------|-------|
| Language | C with Rust bindings |
| Data Structure | B+ tree with memory mapping |
| Concurrency | Single Writer, Multiple Readers (SWMR) |
| Multi-Process | Yes (via file locking) |
| Compression | No |
| Maturity | Very mature, used in OpenLDAP, Meilisearch |

**Strengths:**
- Exceptional read performance (often fastest in benchmarks)
- Multi-process access support
- Zero-copy reads via memory mapping
- Minimal memory footprint

**Considerations:**
- Write amplification with copy-on-write
- Database size must be configured upfront
- No built-in compression
- C dependency

**ManifoldDB Fit:** Ideal for read-heavy workloads or when multi-process access is required. The `heed` crate provides a more ergonomic Rust API.

### 3. Sled

[Sled](https://github.com/spacejam/sled) is a modern, lock-free embedded database written in pure Rust.

| Property | Value |
|----------|-------|
| Language | Pure Rust |
| Data Structure | Log-structured with B+ tree indexes |
| Concurrency | Lock-free, highly concurrent |
| Status | Pre-1.0 (beta) |
| Compression | No |

**Strengths:**
- Lock-free architecture for excellent concurrent performance
- Pure Rust with async support
- Designed for modern hardware (SSD-optimized)
- Crash-safe with log-structured storage

**Considerations:**
- Still in beta (on-disk format may change)
- Higher space amplification than RocksDB
- Active rewrite in progress (komora project)

**ManifoldDB Fit:** Interesting for highly-concurrent workloads, but the pre-1.0 status makes it risky for production. Worth monitoring as it matures.

### 4. Fjall

[Fjall](https://github.com/fjall-rs/fjall) is a newer pure-Rust LSM-tree database with modern features.

| Property | Value |
|----------|-------|
| Language | Pure Rust (forbid-unsafe) |
| Data Structure | LSM-tree |
| Concurrency | SWMR or Multi-writer with OCC |
| Compression | Yes (LZ4, Zlib) |
| Status | 2.x stable |
| Column Families | Yes (partitions) |

**Strengths:**
- Pure Rust with excellent safety properties (`forbid(unsafe)`)
- Built-in compression with configurable algorithms
- Lowest disk space usage among Rust options
- Transactional with cross-partition atomicity
- Active development with regular releases (2025)

**Considerations:**
- Newer than RocksDB/LMDB
- Single-process only
- Smaller community

**ManifoldDB Fit:** Strong candidate as a pure-Rust alternative to RocksDB. The compression support and column families (partitions) are attractive features.

### 5. SQLite (via rusqlite)

[SQLite](https://sqlite.org/) is the world's most deployed database.

**Rust Bindings:** [rusqlite](https://github.com/rusqlite/rusqlite)

| Property | Value |
|----------|-------|
| Language | C with Rust bindings |
| Data Structure | B-tree |
| Concurrency | WAL mode for concurrent reads |
| SQL | Full SQL support |
| Maturity | Extremely mature, 20+ years of development |

**Strengths:**
- Ubiquitous, extremely well-tested
- Excellent tooling (CLI, GUIs, etc.)
- Full SQL support could simplify query layer
- WAL mode enables good concurrency

**Considerations:**
- Key-value interface would be an abstraction over SQL
- More overhead than pure KV stores
- C dependency

**ManifoldDB Fit:** Unusual choice for a graph database backend, but the tooling and maturity are unmatched. Could be interesting for debugging and development scenarios.

### 6. Custom In-Memory Engine

For testing and specific use cases, a custom in-memory engine based on concurrent data structures could be valuable.

**Potential Implementation:**

```rust
pub struct ConcurrentMemoryEngine {
    tables: DashMap<String, DashMap<Vec<u8>, Vec<u8>>>,
}
```

Using `dashmap` or similar lock-free concurrent hashmaps would provide:
- Maximum read concurrency
- No persistence overhead
- Ideal for testing and caching scenarios

---

## Comparison Matrix

| Backend | Pure Rust | Compression | Multi-Process | Read Perf | Write Perf | Stability |
|---------|-----------|-------------|---------------|-----------|------------|-----------|
| Redb | ✓ | ✗ | ✗ | Good | Good | Stable |
| RocksDB | ✗ | ✓ | ✗ | Good | Excellent | Very Stable |
| LMDB | ✗ | ✗ | ✓ | Excellent | Good | Very Stable |
| Sled | ✓ | ✗ | ✗ | Good | Good | Beta |
| Fjall | ✓ | ✓ | ✗ | Good | Good | Stable |
| SQLite | ✗ | Via ext | ✓ | Good | Moderate | Very Stable |

---

## Implementation Strategy

### Phase 1: Backend Trait Refinement

The existing `StorageEngine` trait is well-designed. Minor additions might include:

```rust
pub trait StorageEngine: Send + Sync {
    // Existing methods...

    /// Backend name for diagnostics
    fn backend_name(&self) -> &'static str;

    /// Estimated storage size in bytes
    fn estimated_size(&self) -> Result<u64, StorageError>;

    /// Compact the storage (if supported)
    fn compact(&self) -> Result<(), StorageError> {
        Ok(()) // Default no-op
    }
}
```

### Phase 2: Feature Flags

Use Cargo features to enable optional backends:

```toml
[features]
default = ["redb"]
redb = ["dep:redb"]
rocksdb = ["dep:rocksdb"]
lmdb = ["dep:heed"]
fjall = ["dep:fjall"]
```

### Phase 3: Backend Selection at Runtime

Support runtime selection for development/testing:

```rust
pub enum BackendConfig {
    Redb(RedbConfig),
    RocksDb(RocksDbConfig),
    Lmdb(LmdbConfig),
    Memory(MemoryConfig),
}

let db = DatabaseBuilder::new()
    .backend(BackendConfig::Redb(RedbConfig::default()))
    .open()?;
```

---

## Recommendations

### Primary Recommendation: Add Fjall as Alternative

Fjall is the strongest candidate for a second backend:

1. **Pure Rust**: Maintains ManifoldDB's "no C deps" option
2. **Compression**: Addresses redb's lack of compression
3. **Active Development**: Regular releases with performance improvements
4. **Safety**: The `forbid(unsafe)` approach aligns with Rust best practices

### Secondary Recommendation: Add RocksDB for Production

For deployments where storage efficiency is critical:

1. **Battle-tested**: Decades of production experience
2. **Compression**: Excellent compression ratios
3. **Tunable**: Can be optimized for specific workloads

### Long-term: Server Mode with LMDB

If multi-process access becomes a requirement, consider:

1. Server mode architecture (see architecture-analysis.md)
2. LMDB backend for excellent read performance
3. Connection pooling for multiple clients

---

## References

- [redb](https://github.com/cberner/redb) - Current ManifoldDB backend
- [RocksDB](https://rocksdb.org/) - Meta's LSM-tree database
- [LMDB](http://www.lmdb.tech/doc/) - Lightning Memory-Mapped Database
- [Sled](https://github.com/spacejam/sled) - Pure Rust embedded database
- [Fjall](https://github.com/fjall-rs/fjall) - Pure Rust LSM-tree database
- [rusqlite](https://github.com/rusqlite/rusqlite) - SQLite bindings for Rust

---

*This analysis compares storage backend options based on publicly available documentation, benchmarks, and architectural characteristics.*
