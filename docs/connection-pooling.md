# Connection Pooling Strategies for ManifoldDB

**Analysis Date:** December 2024
**Author:** Garak Fork (bc4ef5b7-b364-49f9-8bb7-0bcc538399d2)
**Updated:** December 2024 — Implementation status added

---

## Implementation Status

| Strategy | Status | Notes |
|----------|--------|-------|
| Database Cloneable | ✅ Implemented | `Database` now uses `Arc<DatabaseInner>` for cheap cloning |
| Read Transaction Pool | ✅ Implemented | `ReadPool` in `manifoldb::transaction` module |
| Batch Transaction Coalescing | ✅ Already existed | `BatchWriter` for write batching |
| Transaction Handle Pool | 📋 Future | Unified pool for read/write |
| Server Mode | 📋 Future | Full client-server architecture |

---

## Context: What "Connection Pooling" Means for Embedded Databases

Traditional connection pooling manages a pool of network connections between clients and a database server. For embedded databases like ManifoldDB, this concept must be reframed because:

1. **No Network**: The database runs in-process; there are no TCP connections to pool
2. **No Server Sessions**: There's no server maintaining session state
3. **Direct Memory Access**: Transactions operate on memory-mapped data directly

However, there are analogous patterns that provide similar benefits:

| Traditional Pool | Embedded Equivalent |
|------------------|---------------------|
| Network connection | Transaction handle |
| Server session | Transaction context |
| Connection limit | Concurrent reader limit |
| Health checking | Transaction refresh |

---

## Current ManifoldDB Transaction Model

ManifoldDB's transaction handling is already sophisticated:

### Transaction Creation Flow

```rust
// Read transaction
let tx = db.begin_read()?;
// Uses: TransactionManager -> StorageEngine.begin_read() -> RedbTransaction

// Write transaction
let mut tx = db.begin()?;
// Uses: TransactionManager -> StorageEngine.begin_write() -> RedbTransaction
```

### What Happens Under the Hood

1. **Transaction ID Assignment**: `next_tx_id.fetch_add(1, Ordering::Relaxed)`
2. **Storage Transaction**: Redb creates a new transaction with snapshot isolation
3. **Wrapper Creation**: `DatabaseTransaction` wraps storage transaction with ManifoldDB operations

### Current Costs

Each `begin_read()` call incurs:
- Atomic counter increment (cheap)
- Redb snapshot creation (moderate)
- Memory allocation for transaction state (moderate)

For high-frequency operations, these costs accumulate.

---

## Strategy 1: Read Transaction Pool

For read-heavy workloads, pool pre-created read transactions.

### Concept

```rust
pub struct ReadPool<E: StorageEngine> {
    engine: Arc<E>,
    available: Mutex<Vec<PooledReadTx<E>>>,
    max_size: usize,
    refresh_interval: Duration,
}

pub struct PooledReadTx<E: StorageEngine> {
    inner: E::Transaction<'static>,  // 'static via lifetime tricks
    created_at: Instant,
}
```

### Usage Pattern

```rust
// Get transaction from pool (fast)
let tx = pool.acquire()?;

// Use normally
let entity = tx.get_entity(id)?;

// Return to pool (doesn't drop)
pool.release(tx);
```

### Implementation Considerations

1. **Lifetime Management**: Read transactions hold a snapshot; too-old snapshots see stale data
2. **Refresh Policy**: After writes occur, pooled transactions should be refreshed
3. **Size Limits**: Don't hold too many read snapshots (memory overhead)

### Recommended Configuration

```rust
pub struct ReadPoolConfig {
    /// Maximum pooled transactions
    pub max_size: usize,  // Default: 16

    /// Refresh after this duration
    pub max_age: Duration,  // Default: 100ms

    /// Refresh after N writes to database
    pub refresh_after_writes: u64,  // Default: 100
}
```

### Benefits

- **Reduced Allocation**: Reuse transaction structures
- **Faster Reads**: Avoid snapshot creation overhead
- **Bounded Memory**: Configurable pool size

### Drawbacks

- **Stale Reads Possible**: If refresh isn't triggered
- **Complexity**: More code to maintain
- **Memory Holding**: Pooled transactions hold resources

---

## Strategy 2: Database Handle Cloning

ManifoldDB's `Database` already supports sharing via cloning internal `Arc`s.

### Current Architecture

```rust
pub struct Database {
    manager: TransactionManager<RedbEngine>,  // Contains Arc<RedbEngine>
    config: Config,
    query_cache: QueryCache,
    prepared_cache: PreparedStatementCache,
    db_metrics: Arc<DatabaseMetrics>,
}
```

### Multi-Handle Pattern

```rust
// Create database
let db = Database::open("data.manifold")?;

// Clone for another thread (shares underlying engine)
let db_clone = db.clone();

std::thread::spawn(move || {
    let tx = db_clone.begin_read()?;
    // ...
});
```

### What's Currently Shareable

- `TransactionManager` contains `Arc<RedbEngine>` - ✓ Shared
- `QueryCache` - Would need to be `Arc<QueryCache>`
- `PreparedStatementCache` - Would need to be `Arc<PreparedStatementCache>`
- `DatabaseMetrics` - Already `Arc<DatabaseMetrics>` - ✓ Shared

### Recommended Enhancement

Make `Database` cheaply cloneable:

```rust
pub struct Database {
    inner: Arc<DatabaseInner>,
}

struct DatabaseInner {
    manager: TransactionManager<RedbEngine>,
    config: Config,
    query_cache: QueryCache,
    prepared_cache: PreparedStatementCache,
    db_metrics: DatabaseMetrics,
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}
```

This allows multiple `Database` handles across threads without resource duplication.

---

## Strategy 3: Batch Transaction Coalescing (Already Implemented!)

ManifoldDB already has sophisticated write batching via `BatchWriter`.

### Current Batch Writer Flow

```
Thread 1: batch_tx.put(...) ──┐
Thread 2: batch_tx.put(...) ──┼──▶ WriteQueue ──▶ Single Storage TX ──▶ Commit
Thread 3: batch_tx.put(...) ──┘
```

### Configuration

```rust
pub struct BatchWriterConfig {
    pub max_batch_size: usize,      // Default: 100
    pub flush_interval: Duration,    // Default: 10ms
    pub enabled: bool,               // Default: true
}
```

### Why This Is Already Connection Pooling

The batch writer:
1. Accepts writes from multiple "connections" (transactions)
2. Coalesces them into fewer storage operations
3. Amortizes commit cost across multiple logical transactions

This is essentially a write-side connection pool.

---

## Strategy 4: Transaction Handle Pool

A more general pool for both read and write transaction handles.

### Design

```rust
pub struct TransactionPool<E: StorageEngine> {
    engine: Arc<E>,
    read_handles: ArrayQueue<PooledHandle>,
    write_semaphore: Semaphore,  // Limit concurrent writes
    config: PoolConfig,
}

pub struct PoolConfig {
    pub max_read_handles: usize,
    pub max_write_handles: usize,  // Usually 1 for SWMR
    pub wait_timeout: Duration,
}
```

### Usage

```rust
// Blocking acquire with timeout
let tx = pool.read()?;
// ... use transaction ...
drop(tx);  // Returns to pool

// Write with semaphore
let mut tx = pool.write()?;
// ... exclusive write access ...
tx.commit()?;  // Releases semaphore
```

---

## Strategy 5: Server Mode (Future Direction)

For true connection pooling semantics, ManifoldDB could offer a server mode.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ManifoldDB Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Connection  │  │ Query       │  │ Transaction         │  │
│  │ Manager     │  │ Dispatcher  │  │ Coordinator         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                           │                                  │
│                    ┌──────▼──────┐                           │
│                    │   Database  │                           │
│                    └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
        ▲                   ▲                   ▲
        │                   │                   │
   ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
   │ Client  │         │ Client  │         │ Client  │
   │   1     │         │   2     │         │   3     │
   └─────────┘         └─────────┘         └─────────┘
```

### Server Mode Benefits

- **True Connection Pooling**: Pool of client connections
- **Query Queuing**: Manage concurrent query load
- **Session Management**: Per-client session state
- **Multi-Process**: Multiple processes access same database
- **Protocol**: gRPC, HTTP, or custom binary protocol

### Server Mode Complexity

- Requires network protocol design
- Need to handle connection lifecycle
- Adds deployment complexity
- Performance overhead from serialization

---

## Recommendations

### Immediate (Low Effort) — ✅ DONE

1. **Document Existing Batch Writer**: The batch writer is already a form of connection pooling for writes. ✅ Documented.

2. **Make Database Cloneable**: ✅ IMPLEMENTED in v0.1.2

```rust
// Database is now Clone via Arc<DatabaseInner>
let db = Database::open("mydb.manifold")?;
let db2 = db.clone();  // Cheap clone - shares underlying engine

std::thread::spawn(move || {
    let tx = db2.begin_read()?;
    // ...
});
```

### Short-Term (Medium Effort) — ✅ DONE

3. **Read Transaction Pool**: ✅ IMPLEMENTED in v0.1.2

```rust
use manifoldb::transaction::{ReadPool, ReadPoolConfig};

let pool = ReadPool::new(engine, ReadPoolConfig::default())?;
let tx = pool.acquire()?;
// ... use transaction ...
pool.notify_write();  // Call after writes to track staleness
```

4. **Transaction Metrics**: ✅ IMPLEMENTED via `DatabaseMetrics`

### Medium-Term (High Effort)

5. **Transaction Handle Pool**: Unified pool for read/write with semaphore-based limits — 📋 Future

### Long-Term (Very High Effort)

6. **Server Mode**: Full client-server architecture with true connection pooling — 📋 Future

---

## Example Implementation: Simple Read Pool

```rust
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct ReadPool<E: StorageEngine> {
    engine: Arc<E>,
    pool: Mutex<Vec<PooledTx<E>>>,
    config: ReadPoolConfig,
    write_counter: AtomicU64,
}

struct PooledTx<E: StorageEngine> {
    tx: E::Transaction<'static>,
    created_at: Instant,
    created_at_write: u64,
}

impl<E: StorageEngine> ReadPool<E> {
    pub fn acquire(&self) -> Result<PooledTx<E>, Error> {
        let mut pool = self.pool.lock()?;

        // Try to get from pool
        if let Some(mut tx) = pool.pop() {
            // Check if still valid
            let age = tx.created_at.elapsed();
            let writes_since = self.write_counter.load(Ordering::Relaxed) - tx.created_at_write;

            if age < self.config.max_age && writes_since < self.config.refresh_threshold {
                return Ok(tx);
            }
            // Transaction too old, drop it and create fresh
        }

        // Create new transaction
        let tx = self.engine.begin_read()?;
        Ok(PooledTx {
            tx,
            created_at: Instant::now(),
            created_at_write: self.write_counter.load(Ordering::Relaxed),
        })
    }

    pub fn release(&self, tx: PooledTx<E>) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.config.max_size {
            pool.push(tx);
        }
        // Otherwise drop the transaction
    }

    pub fn notify_write(&self) {
        self.write_counter.fetch_add(1, Ordering::Relaxed);
    }
}
```

---

## Conclusion

For ManifoldDB's embedded architecture, "connection pooling" translates to:

| Strategy | Status |
|----------|--------|
| **Transaction handle reuse** | ✅ `ReadPool` implemented |
| **Write batching** | ✅ `BatchWriter` implemented |
| **Database sharing** | ✅ `Database` is `Clone` via `Arc` |
| **Server mode** | 📋 Future direction |

As of v0.1.2, the high-impact pooling strategies are implemented:
- `Database::clone()` for sharing across threads
- `ReadPool` for high-frequency read workloads
- `BatchWriter` for coalescing writes

---

*This document analyzes connection pooling strategies appropriate for ManifoldDB's embedded database architecture.*
