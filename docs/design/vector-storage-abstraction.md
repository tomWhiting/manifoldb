# Vector Storage Abstraction Design

## Overview

This document describes a new storage architecture where vectors are stored separately from entities, enabling:

- **Storage efficiency**: Read entities without loading vector data (~6KB per 1536-dim embedding)
- **Multiple embeddings per entity**: Support text, image, summary embeddings per entity
- **Independent compression**: Enable quantization without affecting entity serialization
- **Flexible schema evolution**: Add/remove vector types without entity migrations

## Current State Analysis

### What Exists Today

#### VectorStore (Entity-Centric)
Located in `crates/manifoldb-vector/src/store/vector_store.rs`:
- Stores embeddings keyed by `(EmbeddingName, EntityId)`
- Already separates vectors from entities in storage
- Tables: `vector_spaces` (metadata), `vector_embeddings` (data)
- Key format: `[prefix][space_name_hash][entity_id]`

#### PointStore (Qdrant-Style)
Located in `crates/manifoldb-vector/src/store/point_store.rs`:
- Independent point system with payloads + multiple named vectors
- Tables: `point_collections`, `point_payloads`, `point_dense_vectors`, `point_sparse_vectors`, `point_multi_vectors`
- Already supports multiple named vectors per point

#### Value Type (Vector as Property)
Located in `crates/manifoldb-core/src/types/value.rs`:
- Entities can have `Value::Vector`, `Value::SparseVector`, `Value::MultiVector` properties
- This is the problematic pattern - vectors embedded in entity serialization

#### HNSW Index
Located in `crates/manifoldb-vector/src/index/hnsw.rs`:
- Indexes `EntityId` -> graph position
- Uses `EmbeddingLookup` trait to fetch vectors during search
- Has in-memory graph with persistence

### The Problem

When vectors are stored as entity properties:
1. Loading an entity deserializes the vector (~6KB for 1536-dim)
2. Only one vector per property name per entity
3. Vector format tied to entity serialization (no independent quantization)
4. HNSW must query entity storage to get vectors for distance calculations

## Design Decision

**Recommendation: Unify on PointStore Pattern**

Rather than maintaining two parallel systems (VectorStore for entities, PointStore for points), we should:

1. Keep the existing `VectorStore` for legacy entity-based embeddings
2. Evolve the Collection system to use dedicated vector storage (similar to PointStore)
3. HNSW indexes reference `(CollectionId, PointId/EntityId, VectorName)` tuples

---

## Data Model Design

### Core Types

```rust
/// A unique identifier for a stored vector.
///
/// VectorId is separate from EntityId because:
/// - One entity can have multiple vectors (text, image, summary)
/// - Vectors can be replaced independently
/// - Enables vector-level versioning in future
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorId(u64);

/// Reference to a vector attached to an entity.
///
/// This is the primary addressing scheme for vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VectorRef {
    /// The entity this vector belongs to.
    pub entity_id: EntityId,
    /// The named vector space (e.g., "text_embedding", "image_embedding").
    pub vector_name: String,
}

/// A stored vector with its metadata.
#[derive(Debug, Clone)]
pub struct StoredVector {
    /// Unique vector identifier (for internal use).
    pub id: VectorId,
    /// Reference back to the entity.
    pub entity_id: EntityId,
    /// The vector name within the collection.
    pub vector_name: String,
    /// The actual vector data.
    pub data: VectorData,
    /// When this vector was last updated.
    pub updated_at: u64,
}

/// Vector data supporting different types.
#[derive(Debug, Clone)]
pub enum VectorData {
    /// Dense floating-point vector.
    Dense(Vec<f32>),
    /// Sparse vector with (index, value) pairs.
    Sparse(Vec<(u32, f32)>),
    /// Multi-vector (ColBERT-style token embeddings).
    Multi(Vec<Vec<f32>>),
    /// Binary vector (bit-packed).
    Binary(Vec<u8>),
}
```

### Why VectorRef Instead of VectorId?

**Option A: VectorId (UUID/snowflake)**
```rust
struct Vector {
    id: VectorId,           // Primary key
    entity_id: EntityId,    // Foreign key
    vector_name: String,
    data: Vec<f32>,
}
```

Pros:
- Stable ID even if entity is recreated
- Can version vectors independently

Cons:
- Extra indirection layer
- Need to maintain entity_id -> vector_id mapping
- HNSW would store VectorId, requiring lookup to get EntityId for results

**Option B: VectorRef (EntityId + name) - RECOMMENDED**
```rust
struct VectorRef {
    entity_id: EntityId,
    vector_name: String,
}
```

Pros:
- Direct mapping from entity to vector
- HNSW can still return EntityId in search results
- Simpler orphan handling (delete entity = delete all its vectors)
- Natural key matches query patterns ("get vectors for entity X")

Cons:
- Replacing a vector requires update, not insert
- Vector identity tied to entity identity

**Decision: Use VectorRef**

The VectorRef approach is simpler and matches our primary access patterns. Users query "find similar entities" not "find similar vectors". The entity is the unit of result, vectors are just attributes.

---

## Storage Format Design

### Table Structure

```
Tables:
├── collection_vectors       # Vector data by (collection, entity, vector_name)
├── vector_metadata         # Vector space metadata (dimension, metric)
└── (existing HNSW tables)  # hnsw_* tables for index data
```

### Key Encoding

```rust
/// Key for vector storage.
/// Format: [collection_id: 8 bytes][entity_id: 8 bytes][vector_name_hash: 8 bytes]
fn encode_vector_key(
    collection_id: CollectionId,
    entity_id: EntityId,
    vector_name: &str,
) -> [u8; 24] {
    let mut key = [0u8; 24];
    key[0..8].copy_from_slice(&collection_id.as_u64().to_be_bytes());
    key[8..16].copy_from_slice(&entity_id.as_u64().to_be_bytes());
    key[16..24].copy_from_slice(&hash_vector_name(vector_name).to_be_bytes());
    key
}

/// Secondary key for listing all vectors for an entity.
/// Format: [collection_id: 8 bytes][entity_id: 8 bytes]
fn encode_entity_prefix(
    collection_id: CollectionId,
    entity_id: EntityId,
) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..8].copy_from_slice(&collection_id.as_u64().to_be_bytes());
    key[8..16].copy_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}
```

### Value Encoding

```rust
/// Vector value format for storage.
///
/// Format:
/// [version: 1 byte]
/// [type: 1 byte] (0=dense, 1=sparse, 2=multi, 3=binary)
/// [updated_at: 8 bytes]
/// [data_len: 4 bytes]
/// [data: variable]
///
/// For dense vectors, data is f32 values in little-endian.
/// For sparse vectors, data is (u32 index, f32 value) pairs.
/// For multi vectors, data is [num_vectors: u32][dim: u32][f32 data...].
/// For binary vectors, data is raw bytes.
```

### Orphan Handling

When an entity is deleted, its vectors become orphans. Options:

1. **Cascade delete (RECOMMENDED)**: Delete vectors when entity is deleted
   - Implemented in entity deletion code
   - Clean, no garbage accumulation

2. **Lazy cleanup**: Mark orphans, clean up in background
   - More complex, useful if entity deletion is rare
   - Requires tombstone tracking

3. **Reference counting**: Track vector references
   - Overkill for current requirements

**Decision: Cascade delete**

The entity deletion path should also delete vectors. This is easy to implement and matches user expectations.

```rust
impl CollectionHandle {
    pub fn delete_entity(&self, entity_id: EntityId) -> Result<bool> {
        // Start transaction
        let mut tx = self.engine.begin_write()?;

        // Delete entity data
        let deleted = self.entity_store.delete_tx(&mut tx, entity_id)?;

        if deleted {
            // Delete all vectors for this entity
            let prefix = encode_entity_prefix(self.collection_id, entity_id);
            delete_by_prefix(&mut tx, TABLE_COLLECTION_VECTORS, &prefix)?;

            // Remove from HNSW indexes
            for vector_config in self.collection.vectors().values() {
                if let Some(index) = self.get_index(&vector_config.name)? {
                    index.delete(entity_id)?;
                }
            }
        }

        tx.commit()?;
        Ok(deleted)
    }
}
```

---

## HNSW Integration Design

### Current State

HNSW currently:
- Indexes `EntityId` values
- Uses `EmbeddingLookup` trait to fetch vectors
- Returns `Vec<SearchResult>` with `EntityId` and distance

### Design Options

**Option A: Continue indexing EntityId**

The HNSW index continues to use EntityId as the indexed item. It stores the embedding directly in the graph node.

```rust
// Current HnswNode structure
pub struct HnswNode {
    pub entity_id: EntityId,
    pub embedding: Embedding,  // Vector stored in node
    pub max_layer: usize,
    pub connections: Vec<Vec<EntityId>>,
}
```

Pros:
- Minimal changes to HNSW
- Search results are directly EntityIds (what users want)
- Vector stored in graph = fast distance computation

Cons:
- Duplicates vector storage (once in vector table, once in HNSW graph)
- ~2x storage for indexed vectors

**Option B: Store only EntityId + fetch during search**

```rust
pub struct HnswNode {
    pub entity_id: EntityId,
    // No embedding stored - fetch from vector storage when needed
    pub max_layer: usize,
    pub connections: Vec<Vec<EntityId>>,
}
```

Pros:
- No duplicate storage
- Vectors only in one place

Cons:
- Every distance calculation requires storage lookup
- Much slower search (I/O bound instead of CPU bound)
- Not viable for production performance

**Option C: Index VectorRef, map back to EntityId - RECOMMENDED**

The HNSW index conceptually indexes VectorRef, but internally uses EntityId since there's a 1:1 mapping per vector name.

```rust
/// HNSW index for a specific named vector in a collection.
pub struct CollectionHnswIndex {
    /// The collection this index belongs to.
    collection_id: CollectionId,
    /// The vector name being indexed.
    vector_name: String,
    /// The underlying HNSW index (still uses EntityId).
    hnsw: HnswIndex<E>,
}

impl CollectionHnswIndex {
    /// Insert a vector for an entity.
    pub fn insert(&mut self, entity_id: EntityId, vector: &VectorData) -> Result<()> {
        let embedding = vector.to_embedding()?;
        self.hnsw.insert(entity_id, &embedding)
    }

    /// Search returns EntityIds directly.
    pub fn search(&self, query: &VectorData, k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = query.to_embedding()?;
        self.hnsw.search(&query_embedding, k, None)
    }
}
```

**Decision: Option C (Wrapper with EntityId indexing)**

The HNSW index continues to:
- Use EntityId as the indexed identifier
- Store embeddings in graph nodes for fast distance computation
- Return EntityIds in search results

The only change is semantic: the index is "for a specific named vector" and gets its data from the vector storage.

### Multiple Vectors Per Entity

For multi-modal entities (text + image embeddings):

```rust
// Collection with multiple vector types
let collection = db.create_collection("documents")
    .with_dense_vector("text", 768, DistanceMetric::Cosine)
    .with_dense_vector("image", 512, DistanceMetric::Cosine)
    .build()?;

// Each vector type gets its own HNSW index
// - documents_text_hnsw indexes text vectors
// - documents_image_hnsw indexes image vectors

// Insert entity with both vectors
collection.upsert_entity(entity_id)
    .with_vector("text", text_embedding)
    .with_vector("image", image_embedding)
    .execute()?;

// Search by text similarity
let results = collection.search("text")
    .query(query_text_embedding)
    .limit(10)
    .execute()?;

// Search by image similarity
let results = collection.search("image")
    .query(query_image_embedding)
    .limit(10)
    .execute()?;

// Hybrid search (combine text + image scores)
let results = collection.hybrid_search()
    .query("text", query_text_embedding, 0.7)
    .query("image", query_image_embedding, 0.3)
    .limit(10)
    .execute()?;
```

---

## Migration Strategy

### Approach: Additive Migration

We don't modify existing storage. Instead:

1. Add new vector storage tables
2. Keep Value::Vector for backwards compatibility
3. New collections use the new storage
4. Migration tool for existing data

### Migration Tool

```rust
/// Migrate vectors from entity properties to dedicated storage.
pub struct VectorMigrator<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> VectorMigrator<E> {
    /// Migrate a collection's vectors from property storage to vector storage.
    ///
    /// This reads entities with Vector properties and stores them in the
    /// new vector storage tables.
    ///
    /// # Arguments
    /// - `collection`: The collection to migrate
    /// - `property_mappings`: Maps property names to vector names
    ///   e.g., `[("embedding", "text")]` means the "embedding" property
    ///   becomes the "text" named vector
    pub fn migrate_collection(
        &self,
        collection: &Collection,
        property_mappings: &[(&str, &str)],
    ) -> Result<MigrationStats> {
        let mut stats = MigrationStats::default();

        // Read all entities in the collection
        for entity in self.read_all_entities(collection)? {
            for (property_name, vector_name) in property_mappings {
                if let Some(Value::Vector(data)) = entity.get_property(property_name) {
                    // Store in new vector storage
                    self.store_vector(
                        collection.id(),
                        entity.id,
                        vector_name,
                        VectorData::Dense(data.clone()),
                    )?;
                    stats.vectors_migrated += 1;
                }
                // Handle SparseVector, MultiVector similarly
            }
            stats.entities_processed += 1;
        }

        Ok(stats)
    }

    /// Remove migrated vector properties from entities.
    ///
    /// Call this AFTER verifying migration was successful.
    pub fn cleanup_properties(
        &self,
        collection: &Collection,
        property_names: &[&str],
    ) -> Result<usize> {
        let mut cleaned = 0;

        for mut entity in self.read_all_entities(collection)? {
            let mut modified = false;
            for property_name in property_names {
                if entity.properties.remove(*property_name).is_some() {
                    modified = true;
                }
            }
            if modified {
                self.update_entity(&entity)?;
                cleaned += 1;
            }
        }

        Ok(cleaned)
    }
}

#[derive(Debug, Default)]
pub struct MigrationStats {
    pub entities_processed: usize,
    pub vectors_migrated: usize,
    pub errors: Vec<String>,
}
```

### Backwards Compatibility

For existing databases:
1. `Value::Vector` continues to work
2. Old code reads vectors from entity properties
3. Migration is optional but recommended
4. No breaking changes to entity storage format

---

## API Surface Design

### Low-Level Vector Storage API

```rust
/// Low-level vector storage operations.
pub trait VectorStorage {
    /// Store a vector for an entity.
    fn put_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError>;

    /// Get a vector for an entity.
    fn get_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<Option<VectorData>, VectorError>;

    /// Get all vectors for an entity.
    fn get_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<HashMap<String, VectorData>, VectorError>;

    /// Delete a specific vector.
    fn delete_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<bool, VectorError>;

    /// Delete all vectors for an entity.
    fn delete_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError>;

    /// List all entity IDs with a specific vector.
    fn list_entities_with_vector(
        &self,
        collection_id: CollectionId,
        vector_name: &str,
    ) -> Result<Vec<EntityId>, VectorError>;

    /// Batch operations
    fn put_vectors_batch(
        &self,
        collection_id: CollectionId,
        vectors: &[(EntityId, &str, &VectorData)],
    ) -> Result<(), VectorError>;
}
```

### High-Level Collection API

```rust
impl<E: StorageEngine> CollectionHandle<E> {
    /// Upsert an entity with vectors.
    pub fn upsert(&self, entity_id: EntityId) -> UpsertBuilder<E> {
        UpsertBuilder::new(self, entity_id)
    }

    /// Get vectors for an entity.
    pub fn get_vectors(&self, entity_id: EntityId) -> Result<HashMap<String, VectorData>> {
        self.vector_store.get_all_vectors(self.collection_id, entity_id)
    }

    /// Get a specific vector.
    pub fn get_vector(&self, entity_id: EntityId, name: &str) -> Result<Option<VectorData>> {
        self.vector_store.get_vector(self.collection_id, entity_id, name)
    }

    /// Update a specific vector without touching payload.
    pub fn update_vector(
        &self,
        entity_id: EntityId,
        name: &str,
        data: &VectorData,
    ) -> Result<()> {
        // Store vector
        self.vector_store.put_vector(self.collection_id, entity_id, name, data)?;

        // Update HNSW index
        if let Some(index) = self.get_index(name)? {
            index.insert(entity_id, data.to_embedding()?)?;
        }

        Ok(())
    }
}

/// Builder for upserting entities with vectors.
pub struct UpsertBuilder<'a, E: StorageEngine> {
    handle: &'a CollectionHandle<E>,
    entity_id: EntityId,
    payload: Option<Payload>,
    vectors: HashMap<String, VectorData>,
}

impl<'a, E: StorageEngine> UpsertBuilder<'a, E> {
    pub fn with_payload(mut self, payload: Payload) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn with_vector(mut self, name: impl Into<String>, data: impl Into<VectorData>) -> Self {
        self.vectors.insert(name.into(), data.into());
        self
    }

    pub fn execute(self) -> Result<()> {
        let mut tx = self.handle.engine.begin_write()?;

        // Store/update payload
        if let Some(payload) = self.payload {
            self.handle.store_payload_tx(&mut tx, self.entity_id, &payload)?;
        }

        // Store vectors
        for (name, data) in &self.vectors {
            self.handle.store_vector_tx(&mut tx, self.entity_id, name, data)?;
        }

        tx.commit()?;

        // Update HNSW indexes (outside transaction for now)
        for (name, data) in &self.vectors {
            if let Some(index) = self.handle.get_index(name)? {
                index.insert(self.entity_id, &data.to_embedding()?)?;
            }
        }

        Ok(())
    }
}
```

---

## Implementation Phases

### Phase 1: Core Vector Storage
- [ ] Define `VectorData`, `VectorRef`, `StoredVector` types
- [ ] Implement `CollectionVectorStore` with CRUD operations
- [ ] Add key encoding/decoding functions
- [ ] Add cascade delete in entity deletion path
- [ ] Unit tests for storage operations

### Phase 2: HNSW Integration
- [ ] Create `CollectionHnswIndex` wrapper
- [ ] Update index creation to use vector storage
- [ ] Update index insertion to fetch from vector storage
- [ ] Verify search still returns EntityIds
- [ ] Integration tests for search with new storage

### Phase 3: Collection API Updates
- [ ] Add `UpsertBuilder` with vector support
- [ ] Add `get_vectors`, `get_vector`, `update_vector` methods
- [ ] Update `delete_entity` for cascade delete
- [ ] Deprecate vector-as-property pattern in docs

### Phase 4: Migration Tools
- [ ] Implement `VectorMigrator`
- [ ] Add CLI command for migration
- [ ] Write migration guide documentation
- [ ] Test migration with sample databases

### Phase 5: Cleanup & Documentation
- [ ] Update all documentation
- [ ] Add examples for new API
- [ ] Performance benchmarks comparing old vs new
- [ ] Consider removing Value::Vector (major version)

---

## New Type Definitions

### File: `crates/manifoldb-vector/src/types/vector.rs`

```rust
//! Core types for separated vector storage.

use serde::{Deserialize, Serialize};
use manifoldb_core::EntityId;

/// Reference to a vector attached to an entity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorRef {
    /// The entity this vector belongs to.
    pub entity_id: EntityId,
    /// The named vector space.
    pub vector_name: String,
}

impl VectorRef {
    /// Create a new vector reference.
    pub fn new(entity_id: EntityId, vector_name: impl Into<String>) -> Self {
        Self {
            entity_id,
            vector_name: vector_name.into(),
        }
    }
}

/// Vector data supporting different types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorData {
    /// Dense floating-point vector.
    Dense(Vec<f32>),
    /// Sparse vector with (index, value) pairs.
    Sparse(Vec<(u32, f32)>),
    /// Multi-vector (ColBERT-style token embeddings).
    Multi(Vec<Vec<f32>>),
    /// Binary vector (bit-packed).
    Binary(Vec<u8>),
}

impl VectorData {
    /// Get the dimension of the vector.
    pub fn dimension(&self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Sparse(v) => v.iter().map(|(i, _)| *i as usize + 1).max().unwrap_or(0),
            Self::Multi(v) => v.first().map(|inner| inner.len()).unwrap_or(0),
            Self::Binary(v) => v.len() * 8,
        }
    }

    /// Check if this is a dense vector.
    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense(_))
    }

    /// Check if this is a sparse vector.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    /// Get as dense vector slice.
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            Self::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Convert to Embedding for HNSW (dense vectors only).
    pub fn to_embedding(&self) -> Result<Embedding, VectorError> {
        match self {
            Self::Dense(v) => Embedding::new(v.clone()),
            _ => Err(VectorError::Encoding(
                "Cannot convert non-dense vector to Embedding".to_string()
            )),
        }
    }
}

impl From<Vec<f32>> for VectorData {
    fn from(v: Vec<f32>) -> Self {
        Self::Dense(v)
    }
}

impl From<Vec<(u32, f32)>> for VectorData {
    fn from(v: Vec<(u32, f32)>) -> Self {
        Self::Sparse(v)
    }
}

impl From<Vec<Vec<f32>>> for VectorData {
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self::Multi(v)
    }
}
```

### File: `crates/manifoldb-vector/src/store/collection_vector_store.rs`

```rust
//! Vector storage for collections.
//!
//! This module provides dedicated vector storage separate from entity properties.

use std::collections::HashMap;
use std::ops::Bound;

use manifoldb_core::{CollectionId, EntityId};
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::error::VectorError;
use crate::types::VectorData;

/// Table name for collection vectors.
const TABLE_COLLECTION_VECTORS: &str = "collection_vectors";

/// Version byte for vector storage format.
const VECTOR_FORMAT_VERSION: u8 = 1;

/// Vector type discriminants.
const VECTOR_TYPE_DENSE: u8 = 0;
const VECTOR_TYPE_SPARSE: u8 = 1;
const VECTOR_TYPE_MULTI: u8 = 2;
const VECTOR_TYPE_BINARY: u8 = 3;

/// Vector storage for a collection.
pub struct CollectionVectorStore<E: StorageEngine> {
    engine: E,
}

impl<E: StorageEngine> CollectionVectorStore<E> {
    /// Create a new collection vector store.
    pub const fn new(engine: E) -> Self {
        Self { engine }
    }

    /// Store a vector for an entity.
    pub fn put_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError> {
        let mut tx = self.engine.begin_write()?;
        self.put_vector_tx(&mut tx, collection_id, entity_id, vector_name, data)?;
        tx.commit()?;
        Ok(())
    }

    /// Store a vector within a transaction.
    pub fn put_vector_tx<T: Transaction>(
        &self,
        tx: &mut T,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
        data: &VectorData,
    ) -> Result<(), VectorError> {
        let key = encode_vector_key(collection_id, entity_id, vector_name);
        let value = encode_vector_value(data)?;
        tx.put(TABLE_COLLECTION_VECTORS, &key, &value)?;
        Ok(())
    }

    /// Get a vector for an entity.
    pub fn get_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<Option<VectorData>, VectorError> {
        let tx = self.engine.begin_read()?;
        let key = encode_vector_key(collection_id, entity_id, vector_name);

        match tx.get(TABLE_COLLECTION_VECTORS, &key)? {
            Some(bytes) => Ok(Some(decode_vector_value(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Get all vectors for an entity.
    pub fn get_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<HashMap<String, VectorData>, VectorError> {
        let tx = self.engine.begin_read()?;
        let prefix = encode_entity_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        let mut cursor = tx.range(
            TABLE_COLLECTION_VECTORS,
            Bound::Included(prefix.as_slice()),
            Bound::Excluded(prefix_end.as_slice()),
        )?;

        let mut vectors = HashMap::new();
        while let Some((key, value)) = cursor.next()? {
            if let Some(vector_name) = extract_vector_name(&key) {
                let data = decode_vector_value(&value)?;
                vectors.insert(vector_name, data);
            }
        }

        Ok(vectors)
    }

    /// Delete a specific vector.
    pub fn delete_vector(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
        vector_name: &str,
    ) -> Result<bool, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let key = encode_vector_key(collection_id, entity_id, vector_name);
        let deleted = tx.delete(TABLE_COLLECTION_VECTORS, &key)?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Delete all vectors for an entity.
    pub fn delete_all_vectors(
        &self,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError> {
        let mut tx = self.engine.begin_write()?;
        let count = self.delete_all_vectors_tx(&mut tx, collection_id, entity_id)?;
        tx.commit()?;
        Ok(count)
    }

    /// Delete all vectors for an entity within a transaction.
    pub fn delete_all_vectors_tx<T: Transaction>(
        &self,
        tx: &mut T,
        collection_id: CollectionId,
        entity_id: EntityId,
    ) -> Result<usize, VectorError> {
        let prefix = encode_entity_prefix(collection_id, entity_id);
        let prefix_end = next_prefix(&prefix);

        // Collect keys to delete
        let mut keys_to_delete = Vec::new();
        {
            let cursor = tx.range(
                TABLE_COLLECTION_VECTORS,
                Bound::Included(prefix.as_slice()),
                Bound::Excluded(prefix_end.as_slice()),
            )?;

            let mut cursor = cursor;
            while let Some((key, _)) = cursor.next()? {
                keys_to_delete.push(key);
            }
        }

        // Delete collected keys
        let count = keys_to_delete.len();
        for key in keys_to_delete {
            tx.delete(TABLE_COLLECTION_VECTORS, &key)?;
        }

        Ok(count)
    }
}

// Key encoding functions

fn encode_vector_key(
    collection_id: CollectionId,
    entity_id: EntityId,
    vector_name: &str,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(24);
    key.extend_from_slice(&collection_id.as_u64().to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key.extend_from_slice(&hash_vector_name(vector_name).to_be_bytes());
    key
}

fn encode_entity_prefix(
    collection_id: CollectionId,
    entity_id: EntityId,
) -> Vec<u8> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&collection_id.as_u64().to_be_bytes());
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

fn hash_vector_name(name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

fn extract_vector_name(_key: &[u8]) -> Option<String> {
    // Note: We can't reverse the hash. In practice, we'd need to either:
    // 1. Store vector names in a separate table
    // 2. Include the name in the value
    // 3. Keep a collection-level map of hash -> name
    // For now, return None - this function needs the collection metadata
    None
}

fn next_prefix(prefix: &[u8]) -> Vec<u8> {
    let mut result = prefix.to_vec();
    for byte in result.iter_mut().rev() {
        if *byte < 0xFF {
            *byte += 1;
            return result;
        }
    }
    result.push(0xFF);
    result
}

// Value encoding functions

fn encode_vector_value(data: &VectorData) -> Result<Vec<u8>, VectorError> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut bytes = Vec::new();
    bytes.push(VECTOR_FORMAT_VERSION);

    match data {
        VectorData::Dense(v) => {
            bytes.push(VECTOR_TYPE_DENSE);
            bytes.extend_from_slice(&timestamp.to_be_bytes());
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            for &val in v {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        VectorData::Sparse(v) => {
            bytes.push(VECTOR_TYPE_SPARSE);
            bytes.extend_from_slice(&timestamp.to_be_bytes());
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            for &(idx, val) in v {
                bytes.extend_from_slice(&idx.to_be_bytes());
                bytes.extend_from_slice(&val.to_le_bytes());
            }
        }
        VectorData::Multi(v) => {
            bytes.push(VECTOR_TYPE_MULTI);
            bytes.extend_from_slice(&timestamp.to_be_bytes());
            let num_vectors = v.len() as u32;
            let dim = v.first().map(|inner| inner.len() as u32).unwrap_or(0);
            bytes.extend_from_slice(&num_vectors.to_be_bytes());
            bytes.extend_from_slice(&dim.to_be_bytes());
            for inner in v {
                for &val in inner {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        VectorData::Binary(v) => {
            bytes.push(VECTOR_TYPE_BINARY);
            bytes.extend_from_slice(&timestamp.to_be_bytes());
            bytes.extend_from_slice(&(v.len() as u32).to_be_bytes());
            bytes.extend_from_slice(v);
        }
    }

    Ok(bytes)
}

fn decode_vector_value(bytes: &[u8]) -> Result<VectorData, VectorError> {
    if bytes.len() < 10 {
        return Err(VectorError::Encoding("truncated vector value".to_string()));
    }

    let version = bytes[0];
    if version != VECTOR_FORMAT_VERSION {
        return Err(VectorError::Encoding(format!(
            "unsupported vector format version: {}",
            version
        )));
    }

    let vec_type = bytes[1];
    // Skip timestamp (bytes 2-9)
    let data_len = u32::from_be_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;

    match vec_type {
        VECTOR_TYPE_DENSE => {
            let expected_len = 14 + data_len * 4;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("dense vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(data_len);
            for i in 0..data_len {
                let offset = 14 + i * 4;
                let val = f32::from_le_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                v.push(val);
            }
            Ok(VectorData::Dense(v))
        }
        VECTOR_TYPE_SPARSE => {
            let expected_len = 14 + data_len * 8;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("sparse vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(data_len);
            for i in 0..data_len {
                let offset = 14 + i * 8;
                let idx = u32::from_be_bytes([
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ]);
                let val = f32::from_le_bytes([
                    bytes[offset + 4],
                    bytes[offset + 5],
                    bytes[offset + 6],
                    bytes[offset + 7],
                ]);
                v.push((idx, val));
            }
            Ok(VectorData::Sparse(v))
        }
        VECTOR_TYPE_MULTI => {
            if bytes.len() < 18 {
                return Err(VectorError::Encoding("truncated multi-vector".to_string()));
            }
            let num_vectors = data_len;
            let dim = u32::from_be_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]) as usize;
            let expected_len = 18 + num_vectors * dim * 4;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("multi-vector length mismatch".to_string()));
            }
            let mut v = Vec::with_capacity(num_vectors);
            for i in 0..num_vectors {
                let mut inner = Vec::with_capacity(dim);
                for j in 0..dim {
                    let offset = 18 + (i * dim + j) * 4;
                    let val = f32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    inner.push(val);
                }
                v.push(inner);
            }
            Ok(VectorData::Multi(v))
        }
        VECTOR_TYPE_BINARY => {
            let expected_len = 14 + data_len;
            if bytes.len() != expected_len {
                return Err(VectorError::Encoding("binary vector length mismatch".to_string()));
            }
            Ok(VectorData::Binary(bytes[14..].to_vec()))
        }
        _ => Err(VectorError::Encoding(format!("unknown vector type: {}", vec_type))),
    }
}
```

---

## Summary

This design provides:

1. **Clean separation**: Vectors stored independently from entities
2. **Multiple vectors per entity**: Support for text, image, summary embeddings
3. **Efficient access**: Read entity without loading vectors, read vectors without loading entity
4. **Simple addressing**: VectorRef = (EntityId, vector_name) matches query patterns
5. **Cascade delete**: Deleting entity removes all its vectors
6. **HNSW compatibility**: Index continues to return EntityIds
7. **Backwards compatible**: Existing Value::Vector still works
8. **Clear migration path**: Tools to migrate existing data

The implementation can proceed in phases, with the core storage layer implemented first, followed by HNSW integration and API updates.
