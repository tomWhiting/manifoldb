//! HNSW index registry for managing vector indexes.
//!
//! The registry provides:
//! - Index metadata storage and retrieval
//! - Index lifecycle management (create, load, drop)
//! - Coordination between indexes and storage transactions
//!
//! ## Storage Layout
//!
//! The registry uses the `hnsw_registry` table to store index metadata.
//! Each index entry contains:
//! - Index name
//! - Table name (the entity label)
//! - Column name (the vector property)
//! - Dimension and distance metric
//! - HNSW configuration parameters

use serde::{Deserialize, Serialize};

use manifoldb_core::EntityId;
use manifoldb_storage::{Cursor, Transaction};

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::types::Embedding;

use super::config::HnswConfig;

/// Well-known table name for HNSW registry.
pub const HNSW_REGISTRY_TABLE: &str = "hnsw_registry";

/// Registry entry for an HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndexEntry {
    /// The unique name of the index.
    pub name: String,
    /// The table (entity label) this index is on.
    pub table: String,
    /// The column (property name) containing vectors.
    pub column: String,
    /// The dimension of vectors in this index.
    pub dimension: usize,
    /// The distance metric used.
    pub distance_metric: DistanceMetricSerde,
    /// HNSW M parameter (max connections per node).
    pub m: usize,
    /// HNSW M_max0 parameter (max connections in layer 0).
    pub m_max0: usize,
    /// HNSW ef_construction parameter.
    pub ef_construction: usize,
    /// HNSW ef_search parameter.
    pub ef_search: usize,
    /// The ml parameter as bits.
    pub ml_bits: u64,
    /// Number of PQ segments (0 = disabled).
    #[serde(default)]
    pub pq_segments: usize,
    /// Number of PQ centroids per segment.
    #[serde(default = "default_pq_centroids")]
    pub pq_centroids: usize,
}

fn default_pq_centroids() -> usize {
    256
}

/// Serializable version of DistanceMetric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetricSerde {
    /// L2 (Euclidean) distance.
    Euclidean,
    /// Cosine similarity (1 - cosine).
    Cosine,
    /// Dot product (negated for distance).
    DotProduct,
    /// Manhattan (L1) distance.
    Manhattan,
    /// Chebyshev (L∞) distance.
    Chebyshev,
}

impl From<DistanceMetric> for DistanceMetricSerde {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Euclidean => Self::Euclidean,
            DistanceMetric::Cosine => Self::Cosine,
            DistanceMetric::DotProduct => Self::DotProduct,
            DistanceMetric::Manhattan => Self::Manhattan,
            DistanceMetric::Chebyshev => Self::Chebyshev,
        }
    }
}

impl From<DistanceMetricSerde> for DistanceMetric {
    fn from(metric: DistanceMetricSerde) -> Self {
        match metric {
            DistanceMetricSerde::Euclidean => Self::Euclidean,
            DistanceMetricSerde::Cosine => Self::Cosine,
            DistanceMetricSerde::DotProduct => Self::DotProduct,
            DistanceMetricSerde::Manhattan => Self::Manhattan,
            DistanceMetricSerde::Chebyshev => Self::Chebyshev,
        }
    }
}

impl HnswIndexEntry {
    /// Create a new index entry.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        table: impl Into<String>,
        column: impl Into<String>,
        dimension: usize,
        distance_metric: DistanceMetric,
        config: &HnswConfig,
    ) -> Self {
        Self {
            name: name.into(),
            table: table.into(),
            column: column.into(),
            dimension,
            distance_metric: distance_metric.into(),
            m: config.m,
            m_max0: config.m_max0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            ml_bits: config.ml.to_bits(),
            pq_segments: config.pq_segments,
            pq_centroids: config.pq_centroids,
        }
    }

    /// Get the HNSW configuration from this entry.
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        HnswConfig {
            m: self.m,
            m_max0: self.m_max0,
            ef_construction: self.ef_construction,
            ef_search: self.ef_search,
            ml: f64::from_bits(self.ml_bits),
            pq_segments: self.pq_segments,
            pq_centroids: self.pq_centroids,
            pq_training_samples: 1000, // Default
        }
    }

    /// Get the distance metric from this entry.
    #[must_use]
    pub fn distance_metric(&self) -> DistanceMetric {
        self.distance_metric.into()
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, VectorError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| VectorError::Encoding(format!("failed to serialize index entry: {e}")))
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .map(|(entry, _)| entry)
            .map_err(|e| VectorError::Encoding(format!("failed to deserialize index entry: {e}")))
    }
}

/// Registry for managing HNSW indexes.
///
/// The registry stores index metadata and provides methods for:
/// - Registering new indexes
/// - Looking up index configuration
/// - Listing all indexes for a table
/// - Dropping indexes
pub struct HnswRegistry;

impl HnswRegistry {
    /// Register a new HNSW index.
    ///
    /// This stores the index metadata in the registry table.
    /// The actual index data is stored separately using the persistence module.
    pub fn register<T: Transaction>(tx: &mut T, entry: &HnswIndexEntry) -> Result<(), VectorError> {
        let key = Self::entry_key(&entry.name);
        let value = entry.to_bytes()?;
        tx.put(HNSW_REGISTRY_TABLE, &key, &value)?;
        Ok(())
    }

    /// Get an index entry by name.
    pub fn get<T: Transaction>(tx: &T, name: &str) -> Result<Option<HnswIndexEntry>, VectorError> {
        let key = Self::entry_key(name);
        match tx.get(HNSW_REGISTRY_TABLE, &key)? {
            Some(bytes) => Ok(Some(HnswIndexEntry::from_bytes(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Check if an index exists.
    pub fn exists<T: Transaction>(tx: &T, name: &str) -> Result<bool, VectorError> {
        let key = Self::entry_key(name);
        Ok(tx.get(HNSW_REGISTRY_TABLE, &key)?.is_some())
    }

    /// Drop an index from the registry.
    ///
    /// This only removes the registry entry. The caller is responsible for
    /// cleaning up the actual index data.
    pub fn drop<T: Transaction>(tx: &mut T, name: &str) -> Result<bool, VectorError> {
        let key = Self::entry_key(name);
        Ok(tx.delete(HNSW_REGISTRY_TABLE, &key)?)
    }

    /// List all indexes for a specific table.
    pub fn list_for_table<T: Transaction>(
        tx: &T,
        table: &str,
    ) -> Result<Vec<HnswIndexEntry>, VectorError> {
        use std::ops::Bound;

        let mut entries = Vec::new();

        // Scan all entries in the registry
        let mut cursor = tx.range(HNSW_REGISTRY_TABLE, Bound::Unbounded, Bound::Unbounded)?;

        while let Some((_, value)) = cursor.next()? {
            if let Ok(entry) = HnswIndexEntry::from_bytes(&value) {
                if entry.table == table {
                    entries.push(entry);
                }
            }
        }

        Ok(entries)
    }

    /// List all indexes for a specific table and column.
    pub fn list_for_column<T: Transaction>(
        tx: &T,
        table: &str,
        column: &str,
    ) -> Result<Vec<HnswIndexEntry>, VectorError> {
        let table_entries = Self::list_for_table(tx, table)?;
        Ok(table_entries.into_iter().filter(|e| e.column == column).collect())
    }

    /// List all registered indexes.
    pub fn list_all<T: Transaction>(tx: &T) -> Result<Vec<HnswIndexEntry>, VectorError> {
        use std::ops::Bound;

        let mut entries = Vec::new();

        // Scan all entries in the registry
        let mut cursor = tx.range(HNSW_REGISTRY_TABLE, Bound::Unbounded, Bound::Unbounded)?;

        while let Some((_, value)) = cursor.next()? {
            if let Ok(entry) = HnswIndexEntry::from_bytes(&value) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Generate the storage key for an index entry.
    fn entry_key(name: &str) -> Vec<u8> {
        name.as_bytes().to_vec()
    }
}

/// Trait for looking up embeddings by entity ID.
///
/// This is used when building or updating HNSW indexes to retrieve
/// the vector data for entities.
pub trait EmbeddingLookup {
    /// Get the embedding for an entity from a specific column.
    fn get_embedding(
        &self,
        entity_id: EntityId,
        column: &str,
    ) -> Result<Option<Embedding>, VectorError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;
    use manifoldb_storage::StorageEngine;

    #[test]
    fn test_index_entry_roundtrip() {
        let config = HnswConfig::default();
        let entry = HnswIndexEntry::new(
            "test_index",
            "documents",
            "embedding",
            384,
            DistanceMetric::Cosine,
            &config,
        );

        let bytes = entry.to_bytes().unwrap();
        let decoded = HnswIndexEntry::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.name, "test_index");
        assert_eq!(decoded.table, "documents");
        assert_eq!(decoded.column, "embedding");
        assert_eq!(decoded.dimension, 384);
        assert_eq!(decoded.distance_metric, DistanceMetricSerde::Cosine);
    }

    #[test]
    fn test_registry_crud() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();

        let entry = HnswIndexEntry::new(
            "test_index",
            "documents",
            "embedding",
            384,
            DistanceMetric::Cosine,
            &config,
        );

        // Register
        {
            let mut tx = engine.begin_write().unwrap();
            HnswRegistry::register(&mut tx, &entry).unwrap();
            tx.commit().unwrap();
        }

        // Get
        {
            let tx = engine.begin_read().unwrap();
            let retrieved = HnswRegistry::get(&tx, "test_index").unwrap().unwrap();
            assert_eq!(retrieved.name, "test_index");
            assert!(HnswRegistry::exists(&tx, "test_index").unwrap());
            assert!(!HnswRegistry::exists(&tx, "nonexistent").unwrap());
        }

        // List
        {
            let tx = engine.begin_read().unwrap();
            let entries = HnswRegistry::list_for_table(&tx, "documents").unwrap();
            assert_eq!(entries.len(), 1);
            let all = HnswRegistry::list_all(&tx).unwrap();
            assert_eq!(all.len(), 1);
        }

        // Drop
        {
            let mut tx = engine.begin_write().unwrap();
            assert!(HnswRegistry::drop(&mut tx, "test_index").unwrap());
            tx.commit().unwrap();
        }

        // Verify dropped
        {
            let tx = engine.begin_read().unwrap();
            assert!(!HnswRegistry::exists(&tx, "test_index").unwrap());
        }
    }
}
