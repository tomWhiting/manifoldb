//! HNSW index persistence.
//!
//! This module handles saving and loading HNSW indexes to/from storage.
//!
//! ## Storage Layout
//!
//! The HNSW index uses the following key prefixes:
//! - `0x20` - Index metadata (entry point, max layer, config)
//! - `0x21` - Node data (embedding, max_layer)
//! - `0x22` - Node connections (per-layer neighbor lists)

// Allow unwrap on try_into for fixed-size slice conversions which are guaranteed to succeed
#![allow(clippy::unwrap_used)]

use manifoldb_core::EntityId;
use manifoldb_storage::{Cursor, StorageEngine, Transaction};

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::types::Embedding;

use super::config::HnswConfig;
use super::graph::{HnswGraph, HnswNode};

/// Key prefix for HNSW index metadata.
pub const PREFIX_HNSW_META: u8 = 0x20;

/// Key prefix for HNSW node data.
pub const PREFIX_HNSW_NODE: u8 = 0x21;

/// Key prefix for HNSW node connections.
pub const PREFIX_HNSW_CONNECTIONS: u8 = 0x22;

/// Table name for HNSW index data.
pub fn table_name(index_name: &str) -> String {
    format!("hnsw_{index_name}")
}

/// Encode the metadata key for an index.
fn encode_meta_key() -> Vec<u8> {
    vec![PREFIX_HNSW_META]
}

/// Encode a node key.
fn encode_node_key(entity_id: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(9);
    key.push(PREFIX_HNSW_NODE);
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key
}

/// Encode a connections key for a node at a specific layer.
fn encode_connections_key(entity_id: EntityId, layer: usize) -> Vec<u8> {
    let mut key = Vec::with_capacity(13);
    key.push(PREFIX_HNSW_CONNECTIONS);
    key.extend_from_slice(&entity_id.as_u64().to_be_bytes());
    key.extend_from_slice(&(layer as u32).to_be_bytes());
    key
}

/// Decode a node key.
fn decode_node_key(key: &[u8]) -> Option<EntityId> {
    if key.len() != 9 || key[0] != PREFIX_HNSW_NODE {
        return None;
    }
    let bytes: [u8; 8] = key[1..9].try_into().ok()?;
    Some(EntityId::new(u64::from_be_bytes(bytes)))
}

/// Index metadata stored in the database.
#[derive(Debug, Clone)]
pub struct IndexMetadata {
    /// The dimension of embeddings.
    pub dimension: usize,
    /// The distance metric.
    pub distance_metric: DistanceMetric,
    /// The entry point entity ID, if any.
    pub entry_point: Option<EntityId>,
    /// The maximum layer in the graph.
    pub max_layer: usize,
    /// The M parameter.
    pub m: usize,
    /// The M_max0 parameter.
    pub m_max0: usize,
    /// The ef_construction parameter.
    pub ef_construction: usize,
    /// The ef_search parameter.
    pub ef_search: usize,
    /// The ml parameter (stored as bits).
    pub ml_bits: u64,
}

impl IndexMetadata {
    /// Serialize metadata to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(64);

        // Version byte
        bytes.push(1);

        // Dimension (4 bytes)
        bytes.extend_from_slice(&(self.dimension as u32).to_be_bytes());

        // Distance metric (1 byte)
        bytes.push(match self.distance_metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::DotProduct => 2,
        });

        // Entry point (1 byte flag + 8 bytes if present)
        if let Some(ep) = self.entry_point {
            bytes.push(1);
            bytes.extend_from_slice(&ep.as_u64().to_be_bytes());
        } else {
            bytes.push(0);
        }

        // Max layer (4 bytes)
        bytes.extend_from_slice(&(self.max_layer as u32).to_be_bytes());

        // Config parameters (4 bytes each)
        bytes.extend_from_slice(&(self.m as u32).to_be_bytes());
        bytes.extend_from_slice(&(self.m_max0 as u32).to_be_bytes());
        bytes.extend_from_slice(&(self.ef_construction as u32).to_be_bytes());
        bytes.extend_from_slice(&(self.ef_search as u32).to_be_bytes());

        // ml as bits (8 bytes)
        bytes.extend_from_slice(&self.ml_bits.to_be_bytes());

        bytes
    }

    /// Deserialize metadata from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.is_empty() {
            return Err(VectorError::Encoding("empty metadata".into()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(VectorError::Encoding(format!("unsupported metadata version: {version}")));
        }

        let mut pos = 1;

        let read_u32 = |bytes: &[u8], pos: &mut usize| -> Result<u32, VectorError> {
            if *pos + 4 > bytes.len() {
                return Err(VectorError::Encoding("truncated metadata".into()));
            }
            let val = u32::from_be_bytes(bytes[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(val)
        };

        let read_u64 = |bytes: &[u8], pos: &mut usize| -> Result<u64, VectorError> {
            if *pos + 8 > bytes.len() {
                return Err(VectorError::Encoding("truncated metadata".into()));
            }
            let val = u64::from_be_bytes(bytes[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(val)
        };

        let dimension = read_u32(bytes, &mut pos)? as usize;

        if pos >= bytes.len() {
            return Err(VectorError::Encoding("truncated metadata".into()));
        }
        let distance_metric = match bytes[pos] {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Cosine,
            2 => DistanceMetric::DotProduct,
            b => return Err(VectorError::Encoding(format!("unknown distance metric: {b}"))),
        };
        pos += 1;

        if pos >= bytes.len() {
            return Err(VectorError::Encoding("truncated metadata".into()));
        }
        let has_entry_point = bytes[pos] == 1;
        pos += 1;

        let entry_point =
            if has_entry_point { Some(EntityId::new(read_u64(bytes, &mut pos)?)) } else { None };

        let max_layer = read_u32(bytes, &mut pos)? as usize;
        let m = read_u32(bytes, &mut pos)? as usize;
        let m_max0 = read_u32(bytes, &mut pos)? as usize;
        let ef_construction = read_u32(bytes, &mut pos)? as usize;
        let ef_search = read_u32(bytes, &mut pos)? as usize;
        let ml_bits = read_u64(bytes, &mut pos)?;

        Ok(Self {
            dimension,
            distance_metric,
            entry_point,
            max_layer,
            m,
            m_max0,
            ef_construction,
            ef_search,
            ml_bits,
        })
    }
}

/// Node data stored in the database (without connections).
#[derive(Debug, Clone)]
pub struct NodeData {
    /// The embedding vector.
    pub embedding: Embedding,
    /// The maximum layer this node appears in.
    pub max_layer: usize,
}

impl NodeData {
    /// Serialize node data to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let embedding_bytes = self.embedding.to_bytes();
        let mut bytes = Vec::with_capacity(5 + embedding_bytes.len());

        // Version byte
        bytes.push(1);

        // Max layer (4 bytes)
        bytes.extend_from_slice(&(self.max_layer as u32).to_be_bytes());

        // Embedding bytes
        bytes.extend_from_slice(&embedding_bytes);

        bytes
    }

    /// Deserialize node data from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VectorError> {
        if bytes.len() < 5 {
            return Err(VectorError::Encoding("truncated node data".into()));
        }

        let version = bytes[0];
        if version != 1 {
            return Err(VectorError::Encoding(format!("unsupported node data version: {version}")));
        }

        let max_layer = u32::from_be_bytes(bytes[1..5].try_into().unwrap()) as usize;
        let embedding = Embedding::from_bytes(&bytes[5..])?;

        Ok(Self { embedding, max_layer })
    }
}

/// Serialize a list of neighbor IDs.
fn serialize_connections(neighbors: &[EntityId]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + neighbors.len() * 8);

    // Number of neighbors (4 bytes)
    bytes.extend_from_slice(&(neighbors.len() as u32).to_be_bytes());

    // Each neighbor ID (8 bytes each)
    for &id in neighbors {
        bytes.extend_from_slice(&id.as_u64().to_be_bytes());
    }

    bytes
}

/// Deserialize a list of neighbor IDs.
fn deserialize_connections(bytes: &[u8]) -> Result<Vec<EntityId>, VectorError> {
    if bytes.len() < 4 {
        return Err(VectorError::Encoding("truncated connections data".into()));
    }

    let count = u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let expected_len = 4 + count * 8;

    if bytes.len() < expected_len {
        return Err(VectorError::Encoding("truncated connections data".into()));
    }

    let mut neighbors = Vec::with_capacity(count);
    for i in 0..count {
        let start = 4 + i * 8;
        let id = u64::from_be_bytes(bytes[start..start + 8].try_into().unwrap());
        neighbors.push(EntityId::new(id));
    }

    Ok(neighbors)
}

/// Save index metadata to storage.
pub fn save_metadata<E: StorageEngine>(
    engine: &E,
    table: &str,
    metadata: &IndexMetadata,
) -> Result<(), VectorError> {
    let mut tx = engine.begin_write()?;
    tx.put(table, &encode_meta_key(), &metadata.to_bytes())?;
    tx.commit()?;
    Ok(())
}

/// Load index metadata from storage.
pub fn load_metadata<E: StorageEngine>(
    engine: &E,
    table: &str,
) -> Result<Option<IndexMetadata>, VectorError> {
    let tx = engine.begin_read()?;
    match tx.get(table, &encode_meta_key())? {
        Some(bytes) => Ok(Some(IndexMetadata::from_bytes(&bytes)?)),
        None => Ok(None),
    }
}

/// Save a single node to storage.
pub fn save_node<E: StorageEngine>(
    engine: &E,
    table: &str,
    node: &HnswNode,
) -> Result<(), VectorError> {
    let mut tx = engine.begin_write()?;

    // Save node data
    let node_data = NodeData { embedding: node.embedding.clone(), max_layer: node.max_layer };
    tx.put(table, &encode_node_key(node.entity_id), &node_data.to_bytes())?;

    // Save connections for each layer
    for (layer, neighbors) in node.connections.iter().enumerate() {
        let key = encode_connections_key(node.entity_id, layer);
        tx.put(table, &key, &serialize_connections(neighbors))?;
    }

    tx.commit()?;
    Ok(())
}

/// Load a single node from storage.
pub fn load_node<E: StorageEngine>(
    engine: &E,
    table: &str,
    entity_id: EntityId,
) -> Result<Option<HnswNode>, VectorError> {
    let tx = engine.begin_read()?;

    // Load node data
    let node_data = match tx.get(table, &encode_node_key(entity_id))? {
        Some(bytes) => NodeData::from_bytes(&bytes)?,
        None => return Ok(None),
    };

    // Load connections for each layer
    let mut connections = Vec::with_capacity(node_data.max_layer + 1);
    for layer in 0..=node_data.max_layer {
        let key = encode_connections_key(entity_id, layer);
        let neighbors = match tx.get(table, &key)? {
            Some(bytes) => deserialize_connections(&bytes)?,
            None => Vec::new(),
        };
        connections.push(neighbors);
    }

    Ok(Some(HnswNode {
        entity_id,
        embedding: node_data.embedding,
        max_layer: node_data.max_layer,
        connections,
    }))
}

/// Delete a node from storage.
pub fn delete_node<E: StorageEngine>(
    engine: &E,
    table: &str,
    entity_id: EntityId,
    max_layer: usize,
) -> Result<bool, VectorError> {
    let mut tx = engine.begin_write()?;

    // Delete node data
    let existed = tx.delete(table, &encode_node_key(entity_id))?;

    // Delete connections for each layer
    for layer in 0..=max_layer {
        let key = encode_connections_key(entity_id, layer);
        tx.delete(table, &key)?;
    }

    tx.commit()?;
    Ok(existed)
}

/// Update the connections for a node at a specific layer.
pub fn update_connections<E: StorageEngine>(
    engine: &E,
    table: &str,
    entity_id: EntityId,
    layer: usize,
    neighbors: &[EntityId],
) -> Result<(), VectorError> {
    let mut tx = engine.begin_write()?;
    let key = encode_connections_key(entity_id, layer);
    tx.put(table, &key, &serialize_connections(neighbors))?;
    tx.commit()?;
    Ok(())
}

/// Load the entire graph from storage.
pub fn load_graph<E: StorageEngine>(
    engine: &E,
    table: &str,
    metadata: &IndexMetadata,
) -> Result<HnswGraph, VectorError> {
    let mut graph = HnswGraph::new(metadata.dimension, metadata.distance_metric);
    graph.entry_point = metadata.entry_point;
    graph.max_layer = metadata.max_layer;

    let tx = engine.begin_read()?;

    // Scan for all node keys
    let node_prefix = [PREFIX_HNSW_NODE];
    let node_end = [PREFIX_HNSW_NODE + 1];

    let mut cursor = tx.range(
        table,
        std::ops::Bound::Included(&node_prefix[..]),
        std::ops::Bound::Excluded(&node_end[..]),
    )?;

    let mut entity_ids = Vec::new();
    while let Some((key, _)) = cursor.next()? {
        if let Some(entity_id) = decode_node_key(&key) {
            entity_ids.push(entity_id);
        }
    }
    drop(cursor);
    drop(tx);

    // Load each node
    for entity_id in entity_ids {
        if let Some(node) = load_node(engine, table, entity_id)? {
            graph.nodes.insert(entity_id, node);
        }
    }

    Ok(graph)
}

/// Save the entire graph to storage.
pub fn save_graph<E: StorageEngine>(
    engine: &E,
    table: &str,
    graph: &HnswGraph,
    config: &HnswConfig,
) -> Result<(), VectorError> {
    // Save metadata
    let metadata = IndexMetadata {
        dimension: graph.dimension,
        distance_metric: graph.distance_metric,
        entry_point: graph.entry_point,
        max_layer: graph.max_layer,
        m: config.m,
        m_max0: config.m_max0,
        ef_construction: config.ef_construction,
        ef_search: config.ef_search,
        ml_bits: config.ml.to_bits(),
    };
    save_metadata(engine, table, &metadata)?;

    // Save all nodes
    for node in graph.nodes.values() {
        save_node(engine, table, node)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_embedding(dim: usize, value: f32) -> Embedding {
        Embedding::new(vec![value; dim]).unwrap()
    }

    #[test]
    fn test_metadata_roundtrip() {
        let metadata = IndexMetadata {
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            entry_point: Some(EntityId::new(42)),
            max_layer: 3,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            ml_bits: 0.5_f64.to_bits(),
        };

        let bytes = metadata.to_bytes();
        let decoded = IndexMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.dimension, 128);
        assert_eq!(decoded.distance_metric, DistanceMetric::Cosine);
        assert_eq!(decoded.entry_point, Some(EntityId::new(42)));
        assert_eq!(decoded.max_layer, 3);
        assert_eq!(decoded.m, 16);
        assert_eq!(decoded.m_max0, 32);
        assert_eq!(decoded.ef_construction, 200);
        assert_eq!(decoded.ef_search, 50);
    }

    #[test]
    fn test_metadata_no_entry_point() {
        let metadata = IndexMetadata {
            dimension: 64,
            distance_metric: DistanceMetric::Euclidean,
            entry_point: None,
            max_layer: 0,
            m: 32,
            m_max0: 64,
            ef_construction: 100,
            ef_search: 25,
            ml_bits: 0.3_f64.to_bits(),
        };

        let bytes = metadata.to_bytes();
        let decoded = IndexMetadata::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.entry_point, None);
    }

    #[test]
    fn test_node_data_roundtrip() {
        let embedding = create_test_embedding(4, 1.5);
        let node_data = NodeData { embedding: embedding.clone(), max_layer: 2 };

        let bytes = node_data.to_bytes();
        let decoded = NodeData::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.max_layer, 2);
        assert_eq!(decoded.embedding.as_slice(), embedding.as_slice());
    }

    #[test]
    fn test_connections_roundtrip() {
        let neighbors = vec![EntityId::new(1), EntityId::new(5), EntityId::new(10)];
        let bytes = serialize_connections(&neighbors);
        let decoded = deserialize_connections(&bytes).unwrap();

        assert_eq!(decoded, neighbors);
    }

    #[test]
    fn test_connections_empty() {
        let neighbors: Vec<EntityId> = vec![];
        let bytes = serialize_connections(&neighbors);
        let decoded = deserialize_connections(&bytes).unwrap();

        assert!(decoded.is_empty());
    }

    #[test]
    fn test_save_load_node() {
        let engine = RedbEngine::in_memory().unwrap();
        let table = "test_hnsw";

        let mut node = HnswNode::new(EntityId::new(42), create_test_embedding(4, 1.0), 2);
        node.connections[0] = vec![EntityId::new(1), EntityId::new(2)];
        node.connections[1] = vec![EntityId::new(3)];

        save_node(&engine, table, &node).unwrap();
        let loaded = load_node(&engine, table, EntityId::new(42)).unwrap().unwrap();

        assert_eq!(loaded.entity_id, EntityId::new(42));
        assert_eq!(loaded.max_layer, 2);
        assert_eq!(loaded.connections[0], vec![EntityId::new(1), EntityId::new(2)]);
        assert_eq!(loaded.connections[1], vec![EntityId::new(3)]);
    }

    #[test]
    fn test_save_load_metadata() {
        let engine = RedbEngine::in_memory().unwrap();
        let table = "test_hnsw";

        let metadata = IndexMetadata {
            dimension: 128,
            distance_metric: DistanceMetric::Cosine,
            entry_point: Some(EntityId::new(1)),
            max_layer: 3,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            ml_bits: 0.5_f64.to_bits(),
        };

        save_metadata(&engine, table, &metadata).unwrap();
        let loaded = load_metadata(&engine, table).unwrap().unwrap();

        assert_eq!(loaded.dimension, metadata.dimension);
        assert_eq!(loaded.entry_point, metadata.entry_point);
    }
}
