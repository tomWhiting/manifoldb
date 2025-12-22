//! HNSW index implementation.
//!
//! This is the main HNSW (Hierarchical Navigable Small World) index implementation.
//! It provides approximate nearest neighbor search with O(log N) complexity.

use std::sync::RwLock;

use manifoldb_core::EntityId;
use manifoldb_storage::StorageEngine;

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::types::Embedding;

use super::config::HnswConfig;
use super::graph::{search_layer, select_neighbors_heuristic, Candidate, HnswGraph, HnswNode};
use super::persistence::{
    self, delete_node, load_graph, load_metadata, save_graph, save_metadata, save_node, table_name,
    update_connections, IndexMetadata,
};
use super::traits::{SearchResult, VectorIndex};

/// Random level generator for HNSW.
///
/// Generates node levels using an exponential distribution,
/// as specified in the HNSW paper.
struct LevelGenerator {
    ml: f64,
    rng_state: u64,
}

impl LevelGenerator {
    #[allow(clippy::cast_possible_truncation)] // Intentional: nanos truncation is fine for seeding
    fn new(ml: f64) -> Self {
        // Use a simple seed based on current time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);
        Self { ml, rng_state: seed }
    }

    /// Generate a random level for a new node.
    #[allow(clippy::cast_precision_loss)] // Intentional: precision loss is acceptable for RNG
    #[allow(clippy::cast_possible_truncation)] // Intentional: level is bounded by min(16)
    #[allow(clippy::cast_sign_loss)] // Level is always non-negative after floor
    fn generate_level(&mut self) -> usize {
        // Simple xorshift64 PRNG
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        // Convert to uniform [0, 1)
        let uniform = (x as f64) / (u64::MAX as f64);

        // Exponential distribution: -ln(uniform) * ml
        // Truncate to integer
        let level = (-uniform.ln() * self.ml).floor() as usize;

        // Cap at reasonable maximum to prevent pathological cases
        level.min(16)
    }
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the main index structure that provides approximate nearest neighbor
/// search with configurable precision-performance tradeoffs.
pub struct HnswIndex<E: StorageEngine> {
    /// The storage engine.
    engine: E,
    /// The table name for this index.
    table: String,
    /// The in-memory graph representation.
    graph: RwLock<HnswGraph>,
    /// Index configuration.
    config: HnswConfig,
    /// Level generator for new nodes.
    level_gen: RwLock<LevelGenerator>,
}

impl<E: StorageEngine> HnswIndex<E> {
    /// Create a new HNSW index.
    ///
    /// If an index already exists in storage, it will be loaded.
    /// Otherwise, a new empty index is created.
    ///
    /// # Arguments
    ///
    /// * `engine` - The storage engine to use
    /// * `name` - A unique name for this index
    /// * `dimension` - The dimension of embeddings
    /// * `distance_metric` - The distance metric to use
    /// * `config` - The HNSW configuration parameters
    pub fn new(
        engine: E,
        name: &str,
        dimension: usize,
        distance_metric: DistanceMetric,
        config: HnswConfig,
    ) -> Result<Self, VectorError> {
        let table = table_name(name);

        // Try to load existing index
        if let Some(metadata) = load_metadata(&engine, &table)? {
            // Validate that the index matches
            if metadata.dimension != dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: metadata.dimension,
                    actual: dimension,
                });
            }
            if metadata.distance_metric != distance_metric {
                return Err(VectorError::Encoding(format!(
                    "distance metric mismatch: stored {:?}, requested {:?}",
                    metadata.distance_metric, distance_metric
                )));
            }

            // Load the graph
            let graph = load_graph(&engine, &table, &metadata)?;

            // Reconstruct config from metadata
            let config = HnswConfig {
                m: metadata.m,
                m_max0: metadata.m_max0,
                ef_construction: metadata.ef_construction,
                ef_search: metadata.ef_search,
                ml: f64::from_bits(metadata.ml_bits),
            };

            return Ok(Self {
                engine,
                table,
                graph: RwLock::new(graph),
                config: config.clone(),
                level_gen: RwLock::new(LevelGenerator::new(config.ml)),
            });
        }

        // Create new empty index
        let graph = HnswGraph::new(dimension, distance_metric);

        // Save initial metadata
        let metadata = IndexMetadata {
            dimension,
            distance_metric,
            entry_point: None,
            max_layer: 0,
            m: config.m,
            m_max0: config.m_max0,
            ef_construction: config.ef_construction,
            ef_search: config.ef_search,
            ml_bits: config.ml.to_bits(),
        };
        save_metadata(&engine, &table, &metadata)?;

        Ok(Self {
            engine,
            table,
            graph: RwLock::new(graph),
            config: config.clone(),
            level_gen: RwLock::new(LevelGenerator::new(config.ml)),
        })
    }

    /// Open an existing HNSW index from storage.
    ///
    /// Returns an error if the index does not exist.
    pub fn open(engine: E, name: &str) -> Result<Self, VectorError> {
        let table = table_name(name);

        let metadata = load_metadata(&engine, &table)?.ok_or_else(|| {
            VectorError::SpaceNotFound(format!("HNSW index '{}' not found", name))
        })?;

        let graph = load_graph(&engine, &table, &metadata)?;

        let config = HnswConfig {
            m: metadata.m,
            m_max0: metadata.m_max0,
            ef_construction: metadata.ef_construction,
            ef_search: metadata.ef_search,
            ml: f64::from_bits(metadata.ml_bits),
        };

        Ok(Self {
            engine,
            table,
            graph: RwLock::new(graph),
            config: config.clone(),
            level_gen: RwLock::new(LevelGenerator::new(config.ml)),
        })
    }

    /// Get the configuration for this index.
    #[must_use]
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the distance metric for this index.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::LockPoisoned` if the internal lock is poisoned due to
    /// a prior panic in another thread.
    pub fn distance_metric(&self) -> Result<DistanceMetric, VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(graph.distance_metric)
    }

    /// Persist all changes to storage.
    ///
    /// This is useful after batch inserts to ensure all data is saved.
    ///
    /// # Errors
    ///
    /// Returns `VectorError::LockPoisoned` if the internal lock is poisoned, or
    /// a storage error if persistence fails.
    pub fn flush(&self) -> Result<(), VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;
        save_graph(&self.engine, &self.table, &graph, &self.config)?;
        Ok(())
    }

    /// Insert a node, connecting it to the graph.
    ///
    /// This implements the HNSW insert algorithm (Algorithm 1 from the paper).
    fn insert_internal(
        &self,
        graph: &mut HnswGraph,
        entity_id: EntityId,
        embedding: &Embedding,
    ) -> Result<(), VectorError> {
        // Generate random level for this node
        let node_level =
            self.level_gen.write().map_err(|_| VectorError::LockPoisoned)?.generate_level();

        // Create the new node
        let new_node = HnswNode::new(entity_id, embedding.clone(), node_level);

        // If graph is empty, just insert as entry point
        if graph.is_empty() {
            graph.insert_node(new_node);
            // SAFETY: We just inserted this node above, so it must exist
            #[allow(clippy::unwrap_used)]
            save_node(&self.engine, &self.table, graph.get_node(entity_id).unwrap())?;
            self.update_metadata(graph)?;
            return Ok(());
        }

        // Get current entry point
        // SAFETY: We checked !graph.is_empty() above, so entry_point must exist
        #[allow(clippy::unwrap_used)]
        let entry_point = graph.entry_point.unwrap();
        let current_max_layer = graph.max_layer;

        // Search from top layer down to node_level + 1 (just to find entry point for lower layers)
        let mut current_ep = vec![entry_point];

        for layer in (node_level + 1..=current_max_layer).rev() {
            let candidates = search_layer(graph, embedding, &current_ep, 1, layer);
            current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
            if current_ep.is_empty() {
                current_ep = vec![entry_point];
            }
        }

        // Insert node first (so we can reference it during connection)
        graph.insert_node(new_node);

        // Search and connect at each layer from min(node_level, current_max_layer) down to 0
        let start_layer = node_level.min(current_max_layer);

        for layer in (0..=start_layer).rev() {
            // Search for candidates
            let candidates =
                search_layer(graph, embedding, &current_ep, self.config.ef_construction, layer);

            // Select neighbors using heuristic
            let m = if layer == 0 { self.config.m_max0 } else { self.config.m };
            let neighbors = select_neighbors_heuristic(graph, embedding, &candidates, m, false);

            // Connect the new node to its neighbors
            if let Some(node) = graph.get_node_mut(entity_id) {
                node.set_connections(layer, neighbors.clone());
            }

            // Add bidirectional connections and collect neighbors that need pruning
            let mut neighbors_to_prune = Vec::new();
            let max_conn = if layer == 0 { self.config.m_max0 } else { self.config.m };

            for &neighbor_id in &neighbors {
                if let Some(neighbor) = graph.get_node_mut(neighbor_id) {
                    neighbor.add_connection(layer, entity_id);

                    // Check if neighbor has too many connections
                    if neighbor.connections_at(layer).len() > max_conn {
                        // Collect data needed for pruning (can't do it here due to borrow)
                        let neighbor_conn_ids: Vec<EntityId> =
                            neighbor.connections_at(layer).to_vec();
                        let neighbor_embedding = neighbor.embedding.clone();
                        neighbors_to_prune.push((
                            neighbor_id,
                            neighbor_conn_ids,
                            neighbor_embedding,
                        ));
                    }
                }
            }

            // Now prune neighbors that have too many connections
            for (neighbor_id, neighbor_conn_ids, neighbor_embedding) in neighbors_to_prune {
                // Build candidates by looking up each neighbor's embedding
                let neighbor_connections: Vec<Candidate> = neighbor_conn_ids
                    .iter()
                    .filter_map(|&id| {
                        graph.get_node(id).map(|n| {
                            Candidate::new(id, graph.distance(&neighbor_embedding, &n.embedding))
                        })
                    })
                    .collect();

                // Select best neighbors using heuristic
                let pruned = select_neighbors_heuristic(
                    graph,
                    &neighbor_embedding,
                    &neighbor_connections,
                    max_conn,
                    false,
                );

                // Apply the pruned connections
                if let Some(neighbor) = graph.get_node_mut(neighbor_id) {
                    neighbor.set_connections(layer, pruned.clone());
                }

                // Persist the pruned connections
                update_connections(&self.engine, &self.table, neighbor_id, layer, &pruned)?;
            }

            // Update entry points for next layer
            current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
            if current_ep.is_empty() && !neighbors.is_empty() {
                current_ep = neighbors;
            }
        }

        // Save the new node
        // SAFETY: We inserted this node at line 271, so it must exist
        #[allow(clippy::unwrap_used)]
        save_node(&self.engine, &self.table, graph.get_node(entity_id).unwrap())?;

        // Update connections for all affected neighbors
        for layer in 0..=start_layer {
            if let Some(node) = graph.get_node(entity_id) {
                for &neighbor_id in node.connections_at(layer) {
                    if let Some(neighbor) = graph.get_node(neighbor_id) {
                        update_connections(
                            &self.engine,
                            &self.table,
                            neighbor_id,
                            layer,
                            neighbor.connections_at(layer),
                        )?;
                    }
                }
            }
        }

        // Update entry point if new node has higher layer
        if node_level > current_max_layer {
            graph.entry_point = Some(entity_id);
            graph.max_layer = node_level;
            self.update_metadata(graph)?;
        }

        Ok(())
    }

    /// Update stored metadata to match current graph state.
    fn update_metadata(&self, graph: &HnswGraph) -> Result<(), VectorError> {
        let metadata = IndexMetadata {
            dimension: graph.dimension,
            distance_metric: graph.distance_metric,
            entry_point: graph.entry_point,
            max_layer: graph.max_layer,
            m: self.config.m,
            m_max0: self.config.m_max0,
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
            ml_bits: self.config.ml.to_bits(),
        };
        persistence::save_metadata(&self.engine, &self.table, &metadata)?;
        Ok(())
    }
}

impl<E: StorageEngine> VectorIndex for HnswIndex<E> {
    fn insert(&mut self, entity_id: EntityId, embedding: &Embedding) -> Result<(), VectorError> {
        // Validate dimension
        let mut graph = self.graph.write().map_err(|_| VectorError::LockPoisoned)?;
        if embedding.dimension() != graph.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: graph.dimension,
                actual: embedding.dimension(),
            });
        }

        // If node already exists, remove it first
        if graph.contains(entity_id) {
            self.delete_internal(&mut graph, entity_id)?;
        }

        self.insert_internal(&mut graph, entity_id, embedding)
    }

    fn delete(&mut self, entity_id: EntityId) -> Result<bool, VectorError> {
        let mut graph = self.graph.write().map_err(|_| VectorError::LockPoisoned)?;
        self.delete_internal(&mut graph, entity_id)
    }

    fn search(
        &self,
        query: &Embedding,
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;

        // Validate dimension
        if query.dimension() != graph.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: graph.dimension,
                actual: query.dimension(),
            });
        }

        if graph.is_empty() {
            return Ok(Vec::new());
        }

        // ef_search defaults to configured value, but is always at least k
        let ef = ef_search.unwrap_or(self.config.ef_search).max(k);
        // SAFETY: We checked !graph.is_empty() above, so entry_point must exist
        #[allow(clippy::unwrap_used)]
        let entry_point = graph.entry_point.unwrap();

        // Search from top layer to layer 1, using ef=1 (greedy)
        let mut current_ep = vec![entry_point];

        for layer in (1..=graph.max_layer).rev() {
            let candidates = search_layer(&graph, query, &current_ep, 1, layer);
            current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
            if current_ep.is_empty() {
                current_ep = vec![entry_point];
            }
        }

        // Search layer 0 with full ef
        let candidates = search_layer(&graph, query, &current_ep, ef, 0);

        // Return top k results
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(c.entity_id, c.distance))
            .collect();

        Ok(results)
    }

    fn contains(&self, entity_id: EntityId) -> Result<bool, VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(graph.contains(entity_id))
    }

    fn len(&self) -> Result<usize, VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(graph.len())
    }

    fn dimension(&self) -> Result<usize, VectorError> {
        let graph = self.graph.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(graph.dimension)
    }
}

impl<E: StorageEngine> HnswIndex<E> {
    /// Internal delete implementation that takes a mutable graph reference.
    fn delete_internal(
        &self,
        graph: &mut HnswGraph,
        entity_id: EntityId,
    ) -> Result<bool, VectorError> {
        let node = match graph.remove_node(entity_id) {
            Some(n) => n,
            None => return Ok(false),
        };

        // Delete from storage
        delete_node(&self.engine, &self.table, entity_id, node.max_layer)?;

        // Update metadata if entry point changed
        self.update_metadata(graph)?;

        // Update connections for neighbors that pointed to this node
        // (The graph already removed these connections in remove_node)
        for layer in 0..=node.max_layer {
            for &neighbor_id in &node.connections[layer] {
                if let Some(neighbor) = graph.get_node(neighbor_id) {
                    update_connections(
                        &self.engine,
                        &self.table,
                        neighbor_id,
                        layer,
                        neighbor.connections_at(layer),
                    )?;
                }
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifoldb_storage::backends::RedbEngine;

    fn create_test_embedding(dim: usize, value: f32) -> Embedding {
        Embedding::new(vec![value; dim]).unwrap()
    }

    #[test]
    fn test_create_index() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        assert_eq!(index.dimension().unwrap(), 4);
        assert_eq!(index.len().unwrap(), 0);
        assert!(index.is_empty().unwrap());
    }

    #[test]
    fn test_insert_single() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        assert_eq!(index.len().unwrap(), 1);
        assert!(index.contains(EntityId::new(1)).unwrap());
    }

    #[test]
    fn test_insert_multiple() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        for i in 0..10 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i), &embedding).unwrap();
        }

        assert_eq!(index.len().unwrap(), 10);
    }

    #[test]
    fn test_search_empty() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let query = create_test_embedding(4, 1.0);
        let results = index.search(&query, 5, None).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_search_single() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        let query = create_test_embedding(4, 1.0);
        let results = index.search(&query, 1, None).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, EntityId::new(1));
        assert!(results[0].distance < 1e-6); // Should be very close to 0
    }

    #[test]
    fn test_search_nearest() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert embeddings at increasing distances from origin
        for i in 1..=5 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i as u64), &embedding).unwrap();
        }

        // Query close to the first embedding
        let query = create_test_embedding(4, 1.5);
        let results = index.search(&query, 3, None).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be entity 1 or 2 (closest to 1.5)
        assert!(
            results[0].entity_id == EntityId::new(1) || results[0].entity_id == EntityId::new(2)
        );
    }

    #[test]
    fn test_delete() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        assert!(index.delete(EntityId::new(1)).unwrap());
        assert!(!index.contains(EntityId::new(1)).unwrap());
        assert_eq!(index.len().unwrap(), 0);

        // Delete non-existent should return false
        assert!(!index.delete(EntityId::new(999)).unwrap());
    }

    #[test]
    fn test_update_embedding() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding1 = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding1).unwrap();

        // Insert again with different embedding (should update)
        let embedding2 = create_test_embedding(4, 10.0);
        index.insert(EntityId::new(1), &embedding2).unwrap();

        assert_eq!(index.len().unwrap(), 1);

        // Search should find the updated embedding
        let query = create_test_embedding(4, 10.0);
        let results = index.search(&query, 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(8, 1.0); // Wrong dimension
        let result = index.insert(EntityId::new(1), &embedding);

        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_dimension_mismatch_search() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &embedding).unwrap();

        let query = create_test_embedding(8, 1.0); // Wrong dimension
        let result = index.search(&query, 1, None);

        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_persistence() {
        // Use a temporary file for persistence testing
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("hnsw_persist_test_{}.redb", std::process::id()));

        // Clean up any existing file
        let _ = std::fs::remove_file(&db_path);

        // Create and populate index
        {
            let engine = RedbEngine::open(&db_path).unwrap();
            let config = HnswConfig::new(4);
            let mut index =
                HnswIndex::new(engine, "persist_test", 4, DistanceMetric::Euclidean, config)
                    .unwrap();

            for i in 0..5 {
                let embedding = create_test_embedding(4, i as f32);
                index.insert(EntityId::new(i), &embedding).unwrap();
            }

            index.flush().unwrap();
        }

        // Reopen and verify
        {
            let engine = RedbEngine::open(&db_path).unwrap();
            let index: HnswIndex<RedbEngine> = HnswIndex::open(engine, "persist_test").unwrap();

            assert_eq!(index.len().unwrap(), 5);
            assert_eq!(index.dimension().unwrap(), 4);

            for i in 0..5 {
                assert!(index.contains(EntityId::new(i)).unwrap());
            }
        }

        // Clean up
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_cosine_distance() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index = HnswIndex::new(engine, "test", 4, DistanceMetric::Cosine, config).unwrap();

        // Insert two orthogonal vectors
        let e1 = Embedding::new(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let e2 = Embedding::new(vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        let e3 = Embedding::new(vec![0.5, 0.5, 0.0, 0.0]).unwrap();

        index.insert(EntityId::new(1), &e1).unwrap();
        index.insert(EntityId::new(2), &e2).unwrap();
        index.insert(EntityId::new(3), &e3).unwrap();

        // Query with e1 direction
        let query = Embedding::new(vec![2.0, 0.0, 0.0, 0.0]).unwrap();
        let results = index.search(&query, 3, None).unwrap();

        // e1 should be closest (cosine distance = 0)
        assert_eq!(results[0].entity_id, EntityId::new(1));
    }
}
