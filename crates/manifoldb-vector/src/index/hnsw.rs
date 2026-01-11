//! HNSW index implementation.
//!
//! This is the main HNSW (Hierarchical Navigable Small World) index implementation.
//! It provides approximate nearest neighbor search with O(log N) complexity.
//!
//! ## Architecture
//!
//! The HNSW index stores the graph structure (nodes and their connections) but does NOT
//! store the actual embedding vectors. Instead, vectors are cached in memory during
//! operations and can be provided externally via a vector fetcher pattern.
//!
//! This separation allows:
//! - Vectors to be stored in a single location (CollectionVectorStore)
//! - HNSW to focus purely on the navigation graph
//! - Reduced disk I/O since vectors aren't duplicated in the HNSW index

use std::collections::HashMap;
use std::sync::RwLock;

use manifoldb_core::EntityId;
use manifoldb_storage::{StorageEngine, Transaction};

use crate::distance::DistanceMetric;
use crate::error::VectorError;
use crate::types::Embedding;

use super::config::HnswConfig;
use super::graph::{
    search_layer, search_layer_filtered, select_neighbors_heuristic, Candidate, HnswGraph, HnswNode,
};
use super::persistence::{
    self, delete_node, load_graph, load_metadata, save_graph, save_metadata, save_node, table_name,
    update_connections, IndexMetadata,
};
use super::traits::{FilteredSearchConfig, SearchResult, VectorIndex};

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
///
/// ## Vector Storage
///
/// The HNSW index maintains an in-memory cache of embeddings for all indexed entities.
/// This cache is populated when:
/// - Vectors are inserted via `insert()` or `insert_batch()`
/// - The index is loaded from storage (vectors must be provided separately)
///
/// The graph structure (nodes and connections) is persisted to storage, but the actual
/// embedding vectors are NOT stored in the HNSW tables. Instead, vectors should be
/// persisted separately (e.g., in CollectionVectorStore) and loaded into the cache.
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
    /// In-memory cache of embeddings for all indexed entities.
    /// This cache is used during search operations to compute distances.
    embeddings: RwLock<HashMap<EntityId, Embedding>>,
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
                pq_segments: metadata.pq_segments,
                pq_centroids: metadata.pq_centroids,
                pq_training_samples: 1000, // Default
            };

            // Note: When loading from storage, embeddings cache is empty.
            // Embeddings must be loaded separately (e.g., from CollectionVectorStore)
            // using load_embeddings() before the index can be used for search.
            return Ok(Self {
                engine,
                table,
                graph: RwLock::new(graph),
                config: config.clone(),
                level_gen: RwLock::new(LevelGenerator::new(config.ml)),
                embeddings: RwLock::new(HashMap::new()),
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
            pq_segments: config.pq_segments,
            pq_centroids: config.pq_centroids,
        };
        save_metadata(&engine, &table, &metadata)?;

        Ok(Self {
            engine,
            table,
            graph: RwLock::new(graph),
            config: config.clone(),
            level_gen: RwLock::new(LevelGenerator::new(config.ml)),
            embeddings: RwLock::new(HashMap::new()),
        })
    }

    /// Open an existing HNSW index from storage.
    ///
    /// Returns an error if the index does not exist.
    ///
    /// Note: After opening, the embeddings cache will be empty. You must call
    /// `load_embeddings()` or manually populate the cache before performing searches.
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
            pq_segments: metadata.pq_segments,
            pq_centroids: metadata.pq_centroids,
            pq_training_samples: 1000, // Default
        };

        Ok(Self {
            engine,
            table,
            graph: RwLock::new(graph),
            config: config.clone(),
            level_gen: RwLock::new(LevelGenerator::new(config.ml)),
            embeddings: RwLock::new(HashMap::new()),
        })
    }

    /// Load embeddings into the cache from an iterator.
    ///
    /// This should be called after `open()` to populate the embeddings cache
    /// before performing search operations. The embeddings should be fetched
    /// from the external storage (e.g., CollectionVectorStore).
    ///
    /// # Errors
    ///
    /// Returns `VectorError::LockPoisoned` if the internal lock is poisoned.
    pub fn load_embeddings<I>(&self, embeddings: I) -> Result<(), VectorError>
    where
        I: IntoIterator<Item = (EntityId, Embedding)>,
    {
        let mut cache = self.embeddings.write().map_err(|_| VectorError::LockPoisoned)?;
        for (entity_id, embedding) in embeddings {
            cache.insert(entity_id, embedding);
        }
        Ok(())
    }

    /// Get a single embedding from the cache.
    ///
    /// Returns `None` if the embedding is not in the cache.
    pub fn get_embedding(&self, entity_id: EntityId) -> Result<Option<Embedding>, VectorError> {
        let cache = self.embeddings.read().map_err(|_| VectorError::LockPoisoned)?;
        Ok(cache.get(&entity_id).cloned())
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
    ///
    /// The `embeddings_cache` parameter provides access to embeddings for all nodes.
    /// This should be a reference to the `embeddings` field.
    fn insert_internal(
        &self,
        graph: &mut HnswGraph,
        entity_id: EntityId,
        embedding: &Embedding,
        embeddings_cache: &HashMap<EntityId, Embedding>,
    ) -> Result<(), VectorError> {
        // Create a vector fetcher closure that uses the embeddings cache
        let get_vector = |id: EntityId| -> Option<Embedding> {
            // For the newly inserted node, use the provided embedding
            if id == entity_id {
                return Some(embedding.clone());
            }
            embeddings_cache.get(&id).cloned()
        };

        // Generate random level for this node
        let node_level =
            self.level_gen.write().map_err(|_| VectorError::LockPoisoned)?.generate_level();

        // Create the new node (no embedding stored in node)
        let new_node = HnswNode::new(entity_id, node_level);

        // If graph is empty, just insert as entry point
        if graph.is_empty() {
            graph.insert_node(new_node);
            save_node(
                &self.engine,
                &self.table,
                graph.get_node(entity_id).ok_or(VectorError::NodeNotFound(entity_id))?,
            )?;
            self.update_metadata(graph)?;
            return Ok(());
        }

        // Get current entry point
        let entry_point = graph
            .entry_point
            .ok_or(VectorError::InvalidGraphState("entry_point missing in non-empty graph"))?;
        let current_max_layer = graph.max_layer;

        // Search from top layer down to node_level + 1 (just to find entry point for lower layers)
        let mut current_ep = vec![entry_point];

        for layer in (node_level + 1..=current_max_layer).rev() {
            let candidates = search_layer(graph, embedding, &current_ep, 1, layer, &get_vector);
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
            let candidates = search_layer(
                graph,
                embedding,
                &current_ep,
                self.config.ef_construction,
                layer,
                &get_vector,
            );

            // Select neighbors using heuristic
            let m = if layer == 0 { self.config.m_max0 } else { self.config.m };
            let neighbors =
                select_neighbors_heuristic(graph, embedding, &candidates, m, false, &get_vector);

            // Connect the new node to its neighbors
            if let Some(node) = graph.get_node_mut(entity_id) {
                node.set_connections(layer, neighbors.clone());
            }

            // Add bidirectional connections and collect neighbors that need pruning
            let mut neighbors_to_prune: Vec<(EntityId, Vec<EntityId>)> = Vec::new();
            let max_conn = if layer == 0 { self.config.m_max0 } else { self.config.m };

            for &neighbor_id in &neighbors {
                if let Some(neighbor) = graph.get_node_mut(neighbor_id) {
                    neighbor.add_connection(layer, entity_id);

                    // Check if neighbor has too many connections
                    if neighbor.connections_at(layer).len() > max_conn {
                        // Collect data needed for pruning (can't do it here due to borrow)
                        let neighbor_conn_ids: Vec<EntityId> =
                            neighbor.connections_at(layer).to_vec();
                        neighbors_to_prune.push((neighbor_id, neighbor_conn_ids));
                    }
                }
            }

            // Now prune neighbors that have too many connections
            for (neighbor_id, neighbor_conn_ids) in neighbors_to_prune {
                // Get the neighbor's embedding from the cache
                let neighbor_embedding = match get_vector(neighbor_id) {
                    Some(emb) => emb,
                    None => continue, // Skip if we can't get the embedding
                };

                // Build candidates by looking up each neighbor's embedding
                let neighbor_connections: Vec<Candidate> = neighbor_conn_ids
                    .iter()
                    .filter_map(|&id| {
                        get_vector(id).map(|emb| {
                            Candidate::new(id, graph.distance(&neighbor_embedding, &emb))
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
                    &get_vector,
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
        save_node(
            &self.engine,
            &self.table,
            graph.get_node(entity_id).ok_or(VectorError::NodeNotFound(entity_id))?,
        )?;

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
            pq_segments: self.config.pq_segments,
            pq_centroids: self.config.pq_centroids,
        };
        persistence::save_metadata(&self.engine, &self.table, &metadata)?;
        Ok(())
    }

    /// Batch insert multiple embeddings with optimized persistence.
    ///
    /// This method inserts all nodes into the graph first, then connects them,
    /// and finally persists all changes in a single batch operation.
    ///
    /// The `embeddings_cache` parameter provides access to all embeddings including
    /// previously inserted ones.
    fn insert_batch_internal(
        &self,
        graph: &mut HnswGraph,
        embeddings: &[(EntityId, &Embedding)],
        embeddings_cache: &HashMap<EntityId, Embedding>,
    ) -> Result<(), VectorError> {
        // Create a combined lookup that includes both the cache and the new embeddings
        let new_embeddings_map: HashMap<EntityId, &Embedding> =
            embeddings.iter().map(|(id, e)| (*id, *e)).collect();

        // Create a vector fetcher closure
        let get_vector = |id: EntityId| -> Option<Embedding> {
            // First check the new embeddings being inserted
            if let Some(&emb) = new_embeddings_map.get(&id) {
                return Some(emb.clone());
            }
            // Then check the existing cache
            embeddings_cache.get(&id).cloned()
        };

        // Phase 1: Generate levels and create nodes
        let mut new_nodes: Vec<(EntityId, &Embedding, usize)> =
            Vec::with_capacity(embeddings.len());

        {
            let mut level_gen = self.level_gen.write().map_err(|_| VectorError::LockPoisoned)?;
            for (entity_id, embedding) in embeddings {
                let node_level = level_gen.generate_level();
                new_nodes.push((*entity_id, embedding, node_level));
            }
        }

        // Phase 2: Insert all nodes and connect them (in-memory)
        // Track all nodes that need connection updates persisted
        let mut affected_neighbors: std::collections::HashSet<EntityId> =
            std::collections::HashSet::new();

        for (entity_id, embedding, node_level) in &new_nodes {
            // Create the new node (no embedding stored in node)
            let new_node = HnswNode::new(*entity_id, *node_level);

            // If graph is empty, just insert as entry point
            if graph.is_empty() {
                graph.insert_node(new_node);
                continue;
            }

            // Get current entry point
            let entry_point = graph
                .entry_point
                .ok_or(VectorError::InvalidGraphState("entry_point missing in non-empty graph"))?;
            let current_max_layer = graph.max_layer;

            // Search from top layer down to node_level + 1
            let mut current_ep = vec![entry_point];

            for layer in (node_level + 1..=current_max_layer).rev() {
                let candidates = search_layer(graph, embedding, &current_ep, 1, layer, &get_vector);
                current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
                if current_ep.is_empty() {
                    current_ep = vec![entry_point];
                }
            }

            // Insert node first
            graph.insert_node(new_node);

            // Search and connect at each layer
            let start_layer = (*node_level).min(current_max_layer);

            for layer in (0..=start_layer).rev() {
                let candidates = search_layer(
                    graph,
                    embedding,
                    &current_ep,
                    self.config.ef_construction,
                    layer,
                    &get_vector,
                );

                let m = if layer == 0 { self.config.m_max0 } else { self.config.m };
                let neighbors = select_neighbors_heuristic(
                    graph,
                    embedding,
                    &candidates,
                    m,
                    false,
                    &get_vector,
                );

                // Connect the new node to its neighbors
                if let Some(node) = graph.get_node_mut(*entity_id) {
                    node.set_connections(layer, neighbors.clone());
                }

                // Add bidirectional connections and track affected neighbors
                let max_conn = if layer == 0 { self.config.m_max0 } else { self.config.m };
                let mut neighbors_to_prune: Vec<(EntityId, Vec<EntityId>)> = Vec::new();

                for &neighbor_id in &neighbors {
                    affected_neighbors.insert(neighbor_id);
                    if let Some(neighbor) = graph.get_node_mut(neighbor_id) {
                        neighbor.add_connection(layer, *entity_id);

                        if neighbor.connections_at(layer).len() > max_conn {
                            let neighbor_conn_ids: Vec<EntityId> =
                                neighbor.connections_at(layer).to_vec();
                            neighbors_to_prune.push((neighbor_id, neighbor_conn_ids));
                        }
                    }
                }

                // Prune neighbors that have too many connections
                for (neighbor_id, neighbor_conn_ids) in neighbors_to_prune {
                    let neighbor_embedding = match get_vector(neighbor_id) {
                        Some(emb) => emb,
                        None => continue,
                    };

                    let neighbor_connections: Vec<Candidate> = neighbor_conn_ids
                        .iter()
                        .filter_map(|&id| {
                            get_vector(id).map(|emb| {
                                Candidate::new(id, graph.distance(&neighbor_embedding, &emb))
                            })
                        })
                        .collect();

                    let pruned = select_neighbors_heuristic(
                        graph,
                        &neighbor_embedding,
                        &neighbor_connections,
                        max_conn,
                        false,
                        &get_vector,
                    );

                    if let Some(neighbor) = graph.get_node_mut(neighbor_id) {
                        neighbor.set_connections(layer, pruned);
                    }
                }

                current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
                if current_ep.is_empty() && !neighbors.is_empty() {
                    current_ep = neighbors;
                }
            }

            // Update entry point if new node has higher layer
            if *node_level > current_max_layer {
                graph.entry_point = Some(*entity_id);
                graph.max_layer = *node_level;
            }
        }

        // Phase 3: Batch persist all changes in a single transaction
        let mut tx = self.engine.begin_write()?;

        // Save all new nodes
        for (entity_id, _, _) in &new_nodes {
            if let Some(node) = graph.get_node(*entity_id) {
                persistence::save_node_tx(&mut tx, &self.table, node)?;
            }
        }

        // Save all affected neighbor connections
        for neighbor_id in affected_neighbors {
            if let Some(neighbor) = graph.get_node(neighbor_id) {
                // Only update connections, not the full node
                for (layer, connections) in neighbor.connections.iter().enumerate() {
                    persistence::update_connections_tx(
                        &mut tx,
                        &self.table,
                        neighbor_id,
                        layer,
                        connections,
                    )?;
                }
            }
        }

        // Update metadata
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
            pq_segments: self.config.pq_segments,
            pq_centroids: self.config.pq_centroids,
        };
        persistence::save_metadata_tx(&mut tx, &self.table, &metadata)?;

        // Commit the transaction
        tx.commit()?;

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

        // Get a snapshot of the current embeddings cache for the insert operation
        let embeddings_cache =
            self.embeddings.read().map_err(|_| VectorError::LockPoisoned)?.clone();

        // Insert the node
        self.insert_internal(&mut graph, entity_id, embedding, &embeddings_cache)?;

        // Add the embedding to the cache
        self.embeddings
            .write()
            .map_err(|_| VectorError::LockPoisoned)?
            .insert(entity_id, embedding.clone());

        Ok(())
    }

    fn insert_batch(&mut self, embeddings: &[(EntityId, &Embedding)]) -> Result<(), VectorError> {
        if embeddings.is_empty() {
            return Ok(());
        }

        let mut graph = self.graph.write().map_err(|_| VectorError::LockPoisoned)?;

        // Validate all dimensions first
        for (entity_id, embedding) in embeddings {
            if embedding.dimension() != graph.dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: graph.dimension,
                    actual: embedding.dimension(),
                });
            }

            // If node already exists, remove it first
            if graph.contains(*entity_id) {
                self.delete_internal(&mut graph, *entity_id)?;
            }
        }

        // Get a snapshot of the current embeddings cache
        let embeddings_cache =
            self.embeddings.read().map_err(|_| VectorError::LockPoisoned)?.clone();

        // Batch insert all embeddings
        self.insert_batch_internal(&mut graph, embeddings, &embeddings_cache)?;

        // Add all embeddings to the cache
        let mut cache = self.embeddings.write().map_err(|_| VectorError::LockPoisoned)?;
        for (entity_id, embedding) in embeddings {
            cache.insert(*entity_id, (*embedding).clone());
        }

        Ok(())
    }

    fn delete(&mut self, entity_id: EntityId) -> Result<bool, VectorError> {
        let mut graph = self.graph.write().map_err(|_| VectorError::LockPoisoned)?;
        let deleted = self.delete_internal(&mut graph, entity_id)?;

        // Remove from embeddings cache
        if deleted {
            self.embeddings.write().map_err(|_| VectorError::LockPoisoned)?.remove(&entity_id);
        }

        Ok(deleted)
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

        // Create vector fetcher from the embeddings cache
        let embeddings_cache = self.embeddings.read().map_err(|_| VectorError::LockPoisoned)?;
        let get_vector = |id: EntityId| -> Option<Embedding> { embeddings_cache.get(&id).cloned() };

        // ef_search defaults to configured value, but is always at least k
        let ef = ef_search.unwrap_or(self.config.ef_search).max(k);
        let entry_point = graph
            .entry_point
            .ok_or(VectorError::InvalidGraphState("entry_point missing in non-empty graph"))?;

        // Search from top layer to layer 1, using ef=1 (greedy)
        let mut current_ep = vec![entry_point];

        for layer in (1..=graph.max_layer).rev() {
            let candidates = search_layer(&graph, query, &current_ep, 1, layer, &get_vector);
            current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
            if current_ep.is_empty() {
                current_ep = vec![entry_point];
            }
        }

        // Search layer 0 with full ef
        let candidates = search_layer(&graph, query, &current_ep, ef, 0, &get_vector);

        // Return top k results
        let results: Vec<SearchResult> = candidates
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(c.entity_id, c.distance))
            .collect();

        Ok(results)
    }

    fn search_with_filter<F>(
        &self,
        query: &Embedding,
        k: usize,
        predicate: F,
        ef_search: Option<usize>,
        config: Option<FilteredSearchConfig>,
    ) -> Result<Vec<SearchResult>, VectorError>
    where
        F: Fn(EntityId) -> bool,
    {
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

        // Create vector fetcher from the embeddings cache
        let embeddings_cache = self.embeddings.read().map_err(|_| VectorError::LockPoisoned)?;
        let get_vector = |id: EntityId| -> Option<Embedding> { embeddings_cache.get(&id).cloned() };

        // Get filtered search config
        let filter_config = config.unwrap_or_default();

        // Calculate adjusted ef_search for filtering
        // When filtering, we need to explore more nodes to find k matching results
        let base_ef = ef_search.unwrap_or(self.config.ef_search).max(k);
        let ef = filter_config.adjusted_ef(base_ef, None);

        let entry_point = graph
            .entry_point
            .ok_or(VectorError::InvalidGraphState("entry_point missing in non-empty graph"))?;

        // Search from top layer to layer 1, using ef=1 (greedy)
        // Note: We don't filter in upper layers - just find a good entry point
        let mut current_ep = vec![entry_point];

        for layer in (1..=graph.max_layer).rev() {
            let candidates = search_layer(&graph, query, &current_ep, 1, layer, &get_vector);
            current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
            if current_ep.is_empty() {
                current_ep = vec![entry_point];
            }
        }

        // Search layer 0 with filter applied during traversal
        let candidates =
            search_layer_filtered(&graph, query, &current_ep, ef, 0, &get_vector, predicate);

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

    #[test]
    fn test_search_with_filter() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert embeddings at increasing distances from origin
        for i in 1..=10 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i as u64), &embedding).unwrap();
        }

        // Query close to the first embedding
        let query = create_test_embedding(4, 1.5);

        // Filter to only include even entity IDs
        let predicate = |id: EntityId| id.as_u64() % 2 == 0;

        let results = index.search_with_filter(&query, 3, predicate, None, None).unwrap();

        // Should only return even IDs
        assert!(!results.is_empty());
        for result in &results {
            assert_eq!(result.entity_id.as_u64() % 2, 0);
        }

        // First result should be entity 2 (closest even to 1.5)
        assert_eq!(results[0].entity_id, EntityId::new(2));
    }

    #[test]
    fn test_search_with_filter_empty_match() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert some embeddings
        for i in 1..=5 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i as u64), &embedding).unwrap();
        }

        let query = create_test_embedding(4, 1.0);

        // Filter that matches nothing
        let predicate = |_id: EntityId| false;

        let results = index.search_with_filter(&query, 3, predicate, None, None).unwrap();

        // Should return empty results
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_filter_all_match() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert some embeddings
        for i in 1..=5 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i as u64), &embedding).unwrap();
        }

        let query = create_test_embedding(4, 1.0);

        // Filter that matches everything
        let predicate = |_id: EntityId| true;

        let results = index.search_with_filter(&query, 3, predicate, None, None).unwrap();

        // Should return 3 results
        assert_eq!(results.len(), 3);
        // Results should be the same as regular search
        let regular_results = index.search(&query, 3, None).unwrap();
        assert_eq!(results[0].entity_id, regular_results[0].entity_id);
    }

    #[test]
    fn test_filtered_search_config() {
        let config = FilteredSearchConfig::new()
            .with_min_ef_search(50)
            .with_max_ef_search(1000)
            .with_ef_multiplier(3.0);

        assert_eq!(config.min_ef_search, 50);
        assert_eq!(config.max_ef_search, 1000);
        assert_eq!(config.ef_multiplier, 3.0);

        // Test ef adjustment
        let adjusted = config.adjusted_ef(100, None);
        assert_eq!(adjusted, 300); // 100 * 3.0 = 300

        // With selectivity 0.5, multiplier should be 2.0
        let adjusted_selective = config.adjusted_ef(100, Some(0.5));
        assert_eq!(adjusted_selective, 200);

        // With selectivity 0.1, multiplier should be 10.0
        let adjusted_very_selective = config.adjusted_ef(100, Some(0.1));
        assert_eq!(adjusted_very_selective, 1000); // Clamped to max

        // Test clamping
        let adjusted_min = config.adjusted_ef(10, None);
        assert_eq!(adjusted_min, 50); // Clamped to min
    }

    // ========================================================================
    // Batch Insert Tests
    // ========================================================================

    #[test]
    fn test_insert_batch_empty() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Empty batch should succeed
        index.insert_batch(&[]).unwrap();
        assert_eq!(index.len().unwrap(), 0);
    }

    #[test]
    fn test_insert_batch_single() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let embedding = create_test_embedding(4, 1.0);
        index.insert_batch(&[(EntityId::new(1), &embedding)]).unwrap();

        assert_eq!(index.len().unwrap(), 1);
        assert!(index.contains(EntityId::new(1)).unwrap());
    }

    #[test]
    fn test_insert_batch_multiple() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Create embeddings
        let embeddings: Vec<Embedding> =
            (0..10).map(|i| create_test_embedding(4, i as f32)).collect();

        let batch: Vec<(EntityId, &Embedding)> =
            embeddings.iter().enumerate().map(|(i, e)| (EntityId::new(i as u64), e)).collect();

        index.insert_batch(&batch).unwrap();

        assert_eq!(index.len().unwrap(), 10);
        for i in 0..10 {
            assert!(index.contains(EntityId::new(i)).unwrap());
        }
    }

    #[test]
    fn test_insert_batch_large() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(16);
        let mut index =
            HnswIndex::new(engine, "test", 128, DistanceMetric::Euclidean, config).unwrap();

        // Insert 500 vectors in batch
        let embeddings: Vec<Embedding> =
            (0..500).map(|i| Embedding::new(vec![i as f32 / 500.0; 128]).unwrap()).collect();

        let batch: Vec<(EntityId, &Embedding)> =
            embeddings.iter().enumerate().map(|(i, e)| (EntityId::new(i as u64), e)).collect();

        index.insert_batch(&batch).unwrap();

        assert_eq!(index.len().unwrap(), 500);

        // Verify search still works
        let query = Embedding::new(vec![0.5; 128]).unwrap();
        let results = index.search(&query, 10, None).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_insert_batch_dimension_mismatch() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let good_embedding = create_test_embedding(4, 1.0);
        let bad_embedding = create_test_embedding(8, 2.0); // Wrong dimension

        let batch: Vec<(EntityId, &Embedding)> =
            vec![(EntityId::new(1), &good_embedding), (EntityId::new(2), &bad_embedding)];

        let result = index.insert_batch(&batch);
        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_insert_batch_updates_existing() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert initial embedding
        let original = create_test_embedding(4, 1.0);
        index.insert(EntityId::new(1), &original).unwrap();

        // Batch update with different embedding
        let updated = create_test_embedding(4, 10.0);
        index.insert_batch(&[(EntityId::new(1), &updated)]).unwrap();

        // Should still have 1 entry
        assert_eq!(index.len().unwrap(), 1);

        // Search should find the updated embedding
        let query = create_test_embedding(4, 10.0);
        let results = index.search(&query, 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_insert_batch_search_quality() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(16);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert embeddings at increasing values
        let embeddings: Vec<Embedding> =
            (1..=20).map(|i| create_test_embedding(4, i as f32)).collect();

        let batch: Vec<(EntityId, &Embedding)> = embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| (EntityId::new((i + 1) as u64), e))
            .collect();

        index.insert_batch(&batch).unwrap();

        // Query close to embedding 10
        let query = create_test_embedding(4, 10.5);
        let results = index.search(&query, 5, None).unwrap();

        // Top results should be entities 10 and 11 (values 10.0 and 11.0)
        let top_ids: Vec<u64> = results.iter().map(|r| r.entity_id.as_u64()).collect();
        assert!(top_ids.contains(&10) || top_ids.contains(&11));
    }

    #[test]
    fn test_insert_batch_persistence() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join(format!("hnsw_batch_persist_test_{}.redb", std::process::id()));
        let _ = std::fs::remove_file(&db_path);

        // Create and batch insert
        {
            let engine = RedbEngine::open(&db_path).unwrap();
            let config = HnswConfig::new(4);
            let mut index =
                HnswIndex::new(engine, "batch_test", 4, DistanceMetric::Euclidean, config).unwrap();

            let embeddings: Vec<Embedding> =
                (0..50).map(|i| create_test_embedding(4, i as f32)).collect();

            let batch: Vec<(EntityId, &Embedding)> =
                embeddings.iter().enumerate().map(|(i, e)| (EntityId::new(i as u64), e)).collect();

            index.insert_batch(&batch).unwrap();
            index.flush().unwrap();
        }

        // Reopen and verify
        {
            let engine = RedbEngine::open(&db_path).unwrap();
            let index: HnswIndex<RedbEngine> = HnswIndex::open(engine, "batch_test").unwrap();

            assert_eq!(index.len().unwrap(), 50);
            for i in 0..50 {
                assert!(index.contains(EntityId::new(i)).unwrap());
            }

            // Load embeddings (in real usage, these would come from external storage)
            // Here we regenerate them since the test uses a deterministic function
            let embeddings: Vec<(EntityId, Embedding)> = (0..50)
                .map(|i| (EntityId::new(i as u64), create_test_embedding(4, i as f32)))
                .collect();
            index.load_embeddings(embeddings).unwrap();

            // Verify search works
            let query = create_test_embedding(4, 25.0);
            let results = index.search(&query, 5, None).unwrap();
            assert_eq!(results.len(), 5);
        }

        let _ = std::fs::remove_file(&db_path);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_error_node_not_found_display() {
        let error = VectorError::NodeNotFound(EntityId::new(42));
        let msg = error.to_string();
        assert!(msg.contains("42"), "Error should contain entity ID");
        assert!(msg.contains("node not found"), "Error should describe issue");
    }

    #[test]
    fn test_error_invalid_graph_state_display() {
        let error = VectorError::InvalidGraphState("entry_point missing in non-empty graph");
        let msg = error.to_string();
        assert!(msg.contains("entry_point"), "Error should contain context");
        assert!(msg.contains("invalid graph state"), "Error should describe issue");
    }

    #[test]
    fn test_delete_nonexistent_returns_false() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Try to delete a node that was never inserted
        let result = index.delete(EntityId::new(999));
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_contains_nonexistent_returns_false() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Check for a node that was never inserted
        assert!(!index.contains(EntityId::new(999)).unwrap());
    }

    #[test]
    fn test_search_after_all_deleted() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::new(4);
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Insert some nodes
        for i in 0..5 {
            let embedding = create_test_embedding(4, i as f32);
            index.insert(EntityId::new(i), &embedding).unwrap();
        }

        // Delete all nodes
        for i in 0..5 {
            assert!(index.delete(EntityId::new(i)).unwrap());
        }

        // Search should return empty results
        let query = create_test_embedding(4, 1.0);
        let results = index.search(&query, 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_on_empty_index() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let index = HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        let query = create_test_embedding(4, 1.0);
        let predicate = |_id: EntityId| true;

        let results = index.search_with_filter(&query, 5, predicate, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_insert_empty_vec() {
        let engine = RedbEngine::in_memory().unwrap();
        let config = HnswConfig::default();
        let mut index =
            HnswIndex::new(engine, "test", 4, DistanceMetric::Euclidean, config).unwrap();

        // Empty batch should be a no-op
        let empty: Vec<(EntityId, &Embedding)> = vec![];
        index.insert_batch(&empty).unwrap();
        assert_eq!(index.len().unwrap(), 0);
    }
}
