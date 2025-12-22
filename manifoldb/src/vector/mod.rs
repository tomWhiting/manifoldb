//! Vector index management for HNSW indexes.
//!
//! This module provides functionality for:
//! - Building HNSW indexes from existing entity data
//! - Maintaining indexes during DML operations (INSERT/UPDATE/DELETE)
//! - Loading and persisting indexes to storage
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::vector::{HnswIndexBuilder, IndexManager};
//!
//! // Create an HNSW index on a vector column
//! let mut tx = db.begin()?;
//! HnswIndexBuilder::new("embeddings_idx", "documents", "embedding")
//!     .dimension(384)
//!     .distance_metric(DistanceMetric::Cosine)
//!     .build(&mut tx)?;
//! tx.commit()?;
//! ```

use manifoldb_core::{Entity, EntityId, TransactionError, Value};
use manifoldb_storage::Transaction;
use manifoldb_vector::distance::DistanceMetric;
use manifoldb_vector::index::{
    clear_index_tx, hnsw_table_name, load_graph_tx, load_metadata_tx, save_graph_tx,
    search_layer_filtered, FilteredSearchConfig, HnswConfig, HnswGraph, HnswIndexEntry, HnswNode,
    HnswRegistry, SearchResult,
};
use manifoldb_vector::types::Embedding;

use crate::transaction::DatabaseTransaction;

/// Error type for vector index operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorIndexError {
    /// The index already exists.
    #[error("Index already exists: {0}")]
    IndexExists(String),

    /// The index was not found.
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    /// The table was not found.
    #[error("Table not found: {0}")]
    TableNotFound(String),

    /// The column was not found.
    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    /// Dimension mismatch between vectors.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Invalid vector data.
    #[error("Invalid vector in entity {entity_id}: {reason}")]
    InvalidVector {
        /// Entity ID with invalid vector.
        entity_id: EntityId,
        /// Reason for the error.
        reason: String,
    },

    /// Transaction error.
    #[error("Transaction error: {0}")]
    Transaction(#[from] TransactionError),

    /// Vector operation error.
    #[error("Vector error: {0}")]
    Vector(#[from] manifoldb_vector::error::VectorError),

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(String),
}

/// Builder for creating HNSW indexes.
pub struct HnswIndexBuilder {
    name: String,
    table: String,
    column: String,
    dimension: Option<usize>,
    distance_metric: DistanceMetric,
    config: HnswConfig,
}

impl HnswIndexBuilder {
    /// Create a new HNSW index builder.
    ///
    /// # Arguments
    ///
    /// * `name` - The unique name for this index
    /// * `table` - The table (entity label) this index is on
    /// * `column` - The column (property name) containing vector data
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        table: impl Into<String>,
        column: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            table: table.into(),
            column: column.into(),
            dimension: None,
            distance_metric: DistanceMetric::Cosine,
            config: HnswConfig::default(),
        }
    }

    /// Set the dimension of vectors in this index.
    ///
    /// If not set, the dimension will be inferred from the first vector found.
    #[must_use]
    pub const fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Set the distance metric.
    #[must_use]
    pub const fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set the M parameter (max connections per node).
    #[must_use]
    pub fn m(mut self, m: usize) -> Self {
        self.config.m = m;
        self.config.m_max0 = 2 * m;
        self
    }

    /// Set the ef_construction parameter.
    #[must_use]
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.config.ef_construction = ef;
        self
    }

    /// Set the ef_search parameter.
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.config.ef_search = ef;
        self
    }

    /// Build the index, inserting all existing data.
    ///
    /// This scans all entities with the specified label and inserts their
    /// vector values into the HNSW index.
    pub fn build<T: Transaction>(
        self,
        tx: &mut DatabaseTransaction<T>,
    ) -> Result<(), VectorIndexError> {
        // Check if index already exists
        let storage = tx.storage_ref()?;
        if HnswRegistry::exists(storage, &self.name)? {
            return Err(VectorIndexError::IndexExists(self.name));
        }

        // Get all entities with this label
        let entities = tx.iter_entities(Some(&self.table))?;

        // Collect vectors and determine dimension
        let mut vectors: Vec<(EntityId, Embedding)> = Vec::new();
        let mut inferred_dimension: Option<usize> = self.dimension;

        for entity in &entities {
            if let Some(embedding) = extract_embedding(entity, &self.column)? {
                // Check/infer dimension
                match inferred_dimension {
                    None => inferred_dimension = Some(embedding.len()),
                    Some(dim) if dim != embedding.len() => {
                        return Err(VectorIndexError::DimensionMismatch {
                            expected: dim,
                            actual: embedding.len(),
                        });
                    }
                    _ => {}
                }
                vectors.push((entity.id, embedding));
            }
        }

        // Determine final dimension (use 1 if no vectors found; will be validated on first insert)
        let dimension = inferred_dimension.unwrap_or(1);

        // Build the HNSW graph in memory
        let mut graph = HnswGraph::new(dimension, self.distance_metric);

        // Insert all vectors using the HNSW algorithm
        for (entity_id, embedding) in vectors {
            insert_into_graph(&mut graph, entity_id, embedding, &self.config)?;
        }

        // Save to storage
        let table_name = hnsw_table_name(&self.name);
        let storage = tx.storage_mut_ref()?;
        save_graph_tx(storage, &table_name, &graph, &self.config)?;

        // Register the index
        let entry = HnswIndexEntry::new(
            &self.name,
            &self.table,
            &self.column,
            dimension,
            self.distance_metric,
            &self.config,
        );
        HnswRegistry::register(storage, &entry)?;

        Ok(())
    }
}

/// Drop an HNSW index.
pub fn drop_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    name: &str,
    if_exists: bool,
) -> Result<bool, VectorIndexError> {
    let storage = tx.storage_ref()?;

    // Check if index exists
    if !HnswRegistry::exists(storage, name)? {
        if if_exists {
            return Ok(false);
        }
        return Err(VectorIndexError::IndexNotFound(name.to_string()));
    }

    let table_name = hnsw_table_name(name);

    // Clear the index data
    let storage = tx.storage_mut_ref()?;
    clear_index_tx(storage, &table_name)?;

    // Remove from registry
    HnswRegistry::drop(storage, name)?;

    Ok(true)
}

/// Load an HNSW index graph from storage.
pub fn load_index<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    name: &str,
) -> Result<(HnswGraph, HnswConfig), VectorIndexError> {
    let storage = tx.storage_ref()?;

    // Get the registry entry
    let entry = HnswRegistry::get(storage, name)?
        .ok_or_else(|| VectorIndexError::IndexNotFound(name.to_string()))?;

    let table_name = hnsw_table_name(name);

    // Load metadata
    let metadata = load_metadata_tx(storage, &table_name)?
        .ok_or_else(|| VectorIndexError::IndexNotFound(name.to_string()))?;

    // Load the graph
    let graph = load_graph_tx(storage, &table_name, &metadata)?;

    Ok((graph, entry.config()))
}

/// Find an HNSW index for a given table and column.
pub fn find_index_for_column<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    table: &str,
    column: &str,
) -> Result<Option<String>, VectorIndexError> {
    let storage = tx.storage_ref()?;

    // List all indexes for this table
    let indexes = HnswRegistry::list_for_table(storage, table)?;

    // Find one that matches the column
    for entry in indexes {
        if entry.column == column {
            return Ok(Some(entry.name.clone()));
        }
    }

    Ok(None)
}

/// Perform an HNSW search without filtering.
pub fn search_index<T: Transaction>(
    tx: &DatabaseTransaction<T>,
    index_name: &str,
    query: &Embedding,
    k: usize,
    ef_search: Option<usize>,
) -> Result<Vec<SearchResult>, VectorIndexError> {
    let (graph, config) = load_index(tx, index_name)?;

    if graph.dimension != query.len() {
        return Err(VectorIndexError::DimensionMismatch {
            expected: graph.dimension,
            actual: query.len(),
        });
    }

    if graph.nodes.is_empty() {
        return Ok(Vec::new());
    }

    let entry_point =
        graph.entry_point.ok_or_else(|| VectorIndexError::Storage("no entry point".to_string()))?;

    // ef_search defaults to configured value, but is always at least k
    let ef = ef_search.unwrap_or(config.ef_search).max(k);

    // Search from top layer to layer 1, using ef=1 (greedy)
    let mut current_ep = vec![entry_point];

    for layer in (1..=graph.max_layer).rev() {
        let candidates =
            manifoldb_vector::index::search_layer(&graph, query, &current_ep, 1, layer);
        current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
        if current_ep.is_empty() {
            current_ep = vec![entry_point];
        }
    }

    // Search layer 0 with full ef
    let candidates = manifoldb_vector::index::search_layer(&graph, query, &current_ep, ef, 0);

    // Return top k results
    let results: Vec<SearchResult> = candidates
        .into_iter()
        .take(k)
        .map(|c| SearchResult::new(c.entity_id, c.distance))
        .collect();

    Ok(results)
}

/// Perform a filtered HNSW search.
///
/// This applies the filter during graph traversal, which is more efficient
/// than post-filtering when the filter is selective.
pub fn search_index_filtered<T, F>(
    tx: &DatabaseTransaction<T>,
    index_name: &str,
    query: &Embedding,
    k: usize,
    predicate: F,
    ef_search: Option<usize>,
    filter_config: Option<FilteredSearchConfig>,
) -> Result<Vec<SearchResult>, VectorIndexError>
where
    T: Transaction,
    F: Fn(EntityId) -> bool,
{
    let (graph, config) = load_index(tx, index_name)?;

    if graph.dimension != query.len() {
        return Err(VectorIndexError::DimensionMismatch {
            expected: graph.dimension,
            actual: query.len(),
        });
    }

    if graph.nodes.is_empty() {
        return Ok(Vec::new());
    }

    let entry_point =
        graph.entry_point.ok_or_else(|| VectorIndexError::Storage("no entry point".to_string()))?;

    // Get filtered search config
    let fc = filter_config.unwrap_or_default();

    // Calculate adjusted ef_search for filtering
    let base_ef = ef_search.unwrap_or(config.ef_search).max(k);
    let ef = fc.adjusted_ef(base_ef, None);

    // Search from top layer to layer 1, using ef=1 (greedy)
    // Note: We don't filter in upper layers - just find a good entry point
    let mut current_ep = vec![entry_point];

    for layer in (1..=graph.max_layer).rev() {
        let candidates =
            manifoldb_vector::index::search_layer(&graph, query, &current_ep, 1, layer);
        current_ep = candidates.into_iter().map(|c| c.entity_id).collect();
        if current_ep.is_empty() {
            current_ep = vec![entry_point];
        }
    }

    // Search layer 0 with filter applied during traversal
    let candidates = search_layer_filtered(&graph, query, &current_ep, ef, 0, &predicate);

    // Return top k results
    let results: Vec<SearchResult> = candidates
        .into_iter()
        .take(k)
        .map(|c| SearchResult::new(c.entity_id, c.distance))
        .collect();

    Ok(results)
}

/// Update an entity in the HNSW index.
///
/// If the entity has a vector property that matches an indexed column,
/// the index is updated accordingly.
pub fn update_entity_in_indexes<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    entity: &Entity,
    old_entity: Option<&Entity>,
) -> Result<(), VectorIndexError> {
    // Collect index names and columns first to avoid borrowing issues
    let mut index_operations: Vec<(String, Option<Embedding>, Option<Embedding>)> = Vec::new();

    {
        let storage = tx.storage_ref()?;

        // Find all indexes for this entity's labels
        for label in &entity.labels {
            let indexes = HnswRegistry::list_for_table(storage, label.as_str())?;

            for entry in indexes {
                let old_embedding =
                    old_entity.and_then(|e| extract_embedding(e, &entry.column).ok().flatten());
                let new_embedding = extract_embedding(entity, &entry.column)?;
                index_operations.push((entry.name.clone(), old_embedding, new_embedding));
            }
        }
    }

    // Now perform the operations
    for (index_name, old_embedding, new_embedding) in index_operations {
        match (old_embedding, new_embedding) {
            (None, Some(embedding)) => {
                // Insert: new vector where none existed
                add_to_index(tx, &index_name, entity.id, embedding)?;
            }
            (Some(_), None) => {
                // Delete: vector removed
                remove_from_index(tx, &index_name, entity.id)?;
            }
            (Some(old), Some(new)) if old.as_slice() != new.as_slice() => {
                // Update: vector changed
                remove_from_index(tx, &index_name, entity.id)?;
                add_to_index(tx, &index_name, entity.id, new)?;
            }
            _ => {
                // No change
            }
        }
    }

    Ok(())
}

/// Remove an entity from all HNSW indexes.
pub fn remove_entity_from_indexes<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    entity: &Entity,
) -> Result<(), VectorIndexError> {
    // Collect index names first to avoid borrowing issues
    let mut indexes_to_update: Vec<String> = Vec::new();

    {
        let storage = tx.storage_ref()?;

        // Find all indexes for this entity's labels
        for label in &entity.labels {
            let indexes = HnswRegistry::list_for_table(storage, label.as_str())?;

            for entry in indexes {
                if extract_embedding(entity, &entry.column)?.is_some() {
                    indexes_to_update.push(entry.name.clone());
                }
            }
        }
    }

    // Now perform the removals
    for index_name in indexes_to_update {
        remove_from_index(tx, &index_name, entity.id)?;
    }

    Ok(())
}

/// Add an entity to an HNSW index.
fn add_to_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    index_name: &str,
    entity_id: EntityId,
    embedding: Embedding,
) -> Result<(), VectorIndexError> {
    let (mut graph, config) = load_index(tx, index_name)?;

    // Check dimension
    if graph.dimension != embedding.len() && !graph.nodes.is_empty() {
        return Err(VectorIndexError::DimensionMismatch {
            expected: graph.dimension,
            actual: embedding.len(),
        });
    }

    // Update dimension if this is the first node
    if graph.nodes.is_empty() {
        graph.dimension = embedding.len();
    }

    // Insert into graph
    insert_into_graph(&mut graph, entity_id, embedding, &config)?;

    // Save updated graph
    let table_name = hnsw_table_name(index_name);
    let storage = tx.storage_mut_ref()?;
    save_graph_tx(storage, &table_name, &graph, &config)?;

    Ok(())
}

/// Remove an entity from an HNSW index.
fn remove_from_index<T: Transaction>(
    tx: &mut DatabaseTransaction<T>,
    index_name: &str,
    entity_id: EntityId,
) -> Result<(), VectorIndexError> {
    let (mut graph, config) = load_index(tx, index_name)?;

    // Remove the node
    remove_from_graph(&mut graph, entity_id)?;

    // Save updated graph
    let table_name = hnsw_table_name(index_name);
    let storage = tx.storage_mut_ref()?;
    save_graph_tx(storage, &table_name, &graph, &config)?;

    Ok(())
}

/// Extract an embedding from an entity's property.
fn extract_embedding(entity: &Entity, column: &str) -> Result<Option<Embedding>, VectorIndexError> {
    match entity.get_property(column) {
        Some(Value::Vector(v)) => {
            let embedding = Embedding::new(v.clone()).map_err(|e| {
                VectorIndexError::InvalidVector { entity_id: entity.id, reason: e.to_string() }
            })?;
            Ok(Some(embedding))
        }
        Some(_) => Ok(None), // Non-vector property
        None => Ok(None),    // No such property
    }
}

/// Insert a vector into an HNSW graph.
fn insert_into_graph(
    graph: &mut HnswGraph,
    entity_id: EntityId,
    embedding: Embedding,
    config: &HnswConfig,
) -> Result<(), VectorIndexError> {
    // Generate random level for this node
    let level = generate_level(config.ml, config.m);

    // Create the new node
    let node = HnswNode::new(entity_id, embedding.clone(), level);

    if graph.nodes.is_empty() {
        // First node becomes entry point
        graph.entry_point = Some(entity_id);
        graph.max_layer = level;
        graph.nodes.insert(entity_id, node);
        return Ok(());
    }

    // Find entry point
    let entry_point =
        graph.entry_point.ok_or_else(|| VectorIndexError::Storage("no entry point".to_string()))?;

    // Search for insertion point from top to level+1
    let mut current = entry_point;
    for layer in (level + 1..=graph.max_layer).rev() {
        current = search_layer_greedy(graph, &embedding, current, layer)?;
    }

    // For each layer from level down to 0
    for layer in (0..=level.min(graph.max_layer)).rev() {
        // Find neighbors at this layer
        let neighbors =
            search_layer_candidates(graph, &embedding, current, layer, config.ef_construction)?;

        // Select best neighbors
        let m = if layer == 0 { config.m_max0 } else { config.m };
        let selected: Vec<EntityId> = neighbors.into_iter().take(m).collect();

        // Add connections
        graph.nodes.entry(entity_id).or_insert_with(|| node.clone()).connections[layer] =
            selected.clone();

        // Add reverse connections
        for &neighbor_id in &selected {
            if let Some(neighbor) = graph.nodes.get_mut(&neighbor_id) {
                if layer < neighbor.connections.len() {
                    let neighbor_m = if layer == 0 { config.m_max0 } else { config.m };
                    if !neighbor.connections[layer].contains(&entity_id) {
                        neighbor.connections[layer].push(entity_id);
                        // Prune if necessary
                        if neighbor.connections[layer].len() > neighbor_m {
                            prune_connections(graph, neighbor_id, layer, neighbor_m)?;
                        }
                    }
                }
            }
        }

        // Update current for next layer
        if !selected.is_empty() {
            current = selected[0];
        }
    }

    // Insert the node
    graph.nodes.insert(entity_id, node);

    // Update entry point if needed
    if level > graph.max_layer {
        graph.max_layer = level;
        graph.entry_point = Some(entity_id);
    }

    Ok(())
}

/// Remove a vector from an HNSW graph.
fn remove_from_graph(graph: &mut HnswGraph, entity_id: EntityId) -> Result<(), VectorIndexError> {
    // Get the node's max_layer before removing
    let max_layer = match graph.nodes.get(&entity_id) {
        Some(node) => node.max_layer,
        None => return Ok(()), // Node doesn't exist, nothing to do
    };

    // Remove connections from neighbors
    for layer in 0..=max_layer {
        if let Some(node) = graph.nodes.get(&entity_id) {
            let neighbors = node.connections.get(layer).cloned().unwrap_or_default();
            for neighbor_id in neighbors {
                if let Some(neighbor) = graph.nodes.get_mut(&neighbor_id) {
                    if layer < neighbor.connections.len() {
                        neighbor.connections[layer].retain(|&id| id != entity_id);
                    }
                }
            }
        }
    }

    // Remove the node
    graph.nodes.remove(&entity_id);

    // Update entry point if necessary
    if graph.entry_point == Some(entity_id) {
        // Find a new entry point - prefer highest layer node
        graph.entry_point = graph.nodes.iter().max_by_key(|(_, n)| n.max_layer).map(|(&id, _)| id);

        // Update max_layer
        graph.max_layer = graph.nodes.values().map(|n| n.max_layer).max().unwrap_or(0);
    }

    Ok(())
}

/// Generate a random level for a new node using the standard HNSW formula.
fn generate_level(ml: f64, _m: usize) -> usize {
    use std::hash::{Hash, Hasher};

    // Use a simple hash-based randomization for deterministic testing
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    let random = hasher.finish();

    let r = (random as f64) / (u64::MAX as f64);
    (-r.ln() * ml).floor() as usize
}

/// Search layer greedily, returning the closest node to the query.
fn search_layer_greedy(
    graph: &HnswGraph,
    query: &Embedding,
    entry: EntityId,
    layer: usize,
) -> Result<EntityId, VectorIndexError> {
    let mut current = entry;
    let mut current_dist = compute_distance(graph, query, current)?;

    loop {
        let mut changed = false;

        if let Some(node) = graph.nodes.get(&current) {
            if layer < node.connections.len() {
                for &neighbor_id in &node.connections[layer] {
                    let dist = compute_distance(graph, query, neighbor_id)?;
                    if dist < current_dist {
                        current = neighbor_id;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    Ok(current)
}

/// Search layer for candidates, returning the closest nodes.
fn search_layer_candidates(
    graph: &HnswGraph,
    query: &Embedding,
    entry: EntityId,
    layer: usize,
    ef: usize,
) -> Result<Vec<EntityId>, VectorIndexError> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashSet};

    #[derive(PartialEq)]
    struct Candidate {
        distance: f32,
        id: EntityId,
    }

    impl Eq for Candidate {}

    impl PartialOrd for Candidate {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Candidate {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.distance.partial_cmp(&other.distance).unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let entry_dist = compute_distance(graph, query, entry)?;

    let mut visited = HashSet::new();
    visited.insert(entry);

    // Min-heap for candidates to explore
    let mut candidates = BinaryHeap::new();
    candidates.push(Reverse(Candidate { distance: entry_dist, id: entry }));

    // Max-heap for results (we want to keep the closest ef nodes)
    let mut results = BinaryHeap::new();
    results.push(Candidate { distance: entry_dist, id: entry });

    while let Some(Reverse(Candidate { distance: current_dist, id: current })) = candidates.pop() {
        // Get worst distance in results
        let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

        if current_dist > worst_dist && results.len() >= ef {
            break;
        }

        if let Some(node) = graph.nodes.get(&current) {
            if layer < node.connections.len() {
                for &neighbor_id in &node.connections[layer] {
                    if visited.insert(neighbor_id) {
                        let dist = compute_distance(graph, query, neighbor_id)?;

                        if results.len() < ef || dist < worst_dist {
                            candidates.push(Reverse(Candidate { distance: dist, id: neighbor_id }));
                            results.push(Candidate { distance: dist, id: neighbor_id });

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert results to sorted vector
    let mut result_vec: Vec<_> = results.into_iter().map(|c| (c.distance, c.id)).collect();
    result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(result_vec.into_iter().map(|(_, id)| id).collect())
}

/// Prune connections at a layer to maintain the maximum connection limit.
fn prune_connections(
    graph: &mut HnswGraph,
    node_id: EntityId,
    layer: usize,
    max_connections: usize,
) -> Result<(), VectorIndexError> {
    let node = match graph.nodes.get(&node_id) {
        Some(n) => n,
        None => return Ok(()),
    };

    if layer >= node.connections.len() {
        return Ok(());
    }

    let embedding = node.embedding.clone();
    let connections = node.connections[layer].clone();

    // Sort by distance and keep closest
    let mut distances: Vec<(f32, EntityId)> = Vec::new();
    for &neighbor_id in &connections {
        if let Some(neighbor) = graph.nodes.get(&neighbor_id) {
            let dist =
                compute_distance_embeddings(&embedding, &neighbor.embedding, graph.distance_metric);
            distances.push((dist, neighbor_id));
        }
    }

    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let new_connections: Vec<EntityId> =
        distances.into_iter().take(max_connections).map(|(_, id)| id).collect();

    if let Some(node) = graph.nodes.get_mut(&node_id) {
        if layer < node.connections.len() {
            node.connections[layer] = new_connections;
        }
    }

    Ok(())
}

/// Compute distance between a query and a node.
fn compute_distance(
    graph: &HnswGraph,
    query: &Embedding,
    node_id: EntityId,
) -> Result<f32, VectorIndexError> {
    let node = graph
        .nodes
        .get(&node_id)
        .ok_or_else(|| VectorIndexError::Storage(format!("node not found: {:?}", node_id)))?;

    Ok(compute_distance_embeddings(query, &node.embedding, graph.distance_metric))
}

/// Compute distance between two embeddings.
fn compute_distance_embeddings(a: &Embedding, b: &Embedding, metric: DistanceMetric) -> f32 {
    let a = a.as_slice();
    let b = b.as_slice();

    match metric {
        DistanceMetric::Euclidean => {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
        }
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a == 0.0 || norm_b == 0.0 {
                f32::MAX
            } else {
                1.0 - (dot / (norm_a * norm_b))
            }
        }
        DistanceMetric::DotProduct => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
        DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::Chebyshev => {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::TransactionManager;
    use manifoldb_storage::backends::RedbEngine;

    #[test]
    fn test_hnsw_index_builder() {
        let engine = RedbEngine::in_memory().unwrap();
        let manager = TransactionManager::new(engine);

        // Create some entities with vectors
        {
            let mut tx = manager.begin_write().unwrap();

            let entity1 = tx
                .create_entity()
                .unwrap()
                .with_label("documents")
                .with_property("embedding", vec![0.1f32, 0.2, 0.3, 0.4]);
            tx.put_entity(&entity1).unwrap();

            let entity2 = tx
                .create_entity()
                .unwrap()
                .with_label("documents")
                .with_property("embedding", vec![0.2f32, 0.3, 0.4, 0.5]);
            tx.put_entity(&entity2).unwrap();

            let entity3 = tx
                .create_entity()
                .unwrap()
                .with_label("documents")
                .with_property("embedding", vec![0.3f32, 0.4, 0.5, 0.6]);
            tx.put_entity(&entity3).unwrap();

            tx.commit().unwrap();
        }

        // Create the index
        {
            let mut tx = manager.begin_write().unwrap();

            HnswIndexBuilder::new("test_idx", "documents", "embedding")
                .dimension(4)
                .distance_metric(DistanceMetric::Cosine)
                .m(4)
                .ef_construction(16)
                .build(&mut tx)
                .unwrap();

            tx.commit().unwrap();
        }

        // Verify index was created
        {
            let tx = manager.begin_read().unwrap();
            let (graph, config) = load_index(&tx, "test_idx").unwrap();

            assert_eq!(graph.dimension, 4);
            assert_eq!(graph.nodes.len(), 3);
            assert_eq!(config.m, 4);
        }
    }

    #[test]
    fn test_drop_index() {
        let engine = RedbEngine::in_memory().unwrap();
        let manager = TransactionManager::new(engine);

        // Create an index
        {
            let mut tx = manager.begin_write().unwrap();

            HnswIndexBuilder::new("to_drop", "test", "vec").dimension(4).build(&mut tx).unwrap();

            tx.commit().unwrap();
        }

        // Verify it exists
        {
            let tx = manager.begin_read().unwrap();
            assert!(load_index(&tx, "to_drop").is_ok());
        }

        // Drop it
        {
            let mut tx = manager.begin_write().unwrap();
            assert!(drop_index(&mut tx, "to_drop", false).unwrap());
            tx.commit().unwrap();
        }

        // Verify it's gone
        {
            let tx = manager.begin_read().unwrap();
            assert!(load_index(&tx, "to_drop").is_err());
        }
    }
}
