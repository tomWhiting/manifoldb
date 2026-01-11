//! Node similarity algorithms.
//!
//! This module implements similarity algorithms for comparing graph nodes
//! based on their neighborhood structure or property values.
//!
//! # Algorithms
//!
//! ## Neighborhood-based Similarity
//!
//! - [`jaccard_similarity`] - Jaccard coefficient: |A ∩ B| / |A ∪ B|
//! - [`overlap_coefficient`] - Overlap coefficient: |A ∩ B| / min(|A|, |B|)
//!
//! ## Property-based Similarity
//!
//! - [`cosine_similarity`] - Cosine similarity on property vectors
//!
//! ## Bulk Computation
//!
//! - [`NodeSimilarity`] - Compute similarity for all node pairs
//!
//! # Example
//!
//! ```ignore
//! use manifoldb_graph::analytics::similarity::{jaccard_similarity, NodeSimilarity, NodeSimilarityConfig};
//!
//! // Compute Jaccard similarity between two nodes
//! let similarity = jaccard_similarity(&tx, node_a, node_b, None)?;
//!
//! // Compute similarity for all pairs in the graph
//! let config = NodeSimilarityConfig::default().with_top_k(10);
//! let result = NodeSimilarity::compute(&tx, &config)?;
//! ```

use std::collections::{HashMap, HashSet};

use manifoldb_core::EntityId;
use manifoldb_storage::Transaction;

use crate::index::AdjacencyIndex;
use crate::store::{EdgeStore, GraphError, GraphResult, NodeStore};

use super::pagerank::DEFAULT_MAX_GRAPH_NODES;

/// Configuration for node similarity computation.
#[derive(Debug, Clone)]
pub struct NodeSimilarityConfig {
    /// The similarity algorithm to use.
    pub algorithm: SimilarityAlgorithm,
    /// Optional label filter - only consider nodes with this label.
    pub label_filter: Option<String>,
    /// Optional edge type filter - only consider edges of this type.
    pub edge_type_filter: Option<String>,
    /// Return only the top K most similar pairs.
    pub top_k: Option<usize>,
    /// Minimum similarity threshold to include in results.
    pub similarity_cutoff: f64,
    /// Maximum number of nodes allowed before returning an error.
    pub max_graph_nodes: Option<usize>,
}

/// Similarity algorithm to use.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SimilarityAlgorithm {
    /// Jaccard coefficient: |A ∩ B| / |A ∪ B|
    #[default]
    Jaccard,
    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|)
    Overlap,
    /// Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)
    Cosine,
}

impl Default for NodeSimilarityConfig {
    fn default() -> Self {
        Self {
            algorithm: SimilarityAlgorithm::Jaccard,
            label_filter: None,
            edge_type_filter: None,
            top_k: None,
            similarity_cutoff: 0.0,
            max_graph_nodes: Some(DEFAULT_MAX_GRAPH_NODES),
        }
    }
}

impl NodeSimilarityConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the similarity algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: SimilarityAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set the label filter.
    #[must_use]
    pub fn with_label_filter(mut self, label: impl Into<String>) -> Self {
        self.label_filter = Some(label.into());
        self
    }

    /// Set the edge type filter.
    #[must_use]
    pub fn with_edge_type_filter(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type_filter = Some(edge_type.into());
        self
    }

    /// Set the top K limit.
    #[must_use]
    pub const fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Set the similarity cutoff threshold.
    #[must_use]
    pub const fn with_similarity_cutoff(mut self, cutoff: f64) -> Self {
        self.similarity_cutoff = cutoff;
        self
    }

    /// Set the maximum number of nodes allowed.
    #[must_use]
    pub const fn with_max_graph_nodes(mut self, limit: Option<usize>) -> Self {
        self.max_graph_nodes = limit;
        self
    }
}

/// Result of node similarity computation.
#[derive(Debug, Clone)]
pub struct NodeSimilarityResult {
    /// Similarity scores for node pairs.
    /// Each entry is (node1, node2, similarity).
    pub similarities: Vec<(EntityId, EntityId, f64)>,
}

impl NodeSimilarityResult {
    /// Get the top N most similar pairs.
    pub fn top_n(&self, n: usize) -> Vec<(EntityId, EntityId, f64)> {
        self.similarities.iter().take(n).copied().collect()
    }

    /// Get all pairs above a similarity threshold.
    pub fn above_threshold(&self, threshold: f64) -> Vec<(EntityId, EntityId, f64)> {
        self.similarities.iter().filter(|(_, _, sim)| *sim >= threshold).copied().collect()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.similarities.is_empty()
    }

    /// Get the number of similarity pairs.
    pub fn len(&self) -> usize {
        self.similarities.len()
    }
}

/// Node similarity computation.
///
/// Computes pairwise similarity between nodes based on their neighborhoods.
pub struct NodeSimilarity;

impl NodeSimilarity {
    /// Compute similarity scores for all pairs of nodes.
    ///
    /// This is an expensive operation as it considers all pairs of nodes.
    /// Use `top_k` and `similarity_cutoff` to limit results.
    pub fn compute<T: Transaction>(
        tx: &T,
        config: &NodeSimilarityConfig,
    ) -> GraphResult<NodeSimilarityResult> {
        // Check graph size
        if let Some(limit) = config.max_graph_nodes {
            let node_count = NodeStore::count(tx)?;
            if node_count > limit {
                return Err(GraphError::GraphTooLarge { node_count, limit });
            }
        }

        // Collect nodes (optionally filtered by label)
        let nodes = Self::collect_nodes(tx, config.label_filter.as_deref())?;
        if nodes.is_empty() {
            return Ok(NodeSimilarityResult { similarities: Vec::new() });
        }

        // Build neighbor sets for all nodes
        let neighbor_sets =
            Self::build_neighbor_sets(tx, &nodes, config.edge_type_filter.as_deref())?;

        // Compute similarities for all pairs
        let mut similarities = Vec::new();

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let node1 = nodes[i];
                let node2 = nodes[j];

                let neighbors1 = &neighbor_sets[&node1];
                let neighbors2 = &neighbor_sets[&node2];

                let similarity =
                    Self::compute_set_similarity(neighbors1, neighbors2, config.algorithm);

                if similarity >= config.similarity_cutoff {
                    similarities.push((node1, node2, similarity));
                }
            }
        }

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top_k limit
        if let Some(k) = config.top_k {
            similarities.truncate(k);
        }

        Ok(NodeSimilarityResult { similarities })
    }

    /// Collect all nodes, optionally filtered by label.
    fn collect_nodes<T: Transaction>(
        tx: &T,
        label_filter: Option<&str>,
    ) -> GraphResult<Vec<EntityId>> {
        let nodes = if let Some(label) = label_filter {
            // Use label index for filtering
            let label = manifoldb_core::Label::new(label);
            NodeStore::find_by_label(tx, &label)?
        } else {
            // Collect all nodes
            let mut collected = Vec::new();
            NodeStore::for_each(tx, |entity| {
                collected.push(entity.id);
                true
            })?;
            collected
        };

        Ok(nodes)
    }

    /// Build neighbor sets for all nodes.
    fn build_neighbor_sets<T: Transaction>(
        tx: &T,
        nodes: &[EntityId],
        edge_type_filter: Option<&str>,
    ) -> GraphResult<HashMap<EntityId, HashSet<EntityId>>> {
        let node_set: HashSet<EntityId> = nodes.iter().copied().collect();
        let mut neighbor_sets: HashMap<EntityId, HashSet<EntityId>> =
            HashMap::with_capacity(nodes.len());

        for &node in nodes {
            let neighbors = get_neighbors(tx, node, edge_type_filter, Some(&node_set))?;
            neighbor_sets.insert(node, neighbors);
        }

        Ok(neighbor_sets)
    }

    /// Compute similarity between two neighbor sets.
    fn compute_set_similarity(
        set1: &HashSet<EntityId>,
        set2: &HashSet<EntityId>,
        algorithm: SimilarityAlgorithm,
    ) -> f64 {
        if set1.is_empty() && set2.is_empty() {
            return 0.0;
        }

        let intersection_size = set1.intersection(set2).count();

        match algorithm {
            SimilarityAlgorithm::Jaccard => {
                let union_size = set1.union(set2).count();
                if union_size == 0 {
                    0.0
                } else {
                    intersection_size as f64 / union_size as f64
                }
            }
            SimilarityAlgorithm::Overlap => {
                let min_size = set1.len().min(set2.len());
                if min_size == 0 {
                    0.0
                } else {
                    intersection_size as f64 / min_size as f64
                }
            }
            SimilarityAlgorithm::Cosine => {
                let product = set1.len() * set2.len();
                if product == 0 {
                    0.0
                } else {
                    intersection_size as f64 / (product as f64).sqrt()
                }
            }
        }
    }
}

/// Get neighbors of a node, optionally filtered by edge type.
///
/// # Arguments
///
/// * `tx` - The transaction to use
/// * `node` - The node to get neighbors for
/// * `edge_type_filter` - Optional edge type filter
/// * `valid_nodes` - Optional set of valid nodes to filter neighbors
///
/// # Returns
///
/// A set of neighbor node IDs.
#[allow(clippy::implicit_hasher)]
pub fn get_neighbors<T: Transaction>(
    tx: &T,
    node: EntityId,
    edge_type_filter: Option<&str>,
    valid_nodes: Option<&HashSet<EntityId>>,
) -> GraphResult<HashSet<EntityId>> {
    let mut neighbors = HashSet::new();

    // Get outgoing edges
    let outgoing = AdjacencyIndex::get_outgoing_edge_ids(tx, node)?;
    for edge_id in outgoing {
        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
            // Check edge type filter
            if let Some(filter) = edge_type_filter {
                if edge.edge_type.as_str() != filter {
                    continue;
                }
            }

            // Check if target is valid (if filter provided)
            if let Some(valid) = valid_nodes {
                if !valid.contains(&edge.target) {
                    continue;
                }
            }

            if edge.target != node {
                neighbors.insert(edge.target);
            }
        }
    }

    // Get incoming edges (treat graph as undirected for similarity)
    let incoming = AdjacencyIndex::get_incoming_edge_ids(tx, node)?;
    for edge_id in incoming {
        if let Some(edge) = EdgeStore::get(tx, edge_id)? {
            // Check edge type filter
            if let Some(filter) = edge_type_filter {
                if edge.edge_type.as_str() != filter {
                    continue;
                }
            }

            // Check if source is valid (if filter provided)
            if let Some(valid) = valid_nodes {
                if !valid.contains(&edge.source) {
                    continue;
                }
            }

            if edge.source != node {
                neighbors.insert(edge.source);
            }
        }
    }

    Ok(neighbors)
}

/// Compute Jaccard similarity between two nodes.
///
/// Jaccard coefficient = |A ∩ B| / |A ∪ B|
///
/// Where A and B are the neighbor sets of the two nodes.
///
/// # Arguments
///
/// * `tx` - The transaction to use
/// * `node1` - First node
/// * `node2` - Second node
/// * `edge_type` - Optional edge type filter
///
/// # Returns
///
/// The Jaccard similarity coefficient (0.0 to 1.0).
pub fn jaccard_similarity<T: Transaction>(
    tx: &T,
    node1: EntityId,
    node2: EntityId,
    edge_type: Option<&str>,
) -> GraphResult<f64> {
    let neighbors1 = get_neighbors(tx, node1, edge_type, None)?;
    let neighbors2 = get_neighbors(tx, node2, edge_type, None)?;

    if neighbors1.is_empty() && neighbors2.is_empty() {
        return Ok(0.0);
    }

    let intersection_size = neighbors1.intersection(&neighbors2).count();
    let union_size = neighbors1.union(&neighbors2).count();

    if union_size == 0 {
        Ok(0.0)
    } else {
        Ok(intersection_size as f64 / union_size as f64)
    }
}

/// Compute Overlap coefficient between two nodes.
///
/// Overlap coefficient = |A ∩ B| / min(|A|, |B|)
///
/// Where A and B are the neighbor sets of the two nodes.
///
/// # Arguments
///
/// * `tx` - The transaction to use
/// * `node1` - First node
/// * `node2` - Second node
/// * `edge_type` - Optional edge type filter
///
/// # Returns
///
/// The Overlap coefficient (0.0 to 1.0).
pub fn overlap_coefficient<T: Transaction>(
    tx: &T,
    node1: EntityId,
    node2: EntityId,
    edge_type: Option<&str>,
) -> GraphResult<f64> {
    let neighbors1 = get_neighbors(tx, node1, edge_type, None)?;
    let neighbors2 = get_neighbors(tx, node2, edge_type, None)?;

    let min_size = neighbors1.len().min(neighbors2.len());
    if min_size == 0 {
        return Ok(0.0);
    }

    let intersection_size = neighbors1.intersection(&neighbors2).count();
    Ok(intersection_size as f64 / min_size as f64)
}

/// Compute Cosine similarity between two nodes based on property vectors.
///
/// The properties are treated as a vector, and cosine similarity is computed.
///
/// # Arguments
///
/// * `tx` - The transaction to use
/// * `node1` - First node
/// * `node2` - Second node
/// * `properties` - List of property names to use as vector dimensions
///
/// # Returns
///
/// The Cosine similarity (-1.0 to 1.0, typically 0.0 to 1.0 for non-negative values).
pub fn cosine_similarity_properties<T: Transaction>(
    tx: &T,
    node1: EntityId,
    node2: EntityId,
    properties: &[String],
) -> GraphResult<f64> {
    // Get the entities
    let entity1 = NodeStore::get(tx, node1)?.ok_or(GraphError::EntityNotFound(node1))?;
    let entity2 = NodeStore::get(tx, node2)?.ok_or(GraphError::EntityNotFound(node2))?;

    // Extract property values as f64 vectors
    let vec1 = extract_property_vector(&entity1, properties);
    let vec2 = extract_property_vector(&entity2, properties);

    Ok(compute_cosine_similarity(&vec1, &vec2))
}

/// Extract numeric property values as a vector.
fn extract_property_vector(entity: &manifoldb_core::Entity, properties: &[String]) -> Vec<f64> {
    properties
        .iter()
        .map(|prop| {
            entity
                .get_property(prop)
                .and_then(|v| {
                    // Try to convert to f64
                    v.as_float().or_else(|| v.as_int().map(|i| i as f64))
                })
                .unwrap_or(0.0)
        })
        .collect()
}

/// Compute cosine similarity between two vectors.
fn compute_cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let config = NodeSimilarityConfig::default();
        assert_eq!(config.algorithm, SimilarityAlgorithm::Jaccard);
        assert!(config.label_filter.is_none());
        assert!(config.edge_type_filter.is_none());
        assert!(config.top_k.is_none());
        assert!((config.similarity_cutoff - 0.0).abs() < f64::EPSILON);
        assert_eq!(config.max_graph_nodes, Some(DEFAULT_MAX_GRAPH_NODES));
    }

    #[test]
    fn config_builder() {
        let config = NodeSimilarityConfig::new()
            .with_algorithm(SimilarityAlgorithm::Overlap)
            .with_label_filter("Person")
            .with_edge_type_filter("KNOWS")
            .with_top_k(10)
            .with_similarity_cutoff(0.5)
            .with_max_graph_nodes(Some(1000));

        assert_eq!(config.algorithm, SimilarityAlgorithm::Overlap);
        assert_eq!(config.label_filter, Some("Person".to_string()));
        assert_eq!(config.edge_type_filter, Some("KNOWS".to_string()));
        assert_eq!(config.top_k, Some(10));
        assert!((config.similarity_cutoff - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.max_graph_nodes, Some(1000));
    }

    #[test]
    fn result_methods() {
        let result = NodeSimilarityResult {
            similarities: vec![
                (EntityId::new(1), EntityId::new(2), 0.8),
                (EntityId::new(1), EntityId::new(3), 0.6),
                (EntityId::new(2), EntityId::new(3), 0.4),
            ],
        };

        assert_eq!(result.len(), 3);
        assert!(!result.is_empty());

        let top2 = result.top_n(2);
        assert_eq!(top2.len(), 2);
        assert!((top2[0].2 - 0.8).abs() < f64::EPSILON);

        let above = result.above_threshold(0.5);
        assert_eq!(above.len(), 2);
    }

    #[test]
    fn set_similarity_jaccard() {
        let set1: HashSet<EntityId> = [EntityId::new(1), EntityId::new(2), EntityId::new(3)].into();
        let set2: HashSet<EntityId> = [EntityId::new(2), EntityId::new(3), EntityId::new(4)].into();

        // Intersection: {2, 3} = 2 elements
        // Union: {1, 2, 3, 4} = 4 elements
        // Jaccard = 2/4 = 0.5
        let sim =
            NodeSimilarity::compute_set_similarity(&set1, &set2, SimilarityAlgorithm::Jaccard);
        assert!((sim - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn set_similarity_overlap() {
        let set1: HashSet<EntityId> = [EntityId::new(1), EntityId::new(2), EntityId::new(3)].into();
        let set2: HashSet<EntityId> = [EntityId::new(2), EntityId::new(3)].into();

        // Intersection: {2, 3} = 2 elements
        // min(3, 2) = 2
        // Overlap = 2/2 = 1.0
        let sim =
            NodeSimilarity::compute_set_similarity(&set1, &set2, SimilarityAlgorithm::Overlap);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn set_similarity_cosine() {
        let set1: HashSet<EntityId> = [EntityId::new(1), EntityId::new(2), EntityId::new(3)].into();
        let set2: HashSet<EntityId> = [EntityId::new(2), EntityId::new(3), EntityId::new(4)].into();

        // Intersection: 2
        // sqrt(3 * 3) = 3
        // Cosine = 2/3 ≈ 0.667
        let sim = NodeSimilarity::compute_set_similarity(&set1, &set2, SimilarityAlgorithm::Cosine);
        assert!((sim - (2.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn set_similarity_empty() {
        let empty: HashSet<EntityId> = HashSet::new();
        let set1: HashSet<EntityId> = [EntityId::new(1)].into();

        assert!(
            (NodeSimilarity::compute_set_similarity(&empty, &empty, SimilarityAlgorithm::Jaccard))
                .abs()
                < f64::EPSILON
        );
        assert!(
            (NodeSimilarity::compute_set_similarity(&empty, &set1, SimilarityAlgorithm::Jaccard))
                .abs()
                < f64::EPSILON
        );
        assert!(
            (NodeSimilarity::compute_set_similarity(&empty, &set1, SimilarityAlgorithm::Overlap))
                .abs()
                < f64::EPSILON
        );
        assert!(
            (NodeSimilarity::compute_set_similarity(&empty, &set1, SimilarityAlgorithm::Cosine))
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn cosine_similarity_vectors() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];

        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        // norm1 = sqrt(1 + 4 + 9) = sqrt(14)
        // norm2 = sqrt(16 + 25 + 36) = sqrt(77)
        // cosine = 32 / (sqrt(14) * sqrt(77)) ≈ 0.9746
        let sim = compute_cosine_similarity(&vec1, &vec2);
        assert!((sim - 0.9746).abs() < 0.001);
    }

    #[test]
    fn cosine_similarity_zero_vectors() {
        let zero = vec![0.0, 0.0, 0.0];
        let vec1 = vec![1.0, 2.0, 3.0];

        assert!((compute_cosine_similarity(&zero, &vec1)).abs() < f64::EPSILON);
        assert!((compute_cosine_similarity(&zero, &zero)).abs() < f64::EPSILON);
    }

    #[test]
    fn cosine_similarity_identical() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let sim = compute_cosine_similarity(&vec1, &vec1);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }
}
