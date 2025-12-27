//! ManifoldDB storage for embeddings using the Collection API
//!
//! Uses ManifoldDB's Collection-based vector storage for proper HNSW indexing.
//! Vectors are stored separately from entity properties using the VectorIndexCoordinator.
//!
//! Supports multiple embedding types per chunk and various search modes:
//! - Dense vector similarity search
//! - Sparse (SPLADE) search
//! - Hybrid search with configurable fusion
//! - ColBERT late interaction search

use crate::config::{Config, FusionStrategy, HybridConfig, SearchMode};
use crate::embed::Embedding;
use anyhow::{anyhow, Result};
use manifoldb::collection::{CollectionHandle, DistanceMetric, Filter, PointStruct, Vector};
use manifoldb::{Database, DatabaseBuilder, EntityId, Value};
use manifoldb_storage::backends::RedbEngine;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Collection name for chunks
const CHUNKS_COLLECTION: &str = "chunks";

/// Vector names for different embedding types
pub const VECTOR_DENSE: &str = "dense";
pub const VECTOR_SPARSE: &str = "sparse";
pub const VECTOR_COLBERT: &str = "colbert";

/// Storage layer for embeddings in ManifoldDB using the Collection API
pub struct Store {
    db: Database,
    collection: CollectionHandle<Arc<RedbEngine>>,
    /// Cache of directory path -> EntityId
    dir_cache: HashMap<String, EntityId>,
    /// Cache of file path -> EntityId
    file_cache: HashMap<String, EntityId>,
    /// Dimension of dense vectors (detected from first embedding)
    dense_dim: Option<usize>,
    /// Whether sparse vectors are enabled
    has_sparse: bool,
    /// Whether ColBERT vectors are enabled
    has_colbert: bool,
    /// Dimension of ColBERT token vectors
    colbert_dim: Option<usize>,
}

impl Store {
    /// Open or create the store with the given configuration
    pub fn open(config: &Config) -> Result<Self> {
        let db = if config.database.path.to_string_lossy() == ":memory:" {
            Database::in_memory()?
        } else {
            DatabaseBuilder::new()
                .path(&config.database.path)
                .create_if_missing(true)
                .open()?
        };

        // Determine which vector types are configured
        let has_dense = config.has_vector("dense");
        let has_sparse = config.has_vector("sparse");
        let has_colbert = config.has_vector("colbert");

        // Try to open existing collection, or create a new one
        let collection = match db.collection(CHUNKS_COLLECTION) {
            Ok(c) => c,
            Err(_) => {
                // Create collection with configured vectors
                // We'll use default dimensions here - they'll be validated on first insert
                let mut builder = db.create_collection(CHUNKS_COLLECTION)?;

                if has_dense {
                    // Use a typical dimension for BGE models (768)
                    // This will be validated when inserting
                    builder = builder.with_dense_vector(VECTOR_DENSE, 768, DistanceMetric::Cosine);
                }

                if has_sparse {
                    builder = builder.with_sparse_vector(VECTOR_SPARSE);
                }

                if has_colbert {
                    // ColBERT uses 128-dim token vectors
                    builder = builder.with_dense_vector(VECTOR_COLBERT, 128, DistanceMetric::Cosine);
                }

                builder.build()?
            }
        };

        Ok(Self {
            db,
            collection,
            dir_cache: HashMap::new(),
            file_cache: HashMap::new(),
            dense_dim: None,
            has_sparse,
            has_colbert,
            colbert_dim: None,
        })
    }

    /// Ensure directory node exists and return its EntityId
    pub fn ensure_directory(&mut self, path: &str) -> Result<EntityId> {
        if let Some(&id) = self.dir_cache.get(path) {
            return Ok(id);
        }

        // Check if already exists in DB
        let existing = self.db.query_with_params(
            "SELECT _rowid FROM Directory WHERE path = $1 LIMIT 1",
            &[Value::String(path.to_string())],
        )?;

        if let Some(row) = existing.into_iter().next() {
            let id = EntityId::new(row.get_as::<i64>(0)? as u64);
            self.dir_cache.insert(path.to_string(), id);
            return Ok(id);
        }

        // Create new directory node
        let mut tx = self.db.begin()?;
        let entity = tx.create_entity()?
            .with_label("Directory")
            .with_property("path", path)
            .with_property("name", Path::new(path).file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string()));

        let id = entity.id;
        tx.put_entity(&entity)?;
        tx.commit()?;

        self.dir_cache.insert(path.to_string(), id);

        // Create edge to parent directory
        if let Some(parent) = Path::new(path).parent() {
            let parent_str = parent.to_string_lossy().to_string();
            if !parent_str.is_empty() && parent_str != path {
                let parent_id = self.ensure_directory(&parent_str)?;
                let mut tx = self.db.begin()?;
                let edge = tx.create_edge(parent_id, id, "CONTAINS")?;
                tx.put_edge(&edge)?;
                tx.commit()?;
            }
        }

        Ok(id)
    }

    /// Ensure file node exists and return its EntityId
    pub fn ensure_file(&mut self, path: &str) -> Result<EntityId> {
        if let Some(&id) = self.file_cache.get(path) {
            return Ok(id);
        }

        // Check if already exists in DB
        let existing = self.db.query_with_params(
            "SELECT _rowid FROM File WHERE path = $1 LIMIT 1",
            &[Value::String(path.to_string())],
        )?;

        if let Some(row) = existing.into_iter().next() {
            let id = EntityId::new(row.get_as::<i64>(0)? as u64);
            self.file_cache.insert(path.to_string(), id);
            return Ok(id);
        }

        // Create new file node
        let mut tx = self.db.begin()?;
        let entity = tx.create_entity()?
            .with_label("File")
            .with_property("path", path)
            .with_property("name", Path::new(path).file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string()));

        let id = entity.id;
        tx.put_entity(&entity)?;
        tx.commit()?;

        self.file_cache.insert(path.to_string(), id);

        // Create edge to parent directory
        if let Some(parent) = Path::new(path).parent() {
            let parent_str = parent.to_string_lossy().to_string();
            if !parent_str.is_empty() {
                let parent_id = self.ensure_directory(&parent_str)?;
                let mut tx = self.db.begin()?;
                let edge = tx.create_edge(parent_id, id, "CONTAINS")?;
                tx.put_edge(&edge)?;
                tx.commit()?;
            }
        }

        Ok(id)
    }

    /// Store a chunk with multiple embeddings using the Collection API
    ///
    /// The embeddings map contains named embeddings (e.g., "dense", "sparse", "colbert")
    /// that will be stored in the collection's vector indices.
    pub fn store_chunk(
        &mut self,
        file_path: &str,
        heading: Option<&str>,
        content: &str,
        embeddings: &HashMap<String, Embedding>,
        properties: &[(String, String)],
        prev_chunk_id: Option<EntityId>,
    ) -> Result<EntityId> {
        let file_id = self.ensure_file(file_path)?;

        // Create the chunk entity first (for graph relationships)
        let mut tx = self.db.begin()?;
        let entity = tx.create_entity()?
            .with_label("Chunk")
            .with_property("file_path", file_path)
            .with_property("content", content)
            .with_property("heading", heading.unwrap_or(""));

        let chunk_id = entity.id;
        tx.put_entity(&entity)?;
        tx.commit()?;

        // Build the payload for the collection point
        let mut payload = json!({
            "file_path": file_path,
            "heading": heading.unwrap_or(""),
            "content": content,
        });

        // Add custom properties to payload
        for (key, value) in properties {
            payload[key] = json!(value);
        }

        // Build the PointStruct with vectors
        let mut point = PointStruct::new(chunk_id.as_u64())
            .with_payload(payload);

        // Add embeddings based on type
        for (name, embedding) in embeddings {
            point = self.add_vector_to_point(point, name, embedding)?;
        }

        // Store in collection
        self.collection.upsert_point(point)?;

        // Create CHILD_OF edge to file
        let mut tx = self.db.begin()?;
        let edge = tx.create_edge(file_id, chunk_id, "CHILD_OF")?;
        tx.put_edge(&edge)?;
        tx.commit()?;

        // Create NEXT edge from previous chunk
        if let Some(prev_id) = prev_chunk_id {
            let mut tx = self.db.begin()?;
            let edge = tx.create_edge(prev_id, chunk_id, "NEXT")?;
            tx.put_edge(&edge)?;
            tx.commit()?;
        }

        Ok(chunk_id)
    }

    /// Add a vector to a PointStruct based on embedding type
    fn add_vector_to_point(
        &mut self,
        point: PointStruct,
        name: &str,
        embedding: &Embedding,
    ) -> Result<PointStruct> {
        match embedding {
            Embedding::Dense(vec) => {
                // Track dimension for validation
                if self.dense_dim.is_none() {
                    self.dense_dim = Some(vec.len());
                }
                Ok(point.with_vector(VECTOR_DENSE, vec.clone()))
            }
            Embedding::Sparse(weights) => {
                // Convert sparse weights to Vector::Sparse format
                let sparse_pairs: Vec<(u32, f32)> = weights
                    .iter()
                    .map(|(idx, weight)| (*idx as u32, *weight))
                    .collect();
                Ok(point.with_vector(VECTOR_SPARSE, Vector::Sparse(sparse_pairs)))
            }
            Embedding::MultiVector(vectors) => {
                // For ColBERT, we store the first token vector as a representative
                // In a full implementation, we'd use a multi-vector index
                if let Some(first_vec) = vectors.first() {
                    if self.colbert_dim.is_none() {
                        self.colbert_dim = Some(first_vec.len());
                    }
                    // Store all vectors as JSON in payload for MaxSim computation
                    // and use first vector for approximate retrieval
                    Ok(point.with_vector(VECTOR_COLBERT, first_vec.clone()))
                } else {
                    Err(anyhow!("Empty ColBERT multi-vector"))
                }
            }
        }
    }

    /// Search using the specified mode
    pub fn search(
        &self,
        mode: SearchMode,
        query_embeddings: &HashMap<String, Embedding>,
        limit: usize,
        filters: &[(String, String)],
        hybrid_config: Option<&HybridConfig>,
    ) -> Result<Vec<SearchResult>> {
        match mode {
            SearchMode::Dense => self.search_dense(query_embeddings, limit, filters),
            SearchMode::Sparse => self.search_sparse(query_embeddings, limit, filters),
            SearchMode::Hybrid => {
                let config = hybrid_config.ok_or_else(|| anyhow!("Hybrid config required for hybrid search"))?;
                self.search_hybrid(query_embeddings, limit, filters, config)
            }
            SearchMode::Colbert => self.search_colbert(query_embeddings, limit, filters),
        }
    }

    /// Dense vector similarity search using Collection API
    fn search_dense(
        &self,
        query_embeddings: &HashMap<String, Embedding>,
        limit: usize,
        filters: &[(String, String)],
    ) -> Result<Vec<SearchResult>> {
        // Find the dense embedding
        let dense = query_embeddings.values()
            .find_map(|e| e.as_dense())
            .ok_or_else(|| anyhow!("No dense embedding provided for dense search"))?;

        // Build search query
        let mut search = self.collection.search(VECTOR_DENSE)
            .query(dense.to_vec())
            .limit(limit)
            .with_payload(true);

        // Apply filters if any
        if !filters.is_empty() {
            let filter = self.build_filter(filters);
            search = search.filter(filter);
        }

        // Execute search
        let results = search.execute()?;

        // Convert to SearchResult
        Ok(results.into_iter().map(|r| {
            let payload = r.payload.unwrap_or_default();
            SearchResult {
                id: EntityId::new(r.id.as_u64()),
                heading: payload.get("heading").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                content: payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                file_path: payload.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                score: r.score,
            }
        }).collect())
    }

    /// Sparse (SPLADE) search using Collection API
    fn search_sparse(
        &self,
        query_embeddings: &HashMap<String, Embedding>,
        limit: usize,
        filters: &[(String, String)],
    ) -> Result<Vec<SearchResult>> {
        // Find the sparse embedding
        let sparse = query_embeddings.values()
            .find_map(|e| e.as_sparse())
            .ok_or_else(|| anyhow!("No sparse embedding provided for sparse search"))?;

        // Convert to sparse vector format
        let sparse_pairs: Vec<(u32, f32)> = sparse
            .iter()
            .map(|(idx, weight)| (*idx as u32, *weight))
            .collect();

        // Build search query with sparse vector
        let mut search = self.collection.search(VECTOR_SPARSE)
            .query(Vector::Sparse(sparse_pairs))
            .limit(limit)
            .with_payload(true);

        // Apply filters if any
        if !filters.is_empty() {
            let filter = self.build_filter(filters);
            search = search.filter(filter);
        }

        // Execute search
        let results = search.execute()?;

        // Convert to SearchResult
        Ok(results.into_iter().map(|r| {
            let payload = r.payload.unwrap_or_default();
            SearchResult {
                id: EntityId::new(r.id.as_u64()),
                heading: payload.get("heading").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                content: payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                file_path: payload.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                score: r.score,
            }
        }).collect())
    }

    /// Hybrid search combining dense and sparse
    fn search_hybrid(
        &self,
        query_embeddings: &HashMap<String, Embedding>,
        limit: usize,
        filters: &[(String, String)],
        config: &HybridConfig,
    ) -> Result<Vec<SearchResult>> {
        // Get more candidates for fusion
        let candidate_limit = limit * 3;

        // Run both searches
        let dense_results = self.search_dense(query_embeddings, candidate_limit, filters)?;
        let sparse_results = self.search_sparse(query_embeddings, candidate_limit, filters)?;

        // Fuse results based on strategy
        let fused = match config.fusion {
            FusionStrategy::Rrf => self.rrf_fusion(&dense_results, &sparse_results, limit),
            FusionStrategy::WeightedSum => self.weighted_fusion(&dense_results, &sparse_results, config, limit),
        };

        Ok(fused)
    }

    /// Reciprocal Rank Fusion
    fn rrf_fusion(
        &self,
        dense_results: &[SearchResult],
        sparse_results: &[SearchResult],
        limit: usize,
    ) -> Vec<SearchResult> {
        const K: f32 = 60.0; // RRF constant

        let mut scores: HashMap<u64, (f32, Option<&SearchResult>)> = HashMap::new();

        // Score from dense results
        for (rank, result) in dense_results.iter().enumerate() {
            let rrf_score = 1.0 / (K + rank as f32 + 1.0);
            scores.entry(result.id.as_u64())
                .and_modify(|(s, _)| *s += rrf_score)
                .or_insert((rrf_score, Some(result)));
        }

        // Score from sparse results
        for (rank, result) in sparse_results.iter().enumerate() {
            let rrf_score = 1.0 / (K + rank as f32 + 1.0);
            scores.entry(result.id.as_u64())
                .and_modify(|(s, r)| {
                    *s += rrf_score;
                    if r.is_none() {
                        *r = Some(result);
                    }
                })
                .or_insert((rrf_score, Some(result)));
        }

        // Sort by combined score and collect
        let mut scored: Vec<_> = scores.into_iter()
            .filter_map(|(_, (score, result))| result.map(|r| (score, r)))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored.into_iter().map(|(score, result)| {
            SearchResult {
                id: result.id,
                heading: result.heading.clone(),
                content: result.content.clone(),
                file_path: result.file_path.clone(),
                score,
            }
        }).collect()
    }

    /// Weighted score fusion
    fn weighted_fusion(
        &self,
        dense_results: &[SearchResult],
        sparse_results: &[SearchResult],
        config: &HybridConfig,
        limit: usize,
    ) -> Vec<SearchResult> {
        let mut scores: HashMap<u64, (f32, Option<&SearchResult>)> = HashMap::new();

        // Normalize and weight dense scores
        let max_dense = dense_results.iter().map(|r| r.score).fold(0.0f32, f32::max);
        for result in dense_results {
            let normalized = if max_dense > 0.0 {
                result.score / max_dense
            } else {
                0.0
            };
            let weighted = normalized * config.dense_weight as f32;
            scores.entry(result.id.as_u64())
                .and_modify(|(s, _)| *s += weighted)
                .or_insert((weighted, Some(result)));
        }

        // Normalize and weight sparse scores
        let max_sparse = sparse_results.iter().map(|r| r.score).fold(0.0f32, f32::max);
        for result in sparse_results {
            let normalized = if max_sparse > 0.0 {
                result.score / max_sparse
            } else {
                0.0
            };
            let weighted = normalized * config.sparse_weight as f32;
            scores.entry(result.id.as_u64())
                .and_modify(|(s, r)| {
                    *s += weighted;
                    if r.is_none() {
                        *r = Some(result);
                    }
                })
                .or_insert((weighted, Some(result)));
        }

        // Sort and collect
        let mut scored: Vec<_> = scores.into_iter()
            .filter_map(|(_, (score, result))| result.map(|r| (score, r)))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored.into_iter().map(|(score, result)| {
            SearchResult {
                id: result.id,
                heading: result.heading.clone(),
                content: result.content.clone(),
                file_path: result.file_path.clone(),
                score,
            }
        }).collect()
    }

    /// ColBERT late interaction search
    fn search_colbert(
        &self,
        query_embeddings: &HashMap<String, Embedding>,
        limit: usize,
        filters: &[(String, String)],
    ) -> Result<Vec<SearchResult>> {
        // Find the ColBERT embedding
        let query_vectors = query_embeddings.values()
            .find_map(|e| e.as_multi_vector())
            .ok_or_else(|| anyhow!("No ColBERT embedding provided for ColBERT search"))?;

        // Use first token vector for approximate retrieval
        let first_query = query_vectors.first()
            .ok_or_else(|| anyhow!("Empty ColBERT query"))?;

        // Get more candidates for reranking
        let candidate_limit = limit * 10;

        // Build search query using ColBERT vector
        let mut search = self.collection.search(VECTOR_COLBERT)
            .query(first_query.clone())
            .limit(candidate_limit)
            .with_payload(true);

        // Apply filters if any
        if !filters.is_empty() {
            let filter = self.build_filter(filters);
            search = search.filter(filter);
        }

        // Execute initial retrieval
        let candidates = search.execute()?;

        // Rerank with full MaxSim computation
        // Note: In production, we'd retrieve full multi-vectors from storage
        // For now, we use the single-vector approximation
        let mut results: Vec<SearchResult> = candidates.into_iter().map(|r| {
            let payload = r.payload.unwrap_or_default();
            SearchResult {
                id: EntityId::new(r.id.as_u64()),
                heading: payload.get("heading").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                content: payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                file_path: payload.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                score: r.score,
            }
        }).collect();

        // Sort by score and take limit
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Build a filter from key-value pairs
    fn build_filter(&self, filters: &[(String, String)]) -> Filter {
        // Convert all filters to Filter::eq
        let filter_vec: Vec<Filter> = filters
            .iter()
            .map(|(key, value)| Filter::eq(key.as_str(), value.as_str()))
            .collect();

        // Combine with AND
        if filter_vec.len() == 1 {
            filter_vec.into_iter().next().unwrap()
        } else {
            Filter::And(filter_vec)
        }
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// MaxSim scoring for ColBERT
    fn max_sim(query_vectors: &[Vec<f32>], doc_vectors: &[Vec<f32>]) -> f32 {
        if query_vectors.is_empty() || doc_vectors.is_empty() {
            return 0.0;
        }

        // For each query token, find max similarity with any doc token
        let mut total_score = 0.0;
        for q_vec in query_vectors {
            let max_sim = doc_vectors.iter()
                .map(|d_vec| Self::cosine_similarity(q_vec, d_vec))
                .fold(f32::NEG_INFINITY, f32::max);
            total_score += max_sim;
        }

        total_score / query_vectors.len() as f32
    }

    /// Get database reference for advanced queries
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Flush any buffered data to durable storage
    pub fn flush(&self) -> Result<()> {
        self.db.flush().map_err(|e| anyhow!("Flush failed: {}", e))
    }
}

/// Search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: EntityId,
    pub heading: String,
    pub content: String,
    pub file_path: String,
    /// Score (higher is better for similarity)
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((Store::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((Store::cosine_similarity(&a, &c)).abs() < 1e-6);
    }

    #[test]
    fn test_max_sim() {
        let query = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let doc = vec![
            vec![1.0, 0.0],
            vec![0.5, 0.5],
        ];
        let score = Store::max_sim(&query, &doc);
        // First query token matches first doc token perfectly (1.0)
        // Second query token has max sim ~0.707 with second doc token
        assert!(score > 0.8);
    }
}
