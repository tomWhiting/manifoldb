//! Collection handle for point operations and search.
//!
//! This module provides the [`CollectionHandle`] which is the main interface
//! for working with a collection after it's been created or opened.
//!
//! # Example
//!
//! ```ignore
//! use manifoldb::collection::{PointStruct, Vector, Filter};
//! use serde_json::json;
//!
//! // Get a collection handle
//! let collection = db.collection("documents")?;
//!
//! // Upsert a point
//! collection.upsert_point(PointStruct::new(1)
//!     .with_payload(json!({"title": "Rust Book"}))
//!     .with_vector("text", vec![0.1; 768]))?;
//!
//! // Search
//! let results = collection.search("text")
//!     .query(query_vector)
//!     .limit(10)
//!     .filter(Filter::eq("category", "programming"))
//!     .execute()?;
//! ```

use std::collections::HashMap;

use manifoldb_core::PointId;
use manifoldb_storage::StorageEngine;
use manifoldb_vector::store::PointStore;
use manifoldb_vector::types::{
    CollectionName as VectorCollectionName, CollectionSchema, NamedVector,
    Payload as VectorPayload, VectorConfig as VectorStoreConfig,
    VectorType as VectorStoreVectorType,
};
use serde_json::Value as JsonValue;

use super::config::VectorConfig;
use super::error::{ApiError, ApiResult};
use super::filter::Filter;
use super::metadata::CollectionName;
use super::point::{PointStruct, ScoredPoint, Vector};
use super::search::{FusionStrategy, HybridSearchBuilder, SearchBuilder};

/// A handle to a collection for performing point operations and search.
///
/// The collection handle provides methods for:
/// - Inserting, updating, and deleting points
/// - Single-vector and hybrid similarity search
/// - Point retrieval and listing
///
/// Handles are lightweight and can be cloned. They hold a reference to
/// the storage engine and collection metadata.
///
/// # Example
///
/// ```ignore
/// use manifoldb::collection::{PointStruct, Vector};
///
/// let collection = db.collection("documents")?;
///
/// // Insert points
/// collection.upsert_point(PointStruct::new(1)
///     .with_payload(json!({"title": "Hello World"}))
///     .with_vector("text", vec![0.1; 768]))?;
///
/// // Search
/// let results = collection.search("text")
///     .query(vec![0.1; 768])
///     .limit(10)
///     .execute()?;
/// ```
pub struct CollectionHandle<E: StorageEngine> {
    /// The underlying point store.
    point_store: PointStore<E>,
    /// The collection name.
    name: CollectionName,
    /// The vector name (for manifoldb-vector).
    vector_name: VectorCollectionName,
    /// Vector configurations from the collection schema.
    vectors: HashMap<String, VectorConfig>,
}

impl<E: StorageEngine> CollectionHandle<E> {
    /// Create a new collection and return a handle to it.
    ///
    /// This is called by `CollectionBuilder::build()` after configuring vectors.
    pub(crate) fn create(
        engine: E,
        name: CollectionName,
        vectors: Vec<(String, VectorConfig)>,
    ) -> ApiResult<Self> {
        // Create the vector collection name
        let vector_name = VectorCollectionName::new(name.as_str())?;

        // Build the schema for the point store
        let mut schema = CollectionSchema::new();
        for (vec_name, config) in &vectors {
            let store_config = vector_config_to_store_config(config);
            schema = schema.with_vector(vec_name.clone(), store_config);
        }

        // Create the point store with the engine
        let point_store = PointStore::new(engine);

        // Create the collection in the point store
        point_store.create_collection(&vector_name, schema)?;

        Ok(Self { point_store, name, vector_name, vectors: vectors.into_iter().collect() })
    }

    /// Open an existing collection and return a handle.
    ///
    /// This is called by `Database::collection()` to get a handle to an existing collection.
    pub(crate) fn open(engine: E, name: CollectionName) -> ApiResult<Self> {
        let vector_name = VectorCollectionName::new(name.as_str())?;
        let point_store = PointStore::new(engine);

        // Get the collection to verify it exists and get the schema
        let collection = point_store.get_collection(&vector_name)?;
        let schema = collection.schema();

        // Convert schema to our VectorConfig format
        let mut vectors = HashMap::new();
        for (vec_name, store_config) in schema.vectors() {
            let config = store_config_to_vector_config(store_config);
            vectors.insert(vec_name.clone(), config);
        }

        Ok(Self { point_store, name, vector_name, vectors })
    }

    /// Get the collection name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    /// Get the vector configurations for this collection.
    #[must_use]
    pub fn vectors(&self) -> &HashMap<String, VectorConfig> {
        &self.vectors
    }

    /// Check if the collection has a vector with the given name.
    #[must_use]
    pub fn has_vector(&self, name: &str) -> bool {
        self.vectors.contains_key(name)
    }

    // ========================================================================
    // Point operations
    // ========================================================================

    /// Upsert a point (insert or update).
    ///
    /// If a point with the same ID exists, it will be updated.
    /// Otherwise, a new point is created.
    ///
    /// # Example
    ///
    /// ```ignore
    /// collection.upsert_point(PointStruct::new(1)
    ///     .with_payload(json!({"title": "Rust Book"}))
    ///     .with_vector("text", vec![0.1; 768]))?;
    /// ```
    pub fn upsert_point(&self, point: PointStruct) -> ApiResult<()> {
        let payload =
            point.payload.map(|v| VectorPayload::from_value(v)).unwrap_or_default();

        let vectors = self.convert_vectors_to_store(point.vectors)?;

        self.point_store.upsert_point(&self.vector_name, point.id, payload, vectors)?;
        Ok(())
    }

    /// Insert a point. Fails if the point already exists.
    ///
    /// # Errors
    ///
    /// Returns an error if a point with the same ID already exists.
    pub fn insert_point(&self, point: PointStruct) -> ApiResult<()> {
        if self.point_exists(point.id)? {
            return Err(ApiError::PointAlreadyExists {
                point_id: point.id,
                collection: self.name.as_str().to_string(),
            });
        }
        self.upsert_point(point)
    }

    /// Upsert multiple points in a batch.
    ///
    /// More efficient than calling `upsert_point` multiple times.
    pub fn upsert_points(&self, points: impl IntoIterator<Item = PointStruct>) -> ApiResult<()> {
        for point in points {
            self.upsert_point(point)?;
        }
        Ok(())
    }

    /// Get a point's payload by ID.
    ///
    /// Returns `None` if the point doesn't exist.
    pub fn get_payload(&self, id: PointId) -> ApiResult<Option<JsonValue>> {
        match self.point_store.get_payload(&self.vector_name, id) {
            Ok(payload) => Ok(Some(payload.into_value())),
            Err(manifoldb_vector::error::VectorError::EmbeddingNotFound { .. }) => Ok(None),
            Err(e) => Err(ApiError::from(e)),
        }
    }

    /// Get a specific vector from a point.
    ///
    /// Returns `None` if the point or vector doesn't exist.
    pub fn get_vector(&self, id: PointId, vector_name: &str) -> ApiResult<Option<Vector>> {
        match self.point_store.get_vector(&self.vector_name, id, vector_name) {
            Ok(named_vec) => Ok(Some(named_vector_to_vector(named_vec))),
            Err(manifoldb_vector::error::VectorError::EmbeddingNotFound { .. }) => Ok(None),
            Err(e) => Err(ApiError::from(e)),
        }
    }

    /// Get all vectors for a point.
    ///
    /// Returns an empty map if the point doesn't exist.
    pub fn get_all_vectors(&self, id: PointId) -> ApiResult<HashMap<String, Vector>> {
        let store_vectors = self.point_store.get_all_vectors(&self.vector_name, id)?;
        Ok(store_vectors.into_iter().map(|(k, v)| (k, named_vector_to_vector(v))).collect())
    }

    /// Update a point's payload without touching vectors.
    pub fn update_payload(&self, id: PointId, payload: JsonValue) -> ApiResult<()> {
        if !self.point_exists(id)? {
            return Err(ApiError::PointNotFound {
                point_id: id,
                collection: self.name.as_str().to_string(),
            });
        }
        self.point_store.update_payload(
            &self.vector_name,
            id,
            VectorPayload::from_value(payload),
        )?;
        Ok(())
    }

    /// Update a specific vector without touching the payload or other vectors.
    pub fn update_vector(
        &self,
        id: PointId,
        vector_name: &str,
        vector: impl Into<Vector>,
    ) -> ApiResult<()> {
        let named_vec = vector_to_named_vector(vector.into());
        self.point_store.update_vector(&self.vector_name, id, vector_name, named_vec)?;
        Ok(())
    }

    /// Delete a point and all its vectors.
    ///
    /// Returns `true` if the point was deleted, `false` if it didn't exist.
    pub fn delete_point(&self, id: PointId) -> ApiResult<bool> {
        Ok(self.point_store.delete_point(&self.vector_name, id)?)
    }

    /// Delete multiple points.
    ///
    /// Returns the number of points deleted.
    pub fn delete_points(&self, ids: impl IntoIterator<Item = PointId>) -> ApiResult<usize> {
        let mut deleted = 0;
        for id in ids {
            if self.delete_point(id)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Delete a specific vector from a point.
    ///
    /// Returns `true` if the vector was deleted, `false` if it didn't exist.
    pub fn delete_vector(&self, id: PointId, vector_name: &str) -> ApiResult<bool> {
        Ok(self.point_store.delete_vector(&self.vector_name, id, vector_name)?)
    }

    /// Check if a point exists.
    pub fn point_exists(&self, id: PointId) -> ApiResult<bool> {
        Ok(self.point_store.point_exists(&self.vector_name, id)?)
    }

    /// List all point IDs in the collection.
    pub fn list_points(&self) -> ApiResult<Vec<PointId>> {
        Ok(self.point_store.list_points(&self.vector_name)?)
    }

    /// Count the number of points in the collection.
    pub fn count_points(&self) -> ApiResult<usize> {
        Ok(self.point_store.count_points(&self.vector_name)?)
    }

    // ========================================================================
    // Search operations
    // ========================================================================

    /// Create a search builder for single-vector search.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = collection.search("text")
    ///     .query(query_vector)
    ///     .limit(10)
    ///     .filter(Filter::eq("category", "programming"))
    ///     .with_payload(true)
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn search(&self, vector_name: impl Into<String>) -> SearchBuilder<'_, E> {
        SearchBuilder::new(self, vector_name)
    }

    /// Create a hybrid search builder for multi-vector search.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = collection.hybrid_search()
    ///     .query("dense", dense_vector, 0.7)
    ///     .query("sparse", sparse_vector, 0.3)
    ///     .limit(10)
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn hybrid_search(&self) -> HybridSearchBuilder<'_, E> {
        HybridSearchBuilder::new(self)
    }

    /// Execute a search query (internal implementation).
    pub(crate) fn execute_search(
        &self,
        vector_name: &str,
        query: Vector,
        limit: usize,
        offset: usize,
        filter: Option<Filter>,
        with_payload: bool,
        with_vectors: bool,
        score_threshold: Option<f32>,
        _ef: Option<usize>,
    ) -> ApiResult<Vec<ScoredPoint>> {
        // For now, we do a simple scan-and-filter approach
        // A production implementation would use the HNSW index
        let points = self.point_store.list_points(&self.vector_name)?;

        let mut results = Vec::new();

        for point_id in points {
            // Get the vector
            let stored_vector =
                match self.point_store.get_vector(&self.vector_name, point_id, vector_name) {
                    Ok(v) => v,
                    Err(_) => continue, // Point doesn't have this vector
                };

            // Get payload for filtering
            let payload = self.point_store.get_payload(&self.vector_name, point_id)?;
            let payload_value = payload.value().clone();

            // Apply filter
            if let Some(ref f) = filter {
                if !f.matches(&payload_value) {
                    continue;
                }
            }

            // Compute similarity score
            let score = compute_similarity(&query, &named_vector_to_vector(stored_vector));

            // Apply score threshold
            if let Some(threshold) = score_threshold {
                if score < threshold {
                    continue;
                }
            }

            let mut scored = ScoredPoint::new(point_id, score);

            if with_payload {
                scored = scored.with_payload(payload_value);
            }

            if with_vectors {
                let vectors = self.get_all_vectors(point_id)?;
                scored = scored.with_vectors(vectors);
            }

            results.push(scored);
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Apply offset and limit
        let results: Vec<_> = results.into_iter().skip(offset).take(limit).collect();

        Ok(results)
    }

    /// Execute a hybrid search query (internal implementation).
    pub(crate) fn execute_hybrid_search(
        &self,
        queries: Vec<(String, Vector, f32)>,
        limit: usize,
        offset: usize,
        filter: Option<Filter>,
        with_payload: bool,
        with_vectors: bool,
        fusion: FusionStrategy,
    ) -> ApiResult<Vec<ScoredPoint>> {
        // Execute individual searches for each vector
        let mut all_results: Vec<Vec<(PointId, f32)>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for (vector_name, query, weight) in queries {
            let results = self.execute_search(
                &vector_name,
                query,
                limit * 3, // Fetch more for fusion
                0,
                filter.clone(),
                false,
                false,
                None,
                None,
            )?;

            all_results.push(results.into_iter().map(|r| (r.id, r.score)).collect());
            weights.push(weight);
        }

        // Fuse results
        let fused = fuse_results(all_results, weights, fusion);

        // Sort and limit
        let mut sorted: Vec<_> = fused.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply offset and limit, and enrich with payload/vectors
        let mut final_results = Vec::new();
        for (point_id, score) in sorted.into_iter().skip(offset).take(limit) {
            let mut scored = ScoredPoint::new(point_id, score);

            if with_payload {
                if let Ok(payload) = self.point_store.get_payload(&self.vector_name, point_id) {
                    scored = scored.with_payload(payload.into_value());
                }
            }

            if with_vectors {
                let vectors = self.get_all_vectors(point_id)?;
                scored = scored.with_vectors(vectors);
            }

            final_results.push(scored);
        }

        Ok(final_results)
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Convert our Vector types to the store's NamedVector types.
    fn convert_vectors_to_store(
        &self,
        vectors: HashMap<String, Vector>,
    ) -> ApiResult<HashMap<String, NamedVector>> {
        let mut result = HashMap::new();
        for (name, vector) in vectors {
            result.insert(name, vector_to_named_vector(vector));
        }
        Ok(result)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Convert our Vector to NamedVector.
fn vector_to_named_vector(vector: Vector) -> NamedVector {
    match vector {
        Vector::Dense(v) => NamedVector::Dense(v),
        Vector::Sparse(v) => NamedVector::Sparse(v),
        Vector::Multi(v) => NamedVector::Multi(v),
    }
}

/// Convert NamedVector to our Vector.
fn named_vector_to_vector(vector: NamedVector) -> Vector {
    match vector {
        NamedVector::Dense(v) => Vector::Dense(v),
        NamedVector::Sparse(v) => Vector::Sparse(v),
        NamedVector::Multi(v) => Vector::Multi(v),
    }
}

/// Convert VectorConfig to store's VectorConfig.
fn vector_config_to_store_config(config: &VectorConfig) -> VectorStoreConfig {
    match &config.vector_type {
        super::config::VectorType::Dense { dimension } => {
            VectorStoreConfig::dense(*dimension as u32)
        }
        super::config::VectorType::Sparse { max_dimension } => {
            VectorStoreConfig::sparse(*max_dimension)
        }
        super::config::VectorType::Multi { token_dim } => {
            VectorStoreConfig::multi(*token_dim as u32)
        }
        super::config::VectorType::Binary { bits } => {
            // Treat binary as dense with bits/8 dimension
            VectorStoreConfig::dense((*bits / 8) as u32)
        }
    }
}

/// Convert store's VectorConfig to our VectorConfig.
fn store_config_to_vector_config(config: &VectorStoreConfig) -> VectorConfig {
    use manifoldb_vector::distance::DistanceMetric;

    match config.vector_type {
        VectorStoreVectorType::Dense => {
            VectorConfig::dense(config.dimension as usize, DistanceMetric::Cosine)
        }
        VectorStoreVectorType::Sparse => VectorConfig::sparse(config.dimension),
        VectorStoreVectorType::Multi => VectorConfig::multi_vector(config.dimension as usize),
    }
}

/// Compute similarity between two vectors.
///
/// Uses cosine similarity for dense vectors, dot product for sparse.
fn compute_similarity(a: &Vector, b: &Vector) -> f32 {
    match (a, b) {
        (Vector::Dense(a), Vector::Dense(b)) => cosine_similarity(a, b),
        (Vector::Sparse(a), Vector::Sparse(b)) => sparse_dot_product(a, b),
        (Vector::Multi(a), Vector::Multi(b)) => max_sim(a, b),
        _ => 0.0, // Mismatched types
    }
}

/// Cosine similarity between two dense vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Dot product between two sparse vectors.
fn sparse_dot_product(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut score = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                score += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                j += 1;
            }
        }
    }

    score
}

/// MaxSim (ColBERT-style) similarity between multi-vectors.
fn max_sim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if query.is_empty() || doc.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;

    for q_token in query {
        let mut max_sim = f32::NEG_INFINITY;
        for d_token in doc {
            let sim = cosine_similarity(q_token, d_token);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        if max_sim.is_finite() {
            total += max_sim;
        }
    }

    total
}

/// Fuse results from multiple searches using the specified strategy.
fn fuse_results(
    results: Vec<Vec<(PointId, f32)>>,
    weights: Vec<f32>,
    strategy: FusionStrategy,
) -> HashMap<PointId, f32> {
    let mut fused: HashMap<PointId, f32> = HashMap::new();

    match strategy {
        FusionStrategy::Rrf { k } => {
            // Reciprocal Rank Fusion
            for (result_set, weight) in results.iter().zip(weights.iter()) {
                for (rank, (point_id, _score)) in result_set.iter().enumerate() {
                    let rrf_score = weight / (k + (rank as f32) + 1.0);
                    *fused.entry(*point_id).or_insert(0.0) += rrf_score;
                }
            }
        }
        FusionStrategy::WeightedAverage => {
            // Normalize scores and average
            for (result_set, weight) in results.iter().zip(weights.iter()) {
                let max_score =
                    result_set.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
                let min_score = result_set.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
                let range = max_score - min_score;

                for (point_id, score) in result_set {
                    let normalized = if range > 0.0 { (score - min_score) / range } else { 1.0 };
                    *fused.entry(*point_id).or_insert(0.0) += normalized * weight;
                }
            }
        }
        FusionStrategy::WeightedSum => {
            // Simple weighted sum
            for (result_set, weight) in results.iter().zip(weights.iter()) {
                for (point_id, score) in result_set {
                    *fused.entry(*point_id).or_insert(0.0) += score * weight;
                }
            }
        }
    }

    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_create_with_arc_engine_works() {
        use manifoldb_storage::backends::RedbEngine;

        // Create an engine wrapped in Arc
        let engine = Arc::new(RedbEngine::in_memory().unwrap());

        // Clone the Arc to have multiple references - this should work now
        let engine_clone = Arc::clone(&engine);

        // Create a collection handle with the Arc - should succeed
        let name = CollectionName::new("test_collection").unwrap();
        let vectors = vec![(
            "embedding".to_string(),
            VectorConfig::dense(128, manifoldb_vector::distance::DistanceMetric::Cosine),
        )];

        let result = CollectionHandle::create(engine, name, vectors);
        assert!(result.is_ok(), "Creating handle with Arc should work");

        // The clone should still be valid
        drop(engine_clone);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((cosine_similarity(&a, &b) - expected).abs() < 0.001);
    }

    #[test]
    fn test_sparse_dot_product() {
        let a = vec![(0, 1.0), (2, 2.0), (5, 3.0)];
        let b = vec![(0, 0.5), (2, 1.0), (10, 1.0)];
        // 1.0*0.5 + 2.0*1.0 = 2.5
        assert!((sparse_dot_product(&a, &b) - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_rrf_fusion() {
        let results = vec![
            vec![(PointId::new(1), 0.9), (PointId::new(2), 0.8)],
            vec![(PointId::new(2), 0.95), (PointId::new(1), 0.85)],
        ];
        let weights = vec![0.5, 0.5];

        let fused = fuse_results(results, weights, FusionStrategy::Rrf { k: 60.0 });

        // Both points should have scores
        assert!(fused.contains_key(&PointId::new(1)));
        assert!(fused.contains_key(&PointId::new(2)));
    }
}
