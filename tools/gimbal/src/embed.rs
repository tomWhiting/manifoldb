//! Tessera embedding integration with multi-vector support
//!
//! Provides a unified interface for embedding text using multiple embedding paradigms:
//! - Dense: Single-vector embeddings for semantic similarity
//! - Sparse: SPLADE vocabulary-space embeddings for keyword matching
//! - ColBERT: Multi-vector token embeddings for late interaction

use crate::config::{ModelType, VectorConfig};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use tessera::{
    model_registry::{get_model, ModelInfo},
    TesseraDense, TesseraMultiVector, TesseraSparse,
};

/// Unified embedding result that can hold any embedding type
#[derive(Debug, Clone)]
pub enum Embedding {
    /// Dense single-vector embedding
    Dense(Vec<f32>),
    /// Sparse vocabulary-space embedding (token_id, weight)
    Sparse(Vec<(usize, f32)>),
    /// Multi-vector (ColBERT) token embeddings
    MultiVector(Vec<Vec<f32>>),
}

impl Embedding {
    /// Get dimension of the embedding
    pub fn dimension(&self) -> usize {
        match self {
            Embedding::Dense(v) => v.len(),
            Embedding::Sparse(v) => v.len(), // non-zero count
            Embedding::MultiVector(v) => v.first().map(|t| t.len()).unwrap_or(0),
        }
    }

    /// Get the number of vectors (1 for dense/sparse, N for multi-vector)
    pub fn vector_count(&self) -> usize {
        match self {
            Embedding::Dense(_) | Embedding::Sparse(_) => 1,
            Embedding::MultiVector(v) => v.len(),
        }
    }

    /// Convert to dense embedding reference
    pub fn as_dense(&self) -> Option<&Vec<f32>> {
        match self {
            Embedding::Dense(v) => Some(v),
            _ => None,
        }
    }

    /// Convert to sparse embedding reference
    pub fn as_sparse(&self) -> Option<&Vec<(usize, f32)>> {
        match self {
            Embedding::Sparse(v) => Some(v),
            _ => None,
        }
    }

    /// Convert to multi-vector embedding reference
    pub fn as_multi_vector(&self) -> Option<&Vec<Vec<f32>>> {
        match self {
            Embedding::MultiVector(v) => Some(v),
            _ => None,
        }
    }

    /// Get the model type this embedding represents
    pub fn model_type(&self) -> ModelType {
        match self {
            Embedding::Dense(_) => ModelType::Dense,
            Embedding::Sparse(_) => ModelType::Sparse,
            Embedding::MultiVector(_) => ModelType::Colbert,
        }
    }
}

/// Model metadata resolved from Tessera registry
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model ID
    pub id: String,
    /// Model type
    pub model_type: ModelType,
    /// Embedding dimension
    pub dimension: usize,
    /// Maximum context length (tokens)
    pub context_length: usize,
    /// Human-readable name
    pub name: String,
}

impl ModelMetadata {
    /// Load metadata from Tessera model registry
    pub fn from_registry(model_id: &str) -> Result<Self> {
        let info = get_model(model_id)
            .ok_or_else(|| anyhow!("Model '{}' not found in Tessera registry", model_id))?;

        let model_type = match info.model_type {
            tessera::model_registry::ModelType::Dense => ModelType::Dense,
            tessera::model_registry::ModelType::Sparse => ModelType::Sparse,
            tessera::model_registry::ModelType::Colbert => ModelType::Colbert,
            _ => return Err(anyhow!("Unsupported model type for '{}'", model_id)),
        };

        Ok(Self {
            id: model_id.to_string(),
            model_type,
            dimension: info.embedding_dim.default_dim(),
            context_length: info.context_length,
            name: info.name.to_string(),
        })
    }

    /// Get a reference to the underlying ModelInfo
    pub fn registry_info(model_id: &str) -> Option<&'static ModelInfo> {
        get_model(model_id)
    }
}

/// Single embedder for one model type
pub struct Embedder {
    inner: EmbedderInner,
    metadata: ModelMetadata,
}

enum EmbedderInner {
    Dense(TesseraDense),
    Sparse(TesseraSparse),
    MultiVector(TesseraMultiVector),
}

impl Embedder {
    /// Create a new embedder from a vector configuration
    pub fn from_config(config: &VectorConfig) -> Result<Self> {
        let metadata = ModelMetadata::from_registry(&config.model)?;

        let inner = match metadata.model_type {
            ModelType::Dense => {
                let embedder = TesseraDense::new(&config.model)
                    .with_context(|| format!("Failed to load dense model '{}'", config.model))?;
                EmbedderInner::Dense(embedder)
            }
            ModelType::Sparse => {
                let embedder = TesseraSparse::new(&config.model)
                    .with_context(|| format!("Failed to load sparse model '{}'", config.model))?;
                EmbedderInner::Sparse(embedder)
            }
            ModelType::Colbert => {
                let embedder = TesseraMultiVector::new(&config.model)
                    .with_context(|| format!("Failed to load colbert model '{}'", config.model))?;
                EmbedderInner::MultiVector(embedder)
            }
        };

        Ok(Self { inner, metadata })
    }

    /// Create a new embedder from a model ID
    pub fn new(model_id: &str) -> Result<Self> {
        let config = VectorConfig {
            model: model_id.to_string(),
            enabled: true,
            max_chunk_size: None,
            overlap: None,
        };
        Self::from_config(&config)
    }

    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get the model type
    pub fn model_type(&self) -> ModelType {
        self.metadata.model_type
    }

    /// Get the embedding dimension
    pub fn dimension(&self) -> usize {
        self.metadata.dimension
    }

    /// Get the maximum context length in tokens
    pub fn context_length(&self) -> usize {
        self.metadata.context_length
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Embedding> {
        match &self.inner {
            EmbedderInner::Dense(embedder) => {
                let result = embedder.encode(text)
                    .map_err(|e| anyhow!("Dense encoding failed: {}", e))?;
                Ok(Embedding::Dense(result.embedding.to_vec()))
            }
            EmbedderInner::Sparse(embedder) => {
                let result = embedder.encode(text)
                    .map_err(|e| anyhow!("Sparse encoding failed: {}", e))?;
                Ok(Embedding::Sparse(result.weights))
            }
            EmbedderInner::MultiVector(embedder) => {
                let result = embedder.encode(text)
                    .map_err(|e| anyhow!("Multi-vector encoding failed: {}", e))?;
                let vectors: Vec<Vec<f32>> = result.embeddings
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();
                Ok(Embedding::MultiVector(vectors))
            }
        }
    }

    /// Embed multiple texts (batch processing for efficiency)
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        match &self.inner {
            EmbedderInner::Dense(embedder) => {
                let results = embedder.encode_batch(texts)
                    .map_err(|e| anyhow!("Dense batch encoding failed: {}", e))?;
                Ok(results.into_iter()
                    .map(|r| Embedding::Dense(r.embedding.to_vec()))
                    .collect())
            }
            EmbedderInner::Sparse(embedder) => {
                let results = embedder.encode_batch(texts)
                    .map_err(|e| anyhow!("Sparse batch encoding failed: {}", e))?;
                Ok(results.into_iter()
                    .map(|r| Embedding::Sparse(r.weights))
                    .collect())
            }
            EmbedderInner::MultiVector(embedder) => {
                let results = embedder.encode_batch(texts)
                    .map_err(|e| anyhow!("Multi-vector batch encoding failed: {}", e))?;
                Ok(results.into_iter()
                    .map(|r| {
                        let vectors: Vec<Vec<f32>> = r.embeddings
                            .rows()
                            .into_iter()
                            .map(|row| row.to_vec())
                            .collect();
                        Embedding::MultiVector(vectors)
                    })
                    .collect())
            }
        }
    }

    /// Compute similarity between two texts
    pub fn similarity(&self, text_a: &str, text_b: &str) -> Result<f32> {
        match &self.inner {
            EmbedderInner::Dense(embedder) => {
                embedder.similarity(text_a, text_b)
                    .map_err(|e| anyhow!("Dense similarity failed: {}", e))
            }
            EmbedderInner::Sparse(embedder) => {
                embedder.similarity(text_a, text_b)
                    .map_err(|e| anyhow!("Sparse similarity failed: {}", e))
            }
            EmbedderInner::MultiVector(embedder) => {
                embedder.similarity(text_a, text_b)
                    .map_err(|e| anyhow!("Multi-vector similarity failed: {}", e))
            }
        }
    }
}

/// Collection of named embedders for multi-vector embeddings
pub struct EmbedderSet {
    embedders: HashMap<String, Embedder>,
}

impl EmbedderSet {
    /// Create a new empty embedder set
    pub fn new() -> Self {
        Self {
            embedders: HashMap::new(),
        }
    }

    /// Create embedder set from configuration
    pub fn from_config(vectors: &HashMap<String, VectorConfig>) -> Result<Self> {
        let mut embedders = HashMap::new();

        for (name, config) in vectors.iter().filter(|(_, v)| v.enabled) {
            let embedder = Embedder::from_config(config)?;
            embedders.insert(name.clone(), embedder);
        }

        if embedders.is_empty() {
            return Err(anyhow!("No embedders configured"));
        }

        Ok(Self { embedders })
    }

    /// Create embedder set from configuration, filtered to specific vector names
    pub fn from_config_filtered(
        vectors: &HashMap<String, VectorConfig>,
        filter: &[String],
    ) -> Result<Self> {
        let mut embedders = HashMap::new();

        for (name, config) in vectors.iter().filter(|(_, v)| v.enabled) {
            // Only include if in the filter list
            if filter.iter().any(|f| f == name) {
                let embedder = Embedder::from_config(config)?;
                embedders.insert(name.clone(), embedder);
            }
        }

        if embedders.is_empty() {
            return Err(anyhow!(
                "No embedders found for filter {:?}. Available: {:?}",
                filter,
                vectors.keys().collect::<Vec<_>>()
            ));
        }

        Ok(Self { embedders })
    }

    /// Get an embedder by name
    pub fn get(&self, name: &str) -> Option<&Embedder> {
        self.embedders.get(name)
    }

    /// Get an embedder by model type (returns first match)
    pub fn get_by_type(&self, model_type: ModelType) -> Option<(&String, &Embedder)> {
        self.embedders
            .iter()
            .find(|(_, e)| e.model_type() == model_type)
    }

    /// Check if an embedder exists for a given name
    pub fn has(&self, name: &str) -> bool {
        self.embedders.contains_key(name)
    }

    /// Check if an embedder exists for a given model type
    pub fn has_type(&self, model_type: ModelType) -> bool {
        self.embedders.values().any(|e| e.model_type() == model_type)
    }

    /// Get all embedder names
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.embedders.keys()
    }

    /// Iterate over all embedders
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Embedder)> {
        self.embedders.iter()
    }

    /// Get the number of embedders
    pub fn len(&self) -> usize {
        self.embedders.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.embedders.is_empty()
    }

    /// Embed text with all embedders, returning a map of name -> embedding
    pub fn embed_all(&self, text: &str) -> Result<HashMap<String, Embedding>> {
        let mut results = HashMap::new();
        for (name, embedder) in &self.embedders {
            let embedding = embedder.embed(text)?;
            results.insert(name.clone(), embedding);
        }
        Ok(results)
    }

    /// Get the minimum context length across all embedders
    /// Useful for determining safe chunk sizes
    pub fn min_context_length(&self) -> usize {
        self.embedders
            .values()
            .map(|e| e.context_length())
            .min()
            .unwrap_or(512)
    }

    /// Get context length for a specific vector config
    /// Returns the model's context length if not overridden
    pub fn effective_chunk_size(&self, name: &str, config: &VectorConfig) -> usize {
        config.max_chunk_size.unwrap_or_else(|| {
            self.get(name)
                .map(|e| e.context_length())
                .unwrap_or(512)
        })
    }
}

impl Default for EmbedderSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_type_detection() {
        let dense = Embedding::Dense(vec![1.0, 2.0, 3.0]);
        assert_eq!(dense.model_type(), ModelType::Dense);
        assert_eq!(dense.dimension(), 3);
        assert_eq!(dense.vector_count(), 1);

        let sparse = Embedding::Sparse(vec![(1, 0.5), (5, 0.8)]);
        assert_eq!(sparse.model_type(), ModelType::Sparse);
        assert_eq!(sparse.vector_count(), 1);

        let multi = Embedding::MultiVector(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ]);
        assert_eq!(multi.model_type(), ModelType::Colbert);
        assert_eq!(multi.dimension(), 2);
        assert_eq!(multi.vector_count(), 3);
    }

    #[test]
    fn test_embedding_accessors() {
        let dense = Embedding::Dense(vec![1.0, 2.0, 3.0]);
        assert!(dense.as_dense().is_some());
        assert!(dense.as_sparse().is_none());
        assert!(dense.as_multi_vector().is_none());

        let sparse = Embedding::Sparse(vec![(1, 0.5)]);
        assert!(sparse.as_dense().is_none());
        assert!(sparse.as_sparse().is_some());

        let multi = Embedding::MultiVector(vec![vec![1.0]]);
        assert!(multi.as_multi_vector().is_some());
    }

    #[test]
    fn test_model_metadata_lookup() {
        // This test requires Tessera to be properly configured
        // Skip if model not available
        if let Ok(meta) = ModelMetadata::from_registry("bge-base-en-v1.5") {
            assert_eq!(meta.model_type, ModelType::Dense);
            assert!(meta.context_length > 0);
            assert!(meta.dimension > 0);
        }
    }
}
