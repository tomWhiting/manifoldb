//! Embedding service for generating text embeddings.
//!
//! This module provides a service for generating embeddings from text using
//! the tessera-embeddings library. The service is designed to work with
//! async GraphQL by running embedding operations on blocking threads.

use std::sync::Mutex;

use tessera::TesseraDense;
use thiserror::Error;

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbeddingError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    #[error("Failed to encode text: {0}")]
    EncodeError(String),

    #[error("Model lock error")]
    LockError,
}

/// Information about a supported embedding model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub dimension: i32,
    pub model_type: &'static str,
}

/// Service for generating text embeddings.
///
/// The service uses a Mutex to ensure thread-safe access to the model cache.
/// Embedding operations are inherently blocking (model inference), so callers
/// should use `spawn_blocking` or similar mechanisms in async contexts.
pub struct EmbeddingService {
    /// The currently loaded model (single model at a time for simplicity).
    /// The inner Option allows lazy loading.
    current_model: Mutex<Option<(String, TesseraDense)>>,
}

// SAFETY: EmbeddingService is Send because Mutex<T> is Send when T is Send,
// and the inner types are Send. We use Mutex instead of RwLock because
// TesseraDense may not be Sync (candle tensors).
unsafe impl Sync for EmbeddingService {}

impl EmbeddingService {
    /// Create a new embedding service.
    pub fn new() -> Self {
        Self {
            current_model: Mutex::new(None),
        }
    }

    /// Encode text to a dense embedding vector.
    ///
    /// The model is lazy-loaded on first use. If a different model is requested,
    /// the previous model is unloaded and the new one is loaded.
    ///
    /// Note: This is a blocking operation. In async contexts, use spawn_blocking.
    pub fn encode(&self, text: &str, model_id: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut guard = self.current_model.lock().map_err(|_| EmbeddingError::LockError)?;

        // Check if we need to load a different model
        let needs_load = match &*guard {
            Some((id, _)) if id == model_id => false,
            _ => true,
        };

        if needs_load {
            let model = TesseraDense::new(model_id)
                .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?;
            *guard = Some((model_id.to_string(), model));
        }

        // Now encode
        let (_, model) = guard.as_ref().ok_or(EmbeddingError::LockError)?;
        let embedding = model
            .encode(text)
            .map_err(|e| EmbeddingError::EncodeError(e.to_string()))?;

        // Convert ndarray to Vec<f32>
        Ok(embedding.embedding.to_vec())
    }

    /// Encode multiple texts to dense embedding vectors.
    ///
    /// More efficient than encoding one at a time for batch operations.
    pub fn encode_batch(
        &self,
        texts: &[&str],
        model_id: &str,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut guard = self.current_model.lock().map_err(|_| EmbeddingError::LockError)?;

        // Check if we need to load a different model
        let needs_load = match &*guard {
            Some((id, _)) if id == model_id => false,
            _ => true,
        };

        if needs_load {
            let model = TesseraDense::new(model_id)
                .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?;
            *guard = Some((model_id.to_string(), model));
        }

        // Now encode batch
        let (_, model) = guard.as_ref().ok_or(EmbeddingError::LockError)?;
        let embeddings = model
            .encode_batch(texts)
            .map_err(|e| EmbeddingError::EncodeError(e.to_string()))?;

        // Convert each ndarray to Vec<f32>
        Ok(embeddings.into_iter().map(|e| e.embedding.to_vec()).collect())
    }

    /// List available embedding models.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        use tessera::model_registry::{
            BGE_BASE_EN_V1_5, JINA_EMBEDDINGS_V2_BASE_EN, JINA_EMBEDDINGS_V2_SMALL_EN,
            JINA_EMBEDDINGS_V3, NOMIC_EMBED_V1_5, SNOWFLAKE_ARCTIC_L,
        };

        vec![
            ModelInfo {
                id: BGE_BASE_EN_V1_5.id,
                name: BGE_BASE_EN_V1_5.name,
                dimension: BGE_BASE_EN_V1_5.hidden_dim as i32,
                model_type: "dense",
            },
            ModelInfo {
                id: NOMIC_EMBED_V1_5.id,
                name: NOMIC_EMBED_V1_5.name,
                dimension: NOMIC_EMBED_V1_5.hidden_dim as i32,
                model_type: "dense",
            },
            ModelInfo {
                id: JINA_EMBEDDINGS_V2_SMALL_EN.id,
                name: JINA_EMBEDDINGS_V2_SMALL_EN.name,
                dimension: JINA_EMBEDDINGS_V2_SMALL_EN.hidden_dim as i32,
                model_type: "dense",
            },
            ModelInfo {
                id: JINA_EMBEDDINGS_V2_BASE_EN.id,
                name: JINA_EMBEDDINGS_V2_BASE_EN.name,
                dimension: JINA_EMBEDDINGS_V2_BASE_EN.hidden_dim as i32,
                model_type: "dense",
            },
            ModelInfo {
                id: JINA_EMBEDDINGS_V3.id,
                name: JINA_EMBEDDINGS_V3.name,
                dimension: JINA_EMBEDDINGS_V3.hidden_dim as i32,
                model_type: "dense",
            },
            ModelInfo {
                id: SNOWFLAKE_ARCTIC_L.id,
                name: SNOWFLAKE_ARCTIC_L.name,
                dimension: SNOWFLAKE_ARCTIC_L.hidden_dim as i32,
                model_type: "dense",
            },
        ]
    }

    /// Get the default model ID.
    pub fn default_model(&self) -> &'static str {
        "jina-embeddings-v2-small-en"
    }
}

impl Default for EmbeddingService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models() {
        let service = EmbeddingService::new();
        let models = service.list_models();
        assert!(!models.is_empty());
    }
}
