//! Configuration for gimbal
//!
//! Supports multi-vector embeddings with named vector configurations,
//! each with its own model, chunking settings, and search behavior.
//!
//! # Example Configuration
//!
//! ```toml
//! [database]
//! path = "embeddings.manifold"
//!
//! [chunking]
//! split_on_headers = true
//! header_levels = [1, 2, 3]
//! overlap = 50
//!
//! # Named vectors - use any names you want
//! [vectors.docs]
//! model = "bge-base-en-v1.5"
//!
//! [vectors.code]
//! model = "nomic-embed-v1.5"
//!
//! [vectors.sparse]
//! model = "splade-v3"
//!
//! [vectors.colbert]
//! model = "colbert-v2"
//!
//! # Ingestion sources with vector assignments
//! [[ingest.sources]]
//! name = "documentation"
//! paths = ["./docs", "./research"]
//! extensions = ["md", "txt", "rst"]
//! exclude = [".git", "node_modules"]
//! vectors = ["docs", "sparse", "colbert"]
//!
//! [[ingest.sources]]
//! name = "codebase"
//! paths = ["./src", "./lib"]
//! extensions = ["rs", "py", "ts"]
//! exclude = ["target", "__pycache__"]
//! vectors = ["code", "sparse", "colbert"]
//!
//! # Default for CLI one-off ingests
//! [ingest.default]
//! vectors = ["docs", "sparse"]
//!
//! [search]
//! default_mode = "hybrid"
//!
//! [search.hybrid]
//! dense_weight = 0.7
//! sparse_weight = 0.3
//! fusion = "rrf"
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Main configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Database configuration
    pub database: DatabaseConfig,

    /// Default chunking configuration (inherited by vectors unless overridden)
    #[serde(default)]
    pub chunking: ChunkingConfig,

    /// Named vector configurations
    #[serde(default)]
    pub vectors: HashMap<String, VectorConfig>,

    /// Ingestion configuration with sources
    #[serde(default)]
    pub ingest: IngestConfig,

    /// Search configuration
    #[serde(default)]
    pub search: SearchConfig,
}

impl Default for Config {
    fn default() -> Self {
        let mut vectors = HashMap::new();
        vectors.insert(
            "dense".to_string(),
            VectorConfig {
                model: "bge-base-en-v1.5".to_string(),
                enabled: true,
                max_chunk_size: None,
                overlap: None,
            },
        );

        Self {
            database: DatabaseConfig::default(),
            chunking: ChunkingConfig::default(),
            vectors,
            ingest: IngestConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a TOML file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a TOML file
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(anyhow!("At least one vector configuration is required"));
        }

        for (name, vector) in &self.vectors {
            vector.validate().map_err(|e| anyhow!("Vector '{}': {}", name, e))?;
        }

        self.ingest.validate(&self.vectors)?;
        self.search.validate()?;
        Ok(())
    }

    /// Get a source by name
    pub fn get_source(&self, name: &str) -> Option<&IngestSource> {
        self.ingest.sources.iter().find(|s| s.name == name)
    }

    /// Get vectors to use for a source (or default if not specified)
    pub fn vectors_for_source(&self, source: &IngestSource) -> Vec<String> {
        if source.vectors.is_empty() {
            // Use all enabled vectors
            self.vectors
                .iter()
                .filter(|(_, v)| v.enabled)
                .map(|(name, _)| name.clone())
                .collect()
        } else {
            source.vectors.clone()
        }
    }

    /// Get default vectors for CLI one-off ingests
    pub fn default_ingest_vectors(&self) -> Vec<String> {
        if self.ingest.default.vectors.is_empty() {
            // Use all enabled vectors
            self.vectors
                .iter()
                .filter(|(_, v)| v.enabled)
                .map(|(name, _)| name.clone())
                .collect()
        } else {
            self.ingest.default.vectors.clone()
        }
    }

    /// Get all enabled vector configurations
    pub fn enabled_vectors(&self) -> impl Iterator<Item = (&String, &VectorConfig)> {
        self.vectors.iter().filter(|(_, v)| v.enabled)
    }

    /// Get a specific vector configuration by name
    pub fn get_vector(&self, name: &str) -> Option<&VectorConfig> {
        self.vectors.get(name).filter(|v| v.enabled)
    }

    /// Check if a specific vector type is configured and enabled
    pub fn has_vector(&self, name: &str) -> bool {
        self.get_vector(name).is_some()
    }
}

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the ManifoldDB database
    pub path: PathBuf,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("embeddings.manifold"),
        }
    }
}

/// Ingestion configuration with named sources
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestConfig {
    /// Named ingestion sources
    #[serde(default)]
    pub sources: Vec<IngestSource>,

    /// Default configuration for CLI one-off ingests
    #[serde(default)]
    pub default: IngestDefault,
}

impl IngestConfig {
    /// Validate ingestion configuration
    pub fn validate(&self, vectors: &HashMap<String, VectorConfig>) -> Result<()> {
        for source in &self.sources {
            source.validate(vectors)?;
        }
        self.default.validate(vectors)?;
        Ok(())
    }
}

/// A named ingestion source with paths and vector assignments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSource {
    /// Name of this source (e.g., "documentation", "codebase")
    pub name: String,

    /// Paths to crawl for this source
    #[serde(default)]
    pub paths: Vec<PathBuf>,

    /// File extensions to include (e.g., ["md", "txt", "rs"])
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,

    /// Patterns to exclude (e.g., [".git", "node_modules", "target"])
    #[serde(default)]
    pub exclude: Vec<String>,

    /// Vector names to use for this source (empty = use all enabled vectors)
    #[serde(default)]
    pub vectors: Vec<String>,
}

fn default_extensions() -> Vec<String> {
    vec!["md".to_string(), "txt".to_string()]
}

impl IngestSource {
    /// Validate source configuration
    pub fn validate(&self, vectors: &HashMap<String, VectorConfig>) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow!("Source name cannot be empty"));
        }
        if self.paths.is_empty() {
            return Err(anyhow!("Source '{}': at least one path is required", self.name));
        }
        // Validate that all specified vectors exist
        for vec_name in &self.vectors {
            if !vectors.contains_key(vec_name) {
                return Err(anyhow!(
                    "Source '{}': unknown vector '{}'. Available: {:?}",
                    self.name,
                    vec_name,
                    vectors.keys().collect::<Vec<_>>()
                ));
            }
        }
        Ok(())
    }

    /// Check if a path should be excluded
    pub fn should_exclude(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        self.exclude.iter().any(|pattern| path_str.contains(pattern))
    }

    /// Check if a file extension is included
    pub fn matches_extension(&self, path: &Path) -> bool {
        if self.extensions.is_empty() {
            return true;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.extensions.iter().any(|e| e == ext))
            .unwrap_or(false)
    }
}

/// Default configuration for CLI one-off ingests
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestDefault {
    /// Default vectors to use (empty = use all enabled vectors)
    #[serde(default)]
    pub vectors: Vec<String>,

    /// Default extensions for CLI ingests
    #[serde(default = "default_extensions")]
    pub extensions: Vec<String>,

    /// Default exclude patterns
    #[serde(default)]
    pub exclude: Vec<String>,
}

impl IngestDefault {
    /// Validate default configuration
    pub fn validate(&self, vectors: &HashMap<String, VectorConfig>) -> Result<()> {
        for vec_name in &self.vectors {
            if !vectors.contains_key(vec_name) {
                return Err(anyhow!(
                    "Default ingest: unknown vector '{}'. Available: {:?}",
                    vec_name,
                    vectors.keys().collect::<Vec<_>>()
                ));
            }
        }
        Ok(())
    }
}

/// Named vector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorConfig {
    /// Model ID (e.g., "bge-base-en-v1.5", "splade-cocondenser", "colbert-v2")
    pub model: String,

    /// Whether this vector is enabled (default: true)
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Maximum chunk size in tokens (default: model's context_length)
    /// If not specified, will be determined from model metadata
    pub max_chunk_size: Option<usize>,

    /// Overlap between chunks in tokens (overrides global chunking.overlap)
    pub overlap: Option<usize>,
}

fn default_enabled() -> bool {
    true
}

impl VectorConfig {
    /// Validate the vector configuration
    pub fn validate(&self) -> Result<()> {
        if self.model.is_empty() {
            return Err(anyhow!("Model ID cannot be empty"));
        }
        Ok(())
    }

    /// Get the model type from the model ID
    pub fn model_type(&self) -> ModelType {
        // Determine type from model ID patterns
        let model_lower = self.model.to_lowercase();
        if model_lower.contains("splade") {
            ModelType::Sparse
        } else if model_lower.contains("colbert") {
            ModelType::Colbert
        } else {
            ModelType::Dense
        }
    }
}

/// Model type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Dense,
    Sparse,
    Colbert,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Dense => write!(f, "dense"),
            ModelType::Sparse => write!(f, "sparse"),
            ModelType::Colbert => write!(f, "colbert"),
        }
    }
}

/// Chunking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Split on headers (h1, h2, h3, etc.)
    #[serde(default = "default_split_on_headers")]
    pub split_on_headers: bool,

    /// Header levels to split on (1 = h1, 2 = h2, etc.)
    #[serde(default = "default_header_levels")]
    pub header_levels: Vec<u8>,

    /// Default overlap between chunks in tokens
    #[serde(default)]
    pub overlap: usize,
}

fn default_split_on_headers() -> bool {
    true
}

fn default_header_levels() -> Vec<u8> {
    vec![1, 2, 3]
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            split_on_headers: true,
            header_levels: vec![1, 2, 3],
            overlap: 50,
        }
    }
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Default search mode
    #[serde(default = "default_search_mode")]
    pub default_mode: SearchMode,

    /// Hybrid search configuration
    #[serde(default)]
    pub hybrid: HybridConfig,

    /// Reranking configuration
    #[serde(default)]
    pub rerank: RerankConfig,
}

fn default_search_mode() -> SearchMode {
    SearchMode::Dense
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_mode: SearchMode::Dense,
            hybrid: HybridConfig::default(),
            rerank: RerankConfig::default(),
        }
    }
}

impl SearchConfig {
    /// Validate search configuration
    pub fn validate(&self) -> Result<()> {
        self.hybrid.validate()?;
        Ok(())
    }
}

/// Search mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Dense vector similarity search
    Dense,
    /// Sparse (SPLADE) search
    Sparse,
    /// Hybrid search combining dense and sparse
    Hybrid,
    /// ColBERT late interaction search
    Colbert,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchMode::Dense => write!(f, "dense"),
            SearchMode::Sparse => write!(f, "sparse"),
            SearchMode::Hybrid => write!(f, "hybrid"),
            SearchMode::Colbert => write!(f, "colbert"),
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "dense" => Ok(SearchMode::Dense),
            "sparse" => Ok(SearchMode::Sparse),
            "hybrid" => Ok(SearchMode::Hybrid),
            "colbert" => Ok(SearchMode::Colbert),
            _ => Err(anyhow!("Unknown search mode: {}. Valid modes: dense, sparse, hybrid, colbert", s)),
        }
    }
}

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Weight for dense scores (0.0 - 1.0)
    #[serde(default = "default_dense_weight")]
    pub dense_weight: f64,

    /// Weight for sparse scores (0.0 - 1.0)
    #[serde(default = "default_sparse_weight")]
    pub sparse_weight: f64,

    /// Fusion strategy
    #[serde(default)]
    pub fusion: FusionStrategy,
}

fn default_dense_weight() -> f64 {
    0.7
}

fn default_sparse_weight() -> f64 {
    0.3
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            dense_weight: 0.7,
            sparse_weight: 0.3,
            fusion: FusionStrategy::default(),
        }
    }
}

impl HybridConfig {
    /// Validate hybrid configuration
    pub fn validate(&self) -> Result<()> {
        if self.dense_weight < 0.0 || self.dense_weight > 1.0 {
            return Err(anyhow!("dense_weight must be between 0.0 and 1.0"));
        }
        if self.sparse_weight < 0.0 || self.sparse_weight > 1.0 {
            return Err(anyhow!("sparse_weight must be between 0.0 and 1.0"));
        }
        Ok(())
    }
}

/// Fusion strategy for combining search results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion - combines rankings rather than scores
    #[default]
    Rrf,
    /// Weighted sum of normalized scores
    WeightedSum,
}

impl std::fmt::Display for FusionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionStrategy::Rrf => write!(f, "rrf"),
            FusionStrategy::WeightedSum => write!(f, "weighted_sum"),
        }
    }
}

/// Reranking configuration
///
/// NOTE: Reranking requires a cross-encoder model (e.g., ms-marco-MiniLM, bge-reranker).
/// Check Tessera for available reranker models before enabling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankConfig {
    /// Whether reranking is enabled (requires cross-encoder model support in Tessera)
    #[serde(default)]
    pub enabled: bool,

    /// Model to use for reranking (must be a cross-encoder, NOT ColBERT)
    #[serde(default = "default_rerank_model")]
    pub model: String,

    /// Number of candidates to retrieve before reranking
    #[serde(default = "default_candidates")]
    pub candidates: usize,
}

fn default_rerank_model() -> String {
    // Placeholder - check Tessera for actual cross-encoder support
    String::new()
}

fn default_candidates() -> usize {
    100
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: String::new(), // Not yet implemented - needs cross-encoder support
            candidates: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.vectors.contains_key("dense"));
        assert_eq!(config.search.default_mode, SearchMode::Dense);
    }

    #[test]
    fn test_model_type_detection() {
        let dense = VectorConfig {
            model: "bge-base-en-v1.5".to_string(),
            enabled: true,
            max_chunk_size: None,
            overlap: None,
        };
        assert_eq!(dense.model_type(), ModelType::Dense);

        let sparse = VectorConfig {
            model: "splade-cocondenser".to_string(),
            enabled: true,
            max_chunk_size: None,
            overlap: None,
        };
        assert_eq!(sparse.model_type(), ModelType::Sparse);

        let colbert = VectorConfig {
            model: "colbert-v2".to_string(),
            enabled: true,
            max_chunk_size: None,
            overlap: None,
        };
        assert_eq!(colbert.model_type(), ModelType::Colbert);
    }

    #[test]
    fn test_search_mode_parsing() {
        assert_eq!("dense".parse::<SearchMode>().unwrap(), SearchMode::Dense);
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("COLBERT".parse::<SearchMode>().unwrap(), SearchMode::Colbert);
        assert!("invalid".parse::<SearchMode>().is_err());
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        // Empty vectors should fail
        config.vectors.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_hybrid_validation() {
        let mut hybrid = HybridConfig::default();
        assert!(hybrid.validate().is_ok());

        hybrid.dense_weight = 1.5;
        assert!(hybrid.validate().is_err());

        hybrid.dense_weight = -0.1;
        assert!(hybrid.validate().is_err());
    }
}
