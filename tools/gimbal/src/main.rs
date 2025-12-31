//! Gimbal - CLI for embedding documents with Tessera and storing in ManifoldDB
//!
//! Supports multiple embedding paradigms (dense, sparse, ColBERT) with configurable
//! search modes including hybrid search with RRF fusion.

mod chunk;
mod config;
mod embed;
mod ingest;
mod store;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::SearchMode;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "gimbal")]
#[command(about = "Embed documents with Tessera and store in ManifoldDB")]
#[command(version)]
struct Cli {
    /// Path to config file (default: gimbal.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Database path
    #[arg(short, long, global = true)]
    db: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest documents from configured sources, a directory, file, or stdin
    ///
    /// When no path is given, ingests from all configured sources in the config file.
    /// Use --source to ingest from a specific configured source.
    Ingest {
        /// Directory or file to ingest (omit to use configured sources)
        path: Option<PathBuf>,

        /// Ingest from a specific configured source by name
        #[arg(short, long)]
        source: Option<String>,

        /// Read content from stdin instead of a file
        #[arg(long)]
        stdin: bool,

        /// Logical file path when reading from stdin (for graph organization)
        #[arg(long)]
        file: Option<String>,

        /// Section heading when reading from stdin
        #[arg(long)]
        heading: Option<String>,

        /// File extensions to include (overrides config, default: md, txt)
        #[arg(short, long)]
        ext: Option<Vec<String>>,

        /// Vectors to use (overrides config, default: all enabled)
        #[arg(short, long)]
        vectors: Option<Vec<String>>,

        /// Category tag for all ingested documents
        #[arg(long)]
        category: Option<String>,

        /// Topic tag for all ingested documents
        #[arg(long)]
        topic: Option<String>,

        /// Additional key=value properties
        #[arg(long, value_parser = parse_property)]
        prop: Option<Vec<(String, String)>>,
    },

    /// Search for similar documents
    Search {
        /// Query text
        query: String,

        /// Maximum results to return
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Show full chunk content instead of preview
        #[arg(short, long)]
        full: bool,

        /// Search mode: dense, sparse, hybrid, colbert
        #[arg(short, long)]
        mode: Option<String>,

        /// Filter by category
        #[arg(long)]
        category: Option<String>,

        /// Filter by topic
        #[arg(long)]
        topic: Option<String>,
    },

    /// Initialize a new config file
    Init {
        /// Output path for config file
        #[arg(default_value = "gimbal.toml")]
        path: PathBuf,

        /// Include example multi-vector configuration
        #[arg(long)]
        multi_vector: bool,
    },

    /// List available embedding models
    Models {
        /// Filter by model type: dense, sparse, colbert
        #[arg(short, long)]
        r#type: Option<String>,
    },
}

fn parse_property(s: &str) -> Result<(String, String), String> {
    let parts: Vec<&str> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid property format: {s}. Use key=value"));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

fn main() {
    if let Err(e) = run() {
        // Display root cause directly for clear error messages
        let chain: Vec<_> = e.chain().collect();
        if let Some(root_cause) = chain.last() {
            eprintln!("Error: {}", root_cause);
        }
        // Show context chain if there are additional layers
        if chain.len() > 1 {
            eprintln!("\nContext:");
            for cause in chain.iter().take(chain.len() - 1) {
                eprintln!("  - {}", cause);
            }
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    // Load config
    let config_path = cli.config.unwrap_or_else(|| PathBuf::from("gimbal.toml"));
    let mut cfg = if config_path.exists() {
        config::Config::load(&config_path)?
    } else {
        config::Config::default()
    };

    // Override db path from CLI if provided
    if let Some(db) = cli.db {
        cfg.database.path = db;
    }

    match cli.command {
        Commands::Ingest { path, source, stdin, file, heading, ext, vectors, category, topic, prop } => {
            let properties = build_properties(category, topic, prop);

            if stdin {
                // Pipe mode - read from stdin
                let file_path = file.unwrap_or_else(|| "stdin".to_string());
                let vector_filter = vectors.or_else(|| Some(cfg.default_ingest_vectors()));
                ingest::pipe_stdin(&cfg, &file_path, heading.as_deref(), &properties, vector_filter.as_deref())?;
            } else if let Some(source_name) = source {
                // Ingest from a specific configured source
                ingest::ingest_source(&cfg, &source_name, &properties)?;
            } else if let Some(path) = path {
                // CLI path provided - use default vectors
                let extensions = ext.unwrap_or_else(|| cfg.ingest.default.extensions.clone());
                let extensions = if extensions.is_empty() {
                    vec!["md".to_string(), "txt".to_string()]
                } else {
                    extensions
                };
                let exclude = cfg.ingest.default.exclude.clone();
                let vector_filter = vectors.or_else(|| Some(cfg.default_ingest_vectors()));
                ingest::ingest_path(&cfg, &path, &extensions, &exclude, &properties, vector_filter.as_deref())?;
            } else if !cfg.ingest.sources.is_empty() {
                // No path, no source specified - run all configured sources
                ingest::ingest_all_sources(&cfg, &properties)?;
            } else {
                // No sources configured and no path - show help
                println!("No path provided and no sources configured.");
                println!("Either:");
                println!("  - Provide a path: gimbal ingest ./docs");
                println!("  - Configure sources in gimbal.toml");
                println!("  - Use stdin: echo 'text' | gimbal ingest --stdin");
            }
        }

        Commands::Search { query, limit, full, mode, category, topic } => {
            let filters = build_filters(category, topic);
            let search_mode = mode.map(|m| m.parse::<SearchMode>()).transpose()?;
            ingest::search(&cfg, &query, limit, full, search_mode, &filters)?;
        }

        Commands::Init { path, multi_vector } => {
            let content = if multi_vector {
                generate_multi_vector_config_content()
            } else {
                generate_default_config_content()
            };
            std::fs::write(&path, &content)?;
            println!("Created config file: {}", path.display());
            if multi_vector {
                println!("  Configured with dense, sparse, and ColBERT vectors");
            }
            println!("  Run 'gimbal models' to see all available embedding models");
        }

        Commands::Models { r#type } => {
            list_models(r#type.as_deref())?;
        }
    }

    Ok(())
}

fn build_properties(
    category: Option<String>,
    topic: Option<String>,
    props: Option<Vec<(String, String)>>,
) -> Vec<(String, String)> {
    let mut result = Vec::new();
    if let Some(c) = category {
        result.push(("category".to_string(), c));
    }
    if let Some(t) = topic {
        result.push(("topic".to_string(), t));
    }
    if let Some(p) = props {
        result.extend(p);
    }
    result
}

fn build_filters(category: Option<String>, topic: Option<String>) -> Vec<(String, String)> {
    build_properties(category, topic, None)
}

/// Generate default config content with helpful comments
fn generate_default_config_content() -> String {
    r##"# Gimbal Configuration
# Embed documents with multiple vector types and store in ManifoldDB

[database]
path = "embeddings.manifold"

# =============================================================================
# CHUNKING - Automatic content-aware splitting
# =============================================================================
# Gimbal automatically detects file types and uses appropriate chunking:
#   - Markdown (.md): Split on headers
#   - Code (.rs, .py, .ts, etc.): Tree-sitter AST-based symbol extraction
#   - Other text: Paragraph-based splitting

[chunking]
# Markdown options
split_on_headers = true
header_levels = [1, 2, 3]
overlap = 50

# Code options (tree-sitter based)
code_enabled = true
code_max_size = 8000  # Max bytes per code chunk

# Vector configurations - name them whatever you want!
# Each vector name becomes a separate embedding namespace.
[vectors.dense]
model = "bge-base-en-v1.5"
enabled = true
# max_chunk_size = 512  # optional, defaults to model's context length
# overlap = 50          # optional, overrides chunking.overlap

# Ingestion sources (optional - you can also use CLI directly)
# [[ingest.sources]]
# name = "documentation"
# paths = ["./docs"]
# extensions = ["md", "txt", "rs", "py"]
# exclude = [".git", "node_modules"]
# vectors = ["dense"]  # which vectors to use for this source

# Default vectors for CLI one-off ingests
[ingest.default]
vectors = ["dense"]
extensions = ["md", "txt", "rs", "py", "ts", "js", "go"]
exclude = [".git", "node_modules", "target", "__pycache__", ".venv"]

[search]
default_mode = "dense"

# Hybrid search combines dense and sparse results
[search.hybrid]
dense_weight = 0.7
sparse_weight = 0.3
fusion = "rrf"  # or "weighted_sum"

# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================
# Code (tree-sitter AST parsing):
#   Rust (.rs), Python (.py), TypeScript (.ts), JavaScript (.js),
#   TSX (.tsx), Go (.go), C (.c, .h), C++ (.cpp, .hpp),
#   JSON, YAML, CSS, Bash (.sh)
#
# Documents (markdown-aware):
#   Markdown (.md), Plain text (.txt)
#
# =============================================================================
# AVAILABLE EMBEDDING MODELS (from Tessera)
# Run 'gimbal models' for the full list with dimensions
# =============================================================================
#
# DENSE MODELS (semantic similarity):
#   bge-base-en-v1.5     - 768 dim, 512 ctx  - Strong baseline, fast
#   nomic-embed-v1.5     - 768 dim, 8K ctx   - Matryoshka support
#   snowflake-arctic-l   - 1024 dim, 512 ctx - High performance, Matryoshka
#   jina-embeddings-v3   - 1024 dim, 8K ctx  - Multilingual (89 langs)
#   gte-qwen2-7b         - 3584 dim, 32K ctx - State-of-art, Matryoshka
#   qwen3-embedding-8b   - 4096 dim, 32K ctx - Latest, #1 MTEB multilingual
#   qwen3-embedding-4b   - 2560 dim, 32K ctx - Efficient, strong performance
#   qwen3-embedding-0.6b - 1024 dim, 32K ctx - Compact multilingual
#
# SPARSE MODELS (keyword/lexical matching):
#   splade-v3            - vocab dim, 512 ctx - Best sparse retrieval
#   splade-pp-en-v1      - vocab dim, 512 ctx - Efficient variant
#   splade-pp-en-v2      - vocab dim, 512 ctx - Improved v1
#   minicoil-v1          - 4 dim, 512 ctx     - Ultra-compact sparse
#
# COLBERT MODELS (multi-vector late interaction):
#   colbert-v2           - 128 dim, 512 ctx  - Original Stanford baseline
#   colbert-small        - 96 dim, 512 ctx   - Compact, fast inference
#   jina-colbert-v2      - 768 dim, 8K ctx   - Multilingual, Matryoshka
#   jina-colbert-v2-96   - 96 dim, 8K ctx    - Compact variant
#   gte-modern-colbert   - 768 dim, 8K ctx   - ModernBERT, best performance
#
# UNIFIED MODELS (dense + sparse + colbert in one):
#   bge-m3-multi         - 1024 dim, 8K ctx  - All three modes, 100+ langs
#
# =============================================================================
"##.to_string()
}

/// Generate multi-vector config content with helpful comments
fn generate_multi_vector_config_content() -> String {
    r##"# Gimbal Configuration - Multi-Vector Setup
# Uses dense, sparse, and ColBERT for comprehensive retrieval

[database]
path = "embeddings.manifold"

# =============================================================================
# CHUNKING - Automatic content-aware splitting
# =============================================================================
# Gimbal automatically detects file types and uses appropriate chunking:
#   - Markdown (.md): Split on headers
#   - Code (.rs, .py, .ts, etc.): Tree-sitter AST-based symbol extraction
#   - Other text: Paragraph-based splitting

[chunking]
# Markdown options
split_on_headers = true
header_levels = [1, 2, 3]
overlap = 50

# Code options (tree-sitter based)
code_enabled = true
code_max_size = 8000  # Max bytes per code chunk

# Vector configurations - you can rename these to anything you want!
# Examples: "docs", "code", "research", "semantic", "keywords", etc.

# Dense embeddings for semantic similarity
[vectors.dense]
model = "bge-base-en-v1.5"
enabled = true

# Sparse embeddings for keyword matching (SPLADE)
[vectors.sparse]
model = "splade-v3"
enabled = true
max_chunk_size = 256  # SPLADE works well with shorter chunks
overlap = 25

# ColBERT for fine-grained late interaction
[vectors.colbert]
model = "colbert-v2"
enabled = true

# =============================================================================
# EXAMPLE: Source-based ingestion with vector assignments
# =============================================================================
#
# [[ingest.sources]]
# name = "documentation"
# paths = ["./docs", "./research"]
# extensions = ["md", "txt", "rst"]
# exclude = [".git", "node_modules"]
# vectors = ["dense", "sparse", "colbert"]  # all vectors for docs
#
# [[ingest.sources]]
# name = "codebase"
# paths = ["./src", "./lib"]
# extensions = ["rs", "py", "ts", "js"]
# exclude = ["target", "__pycache__", "node_modules"]
# vectors = ["dense", "sparse"]  # skip colbert for code (overkill)
#
# =============================================================================

# Default vectors for CLI one-off ingests
[ingest.default]
vectors = ["dense", "sparse", "colbert"]
extensions = ["md", "txt"]
exclude = [".git", "node_modules", "target", "__pycache__"]

[search]
default_mode = "hybrid"

# Hybrid search combines dense + sparse with RRF fusion
[search.hybrid]
dense_weight = 0.7
sparse_weight = 0.3
fusion = "rrf"

# =============================================================================
# AVAILABLE EMBEDDING MODELS (from Tessera)
# Run 'gimbal models' for the full list with dimensions
# =============================================================================
#
# DENSE MODELS (semantic similarity):
#   bge-base-en-v1.5     - 768 dim, 512 ctx  - Strong baseline, fast
#   nomic-embed-v1.5     - 768 dim, 8K ctx   - Matryoshka support
#   snowflake-arctic-l   - 1024 dim, 512 ctx - High performance, Matryoshka
#   jina-embeddings-v3   - 1024 dim, 8K ctx  - Multilingual (89 langs)
#   gte-qwen2-7b         - 3584 dim, 32K ctx - State-of-art, Matryoshka
#   qwen3-embedding-8b   - 4096 dim, 32K ctx - Latest, #1 MTEB multilingual
#   qwen3-embedding-4b   - 2560 dim, 32K ctx - Efficient, strong performance
#   qwen3-embedding-0.6b - 1024 dim, 32K ctx - Compact multilingual
#
# SPARSE MODELS (keyword/lexical matching):
#   splade-v3            - vocab dim, 512 ctx - Best sparse retrieval
#   splade-pp-en-v1      - vocab dim, 512 ctx - Efficient variant
#   splade-pp-en-v2      - vocab dim, 512 ctx - Improved v1
#   minicoil-v1          - 4 dim, 512 ctx     - Ultra-compact sparse
#
# COLBERT MODELS (multi-vector late interaction):
#   colbert-v2           - 128 dim, 512 ctx  - Original Stanford baseline
#   colbert-small        - 96 dim, 512 ctx   - Compact, fast inference
#   jina-colbert-v2      - 768 dim, 8K ctx   - Multilingual, Matryoshka
#   jina-colbert-v2-96   - 96 dim, 8K ctx    - Compact variant
#   gte-modern-colbert   - 768 dim, 8K ctx   - ModernBERT, best performance
#
# UNIFIED MODELS (dense + sparse + colbert in one):
#   bge-m3-multi         - 1024 dim, 8K ctx  - All three modes, 100+ langs
#
# =============================================================================
"##.to_string()
}

/// List available embedding models from Tessera registry
fn list_models(type_filter: Option<&str>) -> Result<()> {
    use tessera::model_registry::{models_by_type, ModelType, MODEL_REGISTRY};

    let models: Vec<_> = match type_filter {
        Some("dense") => models_by_type(ModelType::Dense).to_vec(),
        Some("sparse") => models_by_type(ModelType::Sparse).to_vec(),
        Some("colbert") => models_by_type(ModelType::Colbert).to_vec(),
        Some(t) => {
            println!("Unknown model type: {}. Use: dense, sparse, colbert", t);
            return Ok(());
        }
        None => MODEL_REGISTRY.iter().collect(),
    };

    if models.is_empty() {
        println!("No models found.");
        return Ok(());
    }

    println!("Available embedding models:\n");
    println!("{:<25} {:<10} {:<8} {:<8} {}", "ID", "Type", "Dim", "Context", "Name");
    println!("{}", "-".repeat(80));

    for model in models {
        let type_str = match model.model_type {
            ModelType::Dense => "dense",
            ModelType::Sparse => "sparse",
            ModelType::Colbert => "colbert",
            _ => "other",
        };

        println!(
            "{:<25} {:<10} {:<8} {:<8} {}",
            model.id,
            type_str,
            model.embedding_dim.default_dim(),
            model.context_length,
            model.name
        );
    }

    Ok(())
}
