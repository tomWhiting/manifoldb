//! Routes files to appropriate chunking strategies

use std::path::Path;

use anyhow::Result;

use crate::config::ChunkingConfig;
use super::{
    Chunk, ChunkStrategy, CodeChunker, Language, MarkdownChunker, TextChunker,
};

/// Routes files to the appropriate chunker based on file type
pub struct ChunkRouter {
    markdown: MarkdownChunker,
    code: CodeChunker,
    text: TextChunker,
}

impl ChunkRouter {
    /// Create a new router with default settings
    pub fn new() -> Self {
        Self {
            markdown: MarkdownChunker::default(),
            code: CodeChunker::default(),
            text: TextChunker::default(),
        }
    }

    /// Create a router from config
    pub fn from_config(config: &ChunkingConfig) -> Self {
        Self {
            markdown: MarkdownChunker::from_config(config),
            code: CodeChunker::default(),
            text: TextChunker::default(),
        }
    }

    /// Create a router with custom chunkers
    pub fn with_chunkers(
        markdown: MarkdownChunker,
        code: CodeChunker,
        text: TextChunker,
    ) -> Self {
        Self {
            markdown,
            code,
            text,
        }
    }

    /// Chunk content based on file path
    pub fn chunk(&self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext.to_lowercase().as_str() {
            // Markdown
            "md" | "markdown" => self.markdown.chunk(content, path),

            // Code files with tree-sitter support
            "rs" | "py" | "pyi" | "ts" | "mts" | "cts" | "js" | "mjs" | "cjs" | "tsx" | "go"
            | "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" | "json" | "yaml"
            | "yml" | "css" | "sh" | "bash" | "zsh" => {
                // Try code chunker first
                match self.code.chunk(content, path) {
                    Ok(chunks) if !chunks.is_empty() => Ok(chunks),
                    // Fall back to text chunker if code chunking fails
                    _ => self.text.chunk(content, path),
                }
            }

            // Plain text and unknown extensions
            "txt" | "text" | "rst" | "asciidoc" | "adoc" => self.text.chunk(content, path),

            // Unknown - try text chunker
            _ => self.text.chunk(content, path),
        }
    }

    /// Get the chunking strategy that would be used for a path
    pub fn strategy_for(&self, path: &Path) -> ChunkingStrategy {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext.to_lowercase().as_str() {
            "md" | "markdown" => ChunkingStrategy::Markdown,
            "rs" | "py" | "pyi" | "ts" | "mts" | "cts" | "js" | "mjs" | "cjs" | "tsx" | "go"
            | "c" | "h" | "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" | "json" | "yaml"
            | "yml" | "css" | "sh" | "bash" | "zsh" => {
                if let Some(lang) = Language::from_extension(ext) {
                    ChunkingStrategy::Code(lang)
                } else {
                    ChunkingStrategy::Text
                }
            }
            _ => ChunkingStrategy::Text,
        }
    }
}

impl Default for ChunkRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// The chunking strategy used for a file
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    Markdown,
    Code(Language),
    Text,
}

impl std::fmt::Display for ChunkingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkingStrategy::Markdown => write!(f, "markdown"),
            ChunkingStrategy::Code(lang) => write!(f, "code:{}", lang),
            ChunkingStrategy::Text => write!(f, "text"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Language;
    use std::path::PathBuf;

    #[test]
    fn test_strategy_detection() {
        let router = ChunkRouter::new();

        assert_eq!(
            router.strategy_for(&PathBuf::from("README.md")),
            ChunkingStrategy::Markdown
        );
        assert_eq!(
            router.strategy_for(&PathBuf::from("main.rs")),
            ChunkingStrategy::Code(Language::Rust)
        );
        assert_eq!(
            router.strategy_for(&PathBuf::from("script.py")),
            ChunkingStrategy::Code(Language::Python)
        );
        assert_eq!(
            router.strategy_for(&PathBuf::from("notes.txt")),
            ChunkingStrategy::Text
        );
    }

    #[test]
    fn test_markdown_routing() {
        let router = ChunkRouter::new();
        let content = "# Hello\n\nWorld";
        let chunks = router
            .chunk(content, &PathBuf::from("test.md"))
            .unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks[0].heading.is_some());
    }

    #[test]
    fn test_code_routing() {
        let router = ChunkRouter::new();
        let content = "fn main() { println!(\"Hello\"); }";
        let chunks = router
            .chunk(content, &PathBuf::from("main.rs"))
            .unwrap();

        assert!(!chunks.is_empty());
    }
}
