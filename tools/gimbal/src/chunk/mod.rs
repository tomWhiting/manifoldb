//! Content chunking for embedding
//!
//! Provides modular chunking strategies for different content types:
//! - Markdown: Header-aware splitting
//! - Code: Tree-sitter AST-based symbol extraction
//! - Text: Plain text paragraph/sentence splitting

mod markdown;
mod text;
pub mod code;
mod router;

pub use markdown::MarkdownChunker;
pub use text::TextChunker;
pub use code::CodeChunker;
pub use router::ChunkRouter;

use std::ops::Range;
use std::path::Path;

/// Source location for a chunk
#[derive(Debug, Clone)]
pub struct ChunkSource {
    /// Byte range in the original content
    pub byte_range: Range<usize>,
    /// Line range (1-indexed)
    pub line_range: Range<usize>,
}

/// Symbol information for code chunks
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// Symbol name (e.g., "parse_config")
    pub name: String,
    /// Kind of symbol
    pub kind: SymbolKind,
    /// Full signature if available (e.g., "pub fn parse_config(path: &Path) -> Result<Config>")
    pub signature: Option<String>,
    /// Visibility
    pub visibility: Visibility,
    /// Parent symbol name (e.g., impl block or module)
    pub parent: Option<String>,
}

/// Kinds of code symbols
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Class,
    Method,
    Constant,
    Type,
    Field,
    Macro,
    Interface,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolKind::Function => write!(f, "function"),
            SymbolKind::Struct => write!(f, "struct"),
            SymbolKind::Enum => write!(f, "enum"),
            SymbolKind::Trait => write!(f, "trait"),
            SymbolKind::Impl => write!(f, "impl"),
            SymbolKind::Module => write!(f, "module"),
            SymbolKind::Class => write!(f, "class"),
            SymbolKind::Method => write!(f, "method"),
            SymbolKind::Constant => write!(f, "constant"),
            SymbolKind::Type => write!(f, "type"),
            SymbolKind::Field => write!(f, "field"),
            SymbolKind::Macro => write!(f, "macro"),
            SymbolKind::Interface => write!(f, "interface"),
        }
    }
}

/// Symbol visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Visibility {
    Public,
    #[default]
    Private,
    Crate,
    Super,
}

/// Programming language for code chunks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Tsx,
    Go,
    C,
    Cpp,
    Json,
    Yaml,
    Css,
    Markdown,
    Bash,
}

impl Language {
    /// Detect language from file path
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?;
        Self::from_extension(ext)
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "py" | "pyi" => Some(Self::Python),
            "ts" | "mts" | "cts" => Some(Self::TypeScript),
            "js" | "mjs" | "cjs" => Some(Self::JavaScript),
            "tsx" => Some(Self::Tsx),
            "go" => Some(Self::Go),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(Self::Cpp),
            "json" => Some(Self::Json),
            "yaml" | "yml" => Some(Self::Yaml),
            "css" => Some(Self::Css),
            "md" | "markdown" => Some(Self::Markdown),
            "sh" | "bash" | "zsh" => Some(Self::Bash),
            _ => None,
        }
    }

    /// Get the query directory name for this language
    pub fn query_dir_name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::Tsx => "tsx",
            Self::Go => "go",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Css => "css",
            Self::Markdown => "markdown",
            Self::Bash => "bash",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.query_dir_name())
    }
}

/// A chunk of content with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The text content
    pub content: String,
    /// Source location in original file
    pub source: ChunkSource,

    // Markdown-specific fields
    /// Section heading (for markdown)
    pub heading: Option<String>,
    /// Header level 1-6 (for markdown)
    pub header_level: Option<u8>,

    // Code-specific fields
    /// Symbol information (for code)
    pub symbol: Option<SymbolInfo>,
    /// Programming language (for code)
    pub language: Option<Language>,
}

impl Chunk {
    /// Create a new chunk with minimal metadata
    pub fn new(content: String, byte_range: Range<usize>, line_range: Range<usize>) -> Self {
        Self {
            content,
            source: ChunkSource {
                byte_range,
                line_range,
            },
            heading: None,
            header_level: None,
            symbol: None,
            language: None,
        }
    }

    /// Create a markdown chunk
    pub fn markdown(
        content: String,
        byte_range: Range<usize>,
        line_range: Range<usize>,
        heading: Option<String>,
        level: u8,
    ) -> Self {
        Self {
            content,
            source: ChunkSource {
                byte_range,
                line_range,
            },
            heading,
            header_level: Some(level),
            symbol: None,
            language: Some(Language::Markdown),
        }
    }

    /// Create a code chunk
    pub fn code(
        content: String,
        byte_range: Range<usize>,
        line_range: Range<usize>,
        symbol: SymbolInfo,
        language: Language,
    ) -> Self {
        Self {
            content,
            source: ChunkSource {
                byte_range,
                line_range,
            },
            heading: None,
            header_level: None,
            symbol: Some(symbol),
            language: Some(language),
        }
    }
}

/// Strategy for splitting content into chunks
pub trait ChunkStrategy {
    /// Split content into chunks
    fn chunk(&self, content: &str, path: &Path) -> anyhow::Result<Vec<Chunk>>;
}
