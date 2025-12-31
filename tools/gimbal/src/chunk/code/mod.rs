//! Code chunking using tree-sitter AST analysis
//!
//! Parses source code using tree-sitter and extracts semantic chunks
//! based on symbol boundaries (functions, structs, classes, etc.)

mod languages;
mod queries;
mod symbols;

pub use super::Language;

use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Result};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

use super::{Chunk, ChunkStrategy, SymbolKind, Visibility};
use languages::get_grammar;
use queries::get_queries;
use symbols::{build_symbol_info, parse_rust_visibility, symbol_kind_from_node};

/// Code chunker using tree-sitter for AST-aware splitting
pub struct CodeChunker {
    /// Maximum chunk size in bytes
    max_chunk_size: usize,
}

impl CodeChunker {
    /// Create a new code chunker
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    /// Check if a language is supported for code chunking
    pub fn supports_language(lang: Language) -> bool {
        get_queries(lang)
            .map(|q| q.chunking_query().is_some())
            .unwrap_or(false)
    }

    /// Chunk code for a specific language
    pub fn chunk_language(&self, content: &str, lang: Language) -> Result<Vec<Chunk>> {
        let grammar = get_grammar(lang)?;
        let query_set = get_queries(lang).ok_or_else(|| anyhow!("No queries for {}", lang))?;
        let query_source = query_set
            .chunking_query()
            .ok_or_else(|| anyhow!("No chunking query for {}", lang))?;

        // Parse the source code
        let mut parser = Parser::new();
        parser.set_language(&grammar)?;
        let tree = parser
            .parse(content, None)
            .ok_or_else(|| anyhow!("Failed to parse source code"))?;

        // Compile and run the query
        let query = Query::new(&grammar, query_source)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), content.as_bytes());

        // Extract capture names for reference
        let capture_names: Vec<String> = query.capture_names().iter().map(|s| s.to_string()).collect();

        let mut chunks = Vec::new();
        let mut seen_ranges: Vec<Range<usize>> = Vec::new();

        while let Some(match_) = matches.next() {
            let mut item_node = None;
            let mut name_text = None;
            let mut context_text = None;

            for capture in match_.captures.iter() {
                let capture_name = capture_names
                    .get(capture.index as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                let node = capture.node;

                match capture_name {
                    "item" => {
                        item_node = Some(node);
                    }
                    "name" => {
                        name_text = Some(
                            content[node.byte_range()]
                                .to_string(),
                        );
                    }
                    "context" => {
                        if context_text.is_none() {
                            context_text = Some(
                                content[node.byte_range()]
                                    .to_string(),
                            );
                        }
                    }
                    _ => {}
                }
            }

            if let Some(node) = item_node {
                let byte_range = node.byte_range();

                // Skip if we've already seen this range (nested matches)
                if seen_ranges.iter().any(|r| {
                    r.start <= byte_range.start && r.end >= byte_range.end
                }) {
                    continue;
                }
                seen_ranges.push(byte_range.clone());

                let node_content = content[byte_range.clone()].to_string();
                let node_type = node.kind();

                // Calculate line range
                let start_line = node.start_position().row + 1;
                let end_line = node.end_position().row + 1;

                // Determine symbol kind
                let kind = symbol_kind_from_node(node_type, lang)
                    .unwrap_or(SymbolKind::Function);

                // Extract visibility for Rust
                let visibility = if lang == Language::Rust {
                    // Look for visibility modifier in the node's children
                    let vis_text = node
                        .child_by_field_name("visibility")
                        .or_else(|| {
                            // Check first child if it's a visibility_modifier
                            let first = node.child(0)?;
                            if first.kind() == "visibility_modifier" {
                                Some(first)
                            } else {
                                None
                            }
                        })
                        .map(|n| &content[n.byte_range()]);
                    parse_rust_visibility(vis_text)
                } else {
                    Visibility::Private
                };

                // Build signature (first line or context)
                let signature = context_text.or_else(|| {
                    node_content.lines().next().map(|s| s.to_string())
                });

                let symbol = build_symbol_info(
                    name_text.unwrap_or_else(|| "<anonymous>".to_string()),
                    kind,
                    signature,
                    visibility,
                    None, // TODO: track parent context
                );

                // Handle chunks that exceed max size
                if node_content.len() <= self.max_chunk_size {
                    chunks.push(Chunk::code(
                        node_content,
                        byte_range,
                        start_line..end_line + 1,
                        symbol,
                        lang,
                    ));
                } else {
                    // For oversized chunks, we could either:
                    // 1. Include them anyway (current approach for now)
                    // 2. Split them further
                    // For now, include them - embeddings can handle truncation
                    chunks.push(Chunk::code(
                        node_content,
                        byte_range,
                        start_line..end_line + 1,
                        symbol,
                        lang,
                    ));
                }
            }
        }

        // If no chunks were extracted, fall back to the whole file as one chunk
        if chunks.is_empty() {
            let line_count = content.lines().count();
            chunks.push(Chunk::new(
                content.to_string(),
                0..content.len(),
                1..line_count + 1,
            ));
        }

        Ok(chunks)
    }
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self::new(8000) // Default to ~8KB chunks
    }
}

impl ChunkStrategy for CodeChunker {
    fn chunk(&self, content: &str, path: &Path) -> Result<Vec<Chunk>> {
        let lang = Language::from_path(path)
            .ok_or_else(|| anyhow!("Unknown language for {:?}", path))?;

        self.chunk_language(content, lang)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_chunking() {
        let code = r#"
/// A greeting function
pub fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

/// A person struct
pub struct Person {
    name: String,
    age: u32,
}

impl Person {
    pub fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }
}
"#;

        let chunker = CodeChunker::new(8000);
        let chunks = chunker.chunk_language(code, Language::Rust).unwrap();

        // Should have at least function, struct, and impl
        assert!(!chunks.is_empty());

        // Check that we got symbols
        let has_function = chunks.iter().any(|c| {
            c.symbol
                .as_ref()
                .map(|s| s.kind == SymbolKind::Function)
                .unwrap_or(false)
        });
        assert!(has_function);
    }

    #[test]
    fn test_python_chunking() {
        let code = r#"
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

class Person:
    """A person class."""

    def __init__(self, name: str):
        self.name = name
"#;

        let chunker = CodeChunker::new(8000);
        let chunks = chunker.chunk_language(code, Language::Python).unwrap();

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_language_detection() {
        assert!(CodeChunker::supports_language(Language::Rust));
        assert!(CodeChunker::supports_language(Language::Python));
        assert!(CodeChunker::supports_language(Language::TypeScript));
    }
}
