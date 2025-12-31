//! Markdown chunking by headers

use std::path::Path;

use crate::config::ChunkingConfig;
use super::{Chunk, ChunkStrategy};

/// Markdown-aware chunker that splits on headers
pub struct MarkdownChunker {
    /// Header levels to split on (1 = h1, 2 = h2, etc.)
    pub header_levels: Vec<u8>,
    /// Whether header splitting is enabled
    pub split_on_headers: bool,
}

impl MarkdownChunker {
    /// Create a new markdown chunker from config
    pub fn from_config(config: &ChunkingConfig) -> Self {
        Self {
            header_levels: config.header_levels.clone(),
            split_on_headers: config.split_on_headers,
        }
    }

    /// Create a new markdown chunker with default settings
    pub fn new() -> Self {
        Self {
            header_levels: vec![1, 2, 3],
            split_on_headers: true,
        }
    }
}

impl Default for MarkdownChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkStrategy for MarkdownChunker {
    fn chunk(&self, content: &str, _path: &Path) -> anyhow::Result<Vec<Chunk>> {
        Ok(split_markdown_with_chunker(content, self))
    }
}

/// Split markdown text into chunks at header boundaries
///
/// This function accepts a ChunkingConfig for backwards compatibility.
pub fn split_markdown(text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    let chunker = MarkdownChunker::from_config(config);
    split_markdown_with_chunker(text, &chunker)
}

/// Split markdown with a pre-configured chunker
pub fn split_markdown_with_chunker(text: &str, chunker: &MarkdownChunker) -> Vec<Chunk> {
    if !chunker.split_on_headers {
        let line_count = text.lines().count();
        return vec![Chunk::markdown(
            text.to_string(),
            0..text.len(),
            1..line_count.max(1) + 1,
            None,
            0,
        )];
    }

    let mut chunks = Vec::new();
    let mut current_content = String::new();
    let mut current_heading: Option<String> = None;
    let mut current_level: u8 = 0;
    let mut current_line_start: usize = 1;
    let mut current_byte_start: usize = 0;
    let mut line_num = 0;
    let mut byte_pos = 0;

    for line in text.lines() {
        line_num += 1;
        let line_len = line.len();

        // Check if this line is a header
        if let Some((level, heading)) = parse_header(line) {
            // If we should split on this header level
            if chunker.header_levels.contains(&level) {
                // Save current chunk if it has content
                if !current_content.trim().is_empty() {
                    chunks.push(Chunk::markdown(
                        current_content.trim().to_string(),
                        current_byte_start..byte_pos,
                        current_line_start..line_num,
                        current_heading,
                        current_level,
                    ));
                }

                // Start new chunk
                current_content = String::new();
                current_heading = Some(heading);
                current_level = level;
                current_line_start = line_num;
                current_byte_start = byte_pos;
            } else {
                // Include header in current chunk
                current_content.push_str(line);
                current_content.push('\n');
            }
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }

        byte_pos += line_len + 1; // +1 for newline
    }

    // Don't forget the last chunk
    if !current_content.trim().is_empty() {
        chunks.push(Chunk::markdown(
            current_content.trim().to_string(),
            current_byte_start..byte_pos,
            current_line_start..line_num + 1,
            current_heading,
            current_level,
        ));
    }

    chunks
}

/// Parse a markdown header line
fn parse_header(line: &str) -> Option<(u8, String)> {
    let trimmed = line.trim_start();

    // Count leading # characters
    let hash_count = trimmed.chars().take_while(|&c| c == '#').count();

    if hash_count > 0 && hash_count <= 6 {
        let rest = trimmed[hash_count..].trim();
        if !rest.is_empty() || hash_count <= 6 {
            return Some((hash_count as u8, rest.to_string()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Language;

    #[test]
    fn test_parse_header() {
        assert_eq!(parse_header("# Title"), Some((1, "Title".to_string())));
        assert_eq!(parse_header("## Subtitle"), Some((2, "Subtitle".to_string())));
        assert_eq!(parse_header("### Deep"), Some((3, "Deep".to_string())));
        assert_eq!(parse_header("Regular text"), None);
        assert_eq!(parse_header("  # Indented"), Some((1, "Indented".to_string())));
    }

    #[test]
    fn test_split_markdown() {
        let md = r#"# Introduction

This is the intro.

## First Section

Content of first section.

## Second Section

Content of second section.
"#;

        let chunker = MarkdownChunker::new();
        let chunks = split_markdown_with_chunker(md, &chunker);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, Some("Introduction".to_string()));
        assert_eq!(chunks[1].heading, Some("First Section".to_string()));
        assert_eq!(chunks[2].heading, Some("Second Section".to_string()));

        // Check that language is set
        assert_eq!(chunks[0].language, Some(Language::Markdown));
    }

    #[test]
    fn test_byte_ranges() {
        let md = "# Hello\n\nWorld\n";
        let chunker = MarkdownChunker::new();
        let chunks = split_markdown_with_chunker(md, &chunker);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source.byte_range.start, 0);
        assert_eq!(chunks[0].source.line_range.start, 1);
    }
}
