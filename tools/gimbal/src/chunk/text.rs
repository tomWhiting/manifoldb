//! Plain text chunking by paragraphs

use std::path::Path;

use super::{Chunk, ChunkStrategy};

/// Plain text chunker that splits on paragraph boundaries
pub struct TextChunker {
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,
    /// Overlap between chunks in characters
    pub overlap: usize,
}

impl TextChunker {
    /// Create a new text chunker with specified limits
    pub fn new(max_chunk_size: usize, overlap: usize) -> Self {
        Self {
            max_chunk_size,
            overlap,
        }
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000,
            overlap: 200,
        }
    }
}

impl ChunkStrategy for TextChunker {
    fn chunk(&self, content: &str, _path: &Path) -> anyhow::Result<Vec<Chunk>> {
        Ok(split_text(content, self.max_chunk_size, self.overlap))
    }
}

/// Split text into chunks by paragraphs, respecting max size
pub fn split_text(text: &str, max_size: usize, overlap: usize) -> Vec<Chunk> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    if paragraphs.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut current_content = String::new();
    let mut current_byte_start = 0;
    let mut current_line_start = 1;
    let mut byte_pos = 0;
    let mut line_num = 1;

    for para in paragraphs {
        let para_lines = para.lines().count();
        let para_bytes = para.len();

        // If adding this paragraph would exceed max size, save current chunk
        if !current_content.is_empty()
            && current_content.len() + para_bytes + 2 > max_size
        {
            let chunk = Chunk::new(
                current_content.trim().to_string(),
                current_byte_start..byte_pos,
                current_line_start..line_num,
            );
            chunks.push(chunk);

            // Start new chunk with overlap
            if overlap > 0 && current_content.len() > overlap {
                let overlap_start = current_content.len().saturating_sub(overlap);
                current_content = current_content[overlap_start..].to_string();
            } else {
                current_content = String::new();
            }
            current_byte_start = byte_pos.saturating_sub(current_content.len());
            current_line_start = line_num.saturating_sub(
                current_content.lines().count().saturating_sub(1),
            );
        }

        // Add paragraph to current chunk
        if !current_content.is_empty() {
            current_content.push_str("\n\n");
        }
        current_content.push_str(para);

        byte_pos += para_bytes + 2; // +2 for paragraph separator
        line_num += para_lines + 1; // +1 for blank line between paragraphs
    }

    // Don't forget the last chunk
    if !current_content.trim().is_empty() {
        let chunk = Chunk::new(
            current_content.trim().to_string(),
            current_byte_start..byte_pos,
            current_line_start..line_num,
        );
        chunks.push(chunk);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_simple() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = split_text(text, 1000, 0);

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("First"));
        assert!(chunks[0].content.contains("Third"));
    }

    #[test]
    fn test_split_by_size() {
        let text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph.";
        let chunks = split_text(text, 50, 0);

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_empty_text() {
        let chunks = split_text("", 1000, 0);
        assert!(chunks.is_empty());
    }
}
