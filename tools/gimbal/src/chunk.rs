//! Markdown chunking by headers

use crate::config::ChunkingConfig;

/// A chunk of text with metadata
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The text content
    pub content: String,

    /// Section heading (if any)
    pub heading: Option<String>,

    /// Header level (1-6, or 0 if no header)
    pub level: u8,

    /// Line number in original file
    pub line_start: usize,
}

/// Split markdown text into chunks at header boundaries
pub fn split_markdown(text: &str, config: &ChunkingConfig) -> Vec<Chunk> {
    if !config.split_on_headers {
        return vec![Chunk {
            content: text.to_string(),
            heading: None,
            level: 0,
            line_start: 1,
        }];
    }

    let mut chunks = Vec::new();
    let mut current_content = String::new();
    let mut current_heading: Option<String> = None;
    let mut current_level: u8 = 0;
    let mut current_line_start: usize = 1;
    let mut line_num = 0;

    for line in text.lines() {
        line_num += 1;

        // Check if this line is a header
        if let Some((level, heading)) = parse_header(line) {
            // If we should split on this header level
            if config.header_levels.contains(&level) {
                // Save current chunk if it has content
                if !current_content.trim().is_empty() {
                    chunks.push(Chunk {
                        content: current_content.trim().to_string(),
                        heading: current_heading,
                        level: current_level,
                        line_start: current_line_start,
                    });
                }

                // Start new chunk
                current_content = String::new();
                current_heading = Some(heading);
                current_level = level;
                current_line_start = line_num;
            } else {
                // Include header in current chunk
                current_content.push_str(line);
                current_content.push('\n');
            }
        } else {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    // Don't forget the last chunk
    if !current_content.trim().is_empty() {
        chunks.push(Chunk {
            content: current_content.trim().to_string(),
            heading: current_heading,
            level: current_level,
            line_start: current_line_start,
        });
    }

    // Note: max_chunk_size is now handled per-vector in the embedding config
    // The overlap from config is available for future use if needed

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

/// Split chunks that exceed max size
fn apply_max_size(chunks: Vec<Chunk>, max_size: usize, overlap: usize) -> Vec<Chunk> {
    let mut result = Vec::new();

    for chunk in chunks {
        if chunk.content.len() <= max_size {
            result.push(chunk);
        } else {
            // Split by paragraphs first, then by size
            let paragraphs: Vec<&str> = chunk.content.split("\n\n").collect();
            let mut current = String::new();
            let mut part_num = 0;

            for para in paragraphs {
                if current.len() + para.len() + 2 > max_size && !current.is_empty() {
                    // Save current chunk
                    part_num += 1;
                    result.push(Chunk {
                        content: current.trim().to_string(),
                        heading: chunk.heading.as_ref().map(|h| {
                            if part_num > 1 {
                                format!("{} (part {})", h, part_num)
                            } else {
                                h.clone()
                            }
                        }),
                        level: chunk.level,
                        line_start: chunk.line_start,
                    });

                    // Start new chunk with overlap
                    if overlap > 0 && current.len() > overlap {
                        current = current[current.len() - overlap..].to_string();
                    } else {
                        current = String::new();
                    }
                }

                if !current.is_empty() {
                    current.push_str("\n\n");
                }
                current.push_str(para);
            }

            // Final chunk
            if !current.trim().is_empty() {
                part_num += 1;
                result.push(Chunk {
                    content: current.trim().to_string(),
                    heading: chunk.heading.as_ref().map(|h| {
                        if part_num > 1 {
                            format!("{} (part {})", h, part_num)
                        } else {
                            h.clone()
                        }
                    }),
                    level: chunk.level,
                    line_start: chunk.line_start,
                });
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let config = ChunkingConfig::default();
        let chunks = split_markdown(md, &config);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].heading, Some("Introduction".to_string()));
        assert_eq!(chunks[1].heading, Some("First Section".to_string()));
        assert_eq!(chunks[2].heading, Some("Second Section".to_string()));
    }
}
