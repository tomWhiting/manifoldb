//! Ingestion and search logic for multi-vector embeddings
//!
//! Handles crawling, chunking, embedding with multiple models, and storing in ManifoldDB.
//! Supports named sources with per-source vector assignments.

use crate::chunk::split_markdown;
use crate::config::{Config, IngestSource, SearchMode};
use crate::embed::EmbedderSet;
use crate::store::Store;
use anyhow::{anyhow, Result};
use manifoldb::EntityId;
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;
use walkdir::WalkDir;

/// Ingest all configured sources
pub fn ingest_all_sources(config: &Config, properties: &[(String, String)]) -> Result<()> {
    if config.ingest.sources.is_empty() {
        return Err(anyhow!("No sources configured. Add [[ingest.sources]] to your config."));
    }

    println!("Ingesting from {} configured sources...", config.ingest.sources.len());

    let mut store = Store::open(config)?;
    println!("Database opened: {}", config.database.path.display());

    for source in &config.ingest.sources {
        println!("\n=== Source: {} ===", source.name);
        ingest_source_internal(config, &mut store, source, properties)?;
    }

    store.flush()?;
    println!("\nAll sources ingested successfully!");
    Ok(())
}

/// Ingest from a specific configured source by name
pub fn ingest_source(config: &Config, source_name: &str, properties: &[(String, String)]) -> Result<()> {
    let source = config.get_source(source_name)
        .ok_or_else(|| anyhow!(
            "Source '{}' not found. Available sources: {:?}",
            source_name,
            config.ingest.sources.iter().map(|s| &s.name).collect::<Vec<_>>()
        ))?;

    println!("Ingesting from source: {}", source.name);

    let mut store = Store::open(config)?;
    println!("Database opened: {}", config.database.path.display());

    ingest_source_internal(config, &mut store, source, properties)?;

    store.flush()?;
    println!("\nSource '{}' ingested successfully!", source_name);
    Ok(())
}

/// Internal function to ingest a source
fn ingest_source_internal(
    config: &Config,
    store: &mut Store,
    source: &IngestSource,
    properties: &[(String, String)],
) -> Result<()> {
    // Get vectors for this source
    let vector_names = config.vectors_for_source(source);
    println!("  Vectors: {:?}", vector_names);
    println!("  Extensions: {:?}", source.extensions);
    println!("  Exclude: {:?}", source.exclude);

    // Load only the embedders needed for this source
    let embedder_set = EmbedderSet::from_config_filtered(&config.vectors, &vector_names)?;

    println!("  Loading {} embedding models...", embedder_set.len());
    for (name, embedder) in embedder_set.iter() {
        println!(
            "    {} ({}, dim: {})",
            name,
            embedder.metadata().name,
            embedder.dimension()
        );
    }

    let mut file_count = 0;
    let mut chunk_count = 0;

    for path in &source.paths {
        println!("  Crawling: {}", path.display());

        if path.is_file() {
            if source.matches_extension(path) && !source.should_exclude(path) {
                chunk_count += ingest_file_internal(config, &embedder_set, store, path, properties)?;
                file_count += 1;
            }
        } else if path.is_dir() {
            let (files, chunks) = ingest_directory_internal(
                config,
                &embedder_set,
                store,
                path,
                &source.extensions,
                &source.exclude,
                properties,
            )?;
            file_count += files;
            chunk_count += chunks;
        } else {
            eprintln!("  Warning: Path does not exist: {}", path.display());
        }
    }

    println!("  Processed {} files, {} chunks", file_count, chunk_count);
    Ok(())
}

/// Ingest from a CLI-specified path with optional filters
pub fn ingest_path(
    config: &Config,
    path: &Path,
    extensions: &[String],
    exclude: &[String],
    properties: &[(String, String)],
    vector_filter: Option<&[String]>,
) -> Result<()> {
    // Determine which vectors to use
    let vector_names: Vec<String> = match vector_filter {
        Some(names) => names.to_vec(),
        None => config.default_ingest_vectors(),
    };

    println!("Ingesting from: {}", path.display());
    println!("  Vectors: {:?}", vector_names);
    println!("  Extensions: {:?}", extensions);
    if !exclude.is_empty() {
        println!("  Exclude: {:?}", exclude);
    }

    // Load embedders
    let embedder_set = EmbedderSet::from_config_filtered(&config.vectors, &vector_names)?;

    println!("Loading {} embedding models...", embedder_set.len());
    for (name, embedder) in embedder_set.iter() {
        println!(
            "  {} ({}, dim: {}, context: {})",
            name,
            embedder.metadata().name,
            embedder.dimension(),
            embedder.context_length()
        );
    }

    let mut store = Store::open(config)?;
    println!("Database opened: {}", config.database.path.display());

    let (file_count, chunk_count) = if path.is_file() {
        let chunks = ingest_file_internal(config, &embedder_set, &mut store, path, properties)?;
        (1, chunks)
    } else if path.is_dir() {
        ingest_directory_internal(
            config,
            &embedder_set,
            &mut store,
            path,
            extensions,
            exclude,
            properties,
        )?
    } else {
        return Err(anyhow!("Path does not exist: {}", path.display()));
    };

    store.flush()?;
    println!("\nIngestion complete! {} files, {} chunks", file_count, chunk_count);
    Ok(())
}

/// Legacy ingest function for backwards compatibility
pub fn ingest(
    config: &Config,
    path: &Path,
    extensions: &[String],
    properties: &[(String, String)],
) -> Result<()> {
    ingest_path(config, path, extensions, &[], properties, None)
}

/// Ingest a single file, returning the number of chunks created
fn ingest_file_internal(
    config: &Config,
    embedder_set: &EmbedderSet,
    store: &mut Store,
    path: &Path,
    properties: &[(String, String)],
) -> Result<usize> {
    let content = std::fs::read_to_string(path)?;
    let path_str = path.to_string_lossy().to_string();

    println!("Processing: {}", path.display());

    let chunks = split_markdown(&content, &config.chunking);
    println!("  {} chunks", chunks.len());

    let mut prev_chunk_id: Option<EntityId> = None;

    for (i, chunk) in chunks.iter().enumerate() {
        // Embed with configured models
        let embeddings = embedder_set.embed_all(&chunk.content)?;

        let chunk_id = store.store_chunk(
            &path_str,
            chunk.heading.as_deref(),
            &chunk.content,
            &embeddings,
            properties,
            prev_chunk_id,
        )?;

        prev_chunk_id = Some(chunk_id);

        if (i + 1) % 10 == 0 {
            println!("  Embedded {}/{} chunks", i + 1, chunks.len());
        }
    }

    Ok(chunks.len())
}

/// Ingest all files in a directory, returning (file_count, chunk_count)
fn ingest_directory_internal(
    config: &Config,
    embedder_set: &EmbedderSet,
    store: &mut Store,
    path: &Path,
    extensions: &[String],
    exclude: &[String],
    properties: &[(String, String)],
) -> Result<(usize, usize)> {
    let mut file_count = 0;
    let mut chunk_count = 0;

    for entry in WalkDir::new(path).follow_links(true) {
        let entry = entry?;
        let entry_path = entry.path();

        // Skip directories
        if !entry_path.is_file() {
            continue;
        }

        // Check exclusions
        let path_str = entry_path.to_string_lossy();
        if exclude.iter().any(|pattern| path_str.contains(pattern)) {
            continue;
        }

        // Check extension
        let ext = entry_path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if !extensions.is_empty() && !extensions.iter().any(|e| e == ext) {
            continue;
        }

        // Read file content
        let content = match std::fs::read_to_string(entry_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Could not read {}: {}", entry_path.display(), e);
                continue;
            }
        };

        let path_str = entry_path.to_string_lossy().to_string();
        println!("Processing: {}", entry_path.display());

        let chunks = split_markdown(&content, &config.chunking);
        println!("  {} chunks", chunks.len());

        let mut prev_chunk_id: Option<EntityId> = None;

        for chunk in &chunks {
            let embeddings = embedder_set.embed_all(&chunk.content)?;

            let chunk_id = store.store_chunk(
                &path_str,
                chunk.heading.as_deref(),
                &chunk.content,
                &embeddings,
                properties,
                prev_chunk_id,
            )?;

            prev_chunk_id = Some(chunk_id);
            chunk_count += 1;
        }

        file_count += 1;
    }

    Ok((file_count, chunk_count))
}

/// Ingest from stdin (pipe mode)
pub fn pipe_stdin(
    config: &Config,
    file_path: &str,
    heading: Option<&str>,
    properties: &[(String, String)],
    vector_filter: Option<&[String]>,
) -> Result<()> {
    // Read all stdin
    let mut content = String::new();
    io::stdin().read_to_string(&mut content)?;

    if content.trim().is_empty() {
        println!("No input received on stdin");
        return Ok(());
    }

    // Determine which vectors to use
    let vector_names: Vec<String> = match vector_filter {
        Some(names) => names.to_vec(),
        None => config.default_ingest_vectors(),
    };

    println!("Loading embedding models...");
    let embedder_set = EmbedderSet::from_config_filtered(&config.vectors, &vector_names)?;

    for (name, embedder) in embedder_set.iter() {
        println!(
            "  {} ({}, dim: {})",
            name,
            embedder.metadata().name,
            embedder.dimension()
        );
    }

    let mut store = Store::open(config)?;

    println!("Embedding chunk from: {}", file_path);
    let embeddings = embedder_set.embed_all(&content)?;

    let chunk_id = store.store_chunk(
        file_path,
        heading,
        &content,
        &embeddings,
        properties,
        None,
    )?;

    store.flush()?;

    println!("Stored chunk with ID: {:?}", chunk_id);
    Ok(())
}

// ANSI color codes
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[31m";
const BLUE: &str = "\x1b[34m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

/// Search for similar documents
pub fn search(
    config: &Config,
    query: &str,
    limit: usize,
    show_full: bool,
    mode: Option<SearchMode>,
    filters: &[(String, String)],
) -> Result<()> {
    let embedder_set = EmbedderSet::from_config(&config.vectors)?;
    let store = Store::open(config)?;

    // Determine search mode
    let search_mode = mode.unwrap_or(config.search.default_mode);

    // Generate query embeddings with all models
    let query_embeddings = embedder_set.embed_all(query)?;

    // Get hybrid config if needed
    let hybrid_config = if search_mode == SearchMode::Hybrid {
        Some(&config.search.hybrid)
    } else {
        None
    };

    let results = store.search(search_mode, &query_embeddings, limit, filters, hybrid_config)?;

    if results.is_empty() {
        println!("No results found.");
        return Ok(());
    }

    // Calculate score normalization based on mode and actual results
    // For dot product scores (SPLADE/sparse), normalize by max score in results
    let (max_score, score_explanation) = match search_mode {
        SearchMode::Hybrid => {
            let k = 60.0;
            let n_sources = 2.0; // dense + sparse
            (n_sources / (k + 1.0), "RRF")
        }
        SearchMode::Dense | SearchMode::Colbert => {
            // Cosine similarity is already 0-1
            (1.0, "similarity")
        }
        SearchMode::Sparse => {
            // SPLADE uses dot product which returns unbounded scores (typically 5-50+)
            // Normalize by the max score in the result set
            let max_in_results = results.iter().map(|r| r.score).fold(0.0f32, f32::max);
            (max_in_results.max(1.0), "relevance")
        }
    };

    // Header
    println!(
        "\n{DIM}{} search ({}) · {} results · higher = better match{RESET}\n",
        format!("{:?}", search_mode).to_lowercase(),
        score_explanation,
        results.len()
    );

    // Extract query terms for highlighting
    // Keep all terms for phrase matching, filter later for individual highlights
    let query_terms: Vec<&str> = query.split_whitespace().collect();

    for result in &results {
        // Normalize score to percentage
        let pct = ((result.score / max_score) * 100.0).min(100.0);

        // Format header line: <score%> path (turbofish style)
        print!("{BOLD}{CYAN}<{:>3.0}%>{RESET} ", pct);
        print!("{BLUE}{}{RESET}", result.file_path);

        // Add heading if present
        if !result.heading.is_empty() {
            print!("{DIM} · {}{RESET}", result.heading);
        }
        println!();

        // Content with highlighted matches
        let content = if show_full {
            result.content.clone()
        } else {
            // Preview: first 400 chars, collapse newlines
            let preview: String = result.content.chars().take(400).collect();
            let preview = preview.replace('\n', " ");
            let suffix = if result.content.chars().count() > 400 { "..." } else { "" };
            format!("{}{}", preview.trim(), suffix)
        };

        // Highlight matching terms
        let highlighted = highlight_terms(&content, &query_terms);

        // Print content (left-aligned for easy copy/paste)
        println!("{}\n", highlighted);
    }

    Ok(())
}

/// Common stop words to skip when highlighting (unless part of a phrase)
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "it", "its",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "what", "which", "who", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
];

/// Check if a word is a stop word
fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(&word.to_lowercase().as_str())
}

/// Check if a term contains special characters (operators, punctuation)
fn has_special_chars(term: &str) -> bool {
    term.chars().any(|c| !c.is_alphanumeric())
}

/// Check if a term is a single letter (variable name like k, x, n, K, X)
fn is_single_letter(term: &str) -> bool {
    term.len() == 1 && term.chars().next().map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
}

/// Check if a term is an acronym/initialism (all uppercase letters, 2+ chars)
fn is_acronym(term: &str) -> bool {
    term.len() >= 2 && term.chars().all(|c| c.is_ascii_uppercase())
}

/// Highlight query terms in text using ripgrep-style red
/// - Full phrase matching first (includes all terms)
/// - Single letters matched case-sensitively (variables like k, K)
/// - Acronyms matched case-sensitively (ML, API, NASA)
/// - Terms with operators matched literally
/// - Stop words skipped for individual matches (but included in phrases)
fn highlight_terms(text: &str, terms: &[&str]) -> String {
    if terms.is_empty() {
        return text.to_string();
    }

    let text_lower = text.to_lowercase();

    // Categorize terms for individual matching (phrase matching uses all terms)
    let content_terms: Vec<&str> = terms
        .iter()
        .filter(|t| !is_stop_word(t) && !has_special_chars(t) && !is_single_letter(t) && !is_acronym(t) && t.len() >= 3)
        .copied()
        .collect();

    let special_terms: Vec<&str> = terms
        .iter()
        .filter(|t| has_special_chars(t))
        .copied()
        .collect();

    let letter_terms: Vec<&str> = terms
        .iter()
        .filter(|t| is_single_letter(t))
        .copied()
        .collect();

    let acronym_terms: Vec<&str> = terms
        .iter()
        .filter(|t| is_acronym(t))
        .copied()
        .collect();

    // Find all positions to highlight
    let mut matches: Vec<(usize, usize)> = Vec::new();

    // 1. First, try to find the full query as a phrase (for equations etc.)
    let full_query = terms.join(" ");
    if terms.len() >= 3 {
        let mut search_start = 0;
        while let Some(pos) = text[search_start..].find(&full_query) {
            let abs_pos = search_start + pos;
            matches.push((abs_pos, abs_pos + full_query.len()));
            search_start = abs_pos + 1;
        }
        // Also try case-insensitive
        let full_query_lower = full_query.to_lowercase();
        search_start = 0;
        while let Some(pos) = text_lower[search_start..].find(&full_query_lower) {
            let abs_pos = search_start + pos;
            let already = matches.iter().any(|(s, e)| abs_pos >= *s && abs_pos + full_query.len() <= *e);
            if !already {
                matches.push((abs_pos, abs_pos + full_query.len()));
            }
            search_start = abs_pos + 1;
        }
    }

    // 2. Match terms with special characters (literal, case-insensitive)
    for term in &special_terms {
        let term_lower = term.to_lowercase();
        let mut search_start = 0;

        while let Some(pos) = text_lower[search_start..].find(&term_lower) {
            let abs_pos = search_start + pos;
            let end_pos = abs_pos + term.len();

            let already_covered = matches.iter().any(|(s, e)| abs_pos >= *s && end_pos <= *e);
            if !already_covered {
                matches.push((abs_pos, end_pos));
            }
            search_start = abs_pos + 1;
        }
    }

    // 3. Match single letters case-sensitively (variables like k, K, x, n)
    for term in &letter_terms {
        let mut search_start = 0;

        while let Some(pos) = text[search_start..].find(term) {
            let abs_pos = search_start + pos;
            let end_pos = abs_pos + term.len();

            // Check word boundaries
            let before_ok = abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after_ok = end_pos >= text.len() || !text.as_bytes()[end_pos].is_ascii_alphanumeric();

            if before_ok && after_ok {
                let already_covered = matches.iter().any(|(s, e)| abs_pos >= *s && end_pos <= *e);
                if !already_covered {
                    matches.push((abs_pos, end_pos));
                }
            }
            search_start = abs_pos + 1;
        }
    }

    // 4. Match acronyms case-sensitively (ML, API, NASA, etc.)
    for term in &acronym_terms {
        let mut search_start = 0;

        while let Some(pos) = text[search_start..].find(term) {
            let abs_pos = search_start + pos;
            let end_pos = abs_pos + term.len();

            // Check word boundaries
            let before_ok = abs_pos == 0 || !text.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after_ok = end_pos >= text.len() || !text.as_bytes()[end_pos].is_ascii_alphanumeric();

            if before_ok && after_ok {
                let already_covered = matches.iter().any(|(s, e)| abs_pos >= *s && end_pos <= *e);
                if !already_covered {
                    matches.push((abs_pos, end_pos));
                }
            }
            search_start = abs_pos + 1;
        }
    }

    // 5. Match regular content words (case-insensitive, word boundary)
    for term in &content_terms {
        let term_lower = term.to_lowercase();
        let mut search_start = 0;

        while let Some(pos) = text_lower[search_start..].find(&term_lower) {
            let abs_pos = search_start + pos;
            let end_pos = abs_pos + term_lower.len();

            // Check word boundaries
            let before_ok = abs_pos == 0 || !text_lower.as_bytes()[abs_pos - 1].is_ascii_alphanumeric();
            let after_ok = end_pos >= text_lower.len() || !text_lower.as_bytes()[end_pos].is_ascii_alphanumeric();

            if before_ok && after_ok {
                let already_covered = matches.iter().any(|(s, e)| abs_pos >= *s && end_pos <= *e);
                if !already_covered {
                    matches.push((abs_pos, end_pos));
                }
            }
            search_start = abs_pos + 1;
        }
    }

    if matches.is_empty() {
        return text.to_string();
    }

    // Sort by start position and merge overlapping/adjacent ranges
    matches.sort_by_key(|(s, _)| *s);
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in matches {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }

    // Build result string with highlights (single pass)
    let mut result = String::with_capacity(text.len() + merged.len() * 20);
    let mut last_end = 0;

    for (start, end) in merged {
        result.push_str(&text[last_end..start]);
        result.push_str(BOLD);
        result.push_str(RED);
        result.push_str(&text[start..end]);
        result.push_str(RESET);
        last_end = end;
    }
    result.push_str(&text[last_end..]);

    result
}

/// Re-embed existing chunks with new/additional embedders
/// Useful when adding new vector types to an existing database
#[allow(dead_code)]
pub fn reembed(
    config: &Config,
    vector_names: Option<&[String]>,
) -> Result<()> {
    println!("Loading embedding models...");
    let embedder_set = EmbedderSet::from_config(&config.vectors)?;

    // Filter to specified vectors if provided
    let vectors_to_process: Vec<&String> = match vector_names {
        Some(names) => embedder_set.names()
            .filter(|n| names.iter().any(|x| x == *n))
            .collect(),
        None => embedder_set.names().collect(),
    };

    if vectors_to_process.is_empty() {
        println!("No vectors to process.");
        return Ok(());
    }

    for name in &vectors_to_process {
        if let Some(embedder) = embedder_set.get(name) {
            println!(
                "  {} ({}, dim: {})",
                name,
                embedder.metadata().name,
                embedder.dimension()
            );
        }
    }

    let store = Store::open(config)?;

    // Get all chunks
    let results = store.db().query("SELECT _rowid FROM Chunk")?;
    let chunk_count = results.len();
    println!("Found {} chunks to re-embed", chunk_count);

    // For each chunk, generate new embeddings and update
    let tx = store.db().begin_read()?;
    let mut processed = 0;

    for row in results {
        let id = EntityId::new(row.get_as::<i64>(0)? as u64);

        if let Some(entity) = tx.get_entity(id)? {
            if let Some(manifoldb::Value::String(content)) = entity.get_property("content") {
                // Generate embeddings for specified vectors
                let mut _new_embeddings = HashMap::new();
                for name in &vectors_to_process {
                    if let Some(embedder) = embedder_set.get(name) {
                        let embedding = embedder.embed(content)?;
                        _new_embeddings.insert((*name).clone(), embedding);
                    }
                }

                // TODO: Update entity with new embeddings
                // This would require a new method in Store to update just the embeddings
                // For now, we just count what would be processed
                processed += 1;

                if processed % 100 == 0 {
                    println!("  Processed {}/{} chunks", processed, chunk_count);
                }
            }
        }
    }

    println!("Re-embedding complete! Processed {} chunks.", processed);
    println!("Note: Entity updates not yet implemented - this is a dry run.");

    Ok(())
}
