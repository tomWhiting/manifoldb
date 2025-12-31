//! Embedded tree-sitter queries for code chunking
//!
//! Queries are embedded at compile time from assets/queries/

use super::Language;

/// Query set for a language
#[derive(Debug, Clone)]
pub struct QuerySet {
    /// Embedding query (optimized for chunking)
    pub embedding: Option<&'static str>,
    /// Outline query (fallback for symbol extraction)
    pub outline: Option<&'static str>,
}

impl QuerySet {
    /// Get the best query for chunking (prefer embedding, fall back to outline)
    pub fn chunking_query(&self) -> Option<&'static str> {
        self.embedding.or(self.outline)
    }
}

/// Get embedded queries for a language
pub fn get_queries(lang: Language) -> Option<QuerySet> {
    match lang {
        Language::Rust => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/rust/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/rust/outline.scm")),
        }),
        Language::Python => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/python/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/python/outline.scm")),
        }),
        Language::TypeScript => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/typescript/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/typescript/outline.scm")),
        }),
        Language::JavaScript => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/javascript/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/javascript/outline.scm")),
        }),
        Language::Tsx => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/tsx/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/tsx/outline.scm")),
        }),
        Language::Go => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/go/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/go/outline.scm")),
        }),
        Language::C => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/c/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/c/outline.scm")),
        }),
        Language::Cpp => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/cpp/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/cpp/outline.scm")),
        }),
        Language::Json => Some(QuerySet {
            embedding: Some(include_str!("../../../assets/queries/json/embedding.scm")),
            outline: Some(include_str!("../../../assets/queries/json/outline.scm")),
        }),
        Language::Yaml => Some(QuerySet {
            embedding: None,
            outline: Some(include_str!("../../../assets/queries/yaml/outline.scm")),
        }),
        Language::Css => Some(QuerySet {
            embedding: None,
            outline: Some(include_str!("../../../assets/queries/css/outline.scm")),
        }),
        Language::Markdown => Some(QuerySet {
            embedding: None,
            outline: Some(include_str!("../../../assets/queries/markdown/outline.scm")),
        }),
        Language::Bash => Some(QuerySet {
            embedding: None,
            outline: None, // Bash doesn't have outline.scm in Zed
        }),
    }
}
