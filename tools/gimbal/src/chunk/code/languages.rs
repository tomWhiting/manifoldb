//! Language grammar loading for tree-sitter

use anyhow::Result;
use tree_sitter::Language as TsLanguage;

use super::Language;

/// Get the tree-sitter grammar for a language
pub fn get_grammar(lang: Language) -> Result<TsLanguage> {
    match lang {
        Language::Rust => Ok(tree_sitter_rust::LANGUAGE.into()),
        Language::Python => Ok(tree_sitter_python::LANGUAGE.into()),
        Language::TypeScript => Ok(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        Language::JavaScript => Ok(tree_sitter_javascript::LANGUAGE.into()),
        Language::Tsx => Ok(tree_sitter_typescript::LANGUAGE_TSX.into()),
        Language::Go => Ok(tree_sitter_go::LANGUAGE.into()),
        Language::C => Ok(tree_sitter_c::LANGUAGE.into()),
        Language::Cpp => Ok(tree_sitter_cpp::LANGUAGE.into()),
        Language::Json => Ok(tree_sitter_json::LANGUAGE.into()),
        Language::Yaml => Ok(tree_sitter_yaml::LANGUAGE.into()),
        Language::Css => Ok(tree_sitter_css::LANGUAGE.into()),
        Language::Markdown => Ok(tree_sitter_md::LANGUAGE.into()),
        Language::Bash => Ok(tree_sitter_bash::LANGUAGE.into()),
    }
}

/// Check if a language has tree-sitter support
pub fn is_supported(lang: Language) -> bool {
    get_grammar(lang).is_ok()
}

/// List all supported languages
pub fn supported_languages() -> Vec<Language> {
    vec![
        Language::Rust,
        Language::Python,
        Language::TypeScript,
        Language::JavaScript,
        Language::Tsx,
        Language::Go,
        Language::C,
        Language::Cpp,
        Language::Json,
        Language::Yaml,
        Language::Css,
        Language::Markdown,
        Language::Bash,
    ]
}
