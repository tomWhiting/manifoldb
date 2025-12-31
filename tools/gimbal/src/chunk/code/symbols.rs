//! Symbol extraction from tree-sitter AST nodes

use super::super::{SymbolInfo, SymbolKind, Visibility};
use super::Language;

/// Extract symbol kind from a tree-sitter node type
pub fn symbol_kind_from_node(node_type: &str, lang: Language) -> Option<SymbolKind> {
    match lang {
        Language::Rust => rust_symbol_kind(node_type),
        Language::Python => python_symbol_kind(node_type),
        Language::TypeScript | Language::JavaScript | Language::Tsx => js_symbol_kind(node_type),
        Language::Go => go_symbol_kind(node_type),
        Language::C | Language::Cpp => c_symbol_kind(node_type),
        _ => None,
    }
}

fn rust_symbol_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_item" | "function_signature_item" => Some(SymbolKind::Function),
        "struct_item" => Some(SymbolKind::Struct),
        "enum_item" => Some(SymbolKind::Enum),
        "trait_item" => Some(SymbolKind::Trait),
        "impl_item" => Some(SymbolKind::Impl),
        "mod_item" => Some(SymbolKind::Module),
        "const_item" | "static_item" => Some(SymbolKind::Constant),
        "type_item" | "associated_type" => Some(SymbolKind::Type),
        "field_declaration" => Some(SymbolKind::Field),
        "macro_definition" => Some(SymbolKind::Macro),
        _ => None,
    }
}

fn python_symbol_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" => Some(SymbolKind::Function),
        "class_definition" => Some(SymbolKind::Class),
        _ => None,
    }
}

fn js_symbol_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" | "arrow_function" | "function" => Some(SymbolKind::Function),
        "class_declaration" | "class" => Some(SymbolKind::Class),
        "method_definition" => Some(SymbolKind::Method),
        "interface_declaration" => Some(SymbolKind::Interface),
        "type_alias_declaration" => Some(SymbolKind::Type),
        _ => None,
    }
}

fn go_symbol_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_declaration" | "method_declaration" => Some(SymbolKind::Function),
        "type_declaration" => Some(SymbolKind::Type),
        "type_spec" => Some(SymbolKind::Struct), // Could be struct, interface, etc.
        "const_spec" => Some(SymbolKind::Constant),
        _ => None,
    }
}

fn c_symbol_kind(node_type: &str) -> Option<SymbolKind> {
    match node_type {
        "function_definition" | "function_declarator" => Some(SymbolKind::Function),
        "struct_specifier" => Some(SymbolKind::Struct),
        "enum_specifier" => Some(SymbolKind::Enum),
        "type_definition" => Some(SymbolKind::Type),
        _ => None,
    }
}

/// Extract visibility from Rust visibility modifier text
pub fn parse_rust_visibility(vis_text: Option<&str>) -> Visibility {
    match vis_text {
        Some("pub") => Visibility::Public,
        Some(s) if s.starts_with("pub(crate)") => Visibility::Crate,
        Some(s) if s.starts_with("pub(super)") => Visibility::Super,
        _ => Visibility::Private,
    }
}

/// Build a SymbolInfo from extracted data
pub fn build_symbol_info(
    name: String,
    kind: SymbolKind,
    signature: Option<String>,
    visibility: Visibility,
    parent: Option<String>,
) -> SymbolInfo {
    SymbolInfo {
        name,
        kind,
        signature,
        visibility,
        parent,
    }
}
