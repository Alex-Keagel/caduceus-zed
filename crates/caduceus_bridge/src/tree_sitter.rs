//! Tree-sitter code intelligence — AST-based symbol extraction and repo mapping.
//!
//! Replaces the engine's regex-based CodeChunker with real AST parsing
//! using Zed's tree-sitter grammars for accurate symbol boundaries.

use caduceus_omniscience::{Chunker, CodeChunk, SymbolType};
use std::path::Path;

/// Tree-sitter backed chunker implementing the engine's `Chunker` trait.
/// Uses real AST parsing for precise symbol extraction.
pub struct TreeSitterChunker;

impl TreeSitterChunker {
    pub fn new() -> Self {
        Self
    }

    /// Static convenience method — usable without importing Chunker trait
    pub fn chunk_file_static(path: &str, content: &str) -> Vec<CodeChunk> {
        let language = detect_language(path);
        match get_parser(&language) {
            Some(mut parser) => parse_with_tree_sitter(&mut parser, path, content, &language),
            None => {
                let fallback = caduceus_omniscience::CodeChunker::default();
                caduceus_omniscience::Chunker::chunk_file(&fallback, path, content)
            }
        }
    }
}

impl Default for TreeSitterChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker for TreeSitterChunker {
    fn chunk_file(&self, path: &str, content: &str) -> Vec<CodeChunk> {
        let language = detect_language(path);
        match get_parser(&language) {
            Some(mut parser) => parse_with_tree_sitter(&mut parser, path, content, &language),
            None => {
                // Fallback to engine's regex chunker for unsupported languages
                let fallback = caduceus_omniscience::CodeChunker::default();
                caduceus_omniscience::Chunker::chunk_file(&fallback, path, content)
            }
        }
    }
}

/// Generate an Aider-style repo map from a list of files.
/// Returns a compact outline of all symbols across the project.
pub fn generate_repo_map(files: &[(String, String)]) -> String {
    let chunker = TreeSitterChunker::new();
    let mut map = String::with_capacity(files.len() * 100);

    for (path, content) in files {
        let chunks = chunker.chunk_file(path, content);
        if chunks.is_empty() {
            continue;
        }

        map.push_str(&format!("{}:\n", path));
        for chunk in &chunks {
            let indent = "  ";
            let sym = match chunk.symbol_type {
                SymbolType::Function | SymbolType::Method => "fn",
                SymbolType::Struct => "struct",
                SymbolType::Class => "class",
                SymbolType::Enum => "enum",
                SymbolType::Trait | SymbolType::Interface => "trait",
                SymbolType::Import => "use",
                SymbolType::Module => "mod",
                SymbolType::Other => "def",
            };
            map.push_str(&format!(
                "{}{} {} (L{}-{})\n",
                indent, sym, chunk.symbol_name, chunk.start_line, chunk.end_line
            ));
        }
    }

    map
}

// ── Language detection ──────────────────────────────────────────────────────

fn detect_language(path: &str) -> String {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext {
        "rs" => "rust",
        "py" | "pyi" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "go" => "go",
        _ => "unknown",
    }
    .to_string()
}

fn get_parser(language: &str) -> Option<tree_sitter::Parser> {
    let ts_language = match language {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "typescript" | "javascript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        _ => None,
    }?;

    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&ts_language).ok()?;
    Some(parser)
}

// ── AST-based symbol extraction ─────────────────────────────────────────────

fn parse_with_tree_sitter(
    parser: &mut tree_sitter::Parser,
    path: &str,
    content: &str,
    language: &str,
) -> Vec<CodeChunk> {
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let mut chunks = Vec::new();
    let root = tree.root_node();
    let lines: Vec<&str> = content.lines().collect();

    extract_symbols_recursive(root, path, content, &lines, language, &mut chunks, 0);
    chunks
}

fn extract_symbols_recursive(
    node: tree_sitter::Node,
    path: &str,
    content: &str,
    lines: &[&str],
    language: &str,
    chunks: &mut Vec<CodeChunk>,
    depth: usize,
) {
    // Don't recurse too deep
    if depth > 10 {
        return;
    }

    let kind = node.kind();

    // Check if this node is a symbol definition
    if let Some((sym_type, sym_name)) = classify_node(kind, node, content, language) {
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;

        let node_content = if end_line <= lines.len() {
            lines[start_line.saturating_sub(1)..end_line.min(lines.len())]
                .join("\n")
        } else {
            node.utf8_text(content.as_bytes()).unwrap_or("").to_string()
        };

        let content_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            node_content.hash(&mut hasher);
            format!("{:x}", hasher.finish())
        };

        chunks.push(CodeChunk {
            file_path: path.to_string(),
            symbol_name: sym_name,
            symbol_type: sym_type,
            language: language.to_string(),
            start_line,
            end_line,
            content: node_content,
            content_hash,
        });
    }

    // Recurse into children for nested definitions
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_symbols_recursive(child, path, content, lines, language, chunks, depth + 1);
    }
}

fn classify_node(
    kind: &str,
    node: tree_sitter::Node,
    content: &str,
    language: &str,
) -> Option<(SymbolType, String)> {
    match language {
        "rust" => classify_rust_node(kind, node, content),
        "python" => classify_python_node(kind, node, content),
        "typescript" | "javascript" => classify_ts_node(kind, node, content),
        "go" => classify_go_node(kind, node, content),
        _ => None,
    }
}

fn get_child_text<'a>(node: tree_sitter::Node<'a>, field: &str, content: &'a str) -> Option<&'a str> {
    node.child_by_field_name(field)
        .and_then(|n| n.utf8_text(content.as_bytes()).ok())
}

// ── Rust ────────────────────────────────────────────────────────────────────

fn classify_rust_node(kind: &str, node: tree_sitter::Node, content: &str) -> Option<(SymbolType, String)> {
    match kind {
        "function_item" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Function, name.to_string()))
        }
        "struct_item" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Struct, name.to_string()))
        }
        "enum_item" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Enum, name.to_string()))
        }
        "trait_item" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Trait, name.to_string()))
        }
        "impl_item" => {
            let type_node = node.child_by_field_name("type")?;
            let name = type_node.utf8_text(content.as_bytes()).ok()?;
            Some((SymbolType::Class, format!("impl {}", name)))
        }
        "mod_item" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Module, name.to_string()))
        }
        "use_declaration" => {
            let text = node.utf8_text(content.as_bytes()).ok()?;
            let path: String = text.chars().skip(4).take(60).collect(); // skip "use "
            Some((SymbolType::Import, path.trim_end_matches(';').to_string()))
        }
        _ => None,
    }
}

// ── Python ──────────────────────────────────────────────────────────────────

fn classify_python_node(kind: &str, node: tree_sitter::Node, content: &str) -> Option<(SymbolType, String)> {
    match kind {
        "function_definition" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Function, name.to_string()))
        }
        "class_definition" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Class, name.to_string()))
        }
        "import_statement" | "import_from_statement" => {
            let text = node.utf8_text(content.as_bytes()).ok()?;
            let short: String = text.chars().take(60).collect();
            Some((SymbolType::Import, short))
        }
        _ => None,
    }
}

// ── TypeScript/JavaScript ───────────────────────────────────────────────────

fn classify_ts_node(kind: &str, node: tree_sitter::Node, content: &str) -> Option<(SymbolType, String)> {
    match kind {
        "function_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Function, name.to_string()))
        }
        "class_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Class, name.to_string()))
        }
        "interface_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Interface, name.to_string()))
        }
        "enum_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Enum, name.to_string()))
        }
        "method_definition" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Method, name.to_string()))
        }
        "arrow_function" | "function" => {
            // Only if it's a variable declaration like `const foo = () => {}`
            if let Some(parent) = node.parent() {
                if parent.kind() == "variable_declarator" {
                    let name = get_child_text(parent, "name", content)?;
                    return Some((SymbolType::Function, name.to_string()));
                }
            }
            None
        }
        "import_statement" => {
            let text = node.utf8_text(content.as_bytes()).ok()?;
            let short: String = text.chars().take(60).collect();
            Some((SymbolType::Import, short))
        }
        _ => None,
    }
}

// ── Go ──────────────────────────────────────────────────────────────────────

fn classify_go_node(kind: &str, node: tree_sitter::Node, content: &str) -> Option<(SymbolType, String)> {
    match kind {
        "function_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Function, name.to_string()))
        }
        "method_declaration" => {
            let name = get_child_text(node, "name", content)?;
            Some((SymbolType::Method, name.to_string()))
        }
        "type_declaration" => {
            // Contains type_spec children
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "type_spec" {
                    let name = get_child_text(child, "name", content)?;
                    let type_child = child.child_by_field_name("type")?;
                    let sym_type = match type_child.kind() {
                        "struct_type" => SymbolType::Struct,
                        "interface_type" => SymbolType::Interface,
                        _ => SymbolType::Other,
                    };
                    return Some((sym_type, name.to_string()));
                }
            }
            None
        }
        "import_declaration" => {
            let text = node.utf8_text(content.as_bytes()).ok()?;
            let short: String = text.chars().take(60).collect();
            Some((SymbolType::Import, short))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_parsing() {
        let code = r#"
use std::path::Path;

pub struct Config {
    name: String,
    value: u32,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), value: 0 }
    }
}

pub fn main() {
    let c = Config::new("test");
}

pub trait Processor {
    fn process(&self);
}

pub enum Status {
    Active,
    Inactive,
}
"#;
        let chunker = TreeSitterChunker::new();
        let chunks = chunker.chunk_file("src/main.rs", code);

        let names: Vec<&str> = chunks.iter().map(|c| c.symbol_name.as_str()).collect();
        assert!(names.contains(&"Config"), "Should find struct Config, got: {:?}", names);
        assert!(names.contains(&"main"), "Should find fn main, got: {:?}", names);
        assert!(names.contains(&"Processor"), "Should find trait Processor, got: {:?}", names);
        assert!(names.contains(&"Status"), "Should find enum Status, got: {:?}", names);
        assert!(chunks.len() >= 5, "Should find at least 5 symbols, found {}", chunks.len());
    }

    #[test]
    fn test_python_parsing() {
        let code = r#"
import os
from pathlib import Path

class DataProcessor:
    def __init__(self, path: str):
        self.path = path

    def process(self) -> list:
        return []

def main():
    dp = DataProcessor("/tmp")
"#;
        let chunker = TreeSitterChunker::new();
        let chunks = chunker.chunk_file("main.py", code);
        let names: Vec<&str> = chunks.iter().map(|c| c.symbol_name.as_str()).collect();
        assert!(names.contains(&"DataProcessor"), "Should find class, got: {:?}", names);
        assert!(names.contains(&"main"), "Should find function, got: {:?}", names);
    }

    #[test]
    fn test_repo_map() {
        let files = vec![
            ("src/main.rs".to_string(), "pub fn main() {}\npub struct App {}".to_string()),
            ("src/lib.rs".to_string(), "pub fn init() {}\npub trait Service {}".to_string()),
        ];
        let map = generate_repo_map(&files);
        assert!(map.contains("src/main.rs:"), "Should have file header");
        assert!(map.contains("fn main"), "Should list main function");
        assert!(map.contains("struct App"), "Should list App struct");
    }

    #[test]
    fn test_unknown_language_fallback() {
        let chunker = TreeSitterChunker::new();
        let chunks = chunker.chunk_file("data.csv", "a,b,c\n1,2,3");
        // Should use fallback chunker without panicking
        let _ = chunks;
    }
}
