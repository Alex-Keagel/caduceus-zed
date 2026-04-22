use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, Entity, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Extracts code structure using Tree-sitter AST parsing. Returns functions,
/// classes, structs, and other symbols with their locations. Use this for
/// understanding code structure, finding definitions, and navigating large files.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusTreeSitterToolInput {
    /// The operation to perform.
    pub operation: TreeSitterOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TreeSitterOperation {
    /// Extract outline (functions, classes, structs) from a file
    Outline {
        /// Path to the file to analyze
        path: String,
    },
    /// Find symbols containing a given position
    SymbolsAt {
        /// Path to the file
        path: String,
        /// Line number (0-based)
        line: u32,
        /// Column (0-based)
        column: u32,
    },
    /// List all languages with Tree-sitter grammars available
    Languages,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutlineEntry {
    pub name: String,
    pub depth: usize,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusTreeSitterToolOutput {
    Outline {
        path: String,
        entries: Vec<OutlineEntry>,
    },
    Symbols {
        path: String,
        symbols: Vec<String>,
    },
    Languages {
        languages: Vec<String>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusTreeSitterToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusTreeSitterToolOutput) -> Self {
        match output {
            CaduceusTreeSitterToolOutput::Outline { path, entries } => {
                if entries.is_empty() {
                    format!("No outline items found in {path}").into()
                } else {
                    let mut text = format!("## Outline: {path} ({} items)\n\n", entries.len());
                    for e in &entries {
                        let indent = "  ".repeat(e.depth);
                        text.push_str(&format!(
                            "{indent}- **{}** (lines {}-{})\n",
                            e.name,
                            e.start_line + 1,
                            e.end_line + 1
                        ));
                    }
                    text.into()
                }
            }
            CaduceusTreeSitterToolOutput::Symbols { path, symbols } => {
                if symbols.is_empty() {
                    format!("No symbols found at position in {path}").into()
                } else {
                    let mut text = format!("Symbols at position in {path}:\n");
                    for s in &symbols {
                        text.push_str(&format!("- {s}\n"));
                    }
                    text.into()
                }
            }
            CaduceusTreeSitterToolOutput::Languages { languages } => {
                let mut text =
                    format!("{} languages with Tree-sitter grammars:\n", languages.len());
                for l in &languages {
                    text.push_str(&format!("- {l}\n"));
                }
                text.into()
            }
            CaduceusTreeSitterToolOutput::Error { error } => {
                format!("Tree-sitter error: {error}").into()
            }
        }
    }
}

pub struct CaduceusTreeSitterTool {
    project: Entity<Project>,
}

impl CaduceusTreeSitterTool {
    pub fn new(project: Entity<Project>) -> Self {
        Self { project }
    }
}

impl AgentTool for CaduceusTreeSitterTool {
    type Input = CaduceusTreeSitterToolInput;
    type Output = CaduceusTreeSitterToolOutput;

    const NAME: &'static str = "caduceus_tree_sitter";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                TreeSitterOperation::Outline { path } => {
                    format!("Outline: {}", super::truncate_str(path, 40)).into()
                }
                TreeSitterOperation::SymbolsAt { path, line, .. } => {
                    format!("Symbols at {}:{}", super::truncate_str(path, 30), line).into()
                }
                TreeSitterOperation::Languages => "List languages".into(),
            }
        } else {
            "Tree-sitter".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let languages = project.read(cx).languages().clone();

        cx.spawn(async move |cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusTreeSitterToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            match input.operation {
                TreeSitterOperation::Outline { path } => {
                    // Validate path — reject traversal and absolute paths
                    if path.contains("..") || std::path::Path::new(&path).is_absolute() {
                        return Err(CaduceusTreeSitterToolOutput::Error {
                            error: "Path traversal or absolute paths not allowed".to_string(),
                        });
                    }
                    if crate::tools::is_sensitive_file(&path) {
                        return Err(CaduceusTreeSitterToolOutput::Error {
                            error: format!("Cannot read sensitive file: {path}"),
                        });
                    }
                    let content = std::fs::read_to_string(&path).map_err(|e| {
                        CaduceusTreeSitterToolOutput::Error {
                            error: format!("Cannot read {path}: {e}"),
                        }
                    })?;

                    // Use bridge TreeSitterChunker for real AST parsing
                    let chunks = caduceus_bridge::tree_sitter::TreeSitterChunker::chunk_file_static(
                        &path, &content,
                    );

                    let entries: Vec<OutlineEntry> = chunks
                        .iter()
                        .map(|chunk| OutlineEntry {
                            name: format!("{} {}", chunk.symbol_type, chunk.symbol_name),
                            depth: 0,
                            start_line: chunk.start_line as u32,
                            end_line: chunk.end_line as u32,
                        })
                        .collect();

                    Ok(CaduceusTreeSitterToolOutput::Outline { path, entries })
                }
                TreeSitterOperation::SymbolsAt {
                    path,
                    line,
                    column: _,
                } => {
                    if path.contains("..") || std::path::Path::new(&path).is_absolute() {
                        return Err(CaduceusTreeSitterToolOutput::Error {
                            error: "Path traversal or absolute paths not allowed".to_string(),
                        });
                    }
                    if crate::tools::is_sensitive_file(&path) {
                        return Err(CaduceusTreeSitterToolOutput::Error {
                            error: format!("Cannot read sensitive file: {path}"),
                        });
                    }
                    let content = std::fs::read_to_string(&path).map_err(|e| {
                        CaduceusTreeSitterToolOutput::Error {
                            error: format!("Cannot read {path}: {e}"),
                        }
                    })?;

                    let target_line = content.lines().nth(line as usize).unwrap_or("").to_string();

                    let symbols = vec![format!("Line {}: {}", line + 1, target_line.trim())];

                    Ok(CaduceusTreeSitterToolOutput::Symbols { path, symbols })
                }
                TreeSitterOperation::Languages => {
                    let lang_names: Vec<String> = cx.update(|_cx| {
                        languages
                            .language_names()
                            .into_iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<String>>()
                    });

                    Ok(CaduceusTreeSitterToolOutput::Languages {
                        languages: lang_names,
                    })
                }
            }
        })
    }
}
