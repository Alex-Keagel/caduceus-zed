use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Indexes project files for semantic search. Run this before using `caduceus_semantic_search`
/// to populate the search index. Can index an entire directory or re-index a single file.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusIndexToolInput {
    /// Path to index. Can be a directory (indexes recursively) or a single file.
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusIndexToolOutput {
    Success {
        chunks_indexed: usize,
        total_chunks: usize,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusIndexToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusIndexToolOutput) -> Self {
        match output {
            CaduceusIndexToolOutput::Success {
                chunks_indexed,
                total_chunks,
            } => format!(
                "Indexed {chunks_indexed} new chunks. Total chunks in index: {total_chunks}."
            )
            .into(),
            CaduceusIndexToolOutput::Error { error } => {
                format!("Indexing error: {error}").into()
            }
        }
    }
}

pub struct CaduceusIndexTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusIndexTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusIndexTool {
    type Input = CaduceusIndexToolInput;
    type Output = CaduceusIndexToolOutput;

    const NAME: &'static str = "caduceus_index";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!("Index: {}", input.path).into()
        } else {
            "Index project".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let engine = self.engine.clone();
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusIndexToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let path = PathBuf::from(&input.path);

            // Validate path is within project root (prevent indexing arbitrary filesystem paths)
            let canonical_path = path.canonicalize().unwrap_or_else(|_| path.clone());
            let canonical_root = engine.project_root.canonicalize()
                .unwrap_or_else(|_| engine.project_root.clone());
            if !canonical_path.starts_with(&canonical_root) {
                return Err(CaduceusIndexToolOutput::Error {
                    error: format!(
                        "Path '{}' is outside the project root '{}'",
                        path.display(),
                        engine.project_root.display()
                    ),
                });
            }

            let result = if path.is_file() {
                engine.reindex_file(&path).await
            } else {
                engine.index_directory(&path).await
            };

            match result {
                Ok(chunks_indexed) => {
                    let total_chunks = engine.index_chunk_count().await;
                    Ok(CaduceusIndexToolOutput::Success {
                        chunks_indexed,
                        total_chunks,
                    })
                }
                Err(e) => Err(CaduceusIndexToolOutput::Error { error: e }),
            }
        })
    }
}
