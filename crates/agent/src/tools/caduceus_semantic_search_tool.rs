use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Performs semantic code search across the indexed project. Returns code chunks ranked by
/// relevance to the natural-language query. Use this when grep is too literal and you need
/// conceptual matches (e.g., "error handling patterns" or "authentication flow").
///
/// NOTE: Results depend on prior indexing via the `caduceus_index` tool. If no files are
/// indexed, results will be empty.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusSemanticSearchToolInput {
    /// Natural-language query describing what you're looking for.
    pub query: String,

    /// Maximum number of results to return (default: 10).
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusSemanticSearchToolOutput {
    Success { results: Vec<SearchHit> },
    Error { error: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

impl From<CaduceusSemanticSearchToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusSemanticSearchToolOutput) -> Self {
        match output {
            CaduceusSemanticSearchToolOutput::Success { results } => {
                if results.is_empty() {
                    "No semantic matches found. Try indexing the project first with `caduceus_index`.".into()
                } else {
                    let mut text = format!("Found {} semantic matches:\n", results.len());
                    for (i, hit) in results.iter().enumerate() {
                        text.push_str(&format!(
                            "\n{}. {} (score: {:.3})\n```\n{}\n```\n",
                            i + 1,
                            hit.path,
                            hit.score,
                            hit.snippet
                        ));
                    }
                    text.into()
                }
            }
            CaduceusSemanticSearchToolOutput::Error { error } => {
                format!("Semantic search error: {error}").into()
            }
        }
    }
}

pub struct CaduceusSemanticSearchTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusSemanticSearchTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusSemanticSearchTool {
    type Input = CaduceusSemanticSearchToolInput;
    type Output = CaduceusSemanticSearchToolOutput;

    const NAME: &'static str = "caduceus_semantic_search";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!("Semantic search: \"{}\"", crate::tools::truncate_str(&input.query, 50)).into()
        } else {
            "Semantic search".into()
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
                CaduceusSemanticSearchToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            match engine.semantic_search(&input.query, input.top_k).await {
                Ok(hits) => {
                    let results: Vec<SearchHit> = hits
                        .into_iter()
                        .map(|(content, score)| SearchHit {
                            path: content.lines().next().unwrap_or("unknown").to_string(),
                            snippet: content,
                            score,
                        })
                        .collect();
                    Ok(CaduceusSemanticSearchToolOutput::Success { results })
                }
                Err(e) => Err(CaduceusSemanticSearchToolOutput::Error {
                    error: e,
                }),
            }
        })
    }
}
