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

/// Maximum number of semantic search results we will ever return in a single
/// call. Anything higher risks unbounded allocation (a malformed request with
/// `top_k = usize::MAX` would OOM the index walk).
pub(crate) const MAX_TOP_K: usize = 100;

/// Clamp the requested `top_k` to a safe range. Exposed for direct unit tests
/// so the safety bound can be verified without spinning up an engine.
pub(crate) fn clamp_top_k(requested: usize) -> usize {
    requested.clamp(1, MAX_TOP_K)
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
            format!(
                "Semantic search: \"{}\"",
                crate::tools::truncate_str(&input.query, 50)
            )
            .into()
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
            let input =
                input
                    .recv()
                    .await
                    .map_err(|e| CaduceusSemanticSearchToolOutput::Error {
                        error: format!("Failed to receive input: {e}"),
                    })?;

            // Clamp top_k so a malformed/malicious request can't allocate
            // an unbounded result vector (usize::MAX would OOM the index walk).
            let top_k = clamp_top_k(input.top_k);
            match engine
                .semantic_search_as(
                    "tool:caduceus_semantic_search",
                    caduceus_bridge::index_dag::AgentKind::Tool,
                    &input.query,
                    top_k,
                )
                .await
            {
                Ok(hits) => {
                    let results: Vec<SearchHit> = hits
                        .into_iter()
                        .filter(|(content, _)| {
                            let path = content.lines().next().unwrap_or("");
                            !crate::tools::is_sensitive_file(path)
                        })
                        .map(|(content, score)| SearchHit {
                            path: content.lines().next().unwrap_or("unknown").to_string(),
                            snippet: content,
                            score,
                        })
                        .collect();
                    Ok(CaduceusSemanticSearchToolOutput::Success { results })
                }
                Err(e) => Err(CaduceusSemanticSearchToolOutput::Error { error: e }),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for bug #22: `top_k` could be set to `usize::MAX`, which
    /// would let the index walk attempt to allocate billions of result slots
    /// and OOM the process. The clamp must hold the upper bound.
    #[test]
    fn top_k_clamped_to_max() {
        assert_eq!(clamp_top_k(usize::MAX), MAX_TOP_K);
        assert_eq!(clamp_top_k(MAX_TOP_K + 1), MAX_TOP_K);
        assert_eq!(clamp_top_k(1_000_000), MAX_TOP_K);
    }

    /// Regression for bug #22: a request with `top_k = 0` previously returned
    /// no results at best, panicked at worst (some downstream search backends
    /// assume `top_k >= 1`). Clamp the lower bound to 1.
    #[test]
    fn top_k_clamped_to_min() {
        assert_eq!(clamp_top_k(0), 1);
    }

    #[test]
    fn top_k_passthrough_in_range() {
        assert_eq!(clamp_top_k(1), 1);
        assert_eq!(clamp_top_k(10), 10);
        assert_eq!(clamp_top_k(MAX_TOP_K), MAX_TOP_K);
    }
}
