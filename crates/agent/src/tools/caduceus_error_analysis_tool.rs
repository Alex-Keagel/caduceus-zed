use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Analyzes code errors: detects parse errors, classifies error types,
/// and suggests fixes. Use this when debugging build failures or runtime errors.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusErrorAnalysisToolInput {
    /// The code content or error message to analyze.
    pub content: String,
    /// Optional file path for context.
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusErrorAnalysisToolOutput {
    Success {
        language: String,
        errors: Vec<String>,
        analysis: Vec<String>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusErrorAnalysisToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusErrorAnalysisToolOutput) -> Self {
        match output {
            CaduceusErrorAnalysisToolOutput::Success {
                language,
                errors,
                analysis,
            } => {
                let mut text = format!("Language: {language}\n");
                if errors.is_empty() {
                    text.push_str("No parse errors detected.\n");
                } else {
                    text.push_str(&format!("\n## {} Parse Errors\n", errors.len()));
                    for e in &errors {
                        text.push_str(&format!("- {e}\n"));
                    }
                }
                if !analysis.is_empty() {
                    text.push_str("\n## Analysis\n");
                    for a in &analysis {
                        text.push_str(&format!("- {a}\n"));
                    }
                }
                text.into()
            }
            CaduceusErrorAnalysisToolOutput::Error { error } => {
                format!("Error analysis failed: {error}").into()
            }
        }
    }
}

pub struct CaduceusErrorAnalysisTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusErrorAnalysisTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusErrorAnalysisTool {
    type Input = CaduceusErrorAnalysisToolInput;
    type Output = CaduceusErrorAnalysisToolOutput;

    const NAME: &'static str = "caduceus_error_analysis";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            if let Some(path) = &input.path {
                format!("Analyze errors: {path}").into()
            } else {
                "Analyze errors".into()
            }
        } else {
            "Error analysis".into()
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
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusErrorAnalysisToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let path = input.path.as_deref().unwrap_or("unknown");
            let language = engine.detect_language(std::path::Path::new(path));
            let parse_errors = engine.detect_parse_errors(&input.content, &language);
            let analysis = engine.analyze_error(&input.content);

            Ok(CaduceusErrorAnalysisToolOutput::Success {
                language,
                errors: parse_errors,
                analysis: vec![
                    format!("Category: {}", analysis.category),
                    format!("Root cause: {}", analysis.root_cause),
                    format!("Severity: {}", analysis.severity),
                ],
            })
        })
    }
}
