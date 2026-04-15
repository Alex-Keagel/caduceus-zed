use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Infer progress from commits, tests, and files. Also suggest trigger phrases for agents.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusProgressToolInput {
    /// The progress operation to perform.
    pub operation: ProgressOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProgressOperation {
    /// Estimate progress from git commit messages referencing a task title.
    InferFromCommits {
        /// The task title to look for in commit messages.
        task_title: String,
        /// Recent commit messages.
        commit_messages: Vec<String>,
    },
    /// Progress from test suite pass rate (returns 0–100).
    InferFromTests {
        /// Total number of tests.
        total: usize,
        /// Number of passing tests.
        passing: usize,
    },
    /// Progress from file creation ratio (returns 0–100).
    InferFromFiles {
        /// Number of planned files.
        planned: usize,
        /// Number of files already created.
        created: usize,
    },
    /// Weighted average of commit/test/file progress (40/40/20).
    Combined {
        /// Commit-based progress (0–100).
        commits: f64,
        /// Test-based progress (0–100).
        tests: f64,
        /// File-based progress (0–100).
        files: f64,
    },
    /// Suggest trigger phrases for an agent based on its description.
    SuggestTriggers {
        /// A description of the agent's purpose.
        description: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusProgressToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusProgressToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusProgressToolOutput) -> Self {
        match output {
            CaduceusProgressToolOutput::Text { text } => text.into(),
            CaduceusProgressToolOutput::Error { error } => {
                format!("Progress error: {error}").into()
            }
        }
    }
}

pub struct CaduceusProgressTool;

impl CaduceusProgressTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusProgressTool {
    type Input = CaduceusProgressToolInput;
    type Output = CaduceusProgressToolOutput;

    const NAME: &'static str = "caduceus_progress";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                ProgressOperation::InferFromCommits { .. } => "from commits",
                ProgressOperation::InferFromTests { .. } => "from tests",
                ProgressOperation::InferFromFiles { .. } => "from files",
                ProgressOperation::Combined { .. } => "combined",
                ProgressOperation::SuggestTriggers { .. } => "suggest triggers",
            };
            format!("Progress {op}").into()
        } else {
            "Progress inference".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusProgressToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let text = match input.operation {
                ProgressOperation::InferFromCommits {
                    task_title,
                    commit_messages,
                } => {
                    let progress =
                        caduceus_bridge::orchestrator::OrchestratorBridge::infer_from_commits(
                            &task_title,
                            &commit_messages,
                        );
                    format!(
                        "Progress: {:.1}%, Confidence: {:.2}",
                        progress.percentage, progress.confidence
                    )
                }
                ProgressOperation::InferFromTests { total, passing } => {
                    let pct = caduceus_bridge::orchestrator::OrchestratorBridge::infer_from_tests(
                        total, passing,
                    );
                    format!("Test progress: {pct:.1}% ({passing}/{total} passing)")
                }
                ProgressOperation::InferFromFiles { planned, created } => {
                    let pct = caduceus_bridge::orchestrator::OrchestratorBridge::infer_from_files(
                        planned, created,
                    );
                    format!("File progress: {pct:.1}% ({created}/{planned} created)")
                }
                ProgressOperation::Combined {
                    commits,
                    tests,
                    files,
                } => {
                    let combined =
                        caduceus_bridge::orchestrator::OrchestratorBridge::combined_progress(
                            commits, tests, files,
                        );
                    format!(
                        "Combined progress: {combined:.1}% (commits: {commits:.1}, tests: {tests:.1}, files: {files:.1})"
                    )
                }
                ProgressOperation::SuggestTriggers { description } => {
                    let triggers =
                        caduceus_bridge::orchestrator::OrchestratorBridge::suggest_triggers(
                            &description,
                        );
                    if triggers.is_empty() {
                        "No trigger phrases suggested".to_string()
                    } else {
                        triggers
                            .iter()
                            .map(|t| format!("- \"{t}\""))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
            };

            Ok(CaduceusProgressToolOutput::Text { text })
        })
    }
}
