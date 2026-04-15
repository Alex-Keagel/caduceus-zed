use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Parses Product Requirements Documents (PRDs) into structured tasks with
/// dependency analysis and next-step recommendations. Use this to break down
/// large feature requests or specification documents into actionable work items.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusPrdToolInput {
    /// The PRD operation to perform.
    pub operation: PrdOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PrdOperation {
    /// Parse a PRD document into structured tasks
    Parse {
        /// The PRD text content to parse
        text: String,
    },
    /// Analyze parsed tasks and recommend next steps
    RecommendNext {
        /// The PRD text to parse
        text: String,
        /// Indices of already-completed tasks (0-based)
        #[serde(default)]
        completed: Vec<usize>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PrdTaskInfo {
    pub index: usize,
    pub title: String,
    pub description: String,
    pub priority: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TaskRecommendationInfo {
    pub index: usize,
    pub title: String,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusPrdToolOutput {
    Tasks {
        tasks: Vec<PrdTaskInfo>,
        dependencies: Vec<(usize, usize)>,
    },
    Recommendations {
        recommendations: Vec<TaskRecommendationInfo>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusPrdToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusPrdToolOutput) -> Self {
        match output {
            CaduceusPrdToolOutput::Tasks {
                tasks,
                dependencies,
            } => {
                let mut text = format!("## Parsed {} tasks from PRD\n\n", tasks.len());
                for t in &tasks {
                    text.push_str(&format!(
                        "{}. **{}** [{}]\n   {}\n\n",
                        t.index + 1,
                        t.title,
                        t.priority,
                        t.description
                    ));
                }
                if !dependencies.is_empty() {
                    text.push_str("## Dependencies\n");
                    for (from, to) in &dependencies {
                        text.push_str(&format!("- Task {} → Task {} (must complete first)\n", to + 1, from + 1));
                    }
                }
                text.into()
            }
            CaduceusPrdToolOutput::Recommendations { recommendations } => {
                if recommendations.is_empty() {
                    "All tasks completed or no recommendations available.".into()
                } else {
                    let mut text = "## Recommended Next Tasks\n\n".to_string();
                    for r in &recommendations {
                        text.push_str(&format!(
                            "- **Task {}**: {} — {}\n",
                            r.index + 1,
                            r.title,
                            r.reason
                        ));
                    }
                    text.into()
                }
            }
            CaduceusPrdToolOutput::Error { error } => {
                format!("PRD parse error: {error}").into()
            }
        }
    }
}

pub struct CaduceusPrdTool;

impl CaduceusPrdTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusPrdTool {
    type Input = CaduceusPrdToolInput;
    type Output = CaduceusPrdToolOutput;

    const NAME: &'static str = "caduceus_prd";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                PrdOperation::Parse { .. } => "Parse PRD".into(),
                PrdOperation::RecommendNext { .. } => "Recommend next tasks".into(),
            }
        } else {
            "PRD analysis".into()
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
                CaduceusPrdToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            match input.operation {
                PrdOperation::Parse { text } => {
                    let tasks = caduceus_bridge::orchestrator::OrchestratorBridge::parse_prd(&text);
                    let deps =
                        caduceus_bridge::orchestrator::OrchestratorBridge::infer_dependencies(
                            &tasks,
                        );

                    let task_infos: Vec<PrdTaskInfo> = tasks
                        .iter()
                        .enumerate()
                        .map(|(i, t)| PrdTaskInfo {
                            index: i,
                            title: t.title.clone(),
                            description: t.description.clone(),
                            priority: format!("P{}", t.priority),
                        })
                        .collect();

                    Ok(CaduceusPrdToolOutput::Tasks {
                        tasks: task_infos,
                        dependencies: deps,
                    })
                }
                PrdOperation::RecommendNext { text, completed } => {
                    let tasks = caduceus_bridge::orchestrator::OrchestratorBridge::parse_prd(&text);
                    let recs =
                        caduceus_bridge::orchestrator::OrchestratorBridge::recommend_next(
                            &tasks, &completed,
                        );

                    let recommendations: Vec<TaskRecommendationInfo> = recs
                        .into_iter()
                        .map(|r| TaskRecommendationInfo {
                            index: r.task_id,
                            title: format!("Task {}", r.task_id),
                            reason: r.reason,
                        })
                        .collect();

                    Ok(CaduceusPrdToolOutput::Recommendations { recommendations })
                }
            }
        })
    }
}
