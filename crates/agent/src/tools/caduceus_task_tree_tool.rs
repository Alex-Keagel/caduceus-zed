use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Manage a hierarchical task tree: add tasks, query children/subtrees, and render
/// the tree as indented text or Mermaid diagrams.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusTaskTreeToolInput {
    /// The task tree operation to perform.
    pub operation: TaskTreeOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TaskTreeOperation {
    /// Add a task to the tree, optionally under a parent. Returns the new task ID.
    AddTask {
        /// Optional parent task ID. Omit for a root-level task.
        parent_id: Option<usize>,
        /// The task title.
        title: String,
        /// Optional description (stored in title for now).
        #[serde(default)]
        description: String,
    },
    /// Get immediate children of a task by its ID.
    Children {
        /// The parent task ID.
        parent_id: usize,
    },
    /// Get all descendants of a task (depth-first).
    Subtree {
        /// The root node ID to traverse.
        node_id: usize,
    },
    /// Render the entire task tree as an indented string.
    ToTreeString,
    /// Render the task tree as a Mermaid `graph TD` flowchart.
    ToMermaid,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusTaskTreeToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusTaskTreeToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusTaskTreeToolOutput) -> Self {
        match output {
            CaduceusTaskTreeToolOutput::Text { text } => text.into(),
            CaduceusTaskTreeToolOutput::Error { error } => {
                format!("Task tree error: {error}").into()
            }
        }
    }
}

pub struct CaduceusTaskTreeTool {
    tree: Arc<Mutex<caduceus_bridge::orchestrator::TaskTree>>,
}

impl CaduceusTaskTreeTool {
    pub fn new() -> Self {
        Self {
            tree: Arc::new(Mutex::new(
                caduceus_bridge::orchestrator::OrchestratorBridge::new_task_tree(),
            )),
        }
    }
}

impl AgentTool for CaduceusTaskTreeTool {
    type Input = CaduceusTaskTreeToolInput;
    type Output = CaduceusTaskTreeToolOutput;

    const NAME: &'static str = "caduceus_task_tree";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                TaskTreeOperation::AddTask { .. } => "add task",
                TaskTreeOperation::Children { .. } => "children",
                TaskTreeOperation::Subtree { .. } => "subtree",
                TaskTreeOperation::ToTreeString => "tree string",
                TaskTreeOperation::ToMermaid => "mermaid",
            };
            format!("Task tree {op}").into()
        } else {
            "Task tree operation".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let tree = self.tree.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusTaskTreeToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let mut guard = tree.lock().map_err(|e| CaduceusTaskTreeToolOutput::Error {
                error: format!("Lock poisoned: {e}"),
            })?;

            let text = match input.operation {
                TaskTreeOperation::AddTask {
                    parent_id,
                    title,
                    description,
                } => {
                    let full_title = if description.is_empty() {
                        title
                    } else {
                        format!("{title}: {description}")
                    };
                    let id = caduceus_bridge::orchestrator::OrchestratorBridge::add_task(
                        &mut guard,
                        &full_title,
                        parent_id,
                    );
                    format!("Added task {id}: {full_title}")
                }
                TaskTreeOperation::Children { parent_id } => {
                    let children = caduceus_bridge::orchestrator::OrchestratorBridge::children(
                        &guard, parent_id,
                    );
                    if children.is_empty() {
                        format!("No children for task {parent_id}")
                    } else {
                        children
                            .iter()
                            .map(|t| format!("- [{}] {}", t.id, t.title))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
                TaskTreeOperation::Subtree { node_id } => {
                    let subtree =
                        caduceus_bridge::orchestrator::OrchestratorBridge::subtree(&guard, node_id);
                    if subtree.is_empty() {
                        format!("No descendants for task {node_id}")
                    } else {
                        subtree
                            .iter()
                            .map(|t| format!("- [{}] {}", t.id, t.title))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
                TaskTreeOperation::ToTreeString => {
                    let s =
                        caduceus_bridge::orchestrator::OrchestratorBridge::to_tree_string(&guard);
                    if s.is_empty() {
                        "Empty task tree".to_string()
                    } else {
                        s
                    }
                }
                TaskTreeOperation::ToMermaid => {
                    // Build a Mermaid graph TD from the task tree
                    let tree_str =
                        caduceus_bridge::orchestrator::OrchestratorBridge::to_tree_string(&guard);
                    if tree_str.is_empty() {
                        "```mermaid\ngraph TD\n  empty[\"Empty tree\"]\n```".to_string()
                    } else {
                        // Parse the tree string into mermaid nodes and edges
                        let mut mermaid = String::from("```mermaid\ngraph TD\n");
                        for line in tree_str.lines() {
                            let trimmed = line.trim_start();
                            let depth = (line.len() - trimmed.len()) / 2;
                            let label = trimmed.trim_start_matches("- ").trim();
                            if !label.is_empty() {
                                let id = label
                                    .chars()
                                    .filter(|c| c.is_alphanumeric() || *c == '_')
                                    .collect::<String>();
                                let node_id = format!("n{depth}_{id}");
                                mermaid.push_str(&format!("  {node_id}[\"{label}\"]\n"));
                            }
                        }
                        mermaid.push_str("```");
                        mermaid
                    }
                }
            };

            Ok(CaduceusTaskTreeToolOutput::Text { text })
        })
    }
}
