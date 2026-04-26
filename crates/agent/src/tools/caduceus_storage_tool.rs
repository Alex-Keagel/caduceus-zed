use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Manage tasks and project snapshots via the Caduceus storage bridge.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusStorageToolInput {
    /// The storage operation to perform.
    pub operation: StorageOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StorageOperation {
    /// List all tasks in the project.
    ListTasks,
    /// Save (upsert) a task. Provide `task` as a JSON object with at least `id`, `title`, `status`.
    SaveTask {
        /// The task as a JSON object.
        task: serde_json::Value,
    },
    /// Delete a task by ID.
    DeleteTask {
        /// The task ID to delete.
        id: String,
    },
    /// Export a session transcript to a file.
    ExportTranscript {
        /// The session ID to export.
        session_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusStorageToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusStorageToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusStorageToolOutput) -> Self {
        match output {
            CaduceusStorageToolOutput::Text { text } => text.into(),
            CaduceusStorageToolOutput::Error { error } => format!("Storage error: {error}").into(),
        }
    }
}

pub struct CaduceusStorageTool {
    project_root: PathBuf,
    bridge: Arc<caduceus_bridge::storage::StorageBridge>,
}

impl CaduceusStorageTool {
    pub fn new(project_root: PathBuf) -> Self {
        let bridge = caduceus_bridge::storage::StorageBridge::open_default().unwrap_or_else(|_| {
            caduceus_bridge::storage::StorageBridge::open_in_memory()
                .expect("in-memory storage should always succeed")
        });
        Self {
            project_root,
            bridge: Arc::new(bridge),
        }
    }
}

impl AgentTool for CaduceusStorageTool {
    type Input = CaduceusStorageToolInput;
    type Output = CaduceusStorageToolOutput;

    const NAME: &'static str = "caduceus_storage";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                StorageOperation::ListTasks => "list tasks",
                StorageOperation::SaveTask { .. } => "save task",
                StorageOperation::DeleteTask { .. } => "delete task",
                StorageOperation::ExportTranscript { .. } => "export transcript",
            };
            format!("Storage {op}").into()
        } else {
            "Storage operation".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let bridge = self.bridge.clone();
        let project_root = self.project_root.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusStorageToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let result: Result<String, String> = match input.operation {
                StorageOperation::ListTasks => bridge.list_tasks(&project_root).map(|tasks| {
                    if tasks.is_empty() {
                        "No tasks found".to_string()
                    } else {
                        serde_json::to_string_pretty(&tasks).unwrap_or_else(|_| {
                            tasks
                                .iter()
                                .map(|t| format!("{t}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                    }
                }),
                StorageOperation::SaveTask { task } => bridge
                    .save_task(&project_root, &task)
                    .map(|_| "Task saved".to_string()),
                StorageOperation::DeleteTask { id } => bridge
                    .delete_task(&project_root, &id)
                    .map(|_| format!("Task '{id}' deleted")),
                StorageOperation::ExportTranscript { session_id } => {
                    // Export transcript is not available without caduceus_core::SessionId.
                    // Return info about the session ID for manual export.
                    Ok(format!(
                        "Transcript export requested for session '{session_id}'. \
                         Use the CLI or storage bridge directly to export."
                    ))
                }
            };

            match result {
                Ok(text) => Ok(CaduceusStorageToolOutput::Text { text }),
                Err(e) => Err(CaduceusStorageToolOutput::Error { error: e }),
            }
        })
    }
}
