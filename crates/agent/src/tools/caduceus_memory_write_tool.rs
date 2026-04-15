use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Writes to persistent project memory. Use this to store project preferences,
/// architectural decisions, user preferences, or any context that should persist
/// across sessions.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusMemoryWriteToolInput {
    /// The write operation to perform.
    pub operation: MemoryWriteOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemoryWriteOperation {
    /// Store a key-value pair
    Store {
        /// The key to store under
        key: String,
        /// The value to store
        value: String,
    },
    /// Delete a stored memory
    Delete {
        /// The key to delete
        key: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusMemoryWriteToolOutput {
    Success { message: String },
    Error { error: String },
}

impl From<CaduceusMemoryWriteToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusMemoryWriteToolOutput) -> Self {
        match output {
            CaduceusMemoryWriteToolOutput::Success { message } => message.into(),
            CaduceusMemoryWriteToolOutput::Error { error } => {
                format!("Memory write error: {error}").into()
            }
        }
    }
}

pub struct CaduceusMemoryWriteTool {
    project_root: PathBuf,
}

impl CaduceusMemoryWriteTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

impl AgentTool for CaduceusMemoryWriteTool {
    type Input = CaduceusMemoryWriteToolInput;
    type Output = CaduceusMemoryWriteToolOutput;

    const NAME: &'static str = "caduceus_memory_write";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                MemoryWriteOperation::Store { key, .. } => {
                    format!("Store memory: {key}").into()
                }
                MemoryWriteOperation::Delete { key } => {
                    format!("Delete memory: {key}").into()
                }
            }
        } else {
            "Write memory".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project_root = self.project_root.clone();
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusMemoryWriteToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            match input.operation {
                MemoryWriteOperation::Store { key, value } => {
                    match caduceus_bridge::memory::store(&project_root, &key, &value) {
                        Ok(()) => Ok(CaduceusMemoryWriteToolOutput::Success {
                            message: format!("Stored memory: {key}"),
                        }),
                        Err(e) => Err(CaduceusMemoryWriteToolOutput::Error { error: e }),
                    }
                }
                MemoryWriteOperation::Delete { key } => {
                    match caduceus_bridge::memory::delete(&project_root, &key) {
                        Ok(()) => Ok(CaduceusMemoryWriteToolOutput::Success {
                            message: format!("Deleted memory: {key}"),
                        }),
                        Err(e) => Err(CaduceusMemoryWriteToolOutput::Error { error: e }),
                    }
                }
            }
        })
    }
}
