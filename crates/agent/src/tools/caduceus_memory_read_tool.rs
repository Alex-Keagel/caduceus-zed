use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Reads from persistent project memory. Memory persists across sessions and stores
/// key-value pairs for project preferences, decisions, and context.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusMemoryReadToolInput {
    /// The operation to perform.
    pub operation: MemoryReadOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MemoryReadOperation {
    /// Get a specific memory by key
    Get {
        /// The key to look up
        key: String,
    },
    /// List all stored memories
    List,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusMemoryReadToolOutput {
    Value { key: String, value: String },
    Entries { entries: Vec<(String, String)> },
    NotFound { message: String },
    Error { error: String },
}

impl From<CaduceusMemoryReadToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusMemoryReadToolOutput) -> Self {
        match output {
            CaduceusMemoryReadToolOutput::Value { key, value } => format!("{key}: {value}").into(),
            CaduceusMemoryReadToolOutput::Entries { entries } => {
                if entries.is_empty() {
                    "No memories stored.".into()
                } else {
                    let mut text = format!("{} stored memories:\n", entries.len());
                    for (k, v) in &entries {
                        text.push_str(&format!("- {k}: {v}\n"));
                    }
                    text.into()
                }
            }
            CaduceusMemoryReadToolOutput::NotFound { message } => message.into(),
            CaduceusMemoryReadToolOutput::Error { error } => {
                format!("Memory error: {error}").into()
            }
        }
    }
}

pub struct CaduceusMemoryReadTool {
    project_root: PathBuf,
}

impl CaduceusMemoryReadTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

impl AgentTool for CaduceusMemoryReadTool {
    type Input = CaduceusMemoryReadToolInput;
    type Output = CaduceusMemoryReadToolOutput;

    const NAME: &'static str = "caduceus_memory_read";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                MemoryReadOperation::Get { key } => format!("Read memory: {key}").into(),
                MemoryReadOperation::List => "List memories".into(),
            }
        } else {
            "Read memory".into()
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
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusMemoryReadToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            match input.operation {
                MemoryReadOperation::Get { key } => {
                    match caduceus_bridge::memory::get(&project_root, &key) {
                        Some(value) => Ok(CaduceusMemoryReadToolOutput::Value { key, value }),
                        None => Ok(CaduceusMemoryReadToolOutput::NotFound {
                            message: format!("No memory found for key: {key}"),
                        }),
                    }
                }
                MemoryReadOperation::List => {
                    let entries = caduceus_bridge::memory::list(&project_root);
                    Ok(CaduceusMemoryReadToolOutput::Entries { entries })
                }
            }
        })
    }
}
