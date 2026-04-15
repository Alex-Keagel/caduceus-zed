use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Conversation utilities: serialize history, extract sections, compact instructions,
/// and extract learnable memories from exchanges.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusConversationToolInput {
    /// The conversation operation to perform.
    pub operation: ConversationOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConversationOperation {
    /// Serialize a fresh conversation history to JSON (shows the format).
    Serialize,
    /// Extract `(heading, content)` section pairs from markdown text.
    ExtractSections {
        /// The markdown text to extract sections from.
        text: String,
    },
    /// Compact instructions to fit within a character budget.
    Compact {
        /// The content to compact.
        content: String,
        /// Maximum number of characters.
        max_chars: usize,
    },
    /// Extract learnable memories from a user/assistant exchange.
    ExtractMemories {
        /// The user's input message.
        user_input: String,
        /// The assistant's response.
        assistant_response: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusConversationToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusConversationToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusConversationToolOutput) -> Self {
        match output {
            CaduceusConversationToolOutput::Text { text } => text.into(),
            CaduceusConversationToolOutput::Error { error } => {
                format!("Conversation error: {error}").into()
            }
        }
    }
}

pub struct CaduceusConversationTool;

impl CaduceusConversationTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusConversationTool {
    type Input = CaduceusConversationToolInput;
    type Output = CaduceusConversationToolOutput;

    const NAME: &'static str = "caduceus_conversation";

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
                ConversationOperation::Serialize => "serialize",
                ConversationOperation::ExtractSections { .. } => "extract sections",
                ConversationOperation::Compact { .. } => "compact",
                ConversationOperation::ExtractMemories { .. } => "extract memories",
            };
            format!("Conversation {op}").into()
        } else {
            "Conversation operation".into()
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
                CaduceusConversationToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let text = match input.operation {
                ConversationOperation::Serialize => {
                    let history =
                        caduceus_bridge::orchestrator::OrchestratorBridge::new_history();
                    caduceus_bridge::orchestrator::OrchestratorBridge::conversation_serialize(
                        &history,
                    )
                    .unwrap_or_else(|e| format!("Serialize error: {e}"))
                }
                ConversationOperation::ExtractSections { text } => {
                    let sections =
                        caduceus_bridge::orchestrator::OrchestratorBridge::extract_sections(&text);
                    if sections.is_empty() {
                        "No sections found".to_string()
                    } else {
                        sections
                            .iter()
                            .map(|(heading, content)| format!("## {heading}\n{content}"))
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    }
                }
                ConversationOperation::Compact { content, max_chars } => {
                    caduceus_bridge::orchestrator::OrchestratorBridge::compact_instructions(
                        &content, max_chars,
                    )
                }
                ConversationOperation::ExtractMemories {
                    user_input,
                    assistant_response,
                } => {
                    let memories =
                        caduceus_bridge::orchestrator::OrchestratorBridge::extract_memories(
                            &user_input,
                            &assistant_response,
                        );
                    if memories.is_empty() {
                        "No learnable memories extracted".to_string()
                    } else {
                        memories
                            .iter()
                            .enumerate()
                            .map(|(i, m)| format!("{}. {m}", i + 1))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
            };

            Ok(CaduceusConversationToolOutput::Text { text })
        })
    }
}
