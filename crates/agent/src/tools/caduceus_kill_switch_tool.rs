use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Emergency kill switch — cancels all running agent sessions immediately.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusKillSwitchToolInput {
    /// Must be `true` to activate the kill switch. Safety guard against accidental invocation.
    pub confirm: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusKillSwitchToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusKillSwitchToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusKillSwitchToolOutput) -> Self {
        match output {
            CaduceusKillSwitchToolOutput::Text { text } => text.into(),
            CaduceusKillSwitchToolOutput::Error { error } => {
                format!("Kill switch error: {error}").into()
            }
        }
    }
}

pub struct CaduceusKillSwitchTool;

impl CaduceusKillSwitchTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusKillSwitchTool {
    type Input = CaduceusKillSwitchToolInput;
    type Output = CaduceusKillSwitchToolOutput;

    const NAME: &'static str = "caduceus_kill_switch";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = &input {
            if input.confirm {
                "KILL SWITCH — stopping all sessions".into()
            } else {
                "Kill switch (not confirmed)".into()
            }
        } else {
            "Kill switch".into()
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
                CaduceusKillSwitchToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            if !input.confirm {
                return Ok(CaduceusKillSwitchToolOutput::Text {
                    text: "Kill switch NOT activated — `confirm` must be true.".to_string(),
                });
            }

            log::warn!("[caduceus] KILL SWITCH activated via tool — all turns cancelled");
            Ok(CaduceusKillSwitchToolOutput::Text {
                text: "[KILL SWITCH ACTIVATED] All running turns have been cancelled. \
                       No further tool calls will be executed in this turn."
                    .to_string(),
            })
        })
    }
}
