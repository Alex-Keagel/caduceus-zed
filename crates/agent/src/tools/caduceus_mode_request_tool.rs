use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Request a mode change. Mode changes require user approval because they
/// affect what tools are available (privilege rings). The user will see
/// the requested mode and must confirm before it takes effect.
///
/// Available modes:
/// - `plan`: Read-only analysis, produce action plans (Ring 0)
/// - `act`: Execute changes with approval (Ring 1, default)
/// - `research`: Read-only exploration and summarization (Ring 0)
/// - `autopilot`: Fully autonomous — no approval needed (Ring 2)
/// - `architect`: High-level design discussion (Ring 0)
/// - `debug`: Investigate errors with step-by-step trace (Ring 1)
/// - `review`: Code review with structured findings (Ring 0)
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusModeRequestToolInput {
    /// The mode to switch to.
    pub mode: String,
    /// Why you need this mode change — explain to the user.
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusModeRequestToolOutput {
    Success { message: String },
    Error { error: String },
}

impl From<CaduceusModeRequestToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusModeRequestToolOutput) -> Self {
        match output {
            CaduceusModeRequestToolOutput::Success { message } => message.into(),
            CaduceusModeRequestToolOutput::Error { error } => {
                format!("Mode change error: {error}").into()
            }
        }
    }
}

pub struct CaduceusModeRequestTool;

impl CaduceusModeRequestTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusModeRequestTool {
    type Input = CaduceusModeRequestToolInput;
    type Output = CaduceusModeRequestToolOutput;

    const NAME: &'static str = "caduceus_mode_request";

    fn kind() -> acp::ToolKind {
        // Execute kind triggers user confirmation dialog
        acp::ToolKind::Execute
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!(
                "Request mode: {} — {}",
                input.mode,
                super::truncate_str(&input.reason, 40)
            )
            .into()
        } else {
            "Request mode change".into()
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
                CaduceusModeRequestToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let valid_modes = [
                "plan", "act", "research", "autopilot", "architect", "debug", "review",
            ];

            if !valid_modes.contains(&input.mode.as_str()) {
                return Err(CaduceusModeRequestToolOutput::Error {
                    error: format!(
                        "Invalid mode '{}'. Valid modes: {}",
                        input.mode,
                        valid_modes.join(", ")
                    ),
                });
            }

            // The tool kind is Execute, so Zed will show a confirmation dialog.
            // If we reach here, the user approved the mode change.
            // The mode will be applied on the next prompt via session_modes.
            log::info!(
                "[caduceus] Mode change APPROVED: {} (reason: {})",
                input.mode,
                input.reason
            );

            Ok(CaduceusModeRequestToolOutput::Success {
                message: format!(
                    "📋 Mode change to **{}** has been noted. \
                    To apply it, select '{}' from the profile dropdown in the input bar. \
                    The mode change is not automatic — the user controls the active profile.",
                    input.mode, input.mode
                ),
            })
        })
    }
}
