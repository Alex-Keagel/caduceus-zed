use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use caduceus_bridge::security::PermissionsBridge;

/// Policy engine: check permissions, audit, and generate compliance reports.
/// Grant/Revoke operations are intentionally excluded to prevent LLM self-escalation.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusPolicyToolInput {
    /// The policy operation to perform.
    pub operation: PolicyOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyOperation {
    /// Check if a tool is allowed under the current privilege ring.
    Check {
        /// The tool name to check (e.g. "bash", "read", "write").
        tool_name: String,
    },
    /// Generate a full OWASP security/compliance report.
    Report,
    /// Show audit log entries for a session.
    Audit {
        /// Session ID to query audit entries for.
        session_id: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusPolicyToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusPolicyToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusPolicyToolOutput) -> Self {
        match output {
            CaduceusPolicyToolOutput::Text { text } => text.into(),
            CaduceusPolicyToolOutput::Error { error } => {
                format!("Policy error: {error}").into()
            }
        }
    }
}

pub struct CaduceusPolicyTool {
    bridge: Arc<Mutex<PermissionsBridge>>,
}

impl CaduceusPolicyTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            bridge: Arc::new(Mutex::new(PermissionsBridge::new(project_root))),
        }
    }
}

impl AgentTool for CaduceusPolicyTool {
    type Input = CaduceusPolicyToolInput;
    type Output = CaduceusPolicyToolOutput;

    const NAME: &'static str = "caduceus_policy";

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
                PolicyOperation::Check { tool_name } => format!("check {tool_name}"),
                PolicyOperation::Report => "report".to_string(),
                PolicyOperation::Audit { .. } => "audit".to_string(),
            };
            format!("Policy {op}").into()
        } else {
            "Policy query".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let bridge = self.bridge.clone();
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| CaduceusPolicyToolOutput::Error {
                error: format!("Failed to receive input: {e}"),
            })?;

            let guard = bridge.lock().map_err(|e| CaduceusPolicyToolOutput::Error {
                error: format!("Lock poisoned: {e}"),
            })?;

            let text = match input.operation {
                PolicyOperation::Check { tool_name } => {
                    match guard.check_permission(&tool_name) {
                        Ok(()) => format!("✅ Tool '{tool_name}' is ALLOWED under current privilege ring."),
                        Err(reason) => format!("❌ Tool '{tool_name}' is DENIED: {reason}"),
                    }
                }
                PolicyOperation::Report => {
                    let report = guard.generate_security_report();
                    let score = guard.compliance_score();
                    format!(
                        "## Security & Compliance Report\n\nCompliance Score: {:.0}%\n\n{}",
                        score * 100.0,
                        report
                    )
                }
                PolicyOperation::Audit { session_id } => {
                    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| {
                        CaduceusPolicyToolOutput::Error {
                            error: format!("Invalid session ID (expected UUID): {e}"),
                        }
                    })?;
                    let sid = caduceus_core::SessionId(uuid);
                    let entries = guard.entries_for_session(&sid);
                    if entries.is_empty() {
                        format!("No audit entries for session {session_id}")
                    } else {
                        let mut out = format!("Audit log ({} entries):\n", entries.len());
                        for entry in &entries {
                            out.push_str(&format!(
                                "  [{:?}] {} on {} — {:?}\n",
                                entry.decision, entry.capability, entry.resource, entry.timestamp
                            ));
                        }
                        out
                    }
                }
            };

            Ok(CaduceusPolicyToolOutput::Text { text })
        })
    }
}
