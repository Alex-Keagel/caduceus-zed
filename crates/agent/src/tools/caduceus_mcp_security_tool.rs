use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Scans MCP tool definitions for security issues: typosquatting, hidden instructions,
/// and suspicious patterns. Use this before connecting to untrusted MCP servers.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusMcpSecurityToolInput {
    /// The MCP security operation to perform.
    pub operation: McpSecurityOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpSecurityOperation {
    /// Check a tool name for typosquatting against known tools
    CheckTyposquatting { tool_name: String },
    /// Detect hidden instructions in a tool description
    DetectHiddenInstructions { description: String },
    /// Scan a full tool definition for security issues
    ScanToolDefinition { name: String, description: String },
    /// List supported MCP service categories
    ListServiceCategories,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusMcpSecurityToolOutput {
    Success { result: String },
    Error { error: String },
}

impl From<CaduceusMcpSecurityToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusMcpSecurityToolOutput) -> Self {
        match output {
            CaduceusMcpSecurityToolOutput::Success { result } => result.into(),
            CaduceusMcpSecurityToolOutput::Error { error } => {
                format!("MCP security error: {error}").into()
            }
        }
    }
}

pub struct CaduceusMcpSecurityTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusMcpSecurityTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusMcpSecurityTool {
    type Input = CaduceusMcpSecurityToolInput;
    type Output = CaduceusMcpSecurityToolOutput;

    const NAME: &'static str = "caduceus_mcp_security";

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
                McpSecurityOperation::CheckTyposquatting { tool_name } => {
                    format!("Check typosquatting: {tool_name}").into()
                }
                McpSecurityOperation::DetectHiddenInstructions { .. } => {
                    "Detect hidden instructions".into()
                }
                McpSecurityOperation::ScanToolDefinition { name, .. } => {
                    format!("Scan tool: {name}").into()
                }
                McpSecurityOperation::ListServiceCategories => "List MCP categories".into(),
            }
        } else {
            "MCP security".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let engine = self.engine.clone();
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusMcpSecurityToolOutput::Error { error: format!("Failed to receive input: {e}") }
            })?;

            let result = match input.operation {
                McpSecurityOperation::CheckTyposquatting { tool_name } => {
                    let known_tools = &["read_file", "write_file", "execute", "search", "list_dir"];
                    match engine.mcp_check_typosquatting(&tool_name, known_tools) {
                        Some(similar) => format!("⚠️ '{tool_name}' looks like typosquatting of '{similar}'!"),
                        None => format!("✅ '{tool_name}' appears legitimate."),
                    }
                }
                McpSecurityOperation::DetectHiddenInstructions { description } => {
                    let findings = engine.mcp_detect_hidden_instructions(&description);
                    if findings.is_empty() {
                        "✅ No hidden instructions found.".to_string()
                    } else {
                        let mut text = format!("⚠️ {} hidden instructions detected:\n", findings.len());
                        for f in &findings {
                            text.push_str(&format!("- {f}\n"));
                        }
                        text
                    }
                }
                McpSecurityOperation::ScanToolDefinition { name, description } => {
                    let tool_def = serde_json::json!({
                        "name": name,
                        "description": description
                    });
                    let findings = engine.mcp_scan_tool_definition(&tool_def);
                    if findings.is_empty() {
                        format!("✅ Tool '{name}' passed all security checks.")
                    } else {
                        let mut text = format!("⚠️ {} issues found in '{name}':\n", findings.len());
                        for f in &findings {
                            text.push_str(&format!("- [{}] {}\n", f.severity, f.description));
                        }
                        text
                    }
                }
                McpSecurityOperation::ListServiceCategories => {
                    let categories = engine.mcp_service_categories();
                    let mut text = format!("{} MCP service categories:\n", categories.len());
                    for (cat, services) in &categories {
                        text.push_str(&format!("- {} ({})\n", cat, services.join(", ")));
                    }
                    text
                }
            };

            Ok(CaduceusMcpSecurityToolOutput::Success { result })
        })
    }
}
