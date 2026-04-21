use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Lists available code scaffolding templates and generates boilerplate code.
/// Use this to quickly create new components, modules, or project structures.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusScaffoldToolInput {
    /// The scaffold operation to perform.
    pub operation: ScaffoldOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScaffoldOperation {
    /// List available templates for a scaffold type
    ListTemplates {
        /// Type of scaffold: "component", "module", "api", "test", or "project"
        scaffold_type: String,
    },
    /// Generate scaffold code from a description
    Generate {
        /// Natural-language description of what to generate
        description: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusScaffoldToolOutput {
    Templates { templates: Vec<String> },
    Generated { content: String },
    Error { error: String },
}

impl From<CaduceusScaffoldToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusScaffoldToolOutput) -> Self {
        match output {
            CaduceusScaffoldToolOutput::Templates { templates } => {
                if templates.is_empty() {
                    "No templates available for this type.".into()
                } else {
                    let mut text = format!("{} templates available:\n", templates.len());
                    for t in &templates {
                        text.push_str(&format!("- {t}\n"));
                    }
                    text.into()
                }
            }
            CaduceusScaffoldToolOutput::Generated { content } => content.into(),
            CaduceusScaffoldToolOutput::Error { error } => {
                format!("Scaffold error: {error}").into()
            }
        }
    }
}

pub struct CaduceusScaffoldTool;

impl CaduceusScaffoldTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusScaffoldTool {
    type Input = CaduceusScaffoldToolInput;
    type Output = CaduceusScaffoldToolOutput;

    const NAME: &'static str = "caduceus_scaffold";

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
                ScaffoldOperation::ListTemplates { scaffold_type } => {
                    format!("List {scaffold_type} templates").into()
                }
                ScaffoldOperation::Generate { description } => {
                    format!("Scaffold: {}", crate::tools::truncate_str(description, 40)).into()
                }
            }
        } else {
            "Scaffold".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusScaffoldToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            match input.operation {
                ScaffoldOperation::ListTemplates { scaffold_type } => {
                    let st = match scaffold_type.to_lowercase().as_str() {
                        "skill" => caduceus_bridge::tools::ScaffoldType::Skill,
                        "agent" => caduceus_bridge::tools::ScaffoldType::Agent,
                        "instructions" => caduceus_bridge::tools::ScaffoldType::Instructions,
                        "playbook" => caduceus_bridge::tools::ScaffoldType::Playbook,
                        "workflow" => caduceus_bridge::tools::ScaffoldType::Workflow,
                        "prompt" => caduceus_bridge::tools::ScaffoldType::Prompt,
                        "hook" => caduceus_bridge::tools::ScaffoldType::Hook,
                        "mcp" | "mcp_server" => caduceus_bridge::tools::ScaffoldType::McpServer,
                        _ => caduceus_bridge::tools::ScaffoldType::Skill,
                    };
                    let templates =
                        caduceus_bridge::tools::ToolsBridge::list_scaffold_templates(st)
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect();
                    Ok(CaduceusScaffoldToolOutput::Templates { templates })
                }
                ScaffoldOperation::Generate { description } => {
                    let messages = vec![description];
                    let content =
                        caduceus_bridge::orchestrator::OrchestratorBridge::from_conversation(
                            &messages,
                        );
                    Ok(CaduceusScaffoldToolOutput::Generated { content })
                }
            }
        })
    }
}
