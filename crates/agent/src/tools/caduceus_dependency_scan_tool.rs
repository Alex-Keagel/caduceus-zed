use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Scans project dependencies for known vulnerabilities and identifies lock files.
/// Use this to audit your project's dependency security posture.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusDependencyScanToolInput {
    /// Directory to scan for lock files and dependencies. Defaults to project root.
    #[serde(default = "default_dir")]
    pub directory: String,

    /// Optional OSV scanner JSON output to parse for vulnerability details.
    #[serde(default)]
    pub osv_json: Option<String>,
}

fn default_dir() -> String {
    ".".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LockFileInfo {
    pub path: String,
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VulnerabilityInfo {
    pub package: String,
    pub severity: String,
    pub advisory: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusDependencyScanToolOutput {
    Success {
        lock_files: Vec<LockFileInfo>,
        vulnerabilities: Vec<VulnerabilityInfo>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusDependencyScanToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusDependencyScanToolOutput) -> Self {
        match output {
            CaduceusDependencyScanToolOutput::Success {
                lock_files,
                vulnerabilities,
            } => {
                let mut text = String::new();
                text.push_str(&format!("## Lock Files Found: {}\n", lock_files.len()));
                for lf in &lock_files {
                    text.push_str(&format!("- {} ({})\n", lf.path, lf.kind));
                }
                if !vulnerabilities.is_empty() {
                    text.push_str(&format!(
                        "\n## Vulnerabilities: {}\n",
                        vulnerabilities.len()
                    ));
                    for v in &vulnerabilities {
                        text.push_str(&format!(
                            "- [{}] {}: {}\n",
                            v.severity, v.package, v.advisory
                        ));
                    }
                } else {
                    text.push_str("\nNo known vulnerabilities found.\n");
                }
                text.into()
            }
            CaduceusDependencyScanToolOutput::Error { error } => {
                format!("Dependency scan error: {error}").into()
            }
        }
    }
}

pub struct CaduceusDependencyScanTool;

impl CaduceusDependencyScanTool {
    pub fn new() -> Self {
        Self
    }
}

impl AgentTool for CaduceusDependencyScanTool {
    type Input = CaduceusDependencyScanToolInput;
    type Output = CaduceusDependencyScanToolOutput;

    const NAME: &'static str = "caduceus_dependency_scan";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            format!("Scan dependencies: {}", input.directory).into()
        } else {
            "Scan dependencies".into()
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
                CaduceusDependencyScanToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let lock_files: Vec<LockFileInfo> =
                caduceus_bridge::tools::ToolsBridge::detect_lock_files(&input.directory)
                    .into_iter()
                    .map(|(path, kind)| LockFileInfo {
                        path,
                        kind: format!("{:?}", kind),
                    })
                    .collect();

            let vulnerabilities = if let Some(osv_json) = &input.osv_json {
                caduceus_bridge::tools::ToolsBridge::parse_osv_output(osv_json)
                    .into_iter()
                    .map(|v| VulnerabilityInfo {
                        package: v.package,
                        severity: format!("{:?}", v.severity),
                        advisory: v.description,
                    })
                    .collect()
            } else {
                Vec::new()
            };

            Ok(CaduceusDependencyScanToolOutput::Success {
                lock_files,
                vulnerabilities,
            })
        })
    }
}
