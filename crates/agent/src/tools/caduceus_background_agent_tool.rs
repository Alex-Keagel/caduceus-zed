use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Manage background agent configurations: spawn, list, status, and stop.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusBackgroundAgentToolInput {
    /// The background agent operation to perform.
    pub operation: BackgroundAgentOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundAgentOperation {
    /// Create a background agent configuration for future daemon execution.
    Spawn {
        /// Description of the task the background agent should perform.
        task_description: String,
        /// Execution mode: "autopilot", "supervised", "review_only"
        #[serde(default = "default_mode")]
        mode: String,
        /// Whether the agent should auto-commit its changes.
        #[serde(default)]
        auto_commit: bool,
    },
    /// List all background agent configurations.
    List,
    /// Check the status of a background agent.
    Status {
        /// The agent ID to check.
        agent_id: String,
    },
    /// Mark a background agent as stopped.
    Stop {
        /// The agent ID to stop.
        agent_id: String,
    },
}

fn default_mode() -> String {
    "autopilot".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackgroundAgentConfig {
    id: String,
    task: String,
    mode: String,
    status: String,
    auto_commit: bool,
    created_at: String,
    completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusBackgroundAgentToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusBackgroundAgentToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusBackgroundAgentToolOutput) -> Self {
        match output {
            CaduceusBackgroundAgentToolOutput::Text { text } => text.into(),
            CaduceusBackgroundAgentToolOutput::Error { error } => {
                format!("Background agent error: {error}").into()
            }
        }
    }
}

struct AgentConfigStore {
    dir: PathBuf,
}

impl AgentConfigStore {
    fn new(project_root: &PathBuf) -> Self {
        Self {
            dir: project_root.join(".caduceus/agents"),
        }
    }

    fn ensure_dir(&self) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| e.to_string())
    }

    fn save(&self, config: &BackgroundAgentConfig) -> Result<(), String> {
        self.ensure_dir()?;
        let path = self.dir.join(format!("{}.json", config.id));
        let json = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    fn load(&self, id: &str) -> Result<BackgroundAgentConfig, String> {
        let path = self.dir.join(format!("{id}.json"));
        let json = std::fs::read_to_string(&path)
            .map_err(|_| format!("Agent '{id}' not found"))?;
        serde_json::from_str(&json).map_err(|e| format!("Parse error: {e}"))
    }

    fn list_all(&self) -> Vec<BackgroundAgentConfig> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(e) => e,
            Err(_) => return Vec::new(),
        };
        entries
            .flatten()
            .filter(|e| {
                e.path()
                    .extension()
                    .map_or(false, |ext| ext == "json")
            })
            .filter_map(|e| {
                std::fs::read_to_string(e.path())
                    .ok()
                    .and_then(|json| serde_json::from_str(&json).ok())
            })
            .collect()
    }
}

pub struct CaduceusBackgroundAgentTool {
    project_root: PathBuf,
}

impl CaduceusBackgroundAgentTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

impl AgentTool for CaduceusBackgroundAgentTool {
    type Input = CaduceusBackgroundAgentToolInput;
    type Output = CaduceusBackgroundAgentToolOutput;

    const NAME: &'static str = "caduceus_background_agent";

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
                BackgroundAgentOperation::Spawn { .. } => "spawn",
                BackgroundAgentOperation::List => "list",
                BackgroundAgentOperation::Status { .. } => "status",
                BackgroundAgentOperation::Stop { .. } => "stop",
            };
            format!("Background Agent {op}").into()
        } else {
            "Background Agent".into()
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
                CaduceusBackgroundAgentToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let store = AgentConfigStore::new(&self.project_root);

            let text = match input.operation {
                BackgroundAgentOperation::Spawn {
                    task_description,
                    mode,
                    auto_commit,
                } => {
                    let id = uuid::Uuid::new_v4().to_string();
                    let config = BackgroundAgentConfig {
                        id: id.clone(),
                        task: task_description,
                        mode,
                        status: "pending".to_string(),
                        auto_commit,
                        created_at: chrono::Utc::now().to_rfc3339(),
                        completed_at: None,
                    };
                    store.save(&config).map_err(|e| {
                        CaduceusBackgroundAgentToolOutput::Error { error: e }
                    })?;
                    format!("✅ Background agent spawned: {id}\nTask: {}\nMode: {}\nAuto-commit: {}",
                        config.task, config.mode, config.auto_commit)
                }
                BackgroundAgentOperation::List => {
                    let agents = store.list_all();
                    if agents.is_empty() {
                        "No background agents configured.".to_string()
                    } else {
                        let mut out = format!("{} background agents:\n", agents.len());
                        for a in &agents {
                            out.push_str(&format!(
                                "  [{}] {} — {} (mode: {}, auto-commit: {})\n",
                                a.status,
                                &a.id[..8.min(a.id.len())],
                                a.task,
                                a.mode,
                                a.auto_commit,
                            ));
                        }
                        out
                    }
                }
                BackgroundAgentOperation::Status { agent_id } => {
                    let config = store.load(&agent_id).map_err(|e| {
                        CaduceusBackgroundAgentToolOutput::Error { error: e }
                    })?;
                    format!(
                        "Agent: {}\nStatus: {}\nTask: {}\nMode: {}\nCreated: {}\nCompleted: {}",
                        config.id,
                        config.status,
                        config.task,
                        config.mode,
                        config.created_at,
                        config.completed_at.as_deref().unwrap_or("—"),
                    )
                }
                BackgroundAgentOperation::Stop { agent_id } => {
                    let mut config = store.load(&agent_id).map_err(|e| {
                        CaduceusBackgroundAgentToolOutput::Error { error: e }
                    })?;
                    config.status = "stopped".to_string();
                    config.completed_at = Some(chrono::Utc::now().to_rfc3339());
                    store.save(&config).map_err(|e| {
                        CaduceusBackgroundAgentToolOutput::Error { error: e }
                    })?;
                    format!("✅ Agent {} stopped.", agent_id)
                }
            };

            Ok(CaduceusBackgroundAgentToolOutput::Text { text })
        })
    }
}
