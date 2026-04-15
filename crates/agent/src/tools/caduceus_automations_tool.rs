use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use caduceus_bridge::orchestrator::{
    Automation, AutomationAgentConfig, AutomationTrigger,
    BridgeAgentMode, BridgeModelId,
};

/// Manages trigger-based automations: cron, file-watch, webhook, manual, etc.
/// Each automation defines a trigger and a prompt template that fires an agent session.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusAutomationsToolInput {
    /// The automation operation to perform.
    pub operation: AutomationOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AutomationOperation {
    /// List all registered automations.
    List,
    /// Add a new automation.
    Add {
        name: String,
        /// Trigger type: "manual", "cron", "file_change", "github_pr", "github_push", "webhook"
        trigger_type: String,
        /// Trigger config value (cron expression, glob pattern, repo, branch, webhook path)
        #[serde(default)]
        trigger_config: Option<String>,
        /// Prompt template with optional {{event}} placeholder.
        prompt_template: String,
    },
    /// Remove an automation by name.
    Remove { name: String },
    /// Manually trigger an automation by name.
    Trigger { name: String },
    /// Enable an automation by name.
    Enable { name: String },
    /// Disable an automation by name.
    Disable { name: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusAutomationsToolOutput {
    AutomationList { automations: String },
    Added { message: String },
    Removed { message: String },
    Triggered { message: String },
    Toggled { message: String },
    Error { error: String },
}

impl From<CaduceusAutomationsToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusAutomationsToolOutput) -> Self {
        match output {
            CaduceusAutomationsToolOutput::AutomationList { automations } => automations.into(),
            CaduceusAutomationsToolOutput::Added { message } => message.into(),
            CaduceusAutomationsToolOutput::Removed { message } => message.into(),
            CaduceusAutomationsToolOutput::Triggered { message } => message.into(),
            CaduceusAutomationsToolOutput::Toggled { message } => message.into(),
            CaduceusAutomationsToolOutput::Error { error } => {
                format!("Automations error: {error}").into()
            }
        }
    }
}

/// Lightweight synchronous automation store that persists to
/// `.caduceus/automations.json` — avoids tokio dependency.
struct AutomationStore {
    path: PathBuf,
}

impl AutomationStore {
    fn new(project_root: &PathBuf) -> Self {
        Self {
            path: project_root.join(".caduceus/automations.json"),
        }
    }

    fn load(&self) -> Vec<Automation> {
        if !self.path.exists() {
            return Vec::new();
        }
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default()
    }

    fn save(&self, automations: &[Automation]) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let json = serde_json::to_string_pretty(automations).map_err(|e| e.to_string())?;
        std::fs::write(&self.path, json).map_err(|e| e.to_string())
    }
}

pub struct CaduceusAutomationsTool {
    project_root: PathBuf,
}

impl CaduceusAutomationsTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn parse_trigger(
        trigger_type: &str,
        trigger_config: Option<&str>,
    ) -> Result<AutomationTrigger, String> {
        match trigger_type {
            "manual" => Ok(AutomationTrigger::Manual),
            "cron" => {
                let expr = trigger_config.ok_or("cron trigger requires trigger_config")?;
                Ok(AutomationTrigger::Cron(expr.to_string()))
            }
            "file_change" => {
                let pattern = trigger_config.ok_or("file_change trigger requires trigger_config (glob)")?;
                Ok(AutomationTrigger::FileChange { pattern: pattern.to_string() })
            }
            "github_pr" => {
                let repo = trigger_config.ok_or("github_pr trigger requires trigger_config (repo)")?;
                Ok(AutomationTrigger::GitHubPR { repo: repo.to_string() })
            }
            "github_push" => {
                let branch = trigger_config.ok_or("github_push trigger requires trigger_config (branch)")?;
                Ok(AutomationTrigger::GitHubPush { branch: branch.to_string() })
            }
            "webhook" => {
                let path = trigger_config.ok_or("webhook trigger requires trigger_config (path)")?;
                Ok(AutomationTrigger::Webhook { path: path.to_string() })
            }
            other => Err(format!("Unknown trigger type: {other}")),
        }
    }
}

impl AgentTool for CaduceusAutomationsTool {
    type Input = CaduceusAutomationsToolInput;
    type Output = CaduceusAutomationsToolOutput;

    const NAME: &'static str = "caduceus_automations";

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
                AutomationOperation::List => "List automations".into(),
                AutomationOperation::Add { name, .. } => format!("Add automation: {name}").into(),
                AutomationOperation::Remove { name } => format!("Remove: {name}").into(),
                AutomationOperation::Trigger { name } => format!("Trigger: {name}").into(),
                AutomationOperation::Enable { name } => format!("Enable: {name}").into(),
                AutomationOperation::Disable { name } => format!("Disable: {name}").into(),
            }
        } else {
            "Automations".into()
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
                CaduceusAutomationsToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let store = AutomationStore::new(&self.project_root);

            let result = match input.operation {
                AutomationOperation::List => {
                    let autos = store.load();
                    if autos.is_empty() {
                        CaduceusAutomationsToolOutput::AutomationList {
                            automations: "No automations registered.".to_string(),
                        }
                    } else {
                        let mut text = format!("{} automations:\n", autos.len());
                        for a in &autos {
                            let status = if a.enabled { "✓" } else { "✗" };
                            text.push_str(&format!(
                                "  [{status}] {} — {:?} (runs: {})\n",
                                a.name, a.trigger, a.run_count,
                            ));
                        }
                        CaduceusAutomationsToolOutput::AutomationList { automations: text }
                    }
                }
                AutomationOperation::Add {
                    name,
                    trigger_type,
                    trigger_config,
                    prompt_template,
                } => {
                    let mut autos = store.load();
                    if autos.iter().any(|a| a.name == name) {
                        return Err(CaduceusAutomationsToolOutput::Error {
                            error: format!("Automation '{name}' already exists"),
                        });
                    }
                    let trigger = Self::parse_trigger(
                        &trigger_type,
                        trigger_config.as_deref(),
                    )
                    .map_err(|e| CaduceusAutomationsToolOutput::Error { error: e })?;

                    let automation = Automation {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: name.clone(),
                        trigger,
                        agent_config: AutomationAgentConfig {
                            mode: BridgeAgentMode::Act,
                            model: BridgeModelId::new("default"),
                            prompt_template,
                            tools: vec![],
                            max_turns: 10,
                            auto_commit: false,
                            auto_pr: false,
                        },
                        enabled: true,
                        created_at: chrono::Utc::now(),
                        last_run: None,
                        run_count: 0,
                    };
                    autos.push(automation);
                    store.save(&autos).map_err(|e| {
                        CaduceusAutomationsToolOutput::Error { error: e }
                    })?;
                    CaduceusAutomationsToolOutput::Added {
                        message: format!("Added automation: {name}"),
                    }
                }
                AutomationOperation::Remove { name } => {
                    let mut autos = store.load();
                    let len_before = autos.len();
                    autos.retain(|a| a.name != name);
                    if autos.len() == len_before {
                        return Err(CaduceusAutomationsToolOutput::Error {
                            error: format!("Automation '{name}' not found"),
                        });
                    }
                    store.save(&autos).map_err(|e| {
                        CaduceusAutomationsToolOutput::Error { error: e }
                    })?;
                    CaduceusAutomationsToolOutput::Removed {
                        message: format!("Removed automation: {name}"),
                    }
                }
                AutomationOperation::Trigger { name } => {
                    let mut autos = store.load();
                    let auto = autos.iter_mut().find(|a| a.name == name).ok_or_else(|| {
                        CaduceusAutomationsToolOutput::Error {
                            error: format!("Automation '{name}' not found"),
                        }
                    })?;
                    auto.run_count += 1;
                    auto.last_run = Some(chrono::Utc::now());
                    let prompt = auto
                        .agent_config
                        .prompt_template
                        .replace("{{event}}", "manual_trigger");
                    store.save(&autos).map_err(|e| {
                        CaduceusAutomationsToolOutput::Error { error: e }
                    })?;
                    CaduceusAutomationsToolOutput::Triggered {
                        message: format!("Triggered '{name}'. Prompt: {prompt}"),
                    }
                }
                AutomationOperation::Enable { name } => {
                    let mut autos = store.load();
                    let auto = autos.iter_mut().find(|a| a.name == name).ok_or_else(|| {
                        CaduceusAutomationsToolOutput::Error {
                            error: format!("Automation '{name}' not found"),
                        }
                    })?;
                    auto.enabled = true;
                    store.save(&autos).map_err(|e| {
                        CaduceusAutomationsToolOutput::Error { error: e }
                    })?;
                    CaduceusAutomationsToolOutput::Toggled {
                        message: format!("Enabled automation: {name}"),
                    }
                }
                AutomationOperation::Disable { name } => {
                    let mut autos = store.load();
                    let auto = autos.iter_mut().find(|a| a.name == name).ok_or_else(|| {
                        CaduceusAutomationsToolOutput::Error {
                            error: format!("Automation '{name}' not found"),
                        }
                    })?;
                    auto.enabled = false;
                    store.save(&autos).map_err(|e| {
                        CaduceusAutomationsToolOutput::Error { error: e }
                    })?;
                    CaduceusAutomationsToolOutput::Toggled {
                        message: format!("Disabled automation: {name}"),
                    }
                }
            };

            Ok(result)
        })
    }
}
