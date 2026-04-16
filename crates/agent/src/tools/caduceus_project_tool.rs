use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Manage `.caduceus/project.json` — defines repos, roles, and relationships
/// for multi-repo projects.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusProjectToolInput {
    /// The project operation to perform.
    pub operation: ProjectOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectOperation {
    /// Load and return the current project config.
    Load,
    /// Create a new project config.
    Create {
        /// Project name.
        name: String,
        /// Project description.
        description: String,
    },
    /// Add a repository to the project.
    AddRepo {
        /// Short name (key) for the repo.
        name: String,
        /// Filesystem path to the repo.
        path: String,
        /// Role: "frontend", "backend", "shared", etc.
        role: String,
        /// Primary language.
        language: String,
        /// Short description.
        description: String,
    },
    /// Remove a repository from the project.
    RemoveRepo {
        /// Short name of the repo to remove.
        name: String,
    },
    /// List all repositories in the project.
    ListRepos,
    /// Add a relationship between two repos.
    AddRelationship {
        /// Source repo name.
        from: String,
        /// Target repo name.
        to: String,
        /// Relationship type (e.g. "api_consumer", "shared_lib").
        #[serde(rename = "type")]
        relationship_type: String,
        /// Optional contract file path.
        #[serde(default)]
        contract: Option<String>,
    },
    /// Show all relationships.
    ShowRelationships,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project: ProjectMeta,
    #[serde(default)]
    pub repos: BTreeMap<String, RepoEntry>,
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoEntry {
    pub path: String,
    pub role: String,
    pub language: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub from: String,
    pub to: String,
    #[serde(rename = "type")]
    pub relationship_type: String,
    #[serde(default)]
    pub contract: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusProjectToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusProjectToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusProjectToolOutput) -> Self {
        match output {
            CaduceusProjectToolOutput::Text { text } => text.into(),
            CaduceusProjectToolOutput::Error { error } => {
                format!("Project error: {error}").into()
            }
        }
    }
}

pub struct CaduceusProjectTool {
    project_root: PathBuf,
}

impl CaduceusProjectTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn config_path(&self) -> PathBuf {
        self.project_root.join(".caduceus/project.json")
    }

    fn load_config(&self) -> Result<ProjectConfig, String> {
        let path = self.config_path();
        if !path.exists() {
            return Err("No project.json found. Use 'create' to initialize.".into());
        }
        let data = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read project.json: {e}"))?;
        serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse project.json: {e}"))
    }

    fn save_config(&self, config: &ProjectConfig) -> Result<(), String> {
        let path = self.config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create .caduceus dir: {e}"))?;
        }
        let data = serde_json::to_string_pretty(config)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        std::fs::write(&path, data)
            .map_err(|e| format!("Failed to write project.json: {e}"))
    }
}

impl AgentTool for CaduceusProjectTool {
    type Input = CaduceusProjectToolInput;
    type Output = CaduceusProjectToolOutput;

    const NAME: &'static str = "caduceus_project";

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
                ProjectOperation::Load => "load config",
                ProjectOperation::Create { .. } => "create project",
                ProjectOperation::AddRepo { .. } => "add repo",
                ProjectOperation::RemoveRepo { .. } => "remove repo",
                ProjectOperation::ListRepos => "list repos",
                ProjectOperation::AddRelationship { .. } => "add relationship",
                ProjectOperation::ShowRelationships => "show relationships",
            };
            format!("Project {op}").into()
        } else {
            "Project operation".into()
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
                CaduceusProjectToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let result: Result<String, String> = match input.operation {
                ProjectOperation::Load => {
                    self.load_config().and_then(|config| {
                        serde_json::to_string_pretty(&config)
                            .map_err(|e| format!("Serialization error: {e}"))
                    })
                }
                ProjectOperation::Create { name, description } => {
                    let config = ProjectConfig {
                        project: ProjectMeta { name: name.clone(), description },
                        repos: BTreeMap::new(),
                        relationships: Vec::new(),
                    };
                    self.save_config(&config)
                        .map(|_| format!("Project '{name}' created at .caduceus/project.json"))
                }
                ProjectOperation::AddRepo {
                    name,
                    path,
                    role,
                    language,
                    description,
                } => {
                    self.load_config().and_then(|mut config| {
                        config.repos.insert(
                            name.clone(),
                            RepoEntry { path, role, language, description },
                        );
                        self.save_config(&config).map(|_| format!("Repo '{name}' added"))
                    })
                }
                ProjectOperation::RemoveRepo { name } => {
                    self.load_config().and_then(|mut config| {
                        if config.repos.remove(&name).is_some() {
                            config.relationships.retain(|r| r.from != name && r.to != name);
                            self.save_config(&config).map(|_| format!("Repo '{name}' removed"))
                        } else {
                            Err(format!("Repo '{name}' not found"))
                        }
                    })
                }
                ProjectOperation::ListRepos => {
                    self.load_config().map(|config| {
                        if config.repos.is_empty() {
                            "No repos configured".into()
                        } else {
                            let mut text = format!(
                                "Project: {} — {} repos\n\n",
                                config.project.name,
                                config.repos.len()
                            );
                            for (name, repo) in &config.repos {
                                text.push_str(&format!(
                                    "- **{name}** ({}) — {} [{}]\n  {}\n",
                                    repo.role, repo.language, repo.path, repo.description
                                ));
                            }
                            text
                        }
                    })
                }
                ProjectOperation::AddRelationship {
                    from,
                    to,
                    relationship_type,
                    contract,
                } => {
                    self.load_config().and_then(|mut config| {
                        if !config.repos.contains_key(&from) {
                            return Err(format!("Repo '{from}' not found"));
                        }
                        if !config.repos.contains_key(&to) {
                            return Err(format!("Repo '{to}' not found"));
                        }
                        config.relationships.push(Relationship {
                            from: from.clone(),
                            to: to.clone(),
                            relationship_type: relationship_type.clone(),
                            contract,
                        });
                        self.save_config(&config)
                            .map(|_| format!("{from} → {to} ({relationship_type})"))
                    })
                }
                ProjectOperation::ShowRelationships => {
                    self.load_config().map(|config| {
                        if config.relationships.is_empty() {
                            "No relationships configured".into()
                        } else {
                            let mut text = String::from("Relationships:\n");
                            for r in &config.relationships {
                                let contract = r
                                    .contract
                                    .as_deref()
                                    .map(|c| format!(" [contract: {c}]"))
                                    .unwrap_or_default();
                                text.push_str(&format!(
                                    "- {} → {} ({}){}\n",
                                    r.from, r.to, r.relationship_type, contract
                                ));
                            }
                            text
                        }
                    })
                }
            };

            match result {
                Ok(text) => Ok(CaduceusProjectToolOutput::Text { text }),
                Err(e) => Err(CaduceusProjectToolOutput::Error { error: e }),
            }
        })
    }
}
