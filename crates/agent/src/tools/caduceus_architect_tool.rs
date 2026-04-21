use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use super::caduceus_api_registry_tool::ApiRegistry;
use super::caduceus_project_tool::ProjectConfig;

/// Provides architectural views of a multi-repo project: Mermaid diagrams,
/// health scoring, and cross-repo impact analysis.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusArchitectToolInput {
    /// The architect operation to perform.
    pub operation: ArchitectOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArchitectOperation {
    /// Generate a Mermaid diagram from project.json relationships + APIs.
    Diagram,
    /// Compute project health score per repo.
    Health,
    /// Analyze impact of a change in one repo on others.
    Impact {
        /// The repo where the change occurs.
        repo: String,
        /// Description of the change.
        change: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusArchitectToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusArchitectToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusArchitectToolOutput) -> Self {
        match output {
            CaduceusArchitectToolOutput::Text { text } => text.into(),
            CaduceusArchitectToolOutput::Error { error } => {
                format!("Architect error: {error}").into()
            }
        }
    }
}

pub struct CaduceusArchitectTool {
    project_root: PathBuf,
}

impl CaduceusArchitectTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn load_project_config(&self) -> Result<ProjectConfig, String> {
        super::caduceus_project_tool::load_project_config(&self.project_root)
    }

    fn load_api_registry(&self) -> ApiRegistry {
        let path = self.project_root.join(".caduceus/apis.json");
        if !path.exists() {
            return ApiRegistry { apis: Vec::new() };
        }
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or(ApiRegistry { apis: Vec::new() })
    }

    fn resolve_repo_path(&self, repo_path: &str) -> Result<PathBuf, String> {
        super::caduceus_project_tool::resolve_repo_path(&self.project_root, repo_path)
    }

    fn sanitize_mermaid_id(s: &str) -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect()
    }

    fn sanitize_mermaid_label(s: &str) -> String {
        s.replace('"', "'")
            .replace('[', "(")
            .replace(']', ")")
            .replace(';', ",")
            .replace('\n', " ")
    }

    fn generate_diagram(&self, config: &ProjectConfig) -> String {
        let registry = self.load_api_registry();

        let mut mermaid = String::from("```mermaid\ngraph LR\n");

        for (name, repo) in &config.repos {
            let id = Self::sanitize_mermaid_id(name);
            let label = Self::sanitize_mermaid_label(&format!("{} - {}", name, repo.language));
            mermaid.push_str(&format!("  {id}[\"{label}\"]\n"));
        }

        for rel in &config.relationships {
            let from = Self::sanitize_mermaid_id(&rel.from);
            let to = Self::sanitize_mermaid_id(&rel.to);
            let sanitized_type =
                Self::sanitize_mermaid_label(&rel.relationship_type.replace(' ', "_"));
            mermaid.push_str(&format!("  {} -->|{}| {}\n", from, sanitized_type, to));
        }

        // Add API nodes
        for api in &registry.apis {
            let api_node = Self::sanitize_mermaid_id(&api.name.replace('/', "_").replace('.', "_"));
            let label = Self::sanitize_mermaid_label(&format!(
                "{} ({})",
                api.title.as_deref().unwrap_or(&api.name),
                api.schema_type
            ));
            let repo_id = Self::sanitize_mermaid_id(&api.repo);
            mermaid.push_str(&format!("  {api_node}((\"{label}\"))\n"));
            mermaid.push_str(&format!("  {} --- {api_node}\n", repo_id));
        }

        mermaid.push_str("```\n");
        mermaid
    }

    fn compute_health(&self, config: &ProjectConfig) -> String {
        let registry = self.load_api_registry();
        let mut text = String::from("# Project Health Report\n\n");

        for (name, repo) in &config.repos {
            let repo_path = match self.resolve_repo_path(&repo.path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let mut score = 0u32;
            let mut checks = Vec::new();

            // Check for README
            let has_readme =
                repo_path.join("README.md").exists() || repo_path.join("readme.md").exists();
            if has_readme {
                score += 20;
                checks.push("✅ README");
            } else {
                checks.push("❌ README missing");
            }

            // Check for tests directory
            let has_tests = repo_path.join("tests").exists()
                || repo_path.join("test").exists()
                || repo_path.join("__tests__").exists()
                || repo_path.join("spec").exists();
            if has_tests {
                score += 20;
                checks.push("✅ Tests directory");
            } else {
                checks.push("❌ No tests directory");
            }

            // Check for CI config
            let has_ci = repo_path.join(".github/workflows").exists()
                || repo_path.join(".gitlab-ci.yml").exists()
                || repo_path.join("Jenkinsfile").exists();
            if has_ci {
                score += 20;
                checks.push("✅ CI configuration");
            } else {
                checks.push("❌ No CI config");
            }

            // Check for API docs
            let has_api = registry.apis.iter().any(|a| a.repo == *name);
            if has_api {
                score += 20;
                checks.push("✅ API schema");
            } else {
                checks.push("⚪ No API schema (may not apply)");
            }

            // Check for .caduceus index
            let has_index = repo_path.join(".caduceus/index").exists();
            if has_index {
                score += 20;
                checks.push("✅ Caduceus indexed");
            } else {
                checks.push("❌ Not indexed");
            }

            text.push_str(&format!(
                "## {} ({}/100)\n**{}** — {}\n",
                name, score, repo.role, repo.language
            ));
            for check in &checks {
                text.push_str(&format!("  {check}\n"));
            }
            text.push('\n');
        }

        text
    }

    fn analyze_impact(&self, config: &ProjectConfig, repo: &str, change: &str) -> String {
        let mut text = format!("# Impact Analysis: {} change\n\n", repo);
        text.push_str(&format!("**Change**: {change}\n\n"));

        // Find direct dependents
        let dependents: Vec<&crate::tools::caduceus_project_tool::Relationship> = config
            .relationships
            .iter()
            .filter(|r| r.from == repo || r.to == repo)
            .collect();

        if dependents.is_empty() {
            text.push_str("No known relationships — impact is isolated to this repo.\n");
        } else {
            text.push_str("## Affected Relationships\n");
            for rel in &dependents {
                let direction = if rel.from == repo {
                    format!("{} → {} ({})", rel.from, rel.to, rel.relationship_type)
                } else {
                    format!("{} → {} ({})", rel.from, rel.to, rel.relationship_type)
                };
                text.push_str(&format!("- {direction}\n"));

                if let Some(contract) = &rel.contract {
                    text.push_str(&format!("  ⚠️  Contract: {contract} — may need updating\n"));
                }
            }

            // Find transitively affected repos
            let mut affected: Vec<String> = dependents
                .iter()
                .flat_map(|r| {
                    if r.from == repo {
                        vec![r.to.clone()]
                    } else {
                        vec![r.from.clone()]
                    }
                })
                .collect();
            affected.sort();
            affected.dedup();

            text.push_str(&format!(
                "\n## Summary\n- **Directly affected repos**: {}\n- **Repos**: {}\n",
                affected.len(),
                affected.join(", ")
            ));

            // Check APIs
            let registry = self.load_api_registry();
            let repo_apis: Vec<_> = registry.apis.iter().filter(|a| a.repo == repo).collect();
            if !repo_apis.is_empty() {
                text.push_str("\n## API Impact\n");
                for api in &repo_apis {
                    text.push_str(&format!(
                        "- {} ({}) — {} endpoints may be affected\n",
                        api.name, api.schema_type, api.endpoint_count
                    ));
                }
            }
        }

        text
    }
}

impl AgentTool for CaduceusArchitectTool {
    type Input = CaduceusArchitectToolInput;
    type Output = CaduceusArchitectToolOutput;

    const NAME: &'static str = "caduceus_architect";

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
                ArchitectOperation::Diagram => "Generate architecture diagram".into(),
                ArchitectOperation::Health => "Compute project health".into(),
                ArchitectOperation::Impact { repo, .. } => {
                    format!("Impact analysis: {repo}").into()
                }
            }
        } else {
            "Architect view".into()
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
                .map_err(|e| CaduceusArchitectToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let config = self
                .load_project_config()
                .map_err(|e| CaduceusArchitectToolOutput::Error { error: e })?;

            match input.operation {
                ArchitectOperation::Diagram => {
                    let diagram = self.generate_diagram(&config);
                    Ok(CaduceusArchitectToolOutput::Text { text: diagram })
                }
                ArchitectOperation::Health => {
                    let health = self.compute_health(&config);
                    Ok(CaduceusArchitectToolOutput::Text { text: health })
                }
                ArchitectOperation::Impact { repo, change } => {
                    if !config.repos.contains_key(&repo) {
                        return Err(CaduceusArchitectToolOutput::Error {
                            error: format!("Repo '{repo}' not found in project.json"),
                        });
                    }
                    let impact = self.analyze_impact(&config, &repo, &change);
                    Ok(CaduceusArchitectToolOutput::Text { text: impact })
                }
            }
        })
    }
}
