use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use super::caduceus_project_tool::ProjectConfig;

/// Product-level view of a multi-repo project: feature tracking, milestone
/// progress, and next-work recommendations based on Kanban state and project.json.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusProductToolInput {
    /// The product operation to perform.
    pub operation: ProductOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProductOperation {
    /// Overall project status: features, completion, velocity.
    Status,
    /// List features with status and repo mapping.
    Features,
    /// Show milestone progress.
    Milestone {
        /// Name of the milestone to show.
        name: String,
    },
    /// Recommend next work items based on dependencies + priority.
    Next,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusProductToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusProductToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusProductToolOutput) -> Self {
        match output {
            CaduceusProductToolOutput::Text { text } => text.into(),
            CaduceusProductToolOutput::Error { error } => {
                format!("Product error: {error}").into()
            }
        }
    }
}

pub struct CaduceusProductTool {
    project_root: PathBuf,
}

impl CaduceusProductTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn load_project_config(&self) -> Result<ProjectConfig, String> {
        let path = self.project_root.join(".caduceus/project.json");
        if !path.exists() {
            return Err("No project.json found. Use `caduceus_project create` first.".into());
        }
        let data = std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read project.json: {e}"))?;
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse project.json: {e}"))
    }

    fn load_product_section(&self) -> Option<serde_json::Value> {
        let path = self.project_root.join(".caduceus/project.json");
        let data = std::fs::read_to_string(&path).ok()?;
        let val: serde_json::Value = serde_json::from_str(&data).ok()?;
        val.get("product").cloned()
    }

    fn load_kanban_board(&self) -> Option<serde_json::Value> {
        let path = self.project_root.join(".caduceus/kanban.json");
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn generate_status(&self, config: &ProjectConfig) -> String {
        let mut text = format!("# {} — Project Status\n\n", config.project.name);
        text.push_str(&format!("{}\n\n", config.project.description));
        text.push_str(&format!("**Repositories**: {}\n", config.repos.len()));
        text.push_str(&format!(
            "**Relationships**: {}\n\n",
            config.relationships.len()
        ));

        // Kanban summary
        if let Some(board) = self.load_kanban_board() {
            if let Some(cards) = board.get("cards").and_then(|c| c.as_array()) {
                let total = cards.len();
                let done = cards
                    .iter()
                    .filter(|c| c["column_id"].as_str() == Some("done"))
                    .count();
                let in_progress = cards
                    .iter()
                    .filter(|c| c["column_id"].as_str() == Some("in-progress"))
                    .count();
                let review = cards
                    .iter()
                    .filter(|c| c["column_id"].as_str() == Some("review"))
                    .count();
                let backlog = total - done - in_progress - review;

                text.push_str("## Task Progress\n");
                text.push_str(&format!("- **Total**: {total}\n"));
                text.push_str(&format!("- **Done**: {done}\n"));
                text.push_str(&format!("- **In Progress**: {in_progress}\n"));
                text.push_str(&format!("- **Review**: {review}\n"));
                text.push_str(&format!("- **Backlog**: {backlog}\n"));

                if total > 0 {
                    let pct = (done as f64 / total as f64 * 100.0) as u32;
                    text.push_str(&format!("- **Completion**: {pct}%\n"));
                }
                text.push('\n');
            }
        }

        // Product section
        if let Some(product) = self.load_product_section() {
            if let Some(features) = product.get("features").and_then(|f| f.as_array()) {
                text.push_str(&format!("## Features: {}\n", features.len()));
                for feat in features {
                    let name = feat["name"].as_str().unwrap_or("unnamed");
                    let status = feat["status"].as_str().unwrap_or("unknown");
                    text.push_str(&format!("- {name} [{status}]\n"));
                }
                text.push('\n');
            }
        }

        text
    }

    fn list_features(&self) -> String {
        let mut text = String::from("# Features\n\n");

        if let Some(product) = self.load_product_section() {
            if let Some(features) = product.get("features").and_then(|f| f.as_array()) {
                for feat in features {
                    let name = feat["name"].as_str().unwrap_or("unnamed");
                    let status = feat["status"].as_str().unwrap_or("unknown");
                    let desc = feat["description"].as_str().unwrap_or("");
                    let repos = feat["repos"]
                        .as_array()
                        .map(|r| {
                            r.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        })
                        .unwrap_or_default();

                    text.push_str(&format!("## {name} [{status}]\n"));
                    if !desc.is_empty() {
                        text.push_str(&format!("{desc}\n"));
                    }
                    if !repos.is_empty() {
                        text.push_str(&format!("Repos: {repos}\n"));
                    }
                    text.push('\n');
                }
            } else {
                text.push_str("No features defined in project.json `product.features`.\n");
            }
        } else {
            text.push_str("No `product` section in project.json. Add features like:\n");
            text.push_str("```json\n\"product\": {\n  \"features\": [\n    { \"name\": \"Auth\", \"status\": \"in-progress\", \"repos\": [\"backend\", \"frontend\"] }\n  ]\n}\n```\n");
        }

        text
    }

    fn show_milestone(&self, name: &str) -> Result<String, String> {
        let product = self
            .load_product_section()
            .ok_or("No `product` section in project.json")?;

        let milestones = product
            .get("milestones")
            .and_then(|m| m.as_array())
            .ok_or("No milestones defined in project.json `product.milestones`")?;

        let milestone = milestones
            .iter()
            .find(|m| m["name"].as_str() == Some(name))
            .ok_or(format!("Milestone '{name}' not found"))?;

        let mut text = format!("# Milestone: {name}\n\n");

        if let Some(desc) = milestone["description"].as_str() {
            text.push_str(&format!("{desc}\n\n"));
        }
        if let Some(due) = milestone["due"].as_str() {
            text.push_str(&format!("**Due**: {due}\n"));
        }
        if let Some(features) = milestone["features"].as_array() {
            text.push_str(&format!("\n## Features ({})\n", features.len()));

            let all_features = self.load_product_section()
                .and_then(|p| p.get("features").and_then(|f| f.as_array().cloned()));

            for feat_name in features {
                let fname = feat_name.as_str().unwrap_or("?");
                let status = all_features
                    .as_ref()
                    .and_then(|fs| {
                        fs.iter()
                            .find(|f| f["name"].as_str() == Some(fname))
                            .and_then(|f| f["status"].as_str())
                    })
                    .unwrap_or("unknown");
                text.push_str(&format!("- {fname} [{status}]\n"));
            }
        }

        Ok(text)
    }

    fn recommend_next(&self, config: &ProjectConfig) -> String {
        let mut text = String::from("# Recommended Next Work\n\n");

        // Check Kanban for ready cards
        if let Some(board) = self.load_kanban_board() {
            if let Some(cards) = board.get("cards").and_then(|c| c.as_array()) {
                let backlog_cards: Vec<_> = cards
                    .iter()
                    .filter(|c| c["column_id"].as_str() == Some("backlog"))
                    .collect();

                if !backlog_cards.is_empty() {
                    text.push_str("## Ready from Kanban\n");
                    for card in backlog_cards.iter().take(5) {
                        let title = card["title"].as_str().unwrap_or("untitled");
                        let deps = card["dependencies"]
                            .as_array()
                            .map(|d| d.len())
                            .unwrap_or(0);
                        let priority = if deps == 0 { "🟢 no deps" } else { "🟡 has deps" };
                        text.push_str(&format!("- {title} ({priority})\n"));
                    }
                    text.push('\n');
                }
            }
        }

        // Check for unindexed repos
        let mut unindexed = Vec::new();
        for (name, repo) in &config.repos {
            let repo_path = if PathBuf::from(&repo.path).is_absolute() {
                PathBuf::from(&repo.path)
            } else {
                self.project_root.join(&repo.path)
            };
            if !repo_path.join(".caduceus/index").exists() {
                unindexed.push(name.as_str());
            }
        }
        if !unindexed.is_empty() {
            text.push_str("## Maintenance\n");
            text.push_str(&format!(
                "- Index these repos for cross-search: {}\n",
                unindexed.join(", ")
            ));
        }

        // Check for repos without tests
        for (name, repo) in &config.repos {
            let repo_path = if PathBuf::from(&repo.path).is_absolute() {
                PathBuf::from(&repo.path)
            } else {
                self.project_root.join(&repo.path)
            };
            let has_tests = repo_path.join("tests").exists()
                || repo_path.join("test").exists()
                || repo_path.join("__tests__").exists();
            if !has_tests {
                text.push_str(&format!("- ⚠️ Add tests to **{name}**\n"));
            }
        }

        text
    }
}

impl AgentTool for CaduceusProductTool {
    type Input = CaduceusProductToolInput;
    type Output = CaduceusProductToolOutput;

    const NAME: &'static str = "caduceus_product";

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
                ProductOperation::Status => "Project status".into(),
                ProductOperation::Features => "List features".into(),
                ProductOperation::Milestone { name } => {
                    format!("Milestone: {name}").into()
                }
                ProductOperation::Next => "Recommend next work".into(),
            }
        } else {
            "Product view".into()
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
                CaduceusProductToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let config = self.load_project_config().map_err(|e| {
                CaduceusProductToolOutput::Error { error: e }
            })?;

            match input.operation {
                ProductOperation::Status => {
                    let status = self.generate_status(&config);
                    Ok(CaduceusProductToolOutput::Text { text: status })
                }
                ProductOperation::Features => {
                    let features = self.list_features();
                    Ok(CaduceusProductToolOutput::Text { text: features })
                }
                ProductOperation::Milestone { name } => match self.show_milestone(&name) {
                    Ok(text) => Ok(CaduceusProductToolOutput::Text { text }),
                    Err(e) => Err(CaduceusProductToolOutput::Error { error: e }),
                },
                ProductOperation::Next => {
                    let next = self.recommend_next(&config);
                    Ok(CaduceusProductToolOutput::Text { text: next })
                }
            }
        })
    }
}
