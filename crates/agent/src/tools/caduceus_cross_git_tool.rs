use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Coordinated git operations across multiple repositories in a project.
/// Shows status, diffs, and can create branches/commits across all repos at once.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusCrossGitToolInput {
    /// The cross-repo git operation to perform.
    pub operation: CrossGitOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CrossGitOperation {
    /// Show git status across all project repos
    StatusAll,
    /// Show current branch for each repo
    BranchAll,
    /// Show recent commits across all repos (last N per repo)
    LogAll {
        #[serde(default = "default_count")]
        count: usize,
    },
    /// Create the same branch in all repos
    CreateBranchAll { branch_name: String },
    /// Show which repos have uncommitted changes
    DirtyRepos,
}

fn default_count() -> usize {
    3
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusCrossGitToolOutput {
    Success { result: String },
    Error { error: String },
}

impl From<CaduceusCrossGitToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusCrossGitToolOutput) -> Self {
        match output {
            CaduceusCrossGitToolOutput::Success { result } => result.into(),
            CaduceusCrossGitToolOutput::Error { error } => {
                format!("Cross-repo git error: {error}").into()
            }
        }
    }
}

pub struct CaduceusCrossGitTool {
    project_root: PathBuf,
}

impl CaduceusCrossGitTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn load_repos(&self) -> Result<Vec<(String, PathBuf)>, String> {
        let config = super::caduceus_project_tool::load_project_config(&self.project_root)?;
        let mut result: Vec<(String, PathBuf)> = Vec::new();
        for (name, repo) in &config.repos {
            let path =
                super::caduceus_project_tool::resolve_repo_path(&self.project_root, &repo.path)?;
            result.push((name.clone(), path));
        }
        Ok(result)
    }
}

impl AgentTool for CaduceusCrossGitTool {
    type Input = CaduceusCrossGitToolInput;
    type Output = CaduceusCrossGitToolOutput;

    const NAME: &'static str = "caduceus_cross_git";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Execute // Contains CreateBranchAll which mutates repos
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                CrossGitOperation::StatusAll => "Git status (all repos)".into(),
                CrossGitOperation::BranchAll => "Git branches (all repos)".into(),
                CrossGitOperation::LogAll { .. } => "Git log (all repos)".into(),
                CrossGitOperation::CreateBranchAll { branch_name } => format!(
                    "Create branch '{}' (all repos)",
                    crate::tools::truncate_str(branch_name, 20)
                )
                .into(),
                CrossGitOperation::DirtyRepos => "Dirty repos".into(),
            }
        } else {
            "Cross-repo git".into()
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
                .map_err(|e| CaduceusCrossGitToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let repos = self
                .load_repos()
                .map_err(|e| CaduceusCrossGitToolOutput::Error { error: e })?;

            if repos.is_empty() {
                return Ok(CaduceusCrossGitToolOutput::Success {
                    result: "No repos configured in project.json.".to_string(),
                });
            }

            let mut output = String::new();

            match input.operation {
                CrossGitOperation::StatusAll => {
                    output.push_str("## Git Status — All Repos\n\n");
                    for (name, path) in &repos {
                        let git_repo = caduceus_git::GitRepo::discover(path);
                        output.push_str(&format!("### {}\n", name));
                        match git_repo
                            .as_ref()
                            .map_err(|e| e.to_string())
                            .and_then(|r| r.status().map_err(|e| e.to_string()))
                        {
                            Ok(entries) if entries.is_empty() => {
                                output.push_str("Clean ✅\n\n");
                            }
                            Ok(entries) => {
                                for e in &entries {
                                    output.push_str(&format!("  {:?} {}\n", e.status, e.path));
                                }
                                output.push('\n');
                            }
                            Err(e) => output.push_str(&format!("  Error: {e}\n\n")),
                        }
                    }
                }
                CrossGitOperation::BranchAll => {
                    output.push_str("## Branches — All Repos\n\n");
                    for (name, path) in &repos {
                        let git_repo = caduceus_git::GitRepo::discover(path);
                        let branch = git_repo
                            .as_ref()
                            .map_err(|e| e.to_string())
                            .and_then(|r| r.current_branch().map_err(|e| e.to_string()))
                            .unwrap_or_else(|e| format!("error: {e}"));
                        output.push_str(&format!("- **{}**: {}\n", name, branch));
                    }
                }
                CrossGitOperation::LogAll { count } => {
                    output.push_str(&format!(
                        "## Recent Commits — All Repos (last {})\n\n",
                        count
                    ));
                    for (name, path) in &repos {
                        let git_repo = caduceus_git::GitRepo::discover(path);
                        output.push_str(&format!("### {}\n", name));
                        match git_repo
                            .as_ref()
                            .map_err(|e| e.to_string())
                            .and_then(|r| r.log(count).map_err(|e| e.to_string()))
                        {
                            Ok(commits) => {
                                for c in &commits {
                                    let sha = &c.sha[..7.min(c.sha.len())];
                                    output.push_str(&format!(
                                        "  {} {} — {}\n",
                                        sha, c.message, c.author
                                    ));
                                }
                            }
                            Err(e) => output.push_str(&format!("  Error: {e}\n")),
                        }
                        output.push('\n');
                    }
                }
                CrossGitOperation::CreateBranchAll { branch_name } => {
                    output.push_str(&format!(
                        "## Create Branch '{}' — All Repos\n\n",
                        branch_name
                    ));
                    for (name, path) in &repos {
                        // Branch creation is rare — use engine for this specific operation
                        let engine = caduceus_bridge::engine::CaduceusEngine::new(path);
                        match engine.git_create_task_branch(&branch_name) {
                            Ok(branch) => {
                                output.push_str(&format!("- **{}**: ✅ Created {}\n", name, branch))
                            }
                            Err(e) => output.push_str(&format!("- **{}**: ❌ {}\n", name, e)),
                        }
                    }
                }
                CrossGitOperation::DirtyRepos => {
                    output.push_str("## Dirty Repos\n\n");
                    let mut dirty_count = 0;
                    for (name, path) in &repos {
                        let git_repo = caduceus_git::GitRepo::discover(path);
                        match git_repo
                            .as_ref()
                            .map_err(|e| e.to_string())
                            .and_then(|r| r.status().map_err(|e| e.to_string()))
                        {
                            Ok(entries) if !entries.is_empty() => {
                                dirty_count += 1;
                                output.push_str(&format!(
                                    "- **{}**: {} changes\n",
                                    name,
                                    entries.len()
                                ));
                            }
                            Ok(_) => {}
                            Err(_) => {
                                output.push_str(&format!("- **{}**: ⚠️ not a git repo\n", name));
                            }
                        }
                    }
                    if dirty_count == 0 {
                        output.push_str("All repos are clean ✅\n");
                    }
                }
            }

            Ok(CaduceusCrossGitToolOutput::Success { result: output })
        })
    }
}
