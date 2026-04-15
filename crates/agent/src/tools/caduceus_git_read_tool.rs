use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Reads git repository status: current branch, working tree changes, recent commits,
/// diffs, and freshness checks. Use this for read-only git intelligence — understanding
/// what changed, what's staged, and the current state of the repository.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusGitReadToolInput {
    /// The git operation to perform.
    pub operation: GitReadOperation,

    /// Number of recent commits to show (only used with `log`). Default: 10.
    #[serde(default = "default_log_count")]
    pub count: usize,
}

fn default_log_count() -> usize {
    10
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GitReadOperation {
    /// Show current branch name
    Branch,
    /// Show working tree status (modified, added, deleted files)
    Status,
    /// Show combined diff of all unstaged changes
    Diff,
    /// Show staged changes as structured diff entries
    DiffStaged,
    /// Show unstaged changes as structured diff entries
    DiffUnstaged,
    /// Show recent commit history
    Log,
    /// Check if the repo is up-to-date with remote
    Freshness,
    /// Check if local branch has diverged from remote
    Diverged,
    /// List git worktrees
    Worktrees,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusGitReadToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusGitReadToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusGitReadToolOutput) -> Self {
        match output {
            CaduceusGitReadToolOutput::Text { text } => text.into(),
            CaduceusGitReadToolOutput::Error { error } => {
                format!("Git error: {error}").into()
            }
        }
    }
}

pub struct CaduceusGitReadTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusGitReadTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusGitReadTool {
    type Input = CaduceusGitReadToolInput;
    type Output = CaduceusGitReadToolOutput;

    const NAME: &'static str = "caduceus_git_read";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            let op = match &input.operation {
                GitReadOperation::Branch => "branch",
                GitReadOperation::Status => "status",
                GitReadOperation::Diff => "diff",
                GitReadOperation::DiffStaged => "diff staged",
                GitReadOperation::DiffUnstaged => "diff unstaged",
                GitReadOperation::Log => "log",
                GitReadOperation::Freshness => "freshness",
                GitReadOperation::Diverged => "diverged check",
                GitReadOperation::Worktrees => "worktrees",
            };
            format!("Git {op}").into()
        } else {
            "Git read".into()
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
                CaduceusGitReadToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let result = match input.operation {
                GitReadOperation::Branch => engine.git_branch(),
                GitReadOperation::Status => engine.git_status().map(|entries| {
                    if entries.is_empty() {
                        "Working tree clean".to_string()
                    } else {
                        entries
                            .iter()
                            .map(|e| format!("{} {}", e.status, e.path))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }),
                GitReadOperation::Diff => engine.git_diff(),
                GitReadOperation::DiffStaged => engine.git_diff_staged().map(|diffs| {
                    diffs
                        .iter()
                        .map(|d| format!("{}: +{} -{}", d.path, d.additions, d.deletions))
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
                GitReadOperation::DiffUnstaged => engine.git_diff_unstaged().map(|diffs| {
                    diffs
                        .iter()
                        .map(|d| format!("{}: +{} -{}", d.path, d.additions, d.deletions))
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
                GitReadOperation::Log => engine.git_log(input.count).map(|commits| {
                    commits
                        .iter()
                        .map(|c| format!("{} {} — {}", &c.sha[..7.min(c.sha.len())], c.message, c.author))
                        .collect::<Vec<_>>()
                        .join("\n")
                }),
                GitReadOperation::Freshness => engine.git_check_freshness().map(|f| {
                    format!(
                        "Commits behind: {}, Diverged: {}, Stale: {}",
                        f.commits_behind, f.is_diverged, f.is_stale
                    )
                }),
                GitReadOperation::Diverged => engine
                    .git_check_diverged()
                    .map(|d| if d { "Diverged from remote".to_string() } else { "In sync with remote".to_string() }),
                GitReadOperation::Worktrees => engine.git_list_worktrees().map(|wts| {
                    if wts.is_empty() {
                        "No worktrees".to_string()
                    } else {
                        wts.iter()
                            .map(|w| format!("{}: {}", w.branch, w.path))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }),
            };

            match result {
                Ok(text) => Ok(CaduceusGitReadToolOutput::Text { text }),
                Err(e) => Err(CaduceusGitReadToolOutput::Error { error: e }),
            }
        })
    }
}
