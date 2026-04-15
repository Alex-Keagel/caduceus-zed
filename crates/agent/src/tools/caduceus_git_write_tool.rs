use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Performs git write operations: staging files, creating commits, and managing branches.
/// These are destructive operations that modify the repository state.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusGitWriteToolInput {
    /// The git write operation to perform.
    pub operation: GitWriteOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GitWriteOperation {
    /// Stage specific files
    Stage {
        /// File paths to stage
        paths: Vec<String>,
    },
    /// Create a commit with a message
    Commit {
        /// Commit message
        message: String,
    },
    /// Stage all changes and commit
    CommitAll {
        /// Commit message
        message: String,
    },
    /// Create a new task branch
    CreateTaskBranch {
        /// Task name (used to derive branch name)
        task_name: String,
    },
    /// Create a new worktree
    CreateWorktree {
        /// Branch name
        branch: String,
        /// Directory path for the worktree
        path: String,
    },
    /// Remove a worktree
    RemoveWorktree {
        /// Directory path of the worktree to remove
        path: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusGitWriteToolOutput {
    Success { message: String },
    Error { error: String },
}

impl From<CaduceusGitWriteToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusGitWriteToolOutput) -> Self {
        match output {
            CaduceusGitWriteToolOutput::Success { message } => message.into(),
            CaduceusGitWriteToolOutput::Error { error } => {
                format!("Git write error: {error}").into()
            }
        }
    }
}

pub struct CaduceusGitWriteTool {
    engine: Arc<caduceus_bridge::engine::CaduceusEngine>,
}

impl CaduceusGitWriteTool {
    pub fn new(engine: Arc<caduceus_bridge::engine::CaduceusEngine>) -> Self {
        Self { engine }
    }
}

impl AgentTool for CaduceusGitWriteTool {
    type Input = CaduceusGitWriteToolInput;
    type Output = CaduceusGitWriteToolOutput;

    const NAME: &'static str = "caduceus_git_write";

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
                GitWriteOperation::Stage { paths } => {
                    format!("Stage {} file(s)", paths.len())
                }
                GitWriteOperation::Commit { message } => {
                    format!("Commit: {}", &message[..message.len().min(40)])
                }
                GitWriteOperation::CommitAll { message } => {
                    format!("Commit all: {}", &message[..message.len().min(40)])
                }
                GitWriteOperation::CreateTaskBranch { task_name } => {
                    format!("Create branch for: {task_name}")
                }
                GitWriteOperation::CreateWorktree { branch, .. } => {
                    format!("Create worktree: {branch}")
                }
                GitWriteOperation::RemoveWorktree { path } => {
                    format!("Remove worktree: {path}")
                }
            };
            op.into()
        } else {
            "Git write".into()
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
                CaduceusGitWriteToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let result = match input.operation {
                GitWriteOperation::Stage { paths } => engine
                    .git_stage_paths(&paths)
                    .map(|_| format!("Staged {} file(s)", paths.len())),
                GitWriteOperation::Commit { message } => engine
                    .git_commit(&message)
                    .map(|r| format!("Committed: ({})", &r.sha[..7.min(r.sha.len())])),
                GitWriteOperation::CommitAll { message } => engine
                    .git_commit_all(&message)
                    .map(|sha| format!("Committed all changes: {sha}")),
                GitWriteOperation::CreateTaskBranch { task_name } => engine
                    .git_create_task_branch(&task_name)
                    .map(|branch| format!("Created branch: {branch}")),
                GitWriteOperation::CreateWorktree { branch, path } => engine
                    .git_create_worktree(&branch, &path)
                    .map(|_| format!("Created worktree at {path} for branch {branch}")),
                GitWriteOperation::RemoveWorktree { path } => engine
                    .git_remove_worktree(&path)
                    .map(|_| format!("Removed worktree at {path}")),
            };

            match result {
                Ok(message) => Ok(CaduceusGitWriteToolOutput::Success { message }),
                Err(e) => Err(CaduceusGitWriteToolOutput::Error { error: e }),
            }
        })
    }
}
