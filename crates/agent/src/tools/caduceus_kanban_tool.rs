use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use caduceus_bridge::engine::CaduceusEngine;
use caduceus_bridge::orchestrator::{KanbanBoard, KanbanCard};

use crate::tools::caduceus_file_lock::acquire_file_lock;

/// Manages a Kanban board for multi-agent task orchestration. Each card represents
/// a task that can be assigned to an agent with its own git worktree branch.
/// Cards have dependency chains — downstream cards auto-start when dependencies complete.
/// Columns: Backlog → In Progress → Review → Done.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusKanbanToolInput {
    /// The Kanban operation to perform.
    pub operation: KanbanOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum KanbanOperation {
    /// Show the full board state
    ShowBoard,
    /// Add a new card to the backlog
    AddCard {
        title: String,
        description: String,
        /// Optional git branch for worktree isolation
        #[serde(default)]
        worktree_branch: Option<String>,
        /// Enable auto-commit on completion
        #[serde(default)]
        auto_commit: bool,
    },
    /// Move a card to a different column
    MoveCard {
        card_id: String,
        /// Target column: "backlog", "in-progress", "review", or "done"
        column_id: String,
    },
    /// Link two cards: `dependency_id` must complete before `card_id` can start
    LinkCards {
        dependency_id: String,
        card_id: String,
    },
    /// Mark a card as complete — auto-starts any unblocked dependents
    CompleteCard { card_id: String },
    /// List cards that are ready to start (all dependencies met)
    ReadyCards,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusKanbanToolOutput {
    Board {
        board: String,
    },
    CardAdded {
        card_id: String,
        message: String,
    },
    Moved {
        message: String,
    },
    Linked {
        message: String,
    },
    Completed {
        message: String,
        auto_started: Vec<String>,
    },
    Ready {
        cards: Vec<String>,
    },
    Error {
        error: String,
    },
}

impl From<CaduceusKanbanToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusKanbanToolOutput) -> Self {
        match output {
            CaduceusKanbanToolOutput::Board { board } => board.into(),
            CaduceusKanbanToolOutput::CardAdded { card_id, message } => {
                format!("{message} (id: {card_id})").into()
            }
            CaduceusKanbanToolOutput::Moved { message } => message.into(),
            CaduceusKanbanToolOutput::Linked { message } => message.into(),
            CaduceusKanbanToolOutput::Completed {
                message,
                auto_started,
            } => {
                if auto_started.is_empty() {
                    message.into()
                } else {
                    format!("{message}\nAuto-started: {}", auto_started.join(", ")).into()
                }
            }
            CaduceusKanbanToolOutput::Ready { cards } => {
                if cards.is_empty() {
                    "No cards ready — all have pending dependencies.".into()
                } else {
                    let mut text = format!("{} cards ready to start:\n", cards.len());
                    for c in &cards {
                        text.push_str(&format!("- {c}\n"));
                    }
                    text.into()
                }
            }
            CaduceusKanbanToolOutput::Error { error } => format!("Kanban error: {error}").into(),
        }
    }
}

pub struct CaduceusKanbanTool {
    project_root: PathBuf,
    engine: Arc<CaduceusEngine>,
}

impl CaduceusKanbanTool {
    pub fn new(project_root: PathBuf, engine: Arc<CaduceusEngine>) -> Self {
        Self {
            project_root,
            engine,
        }
    }

    fn worktree_path(branch: &str) -> Result<String, String> {
        // SEC-19: Validate branch names to prevent path traversal
        if branch.is_empty()
            || branch.contains("..")
            || branch.contains('/')
            || branch.contains('\\')
            || branch.starts_with('-')
        {
            return Err(format!(
                "Invalid branch name '{}' — must not contain ../ or path separators",
                branch
            ));
        }
        if branch.len() > 64 {
            return Err("Branch name too long (max 64 chars)".to_string());
        }
        Ok(format!(".caduceus/worktrees/{branch}"))
    }

    fn load_or_create_board(&self) -> Result<KanbanBoard, String> {
        KanbanBoard::load_or_new(&self.project_root, "Caduceus Board").map_err(|e| format!("{e}"))
    }

    fn save_board(&self, board: &KanbanBoard) -> Result<(), String> {
        board
            .save_to_workspace(&self.project_root)
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }

    fn render_board(board: &KanbanBoard) -> String {
        let mut text = format!("# {} ({})\n\n", board.name, board.id);
        for col in &board.columns {
            let cards_in_col: Vec<&KanbanCard> = board
                .cards
                .iter()
                .filter(|c| c.column_id == col.id)
                .collect();
            text.push_str(&format!("## {} ({} cards)\n", col.name, cards_in_col.len()));
            for card in cards_in_col {
                let status = format!("{:?}", card.status);
                let branch = card.worktree_branch.as_deref().unwrap_or("no branch");
                let deps = if card.dependencies.is_empty() {
                    String::new()
                } else {
                    format!(" ← depends on: {}", card.dependencies.join(", "))
                };
                text.push_str(&format!(
                    "- [{}] **{}** ({}){}\n  {}\n",
                    status, card.title, branch, deps, card.description
                ));
            }
            text.push('\n');
        }
        text
    }
}

impl AgentTool for CaduceusKanbanTool {
    type Input = CaduceusKanbanToolInput;
    type Output = CaduceusKanbanToolOutput;

    const NAME: &'static str = "caduceus_kanban";

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
                KanbanOperation::ShowBoard => "Show Kanban board".into(),
                KanbanOperation::AddCard { title, .. } => format!("Add card: {title}").into(),
                KanbanOperation::MoveCard { card_id, column_id } => {
                    format!("Move {card_id} → {column_id}").into()
                }
                KanbanOperation::LinkCards { .. } => "Link cards".into(),
                KanbanOperation::CompleteCard { card_id } => format!("Complete: {card_id}").into(),
                KanbanOperation::ReadyCards => "List ready cards".into(),
            }
        } else {
            "Kanban".into()
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
                .map_err(|e| CaduceusKanbanToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let lock_path = self.project_root.join(".caduceus/kanban.json");
            let _lock = acquire_file_lock(&lock_path)
                .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;

            let mut board = self
                .load_or_create_board()
                .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;

            let result = match input.operation {
                KanbanOperation::ShowBoard => CaduceusKanbanToolOutput::Board {
                    board: Self::render_board(&board),
                },
                KanbanOperation::AddCard {
                    title,
                    description,
                    worktree_branch,
                    auto_commit,
                } => {
                    let mut card = KanbanCard::new(&title, &description);
                    card.worktree_branch = worktree_branch;
                    card.auto_commit = auto_commit;
                    let card_id = card.id.clone();
                    board
                        .add_card(card)
                        .map_err(|e| CaduceusKanbanToolOutput::Error {
                            error: format!("{e}"),
                        })?;
                    self.save_board(&board)
                        .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                    CaduceusKanbanToolOutput::CardAdded {
                        card_id,
                        message: format!("Added card: {title}"),
                    }
                }
                KanbanOperation::MoveCard { card_id, column_id } => {
                    // Create worktree when moving to in-progress
                    if column_id == "in-progress" {
                        if let Some(card) = board.cards.iter().find(|c| c.id == card_id) {
                            if let Some(branch) = &card.worktree_branch {
                                let wt_path = Self::worktree_path(branch)
                                    .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                                let _ = self.engine.git_create_worktree(branch, &wt_path);
                            }
                        }
                    }
                    board.move_card(&card_id, &column_id).map_err(|e| {
                        CaduceusKanbanToolOutput::Error {
                            error: format!("{e}"),
                        }
                    })?;
                    self.save_board(&board)
                        .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                    CaduceusKanbanToolOutput::Moved {
                        message: format!("Moved {card_id} → {column_id}"),
                    }
                }
                KanbanOperation::LinkCards {
                    dependency_id,
                    card_id,
                } => {
                    board.link_cards(&dependency_id, &card_id).map_err(|e| {
                        CaduceusKanbanToolOutput::Error {
                            error: format!("{e}"),
                        }
                    })?;
                    self.save_board(&board)
                        .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                    CaduceusKanbanToolOutput::Linked {
                        message: format!("{card_id} now depends on {dependency_id}"),
                    }
                }
                KanbanOperation::CompleteCard { card_id } => {
                    // Capture worktree branch before completing (card may move)
                    let completed_branch = board
                        .cards
                        .iter()
                        .find(|c| c.id == card_id)
                        .and_then(|c| c.worktree_branch.clone());

                    let auto_started = board.on_card_complete(&card_id).map_err(|e| {
                        CaduceusKanbanToolOutput::Error {
                            error: format!("{e}"),
                        }
                    })?;

                    // Remove worktree for the completed card
                    if let Some(branch) = &completed_branch {
                        let wt_path = Self::worktree_path(branch)
                            .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                        let _ = self.engine.git_remove_worktree(&wt_path);
                    }

                    // Create worktrees for auto-started dependents
                    for started_id in &auto_started {
                        let id_part = started_id.split(':').next().unwrap_or(started_id);
                        if let Some(card) = board.cards.iter().find(|c| c.id == id_part) {
                            if let Some(branch) = &card.worktree_branch {
                                let wt_path = Self::worktree_path(branch)
                                    .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                                let _ = self.engine.git_create_worktree(branch, &wt_path);
                            }
                        }
                    }

                    self.save_board(&board)
                        .map_err(|e| CaduceusKanbanToolOutput::Error { error: e })?;
                    CaduceusKanbanToolOutput::Completed {
                        message: format!("Completed: {card_id}"),
                        auto_started,
                    }
                }
                KanbanOperation::ReadyCards => {
                    let ready: Vec<String> = board
                        .ready_cards()
                        .iter()
                        .map(|c| format!("{}: {}", c.id, c.title))
                        .collect();
                    CaduceusKanbanToolOutput::Ready { cards: ready }
                }
            };

            Ok(result)
        })
    }
}
