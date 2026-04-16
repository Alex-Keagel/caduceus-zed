use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use super::caduceus_project_tool::ProjectConfig;

/// Federated semantic search across all repos in a multi-repo project.
/// Reads `.caduceus/project.json` to discover repos, then searches each one.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusCrossSearchToolInput {
    /// The cross-search operation to perform.
    pub operation: CrossSearchOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CrossSearchOperation {
    /// Semantic search across ALL project repos.
    Search {
        /// Natural-language query.
        query: String,
    },
    /// List all indexed repos with chunk counts.
    ListRepos,
    /// Index a specific repo from project.json.
    IndexRepo {
        /// Short name of the repo (key in project.json).
        name: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CrossSearchHit {
    pub repo: String,
    pub path: String,
    pub snippet: String,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusCrossSearchToolOutput {
    SearchResults { results: Vec<CrossSearchHit> },
    RepoList { repos: Vec<RepoIndexInfo> },
    Indexed { message: String },
    Error { error: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RepoIndexInfo {
    pub name: String,
    pub path: String,
    pub chunk_count: usize,
}

impl From<CaduceusCrossSearchToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusCrossSearchToolOutput) -> Self {
        match output {
            CaduceusCrossSearchToolOutput::SearchResults { results } => {
                if results.is_empty() {
                    "No cross-repo matches found. Ensure repos are indexed via `index_repo`.".into()
                } else {
                    let mut text = format!("Found {} cross-repo matches:\n", results.len());
                    for (i, hit) in results.iter().enumerate() {
                        text.push_str(&format!(
                            "\n{}. [{}] {} (score: {:.3})\n```\n{}\n```\n",
                            i + 1,
                            hit.repo,
                            hit.path,
                            hit.score,
                            hit.snippet
                        ));
                    }
                    text.into()
                }
            }
            CaduceusCrossSearchToolOutput::RepoList { repos } => {
                if repos.is_empty() {
                    "No repos configured in project.json.".into()
                } else {
                    let mut text = format!("{} repos in project:\n", repos.len());
                    for r in &repos {
                        text.push_str(&format!(
                            "- **{}** ({}) — {} chunks indexed\n",
                            r.name, r.path, r.chunk_count
                        ));
                    }
                    text.into()
                }
            }
            CaduceusCrossSearchToolOutput::Indexed { message } => message.into(),
            CaduceusCrossSearchToolOutput::Error { error } => {
                format!("Cross-search error: {error}").into()
            }
        }
    }
}

pub struct CaduceusCrossSearchTool {
    project_root: PathBuf,
}

impl CaduceusCrossSearchTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn load_project_config(&self) -> Result<ProjectConfig, String> {
        super::caduceus_project_tool::load_project_config(&self.project_root)
    }

    fn resolve_repo_path(&self, repo_path: &str) -> Result<PathBuf, String> {
        super::caduceus_project_tool::resolve_repo_path(&self.project_root, repo_path)
    }

    fn count_index_chunks(&self, repo_path: &PathBuf) -> usize {
        let index_dir = repo_path.join(".caduceus").join("index");
        if index_dir.exists() {
            std::fs::read_dir(&index_dir)
                .map(|entries| entries.count())
                .unwrap_or(0)
        } else {
            0
        }
    }
}

impl AgentTool for CaduceusCrossSearchTool {
    type Input = CaduceusCrossSearchToolInput;
    type Output = CaduceusCrossSearchToolOutput;

    const NAME: &'static str = "caduceus_cross_search";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Search
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Ok(input) = input {
            match &input.operation {
                CrossSearchOperation::Search { query } => {
                    format!(
                        "Cross-repo search: \"{}\"",
                        crate::tools::truncate_str(query, 40)
                    )
                    .into()
                }
                CrossSearchOperation::ListRepos => "List project repos".into(),
                CrossSearchOperation::IndexRepo { name } => {
                    format!("Index repo: {name}").into()
                }
            }
        } else {
            "Cross-repo search".into()
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
                CaduceusCrossSearchToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let config = self.load_project_config().map_err(|e| {
                CaduceusCrossSearchToolOutput::Error { error: e }
            })?;

            match input.operation {
                CrossSearchOperation::Search { query } => {
                    let mut all_results = Vec::new();
                    for (name, repo) in &config.repos {
                        let repo_path = match self.resolve_repo_path(&repo.path) {
                            Ok(p) => p,
                            Err(e) => {
                                log::warn!("[caduceus] Skipping repo {name}: {e}");
                                continue;
                            }
                        };
                        let engine =
                            caduceus_bridge::engine::CaduceusEngine::new(&repo_path);
                        match engine.semantic_search(&query, 5).await {
                            Ok(hits) => {
                                for (content, score) in hits {
                                    all_results.push(CrossSearchHit {
                                        repo: name.clone(),
                                        path: content
                                            .lines()
                                            .next()
                                            .unwrap_or("unknown")
                                            .to_string(),
                                        snippet: content,
                                        score,
                                    });
                                }
                            }
                            Err(e) => {
                                log::warn!(
                                    "[caduceus] Cross-search failed for repo {name}: {e}"
                                );
                            }
                        }
                    }
                    all_results
                        .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    all_results.truncate(10);
                    Ok(CaduceusCrossSearchToolOutput::SearchResults {
                        results: all_results,
                    })
                }
                CrossSearchOperation::ListRepos => {
                    let repos: Vec<RepoIndexInfo> = config
                        .repos
                        .iter()
                        .filter_map(|(name, repo)| {
                            match self.resolve_repo_path(&repo.path) {
                                Ok(repo_path) => Some(RepoIndexInfo {
                                    name: name.clone(),
                                    path: repo.path.clone(),
                                    chunk_count: self.count_index_chunks(&repo_path),
                                }),
                                Err(e) => {
                                    log::warn!("[caduceus] Skipping repo {name}: {e}");
                                    None
                                }
                            }
                        })
                        .collect();
                    Ok(CaduceusCrossSearchToolOutput::RepoList { repos })
                }
                CrossSearchOperation::IndexRepo { name } => {
                    let repo = config.repos.get(&name).ok_or_else(|| {
                        CaduceusCrossSearchToolOutput::Error {
                            error: format!("Repo '{name}' not found in project.json"),
                        }
                    })?;
                    let repo_path = self.resolve_repo_path(&repo.path).map_err(|e| {
                        CaduceusCrossSearchToolOutput::Error { error: e }
                    })?;
                    let engine =
                        caduceus_bridge::engine::CaduceusEngine::new(&repo_path);
                    match engine.index_directory(&repo_path).await {
                        Ok(count) => Ok(CaduceusCrossSearchToolOutput::Indexed {
                            message: format!(
                                "Indexed repo '{name}' at {} — {count} chunks",
                                repo_path.display()
                            ),
                        }),
                        Err(e) => Err(CaduceusCrossSearchToolOutput::Error {
                            error: format!("Failed to index repo '{name}': {e}"),
                        }),
                    }
                }
            }
        })
    }
}
