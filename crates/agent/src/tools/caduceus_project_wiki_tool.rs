use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use super::caduceus_project_tool::ProjectConfig;

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

/// Manage a file-based project wiki under `.caduceus/wiki/`. Supports reading,
/// writing, listing, searching, and auto-populating wiki pages from project config.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusProjectWikiToolInput {
    /// The wiki operation to perform.
    pub operation: ProjectWikiOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectWikiOperation {
    /// Read a wiki page by path (e.g. "project", "repos/frontend", "apis/backend-api").
    ReadPage {
        /// Page path relative to the wiki root (without .md extension).
        path: String,
    },
    /// Write or update a wiki page.
    WritePage {
        /// Page path relative to the wiki root (without .md extension).
        path: String,
        /// Markdown content for the page.
        content: String,
    },
    /// List all wiki pages hierarchically.
    ListPages,
    /// Search across all wiki pages for a query string.
    Search {
        /// The text to search for (case-insensitive substring match).
        query: String,
    },
    /// Auto-populate wiki pages from project.json and repo contents.
    AutoPopulate,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusProjectWikiToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusProjectWikiToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusProjectWikiToolOutput) -> Self {
        match output {
            CaduceusProjectWikiToolOutput::Text { text } => text.into(),
            CaduceusProjectWikiToolOutput::Error { error } => {
                format!("Project wiki error: {error}").into()
            }
        }
    }
}

pub struct CaduceusProjectWikiTool {
    project_root: PathBuf,
}

impl CaduceusProjectWikiTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn wiki_root(&self) -> PathBuf {
        self.project_root.join(".caduceus/wiki")
    }

    fn page_path(&self, page: &str) -> Result<PathBuf, String> {
        let rel = Path::new(page);
        if rel.is_absolute() {
            return Err("Wiki page path must be relative".to_string());
        }
        let mut safe = self.wiki_root();
        for component in rel.components() {
            match component {
                std::path::Component::Normal(seg) => safe.push(seg),
                _ => return Err(format!("Invalid wiki page path: '{page}'")),
            }
        }
        // Append .md without replacing existing extensions (e.g., "setup.guide" → "setup.guide.md")
        let mut name = safe.file_name().unwrap_or_default().to_os_string();
        name.push(".md");
        safe.set_file_name(name);
        Ok(safe)
    }

    fn read_page(&self, page: &str) -> Result<String, String> {
        let path = self.page_path(page)?;
        if !path.exists() {
            return Err(format!("Page '{page}' not found"));
        }
        std::fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read page '{page}': {e}"))
    }

    fn write_page(&self, page: &str, content: &str) -> Result<(), String> {
        let path = self.page_path(page)?;
        let _lock = super::caduceus_file_lock::acquire_file_lock(&path)
            .map_err(|e| format!("Failed to lock wiki page '{page}': {e}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create wiki directory: {e}"))?;
        }
        std::fs::write(&path, content)
            .map_err(|e| format!("Failed to write page '{page}': {e}"))
    }

    fn list_pages_recursive(&self, dir: &Path, prefix: &str) -> Vec<String> {
        let mut pages = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return pages,
        };
        let mut entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if path.is_dir() {
                let sub_prefix = if prefix.is_empty() {
                    name.clone()
                } else {
                    format!("{prefix}/{name}")
                };
                pages.extend(self.list_pages_recursive(&path, &sub_prefix));
            } else if name.ends_with(".md") {
                let page_name = name.trim_end_matches(".md");
                let full = if prefix.is_empty() {
                    page_name.to_string()
                } else {
                    format!("{prefix}/{page_name}")
                };
                pages.push(full);
            }
        }
        pages
    }

    fn search_pages(&self, query: &str) -> Result<Vec<(String, Vec<String>)>, String> {
        let wiki_root = self.wiki_root();
        if !wiki_root.exists() {
            return Ok(Vec::new());
        }
        let pages = self.list_pages_recursive(&wiki_root, "");
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        for page in pages {
            let content = self.read_page(&page).unwrap_or_default();
            let matching_lines: Vec<String> = content
                .lines()
                .filter(|line| line.to_lowercase().contains(&query_lower))
                .map(|l| l.to_string())
                .collect();
            if !matching_lines.is_empty() {
                results.push((page, matching_lines));
            }
        }
        Ok(results)
    }

    fn load_project_config(&self) -> Result<ProjectConfig, String> {
        super::caduceus_project_tool::load_project_config(&self.project_root)
    }

    fn auto_populate(&self) -> Result<String, String> {
        let config = self.load_project_config()?;
        let mut generated = Vec::new();

        // project.md
        let mut project_md = format!(
            "# {}\n\n{}\n\n## Repos\n\n",
            config.project.name, config.project.description
        );
        for (name, repo) in &config.repos {
            project_md.push_str(&format!(
                "- **{name}** — {} ({}) at `{}`\n",
                repo.description, repo.language, repo.path
            ));
        }
        self.write_page("project", &project_md)?;
        generated.push("project".to_string());

        // repos/{name}.md
        for (name, repo) in &config.repos {
            let mut page = format!(
                "# {name}\n\n- **Role**: {}\n- **Language**: {}\n- **Path**: `{}`\n\n{}\n",
                repo.role, repo.language, repo.path, repo.description
            );

            // Try reading README from the repo
            let repo_path = expand_tilde(&repo.path);
            if let Ok(readme) = std::fs::read_to_string(repo_path.join("README.md")) {
                let truncated: String = readme.lines().take(50).collect::<Vec<_>>().join("\n");
                page.push_str(&format!("\n## README (excerpt)\n\n{truncated}\n"));
            }

            let page_path = format!("repos/{name}");
            self.write_page(&page_path, &page)?;
            generated.push(page_path);
        }

        // apis/index.md — scan for API schemas
        let mut api_page = String::from("# API Index\n\n");
        let mut found_apis = false;
        for (name, repo) in &config.repos {
            let repo_path = expand_tilde(&repo.path);
            let api_files = ["openapi.yaml", "openapi.json", "swagger.json", "swagger.yaml"];
            for api_file in &api_files {
                if repo_path.join(api_file).exists() {
                    api_page.push_str(&format!("- **{name}**: `{api_file}`\n"));
                    found_apis = true;
                }
            }
            // Check for .proto files
            if let Ok(entries) = std::fs::read_dir(&repo_path) {
                for entry in entries.flatten() {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if fname.ends_with(".proto") {
                        api_page.push_str(&format!("- **{name}**: `{fname}` (protobuf)\n"));
                        found_apis = true;
                    }
                }
            }
        }
        if !found_apis {
            api_page.push_str("No API schemas found in configured repos.\n");
        }
        self.write_page("apis/index", &api_page)?;
        generated.push("apis/index".to_string());

        // architecture.md — relationships
        let mut arch_page = format!(
            "# Architecture: {}\n\n{}\n\n",
            config.project.name, config.project.description
        );
        if config.relationships.is_empty() {
            arch_page.push_str("No relationships defined.\n");
        } else {
            arch_page.push_str("## Relationships\n\n");
            for r in &config.relationships {
                let contract = r
                    .contract
                    .as_deref()
                    .map(|c| format!(" (contract: `{c}`)"))
                    .unwrap_or_default();
                arch_page.push_str(&format!(
                    "- {} → {} [{}]{}\n",
                    r.from, r.to, r.relationship_type, contract
                ));
            }
        }
        self.write_page("architecture", &arch_page)?;
        generated.push("architecture".to_string());

        Ok(format!(
            "Auto-populated {} wiki pages:\n{}",
            generated.len(),
            generated.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n")
        ))
    }
}

impl AgentTool for CaduceusProjectWikiTool {
    type Input = CaduceusProjectWikiToolInput;
    type Output = CaduceusProjectWikiToolOutput;

    const NAME: &'static str = "caduceus_project_wiki";

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
                ProjectWikiOperation::ReadPage { .. } => "read page",
                ProjectWikiOperation::WritePage { .. } => "write page",
                ProjectWikiOperation::ListPages => "list pages",
                ProjectWikiOperation::Search { .. } => "search",
                ProjectWikiOperation::AutoPopulate => "auto-populate",
            };
            format!("Project wiki {op}").into()
        } else {
            "Project wiki operation".into()
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
                CaduceusProjectWikiToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let result: Result<String, String> = match input.operation {
                ProjectWikiOperation::ReadPage { path } => self.read_page(&path),
                ProjectWikiOperation::WritePage { path, content } => {
                    self.write_page(&path, &content)
                        .map(|_| format!("Page '{path}' saved"))
                }
                ProjectWikiOperation::ListPages => {
                    let wiki_root = self.wiki_root();
                    if !wiki_root.exists() {
                        Ok("No wiki pages found. Use 'auto_populate' or 'write_page' to create pages.".into())
                    } else {
                        let pages = self.list_pages_recursive(&wiki_root, "");
                        if pages.is_empty() {
                            Ok("No wiki pages found.".into())
                        } else {
                            let mut text = format!("{} wiki pages:\n", pages.len());
                            for p in &pages {
                                text.push_str(&format!("- {p}\n"));
                            }
                            Ok(text)
                        }
                    }
                }
                ProjectWikiOperation::Search { query } => {
                    match self.search_pages(&query) {
                        Err(e) => Err(e),
                        Ok(results) if results.is_empty() => {
                            Ok(format!("No pages matching '{query}'"))
                        }
                        Ok(results) => {
                            let mut text = format!(
                                "Found matches in {} pages:\n\n",
                                results.len()
                            );
                            for (page, lines) in &results {
                                text.push_str(&format!("**{page}**:\n"));
                                for line in lines.iter().take(5) {
                                    text.push_str(&format!("  > {line}\n"));
                                }
                                if lines.len() > 5 {
                                    text.push_str(&format!(
                                        "  ... and {} more matches\n",
                                        lines.len() - 5
                                    ));
                                }
                                text.push('\n');
                            }
                            Ok(text)
                        }
                    }
                }
                ProjectWikiOperation::AutoPopulate => self.auto_populate(),
            };

            match result {
                Ok(text) => Ok(CaduceusProjectWikiToolOutput::Text { text }),
                Err(e) => Err(CaduceusProjectWikiToolOutput::Error { error: e }),
            }
        })
    }
}
