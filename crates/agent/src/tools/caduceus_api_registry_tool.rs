use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

use super::caduceus_project_tool::ProjectConfig;

/// Discovers and catalogs API schemas (OpenAPI, Swagger, gRPC, GraphQL) across
/// all repos in a multi-repo project. Results are cached in `.caduceus/apis.json`.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusApiRegistryToolInput {
    /// The API registry operation to perform.
    pub operation: ApiRegistryOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApiRegistryOperation {
    /// Scan all project repos for API schema files.
    Scan,
    /// List all discovered APIs.
    List,
    /// Show details of a specific API.
    Show {
        /// Name/identifier of the API to show.
        api_name: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEntry {
    pub name: String,
    pub repo: String,
    pub schema_type: String,
    pub file_path: String,
    pub title: Option<String>,
    pub version: Option<String>,
    pub endpoint_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRegistry {
    pub apis: Vec<ApiEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusApiRegistryToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusApiRegistryToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusApiRegistryToolOutput) -> Self {
        match output {
            CaduceusApiRegistryToolOutput::Text { text } => text.into(),
            CaduceusApiRegistryToolOutput::Error { error } => {
                format!("API registry error: {error}").into()
            }
        }
    }
}

pub struct CaduceusApiRegistryTool {
    project_root: PathBuf,
}

impl CaduceusApiRegistryTool {
    pub fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }

    fn registry_path(&self) -> PathBuf {
        self.project_root.join(".caduceus/apis.json")
    }

    fn load_project_config(&self) -> Result<ProjectConfig, String> {
        super::caduceus_project_tool::load_project_config(&self.project_root)
    }

    fn load_registry(&self) -> Result<ApiRegistry, String> {
        let path = self.registry_path();
        if !path.exists() {
            return Ok(ApiRegistry { apis: Vec::new() });
        }
        let data =
            std::fs::read_to_string(&path).map_err(|e| format!("Failed to read apis.json: {e}"))?;
        serde_json::from_str(&data).map_err(|e| format!("Failed to parse apis.json: {e}"))
    }

    fn save_registry(&self, registry: &ApiRegistry) -> Result<(), String> {
        let path = self.registry_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create .caduceus dir: {e}"))?;
        }
        let data = serde_json::to_string_pretty(registry)
            .map_err(|e| format!("Failed to serialize registry: {e}"))?;
        std::fs::write(&path, data).map_err(|e| format!("Failed to write apis.json: {e}"))
    }

    fn resolve_repo_path(&self, repo_path: &str) -> Result<PathBuf, String> {
        super::caduceus_project_tool::resolve_repo_path(&self.project_root, repo_path)
    }

    fn scan_directory_for_apis(dir: &Path, repo_name: &str) -> Vec<ApiEntry> {
        let schema_patterns: &[(&str, &str)] = &[
            ("openapi.yaml", "openapi"),
            ("openapi.yml", "openapi"),
            ("openapi.json", "openapi"),
            ("swagger.json", "swagger"),
            ("swagger.yaml", "swagger"),
            ("swagger.yml", "swagger"),
        ];
        let extension_patterns: &[(&str, &str)] = &[
            ("proto", "grpc"),
            ("graphql", "graphql"),
            ("gql", "graphql"),
        ];

        let mut apis = Vec::new();
        let mut stack: Vec<(PathBuf, usize)> = vec![(dir.to_path_buf(), 0)];

        while let Some((current_dir, depth)) = stack.pop() {
            if depth > 5 {
                continue;
            }
            let entries = match std::fs::read_dir(&current_dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with('.')
                    || file_name == "node_modules"
                    || file_name == "target"
                    || file_name == "vendor"
                {
                    continue;
                }

                let path = entry.path();
                let file_type = match entry.file_type() {
                    Ok(ft) => ft,
                    Err(_) => continue,
                };

                if file_type.is_dir() {
                    stack.push((path, depth + 1));
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                let file_path = path.to_string_lossy().to_string();

                // Check exact filename matches
                for (pattern, schema_type) in schema_patterns {
                    if file_name == *pattern {
                        let (title, version, endpoint_count) = if *schema_type == "openapi"
                            || *schema_type == "swagger"
                        {
                            Self::parse_openapi_basic(&path)
                        } else {
                            (None, None, 0)
                        };
                        apis.push(ApiEntry {
                            name: format!("{repo_name}/{file_name}"),
                            repo: repo_name.to_string(),
                            schema_type: schema_type.to_string(),
                            file_path: file_path.clone(),
                            title,
                            version,
                            endpoint_count,
                        });
                        break;
                    }
                }

                // Check extension matches
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    for (pattern_ext, schema_type) in extension_patterns {
                        if ext == *pattern_ext {
                            apis.push(ApiEntry {
                                name: format!("{repo_name}/{file_name}"),
                                repo: repo_name.to_string(),
                                schema_type: schema_type.to_string(),
                                file_path,
                                title: None,
                                version: None,
                                endpoint_count: 0,
                            });
                            break;
                        }
                    }
                }
            }
        }
        apis
    }

    fn parse_openapi_basic(path: &Path) -> (Option<String>, Option<String>, usize) {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return (None, None, 0),
        };

        // Try JSON first, then YAML-like parsing (just grep for patterns)
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            let title = val["info"]["title"].as_str().map(String::from);
            let version = val["info"]["version"].as_str().map(String::from);
            let endpoint_count = val["paths"]
                .as_object()
                .map(|p| p.len())
                .unwrap_or(0);
            return (title, version, endpoint_count);
        }

        // Rough YAML extraction
        let mut title = None;
        let mut version = None;
        let mut endpoint_count = 0;
        let mut in_paths = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("title:") {
                title = Some(trimmed.trim_start_matches("title:").trim().trim_matches('"').to_string());
            } else if trimmed.starts_with("version:") && version.is_none() {
                version = Some(
                    trimmed
                        .trim_start_matches("version:")
                        .trim()
                        .trim_matches('"')
                        .to_string(),
                );
            } else if line == "paths:" || line.starts_with("paths:") {
                in_paths = true;
            } else if in_paths && !line.starts_with(' ') && !line.starts_with('\t') {
                in_paths = false;
            } else if in_paths && line.starts_with("  /") {
                endpoint_count += 1;
            }
        }

        (title, version, endpoint_count)
    }
}

impl AgentTool for CaduceusApiRegistryTool {
    type Input = CaduceusApiRegistryToolInput;
    type Output = CaduceusApiRegistryToolOutput;

    const NAME: &'static str = "caduceus_api_registry";

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
                ApiRegistryOperation::Scan => "Scan for APIs".into(),
                ApiRegistryOperation::List => "List APIs".into(),
                ApiRegistryOperation::Show { api_name } => {
                    format!("Show API: {api_name}").into()
                }
            }
        } else {
            "API Registry".into()
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
                CaduceusApiRegistryToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            match input.operation {
                ApiRegistryOperation::Scan => {
                    let config = self.load_project_config().map_err(|e| {
                        CaduceusApiRegistryToolOutput::Error { error: e }
                    })?;

                    let mut all_apis = Vec::new();
                    for (name, repo) in &config.repos {
                        let repo_path = match self.resolve_repo_path(&repo.path) {
                            Ok(p) => p,
                            Err(e) => {
                                log::warn!("[caduceus] Skipping repo {name}: {e}");
                                continue;
                            }
                        };
                        if repo_path.exists() {
                            let apis = Self::scan_directory_for_apis(&repo_path, name);
                            all_apis.extend(apis);
                        }
                    }

                    let count = all_apis.len();
                    let registry = ApiRegistry { apis: all_apis };
                    self.save_registry(&registry).map_err(|e| {
                        CaduceusApiRegistryToolOutput::Error { error: e }
                    })?;

                    Ok(CaduceusApiRegistryToolOutput::Text {
                        text: format!("Scanned {} repos, found {} API schemas. Saved to .caduceus/apis.json",
                            config.repos.len(), count),
                    })
                }
                ApiRegistryOperation::List => {
                    let registry = self.load_registry().map_err(|e| {
                        CaduceusApiRegistryToolOutput::Error { error: e }
                    })?;

                    if registry.apis.is_empty() {
                        Ok(CaduceusApiRegistryToolOutput::Text {
                            text: "No APIs discovered. Run `scan` first.".into(),
                        })
                    } else {
                        let mut text = format!("{} APIs discovered:\n\n", registry.apis.len());
                        for api in &registry.apis {
                            let title = api.title.as_deref().unwrap_or("untitled");
                            let version = api.version.as_deref().unwrap_or("?");
                            text.push_str(&format!(
                                "- **{}** [{}] ({}, v{}) — {} endpoints\n",
                                api.name, api.repo, api.schema_type, version, api.endpoint_count
                            ));
                            text.push_str(&format!("  {title}\n"));
                        }
                        Ok(CaduceusApiRegistryToolOutput::Text { text })
                    }
                }
                ApiRegistryOperation::Show { api_name } => {
                    let registry = self.load_registry().map_err(|e| {
                        CaduceusApiRegistryToolOutput::Error { error: e }
                    })?;

                    if let Some(api) = registry.apis.iter().find(|a| a.name == api_name) {
                        let mut text = format!("# {}\n\n", api.name);
                        text.push_str(&format!("- **Repo**: {}\n", api.repo));
                        text.push_str(&format!("- **Type**: {}\n", api.schema_type));
                        text.push_str(&format!("- **File**: {}\n", api.file_path));
                        if let Some(title) = &api.title {
                            text.push_str(&format!("- **Title**: {title}\n"));
                        }
                        if let Some(version) = &api.version {
                            text.push_str(&format!("- **Version**: {version}\n"));
                        }
                        text.push_str(&format!("- **Endpoints**: {}\n", api.endpoint_count));

                        // Try to show a preview of the schema file —
                        // canonicalize and contain to project root before reading
                        // so that an attacker who edits .caduceus/apis.json (or
                        // a symlink it points at) cannot make us read /etc/passwd.
                        let raw = PathBuf::from(&api.file_path);
                        let project_canonical = self
                            .project_root
                            .canonicalize()
                            .unwrap_or_else(|_| self.project_root.clone());
                        let candidate = if raw.is_absolute() {
                            raw.clone()
                        } else {
                            self.project_root.join(&raw)
                        };
                        if let Ok(canonical) = candidate.canonicalize() {
                            if canonical.starts_with(&project_canonical)
                                && !crate::tools::is_sensitive_file(&canonical.to_string_lossy())
                            {
                                if let Ok(content) = std::fs::read_to_string(&canonical) {
                                    let preview: String = content.chars().take(1000).collect();
                                    text.push_str(&format!("\n### Preview\n```\n{preview}\n```\n"));
                                }
                            }
                        }

                        Ok(CaduceusApiRegistryToolOutput::Text { text })
                    } else {
                        Err(CaduceusApiRegistryToolOutput::Error {
                            error: format!("API '{api_name}' not found. Run `list` to see available APIs."),
                        })
                    }
                }
            }
        })
    }
}
