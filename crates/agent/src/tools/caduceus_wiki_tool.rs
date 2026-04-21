use std::path::PathBuf;
use std::sync::Arc;

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Read and write project wiki pages via the Caduceus storage bridge.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusWikiToolInput {
    /// The wiki operation to perform.
    pub operation: WikiOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WikiOperation {
    /// List all wiki pages.
    ListPages,
    /// Read a wiki page by slug/title.
    ReadPage {
        /// The page slug to read.
        title: String,
    },
    /// Write (upsert) a wiki page.
    WritePage {
        /// The page slug.
        title: String,
        /// The page content (markdown).
        content: String,
    },
    /// Delete a wiki page.
    DeletePage {
        /// The page slug to delete.
        title: String,
    },
    /// Search wiki pages by query.
    SearchPages {
        /// The search query.
        query: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusWikiToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusWikiToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusWikiToolOutput) -> Self {
        match output {
            CaduceusWikiToolOutput::Text { text } => text.into(),
            CaduceusWikiToolOutput::Error { error } => format!("Wiki error: {error}").into(),
        }
    }
}

pub struct CaduceusWikiTool {
    project_root: PathBuf,
    bridge: Arc<caduceus_bridge::storage::StorageBridge>,
}

impl CaduceusWikiTool {
    pub fn new(project_root: PathBuf) -> Self {
        let bridge = caduceus_bridge::storage::StorageBridge::open_default().unwrap_or_else(|_| {
            caduceus_bridge::storage::StorageBridge::open_in_memory()
                .expect("in-memory storage should always succeed")
        });
        Self {
            project_root,
            bridge: Arc::new(bridge),
        }
    }
}

impl AgentTool for CaduceusWikiTool {
    type Input = CaduceusWikiToolInput;
    type Output = CaduceusWikiToolOutput;

    const NAME: &'static str = "caduceus_wiki";

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
                WikiOperation::ListPages => "list pages",
                WikiOperation::ReadPage { .. } => "read page",
                WikiOperation::WritePage { .. } => "write page",
                WikiOperation::DeletePage { .. } => "delete page",
                WikiOperation::SearchPages { .. } => "search pages",
            };
            format!("Wiki {op}").into()
        } else {
            "Wiki operation".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let bridge = self.bridge.clone();
        let project_root = self.project_root.clone();
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CaduceusWikiToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                })?;

            let result: Result<String, String> = match input.operation {
                WikiOperation::ListPages => bridge.list_pages(&project_root).map(|pages| {
                    if pages.is_empty() {
                        "No wiki pages found".to_string()
                    } else {
                        pages
                            .iter()
                            .map(|p| format!("- {}", p.slug))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }),
                WikiOperation::ReadPage { title } => bridge.read_page(&project_root, &title),
                WikiOperation::WritePage { title, content } => bridge
                    .write_page(&project_root, &title, &content)
                    .map(|_| format!("Page '{title}' saved")),
                WikiOperation::DeletePage { title } => bridge
                    .delete_page(&project_root, &title)
                    .map(|_| format!("Page '{title}' deleted")),
                WikiOperation::SearchPages { query } => {
                    bridge.search_pages(&project_root, &query).map(|pages| {
                        if pages.is_empty() {
                            format!("No pages matching '{query}'")
                        } else {
                            pages
                                .iter()
                                .map(|p| format!("- {}", p.slug))
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                    })
                }
            };

            match result {
                Ok(text) => Ok(CaduceusWikiToolOutput::Text { text }),
                Err(e) => Err(CaduceusWikiToolOutput::Error { error: e }),
            }
        })
    }
}
