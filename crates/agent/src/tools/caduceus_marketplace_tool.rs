use std::sync::{Arc, Mutex};

use agent_client_protocol as acp;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Browse the Caduceus skill marketplace: search, rank, generate, and list evolved skills.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct CaduceusMarketplaceToolInput {
    /// The marketplace operation to perform.
    pub operation: MarketplaceOperation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MarketplaceOperation {
    /// Search marketplace entries by query string.
    Search {
        /// The search query.
        query: String,
    },
    /// List top N entries by download count.
    TopDownloaded {
        /// Maximum number of results to return.
        limit: usize,
    },
    /// Generate a SKILL.md file from a name and description.
    GenerateSkill {
        /// The skill name (kebab-case).
        name: String,
        /// A description of what the skill does.
        description: String,
    },
    /// Suggest a kebab-case skill name from a description.
    SuggestName {
        /// A description to derive a name from.
        description: String,
    },
    /// List all evolved skills.
    ListEvolved,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaduceusMarketplaceToolOutput {
    Text { text: String },
    Error { error: String },
}

impl From<CaduceusMarketplaceToolOutput> for LanguageModelToolResultContent {
    fn from(output: CaduceusMarketplaceToolOutput) -> Self {
        match output {
            CaduceusMarketplaceToolOutput::Text { text } => text.into(),
            CaduceusMarketplaceToolOutput::Error { error } => {
                format!("Marketplace error: {error}").into()
            }
        }
    }
}

pub struct CaduceusMarketplaceTool {
    bridge: Arc<Mutex<caduceus_bridge::marketplace::MarketplaceBridge>>,
}

impl CaduceusMarketplaceTool {
    pub fn new() -> Self {
        Self {
            bridge: Arc::new(Mutex::new(
                caduceus_bridge::marketplace::MarketplaceBridge::new(),
            )),
        }
    }
}

impl AgentTool for CaduceusMarketplaceTool {
    type Input = CaduceusMarketplaceToolInput;
    type Output = CaduceusMarketplaceToolOutput;

    const NAME: &'static str = "caduceus_marketplace";

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
                MarketplaceOperation::Search { .. } => "search",
                MarketplaceOperation::TopDownloaded { .. } => "top downloaded",
                MarketplaceOperation::GenerateSkill { .. } => "generate skill",
                MarketplaceOperation::SuggestName { .. } => "suggest name",
                MarketplaceOperation::ListEvolved => "list evolved",
            };
            format!("Marketplace {op}").into()
        } else {
            "Marketplace query".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let bridge = self.bridge.clone();
        cx.spawn(async move |_cx| {
            let input = input.recv().await.map_err(|e| {
                CaduceusMarketplaceToolOutput::Error {
                    error: format!("Failed to receive input: {e}"),
                }
            })?;

            let guard = bridge.lock().map_err(|e| {
                CaduceusMarketplaceToolOutput::Error {
                    error: format!("Lock poisoned: {e}"),
                }
            })?;

            let text = match input.operation {
                MarketplaceOperation::Search { query } => {
                    let results = guard.search(&query);
                    if results.is_empty() {
                        format!("No entries matching '{query}'")
                    } else {
                        results
                            .iter()
                            .map(|e| {
                                format!(
                                    "- {} (v{}) by {} — {} downloads",
                                    e.name, e.version, e.author, e.downloads
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
                MarketplaceOperation::TopDownloaded { limit } => {
                    let results = guard.top_downloaded(limit);
                    if results.is_empty() {
                        "No entries in registry".to_string()
                    } else {
                        results
                            .iter()
                            .map(|e| format!("- {} ({} downloads)", e.name, e.downloads))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
                MarketplaceOperation::GenerateSkill { name, description } => {
                    caduceus_bridge::marketplace::MarketplaceBridge::generate_skill_md(
                        &name,
                        &description,
                        &[],
                        &[],
                    )
                }
                MarketplaceOperation::SuggestName { description } => {
                    caduceus_bridge::marketplace::MarketplaceBridge::suggest_skill_name(&[
                        description,
                    ])
                }
                MarketplaceOperation::ListEvolved => {
                    let evolved = guard.list_evolved();
                    if evolved.is_empty() {
                        "No evolved skills yet".to_string()
                    } else {
                        evolved
                            .iter()
                            .map(|s| {
                                format!(
                                    "- {} (v{}, quality: {:.2})",
                                    s.name, s.version, s.quality_score
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
            };

            Ok(CaduceusMarketplaceToolOutput::Text { text })
        })
    }
}
