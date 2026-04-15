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
    /// Ingest session data for pattern evolution.
    IngestSession {
        /// Session identifier to track.
        session_id: String,
        /// Patterns observed in this session.
        patterns: Vec<String>,
    },
    /// Show top emerging patterns from aggregated sessions.
    TopPatterns {
        /// Maximum number of patterns to return.
        count: usize,
    },
    /// Trigger skill evolution from accumulated patterns.
    Evolve,
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
                MarketplaceOperation::IngestSession { .. } => "ingest session",
                MarketplaceOperation::TopPatterns { .. } => "top patterns",
                MarketplaceOperation::Evolve => "evolve",
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

            let mut guard = bridge.lock().map_err(|e| {
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
                MarketplaceOperation::IngestSession {
                    session_id,
                    patterns,
                } => {
                    let pattern_refs: Vec<&str> = patterns.iter().map(|s| s.as_str()).collect();
                    guard.ingest_session(&session_id, &pattern_refs.iter().map(|s| s.to_string()).collect::<Vec<_>>());
                    format!("✅ Ingested {} patterns from session {session_id}", patterns.len())
                }
                MarketplaceOperation::TopPatterns { count } => {
                    let patterns = guard.top_patterns(count);
                    if patterns.is_empty() {
                        "No patterns aggregated yet.".to_string()
                    } else {
                        patterns
                            .iter()
                            .map(|p| format!("- {} (count: {})", p.pattern, p.occurrences))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                }
                MarketplaceOperation::Evolve => {
                    let aggregated = guard.aggregate();
                    let session_count = aggregated.len();
                    if !guard.should_evolve(session_count) {
                        "Not enough data to trigger evolution yet.".to_string()
                    } else {
                        let summaries: Vec<caduceus_marketplace::SessionSummary> = aggregated
                            .iter()
                            .enumerate()
                            .map(|(i, p)| caduceus_marketplace::SessionSummary {
                                session_id: format!("agg-{i}"),
                                patterns: vec![p.pattern.clone()],
                                tools_used: vec![],
                                success: true,
                            })
                            .collect();
                        let evolved = guard.evolve_from_summaries(&summaries);
                        if evolved.is_empty() {
                            "Evolution produced no new skills.".to_string()
                        } else {
                            let names: Vec<_> = evolved.iter().map(|s| s.name.clone()).collect();
                            for skill in evolved {
                                guard.register_evolved(skill);
                            }
                            format!("🧬 Evolved {} skills: {}", names.len(), names.join(", "))
                        }
                    }
                }
            };

            Ok(CaduceusMarketplaceToolOutput::Text { text })
        })
    }
}
