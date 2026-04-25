use agent_client_protocol as acp;
use anyhow::Result;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::dag_diversity::{DiverseModel, assign_diverse_models};
use crate::{AgentTool, ToolCallEventStream, ToolInput};

/// Suggest a vendor-diverse set of language models for parallel sub-agent
/// fan-out.
///
/// Use this BEFORE issuing multiple `spawn_agent` calls so each sub-branch
/// runs on a different vendor's best-available model. Returns up to `n`
/// `{model_id, provider_id, family}` triples spread across distinct families
/// (claude / gpt / gemini / grok / …) using round-robin selection. Within
/// each family, the alphabetically-latest id wins (a coarse "newest" proxy).
///
/// Plug each returned `model_id` into `spawn_agent`'s `model` field. Diverse
/// vendors mean mistakes in one model family are less likely to be repeated
/// by another — surface area for cross-checking grows.
///
/// Authentication state is respected: only providers the user has logged
/// into are surfaced.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SuggestModelsToolInput {
    /// How many vendor-diverse model IDs to return. The actual count may be
    /// smaller if the registry has fewer authenticated families.
    pub n: usize,
    /// If set, the master's own model id is excluded from the suggestion so
    /// the fan-out is disjoint from the master itself.
    #[serde(default)]
    pub exclude: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SuggestModelsToolOutput {
    Success {
        models: Vec<DiverseModel>,
        families_seen: Vec<String>,
        note: String,
    },
    Error {
        error: String,
    },
}

impl From<SuggestModelsToolOutput> for LanguageModelToolResultContent {
    fn from(out: SuggestModelsToolOutput) -> Self {
        match out {
            SuggestModelsToolOutput::Success {
                models,
                families_seen,
                note,
            } => {
                let mut s = format!(
                    "Suggested {} vendor-diverse model(s) across {} family/families: {}\n\n",
                    models.len(),
                    families_seen.len(),
                    families_seen.join(", "),
                );
                for (i, m) in models.iter().enumerate() {
                    s.push_str(&format!(
                        "{}. model_id=\"{}\"  provider={}  family={}\n",
                        i + 1,
                        m.model_id,
                        m.provider_id,
                        m.family
                    ));
                }
                s.push_str(&format!("\n{note}\n"));
                s.into()
            }
            SuggestModelsToolOutput::Error { error } => {
                format!("suggest_models error: {error}").into()
            }
        }
    }
}

pub struct SuggestModelsTool;

impl SuggestModelsTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SuggestModelsTool {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentTool for SuggestModelsTool {
    type Input = SuggestModelsToolInput;
    type Output = SuggestModelsToolOutput;

    const NAME: &'static str = "suggest_models";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Think
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => format!("Suggesting {} diverse model(s)", i.n).into(),
            Err(_) => "Suggesting diverse models".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        // We can't access `cx: &App` inside the async block, so capture the
        // input synchronously via channels and resolve in the spawned task.
        // The registry call needs `&App`, so we route it through cx.spawn's
        // AsyncApp which can `update(|cx| ...)`.
        cx.spawn(async move |cx| {
            let parsed = input
                .recv()
                .await
                .map_err(|e| SuggestModelsToolOutput::Error {
                    error: format!("Failed to receive tool input: {e}"),
                })?;
            if parsed.n == 0 {
                return Err(SuggestModelsToolOutput::Error {
                    error: "n must be >= 1".into(),
                });
            }
            let models: Vec<DiverseModel> =
                cx.update(|cx| assign_diverse_models(parsed.n, parsed.exclude.as_deref(), cx));
            if models.is_empty() {
                return Err(SuggestModelsToolOutput::Error {
                    error: "no authenticated models available — please configure a provider"
                        .into(),
                });
            }
            let mut families_seen: Vec<String> =
                models.iter().map(|m| m.family.clone()).collect();
            families_seen.sort();
            families_seen.dedup();
            let note = if families_seen.len() == 1 {
                "Only one vendor family is authenticated — diversity is degenerate. \
                 Consider configuring additional providers (Anthropic, OpenAI, Google, …) \
                 before fanning out a non-trivial DAG."
                    .to_string()
            } else {
                "Pass each model_id into spawn_agent's `model` field. \
                 For non-trivial DAGs prefer at least 2-3 distinct families."
                    .to_string()
            };
            Ok(SuggestModelsToolOutput::Success {
                models,
                families_seen,
                note,
            })
        })
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let content: LanguageModelToolResultContent = output.into();
        let text = match content {
            LanguageModelToolResultContent::Text(t) => t.to_string(),
            other => format!("{other:?}"),
        };
        event_stream.update_fields_with_meta(
            acp::ToolCallUpdateFields::new().content(vec![text.into()]),
            None,
        );
        Ok(())
    }
}
