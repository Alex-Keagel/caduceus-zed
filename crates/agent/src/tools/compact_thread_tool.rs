use agent_client_protocol as acp;
use anyhow::Result;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    AgentTool, ToolCallEventStream, ToolInput, db::ThreadsDatabase,
    tools::read_thread_tool::render_thread_markdown,
};

const DEFAULT_MAX_CHARS: usize = 4000;

/// Load a persisted thread by session_id and return a compact summary
/// suitable for inclusion in the orchestrator's context. The full transcript
/// is rendered to Markdown, then run through the same heading-aware
/// compactor used by Caduceus instructions (`OrchestratorBridge::
/// compact_instructions`). The first user message and the final assistant
/// reply are preserved verbatim where they fit; the middle is truncated.
///
/// Use this when you only need the gist of another thread (typically what
/// an orchestrator wants from a sub-agent) — pair with `read_thread` if
/// you need the verbatim transcript.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct CompactThreadToolInput {
    /// The session_id of the thread to compact.
    pub session_id: String,
    /// Maximum characters in the compacted summary. Default 4000.
    #[serde(default)]
    pub max_chars: Option<usize>,
    /// Optional focus hint to bias the summary (free-form text appended as a
    /// preamble). The compactor itself does not yet condition on this, but
    /// the field is forwarded so future implementations can.
    #[serde(default)]
    pub focus: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CompactThreadToolOutput {
    Success {
        session_id: String,
        title: String,
        message_count: usize,
        original_chars: usize,
        compacted_chars: usize,
        summary: String,
    },
    Error {
        error: String,
    },
}

impl From<CompactThreadToolOutput> for LanguageModelToolResultContent {
    fn from(out: CompactThreadToolOutput) -> Self {
        match out {
            CompactThreadToolOutput::Success {
                session_id,
                title,
                message_count,
                original_chars,
                compacted_chars,
                summary,
            } => format!(
                "# {title} (compacted: {compacted_chars}/{original_chars} chars, \
                 {message_count} messages, session_id={session_id})\n\n{summary}"
            )
            .into(),
            CompactThreadToolOutput::Error { error } => {
                format!("compact_thread error: {error}").into()
            }
        }
    }
}

pub struct CompactThreadTool;

impl CompactThreadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CompactThreadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentTool for CompactThreadTool {
    type Input = CompactThreadToolInput;
    type Output = CompactThreadToolOutput;

    const NAME: &'static str = "compact_thread";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => format!("Compacting thread {}", short_id(&i.session_id)).into(),
            Err(_) => "Compacting thread".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let db_future = ThreadsDatabase::connect(cx);
        cx.spawn(async move |_cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| CompactThreadToolOutput::Error {
                    error: format!("Failed to receive tool input: {e}"),
                })?;

            let max_chars = input.max_chars.unwrap_or(DEFAULT_MAX_CHARS).max(256);
            let session_id = acp::SessionId::new(input.session_id.clone());

            let db = db_future
                .await
                .map_err(|e| CompactThreadToolOutput::Error {
                    error: format!("DB connect failed: {e}"),
                })?;
            let thread = db
                .load_thread(session_id)
                .await
                .map_err(|e| CompactThreadToolOutput::Error {
                    error: format!("load_thread failed: {e}"),
                })?
                .ok_or_else(|| CompactThreadToolOutput::Error {
                    error: format!("no persisted thread with session_id={}", input.session_id),
                })?;

            let mut transcript = render_thread_markdown(&thread);
            if let Some(focus) = input.focus.as_deref().filter(|s| !s.is_empty()) {
                transcript = format!("# Focus: {focus}\n\n{transcript}");
            }
            let original_chars = transcript.chars().count();
            let summary = caduceus_bridge::orchestrator::OrchestratorBridge::compact_instructions(
                &transcript,
                max_chars,
            );
            let compacted_chars = summary.chars().count();

            Ok(CompactThreadToolOutput::Success {
                session_id: input.session_id,
                title: thread.title.to_string(),
                message_count: thread.messages.len(),
                original_chars,
                compacted_chars,
                summary,
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
        let text: String = match output {
            CompactThreadToolOutput::Success {
                session_id,
                title,
                message_count,
                original_chars,
                compacted_chars,
                summary,
            } => format!(
                "# {title} (compacted: {compacted_chars}/{original_chars} chars, \
                 {message_count} messages, session_id={session_id})\n\n{summary}"
            ),
            CompactThreadToolOutput::Error { error } => {
                format!("compact_thread error: {error}")
            }
        };
        event_stream.update_fields_with_meta(
            acp::ToolCallUpdateFields::new().content(vec![text.into()]),
            None,
        );
        Ok(())
    }
}

fn short_id(s: &str) -> String {
    if s.len() <= 8 {
        s.to_string()
    } else {
        format!("{}…", &s[..8])
    }
}

#[cfg(test)]
mod tests {
    use crate::db::DbThread;
    use crate::tools::read_thread_tool::render_thread_markdown;
    use crate::{AgentMessage, AgentMessageContent, Message, UserMessage, UserMessageContent};
    use acp_thread::UserMessageId;
    use chrono::Utc;
    use gpui::SharedString;

    fn long_thread() -> DbThread {
        let mut messages = Vec::new();
        for i in 0..40 {
            messages.push(Message::User(UserMessage {
                id: UserMessageId::new(),
                content: vec![UserMessageContent::Text(format!(
                    "user message #{i} with enough body text to make the transcript large \
                     enough that compaction has something to do — repeated padding padding \
                     padding padding padding padding padding padding padding padding."
                ))],
            }));
            messages.push(Message::Agent(AgentMessage {
                id: crate::AgentMessageId::new(),
                content: vec![AgentMessageContent::Text(format!(
                    "assistant reply #{i} with similar long body text to balance the load \
                     padding padding padding padding padding padding padding padding."
                ))],
                tool_results: Default::default(),
                reasoning_details: None,
            }));
        }
        DbThread {
            title: SharedString::from("Long".to_string()),
            messages,
            updated_at: Utc::now(),
            detailed_summary: None,
            initial_project_snapshot: None,
            cumulative_token_usage: Default::default(),
            request_token_usage: Default::default(),
            model: None,
            profile: None,
            imported: false,
            subagent_context: None,
            speed: None,
            thinking_enabled: false,
            thinking_effort: None,
            draft_prompt: None,
            pinned: Vec::new(),
            ui_scroll_position: None,
        }
    }

    #[test]
    fn compact_respects_budget_and_shrinks_long_transcripts() {
        let thread = long_thread();
        let transcript = render_thread_markdown(&thread);
        let original = transcript.chars().count();

        let max_chars = 800;
        let summary = caduceus_bridge::orchestrator::OrchestratorBridge::compact_instructions(
            &transcript,
            max_chars,
        );
        let compacted = summary.chars().count();

        assert!(
            original > max_chars,
            "fixture must produce transcript larger than budget; got {original}"
        );
        // Heading-aware compactor preserves boundaries, so it may overshoot
        // the budget slightly. The real contract is "shrinks substantially";
        // allow up to 20% slack on the hard cap.
        let slack = max_chars + max_chars / 5;
        assert!(
            compacted <= slack,
            "compacted ({compacted}) must be <= budget+slack ({slack})"
        );
        assert!(
            compacted < original / 2,
            "compaction must shrink transcript by >50%; original={original} compacted={compacted}"
        );
        assert!(compacted > 0, "compaction should not return empty");
    }
}
