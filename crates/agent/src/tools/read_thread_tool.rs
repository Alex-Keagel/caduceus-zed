use agent_client_protocol as acp;
use anyhow::Result;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{AgentTool, ToolCallEventStream, ToolInput, db::ThreadsDatabase};

/// Load a persisted thread by its session_id and return the full transcript
/// as Markdown. Use this to consult or compare against another agent's prior
/// conversation. Pair with `compact_thread` when you only need a summary.
///
/// The session_id is the value shown next to a thread in the History panel
/// (and returned by `spawn_agent` on success). Only persisted threads are
/// readable — a brand-new in-memory thread that has never produced an
/// assistant turn will not yet be in the database.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct ReadThreadToolInput {
    /// The session_id of the thread to load.
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReadThreadToolOutput {
    Success {
        session_id: String,
        title: String,
        message_count: usize,
        transcript: String,
    },
    Error {
        error: String,
    },
}

impl From<ReadThreadToolOutput> for LanguageModelToolResultContent {
    fn from(out: ReadThreadToolOutput) -> Self {
        match out {
            ReadThreadToolOutput::Success {
                session_id,
                title,
                message_count,
                transcript,
            } => format!(
                "# {title} ({message_count} messages, session_id={session_id})\n\n{transcript}"
            )
            .into(),
            ReadThreadToolOutput::Error { error } => format!("read_thread error: {error}").into(),
        }
    }
}

pub struct ReadThreadTool;

impl ReadThreadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadThreadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentTool for ReadThreadTool {
    type Input = ReadThreadToolInput;
    type Output = ReadThreadToolOutput;

    const NAME: &'static str = "read_thread";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => format!("Reading thread {}", short_id(&i.session_id)).into(),
            Err(_) => "Reading thread".into(),
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
                .map_err(|e| ReadThreadToolOutput::Error {
                    error: format!("Failed to receive tool input: {e}"),
                })?;
            let session_id = acp::SessionId::new(input.session_id.clone());

            let db = db_future
                .await
                .map_err(|e| ReadThreadToolOutput::Error {
                    error: format!("DB connect failed: {e}"),
                })?;
            let thread = db
                .load_thread(session_id)
                .await
                .map_err(|e| ReadThreadToolOutput::Error {
                    error: format!("load_thread failed: {e}"),
                })?
                .ok_or_else(|| ReadThreadToolOutput::Error {
                    error: format!("no persisted thread with session_id={}", input.session_id),
                })?;

            let transcript = render_thread_markdown(&thread);
            Ok(ReadThreadToolOutput::Success {
                session_id: input.session_id,
                title: thread.title.to_string(),
                message_count: thread.messages.len(),
                transcript,
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
            ReadThreadToolOutput::Success {
                session_id,
                title,
                message_count,
                transcript,
            } => format!(
                "# {title} ({message_count} messages, session_id={session_id})\n\n{transcript}"
            ),
            ReadThreadToolOutput::Error { error } => format!("read_thread error: {error}"),
        };
        event_stream.update_fields_with_meta(
            acp::ToolCallUpdateFields::new().content(vec![text.into()]),
            None,
        );
        Ok(())
    }
}

pub(crate) fn render_thread_markdown(thread: &crate::db::DbThread) -> String {
    let mut out = String::new();
    for (i, msg) in thread.messages.iter().enumerate() {
        let role = match msg {
            crate::Message::User(_) => "User",
            crate::Message::Agent(_) => "Assistant",
            crate::Message::Resume(_) => "Resume",
        };
        out.push_str(&format!("## {} (#{i})\n\n", role));
        out.push_str(&msg.to_markdown());
        if !out.ends_with("\n\n") {
            out.push_str("\n\n");
        }
    }
    if out.is_empty() {
        out.push_str("(no messages)\n");
    }
    out
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
    use super::*;
    use crate::db::DbThread;
    use crate::{AgentMessage, AgentMessageContent, Message, UserMessage, UserMessageContent};
    use acp_thread::UserMessageId;
    use chrono::Utc;
    use gpui::SharedString;

    fn fixture_thread(title: &str, n: usize) -> DbThread {
        let mut messages = Vec::with_capacity(n * 2);
        for i in 0..n {
            messages.push(Message::User(UserMessage {
                id: UserMessageId::new(),
                content: vec![UserMessageContent::Text(format!("user msg #{i}"))],
            }));
            messages.push(Message::Agent(AgentMessage {
                id: crate::AgentMessageId::new(),
                content: vec![AgentMessageContent::Text(format!(
                    "assistant reply to #{i}"
                ))],
                tool_results: Default::default(),
                reasoning_details: None,
            }));
        }
        DbThread {
            title: SharedString::from(title.to_string()),
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
    fn render_includes_role_headings_and_content() {
        let thread = fixture_thread("Demo", 2);
        let md = render_thread_markdown(&thread);
        assert!(md.contains("## User (#0)"), "{md}");
        assert!(md.contains("## Assistant (#1)"), "{md}");
        assert!(md.contains("## User (#2)"), "{md}");
        assert!(md.contains("user msg #0"), "{md}");
        assert!(md.contains("assistant reply to #1"), "{md}");
    }

    #[test]
    fn render_empty_thread_returns_placeholder() {
        let thread = fixture_thread("Empty", 0);
        let md = render_thread_markdown(&thread);
        assert_eq!(md, "(no messages)\n");
    }
}
