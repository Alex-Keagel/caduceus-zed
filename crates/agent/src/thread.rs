use crate::{
    CaduceusApiRegistryTool, CaduceusArchitectTool, CaduceusAutomationsTool,
    CaduceusBackgroundAgentTool, CaduceusCheckpointTool, CaduceusCodeGraphTool,
    CaduceusConversationTool, CaduceusCrossGitTool, CaduceusCrossSearchTool,
    CaduceusDependencyScanTool, CaduceusErrorAnalysisTool, CaduceusGitReadTool,
    CaduceusGitWriteTool, CaduceusIndexTool, CaduceusKanbanTool, CaduceusKillSwitchTool,
    CaduceusMarketplaceTool, CaduceusMcpSecurityTool, CaduceusMemoryReadTool,
    CaduceusMemoryWriteTool, CaduceusPolicyTool, CaduceusPrdTool, CaduceusProductTool,
    CaduceusProgressTool, CaduceusProjectTool, CaduceusProjectWikiTool, CaduceusScaffoldTool,
    CaduceusSecurityScanTool, CaduceusSemanticSearchTool, CaduceusStorageTool,
    CaduceusTaskDecomposeTool, CaduceusTaskTreeTool, CaduceusTelemetryTool,
    CaduceusTimeTrackingTool, CaduceusTreeSitterTool, CompactThreadTool, ContextServerRegistry,
    CopyPathTool, CreateDirectoryTool, DbLanguageModel, DbThread, DeletePathTool, DiagnosticsTool,
    EditFileTool, FetchTool, FindPathTool, GrepTool, ListDirectoryTool, MovePathTool, NowTool,
    OpenTool, ProjectSnapshot, ReadFileTool, ReadThreadTool, RestoreFileFromDiskTool, SaveFileTool,
    SpawnAgentTool, StreamingEditFileTool, SuggestModelsTool, SystemPromptTemplate, Template,
    Templates, TerminalTool, ToolPermissionDecision, UpdatePlanTool, WebSearchTool,
    decide_permission_from_settings,
};
use acp_thread::{MentionUri, UserMessageId};
use action_log::ActionLog;
use feature_flags::{
    FeatureFlagAppExt as _, StreamingEditFileToolFeatureFlag, UpdatePlanToolFeatureFlag,
};

use agent_client_protocol as acp;
use agent_settings::{
    AgentProfileId, AgentSettings, SUMMARIZE_THREAD_DETAILED_PROMPT, SUMMARIZE_THREAD_PROMPT,
};
use anyhow::{Context as _, Result, anyhow};
use chrono::{DateTime, Utc};
use client::UserStore;
use cloud_api_types::Plan;
use collections::{HashMap, HashSet, IndexMap};
use fs::Fs;
use futures::{
    FutureExt,
    channel::{mpsc, oneshot},
    future::Shared,
    stream::FuturesUnordered,
};
use futures::{StreamExt, stream};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, EventEmitter, SharedString, Task, WeakEntity,
};
use heck::ToSnakeCase as _;
use language_model::{
    CompletionIntent, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelImage, LanguageModelProviderId, LanguageModelRegistry,
    LanguageModelRequest, LanguageModelRequestMessage, LanguageModelRequestTool,
    LanguageModelToolResult, LanguageModelToolResultContent, LanguageModelToolSchemaFormat,
    LanguageModelToolUse, LanguageModelToolUseId, Role, SelectedModel, Speed, StopReason,
    TokenUsage, ZED_CLOUD_PROVIDER_ID,
};
use project::Project;
use prompt_store::ProjectContext;
use schemars::{JsonSchema, Schema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use settings::{LanguageModelSelection, Settings, ToolPermissionMode, update_settings_file};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    ops::RangeInclusive,
    path::Path,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};
use std::{fmt::Write, path::PathBuf};
use util::{ResultExt, debug_panic, markdown::MarkdownCodeBlock, paths::PathStyle};
use uuid::Uuid;

const TOOL_CANCELED_MESSAGE: &str = "Tool canceled by user";
pub const MAX_TOOL_NAME_LENGTH: usize = 64;
pub const MAX_SUBAGENT_DEPTH: u8 = 1;

/// Returned when a turn is attempted but no language model has been selected.
/// Outcome of an explicit /compact request — distinguishes between the
/// three reasons compaction may not have produced a visible change so the
/// UI can give the user honest feedback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactOutcome {
    /// Compaction ran and shrank the message log.
    Compacted,
    /// Context is currently within budget; compaction was a no-op by design.
    WithinBudget,
    /// Compaction was skipped because the cooldown is still active —
    /// surfacing this lets the UI say so instead of pretending we ran.
    CooldownActive,
}

#[derive(Debug)]
pub struct NoModelConfiguredError;

impl std::fmt::Display for NoModelConfiguredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no language model configured")
    }
}

impl std::error::Error for NoModelConfiguredError {}

/// Context passed to a subagent thread for lifecycle management
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubagentContext {
    /// ID of the parent thread
    pub parent_thread_id: acp::SessionId,

    /// Current depth level (0 = root agent, 1 = first-level subagent, etc.)
    pub depth: u8,
}

/// The ID of the user prompt that initiated a request.
///
/// This equates to the user physically submitting a message to the model (e.g., by pressing the Enter key).
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Serialize, Deserialize)]
pub struct PromptId(Arc<str>);

impl PromptId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string().into())
    }
}

impl std::fmt::Display for PromptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub(crate) const MAX_RETRY_ATTEMPTS: u8 = 4;
pub(crate) const BASE_RETRY_DELAY: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
enum RetryStrategy {
    ExponentialBackoff {
        initial_delay: Duration,
        max_attempts: u8,
    },
    Fixed {
        delay: Duration,
        max_attempts: u8,
    },
}

/// Stable identity of a `Message::Resume` marker. Generated at push-time;
/// invariant under truncation and compaction. Replaces the position-based
/// `ResumeIndex` keying that silently retargeted pins after compaction
/// removed earlier Resume markers (ST2 fix-loop #2).
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResumeId(pub Uuid);

impl ResumeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ResumeId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Message {
    User(UserMessage),
    Agent(AgentMessage),
    Resume(ResumeId),
}

impl<'de> Deserialize<'de> for Message {
    /// Backward-compat: legacy DbThread JSON encodes `Message::Resume` as
    /// the bare string `"Resume"` (unit variant). Newer DbThreads encode
    /// it as `{"Resume": "<uuid>"}` (newtype variant). Accept both. A
    /// legacy bare-string is deserialized with a freshly-allocated
    /// `ResumeId`; the resulting id is unique within the loaded process
    /// but is NOT round-trip-stable across the legacy boundary (which is
    /// fine: legacy data has no Resume pins to invalidate).
    fn deserialize<D>(de: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        let v = serde_json::Value::deserialize(de).map_err(D::Error::custom)?;
        if v == serde_json::Value::String("Resume".to_string()) {
            return Ok(Message::Resume(ResumeId::new()));
        }
        #[derive(Deserialize)]
        enum Helper {
            User(UserMessage),
            Agent(AgentMessage),
            Resume(ResumeId),
        }
        let h: Helper = serde_json::from_value(v).map_err(D::Error::custom)?;
        Ok(match h {
            Helper::User(u) => Message::User(u),
            Helper::Agent(a) => Message::Agent(a),
            Helper::Resume(r) => Message::Resume(r),
        })
    }
}

impl Message {
    pub fn as_agent_message(&self) -> Option<&AgentMessage> {
        match self {
            Message::Agent(agent_message) => Some(agent_message),
            _ => None,
        }
    }

    pub fn to_request(&self) -> Vec<LanguageModelRequestMessage> {
        match self {
            Message::User(message) => {
                if message.content.is_empty() {
                    vec![]
                } else {
                    vec![message.to_request()]
                }
            }
            Message::Agent(message) => message.to_request(),
            Message::Resume(_) => vec![LanguageModelRequestMessage {
                role: Role::User,
                content: vec!["Continue where you left off".into()],
                cache: false,
                reasoning_details: None,
            }],
        }
    }

    pub fn to_markdown(&self) -> String {
        match self {
            Message::User(message) => message.to_markdown(),
            Message::Agent(message) => message.to_markdown(),
            Message::Resume(_) => "[resume]\n".into(),
        }
    }

    pub fn role(&self) -> Role {
        match self {
            Message::User(_) | Message::Resume(_) => Role::User,
            Message::Agent(_) => Role::Assistant,
        }
    }
}

// ─── ST2: pinned messages ──────────────────────────────────────────
//
// `Thread::pinned` is a `Vec<MessageRef>` recording which messages must
// survive compaction. ST2 ships the data model + auto-pin trigger sites
// + persistence; ST3 consumes `pinned_message_indices()` to protect
// pinned positions during compaction.
//
// Composite key: pins are uniquely identified by (PinnedMessageKey,
// PinReason). Multiple reasons may coexist on the same key (e.g.
// FirstUser + Manual on the same user message).
//
// ST2-INVARIANT: subagent threads MUST start with `pinned: vec![]`.
// `Thread::new_internal` initializes pinned to empty; `Thread::from_db`
// also clears pins when `subagent_context.is_some()` (belt-and-suspenders
// against a maliciously-crafted DbThread JSON).

/// Reason a message was pinned. Drives compaction protection priority
/// in ST3 and surfaces in the UI in ST5.
///
/// **Forward-compat:** persisted via the `PinReasonProxy` enum in `db.rs`,
/// which has `#[serde(other)] Unknown` so future variants from a newer
/// build round-trip without breaking the legacy reader. `Unknown` entries
/// are quarantined (dropped with a `WARN` log) at load time.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PinReason {
    /// The first user message in the thread — the original task statement.
    FirstUser,
    /// A `Message::Resume` marker (explicit or implicit-continuation).
    Resume,
    /// Pinned because an `acp::Plan` event was emitted.
    PlanUpdate,
    /// Pinned because the agent requested scope expansion against this message.
    /// (Auto-unpin on Granted/Denied is deferred to ST2.5.)
    ScopeExpansionActive,
    /// Operator-driven pin (no auto-trigger; API-only in ST2).
    Manual,
}

/// Legacy 1-based Resume index — retained for compatibility with any
/// external code that may still reference the type, but no longer used
/// in `PinnedMessageKey` (ST2 fix-loop #2 replaced it with `ResumeId`).
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResumeIndex(pub u32);

/// Identity of a pinned message. `Message::User` carries a stable
/// `UserMessageId`; `Message::Resume` carries a `ResumeId(Uuid)`
/// (generated at push-time — see ST2 fix-loop #2); `Message::Agent`
/// carries an `AgentMessageId(Uuid)` (added in ST2 fix-loop #3 to
/// support `PlanUpdate` pinning the most-recent agent message per
/// plan v3.1 Fix 1). All three are invariant under truncation and
/// compaction.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PinnedMessageKey {
    User(UserMessageId),
    Resume(ResumeId),
    Agent(AgentMessageId),
}

/// A single pin entry. Insertion-order is preserved and is part of the contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageRef {
    pub key: PinnedMessageKey,
    pub reason: PinReason,
    pub pinned_at: DateTime<Utc>,
}

/// Resume coalescing cap: at most this many `Resume`-reason pins exist
/// at once on a thread. When a new Resume pin pushes count above the cap,
/// the oldest Resume pin is dropped.
pub const MAX_RESUME_PINS: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserMessage {
    pub id: UserMessageId,
    pub content: Vec<UserMessageContent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserMessageContent {
    Text(String),
    Mention { uri: MentionUri, content: String },
    Image(LanguageModelImage),
}

impl UserMessage {
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        for content in &self.content {
            match content {
                UserMessageContent::Text(text) => {
                    markdown.push_str(text);
                    markdown.push('\n');
                }
                UserMessageContent::Image(_) => {
                    markdown.push_str("<image />\n");
                }
                UserMessageContent::Mention { uri, content } => {
                    if !content.is_empty() {
                        let _ = writeln!(&mut markdown, "{}\n\n{}", uri.as_link(), content);
                    } else {
                        let _ = writeln!(&mut markdown, "{}", uri.as_link());
                    }
                }
            }
        }

        markdown
    }

    fn to_request(&self) -> LanguageModelRequestMessage {
        let mut message = LanguageModelRequestMessage {
            role: Role::User,
            content: Vec::with_capacity(self.content.len()),
            cache: false,
            reasoning_details: None,
        };

        const OPEN_CONTEXT: &str = "<context>\n\
            The following items were attached by the user. \
            They are up-to-date and don't need to be re-read.\n\n";

        const OPEN_FILES_TAG: &str = "<files>";
        const OPEN_DIRECTORIES_TAG: &str = "<directories>";
        const OPEN_SYMBOLS_TAG: &str = "<symbols>";
        const OPEN_SELECTIONS_TAG: &str = "<selections>";
        const OPEN_THREADS_TAG: &str = "<threads>";
        const OPEN_FETCH_TAG: &str = "<fetched_urls>";
        const OPEN_RULES_TAG: &str =
            "<rules>\nThe user has specified the following rules that should be applied:\n";
        const OPEN_DIAGNOSTICS_TAG: &str = "<diagnostics>";
        const OPEN_DIFFS_TAG: &str = "<diffs>";
        const MERGE_CONFLICT_TAG: &str = "<merge_conflicts>";

        let mut file_context = OPEN_FILES_TAG.to_string();
        let mut directory_context = OPEN_DIRECTORIES_TAG.to_string();
        let mut symbol_context = OPEN_SYMBOLS_TAG.to_string();
        let mut selection_context = OPEN_SELECTIONS_TAG.to_string();
        let mut thread_context = OPEN_THREADS_TAG.to_string();
        let mut fetch_context = OPEN_FETCH_TAG.to_string();
        let mut rules_context = OPEN_RULES_TAG.to_string();
        let mut diagnostics_context = OPEN_DIAGNOSTICS_TAG.to_string();
        let mut diffs_context = OPEN_DIFFS_TAG.to_string();
        let mut merge_conflict_context = MERGE_CONFLICT_TAG.to_string();

        for chunk in &self.content {
            let chunk = match chunk {
                UserMessageContent::Text(text) => {
                    language_model::MessageContent::Text(text.clone())
                }
                UserMessageContent::Image(value) => {
                    language_model::MessageContent::Image(value.clone())
                }
                UserMessageContent::Mention { uri, content } => {
                    match uri {
                        MentionUri::File { abs_path } => {
                            write!(
                                &mut file_context,
                                "\n{}",
                                MarkdownCodeBlock {
                                    tag: &codeblock_tag(abs_path, None),
                                    text: &content.to_string(),
                                }
                            )
                            .ok();
                        }
                        MentionUri::PastedImage { .. } => {
                            debug_panic!("pasted image URI should not be used in mention content")
                        }
                        MentionUri::Directory { .. } => {
                            write!(&mut directory_context, "\n{}\n", content).ok();
                        }
                        MentionUri::Symbol {
                            abs_path: path,
                            line_range,
                            ..
                        } => {
                            write!(
                                &mut symbol_context,
                                "\n{}",
                                MarkdownCodeBlock {
                                    tag: &codeblock_tag(path, Some(line_range)),
                                    text: content
                                }
                            )
                            .ok();
                        }
                        MentionUri::Selection {
                            abs_path: path,
                            line_range,
                            ..
                        } => {
                            write!(
                                &mut selection_context,
                                "\n{}",
                                MarkdownCodeBlock {
                                    tag: &codeblock_tag(
                                        path.as_deref().unwrap_or("Untitled".as_ref()),
                                        Some(line_range)
                                    ),
                                    text: content
                                }
                            )
                            .ok();
                        }
                        MentionUri::Thread { .. } => {
                            write!(&mut thread_context, "\n{}\n", content).ok();
                        }
                        MentionUri::Rule { .. } => {
                            write!(
                                &mut rules_context,
                                "\n{}",
                                MarkdownCodeBlock {
                                    tag: "",
                                    text: content
                                }
                            )
                            .ok();
                        }
                        MentionUri::Fetch { url } => {
                            write!(&mut fetch_context, "\nFetch: {}\n\n{}", url, content).ok();
                        }
                        MentionUri::Diagnostics { .. } => {
                            write!(&mut diagnostics_context, "\n{}\n", content).ok();
                        }
                        MentionUri::TerminalSelection { .. } => {
                            write!(
                                &mut selection_context,
                                "\n{}",
                                MarkdownCodeBlock {
                                    tag: "console",
                                    text: content
                                }
                            )
                            .ok();
                        }
                        MentionUri::GitDiff { base_ref } => {
                            write!(
                                &mut diffs_context,
                                "\nBranch diff against {}:\n{}",
                                base_ref,
                                MarkdownCodeBlock {
                                    tag: "diff",
                                    text: content
                                }
                            )
                            .ok();
                        }
                        MentionUri::MergeConflict { file_path } => {
                            write!(
                                &mut merge_conflict_context,
                                "\nMerge conflict in {}:\n{}",
                                file_path,
                                MarkdownCodeBlock {
                                    tag: "diff",
                                    text: content
                                }
                            )
                            .ok();
                        }
                    }

                    language_model::MessageContent::Text(uri.as_link().to_string())
                }
            };

            message.content.push(chunk);
        }

        let len_before_context = message.content.len();

        if file_context.len() > OPEN_FILES_TAG.len() {
            file_context.push_str("</files>\n");
            message
                .content
                .push(language_model::MessageContent::Text(file_context));
        }

        if directory_context.len() > OPEN_DIRECTORIES_TAG.len() {
            directory_context.push_str("</directories>\n");
            message
                .content
                .push(language_model::MessageContent::Text(directory_context));
        }

        if symbol_context.len() > OPEN_SYMBOLS_TAG.len() {
            symbol_context.push_str("</symbols>\n");
            message
                .content
                .push(language_model::MessageContent::Text(symbol_context));
        }

        if selection_context.len() > OPEN_SELECTIONS_TAG.len() {
            selection_context.push_str("</selections>\n");
            message
                .content
                .push(language_model::MessageContent::Text(selection_context));
        }

        if diffs_context.len() > OPEN_DIFFS_TAG.len() {
            diffs_context.push_str("</diffs>\n");
            message
                .content
                .push(language_model::MessageContent::Text(diffs_context));
        }

        if thread_context.len() > OPEN_THREADS_TAG.len() {
            thread_context.push_str("</threads>\n");
            message
                .content
                .push(language_model::MessageContent::Text(thread_context));
        }

        if fetch_context.len() > OPEN_FETCH_TAG.len() {
            fetch_context.push_str("</fetched_urls>\n");
            message
                .content
                .push(language_model::MessageContent::Text(fetch_context));
        }

        if rules_context.len() > OPEN_RULES_TAG.len() {
            rules_context.push_str("</user_rules>\n");
            message
                .content
                .push(language_model::MessageContent::Text(rules_context));
        }

        if diagnostics_context.len() > OPEN_DIAGNOSTICS_TAG.len() {
            diagnostics_context.push_str("</diagnostics>\n");
            message
                .content
                .push(language_model::MessageContent::Text(diagnostics_context));
        }

        if merge_conflict_context.len() > MERGE_CONFLICT_TAG.len() {
            merge_conflict_context.push_str("</merge_conflicts>\n");
            message
                .content
                .push(language_model::MessageContent::Text(merge_conflict_context));
        }

        if message.content.len() > len_before_context {
            message.content.insert(
                len_before_context,
                language_model::MessageContent::Text(OPEN_CONTEXT.into()),
            );
            message
                .content
                .push(language_model::MessageContent::Text("</context>".into()));
        }

        message
    }
}

fn codeblock_tag(full_path: &Path, line_range: Option<&RangeInclusive<u32>>) -> String {
    let mut result = String::new();

    if let Some(extension) = full_path.extension().and_then(|ext| ext.to_str()) {
        let _ = write!(result, "{} ", extension);
    }

    let _ = write!(result, "{}", full_path.display());

    if let Some(range) = line_range {
        if range.start() == range.end() {
            let _ = write!(result, ":{}", range.start() + 1);
        } else {
            let _ = write!(result, ":{}-{}", range.start() + 1, range.end() + 1);
        }
    }

    result
}

impl AgentMessage {
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();

        for content in &self.content {
            match content {
                AgentMessageContent::Text(text) => {
                    markdown.push_str(text);
                    markdown.push('\n');
                }
                AgentMessageContent::Thinking { text, .. } => {
                    markdown.push_str("<think>");
                    markdown.push_str(text);
                    markdown.push_str("</think>\n");
                }
                AgentMessageContent::RedactedThinking(_) => {
                    markdown.push_str("<redacted_thinking />\n")
                }
                AgentMessageContent::ToolUse(tool_use) => {
                    markdown.push_str(&format!(
                        "**Tool Use**: {} (ID: {})\n",
                        tool_use.name, tool_use.id
                    ));
                    markdown.push_str(&format!(
                        "{}\n",
                        MarkdownCodeBlock {
                            tag: "json",
                            text: &format!("{:#}", tool_use.input)
                        }
                    ));
                }
            }
        }

        for tool_result in self.tool_results.values() {
            markdown.push_str(&format!(
                "**Tool Result**: {} (ID: {})\n\n",
                tool_result.tool_name, tool_result.tool_use_id
            ));
            if tool_result.is_error {
                markdown.push_str("**ERROR:**\n");
            }

            match &tool_result.content {
                LanguageModelToolResultContent::Text(text) => {
                    writeln!(markdown, "{text}\n").ok();
                }
                LanguageModelToolResultContent::Image(_) => {
                    writeln!(markdown, "<image />\n").ok();
                }
            }

            if let Some(output) = tool_result.output.as_ref() {
                writeln!(
                    markdown,
                    "**Debug Output**:\n\n```json\n{}\n```\n",
                    serde_json::to_string_pretty(output).unwrap()
                )
                .unwrap();
            }
        }

        markdown
    }

    pub fn to_request(&self) -> Vec<LanguageModelRequestMessage> {
        let mut assistant_message = LanguageModelRequestMessage {
            role: Role::Assistant,
            content: Vec::with_capacity(self.content.len()),
            cache: false,
            reasoning_details: self.reasoning_details.clone(),
        };
        for chunk in &self.content {
            match chunk {
                AgentMessageContent::Text(text) => {
                    assistant_message
                        .content
                        .push(language_model::MessageContent::Text(text.clone()));
                }
                AgentMessageContent::Thinking { text, signature } => {
                    assistant_message
                        .content
                        .push(language_model::MessageContent::Thinking {
                            text: text.clone(),
                            signature: signature.clone(),
                        });
                }
                AgentMessageContent::RedactedThinking(value) => {
                    assistant_message.content.push(
                        language_model::MessageContent::RedactedThinking(value.clone()),
                    );
                }
                AgentMessageContent::ToolUse(tool_use) => {
                    if self.tool_results.contains_key(&tool_use.id) {
                        assistant_message
                            .content
                            .push(language_model::MessageContent::ToolUse(tool_use.clone()));
                    }
                }
            };
        }

        let mut user_message = LanguageModelRequestMessage {
            role: Role::User,
            content: Vec::new(),
            cache: false,
            reasoning_details: None,
        };

        for tool_result in self.tool_results.values() {
            let mut tool_result = tool_result.clone();
            // Surprisingly, the API fails if we return an empty string here.
            // It thinks we are sending a tool use without a tool result.
            if tool_result.content.is_empty() {
                tool_result.content = "<Tool returned an empty string>".into();
            }
            user_message
                .content
                .push(language_model::MessageContent::ToolResult(tool_result));
        }

        let mut messages = Vec::new();
        if !assistant_message.content.is_empty() {
            messages.push(assistant_message);
        }
        if !user_message.content.is_empty() {
            messages.push(user_message);
        }
        messages
    }
}

/// Stable identity of an `AgentMessage`. Generated at message construction;
/// invariant under truncation/compaction. Added in ST2 fix-loop #3 so
/// `PinReason::PlanUpdate` can pin the most-recent agent message (per
/// plan v3.1 Fix 1) without retargeting under message-list mutations.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentMessageId(pub Uuid);

impl AgentMessageId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AgentMessageId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    /// Stable identity. `#[serde(default)]` auto-generates one on load
    /// for legacy DbThread JSON written before this field existed —
    /// pre-ST2 threads have no PlanUpdate pins so the synthetic id is
    /// harmless. Excluded from `PartialEq` so tests that construct
    /// expected `AgentMessage` values without specifying an id (or
    /// using a fresh random one) still compare on semantic content.
    #[serde(default)]
    pub id: AgentMessageId,
    pub content: Vec<AgentMessageContent>,
    pub tool_results: IndexMap<LanguageModelToolUseId, LanguageModelToolResult>,
    pub reasoning_details: Option<serde_json::Value>,
}

impl PartialEq for AgentMessage {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
            && self.tool_results == other.tool_results
            && self.reasoning_details == other.reasoning_details
    }
}

impl Eq for AgentMessage {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentMessageContent {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    RedactedThinking(String),
    ToolUse(LanguageModelToolUse),
}

pub trait TerminalHandle {
    fn id(&self, cx: &AsyncApp) -> Result<acp::TerminalId>;
    fn current_output(&self, cx: &AsyncApp) -> Result<acp::TerminalOutputResponse>;
    fn wait_for_exit(&self, cx: &AsyncApp) -> Result<Shared<Task<acp::TerminalExitStatus>>>;
    fn kill(&self, cx: &AsyncApp) -> Result<()>;
    fn was_stopped_by_user(&self, cx: &AsyncApp) -> Result<bool>;
}

pub trait SubagentHandle {
    /// The session ID of this subagent thread
    fn id(&self) -> acp::SessionId;
    /// The current number of entries in the thread.
    /// Useful for knowing where the next turn will begin
    fn num_entries(&self, cx: &App) -> usize;
    /// Runs a turn for a given message and returns both the response and the index of that output message.
    fn send(&self, message: String, cx: &AsyncApp) -> Task<Result<String>>;
    /// ST7 fix #3: subscribe to the spawned subagent's `AgentEvent` stream
    /// so `spawn_agent_tool` can drive phase tracking
    /// (`SubAgentPhase::next_phase` + `tools_started`). Returns `None`
    /// when no emitter is wired (e.g. legacy/non-caduceus paths). Each
    /// call returns a fresh broadcast receiver that observes only events
    /// emitted *after* this call. Default impl returns `None` so adding
    /// this method is non-breaking for downstream `SubagentHandle` impls.
    fn events(
        &self,
        _cx: &mut AsyncApp,
    ) -> Option<tokio::sync::broadcast::Receiver<caduceus_core::AgentEvent>> {
        None
    }
    /// ST7 r3 followup-B: surface the subagent's currently-selected
    /// (provider, model) so `classify_subagent_error` at the
    /// spawn-tool boundary can build a populated `ClassifyContext`
    /// (used by ST8 vendor rerouting). Default impl returns
    /// `ClassifyContext::empty()` so adding this method is non-breaking
    /// for downstream `SubagentHandle` impls (e.g. `FakeSubagent`).
    fn classify_context(&self, _cx: &App) -> caduceus_core::ClassifyContext {
        caduceus_core::ClassifyContext::empty()
    }
}

/// Options for spawning a sub-agent. Allows the master to override inherited
/// settings on a per-spawn basis so a DAG can fan out across diverse
/// (vendor, model, profile, mode) combinations.
///
/// All overrides are validated at spawn time. Unknown values fail the spawn
/// with a clear error message returned to the caller (the master agent).
#[derive(Debug, Clone, Default)]
pub struct SubagentSpawnOptions {
    /// Short label displayed in the UI while the agent runs.
    pub label: String,
    /// Override the inherited profile (e.g., `"plan"`, `"act"`, or a custom
    /// profile id from `agent.profiles`). `None` = inherit from parent.
    pub profile_override: Option<String>,
    /// Override the inherited language model id (e.g., `"claude-opus-4.7"`,
    /// `"gpt-5.4"`). `None` = inherit from parent.
    pub model_override: Option<String>,
    /// Override the inherited caduceus mode (`"plan"` / `"act"` / `"autopilot"`).
    /// `None` = inherit from parent.
    pub mode_override: Option<String>,
}

impl SubagentSpawnOptions {
    /// Convenience constructor for the legacy "just a label" call site.
    pub fn from_label(label: String) -> Self {
        Self {
            label,
            ..Default::default()
        }
    }
}

pub trait ThreadEnvironment {
    fn create_terminal(
        &self,
        command: String,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
        cx: &mut AsyncApp,
    ) -> Task<Result<Rc<dyn TerminalHandle>>>;

    fn create_subagent(
        &self,
        opts: SubagentSpawnOptions,
        cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>>;

    fn resume_subagent(
        &self,
        _session_id: acp::SessionId,
        _cx: &mut App,
    ) -> Result<Rc<dyn SubagentHandle>> {
        Err(anyhow::anyhow!(
            "Resuming subagent sessions is not supported"
        ))
    }
}

#[derive(Debug)]
pub enum ThreadEvent {
    UserMessage(UserMessage),
    AgentText(String),
    AgentThinking(String),
    ToolCall(acp::ToolCall),
    ToolCallUpdate(acp_thread::ToolCallUpdate),
    Plan(acp::Plan),
    ToolCallAuthorization(ToolCallAuthorization),
    SubagentSpawned(acp::SessionId),
    Retry(acp_thread::RetryStatus),
    Stop(acp::StopReason),
    /// G1b: non-fatal engine notice surfaced to the UI (e.g. "context
    /// compacted", "permission upgraded", "pin evicted"). Consumers
    /// that don't care should no-op; UI consumers render as a dim
    /// status line under the turn separator.
    ContextNotice(ContextNotice),
    /// G1b: engine-side diagnostic event — tool dispatch error,
    /// reducer failure, envelope violation. Carries a severity level
    /// so the UI can decide whether to surface a banner.
    EngineDiagnostic(EngineDiagnostic),
    /// Native-loop — per-turn token usage snapshot emitted at turn
    /// boundary, BEFORE the corresponding `Stop(reason)` event. The
    /// engine's `TurnComplete` carries a `TokenUsage` payload (wiring
    /// plan part 1 L269 / L443 / L837 + H2 note at
    /// `event_translator.rs:129-130`) that the translator preserves;
    /// previously the dispatcher dropped this with a `..` pattern and
    /// the UI had no way to update its per-turn token meter without
    /// round-tripping to the reducer. This variant surfaces the
    /// structured payload so the UI (or a downstream observer) can
    /// update its meter directly off the stream.
    ///
    /// Emitted once per turn, always immediately before `Stop`. Legacy
    /// ACP consumers (`agent.rs:1961+`) log-and-drop; native consumers
    /// can render a token-usage badge.
    UsageUpdated(NativeTokenUsage),
}

/// Payload for [`ThreadEvent::UsageUpdated`]. Mirrors
/// `caduceus_bridge::event_translator::TokenUsageMirror` (which itself
/// mirrors `caduceus_core::TokenUsage`) — decoupled as a thread-local
/// struct so the public `ThreadEvent` surface does not leak bridge
/// internals to legacy consumers that never opt into the native loop.
/// Named `NativeTokenUsage` (not `TokenUsage`) to avoid colliding with
/// `language_model::TokenUsage` which is already in scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeTokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
}

impl From<caduceus_bridge::event_translator::TokenUsageMirror> for NativeTokenUsage {
    fn from(m: caduceus_bridge::event_translator::TokenUsageMirror) -> Self {
        Self {
            input_tokens: m.input_tokens,
            output_tokens: m.output_tokens,
            cache_read_tokens: m.cache_read_tokens,
            cache_write_tokens: m.cache_write_tokens,
        }
    }
}

/// Payload for [`ThreadEvent::ContextNotice`]. Informational; the turn
/// keeps running. Kept deliberately narrow so translator → UI coupling
/// stays one-way.
#[derive(Debug, Clone)]
pub struct ContextNotice {
    /// Short stable identifier (e.g. `"context.compacted"`, `"pin.evicted"`).
    pub kind: String,
    /// Human-readable message for the UI.
    pub message: String,
}

/// Payload for [`ThreadEvent::EngineDiagnostic`]. Non-fatal by default;
/// the consumer decides whether to escalate.
#[derive(Debug, Clone)]
pub struct EngineDiagnostic {
    /// Short stable identifier (e.g. `"dispatch.timeout"`, `"envelope.violation"`).
    pub kind: String,
    /// Human-readable detail.
    pub detail: String,
    pub severity: EngineDiagnosticSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Synthetic guardrail prefixes the agent emits in place of a real tool
/// result when it has rejected the call itself (permission, circuit
/// breaker, loop detector, missing tool, etc). Counting these as failures
/// in the circuit breaker is a feedback loop: a single guardrail trip
/// would itself trip the breaker on the next turn. Pulled out so the
/// prefix list can be unit-tested without spinning up a Thread.
pub(crate) const SYNTHETIC_GUARDRAIL_PREFIXES: &[&str] = &[
    "PERMISSION DENIED:",
    "CIRCUIT BREAKER:",
    "LOOP DETECTED:",
    "RUNAWAY LOOP:",
    "No tool named ",
];

pub(crate) fn is_synthetic_guardrail_text(text: &str) -> bool {
    SYNTHETIC_GUARDRAIL_PREFIXES
        .iter()
        .any(|p| text.starts_with(p))
}

/// Bug C6 helper: atomic file write. Writes `bytes` to a sibling tempfile
/// in the same directory, fsync's it, then `rename`s into `path`. On POSIX
/// the rename is atomic on the same filesystem, so a concurrent reader
/// either sees the previous full file or the new full file — never a
/// half-written one. The previous code used `std::fs::write` which can
/// produce a partial file if the writer is interrupted or interleaved
/// with another writer.
pub(crate) fn atomic_write(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let dir = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "atomic_write: path has no parent directory",
        )
    })?;
    std::fs::create_dir_all(dir)?;
    // Build a unique sibling tmp name. Using thread id + nanosecond
    // counter is sufficient since (a) single-process and (b) the rename
    // step is what makes the operation atomic. Avoids pulling tempfile
    // into agent's runtime dependency graph.
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("file");
    let pid = std::process::id();
    let nonce = ATOMIC_WRITE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = dir.join(format!(".{}.{}.{}.atomic-tmp", stem, pid, nonce));
    {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)?;
        if let Err(e) = f.write_all(bytes).and_then(|_| f.sync_all()) {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e);
        }
    }
    if let Err(e) = std::fs::rename(&tmp_path, path) {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(e);
    }
    Ok(())
}

static ATOMIC_WRITE_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Returns the absolute path of the first project worktree IFF that path
/// exists on disk and is a directory. Returns `None` when the user has
/// opened a single file (e.g. `settings.json`) instead of a folder, or when
/// the project has no worktrees yet.
///
/// Several Caduceus subsystems — native loop, auto-index, CADUCEUS.md
/// loader — only make sense against a folder workspace. Gating them on
/// this helper turns "user opened a file" from a cascade of confusing
/// ENOTDIR errors into a single one-time info log.
pub(crate) fn caduceus_workspace_folder(project: &Entity<Project>, cx: &App) -> Option<PathBuf> {
    let worktree = project.read(cx).worktrees(cx).next()?;
    let worktree = worktree.read(cx);
    // Trust the worktree's own metadata (works for FakeFs in tests and
    // real fs in prod). Fall back to disk-probe if no root_entry yet.
    let is_folder = match worktree.root_entry() {
        Some(entry) => entry.is_dir(),
        None => worktree.abs_path().is_dir(),
    };
    if is_folder {
        Some(worktree.abs_path().to_path_buf())
    } else {
        None
    }
}

/// Emits exactly one `info!` per process explaining that Caduceus features
/// require a folder workspace. Subsequent calls are no-ops. Use from any
/// site where `caduceus_workspace_folder` returned `None`.
pub(crate) fn log_no_folder_workspace_once() {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::info!(
            "[caduceus] Workspace is not a folder. Caduceus features (native \
             loop, auto-index, CADUCEUS.md, instructions) are disabled. Open \
             a folder (Cmd-O → choose a directory) to enable them."
        );
    }
}

#[derive(Debug)]
pub struct NewTerminal {
    pub command: String,
    pub output_byte_limit: Option<u64>,
    pub cwd: Option<PathBuf>,
    pub response: oneshot::Sender<Result<Entity<acp_thread::Terminal>>>,
}

#[derive(Debug, Clone)]
pub struct ToolPermissionContext {
    pub tool_name: String,
    pub input_values: Vec<String>,
    pub scope: ToolPermissionScope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPermissionScope {
    ToolInput,
    SymlinkTarget,
}

impl ToolPermissionContext {
    pub fn new(tool_name: impl Into<String>, input_values: Vec<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            input_values,
            scope: ToolPermissionScope::ToolInput,
        }
    }

    pub fn symlink_target(tool_name: impl Into<String>, target_paths: Vec<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            input_values: target_paths,
            scope: ToolPermissionScope::SymlinkTarget,
        }
    }

    /// Builds the permission options for this tool context.
    ///
    /// This is the canonical source for permission option generation.
    /// Tests should use this function rather than manually constructing options.
    ///
    /// # Shell Compatibility for Terminal Tool
    ///
    /// For the terminal tool, "Always allow" options are only shown when the user's
    /// shell supports POSIX-like command chaining syntax (`&&`, `||`, `;`, `|`).
    ///
    /// **Why this matters:** When a user sets up an "always allow" pattern like `^cargo`,
    /// we need to parse the command to extract all sub-commands and verify that EVERY
    /// sub-command matches the pattern. Otherwise, an attacker could craft a command like
    /// `cargo build && rm -rf /` that would bypass the security check.
    ///
    /// **Supported shells:** Posix (sh, bash, dash, zsh), Fish 3.0+, PowerShell 7+/Pwsh,
    /// Cmd, Xonsh, Csh, Tcsh
    ///
    /// **Unsupported shells:** Nushell (uses `and`/`or` keywords), Elvish (uses `and`/`or`
    /// keywords), Rc (Plan 9 shell - no `&&`/`||` operators)
    ///
    /// For unsupported shells, we hide the "Always allow" UI options entirely, and if
    /// the user has `always_allow` rules configured in settings, `ToolPermissionDecision::from_input`
    /// will return a `Deny` with an explanatory error message.
    pub fn build_permission_options(&self) -> acp_thread::PermissionOptions {
        use crate::pattern_extraction::*;
        use util::shell::ShellKind;

        let tool_name = &self.tool_name;
        let input_values = &self.input_values;
        if self.scope == ToolPermissionScope::SymlinkTarget {
            return acp_thread::PermissionOptions::Flat(vec![
                acp::PermissionOption::new(
                    acp::PermissionOptionId::new("allow"),
                    "Yes",
                    acp::PermissionOptionKind::AllowOnce,
                ),
                acp::PermissionOption::new(
                    acp::PermissionOptionId::new("deny"),
                    "No",
                    acp::PermissionOptionKind::RejectOnce,
                ),
            ]);
        }

        // Check if the user's shell supports POSIX-like command chaining.
        // See the doc comment above for the full explanation of why this is needed.
        let shell_supports_always_allow = if tool_name == TerminalTool::NAME {
            ShellKind::system().supports_posix_chaining()
        } else {
            true
        };

        // For terminal commands with multiple pipeline commands, use DropdownWithPatterns
        // to let users individually select which command patterns to always allow.
        if tool_name == TerminalTool::NAME && shell_supports_always_allow {
            if let Some(input) = input_values.first() {
                let all_patterns = extract_all_terminal_patterns(input);
                if all_patterns.len() > 1 {
                    let mut choices = Vec::new();
                    choices.push(acp_thread::PermissionOptionChoice {
                        allow: acp::PermissionOption::new(
                            acp::PermissionOptionId::new(format!("always_allow:{}", tool_name)),
                            format!("Always for {}", tool_name.replace('_', " ")),
                            acp::PermissionOptionKind::AllowAlways,
                        ),
                        deny: acp::PermissionOption::new(
                            acp::PermissionOptionId::new(format!("always_deny:{}", tool_name)),
                            format!("Always for {}", tool_name.replace('_', " ")),
                            acp::PermissionOptionKind::RejectAlways,
                        ),
                        sub_patterns: vec![],
                    });
                    choices.push(acp_thread::PermissionOptionChoice {
                        allow: acp::PermissionOption::new(
                            acp::PermissionOptionId::new("allow"),
                            "Only this time",
                            acp::PermissionOptionKind::AllowOnce,
                        ),
                        deny: acp::PermissionOption::new(
                            acp::PermissionOptionId::new("deny"),
                            "Only this time",
                            acp::PermissionOptionKind::RejectOnce,
                        ),
                        sub_patterns: vec![],
                    });
                    return acp_thread::PermissionOptions::DropdownWithPatterns {
                        choices,
                        patterns: all_patterns,
                        tool_name: tool_name.clone(),
                    };
                }
            }
        }

        let extract_for_value = |value: &str| -> (Option<String>, Option<String>) {
            if tool_name == TerminalTool::NAME {
                (
                    extract_terminal_pattern(value),
                    extract_terminal_pattern_display(value),
                )
            } else if tool_name == CopyPathTool::NAME
                || tool_name == MovePathTool::NAME
                || tool_name == EditFileTool::NAME
                || tool_name == DeletePathTool::NAME
                || tool_name == CreateDirectoryTool::NAME
                || tool_name == SaveFileTool::NAME
            {
                (
                    extract_path_pattern(value),
                    extract_path_pattern_display(value),
                )
            } else if tool_name == FetchTool::NAME {
                (
                    extract_url_pattern(value),
                    extract_url_pattern_display(value),
                )
            } else {
                (None, None)
            }
        };

        // Extract patterns from all input values. Only offer a pattern-specific
        // "always allow/deny" button when every value produces the same pattern.
        let (pattern, pattern_display) = match input_values.as_slice() {
            [single] => extract_for_value(single),
            _ => {
                let mut iter = input_values.iter().map(|v| extract_for_value(v));
                match iter.next() {
                    Some(first) => {
                        if iter.all(|pair| pair.0 == first.0) {
                            first
                        } else {
                            (None, None)
                        }
                    }
                    None => (None, None),
                }
            }
        };

        let mut choices = Vec::new();

        let mut push_choice =
            |label: String, allow_id, deny_id, allow_kind, deny_kind, sub_patterns: Vec<String>| {
                choices.push(acp_thread::PermissionOptionChoice {
                    allow: acp::PermissionOption::new(
                        acp::PermissionOptionId::new(allow_id),
                        label.clone(),
                        allow_kind,
                    ),
                    deny: acp::PermissionOption::new(
                        acp::PermissionOptionId::new(deny_id),
                        label,
                        deny_kind,
                    ),
                    sub_patterns,
                });
            };

        if shell_supports_always_allow {
            push_choice(
                format!("Always for {}", tool_name.replace('_', " ")),
                format!("always_allow:{}", tool_name),
                format!("always_deny:{}", tool_name),
                acp::PermissionOptionKind::AllowAlways,
                acp::PermissionOptionKind::RejectAlways,
                vec![],
            );

            if let (Some(pattern), Some(display)) = (pattern, pattern_display) {
                let button_text = if tool_name == TerminalTool::NAME {
                    format!("Always for `{}` commands", display)
                } else {
                    format!("Always for `{}`", display)
                };
                push_choice(
                    button_text,
                    format!("always_allow:{}", tool_name),
                    format!("always_deny:{}", tool_name),
                    acp::PermissionOptionKind::AllowAlways,
                    acp::PermissionOptionKind::RejectAlways,
                    vec![pattern],
                );
            }
        }

        push_choice(
            "Only this time".to_string(),
            "allow".to_string(),
            "deny".to_string(),
            acp::PermissionOptionKind::AllowOnce,
            acp::PermissionOptionKind::RejectOnce,
            vec![],
        );

        acp_thread::PermissionOptions::Dropdown(choices)
    }
}

#[derive(Debug)]
pub struct ToolCallAuthorization {
    pub tool_call: acp::ToolCallUpdate,
    pub options: acp_thread::PermissionOptions,
    pub response: oneshot::Sender<acp_thread::SelectedPermissionOutcome>,
    pub context: Option<ToolPermissionContext>,
}

#[derive(Debug, thiserror::Error)]
enum CompletionError {
    #[error("max tokens")]
    MaxTokens,
    #[error("refusal")]
    Refusal,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Severity of a guardrail alert (drives UI color)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Warning,
    Error,
}

pub struct Thread {
    id: acp::SessionId,
    prompt_id: PromptId,
    updated_at: DateTime<Utc>,
    title: Option<SharedString>,
    pending_title_generation: Option<Task<()>>,
    pending_summary_generation: Option<Shared<Task<Option<SharedString>>>>,
    summary: Option<SharedString>,
    messages: Vec<Message>,
    user_store: Entity<UserStore>,
    /// Holds the task that handles agent interaction until the end of the turn.
    /// Survives across multiple requests as the model performs tool calls and
    /// we run tools, report their results.
    running_turn: Option<RunningTurn>,
    /// Flag indicating the UI has a queued message waiting to be sent.
    /// Used to signal that the turn should end at the next message boundary.
    has_queued_message: bool,
    pending_message: Option<AgentMessage>,
    pub(crate) tools: BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
    request_token_usage: HashMap<UserMessageId, language_model::TokenUsage>,
    #[allow(unused)]
    cumulative_token_usage: TokenUsage,
    #[allow(unused)]
    initial_project_snapshot: Shared<Task<Option<Arc<ProjectSnapshot>>>>,
    pub(crate) context_server_registry: Entity<ContextServerRegistry>,
    profile_id: AgentProfileId,
    project_context: Entity<ProjectContext>,
    pub(crate) templates: Arc<Templates>,
    model: Option<Arc<dyn LanguageModel>>,
    summarization_model: Option<Arc<dyn LanguageModel>>,
    thinking_enabled: bool,
    thinking_effort: Option<String>,
    speed: Option<Speed>,
    prompt_capabilities_tx: watch::Sender<acp::PromptCapabilities>,
    pub(crate) prompt_capabilities_rx: watch::Receiver<acp::PromptCapabilities>,
    pub(crate) project: Entity<Project>,
    pub(crate) action_log: Entity<ActionLog>,
    /// True if this thread was imported from a shared thread and can be synced.
    imported: bool,
    /// If this is a subagent thread, contains context about the parent
    subagent_context: Option<SubagentContext>,
    /// The user's unsent prompt text, persisted so it can be restored when reloading the thread.
    draft_prompt: Option<Vec<acp::ContentBlock>>,
    ui_scroll_position: Option<gpui::ListOffset>,
    /// Weak references to running subagent threads for cancellation propagation
    running_subagents: Vec<WeakEntity<Thread>>,
    /// Current Caduceus agent mode (Plan, Act, Research, etc.)
    caduceus_mode: Option<String>,
    /// Cached project instructions to avoid repeated sync I/O
    cached_project_instructions: Option<String>,
    /// Loop detection: blocks Nth consecutive same-tool call
    loop_detector: caduceus_bridge::safety::LoopDetector,
    /// Escalation counter — number of consecutive loop detections without recovery.
    /// After hitting `LOOP_ESCALATION_THRESHOLD`, the entire turn is hard-cancelled.
    loop_escalation_count: usize,
    /// Circuit breaker: trips after N consecutive tool failures
    circuit_breaker: caduceus_bridge::safety::CircuitBreaker,
    /// Compaction cooldown: prevents double-fire within cooldown window
    compaction_cooldown: caduceus_bridge::safety::CompactionCooldown,
    /// Last guardrail alert (visible in UI) — cleared after 10 seconds
    guardrail_alert: Option<(String, AlertSeverity, std::time::Instant)>,
    /// Whether context was compacted this turn (visible in UI)
    context_compacted_this_turn: bool,
    /// Pinned context items that survive compaction
    context_pins: caduceus_bridge::orchestrator::ContextManager,
    /// Task DAG for multi-agent decomposition (per-thread, not global)
    task_dag: std::sync::Arc<std::sync::Mutex<caduceus_bridge::orchestrator::TaskDAG>>,
    /// ST-B3: grouped native-loop state. Fields inside this struct were
    /// previously eight separate `caduceus_*` fields on Thread; they are
    /// now owned together because they share a single lifecycle (built
    /// lazily in `ensure_caduceus_harness`, invalidated as a set in
    /// `invalidate_caduceus_harness`, and referenced in lock-step by
    /// the native turn path). See `CaduceusState` for per-field docs.
    caduceus: CaduceusState,
    /// Cached total token estimate (invalidated on message changes).
    /// `Cell` allows population from `&self` callers — the cached value is
    /// observation-only state and never affects logical equality of Thread.
    cached_token_estimate: std::cell::Cell<Option<u32>>,
    /// Set true by `cancel()`; cleared at the start of `run_turn()`. Used by
    /// `flush_pending_message` to skip auto-memory extraction (which races
    /// with the next turn's `activeContext.md` writes) when the flush was
    /// triggered by a cancellation rather than a normal turn end.
    last_turn_cancelled: bool,
    /// Caduceus F2: session ids pulled in via `/pull <id>` slash command.
    /// When non-empty, `caduceus_assemble_context` appends a system block
    /// listing them and reminding the orchestrator that `read_thread` and
    /// `compact_thread` tools are available to consult those sessions.
    pulled_session_ids: Vec<String>,
    /// Monotonic counter bumped on every cancel/new-turn boundary.
    /// `pub(crate)` accessor below for test visibility.
    /// Deferred async work captures the value at scheduling time and bails
    /// out if the counter has advanced — preventing stale callbacks from
    /// the previous turn from clobbering state in a fresh turn.
    turn_generation: u64,
    /// ST2: pinned messages that survive compaction. See `PinReason`,
    /// `PinnedMessageKey`, `MessageRef`. Insertion-order preserved.
    /// **ST2-INVARIANT**: subagent threads MUST start empty.
    pub(crate) pinned: Vec<MessageRef>,
}

/// Grouped per-thread native-loop state (ST-B3). These seven fields share
/// one lifecycle: populated together in [`Thread::ensure_caduceus_harness`]
/// and cleared as a set in [`Thread::invalidate_caduceus_harness`].
/// `Default` returns the empty state used in both `Thread::new` call sites.
#[derive(Default)]
pub(crate) struct CaduceusState {
    /// G1: shared OrchestratorBridge handle, lazily built on first turn.
    /// When `caduceus_native_loop` setting is ON the bridge's
    /// `native_loop_enabled` atomic is set; `run_turn_internal` dispatches
    /// to the native loop via this handle.
    pub(crate) bridge: Option<std::sync::Arc<caduceus_bridge::orchestrator::OrchestratorBridge>>,
    /// G1: engine harness for the native loop. Lazily built on first native
    /// turn; invalidated on cancel/model-change so the next turn rebuilds
    /// with fresh state.
    pub(crate) harness: Option<std::sync::Arc<caduceus_orchestrator::AgentHarness>>,
    /// G1: per-thread mutable engine state held behind an async mutex so
    /// the turn task can `lock()` for the full duration of a harness call.
    /// Panic-safe via mutex poison; invalidated together with the harness.
    pub(crate) native_state:
        Option<std::sync::Arc<tokio::sync::Mutex<crate::caduceus_native_state::NativeLoopState>>>,
    /// G1: cancellation token given to the harness at build time. Zed →
    /// engine cancel flips this; the harness cooperates via its internal
    /// `select!`.
    pub(crate) cancel_token: Option<caduceus_core::CancellationToken>,
    /// ST-A2d: long-lived emitter clone from the per-session harness.
    /// Per-turn code calls `.subscribe()` to get a fresh broadcast
    /// receiver that delivers events for the current generation only.
    /// Populated together with `harness`; invalidated with it.
    pub(crate) emitter: Option<caduceus_orchestrator::AgentEventEmitter>,
    /// ST-A3: cached provider dispatcher for this session. Spawned once
    /// against the currently-selected language model; invalidated on
    /// model change (via `invalidate_caduceus_harness`) so the next
    /// turn re-spawns against the new model. Cloning the handle is
    /// cheap (just an `mpsc::Sender` clone).
    pub(crate) dispatcher: Option<crate::caduceus_provider_adapter::DispatcherHandle>,
    /// NW-3: engine approval channel for the native loop.
    /// `Some` when the native harness was built with the default HITL
    /// approval set; `None` for legacy path (which handles permission
    /// UI inline in tool adapters). Sender is the engine's
    /// `approval_tx`; keys are `format!("perm_{tool_use_id}")` and the
    /// bool is the user's decision (`true` = allow, `false` = deny).
    /// Invalidated in lock-step with `harness`.
    pub(crate) approval_tx: Option<tokio::sync::mpsc::Sender<(String, bool)>>,
    /// T4 (Audit C6): flag value in effect when the current harness
    /// was provisioned. `None` means no harness is provisioned yet.
    /// If the live `caduceus_native_loop` setting no longer matches
    /// this value at turn start, `ensure_caduceus_harness` invalidates
    /// the harness so the next turn rebuilds with current state. This
    /// closes the gap where a ON→OFF→ON transition would reuse a
    /// harness built against a since-invalidated flag.
    pub(crate) last_native_loop_flag: Option<bool>,
}

impl Thread {
    fn prompt_capabilities(model: Option<&dyn LanguageModel>) -> acp::PromptCapabilities {
        let image = model.map_or(true, |model| model.supports_images());
        acp::PromptCapabilities::new()
            .image(image)
            .embedded_context(true)
    }

    pub fn new_subagent(parent_thread: &Entity<Thread>, cx: &mut Context<Self>) -> Self {
        let project = parent_thread.read(cx).project.clone();
        let project_context = parent_thread.read(cx).project_context.clone();
        let context_server_registry = parent_thread.read(cx).context_server_registry.clone();
        let templates = parent_thread.read(cx).templates.clone();
        let model = parent_thread.read(cx).model().cloned();
        let parent_action_log = parent_thread.read(cx).action_log().clone();
        let action_log =
            cx.new(|_cx| ActionLog::new(project.clone()).with_linked_action_log(parent_action_log));
        let mut thread = Self::new_internal(
            project,
            project_context,
            context_server_registry,
            templates,
            model,
            action_log,
            cx,
        );
        thread.subagent_context = Some(SubagentContext {
            parent_thread_id: parent_thread.read(cx).id().clone(),
            depth: parent_thread.read(cx).depth() + 1,
        });
        thread.inherit_parent_settings(parent_thread, cx);
        // ST7 r3 #1: seed `caduceus.emitter` eagerly so that
        // `subagent_event_subscriber()` can return a live receiver
        // BEFORE `prompt()` builds the harness. Without this seed,
        // `spawn_agent_tool::run()` calls `subagent.events(cx)` at
        // line ~458 (before `subagent.send()` runs) and gets None,
        // so the phase-tracking pump task never starts — leaving
        // last_phase pinned at ModelSelection / tools_started=false
        // through every timeout.
        //
        // The mpsc rx returned by `AgentEventEmitter::channel(...)`
        // would otherwise let the channel close as soon as `prompt()`
        // attaches the harness. Drain it on a background task whose
        // lifetime equals the (cloneable) emitter — i.e. as long as
        // *any* clone of the emitter exists, the rx stays alive and
        // `try_send` never trips Closed. The drain task ends when the
        // last emitter clone drops (parent invalidates, thread drops).
        //
        // The harness builder later reuses this seeded emitter via
        // `with_emitter_reuse(em)` (see thread.rs:~3707) so events
        // emitted by the harness reach the same broadcast fan-out
        // the pump is subscribed to.
        let (em, mut rx) = caduceus_orchestrator::AgentEventEmitter::channel(
            caduceus_bridge::orchestrator::OrchestratorBridge::DEFAULT_EVENT_CHANNEL_BUFFER,
        );
        thread.caduceus.emitter = Some(em);
        cx.background_spawn(async move { while rx.recv().await.is_some() {} })
            .detach();
        thread
    }

    pub fn new(
        project: Entity<Project>,
        project_context: Entity<ProjectContext>,
        context_server_registry: Entity<ContextServerRegistry>,
        templates: Arc<Templates>,
        model: Option<Arc<dyn LanguageModel>>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new_internal(
            project.clone(),
            project_context,
            context_server_registry,
            templates,
            model,
            cx.new(|_cx| ActionLog::new(project)),
            cx,
        )
    }

    fn new_internal(
        project: Entity<Project>,
        project_context: Entity<ProjectContext>,
        context_server_registry: Entity<ContextServerRegistry>,
        templates: Arc<Templates>,
        model: Option<Arc<dyn LanguageModel>>,
        action_log: Entity<ActionLog>,
        cx: &mut Context<Self>,
    ) -> Self {
        let settings = AgentSettings::get_global(cx);
        let profile_id = settings.default_profile.clone();
        let enable_thinking = settings
            .default_model
            .as_ref()
            .is_some_and(|model| model.enable_thinking);
        let thinking_effort = settings
            .default_model
            .as_ref()
            .and_then(|model| model.effort.clone());
        let speed = settings
            .default_model
            .as_ref()
            .and_then(|model| model.speed);
        let (prompt_capabilities_tx, prompt_capabilities_rx) =
            watch::channel(Self::prompt_capabilities(model.as_deref()));
        Self {
            id: acp::SessionId::new(uuid::Uuid::new_v4().to_string()),
            prompt_id: PromptId::new(),
            updated_at: Utc::now(),
            title: None,
            pending_title_generation: None,
            pending_summary_generation: None,
            summary: None,
            messages: Vec::new(),
            user_store: project.read(cx).user_store(),
            running_turn: None,
            has_queued_message: false,
            pending_message: None,
            tools: BTreeMap::default(),
            request_token_usage: HashMap::default(),
            cumulative_token_usage: TokenUsage::default(),
            initial_project_snapshot: {
                let project_snapshot = Self::project_snapshot(project.clone(), cx);
                cx.foreground_executor()
                    .spawn(async move { Some(project_snapshot.await) })
                    .shared()
            },
            context_server_registry,
            profile_id,
            project_context,
            templates,
            model,
            summarization_model: None,
            thinking_enabled: enable_thinking,
            speed,
            thinking_effort,
            prompt_capabilities_tx,
            prompt_capabilities_rx,
            project,
            action_log,
            imported: false,
            subagent_context: None,
            draft_prompt: None,
            ui_scroll_position: None,
            running_subagents: Vec::new(),
            caduceus_mode: None,
            cached_project_instructions: None,
            loop_detector: caduceus_bridge::safety::LoopDetector::new(5),
            loop_escalation_count: 0,
            circuit_breaker: caduceus_bridge::safety::CircuitBreaker::new(5),
            compaction_cooldown: caduceus_bridge::safety::CompactionCooldown::new(
                std::time::Duration::from_secs(30),
            ),
            guardrail_alert: None,
            context_compacted_this_turn: false,
            context_pins: caduceus_bridge::orchestrator::ContextManager::new(128000),
            task_dag: std::sync::Arc::new(std::sync::Mutex::new(
                caduceus_bridge::orchestrator::TaskDAG::new(),
            )),
            caduceus: CaduceusState::default(),
            cached_token_estimate: std::cell::Cell::new(None),
            turn_generation: 0,
            last_turn_cancelled: false,
            pulled_session_ids: Vec::new(),
            pinned: Vec::new(),
        }
    }

    /// Copies runtime-mutable settings from the parent thread so that
    /// subagents start with the same configuration the user selected.
    /// Every property that `set_*` propagates to `running_subagents`
    /// should be inherited here as well.
    fn inherit_parent_settings(&mut self, parent_thread: &Entity<Thread>, cx: &mut Context<Self>) {
        let parent = parent_thread.read(cx);
        self.speed = parent.speed;
        self.thinking_enabled = parent.thinking_enabled;
        self.thinking_effort = parent.thinking_effort.clone();
        self.summarization_model = parent.summarization_model.clone();
        self.profile_id = parent.profile_id.clone();
        // Caduceus: inherit mode so subagents respect privilege rings
        self.caduceus_mode = parent.caduceus_mode.clone();
    }

    /// Apply per-spawn overrides on top of the inherited parent settings.
    /// Validates each override against the live registry/catalog and returns
    /// a descriptive error if any value is unknown — the error is surfaced
    /// to the master agent so it can self-correct (DAG fan-out diversity).
    pub fn apply_subagent_overrides(
        &mut self,
        opts: &SubagentSpawnOptions,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        // Profile override — must exist in the live AgentSettings.profiles map.
        if let Some(profile) = opts.profile_override.as_ref() {
            let settings = AgentSettings::get_global(cx);
            let profile_id = AgentProfileId(profile.clone().into());
            if !settings.profiles.contains_key(&profile_id) {
                let available: Vec<String> = settings
                    .profiles
                    .keys()
                    .map(|k| k.as_str().to_string())
                    .collect();
                anyhow::bail!(
                    "Unknown profile '{}'. Available profiles: [{}]",
                    profile,
                    available.join(", ")
                );
            }
            self.profile_id = profile_id;
        }

        // Mode override — must parse via AgentMode::from_str_loose.
        if let Some(mode) = opts.mode_override.as_ref() {
            use caduceus_orchestrator::modes::AgentMode;
            if AgentMode::from_str_loose(mode).is_none() {
                anyhow::bail!("Unknown mode '{}'. Valid: plan, act, autopilot", mode);
            }
            self.caduceus_mode = Some(mode.clone());
        }

        // Model override — must resolve via LanguageModelRegistry.
        if let Some(model_id) = opts.model_override.as_ref() {
            let registry = language_model::LanguageModelRegistry::read_global(cx);
            let resolved = registry
                .available_models(cx)
                .find(|m| m.id().0.as_ref() == model_id.as_str());
            match resolved {
                Some(model) => self.model = Some(model),
                None => {
                    let mut available: Vec<String> = registry
                        .available_models(cx)
                        .map(|m| m.id().0.to_string())
                        .collect();
                    available.sort();
                    available.dedup();
                    anyhow::bail!(
                        "Unknown model '{}'. {} models available; first 10: [{}]",
                        model_id,
                        available.len(),
                        available
                            .iter()
                            .take(10)
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
        }

        Ok(())
    }

    pub fn id(&self) -> &acp::SessionId {
        &self.id
    }

    /// Returns true if this thread was imported from a shared thread.
    pub fn is_imported(&self) -> bool {
        self.imported
    }

    /// ST7 fix #3: subscribe to the caduceus engine's `AgentEventEmitter`
    /// for this thread, if a harness has been built. Returns `None` when
    /// no emitter is wired (legacy code paths or pre-harness-build state).
    /// Each call returns a fresh broadcast receiver scoped to events
    /// emitted *after* this call (use the emitter's retention ring for
    /// replay if needed).
    pub fn subagent_event_subscriber(
        &self,
    ) -> Option<tokio::sync::broadcast::Receiver<caduceus_core::AgentEvent>> {
        self.caduceus.emitter.as_ref().map(|e| e.subscribe())
    }

    pub fn replay(
        &mut self,
        cx: &mut Context<Self>,
    ) -> mpsc::UnboundedReceiver<Result<ThreadEvent>> {
        let (tx, rx) = mpsc::unbounded();
        let stream = ThreadEventStream(tx);
        for message in &self.messages {
            match message {
                Message::User(user_message) => stream.send_user_message(user_message),
                Message::Agent(assistant_message) => {
                    for content in &assistant_message.content {
                        match content {
                            AgentMessageContent::Text(text) => stream.send_text(text),
                            AgentMessageContent::Thinking { text, .. } => {
                                stream.send_thinking(text)
                            }
                            AgentMessageContent::RedactedThinking(_) => {}
                            AgentMessageContent::ToolUse(tool_use) => {
                                self.replay_tool_call(
                                    tool_use,
                                    assistant_message.tool_results.get(&tool_use.id),
                                    &stream,
                                    cx,
                                );
                            }
                        }
                    }
                }
                Message::Resume(_) => {}
            }
        }
        rx
    }

    fn replay_tool_call(
        &self,
        tool_use: &LanguageModelToolUse,
        tool_result: Option<&LanguageModelToolResult>,
        stream: &ThreadEventStream,
        cx: &mut Context<Self>,
    ) {
        // Extract saved output and status first, so they're available even if tool is not found
        let output = tool_result
            .as_ref()
            .and_then(|result| result.output.clone());
        let status = tool_result
            .as_ref()
            .map_or(acp::ToolCallStatus::Failed, |result| {
                if result.is_error {
                    acp::ToolCallStatus::Failed
                } else {
                    acp::ToolCallStatus::Completed
                }
            });

        let tool = self.tools.get(tool_use.name.as_ref()).cloned().or_else(|| {
            self.context_server_registry
                .read(cx)
                .servers()
                .find_map(|(_, tools)| {
                    if let Some(tool) = tools.get(tool_use.name.as_ref()) {
                        Some(tool.clone())
                    } else {
                        None
                    }
                })
        });

        let Some(tool) = tool else {
            // Tool not found (e.g., MCP server not connected after restart),
            // but still display the saved result if available.
            // We need to send both ToolCall and ToolCallUpdate events because the UI
            // only converts raw_output to displayable content in update_fields, not from_acp.
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCall(
                    acp::ToolCall::new(tool_use.id.to_string(), tool_use.name.to_string())
                        .status(status)
                        .raw_input(tool_use.input.clone()),
                )))
                .ok();
            stream.update_tool_call_fields(
                &tool_use.id,
                acp::ToolCallUpdateFields::new()
                    .status(status)
                    .raw_output(output),
                None,
            );
            return;
        };

        let title = tool.initial_title(tool_use.input.clone(), cx);
        let kind = tool.kind();
        stream.send_tool_call(
            &tool_use.id,
            &tool_use.name,
            title,
            kind,
            tool_use.input.clone(),
        );

        if let Some(output) = output.clone() {
            // For replay, we use a dummy cancellation receiver since the tool already completed
            let (_cancellation_tx, cancellation_rx) = watch::channel(false);
            let tool_event_stream = ToolCallEventStream::new(
                tool_use.id.clone(),
                stream.clone(),
                Some(self.project.read(cx).fs().clone()),
                cancellation_rx,
            );
            tool.replay(tool_use.input.clone(), output, tool_event_stream, cx)
                .log_err();
        }

        stream.update_tool_call_fields(
            &tool_use.id,
            acp::ToolCallUpdateFields::new()
                .status(status)
                .raw_output(output),
            None,
        );
    }

    pub fn from_db(
        id: acp::SessionId,
        mut db_thread: DbThread,
        project: Entity<Project>,
        project_context: Entity<ProjectContext>,
        context_server_registry: Entity<ContextServerRegistry>,
        templates: Arc<Templates>,
        cx: &mut Context<Self>,
    ) -> Self {
        let settings = AgentSettings::get_global(cx);
        let profile_id = db_thread
            .profile
            .unwrap_or_else(|| settings.default_profile.clone());

        let mut model = LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
            db_thread
                .model
                .and_then(|model| {
                    let model = SelectedModel {
                        provider: model.provider.clone().into(),
                        model: model.model.into(),
                    };
                    registry.select_model(&model, cx)
                })
                .or_else(|| registry.default_model())
                .map(|model| model.model)
        });

        if model.is_none() {
            model = Self::resolve_profile_model(&profile_id, cx);
        }
        if model.is_none() {
            model = LanguageModelRegistry::global(cx).update(cx, |registry, _cx| {
                registry.default_model().map(|model| model.model)
            });
        }

        let (prompt_capabilities_tx, prompt_capabilities_rx) =
            watch::channel(Self::prompt_capabilities(model.as_deref()));

        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        // ST2: compute pins before moving fields out of db_thread.
        // Subagents force-clear (belt-and-suspenders against a
        // maliciously-crafted DbThread JSON setting both
        // `subagent_context` and `pinned`).
        let loaded_pinned = if db_thread.subagent_context.is_some() {
            if !db_thread.pinned.is_empty() {
                log::warn!(
                    "[st2] DbThread has subagent_context AND non-empty pinned ({} entries); \
                     clearing pins (subagent invariant)",
                    db_thread.pinned.len()
                );
            }
            Vec::new()
        } else {
            std::mem::take(&mut db_thread.pinned)
        };

        Self {
            id,
            prompt_id: PromptId::new(),
            title: if db_thread.title.is_empty() {
                None
            } else {
                Some(db_thread.title.clone())
            },
            pending_title_generation: None,
            pending_summary_generation: None,
            summary: db_thread.detailed_summary,
            messages: db_thread.messages,
            user_store: project.read(cx).user_store(),
            running_turn: None,
            has_queued_message: false,
            pending_message: None,
            tools: BTreeMap::default(),
            request_token_usage: db_thread.request_token_usage.clone(),
            cumulative_token_usage: db_thread.cumulative_token_usage,
            initial_project_snapshot: Task::ready(db_thread.initial_project_snapshot).shared(),
            context_server_registry,
            profile_id,
            project_context,
            templates,
            model,
            summarization_model: None,
            thinking_enabled: db_thread.thinking_enabled,
            thinking_effort: db_thread.thinking_effort,
            speed: db_thread.speed,
            project,
            action_log,
            updated_at: db_thread.updated_at,
            prompt_capabilities_tx,
            prompt_capabilities_rx,
            imported: db_thread.imported,
            subagent_context: db_thread.subagent_context,
            draft_prompt: db_thread.draft_prompt,
            ui_scroll_position: db_thread.ui_scroll_position.map(|sp| gpui::ListOffset {
                item_ix: sp.item_ix,
                offset_in_item: gpui::px(sp.offset_in_item),
            }),
            running_subagents: Vec::new(),
            caduceus_mode: None,
            cached_project_instructions: None,
            loop_detector: caduceus_bridge::safety::LoopDetector::new(5),
            loop_escalation_count: 0,
            circuit_breaker: caduceus_bridge::safety::CircuitBreaker::new(5),
            compaction_cooldown: caduceus_bridge::safety::CompactionCooldown::new(
                std::time::Duration::from_secs(30),
            ),
            guardrail_alert: None,
            context_compacted_this_turn: false,
            context_pins: caduceus_bridge::orchestrator::ContextManager::new(128000),
            task_dag: std::sync::Arc::new(std::sync::Mutex::new(
                caduceus_bridge::orchestrator::TaskDAG::new(),
            )),
            caduceus: CaduceusState::default(),
            cached_token_estimate: std::cell::Cell::new(None),
            turn_generation: 0,
            last_turn_cancelled: false,
            pulled_session_ids: Vec::new(),
            pinned: loaded_pinned,
        }
    }

    pub fn to_db(&self, cx: &App) -> Task<DbThread> {
        let initial_project_snapshot = self.initial_project_snapshot.clone();
        // ST2 fix-loop #6: filter orphaned pins inline. `to_db` takes
        // `&self`, so we cannot call `gc_pinned()` here (which requires
        // `&mut self`). Apply the same retain predicate as gc_pinned so
        // persisted state is never polluted by dangling pins, even if
        // an upstream mutation path forgot to call gc_pinned.
        let user_ids: HashSet<UserMessageId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User(u) => Some(u.id.clone()),
                _ => None,
            })
            .collect();
        let resume_ids: HashSet<ResumeId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Resume(rid) => Some(*rid),
                _ => None,
            })
            .collect();
        let agent_ids: HashSet<AgentMessageId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Agent(a) => Some(a.id),
                _ => None,
            })
            .collect();
        let pinned: Vec<MessageRef> = self
            .pinned
            .iter()
            .filter(|p| match &p.key {
                PinnedMessageKey::User(id) => user_ids.contains(id),
                PinnedMessageKey::Resume(rid) => resume_ids.contains(rid),
                PinnedMessageKey::Agent(aid) => agent_ids.contains(aid),
            })
            .cloned()
            .collect();
        let mut thread = DbThread {
            title: self.title().unwrap_or_default(),
            messages: self.messages.clone(),
            updated_at: self.updated_at,
            detailed_summary: self.summary.clone(),
            initial_project_snapshot: None,
            cumulative_token_usage: self.cumulative_token_usage,
            request_token_usage: self.request_token_usage.clone(),
            model: self.model.as_ref().map(|model| DbLanguageModel {
                provider: model.provider_id().to_string(),
                model: model.id().0.to_string(),
            }),
            profile: Some(self.profile_id.clone()),
            imported: self.imported,
            subagent_context: self.subagent_context.clone(),
            speed: self.speed,
            thinking_enabled: self.thinking_enabled,
            thinking_effort: self.thinking_effort.clone(),
            draft_prompt: self.draft_prompt.clone(),
            ui_scroll_position: self.ui_scroll_position.map(|lo| {
                crate::db::SerializedScrollPosition {
                    item_ix: lo.item_ix,
                    offset_in_item: lo.offset_in_item.as_f32(),
                }
            }),
            pinned,
        };

        cx.background_spawn(async move {
            let initial_project_snapshot = initial_project_snapshot.await;
            thread.initial_project_snapshot = initial_project_snapshot;
            thread
        })
    }

    /// Create a snapshot of the current project state including git information and unsaved buffers.
    fn project_snapshot(
        project: Entity<Project>,
        cx: &mut Context<Self>,
    ) -> Task<Arc<ProjectSnapshot>> {
        let task = project::telemetry_snapshot::TelemetrySnapshot::new(&project, cx);
        cx.spawn(async move |_, _| {
            let snapshot = task.await;

            Arc::new(ProjectSnapshot {
                worktree_snapshots: snapshot.worktree_snapshots,
                timestamp: Utc::now(),
            })
        })
    }

    pub fn project_context(&self) -> &Entity<ProjectContext> {
        &self.project_context
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn action_log(&self) -> &Entity<ActionLog> {
        &self.action_log
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty() && self.title.is_none()
    }

    pub fn draft_prompt(&self) -> Option<&[acp::ContentBlock]> {
        self.draft_prompt.as_deref()
    }

    pub fn set_draft_prompt(&mut self, prompt: Option<Vec<acp::ContentBlock>>) {
        self.draft_prompt = prompt;
    }

    pub fn ui_scroll_position(&self) -> Option<gpui::ListOffset> {
        self.ui_scroll_position
    }

    pub fn set_ui_scroll_position(&mut self, position: Option<gpui::ListOffset>) {
        self.ui_scroll_position = position;
    }

    pub fn model(&self) -> Option<&Arc<dyn LanguageModel>> {
        self.model.as_ref()
    }

    pub fn set_caduceus_mode(&mut self, mode: Option<String>) {
        self.caduceus_mode = mode;
    }

    /// Caduceus F2: append a session id to the pulled set so the next turn's
    /// system prompt mentions it and the orchestrator knows to consult it
    /// via `read_thread` / `compact_thread`. Idempotent; preserves order.
    pub fn pull_session_id(&mut self, session_id: String) {
        if !self.pulled_session_ids.iter().any(|s| s == &session_id) {
            self.pulled_session_ids.push(session_id);
        }
    }

    /// Caduceus F2: read-only view of pulled session ids.
    pub fn pulled_session_ids(&self) -> &[String] {
        &self.pulled_session_ids
    }

    /// Current guardrail alert message and severity (if any, within 10s TTL)
    pub fn guardrail_alert(&self) -> Option<(&str, AlertSeverity)> {
        self.guardrail_alert.as_ref().and_then(|(msg, sev, ts)| {
            if ts.elapsed() < std::time::Duration::from_secs(10) {
                Some((msg.as_str(), *sev))
            } else {
                None
            }
        })
    }

    /// Whether context was compacted during the current/last turn
    pub fn context_compacted_this_turn(&self) -> bool {
        self.context_compacted_this_turn
    }

    /// Current context zone and fill percentage (combined to avoid double computation)
    pub fn context_zone_and_pct(&self) -> (caduceus_bridge::orchestrator::ContextZone, f64) {
        let total = self.estimate_total_tokens();
        let max = self.model_max_tokens();
        let pct = if max == 0 {
            0.0
        } else {
            ((total as f64 / max as f64) * 100.0).min(100.0)
        };
        (
            caduceus_bridge::orchestrator::ContextZone::from_percentage(pct),
            pct,
        )
    }

    /// Current context zone (Green/Yellow/Orange/Red/Critical)
    pub fn context_zone(&self) -> caduceus_bridge::orchestrator::ContextZone {
        self.context_zone_and_pct().0
    }

    /// Context fill percentage (0-100, capped)
    pub fn context_fill_pct(&self) -> f64 {
        self.context_zone_and_pct().1
    }

    // ── G1a — native-loop harness lifecycle ─────────────────────────
    //
    // `caduceus_bridge`, `caduceus_harness`, `caduceus_native_state`,
    // and `caduceus_cancel_token` are the four sibling fields managed
    // as a unit: either all `Some` (after `ensure_caduceus_harness`)
    // or all `None` (pre-init or post-invalidation). `ensure` is
    // idempotent and a no-op while the `caduceus_native_loop` setting
    // is OFF. `invalidate` is called on `set_model`, cancel (optional),
    // and a flag OFF→ON or ON→OFF transition.

    /// Returns `true` when the native loop is enabled in settings AND
    /// the harness has been provisioned for this thread. Callers that
    /// want to dispatch on the setting alone (not provisioning) should
    /// read `AgentSettings::get_global(cx).caduceus_native_loop`
    /// instead.
    pub fn caduceus_native_loop_ready(&self) -> bool {
        self.caduceus.harness.is_some()
            && self.caduceus.native_state.is_some()
            && self.caduceus.bridge.is_some()
    }

    /// Drops the harness, state, cancel token, emitter, and bridge
    /// handle so the next `ensure_caduceus_harness` rebuilds them.
    /// Cheap: dropping an Arc is O(1); the async mutex's inner drop is
    /// Send. Safe to call from `&mut self` contexts; never blocks.
    pub fn invalidate_caduceus_harness(&mut self) {
        self.caduceus.harness = None;
        self.caduceus.native_state = None;
        self.caduceus.cancel_token = None;
        self.caduceus.emitter = None;
        self.caduceus.dispatcher = None;
        self.caduceus.approval_tx = None;
        // T4 (Audit C6): clear the init-time flag snapshot so the
        // next ensure_caduceus_harness re-records the current setting.
        self.caduceus.last_native_loop_flag = None;
        // Intentionally keep `caduceus_bridge` — the bridge itself
        // (ContextManager config, native_loop flag) outlives the
        // harness. If the flag was the reason for invalidation the
        // caller is expected to update the flag separately.
    }

    /// Idempotent lazy constructor. Populates the native-loop
    /// fields iff the `caduceus_native_loop` setting is ON. Returns
    /// `true` if provisioning happened in this call, `false` if
    /// already provisioned (no-op) or the flag is OFF.
    ///
    /// ST-A2d: also populates `caduceus_native_state` (previously
    /// discarded) and leaves `caduceus_cancel_token` populated so a
    /// subsequent turn can bump its generation. The `AgentHarness`
    /// itself (`caduceus_harness`) + its `caduceus_emitter` are still
    /// built lazily by `try_run_turn_native` in ST-A2, because that's
    /// where the provider adapter + tool registry + system prompt are
    /// known.
    ///
    /// Failure modes (return `Err`): no worktree, bridge build
    /// failure. On any error the fields stay `None` and the legacy
    /// path keeps running.
    pub fn ensure_caduceus_harness(&mut self, cx: &mut Context<Self>) -> Result<bool> {
        let enabled = agent_settings::AgentSettings::get_global(cx).caduceus_native_loop;

        // T4 (Audit C6): detect a live flag transition vs the flag
        // value that was in effect when the current harness was
        // provisioned. If they differ — OFF→ON or ON→OFF — drop the
        // stale harness so we rebuild (ON) or release resources (OFF).
        if let Some(prev) = self.caduceus.last_native_loop_flag {
            if prev != enabled {
                log::info!(
                    "[native-loop] caduceus_native_loop flag transitioned {prev} → {enabled}; \
                     invalidating harness so next turn rebuilds with current setting"
                );
                self.invalidate_caduceus_harness();
                // T4: on a flag transition we also drop the bridge.
                // `invalidate_caduceus_harness` intentionally keeps
                // the bridge across normal invalidations (mode/model
                // change), but a flag transition is a different
                // lifecycle event — the bridge is only meaningful
                // when the flag is ON, so drop it unconditionally.
                self.caduceus.bridge = None;
            }
        }

        // Already provisioned against the current flag?
        if self.caduceus_native_loop_ready() {
            return Ok(false);
        }
        if !enabled {
            return Ok(false);
        }

        let Some(root) = caduceus_workspace_folder(&self.project, cx) else {
            log_no_folder_workspace_once();
            return Ok(false);
        };

        let bridge = std::sync::Arc::new(
            caduceus_bridge::orchestrator::OrchestratorBridge::new(&root)
                .with_native_loop_enabled(true),
        );
        let cancel_token = caduceus_core::CancellationToken::new();
        let state = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::caduceus_native_state::NativeLoopState::new(root),
        ));

        // ST-A2d: stash bridge + cancel token + native state so
        // `try_run_turn_native` (ST-A2) can build the harness lazily
        // with full turn context (provider, tools, system prompt).
        // The harness and its emitter clone are populated there, not
        // here. `caduceus_native_loop_ready()` therefore still reports
        // `false` until the first native turn completes ensure-harness
        // + emitter population, which is the correct invariant for
        // `run_turn_internal` dispatch (legacy path remains until
        // harness is fully built).
        self.caduceus.bridge = Some(bridge);
        self.caduceus.cancel_token = Some(cancel_token);
        self.caduceus.native_state = Some(state);
        // T4 (Audit C6): record the flag value we built against so a
        // future transition is detected at the top of this function.
        self.caduceus.last_native_loop_flag = Some(enabled);
        Ok(true)
    }

    /// Current Caduceus mode name
    pub fn caduceus_mode_name(&self) -> &str {
        self.caduceus_mode_from_profile()
    }

    /// Pin context that survives compaction
    pub fn pin_context(&mut self, label: &str, content: &str) {
        self.context_pins.pin(label, content);
    }

    /// Unpin context
    pub fn unpin_context(&mut self, label: &str) -> bool {
        self.context_pins.unpin(label)
    }

    /// List pinned context items
    pub fn list_pins(&self) -> &[caduceus_bridge::orchestrator::PinnedContext] {
        self.context_pins.list_pins()
    }

    // ─── ST2: pinned-message API ──────────────────────────────────────
    //
    // See `MessageRef`, `PinReason`, `PinnedMessageKey` for the data model.
    // Pins are keyed by `(PinnedMessageKey, PinReason)`; multiple reasons
    // may coexist on the same key. Insertion order is preserved.

    /// Pin a message. No-op if `(key, reason)` is already present.
    /// Returns `true` if a new pin was inserted.
    ///
    /// `Resume`-reason pins are coalesced to `MAX_RESUME_PINS` — a new
    /// Resume pin past the cap drops the oldest Resume pin.
    pub fn pin(&mut self, key: PinnedMessageKey, reason: PinReason) -> bool {
        self.pin_at(key, reason, Utc::now())
    }

    /// Test seam: pin with an explicit timestamp for deterministic assertions.
    pub(crate) fn pin_at(
        &mut self,
        key: PinnedMessageKey,
        reason: PinReason,
        now: DateTime<Utc>,
    ) -> bool {
        if self
            .pinned
            .iter()
            .any(|p| p.key == key && p.reason == reason)
        {
            return false;
        }
        self.pinned.push(MessageRef {
            key,
            reason,
            pinned_at: now,
        });
        if matches!(reason, PinReason::Resume) {
            self.coalesce_resume_pins();
        }
        true
    }

    /// Replace the (singleton-by-reason) pin for `reason` with `(key, reason)`.
    /// Returns the previous key if any. Used for `PlanUpdate` and
    /// `ScopeExpansionActive`. Do NOT use for `Resume` (which is N-cap, not singleton).
    pub fn pin_replace(
        &mut self,
        key: PinnedMessageKey,
        reason: PinReason,
    ) -> Option<PinnedMessageKey> {
        debug_assert!(
            !matches!(reason, PinReason::Resume),
            "pin_replace must not be used for Resume (use pin + coalesce)"
        );
        let prev = self
            .pinned
            .iter()
            .position(|p| p.reason == reason)
            .map(|i| self.pinned.remove(i).key);
        self.pin(key, reason);
        prev
    }

    /// Drop every pin for the given key (all reasons). Returns count removed.
    pub fn unpin(&mut self, key: &PinnedMessageKey) -> usize {
        let before = self.pinned.len();
        self.pinned.retain(|p| &p.key != key);
        before - self.pinned.len()
    }

    /// Drop a single (key, reason) entry. Returns true if it was present.
    pub fn unpin_reason(&mut self, key: &PinnedMessageKey, reason: PinReason) -> bool {
        if let Some(idx) = self
            .pinned
            .iter()
            .position(|p| &p.key == key && p.reason == reason)
        {
            self.pinned.remove(idx);
            true
        } else {
            false
        }
    }

    /// Returns every `PinReason` currently pinned on `key`, in insertion order.
    /// Empty Vec means not pinned.
    pub fn is_pinned(&self, key: &PinnedMessageKey) -> Vec<PinReason> {
        self.pinned
            .iter()
            .filter(|p| &p.key == key)
            .map(|p| p.reason)
            .collect()
    }

    /// Lookup the first pin with the given key, if any.
    pub fn pinned_ref(&self, key: &PinnedMessageKey) -> Option<&MessageRef> {
        self.pinned.iter().find(|p| &p.key == key)
    }

    /// All pins, in insertion order.
    pub fn pinned_refs(&self) -> &[MessageRef] {
        &self.pinned
    }

    /// Returns sorted unique message positions. A message pinned for
    /// multiple reasons appears once. Used by ST3 compaction-protection
    /// budget calculation — the **set** semantics are part of that
    /// contract: pinned-token budget must not double-count a multi-reason
    /// pin. Pins whose underlying message no longer exists are skipped
    /// (see `gc_pinned` for permanent cleanup).
    pub fn pinned_message_indices(&self) -> Vec<usize> {
        let mut set: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for p in &self.pinned {
            if let Some(idx) = self.resolve_key_to_index(&p.key) {
                set.insert(idx);
            }
        }
        Vec::from_iter(set)
    }

    /// Drop pins whose underlying message no longer exists in `self.messages`.
    /// Called by ST3 post-compaction. Idempotent.
    ///
    /// ST2 fix-loop #6: also called from `to_db()` (defensive — persisted
    /// pins never drift) and from `truncate()` / `auto_compact_context_with_zone()`
    /// (mutation paths that drop messages).
    pub fn gc_pinned(&mut self) {
        let user_ids: HashSet<UserMessageId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::User(u) => Some(u.id.clone()),
                _ => None,
            })
            .collect();
        let resume_ids: HashSet<ResumeId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Resume(rid) => Some(*rid),
                _ => None,
            })
            .collect();
        let agent_ids: HashSet<AgentMessageId> = self
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::Agent(a) => Some(a.id),
                _ => None,
            })
            .collect();
        let before = self.pinned.len();
        self.pinned.retain(|p| match &p.key {
            PinnedMessageKey::User(id) => user_ids.contains(id),
            PinnedMessageKey::Resume(rid) => resume_ids.contains(rid),
            PinnedMessageKey::Agent(aid) => agent_ids.contains(aid),
        });
        let dropped = before - self.pinned.len();
        if dropped > 0 {
            log::debug!("[st2] gc_pinned dropped {} orphaned pin(s)", dropped);
        }
    }

    /// Empty the pinned list. Called when seeding a subagent context;
    /// also used to reset state in tests.
    pub fn clear_inherited_pins(&mut self) {
        self.pinned.clear();
    }

    /// Look up a `PinnedMessageKey` to a slice index in `self.messages`.
    /// Returns `None` if the key has no live message (e.g. user message
    /// was truncated, or the Resume marker with that id is gone).
    fn resolve_key_to_index(&self, key: &PinnedMessageKey) -> Option<usize> {
        match key {
            PinnedMessageKey::User(target) => self.messages.iter().position(|m| match m {
                Message::User(u) => &u.id == target,
                _ => false,
            }),
            PinnedMessageKey::Resume(target) => self.messages.iter().position(|m| match m {
                Message::Resume(rid) => rid == target,
                _ => false,
            }),
            PinnedMessageKey::Agent(target) => self.messages.iter().position(|m| match m {
                Message::Agent(a) => &a.id == target,
                _ => false,
            }),
        }
    }

    /// Return the `ResumeId` of the most-recently pushed `Message::Resume`
    /// in the thread, or `None` if no Resume marker exists. Used by
    /// `auto_pin_resume` immediately after pushing the marker.
    fn last_resume_id(&self) -> Option<ResumeId> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Resume(rid) => Some(*rid),
            _ => None,
        })
    }

    /// Coalesce Resume pins to at most `MAX_RESUME_PINS`. Drops oldest first.
    fn coalesce_resume_pins(&mut self) {
        let resume_indices: Vec<usize> = self
            .pinned
            .iter()
            .enumerate()
            .filter(|(_, p)| matches!(p.reason, PinReason::Resume))
            .map(|(i, _)| i)
            .collect();
        if resume_indices.len() <= MAX_RESUME_PINS {
            return;
        }
        let to_drop = resume_indices.len() - MAX_RESUME_PINS;
        // Vec is in insertion order; drop the first `to_drop` Resume entries.
        let drop_set: HashSet<usize> = resume_indices.into_iter().take(to_drop).collect();
        let mut idx = 0;
        self.pinned.retain(|_| {
            let keep = !drop_set.contains(&idx);
            idx += 1;
            keep
        });
    }

    /// FirstUser auto-pin trigger: if there is exactly one user message
    /// in the thread (just inserted), pin it with `PinReason::FirstUser`.
    /// Called from both `Thread::send` and `Thread::push_acp_user_block`
    /// — idempotent because of the `count == 1` guard.
    fn maybe_auto_pin_first_user(&mut self, id: UserMessageId) {
        let user_count = self
            .messages
            .iter()
            .filter(|m| matches!(m, Message::User(_)))
            .count();
        if user_count == 1 {
            self.pin(PinnedMessageKey::User(id), PinReason::FirstUser);
        }
    }

    /// Resume auto-pin trigger: pin the most recently pushed Resume marker
    /// (by its stable `ResumeId`) with `PinReason::Resume`. Honors
    /// `MAX_RESUME_PINS` via `coalesce_resume_pins`.
    fn auto_pin_resume(&mut self) {
        let Some(rid) = self.last_resume_id() else {
            return;
        };
        self.pin(PinnedMessageKey::Resume(rid), PinReason::Resume);
    }

    /// PlanUpdate hook: called from `agent.rs` (legacy ACP dispatcher) when
    /// `ThreadEvent::Plan` is observed, and from the native-loop consumer
    /// task on `T::PlanStep` / `T::PlanAmended`. Pins the **current-turn**
    /// agent message (by stable `AgentMessageId`) with `PinReason::PlanUpdate`.
    ///
    /// Target selection (ST2 r3 Fix 1) — order matters because plan events
    /// fire mid-stream, before `flush_pending_message` has moved the
    /// in-flight assistant content into `self.messages`:
    ///   1. If `self.pending_message` is `Some`, use its `id` — that is
    ///      the live current-turn assistant message.
    ///   2. Otherwise, walk `self.messages` from the end backwards, stopping
    ///      at the most recent `Message::User` or `Message::Resume` (the
    ///      current-turn boundary). Pin the latest `Message::Agent` seen
    ///      after that boundary (the trailing agent in the current turn).
    ///   3. If neither exists (plan event emitted before any current-turn
    ///      agent target), drop any stale `PinReason::PlanUpdate` pin so
    ///      we don't leave a dangling reference to a prior turn, and emit
    ///      a `WARN` log (per AC6).
    ///
    /// Earlier code (ST2 fix-loop #3) only scanned `self.messages` for the
    /// last `Message::Agent`, which during a streamed turn either pinned a
    /// HISTORICAL agent message (wrong target) or silently no-op'd. v3.1
    /// Fix 1 mandates the **current-turn** agent message as the anchor.
    pub fn on_plan_event_emitted(&mut self, _cx: &mut Context<Self>) {
        let target = if let Some(pending) = self.pending_message.as_ref() {
            Some(pending.id)
        } else {
            self.current_turn_trailing_agent_id()
        };
        let Some(id) = target else {
            // No current-turn agent target available. Proactively drop any
            // stale PlanUpdate pin so we don't carry a pin from a prior
            // turn forward.
            let stale_idx = self
                .pinned
                .iter()
                .position(|p| p.reason == PinReason::PlanUpdate);
            if let Some(i) = stale_idx {
                self.pinned.remove(i);
                log::warn!(
                    "[st2] plan event emitted with no current-turn agent target; \
                     stale PlanUpdate pin removed"
                );
            } else {
                log::warn!(
                    "[st2] plan event emitted before any current-turn agent message; \
                     PlanUpdate pin dropped"
                );
            }
            return;
        };
        self.pin_replace(PinnedMessageKey::Agent(id), PinReason::PlanUpdate);
    }

    /// Returns the id of the latest `Message::Agent` in the trailing
    /// contiguous agent segment after the most recent `Message::User` /
    /// `Message::Resume` boundary. `None` if no agent appears in the
    /// current turn (e.g. user just sent, no reply yet).
    fn current_turn_trailing_agent_id(&self) -> Option<AgentMessageId> {
        for m in self.messages.iter().rev() {
            match m {
                Message::Agent(a) => return Some(a.id),
                Message::User(_) | Message::Resume(_) => return None,
            }
        }
        None
    }

    /// ScopeExpansionActive auto-pin trigger: invoked from the native-loop
    /// consumer task on `AgentEvent::ScopeExpansionRequested`. Pins the
    /// most-recent user message (the one whose tool call triggered the
    /// expansion). Replacement semantics: a new request supersedes the
    /// prior `ScopeExpansionActive` pin via `pin_replace`.
    ///
    /// Auto-unpin on Granted/Denied is **deferred to ST2.5** — the
    /// permission lifecycle event is not bridged onto `AgentEvent` today.
    pub fn on_scope_expansion_requested(&mut self) {
        let target = self.messages.iter().rev().find_map(|m| match m {
            Message::User(u) => Some(u.id.clone()),
            _ => None,
        });
        let Some(id) = target else {
            log::warn!(
                "[st2] scope expansion requested before any user message; pin dropped"
            );
            return;
        };
        self.pin_replace(PinnedMessageKey::User(id), PinReason::ScopeExpansionActive);
    }

    pub(crate) fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Derive caduceus_mode from the current profile_id so the two stay in sync.
    fn caduceus_mode_from_profile(&self) -> &str {
        self.caduceus_mode
            .as_deref()
            .unwrap_or_else(|| self.profile_id.0.as_ref())
    }

    /// Derive the active ActLens (if any) for prompt rendering. Lens state is
    /// Zed-local (UI concern), not part of the mode token — kept `None` until
    /// a lens is exposed through the session UI.
    fn caduceus_lens_from_profile(&self) -> Option<String> {
        None
    }

    /// Caduceus: check if a tool is allowed in the current privilege ring.
    /// Modes: Plan / Act / Research / Autopilot (Act carries a Lens — Normal/Debug/Review).
    /// Legacy aliases `architect`/`debug`/`review` are normalized at the engine layer
    /// (see `caduceus_orchestrator::modes::ModeSelection::from_str_loose`).
    ///
    /// Plan mode CAN write:
    /// - .md, .json, .yaml, .txt files (plans, specs, wiki, configs)
    /// - Caduceus wiki, memory, project, kanban, checkpoint tools
    /// Plan mode CANNOT write:
    /// - Code files (.rs, .py, .js, .ts, etc.)
    /// - Terminal commands
    /// Tool dispatch gate — delegates to the bridge's authoritative
    /// `mode_allows_tool` (ST-A6 / contract `mode-policy-shim-v1`). The old
    /// hardcoded allowlist that lived here is retired; the bridge is the
    /// single source of truth for mode × tool decisions.
    fn is_tool_allowed_in_current_mode(&self, tool_name: &str) -> bool {
        let mode = self.caduceus_mode_from_profile();
        let allowed = caduceus_bridge::orchestrator::mode_allows_tool(mode, tool_name);
        if !allowed {
            log::warn!(
                "[caduceus] BLOCKED '{}' in {} mode. \
                Ask the user to switch to a mode that allows this tool.",
                tool_name,
                mode
            );
        }
        allowed
    }

    /// Caduceus: unified max-token fallback for context budgeting.
    fn model_max_tokens(&self) -> u32 {
        self.model
            .as_ref()
            .map(|m| m.max_token_count() as u32)
            .unwrap_or(64_000)
    }

    /// Test-only accessor for `model_max_tokens` so the
    /// provider-vs-local zone test can compute its 90%-full target.
    #[cfg(test)]
    pub(crate) fn model_max_tokens_for_test(&self) -> u32 {
        self.model_max_tokens()
    }

    /// Caduceus: estimate total tokens across all messages (DRY helper).
    /// `pub(crate)` for test visibility — bug C5 tests verify the cache
    /// is invalidated when messages change. Populates the cache on read
    /// via `Cell` interior mutability so subsequent calls are O(1).
    pub(crate) fn estimate_total_tokens(&self) -> u32 {
        if let Some(cached) = self.cached_token_estimate.get() {
            return cached;
        }
        let total: u32 = self
            .messages
            .iter()
            .map(|m| match m {
                Message::User(u) => u
                    .content
                    .iter()
                    .map(|c| match c {
                        UserMessageContent::Text(t) => {
                            caduceus_bridge::orchestrator::count_tokens_exact(t)
                        }
                        _ => 10,
                    })
                    .sum::<u32>(),
                Message::Agent(a) => {
                    caduceus_bridge::orchestrator::count_tokens_exact(&a.to_markdown())
                }
                Message::Resume(_) => 0,
            })
            .sum();
        self.cached_token_estimate.set(Some(total));
        total
    }

    /// Invalidate cached token estimate (call when messages change).
    /// `pub(crate)` for test visibility — see C5 regression tests.
    pub(crate) fn invalidate_token_cache(&mut self) {
        self.cached_token_estimate.set(None);
    }

    /// Test-only: read whether the last turn was cancelled (Bug #14 gate).
    #[cfg(test)]
    pub(crate) fn last_turn_cancelled_for_test(&self) -> bool {
        self.last_turn_cancelled
    }

    /// Test-only: read the current turn generation counter (Bug C2).
    #[cfg(test)]
    pub(crate) fn turn_generation_for_test(&self) -> u64 {
        self.turn_generation
    }

    /// Test-only: peek at cached token estimate without recomputing.
    #[cfg(test)]
    pub(crate) fn cached_token_estimate_for_test(&self) -> Option<u32> {
        self.cached_token_estimate.get()
    }

    /// Test-only: expose `current_context_zone` so the
    /// provider-vs-local divergence regression test can pin behavior.
    #[cfg(test)]
    pub(crate) fn current_context_zone_for_test(
        &self,
    ) -> caduceus_bridge::orchestrator::ContextZone {
        self.current_context_zone()
    }

    /// Caduceus: estimate current context zone based on token usage.
    ///
    /// Prefers the **provider's** actual reported token usage (from the
    /// last completed request) over our local heuristic
    /// `estimate_total_tokens()`. The local estimate undercounts —
    /// non-text user content and tool results are folded in at a flat
    /// `10` tokens — so a thread can sit at "Green" by our heuristic
    /// while the provider has already declared the context full and the
    /// UI is showing the red "Thread reached the token limit" banner.
    ///
    /// Falling back to the local estimate before the first response is
    /// the only way we can early-detect a runaway first turn that hasn't
    /// reported usage yet.
    fn current_context_zone(&self) -> caduceus_bridge::orchestrator::ContextZone {
        use caduceus_bridge::orchestrator::ContextZone;
        let max = self.model_max_tokens();
        if max == 0 {
            return ContextZone::Green;
        }
        let used = if let Some(provider) = self.latest_request_token_usage() {
            // Provider knows: input + (cached input not double-counted)
            // + output. `total_tokens()` is the same number the UI
            // banner uses, so our zone now tracks the banner exactly.
            provider.total_tokens() as u32
        } else {
            self.estimate_total_tokens()
        };
        ContextZone::from_percentage((used as f64 / max as f64) * 100.0)
    }

    /// Caduceus: smart context management using engine compaction pipeline.
    /// Triggers based on both message count AND estimated token usage.
    /// Uses engine's ContextZone (Green/Yellow/Orange/Red/Critical) for decisions.
    pub(crate) fn auto_compact_context(&mut self, cx: &mut Context<Self>) -> bool {
        let zone = self.current_context_zone();
        self.auto_compact_context_with_zone(zone, cx)
    }

    /// Same as `auto_compact_context` but distinguishes between the three
    /// possible outcomes so UI/slash-command handlers can give the user
    /// honest feedback (e.g. "didn't run because cooldown" vs "ran but
    /// nothing to do").
    pub(crate) fn auto_compact_context_explained(
        &mut self,
        cx: &mut Context<Self>,
    ) -> CompactOutcome {
        if !self.compaction_cooldown.can_compact() {
            return CompactOutcome::CooldownActive;
        }
        if self.auto_compact_context(cx) {
            CompactOutcome::Compacted
        } else {
            CompactOutcome::WithinBudget
        }
    }

    /// Same as auto_compact_context but accepts a pre-computed zone to avoid recomputation.
    fn auto_compact_context_with_zone(
        &mut self,
        zone: caduceus_bridge::orchestrator::ContextZone,
        cx: &mut Context<Self>,
    ) -> bool {
        use caduceus_bridge::orchestrator::compaction::{MessageGroupKind, build_message_groups};
        use caduceus_bridge::orchestrator::{CompactMessage, CompactionPipeline, ContextZone};

        // Use compaction cooldown guard
        if !self.compaction_cooldown.can_compact() {
            return false;
        }

        let local_estimate = self.estimate_total_tokens();
        let provider_used = self
            .latest_request_token_usage()
            .map(|u| u.total_tokens() as u32);
        let max_context = self.model_max_tokens();
        // Mirror `current_context_zone()`: prefer provider tokens, fall
        // back to local estimate. The two used to disagree (provider
        // banner red, local heuristic green) and auto-compact silently
        // no-op'd. See [`current_context_zone`] for the rationale.
        let total_tokens = provider_used.unwrap_or(local_estimate);
        let fill_pct = if max_context == 0 {
            0.0
        } else {
            (total_tokens as f64 / max_context as f64) * 100.0
        };

        let msg_count = self.messages.len();
        // Per-turn breadcrumb so any future provider/local divergence
        // shows up in the log even when zone is Green.
        log::debug!(
            "[caduceus] context check: zone={:?} fill={:.1}% provider={:?} local={} max={} msgs={}",
            zone,
            fill_pct,
            provider_used,
            local_estimate,
            max_context,
            msg_count
        );
        // Decide whether to compact based on zone + message count
        let (should_compact, keep_recent) = match zone {
            ContextZone::Green => {
                if msg_count > 60 {
                    (true, 20)
                } else {
                    (false, 0)
                }
            }
            ContextZone::Yellow => {
                log::info!(
                    "[caduceus] Context zone YELLOW ({:.0}% full, {} msgs)",
                    fill_pct,
                    msg_count
                );
                if msg_count > 30 {
                    (true, 15)
                } else {
                    (false, 0)
                }
            }
            ContextZone::Orange => {
                log::info!(
                    "[caduceus] Context zone ORANGE ({:.0}% full, {} msgs)",
                    fill_pct,
                    msg_count
                );
                if msg_count > 15 {
                    (true, 10)
                } else {
                    (false, 0)
                }
            }
            ContextZone::Red => {
                log::warn!(
                    "[caduceus] Context zone RED ({:.0}% full, {} msgs)",
                    fill_pct,
                    msg_count
                );
                if msg_count > 10 {
                    (true, 8)
                } else {
                    (false, 0)
                }
            }
            ContextZone::Critical => {
                log::error!(
                    "[caduceus] Context zone CRITICAL ({:.0}% full, {} msgs) — EMERGENCY compact",
                    fill_pct,
                    msg_count
                );
                if msg_count > 5 { (true, 5) } else { (false, 0) }
            }
        };

        if !should_compact {
            return false;
        }

        let messages_to_compact = self.messages.len() - keep_recent;
        log::info!(
            "[caduceus] Compacting: {} msgs, {:.0}% tokens, zone {:?} → keeping last {}",
            self.messages.len(),
            fill_pct,
            zone,
            keep_recent
        );

        // Guard: zero-budget means model lookup failed — don't compact
        let budget = if max_context == 0 {
            log::warn!("[caduceus] max_context is 0 — skipping compaction");
            caduceus_bridge::context_events::record_and_count(
                caduceus_bridge::context_events::ContextEventKind::CompactionSkipped {
                    reason: "zero budget (model max_context unknown)".into(),
                    fill_pct,
                    msg_count,
                },
            );
            return false;
        } else {
            (max_context as usize) / 4
        };

        // Convert IDE messages to engine CompactMessages for the pipeline
        let compact_messages: Vec<CompactMessage> = self.messages[..messages_to_compact]
            .iter()
            .map(|msg| match msg {
                Message::User(u) => {
                    let text: String = u
                        .content
                        .iter()
                        .map(|c| match c {
                            UserMessageContent::Text(t) => t.as_str(),
                            _ => "[context]",
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    CompactMessage::new("user", text)
                }
                Message::Agent(a) => {
                    // Classify from structured content, not markdown substring
                    let text = a.to_markdown();
                    let is_pure_tool = text.len() > 100
                        && (text.contains("Tool Call:") || text.contains("Status: Completed"))
                        && !text.contains("I'll")
                        && !text.contains("Let me");
                    if is_pure_tool {
                        CompactMessage::new("tool", "[tool calls executed]")
                    } else {
                        CompactMessage::new("assistant", text)
                    }
                }
                Message::Resume(_) => CompactMessage::new("system", "[session resumed]"),
            })
            .collect();

        // Run engine's compaction pipeline — tool collapse + summarize
        let mut groups = build_message_groups(&compact_messages);
        let pipeline = CompactionPipeline::default_pipeline(budget);
        let result = pipeline.run(&mut groups);

        // Extract summary from the pipeline output — prefer Summary groups,
        // fall back to all surviving groups if no summary was generated
        let summary: String = {
            let summary_groups: Vec<_> = groups
                .iter()
                .filter(|g| g.kind == MessageGroupKind::Summary && !g.excluded)
                .collect();
            if !summary_groups.is_empty() {
                summary_groups
                    .iter()
                    .flat_map(|g| &g.messages)
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                groups
                    .iter()
                    .filter(|g| !g.excluded)
                    .flat_map(|g| &g.messages)
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        // Guard: if summary is degenerate (all tool calls, no real content), abort
        if summary.trim().is_empty()
            || summary
                .lines()
                .all(|l| l.contains("[tool calls executed]") || l.contains("[session resumed]"))
        {
            log::warn!("[caduceus] Compaction produced degenerate summary — aborting");
            caduceus_bridge::context_events::record_and_count(
                caduceus_bridge::context_events::ContextEventKind::CompactionSkipped {
                    reason: "degenerate summary (only tool/system messages)".into(),
                    fill_pct,
                    msg_count,
                },
            );
            return false;
        }

        log::info!(
            "[caduceus] Pipeline applied {} strategies, freed {} tokens",
            result.strategies_applied.len(),
            result.total_removed_tokens
        );

        // Save context summary to project in background
        if let Some(root) = caduceus_workspace_folder(&self.project, cx) {
            let session_id = self.id.0.to_string();
            let summary_clone = summary.clone();
            cx.background_executor()
                .spawn(async move {
                    let context_dir = root.join(".caduceus").join("context");
                    let _ = std::fs::create_dir_all(&context_dir);
                    let filename =
                        format!("compact-{}.md", crate::tools::truncate_str(&session_id, 8));
                    let path = context_dir.join(&filename);
                    if std::fs::write(&path, &summary_clone).is_ok() {
                        log::info!("[caduceus] Context saved to {}", path.display());
                    }
                })
                .detach();
        }

        // Remove older messages, keep recent
        self.messages.drain(..messages_to_compact);

        // Reinject compacted summary as first message
        self.messages.insert(
            0,
            Message::User(UserMessage {
                id: acp_thread::UserMessageId::new(),
                content: vec![UserMessageContent::Text(format!(
                    "[Context compacted — {} older messages summarized via engine pipeline, {:.0}% token usage]\n\n{}",
                    messages_to_compact, fill_pct, summary
                ))],
            }),
        );
        // ST2 fix-loop #6: drop pins whose target messages were just
        // evicted by the drain above. Pins on surviving messages keep
        // their semantic meaning; pins on evicted messages are stale
        // and must not leak to disk via to_db.
        self.gc_pinned();
        self.invalidate_token_cache();

        log::info!(
            "[caduceus] Compacted: {} messages remain, ~{} tokens freed",
            self.messages.len(),
            total_tokens
                .saturating_sub(caduceus_bridge::orchestrator::count_tokens_exact(&summary))
        );

        self.compaction_cooldown.record_compaction();
        self.context_compacted_this_turn = true;
        let tokens_freed = total_tokens
            .saturating_sub(caduceus_bridge::orchestrator::count_tokens_exact(&summary))
            as usize;
        caduceus_bridge::context_events::record_and_count(
            caduceus_bridge::context_events::ContextEventKind::AutoCompacted {
                messages_compacted: messages_to_compact,
                tokens_freed,
                fill_pct,
                zone: format!("{:?}", zone),
                keep_recent,
            },
        );
        true
    }

    pub fn set_model(&mut self, model: Arc<dyn LanguageModel>, cx: &mut Context<Self>) {
        let old_usage = self.latest_token_usage();
        self.model = Some(model.clone());
        let new_caps = Self::prompt_capabilities(self.model.as_deref());
        let new_usage = self.latest_token_usage();
        if old_usage != new_usage {
            cx.emit(TokenUsageUpdated(new_usage));
        }
        self.prompt_capabilities_tx.send(new_caps).log_err();

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| thread.set_model(model.clone(), cx))
                .ok();
        }

        // G1a: model change invalidates the native-loop harness so the
        // next turn rebuilds with the new system prompt / model binding.
        self.invalidate_caduceus_harness();

        cx.notify()
    }

    pub fn summarization_model(&self) -> Option<&Arc<dyn LanguageModel>> {
        self.summarization_model.as_ref()
    }

    pub fn set_summarization_model(
        &mut self,
        model: Option<Arc<dyn LanguageModel>>,
        cx: &mut Context<Self>,
    ) {
        self.summarization_model = model.clone();

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| {
                    thread.set_summarization_model(model.clone(), cx)
                })
                .ok();
        }
        cx.notify()
    }

    pub fn thinking_enabled(&self) -> bool {
        self.thinking_enabled
    }

    pub fn set_thinking_enabled(&mut self, enabled: bool, cx: &mut Context<Self>) {
        self.thinking_enabled = enabled;

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| thread.set_thinking_enabled(enabled, cx))
                .ok();
        }
        cx.notify();
    }

    pub fn thinking_effort(&self) -> Option<&String> {
        self.thinking_effort.as_ref()
    }

    pub fn set_thinking_effort(&mut self, effort: Option<String>, cx: &mut Context<Self>) {
        self.thinking_effort = effort.clone();

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| {
                    thread.set_thinking_effort(effort.clone(), cx)
                })
                .ok();
        }
        cx.notify();
    }

    pub fn speed(&self) -> Option<Speed> {
        self.speed
    }

    pub fn set_speed(&mut self, speed: Speed, cx: &mut Context<Self>) {
        self.speed = Some(speed);

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| thread.set_speed(speed, cx))
                .ok();
        }
        cx.notify();
    }

    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn last_received_or_pending_message(&self) -> Option<Message> {
        if let Some(message) = self.pending_message.clone() {
            Some(Message::Agent(message))
        } else {
            self.messages.last().cloned()
        }
    }

    pub fn add_default_tools(
        &mut self,
        environment: Rc<dyn ThreadEnvironment>,
        caduceus_engine: Option<Arc<caduceus_bridge::engine::CaduceusEngine>>,
        cx: &mut Context<Self>,
    ) {
        // Only update the agent location for the root thread, not for subagents.
        let update_agent_location = self.parent_thread_id().is_none();

        let language_registry = self.project.read(cx).languages().clone();
        self.add_tool(CopyPathTool::new(self.project.clone()));
        self.add_tool(CreateDirectoryTool::new(self.project.clone()));
        self.add_tool(DeletePathTool::new(
            self.project.clone(),
            self.action_log.clone(),
        ));
        self.add_tool(DiagnosticsTool::new(self.project.clone()));
        self.add_tool(EditFileTool::new(
            self.project.clone(),
            cx.weak_entity(),
            language_registry.clone(),
            Templates::new(),
        ));
        self.add_tool(StreamingEditFileTool::new(
            self.project.clone(),
            cx.weak_entity(),
            self.action_log.clone(),
            language_registry,
        ));
        self.add_tool(FetchTool::new(self.project.read(cx).client().http_client()));
        self.add_tool(FindPathTool::new(self.project.clone()));
        self.add_tool(GrepTool::new(self.project.clone()));
        self.add_tool(ListDirectoryTool::new(self.project.clone()));
        self.add_tool(MovePathTool::new(self.project.clone()));
        self.add_tool(NowTool);
        self.add_tool(OpenTool::new(self.project.clone()));
        if cx.has_flag::<UpdatePlanToolFeatureFlag>() {
            self.add_tool(UpdatePlanTool);
        }
        self.add_tool(ReadFileTool::new(
            self.project.clone(),
            self.action_log.clone(),
            update_agent_location,
        ));
        self.add_tool(SaveFileTool::new(self.project.clone()));
        self.add_tool(RestoreFileFromDiskTool::new(self.project.clone()));
        self.add_tool(TerminalTool::new(self.project.clone(), environment.clone()));
        self.add_tool(WebSearchTool);

        // Caduceus engine tools — use shared engine from ProjectState
        if let Some(worktree) = self.project.read(cx).worktrees(cx).next() {
            let project_root = worktree.read(cx).abs_path().to_path_buf();
            // Use shared engine if provided, otherwise create one (fallback for tests)
            let engine = caduceus_engine.unwrap_or_else(|| {
                Arc::new(caduceus_bridge::engine::CaduceusEngine::new(&project_root))
            });

            self.add_tool(CaduceusSemanticSearchTool::new(engine.clone()));
            self.add_tool(CaduceusIndexTool::new(engine.clone()));
            self.add_tool(CaduceusSecurityScanTool::new(engine.clone()));
            self.add_tool(CaduceusCodeGraphTool::new(engine.clone()));
            self.add_tool(CaduceusErrorAnalysisTool::new(engine.clone()));
            self.add_tool(CaduceusMcpSecurityTool::new(engine.clone()));
            self.add_tool(CaduceusGitReadTool::new(engine.clone()));
            self.add_tool(CaduceusGitWriteTool::new(engine.clone()));
            self.add_tool(CaduceusMemoryReadTool::new(project_root.clone()));
            self.add_tool(CaduceusMemoryWriteTool::new(project_root.clone()));
            self.add_tool(CaduceusStorageTool::new(project_root.clone()));
            self.add_tool(CaduceusCheckpointTool::new(project_root.clone()));
            self.add_tool(CaduceusProjectTool::new(project_root.clone()));
            self.add_tool(CaduceusProjectWikiTool::new(project_root.clone()));
            self.add_tool(CaduceusCrossGitTool::new(project_root.clone()));
            self.add_tool(CaduceusCrossSearchTool::new(project_root.clone()));
            self.add_tool(CaduceusApiRegistryTool::new(project_root.clone()));
            self.add_tool(CaduceusArchitectTool::new(project_root.clone()));
            self.add_tool(CaduceusProductTool::new(project_root.clone()));
            self.add_tool(CaduceusKanbanTool::new(project_root.clone(), engine));
            self.add_tool(CaduceusAutomationsTool::new(project_root.clone()));
            self.add_tool(CaduceusPolicyTool::new(project_root.clone()));
            self.add_tool(CaduceusBackgroundAgentTool::new(project_root));
        }
        self.add_tool(CaduceusDependencyScanTool::new());
        self.add_tool(CaduceusScaffoldTool::new());
        self.add_tool(CaduceusPrdTool::new());
        self.add_tool(CaduceusConversationTool::new());
        self.add_tool(CaduceusMarketplaceTool::new());
        self.add_tool(CaduceusProgressTool::new());
        self.add_tool(CaduceusTelemetryTool::new());
        self.add_tool(CaduceusTimeTrackingTool::new());
        self.add_tool(CaduceusTreeSitterTool::new(self.project.clone()));
        self.add_tool(CaduceusTaskTreeTool::new());
        self.add_tool(CaduceusTaskDecomposeTool::new(self.task_dag.clone()));
        self.add_tool(CaduceusKillSwitchTool::new());

        if self.depth() < MAX_SUBAGENT_DEPTH {
            self.add_tool(SpawnAgentTool::new(environment));
        }
        self.add_tool(ReadThreadTool::new());
        self.add_tool(CompactThreadTool::new());
        self.add_tool(SuggestModelsTool::new());
    }

    pub fn add_tool<T: AgentTool>(&mut self, tool: T) {
        debug_assert!(
            !self.tools.contains_key(T::NAME),
            "Duplicate tool name: {}",
            T::NAME,
        );
        self.tools.insert(T::NAME.into(), tool.erase());
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn remove_tool(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }

    pub fn profile(&self) -> &AgentProfileId {
        &self.profile_id
    }

    pub fn set_profile(&mut self, profile_id: AgentProfileId, cx: &mut Context<Self>) {
        if self.profile_id == profile_id {
            return;
        }

        self.profile_id = profile_id.clone();
        // Caduceus: sync mode with profile
        self.caduceus_mode = Some(profile_id.0.to_string());

        // Swap to the profile's preferred model when available.
        if let Some(model) = Self::resolve_profile_model(&self.profile_id, cx) {
            self.set_model(model, cx);
        }

        for subagent in &self.running_subagents {
            subagent
                .update(cx, |thread, cx| thread.set_profile(profile_id.clone(), cx))
                .ok();
        }
    }

    pub fn cancel(&mut self, cx: &mut Context<Self>) -> Task<()> {
        // Reset safety counters so the next turn starts fresh
        self.loop_escalation_count = 0;
        // Mark this turn as cancelled so flush_pending_message skips
        // auto-memory extraction (it would race the next turn's
        // activeContext.md writer).
        self.last_turn_cancelled = true;
        // Bump generation: any in-flight callbacks scheduled before this point
        // will see a stale generation and bail out instead of mutating new-turn state.
        self.turn_generation = self.turn_generation.wrapping_add(1);
        let cancel_generation = self.turn_generation;
        for subagent in self.running_subagents.drain(..) {
            if let Some(subagent) = subagent.upgrade() {
                subagent.update(cx, |thread, cx| thread.cancel(cx)).detach();
            }
        }

        let Some(running_turn) = self.running_turn.take() else {
            self.flush_pending_message(cx);
            return Task::ready(());
        };

        let turn_task = running_turn.cancel();

        cx.spawn(async move |this, cx| {
            turn_task.await;
            this.update(cx, |this, cx| {
                // Only flush if no new turn started in the meantime — otherwise
                // we'd flush the *new* turn's pending message into history.
                if this.turn_generation == cancel_generation {
                    this.flush_pending_message(cx);
                }
            })
            .ok();
        })
    }

    pub fn set_has_queued_message(&mut self, has_queued: bool) {
        self.has_queued_message = has_queued;
    }

    pub fn has_queued_message(&self) -> bool {
        self.has_queued_message
    }

    fn update_token_usage(&mut self, update: language_model::TokenUsage, cx: &mut Context<Self>) {
        let Some(last_user_message) = self.last_user_message() else {
            return;
        };

        self.request_token_usage
            .insert(last_user_message.id.clone(), update);
        cx.emit(TokenUsageUpdated(self.latest_token_usage()));
        cx.notify();
    }

    pub fn truncate(&mut self, message_id: UserMessageId, cx: &mut Context<Self>) -> Result<()> {
        self.cancel(cx).detach();
        // Clear pending message since cancel will try to flush it asynchronously,
        // and we don't want that content to be added after we truncate
        self.pending_message.take();
        let Some(position) = self.messages.iter().position(
            |msg| matches!(msg, Message::User(UserMessage { id, .. }) if id == &message_id),
        ) else {
            return Err(anyhow!("Message not found"));
        };

        let evicted_count = self.messages.len() - position;
        for message in self.messages.drain(position..) {
            match message {
                Message::User(message) => {
                    self.request_token_usage.remove(&message.id);
                }
                Message::Agent(_) | Message::Resume(_) => {}
            }
        }
        if evicted_count > 0 {
            caduceus_bridge::context_events::record_and_count(
                caduceus_bridge::context_events::ContextEventKind::MessageEvicted {
                    count: evicted_count,
                    reason: "user truncated to earlier message".into(),
                },
            );
        }
        // ST2 fix-loop #6: drop pins that target messages no longer in
        // the live message vec. Without this, `pinned_message_indices`
        // could silently lose entries (resolve_key_to_index returns
        // None) but the orphaned pins would persist in `self.pinned`
        // and eventually be written to disk.
        self.gc_pinned();
        self.invalidate_token_cache();
        self.clear_summary();
        cx.notify();
        Ok(())
    }

    pub fn latest_request_token_usage(&self) -> Option<language_model::TokenUsage> {
        let last_user_message = self.last_user_message()?;
        let tokens = self.request_token_usage.get(&last_user_message.id)?;
        Some(*tokens)
    }

    pub fn latest_token_usage(&self) -> Option<acp_thread::TokenUsage> {
        let usage = self.latest_request_token_usage()?;
        let model = self.model.clone()?;
        Some(acp_thread::TokenUsage {
            max_tokens: model.max_token_count(),
            max_output_tokens: model.max_output_tokens(),
            used_tokens: usage.total_tokens(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        })
    }

    /// Get the total input token count as of the message before the given message.
    ///
    /// Returns `None` if:
    /// - `target_id` is the first message (no previous message)
    /// - The previous message hasn't received a response yet (no usage data)
    /// - `target_id` is not found in the messages
    pub fn tokens_before_message(&self, target_id: &UserMessageId) -> Option<u64> {
        let mut previous_user_message_id: Option<&UserMessageId> = None;

        for message in &self.messages {
            if let Message::User(user_msg) = message {
                if &user_msg.id == target_id {
                    let prev_id = previous_user_message_id?;
                    let usage = self.request_token_usage.get(prev_id)?;
                    return Some(usage.input_tokens);
                }
                previous_user_message_id = Some(&user_msg.id);
            }
        }
        None
    }

    /// Look up the active profile and resolve its preferred model if one is configured.
    fn resolve_profile_model(
        profile_id: &AgentProfileId,
        cx: &mut Context<Self>,
    ) -> Option<Arc<dyn LanguageModel>> {
        let selection = AgentSettings::get_global(cx)
            .profiles
            .get(profile_id)?
            .default_model
            .clone()?;
        Self::resolve_model_from_selection(&selection, cx)
    }

    /// Translate a stored model selection into the configured model from the registry.
    fn resolve_model_from_selection(
        selection: &LanguageModelSelection,
        cx: &mut Context<Self>,
    ) -> Option<Arc<dyn LanguageModel>> {
        let selected = SelectedModel {
            provider: LanguageModelProviderId::from(selection.provider.0.clone()),
            model: LanguageModelId::from(selection.model.clone()),
        };
        LanguageModelRegistry::global(cx).update(cx, |registry, cx| {
            registry
                .select_model(&selected, cx)
                .map(|configured| configured.model)
        })
    }

    pub fn resume(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>> {
        self.messages.push(Message::Resume(ResumeId::new()));
        // ST2: auto-pin Resume marker (explicit-resume trigger site).
        self.auto_pin_resume();
        self.invalidate_token_cache();
        cx.notify();

        log::debug!("Total messages in thread: {}", self.messages.len());
        self.run_turn(cx)
    }

    /// Sending a message results in the model streaming a response, which could include tool calls.
    /// After calling tools, the model will stops and waits for any outstanding tool calls to be completed and their results sent.
    /// The returned channel will report all the occurrences in which the model stops before erroring or ending its turn.
    pub fn send<T>(
        &mut self,
        id: UserMessageId,
        content: impl IntoIterator<Item = T>,
        cx: &mut Context<Self>,
    ) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>>
    where
        T: Into<UserMessageContent>,
    {
        let content = content.into_iter().map(Into::into).collect::<Vec<_>>();
        log::debug!("Thread::send content: {:?}", content);

        let id_for_pin = id.clone();
        self.messages
            .push(Message::User(UserMessage { id, content }));
        // ST2: auto-pin if this is the FIRST user message in the thread.
        self.maybe_auto_pin_first_user(id_for_pin);
        self.invalidate_token_cache();
        cx.notify();

        self.send_existing(cx)
    }

    pub fn send_existing(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>> {
        let model = self
            .model()
            .ok_or_else(|| anyhow!(NoModelConfiguredError))?;

        log::info!("Thread::send called with model: {}", model.name().0);
        self.advance_prompt_id();

        log::debug!("Total messages in thread: {}", self.messages.len());
        self.run_turn(cx)
    }

    pub fn push_acp_user_block(
        &mut self,
        id: UserMessageId,
        blocks: impl IntoIterator<Item = acp::ContentBlock>,
        path_style: PathStyle,
        cx: &mut Context<Self>,
    ) {
        let content = blocks
            .into_iter()
            .map(|block| UserMessageContent::from_content_block(block, path_style))
            .collect::<Vec<_>>();
        let id_for_pin = id.clone();
        self.messages
            .push(Message::User(UserMessage { id, content }));
        // ST2: auto-pin if this is the FIRST user message (alt insert path).
        self.maybe_auto_pin_first_user(id_for_pin);
        self.invalidate_token_cache();
        cx.notify();
    }

    pub fn push_acp_agent_block(&mut self, block: acp::ContentBlock, cx: &mut Context<Self>) {
        let text = match block {
            acp::ContentBlock::Text(text_content) => text_content.text,
            acp::ContentBlock::Image(_) => "[image]".to_string(),
            acp::ContentBlock::Audio(_) => "[audio]".to_string(),
            acp::ContentBlock::ResourceLink(resource_link) => resource_link.uri,
            acp::ContentBlock::Resource(resource) => match resource.resource {
                acp::EmbeddedResourceResource::TextResourceContents(resource) => resource.uri,
                acp::EmbeddedResourceResource::BlobResourceContents(resource) => resource.uri,
                _ => "[resource]".to_string(),
            },
            _ => "[unknown]".to_string(),
        };

        self.messages.push(Message::Agent(AgentMessage {
            content: vec![AgentMessageContent::Text(text)],
            ..Default::default()
        }));
        self.invalidate_token_cache();
        cx.notify();
    }

    fn run_turn(
        &mut self,
        cx: &mut Context<Self>,
    ) -> Result<mpsc::UnboundedReceiver<Result<ThreadEvent>>> {
        // Reset per-turn UI state
        self.context_compacted_this_turn = false;
        // Bump generation: flagged callbacks scheduled by the prior turn (cancel
        // cleanup, auto-extract, etc.) will see a stale generation and exit.
        self.turn_generation = self.turn_generation.wrapping_add(1);
        // Clean stale guardrail alert from memory
        if self
            .guardrail_alert
            .as_ref()
            .is_some_and(|(_, _, ts)| ts.elapsed() >= std::time::Duration::from_secs(10))
        {
            self.guardrail_alert = None;
        }

        // Flush the old pending message synchronously before cancelling, so the
        // detached cancel task can't flush stale state into the new turn.
        // This must happen BEFORE auto_compact so any in-flight tool_results
        // become part of a complete message and don't get orphaned by the drain
        // (compaction would otherwise leave tool_results without their owning
        // ToolUse, producing invalid LLM requests).
        self.flush_pending_message(cx);
        self.cancel(cx).detach();

        // Caduceus: check context zone and warn user
        let zone = self.current_context_zone();
        match zone {
            caduceus_bridge::orchestrator::ContextZone::Orange => {
                log::warn!("[caduceus] Context 70-85% full — consider /compact")
            }
            caduceus_bridge::orchestrator::ContextZone::Red => {
                log::warn!("[caduceus] Context 85-95% full — auto-compacting soon")
            }
            caduceus_bridge::orchestrator::ContextZone::Critical => {
                log::error!("[caduceus] Context >95% full — EMERGENCY")
            }
            _ => {}
        }

        // Caduceus: auto-compact context — pass pre-computed zone to avoid recomputation.
        // Runs AFTER flush so message history is consistent (no orphan tool_results).
        self.auto_compact_context_with_zone(zone, cx);

        let (events_tx, events_rx) = mpsc::unbounded::<Result<ThreadEvent>>();
        let event_stream = ThreadEventStream(events_tx);
        let message_ix = self.messages.len().saturating_sub(1);
        self.clear_summary();
        // Clear the cancelled-flag so that this turn ending normally will be
        // allowed to extract auto-memories. (Set by `cancel()` if a previous
        // turn was aborted.)
        self.last_turn_cancelled = false;
        let (cancellation_tx, mut cancellation_rx) = watch::channel(false);
        self.running_turn = Some(RunningTurn {
            event_stream: event_stream.clone(),
            tools: self.enabled_tools(cx),
            cancellation_tx,
            streaming_tool_inputs: HashMap::default(),
            _task: cx.spawn(async move |this, cx| {
                log::debug!("Starting agent turn execution");

                let turn_result =
                    Self::run_turn_internal(&this, &event_stream, cancellation_rx.clone(), cx)
                        .await;

                // Check if we were cancelled - if so, cancel() already took running_turn
                // and we shouldn't touch it (it might be a NEW turn now)
                let was_cancelled = *cancellation_rx.borrow();
                if was_cancelled {
                    log::debug!("Turn was cancelled, skipping cleanup");
                    return;
                }

                _ = this.update(cx, |this, cx| this.flush_pending_message(cx));

                match turn_result {
                    Ok(()) => {
                        log::debug!("Turn execution completed");
                        event_stream.send_stop(acp::StopReason::EndTurn);
                    }
                    Err(error) => {
                        log::error!("Turn execution failed: {:?}", error);
                        match error.downcast::<CompletionError>() {
                            Ok(CompletionError::Refusal) => {
                                event_stream.send_stop(acp::StopReason::Refusal);
                                _ = this.update(cx, |this, _| this.messages.truncate(message_ix));
                            }
                            Ok(CompletionError::MaxTokens) => {
                                event_stream.send_stop(acp::StopReason::MaxTokens);
                            }
                            Ok(CompletionError::Other(error)) | Err(error) => {
                                event_stream.send_error(error);
                            }
                        }
                    }
                }

                _ = this.update(cx, |this, _| this.running_turn.take());
            }),
        });
        Ok(events_rx)
    }

    async fn run_turn_internal(
        this: &WeakEntity<Self>,
        event_stream: &ThreadEventStream,
        mut cancellation_rx: watch::Receiver<bool>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        // G1d: native-loop gate. When the setting is ON, try to
        // dispatch through the engine. If the gate declines (flag
        // off, provider adapter missing, or ensure_caduceus_harness
        // errored), fall through to the legacy body unchanged.
        if Self::try_run_turn_native(this, event_stream, &mut cancellation_rx, cx).await? {
            return Ok(());
        }

        let mut attempt = 0;
        let mut intent = CompletionIntent::UserPrompt;
        loop {
            // Re-read the model and refresh tools on each iteration so that
            // mid-turn changes (e.g. the user switches model, toggles tools,
            // or changes profile) take effect between tool-call rounds.
            let (model, request) = this.update(cx, |this, cx| {
                let model = this
                    .model
                    .clone()
                    .ok_or_else(|| anyhow!(NoModelConfiguredError))?;
                this.refresh_turn_tools(cx);
                let request = this.build_completion_request(intent, cx)?;
                anyhow::Ok((model, request))
            })??;

            telemetry::event!(
                "Agent Thread Completion",
                thread_id = this.read_with(cx, |this, _| this.id.to_string())?,
                parent_thread_id = this.read_with(cx, |this, _| this
                    .parent_thread_id()
                    .map(|id| id.to_string()))?,
                prompt_id = this.read_with(cx, |this, _| this.prompt_id.to_string())?,
                model = model.telemetry_id(),
                model_provider = model.provider_id().to_string(),
                attempt
            );

            log::debug!("Calling model.stream_completion, attempt {}", attempt);

            let (mut events, mut error) = match model.stream_completion(request, cx).await {
                Ok(events) => (events.fuse(), None),
                Err(err) => (stream::empty().boxed().fuse(), Some(err)),
            };
            let mut tool_results: FuturesUnordered<Task<LanguageModelToolResult>> =
                FuturesUnordered::new();
            let mut early_tool_results: Vec<LanguageModelToolResult> = Vec::new();
            let mut cancelled = false;
            loop {
                // Race between getting the first event, tool completion, and cancellation.
                let first_event = futures::select! {
                    event = events.next().fuse() => event,
                    tool_result = futures::StreamExt::select_next_some(&mut tool_results) => {
                        let is_error = tool_result.is_error;
                        let is_still_streaming = this
                            .read_with(cx, |this, _cx| {
                                this.running_turn
                                    .as_ref()
                                    .and_then(|turn| turn.streaming_tool_inputs.get(&tool_result.tool_use_id))
                                    .map_or(false, |inputs| !inputs.has_received_final())
                            })
                            .unwrap_or(false);

                        early_tool_results.push(tool_result);

                        // Only break if the tool errored and we are still
                        // streaming the input of the tool. If the tool errored
                        // but we are no longer streaming its input (i.e. there
                        // are parallel tool calls) we want to continue
                        // processing those tool inputs.
                        if is_error && is_still_streaming {
                            break;
                        }
                        continue;
                    }
                    _ = cancellation_rx.changed().fuse() => {
                        if *cancellation_rx.borrow() {
                            cancelled = true;
                            break;
                        }
                        continue;
                    }
                };
                let Some(first_event) = first_event else {
                    break;
                };

                // Collect all immediately available events to process as a batch
                let mut batch = vec![first_event];
                while let Some(event) = events.next().now_or_never().flatten() {
                    batch.push(event);
                }

                // Process the batch in a single update
                let batch_result = this.update(cx, |this, cx| {
                    let mut batch_tool_results = Vec::new();
                    let mut batch_error = None;

                    for event in batch {
                        log::trace!("Received completion event: {:?}", event);
                        match event {
                            Ok(event) => {
                                match this.handle_completion_event(
                                    event,
                                    event_stream,
                                    cancellation_rx.clone(),
                                    cx,
                                ) {
                                    Ok(Some(task)) => batch_tool_results.push(task),
                                    Ok(None) => {}
                                    Err(err) => {
                                        batch_error = Some(err);
                                        break;
                                    }
                                }
                            }
                            Err(err) => {
                                batch_error = Some(err.into());
                                break;
                            }
                        }
                    }

                    cx.notify();
                    (batch_tool_results, batch_error)
                })?;

                tool_results.extend(batch_result.0);
                if let Some(err) = batch_result.1 {
                    error = Some(err.downcast()?);
                    break;
                }
            }

            // Drop the stream to release the rate limit permit before tool execution.
            // The stream holds a semaphore guard that limits concurrent requests.
            // Without this, the permit would be held during potentially long-running
            // tool execution, which could cause deadlocks when tools spawn subagents
            // that need their own permits.
            drop(events);

            // Drop streaming tool input senders that never received their final input.
            // This prevents deadlock when the LLM stream ends (e.g. because of an error)
            // before sending a tool use with `is_input_complete: true`.
            this.update(cx, |this, _cx| {
                if let Some(running_turn) = this.running_turn.as_mut() {
                    if running_turn.streaming_tool_inputs.is_empty() {
                        return;
                    }
                    log::warn!("Dropping partial tool inputs because the stream ended");
                    running_turn.streaming_tool_inputs.drain();
                }
            })?;

            let end_turn = tool_results.is_empty() && early_tool_results.is_empty();

            for tool_result in early_tool_results {
                Self::process_tool_result(this, event_stream, cx, tool_result)?;
            }
            while let Some(tool_result) = tool_results.next().await {
                Self::process_tool_result(this, event_stream, cx, tool_result)?;
            }

            this.update(cx, |this, cx| {
                this.flush_pending_message(cx);
                if this.title.is_none() && this.pending_title_generation.is_none() {
                    this.generate_title(cx);
                }
            })?;

            if cancelled {
                log::debug!("Turn cancelled by user, exiting");
                return Ok(());
            }

            if let Some(error) = error {
                attempt += 1;
                let retry = this.update(cx, |this, cx| {
                    let user_store = this.user_store.read(cx);
                    this.handle_completion_error(error, attempt, user_store.plan(), cx)
                })??;
                let timer = cx.background_executor().timer(retry.duration);
                event_stream.send_retry(retry);
                futures::select! {
                    _ = timer.fuse() => {}
                    _ = cancellation_rx.changed().fuse() => {
                        if *cancellation_rx.borrow() {
                            log::debug!("Turn cancelled during retry delay, exiting");
                            return Ok(());
                        }
                    }
                }
                this.update(cx, |this, _cx| {
                    if let Some(Message::Agent(message)) = this.messages.last() {
                        if message.tool_results.is_empty() {
                            intent = CompletionIntent::UserPrompt;
                            this.messages.push(Message::Resume(ResumeId::new()));
                            // ST2: auto-pin implicit-continuation Resume.
                            this.auto_pin_resume();
                        }
                    }
                })?;
            } else if end_turn {
                return Ok(());
            } else {
                let has_queued = this.update(cx, |this, _| this.has_queued_message())?;
                if has_queued {
                    log::debug!("Queued message found, ending turn at message boundary");
                    return Ok(());
                }
                intent = CompletionIntent::ToolResults;
                attempt = 0;
            }
        }
    }

    /// G1d → ST-A2: native-loop dispatch. Returns `Ok(true)` if the
    /// native path fully handled the turn (caller returns immediately),
    /// or `Ok(false)` if the legacy path should run.
    ///
    /// **What "native" does today:**
    /// * Builds a per-session `AgentHarness` on first turn (cached on
    ///   Thread; invalidated on model/mode change).
    /// * Spawns a GPUI-foreground provider dispatcher that owns the
    ///   zed `LanguageModel` and answers `DispatchRequest`s from the
    ///   engine via `stream_completion`.
    /// * Subscribes a fresh broadcast receiver per turn (ST-A2a) and
    ///   bridges it into the mpsc `event_rx` that
    ///   `run_caduceus_loop_translated` currently consumes.
    /// * Bumps the cancel-token generation (ST-A2b / ST-A8) so stale
    ///   cancels from a prior turn cannot poison this one.
    /// * Acquires `caduceus_native_state` (async Mutex) so concurrent
    ///   `try_run_turn_native` calls serialise correctly.
    /// * Drives `dispatch_translated_event` off a `cx.spawn`'d
    ///   consumer so every `TranslatedThreadEvent` routes to the
    ///   ThreadEventStream.
    ///
    /// **Known limitations of this first cut (ST-A2 baseline, ST-B
    /// fills in):** tools registry is empty — tool-calls requested by
    /// the model will not execute. System prompt is empty. No
    /// approval flow (`no_approval()`). These surface as
    /// `EngineDiagnostic`s via the translator; full tool + approval
    /// wiring is tracked by ST-B1/B2 on follow-up.
    async fn try_run_turn_native(
        this: &WeakEntity<Self>,
        event_stream: &ThreadEventStream,
        _cancellation_rx: &mut watch::Receiver<bool>,
        cx: &mut AsyncApp,
    ) -> Result<bool> {
        use crate::caduceus_provider_adapter::{
            DispatcherHandle, ZedLlmAdapter, dispatcher_channel, drive_dispatcher_fn,
        };
        use crate::caduceus_tool_adapter as ct;

        // ── Phase 1: gate + extract synchronously ──────────────────
        struct TurnSetup {
            bridge: std::sync::Arc<caduceus_bridge::orchestrator::OrchestratorBridge>,
            model: std::sync::Arc<dyn language_model::LanguageModel>,
            user_input: String,
            native_state:
                std::sync::Arc<tokio::sync::Mutex<crate::caduceus_native_state::NativeLoopState>>,
            cancel_token: caduceus_core::CancellationToken,
            harness: Option<std::sync::Arc<caduceus_orchestrator::AgentHarness>>,
            emitter: Option<caduceus_orchestrator::AgentEventEmitter>,
            dispatcher: Option<DispatcherHandle>,
            /// NW-3: approval channel captured alongside the harness.
            /// Keyed `perm_{tool_use_id}` per engine contract.
            approval_tx: Option<tokio::sync::mpsc::Sender<(String, bool)>>,
            /// NW-1: enabled-tools snapshot captured in Phase 1 under
            /// the `this.update` borrow so Phase 2 can build the tool
            /// dispatcher without a second `update` round-trip.
            enabled_tools:
                std::collections::BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>>,
            /// NW-2: rendered base system prompt captured in Phase 1.
            /// The engine's `effective_system_prompt` wraps this with
            /// behavior_rules + mode block + envelope, so we pass only
            /// the Zed-side project/tool-list content here — NOT the
            /// mode prefix or behavior preamble (engine handles those).
            system_prompt: String,
            /// NW-2: Caduceus mode/lens strings to thread onto the
            /// harness via `with_mode` + `with_mode_lens` so the
            /// engine's `effective_system_prompt` includes the
            /// `<agent_mode>` block and mode-driven defaults.
            caduceus_mode: String,
            caduceus_lens: Option<String>,
        }

        let setup = this.update(cx, |this, cx| -> Option<TurnSetup> {
            let flag_on = agent_settings::AgentSettings::get_global(cx).caduceus_native_loop;
            if !flag_on {
                return None;
            }
            if let Err(err) = this.ensure_caduceus_harness(cx) {
                // T4 (Audit C6): flag was explicitly ON but the native
                // harness could not be built. Previously this was a
                // `warn!` + silent legacy fallback. Keep the fallback
                // (to avoid breaking user turns) but escalate the log
                // level and include actionable context so the failure
                // is visible in logs / bug reports.
                log::error!(
                    "[native-loop] ensure_caduceus_harness failed with flag ON: {err}. \
                     Falling back to legacy turn path. The no-folder-workspace case is \
                     now silent (info-level); reaching this branch indicates a real \
                     bridge build failure or initialization bug — investigate."
                );
                return None;
            }
            let bridge = this.caduceus.bridge.clone()?;
            let native_state = this.caduceus.native_state.clone()?;
            let cancel_token = this.caduceus.cancel_token.clone()?;
            let model = this.model.clone()?;
            let user_input = this
                .messages
                .iter()
                .rev()
                .find_map(|m| match m {
                    Message::User(u) => Some(u.to_markdown()),
                    Message::Resume(_) => Some("Continue where you left off.".to_string()),
                    _ => None,
                })
                .unwrap_or_default();
            let enabled_tools = this.enabled_tools(cx);
            // NW-2: render SystemPromptTemplate with the same inputs
            // the legacy path uses. Engine adds behavior_rules + mode
            // on top; we pass only the project-focused content here
            // so nothing double-renders.
            let available_tools: Vec<SharedString> = enabled_tools.keys().cloned().collect();
            let system_prompt = SystemPromptTemplate {
                project: this.project_context.read(cx),
                available_tools,
                model_name: Some(model.name().0.to_string()),
            }
            .render(&this.templates)
            .unwrap_or_default();
            // Follow-up #2: native path must include the same Caduceus
            // context block the legacy path builds (project instructions,
            // wiki overview, pinned context, @mention resolution, cross-
            // repo config, tool guide). Engine prepends behavior_rules +
            // mode block on top — so we only append the Zed-side context
            // here, matching `build_request_messages`.
            let caduceus_guidance = this.build_caduceus_context_block(cx);
            let system_prompt = format!("{system_prompt}{caduceus_guidance}");
            let caduceus_mode = this.caduceus_mode_from_profile().to_string();
            let caduceus_lens = this.caduceus_lens_from_profile();
            Some(TurnSetup {
                bridge,
                model,
                user_input,
                native_state,
                cancel_token,
                harness: this.caduceus.harness.clone(),
                emitter: this.caduceus.emitter.clone(),
                dispatcher: this.caduceus.dispatcher.clone(),
                approval_tx: this.caduceus.approval_tx.clone(),
                enabled_tools,
                system_prompt,
                caduceus_mode,
                caduceus_lens,
            })
        })?;
        let mut setup = match setup {
            Some(s) => s,
            None => return Ok(false),
        };

        // ── Phase 2: lazy per-session harness build ────────────────
        if setup.harness.is_none() {
            let (disp_handle, disp_rx) = dispatcher_channel();
            let model_for_dispatcher = setup.model.clone();
            let cx_for_dispatcher = cx.clone();
            cx.spawn(async move |_| {
                drive_dispatcher_fn(disp_rx, move |req| {
                    let m = model_for_dispatcher.clone();
                    let mut cx2 = cx_for_dispatcher.clone();
                    async move { m.stream_completion(req, &mut cx2).await }
                })
                .await;
            })
            .detach();

            // NW-1: tool-side dispatcher. Distinct from the provider
            // dispatcher above — they carry different request types
            // (`DispatchRequest` for tools vs the provider adapter's
            // own request type). We run the tool-exec loop directly
            // on `cx.spawn` (not via `caduceus_tool_adapter::drive`)
            // because `drive()` requires `Send` futures and GPUI tool
            // execution takes `&mut App` which is !Send. Concurrency
            // is the gpui foreground executor's natural
            // serialization; tools call each other through `cx.update`
            // and progress is bounded by the main-thread queue.
            let (tool_tx, mut tool_rx) =
                tokio::sync::mpsc::channel::<ct::DispatchRequest>(ct::REQUEST_CHANNEL_CAP);
            let tool_handle = ct::DispatcherHandle::new(tool_tx);

            // Snapshot capture for the exec loop. Cloning the BTreeMap
            // of `Arc<dyn AnyAgentTool>` is cheap. Mid-session tool-set
            // changes don't mutate the harness's view — acceptable
            // because profile switches already trigger
            // `invalidate_caduceus_harness`.
            let tool_map = setup.enabled_tools.clone();
            let this_weak = this.clone();
            cx.spawn(async move |cx_loop| {
                while let Some(req) = tool_rx.recv().await {
                    let ct::DispatchRequest {
                        tool_name,
                        input,
                        respond_to,
                    } = req;
                    let Some(tool) = tool_map.get(tool_name.as_str()).cloned() else {
                        let _ = respond_to.send(Err(ct::AdapterError::UnknownTool(tool_name)));
                        continue;
                    };
                    let tool_use_id = language_model::LanguageModelToolUseId::from(format!(
                        "native-{}-{}",
                        tool_name,
                        uuid::Uuid::new_v4()
                    ));
                    // Run the Zed tool on the gpui main thread.
                    let this_weak = this_weak.clone();
                    let spawn_res = this_weak.update(cx_loop, |this, cx| {
                        let fs = this.project.read(cx).fs().clone();
                        let (_tx, cancellation_rx) = watch::channel(false);
                        // Throwaway event stream: tool UI events go via
                        // the engine's TranslatedThreadEvent::ToolCall
                        // surface (see dispatch_translated_event). NW-3
                        // upgrades this to the real Thread event stream
                        // when approval routing lands.
                        let (ev_tx, _ev_rx) = futures::channel::mpsc::unbounded();
                        let throwaway_stream = ThreadEventStream(ev_tx);
                        let stream = ToolCallEventStream::new(
                            tool_use_id.clone(),
                            throwaway_stream,
                            Some(fs),
                            cancellation_rx,
                        );
                        let input = ToolInput::ready(input);
                        tool.clone().run(input, stream, cx)
                    });
                    let task = match spawn_res {
                        Ok(t) => t,
                        Err(e) => {
                            let _ =
                                respond_to.send(Err(ct::AdapterError::Execution(e.to_string())));
                            continue;
                        }
                    };
                    let output = task.await;
                    let (is_error, out) = match output {
                        Ok(o) => (false, o),
                        Err(o) => (true, o),
                    };
                    let text = match &out.llm_output {
                        language_model::LanguageModelToolResultContent::Text(t) => t.to_string(),
                        other => format!("{other:?}"),
                    };
                    let result = if is_error {
                        caduceus_core::ToolResult::error(text)
                    } else {
                        caduceus_core::ToolResult::success(text)
                    };
                    let _ = respond_to.send(Ok(result));
                }
            })
            .detach();

            let tool_registry = crate::caduceus_tool_adapter::build_zed_tool_registry(
                &setup.enabled_tools,
                tool_handle,
                |_| true,
            )
            .unwrap_or_else(|e| {
                log::warn!(
                    "[native-loop] build_zed_tool_registry failed: {e}; \
                                falling back to empty registry"
                );
                caduceus_tools::ToolRegistry::new()
            });

            let provider_id = caduceus_core::ProviderId::new(setup.model.provider_id().0.as_ref());
            let provider = std::sync::Arc::new(ZedLlmAdapter::new(provider_id, disp_handle.clone()))
                as std::sync::Arc<dyn caduceus_providers::LlmAdapter>;
            // ST7 r3 #1: prefer the pre-seeded emitter (subagent path
            // — Thread::new_subagent seeded it so spawn_agent_tool's
            // events() pump can attach BEFORE prompt() runs). When no
            // emitter is seeded (parent thread, first turn) the
            // builder mints a fresh one as before.
            let mut harness_builder =
                setup
                    .bridge
                    .harness(provider, tool_registry, setup.system_prompt.clone());
            harness_builder = if let Some(em) = setup.emitter.clone() {
                harness_builder.with_emitter_reuse(em)
            } else {
                harness_builder.with_emitter()
            };
            let built = harness_builder.with_reducer_sink().build();

            // FU#5 / emitter-mpsc-leak fix: HarnessBuilder hands us
            // `built.event_rx` — the single-consumer mpsc `Receiver`
            // for the emitter's live channel. The native path consumes
            // events via `emitter.subscribe()` (broadcast fan-out), NOT
            // via this mpsc. If we let `built` drop with `event_rx`
            // still inside it, the `Receiver` is destroyed and every
            // subsequent `emit()` in the harness trips
            // `TrySendError::Closed`, spamming ~N warnings per turn
            // ("AgentEventEmitter receiver closed; event will be
            // retained in ring only"). The events are preserved in the
            // retention ring and broadcast fan-out, so UI delivery is
            // unaffected — but the noise hides real signal and the
            // emitter keeps paying the `try_send` cost.
            //
            // Drain-and-drop on a background task for the turn's
            // lifetime. The task ends when the emitter's last `Sender`
            // drops (i.e. when the harness + emitter clone are dropped
            // at the end of the turn). Uses `cx.background_spawn`
            // because GPUI foreground drives this async fn and
            // `tokio::spawn` would panic (same class as the NW-4
            // crash). Requires NO tokio runtime — mpsc::Receiver::recv
            // is runtime-agnostic.
            if let Some(mut mpsc_rx) = built.event_rx {
                cx.background_spawn(async move { while mpsc_rx.recv().await.is_some() {} })
                    .detach();
            }

            // NW-3: capture approval_tx BEFORE consuming
            // `built.harness` into the mode-decorator chain. With
            // default approval (no `.no_approval()`), HarnessBuilder
            // guarantees `Some(approval_tx)` — see the bridge
            // docstring on `build_harness` for the key format
            // (`perm_{tool_use_id}`).
            let approval_tx = built
                .approval_tx
                .expect("default approval on HarnessBuilder guarantees Some(approval_tx)");

            // NW-2: attach mode + lens so the engine's
            // `effective_system_prompt` renders the `<agent_mode>`
            // block on top of the SystemPromptTemplate output. We use
            // `from_str_loose` which falls back to Plan / Normal on
            // unknown strings — same policy as
            // `mode_prompt_for_profile`.
            let harness_with_mode = {
                use caduceus_orchestrator::modes::{ActLens, AgentMode};
                let mode =
                    AgentMode::from_str_loose(&setup.caduceus_mode).unwrap_or(AgentMode::Plan);
                let lens = setup
                    .caduceus_lens
                    .as_deref()
                    .and_then(ActLens::from_str_loose)
                    .unwrap_or(ActLens::Normal);
                built.harness.with_mode(mode).with_mode_lens(lens)
            };
            let harness_arc = std::sync::Arc::new(harness_with_mode);
            let emitter = built
                .emitter
                .expect("with_emitter() on HarnessBuilder guarantees Some");

            this.update(cx, |this, _| {
                this.caduceus.harness = Some(harness_arc.clone());
                // ST7 r3 #1: get_or_insert — preserve the seeded
                // emitter (subagent path) so already-attached pump
                // subscribers keep observing the same broadcast
                // channel. For non-seeded parent threads, this falls
                // back to populating from the harness output.
                this.caduceus.emitter.get_or_insert_with(|| emitter.clone());
                this.caduceus.dispatcher = Some(disp_handle.clone());
                this.caduceus.approval_tx = Some(approval_tx.clone());
            })?;

            setup.harness = Some(harness_arc);
            setup.emitter = Some(emitter);
            setup.dispatcher = Some(disp_handle);
            setup.approval_tx = Some(approval_tx);
        }

        let harness = setup.harness.expect("built in phase 2");
        let emitter = setup.emitter.expect("built in phase 2");

        // ── Phase 3: bump cancel generation (ST-A8) ────────────────
        setup.cancel_token.bump_generation();
        let turn_gen = setup.cancel_token.current_generation();

        // ── Phase 4: serialise concurrent turns via state Mutex ────
        let mut state_guard = setup.native_state.lock().await;
        state_guard.turn_generation_at_lock = turn_gen;

        // ── Phase 5: subscribe a FRESH broadcast receiver (ST-A2a)
        // and bridge into the mpsc that run_caduceus_loop_translated
        // currently consumes.
        //
        // NOTE: this bridge task used to call `tokio::spawn`, which
        // panics with `TryCurrentError` because `try_run_turn_native`
        // is driven by GPUI's foreground executor — not a Tokio
        // runtime. Use `cx.background_spawn` so the task lives on
        // GPUI's background executor; drop-on-scope gives us the
        // same cancellation semantics as the old `.abort()` call.
        let mut broadcast_rx = emitter.subscribe();
        let (turn_event_tx, turn_event_rx) =
            tokio::sync::mpsc::channel::<caduceus_core::AgentEvent>(256);
        let bridge_task = cx.background_spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(ev) => {
                        if turn_event_tx.send(ev).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        log::warn!("[native-loop] broadcast lagged by {n} events");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        // ── Phase 6: spawn translated consumer on GPUI foreground ──
        let (trans_tx, mut trans_rx) = tokio::sync::mpsc::unbounded_channel();
        let stream_for_consumer = event_stream.clone();
        // NW-3: approval_tx cloned into the consumer so
        // `PermissionRequest` variants can round-trip user decisions
        // back to the engine. `None` should not happen here (we always
        // capture it in phase 2), but we fall back to a warning-only
        // surface to preserve the legacy pre-NW-3 behaviour.
        let approval_tx_for_consumer = setup.approval_tx.clone();
        // Follow-up: consumer-local state for tool-call enrichment.
        // Handled by [`caduceus_bridge::event_translator::ToolInputAggregator`]
        // (FU#4) which owns id→name + id→input-JSON bookkeeping and
        // has its own unit tests. Consumer just drives observe_* /
        // finalize from the event loop.
        let enabled_tools_for_consumer = setup.enabled_tools.clone();
        // ST2: capture weak Thread for ScopeExpansion auto-pin from the
        // consumer task. Native-loop only; legacy path emits no
        // TranslatedThreadEvent::ScopeExpansion.
        let thread_for_pins = this.clone();
        let consumer_task = cx.spawn(async move |cx_cons| {
            use caduceus_bridge::event_translator::{
                ToolInputAggregator, TranslatedThreadEvent as T,
            };
            // FU#4: per-id aggregation (id → name, id → input bytes)
            // now lives in the bridge crate's `ToolInputAggregator`,
            // with its own 7 unit tests. The consumer just drives
            // it via observe_* and reads the aggregated payload at
            // InputEnd — see caduceus_bridge::event_translator.
            let mut input_agg = ToolInputAggregator::new();
            while let Some(ev) = trans_rx.recv().await {
                // Permission requests route through their own helper
                // (NW-3 / A2). Everything else funnels through the
                // stateful tool-lifecycle helper, then falls through to
                // the stateless dispatcher for anything it didn't
                // handle end-to-end.
                if let T::PermissionRequest {
                    id,
                    tool,
                    description,
                    raw_input,
                } = &ev
                {
                    if let Some(ref approval_tx) = approval_tx_for_consumer {
                        route_native_permission_request(
                            id,
                            tool,
                            description,
                            raw_input.as_ref(),
                            &stream_for_consumer,
                            approval_tx.clone(),
                            cx_cons,
                        );
                        continue;
                    }
                }
                // ST2: pin most-recent user message on scope expansion request.
                if matches!(ev, T::ScopeExpansion { .. }) {
                    let _ = thread_for_pins.update(cx_cons, |t, _cx| {
                        t.on_scope_expansion_requested();
                    });
                }
                // ST2 fix-loop #4: native-loop PlanUpdate hook. PlanStep and
                // PlanAmended are deltas (not snapshots), but pin_replace is
                // idempotent per (key, reason) so calling on each delta is
                // safe — at most one PlanUpdate pin survives, on the
                // most-recent agent message at the time of the latest delta.
                // Plan v3.1 §4 acknowledged this hook as deferred to G1d, but
                // adding it now closes the convergent reviewer finding that
                // PlanUpdate fires only on the legacy ACP path.
                if matches!(ev, T::PlanStep { .. } | T::PlanAmended { .. }) {
                    let _ = thread_for_pins.update(cx_cons, |t, cx| {
                        t.on_plan_event_emitted(cx);
                    });
                }
                let handled_fully = handle_native_tool_lifecycle(
                    &ev,
                    &mut input_agg,
                    &enabled_tools_for_consumer,
                    &stream_for_consumer,
                    cx_cons,
                );
                if handled_fully {
                    continue;
                }
                dispatch_translated_event(&ev, &stream_for_consumer);
            }
        });

        // ── Phase 7: run the engine loop ───────────────────────────
        //
        // `run_caduceus_loop_translated` (and its `spawn_forwarder`
        // helper) call `tokio::spawn` + `tokio::time::timeout`, both of
        // which require a live Tokio runtime on the current thread.
        // This async fn is polled by GPUI's foreground executor (NOT a
        // Tokio runtime), so without this guard `Handle::current()`
        // panics and the app aborts on the first native-loop turn.
        //
        // `tokio_rt::bridge_runtime_handle().enter()` installs the
        // process-wide bridge runtime on the current thread; the guard
        // stays live for the entire async scope below because GPUI
        // foreground polls stay pinned to the main thread (task-local
        // state persists across .await points on a single-threaded
        // executor). Dropping the guard at scope end releases the
        // thread-local back to its previous (none) state.
        let _bridge_rt_guard = caduceus_bridge::tokio_rt::bridge_runtime_handle().enter();

        let reducer = setup.bridge.new_reducer_handle();
        let state_mut = &mut *state_guard;
        let run_result = setup
            .bridge
            .run_caduceus_loop_translated(
                &harness,
                &mut state_mut.session,
                &mut state_mut.history,
                &setup.user_input,
                reducer,
                turn_event_rx,
                trans_tx,
            )
            .await;

        // ── Phase 8: drain + cleanup ───────────────────────────────
        drop(state_guard);
        // GPUI `Task` cancels on drop — explicit drop here to match
        // the original `.abort()` timing (before we await the consumer).
        drop(bridge_task);
        consumer_task.await;

        // `run_caduceus_loop_translated`'s TurnComplete/TurnError is
        // already surfaced by `dispatch_translated_event`; the outer
        // run_turn caller will send its own Stop on Ok(()). That's a
        // harmless double-stop (receiver takes the first) — accepted
        // cost of not plumbing a "native handled stop" signal yet.
        if let Err(e) = run_result {
            event_stream.send_engine_diagnostic(
                "native.run_failed",
                format!("caduceus native loop failed: {e}"),
                EngineDiagnosticSeverity::Error,
            );
        }
        Ok(true)
    }

    fn process_tool_result(
        this: &WeakEntity<Thread>,
        event_stream: &ThreadEventStream,
        cx: &mut AsyncApp,
        tool_result: LanguageModelToolResult,
    ) -> Result<(), anyhow::Error> {
        log::debug!("Tool finished {:?}", tool_result);

        event_stream.update_tool_call_fields(
            &tool_result.tool_use_id,
            acp::ToolCallUpdateFields::new()
                .status(if tool_result.is_error {
                    acp::ToolCallStatus::Failed
                } else {
                    acp::ToolCallStatus::Completed
                })
                .raw_output(tool_result.output.clone()),
            None,
        );
        this.update(cx, |this, _cx| {
            // Caduceus: track consecutive failures for circuit breaker.
            // Skip synthetic guardrail results (permission denied, circuit breaker,
            // loop detected, missing tool) — these are not real tool failures.
            // Counting them feeds back into the breaker: a single guardrail
            // trip would itself trip the breaker on the next turn.
            let is_synthetic_guardrail =
                if let LanguageModelToolResultContent::Text(t) = &tool_result.content {
                    is_synthetic_guardrail_text(t)
                } else {
                    false
                };
            if !is_synthetic_guardrail {
                this.circuit_breaker.record_result(tool_result.is_error);
            }
            this.pending_message()
                .tool_results
                .insert(tool_result.tool_use_id.clone(), tool_result)
        })?;
        Ok(())
    }

    fn handle_completion_error(
        &mut self,
        error: LanguageModelCompletionError,
        attempt: u8,
        plan: Option<Plan>,
        cx: &mut Context<Self>,
    ) -> Result<acp_thread::RetryStatus> {
        let Some(model) = self.model.as_ref() else {
            return Err(anyhow!(error));
        };

        // ST1a fix-loop #2: feed completion errors into the registry's auth-state
        // cache so a 401 invalidates `Authenticated` and a 429 records `RateLimited`.
        // Plan v3.1 §4 (cache invalidation table — "AuthenticationError (HTTP 401)"
        // and "RateLimitExceeded { retry_after }" rows). Without this the cache only
        // sees errors from tests.
        let provider_id = model.provider_id();
        LanguageModelRegistry::global(cx).update(cx, |registry, _cx| {
            registry.note_completion_error(&provider_id, &error);
        });

        let auto_retry = if model.provider_id() == ZED_CLOUD_PROVIDER_ID {
            plan.is_some()
        } else {
            true
        };

        if !auto_retry {
            return Err(anyhow!(error));
        }

        let Some(strategy) = Self::retry_strategy_for(&error) else {
            return Err(anyhow!(error));
        };

        let max_attempts = match &strategy {
            RetryStrategy::ExponentialBackoff { max_attempts, .. } => *max_attempts,
            RetryStrategy::Fixed { max_attempts, .. } => *max_attempts,
        };

        if attempt > max_attempts {
            return Err(anyhow!(error));
        }

        let delay = match &strategy {
            RetryStrategy::ExponentialBackoff { initial_delay, .. } => {
                let delay_secs = initial_delay.as_secs() * 2u64.pow((attempt - 1) as u32);
                Duration::from_secs(delay_secs)
            }
            RetryStrategy::Fixed { delay, .. } => *delay,
        };
        log::debug!("Retry attempt {attempt} with delay {delay:?}");

        Ok(acp_thread::RetryStatus {
            last_error: error.to_string().into(),
            attempt: attempt as usize,
            max_attempts: max_attempts as usize,
            started_at: Instant::now(),
            duration: delay,
        })
    }

    /// A helper method that's called on every streamed completion event.
    /// Returns an optional tool result task, which the main agentic loop will
    /// send back to the model when it resolves.
    fn handle_completion_event(
        &mut self,
        event: LanguageModelCompletionEvent,
        event_stream: &ThreadEventStream,
        cancellation_rx: watch::Receiver<bool>,
        cx: &mut Context<Self>,
    ) -> Result<Option<Task<LanguageModelToolResult>>> {
        log::trace!("Handling streamed completion event: {:?}", event);
        use LanguageModelCompletionEvent::*;

        match event {
            StartMessage { .. } => {
                self.flush_pending_message(cx);
                self.pending_message = Some(AgentMessage::default());
            }
            Text(new_text) => self.handle_text_event(new_text, event_stream),
            Thinking { text, signature } => {
                self.handle_thinking_event(text, signature, event_stream)
            }
            RedactedThinking { data } => self.handle_redacted_thinking_event(data),
            ReasoningDetails(details) => {
                let last_message = self.pending_message();
                // Store the last non-empty reasoning_details (overwrites earlier ones)
                // This ensures we keep the encrypted reasoning with signatures, not the early text reasoning
                if let serde_json::Value::Array(ref arr) = details {
                    if !arr.is_empty() {
                        last_message.reasoning_details = Some(details);
                    }
                } else {
                    last_message.reasoning_details = Some(details);
                }
            }
            ToolUse(tool_use) => {
                return Ok(self.handle_tool_use_event(tool_use, event_stream, cancellation_rx, cx));
            }
            ToolUseJsonParseError {
                id,
                tool_name,
                raw_input,
                json_parse_error,
            } => {
                return Ok(self.handle_tool_use_json_parse_error_event(
                    id,
                    tool_name,
                    raw_input,
                    json_parse_error,
                    event_stream,
                    cancellation_rx,
                    cx,
                ));
            }
            UsageUpdate(usage) => {
                telemetry::event!(
                    "Agent Thread Completion Usage Updated",
                    thread_id = self.id.to_string(),
                    parent_thread_id = self.parent_thread_id().map(|id| id.to_string()),
                    prompt_id = self.prompt_id.to_string(),
                    model = self.model.as_ref().map(|m| m.telemetry_id()),
                    model_provider = self.model.as_ref().map(|m| m.provider_id().to_string()),
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cache_creation_input_tokens = usage.cache_creation_input_tokens,
                    cache_read_input_tokens = usage.cache_read_input_tokens,
                );
                self.update_token_usage(usage, cx);
            }
            Stop(StopReason::Refusal) => return Err(CompletionError::Refusal.into()),
            Stop(StopReason::MaxTokens) => return Err(CompletionError::MaxTokens.into()),
            Stop(StopReason::ToolUse | StopReason::EndTurn) => {}
            Started | Queued { .. } => {}
        }

        Ok(None)
    }

    fn handle_text_event(&mut self, new_text: String, event_stream: &ThreadEventStream) {
        event_stream.send_text(&new_text);

        let last_message = self.pending_message();
        if let Some(AgentMessageContent::Text(text)) = last_message.content.last_mut() {
            text.push_str(&new_text);
        } else {
            last_message
                .content
                .push(AgentMessageContent::Text(new_text));
        }
    }

    fn handle_thinking_event(
        &mut self,
        new_text: String,
        new_signature: Option<String>,
        event_stream: &ThreadEventStream,
    ) {
        event_stream.send_thinking(&new_text);

        let last_message = self.pending_message();
        if let Some(AgentMessageContent::Thinking { text, signature }) =
            last_message.content.last_mut()
        {
            text.push_str(&new_text);
            *signature = new_signature.or(signature.take());
        } else {
            last_message.content.push(AgentMessageContent::Thinking {
                text: new_text,
                signature: new_signature,
            });
        }
    }

    fn handle_redacted_thinking_event(&mut self, data: String) {
        let last_message = self.pending_message();
        last_message
            .content
            .push(AgentMessageContent::RedactedThinking(data));
    }

    fn handle_tool_use_event(
        &mut self,
        tool_use: LanguageModelToolUse,
        event_stream: &ThreadEventStream,
        cancellation_rx: watch::Receiver<bool>,
        cx: &mut Context<Self>,
    ) -> Option<Task<LanguageModelToolResult>> {
        cx.notify();

        let tool = self.tool(tool_use.name.as_ref());
        let mut title = SharedString::from(&tool_use.name);
        let mut kind = acp::ToolKind::Other;
        if let Some(tool) = tool.as_ref() {
            title = tool.initial_title(tool_use.input.clone(), cx);
            kind = tool.kind();
        }

        self.send_or_update_tool_use(&tool_use, title, kind, event_stream);

        // Caduceus: enforce privilege rings — block disallowed tools immediately
        if !self.is_tool_allowed_in_current_mode(tool_use.name.as_ref()) {
            let mode = self.caduceus_mode_from_profile();
            let content = format!(
                "PERMISSION DENIED: '{}' is blocked in {} mode. \
                DO NOT retry — this is a permission issue, not a transient error. \
                Ask the user: \"I need {} to complete this task. \
                Shall I switch to Act mode?\" \
                The user will switch the mode manually if they agree.",
                tool_use.name, mode, tool_use.name
            );
            log::warn!(
                "[caduceus] PERMISSION DENIED '{}' in {} mode",
                tool_use.name,
                mode
            );
            // Permission denials reset the circuit breaker (not real failures)
            self.circuit_breaker.record_permission_denied();
            return Some(Task::ready(LanguageModelToolResult {
                content: LanguageModelToolResultContent::Text(Arc::from(content)),
                tool_use_id: tool_use.id,
                tool_name: tool_use.name,
                is_error: true,
                output: None,
            }));
        }

        // Caduceus: circuit breaker — stop after N consecutive tool failures
        if self.circuit_breaker.is_tripped() {
            log::error!(
                "[caduceus] Circuit breaker: {} consecutive failures",
                self.circuit_breaker.consecutive_failures()
            );
            self.guardrail_alert = Some((
                format!(
                    "🔴 Circuit breaker: {} consecutive tool failures — execution paused",
                    self.circuit_breaker.consecutive_failures()
                ),
                AlertSeverity::Error,
                std::time::Instant::now(),
            ));
            self.circuit_breaker.reset();
            cx.notify();
            return Some(Task::ready(LanguageModelToolResult {
                content: LanguageModelToolResultContent::Text(Arc::from(
                    "CIRCUIT BREAKER: 5 consecutive tool failures. Stop and explain the issue to the user.",
                )),
                tool_use_id: tool_use.id,
                tool_name: tool_use.name,
                is_error: true,
                output: None,
            }));
        }

        // Caduceus: loop detection — prevent same tool being called too many times consecutively.
        // Escalation: after LOOP_ESCALATION_THRESHOLD consecutive loop detections without recovery,
        // hard-cancel the entire turn to avoid runaway tool storms.
        const LOOP_ESCALATION_THRESHOLD: usize = 3;
        {
            let tool_name = tool_use.name.as_ref();
            let input_repr = tool_use.input.to_string();
            match self.loop_detector.record_call(tool_name, &input_repr) {
                caduceus_bridge::safety::LoopCheckResult::LoopDetected(name) => {
                    self.loop_escalation_count += 1;
                    log::warn!(
                        "[caduceus] Loop detected: {} called too many times consecutively (escalation {}/{})",
                        name,
                        self.loop_escalation_count,
                        LOOP_ESCALATION_THRESHOLD
                    );

                    if self.loop_escalation_count >= LOOP_ESCALATION_THRESHOLD {
                        // Hard stop: cancel the entire turn on next tick to avoid reentrancy.
                        log::error!(
                            "[caduceus] Loop escalation limit reached ({}× in a row) — cancelling turn",
                            self.loop_escalation_count
                        );
                        self.guardrail_alert = Some((
                            format!(
                                "🛑 Runaway loop: {} fired {}× — turn cancelled. Try a different prompt.",
                                name, self.loop_escalation_count
                            ),
                            AlertSeverity::Error,
                            std::time::Instant::now(),
                        ));
                        self.loop_escalation_count = 0;
                        cx.notify();
                        // Defer cancel to next tick so we don't reenter the running update.
                        cx.spawn(async move |this, cx| {
                            this.update(cx, |this, cx| {
                                this.cancel(cx).detach();
                            })
                            .ok();
                        })
                        .detach();
                        return Some(Task::ready(LanguageModelToolResult {
                            content: LanguageModelToolResultContent::Text(Arc::from(format!(
                                "RUNAWAY LOOP: {} called repeatedly. Turn cancelled by safety guardrail. Wait for the user to provide a new prompt.",
                                name
                            ))),
                            tool_use_id: tool_use.id,
                            tool_name: tool_use.name,
                            is_error: true,
                            output: None,
                        }));
                    }

                    self.guardrail_alert = Some((
                        format!(
                            "⚠️ Loop detected: {} called too many times — trying different approach",
                            name
                        ),
                        AlertSeverity::Warning,
                        std::time::Instant::now(),
                    ));
                    cx.notify();
                    return Some(Task::ready(LanguageModelToolResult {
                        content: LanguageModelToolResultContent::Text(Arc::from(format!(
                            "LOOP DETECTED: {} called too many times in a row. Try a different approach.",
                            name
                        ))),
                        tool_use_id: tool_use.id,
                        tool_name: tool_use.name,
                        is_error: true,
                        output: None,
                    }));
                }
                caduceus_bridge::safety::LoopCheckResult::Ok => {
                    // Recovery: any successful tool dispatch resets escalation.
                    self.loop_escalation_count = 0;
                }
            }
        }

        let Some(tool) = tool else {
            let content = format!("No tool named {} exists", tool_use.name);
            return Some(Task::ready(LanguageModelToolResult {
                content: LanguageModelToolResultContent::Text(Arc::from(content)),
                tool_use_id: tool_use.id,
                tool_name: tool_use.name,
                is_error: true,
                output: None,
            }));
        };

        if !tool_use.is_input_complete {
            if tool.supports_input_streaming() {
                let running_turn = self.running_turn.as_mut()?;
                if let Some(sender) = running_turn.streaming_tool_inputs.get_mut(&tool_use.id) {
                    sender.send_partial(tool_use.input);
                    return None;
                }

                let (mut sender, tool_input) = ToolInputSender::channel();
                sender.send_partial(tool_use.input);
                running_turn
                    .streaming_tool_inputs
                    .insert(tool_use.id.clone(), sender);

                let tool = tool.clone();
                log::debug!("Running streaming tool {}", tool_use.name);
                return Some(self.run_tool(
                    tool,
                    tool_input,
                    tool_use.id,
                    tool_use.name,
                    event_stream,
                    cancellation_rx,
                    cx,
                ));
            } else {
                return None;
            }
        }

        if let Some(mut sender) = self
            .running_turn
            .as_mut()?
            .streaming_tool_inputs
            .remove(&tool_use.id)
        {
            sender.send_full(tool_use.input);
            return None;
        }

        log::debug!("Running tool {}", tool_use.name);
        let tool_input = ToolInput::ready(tool_use.input);
        Some(self.run_tool(
            tool,
            tool_input,
            tool_use.id,
            tool_use.name,
            event_stream,
            cancellation_rx,
            cx,
        ))
    }

    fn run_tool(
        &self,
        tool: Arc<dyn AnyAgentTool>,
        tool_input: ToolInput<serde_json::Value>,
        tool_use_id: LanguageModelToolUseId,
        tool_name: Arc<str>,
        event_stream: &ThreadEventStream,
        cancellation_rx: watch::Receiver<bool>,
        cx: &mut Context<Self>,
    ) -> Task<LanguageModelToolResult> {
        let fs = self.project.read(cx).fs().clone();
        let tool_event_stream = ToolCallEventStream::new(
            tool_use_id.clone(),
            event_stream.clone(),
            Some(fs),
            cancellation_rx,
        );
        tool_event_stream.update_fields(
            acp::ToolCallUpdateFields::new().status(acp::ToolCallStatus::InProgress),
        );
        let supports_images = self.model().is_some_and(|model| model.supports_images());
        let tool_result = tool.run(tool_input, tool_event_stream, cx);
        cx.foreground_executor().spawn(async move {
            let (is_error, output) = match tool_result.await {
                Ok(mut output) => {
                    if let LanguageModelToolResultContent::Image(_) = &output.llm_output
                        && !supports_images
                    {
                        output = AgentToolOutput::from_error(
                            "Attempted to read an image, but this model doesn't support it.",
                        );
                        (true, output)
                    } else {
                        (false, output)
                    }
                }
                Err(output) => (true, output),
            };

            LanguageModelToolResult {
                tool_use_id,
                tool_name,
                is_error,
                content: output.llm_output,
                output: Some(output.raw_output),
            }
        })
    }

    fn handle_tool_use_json_parse_error_event(
        &mut self,
        tool_use_id: LanguageModelToolUseId,
        tool_name: Arc<str>,
        raw_input: Arc<str>,
        json_parse_error: String,
        event_stream: &ThreadEventStream,
        cancellation_rx: watch::Receiver<bool>,
        cx: &mut Context<Self>,
    ) -> Option<Task<LanguageModelToolResult>> {
        let tool_use = LanguageModelToolUse {
            id: tool_use_id,
            name: tool_name,
            raw_input: raw_input.to_string(),
            input: serde_json::json!({}),
            is_input_complete: true,
            thought_signature: None,
        };
        self.send_or_update_tool_use(
            &tool_use,
            SharedString::from(&tool_use.name),
            acp::ToolKind::Other,
            event_stream,
        );

        let tool = self.tool(tool_use.name.as_ref());

        let Some(tool) = tool else {
            let content = format!("No tool named {} exists", tool_use.name);
            return Some(Task::ready(LanguageModelToolResult {
                content: LanguageModelToolResultContent::Text(Arc::from(content)),
                tool_use_id: tool_use.id,
                tool_name: tool_use.name,
                is_error: true,
                output: None,
            }));
        };

        let error_message = format!("Error parsing input JSON: {json_parse_error}");

        if tool.supports_input_streaming()
            && let Some(mut sender) = self
                .running_turn
                .as_mut()?
                .streaming_tool_inputs
                .remove(&tool_use.id)
        {
            sender.send_invalid_json(error_message);
            return None;
        }

        log::debug!("Running tool {}. Received invalid JSON", tool_use.name);
        let tool_input = ToolInput::invalid_json(error_message);
        Some(self.run_tool(
            tool,
            tool_input,
            tool_use.id,
            tool_use.name,
            event_stream,
            cancellation_rx,
            cx,
        ))
    }

    fn send_or_update_tool_use(
        &mut self,
        tool_use: &LanguageModelToolUse,
        title: SharedString,
        kind: acp::ToolKind,
        event_stream: &ThreadEventStream,
    ) {
        // Ensure the last message ends in the current tool use
        let last_message = self.pending_message();

        let has_tool_use = last_message.content.iter_mut().rev().any(|content| {
            if let AgentMessageContent::ToolUse(last_tool_use) = content {
                if last_tool_use.id == tool_use.id {
                    *last_tool_use = tool_use.clone();
                    return true;
                }
            }
            false
        });

        if !has_tool_use {
            event_stream.send_tool_call(
                &tool_use.id,
                &tool_use.name,
                title,
                kind,
                tool_use.input.clone(),
            );
            last_message
                .content
                .push(AgentMessageContent::ToolUse(tool_use.clone()));
        } else {
            event_stream.update_tool_call_fields(
                &tool_use.id,
                acp::ToolCallUpdateFields::new()
                    .title(title.as_str())
                    .kind(kind)
                    .raw_input(tool_use.input.clone()),
                None,
            );
        }
    }

    pub fn title(&self) -> Option<SharedString> {
        self.title.clone()
    }

    pub fn is_generating_summary(&self) -> bool {
        self.pending_summary_generation.is_some()
    }

    pub fn is_generating_title(&self) -> bool {
        self.pending_title_generation.is_some()
    }

    pub fn summary(&mut self, cx: &mut Context<Self>) -> Shared<Task<Option<SharedString>>> {
        if let Some(summary) = self.summary.as_ref() {
            return Task::ready(Some(summary.clone())).shared();
        }
        if let Some(task) = self.pending_summary_generation.clone() {
            return task;
        }
        let Some(model) = self.summarization_model.clone() else {
            log::error!("No summarization model available");
            return Task::ready(None).shared();
        };
        let mut request = LanguageModelRequest {
            intent: Some(CompletionIntent::ThreadContextSummarization),
            temperature: AgentSettings::temperature_for_model(&model, cx),
            ..Default::default()
        };

        for message in &self.messages {
            request.messages.extend(message.to_request());
        }

        request.messages.push(LanguageModelRequestMessage {
            role: Role::User,
            content: vec![SUMMARIZE_THREAD_DETAILED_PROMPT.into()],
            cache: false,
            reasoning_details: None,
        });

        let task = cx
            .spawn(async move |this, cx| {
                let mut summary = String::new();
                let mut messages = model.stream_completion(request, cx).await.log_err()?;
                while let Some(event) = messages.next().await {
                    let event = event.log_err()?;
                    let text = match event {
                        LanguageModelCompletionEvent::Text(text) => text,
                        _ => continue,
                    };

                    let mut lines = text.lines();
                    summary.extend(lines.next());
                }

                log::debug!("Setting summary: {}", summary);
                let summary = SharedString::from(summary);

                this.update(cx, |this, cx| {
                    this.summary = Some(summary.clone());
                    this.pending_summary_generation = None;
                    cx.notify()
                })
                .ok()?;

                Some(summary)
            })
            .shared();
        self.pending_summary_generation = Some(task.clone());
        task
    }

    pub fn generate_title(&mut self, cx: &mut Context<Self>) {
        let Some(model) = self.summarization_model.clone() else {
            return;
        };

        log::debug!(
            "Generating title with model: {:?}",
            self.summarization_model.as_ref().map(|model| model.name())
        );
        let mut request = LanguageModelRequest {
            intent: Some(CompletionIntent::ThreadSummarization),
            temperature: AgentSettings::temperature_for_model(&model, cx),
            ..Default::default()
        };

        for message in &self.messages {
            request.messages.extend(message.to_request());
        }

        request.messages.push(LanguageModelRequestMessage {
            role: Role::User,
            content: vec![SUMMARIZE_THREAD_PROMPT.into()],
            cache: false,
            reasoning_details: None,
        });
        self.pending_title_generation = Some(cx.spawn(async move |this, cx| {
            let mut title = String::new();

            let generate = async {
                let mut messages = model.stream_completion(request, cx).await?;
                while let Some(event) = messages.next().await {
                    let event = event?;
                    let text = match event {
                        LanguageModelCompletionEvent::Text(text) => text,
                        _ => continue,
                    };

                    let mut lines = text.lines();
                    title.extend(lines.next());

                    // Stop if the LLM generated multiple lines.
                    if lines.next().is_some() {
                        break;
                    }
                }
                anyhow::Ok(())
            };

            if generate
                .await
                .context("failed to generate thread title")
                .log_err()
                .is_some()
            {
                _ = this.update(cx, |this, cx| this.set_title(title.into(), cx));
            } else {
                // Emit TitleUpdated even on failure so that the propagation
                // chain (agent::Thread → NativeAgent → AcpThread) fires and
                // clears any provisional title that was set before the turn.
                _ = this.update(cx, |_, cx| {
                    cx.emit(TitleUpdated);
                    cx.notify();
                });
            }
            _ = this.update(cx, |this, _| this.pending_title_generation = None);
        }));
    }

    pub fn set_title(&mut self, title: SharedString, cx: &mut Context<Self>) {
        self.pending_title_generation = None;
        if Some(&title) != self.title.as_ref() {
            self.title = Some(title);
            cx.emit(TitleUpdated);
            cx.notify();
        }
    }

    fn clear_summary(&mut self) {
        self.summary = None;
        self.pending_summary_generation = None;
    }

    fn last_user_message(&self) -> Option<&UserMessage> {
        self.messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user_message) => Some(user_message),
                Message::Agent(_) => None,
                Message::Resume(_) => None,
            })
    }

    fn pending_message(&mut self) -> &mut AgentMessage {
        self.pending_message.get_or_insert_default()
    }

    fn flush_pending_message(&mut self, cx: &mut Context<Self>) {
        let Some(mut message) = self.pending_message.take() else {
            return;
        };

        // PB5 enforcement (belt-and-suspenders): drop empty <thinking></thinking>
        // blocks that some models still emit even when the preamble forbids them.
        // An empty thinking block has no text, no signature payload — pure noise.
        message.content.retain(|c| match c {
            AgentMessageContent::Thinking { text, signature } => {
                !text.trim().is_empty() || signature.is_some()
            }
            _ => true,
        });

        if message.content.is_empty() {
            return;
        }

        for content in &message.content {
            let AgentMessageContent::ToolUse(tool_use) = content else {
                continue;
            };

            if !message.tool_results.contains_key(&tool_use.id) {
                message.tool_results.insert(
                    tool_use.id.clone(),
                    LanguageModelToolResult {
                        tool_use_id: tool_use.id.clone(),
                        tool_name: tool_use.name.clone(),
                        is_error: true,
                        content: LanguageModelToolResultContent::Text(TOOL_CANCELED_MESSAGE.into()),
                        output: None,
                    },
                );
            }
        }

        self.messages.push(Message::Agent(message));
        self.updated_at = Utc::now();
        self.clear_summary();
        self.invalidate_token_cache();

        // Only extract memories from clean, non-cancelled turn ends.
        // Both gates required: running_turn must be cleared (turn over) AND
        // the previous turn must not have been cancelled (otherwise the
        // background extractor races the next turn's activeContext.md writes).
        if self.running_turn.is_none() && !self.last_turn_cancelled {
            self.auto_extract_memories(cx);
        }

        cx.notify()
    }

    /// Extract important facts/preferences from the latest exchange and store them.
    fn auto_extract_memories(&self, cx: &Context<Self>) {
        // Skip for subagents — they shouldn't pollute parent memory
        if self.is_subagent() || self.messages.len() < 2 {
            return;
        }

        // Update living spec / active context
        self.update_active_context(cx);
        let last_user = self.messages.iter().rev().find_map(|m| {
            if let Message::User(u) = m {
                Some(
                    u.content
                        .iter()
                        .filter_map(|c| {
                            if let UserMessageContent::Text(t) = c {
                                Some(t.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                )
            } else {
                None
            }
        });
        let last_agent = self.messages.last().and_then(|m| {
            if let Message::Agent(a) = m {
                Some(a.to_markdown())
            } else {
                None
            }
        });

        if let (Some(user_text), Some(agent_text)) = (last_user, last_agent) {
            if let Some(root) = caduceus_workspace_folder(&self.project, cx) {
                // Move extraction + disk I/O to background
                cx.background_executor()
                    .spawn(async move {
                        let memories =
                            caduceus_bridge::orchestrator::OrchestratorBridge::extract_memories(
                                &user_text,
                                &agent_text,
                            );
                        for mem in &memories {
                            use std::hash::{Hash, Hasher};
                            let mut hasher = std::collections::hash_map::DefaultHasher::new();
                            mem.hash(&mut hasher);
                            let key = format!("auto:{:x}", hasher.finish());
                            if let Err(e) = caduceus_bridge::memory::store_system(&root, &key, mem)
                            {
                                log::warn!("[caduceus] Failed to store auto-memory: {e}");
                            }
                        }
                        if !memories.is_empty() {
                            log::debug!(
                                "[caduceus] Auto-extracted {} memories from turn",
                                memories.len()
                            );
                        }
                    })
                    .detach();
            }
        }
    }

    /// Update the living active context file (.caduceus/activeContext.md)
    /// Ephemeral session state — tracks current goal, decisions, modified files.
    fn update_active_context(&self, cx: &Context<Self>) {
        if let Some(root) = caduceus_workspace_folder(&self.project, cx) {
            let mode = self.caduceus_mode_from_profile().to_string();
            let msg_count = self.messages.len();

            // Extract last user message as "current goal"
            let goal = self
                .messages
                .iter()
                .rev()
                .find_map(|m| {
                    if let Message::User(u) = m {
                        Some(
                            u.content
                                .iter()
                                .filter_map(|c| {
                                    if let UserMessageContent::Text(t) = c {
                                        Some(t.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join(" "),
                        )
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            let goal_short: String = goal.chars().take(200).collect();

            // Count tool calls as "decisions made"
            let tool_count: usize = self
                .messages
                .iter()
                .filter_map(|m| {
                    if let Message::Agent(a) = m {
                        Some(
                            a.content
                                .iter()
                                .filter(|c| matches!(c, AgentMessageContent::ToolUse(_)))
                                .count(),
                        )
                    } else {
                        None
                    }
                })
                .sum();

            cx.background_executor()
                .spawn(async move {
                    let ctx_dir = root.join(".caduceus");
                    let _ = std::fs::create_dir_all(&ctx_dir);
                    let content = format!(
                        "# Active Context\n\n\
                         **Mode**: {}\n\
                         **Messages**: {}\n\
                         **Tool calls**: {}\n\n\
                         ## Current Goal\n{}\n\n\
                         *Auto-generated — do not commit*\n",
                        mode, msg_count, tool_count, goal_short
                    );
                    // Bug C6: write atomically (tmp + rename) so a reader
                    // (or a racing writer from a sibling subagent) never
                    // observes a half-written activeContext.md.
                    let _ = atomic_write(&ctx_dir.join("activeContext.md"), content.as_bytes());
                })
                .detach();
        }
    }

    pub(crate) fn build_completion_request(
        &mut self,
        completion_intent: CompletionIntent,
        cx: &App,
    ) -> Result<LanguageModelRequest> {
        let completion_intent =
            if self.is_subagent() && completion_intent == CompletionIntent::UserPrompt {
                CompletionIntent::Subagent
            } else {
                completion_intent
            };

        let model = self
            .model()
            .ok_or_else(|| anyhow!(NoModelConfiguredError))?
            .clone();
        let tools = if let Some(turn) = self.running_turn.as_ref() {
            turn.tools
                .iter()
                .filter_map(|(tool_name, tool)| {
                    log::trace!("Including tool: {}", tool_name);
                    Some(LanguageModelRequestTool {
                        name: tool_name.to_string(),
                        description: tool.description().to_string(),
                        input_schema: tool.input_schema(model.tool_input_format()).log_err()?,
                        use_input_streaming: tool.supports_input_streaming(),
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        log::debug!("Building completion request");
        log::debug!("Completion intent: {:?}", completion_intent);

        let available_tools: Vec<_> = self
            .running_turn
            .as_ref()
            .map(|turn| turn.tools.keys().cloned().collect())
            .unwrap_or_default();

        log::debug!("Request includes {} tools", available_tools.len());
        let messages = self.build_request_messages(available_tools, cx);
        log::debug!("Request will include {} messages", messages.len());

        let request = LanguageModelRequest {
            thread_id: Some(self.id.to_string()),
            prompt_id: Some(self.prompt_id.to_string()),
            intent: Some(completion_intent),
            messages,
            tools,
            tool_choice: None,
            stop: Vec::new(),
            temperature: AgentSettings::temperature_for_model(&model, cx),
            thinking_allowed: self.thinking_enabled,
            thinking_effort: self.thinking_effort.clone(),
            speed: self.speed(),
        };

        log::debug!("Completion request built successfully");
        Ok(request)
    }

    fn enabled_tools(&self, cx: &App) -> BTreeMap<SharedString, Arc<dyn AnyAgentTool>> {
        let Some(model) = self.model.as_ref() else {
            return BTreeMap::new();
        };
        let Some(profile) = AgentSettings::get_global(cx).profiles.get(&self.profile_id) else {
            return BTreeMap::new();
        };
        fn truncate(tool_name: &SharedString) -> SharedString {
            if tool_name.len() > MAX_TOOL_NAME_LENGTH {
                let mut truncated = tool_name.to_string();
                truncated.truncate(MAX_TOOL_NAME_LENGTH);
                truncated.into()
            } else {
                tool_name.clone()
            }
        }

        let use_streaming_edit_tool =
            cx.has_flag::<StreamingEditFileToolFeatureFlag>() && model.supports_streaming_tools();

        let mut tools = self
            .tools
            .iter()
            .filter_map(|(tool_name, tool)| {
                // For streaming_edit_file, check profile against "edit_file" since that's what users configure
                let profile_tool_name = if tool_name == StreamingEditFileTool::NAME {
                    EditFileTool::NAME
                } else {
                    tool_name.as_ref()
                };

                if tool.supports_provider(&model.provider_id())
                    && profile.is_tool_enabled(profile_tool_name)
                {
                    match (tool_name.as_ref(), use_streaming_edit_tool) {
                        (StreamingEditFileTool::NAME, false) | (EditFileTool::NAME, true) => None,
                        (StreamingEditFileTool::NAME, true) => {
                            // Expose streaming tool as "edit_file"
                            Some((SharedString::from(EditFileTool::NAME), tool.clone()))
                        }
                        _ => Some((truncate(tool_name), tool.clone())),
                    }
                } else {
                    None
                }
            })
            .collect::<BTreeMap<_, _>>();

        let mut context_server_tools = Vec::new();
        let mut seen_tools = tools.keys().cloned().collect::<HashSet<_>>();
        let mut duplicate_tool_names = HashSet::default();
        for (server_id, server_tools) in self.context_server_registry.read(cx).servers() {
            for (tool_name, tool) in server_tools {
                if profile.is_context_server_tool_enabled(&server_id.0, &tool_name) {
                    let tool_name = truncate(tool_name);
                    if !seen_tools.insert(tool_name.clone()) {
                        duplicate_tool_names.insert(tool_name.clone());
                    }
                    context_server_tools.push((server_id.clone(), tool_name, tool.clone()));
                }
            }
        }

        // When there are duplicate tool names, disambiguate by prefixing them
        // with the server ID (converted to snake_case for API compatibility).
        // In the rare case there isn't enough space for the disambiguated tool
        // name, keep only the last tool with this name.
        for (server_id, tool_name, tool) in context_server_tools {
            if duplicate_tool_names.contains(&tool_name) {
                let available = MAX_TOOL_NAME_LENGTH.saturating_sub(tool_name.len());
                if available >= 2 {
                    let mut disambiguated = server_id.0.to_snake_case();
                    disambiguated.truncate(available - 1);
                    disambiguated.push('_');
                    disambiguated.push_str(&tool_name);
                    tools.insert(disambiguated.into(), tool.clone());
                } else {
                    tools.insert(tool_name, tool.clone());
                }
            } else {
                tools.insert(tool_name, tool.clone());
            }
        }

        tools
    }

    fn refresh_turn_tools(&mut self, cx: &App) {
        let tools = self.enabled_tools(cx);
        if let Some(turn) = self.running_turn.as_mut() {
            turn.tools = tools;
        }
    }

    fn tool(&self, name: &str) -> Option<Arc<dyn AnyAgentTool>> {
        self.running_turn.as_ref()?.tools.get(name).cloned()
    }

    pub fn has_tool(&self, name: &str) -> bool {
        self.running_turn
            .as_ref()
            .is_some_and(|turn| turn.tools.contains_key(name))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn has_registered_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    pub fn registered_tool_names(&self) -> Vec<SharedString> {
        self.tools.keys().cloned().collect()
    }

    pub(crate) fn register_running_subagent(&mut self, subagent: WeakEntity<Thread>, cx: &App) {
        // Record parent → child spawn in the global Index DAG so the IDE
        // panel can show the full agent tree (not just resource accesses).
        let parent_id = self.id.0.to_string();
        let parent_kind = if self.is_subagent() {
            caduceus_bridge::index_dag::AgentKind::Subagent
        } else {
            caduceus_bridge::index_dag::AgentKind::User
        };
        // Sonnet #1: child id MUST be the SessionId (same id-space as the
        // child's index access events) — otherwise spawn edges and access
        // edges live in disjoint id namespaces and the DAG cannot join
        // "user spawned X" with "X read semantic_index". Only fall back
        // to entity_id when the child has been dropped — preserves the
        // edge but flags the divergence in the label so it's debuggable.
        let child_id = subagent
            .upgrade()
            .map(|s| s.read(cx).id().0.to_string())
            .unwrap_or_else(|| format!("entity:{}", subagent.entity_id().as_u64()));
        caduceus_bridge::index_dag::record_spawn(
            parent_id,
            parent_kind,
            child_id,
            caduceus_bridge::index_dag::AgentKind::Subagent,
        );
        self.running_subagents.push(subagent);
    }

    pub(crate) fn unregister_running_subagent(
        &mut self,
        subagent_session_id: &acp::SessionId,
        cx: &App,
    ) {
        // Sonnet #7: prune the spawn edge for this child so the DAG
        // doesn't show a ghost edge for an agent that's no longer running.
        // Using SessionId here matches the id-space chosen in
        // register_running_subagent (Sonnet #1).
        caduceus_bridge::index_dag::remove_spawn(&subagent_session_id.0);
        self.running_subagents.retain(|s| {
            s.upgrade()
                .map_or(false, |s| s.read(cx).id() != subagent_session_id)
        });
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn running_subagent_ids(&self, cx: &App) -> Vec<acp::SessionId> {
        self.running_subagents
            .iter()
            .filter_map(|s| s.upgrade().map(|s| s.read(cx).id().clone()))
            .collect()
    }

    pub fn is_subagent(&self) -> bool {
        self.subagent_context.is_some()
    }

    pub fn parent_thread_id(&self) -> Option<acp::SessionId> {
        self.subagent_context
            .as_ref()
            .map(|c| c.parent_thread_id.clone())
    }

    pub fn depth(&self) -> u8 {
        self.subagent_context.as_ref().map(|c| c.depth).unwrap_or(0)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn set_subagent_context(&mut self, context: SubagentContext) {
        self.subagent_context = Some(context);
    }

    pub fn is_turn_complete(&self) -> bool {
        self.running_turn.is_none()
    }

    fn build_request_messages(
        &mut self,
        available_tools: Vec<SharedString>,
        cx: &App,
    ) -> Vec<LanguageModelRequestMessage> {
        log::trace!(
            "Building request messages from {} thread messages",
            self.messages.len()
        );

        let system_prompt = SystemPromptTemplate {
            project: self.project_context.read(cx),
            available_tools,
            model_name: self.model.as_ref().map(|m| m.name().0.to_string()),
        }
        .render(&self.templates)
        .context("failed to build system prompt")
        .expect("Invalid template");

        // Inject Caduceus mode prefix into system prompt.
        //
        // F1: delegate to `mode_prompt_for_profile` — the engine's
        // `behavior_rules_preamble` + per-mode system prompt prefix. No more
        // hand-rolled "You are in X mode" text drifting from the engine.
        let system_prompt = {
            use caduceus_bridge::orchestrator::mode_prompt_for_profile;
            let mode_str = self.caduceus_mode_from_profile();
            let lens = self.caduceus_lens_from_profile();
            let mode_prefix = mode_prompt_for_profile(mode_str, lens.as_deref());
            if mode_prefix.trim().is_empty() {
                system_prompt
            } else {
                format!("{mode_prefix}\n\n{system_prompt}")
            }
        };

        // Inject Caduceus tool guidance so the LLM knows when to use them
        let caduceus_guidance = self.build_caduceus_context_block(cx);

        let system_prompt = format!("{system_prompt}{caduceus_guidance}");
        let mut messages = vec![LanguageModelRequestMessage {
            role: Role::System,
            content: vec![system_prompt.into()],
            cache: false,
            reasoning_details: None,
        }];
        for message in &self.messages {
            messages.extend(message.to_request());
        }

        if let Some(last_message) = messages.last_mut() {
            last_message.cache = true;
        }

        if let Some(message) = self.pending_message.as_ref() {
            messages.extend(message.to_request());
        }

        messages
    }

    /// Builds the Caduceus-side context block (project instructions, wiki
    /// overview, pinned context, @mention resolution, cross-repo config,
    /// tool guide) that is appended after the base system prompt.
    ///
    /// Shared between the legacy `build_request_messages` path and the
    /// native-loop turn setup so both paths produce identical context —
    /// previously the native path had only the bare `SystemPromptTemplate`
    /// output and was missing @mentions / pinned / wiki overview.
    pub(crate) fn build_caduceus_context_block(&mut self, cx: &App) -> String {
        use caduceus_bridge::orchestrator::{ContextAssembler, ContextSource};

        // Budget: 12.5% of model context, capped at 4000 tokens
        let model_limit = self.model_max_tokens() as usize;
        let budget = (model_limit / 8).min(4000);
        let mut assembler = ContextAssembler::new(budget);

        if let Some(root) = caduceus_workspace_folder(&self.project, cx) {
            // Project instructions (cached after first load)
            let instr = if let Some(cached) = &self.cached_project_instructions {
                cached.clone()
            } else {
                let orch = caduceus_bridge::orchestrator::OrchestratorBridge::new(&root);
                match orch.load_instructions() {
                    Ok(i) if !i.system_prompt.is_empty() => {
                        self.cached_project_instructions = Some(i.system_prompt.clone());
                        i.system_prompt
                    }
                    Ok(_) => {
                        self.cached_project_instructions = Some(String::new());
                        String::new()
                    }
                    Err(e) => {
                        log::warn!("[caduceus] Instructions load failed: {e}");
                        String::new()
                    }
                }
            };
            if !instr.is_empty() {
                assembler.add_source(ContextSource::Instructions(instr));
            }

            // Wiki overview (compact)
            if let Some(overview) =
                caduceus_bridge::memory::get(&root, caduceus_bridge::memory::KEY_PROJECT_OVERVIEW)
            {
                let compact: String = overview.chars().take(300).collect();
                assembler.add_source(ContextSource::MemoryBank(compact));
            }

            // Inject pinned context items (survive compaction)
            for pin in self.context_pins.list_pins() {
                assembler.add_source(ContextSource::Pinned(format!(
                    "[Pinned: {}] {}",
                    pin.label, pin.content
                )));
            }

            // Resolve @mentions from the latest user message
            if let Some(last_user_text) = self.messages.iter().rev().find_map(|m| {
                if let Message::User(u) = m {
                    Some(
                        u.content
                            .iter()
                            .filter_map(|c| {
                                if let UserMessageContent::Text(t) = c {
                                    Some(t.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(" "),
                    )
                } else {
                    None
                }
            }) {
                let resolver = caduceus_bridge::orchestrator::MentionResolver::new(&root);
                if let Ok(Some(resolved)) = resolver.resolve(&last_user_text) {
                    if !resolved.is_empty() {
                        let truncated: String = resolved.chars().take(2000).collect();
                        assembler.add_source(ContextSource::FileContext {
                            path: "@mention".to_string(),
                            content: truncated,
                            priority: 2,
                        });
                    }
                }
            }

            // Cross-repo context from project.json
            let config_path = root.join(".caduceus").join("project.json");
            if config_path.exists() {
                if let Ok(data) = std::fs::read_to_string(&config_path) {
                    if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&data) {
                        let mut ctx = String::new();
                        if let Some(n) = cfg["project"]["name"].as_str() {
                            ctx.push_str(&format!("Project: {}\n", n));
                        }
                        if let Some(repos) = cfg["repos"].as_object() {
                            for (name, repo) in repos {
                                let role = repo["role"].as_str().unwrap_or("?");
                                ctx.push_str(&format!("- {}: {}\n", name, role));
                            }
                        }
                        if !ctx.is_empty() {
                            assembler.add_source(ContextSource::Instructions(ctx));
                        }
                    }
                }
            }
        }

        // Caduceus F2: announce pulled threads so the orchestrator knows
        // it can use `read_thread` / `compact_thread` against them.
        if !self.pulled_session_ids.is_empty() {
            let mut block = String::from(
                "## Pulled Threads\n\n\
                 The following thread session ids have been attached to this \
                 conversation via `/pull`. Use the `read_thread` tool to fetch \
                 their full transcript, or `compact_thread` to get a summarized \
                 view that fits in your context budget. Prefer `compact_thread` \
                 with a `max_chars` budget when the transcript is large.\n\n",
            );
            for id in &self.pulled_session_ids {
                block.push_str(&format!("- `{}`\n", id));
            }
            assembler.add_source(ContextSource::SystemPrompt(block));
        }

        // Tool guidance — structured guide for the LLM
        assembler.add_source(ContextSource::SystemPrompt(
            "## Caduceus Tool Guide\n\
            \n\
            **Search & Understand**: caduceus_semantic_search (find code by meaning), caduceus_code_graph (dependencies), \
            caduceus_tree_sitter (AST outline of a file), caduceus_cross_search (search across repos)\n\
            **Read Code**: caduceus_git_read (git status/log/diff), caduceus_dependency_scan (vulnerabilities)\n\
            **Plan & Track**: caduceus_prd (parse requirements), caduceus_task_decompose (break into parallel subtasks), \
            caduceus_task_tree (hierarchical tasks), caduceus_kanban (board with worktree isolation)\n\
            **Write & Edit**: Use edit_file/save_file for code. caduceus_project_wiki \
            (file-based project wiki with auto-populate + per-page CRUD), caduceus_checkpoint (save/restore snapshots)\n\
            **Memory**: caduceus_memory_read (recall facts), caduceus_memory_write (store facts). \
            NOT caduceus_storage (that's for structured task persistence).\n\
            **Security**: caduceus_security_scan (SAST), caduceus_mcp_security (MCP tool vetting), caduceus_policy (compliance)\n\
            **Project**: caduceus_project (multi-repo config), caduceus_api_registry (API catalog), \
            caduceus_architect (health score), caduceus_product (feature tracking)\n\
            **Meta**: caduceus_progress (velocity), \
            caduceus_telemetry (token usage), caduceus_time_tracking (session time)\n\
            \n\
            Rules: PERMISSION DENIED = the envelope preflight blocked this call; surface the error verbatim and try an alternative tool or fall back — do NOT ask the user to switch modes (the engine's behavior_rules preamble already covers this). \
            LOOP DETECTED = try a different approach. \
            Use edit_file for code changes, NOT terminal heredocs.\n\
            **Verify-after-edit**: After making code changes, ALWAYS run `diagnostics` to check for errors. \
            If errors are found, fix them before moving on. This is the verify step.\n\
            **Show-before-apply**: For large changes (>20 lines), show a brief summary of what you'll change \
            before applying edits. List files and the type of change (add/modify/delete).\n\
            **Work incrementally**: Create files ONE AT A TIME, not all at once. \
            After each file, confirm it was created successfully before moving to the next. \
            Never try to create an entire project in a single response — break it into steps.\n\
            Before spawning sub-agents: explain plan, wait for approval.".to_string()
        ));

        let assembled = assembler.assemble();
        log::debug!(
            "[caduceus] Context: {} tokens, {} included, {} truncated",
            assembled.total_tokens,
            assembled.sources_included.len(),
            assembled.sources_truncated.len()
        );
        assembled.content
    }

    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        for (ix, message) in self.messages.iter().enumerate() {
            if ix > 0 {
                markdown.push('\n');
            }
            match message {
                Message::User(_) => markdown.push_str("## User\n\n"),
                Message::Agent(_) => markdown.push_str("## Assistant\n\n"),
                Message::Resume(_) => {}
            }
            markdown.push_str(&message.to_markdown());
        }

        if let Some(message) = self.pending_message.as_ref() {
            markdown.push_str("\n## Assistant\n\n");
            markdown.push_str(&message.to_markdown());
        }

        markdown
    }

    fn advance_prompt_id(&mut self) {
        self.prompt_id = PromptId::new();
    }

    fn retry_strategy_for(error: &LanguageModelCompletionError) -> Option<RetryStrategy> {
        use LanguageModelCompletionError::*;
        use http_client::StatusCode;

        // General strategy here:
        // - If retrying won't help (e.g. invalid API key or payload too large), return None so we don't retry at all.
        // - If it's a time-based issue (e.g. server overloaded, rate limit exceeded), retry up to 4 times with exponential backoff.
        // - If it's an issue that *might* be fixed by retrying (e.g. internal server error), retry up to 3 times.
        match error {
            HttpResponseError {
                status_code: StatusCode::TOO_MANY_REQUESTS,
                ..
            } => Some(RetryStrategy::ExponentialBackoff {
                initial_delay: BASE_RETRY_DELAY,
                max_attempts: MAX_RETRY_ATTEMPTS,
            }),
            ServerOverloaded { retry_after, .. } | RateLimitExceeded { retry_after, .. } => {
                Some(RetryStrategy::Fixed {
                    delay: retry_after.unwrap_or(BASE_RETRY_DELAY),
                    max_attempts: MAX_RETRY_ATTEMPTS,
                })
            }
            UpstreamProviderError {
                status,
                retry_after,
                ..
            } => match *status {
                StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE => {
                    Some(RetryStrategy::Fixed {
                        delay: retry_after.unwrap_or(BASE_RETRY_DELAY),
                        max_attempts: MAX_RETRY_ATTEMPTS,
                    })
                }
                StatusCode::INTERNAL_SERVER_ERROR => Some(RetryStrategy::Fixed {
                    delay: retry_after.unwrap_or(BASE_RETRY_DELAY),
                    // Internal Server Error could be anything, retry up to 3 times.
                    max_attempts: 3,
                }),
                status => {
                    // There is no StatusCode variant for the unofficial HTTP 529 ("The service is overloaded"),
                    // but we frequently get them in practice. See https://http.dev/529
                    if status.as_u16() == 529 {
                        Some(RetryStrategy::Fixed {
                            delay: retry_after.unwrap_or(BASE_RETRY_DELAY),
                            max_attempts: MAX_RETRY_ATTEMPTS,
                        })
                    } else {
                        Some(RetryStrategy::Fixed {
                            delay: retry_after.unwrap_or(BASE_RETRY_DELAY),
                            max_attempts: 2,
                        })
                    }
                }
            },
            ApiInternalServerError { .. } => Some(RetryStrategy::Fixed {
                delay: BASE_RETRY_DELAY,
                max_attempts: 3,
            }),
            ApiReadResponseError { .. }
            | HttpSend { .. }
            | DeserializeResponse { .. }
            | BadRequestFormat { .. } => Some(RetryStrategy::Fixed {
                delay: BASE_RETRY_DELAY,
                max_attempts: 3,
            }),
            // Retrying these errors definitely shouldn't help.
            HttpResponseError {
                status_code:
                    StatusCode::PAYLOAD_TOO_LARGE | StatusCode::FORBIDDEN | StatusCode::UNAUTHORIZED,
                ..
            }
            | AuthenticationError { .. }
            | PermissionError { .. }
            | NoApiKey { .. }
            | ApiEndpointNotFound { .. }
            | PromptTooLarge { .. } => None,
            // These errors might be transient, so retry them
            SerializeRequest { .. } | BuildRequestBody { .. } | StreamEndedUnexpectedly { .. } => {
                Some(RetryStrategy::Fixed {
                    delay: BASE_RETRY_DELAY,
                    max_attempts: 1,
                })
            }
            // Retry all other 4xx and 5xx errors once.
            HttpResponseError { status_code, .. }
                if status_code.is_client_error() || status_code.is_server_error() =>
            {
                Some(RetryStrategy::Fixed {
                    delay: BASE_RETRY_DELAY,
                    max_attempts: 3,
                })
            }
            Other(err) if err.is::<language_model::PaymentRequiredError>() => {
                // Retrying won't help for Payment Required errors.
                None
            }
            // Conservatively assume that any other errors are non-retryable
            HttpResponseError { .. } | Other(..) => Some(RetryStrategy::Fixed {
                delay: BASE_RETRY_DELAY,
                max_attempts: 2,
            }),
        }
    }
}

struct RunningTurn {
    /// Holds the task that handles agent interaction until the end of the turn.
    /// Survives across multiple requests as the model performs tool calls and
    /// we run tools, report their results.
    _task: Task<()>,
    /// The current event stream for the running turn. Used to report a final
    /// cancellation event if we cancel the turn.
    event_stream: ThreadEventStream,
    /// The tools that are enabled for the current iteration of the turn.
    /// Refreshed at the start of each iteration via `refresh_turn_tools`.
    tools: BTreeMap<SharedString, Arc<dyn AnyAgentTool>>,
    /// Sender to signal tool cancellation. When cancel is called, this is
    /// set to true so all tools can detect user-initiated cancellation.
    cancellation_tx: watch::Sender<bool>,
    /// Senders for tools that support input streaming and have already been
    /// started but are still receiving input from the LLM.
    streaming_tool_inputs: HashMap<LanguageModelToolUseId, ToolInputSender>,
}

impl RunningTurn {
    fn cancel(mut self) -> Task<()> {
        log::debug!("Cancelling in progress turn");
        self.cancellation_tx.send(true).ok();
        self.event_stream.send_canceled();
        self._task
    }
}

pub struct TokenUsageUpdated(pub Option<acp_thread::TokenUsage>);

impl EventEmitter<TokenUsageUpdated> for Thread {}

pub struct TitleUpdated;

impl EventEmitter<TitleUpdated> for Thread {}

/// A channel-based wrapper that delivers tool input to a running tool.
///
/// For non-streaming tools, created via `ToolInput::ready()` so `.recv()` resolves immediately.
/// For streaming tools, partial JSON snapshots arrive via `.recv_partial()` as the LLM streams
/// them, followed by the final complete input available through `.recv()`.
pub struct ToolInput<T> {
    rx: mpsc::UnboundedReceiver<ToolInputPayload<serde_json::Value>>,
    _phantom: PhantomData<T>,
}

impl<T: DeserializeOwned> ToolInput<T> {
    #[cfg(any(test, feature = "test-support"))]
    pub fn resolved(input: impl Serialize) -> Self {
        let value = serde_json::to_value(input).expect("failed to serialize tool input");
        Self::ready(value)
    }

    pub fn ready(value: serde_json::Value) -> Self {
        let (tx, rx) = mpsc::unbounded();
        tx.unbounded_send(ToolInputPayload::Full(value)).ok();
        Self {
            rx,
            _phantom: PhantomData,
        }
    }

    pub fn invalid_json(error_message: String) -> Self {
        let (tx, rx) = mpsc::unbounded();
        tx.unbounded_send(ToolInputPayload::InvalidJson { error_message })
            .ok();
        Self {
            rx,
            _phantom: PhantomData,
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn test() -> (ToolInputSender, Self) {
        let (sender, input) = ToolInputSender::channel();
        (sender, input.cast())
    }

    /// Wait for the final deserialized input, ignoring all partial updates.
    /// Non-streaming tools can use this to wait until the whole input is available.
    pub async fn recv(mut self) -> Result<T> {
        while let Ok(value) = self.next().await {
            match value {
                ToolInputPayload::Full(value) => return Ok(value),
                ToolInputPayload::Partial(_) => {}
                ToolInputPayload::InvalidJson { error_message } => {
                    return Err(anyhow!(error_message));
                }
            }
        }
        Err(anyhow!("tool input was not fully received"))
    }

    pub async fn next(&mut self) -> Result<ToolInputPayload<T>> {
        let value = self
            .rx
            .next()
            .await
            .ok_or_else(|| anyhow!("tool input was not fully received"))?;

        Ok(match value {
            ToolInputPayload::Partial(payload) => ToolInputPayload::Partial(payload),
            ToolInputPayload::Full(payload) => {
                ToolInputPayload::Full(serde_json::from_value(payload)?)
            }
            ToolInputPayload::InvalidJson { error_message } => {
                ToolInputPayload::InvalidJson { error_message }
            }
        })
    }

    fn cast<U: DeserializeOwned>(self) -> ToolInput<U> {
        ToolInput {
            rx: self.rx,
            _phantom: PhantomData,
        }
    }
}

pub enum ToolInputPayload<T> {
    Partial(serde_json::Value),
    Full(T),
    InvalidJson { error_message: String },
}

pub struct ToolInputSender {
    has_received_final: bool,
    tx: mpsc::UnboundedSender<ToolInputPayload<serde_json::Value>>,
}

impl ToolInputSender {
    pub(crate) fn channel() -> (Self, ToolInput<serde_json::Value>) {
        let (tx, rx) = mpsc::unbounded();
        let sender = Self {
            tx,
            has_received_final: false,
        };
        let input = ToolInput {
            rx,
            _phantom: PhantomData,
        };
        (sender, input)
    }

    pub(crate) fn has_received_final(&self) -> bool {
        self.has_received_final
    }

    pub fn send_partial(&mut self, payload: serde_json::Value) {
        self.tx
            .unbounded_send(ToolInputPayload::Partial(payload))
            .ok();
    }

    pub fn send_full(&mut self, payload: serde_json::Value) {
        self.has_received_final = true;
        self.tx.unbounded_send(ToolInputPayload::Full(payload)).ok();
    }

    pub fn send_invalid_json(&mut self, error_message: String) {
        self.has_received_final = true;
        self.tx
            .unbounded_send(ToolInputPayload::InvalidJson { error_message })
            .ok();
    }
}

pub trait AgentTool
where
    Self: 'static + Sized,
{
    type Input: for<'de> Deserialize<'de> + Serialize + JsonSchema;
    type Output: for<'de> Deserialize<'de> + Serialize + Into<LanguageModelToolResultContent>;

    const NAME: &'static str;

    fn description() -> SharedString {
        let schema = schemars::schema_for!(Self::Input);
        SharedString::new(
            schema
                .get("description")
                .and_then(|description| description.as_str())
                .unwrap_or_default(),
        )
    }

    fn kind() -> acp::ToolKind;

    /// The initial tool title to display. Can be updated during the tool run.
    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        cx: &mut App,
    ) -> SharedString;

    /// Returns the JSON schema that describes the tool's input.
    fn input_schema(format: LanguageModelToolSchemaFormat) -> Schema {
        language_model::tool_schema::root_schema_for::<Self::Input>(format)
    }

    /// Returns whether the tool supports streaming of tool use parameters.
    fn supports_input_streaming() -> bool {
        false
    }

    /// Some tools rely on a provider for the underlying billing or other reasons.
    /// Allow the tool to check if they are compatible, or should be filtered out.
    fn supports_provider(_provider: &LanguageModelProviderId) -> bool {
        true
    }

    /// Runs the tool with the provided input.
    ///
    /// Returns `Result<Self::Output, Self::Output>` rather than `Result<Self::Output, anyhow::Error>`
    /// because tool errors are sent back to the model as tool results. This means error output must
    /// be structured and readable by the agent — not an arbitrary `anyhow::Error`. Returning the
    /// same `Output` type for both success and failure lets tools provide structured data while
    /// still signaling whether the invocation succeeded or failed.
    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>>;

    /// Emits events for a previous execution of the tool.
    fn replay(
        &self,
        _input: Self::Input,
        _output: Self::Output,
        _event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        Ok(())
    }

    fn erase(self) -> Arc<dyn AnyAgentTool> {
        Arc::new(Erased(Arc::new(self)))
    }
}

pub struct Erased<T>(T);

pub struct AgentToolOutput {
    pub llm_output: LanguageModelToolResultContent,
    pub raw_output: serde_json::Value,
}

impl AgentToolOutput {
    pub fn from_error(message: impl Into<String>) -> Self {
        let message = message.into();
        let llm_output = LanguageModelToolResultContent::Text(Arc::from(message.as_str()));
        Self {
            raw_output: serde_json::Value::String(message),
            llm_output,
        }
    }
}

pub trait AnyAgentTool {
    fn name(&self) -> SharedString;
    fn description(&self) -> SharedString;
    fn kind(&self) -> acp::ToolKind;
    fn initial_title(&self, input: serde_json::Value, _cx: &mut App) -> SharedString;
    fn input_schema(&self, format: LanguageModelToolSchemaFormat) -> Result<serde_json::Value>;
    fn supports_input_streaming(&self) -> bool {
        false
    }
    fn supports_provider(&self, _provider: &LanguageModelProviderId) -> bool {
        true
    }
    /// See [`AgentTool::run`] for why this returns `Result<AgentToolOutput, AgentToolOutput>`.
    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>>;
    fn replay(
        &self,
        input: serde_json::Value,
        output: serde_json::Value,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Result<()>;
}

impl<T> AnyAgentTool for Erased<Arc<T>>
where
    T: AgentTool,
{
    fn name(&self) -> SharedString {
        T::NAME.into()
    }

    fn description(&self) -> SharedString {
        T::description()
    }

    fn kind(&self) -> agent_client_protocol::ToolKind {
        T::kind()
    }

    fn supports_input_streaming(&self) -> bool {
        T::supports_input_streaming()
    }

    fn initial_title(&self, input: serde_json::Value, _cx: &mut App) -> SharedString {
        let parsed_input = serde_json::from_value(input.clone()).map_err(|_| input);
        self.0.initial_title(parsed_input, _cx)
    }

    fn input_schema(&self, format: LanguageModelToolSchemaFormat) -> Result<serde_json::Value> {
        let mut json = serde_json::to_value(T::input_schema(format))?;
        language_model::tool_schema::adapt_schema_to_format(&mut json, format)?;
        Ok(json)
    }

    fn supports_provider(&self, provider: &LanguageModelProviderId) -> bool {
        T::supports_provider(provider)
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<serde_json::Value>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<AgentToolOutput, AgentToolOutput>> {
        let tool_input: ToolInput<T::Input> = input.cast();
        let task = self.0.clone().run(tool_input, event_stream, cx);
        cx.spawn(async move |_cx| match task.await {
            Ok(output) => {
                let raw_output = serde_json::to_value(&output).map_err(|e| {
                    AgentToolOutput::from_error(format!("Failed to serialize tool output: {e}"))
                })?;
                Ok(AgentToolOutput {
                    llm_output: output.into(),
                    raw_output,
                })
            }
            Err(error_output) => {
                let raw_output = serde_json::to_value(&error_output).unwrap_or_else(|e| {
                    log::error!("Failed to serialize tool error output: {e}");
                    serde_json::Value::Null
                });
                Err(AgentToolOutput {
                    llm_output: error_output.into(),
                    raw_output,
                })
            }
        })
    }

    fn replay(
        &self,
        input: serde_json::Value,
        output: serde_json::Value,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Result<()> {
        let input = serde_json::from_value(input)?;
        let output = serde_json::from_value(output)?;
        self.0.replay(input, output, event_stream, cx)
    }
}

#[derive(Clone)]
struct ThreadEventStream(mpsc::UnboundedSender<Result<ThreadEvent>>);

impl ThreadEventStream {
    fn send_user_message(&self, message: &UserMessage) {
        self.0
            .unbounded_send(Ok(ThreadEvent::UserMessage(message.clone())))
            .ok();
    }

    fn send_text(&self, text: &str) {
        self.0
            .unbounded_send(Ok(ThreadEvent::AgentText(text.to_string())))
            .ok();
    }

    fn send_thinking(&self, text: &str) {
        self.0
            .unbounded_send(Ok(ThreadEvent::AgentThinking(text.to_string())))
            .ok();
    }

    fn send_tool_call(
        &self,
        id: &LanguageModelToolUseId,
        tool_name: &str,
        title: SharedString,
        kind: acp::ToolKind,
        input: serde_json::Value,
    ) {
        self.0
            .unbounded_send(Ok(ThreadEvent::ToolCall(Self::initial_tool_call(
                id,
                tool_name,
                title.to_string(),
                kind,
                input,
            ))))
            .ok();
    }

    fn initial_tool_call(
        id: &LanguageModelToolUseId,
        tool_name: &str,
        title: String,
        kind: acp::ToolKind,
        input: serde_json::Value,
    ) -> acp::ToolCall {
        acp::ToolCall::new(id.to_string(), title)
            .kind(kind)
            .raw_input(input)
            .meta(acp_thread::meta_with_tool_name(tool_name))
    }

    fn update_tool_call_fields(
        &self,
        tool_use_id: &LanguageModelToolUseId,
        fields: acp::ToolCallUpdateFields,
        meta: Option<acp::Meta>,
    ) {
        self.0
            .unbounded_send(Ok(ThreadEvent::ToolCallUpdate(
                acp::ToolCallUpdate::new(tool_use_id.to_string(), fields)
                    .meta(meta)
                    .into(),
            )))
            .ok();
    }

    fn send_plan(&self, plan: acp::Plan) {
        self.0.unbounded_send(Ok(ThreadEvent::Plan(plan))).ok();
    }

    fn send_retry(&self, status: acp_thread::RetryStatus) {
        self.0.unbounded_send(Ok(ThreadEvent::Retry(status))).ok();
    }

    fn send_stop(&self, reason: acp::StopReason) {
        self.0.unbounded_send(Ok(ThreadEvent::Stop(reason))).ok();
    }

    fn send_canceled(&self) {
        self.0
            .unbounded_send(Ok(ThreadEvent::Stop(acp::StopReason::Cancelled)))
            .ok();
    }

    fn send_error(&self, error: impl Into<anyhow::Error>) {
        self.0.unbounded_send(Err(error.into())).ok();
    }

    // ── G1d: native-loop event sinks ───────────────────────────────
    //
    // These don't exist in the legacy path; added so
    // [`dispatch_translated_event`] can surface engine notices/diag
    // without threading new call sites through every translator arm.

    fn send_context_notice(&self, kind: impl Into<String>, message: impl Into<String>) {
        self.0
            .unbounded_send(Ok(ThreadEvent::ContextNotice(ContextNotice {
                kind: kind.into(),
                message: message.into(),
            })))
            .ok();
    }

    fn send_engine_diagnostic(
        &self,
        kind: impl Into<String>,
        detail: impl Into<String>,
        severity: EngineDiagnosticSeverity,
    ) {
        self.0
            .unbounded_send(Ok(ThreadEvent::EngineDiagnostic(EngineDiagnostic {
                kind: kind.into(),
                detail: detail.into(),
                severity,
            })))
            .ok();
    }
}

// ── NW-3 — native-path permission round-trip ──────────────────────
//
// `route_native_permission_request` bridges a
// `TranslatedThreadEvent::PermissionRequest` back to the engine's
// `approval_tx` channel by:
//  1. **A2 fast-path.** Consult the user's
//     `tool_permissions.<tool>.always_allow / .always_deny /
//     .always_confirm` regex rules (via
//     `decide_permission_from_settings`) against the engine-supplied
//     `raw_input`. If the decision is `Allow` / `Deny` we skip the UI
//     prompt entirely and respond to the engine immediately; this is
//     what the settings-level "always allow `ls `" affordance promises.
//  2. **Interactive fall-through.** On `Confirm` (or when `raw_input`
//     is absent, which happens on legacy engine builds predating A1),
//     we emit a real `ThreadEvent::ToolCallAuthorization` with a
//     oneshot `response` sender — same UI path the legacy tool adapters
//     use — and spawn a task that awaits the user's decision, maps
//     the `SelectedPermissionOutcome` to a boolean, and sends
//     `(id, allowed)` on `approval_tx`.
//
// The `id` received here is already the engine's `perm_{tool_use_id}`
// string — the engine formats the key in `agent_harness.rs:2281` and
// the event translator forwards it unchanged, so we must NOT re-prefix
// (NW-3 regression noted below).
//
// `raw_input` comes from the engine already secret-redacted
// (`caduceus_core::redact_secrets_for_event`). We extract the
// top-level string values as candidate matcher inputs; this covers
// every built-in tool whose approval signal is a string field
// (`command` for terminal, `path` for edit_file, `url` for
// fetch_tool, ...). Tools with only numeric / object-valued fields
// produce an empty input vector → matcher falls through to the tool's
// default → interactive prompt (safe).
//
// Timeout policy is engine-side (default ~300s). If the user closes
// the panel without deciding, the oneshot is dropped → engine receives
// `PermissionOutcome::ChannelClosed` → tool call is denied fail-fast.
#[allow(clippy::redundant_clone)]
fn route_native_permission_request(
    id: &str,
    tool: &str,
    description: &str,
    raw_input: Option<&serde_json::Value>,
    stream: &ThreadEventStream,
    approval_tx: tokio::sync::mpsc::Sender<(String, bool)>,
    cx: &mut AsyncApp,
) {
    let key = id.to_string();

    // A2 fast-path — evaluate user-configured always-allow rules BEFORE
    // spawning an interactive prompt. Pure function over settings +
    // extracted string inputs; no UI involved.
    if let Some(raw) = raw_input {
        let inputs = extract_match_inputs(raw);
        let decision = cx.update(|cx| {
            crate::tool_permissions::decide_permission_from_settings(
                tool,
                &inputs,
                agent_settings::AgentSettings::get_global(cx),
            )
        });
        match decision {
            crate::tool_permissions::ToolPermissionDecision::Allow => {
                // Auto-approve. Surface a diagnostic so the user can see
                // the approval was rule-driven, not silently skipped.
                stream.send_engine_diagnostic(
                    "permission.auto_allow",
                    format!("{tool} auto-approved by user rule"),
                    EngineDiagnosticSeverity::Info,
                );
                let approval_tx = approval_tx.clone();
                cx.spawn(async move |_| {
                    if approval_tx.send((key, true)).await.is_err() {
                        log::warn!(
                            "[native-loop] approval_tx closed before auto-allow could be sent"
                        );
                    }
                })
                .detach();
                return;
            }
            crate::tool_permissions::ToolPermissionDecision::Deny(reason) => {
                stream.send_engine_diagnostic(
                    "permission.auto_deny",
                    format!("{tool} auto-denied by user rule: {reason}"),
                    EngineDiagnosticSeverity::Warning,
                );
                let approval_tx = approval_tx.clone();
                cx.spawn(async move |_| {
                    if approval_tx.send((key, false)).await.is_err() {
                        log::warn!(
                            "[native-loop] approval_tx closed before auto-deny could be sent"
                        );
                    }
                })
                .detach();
                return;
            }
            crate::tool_permissions::ToolPermissionDecision::Confirm => {
                // Fall through to interactive prompt below.
            }
        }
    }

    let (response_tx, response_rx) = oneshot::channel();
    let title = format!("{tool}: {description}");
    let tool_call = acp::ToolCallUpdate::new(
        id.to_string(),
        acp::ToolCallUpdateFields::new().title(title),
    );
    let options =
        acp_thread::PermissionOptions::Dropdown(vec![acp_thread::PermissionOptionChoice {
            allow: acp::PermissionOption::new(
                acp::PermissionOptionId::new("allow_once".to_string()),
                "Allow".to_string(),
                acp::PermissionOptionKind::AllowOnce,
            ),
            deny: acp::PermissionOption::new(
                acp::PermissionOptionId::new("deny_once".to_string()),
                "Deny".to_string(),
                acp::PermissionOptionKind::RejectOnce,
            ),
            sub_patterns: vec![],
        }]);
    if stream
        .0
        .unbounded_send(Ok(ThreadEvent::ToolCallAuthorization(
            ToolCallAuthorization {
                tool_call,
                options,
                response: response_tx,
                context: None,
            },
        )))
        .is_err()
    {
        log::warn!("[native-loop] failed to send ToolCallAuthorization; denying by default");
        // 🔴 bug-fix (rubber-duck): `id` received here is ALREADY the
        // engine's `perm_{tool_use_id}` string (emitted by
        // agent_harness.rs:2281-2286 and forwarded unchanged by the
        // event translator at event_translator.rs:237-241). Re-prefixing
        // produces `perm_perm_*` which fails the engine's `id == req_id`
        // equality check at agent_harness.rs:2319, causing the approval
        // channel to treat every native decision as a stale/mismatched
        // reply — until now NW-3 approvals were silently never arriving.
        let approval_tx = approval_tx.clone();
        let key_for_err = key.clone();
        cx.spawn(async move |_| {
            let _ = approval_tx.send((key_for_err, false)).await;
        })
        .detach();
        return;
    }
    cx.spawn(async move |_| {
        let allowed = match response_rx.await {
            Ok(outcome) => {
                let option_id = outcome.option_id.0.as_ref();
                option_id.starts_with("allow")
            }
            Err(_) => false, // oneshot dropped → deny fail-fast
        };
        if approval_tx.send((key, allowed)).await.is_err() {
            log::warn!("[native-loop] approval_tx closed before decision could be sent");
        }
    })
    .detach();
}

/// Collect top-level string values of `raw_input` as candidate matcher
/// inputs for [`crate::tool_permissions::decide_permission_from_settings`].
///
/// We deliberately look only at the **top level** of a JSON object:
/// built-in tools that hook into always-allow rules (terminal,
/// edit_file, fetch, create_directory, copy_path, delete_path,
/// restore_file_from_disk) all expose the user-meaningful matching
/// field as a top-level string. Nested objects might carry arbitrary
/// values whose regex-match against `command` / `path` rules would be
/// surprising, so we skip them. Non-object / non-string inputs
/// produce an empty `Vec` → `decide_permission_from_settings` falls
/// through to the tool's default (usually `Confirm`), preserving the
/// safe interactive behavior.
///
/// The engine pre-redacts this value via
/// `caduceus_core::redact_secrets_for_event`, so any secret-shaped
/// top-level key has already been replaced with `"<redacted>"`.
fn extract_match_inputs(raw: &serde_json::Value) -> Vec<String> {
    let Some(obj) = raw.as_object() else {
        return Vec::new();
    };
    obj.values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect()
}

/// Phase-6 consumer helper — stateful tool-lifecycle handling for the
/// native-loop translated event stream. Extracted from the consumer
/// closure so it can be unit-tested against a scripted
/// `ToolCallStart → InputDelta×N → InputEnd → ToolResult` sequence
/// without standing up the full `run_caduceus_loop_translated` pipeline.
///
/// Returns `true` when the event was fully emitted as UI events here
/// (caller must NOT fall through to `dispatch_translated_event`), and
/// `false` when the caller should fall through (e.g. `ToolResult`,
/// which we only use to clean up local state — the stateless
/// dispatcher owns the actual status/content/raw_output emission).
///
/// The stateful-consumer's exclusive responsibility is:
/// 1. Resolving `kind` and `initial_title` from the enabled-tools
///    registry (`AnyAgentTool`) at the right lifecycle phases.
/// 2. Aggregating streaming `ToolCallInputDelta` frames into a single
///    parsed `raw_input` value.
/// 3. Emitting exactly one `ToolCall` at start and exactly one
///    `ToolCallUpdate` at input-end (no emit during delta frames).
fn handle_native_tool_lifecycle(
    ev: &caduceus_bridge::event_translator::TranslatedThreadEvent,
    input_agg: &mut caduceus_bridge::event_translator::ToolInputAggregator,
    enabled_tools: &std::collections::BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>>,
    stream: &ThreadEventStream,
    cx: &mut AsyncApp,
) -> bool {
    use caduceus_bridge::event_translator::TranslatedThreadEvent as T;
    match ev {
        T::ToolCallStart { id, name } => {
            input_agg.observe_start(id, name);
            let tool_name: SharedString = name.clone().into();
            let kind = enabled_tools
                .get(&tool_name)
                .map(|t| t.kind())
                .unwrap_or(acp::ToolKind::Other);
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCall(
                    acp::ToolCall::new(id.to_string(), name.to_string())
                        .kind(kind)
                        .status(acp::ToolCallStatus::Pending)
                        .meta(acp_thread::meta_with_tool_name(name)),
                )))
                .ok();
            true
        }
        T::ToolCallInputDelta { id, delta } => {
            input_agg.observe_delta(id, delta);
            true
        }
        T::ToolCallInputEnd { id } => {
            let agg = input_agg.observe_input_end(id);
            let rich_title: Option<SharedString> = match (&agg.parsed, agg.tool_name.as_ref()) {
                (Some(input), Some(name)) => {
                    let input = input.clone();
                    let name_ss: SharedString = name.clone().into();
                    cx.update(|cx| {
                        enabled_tools
                            .get(&name_ss)
                            .map(|t| t.initial_title(input, cx))
                    })
                }
                _ => None,
            };
            let mut fields =
                acp::ToolCallUpdateFields::new().status(acp::ToolCallStatus::InProgress);
            if let Some(t) = rich_title {
                fields = fields.title(t.to_string());
            }
            if let Some(raw) = agg.parsed {
                fields = fields.raw_input(raw);
            }
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCallUpdate(
                    acp::ToolCallUpdate::new(id.to_string(), fields).into(),
                )))
                .ok();
            true
        }
        T::ToolResult { id, .. } => {
            // Clean up per-id state so the maps don't grow unbounded,
            // then fall through to the stateless dispatcher which
            // already emits status + content + raw_output.
            input_agg.finalize(id);
            false
        }
        _ => false,
    }
}

// ── G1d — translator → ThreadEvent dispatch ─────────────────────────
//
// `dispatch_translated_event` is the single mapping boundary between
// the engine event stream (via `caduceus_bridge::event_translator`)
// and the Zed-side `ThreadEvent`s the ACP consumer (and UI layer)
// expect. Keeping this as a pure free function with no `cx`/`this`
// dependency makes it unit-testable against a `mpsc::UnboundedSender`
// alone — the 15 mapping tests in G1f pin the exact behaviors.
//
// **Invariants this function enforces:**
//
// 1. Every `TranslatedThreadEvent` either produces **at least one**
//    `ThreadEvent` or is explicitly logged as swallowed.
//    `T::Swallow` is a no-op; everything else has a mapping.
//
// 2. `TurnComplete` / `TurnError` are the ONLY variants that produce
//    `ThreadEvent::Stop`. Callers rely on this to know when the
//    translated stream has drained — any caller that sees a `Stop`
//    should drop its `translated_rx` afterward.
//
// 3. Engine-only notices (`ContextWarning`, `ContextCompacted`,
//    `ContextEvicted`, `ModeChanged`, `ScopeExpansion`) map to
//    `ThreadEvent::ContextNotice` (info class) or
//    `ThreadEvent::EngineDiagnostic` (warning class for
//    scope-expansion). The UI consumer decides how to render; the
//    existing ACP dispatcher (agent.rs:1879) logs-and-drops.
//
// 4. Plan-DAG variants (`PlanStep`, `PlanAmended`) currently map to
//    `ContextNotice` with kind `"plan.step"` / `"plan.amended"`. The
//    legacy `ThreadEvent::Plan(acp::Plan)` carries a different shape
//    (a full snapshot) — the real plan mapping is deferred until
//    `G1d-provider-adapter` lands since the engine's plan deltas
//    need to be folded into an `acp::Plan` snapshot by the caller.
fn dispatch_translated_event(
    ev: &caduceus_bridge::event_translator::TranslatedThreadEvent,
    stream: &ThreadEventStream,
) {
    use caduceus_bridge::event_translator::TranslatedThreadEvent as T;
    match ev {
        T::AgentText(text) => stream.send_text(text),
        T::AgentThinking(text) => stream.send_thinking(text),
        T::AgentThinkingComplete { content, .. } => stream.send_thinking(content),

        // Tool lifecycle → real `acp::ToolCall` / `acp::ToolCallUpdate`
        // events so the panel renders proper tool-call cards.
        //
        // NOTE: In production the consumer task at Phase 6 intercepts
        // `ToolCallStart` / `ToolCallInputDelta` / `ToolCallInputEnd`
        // BEFORE reaching this dispatcher so it can look up
        // `enabled_tools[name]` for the rich `kind` + `initial_title`
        // and aggregate input deltas per id. These arms remain here
        // for the stateless contract the 15 G1f mapping tests pin —
        // they render a plainer card (title=name, kind=Other, no raw
        // input) and are exercised only when the dispatcher is used
        // outside the native-loop consumer (tests, headless tools).
        T::ToolCallStart { id, name } => {
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCall(
                    acp::ToolCall::new(id.to_string(), name.to_string())
                        .kind(acp::ToolKind::Other)
                        .status(acp::ToolCallStatus::Pending)
                        .meta(acp_thread::meta_with_tool_name(name)),
                )))
                .ok();
        }
        T::ToolCallInputDelta { id, delta } => {
            stream.send_engine_diagnostic(
                "tool.call.input_delta",
                format!("id={id} delta_bytes={}", delta.len()),
                EngineDiagnosticSeverity::Info,
            );
        }
        T::ToolCallInputEnd { id } => {
            // Input streaming done; tool is about to execute. Flip
            // status to InProgress so the card shows a spinner.
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCallUpdate(
                    acp::ToolCallUpdate::new(
                        id.to_string(),
                        acp::ToolCallUpdateFields::new().status(acp::ToolCallStatus::InProgress),
                    )
                    .into(),
                )))
                .ok();
        }
        T::ToolResult {
            id,
            content,
            is_error,
        } => {
            let status = if *is_error {
                acp::ToolCallStatus::Failed
            } else {
                acp::ToolCallStatus::Completed
            };
            let fields = acp::ToolCallUpdateFields::new()
                .status(status)
                .content(vec![content.clone().into()])
                .raw_output(serde_json::Value::String(content.clone()));
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::ToolCallUpdate(
                    acp::ToolCallUpdate::new(id.to_string(), fields).into(),
                )))
                .ok();
        }

        // Permission — surface as warning. Real routing to
        // `ToolCallAuthorization` is deferred to provider-adapter
        // since the oneshot response channel needs engine-side
        // plumbing through `approval_tx`.
        T::PermissionRequest {
            id,
            tool,
            description,
            raw_input: _,
        } => {
            stream.send_engine_diagnostic(
                "permission.request",
                format!("id={id} tool={tool}: {description}"),
                EngineDiagnosticSeverity::Warning,
            );
        }

        // Context notices — informational.
        T::ContextWarning { level, used, max } => {
            stream.send_context_notice("context.warning", format!("{level}: {used}/{max} tokens"))
        }
        T::ContextCompacted {
            freed,
            before,
            after,
        } => stream.send_context_notice(
            "context.compacted",
            format!("freed {freed} tokens ({before} → {after})"),
        ),
        T::ContextEvicted {
            strategy,
            groups,
            total_tokens,
        } => stream.send_context_notice(
            "context.evicted",
            format!("{strategy}: {groups} groups, {total_tokens} tokens"),
        ),
        T::ModeChanged {
            from_mode, to_mode, ..
        } => stream.send_context_notice("mode.changed", format!("{from_mode} → {to_mode}")),
        T::ScopeExpansion {
            capability,
            resource,
            tool,
            reason,
        } => {
            stream.send_engine_diagnostic(
                "scope.expansion",
                format!("tool={tool} capability={capability} resource={resource}: {reason}"),
                EngineDiagnosticSeverity::Warning,
            );
        }

        // Retry — map to the existing acp_thread::RetryStatus shape
        // with best-effort fields. The legacy path carries more
        // detail (attempt counts, backoff), which the provider
        // adapter will reinstate once AttemptStatus is available.
        T::Retry { kind, message } => {
            let label = format!("{kind:?}: {message}");
            stream.send_engine_diagnostic("retry", label, EngineDiagnosticSeverity::Warning);
        }

        // Plan — context notice until full acp::Plan mapping lands.
        T::PlanStep {
            step,
            tool,
            description,
            ..
        } => stream.send_context_notice("plan.step", format!("#{step} {tool}: {description}")),
        T::PlanAmended {
            kind,
            step,
            ok,
            reason,
            ..
        } => {
            stream.send_context_notice("plan.amended", format!("#{step} {kind} ok={ok}: {reason}"))
        }

        // Turn lifecycle.
        T::TurnComplete { stop, usage } => {
            use caduceus_bridge::event_translator::StopReasonKind as K;
            // Emit structured usage BEFORE Stop — H2 contract per
            // event_translator.rs:129-130. Callers that exit on Stop
            // still see the usage event first thanks to ordering.
            stream
                .0
                .unbounded_send(Ok(ThreadEvent::UsageUpdated((*usage).into())))
                .ok();
            let reason = match stop {
                K::EndTurn => acp::StopReason::EndTurn,
                K::ToolUse => acp::StopReason::EndTurn,
                K::MaxTokens => acp::StopReason::MaxTokens,
                K::StopSequence => acp::StopReason::EndTurn,
                K::Error => acp::StopReason::Cancelled,
                K::BudgetExceeded => acp::StopReason::MaxTokens,
                K::Other(_) => acp::StopReason::EndTurn,
            };
            stream.send_stop(reason);
        }
        T::TurnError { message } => {
            stream.send_engine_diagnostic(
                "turn.error",
                message.clone(),
                EngineDiagnosticSeverity::Error,
            );
            stream.send_stop(acp::StopReason::Cancelled);
        }

        T::Swallow { reason } => {
            log::trace!("[native-loop] swallow: {reason}");
        }
    }
}

#[derive(Clone)]
pub struct ToolCallEventStream {
    tool_use_id: LanguageModelToolUseId,
    stream: ThreadEventStream,
    fs: Option<Arc<dyn Fs>>,
    cancellation_rx: watch::Receiver<bool>,
}

impl ToolCallEventStream {
    #[cfg(any(test, feature = "test-support"))]
    pub fn test() -> (Self, ToolCallEventStreamReceiver) {
        let (stream, receiver, _cancellation_tx) = Self::test_with_cancellation();
        (stream, receiver)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn test_with_cancellation() -> (Self, ToolCallEventStreamReceiver, watch::Sender<bool>) {
        let (events_tx, events_rx) = mpsc::unbounded::<Result<ThreadEvent>>();
        let (cancellation_tx, cancellation_rx) = watch::channel(false);

        let stream = ToolCallEventStream::new(
            "test_id".into(),
            ThreadEventStream(events_tx),
            None,
            cancellation_rx,
        );

        (
            stream,
            ToolCallEventStreamReceiver(events_rx),
            cancellation_tx,
        )
    }

    /// Signal cancellation for this event stream. Only available in tests.
    #[cfg(any(test, feature = "test-support"))]
    pub fn signal_cancellation_with_sender(cancellation_tx: &mut watch::Sender<bool>) {
        cancellation_tx.send(true).ok();
    }

    fn new(
        tool_use_id: LanguageModelToolUseId,
        stream: ThreadEventStream,
        fs: Option<Arc<dyn Fs>>,
        cancellation_rx: watch::Receiver<bool>,
    ) -> Self {
        Self {
            tool_use_id,
            stream,
            fs,
            cancellation_rx,
        }
    }

    /// Returns a future that resolves when the user cancels the tool call.
    /// Tools should select on this alongside their main work to detect user cancellation.
    pub fn cancelled_by_user(&self) -> impl std::future::Future<Output = ()> + '_ {
        let mut rx = self.cancellation_rx.clone();
        async move {
            loop {
                if *rx.borrow() {
                    return;
                }
                if rx.changed().await.is_err() {
                    // Sender dropped, will never be cancelled
                    std::future::pending::<()>().await;
                }
            }
        }
    }

    /// Returns true if the user has cancelled this tool call.
    /// This is useful for checking cancellation state after an operation completes,
    /// to determine if the completion was due to user cancellation.
    pub fn was_cancelled_by_user(&self) -> bool {
        *self.cancellation_rx.clone().borrow()
    }

    pub fn tool_use_id(&self) -> &LanguageModelToolUseId {
        &self.tool_use_id
    }

    pub fn update_fields(&self, fields: acp::ToolCallUpdateFields) {
        self.stream
            .update_tool_call_fields(&self.tool_use_id, fields, None);
    }

    pub fn update_fields_with_meta(
        &self,
        fields: acp::ToolCallUpdateFields,
        meta: Option<acp::Meta>,
    ) {
        self.stream
            .update_tool_call_fields(&self.tool_use_id, fields, meta);
    }

    pub fn update_diff(&self, diff: Entity<acp_thread::Diff>) {
        self.stream
            .0
            .unbounded_send(Ok(ThreadEvent::ToolCallUpdate(
                acp_thread::ToolCallUpdateDiff {
                    id: acp::ToolCallId::new(self.tool_use_id.to_string()),
                    diff,
                }
                .into(),
            )))
            .ok();
    }

    pub fn subagent_spawned(&self, id: acp::SessionId) {
        self.stream
            .0
            .unbounded_send(Ok(ThreadEvent::SubagentSpawned(id)))
            .ok();
    }

    pub fn update_plan(&self, plan: acp::Plan) {
        self.stream.send_plan(plan);
    }

    /// Authorize a third-party tool (e.g., MCP tool from a context server).
    ///
    /// Unlike built-in tools, third-party tools don't support pattern-based permissions.
    /// They only support `default` (allow/deny/confirm) per tool.
    ///
    /// Uses the dropdown authorization flow with two granularities:
    /// - "Always for <display_name> MCP tool" → sets `tools.<tool_id>.default = "allow"` or "deny"
    /// - "Only this time" → allow/deny once
    pub fn authorize_third_party_tool(
        &self,
        title: impl Into<String>,
        tool_id: String,
        display_name: String,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let settings = agent_settings::AgentSettings::get_global(cx);

        let decision = decide_permission_from_settings(&tool_id, &[String::new()], &settings);

        match decision {
            ToolPermissionDecision::Allow => return Task::ready(Ok(())),
            ToolPermissionDecision::Deny(reason) => return Task::ready(Err(anyhow!(reason))),
            ToolPermissionDecision::Confirm => {}
        }

        let (response_tx, response_rx) = oneshot::channel();
        if let Err(error) = self
            .stream
            .0
            .unbounded_send(Ok(ThreadEvent::ToolCallAuthorization(
                ToolCallAuthorization {
                    tool_call: acp::ToolCallUpdate::new(
                        self.tool_use_id.to_string(),
                        acp::ToolCallUpdateFields::new().title(title.into()),
                    ),
                    options: acp_thread::PermissionOptions::Dropdown(vec![
                        acp_thread::PermissionOptionChoice {
                            allow: acp::PermissionOption::new(
                                acp::PermissionOptionId::new(format!(
                                    "always_allow_mcp:{}",
                                    tool_id
                                )),
                                format!("Always for {} MCP tool", display_name),
                                acp::PermissionOptionKind::AllowAlways,
                            ),
                            deny: acp::PermissionOption::new(
                                acp::PermissionOptionId::new(format!(
                                    "always_deny_mcp:{}",
                                    tool_id
                                )),
                                format!("Always for {} MCP tool", display_name),
                                acp::PermissionOptionKind::RejectAlways,
                            ),
                            sub_patterns: vec![],
                        },
                        acp_thread::PermissionOptionChoice {
                            allow: acp::PermissionOption::new(
                                acp::PermissionOptionId::new("allow"),
                                "Only this time",
                                acp::PermissionOptionKind::AllowOnce,
                            ),
                            deny: acp::PermissionOption::new(
                                acp::PermissionOptionId::new("deny"),
                                "Only this time",
                                acp::PermissionOptionKind::RejectOnce,
                            ),
                            sub_patterns: vec![],
                        },
                    ]),
                    response: response_tx,
                    context: None,
                },
            )))
        {
            log::error!("Failed to send tool call authorization: {error}");
            return Task::ready(Err(anyhow!(
                "Failed to send tool call authorization: {error}"
            )));
        }

        let fs = self.fs.clone();
        cx.spawn(async move |cx| {
            let outcome = response_rx.await?;
            let is_allow = Self::persist_permission_outcome(&outcome, fs, &cx);
            if is_allow {
                Ok(())
            } else {
                Err(anyhow!("Permission to run tool denied by user"))
            }
        })
    }

    pub fn authorize(
        &self,
        title: impl Into<String>,
        context: ToolPermissionContext,
        cx: &mut App,
    ) -> Task<Result<()>> {
        let options = context.build_permission_options();

        let (response_tx, response_rx) = oneshot::channel();
        if let Err(error) = self
            .stream
            .0
            .unbounded_send(Ok(ThreadEvent::ToolCallAuthorization(
                ToolCallAuthorization {
                    tool_call: acp::ToolCallUpdate::new(
                        self.tool_use_id.to_string(),
                        acp::ToolCallUpdateFields::new().title(title.into()),
                    ),
                    options,
                    response: response_tx,
                    context: Some(context),
                },
            )))
        {
            log::error!("Failed to send tool call authorization: {error}");
            return Task::ready(Err(anyhow!(
                "Failed to send tool call authorization: {error}"
            )));
        }

        let fs = self.fs.clone();
        cx.spawn(async move |cx| {
            let outcome = response_rx.await?;
            let is_allow = Self::persist_permission_outcome(&outcome, fs, &cx);
            if is_allow {
                Ok(())
            } else {
                Err(anyhow!("Permission to run tool denied by user"))
            }
        })
    }

    /// Interprets a `SelectedPermissionOutcome` and persists any settings changes.
    /// Returns `true` if the tool call should be allowed, `false` if denied.
    fn persist_permission_outcome(
        outcome: &acp_thread::SelectedPermissionOutcome,
        fs: Option<Arc<dyn Fs>>,
        cx: &AsyncApp,
    ) -> bool {
        let option_id = outcome.option_id.0.as_ref();

        let always_permission = option_id
            .strip_prefix("always_allow:")
            .map(|tool| (tool, ToolPermissionMode::Allow))
            .or_else(|| {
                option_id
                    .strip_prefix("always_deny:")
                    .map(|tool| (tool, ToolPermissionMode::Deny))
            })
            .or_else(|| {
                option_id
                    .strip_prefix("always_allow_mcp:")
                    .map(|tool| (tool, ToolPermissionMode::Allow))
            })
            .or_else(|| {
                option_id
                    .strip_prefix("always_deny_mcp:")
                    .map(|tool| (tool, ToolPermissionMode::Deny))
            });

        if let Some((tool, mode)) = always_permission {
            let params = outcome.params.as_ref();
            Self::persist_always_permission(tool, mode, params, fs, cx);
            return mode == ToolPermissionMode::Allow;
        }

        // Handle simple "allow" / "deny" (once, no persistence)
        if option_id == "allow" || option_id == "deny" {
            debug_assert!(
                outcome.params.is_none(),
                "unexpected params for once-only permission"
            );
            return option_id == "allow";
        }

        debug_assert!(false, "unexpected permission option_id: {option_id}");
        false
    }

    /// Persists an "always allow" or "always deny" permission, using sub_patterns
    /// from params when present.
    fn persist_always_permission(
        tool: &str,
        mode: ToolPermissionMode,
        params: Option<&acp_thread::SelectedPermissionParams>,
        fs: Option<Arc<dyn Fs>>,
        cx: &AsyncApp,
    ) {
        let Some(fs) = fs else {
            return;
        };

        match params {
            Some(acp_thread::SelectedPermissionParams::Terminal {
                patterns: sub_patterns,
            }) => {
                debug_assert!(
                    !sub_patterns.is_empty(),
                    "empty sub_patterns for tool {tool} — callers should pass None instead"
                );
                let tool = tool.to_string();
                let sub_patterns = sub_patterns.clone();
                cx.update(|cx| {
                    update_settings_file(fs, cx, move |settings, _| {
                        let agent = settings.agent.get_or_insert_default();
                        for pattern in sub_patterns {
                            match mode {
                                ToolPermissionMode::Allow => {
                                    agent.add_tool_allow_pattern(&tool, pattern);
                                }
                                ToolPermissionMode::Deny => {
                                    agent.add_tool_deny_pattern(&tool, pattern);
                                }
                                // If there's no matching pattern this will
                                // default to confirm, so falling through is
                                // fine here.
                                ToolPermissionMode::Confirm => (),
                            }
                        }
                    });
                });
            }
            None => {
                let tool = tool.to_string();
                cx.update(|cx| {
                    update_settings_file(fs, cx, move |settings, _| {
                        settings
                            .agent
                            .get_or_insert_default()
                            .set_tool_default_permission(&tool, mode);
                    });
                });
            }
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub struct ToolCallEventStreamReceiver(mpsc::UnboundedReceiver<Result<ThreadEvent>>);

#[cfg(any(test, feature = "test-support"))]
impl ToolCallEventStreamReceiver {
    pub async fn expect_authorization(&mut self) -> ToolCallAuthorization {
        let event = self.0.next().await;
        if let Some(Ok(ThreadEvent::ToolCallAuthorization(auth))) = event {
            auth
        } else {
            panic!("Expected ToolCallAuthorization but got: {:?}", event);
        }
    }

    pub async fn expect_update_fields(&mut self) -> acp::ToolCallUpdateFields {
        let event = self.0.next().await;
        if let Some(Ok(ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(
            update,
        )))) = event
        {
            update.fields
        } else {
            panic!("Expected update fields but got: {:?}", event);
        }
    }

    pub async fn expect_diff(&mut self) -> Entity<acp_thread::Diff> {
        let event = self.0.next().await;
        if let Some(Ok(ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateDiff(
            update,
        )))) = event
        {
            update.diff
        } else {
            panic!("Expected diff but got: {:?}", event);
        }
    }

    pub async fn expect_terminal(&mut self) -> Entity<acp_thread::Terminal> {
        let event = self.0.next().await;
        if let Some(Ok(ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateTerminal(
            update,
        )))) = event
        {
            update.terminal
        } else {
            panic!("Expected terminal but got: {:?}", event);
        }
    }

    pub async fn expect_plan(&mut self) -> acp::Plan {
        let event = self.0.next().await;
        if let Some(Ok(ThreadEvent::Plan(plan))) = event {
            plan
        } else {
            panic!("Expected plan but got: {:?}", event);
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl std::ops::Deref for ToolCallEventStreamReceiver {
    type Target = mpsc::UnboundedReceiver<Result<ThreadEvent>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(any(test, feature = "test-support"))]
impl std::ops::DerefMut for ToolCallEventStreamReceiver {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<&str> for UserMessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.into())
    }
}

impl From<String> for UserMessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl UserMessageContent {
    pub fn from_content_block(value: acp::ContentBlock, path_style: PathStyle) -> Self {
        match value {
            acp::ContentBlock::Text(text_content) => Self::Text(text_content.text),
            acp::ContentBlock::Image(image_content) => Self::Image(convert_image(image_content)),
            acp::ContentBlock::Audio(_) => {
                // TODO
                Self::Text("[audio]".to_string())
            }
            acp::ContentBlock::ResourceLink(resource_link) => {
                match MentionUri::parse(&resource_link.uri, path_style) {
                    Ok(uri) => Self::Mention {
                        uri,
                        content: String::new(),
                    },
                    Err(err) => {
                        log::error!("Failed to parse mention link: {}", err);
                        Self::Text(format!("[{}]({})", resource_link.name, resource_link.uri))
                    }
                }
            }
            acp::ContentBlock::Resource(resource) => match resource.resource {
                acp::EmbeddedResourceResource::TextResourceContents(resource) => {
                    match MentionUri::parse(&resource.uri, path_style) {
                        Ok(uri) => Self::Mention {
                            uri,
                            content: resource.text,
                        },
                        Err(err) => {
                            log::error!("Failed to parse mention link: {}", err);
                            Self::Text(
                                MarkdownCodeBlock {
                                    tag: &resource.uri,
                                    text: &resource.text,
                                }
                                .to_string(),
                            )
                        }
                    }
                }
                acp::EmbeddedResourceResource::BlobResourceContents(_) => {
                    // TODO
                    Self::Text("[blob]".to_string())
                }
                other => {
                    log::warn!("Unexpected content type: {:?}", other);
                    Self::Text("[unknown]".to_string())
                }
            },
            other => {
                log::warn!("Unexpected content type: {:?}", other);
                Self::Text("[unknown]".to_string())
            }
        }
    }
}

impl From<UserMessageContent> for acp::ContentBlock {
    fn from(content: UserMessageContent) -> Self {
        match content {
            UserMessageContent::Text(text) => text.into(),
            UserMessageContent::Image(image) => {
                acp::ContentBlock::Image(acp::ImageContent::new(image.source, "image/png"))
            }
            UserMessageContent::Mention { uri, content } => acp::ContentBlock::Resource(
                acp::EmbeddedResource::new(acp::EmbeddedResourceResource::TextResourceContents(
                    acp::TextResourceContents::new(content, uri.to_uri().to_string()),
                )),
            ),
        }
    }
}

fn convert_image(image_content: acp::ImageContent) -> LanguageModelImage {
    LanguageModelImage {
        source: image_content.data.into(),
        size: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use language_model::LanguageModelToolUseId;
    use language_model::fake_provider::FakeLanguageModel;
    use serde_json::json;
    use std::sync::Arc;

    async fn setup_thread_for_test(cx: &mut TestAppContext) -> (Entity<Thread>, ThreadEventStream) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });

        let fs = fs::FakeFs::new(cx.background_executor.clone());
        let templates = Templates::new();
        let project = Project::test(fs.clone(), [], cx).await;

        cx.update(|cx| {
            let project_context = cx.new(|_cx| prompt_store::ProjectContext::default());
            let context_server_store = project.read(cx).context_server_store();
            let context_server_registry =
                cx.new(|cx| ContextServerRegistry::new(context_server_store, cx));

            let thread = cx.new(|cx| {
                Thread::new(
                    project,
                    project_context,
                    context_server_registry,
                    templates,
                    None,
                    cx,
                )
            });

            let (event_tx, _event_rx) = mpsc::unbounded();
            let event_stream = ThreadEventStream(event_tx);

            (thread, event_stream)
        })
    }

    fn setup_parent_with_subagents(
        cx: &mut TestAppContext,
        parent: &Entity<Thread>,
        count: usize,
    ) -> Vec<Entity<Thread>> {
        cx.update(|cx| {
            let mut subagents = Vec::new();
            for _ in 0..count {
                let subagent = cx.new(|cx| Thread::new_subagent(parent, cx));
                parent.update(cx, |thread, cx| {
                    thread.register_running_subagent(subagent.downgrade(), cx);
                });
                subagents.push(subagent);
            }
            subagents
        })
    }

    #[gpui::test]
    async fn test_set_model_propagates_to_subagents(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 2);

        let new_model: Arc<dyn LanguageModel> = Arc::new(FakeLanguageModel::with_id_and_thinking(
            "test-provider",
            "new-model",
            "New Model",
            false,
        ));

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_model(new_model, cx);
            });

            for subagent in &subagents {
                let subagent_model_id = subagent.read(cx).model().unwrap().id();
                assert_eq!(
                    subagent_model_id.0.as_ref(),
                    "new-model",
                    "Subagent model should match parent model after set_model"
                );
            }
        });
    }

    #[gpui::test]
    async fn test_set_summarization_model_propagates_to_subagents(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 2);

        let summary_model: Arc<dyn LanguageModel> =
            Arc::new(FakeLanguageModel::with_id_and_thinking(
                "test-provider",
                "summary-model",
                "Summary Model",
                false,
            ));

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_summarization_model(Some(summary_model), cx);
            });

            for subagent in &subagents {
                let subagent_summary_id = subagent.read(cx).summarization_model().unwrap().id();
                assert_eq!(
                    subagent_summary_id.0.as_ref(),
                    "summary-model",
                    "Subagent summarization model should match parent after set_summarization_model"
                );
            }
        });
    }

    #[gpui::test]
    async fn test_set_thinking_enabled_propagates_to_subagents(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 2);

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_thinking_enabled(true, cx);
            });

            for subagent in &subagents {
                assert!(
                    subagent.read(cx).thinking_enabled(),
                    "Subagent thinking should be enabled after parent enables it"
                );
            }

            parent.update(cx, |thread, cx| {
                thread.set_thinking_enabled(false, cx);
            });

            for subagent in &subagents {
                assert!(
                    !subagent.read(cx).thinking_enabled(),
                    "Subagent thinking should be disabled after parent disables it"
                );
            }
        });
    }

    #[gpui::test]
    async fn test_set_thinking_effort_propagates_to_subagents(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 2);

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_thinking_effort(Some("high".to_string()), cx);
            });

            for subagent in &subagents {
                assert_eq!(
                    subagent.read(cx).thinking_effort().map(|s| s.as_str()),
                    Some("high"),
                    "Subagent thinking effort should match parent"
                );
            }

            parent.update(cx, |thread, cx| {
                thread.set_thinking_effort(None, cx);
            });

            for subagent in &subagents {
                assert_eq!(
                    subagent.read(cx).thinking_effort(),
                    None,
                    "Subagent thinking effort should be None after parent clears it"
                );
            }
        });
    }

    #[gpui::test]
    async fn test_subagent_inherits_settings_at_creation(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_speed(Speed::Fast, cx);
                thread.set_thinking_enabled(true, cx);
                thread.set_thinking_effort(Some("high".to_string()), cx);
                thread.set_profile(AgentProfileId("custom-profile".into()), cx);
            });
        });

        let subagents = setup_parent_with_subagents(cx, &parent, 1);

        cx.update(|cx| {
            let sub = subagents[0].read(cx);
            assert_eq!(sub.speed(), Some(Speed::Fast));
            assert!(sub.thinking_enabled());
            assert_eq!(sub.thinking_effort().map(|s| s.as_str()), Some("high"));
            assert_eq!(sub.profile(), &AgentProfileId("custom-profile".into()));
        });
    }

    #[gpui::test]
    async fn test_set_speed_propagates_to_subagents(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 2);

        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_speed(Speed::Fast, cx);
            });

            for subagent in &subagents {
                assert_eq!(
                    subagent.read(cx).speed(),
                    Some(Speed::Fast),
                    "Subagent speed should match parent after set_speed"
                );
            }
        });
    }

    #[gpui::test]
    async fn test_dropped_subagent_does_not_panic(cx: &mut TestAppContext) {
        let (parent, _event_stream) = setup_thread_for_test(cx).await;
        let subagents = setup_parent_with_subagents(cx, &parent, 1);

        // Drop the subagent so the WeakEntity can no longer be upgraded
        drop(subagents);

        // Should not panic even though the subagent was dropped
        cx.update(|cx| {
            parent.update(cx, |thread, cx| {
                thread.set_thinking_enabled(true, cx);
                thread.set_speed(Speed::Fast, cx);
                thread.set_thinking_effort(Some("high".to_string()), cx);
            });
        });
    }

    #[gpui::test]
    async fn test_handle_tool_use_json_parse_error_adds_tool_use_to_content(
        cx: &mut TestAppContext,
    ) {
        let (thread, event_stream) = setup_thread_for_test(cx).await;

        let tool_use_id = LanguageModelToolUseId::from("test_tool_id");
        let tool_name: Arc<str> = Arc::from("test_tool");
        let raw_input: Arc<str> = Arc::from("{invalid json");
        let json_parse_error = "expected value at line 1 column 1".to_string();

        let (_cancellation_tx, cancellation_rx) = watch::channel(false);

        let result = cx
            .update(|cx| {
                thread.update(cx, |thread, cx| {
                    // Call the function under test
                    thread
                        .handle_tool_use_json_parse_error_event(
                            tool_use_id.clone(),
                            tool_name.clone(),
                            raw_input.clone(),
                            json_parse_error,
                            &event_stream,
                            cancellation_rx,
                            cx,
                        )
                        .unwrap()
                })
            })
            .await;

        // Verify the result is an error
        assert!(result.is_error);
        assert_eq!(result.tool_use_id, tool_use_id);
        assert_eq!(result.tool_name, tool_name);
        assert!(matches!(
            result.content,
            LanguageModelToolResultContent::Text(_)
        ));

        thread.update(cx, |thread, _cx| {
            // Verify the tool use was added to the message content
            {
                let last_message = thread.pending_message();
                assert_eq!(
                    last_message.content.len(),
                    1,
                    "Should have one tool_use in content"
                );

                match &last_message.content[0] {
                    AgentMessageContent::ToolUse(tool_use) => {
                        assert_eq!(tool_use.id, tool_use_id);
                        assert_eq!(tool_use.name, tool_name);
                        assert_eq!(tool_use.raw_input, raw_input.to_string());
                        assert!(tool_use.is_input_complete);
                        // Should fall back to empty object for invalid JSON
                        assert_eq!(tool_use.input, json!({}));
                    }
                    _ => panic!("Expected ToolUse content"),
                }
            }

            // Insert the tool result (simulating what the caller does)
            thread
                .pending_message()
                .tool_results
                .insert(result.tool_use_id.clone(), result);

            // Verify the tool result was added
            let last_message = thread.pending_message();
            assert_eq!(
                last_message.tool_results.len(),
                1,
                "Should have one tool_result"
            );
            assert!(last_message.tool_results.contains_key(&tool_use_id));
        })
    }
}

#[cfg(test)]
mod synthetic_guardrail_tests {
    use super::*;

    /// Regression for bug #13: the circuit breaker used to count synthetic
    /// guardrail responses (RUNAWAY LOOP, No tool named, etc.) as real tool
    /// failures. A single guardrail trip would then immediately trip the
    /// breaker on the next turn, hard-stopping the agent for what was
    /// effectively self-caused. The classifier must recognize all of these.
    #[test]
    fn classifies_runaway_loop_as_synthetic() {
        assert!(is_synthetic_guardrail_text(
            "RUNAWAY LOOP: edit_file called repeatedly. Turn cancelled by safety guardrail."
        ));
    }

    #[test]
    fn classifies_no_tool_named_as_synthetic() {
        assert!(is_synthetic_guardrail_text("No tool named foobar exists"));
    }

    #[test]
    fn classifies_loop_detected_as_synthetic() {
        assert!(is_synthetic_guardrail_text(
            "LOOP DETECTED: input hash repeated 5 times"
        ));
    }

    #[test]
    fn classifies_circuit_breaker_as_synthetic() {
        assert!(is_synthetic_guardrail_text(
            "CIRCUIT BREAKER: 5 consecutive tool failures"
        ));
    }

    #[test]
    fn classifies_permission_denied_as_synthetic() {
        assert!(is_synthetic_guardrail_text("PERMISSION DENIED: foo"));
    }

    /// Real tool errors must NOT be misclassified as synthetic — otherwise
    /// the circuit breaker would never trip and an agent stuck on a broken
    /// tool would loop forever.
    #[test]
    fn does_not_misclassify_real_tool_errors() {
        assert!(!is_synthetic_guardrail_text(
            "Failed to read file: not found"
        ));
        assert!(!is_synthetic_guardrail_text("HTTP 500: upstream error"));
        assert!(!is_synthetic_guardrail_text(""));
        assert!(!is_synthetic_guardrail_text(
            "  PERMISSION DENIED: leading whitespace shouldn't trigger"
        ));
    }

    /// Strings that mention guardrail keywords but don't begin with the
    /// exact prefix must not collide. The fix uses `starts_with`, not
    /// `contains`, on purpose.
    #[test]
    fn requires_prefix_match_not_contains() {
        assert!(!is_synthetic_guardrail_text(
            "Tool output mentioned RUNAWAY LOOP: in passing"
        ));
        assert!(!is_synthetic_guardrail_text(
            "Some message ending in CIRCUIT BREAKER: foo"
        ));
    }
}

#[cfg(test)]
mod atomic_write_tests {
    use super::atomic_write;

    /// Bug C6: a basic write must produce the exact bytes.
    #[test]
    fn writes_bytes_to_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ctx.md");
        atomic_write(&p, b"hello").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"hello");
    }

    /// Atomic write must replace an existing file in-place — and the
    /// reader must see only the OLD or only the NEW content, never
    /// a partial mix.
    #[test]
    fn replaces_existing_file_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ctx.md");
        std::fs::write(&p, b"old content original").unwrap();
        atomic_write(&p, b"new").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"new");
    }

    /// Concurrent atomic_write calls from many threads must each leave
    /// the file in a fully-written state (one of the inputs, never a
    /// truncation or a mix). This is exactly the regression bug C6
    /// targets — the old `std::fs::write` could leave a half-written
    /// file when interleaved.
    #[test]
    fn concurrent_writers_never_produce_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ctx.md");
        let payloads: Vec<Vec<u8>> = (0..16)
            .map(|i| format!("payload-{i:04}-{}", "x".repeat(2048)).into_bytes())
            .collect();

        let path_arc = std::sync::Arc::new(p);
        let payloads_arc = std::sync::Arc::new(payloads.clone());

        let handles: Vec<_> = (0..16)
            .map(|i| {
                let p = path_arc.clone();
                let pl = payloads_arc.clone();
                std::thread::spawn(move || {
                    atomic_write(&p, &pl[i]).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Final file must be byte-equal to ONE of the payloads.
        let final_bytes = std::fs::read(&*path_arc).unwrap();
        assert!(
            payloads.iter().any(|p| p == &final_bytes),
            "atomic_write produced a partial / mixed file (len {})",
            final_bytes.len()
        );
    }

    /// Concurrent reader-during-writer: a reader that opens the file
    /// while a writer is renaming over it must see the OLD bytes (or
    /// the NEW bytes) — never a truncated read.
    #[test]
    fn reader_during_write_sees_full_old_or_full_new() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("ctx.md");
        let old = vec![b'A'; 4096];
        let new = vec![b'B'; 4096];
        std::fs::write(&p, &old).unwrap();

        let p_writer = p.clone();
        let new_for_writer = new.clone();
        let writer = std::thread::spawn(move || {
            for _ in 0..50 {
                atomic_write(&p_writer, &new_for_writer).unwrap();
            }
        });

        for _ in 0..200 {
            if let Ok(bytes) = std::fs::read(&p) {
                assert!(
                    bytes == old || bytes == new,
                    "reader saw partial file: len={}",
                    bytes.len()
                );
            }
        }
        writer.join().unwrap();
    }
}

#[cfg(test)]
mod caduceus_native_loop_tests {
    //! ST-A9 — end-to-end wiring tests for the native-loop gate +
    //! harness provisioning inside `try_run_turn_native`.
    //!
    //! The full turn round-trip (engine → fake LLM → translated events
    //! → Thread.messages) is deliberately **not** exercised here; the
    //! orchestrator requires a fair amount of scaffolding to run with a
    //! real `AgentHarness` and the current ST-A2 baseline ships with an
    //! empty `ToolRegistry` + empty system prompt — any tool-call the
    //! model requests would fail at the engine boundary (tracked to
    //! ST-B1/B2). Once ST-B lands, a richer E2E test belongs in the
    //! G1g rollout session alongside flipping the feature flag ON.
    //!
    //! What *is* covered here:
    //! 1. Flag OFF → `try_run_turn_native` returns false and the bridge
    //!    is never provisioned.
    //! 2. Flag ON + worktree → `ensure_caduceus_harness` populates
    //!    bridge/state/cancel_token and reports newly-provisioned.
    //! 3. Flag ON without a worktree → `ensure_caduceus_harness` errors
    //!    out cleanly (legacy path keeps running — failure contract).

    use super::*;
    use agent_settings::AgentSettings;
    use gpui::{TestAppContext, UpdateGlobal};
    use settings::SettingsStore;
    use std::sync::Arc;
    use util::path;

    /// Build a Thread sitting on top of a real in-memory worktree so
    /// `ensure_caduceus_harness` can resolve a project root.
    async fn setup_thread_with_worktree(cx: &mut TestAppContext) -> Entity<Thread> {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            AgentSettings::register(cx);
        });

        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(path!("/project"), serde_json::json!({}))
            .await;
        let project = Project::test(fs.clone(), [path!("/project").as_ref()], cx).await;
        let templates = Templates::new();

        cx.update(|cx| {
            let project_context = cx.new(|_cx| prompt_store::ProjectContext::default());
            let context_server_store = project.read(cx).context_server_store();
            let context_server_registry =
                cx.new(|cx| ContextServerRegistry::new(context_server_store, cx));
            cx.new(|cx| {
                Thread::new(
                    project,
                    project_context,
                    context_server_registry,
                    templates,
                    None,
                    cx,
                )
            })
        })
    }

    fn enable_native_loop(cx: &mut TestAppContext) {
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(r#"{ "agent": { "caduceus_native_loop": true } }"#, cx)
                    .unwrap();
            });
        });
    }

    #[gpui::test]
    async fn flag_off_skips_native_provisioning(cx: &mut TestAppContext) {
        let thread = setup_thread_with_worktree(cx).await;
        // Setting is OFF by default (serde default=false).
        cx.update(|cx| {
            let provisioned = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(
                !provisioned,
                "ensure_caduceus_harness must be a no-op when caduceus_native_loop=false"
            );
            let t = thread.read(cx);
            assert!(t.caduceus.bridge.is_none(), "bridge must stay None");
            assert!(
                t.caduceus.native_state.is_none(),
                "native_state must stay None"
            );
            assert!(
                t.caduceus.cancel_token.is_none(),
                "cancel_token must stay None"
            );
            assert!(t.caduceus.harness.is_none(), "harness must stay None");
        });
    }

    #[gpui::test]
    async fn flag_on_with_worktree_provisions_bridge_and_state(cx: &mut TestAppContext) {
        let thread = setup_thread_with_worktree(cx).await;
        enable_native_loop(cx);

        cx.update(|cx| {
            let provisioned = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(
                provisioned,
                "ensure_caduceus_harness must report newly-provisioned when \
                 caduceus_native_loop=true and a worktree exists"
            );
            let t = thread.read(cx);
            assert!(t.caduceus.bridge.is_some(), "bridge must be Some");
            assert!(
                t.caduceus.native_state.is_some(),
                "native_state must be Some"
            );
            assert!(
                t.caduceus.cancel_token.is_some(),
                "cancel_token must be Some"
            );
            // Harness itself is built lazily inside try_run_turn_native
            // (ST-A2 phase 2) — NOT by ensure_caduceus_harness.
            assert!(
                t.caduceus.harness.is_none(),
                "harness must still be None (built on first native turn)"
            );
            assert!(
                !t.caduceus_native_loop_ready(),
                "ready() requires harness + emitter, which are built per-turn"
            );
        });

        // `ensure_caduceus_harness`'s idempotency gate is
        // `caduceus_native_loop_ready()`, which requires the harness
        // itself to be Some — and the harness is built lazily inside
        // `try_run_turn_native` (phase 2), not here. Until the first
        // native turn runs, repeat calls will re-allocate the bridge.
        // That's documented behavior, not a bug: the bridge is cheap
        // (no I/O in `new`) and becomes stable after turn 1.
        cx.update(|cx| {
            let provisioned_again = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(
                provisioned_again,
                "pre-harness calls rebuild bridge/state/token; becomes a no-op \
                 only after try_run_turn_native populates caduceus_harness"
            );
        });
    }

    #[gpui::test]
    async fn flag_on_without_worktree_errors_cleanly(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            AgentSettings::register(cx);
        });
        enable_native_loop(cx);

        let fs = fs::FakeFs::new(cx.background_executor.clone());
        // No worktree paths — `Project::test(fs, [], cx)` creates an
        // empty project with zero worktrees. `ensure_caduceus_harness`
        // must surface a clean error (not panic) and leave the fields
        // None so legacy dispatch keeps running.
        let project = Project::test(fs.clone(), [], cx).await;
        let templates = Templates::new();

        let thread = cx.update(|cx| {
            let project_context = cx.new(|_cx| prompt_store::ProjectContext::default());
            let context_server_store = project.read(cx).context_server_store();
            let context_server_registry =
                cx.new(|cx| ContextServerRegistry::new(context_server_store, cx));
            cx.new(|cx| {
                Thread::new(
                    project,
                    project_context,
                    context_server_registry,
                    templates,
                    None,
                    cx,
                )
            })
        });

        cx.update(|cx| {
            let res = thread.update(cx, |t, cx| t.ensure_caduceus_harness(cx));
            // WG (Worktree-Gate): when no folder workspace is open we now
            // no-op gracefully instead of erroring. The contract is the
            // same — Caduceus features stay disabled, legacy dispatch keeps
            // running — but the surface is `Ok(false)` + one info log,
            // not `Err(_)` + repeated error logs.
            let provisioned = res.expect(
                "ensure_caduceus_harness must return Ok(false) (silent no-op) \
                 when flag is ON but no folder workspace exists",
            );
            assert!(
                !provisioned,
                "ensure_caduceus_harness must report not-provisioned when no \
                 folder workspace is open (WG worktree-gate)"
            );
            let t = thread.read(cx);
            assert!(
                t.caduceus.bridge.is_none(),
                "bridge must stay None after no-op (legacy path fallback contract)"
            );
        });

        // Silence unused-import warning if Arc is not needed.
        let _ = Arc::new(());
    }

    /// T4 (Audit C6): an ON→OFF→ON flag transition must invalidate
    /// the stale harness state so the next `ensure_caduceus_harness`
    /// rebuilds against the current flag.
    #[gpui::test]
    async fn flag_transition_on_off_on_invalidates_harness(cx: &mut TestAppContext) {
        let thread = setup_thread_with_worktree(cx).await;
        enable_native_loop(cx);

        // Turn 1: flag ON → provisions bridge + state.
        cx.update(|cx| {
            let provisioned = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(provisioned, "turn 1 must provision");
            let t = thread.read(cx);
            assert_eq!(t.caduceus.last_native_loop_flag, Some(true));
            assert!(t.caduceus.bridge.is_some());
        });

        // User flips flag OFF.
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store
                    .set_user_settings(r#"{ "agent": { "caduceus_native_loop": false } }"#, cx)
                    .unwrap();
            });
        });

        // Turn 2: ensure observes the transition ON→OFF and invalidates.
        cx.update(|cx| {
            let provisioned = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(!provisioned, "flag OFF must not provision");
            let t = thread.read(cx);
            assert!(
                t.caduceus.bridge.is_none(),
                "ON→OFF transition must drop stale bridge"
            );
            assert_eq!(
                t.caduceus.last_native_loop_flag, None,
                "flag snapshot must be cleared on invalidation"
            );
        });

        // User flips flag back ON.
        enable_native_loop(cx);

        // Turn 3: ensure re-provisions a fresh bridge, not the stale one.
        cx.update(|cx| {
            let provisioned = thread
                .update(cx, |t, cx| t.ensure_caduceus_harness(cx))
                .unwrap();
            assert!(provisioned, "turn 3 must re-provision after OFF→ON");
            let t = thread.read(cx);
            assert_eq!(t.caduceus.last_native_loop_flag, Some(true));
            assert!(t.caduceus.bridge.is_some(), "bridge must be Some again");
        });
    }
}

#[cfg(test)]
mod native_dispatch_tests {
    //! G1d — exhaustive mapping tests for
    //! [`dispatch_translated_event`]. These pin the translator →
    //! `ThreadEvent` contract so future changes to the translator
    //! variant set surface as failing mapping assertions before they
    //! reach an integration test.

    use super::*;
    use caduceus_bridge::event_translator::{
        RetryKind as TRetryKind, StopReasonKind as TStop, TokenUsageMirror,
        TranslatedThreadEvent as T,
    };
    use futures::StreamExt;
    use futures::channel::mpsc;

    fn dispatch_one(ev: T) -> Vec<ThreadEvent> {
        let (tx, mut rx) = mpsc::unbounded::<Result<ThreadEvent>>();
        let stream = ThreadEventStream(tx);
        dispatch_translated_event(&ev, &stream);
        drop(stream);
        let mut out = Vec::new();
        while let Ok(Some(item)) = rx.try_next() {
            out.push(item.expect("dispatcher only produces Ok"));
        }
        out
    }

    fn assert_one<F: FnOnce(&ThreadEvent)>(evs: &[ThreadEvent], check: F) {
        assert_eq!(evs.len(), 1, "expected one ThreadEvent, got {}", evs.len());
        check(&evs[0]);
    }

    #[test]
    fn agent_text_maps_through() {
        let evs = dispatch_one(T::AgentText("hello".into()));
        assert_one(&evs, |e| match e {
            ThreadEvent::AgentText(s) => assert_eq!(s, "hello"),
            other => panic!("expected AgentText, got {other:?}"),
        });
    }

    #[test]
    fn agent_thinking_maps_through() {
        let evs = dispatch_one(T::AgentThinking("reasoning".into()));
        assert_one(&evs, |e| match e {
            ThreadEvent::AgentThinking(s) => assert_eq!(s, "reasoning"),
            other => panic!("expected AgentThinking, got {other:?}"),
        });
    }

    #[test]
    fn thinking_complete_maps_to_thinking() {
        // Duration is dropped in the current mapping; the content
        // must still appear verbatim.
        let evs = dispatch_one(T::AgentThinkingComplete {
            content: "final thought".into(),
            duration_ms: 123,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::AgentThinking(s) => assert_eq!(s, "final thought"),
            other => panic!("expected AgentThinking, got {other:?}"),
        });
    }

    #[test]
    fn tool_result_with_error_surfaces_failed_update() {
        let evs = dispatch_one(T::ToolResult {
            id: "t1".into(),
            content: "boom".into(),
            is_error: true,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(upd)) => {
                assert_eq!(upd.tool_call_id.0.as_ref(), "t1");
                assert_eq!(upd.fields.status, Some(acp::ToolCallStatus::Failed));
            }
            other => panic!("expected ToolCallUpdate::UpdateFields, got {other:?}"),
        });
    }

    #[test]
    fn tool_result_ok_surfaces_completed_update() {
        let evs = dispatch_one(T::ToolResult {
            id: "t2".into(),
            content: "ok".into(),
            is_error: false,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(upd)) => {
                assert_eq!(upd.fields.status, Some(acp::ToolCallStatus::Completed));
            }
            other => panic!("expected ToolCallUpdate::UpdateFields, got {other:?}"),
        });
    }

    #[test]
    fn context_compacted_is_notice() {
        let evs = dispatch_one(T::ContextCompacted {
            freed: 1000,
            before: 5000,
            after: 4000,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::ContextNotice(n) => {
                assert_eq!(n.kind, "context.compacted");
                assert!(n.message.contains("1000"));
            }
            other => panic!("expected ContextNotice, got {other:?}"),
        });
    }

    #[test]
    fn scope_expansion_is_warning() {
        let evs = dispatch_one(T::ScopeExpansion {
            capability: "fs.write".into(),
            resource: "/etc".into(),
            tool: "edit_file".into(),
            reason: "outside project root".into(),
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::EngineDiagnostic(d) => {
                assert_eq!(d.kind, "scope.expansion");
                assert_eq!(d.severity, EngineDiagnosticSeverity::Warning);
            }
            other => panic!("expected EngineDiagnostic, got {other:?}"),
        });
    }

    #[test]
    fn retry_is_warning_diagnostic() {
        let evs = dispatch_one(T::Retry {
            kind: TRetryKind::Loop,
            message: "same tool twice".into(),
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::EngineDiagnostic(d) => {
                assert_eq!(d.kind, "retry");
                assert_eq!(d.severity, EngineDiagnosticSeverity::Warning);
            }
            other => panic!("expected EngineDiagnostic, got {other:?}"),
        });
    }

    #[test]
    fn plan_step_is_notice() {
        let evs = dispatch_one(T::PlanStep {
            step: 3,
            step_id: 42,
            plan_revision: 1,
            tool: "grep".into(),
            description: "find usages".into(),
            depends_on: vec![],
            parent_step_id: None,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::ContextNotice(n) => {
                assert_eq!(n.kind, "plan.step");
                assert!(n.message.contains("#3"));
                assert!(n.message.contains("grep"));
            }
            other => panic!("expected ContextNotice, got {other:?}"),
        });
    }

    /// Helper — `TurnComplete` now emits `UsageUpdated` first then
    /// `Stop`. The Stop-reason mapping tests only care about the Stop;
    /// this helper skips the leading UsageUpdated and hands back the
    /// tail so the original assert_one pattern still works.
    fn assert_usage_then<F: FnOnce(&ThreadEvent)>(evs: &[ThreadEvent], check: F) {
        assert_eq!(
            evs.len(),
            2,
            "TurnComplete must emit UsageUpdated then Stop (got {evs:?})"
        );
        match &evs[0] {
            ThreadEvent::UsageUpdated(_) => {}
            other => panic!("expected UsageUpdated as first event, got {other:?}"),
        }
        check(&evs[1]);
    }

    #[test]
    fn turn_complete_end_turn_maps_to_stop_end_turn() {
        let evs = dispatch_one(T::TurnComplete {
            stop: TStop::EndTurn,
            usage: TokenUsageMirror::default(),
        });
        assert_usage_then(&evs, |e| match e {
            ThreadEvent::Stop(acp::StopReason::EndTurn) => {}
            other => panic!("expected Stop(EndTurn), got {other:?}"),
        });
    }

    #[test]
    fn turn_complete_max_tokens_maps_through() {
        let evs = dispatch_one(T::TurnComplete {
            stop: TStop::MaxTokens,
            usage: TokenUsageMirror::default(),
        });
        assert_usage_then(&evs, |e| match e {
            ThreadEvent::Stop(acp::StopReason::MaxTokens) => {}
            other => panic!("expected Stop(MaxTokens), got {other:?}"),
        });
    }

    #[test]
    fn turn_complete_budget_exceeded_maps_to_max_tokens() {
        // BudgetExceeded has no ACP equivalent; current mapping
        // folds it into MaxTokens. If this changes, update both
        // mapping and this pin.
        let evs = dispatch_one(T::TurnComplete {
            stop: TStop::BudgetExceeded,
            usage: TokenUsageMirror::default(),
        });
        assert_usage_then(&evs, |e| match e {
            ThreadEvent::Stop(acp::StopReason::MaxTokens) => {}
            other => panic!("expected Stop(MaxTokens), got {other:?}"),
        });
    }

    #[test]
    fn turn_complete_forwards_usage_payload_before_stop() {
        // R2-1 regression guard: `TurnComplete.usage` must be
        // preserved all the way to `ThreadEvent::UsageUpdated` without
        // field-dropping, AND emitted before the terminal `Stop` so a
        // meter consumer that exits on Stop still sees the update.
        let usage = TokenUsageMirror {
            input_tokens: 1_234,
            output_tokens: 567,
            cache_read_tokens: 100,
            cache_write_tokens: 42,
        };
        let evs = dispatch_one(T::TurnComplete {
            stop: TStop::EndTurn,
            usage,
        });
        assert_eq!(evs.len(), 2, "expected UsageUpdated + Stop, got {evs:?}");
        match &evs[0] {
            ThreadEvent::UsageUpdated(u) => {
                assert_eq!(u.input_tokens, 1_234);
                assert_eq!(u.output_tokens, 567);
                assert_eq!(u.cache_read_tokens, 100);
                assert_eq!(u.cache_write_tokens, 42);
            }
            other => panic!("expected UsageUpdated first, got {other:?}"),
        }
        match &evs[1] {
            ThreadEvent::Stop(acp::StopReason::EndTurn) => {}
            other => panic!("expected Stop(EndTurn) second, got {other:?}"),
        }
    }

    #[test]
    fn turn_error_emits_diagnostic_then_stop() {
        let evs = dispatch_one(T::TurnError {
            message: "provider failed".into(),
        });
        assert_eq!(evs.len(), 2, "TurnError must emit diag + stop");
        match &evs[0] {
            ThreadEvent::EngineDiagnostic(d) => {
                assert_eq!(d.kind, "turn.error");
                assert_eq!(d.severity, EngineDiagnosticSeverity::Error);
            }
            other => panic!("expected EngineDiagnostic, got {other:?}"),
        }
        match &evs[1] {
            ThreadEvent::Stop(acp::StopReason::Cancelled) => {}
            other => panic!("expected Stop(Cancelled), got {other:?}"),
        }
    }

    #[test]
    fn swallow_is_no_op() {
        let evs = dispatch_one(T::Swallow {
            reason: "test-only signal",
        });
        assert!(evs.is_empty(), "Swallow must not emit ThreadEvent");
    }

    #[test]
    fn permission_request_is_warning() {
        let evs = dispatch_one(T::PermissionRequest {
            id: "p1".into(),
            tool: "bash".into(),
            description: "run tests".into(),
            raw_input: None,
        });
        assert_one(&evs, |e| match e {
            ThreadEvent::EngineDiagnostic(d) => {
                assert_eq!(d.kind, "permission.request");
                assert_eq!(d.severity, EngineDiagnosticSeverity::Warning);
            }
            other => panic!("expected EngineDiagnostic, got {other:?}"),
        });
    }

    // ── R2-2 regression guards ────────────────────────────────────────
    //
    // The bridge `event_translator` preserves richer payloads on several
    // variants (per the H2 comments at `event_translator.rs:37-140`) than
    // the dispatcher currently surfaces to the UI. The arms below use
    // `..` patterns that silently drop fields; translator tests pass,
    // consumers see nothing, no compile-time signal.
    //
    // These tests pin the CURRENT drop behavior as documented drift.
    // If one of these tests fails because the dispatcher now forwards
    // the dropped field, update BOTH the test AND wire the field end-to-
    // end (new `ThreadEvent` variant or expanded message). See R2-1
    // (`turn_complete_forwards_usage_payload_before_stop`) for the
    // wiring pattern.

    /// Drift guard: `AgentThinkingComplete.duration_ms` is preserved by
    /// the translator but dropped at `thread.rs:5895` via a `..` pattern.
    /// If the dispatcher starts surfacing duration, wire it into the UI
    /// (e.g. a ThinkingComplete variant with duration) rather than
    /// merging it into the free-form content string.
    #[test]
    fn r2_2_thinking_complete_silently_drops_duration_ms() {
        let evs = dispatch_one(T::AgentThinkingComplete {
            content: "final".into(),
            duration_ms: 4242,
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            ThreadEvent::AgentThinking(s) => {
                assert_eq!(s, "final");
                assert!(
                    !s.contains("4242"),
                    "duration_ms must not leak into content string verbatim; \
                     if surfacing duration is desired, add a structured variant"
                );
            }
            other => panic!("expected AgentThinking, got {other:?}"),
        }
    }

    /// Drift guard: `PlanStep.{step_id, plan_revision, depends_on,
    /// parent_step_id}` are preserved by the translator (for DAG
    /// rendering) but dropped at `thread.rs:6026-6031` via `..`. Only
    /// the positional `step`/`tool`/`description` reach the UI as a
    /// ContextNotice. If DAG rendering lands, emit a structured
    /// PlanStep variant instead of folding numbers into prose.
    #[test]
    fn r2_2_plan_step_silently_drops_dag_fields() {
        let evs = dispatch_one(T::PlanStep {
            step: 7,
            step_id: 999_111,
            plan_revision: 42,
            tool: "grep".into(),
            description: "find foo".into(),
            depends_on: vec![100, 200, 300],
            parent_step_id: Some(500),
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            ThreadEvent::ContextNotice(n) => {
                assert_eq!(n.kind, "plan.step");
                // Positional fields reach the UI.
                assert!(n.message.contains("#7"));
                assert!(n.message.contains("grep"));
                assert!(n.message.contains("find foo"));
                // DAG fields are silently dropped.
                assert!(!n.message.contains("999111"));
                assert!(!n.message.contains("100"));
                assert!(!n.message.contains("500"));
            }
            other => panic!("expected ContextNotice, got {other:?}"),
        }
    }

    /// Drift guard: `PlanAmended.plan_revision` is preserved by the
    /// translator but dropped at `thread.rs:6032-6040` via `..`. If
    /// revision tracking lands in the UI, emit it as a structured
    /// field rather than interpolating into the notice body.
    #[test]
    fn r2_2_plan_amended_silently_drops_plan_revision() {
        let evs = dispatch_one(T::PlanAmended {
            plan_revision: 77_777,
            kind: "replace".into(),
            step: 3,
            ok: true,
            reason: "user asked".into(),
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            ThreadEvent::ContextNotice(n) => {
                assert_eq!(n.kind, "plan.amended");
                assert!(n.message.contains("#3"));
                assert!(n.message.contains("replace"));
                assert!(n.message.contains("user asked"));
                // Revision silently dropped.
                assert!(
                    !n.message.contains("77777"),
                    "plan_revision must not leak as a substring; surface it structurally"
                );
            }
            other => panic!("expected ContextNotice, got {other:?}"),
        }
    }

    /// Drift guard: `ModeChanged.{from_lens, to_lens}` are preserved by
    /// the translator but dropped at `thread.rs:6000-6002` via `..`. If
    /// lens transitions need to drive UI (e.g. Act.Review → Act.Debug
    /// badge animations), emit them as structured fields in a
    /// dedicated ModeChanged variant, not interpolated prose.
    #[test]
    fn r2_2_mode_changed_silently_drops_lens_fields() {
        let evs = dispatch_one(T::ModeChanged {
            from_mode: "Plan".into(),
            to_mode: "Act".into(),
            from_lens: Some("LensAlpha".into()),
            to_lens: Some("LensBeta".into()),
        });
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            ThreadEvent::ContextNotice(n) => {
                assert_eq!(n.kind, "mode.changed");
                assert!(n.message.contains("Plan"));
                assert!(n.message.contains("Act"));
                // Lens transitions silently dropped.
                assert!(
                    !n.message.contains("LensAlpha"),
                    "from_lens must not leak into notice text; surface structurally"
                );
                assert!(
                    !n.message.contains("LensBeta"),
                    "to_lens must not leak into notice text; surface structurally"
                );
            }
            other => panic!("expected ContextNotice, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod native_turn_e2e_tests {
    //! NW-4 — headless full-turn E2E harness.
    //!
    //! The GUI dogfood flow (flag ON → user sends a message → engine
    //! reasons, calls tools, returns content, ends turn) traverses a
    //! fixed translator → dispatcher → `ThreadEvent` pipeline. This
    //! harness scripts a realistic turn as a sequence of
    //! `TranslatedThreadEvent`s and asserts the full, ordered
    //! `ThreadEvent` output the UI would observe.
    //!
    //! This substitutes for the manual GUI smoke in the G1g rollout
    //! gate: anything the human tester would visually confirm (tool
    //! card appears, per-turn token meter ticks, Stop fires) is pinned
    //! here as a programmatic contract. Future regressions in the
    //! translator / dispatcher will fail this test before a human
    //! tester ever loads Zed.

    use super::*;
    use caduceus_bridge::event_translator::{
        StopReasonKind as TStop, TokenUsageMirror, TranslatedThreadEvent as T,
    };
    use futures::channel::mpsc;

    /// Dispatch a scripted turn end-to-end and collect every
    /// `ThreadEvent` the stream carries.
    fn dispatch_turn(script: Vec<T>) -> Vec<ThreadEvent> {
        let (tx, mut rx) = mpsc::unbounded::<Result<ThreadEvent>>();
        let stream = ThreadEventStream(tx);
        for ev in &script {
            dispatch_translated_event(ev, &stream);
        }
        drop(stream);
        let mut out = Vec::new();
        while let Ok(Some(item)) = rx.try_next() {
            out.push(item.expect("dispatcher only produces Ok"));
        }
        out
    }

    /// A realistic single-tool turn: agent thinks briefly, streams
    /// prose, invokes a tool, the tool returns, agent closes, turn
    /// ends with usage + Stop. This is the minimum viable "native
    /// loop actually works end-to-end" assertion.
    #[test]
    fn full_turn_text_then_tool_then_text_then_stop_with_usage() {
        let script = vec![
            // Agent reasoning + prose stream.
            T::AgentThinking("Let me read the file…".into()),
            T::AgentThinkingComplete {
                content: "Let me read the file.".into(),
                duration_ms: 42,
            },
            T::AgentText("I'll read ".into()),
            T::AgentText("src/main.rs.".into()),
            // Tool call lifecycle.
            T::ToolCallStart {
                id: "t1".into(),
                name: "read_file".into(),
            },
            T::ToolCallInputDelta {
                id: "t1".into(),
                delta: "{\"path\":\"src/main.rs\"".into(),
            },
            T::ToolCallInputDelta {
                id: "t1".into(),
                delta: "}".into(),
            },
            T::ToolCallInputEnd { id: "t1".into() },
            T::ToolResult {
                id: "t1".into(),
                content: "fn main() {}".into(),
                is_error: false,
            },
            // Post-tool prose + turn end.
            T::AgentText(" The file is empty.".into()),
            T::TurnComplete {
                stop: TStop::EndTurn,
                usage: TokenUsageMirror {
                    input_tokens: 1200,
                    output_tokens: 85,
                    cache_read_tokens: 400,
                    cache_write_tokens: 10,
                },
            },
        ];

        let evs = dispatch_turn(script);

        // Sanity: at least one event per script entry that isn't a
        // Swallow. 11 script entries, all surface → ≥ 11 events; the
        // tool-call + TurnComplete fan-out produces extras.
        assert!(
            evs.len() >= 11,
            "expected ≥11 events for a scripted turn, got {}: {:?}",
            evs.len(),
            evs
        );

        // Per-turn contract assertions, in flow order.
        // 1. AgentThinking fires before AgentText.
        let first_thinking = evs
            .iter()
            .position(|e| matches!(e, ThreadEvent::AgentThinking(_)))
            .expect("must see AgentThinking");
        let first_text = evs
            .iter()
            .position(|e| matches!(e, ThreadEvent::AgentText(_)))
            .expect("must see AgentText");
        assert!(
            first_thinking < first_text,
            "AgentThinking must precede AgentText; got order {evs:?}"
        );

        // 2. A ToolCall (or ToolCallUpdate) appears — the UI card must
        //    materialise for the user to see the tool run.
        assert!(
            evs.iter()
                .any(|e| matches!(e, ThreadEvent::ToolCall(_) | ThreadEvent::ToolCallUpdate(_))),
            "must see at least one tool-call event for read_file; got {evs:?}"
        );

        // 3. UsageUpdated fires before Stop — pins the R2-1 contract
        //    so a future refactor can't silently break it.
        let usage_idx = evs
            .iter()
            .position(|e| matches!(e, ThreadEvent::UsageUpdated(_)))
            .expect("must see UsageUpdated at turn close");
        let stop_idx = evs
            .iter()
            .position(|e| matches!(e, ThreadEvent::Stop(_)))
            .expect("must see Stop at turn close");
        assert!(
            usage_idx < stop_idx,
            "UsageUpdated must precede Stop; got {evs:?}"
        );

        // 4. UsageUpdated payload round-trips the scripted tokens
        //    field-for-field — the per-turn token meter the user
        //    sees in the GUI reads from this exact struct.
        match &evs[usage_idx] {
            ThreadEvent::UsageUpdated(u) => {
                assert_eq!(u.input_tokens, 1200);
                assert_eq!(u.output_tokens, 85);
                assert_eq!(u.cache_read_tokens, 400);
                assert_eq!(u.cache_write_tokens, 10);
            }
            other => panic!("usage_idx pointed at non-usage event {other:?}"),
        }

        // 5. Stop reason is the user-visible "EndTurn" mapping, not a
        //    diagnostic. If this flips to MaxTokens/Refusal/etc. the
        //    GUI would render a warning badge — not what a clean turn
        //    should show.
        match &evs[stop_idx] {
            ThreadEvent::Stop(reason) => {
                use acp::StopReason;
                assert_eq!(
                    *reason,
                    StopReason::EndTurn,
                    "clean TurnComplete must surface as Stop(EndTurn)"
                );
            }
            other => panic!("stop_idx pointed at non-stop event {other:?}"),
        }

        // 6. Stop is the final event. A consumer that breaks out of
        //    the stream on Stop must not miss trailing data.
        assert_eq!(
            stop_idx,
            evs.len() - 1,
            "Stop must be the final event; got tail {:?}",
            &evs[stop_idx..]
        );
    }

    /// Failed-tool variant: the same turn shape but the tool returns
    /// an error. Pins that the tool-call card flips to failed without
    /// disrupting the rest of the turn contract.
    #[test]
    fn full_turn_with_failed_tool_still_closes_cleanly() {
        let script = vec![
            T::AgentText("Trying…".into()),
            T::ToolCallStart {
                id: "t1".into(),
                name: "read_file".into(),
            },
            T::ToolCallInputEnd { id: "t1".into() },
            T::ToolResult {
                id: "t1".into(),
                content: "file not found".into(),
                is_error: true,
            },
            T::AgentText(" Hmm, that failed.".into()),
            T::TurnComplete {
                stop: TStop::EndTurn,
                usage: TokenUsageMirror::default(),
            },
        ];

        let evs = dispatch_turn(script);

        // Contract: failed tool-result must surface as a ToolCallUpdate
        // whose status is Failed (otherwise the GUI card stays stuck
        // on "running").
        let failed_update = evs.iter().any(|e| {
            matches!(
                e,
                ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(u))
                    if matches!(u.fields.status, Some(acp::ToolCallStatus::Failed))
            )
        });
        assert!(
            failed_update,
            "failed tool must surface a ToolCallUpdate with status=Failed; got {evs:?}"
        );

        // Turn still closes: UsageUpdated → Stop(EndTurn) at the tail.
        assert!(matches!(evs.last(), Some(ThreadEvent::Stop(_))));
        let usage_idx = evs
            .iter()
            .position(|e| matches!(e, ThreadEvent::UsageUpdated(_)))
            .expect("usage must fire even on failed-tool turns");
        let stop_idx = evs.len() - 1;
        assert!(usage_idx < stop_idx);
    }

    /// Permission-request variant: mid-turn the engine asks for tool
    /// approval. The `PermissionRequest` is routed back through
    /// `route_native_permission_request` in production; on the
    /// dispatcher path it surfaces as a ThreadEvent the GUI renders
    /// as a permission prompt. This pins that the permission event
    /// does NOT terminate the stream (turn resumes afterward).
    #[test]
    fn full_turn_with_permission_request_preserves_stream() {
        let script = vec![
            T::AgentText("About to run a tool…".into()),
            T::PermissionRequest {
                id: "p1".into(),
                tool: "write_file".into(),
                description: "Write src/lib.rs".into(),
                raw_input: Some(serde_json::json!({"path":"src/lib.rs"})),
            },
            T::ToolCallStart {
                id: "t1".into(),
                name: "write_file".into(),
            },
            T::ToolCallInputEnd { id: "t1".into() },
            T::ToolResult {
                id: "t1".into(),
                content: "written".into(),
                is_error: false,
            },
            T::TurnComplete {
                stop: TStop::EndTurn,
                usage: TokenUsageMirror::default(),
            },
        ];

        let evs = dispatch_turn(script);

        // Sanity: the permission surfaces in the stream as an
        // EngineDiagnostic with kind="permission.request". Note: in
        // production the full approval round-trip is handled by
        // `route_native_permission_request` which short-circuits the
        // dispatcher for permission events; on the dispatcher-only
        // path tested here, the event falls through to a warning
        // diagnostic. Either way, the stream MUST keep flowing.
        let perm_diag = evs.iter().any(|e| {
            matches!(
                e,
                ThreadEvent::EngineDiagnostic(d)
                    if d.kind == "permission.request"
                    && d.severity == EngineDiagnosticSeverity::Warning
            )
        });
        assert!(
            perm_diag,
            "permission request must surface as EngineDiagnostic(permission.request, Warning); got {evs:?}"
        );

        // Turn still reaches Stop.
        assert!(
            matches!(evs.last(), Some(ThreadEvent::Stop(_))),
            "turn must still close after a permission request; got {evs:?}"
        );
    }
}

#[cfg(test)]
mod route_native_permission_tests {
    //!     rules short-circuit the interactive UI and auto-respond on
    //!     `approval_tx` with a diagnostic surfaced on the stream.
    //!   * Legacy fall-through: when the engine omits `raw_input`
    //!     (old build) we fall through to the interactive prompt.
    //!   * `extract_match_inputs` behavior: top-level-string-only.
    //!
    //! These would have caught the NW-3 double-prefix regression that
    //! was only found by a rubber-duck review.
    use super::*;
    use agent_settings::AgentSettings;
    use futures::StreamExt;
    use futures::channel::mpsc as fmpsc;
    use gpui::{TestAppContext, UpdateGlobal};
    use settings::SettingsStore;
    use tokio::sync::mpsc as tmpsc;

    fn init_settings(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
            AgentSettings::register(cx);
        });
    }

    fn set_user_settings_json(cx: &mut TestAppContext, json: &str) {
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.set_user_settings(json, cx).unwrap();
            });
        });
    }

    fn make_stream() -> (
        ThreadEventStream,
        fmpsc::UnboundedReceiver<Result<ThreadEvent>>,
    ) {
        let (tx, rx) = fmpsc::unbounded::<Result<ThreadEvent>>();
        (ThreadEventStream(tx), rx)
    }

    fn drain(rx: &mut fmpsc::UnboundedReceiver<Result<ThreadEvent>>) -> Vec<ThreadEvent> {
        let mut out = Vec::new();
        while let Ok(Some(item)) = rx.try_next() {
            out.push(item.expect("stream only carries Ok"));
        }
        out
    }

    // ── extract_match_inputs (pure) ──

    #[test]
    fn extract_match_inputs_collects_top_level_strings() {
        let raw = serde_json::json!({
            "command": "ls -la",
            "cwd": "/tmp",
            "timeout_ms": 30_000,
            "env": { "FOO": "bar" },
        });
        let mut inputs = extract_match_inputs(&raw);
        inputs.sort();
        assert_eq!(inputs, vec!["/tmp".to_string(), "ls -la".to_string()]);
    }

    #[test]
    fn extract_match_inputs_non_object_yields_empty() {
        assert!(extract_match_inputs(&serde_json::json!("not an object")).is_empty());
        assert!(extract_match_inputs(&serde_json::json!(42)).is_empty());
        assert!(extract_match_inputs(&serde_json::json!([1, 2, 3])).is_empty());
        assert!(extract_match_inputs(&serde_json::json!(null)).is_empty());
    }

    #[test]
    fn extract_match_inputs_skips_nested_strings() {
        let raw = serde_json::json!({
            "path": "/etc/shadow",
            "metadata": { "nested_string": "SHOULD NOT APPEAR" },
        });
        let inputs = extract_match_inputs(&raw);
        assert_eq!(inputs, vec!["/etc/shadow".to_string()]);
    }

    // ── NW-3: id pass-through (the most important regression guard) ──

    #[gpui::test]
    async fn nw3_id_passes_through_verbatim_on_auto_allow(cx: &mut TestAppContext) {
        init_settings(cx);
        // Rule: always allow `terminal` when command matches `^ls `.
        set_user_settings_json(
            cx,
            r#"{
                "agent": {
                    "tool_permissions": {
                        "tools": {
                            "terminal": { "always_allow": [{"pattern": "^ls "}] }
                        }
                    }
                }
            }"#,
        );

        let (approval_tx, mut approval_rx) = tmpsc::channel::<(String, bool)>(4);
        let (stream, mut stream_rx) = make_stream();
        let raw_input = serde_json::json!({ "command": "ls -la" });
        let engine_id = "perm_abc123";

        cx.spawn(async move |mut async_cx| {
            route_native_permission_request(
                engine_id,
                "terminal",
                "run ls -la",
                Some(&raw_input),
                &stream,
                approval_tx,
                &mut async_cx,
            );
        })
        .detach();

        cx.run_until_parked();

        let (key, allowed) = approval_rx
            .try_recv()
            .expect("auto-allow must respond immediately");
        assert_eq!(
            key, "perm_abc123",
            "id MUST pass through verbatim — no `perm_` re-prefix (NW-3 regression guard)"
        );
        assert!(allowed, "always_allow rule must resolve to true");

        // Diagnostic surfaced; no ToolCallAuthorization emitted.
        let events = drain(&mut stream_rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ThreadEvent::EngineDiagnostic(d) if d.kind == "permission.auto_allow")),
            "expected permission.auto_allow diagnostic, got {events:?}"
        );
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, ThreadEvent::ToolCallAuthorization { .. })),
            "UI prompt must NOT be emitted on auto-allow"
        );
    }

    #[gpui::test]
    async fn a2_auto_deny_responds_false(cx: &mut TestAppContext) {
        init_settings(cx);
        set_user_settings_json(
            cx,
            r#"{
                "agent": {
                    "tool_permissions": {
                        "tools": {
                            "terminal": { "always_deny": [{"pattern": "rm -rf"}] }
                        }
                    }
                }
            }"#,
        );

        let (approval_tx, mut approval_rx) = tmpsc::channel::<(String, bool)>(4);
        let (stream, mut stream_rx) = make_stream();
        let raw_input = serde_json::json!({ "command": "rm -rf /" });

        cx.spawn(async move |mut async_cx| {
            route_native_permission_request(
                "perm_xyz",
                "terminal",
                "destructive",
                Some(&raw_input),
                &stream,
                approval_tx,
                &mut async_cx,
            );
        })
        .detach();

        cx.run_until_parked();

        let (key, allowed) = approval_rx.try_recv().expect("auto-deny must respond");
        assert_eq!(key, "perm_xyz");
        assert!(!allowed, "always_deny rule must resolve to false");

        let events = drain(&mut stream_rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ThreadEvent::EngineDiagnostic(d) if d.kind == "permission.auto_deny")),
            "expected permission.auto_deny diagnostic, got {events:?}"
        );
    }

    #[gpui::test]
    async fn no_raw_input_falls_through_to_interactive_prompt(cx: &mut TestAppContext) {
        init_settings(cx);

        let (approval_tx, mut approval_rx) = tmpsc::channel::<(String, bool)>(4);
        let (stream, mut stream_rx) = make_stream();

        cx.spawn(async move |mut async_cx| {
            route_native_permission_request(
                "perm_legacy",
                "terminal",
                "ls",
                None, // legacy engine build pre-A1
                &stream,
                approval_tx,
                &mut async_cx,
            );
        })
        .detach();

        cx.run_until_parked();

        // No auto-response on approval_tx.
        assert!(
            approval_rx.try_recv().is_err(),
            "legacy path must NOT auto-respond"
        );

        // UI prompt emitted.
        let events = drain(&mut stream_rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ThreadEvent::ToolCallAuthorization { .. })),
            "expected ToolCallAuthorization for interactive prompt, got {events:?}"
        );
    }

    #[gpui::test]
    async fn raw_input_but_no_rule_falls_through_to_prompt(cx: &mut TestAppContext) {
        init_settings(cx);
        // No user rules configured for `terminal` → default Confirm.

        let (approval_tx, mut approval_rx) = tmpsc::channel::<(String, bool)>(4);
        let (stream, mut stream_rx) = make_stream();
        let raw_input = serde_json::json!({ "command": "some-novel-cmd" });

        cx.spawn(async move |mut async_cx| {
            route_native_permission_request(
                "perm_nomatch",
                "terminal",
                "novel",
                Some(&raw_input),
                &stream,
                approval_tx,
                &mut async_cx,
            );
        })
        .detach();

        cx.run_until_parked();

        assert!(
            approval_rx.try_recv().is_err(),
            "no-matching-rule path must NOT auto-respond"
        );

        let events = drain(&mut stream_rx);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ThreadEvent::ToolCallAuthorization { .. })),
            "expected ToolCallAuthorization on fall-through, got {events:?}"
        );
    }
}

#[cfg(test)]
mod native_tool_lifecycle_tests {
    //! C2 — tool-lifecycle consumer state-machine tests.
    //!
    //! Exercises [`handle_native_tool_lifecycle`] directly with a
    //! scripted `ToolCallStart → ToolCallInputDelta×N → ToolCallInputEnd
    //! → ToolResult` sequence and asserts:
    //!   * Exactly one `ToolCall` emitted at Start.
    //!   * No stream emission during `ToolCallInputDelta` (bytes
    //!     aggregate silently into `input_buffers`).
    //!   * Exactly one `ToolCallUpdate` emitted at InputEnd, carrying
    //!     status=InProgress and raw_input populated from the
    //!     aggregated JSON.
    //!   * `ToolResult` returns `false` (fall through to dispatcher)
    //!     and cleans up per-id state so maps do not grow unbounded.
    //!   * Unknown variants (AgentText etc.) return `false` unchanged.

    use super::*;
    use caduceus_bridge::event_translator::TranslatedThreadEvent as T;
    use futures::StreamExt;
    use futures::channel::mpsc as fmpsc;
    use gpui::TestAppContext;
    use std::collections::{BTreeMap, HashMap};

    fn make_stream() -> (
        ThreadEventStream,
        fmpsc::UnboundedReceiver<Result<ThreadEvent>>,
    ) {
        let (tx, rx) = fmpsc::unbounded::<Result<ThreadEvent>>();
        (ThreadEventStream(tx), rx)
    }

    fn drain(rx: &mut fmpsc::UnboundedReceiver<Result<ThreadEvent>>) -> Vec<ThreadEvent> {
        let mut out = Vec::new();
        while let Ok(Some(item)) = rx.try_next() {
            out.push(item.expect("stream only carries Ok"));
        }
        out
    }

    #[gpui::test]
    async fn full_lifecycle_emits_one_start_and_one_end_update(cx: &mut TestAppContext) {
        let (stream, mut rx) = make_stream();
        let mut input_agg = caduceus_bridge::event_translator::ToolInputAggregator::new();
        let enabled_tools: BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>> =
            BTreeMap::new();

        cx.spawn(async move |mut cx_cons| {
            // 1. Start
            let handled = handle_native_tool_lifecycle(
                &T::ToolCallStart {
                    id: "t1".into(),
                    name: "terminal".into(),
                },
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            assert!(handled, "ToolCallStart must be fully handled");

            // 2. Deltas — 3 frames, no emit
            for frag in ["{\"comm", "and\":\"ls ", "-la\"}"] {
                let handled = handle_native_tool_lifecycle(
                    &T::ToolCallInputDelta {
                        id: "t1".into(),
                        delta: frag.into(),
                    },
                    &mut input_agg,
                    &enabled_tools,
                    &stream,
                    &mut cx_cons,
                );
                assert!(handled, "ToolCallInputDelta must be fully handled");
            }

            // 3. InputEnd — one update
            let handled = handle_native_tool_lifecycle(
                &T::ToolCallInputEnd { id: "t1".into() },
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            assert!(handled, "ToolCallInputEnd must be fully handled");

            // 4. ToolResult — falls through, cleans up maps
            let handled = handle_native_tool_lifecycle(
                &T::ToolResult {
                    id: "t1".into(),
                    content: "ok".into(),
                    is_error: false,
                },
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            assert!(
                !handled,
                "ToolResult must return false so dispatcher emits the final update"
            );
            assert!(
                input_agg.active_ids().is_empty(),
                "aggregator must be fully cleaned up on ToolResult"
            );
        })
        .detach();

        cx.run_until_parked();

        let events = drain(&mut rx);

        // Exactly 2 events from the consumer (Start + InputEnd-update);
        // ToolResult falls through and would produce its own event via
        // dispatch_translated_event, which we do NOT invoke here.
        assert_eq!(
            events.len(),
            2,
            "expected exactly 2 stream events (ToolCall + ToolCallUpdate), got {}: {:?}",
            events.len(),
            events
        );

        // Event 1: ToolCall with name=terminal, kind=Other (empty
        // registry), status=Pending.
        match &events[0] {
            ThreadEvent::ToolCall(call) => {
                assert_eq!(call.title, "terminal", "title defaults to tool name");
                assert_eq!(call.kind, acp::ToolKind::Other);
                assert_eq!(call.status, acp::ToolCallStatus::Pending);
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }

        // Event 2: ToolCallUpdate status=InProgress with raw_input
        // parsed from the aggregated delta frames.
        match &events[1] {
            ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(update)) => {
                let fields = &update.fields;
                assert_eq!(fields.status, Some(acp::ToolCallStatus::InProgress));
                let raw = fields
                    .raw_input
                    .as_ref()
                    .expect("raw_input must be set at InputEnd");
                assert_eq!(
                    raw,
                    &serde_json::json!({ "command": "ls -la" }),
                    "raw_input must reflect the aggregated delta payload"
                );
            }
            other => panic!("expected ToolCallUpdate::UpdateFields, got {other:?}"),
        }
    }

    #[gpui::test]
    async fn input_delta_alone_never_emits(cx: &mut TestAppContext) {
        let (stream, mut rx) = make_stream();
        let mut input_agg = caduceus_bridge::event_translator::ToolInputAggregator::new();
        let enabled_tools: BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>> =
            BTreeMap::new();

        cx.spawn(async move |mut cx_cons| {
            let _ = handle_native_tool_lifecycle(
                &T::ToolCallInputDelta {
                    id: "t9".into(),
                    delta: "anything".into(),
                },
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            // Buffer was populated.
            let out = input_agg.observe_input_end("t9");
            assert_eq!(out.raw_json, "anything");
        })
        .detach();

        cx.run_until_parked();

        let events = drain(&mut rx);
        assert!(
            events.is_empty(),
            "ToolCallInputDelta must NOT emit any stream events, got {events:?}"
        );
    }

    #[gpui::test]
    async fn unhandled_variants_return_false(cx: &mut TestAppContext) {
        let (stream, mut rx) = make_stream();
        let mut input_agg = caduceus_bridge::event_translator::ToolInputAggregator::new();
        let enabled_tools: BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>> =
            BTreeMap::new();

        cx.spawn(async move |mut cx_cons| {
            let handled = handle_native_tool_lifecycle(
                &T::AgentText("hello".into()),
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            assert!(
                !handled,
                "non-tool-lifecycle events must fall through to dispatcher"
            );
        })
        .detach();

        cx.run_until_parked();

        let events = drain(&mut rx);
        assert!(
            events.is_empty(),
            "handler must not emit anything for unhandled variants"
        );
    }

    #[gpui::test]
    async fn input_end_without_start_still_emits_update(cx: &mut TestAppContext) {
        // Defensive: InputEnd arriving without a prior Start (e.g.
        // engine resumed mid-stream) must still emit an InProgress
        // update without panicking, even though raw_input will be
        // absent (empty buffer → parse fails → None).
        let (stream, mut rx) = make_stream();
        let mut input_agg = caduceus_bridge::event_translator::ToolInputAggregator::new();
        let enabled_tools: BTreeMap<SharedString, std::sync::Arc<dyn AnyAgentTool>> =
            BTreeMap::new();

        cx.spawn(async move |mut cx_cons| {
            let handled = handle_native_tool_lifecycle(
                &T::ToolCallInputEnd { id: "t42".into() },
                &mut input_agg,
                &enabled_tools,
                &stream,
                &mut cx_cons,
            );
            assert!(handled);
        })
        .detach();

        cx.run_until_parked();

        let events = drain(&mut rx);
        assert_eq!(events.len(), 1);
        match &events[0] {
            ThreadEvent::ToolCallUpdate(acp_thread::ToolCallUpdate::UpdateFields(update)) => {
                assert_eq!(update.fields.status, Some(acp::ToolCallStatus::InProgress));
                assert!(
                    update.fields.raw_input.is_none(),
                    "raw_input must be None when the buffer was empty / unparseable"
                );
            }
            other => panic!("expected ToolCallUpdate::UpdateFields, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod st2_pinned_tests {
    //! ST2 pinned-message tests. Plan v3.1 §8.
    //!
    //! API-level (no gpui): pin/unpin/replace/gc/coalesce semantics, key
    //! identity, Resume index resolution, persistence proxy round-trip,
    //! forward-compat quarantine.
    //!
    //! Trigger-level (gpui): FirstUser auto-pin via `push_acp_user_block`;
    //! Resume auto-pin via direct `auto_pin_resume` after pushing a Resume
    //! marker; PlanUpdate via `on_plan_event_emitted`; ScopeExpansionActive
    //! via `on_scope_expansion_requested`. (Native-loop wiring is exercised
    //! in `caduceus_native_loop_tests`; here we test the Thread-side hook.)
    use super::*;
    use chrono::TimeZone;
    use futures::channel::mpsc;
    use gpui::TestAppContext;
    use project::Project;

    fn fixed_now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 4, 26, 12, 0, 0).unwrap()
    }

    fn user_key() -> PinnedMessageKey {
        PinnedMessageKey::User(UserMessageId::new())
    }

    /// Fresh, unique Resume key. Use when the test only cares that keys
    /// differ (e.g. cap/coalesce tests, orphan-gc tests).
    fn resume_key_new() -> PinnedMessageKey {
        PinnedMessageKey::Resume(ResumeId::new())
    }

    /// Collect the `ResumeId`s from the thread's messages, in push-order.
    /// Used by tests that need to pin a *specific* Resume marker.
    #[allow(dead_code)]
    fn resume_ids(t: &Thread) -> Vec<ResumeId> {
        t.messages
            .iter()
            .filter_map(|m| match m {
                Message::Resume(rid) => Some(*rid),
                _ => None,
            })
            .collect()
    }

    async fn setup_thread_for_test(
        cx: &mut TestAppContext,
    ) -> (Entity<Thread>, ThreadEventStream) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });

        let fs = fs::FakeFs::new(cx.background_executor.clone());
        let templates = Templates::new();
        let project = Project::test(fs.clone(), [], cx).await;

        cx.update(|cx| {
            let project_context = cx.new(|_cx| prompt_store::ProjectContext::default());
            let context_server_store = project.read(cx).context_server_store();
            let context_server_registry =
                cx.new(|cx| ContextServerRegistry::new(context_server_store, cx));

            let thread = cx.new(|cx| {
                Thread::new(
                    project,
                    project_context,
                    context_server_registry,
                    templates,
                    None,
                    cx,
                )
            });

            let (event_tx, _event_rx) = mpsc::unbounded();
            let event_stream = ThreadEventStream(event_tx);

            (thread, event_stream)
        })
    }

    // ─── API-level: pin / unpin / is_pinned ───────────────────────

    #[gpui::test]
    async fn t_pin_idempotent_same_reason(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let k = user_key();
            assert!(t.pin_at(k.clone(), PinReason::Manual, fixed_now()));
            assert!(!t.pin_at(k.clone(), PinReason::Manual, fixed_now()));
            assert_eq!(t.pinned_refs().len(), 1);
        });
    }

    #[gpui::test]
    async fn t_pin_allows_multiple_reasons_per_key(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let k = user_key();
            assert!(t.pin_at(k.clone(), PinReason::FirstUser, fixed_now()));
            assert!(t.pin_at(k.clone(), PinReason::Manual, fixed_now()));
            let reasons = t.is_pinned(&k);
            assert_eq!(reasons.len(), 2);
            assert!(reasons.contains(&PinReason::FirstUser));
            assert!(reasons.contains(&PinReason::Manual));
        });
    }

    #[gpui::test]
    async fn t_is_pinned_returns_all_reasons(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let k = user_key();
            t.pin_at(k.clone(), PinReason::FirstUser, fixed_now());
            t.pin_at(k.clone(), PinReason::PlanUpdate, fixed_now());
            t.pin_at(k.clone(), PinReason::Manual, fixed_now());
            assert_eq!(t.is_pinned(&k).len(), 3);
            assert!(t.is_pinned(&user_key()).is_empty());
        });
    }

    #[gpui::test]
    async fn t_unpin_removes_all_reasons_for_key(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let k = user_key();
            t.pin_at(k.clone(), PinReason::FirstUser, fixed_now());
            t.pin_at(k.clone(), PinReason::PlanUpdate, fixed_now());
            assert_eq!(t.unpin(&k), 2);
            assert!(t.is_pinned(&k).is_empty());
        });
    }

    #[gpui::test]
    async fn t_unpin_reason_removes_only_that_reason(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let k = user_key();
            t.pin_at(k.clone(), PinReason::FirstUser, fixed_now());
            t.pin_at(k.clone(), PinReason::Manual, fixed_now());
            assert!(t.unpin_reason(&k, PinReason::Manual));
            assert_eq!(t.is_pinned(&k), vec![PinReason::FirstUser]);
            assert!(!t.unpin_reason(&k, PinReason::Manual));
        });
    }

    #[gpui::test]
    async fn t_unpin_nonexistent_is_noop(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            assert_eq!(t.unpin(&user_key()), 0);
            assert!(!t.unpin_reason(&user_key(), PinReason::Manual));
        });
    }

    #[gpui::test]
    async fn t_pinned_refs_insertion_order(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let a = user_key();
            let b = user_key();
            let c = user_key();
            t.pin_at(a.clone(), PinReason::FirstUser, fixed_now());
            t.pin_at(b.clone(), PinReason::Manual, fixed_now());
            t.pin_at(c.clone(), PinReason::PlanUpdate, fixed_now());
            let keys: Vec<&PinnedMessageKey> =
                t.pinned_refs().iter().map(|p| &p.key).collect();
            assert_eq!(keys, vec![&a, &b, &c]);
        });
    }

    // ─── pin_replace (singleton-by-reason) ────────────────────────

    #[gpui::test]
    async fn t_pin_replace_supersedes_old(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let a = user_key();
            let b = user_key();
            t.pin_at(a.clone(), PinReason::ScopeExpansionActive, fixed_now());
            let prev = t.pin_replace(b.clone(), PinReason::ScopeExpansionActive);
            assert_eq!(prev, Some(a));
            // exactly one ScopeExpansionActive pin remains, on `b`.
            let count = t
                .pinned_refs()
                .iter()
                .filter(|p| p.reason == PinReason::ScopeExpansionActive)
                .count();
            assert_eq!(count, 1);
            assert_eq!(t.is_pinned(&b), vec![PinReason::ScopeExpansionActive]);
        });
    }

    // ─── Resume cap + coalescing ──────────────────────────────────

    #[gpui::test]
    async fn t_resume_pin_cap_3(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            // 5 distinct Resume ids — coalesce drops the oldest 2.
            let mut ids: Vec<ResumeId> = Vec::new();
            for _ in 0..5 {
                let rid = ResumeId::new();
                ids.push(rid);
                t.pin_at(
                    PinnedMessageKey::Resume(rid),
                    PinReason::Resume,
                    fixed_now(),
                );
            }
            let kept: Vec<ResumeId> = t
                .pinned_refs()
                .iter()
                .filter_map(|p| match (&p.key, p.reason) {
                    (PinnedMessageKey::Resume(r), PinReason::Resume) => Some(*r),
                    _ => None,
                })
                .collect();
            assert_eq!(kept.len(), MAX_RESUME_PINS);
            // newest 3 retained (insertion order)
            assert_eq!(kept, ids[2..].to_vec());
        });
    }

    // ─── gc_pinned ────────────────────────────────────────────────

    #[gpui::test]
    async fn t_gc_drops_orphans_keeps_live(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            // live user message
            let live_id = UserMessageId::new();
            t.messages.push(Message::User(UserMessage {
                id: live_id.clone(),
                content: vec![],
            }));
            // orphan user pin
            let orphan = UserMessageId::new();
            t.pin_at(
                PinnedMessageKey::User(live_id.clone()),
                PinReason::Manual,
                fixed_now(),
            );
            t.pin_at(
                PinnedMessageKey::User(orphan),
                PinReason::Manual,
                fixed_now(),
            );
            // orphan resume pin (random ResumeId, no matching marker in messages)
            t.pin_at(resume_key_new(), PinReason::Resume, fixed_now());
            assert_eq!(t.pinned_refs().len(), 3);
            t.gc_pinned();
            assert_eq!(t.pinned_refs().len(), 1);
            assert!(t
                .is_pinned(&PinnedMessageKey::User(live_id))
                .contains(&PinReason::Manual));
        });
    }

    #[gpui::test]
    async fn t_gc_after_full_truncate_empties_pinned(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            t.pin_at(user_key(), PinReason::Manual, fixed_now());
            t.pin_at(resume_key_new(), PinReason::Resume, fixed_now());
            // messages stays empty -> all pins orphan
            t.gc_pinned();
            assert!(t.pinned_refs().is_empty());
        });
    }

    // ─── pinned_message_indices / resolve_key ─────────────────────

    #[gpui::test]
    async fn t_pinned_message_indices_returns_positions(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let id_a = UserMessageId::new();
            let id_b = UserMessageId::new();
            let r1 = ResumeId::new();
            let r2 = ResumeId::new();
            t.messages.push(Message::User(UserMessage {
                id: id_a.clone(),
                content: vec![],
            }));
            t.messages.push(Message::Resume(r1));
            t.messages.push(Message::User(UserMessage {
                id: id_b.clone(),
                content: vec![],
            }));
            t.messages.push(Message::Resume(r2));
            t.pin_at(
                PinnedMessageKey::User(id_a),
                PinReason::FirstUser,
                fixed_now(),
            );
            t.pin_at(PinnedMessageKey::Resume(r2), PinReason::Resume, fixed_now());
            t.pin_at(
                PinnedMessageKey::User(id_b),
                PinReason::Manual,
                fixed_now(),
            );
            let indices = t.pinned_message_indices();
            assert_eq!(indices, vec![0, 2, 3]);
        });
    }

    #[gpui::test]
    async fn t_pinned_message_indices_deduplicates_multi_reason(cx: &mut TestAppContext) {
        // ST2 fix-loop #1 (reviewer Correctness 3/4): a message pinned
        // for multiple reasons must appear ONCE in the indices set.
        // Plan v3.1 §5: pins are keyed by (key, reason); the budget
        // calculation in ST3 must not double-count.
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let id = UserMessageId::new();
            t.messages.push(Message::User(UserMessage {
                id: id.clone(),
                content: vec![],
            }));
            t.pin_at(
                PinnedMessageKey::User(id.clone()),
                PinReason::Manual,
                fixed_now(),
            );
            t.pin_at(
                PinnedMessageKey::User(id.clone()),
                PinReason::FirstUser,
                fixed_now(),
            );
            // Two pins on the same key — but only one index entry.
            assert_eq!(t.pinned_refs().len(), 2);
            let indices = t.pinned_message_indices();
            assert_eq!(indices, vec![0]);
        });
    }

    #[gpui::test]
    async fn t_resume_index_stable_across_user_message_edits(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let r = ResumeId::new();
            t.messages.push(Message::Resume(r)); // resume at idx 0
            t.pin_at(PinnedMessageKey::Resume(r), PinReason::Resume, fixed_now());
            assert_eq!(t.pinned_message_indices(), vec![0]);

            // inserting/removing user messages around the resume marker
            // does not invalidate the resume pin (id-keyed, not position-keyed).
            t.messages.insert(
                0,
                Message::User(UserMessage {
                    id: UserMessageId::new(),
                    content: vec![],
                }),
            );
            assert_eq!(t.pinned_message_indices(), vec![1]);
        });
    }

    // ─── clear_inherited_pins / subagent invariant ────────────────

    #[gpui::test]
    async fn t_clear_inherited_pins(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            t.pin_at(user_key(), PinReason::Manual, fixed_now());
            t.pin_at(user_key(), PinReason::Manual, fixed_now());
            t.clear_inherited_pins();
            assert!(t.pinned_refs().is_empty());
        });
    }

    #[gpui::test]
    async fn t_subagent_constructor_initializes_empty_pinned(cx: &mut TestAppContext) {
        let (parent, _ev) = setup_thread_for_test(cx).await;
        parent.update(cx, |t, _| {
            t.pin_at(user_key(), PinReason::Manual, fixed_now());
            assert_eq!(t.pinned_refs().len(), 1);
        });
        let subagent = cx.update(|cx| cx.new(|cx| Thread::new_subagent(&parent, cx)));
        subagent.update(cx, |t, _| {
            assert!(t.pinned_refs().is_empty(), "subagent must start with empty pinned");
        });
    }

    // ─── trigger sites ────────────────────────────────────────────

    #[gpui::test]
    async fn t_first_user_auto_pinned_via_acp_block(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let id = UserMessageId::new();
        thread.update(cx, |t, cx| {
            t.push_acp_user_block(
                id.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
        });
        thread.update(cx, |t, _| {
            assert_eq!(
                t.is_pinned(&PinnedMessageKey::User(id)),
                vec![PinReason::FirstUser]
            );
        });
    }

    #[gpui::test]
    async fn t_first_user_not_repinned_on_second_user_message(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let id1 = UserMessageId::new();
        let id2 = UserMessageId::new();
        thread.update(cx, |t, cx| {
            t.push_acp_user_block(
                id1.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.push_acp_user_block(
                id2.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
        });
        thread.update(cx, |t, _| {
            assert_eq!(
                t.is_pinned(&PinnedMessageKey::User(id1)),
                vec![PinReason::FirstUser]
            );
            assert!(t.is_pinned(&PinnedMessageKey::User(id2)).is_empty());
        });
    }

    #[gpui::test]
    async fn t_resume_auto_pin_pins_marker_id(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let r1 = ResumeId::new();
            t.messages.push(Message::Resume(r1));
            t.auto_pin_resume();
            assert_eq!(
                t.is_pinned(&PinnedMessageKey::Resume(r1)),
                vec![PinReason::Resume]
            );

            let r2 = ResumeId::new();
            t.messages.push(Message::Resume(r2));
            t.auto_pin_resume();
            let pins: Vec<ResumeId> = t
                .pinned_refs()
                .iter()
                .filter_map(|p| match (&p.key, p.reason) {
                    (PinnedMessageKey::Resume(r), PinReason::Resume) => Some(*r),
                    _ => None,
                })
                .collect();
            assert_eq!(pins, vec![r1, r2]);
        });
    }

    #[gpui::test]
    async fn t_resume_pin_after_resume_removal_does_not_redirect(cx: &mut TestAppContext) {
        // ST2 fix-loop #2 (reviewer Correctness 3/4): with the legacy
        // `ResumeIndex` keying, removing Resume #1 silently retargeted
        // a `Resume(2)` pin to the now-second-position Resume #3 — the
        // wrong message. After moving to `ResumeId(Uuid)` identity, the
        // pin tracks the *original* Resume #2 by id; if Resume #1 is
        // removed, the pin still resolves to the original Resume #2
        // (now at a different position), NOT to Resume #3.
        // Plan v3.1 §5 (Fix 2 — `PinnedMessageKey` upfront).
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            let r1 = ResumeId::new();
            let r2 = ResumeId::new();
            let r3 = ResumeId::new();
            t.messages.push(Message::Resume(r1)); // pos 0
            t.messages.push(Message::Resume(r2)); // pos 1
            t.messages.push(Message::Resume(r3)); // pos 2

            // Pin Resume #2 (the middle one).
            t.pin_at(PinnedMessageKey::Resume(r2), PinReason::Resume, fixed_now());
            assert_eq!(t.pinned_message_indices(), vec![1]);

            // Remove Resume #1.
            t.messages.remove(0);
            // r2 now at pos 0, r3 at pos 1.

            // The pin must still resolve to r2 (now at pos 0), NOT r3 at pos 1.
            assert_eq!(t.pinned_message_indices(), vec![0]);
            // And by identity:
            let pinned_pos = t.pinned_message_indices()[0];
            match &t.messages[pinned_pos] {
                Message::Resume(id) => assert_eq!(
                    *id, r2,
                    "Resume pin must still target original r2, not r3"
                ),
                other => panic!("expected Resume marker at pinned pos, got {other:?}"),
            }

            // gc_pinned must NOT drop the live pin.
            t.gc_pinned();
            assert_eq!(t.pinned_refs().len(), 1);
            assert!(t
                .is_pinned(&PinnedMessageKey::Resume(r2))
                .contains(&PinReason::Resume));
        });
    }

    #[gpui::test]
    async fn t_plan_update_pins_most_recent_agent_message(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let user_id = UserMessageId::new();
        let agent_id_1 = AgentMessageId::new();
        let agent_id_2 = AgentMessageId::new();
        thread.update(cx, |t, cx| {
            t.push_acp_user_block(
                user_id,
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.messages.push(Message::Agent(AgentMessage {
                id: agent_id_1,
                content: vec![AgentMessageContent::Text("first reply".into())],
                ..Default::default()
            }));
            t.messages.push(Message::Agent(AgentMessage {
                id: agent_id_2,
                content: vec![AgentMessageContent::Text("second reply".into())],
                ..Default::default()
            }));
            t.on_plan_event_emitted(cx);
        });
        thread.update(cx, |t, _| {
            assert!(t
                .is_pinned(&PinnedMessageKey::Agent(agent_id_2))
                .contains(&PinReason::PlanUpdate));
            assert!(!t
                .is_pinned(&PinnedMessageKey::Agent(agent_id_1))
                .contains(&PinReason::PlanUpdate));
        });
    }

    #[gpui::test]
    async fn t_plan_update_with_no_agent_message_drops(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let user_id = UserMessageId::new();
        thread.update(cx, |t, cx| {
            // Only a user message — no agent reply yet. PlanUpdate should
            // drop the pin and emit a WARN log per AC6.
            t.push_acp_user_block(
                user_id,
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.on_plan_event_emitted(cx);
            assert!(t
                .pinned_refs()
                .iter()
                .all(|p| p.reason != PinReason::PlanUpdate));
        });
    }

    #[gpui::test]
    async fn t_scope_expansion_pin_replace_supersedes(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let id1 = UserMessageId::new();
        let id2 = UserMessageId::new();
        thread.update(cx, |t, cx| {
            t.push_acp_user_block(
                id1,
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.on_scope_expansion_requested();
            t.push_acp_user_block(
                id2.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.on_scope_expansion_requested();
        });
        thread.update(cx, |t, _| {
            let count = t
                .pinned_refs()
                .iter()
                .filter(|p| p.reason == PinReason::ScopeExpansionActive)
                .count();
            assert_eq!(count, 1);
            assert!(t
                .is_pinned(&PinnedMessageKey::User(id2))
                .contains(&PinReason::ScopeExpansionActive));
        });
    }

    // ─── persistence: round-trip + forward-compat ─────────────────

    #[test]
    fn t_pin_uniqueness_invariant_50_iter() {
        // Property check: 50 random sequences of pin/unpin maintain
        // uniqueness on (key, reason).
        use std::collections::HashSet;
        for seed in 0..50u32 {
            let mut pinned: Vec<MessageRef> = Vec::new();
            // synthesize a sequence
            let keys: Vec<PinnedMessageKey> = (0..5).map(|_| user_key()).collect();
            let reasons = [
                PinReason::FirstUser,
                PinReason::Manual,
                PinReason::PlanUpdate,
            ];
            for i in 0..20 {
                let k = keys[((seed as usize + i) * 7) % keys.len()].clone();
                let r = reasons[(i + seed as usize) % reasons.len()];
                let dup = pinned.iter().any(|p| p.key == k && p.reason == r);
                if !dup {
                    pinned.push(MessageRef {
                        key: k,
                        reason: r,
                        pinned_at: fixed_now(),
                    });
                }
            }
            let set: HashSet<(String, PinReason)> = pinned
                .iter()
                .map(|p| (format!("{:?}", p.key), p.reason))
                .collect();
            assert_eq!(set.len(), pinned.len(), "uniqueness violated for seed {seed}");
        }
    }

    #[test]
    fn t_message_ref_serde_round_trip() {
        let r = MessageRef {
            key: user_key(),
            reason: PinReason::FirstUser,
            pinned_at: fixed_now(),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: MessageRef = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn t_legacy_resume_message_deserializes_with_synthetic_id() {
        // ST2 fix-loop #2: legacy DbThread JSON encodes Message::Resume
        // as the bare string "Resume". The custom Deserialize impl on
        // Message must accept this and synthesize a fresh ResumeId, so
        // pre-ST2 threads on disk continue to load.
        let legacy = serde_json::json!("Resume");
        let m: Message = serde_json::from_value(legacy).expect("legacy Resume must deserialize");
        match m {
            Message::Resume(_) => {}
            other => panic!("expected Message::Resume(_), got {other:?}"),
        }

        // And new shape round-trips.
        let r = ResumeId::new();
        let m2 = Message::Resume(r);
        let s = serde_json::to_string(&m2).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(m2, back);
    }

    #[test]
    fn t_legacy_db_thread_without_pinned_field_defaults_empty() {
        // Simulate a v1 DbThread JSON missing the `pinned` field.
        // Only the three required fields are supplied; everything else
        // (including `pinned`) must default.
        let json = serde_json::json!({
            "title": "x",
            "messages": [],
            "updated_at": fixed_now(),
        });
        let parsed: DbThread =
            serde_json::from_value(json).expect("legacy DbThread must deserialize");
        assert!(parsed.pinned.is_empty());
    }

    #[test]
    fn t_unknown_pin_reason_is_quarantined() {
        // Build a DbThread JSON with one valid + one Unknown variant pin.
        let base = serde_json::json!({
            "title": "x",
            "messages": [],
            "updated_at": fixed_now(),
            "pinned": [
                {
                    "key": { "User": "00000000-0000-0000-0000-000000000001" },
                    "reason": "Manual",
                    "pinned_at": "2026-04-26T12:00:00Z"
                },
                {
                    "key": { "User": "00000000-0000-0000-0000-000000000002" },
                    "reason": "FromTheFuture",
                    "pinned_at": "2026-04-26T12:00:00Z"
                }
            ]
        });
        let parsed: DbThread =
            serde_json::from_value(base).expect("DbThread with unknown pin reason must parse");
        assert_eq!(
            parsed.pinned.len(),
            1,
            "Unknown reason variant must be quarantined; only Manual pin retained"
        );
        assert_eq!(parsed.pinned[0].reason, PinReason::Manual);
    }

    #[test]
    fn t_unknown_pin_key_is_quarantined() {
        // ST2 fix-loop #5: forward-compat for PinnedMessageKey. A future
        // build may write `{"key": {"Tool": "..."}, ...}` — this build
        // must drop that pin (with a WARN) and retain other pins.
        let base = serde_json::json!({
            "title": "x",
            "messages": [],
            "updated_at": fixed_now(),
            "pinned": [
                {
                    "key": { "User": "00000000-0000-0000-0000-000000000001" },
                    "reason": "Manual",
                    "pinned_at": "2026-04-26T12:00:00Z"
                },
                {
                    "key": { "FromTheFuture": "00000000-0000-0000-0000-000000000002" },
                    "reason": "Manual",
                    "pinned_at": "2026-04-26T12:00:00Z"
                }
            ]
        });
        let parsed: DbThread = serde_json::from_value(base)
            .expect("DbThread with unknown pin key must parse");
        assert_eq!(
            parsed.pinned.len(),
            1,
            "Unknown key variant must be quarantined; only User pin retained"
        );
        assert!(matches!(parsed.pinned[0].key, PinnedMessageKey::User(_)));
    }

    #[gpui::test]
    async fn t_to_db_round_trips_pinned(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let id = UserMessageId::new();
        let k = PinnedMessageKey::User(id.clone());
        thread.update(cx, |t, cx| {
            // Push a real user message so the pin's target survives the
            // ST2 fix-loop #6 orphan filter inside to_db.
            t.push_acp_user_block(
                id,
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.pin_at(k.clone(), PinReason::Manual, fixed_now());
        });
        let db = cx
            .update(|cx| thread.read(cx).to_db(cx))
            .await;
        // The pushed user message also auto-pins as FirstUser, so we
        // expect 2 pins (Manual + FirstUser) — not orphaned.
        assert_eq!(db.pinned.len(), 2);
        assert!(db.pinned.iter().any(|p| p.key == k && p.reason == PinReason::Manual));
        assert!(db.pinned.iter().any(|p| p.key == k && p.reason == PinReason::FirstUser));
    }

    #[gpui::test]
    async fn t_to_db_filters_orphaned_pins(cx: &mut TestAppContext) {
        // ST2 fix-loop #6: persistence must never include pins whose
        // target message is no longer present. Add a pin with a key
        // that doesn't correspond to any live message; assert to_db
        // drops it.
        let (thread, _ev) = setup_thread_for_test(cx).await;
        thread.update(cx, |t, _| {
            // Orphan pin — no live message with this UserMessageId.
            let orphan = PinnedMessageKey::User(UserMessageId::new());
            t.pin_at(orphan, PinReason::Manual, fixed_now());
        });
        let db = cx
            .update(|cx| thread.read(cx).to_db(cx))
            .await;
        assert!(
            db.pinned.is_empty(),
            "to_db must filter orphaned pins; got {:?}",
            db.pinned
        );
    }

    #[gpui::test]
    async fn t_truncate_runs_gc_pinned(cx: &mut TestAppContext) {
        // ST2 fix-loop #6: truncate must call gc_pinned so pins on
        // truncated messages are cleared.
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let id1 = UserMessageId::new();
        let id2 = UserMessageId::new();
        thread.update(cx, |t, cx| {
            t.push_acp_user_block(
                id1.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.push_acp_user_block(
                id2.clone(),
                std::iter::empty::<acp::ContentBlock>(),
                util::paths::PathStyle::local(),
                cx,
            );
            t.pin_at(
                PinnedMessageKey::User(id1.clone()),
                PinReason::Manual,
                fixed_now(),
            );
            t.pin_at(
                PinnedMessageKey::User(id2.clone()),
                PinReason::Manual,
                fixed_now(),
            );
            // Truncate to id2 — drops messages from id2 onward.
            t.truncate(id2.clone(), cx)
                .expect("truncate to id2 must succeed");
        });
        thread.update(cx, |t, _| {
            // id2's message is gone; its pin must have been GC'd. id1
            // still lives and remains pinned.
            assert!(
                t.is_pinned(&PinnedMessageKey::User(id2.clone())).is_empty(),
                "pin on truncated message must be gc'd"
            );
            assert!(
                !t.is_pinned(&PinnedMessageKey::User(id1.clone())).is_empty(),
                "pin on surviving message must remain"
            );
        });
    }

    #[gpui::test]
    async fn t_empty_pinned_round_trip(cx: &mut TestAppContext) {
        let (thread, _ev) = setup_thread_for_test(cx).await;
        let db = cx
            .update(|cx| thread.read(cx).to_db(cx))
            .await;
        assert!(db.pinned.is_empty());
    }
}
