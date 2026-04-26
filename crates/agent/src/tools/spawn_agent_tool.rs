use acp_thread::{SUBAGENT_SESSION_INFO_META_KEY, SubagentSessionInfo};
use agent_client_protocol as acp;
use anyhow::Result;
use caduceus_core::{SubAgentFailure, SubAgentPhase, TimeoutFailure};
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use crate::{AgentTool, ThreadEnvironment, ToolCallEventStream, ToolInput};

/// ST7: per-spawn timeout bounds for `spawn_agent` (input override).
/// Defaults to 15 minutes when `timeout_secs` is `None`.
pub const DEFAULT_SPAWN_TIMEOUT_SECS: u64 = 900;
pub const MIN_SPAWN_TIMEOUT_SECS: u64 = 1;
pub const MAX_SPAWN_TIMEOUT_SECS: u64 = 3600;

/// Validate a per-spawn timeout. `None` → 15min default. Out-of-range values
/// produce `SubAgentFailure::InternalError { kind: "invalid_timeout" }` per
/// plan v3.1 T-VAL1 (NOT a `ProviderError` — domain mismatch fix).
pub fn validate_timeout(t: Option<u64>) -> std::result::Result<Duration, SubAgentFailure> {
    match t {
        None => Ok(Duration::from_secs(DEFAULT_SPAWN_TIMEOUT_SECS)),
        Some(v) if (MIN_SPAWN_TIMEOUT_SECS..=MAX_SPAWN_TIMEOUT_SECS).contains(&v) => {
            Ok(Duration::from_secs(v))
        }
        Some(v) => Err(SubAgentFailure::InternalError {
            kind: "invalid_timeout".into(),
            message: format!(
                "timeout_secs={v} out of range [{MIN_SPAWN_TIMEOUT_SECS}..={MAX_SPAWN_TIMEOUT_SECS}]"
            ),
        }),
    }
}

/// ST7: classify an `anyhow::Error` returned by `subagent.send()` into a
/// structured [`SubAgentFailure`]. Order:
///
/// 1. Downcast to a typed `SubAgentFailure` (already classified upstream
///    — e.g. `RecursionLimitExceeded` at `agent.rs:2832`,
///    `ModelRefusal` at `agent.rs:3073`, or any failure wrapped by the
///    new `Err` arm at `agent.rs:3078`).
/// 2. Downcast to a typed [`caduceus_core::CaduceusError`] (live-path fix
///    — must-fix #1, plan v3 §A: 429 / 503 / timeout / cancel from the
///    provider stack reach this site as `anyhow!(CaduceusError::*)`; the
///    classifier maps them to `RetryClass` correctly).
/// 3. String-pattern fallback for the small set of canned `anyhow!(...)`
///    messages emitted directly by `agent.rs` (`MaxTokens` /
///    `MaxTurnRequests` / `User canceled` / refusal text /
///    no-response).
/// 4. Otherwise → `InternalError { kind: "subagent_send_failed" }`.
pub fn classify_subagent_error(
    err: &anyhow::Error,
    last_phase: SubAgentPhase,
    tools_started: bool,
    elapsed_secs: u64,
    timeout_secs: u64,
) -> SubAgentFailure {
    if let Some(typed) = err.downcast_ref::<SubAgentFailure>() {
        return typed.clone();
    }
    if let Some(cd_err) = err.downcast_ref::<caduceus_core::CaduceusError>() {
        // Empty context at this boundary: zed's `spawn_agent_tool` does
        // not yet thread provider/model into the classify call site
        // (filed as ST7-followup-B; provider/model populate from the
        // dispatcher boundary in caduceus-orchestrator). The ClassifyContext
        // typing makes the upgrade non-breaking.
        return caduceus_core::classify_caduceus_error(
            cd_err,
            &caduceus_core::ClassifyContext::empty(),
            last_phase,
            tools_started,
            elapsed_secs,
            timeout_secs,
        );
    }
    let msg = err.to_string();
    if msg == "User canceled" {
        return SubAgentFailure::UserCancel;
    }
    if msg.contains("Maximum subagent depth") {
        return SubAgentFailure::InternalError {
            kind: "recursion_limit".into(),
            message: msg,
        };
    }
    if msg.contains("refused to process") {
        return SubAgentFailure::ModelRefusal { refusal_text: msg };
    }
    if msg == "No response from the agent. You can try messaging again." {
        return SubAgentFailure::InternalError {
            kind: "no_response".into(),
            message: msg,
        };
    }
    SubAgentFailure::InternalError {
        kind: "subagent_send_failed".into(),
        message: msg,
    }
}

/// Spawn a sub-agent for a well-scoped task.
///
/// ### Designing delegated subtasks
/// - An agent does not see your conversation history. Include all relevant context (file paths, requirements, constraints) in the message.
/// - Subtasks must be concrete, well-defined, and self-contained.
/// - Delegated subtasks must materially advance the main task.
/// - Do not duplicate work between your work and delegated subtasks.
/// - Do not use this tool for tasks you could accomplish directly with one or two tool calls.
/// - When you delegate work, focus on coordinating and synthesizing results instead of duplicating the same work yourself.
/// - Avoid issuing multiple delegate calls for the same unresolved subproblem unless the new delegated task is genuinely different and necessary.
/// - Narrow the delegated ask to the concrete output you need next.
/// - For code-edit subtasks, decompose work so each delegated task has a disjoint write set.
/// - When sending a follow-up using an existing agent session_id, the agent already has the context from the previous turn. Send only a short, direct message. Do NOT repeat the original task or context.
///
/// ### Parallel delegation patterns
/// - Run multiple independent information-seeking subtasks in parallel when you have distinct questions that can be answered independently.
/// - Split implementation into disjoint codebase slices and spawn multiple agents for them in parallel when the write scopes do not overlap.
/// - When a plan has multiple independent steps, prefer delegating those steps in parallel rather than serializing them unnecessarily.
/// - Reuse the returned session_id when you want to follow up on the same delegated subproblem instead of creating a duplicate session.
///
/// ### Vendor-diverse DAG fan-out (RECOMMENDED for non-trivial tasks)
/// - When delegating multiple independent subtasks, **diversify across model
///   vendors** by setting `model` and `profile` per spawn. The runtime
///   defaults are inheritance-from-parent, but a master orchestrator should
///   route each subtask to the best-fit (vendor, model, profile) tuple.
/// - Example pattern for a 3-way research / scan / test fan-out:
///     spawn_agent({ label, message, profile: "plan",
///                   model: "gpt-5.4",          mode: "plan" })
///     spawn_agent({ label, message, profile: "act",
///                   model: "claude-opus-4.7",  mode: "act"  })
///     spawn_agent({ label, message, profile: "act",
///                   model: "claude-sonnet-4.6",mode: "act"  })
/// - Validation is strict: an unknown profile / model / mode fails the spawn
///   with the available choices listed in the error.
///
/// ### Output
/// - You will receive only the agent's final message as output.
/// - Successful calls return a session_id that you can use for follow-up messages.
/// - Error results may also include a session_id if a session was already created.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SpawnAgentToolInput {
    /// Short label displayed in the UI while the agent runs (e.g., "Researching alternatives")
    pub label: String,
    /// The prompt for the agent. For new sessions, include full context needed for the task. For follow-ups (with session_id), you can rely on the agent already having the previous message.
    pub message: String,
    /// Session ID of an existing agent session to continue instead of creating a new one.
    #[serde(default)]
    pub session_id: Option<acp::SessionId>,
    /// Optional override: profile id (e.g., "plan", "act", "autopilot", or a
    /// custom profile from settings). Omit to inherit from the parent agent.
    #[serde(default)]
    pub profile: Option<String>,
    /// Optional override: language model id (e.g., "claude-opus-4.7",
    /// "gpt-5.4", "claude-sonnet-4.6"). Omit to inherit from the parent.
    /// Use this to fan out across vendor-diverse models in a DAG.
    #[serde(default)]
    pub model: Option<String>,
    /// Optional override: caduceus mode ("plan", "act", "autopilot"). Omit
    /// to inherit from the parent agent.
    #[serde(default)]
    pub mode: Option<String>,
    /// ST7: per-spawn wall-clock timeout in seconds. `None` → 15-minute
    /// default. Clamped to `[1, 3600]`; out-of-range values fail the spawn
    /// with `failure_type = "InternalError"` (`kind = "invalid_timeout"`).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
pub enum SpawnAgentToolOutput {
    Success {
        session_id: acp::SessionId,
        output: String,
        session_info: SubagentSessionInfo,
    },
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        session_id: Option<acp::SessionId>,
        error: String,
        /// ST7: stable discriminant string mirroring
        /// [`SubAgentFailure::kind_str`]. `None` for legacy / unclassified
        /// errors so existing consumers keep round-tripping.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        failure_type: Option<String>,
        /// ST7: the full `SubAgentFailure` payload (the `details` field of
        /// the tagged enum, opaque to non-LLM consumers). `None` when the
        /// failure could not be classified.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        failure_details: Option<serde_json::Value>,
        session_info: Option<SubagentSessionInfo>,
    },
}

impl SpawnAgentToolOutput {
    /// Build an `Error` variant from a structured [`SubAgentFailure`],
    /// populating `failure_type` and `failure_details` from the typed
    /// payload while preserving `error` as a human-readable string for
    /// backward compat.
    pub fn from_failure(
        session_id: Option<acp::SessionId>,
        failure: &SubAgentFailure,
        session_info: Option<SubagentSessionInfo>,
    ) -> Self {
        let failure_type = Some(failure.kind_str().to_string());
        let failure_details = serde_json::to_value(failure)
            .ok()
            .and_then(|v| v.get("details").cloned());
        SpawnAgentToolOutput::Error {
            session_id,
            error: failure.to_string(),
            failure_type,
            failure_details,
            session_info,
        }
    }
}

impl From<SpawnAgentToolOutput> for LanguageModelToolResultContent {
    fn from(output: SpawnAgentToolOutput) -> Self {
        match output {
            SpawnAgentToolOutput::Success {
                session_id,
                output,
                session_info: _, // Don't show this to the model
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "output": output }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Error {
                session_id,
                error,
                failure_type,
                failure_details,
                session_info: _, // Don't show this to the model
            } => {
                // ST7 / v3.1 Fix 3: ALWAYS emit `session_id` (even
                // `null`) for byte-equivalent backward compat with the
                // pre-ST7 `json!({ "session_id": ..., "error": ... })`
                // shape. New keys (`failure_type`, `failure_details`)
                // are gated on `Option` so the no-classification path
                // is byte-equivalent too.
                let mut o = serde_json::Map::new();
                o.insert(
                    "session_id".into(),
                    match session_id {
                        Some(sid) => serde_json::to_value(sid)
                            .unwrap_or(serde_json::Value::Null),
                        None => serde_json::Value::Null,
                    },
                );
                o.insert("error".into(), serde_json::Value::String(error));
                if let Some(ft) = failure_type {
                    o.insert("failure_type".into(), serde_json::Value::String(ft));
                }
                if let Some(fd) = failure_details {
                    o.insert("failure_details".into(), fd);
                }
                serde_json::to_string(&serde_json::Value::Object(o))
                    .unwrap_or_else(|e| {
                        format!("Failed to serialize spawn_agent output: {e}")
                    })
                    .into()
            }
        }
    }
}

/// Tool that spawns an agent thread to work on a task.
pub struct SpawnAgentTool {
    environment: Rc<dyn ThreadEnvironment>,
}

impl SpawnAgentTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self { environment }
    }
}

impl AgentTool for SpawnAgentTool {
    type Input = SpawnAgentToolInput;
    type Output = SpawnAgentToolOutput;

    const NAME: &'static str = "spawn_agent";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => i.label.into(),
            Err(value) => value
                .get("label")
                .and_then(|v| v.as_str())
                .map(|s| SharedString::from(s.to_owned()))
                .unwrap_or_else(|| "Spawning agent".into()),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: format!("Failed to receive tool input: {e}"),
                    failure_type: None,
                    failure_details: None,
                    session_info: None,
                })?;

            // ST7: validate timeout before any side-effects (subagent
            // creation, telemetry). Out-of-range → InternalError, not
            // ProviderError (T-VAL1, plan v3.1 critic B9 fix).
            let spawn_timeout = match validate_timeout(input.timeout_secs) {
                Ok(d) => d,
                Err(failure) => {
                    return Err(SpawnAgentToolOutput::from_failure(
                        None, &failure, None,
                    ));
                }
            };

            let (subagent, mut session_info) = cx.update(|cx| {
                let subagent = if let Some(session_id) = input.session_id {
                    self.environment.resume_subagent(session_id, cx)
                } else {
                    let opts = crate::SubagentSpawnOptions {
                        label: input.label.clone(),
                        profile_override: input.profile.clone(),
                        model_override: input.model.clone(),
                        mode_override: input.mode.clone(),
                    };
                    self.environment.create_subagent(opts, cx)
                };
                let subagent = subagent.map_err(|err| {
                    // Best-effort classification at the create boundary.
                    let failure = classify_subagent_error(
                        &err,
                        SubAgentPhase::ModelSelection,
                        false,
                        0,
                        spawn_timeout.as_secs(),
                    );
                    SpawnAgentToolOutput::from_failure(None, &failure, None)
                })?;
                let session_info = SubagentSessionInfo {
                    session_id: subagent.id(),
                    message_start_index: subagent.num_entries(cx),
                    message_end_index: None,
                };

                event_stream.subagent_spawned(subagent.id());
                event_stream.update_fields_with_meta(
                    acp::ToolCallUpdateFields::new(),
                    Some(acp::Meta::from_iter([(
                        SUBAGENT_SESSION_INFO_META_KEY.into(),
                        serde_json::json!(&session_info),
                    )])),
                );

                Ok((subagent, session_info))
            })?;

            // ST7: per-spawn local closure state (NOT a shared map). Phase
            // tracking is a future hook — today we don't yet wire the
            // subagent's AgentEvent stream into this closure (that lands
            // with ST7-prereq), so phase stays at the `ModelSelection`
            // default and `tools_started` stays false unless a future
            // change updates them in-place.
            //
            // Uses gpui's `BackgroundExecutor::timer` (NOT `tokio::time`)
            // because the agent crate's tests run under gpui's
            // foreground/background executors, not a Tokio runtime.
            // `Instant::now()` is `std::time::Instant` for the same
            // reason — `tokio::time::Instant` panics off-runtime.
            let started_at = std::time::Instant::now();
            let last_phase = SubAgentPhase::ModelSelection;
            let tools_started = false;

            // ST7: race the `subagent.send()` future against the
            // configured spawn timeout via biased `select!` — timeout
            // arm wins ties (timer-vs-future poll order is
            // platform-stable). When an explicit cancel token reaches
            // this site (ST7-prereq), prepend it as the first arm so
            // cancel beats both timeout and completion.
            let timer = cx.background_executor().timer(spawn_timeout);
            let send_fut = subagent.send(input.message, cx);
            let send_result = {
                use futures::FutureExt;
                let mut timer = timer.fuse();
                let mut send_fut = send_fut.fuse();
                futures::select_biased! {
                    () = timer => {
                        let elapsed = started_at.elapsed().as_secs();
                        let timeout_failure = SubAgentFailure::Timeout(TimeoutFailure::new(
                            elapsed,
                            spawn_timeout.as_secs(),
                            last_phase,
                            tools_started,
                        ));
                        Err(anyhow::Error::new(timeout_failure))
                    }
                    r = send_fut => r,
                }
            };

            let status = if send_result.is_ok() {
                "completed"
            } else {
                "error"
            };
            telemetry::event!(
                "Subagent Completed",
                subagent_session = session_info.session_id.to_string(),
                status,
            );

            session_info.message_end_index =
                cx.update(|cx| Some(subagent.num_entries(cx).saturating_sub(1)));

            let meta = Some(acp::Meta::from_iter([(
                SUBAGENT_SESSION_INFO_META_KEY.into(),
                serde_json::json!(&session_info),
            )]));

            let (output, result) = match send_result {
                Ok(output) => (
                    output.clone(),
                    Ok(SpawnAgentToolOutput::Success {
                        session_id: session_info.session_id.clone(),
                        session_info,
                        output,
                    }),
                ),
                Err(e) => {
                    let elapsed = started_at.elapsed().as_secs();
                    let failure = classify_subagent_error(
                        &e,
                        last_phase,
                        tools_started,
                        elapsed,
                        spawn_timeout.as_secs(),
                    );
                    let error = failure.to_string();
                    (
                        error,
                        Err(SpawnAgentToolOutput::from_failure(
                            Some(session_info.session_id.clone()),
                            &failure,
                            Some(session_info),
                        )),
                    )
                }
            };
            event_stream.update_fields_with_meta(
                acp::ToolCallUpdateFields::new().content(vec![output.into()]),
                meta,
            );
            result
        })
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let (content, session_info) = match output {
            SpawnAgentToolOutput::Success {
                output,
                session_info,
                ..
            } => (output.into(), Some(session_info)),
            SpawnAgentToolOutput::Error {
                error,
                session_info,
                ..
            } => (error.into(), session_info),
        };

        let meta = session_info.map(|session_info| {
            acp::Meta::from_iter([(
                SUBAGENT_SESSION_INFO_META_KEY.into(),
                serde_json::json!(&session_info),
            )])
        });
        event_stream.update_fields_with_meta(
            acp::ToolCallUpdateFields::new().content(vec![content]),
            meta,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_core::{ProviderErrorFailure, RetryClass};

    // ---- T-VAL1: timeout_secs validation boundaries ----
    #[test]
    fn validate_timeout_default_is_900s() {
        let d = validate_timeout(None).expect("default must accept");
        assert_eq!(d, Duration::from_secs(DEFAULT_SPAWN_TIMEOUT_SECS));
        assert_eq!(d, Duration::from_secs(900));
    }

    #[test]
    fn validate_timeout_clamp_boundaries() {
        // Lower boundary
        assert!(validate_timeout(Some(0)).is_err(), "0 must reject");
        assert!(validate_timeout(Some(1)).is_ok(), "1 must accept");
        // Upper boundary
        assert!(validate_timeout(Some(3600)).is_ok(), "3600 must accept");
        assert!(validate_timeout(Some(3601)).is_err(), "3601 must reject");
        assert!(validate_timeout(Some(7200)).is_err(), "7200 must reject");
        // Mid-range
        assert!(validate_timeout(Some(600)).is_ok(), "600 must accept");
    }

    #[test]
    fn validate_timeout_out_of_range_is_internalerror_not_providererror() {
        // T-VAL1 / B9: domain mismatch fix — invalid_timeout is
        // InternalError, NOT ProviderError.
        let err = validate_timeout(Some(0)).unwrap_err();
        match err {
            SubAgentFailure::InternalError { kind, .. } => {
                assert_eq!(kind, "invalid_timeout");
            }
            other => panic!("expected InternalError, got {other:?}"),
        }
    }

    // ---- T9: Error JSON contains failure_type / failure_details / error / session_id ----
    #[test]
    fn failure_serializes_with_failure_type_field() {
        let failure = SubAgentFailure::Timeout(TimeoutFailure::new(
            901,
            900,
            SubAgentPhase::ToolExecution,
            true,
        ));
        let out = SpawnAgentToolOutput::from_failure(None, &failure, None);
        let content: LanguageModelToolResultContent = out.into();
        let s: String = match content {
            LanguageModelToolResultContent::Text(t) => t.to_string(),
            other => panic!("expected text, got {other:?}"),
        };
        let v: serde_json::Value = serde_json::from_str(&s).expect("must parse");
        assert!(v.get("session_id").is_some(), "session_id key must be present");
        assert!(v["session_id"].is_null(), "session_id must be null when None");
        assert!(v.get("error").is_some(), "error must be present");
        assert_eq!(v["failure_type"], "Timeout");
        assert_eq!(v["failure_details"]["elapsed_secs"], 901);
        assert_eq!(v["failure_details"]["timeout_secs"], 900);
        assert_eq!(v["failure_details"]["last_phase"], "ToolExecution");
        assert_eq!(v["failure_details"]["tools_started"], true);
    }

    // ---- T-COMPAT1 (v3.1 Fix 3): session_id: null preserved ----
    #[test]
    fn error_without_classification_preserves_session_id_null() {
        let out = SpawnAgentToolOutput::Error {
            session_id: None,
            error: "x".into(),
            failure_type: None,
            failure_details: None,
            session_info: None,
        };
        let content: LanguageModelToolResultContent = out.into();
        let s: String = match content {
            LanguageModelToolResultContent::Text(t) => t.to_string(),
            other => panic!("expected text, got {other:?}"),
        };
        // Byte-equivalence with pre-ST7 shape: must contain "session_id":null
        assert!(
            s.contains("\"session_id\":null"),
            "must emit session_id:null for backward compat, got: {s}"
        );
        // No new keys added when classification absent.
        assert!(!s.contains("failure_type"));
        assert!(!s.contains("failure_details"));
    }

    // ---- T11: old consumer JSON shape still parses ----
    #[test]
    fn old_consumer_unaffected_no_classification() {
        let out = SpawnAgentToolOutput::Error {
            session_id: None,
            error: "boom".into(),
            failure_type: None,
            failure_details: None,
            session_info: None,
        };
        let content: LanguageModelToolResultContent = out.into();
        let s: String = match content {
            LanguageModelToolResultContent::Text(t) => t.to_string(),
            _ => panic!(),
        };
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["error"], "boom");
        assert!(v["session_id"].is_null());
        assert!(v.get("failure_type").is_none());
        assert!(v.get("failure_details").is_none());
    }

    // ---- T17: InternalError is fallback only ----
    #[test]
    fn classify_subagent_error_known_strings_match_typed_variants() {
        use anyhow::anyhow;
        let err = anyhow!("User canceled");
        assert!(matches!(
            classify_subagent_error(&err, SubAgentPhase::ModelSelection, false, 0, 900),
            SubAgentFailure::UserCancel
        ));

        let err = anyhow!("The agent refused to process that prompt. Try again.");
        assert!(matches!(
            classify_subagent_error(&err, SubAgentPhase::ProviderCall, false, 0, 900),
            SubAgentFailure::ModelRefusal { .. }
        ));

        let err = anyhow!("synthetic random failure");
        let f = classify_subagent_error(&err, SubAgentPhase::Unknown, false, 0, 900);
        match f {
            SubAgentFailure::InternalError { kind, .. } => {
                assert_eq!(kind, "subagent_send_failed");
            }
            other => panic!("expected InternalError, got {other:?}"),
        }
    }

    // ---- T17b: typed failures pass through downcast ----
    #[test]
    fn classify_subagent_error_downcasts_typed_failure() {
        let mut p = ProviderErrorFailure::new("rate limited", RetryClass::Backoff);
        p.retry_after_secs = Some(30);
        p.http_status = Some(429);
        let typed = SubAgentFailure::ProviderError(p);
        let err = anyhow::Error::new(typed.clone());
        let out = classify_subagent_error(&err, SubAgentPhase::ProviderCall, false, 0, 900);
        match out {
            SubAgentFailure::ProviderError(p) => {
                assert_eq!(p.retry_after_secs, Some(30));
                assert_eq!(p.retry_class, RetryClass::Backoff);
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    // ---- Fix-loop #1: live path preserves CaduceusError -> typed ProviderError ----
    #[test]
    fn classify_subagent_error_downcasts_caduceus_error_ratelimit() {
        // Simulates the new agent.rs Err arm: provider returned a typed
        // CaduceusError::RateLimited which got wrapped in anyhow::Error at
        // task.fuse(). The classifier MUST run classify_caduceus_error and
        // surface a ProviderError with RetryClass::Backoff — not collapse
        // to InternalError("subagent_send_failed") via string fallback.
        use caduceus_core::{CaduceusError, RetryClass};
        let err = anyhow::Error::new(CaduceusError::RateLimited {
            retry_after_secs: 30,
        });
        let out =
            classify_subagent_error(&err, SubAgentPhase::ProviderCall, false, 0, 900);
        match out {
            SubAgentFailure::ProviderError(p) => {
                assert_eq!(p.retry_class, RetryClass::Backoff);
                assert_eq!(p.retry_after_secs, Some(30));
                assert_eq!(p.http_status, Some(429));
            }
            other => panic!("expected ProviderError(Backoff), got {other:?}"),
        }
    }

    #[test]
    fn classify_subagent_error_downcasts_caduceus_provider_timeout_to_backoff() {
        use caduceus_core::{CaduceusError, RetryClass};
        let err = anyhow::Error::new(CaduceusError::ProviderTimeout {
            elapsed_ms: 6000,
            limit_ms: 5000,
            context: "chat".into(),
        });
        let out =
            classify_subagent_error(&err, SubAgentPhase::ProviderCall, false, 0, 900);
        match out {
            SubAgentFailure::ProviderError(p) => {
                assert_eq!(p.retry_class, RetryClass::Backoff);
            }
            other => panic!("expected ProviderError, got {other:?}"),
        }
    }

    #[test]
    fn classify_subagent_error_downcasts_caduceus_cancelled_to_user_cancel() {
        use caduceus_core::CaduceusError;
        let err = anyhow::Error::new(CaduceusError::Cancelled);
        let out =
            classify_subagent_error(&err, SubAgentPhase::ProviderCall, false, 0, 900);
        assert!(matches!(out, SubAgentFailure::UserCancel));
    }

    // ---- T-PHASE3-revised: ToolResultEnd does NOT exit ToolExecution ----
    #[test]
    fn tool_result_end_does_not_exit_tool_execution() {
        use caduceus_core::{AgentEvent, ToolCallId};
        let phase = SubAgentPhase::ToolExecution;
        let next = phase.next_phase(&AgentEvent::ToolResultEnd {
            id: ToolCallId::new("x"),
            content: String::new(),
            is_error: false,
        });
        assert_eq!(
            next,
            SubAgentPhase::ToolExecution,
            "ToolResultEnd MUST NOT transition out of ToolExecution"
        );
        // Only the next provider-side event triggers the transition:
        let next = phase.next_phase(&AgentEvent::TextDelta { text: "x".into() });
        assert_eq!(next, SubAgentPhase::ProviderCall);
    }

    #[test]
    fn phase_transitions_from_real_agent_event_variants() {
        use caduceus_core::{AgentEvent, ToolCallId};
        // ModelSelection -> ProviderCall on first ThinkingStarted
        let p = SubAgentPhase::ModelSelection
            .next_phase(&AgentEvent::ThinkingStarted { iteration: 0 });
        assert_eq!(p, SubAgentPhase::ProviderCall);
        // ProviderCall -> ToolExecution on ToolCallStart
        let p = SubAgentPhase::ProviderCall.next_phase(&AgentEvent::ToolCallStart {
            id: ToolCallId::new("1"),
            name: "read_file".into(),
        });
        assert_eq!(p, SubAgentPhase::ToolExecution);
        // any -> ContextManagement on ContextWarning
        let p = SubAgentPhase::ProviderCall.next_phase(&AgentEvent::ContextWarning {
            level: "warning_85".into(),
            used_tokens: 85,
            max_tokens: 100,
        });
        assert_eq!(p, SubAgentPhase::ContextManagement);
    }
}
