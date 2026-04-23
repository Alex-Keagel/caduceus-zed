//! G0 — `AgentEvent` → `TranslatedThreadEvent` translator.
//!
//! Pure function mapping engine events to UI events. Lives in the bridge
//! (depends only on `caduceus-core`) so it stays off the Zed dependency
//! graph and can be unit-tested without GPUI.
//!
//! The translator is deliberately lossy: engine events that only matter
//! to the `SessionStateReducer` (introspection, step boundaries,
//! execution-tree pings, telemetry-only events) translate to
//! [`TranslatedThreadEvent::Swallow`] with a reason. Thread-side code
//! matches on the output variants and emits the corresponding concrete
//! `ThreadEvent` / `acp::*` types.
//!
//! Translation is 1-to-N: a single `AgentEvent` may expand into multiple
//! UI events (e.g. `ToolResultEnd` → `ToolUpdate{status:Completed}` +
//! `ToolUpdate{content:…}`). The output ordering matters and is stable
//! across calls.

use caduceus_core::AgentEvent;
use smallvec::{SmallVec, smallvec};

/// Return type for [`translate`]. Most engine events map to exactly one
/// translated event; sizing the inline buffer to 1 avoids a heap
/// allocation on the fast path (ST-C1 — audit C9 hot-path allocations).
pub type TranslatedEvents = SmallVec<[TranslatedThreadEvent; 1]>;

/// Bridge-owned mirror of the ThreadEvent shapes the translator can
/// produce. Thread-side code maps each variant to its concrete
/// `ThreadEvent` / `acp::*` payload; keeping the enum here avoids
/// leaking Zed types (`acp`, `acp_thread`) into the bridge crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranslatedThreadEvent {
    /// Append assistant text (streaming delta or full chunk).
    AgentText(String),
    /// Reasoning / thinking delta.
    AgentThinking(String),
    /// Terminal reasoning chunk with duration telemetry (H2 — previously
    /// `ReasoningComplete.duration_ms` was dropped). UI may render a
    /// "thought for Xms" collapsed marker.
    AgentThinkingComplete { content: String, duration_ms: u64 },
    /// New tool call starting. `id` is the engine-assigned tool_use_id.
    ToolCallStart { id: String, name: String },
    /// Streaming tool-input delta (for UIs that render arg streaming).
    ToolCallInputDelta { id: String, delta: String },
    /// Tool call finished accumulating its input arguments. UI may now
    /// render the full args if it was buffering.
    ToolCallInputEnd { id: String },
    /// Tool result arrived. `is_error=true` signals a failed execution
    /// (UI surfaces red status).
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
    /// Permission approval needed. `id` matches the engine's request
    /// id; thread.rs routes the user's decision back via `approval_tx`.
    PermissionRequest {
        id: String,
        tool: String,
        description: String,
    },
    /// Context-usage warning from the engine (~70% / ~85% / ~95%).
    /// Level is the raw string (`"warning_70"` etc.) — UI may map to
    /// an icon / color.
    ContextWarning { level: String, used: u32, max: u32 },
    /// Context auto-compaction happened. Delta is `before - after` tokens.
    ContextCompacted { freed: u32, before: u32, after: u32 },
    /// Message groups evicted from active window. `total_tokens` is a
    /// pre-summed convenience; UI may render per-group detail from the
    /// engine event via a separate subscription if needed.
    ContextEvicted {
        strategy: String,
        groups: u32,
        total_tokens: u32,
    },
    /// Retry was triggered — loop detector, circuit breaker, or
    /// tool-timeout. UI renders this as a transient retry banner; the
    /// engine keeps running.
    Retry { kind: RetryKind, message: String },
    /// A new plan step was recorded in Plan mode, awaiting execution.
    /// H2 — `step_id`, `depends_on`, `parent_step_id`, `plan_revision`
    /// preserved so the UI can render the P13 Features DAG (edges
    /// require stable ids; the positional `step` number shifts on
    /// amendment and MUST NOT be used for cross-step references).
    PlanStep {
        step: usize,
        step_id: u64,
        plan_revision: u64,
        tool: String,
        description: String,
        depends_on: Vec<u64>,
        parent_step_id: Option<u64>,
    },
    /// An existing plan was amended. H2 — `kind`/`step`/`ok`/`reason`
    /// preserved so the UI can render "plan changed because <reason>"
    /// and distinguish `ok=false` (rejected) from applied amendments
    /// without re-querying the plan.
    PlanAmended {
        plan_revision: u64,
        kind: String,
        step: usize,
        ok: bool,
        reason: String,
    },
    /// Scope expansion (envelope trip) — always re-prompts the user
    /// regardless of approval cadence. UI should prominently surface.
    ScopeExpansion {
        capability: String,
        resource: String,
        tool: String,
        reason: String,
    },
    /// Mode or lens changed mid-session. UI updates mode badge.
    /// H2 — `from_*` preserved so the UI can render transition
    /// animations / diff badges.
    ModeChanged {
        from_mode: String,
        to_mode: String,
        from_lens: Option<String>,
        to_lens: Option<String>,
    },
    /// Turn finished. UI emits its `Stop` event with the mapped reason.
    /// H2 — token usage preserved so the UI can update the per-turn
    /// token meter without round-tripping to the reducer.
    TurnComplete {
        stop: StopReasonKind,
        usage: TokenUsageMirror,
    },
    /// Engine-level error during the turn. Thread.rs emits `Stop(Error)`
    /// after flushing pending messages.
    TurnError { message: String },
    /// Not surfaced to UI. `reason` documents why (tests + debug logs).
    Swallow { reason: &'static str },
}

/// Retry categories the UI distinguishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryKind {
    /// Loop detector fired — same tool+args repeated.
    Loop,
    /// Circuit breaker tripped — consecutive failures.
    CircuitBreaker,
    /// A single tool exceeded its wall-clock budget.
    ToolTimeout,
    /// A tool was aborted because run-level cancellation fired.
    ToolCancelled,
}

/// Coarse stop reason the thread-side maps to `acp::StopReason`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReasonKind {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// H2 — dedicated variant (was previously aliased to `EndTurn`,
    /// losing the provider signal that a custom stop sequence matched).
    StopSequence,
    Error,
    BudgetExceeded,
    /// Fallback for forward-compat variants.
    Other(String),
}

/// Bridge-local mirror of [`caduceus_core::TokenUsage`] — kept here so
/// the translator output enum has no dependency on engine-only types
/// beyond `AgentEvent` (which is already the translator's input).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsageMirror {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_tokens: u32,
    pub cache_write_tokens: u32,
}

impl From<&caduceus_core::TokenUsage> for TokenUsageMirror {
    fn from(u: &caduceus_core::TokenUsage) -> Self {
        Self {
            input_tokens: u.input_tokens,
            output_tokens: u.output_tokens,
            cache_read_tokens: u.cache_read_tokens,
            cache_write_tokens: u.cache_write_tokens,
        }
    }
}

/// Translate a single engine event into zero-or-more UI events.
///
/// Returning a `Vec` (rather than an `Option`) because several
/// engine events need to fan out into multiple UI events — notably
/// `ContextGroupsEvicted` which may carry enough data to render
/// both a toast AND a status-bar change.
pub fn translate(ev: &AgentEvent) -> TranslatedEvents {
    use TranslatedThreadEvent as T;
    match ev {
        // ── Streaming text ─────────────────────────────────────────────
        AgentEvent::TextDelta { text } => smallvec![T::AgentText(text.clone())],
        AgentEvent::ReasoningDelta { content } => smallvec![T::AgentThinking(content.clone())],
        AgentEvent::ReasoningComplete {
            content,
            duration_ms,
        } => smallvec![T::AgentThinkingComplete {
            content: content.clone(),
            duration_ms: *duration_ms,
        }],
        AgentEvent::ThinkingStarted { .. } => smallvec![T::Swallow {
            reason: "thinking-started is a reducer signal; UI renders on first delta",
        }],

        // ── Tool lifecycle ────────────────────────────────────────────
        AgentEvent::ToolCallStart { id, name } => smallvec![T::ToolCallStart {
            id: id.0.clone(),
            name: name.clone(),
        }],
        AgentEvent::ToolCallInput { id, delta } => smallvec![T::ToolCallInputDelta {
            id: id.0.clone(),
            delta: delta.clone(),
        }],
        AgentEvent::ToolCallEnd { id } => smallvec![T::ToolCallInputEnd { id: id.0.clone() }],
        AgentEvent::ToolResultStart { .. } => smallvec![T::Swallow {
            reason: "tool-result-start — wait for ToolResultEnd to emit a UI update",
        }],
        AgentEvent::ToolResultEnd {
            id,
            content,
            is_error,
        } => smallvec![T::ToolResult {
            id: id.0.clone(),
            content: content.clone(),
            is_error: *is_error,
        }],

        // ── Permission ────────────────────────────────────────────────
        AgentEvent::PermissionRequest {
            id,
            capability,
            description,
        } => smallvec![T::PermissionRequest {
            id: id.clone(),
            tool: capability.clone(),
            description: description.clone(),
        }],
        AgentEvent::PermissionDecision { .. } | AgentEvent::ApprovalDecided { .. } => {
            smallvec![T::Swallow {
                reason: "permission resolution — UI already updated on user action",
            }]
        }
        AgentEvent::ScopeExpansionRequested {
            capability,
            resource,
            reason,
            tool,
        } => smallvec![T::ScopeExpansion {
            capability: capability.clone(),
            resource: resource.clone(),
            tool: tool.clone(),
            reason: reason.clone(),
        }],

        // ── Retry signals ─────────────────────────────────────────────
        AgentEvent::LoopDetected {
            tool_name,
            consecutive_count,
        } => smallvec![T::Retry {
            kind: RetryKind::Loop,
            message: format!("Loop detected on '{tool_name}' ({consecutive_count} consecutive)"),
        }],
        AgentEvent::CircuitBreakerTriggered {
            consecutive_failures,
            last_tools,
        } => smallvec![T::Retry {
            kind: RetryKind::CircuitBreaker,
            message: format!(
                "Circuit breaker: {consecutive_failures} failures; last tools {last_tools:?}"
            ),
        }],
        AgentEvent::ToolTimedOut {
            tool,
            timeout_secs,
            elapsed_ms,
        } => smallvec![T::Retry {
            kind: RetryKind::ToolTimeout,
            message: format!(
                "Tool '{tool}' timed out after {elapsed_ms}ms (budget {timeout_secs}s)"
            ),
        }],
        AgentEvent::ToolCancelled { tool, elapsed_ms } => smallvec![T::Retry {
            kind: RetryKind::ToolCancelled,
            message: format!("Tool '{tool}' cancelled after {elapsed_ms}ms"),
        }],

        // ── Context management ────────────────────────────────────────
        AgentEvent::ContextWarning {
            level,
            used_tokens,
            max_tokens,
        } => smallvec![T::ContextWarning {
            level: level.clone(),
            used: *used_tokens,
            max: *max_tokens,
        }],
        AgentEvent::ContextCompacted {
            freed_tokens,
            before,
            after,
        } => smallvec![T::ContextCompacted {
            freed: *freed_tokens,
            before: *before,
            after: *after,
        }],
        AgentEvent::ContextGroupsEvicted {
            strategy,
            groups,
            total_tokens,
        } => smallvec![T::ContextEvicted {
            strategy: strategy.clone(),
            groups: groups.len() as u32,
            total_tokens: *total_tokens,
        }],

        // ── Plan mode ─────────────────────────────────────────────────
        AgentEvent::PlanStepPending {
            step,
            step_id,
            plan_revision,
            tool_name,
            description,
            depends_on,
            parent_step_id,
            ..
        } => smallvec![T::PlanStep {
            step: *step,
            step_id: step_id.0,
            plan_revision: *plan_revision,
            tool: tool_name.clone(),
            description: description.clone(),
            depends_on: depends_on.iter().map(|s| s.0).collect(),
            parent_step_id: parent_step_id.as_ref().map(|s| s.0),
        }],
        AgentEvent::PlanAmended {
            kind,
            step,
            ok,
            reason,
            plan_revision,
        } => smallvec![T::PlanAmended {
            plan_revision: *plan_revision,
            kind: kind.clone(),
            step: *step,
            ok: *ok,
            reason: reason.clone(),
        }],
        AgentEvent::AwaitingApproval { .. } => smallvec![T::Swallow {
            reason: "awaiting-approval — reducer owns the plan-panel render path",
        }],

        // ── Mode / lens ───────────────────────────────────────────────
        AgentEvent::ModeChanged {
            from_mode,
            to_mode,
            from_lens,
            to_lens,
        } => smallvec![T::ModeChanged {
            from_mode: from_mode.clone(),
            to_mode: to_mode.clone(),
            from_lens: from_lens.clone(),
            to_lens: to_lens.clone(),
        }],

        // ── Turn lifecycle ────────────────────────────────────────────
        AgentEvent::TurnComplete { stop_reason, usage } => smallvec![T::TurnComplete {
            stop: map_stop_reason(stop_reason),
            usage: usage.into(),
        }],
        AgentEvent::Error { message } => smallvec![T::TurnError {
            message: message.clone(),
        }],

        // ── Reducer-only events (no UI surface today) ─────────────────
        AgentEvent::Introspection(_) => smallvec![T::Swallow {
            reason: "introspection — consumed by dag_state reducer",
        }],
        AgentEvent::StepStarted { .. } | AgentEvent::StepCompleted { .. } => smallvec![T::Swallow {
            reason: "step boundary — reducer/telemetry only",
        }],
        AgentEvent::ExecutionTreeNode { .. } | AgentEvent::ExecutionTreeUpdate { .. } => {
            smallvec![T::Swallow {
                reason: "execution-tree — rendered by a dedicated panel via reducer",
            }]
        }
        AgentEvent::MessagePart { .. } => smallvec![T::Swallow {
            reason: "structured message part — AI-elements rendering path, not ThreadEvent",
        }],
        AgentEvent::SessionPhaseChanged { .. } => smallvec![T::Swallow {
            reason: "session-phase — reducer/status-bar only",
        }],
        AgentEvent::RoutingDecision { .. } => smallvec![T::Swallow {
            reason: "routing decision — reducer/introspection panel only",
        }],
        AgentEvent::EventBufferOverflow { .. } => smallvec![T::Swallow {
            reason: "buffer overflow — replay via emitter, not a UI event",
        }],
        AgentEvent::DrainedStaleApproval { .. } => smallvec![T::Swallow {
            reason: "stale approval drain — telemetry only",
        }],
        AgentEvent::TokenLogprobSummary { .. } => smallvec![T::Swallow {
            reason: "logprob summary — reducer renders confidence dot",
        }],
        AgentEvent::BudgetUpdated { .. } => smallvec![T::Swallow {
            reason: "budget updated — reducer/status-bar",
        }],
        AgentEvent::CheckpointCreated { .. } | AgentEvent::CheckpointReverted { .. } => {
            smallvec![T::Swallow {
                reason: "checkpoint — timeline panel via reducer",
            }]
        }
        AgentEvent::BackgroundAgentDone { .. } => smallvec![T::Swallow {
            reason: "background agent — notification channel, not main thread",
        }],
        AgentEvent::CritiqueCall { .. } => smallvec![T::Swallow {
            reason: "critique call — reducer",
        }],
        AgentEvent::VerificationStarted { .. } | AgentEvent::VerificationCompleted { .. } => {
            smallvec![T::Swallow {
                reason: "verification phase — reducer",
            }]
        }
        AgentEvent::TestGateStarted { .. } | AgentEvent::TestGateCompleted { .. } => {
            smallvec![T::Swallow {
                reason: "test gate — reducer",
            }]
        }
        AgentEvent::ParallelToolBatchStarted { .. }
        | AgentEvent::ParallelToolBatchCompleted { .. } => smallvec![T::Swallow {
            reason: "parallel batch diagnostics — reducer",
        }],
        AgentEvent::ReflexionRecorded { .. } => smallvec![T::Swallow {
            reason: "reflexion — rendered inline on the failed tool_result, not a separate event",
        }],
        AgentEvent::CriticVerdict { .. } => smallvec![T::Swallow {
            reason: "critic verdict — reducer",
        }],

        // Forward-compat catch-all.
        AgentEvent::Unknown => smallvec![T::Swallow {
            reason: "unknown event — client may be older than engine",
        }],

        // #[non_exhaustive] safety net: future engine additions fall through here
        // until the translator is extended to handle them explicitly.
        _ => smallvec![T::Swallow {
            reason: "unrecognized AgentEvent variant — bridge translator outdated",
        }],
    }
}

fn map_stop_reason(sr: &caduceus_core::StopReason) -> StopReasonKind {
    use caduceus_core::StopReason as S;
    match sr {
        S::EndTurn => StopReasonKind::EndTurn,
        S::ToolUse => StopReasonKind::ToolUse,
        S::MaxTokens => StopReasonKind::MaxTokens,
        S::StopSequence => StopReasonKind::StopSequence,
        S::Error => StopReasonKind::Error,
        S::BudgetExceeded => StopReasonKind::BudgetExceeded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_core::{AgentEvent, StopReason, TokenUsage};

    fn one(ev: &AgentEvent) -> TranslatedThreadEvent {
        let out = translate(ev);
        assert_eq!(out.len(), 1, "expected single event, got {out:?}");
        out.into_iter().next().unwrap()
    }

    #[test]
    fn text_delta_translates() {
        let got = one(&AgentEvent::TextDelta {
            text: "hello".into(),
        });
        assert_eq!(got, TranslatedThreadEvent::AgentText("hello".into()));
    }

    #[test]
    fn reasoning_delta_and_complete_both_emit_thinking() {
        let d = one(&AgentEvent::ReasoningDelta {
            content: "a".into(),
        });
        assert_eq!(d, TranslatedThreadEvent::AgentThinking("a".into()));
        let c = one(&AgentEvent::ReasoningComplete {
            content: "done".into(),
            duration_ms: 1234,
        });
        assert_eq!(
            c,
            TranslatedThreadEvent::AgentThinkingComplete {
                content: "done".into(),
                duration_ms: 1234,
            }
        );
    }

    #[test]
    fn tool_lifecycle_full_sequence() {
        let id = caduceus_core::ToolCallId("t1".to_string());
        let start = one(&AgentEvent::ToolCallStart {
            id: id.clone(),
            name: "bash".into(),
        });
        assert!(
            matches!(start, TranslatedThreadEvent::ToolCallStart { ref name, .. } if name == "bash")
        );

        let delta = one(&AgentEvent::ToolCallInput {
            id: id.clone(),
            delta: "{\"cmd\":".into(),
        });
        assert!(matches!(
            delta,
            TranslatedThreadEvent::ToolCallInputDelta { .. }
        ));

        let end = one(&AgentEvent::ToolCallEnd { id: id.clone() });
        assert!(matches!(
            end,
            TranslatedThreadEvent::ToolCallInputEnd { .. }
        ));

        let result = one(&AgentEvent::ToolResultEnd {
            id,
            content: "ok".into(),
            is_error: false,
        });
        assert_eq!(
            result,
            TranslatedThreadEvent::ToolResult {
                id: "t1".into(),
                content: "ok".into(),
                is_error: false
            }
        );
    }

    #[test]
    fn permission_request_translates() {
        let got = one(&AgentEvent::PermissionRequest {
            id: "p1".into(),
            capability: "bash".into(),
            description: "run ls".into(),
        });
        match got {
            TranslatedThreadEvent::PermissionRequest { id, tool, .. } => {
                assert_eq!(id, "p1");
                assert_eq!(tool, "bash");
            }
            other => panic!("expected PermissionRequest, got {other:?}"),
        }
    }

    #[test]
    fn loop_detected_becomes_retry() {
        let got = one(&AgentEvent::LoopDetected {
            tool_name: "grep".into(),
            consecutive_count: 4,
        });
        assert!(matches!(
            got,
            TranslatedThreadEvent::Retry {
                kind: RetryKind::Loop,
                ..
            }
        ));
    }

    #[test]
    fn circuit_breaker_becomes_retry() {
        let got = one(&AgentEvent::CircuitBreakerTriggered {
            consecutive_failures: 5,
            last_tools: vec!["bash".into()],
        });
        assert!(matches!(
            got,
            TranslatedThreadEvent::Retry {
                kind: RetryKind::CircuitBreaker,
                ..
            }
        ));
    }

    #[test]
    fn tool_timeout_and_cancel_become_retry() {
        let t = one(&AgentEvent::ToolTimedOut {
            tool: "bash".into(),
            timeout_secs: 30,
            elapsed_ms: 30_500,
        });
        assert!(matches!(
            t,
            TranslatedThreadEvent::Retry {
                kind: RetryKind::ToolTimeout,
                ..
            }
        ));
        let c = one(&AgentEvent::ToolCancelled {
            tool: "bash".into(),
            elapsed_ms: 200,
        });
        assert!(matches!(
            c,
            TranslatedThreadEvent::Retry {
                kind: RetryKind::ToolCancelled,
                ..
            }
        ));
    }

    #[test]
    fn context_warning_and_compaction_and_eviction() {
        let w = one(&AgentEvent::ContextWarning {
            level: "warning_70".into(),
            used_tokens: 70_000,
            max_tokens: 100_000,
        });
        assert_eq!(
            w,
            TranslatedThreadEvent::ContextWarning {
                level: "warning_70".into(),
                used: 70_000,
                max: 100_000
            }
        );
        let c = one(&AgentEvent::ContextCompacted {
            freed_tokens: 1_000,
            before: 70_000,
            after: 69_000,
        });
        assert_eq!(
            c,
            TranslatedThreadEvent::ContextCompacted {
                freed: 1_000,
                before: 70_000,
                after: 69_000
            }
        );
        let e = one(&AgentEvent::ContextGroupsEvicted {
            strategy: "truncate-oldest".into(),
            groups: vec![],
            total_tokens: 500,
        });
        assert_eq!(
            e,
            TranslatedThreadEvent::ContextEvicted {
                strategy: "truncate-oldest".into(),
                groups: 0,
                total_tokens: 500
            }
        );
    }

    #[test]
    fn plan_step_and_amend() {
        let s = one(&AgentEvent::PlanStepPending {
            step: 1,
            step_id: caduceus_core::StepId(42),
            revision: 0,
            plan_revision: 1,
            tool_name: "bash".into(),
            description: "run tests".into(),
            depends_on: vec![caduceus_core::StepId(41)],
            parent_step_id: Some(caduceus_core::StepId(40)),
        });
        assert_eq!(
            s,
            TranslatedThreadEvent::PlanStep {
                step: 1,
                step_id: 42,
                plan_revision: 1,
                tool: "bash".into(),
                description: "run tests".into(),
                depends_on: vec![41],
                parent_step_id: Some(40),
            }
        );

        let a = one(&AgentEvent::PlanAmended {
            kind: "replace".into(),
            step: 2,
            ok: false,
            reason: "invalid step ref".into(),
            plan_revision: 7,
        });
        assert_eq!(
            a,
            TranslatedThreadEvent::PlanAmended {
                plan_revision: 7,
                kind: "replace".into(),
                step: 2,
                ok: false,
                reason: "invalid step ref".into(),
            }
        );
    }

    #[test]
    fn scope_expansion_surfaces() {
        let got = one(&AgentEvent::ScopeExpansionRequested {
            capability: "write".into(),
            resource: "/etc/passwd".into(),
            reason: "NotInAllowList".into(),
            tool: "edit_file".into(),
        });
        match got {
            TranslatedThreadEvent::ScopeExpansion {
                capability, tool, ..
            } => {
                assert_eq!(capability, "write");
                assert_eq!(tool, "edit_file");
            }
            other => panic!("expected ScopeExpansion, got {other:?}"),
        }
    }

    #[test]
    fn mode_changed_preserves_from_and_to() {
        let got = one(&AgentEvent::ModeChanged {
            from_mode: "plan".into(),
            to_mode: "act".into(),
            from_lens: None,
            to_lens: Some("fast".into()),
        });
        assert_eq!(
            got,
            TranslatedThreadEvent::ModeChanged {
                from_mode: "plan".into(),
                to_mode: "act".into(),
                from_lens: None,
                to_lens: Some("fast".into()),
            }
        );
    }

    #[test]
    fn turn_complete_maps_stop_reasons_and_preserves_usage() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 30,
            cache_write_tokens: 5,
        };
        let cases = [
            (StopReason::EndTurn, StopReasonKind::EndTurn),
            (StopReason::ToolUse, StopReasonKind::ToolUse),
            (StopReason::MaxTokens, StopReasonKind::MaxTokens),
            // H2: StopSequence now has its own variant (was previously
            // aliased to EndTurn, losing the provider signal).
            (StopReason::StopSequence, StopReasonKind::StopSequence),
            (StopReason::Error, StopReasonKind::Error),
            (StopReason::BudgetExceeded, StopReasonKind::BudgetExceeded),
        ];
        for (sr, expected) in cases {
            let got = one(&AgentEvent::TurnComplete {
                stop_reason: sr,
                usage: usage.clone(),
            });
            assert_eq!(
                got,
                TranslatedThreadEvent::TurnComplete {
                    stop: expected,
                    usage: TokenUsageMirror {
                        input_tokens: 100,
                        output_tokens: 50,
                        cache_read_tokens: 30,
                        cache_write_tokens: 5,
                    }
                }
            );
        }
    }

    #[test]
    fn error_event_surfaces() {
        let got = one(&AgentEvent::Error {
            message: "provider down".into(),
        });
        assert_eq!(
            got,
            TranslatedThreadEvent::TurnError {
                message: "provider down".into()
            }
        );
    }

    #[test]
    fn reducer_only_events_swallow() {
        for ev in [
            AgentEvent::StepStarted { step_id: 1 },
            AgentEvent::StepCompleted {
                step_id: 1,
                ok: true,
            },
            AgentEvent::SessionPhaseChanged {
                phase: caduceus_core::SessionPhase::Running,
            },
            AgentEvent::EventBufferOverflow {
                dropped_since_last: 3,
            },
            AgentEvent::Unknown,
        ] {
            let got = one(&ev);
            assert!(
                matches!(got, TranslatedThreadEvent::Swallow { .. }),
                "{ev:?} should swallow, got {got:?}"
            );
        }
    }

    #[test]
    fn thinking_started_swallowed() {
        let got = one(&AgentEvent::ThinkingStarted { iteration: 0 });
        assert!(matches!(got, TranslatedThreadEvent::Swallow { .. }));
    }

    #[test]
    fn tool_result_start_swallowed() {
        let got = one(&AgentEvent::ToolResultStart {
            id: caduceus_core::ToolCallId("t1".to_string()),
            name: "bash".into(),
        });
        assert!(matches!(got, TranslatedThreadEvent::Swallow { .. }));
    }

    #[test]
    fn permission_resolution_swallowed() {
        let d = one(&AgentEvent::PermissionDecision {
            id: "p".into(),
            capability: "bash".into(),
            outcome: caduceus_core::PermissionOutcome::Approved,
        });
        assert!(matches!(d, TranslatedThreadEvent::Swallow { .. }));

        let a = one(&AgentEvent::ApprovalDecided {
            tool: "bash".into(),
            decision: caduceus_core::ApprovalDecision::Approved,
            latency_ms: 100,
        });
        assert!(matches!(a, TranslatedThreadEvent::Swallow { .. }));
    }

    #[test]
    fn returns_vec_always_nonempty() {
        // Regression guard: translator must always return at least one
        // output (even if it's a Swallow). Callers rely on this to
        // avoid silent drops without a logged reason.
        for ev in [
            AgentEvent::TextDelta { text: "x".into() },
            AgentEvent::StepStarted { step_id: 1 },
            AgentEvent::Unknown,
        ] {
            assert!(!translate(&ev).is_empty(), "empty translation of {ev:?}");
        }
    }
}
