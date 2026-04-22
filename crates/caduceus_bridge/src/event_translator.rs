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
    ContextWarning {
        level: String,
        used: u32,
        max: u32,
    },
    /// Context auto-compaction happened. Delta is `before - after` tokens.
    ContextCompacted {
        freed: u32,
        before: u32,
        after: u32,
    },
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
    Retry {
        kind: RetryKind,
        message: String,
    },
    /// A new plan step was recorded in Plan mode, awaiting execution.
    PlanStep {
        step: usize,
        tool: String,
        description: String,
    },
    /// An existing plan was amended. Thread.rs should re-query the full
    /// plan rather than try to apply the delta from this event alone.
    PlanAmended { plan_revision: u64 },
    /// Scope expansion (envelope trip) — always re-prompts the user
    /// regardless of approval cadence. UI should prominently surface.
    ScopeExpansion {
        capability: String,
        resource: String,
        tool: String,
        reason: String,
    },
    /// Mode or lens changed mid-session. UI updates mode badge.
    ModeChanged {
        to_mode: String,
        to_lens: Option<String>,
    },
    /// Turn finished. UI emits its `Stop` event with the mapped reason.
    TurnComplete { stop: StopReasonKind },
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
    Error,
    BudgetExceeded,
    /// Fallback for forward-compat variants.
    Other(String),
}

/// Translate a single engine event into zero-or-more UI events.
///
/// Returning a `Vec` (rather than an `Option`) because several
/// engine events need to fan out into multiple UI events — notably
/// `ContextGroupsEvicted` which may carry enough data to render
/// both a toast AND a status-bar change.
pub fn translate(ev: &AgentEvent) -> Vec<TranslatedThreadEvent> {
    use TranslatedThreadEvent as T;
    match ev {
        // ── Streaming text ─────────────────────────────────────────────
        AgentEvent::TextDelta { text } => vec![T::AgentText(text.clone())],
        AgentEvent::ReasoningDelta { content } => vec![T::AgentThinking(content.clone())],
        AgentEvent::ReasoningComplete { content, .. } => {
            // Emit the final reasoning content so UIs that buffer
            // deltas can reconcile with the canonical string. Duplicate
            // display is an accepted minor cost for correctness across
            // providers that only send a terminal chunk.
            vec![T::AgentThinking(content.clone())]
        }
        AgentEvent::ThinkingStarted { .. } => vec![T::Swallow {
            reason: "thinking-started is a reducer signal; UI renders on first delta",
        }],

        // ── Tool lifecycle ────────────────────────────────────────────
        AgentEvent::ToolCallStart { id, name } => vec![T::ToolCallStart {
            id: id.0.clone(),
            name: name.clone(),
        }],
        AgentEvent::ToolCallInput { id, delta } => vec![T::ToolCallInputDelta {
            id: id.0.clone(),
            delta: delta.clone(),
        }],
        AgentEvent::ToolCallEnd { id } => vec![T::ToolCallInputEnd { id: id.0.clone() }],
        AgentEvent::ToolResultStart { .. } => vec![T::Swallow {
            reason: "tool-result-start — wait for ToolResultEnd to emit a UI update",
        }],
        AgentEvent::ToolResultEnd {
            id,
            content,
            is_error,
        } => vec![T::ToolResult {
            id: id.0.clone(),
            content: content.clone(),
            is_error: *is_error,
        }],

        // ── Permission ────────────────────────────────────────────────
        AgentEvent::PermissionRequest {
            id,
            capability,
            description,
        } => vec![T::PermissionRequest {
            id: id.clone(),
            tool: capability.clone(),
            description: description.clone(),
        }],
        AgentEvent::PermissionDecision { .. } | AgentEvent::ApprovalDecided { .. } => {
            vec![T::Swallow {
                reason: "permission resolution — UI already updated on user action",
            }]
        }
        AgentEvent::ScopeExpansionRequested {
            capability,
            resource,
            reason,
            tool,
        } => vec![T::ScopeExpansion {
            capability: capability.clone(),
            resource: resource.clone(),
            tool: tool.clone(),
            reason: reason.clone(),
        }],

        // ── Retry signals ─────────────────────────────────────────────
        AgentEvent::LoopDetected {
            tool_name,
            consecutive_count,
        } => vec![T::Retry {
            kind: RetryKind::Loop,
            message: format!(
                "Loop detected on '{tool_name}' ({consecutive_count} consecutive)"
            ),
        }],
        AgentEvent::CircuitBreakerTriggered {
            consecutive_failures,
            last_tools,
        } => vec![T::Retry {
            kind: RetryKind::CircuitBreaker,
            message: format!(
                "Circuit breaker: {consecutive_failures} failures; last tools {last_tools:?}"
            ),
        }],
        AgentEvent::ToolTimedOut {
            tool,
            timeout_secs,
            elapsed_ms,
        } => vec![T::Retry {
            kind: RetryKind::ToolTimeout,
            message: format!("Tool '{tool}' timed out after {elapsed_ms}ms (budget {timeout_secs}s)"),
        }],
        AgentEvent::ToolCancelled { tool, elapsed_ms } => vec![T::Retry {
            kind: RetryKind::ToolCancelled,
            message: format!("Tool '{tool}' cancelled after {elapsed_ms}ms"),
        }],

        // ── Context management ────────────────────────────────────────
        AgentEvent::ContextWarning {
            level,
            used_tokens,
            max_tokens,
        } => vec![T::ContextWarning {
            level: level.clone(),
            used: *used_tokens,
            max: *max_tokens,
        }],
        AgentEvent::ContextCompacted {
            freed_tokens,
            before,
            after,
        } => vec![T::ContextCompacted {
            freed: *freed_tokens,
            before: *before,
            after: *after,
        }],
        AgentEvent::ContextGroupsEvicted {
            strategy,
            groups,
            total_tokens,
        } => vec![T::ContextEvicted {
            strategy: strategy.clone(),
            groups: groups.len() as u32,
            total_tokens: *total_tokens,
        }],

        // ── Plan mode ─────────────────────────────────────────────────
        AgentEvent::PlanStepPending {
            step,
            tool_name,
            description,
            ..
        } => vec![T::PlanStep {
            step: *step,
            tool: tool_name.clone(),
            description: description.clone(),
        }],
        AgentEvent::PlanAmended { plan_revision, .. } => vec![T::PlanAmended {
            plan_revision: *plan_revision,
        }],
        AgentEvent::AwaitingApproval { .. } => vec![T::Swallow {
            reason: "awaiting-approval — reducer owns the plan-panel render path",
        }],

        // ── Mode / lens ───────────────────────────────────────────────
        AgentEvent::ModeChanged {
            to_mode, to_lens, ..
        } => vec![T::ModeChanged {
            to_mode: to_mode.clone(),
            to_lens: to_lens.clone(),
        }],

        // ── Turn lifecycle ────────────────────────────────────────────
        AgentEvent::TurnComplete { stop_reason, .. } => vec![T::TurnComplete {
            stop: map_stop_reason(stop_reason),
        }],
        AgentEvent::Error { message } => vec![T::TurnError {
            message: message.clone(),
        }],

        // ── Reducer-only events (no UI surface today) ─────────────────
        AgentEvent::Introspection(_) => vec![T::Swallow {
            reason: "introspection — consumed by dag_state reducer",
        }],
        AgentEvent::StepStarted { .. } | AgentEvent::StepCompleted { .. } => vec![T::Swallow {
            reason: "step boundary — reducer/telemetry only",
        }],
        AgentEvent::ExecutionTreeNode { .. } | AgentEvent::ExecutionTreeUpdate { .. } => {
            vec![T::Swallow {
                reason: "execution-tree — rendered by a dedicated panel via reducer",
            }]
        }
        AgentEvent::MessagePart { .. } => vec![T::Swallow {
            reason: "structured message part — AI-elements rendering path, not ThreadEvent",
        }],
        AgentEvent::SessionPhaseChanged { .. } => vec![T::Swallow {
            reason: "session-phase — reducer/status-bar only",
        }],
        AgentEvent::RoutingDecision { .. } => vec![T::Swallow {
            reason: "routing decision — reducer/introspection panel only",
        }],
        AgentEvent::EventBufferOverflow { .. } => vec![T::Swallow {
            reason: "buffer overflow — replay via emitter, not a UI event",
        }],
        AgentEvent::DrainedStaleApproval { .. } => vec![T::Swallow {
            reason: "stale approval drain — telemetry only",
        }],
        AgentEvent::TokenLogprobSummary { .. } => vec![T::Swallow {
            reason: "logprob summary — reducer renders confidence dot",
        }],
        AgentEvent::BudgetUpdated { .. } => vec![T::Swallow {
            reason: "budget updated — reducer/status-bar",
        }],
        AgentEvent::CheckpointCreated { .. } | AgentEvent::CheckpointReverted { .. } => {
            vec![T::Swallow {
                reason: "checkpoint — timeline panel via reducer",
            }]
        }
        AgentEvent::BackgroundAgentDone { .. } => vec![T::Swallow {
            reason: "background agent — notification channel, not main thread",
        }],
        AgentEvent::CritiqueCall { .. } => vec![T::Swallow {
            reason: "critique call — reducer",
        }],
        AgentEvent::VerificationStarted { .. } | AgentEvent::VerificationCompleted { .. } => {
            vec![T::Swallow {
                reason: "verification phase — reducer",
            }]
        }
        AgentEvent::TestGateStarted { .. } | AgentEvent::TestGateCompleted { .. } => {
            vec![T::Swallow {
                reason: "test gate — reducer",
            }]
        }
        AgentEvent::ParallelToolBatchStarted { .. }
        | AgentEvent::ParallelToolBatchCompleted { .. } => vec![T::Swallow {
            reason: "parallel batch diagnostics — reducer",
        }],
        AgentEvent::ReflexionRecorded { .. } => vec![T::Swallow {
            reason: "reflexion — rendered inline on the failed tool_result, not a separate event",
        }],
        AgentEvent::CriticVerdict { .. } => vec![T::Swallow {
            reason: "critic verdict — reducer",
        }],

        // Forward-compat catch-all.
        AgentEvent::Unknown => vec![T::Swallow {
            reason: "unknown event — client may be older than engine",
        }],
    }
}

fn map_stop_reason(sr: &caduceus_core::StopReason) -> StopReasonKind {
    use caduceus_core::StopReason as S;
    match sr {
        S::EndTurn => StopReasonKind::EndTurn,
        S::ToolUse => StopReasonKind::ToolUse,
        S::MaxTokens => StopReasonKind::MaxTokens,
        S::StopSequence => StopReasonKind::EndTurn,
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
            duration_ms: 1,
        });
        assert_eq!(c, TranslatedThreadEvent::AgentThinking("done".into()));
    }

    #[test]
    fn tool_lifecycle_full_sequence() {
        let id = caduceus_core::ToolCallId("t1".to_string());
        let start = one(&AgentEvent::ToolCallStart {
            id: id.clone(),
            name: "bash".into(),
        });
        assert!(matches!(start, TranslatedThreadEvent::ToolCallStart { ref name, .. } if name == "bash"));

        let delta = one(&AgentEvent::ToolCallInput {
            id: id.clone(),
            delta: "{\"cmd\":".into(),
        });
        assert!(matches!(
            delta,
            TranslatedThreadEvent::ToolCallInputDelta { .. }
        ));

        let end = one(&AgentEvent::ToolCallEnd { id: id.clone() });
        assert!(matches!(end, TranslatedThreadEvent::ToolCallInputEnd { .. }));

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
            depends_on: vec![],
            parent_step_id: None,
        });
        assert!(matches!(s, TranslatedThreadEvent::PlanStep { step: 1, .. }));

        let a = one(&AgentEvent::PlanAmended {
            kind: "replace".into(),
            step: 1,
            ok: true,
            reason: "ok".into(),
            plan_revision: 7,
        });
        assert_eq!(
            a,
            TranslatedThreadEvent::PlanAmended { plan_revision: 7 }
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
    fn mode_changed_translates() {
        let got = one(&AgentEvent::ModeChanged {
            from_mode: "plan".into(),
            to_mode: "act".into(),
            from_lens: None,
            to_lens: Some("fast".into()),
        });
        assert_eq!(
            got,
            TranslatedThreadEvent::ModeChanged {
                to_mode: "act".into(),
                to_lens: Some("fast".into())
            }
        );
    }

    #[test]
    fn turn_complete_maps_stop_reasons() {
        let usage = TokenUsage::default();
        let cases = [
            (StopReason::EndTurn, StopReasonKind::EndTurn),
            (StopReason::ToolUse, StopReasonKind::ToolUse),
            (StopReason::MaxTokens, StopReasonKind::MaxTokens),
            (StopReason::StopSequence, StopReasonKind::EndTurn),
            (StopReason::Error, StopReasonKind::Error),
            (StopReason::BudgetExceeded, StopReasonKind::BudgetExceeded),
        ];
        for (sr, expected) in cases {
            let got = one(&AgentEvent::TurnComplete {
                stop_reason: sr,
                usage: usage.clone(),
            });
            assert_eq!(got, TranslatedThreadEvent::TurnComplete { stop: expected });
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
            AgentEvent::TextDelta {
                text: "x".into(),
            },
            AgentEvent::StepStarted { step_id: 1 },
            AgentEvent::Unknown,
        ] {
            assert!(!translate(&ev).is_empty(), "empty translation of {ev:?}");
        }
    }
}
