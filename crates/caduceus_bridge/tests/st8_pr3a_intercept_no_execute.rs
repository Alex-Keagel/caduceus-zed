//! ST8-PR3A — End-to-end regression for resume-on-grant + Intercept.
//!
//! This test closes the rubber-duck CRITICAL #1 verification gap left open
//! by caduceus PR-2 (the orchestrator-side resume-on-grant landing). The
//! invariant under test is the central safety property of plan-mode resume:
//!
//!   When a Deny is upgraded to Intercept by the granted envelope, the tool
//!   MUST NOT actually execute. The tool result MUST be the simulation
//!   message, and `GrantResolved` MUST report `granted:simulated`.
//!
//! Pipeline exercised:
//!   1. MockLlmAdapter scripts a single `write_file` tool call, then EndTurn.
//!   2. AgentHarness preflights the call against an active envelope whose
//!      `write` is `PathAllowlist::closed()` → Decision::Deny.
//!   3. Resume-on-grant catches the Deny, emits `GrantPending`, parks the
//!      tool, and waits for `submit_grant`.
//!   4. The test's grant-deliver task observes `GrantPending` and submits
//!      `GrantOutcome::Granted { updated: <plan-style envelope> }`. The
//!      granted envelope differs from the active one only in
//!      `write.intercept_denied = true`, so it widens monotonically and
//!      maps the same path to Intercept (simulation) instead of Deny.
//!   5. AgentHarness re-runs preflight against the granted envelope, gets
//!      `Decision::Intercept`, synthesizes the simulation tool result,
//!      emits `GrantResolved { outcome: "granted:simulated" }`, and feeds
//!      the simulation result back into the loop without ever invoking
//!      the recording tool.
//!
//! Asserts:
//!   • The recording tool's invocation counter is exactly 0.
//!   • A `GrantPending` and a matching `GrantResolved { outcome: "granted:simulated" }`
//!     were observed on the emitter broadcast.
//!   • The harness `run` future resolved Ok.
//!
//! This is the parity fixture for the bridge-level `with_resume_on_grant`
//! and `with_grant_timeout` forwarding shipped in PR-3A.

use async_trait::async_trait;
use caduceus_bridge::orchestrator::OrchestratorBridge;
use caduceus_core::{AgentEvent, StopReason, ToolResult, ToolSpec, ToolUse};
use caduceus_permissions::GrantOutcome;
use caduceus_permissions::envelope::{PathAllowlist, PermissionEnvelope};
use caduceus_providers::ChatResponse;
use caduceus_providers::mock::MockLlmAdapter;
use caduceus_tools::{Tool, ToolRegistry};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Tool registered as `write_file` so `classify_tool_call` routes it through
/// the Write capability. Records invocations on a shared counter — the test
/// asserts this stays at 0 throughout the run.
struct RecordingWriteTool {
    counter: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for RecordingWriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Recording write tool (must NOT execute under Intercept).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
            required_capability: None,
        }
    }

    async fn call(&self, _input: Value) -> caduceus_core::Result<ToolResult> {
        // If we ever reach here, the test fails — Intercept must short-circuit
        // the tool dispatcher entirely.
        self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(ToolResult::success("recording-tool-invoked"))
    }
}

fn write_tool_call_response(id: &str, path: &str) -> ChatResponse {
    ChatResponse {
        content: String::new(),
        input_tokens: 10,
        output_tokens: 20,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        stop_reason: StopReason::ToolUse,
        tool_calls: vec![ToolUse {
            id: id.into(),
            name: "write_file".into(),
            input: json!({ "path": path, "content": "x" }),
        }],
        logprobs: None,
        thinking: String::new(),
    }
}

fn end_turn_response(text: &str) -> ChatResponse {
    ChatResponse {
        content: text.into(),
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        stop_reason: StopReason::EndTurn,
        tool_calls: vec![],
        logprobs: None,
        thinking: String::new(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn st8_intercept_does_not_execute_recording_tool() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let tmp = tempfile::tempdir().expect("tempdir");
    let bridge = OrchestratorBridge::new(tmp.path()).with_native_loop_enabled(true);

    let counter = Arc::new(AtomicUsize::new(0));
    let recording = Arc::new(RecordingWriteTool {
        counter: counter.clone(),
    });
    let mut tools = ToolRegistry::new();
    tools.register(recording);

    // Two-turn script: tool call → tool result fed back → EndTurn.
    let provider = Arc::new(MockLlmAdapter::new(vec![
        write_tool_call_response("tool-call-1", "src/forbidden.rs"),
        end_turn_response("done"),
    ]));

    // Active envelope: identical to plan_preset BUT with write fully closed
    // (intercept_denied=false, empty allow). This forces Decision::Deny on
    // any write — the trigger for resume-on-grant.
    let mut env_active = PermissionEnvelope::plan_preset();
    env_active.write = PathAllowlist::closed();

    // Granted envelope: vanilla plan_preset → write.intercept_denied=true →
    // same path now maps to Decision::Intercept. Differs from `env_active`
    // ONLY on `write.intercept_denied` (false → true), so monotonic-widening
    // validation passes (read/network/exec/cadence/scope/policy all match).
    let env_grant = PermissionEnvelope::plan_preset();

    let built = bridge
        .harness(provider, tools, "you are a test agent")
        .envelope(env_active)
        .with_resume_on_grant(true)
        .with_grant_timeout(Duration::from_secs(3))
        .with_emitter()
        .no_approval()
        .build();

    let harness = Arc::new(built.harness);
    let emitter = built.emitter.expect("with_emitter() returns Some emitter");

    // Subscribe BEFORE spawning anything so we never miss GrantPending.
    let mut event_sub = emitter.subscribe();

    // Grant-deliver task: watches the broadcast for GrantPending and
    // immediately submits a Granted outcome. Records all events for the
    // post-run assertion phase.
    let h_grant = harness.clone();
    let env_grant_for_task = env_grant.clone();
    let collected_events = Arc::new(tokio::sync::Mutex::new(Vec::<AgentEvent>::new()));
    let collected_for_task = collected_events.clone();

    let grant_task = tokio::spawn(async move {
        loop {
            match event_sub.recv().await {
                Ok(ev) => {
                    let is_pending = matches!(ev, AgentEvent::GrantPending { .. });
                    let pending_id = if let AgentEvent::GrantPending { tool_use_id, .. } = &ev {
                        Some(tool_use_id.clone())
                    } else {
                        None
                    };
                    collected_for_task.lock().await.push(ev);
                    if is_pending {
                        if let Some(id) = pending_id {
                            let _ = h_grant
                                .submit_grant(
                                    &id,
                                    GrantOutcome::Granted {
                                        updated: env_grant_for_task.clone(),
                                    },
                                )
                                .await;
                        }
                    }
                }
                // Lagged → keep going. Closed → emitter dropped, end of stream.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    let mut state = bridge.new_session("mock", "mock-model");
    let mut history = OrchestratorBridge::new_history();

    // ── Act ──────────────────────────────────────────────────────────────
    let run_result = harness
        .run(&mut state, &mut history, "please write the file")
        .await;

    // Drop the emitter so the broadcast channel closes and the grant task
    // can exit its recv loop. The emitter was cloned into the harness, so
    // we must also drop our local handle.
    drop(emitter);
    // Give the grant task a brief window to drain remaining events before
    // we abort it. With worker_threads=2 this is essentially instant; we
    // bound it to keep the test deterministic.
    let _ = tokio::time::timeout(Duration::from_millis(200), async {
        // Cannot await the join handle here without dropping all emitter
        // references, which we just did. The harness still holds one;
        // simply yield until the task finishes naturally or the timeout
        // expires.
    })
    .await;
    grant_task.abort();
    let _ = grant_task.await;

    // ── Assert ───────────────────────────────────────────────────────────
    assert!(
        run_result.is_ok(),
        "harness.run failed under Intercept simulation: {run_result:?}"
    );

    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "Intercept must short-circuit the dispatcher — recording tool must NEVER be invoked"
    );

    let events = collected_events.lock().await.clone();

    let saw_pending = events
        .iter()
        .any(|e| matches!(e, AgentEvent::GrantPending { .. }));
    assert!(
        saw_pending,
        "expected at least one GrantPending event; got {events:#?}"
    );

    let saw_granted_simulated = events.iter().any(|e| {
        matches!(
            e,
            AgentEvent::GrantResolved { outcome, .. } if outcome == "granted:simulated"
        )
    });
    assert!(
        saw_granted_simulated,
        "expected GrantResolved with outcome=\"granted:simulated\"; got {events:#?}"
    );
}
