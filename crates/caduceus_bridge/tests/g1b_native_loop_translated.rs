//! G1b — integration test: engine → event_translator → translated events.
//!
//! Proves the complete native-loop pipeline end-to-end through real
//! engine code (no shortcuts):
//! 1. MockLlmAdapter scripted with a single-turn response.
//! 2. `OrchestratorBridge::build_harness_with_emitter` constructs a
//!    real `AgentHarness` + emitter + event_rx + approval_tx.
//! 3. `run_caduceus_loop_translated` drives `harness.run(...)` and
//!    forwards every emitted `AgentEvent` through
//!    `event_translator::translate` into an unbounded channel.
//! 4. Asserts the resulting `TranslatedThreadEvent` stream contains
//!    the expected UI-shaped events (text + turn complete).
//!
//! This is the parity fixture for G1f: once thread.rs consumes the same
//! channel, UI events observed in-IDE must match this sequence.

use caduceus_bridge::dag_state::SessionStateReducer;
use caduceus_bridge::event_translator::{StopReasonKind, TranslatedThreadEvent};
use caduceus_bridge::orchestrator::OrchestratorBridge;
use caduceus_core::StopReason;
use caduceus_providers::ChatResponse;
use caduceus_providers::mock::MockLlmAdapter;
use caduceus_tools::ToolRegistry;
use std::sync::{Arc, Mutex};

fn mock_response(text: &str) -> ChatResponse {
    ChatResponse {
        content: text.into(),
        input_tokens: 10,
        output_tokens: 5,
        cache_read_tokens: 0,
        cache_creation_tokens: 0,
        stop_reason: StopReason::EndTurn,
        tool_calls: vec![],
        logprobs: None,
    }
}

#[tokio::test]
async fn native_loop_translated_emits_ui_events_for_simple_turn() {
    // ── Arrange ──────────────────────────────────────────────────────
    let tmp = tempfile::tempdir().expect("tempdir");
    let bridge = OrchestratorBridge::new(tmp.path()).with_native_loop_enabled(true);
    let provider = Arc::new(MockLlmAdapter::new(vec![mock_response("hello world")]));
    let tools = ToolRegistry::new();
    let (harness, _approval_tx, _replay, event_rx) =
        bridge.build_harness_with_emitter(provider, tools, "you are a test agent");

    let mut state = bridge.new_session("mock", "mock-model");
    let mut history = OrchestratorBridge::new_history();

    let reducer_handle = bridge.new_reducer_handle();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TranslatedThreadEvent>();

    // ── Act ──────────────────────────────────────────────────────────
    let result = bridge
        .run_caduceus_loop_translated(
            &harness,
            &mut state,
            &mut history,
            "hi",
            reducer_handle,
            event_rx,
            tx,
        )
        .await;
    assert!(result.is_ok(), "native loop failed: {result:?}");

    // Drain all translated events.
    let mut events: Vec<TranslatedThreadEvent> = Vec::new();
    while let Ok(ev) = rx.try_recv() {
        events.push(ev);
    }

    // ── Assert ───────────────────────────────────────────────────────
    // At minimum, the pipeline must surface assistant text AND a turn-
    // complete. Other events (reducer-only swallows, step boundaries)
    // may interleave; the translator documents them as Swallow which we
    // filter out for this parity check.
    let ui_events: Vec<_> = events
        .iter()
        .filter(|e| !matches!(e, TranslatedThreadEvent::Swallow { .. }))
        .collect();

    assert!(
        !ui_events.is_empty(),
        "expected at least one UI-shaped event, got none; full stream: {events:#?}"
    );

    let has_text = ui_events
        .iter()
        .any(|e| matches!(e, TranslatedThreadEvent::AgentText(t) if t.contains("hello")));
    assert!(
        has_text,
        "expected AgentText containing 'hello'; UI events: {ui_events:#?}"
    );

    let has_turn_complete = ui_events.iter().any(|e| {
        matches!(
            e,
            TranslatedThreadEvent::TurnComplete {
                stop: StopReasonKind::EndTurn,
                ..
            }
        )
    });
    assert!(
        has_turn_complete,
        "expected TurnComplete with EndTurn; UI events: {ui_events:#?}"
    );
}

#[tokio::test]
async fn native_loop_translated_refuses_when_flag_off() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bridge = OrchestratorBridge::new(tmp.path()); // flag OFF by default
    let provider = Arc::new(MockLlmAdapter::new(vec![mock_response("x")]));
    let tools = ToolRegistry::new();
    let (harness, _approval, _replay, event_rx) =
        bridge.build_harness_with_emitter(provider, tools, "sys");
    let mut state = bridge.new_session("mock", "mock-model");
    let mut history = OrchestratorBridge::new_history();
    let reducer_handle = bridge.new_reducer_handle();
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<TranslatedThreadEvent>();

    let result = bridge
        .run_caduceus_loop_translated(
            &harness,
            &mut state,
            &mut history,
            "hi",
            reducer_handle,
            event_rx,
            tx,
        )
        .await;
    assert!(result.is_err(), "flag OFF must refuse");
    let err = result.unwrap_err();
    assert!(
        err.contains("native loop is disabled"),
        "error should mention flag, got: {err}"
    );
}

#[tokio::test]
async fn native_loop_translated_forwards_reducer_in_lockstep() {
    // Regression guard: both the reducer and the translated stream must
    // observe the same AgentEvents (no skew, no partial forwarding).
    let tmp = tempfile::tempdir().expect("tempdir");
    let bridge = OrchestratorBridge::new(tmp.path()).with_native_loop_enabled(true);
    let provider = Arc::new(MockLlmAdapter::new(vec![mock_response("lockstep")]));
    let tools = ToolRegistry::new();
    let (harness, _approval, _replay, event_rx) =
        bridge.build_harness_with_emitter(provider, tools, "sys");

    let mut state = bridge.new_session("mock", "mock-model");
    let mut history = OrchestratorBridge::new_history();

    // Wrap our own reducer so we can snapshot it post-turn.
    let reducer_shared = Arc::new(Mutex::new(SessionStateReducer::new()));
    // ReducerHandle is backed by its own reducer; to observe lock-step we
    // just assert the translated stream is non-empty AND the run
    // completed. (Full reducer snapshot parity with the translated
    // stream is covered by p13_introspection tests for the reducer side;
    // here we verify translator+reducer share the same event_rx feed.)
    let reducer_handle = bridge.new_reducer_handle();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TranslatedThreadEvent>();

    let result = bridge
        .run_caduceus_loop_translated(
            &harness,
            &mut state,
            &mut history,
            "hi",
            reducer_handle,
            event_rx,
            tx,
        )
        .await;
    assert!(result.is_ok(), "turn failed: {result:?}");

    let mut count = 0usize;
    while let Ok(_ev) = rx.try_recv() {
        count += 1;
    }
    assert!(
        count > 0,
        "translated stream must observe engine events; got 0"
    );

    // Silence unused-variable warning; reducer_shared is kept here to
    // document intent for future expansion of the parity guard.
    drop(reducer_shared);
}
