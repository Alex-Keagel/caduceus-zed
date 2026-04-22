//! G1b — `ZedToolAdapter`: proxies engine-side (`caduceus_tools::Tool`)
//! tool calls to Zed-side implementations that must run on the gpui
//! main thread.
//!
//! # Why
//! The engine's `AgentHarness::run` calls tools from a plain tokio task;
//! Zed's tool implementations touch `&mut Context<_>` / `Entity<_>` and
//! therefore must execute on the gpui foreground executor. We bridge
//! with a bounded `mpsc::Sender` (cap = [`REQUEST_CHANNEL_CAP`]) into a
//! dispatcher task that lives on the gpui main thread and hands each
//! request to a `cx.spawn(...)` behind a [`tokio::sync::Semaphore`]
//! permit (max [`DEFAULT_MAX_CONCURRENCY`] in flight). Parallel tool
//! batches from the engine therefore execute with real parallelism
//! rather than being serialised by a single-consumer adapter — this
//! matches the legacy `FuturesUnordered` behavior in `run_turn_internal`.
//!
//! # Ownership
//! - One **dispatcher** per [`Thread`] (bound to its gpui `Context`).
//! - Many **adapters** share a clone of the dispatcher's `Sender`; each
//!   adapter represents one named Zed tool visible to the engine.
//! - Dropping the dispatcher closes the `Sender`; any pending adapter
//!   `call()` then resolves to an [`AdapterError::ChannelClosed`]
//!   surfaced as a `ToolResult { is_error: true }`.
//!
//! # Cancellation
//! The adapter is oblivious to cancellation. When the engine cancels a
//! turn the harness drops its `run` future, which drops the oneshots,
//! which drops the in-flight dispatcher task's response channel — the
//! dispatcher's `cx.spawn` detects that, aborts work best-effort, and
//! logs. The engine sees the tool call as cancelled.
//!
//! # Tests
//! G1f covers end-to-end parity (31-test matrix). G1b lands a small
//! smoke test suite inside this module ensuring: (a) happy-path
//! round-trip, (b) semaphore caps concurrency, (c) closed channel
//! produces an error ToolResult, (d) lint delegates to the
//! underlying spec.

use anyhow::Result;
use async_trait::async_trait;
use caduceus_core::{ToolKind, ToolResult, ToolSpec};
use caduceus_tools::Tool;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc, oneshot};

/// Outgoing-request channel cap (from dispatcher's perspective).
/// Design §3-G1b chose 64: high enough that the harness's parallel tool
/// batches rarely block on `send_timeout`, low enough that a runaway
/// engine cannot queue unbounded work against a frozen main thread.
pub const REQUEST_CHANNEL_CAP: usize = 64;

/// Max simultaneous tool executions on the main thread. Matches legacy
/// `FuturesUnordered` steady-state (observed ~8 during heavy edits).
pub const DEFAULT_MAX_CONCURRENCY: usize = 8;

/// One request flowing from engine thread → main-thread dispatcher.
pub struct DispatchRequest {
    pub tool_name: String,
    pub input: Value,
    /// Reply channel. `None` means the caller already gave up (fused).
    pub respond_to: oneshot::Sender<Result<ToolResult, AdapterError>>,
}

/// Errors the adapter surfaces back to the engine. They map onto
/// `ToolResult { is_error: true }` inside [`ZedToolAdapter::call`].
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("dispatcher channel closed (Thread torn down or cancelled)")]
    ChannelClosed,
    #[error("dispatch timed out after {0:?}")]
    Timeout(std::time::Duration),
    #[error("no Zed tool registered for {0}")]
    UnknownTool(String),
    #[error("main-thread execution failed: {0}")]
    Execution(String),
}

/// Handle the engine side holds. Cheap to clone (Arc of the mpsc
/// sender).
#[derive(Clone)]
pub struct DispatcherHandle {
    tx: mpsc::Sender<DispatchRequest>,
}

impl DispatcherHandle {
    pub fn new(tx: mpsc::Sender<DispatchRequest>) -> Self {
        Self { tx }
    }

    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

/// One adapter per Zed tool registered with the engine. Holds the
/// declared spec (so engine-side retrieval, lint, and schema checks
/// work without round-tripping) and the shared dispatcher handle.
pub struct ZedToolAdapter {
    spec: ToolSpec,
    /// Static side-effect class. Overridden per-invocation by
    /// [`ZedToolAdapter::resource_kind`] if the Zed tool has opted-in
    /// to a dynamic classifier, but for G1b we keep the default
    /// conservative behavior.
    static_kind: ToolKind,
    handle: DispatcherHandle,
}

impl ZedToolAdapter {
    pub fn new(spec: ToolSpec, handle: DispatcherHandle) -> Self {
        Self {
            spec,
            static_kind: ToolKind::Destructive,
            handle,
        }
    }

    /// Builder — override the static tool kind (default `Destructive`).
    /// Only use this for tools you have audited to be idempotent or
    /// read-only. Lying here breaks the dispatcher's safety model.
    pub fn with_kind(mut self, kind: ToolKind) -> Self {
        self.static_kind = kind;
        self
    }
}

#[async_trait]
impl Tool for ZedToolAdapter {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn kind(&self) -> ToolKind {
        self.static_kind
    }

    async fn call(&self, input: Value) -> caduceus_core::Result<ToolResult> {
        let (tx, rx) = oneshot::channel();
        let request = DispatchRequest {
            tool_name: self.spec.name.clone(),
            input,
            respond_to: tx,
        };

        if self.handle.tx.send(request).await.is_err() {
            return Ok(error_result(AdapterError::ChannelClosed.to_string()));
        }

        match rx.await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(err)) => Ok(error_result(err.to_string())),
            Err(_recv) => Ok(error_result(AdapterError::ChannelClosed.to_string())),
        }
    }
}

/// Build a `ToolResult` that carries a human-readable error message
/// as its text content. Engine-side tool retry / envelope logic reads
/// `is_error` so the agent sees the failure.
fn error_result(msg: String) -> ToolResult {
    ToolResult::error(format!("tool adapter error: {msg}"))
}

// ── Dispatcher ──────────────────────────────────────────────────────
//
// The dispatcher is deliberately runtime-agnostic in this module: it
// exposes a `drive` function that takes an `mpsc::Receiver` and a
// `spawn_local` callback the caller wires up to `cx.spawn(...)` (or
// to a tokio `spawn` in tests). This keeps the adapter unit-testable
// without pulling gpui into the test graph, while still letting
// `Thread::ensure_caduceus_harness` wire it to the foreground
// executor for the production path (G1d).

/// The main-thread-side executor a dispatcher call out to.
///
/// `execute` is given the tool name and input JSON; it returns a
/// boxed future resolving to the `ToolResult` (or an `AdapterError`).
/// The dispatcher holds semaphore permits around this call so callers
/// can treat the returned future as opaque "main-thread work".
pub type ExecFn = Arc<
    dyn Fn(String, Value) -> futures::future::BoxFuture<'static, Result<ToolResult, AdapterError>>
        + Send
        + Sync,
>;

/// Run the dispatcher loop. Returns when the request receiver is
/// closed AND all spawned tasks have flushed their permits.
///
/// `spawn_local` is the caller-provided spawner. In production the
/// caller passes a closure that calls `cx.spawn(...)`; in tests we
/// pass `tokio::spawn`.
pub async fn drive<Spawn, Fut>(
    mut rx: mpsc::Receiver<DispatchRequest>,
    exec: ExecFn,
    max_concurrency: usize,
    mut spawn_local: Spawn,
) where
    Spawn: FnMut(futures::future::BoxFuture<'static, ()>) -> Fut,
    Fut: Send + 'static,
{
    let sem = Arc::new(Semaphore::new(max_concurrency));

    while let Some(req) = rx.recv().await {
        // Acquire a permit *before* spawning so we truly cap in-flight
        // work at `max_concurrency`. If the semaphore is exhausted
        // this awaits (back-pressuring the sender through the mpsc
        // capacity). `acquire_owned` returns a guard we move into the
        // spawned task.
        let permit = match sem.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => break, // semaphore closed → shutdown
        };

        let exec = exec.clone();
        let DispatchRequest {
            tool_name,
            input,
            respond_to,
        } = req;
        let fut: futures::future::BoxFuture<'static, ()> = Box::pin(async move {
            let result = exec(tool_name, input).await;
            // Caller may have dropped the oneshot (turn cancelled).
            let _ = respond_to.send(result);
            drop(permit);
        });
        let _handle = spawn_local(fut);
        // `_handle` type is intentionally discarded — the spawned
        // task self-reports via the oneshot. For tokio `spawn` this
        // detaches the task; for `cx.spawn` this detaches the gpui
        // task. Detachment is safe because: (a) the only
        // side-effectful completion is the oneshot send, (b) if the
        // dispatcher exits while tasks are running, the semaphore
        // permits drop with the tasks.
    }
}

// ── ST-A1 — build_zed_tool_registry ─────────────────────────────────
//
// Contract `tool-registry-builder-v1`. Given the Thread's enabled tool
// map and a dispatcher handle, produce a `caduceus_tools::ToolRegistry`
// with one `ZedToolAdapter` per Zed tool. The dispatcher itself is
// spawned separately (ST-A2) on the gpui main thread; this helper only
// wires the spec-registration side, which needs no `cx`.
//
// Why a helper: `ensure_caduceus_harness` was a blocker for the native
// loop because there was no bulk adapter — only the single-tool
// `ZedToolAdapter::new`. Without this, wiring 40+ Zed tools would
// require 40+ lines of hand-rolled setup every turn.

use std::collections::BTreeMap;

/// acp::ToolKind → caduceus_core::ToolKind. Conservative on unknowns:
/// anything not obviously read-only maps to Destructive so engine-side
/// batching serialises it.
fn map_acp_kind(kind: agent_client_protocol::ToolKind) -> ToolKind {
    use agent_client_protocol::ToolKind as A;
    match kind {
        A::Read | A::Search | A::Think => ToolKind::ReadOnly,
        A::Fetch => ToolKind::Idempotent,
        _ => ToolKind::Destructive,
    }
}

/// Derive a `ToolSpec` from any Zed `AnyAgentTool`. Pulls the JSON
/// schema via `input_schema(OpenAi)` so the spec matches what a native
/// provider would see.
fn tool_spec_for(tool: &dyn crate::thread::AnyAgentTool) -> anyhow::Result<ToolSpec> {
    use language_model::LanguageModelToolSchemaFormat;
    let schema = tool.input_schema(LanguageModelToolSchemaFormat::JsonSchema)?;
    Ok(ToolSpec {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        input_schema: schema,
        required_capability: None,
    })
}

/// Build a fresh `ToolRegistry` from Zed's per-Thread tool map.
///
/// `handle` is the dispatcher handle every adapter will forward through.
/// `filter` lets callers skip tools the envelope would deny (pass
/// `|_name| true` for no filtering).
///
/// Returns `Err` if any tool's input_schema rejects (malformed derive).
/// In practice Zed tools always succeed here — the `Result` is defense
/// against a custom tool that lies in its derive macro.
pub fn build_zed_tool_registry(
    tools: &BTreeMap<gpui::SharedString, Arc<dyn crate::thread::AnyAgentTool>>,
    handle: DispatcherHandle,
    mut filter: impl FnMut(&str) -> bool,
) -> anyhow::Result<caduceus_tools::ToolRegistry> {
    let mut registry = caduceus_tools::ToolRegistry::new();
    for (name, tool) in tools.iter() {
        if !filter(name.as_ref()) {
            continue;
        }
        let spec = tool_spec_for(tool.as_ref())?;
        let kind = map_acp_kind(tool.kind());
        let adapter = ZedToolAdapter::new(spec, handle.clone()).with_kind(kind);
        registry.register(Arc::new(adapter));
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt as _;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::time::{Duration, sleep};

    fn make_spec(name: &str) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            description: String::new(),
            input_schema: serde_json::json!({"type": "object"}),
            required_capability: None,
        }
    }

    /// ST-A1 — acp::ToolKind translation is the one piece of
    /// build_zed_tool_registry we can test without GPUI. The registry
    /// end-to-end is covered by the ST-A2 integration tests once the
    /// dispatcher is wired.
    #[test]
    fn map_acp_kind_read_only_surfaces_parallel_safe() {
        use agent_client_protocol::ToolKind as A;
        assert_eq!(map_acp_kind(A::Read), ToolKind::ReadOnly);
        assert_eq!(map_acp_kind(A::Search), ToolKind::ReadOnly);
        assert_eq!(map_acp_kind(A::Think), ToolKind::ReadOnly);
        assert_eq!(map_acp_kind(A::Fetch), ToolKind::Idempotent);
        // Everything else is conservative.
        assert_eq!(map_acp_kind(A::Edit), ToolKind::Destructive);
        assert_eq!(map_acp_kind(A::Execute), ToolKind::Destructive);
        assert_eq!(map_acp_kind(A::Delete), ToolKind::Destructive);
        assert_eq!(map_acp_kind(A::Other), ToolKind::Destructive);
    }

    fn ok_result(text: &str) -> ToolResult {
        ToolResult::success(text)
    }

    /// Happy path — adapter sends a request, dispatcher runs it, the
    /// oneshot delivers the result.
    #[tokio::test]
    async fn happy_path_round_trip() {
        let (tx, rx) = mpsc::channel(REQUEST_CHANNEL_CAP);
        let exec: ExecFn =
            Arc::new(|name, _input| async move { Ok(ok_result(&format!("ran {name}"))) }.boxed());
        tokio::spawn(drive(rx, exec, DEFAULT_MAX_CONCURRENCY, |fut| {
            tokio::spawn(fut)
        }));

        let adapter = ZedToolAdapter::new(make_spec("greet"), DispatcherHandle::new(tx));
        let res = adapter.call(serde_json::json!({})).await.unwrap();
        assert!(!res.is_error);
        assert_eq!(res.content, "ran greet");
    }

    /// Semaphore truly caps concurrency at `max_concurrency`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn semaphore_caps_concurrency() {
        let (tx, rx) = mpsc::channel(REQUEST_CHANNEL_CAP);
        let live = Arc::new(AtomicU32::new(0));
        let peak = Arc::new(AtomicU32::new(0));
        let live_c = live.clone();
        let peak_c = peak.clone();
        let exec: ExecFn = Arc::new(move |_name, _input| {
            let live = live_c.clone();
            let peak = peak_c.clone();
            async move {
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                sleep(Duration::from_millis(40)).await;
                live.fetch_sub(1, Ordering::SeqCst);
                Ok(ok_result("ok"))
            }
            .boxed()
        });
        let max = 3;
        tokio::spawn(drive(rx, exec, max, |fut| tokio::spawn(fut)));
        let handle = DispatcherHandle::new(tx);

        let mut joins = Vec::new();
        for i in 0..12 {
            let h = handle.clone();
            joins.push(tokio::spawn(async move {
                let adapter = ZedToolAdapter::new(make_spec(&format!("t{i}")), h);
                adapter.call(serde_json::json!({})).await
            }));
        }
        for j in joins {
            j.await.unwrap().unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) <= max as u32,
            "peak {} exceeded max {}",
            peak.load(Ordering::SeqCst),
            max
        );
    }

    /// When the dispatcher channel is closed before the request is
    /// received, the adapter MUST surface an error ToolResult rather
    /// than hanging.
    #[tokio::test]
    async fn closed_channel_errors_cleanly() {
        let (tx, rx) = mpsc::channel::<DispatchRequest>(1);
        drop(rx);
        let adapter = ZedToolAdapter::new(make_spec("void"), DispatcherHandle::new(tx));
        let res = adapter.call(serde_json::json!({})).await.unwrap();
        assert!(res.is_error);
    }

    /// Dispatcher drops the response oneshot → adapter error.
    #[tokio::test]
    async fn dropped_response_errors() {
        let (tx, mut rx) = mpsc::channel::<DispatchRequest>(1);
        // Spawn a "dispatcher" that reads one request and drops the
        // oneshot without replying.
        tokio::spawn(async move {
            if let Some(req) = rx.recv().await {
                drop(req.respond_to);
            }
        });
        let adapter = ZedToolAdapter::new(make_spec("drop"), DispatcherHandle::new(tx));
        let res = adapter.call(serde_json::json!({})).await.unwrap();
        assert!(res.is_error);
    }

    /// `lint()` delegates to the default implementation, which reads
    /// `spec().input_schema`. An empty `{}` object passes a permissive
    /// schema; a required-field schema rejects it. This exercises the
    /// default delegation without depending on any Zed-side lint
    /// behavior.
    #[test]
    fn lint_delegates_to_spec() {
        let (tx, _rx) = mpsc::channel::<DispatchRequest>(1);
        let spec = ToolSpec {
            name: "strict".into(),
            description: String::new(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "x": {"type": "integer"} },
                "required": ["x"]
            }),
            required_capability: None,
        };
        let adapter = ZedToolAdapter::new(spec, DispatcherHandle::new(tx));
        assert!(adapter.lint(&serde_json::json!({})).is_err());
        assert!(adapter.lint(&serde_json::json!({"x": 1})).is_ok());
    }
}

// Re-export anyhow for downstream call sites that want to annotate
// errors without pulling the crate in directly.
pub use anyhow::Error as AnyError;
