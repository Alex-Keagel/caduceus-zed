//! Orchestrator bridge — agent harness, conversation history, instructions,
//! PRD parsing, task management, progress inference, scaffolding, and tree visualization.

use caduceus_core::AgentEvent;
use caduceus_core::{ProviderId, SessionState};
use caduceus_mcp::{
    McpServerManager,
    mcp_tool_bridge::{McpInvoker, McpToolBridge},
};
use caduceus_orchestrator::{
    AgentEventEmitter, AgentHarness, AgentScaffolder, ConversationHistory, PrdParser,
    ProgressInferrer, SkillScaffolder, TaskRecommender, execute_tool_calls,
    instructions::{self, InstructionLoader, InstructionSet},
};
use caduceus_providers::LlmAdapter;
use caduceus_tools::ToolRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::PermissionEnvelope;
use crate::bridge_error::BridgeError;
use crate::{BuiltinScopedContextInjector, ContextInjector};

// Re-export orchestrator types for consumers.
pub use caduceus_orchestrator::{
    ExecutionTreeViz, HierarchicalTask, InferredProgress, PrdTask, TaskRecommendation, TaskTree,
    TimeEntry, TimeTracker, VizTreeNode,
    automations::{Automation, AutomationAgentConfig, AutomationRegistry, AutomationTrigger},
    background::{BackgroundAgent, BackgroundStatus},
    compaction::{self, CompactMessage, CompactionPipeline, CompactionTrigger, ContextStats},
    context::{
        self, AssembledContext, ContextAssembler, ContextManager, ContextSource, ContextZone,
        PinnedContext, estimate_tokens,
    },
    headless::{HeadlessConfig, HeadlessResult, OutputFormat as HeadlessOutputFormat},
    kanban::{CardStatus, KanbanBoard, KanbanCard, KanbanColumn},
    mentions::MentionResolver,
    workers::{AgentConfig, SharedContext, TaskDAG, TaskDefinition, TaskStatus, TeamResult},
};

// Re-export types needed by tools.
pub use caduceus_core::ModelId;
pub use caduceus_core::TokenBudget;
pub use caduceus_orchestrator::automations::AutomationResult;
pub use caduceus_orchestrator::broadcast_bus::{BroadcastBus, BusError, BusMessage};
pub use caduceus_orchestrator::checkpoint::{
    BatchState, CheckpointError, CheckpointId, CheckpointStore, FileSnapshot, ToolBatchCheckpoint,
};
pub use caduceus_orchestrator::compaction_scorer::{
    self as bridge_compaction_scorer, BradleyTerryModel, Pair,
};
pub use caduceus_orchestrator::compaction_telemetry::{
    CompactionEvent, CompactionTelemetry, StrategyName,
};
pub use caduceus_orchestrator::context_fold::{
    ExpandError, FoldedTranscript, TranscriptId, TranscriptStore,
};
pub use caduceus_orchestrator::learned_selector::{LearnedSelector, SelectionMode};
pub use caduceus_orchestrator::memory_blocks::{
    ArchivalSummary, BlockLimits, CompactionReport as MemoryCompactionReport, MemoryBlocks,
    WorkingMessage,
};
pub use caduceus_orchestrator::modes::AgentMode;
pub use caduceus_orchestrator::modes::{
    ActionPlan, AmendError, AppliedAmendment, PlanAmendment, PlannedAction,
};
pub use caduceus_orchestrator::notifications::{
    self as bridge_notifications, NOTIFICATIONS_CHANNEL, Notification,
    Severity as NotificationSeverity,
};

// ── P7.1 — StepId on SessionState (G26) ─────────────────────────────────
pub use caduceus_core::StepId;

// ── P7.2 — OpenTelemetry GenAI mapper (G23) ─────────────────────────────
pub use caduceus_telemetry::genai::{
    GenAiContext, GenAiMapper, GenAiSpan, GenAiSpanExporter, GenAiValue, JsonlGenAiExporter,
};

// ── P7.3 — Trajectory recorder / replayer (G22) ─────────────────────────
pub use caduceus_eval::{
    RecordingLlmAdapter, ReplayingLlmAdapter, Trajectory, TrajectoryEntry, TrajectoryRecorder,
};

// ── P8.1 / P8.2 / P8.4 — Step verification & process-reward ─────────────
pub use caduceus_core::{
    EnsembleCombiner, EnsembleStepVerifier, ObservedToolCall, OffStepVerifier, StepScore,
    StepVerifier, StepView,
};
pub use caduceus_orchestrator::rollout_prm::RolloutPrmVerifier;

/// P3.1 — Apply a [`PlanAmendment`] to a [`ActionPlan`] from the
/// IPC layer. Accepts the JSON shape produced by the React panel and returns
/// either the new plan revision metadata or a typed error so the UI can
/// re‑render. Stale revisions are surfaced explicitly, never silently merged.
///
/// The caller (Tauri command in the IDE shell) is expected to:
/// 1. Hold the live `ActionPlan` (one per turn).
/// 2. Pass the user's edit as a `PlanAmendment` JSON payload.
/// 3. Forward the returned [`caduceus_core::AgentEvent::PlanAmended`] event
///    to the UI emitter.
pub fn apply_plan_amendment(
    plan: &mut ActionPlan,
    amendment_json: &str,
) -> Result<AppliedAmendment, AmendError> {
    let amendment: PlanAmendment =
        serde_json::from_str(amendment_json).map_err(|_| AmendError::StaleRevision {
            expected: 0,
            actual: plan.revision,
        })?;
    plan.apply_amendment(amendment)
}

/// P3.1 — Render a snapshot of the current [`ActionPlan`] as JSON for
/// the UI. The shape mirrors the per‑step revision so the React panel can
/// detect divergence on the next amendment attempt.
pub fn snapshot_plan_json(plan: &ActionPlan) -> Result<String, BridgeError> {
    serde_json::to_string(plan).map_err(Into::into)
}

// ── P3.3 — per-tool-batch checkpoint + 1-click revert ─────────────────────
//
// The IDE shell owns a [`CheckpointStore`] for each agent session.
// Tool wrappers must call [`begin_checkpoint`] before mutating files and
// [`record_checkpoint_edit`] for each pre-image. After the batch finishes
// (success or failure), the bridge calls [`commit_checkpoint`] to freeze the
// snapshot. The React side renders the timeline via [`list_checkpoints_json`]
// and triggers a one-click revert through [`revert_checkpoint`], which
// returns the pre-images so the host can write them back to disk.
//
// The store is purely an in-memory ring (default cap 64); no I/O happens
// inside this module — keeping the bridge filesystem-agnostic and trivial
// to test under both Tauri and the headless CLI runner.

/// Construct an empty checkpoint store with the orchestrator's default
/// capacity (64). Wrap in `Arc<Mutex<_>>` to share with the harness via
/// `AgentHarness::with_checkpoint_store`.
pub fn new_checkpoint_store() -> CheckpointStore {
    CheckpointStore::default()
}

/// Begin a new tool-batch checkpoint. `tool_summary` is rendered verbatim
/// in the UI timeline (e.g. `"edit_file: src/lib.rs"`). `now_secs` is wall‑
/// clock seconds since epoch; the bridge passes
/// `SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()`.
pub fn begin_checkpoint(
    store: &mut CheckpointStore,
    turn_index: u32,
    tool_summary: impl Into<String>,
    now_secs: u64,
) -> CheckpointId {
    store.begin_batch(turn_index, tool_summary.into(), now_secs)
}

/// Record a pre-image for `path`. Pass `before = None` if the file did
/// not exist before this batch (revert then deletes it). Returns
/// `Err(CheckpointError::Unknown)` if the id is stale (evicted or
/// invalid) and `Err(CheckpointError::AlreadyClosed)` if the batch
/// has already been committed or reverted.
pub fn record_checkpoint_edit(
    store: &mut CheckpointStore,
    id: CheckpointId,
    path: PathBuf,
    before: Option<String>,
) -> Result<(), CheckpointError> {
    store.record_edit(id, path, before)
}

/// Freeze a checkpoint after the tool batch finishes. Idempotent in the
/// sense that committing twice surfaces `AlreadyClosed` — the bridge
/// should treat that as success when the UI has not yet updated.
pub fn commit_checkpoint(
    store: &mut CheckpointStore,
    id: CheckpointId,
) -> Result<(), CheckpointError> {
    store.commit(id).map(|_| ())
}

/// One-click revert. Returns the recorded pre-images so the host can
/// apply them. Marks the batch `Reverted` on success; subsequent reverts
/// of the same id return `AlreadyClosed`. The harness, when wired,
/// emits `CheckpointReverted` so the timeline updates.
pub fn revert_checkpoint(
    store: &mut CheckpointStore,
    id: CheckpointId,
) -> Result<Vec<FileSnapshot>, CheckpointError> {
    store.revert(id)
}

/// Render the full checkpoint timeline as JSON for the React panel.
/// Order is newest-first (matches `CheckpointStore::list`); the panel
/// can render directly without reversing.
pub fn list_checkpoints_json(store: &CheckpointStore) -> Result<String, BridgeError> {
    let v: Vec<&ToolBatchCheckpoint> = store.list();
    serde_json::to_string(&v).map_err(Into::into)
}

// ── P3.4 — background notifications fabric ───────────────────────────────
//
// The IDE shell creates a single [`BroadcastBus`] per session and
// stores it next to the harness. Background workers (automations, cron,
// MCP reload) call [`publish_notification`] / [`publish_automation_completion`]
// when they finish; the bridge subscribes via [`subscribe_notifications`]
// and forwards each `Notification` to the React panel as a toast.
//
// All publishers are `&self` and lock-free on the hot path. A publish
// to a channel with zero subscribers is a silent drop, not an error.

/// Subscribe to the notifications channel. Always succeeds.
pub fn subscribe_notifications(bus: &BroadcastBus) -> tokio::sync::broadcast::Receiver<BusMessage> {
    bridge_notifications::subscribe(bus)
}

/// Publish a typed notification. Returns the number of receivers, or
/// `BusError::NoSubscribers` if nobody is listening.
pub fn publish_notification(bus: &BroadcastBus, n: Notification) -> Result<usize, BusError> {
    bridge_notifications::publish(bus, n)
}

/// Sugar: convert an [`AutomationResult`] to a notification and
/// publish it. Failed runs become `Severity::Error`.
pub fn publish_automation_completion(
    bus: &BroadcastBus,
    result: &AutomationResult,
) -> Result<usize, BusError> {
    bridge_notifications::publish_automation_completion(bus, result)
}

/// Render a [`BusMessage`] payload as a typed
/// [`Notification`]. Returns `Err` only if the message did not
/// originate from the notifications publisher (malformed JSON).
pub fn parse_notification(msg: &BusMessage) -> Result<Notification, BridgeError> {
    serde_json::from_str(&msg.content).map_err(Into::into)
}

// ── P4.1 — MemoryBlocks bridge ───────────────────────────────────────────
//
// The IDE shell creates one `MemoryBlocks` per session and
// shares it with the harness via `AgentHarness::with_memory_blocks`.
// The React panel renders the persona / project / working / archival
// blocks via `snapshot_memory_blocks_json`; tool wrappers append turns
// via `append_working_message`. Compaction is triggered explicitly by
// the orchestrator harness; `compact_memory_blocks` is exposed so the
// IDE can also trigger it on demand (e.g. "/compact" slash).

/// Construct an empty memory-blocks store with the supplied limits.
pub fn new_memory_blocks(limits: BlockLimits) -> MemoryBlocks {
    MemoryBlocks::new(limits)
}

/// Set the persona block. Returns the number of chars dropped due to
/// the cap (0 if under limit).
pub fn set_memory_persona(blocks: &mut MemoryBlocks, text: impl Into<String>) -> usize {
    blocks.set_persona(text)
}

/// Set the project-context block. Returns chars dropped.
pub fn set_memory_project_context(blocks: &mut MemoryBlocks, text: impl Into<String>) -> usize {
    blocks.set_project_context(text)
}

/// Append a single message to working history.
pub fn append_working_message(blocks: &mut MemoryBlocks, msg: WorkingMessage) {
    blocks.append_working(msg);
}

/// Trigger a compaction pass. Returns telemetry the React panel can
/// surface as a toast ("compacted N tool turns into 1 summary").
pub fn compact_memory_blocks(blocks: &mut MemoryBlocks) -> MemoryCompactionReport {
    blocks.compact()
}

/// Render the entire memory-blocks snapshot as JSON for the panel.
pub fn snapshot_memory_blocks_json(blocks: &MemoryBlocks) -> Result<String, BridgeError> {
    serde_json::to_string(blocks).map_err(Into::into)
}

// ── P4.2 — context_fold bridge ───────────────────────────────────────────
//
// Long tool transcripts (e.g. a 4k-token bash output) are folded into
// a one-line placeholder once they exceed
// `context_fold::DEFAULT_FOLD_THRESHOLD_CHARS`. The original is kept
// in a side-store and can be expanded back on demand via the React
// panel's "show full output" affordance.

/// Construct an empty transcript store (default capacity).
pub fn new_transcript_store() -> TranscriptStore {
    TranscriptStore::default()
}

/// Fold a transcript: returns the placeholder + a stable id the panel
/// uses to expand later. `subagent` and `outcome` are short labels
/// rendered in the placeholder bubble (e.g. `"bash"`, `"ok"`).
pub fn fold_transcript(
    store: &mut TranscriptStore,
    subagent: impl Into<String>,
    outcome: impl Into<String>,
    full: impl Into<String>,
) -> FoldedTranscript {
    store.fold(subagent, outcome, full.into())
}

/// Expand a previously-folded transcript by id. Returns
/// `ExpandError::Unknown` when the id has been evicted (FIFO)
/// and `Expired` when the entry is past TTL.
pub fn expand_transcript(store: &TranscriptStore, id: TranscriptId) -> Result<String, ExpandError> {
    store.expand(id).map(str::to_owned)
}

/// Number of folded transcripts currently retained.
pub fn folded_transcript_count(store: &TranscriptStore) -> usize {
    store.len()
}

// ── P4.4 — per-model token budget snapshot ───────────────────────────────
//
// The IDE renders a budget bar above the prompt input. Without this
// helper, the panel would have to wait for the first turn (which
// invokes `apply_model_budget_for_turn`) before it could display
// context limits. This wrapper exposes the static
// `TokenBudget::model_spec` table so the panel pre-renders correctly.

/// Returns `(context_limit_tokens, reserved_output_tokens)` for a
/// given model id. Falls back to the conservative defaults for
/// unknown models.
pub fn model_budget_spec(model_id: &str) -> (u32, u32) {
    TokenBudget::model_spec(model_id)
}

/// JSON shape: `{"model":"…","context_limit":…,"reserved_output":…}`.
pub fn model_budget_spec_json(model_id: &str) -> String {
    let (ctx, reserved) = model_budget_spec(model_id);
    format!(
        "{{\"model\":\"{}\",\"context_limit\":{},\"reserved_output\":{}}}",
        model_id.replace('"', "\\\""),
        ctx,
        reserved
    )
}

// ── P5.1 — CompactionTelemetry bridge ────────────────────────────────────
//
// The harness records a CompactionEvent every time a strategy fires.
// The IDE shell snapshots / drains the ring for offline training and
// for the at-a-glance dashboard. The bridge exposes a thin facade so
// the React panel can:
//   * push events from JS-driven dry runs (rare),
//   * mark the most recent event at a given turn as having caused a
//     downstream re-ask (rich label for the trainer),
//   * drain the ring as JSONL for export, and
//   * snapshot per-strategy stats for the dashboard.

/// Construct an empty telemetry ring with the orchestrator default
/// capacity (1024 events ≈ weeks of typical activity).
pub fn new_compaction_telemetry() -> CompactionTelemetry {
    CompactionTelemetry::default()
}

/// Append an event to the ring. Oldest event is evicted FIFO when
/// the ring is full.
pub fn record_compaction_event(tel: &mut CompactionTelemetry, ev: CompactionEvent) {
    tel.record(ev);
}

/// Mark the most-recent compaction at `turn_index` as having caused
/// (or not) a downstream re-ask. Returns `true` if a matching event
/// was found and updated. Used by the next-turn re-ask detector.
pub fn mark_compaction_re_ask(
    tel: &mut CompactionTelemetry,
    turn_index: u32,
    re_asked: bool,
) -> bool {
    tel.mark_re_ask(turn_index, re_asked)
}

/// Snapshot the ring (newest-last) as JSONL for export. Does not
/// clear the ring.
pub fn compaction_telemetry_jsonl(tel: &CompactionTelemetry) -> String {
    tel.to_jsonl()
}

/// Drain all events as JSONL AND clear the ring. Use when the IDE
/// pushes a batch to disk after a session ends.
pub fn drain_compaction_telemetry_jsonl(tel: &mut CompactionTelemetry) -> String {
    tel.drain_jsonl()
}

/// Per-strategy aggregates as JSON for the dashboard. Each row:
/// `{"strategy":"…","count":N,"mean_tokens_dropped":F,"re_ask_rate":F|null}`.
pub fn compaction_telemetry_stats_json(tel: &CompactionTelemetry) -> String {
    let rows = tel.per_strategy_stats();
    let json: Vec<serde_json::Value> = rows
        .into_iter()
        .map(|(s, count, mean, rate)| {
            serde_json::json!({
                "strategy": s,
                "count": count,
                "mean_tokens_dropped": mean,
                "re_ask_rate": rate,
            })
        })
        .collect();
    serde_json::to_string(&json).unwrap_or_else(|_| "[]".into())
}

// ── P5.2 — Bradley–Terry scorer bridge ───────────────────────────────────
//
// Once telemetry has accumulated enough labelled events, the IDE
// triggers a fit pass either in-process (small datasets) or out-of-
// process via the JSONL export. Both paths are exposed.

/// Fit a Bradley–Terry model directly from a slice of telemetry
/// events. Convenience for in-process training when the dataset is
/// small (a few hundred events).
pub fn fit_bradley_terry(events: &[CompactionEvent]) -> BradleyTerryModel {
    let pairs = bridge_compaction_scorer::pairs_from_events(events);
    bridge_compaction_scorer::fit(&pairs)
}

/// Train a Bradley–Terry model from JSONL exported by
/// [`drain_compaction_telemetry_jsonl`]. Returns a fully-fit model
/// (default-empty if the JSONL is malformed or empty).
pub fn train_bradley_terry_from_jsonl(jsonl: &str) -> BradleyTerryModel {
    bridge_compaction_scorer::train_from_jsonl(jsonl)
}

/// Snapshot a trained model as JSON for persistence under
/// `.caduceus/models/compaction.json`.
pub fn snapshot_bradley_terry_json(model: &BradleyTerryModel) -> Result<String, BridgeError> {
    serde_json::to_string(model).map_err(Into::into)
}

/// Restore a model from a JSON blob. Errors surface as `Err(String)`
/// so the IDE can fall back to the heuristic selector.
pub fn load_bradley_terry_json(json: &str) -> Result<BradleyTerryModel, BridgeError> {
    serde_json::from_str(json).map_err(Into::into)
}

// ── P5.3 — Learned compaction-strategy selector bridge ──────────────────
//
// The selector wraps a trained BradleyTerryModel and re-orders
// candidate strategies. The IDE owns one selector per session,
// reloaded from disk on startup.

/// Construct a learned selector from a trained model. Defaults to
/// `Auto` mode (use the model only when at least 2 strategies have
/// been observed, otherwise fall back to the heuristic order).
pub fn new_learned_selector(model: BradleyTerryModel) -> LearnedSelector {
    LearnedSelector::new(model)
}

/// Override the selection mode (Heuristic / Learned / Auto). Used
/// by the IDE settings panel.
pub fn set_selector_mode(selector: LearnedSelector, mode: SelectionMode) -> LearnedSelector {
    selector.with_mode(mode)
}

/// Re-order `candidates` from best to worst per the selector. Equal
/// scores preserve input order (deterministic).
pub fn rank_candidates<'a>(selector: &LearnedSelector, candidates: &[&'a str]) -> Vec<&'a str> {
    selector.rank(candidates)
}

/// Pick the top candidate plus the score margin over the runner-up.
/// `margin` is 0.0 when only one candidate is given. The IDE can show
/// an "uncertain" badge when the margin is small (< 0.1).
pub fn select_with_confidence<'a>(
    selector: &LearnedSelector,
    candidates: &[&'a str],
) -> Option<(&'a str, f64)> {
    selector.select_with_confidence(candidates)
}

/// Exact token count using tiktoken (cl100k_base, used by GPT-4/Claude).
/// Falls back to the heuristic estimate if tokenizer initialization fails.
pub fn count_tokens_exact(text: &str) -> u32 {
    use std::sync::OnceLock;
    static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();

    let bpe = BPE.get_or_init(|| tiktoken_rs::get_bpe_from_model("gpt-4o").ok());

    match bpe {
        Some(encoder) => encoder.encode_ordinary(text).len() as u32,
        None => estimate_tokens(text), // fallback to heuristic
    }
}

/// Wrapper around the AgentHarness for the bridge.
pub struct OrchestratorBridge {
    project_root: PathBuf,
    /// Optional speculative tool-result cache (P12.2). When set, every
    /// harness produced by `build_harness*` is wired with a clone via
    /// `AgentHarness::with_speculative_cache`, enabling tool-call hits
    /// to short-circuit the spawn loop.
    speculative_cache: Option<caduceus_tools::SpeculativeCache>,
    /// Optional per-project Reflexion memory (P12.4). Wrapped in
    /// `Arc<Mutex<…>>` so the bridge, harness, and IPC handlers all
    /// share the same instance.
    reflexion: Option<Arc<std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>>>,
    /// Optional Tree-of-Thoughts planner config (P12.3) attached to
    /// every harness.
    tot_config: Option<caduceus_orchestrator::PlannerConfig>,
    /// ST-B4a / `native-loop-prep-v1` — runtime flag that gates the
    /// engine-driven turn loop (`run_caduceus_loop`). Off by default.
    /// Stored as `Arc<AtomicBool>` so callers can clone a cheap handle
    /// once and mutate the flag from settings without re-wiring the
    /// bridge. `Relaxed`/`Acquire` ordering is sufficient — the flag
    /// only gates entry into the loop; inside the loop, data flow is
    /// coordinated by the harness's own synchronisation primitives.
    native_loop_enabled: Arc<std::sync::atomic::AtomicBool>,
}

/// Cheap, `Clone`-able handle to an `AgentEventEmitter`'s retention ring.
///
/// The IDE stores one of these per session and calls
/// [`OrchestratorBridge::replay_session_events`] from the agent panel on
/// (re)mount to rebuild the event timeline without losing events emitted
/// while the UI was disconnected (gap G17). The handle shares the same
/// `Arc`-backed ring as the emitter held by the harness, so it is always
/// in sync with the live channel.
#[derive(Clone)]
pub struct ReplayHandle {
    emitter: AgentEventEmitter,
}

impl ReplayHandle {
    fn new(emitter: AgentEventEmitter) -> Self {
        Self { emitter }
    }

    /// Snapshot the retention ring (oldest-first). See
    /// [`AgentEventEmitter::replay`] for semantics.
    pub fn replay(&self) -> Vec<AgentEvent> {
        self.emitter.replay()
    }

    /// Configured retention capacity of the underlying emitter ring.
    pub fn retention_cap(&self) -> usize {
        self.emitter.retention_cap()
    }

    /// P6.2 / G27 — Number of events dropped from the live mpsc channel
    /// since the last successful emit. Surfaced for diagnostics; the UI
    /// timeline observes the same drop via the synthetic
    /// `EventBufferOverflow { dropped_since_last }` event injected on the
    /// next successful emit.
    pub fn dropped_since_last(&self) -> u64 {
        self.emitter.dropped_since_last()
    }
}

// ── P6.7 — MCP error kind tagging ───────────────────────────────────────
pub use caduceus_mcp::error::McpErrorKind;

/// P6.7 — Stable string label for an [`McpErrorKind`]. The IDE uses
/// this to drive icon/colour selection on MCP failures (transient vs
/// permanent vs auth vs config vs not_found vs permission) without
/// coupling to the Rust enum repr.
pub fn mcp_error_kind_label(kind: McpErrorKind) -> &'static str {
    kind.label()
}

/// P6.7 — Convenience predicate mirroring [`McpErrorKind::is_retryable`]
/// so the IPC layer can short-circuit retry orchestration without
/// importing the `caduceus-mcp` crate directly.
pub fn mcp_error_kind_is_retryable(kind: McpErrorKind) -> bool {
    kind.is_retryable()
}

// ── P6.8 — SharedContext write-collision audit (G30) ────────────────────
pub use caduceus_orchestrator::workers::ContextWriteRecord;

/// P6.8 / G30 — Build a `SharedContext` with the per-write audit trail
/// enabled. The IDE uses the resulting `writes()`/`collisions()` snapshots
/// to render a "branch contention" indicator in the multi-agent panel.
pub fn new_shared_context_with_audit() -> SharedContext {
    SharedContext::new().with_write_audit()
}

/// JSON snapshot of every recorded write (oldest-first). Each entry is
/// `{ "branch": "<…>", "key": "<…>", "overwrote": <bool>, "seq": <u64> }`.
/// Cheap; safe to call from any task — internally takes the read-half of
/// the inner `RwLock`.
pub async fn recorded_writes_json(ctx: &SharedContext) -> String {
    serialise_writes(&ctx.writes().await)
}

/// JSON snapshot restricted to writes that overwrote a previous value
/// (collisions). Empty array when no contention has occurred.
pub async fn recorded_collisions_json(ctx: &SharedContext) -> String {
    serialise_writes(&ctx.collisions().await)
}

fn serialise_writes(writes: &[ContextWriteRecord]) -> String {
    let entries: Vec<serde_json::Value> = writes
        .iter()
        .map(|w| {
            serde_json::json!({
                "branch": w.key.branch,
                "key": w.key.key,
                "overwrote": w.overwrote,
                "seq": w.seq,
            })
        })
        .collect();
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Convenience predicate used by the UI to flip a "contention" badge
/// without serialising the full audit list.
pub async fn shared_context_had_collisions(ctx: &SharedContext) -> bool {
    ctx.had_collisions().await
}

// ── P6.9 — Schema-versioned AgentEvent envelope (G33) ───────────────────
pub use caduceus_core::{
    AGENT_EVENT_SCHEMA_VERSION as BRIDGE_AGENT_EVENT_SCHEMA_VERSION, VersionedAgentEvent,
};

/// P6.9 / G33 — Wrap an `AgentEvent` in the current schema envelope and
/// serialise to JSON. The IDE persists this form to the on-disk session
/// log so a future build (newer or older schema) can decode without losing
/// the original payload — unknown variants fall back to `AgentEvent::Unknown`.
pub fn wrap_event_json(event: &AgentEvent) -> Result<String, BridgeError> {
    let envelope = VersionedAgentEvent::current(event.clone());
    serde_json::to_string(&envelope).map_err(Into::into)
}

/// Decode a versioned envelope. Returns the inner event together with a
/// flag set when the producer's schema version is newer than this build —
/// the UI can then surface a "client out of date" hint.
pub fn parse_versioned_event_json(s: &str) -> Result<(AgentEvent, bool), BridgeError> {
    let envelope: VersionedAgentEvent = serde_json::from_str(s)?;
    let from_newer = envelope.is_from_newer_producer();
    Ok((envelope.event, from_newer))
}

// ── F4 / G1c — Envelope preflight exposed to thread-side code ────────────
pub use caduceus_orchestrator::{PreflightOutcome, preflight_envelope_of};

/// Thread-side envelope preflight. Thin wrapper around
/// [`preflight_envelope_of`] so Zed's tool-dispatch path can block or
/// intercept tool calls BEFORE invoking the harness (legacy path) and
/// without duplicating the capability-classification logic that lives
/// in the orchestrator. Returns [`PreflightOutcome::Allow`] when no
/// envelope is attached — callers then proceed to normal dispatch.
pub fn check_tool_envelope(
    envelope: Option<&PermissionEnvelope>,
    tool_name: &str,
    input: &serde_json::Value,
) -> PreflightOutcome {
    match envelope {
        None => PreflightOutcome::Allow,
        Some(env) => preflight_envelope_of(env, tool_name, input),
    }
}

impl OrchestratorBridge {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            speculative_cache: None,
            reflexion: None,
            tot_config: None,
            native_loop_enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    // ── P12 primitive wiring (production side) ───────────────────────────

    /// Mint a fresh [`ReducerHandle`] for a new session. Cheap; the handle
    /// owns a private [`SessionStateReducer`] behind an `Arc<Mutex<…>>` so
    /// every collaborator (critique fan-out driver, IPC replay loop, UI
    /// remount path) sees the same reducer state. Pass the handle into
    /// `spawn_critique_fanout_with_introspection` via `handle.as_sink()`
    /// and into the session event replay path via `handle.ingest_event`
    /// / `handle.ingest_many` — one reducer per session, three filtered
    /// projections out.
    pub fn new_reducer_handle(&self) -> crate::dag_state::ReducerHandle {
        crate::dag_state::ReducerHandle::new()
    }

    /// Attach a [`SpeculativeCache`](caduceus_tools::SpeculativeCache) to
    /// every subsequent harness produced by `build_harness*`. Cheaply
    /// cloneable; the same `Arc`-backed map is shared by the bridge,
    /// harness, and any external prefetcher (UI worker) that calls
    /// [`OrchestratorBridge::speculative_cache`].
    pub fn with_speculative_cache(mut self, cache: caduceus_tools::SpeculativeCache) -> Self {
        self.speculative_cache = Some(cache);
        self
    }

    /// Convenience constructor — installs a default-TTL cache (60 s).
    pub fn with_default_speculative_cache(self) -> Self {
        self.with_speculative_cache(caduceus_tools::SpeculativeCache::new(
            std::time::Duration::from_secs(60),
        ))
    }

    /// Borrow the speculative cache so an external prefetcher (e.g. a
    /// file-watcher worker that predicts the next `read_file` call) can
    /// `reserve` / `complete` slots.
    pub fn speculative_cache(&self) -> Option<&caduceus_tools::SpeculativeCache> {
        self.speculative_cache.as_ref()
    }

    /// Attach a Reflexion memory store. Wrapped in `Arc<Mutex<…>>` so
    /// the bridge can both feed the harness AND expose `record_outcome`
    /// to the IPC layer.
    pub fn with_reflexion(
        mut self,
        memory: Arc<std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>>,
    ) -> Self {
        self.reflexion = Some(memory);
        self
    }

    /// Convenience constructor — 16-slot, no TTL.
    pub fn with_default_reflexion(self) -> Self {
        let mem = caduceus_orchestrator::reflexion::ReflexionMemory::new(16);
        self.with_reflexion(Arc::new(std::sync::Mutex::new(mem)))
    }

    /// Borrow the reflexion handle so the IPC layer can push attempt
    /// outcomes from the agent panel after each turn completes.
    pub fn reflexion(
        &self,
    ) -> Option<&Arc<std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>>> {
        self.reflexion.as_ref()
    }

    /// Record an attempt outcome into the bound reflexion memory using
    /// the supplied `Reflector`. Returns the reflection that was
    /// stored, or `None` if no memory is bound or the reflector
    /// produced nothing.
    pub fn record_attempt_outcome<R>(
        &self,
        reflector: &R,
        task_tag: &str,
        outcome: &caduceus_orchestrator::reflexion::AttemptOutcome,
    ) -> Option<caduceus_orchestrator::reflexion::Reflection>
    where
        R: caduceus_orchestrator::reflexion::Reflector,
    {
        let mem = self.reflexion.as_ref()?;
        let mut guard = mem.lock().ok()?;
        guard.record_outcome(reflector, task_tag, outcome)
    }

    /// Set the default Tree-of-Thoughts planner config used for any
    /// future `plan_with_tot` calls on harnesses produced by this
    /// bridge.
    pub fn with_tot_config(mut self, cfg: caduceus_orchestrator::PlannerConfig) -> Self {
        self.tot_config = Some(cfg);
        self
    }

    /// Convenience: install the planner config defaults
    /// (`PlannerConfig::default`).
    pub fn with_default_tot_config(self) -> Self {
        self.with_tot_config(caduceus_orchestrator::PlannerConfig::default())
    }

    /// Borrow the bound ToT planner config (if any).
    pub fn tot_config(&self) -> Option<&caduceus_orchestrator::PlannerConfig> {
        self.tot_config.as_ref()
    }

    /// Internal helper — chains every bound P12 primitive onto a freshly
    /// built `AgentHarness`. Centralised so the four `build_harness*`
    /// factories don't drift.
    fn attach_p12(&self, mut h: AgentHarness) -> AgentHarness {
        if let Some(cache) = self.speculative_cache.clone() {
            h = h.with_speculative_cache(cache);
        }
        if let Some(mem) = self.reflexion.clone() {
            h = h.with_reflexion(mem);
        }
        if let Some(cfg) = self.tot_config {
            h = h.with_tot_config(cfg);
        }
        h
    }

    // ── MCP → Tool registration (P11.4 production wiring) ────────────────

    /// Walk every tool advertised by `manager`, wrap each in an
    /// [`McpToolBridge`] whose invoker forwards to
    /// [`McpServerManager::call_tool`], and register the bridge in
    /// `registry`. Returns the number of tools that were registered.
    ///
    /// The invoker captures `Arc<McpServerManager>` so the bridge stays
    /// valid even after the registration call returns.
    pub async fn register_mcp_tools(
        manager: Arc<McpServerManager>,
        registry: &mut ToolRegistry,
    ) -> usize {
        let defs = manager.all_tools().await;
        let mut count = 0usize;
        for def in defs {
            let manager_for_invoker = manager.clone();
            let tool_name = def.name.clone();
            let invoker: McpInvoker = Arc::new(move |input| {
                let mgr = manager_for_invoker.clone();
                let name = tool_name.clone();
                Box::pin(async move {
                    match mgr.call_tool(&name, input).await {
                        Ok(value) => Ok(caduceus_core::ToolResult::success(value.to_string())),
                        Err(e) => Ok(caduceus_core::ToolResult::error(format!(
                            "mcp call '{name}' failed: {e}"
                        ))),
                    }
                })
            });
            let bridge = McpToolBridge::new(&def, invoker);
            registry.register(Arc::new(bridge));
            count += 1;
        }
        count
    }

    /// P13.7 (G‑R3.1) — hot‑reload variant of [`register_mcp_tools`].
    ///
    /// Re‑snapshots `manager.all_tools()` and computes the set
    /// difference against the names currently registered in
    /// `registry` that look like MCP tools (i.e. were last produced
    /// by this very function or `register_mcp_tools`). Adds any tools
    /// the manager now exposes that aren't in the registry; removes
    /// any tool from the registry whose name no longer appears in the
    /// manager surface; leaves stable tools untouched.
    ///
    /// The function is idempotent — calling it twice with no manager
    /// changes returns `(0, 0)`. Returns `(added, removed)` so callers
    /// can emit a single hot‑reload event.
    ///
    /// This is the engine‑side half of MCP hot‑reload; the manager
    /// produces drift events via [`McpServerManager::drift_for`]; the
    /// bridge consumes them by re‑running this diff. Tools that
    /// MUTATED (same name, new schema) are NOT detected here — the
    /// manager's `apply_tool_refresh` already replaces the cached
    /// `McpToolDef`, so the next *invocation* will dispatch through
    /// the new schema. Catching mutation here would require versioned
    /// schemas in `Tool::spec()`, which we don't have yet.
    pub async fn register_mcp_tools_diff(
        manager: Arc<McpServerManager>,
        registry: &mut ToolRegistry,
        prior_mcp_names: &std::collections::HashSet<String>,
    ) -> (usize, usize, std::collections::HashSet<String>) {
        let defs = manager.all_tools().await;
        let live: std::collections::HashSet<String> = defs.iter().map(|d| d.name.clone()).collect();

        // Removals first so a name reused by a different server is
        // re‑bound cleanly.
        let mut removed = 0usize;
        for stale in prior_mcp_names.difference(&live) {
            if registry.remove(stale) {
                removed += 1;
            }
        }

        // Additions: only register tools we don't already have.
        let mut added = 0usize;
        for def in defs {
            if prior_mcp_names.contains(&def.name) && registry.get(&def.name).is_some() {
                continue;
            }
            let manager_for_invoker = manager.clone();
            let tool_name = def.name.clone();
            let invoker: McpInvoker = Arc::new(move |input| {
                let mgr = manager_for_invoker.clone();
                let name = tool_name.clone();
                Box::pin(async move {
                    match mgr.call_tool(&name, input).await {
                        Ok(value) => Ok(caduceus_core::ToolResult::success(value.to_string())),
                        Err(e) => Ok(caduceus_core::ToolResult::error(format!(
                            "mcp call '{name}' failed: {e}"
                        ))),
                    }
                })
            });
            let bridge = McpToolBridge::new(&def, invoker);
            registry.register(Arc::new(bridge));
            added += 1;
        }

        (added, removed, live)
    }

    /// Load workspace instructions from .caduceus/ hierarchy.
    pub fn load_instructions(&self) -> Result<InstructionSet, BridgeError> {
        let loader = InstructionLoader::new(&self.project_root);
        loader
            .load()
            .map_err(|e| BridgeError::workspace(e.to_string()))
    }

    /// Create a new conversation history.
    pub fn new_history() -> ConversationHistory {
        ConversationHistory::new()
    }

    /// Create a new session state.
    pub fn new_session(&self, provider: &str, model: &str) -> SessionState {
        SessionState::new(
            &self.project_root,
            ProviderId::new(provider),
            ModelId::new(model),
        )
    }

    /// Extract learnable memories from a conversation exchange.
    pub fn extract_memories(user_input: &str, assistant_response: &str) -> Vec<String> {
        caduceus_orchestrator::extract_memories(user_input, assistant_response)
    }

    // ── Parallel tool execution ──────────────────────────────────────────

    /// Execute multiple tool calls in parallel.
    pub async fn execute_tool_calls(
        registry: &ToolRegistry,
        calls: &[(String, String, serde_json::Value)],
    ) -> Vec<(String, String, bool)> {
        execute_tool_calls(registry, calls).await
    }

    // ── Conversation history persistence ─────────────────────────────────

    /// Serialize conversation history to JSON.
    pub fn conversation_serialize(history: &ConversationHistory) -> Result<String, BridgeError> {
        history
            .serialize()
            .map_err(|e| BridgeError::workspace(e.to_string()))
    }

    /// Deserialize conversation history from JSON.
    pub fn conversation_deserialize(json: &str) -> Result<ConversationHistory, BridgeError> {
        ConversationHistory::deserialize(json).map_err(|e| BridgeError::workspace(e.to_string()))
    }

    /// Truncate conversation history, keeping at most `max` messages.
    pub fn conversation_truncate(history: &mut ConversationHistory, max: usize) {
        history.truncate_oldest(max);
    }

    /// Clear all messages from conversation history.
    pub fn conversation_clear(history: &mut ConversationHistory) {
        history.clear();
    }

    // ── Instruction compaction ───────────────────────────────────────────

    /// Compact instructions to fit within a character budget.
    pub fn compact_instructions(content: &str, max_chars: usize) -> String {
        instructions::compact_instructions(content, max_chars)
    }

    // NOTE: semantic_match_score is private in caduceus-orchestrator::instructions — skipped.

    /// Default set of tools that require human-in-the-loop approval before
    /// execution. These are the tools whose effects are observable outside the
    /// process (filesystem mutations, shell execution, network writes) and so
    /// must not run silently. Override per-call via [`build_harness_with_approval`].
    pub const DEFAULT_APPROVAL_TOOLS: &'static [&'static str] = &[
        "bash",
        "shell",
        "write_file",
        "edit_file",
        "delete_file",
        "create_file",
        "apply_patch",
    ];

    // ── Fluent harness builder (ST-F2) ────────────────────────────────
    //
    // Contract `harness-builder-v2`. Single entry point that replaces
    // the 9 `build_harness_*` variants. Old variants kept as thin
    // `#[deprecated]` wrappers so callers migrate gradually without
    // breaking tests.

    /// Start a new [`HarnessBuilder`] against this bridge.
    ///
    /// Default configuration matches the pre-ST-F2 `build_harness()`
    /// shape: HITL approval on [`DEFAULT_APPROVAL_TOOLS`], no envelope,
    /// no introspection sink, no event emitter, no context injector,
    /// 200k context window, 120s tool timeout.
    pub fn harness(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: impl Into<String>,
    ) -> HarnessBuilder<'_> {
        HarnessBuilder {
            bridge: self,
            provider,
            tools,
            system_prompt: system_prompt.into(),
            approval: ApprovalChoice::Default,
            envelope: None,
            sink: SinkChoice::None,
            emitter: EmitterChoice::None,
            injector: InjectorChoice::None,
            max_context_tokens: 200_000,
            tool_timeout: std::time::Duration::from_secs(120),
            resume_on_grant: false,
            grant_timeout: None,
        }
    }

    /// Build a full agent harness with tools, instructions, and HITL approval
    /// wired for the [`DEFAULT_APPROVAL_TOOLS`] set.
    ///
    /// Returns `(harness, approval_tx)`. Callers **must** route `approval_tx`
    /// to the IDE permission-prompt UI: when the user approves or denies a
    /// `PermissionRequest` event, send `(format!("perm_{tool_use_id}"), approved)`
    /// on this channel. Dropping the sender without responding fails-fast as
    /// `PermissionOutcome::ChannelClosed` rather than the 300s timeout.
    ///
    /// This is the production entry point — it must be used in place of
    /// [`build_harness_no_approval`] for any flow that touches user data.
    pub fn build_harness(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> (AgentHarness, tokio::sync::mpsc::Sender<(String, bool)>) {
        let built = self.harness(provider, tools, system_prompt).build();
        (built.harness, built.approval_tx.expect("default approval"))
    }

    /// Build a harness with a caller-supplied approval set. Pass an empty
    /// slice for an autopilot harness (escape hatch for tests / headless
    /// agents); prefer [`build_harness`] in production.
    pub fn build_harness_with_approval<S: AsRef<str>>(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
        approval_tools: &[S],
    ) -> (AgentHarness, tokio::sync::mpsc::Sender<(String, bool)>) {
        let built = self
            .harness(provider, tools, system_prompt)
            .approval_tools(approval_tools.iter().map(|s| s.as_ref().to_string()))
            .build();
        (built.harness, built.approval_tx.expect("approval set"))
    }

    /// Build a harness with HITL approval disabled. Reserved for non-interactive
    /// surfaces (headless mode, scripted tests). Production UI flows must use
    /// [`build_harness`].
    pub fn build_harness_no_approval(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> AgentHarness {
        let built = self
            .harness(provider, tools, system_prompt)
            .no_approval()
            .build();
        built.harness
    }

    /// ST-B1 / contract `harness-sink-v1` — build a harness pre-wired with
    /// a `ReducerHandle` as its [`IntrospectionSink`], plus HITL approval
    /// on the [`DEFAULT_APPROVAL_TOOLS`] set.
    ///
    /// Returns `(harness, approval_tx, reducer_handle)`.
    pub fn build_harness_with_sink(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> (
        AgentHarness,
        tokio::sync::mpsc::Sender<(String, bool)>,
        crate::dag_state::ReducerHandle,
    ) {
        let built = self
            .harness(provider, tools, system_prompt)
            .with_reducer_sink()
            .build();
        (
            built.harness,
            built.approval_tx.expect("default approval"),
            built.reducer_handle.expect("sink requested"),
        )
    }

    /// ST-B2 / contract `envelope-surface-v1` — build a harness pre-wired
    /// with a caller-supplied [`PermissionEnvelope`] (plus HITL
    /// approval on the default tool set).
    pub fn build_harness_with_envelope(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
        envelope: PermissionEnvelope,
    ) -> (AgentHarness, tokio::sync::mpsc::Sender<(String, bool)>) {
        let built = self
            .harness(provider, tools, system_prompt)
            .envelope(envelope)
            .build();
        (built.harness, built.approval_tx.expect("default approval"))
    }

    /// ST-B2 / contract `envelope-surface-v1` combined with `harness-sink-v1`.
    /// Returns `(harness, approval_tx, reducer_handle)`.
    pub fn build_harness_with_envelope_and_sink(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
        envelope: PermissionEnvelope,
    ) -> (
        AgentHarness,
        tokio::sync::mpsc::Sender<(String, bool)>,
        crate::dag_state::ReducerHandle,
    ) {
        let built = self
            .harness(provider, tools, system_prompt)
            .envelope(envelope)
            .with_reducer_sink()
            .build();
        (
            built.harness,
            built.approval_tx.expect("default approval"),
            built.reducer_handle.expect("sink requested"),
        )
    }

    /// ST-B3 / contract `context-injector-v1` — harness pre-wired with envelope,
    /// introspection sink, and a (possibly default) [`ContextInjector`].
    /// Returns `(harness, approval_tx, reducer_handle)`.
    pub fn build_harness_with_envelope_sink_and_injector(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
        envelope: PermissionEnvelope,
        injector: Option<Arc<dyn ContextInjector>>,
    ) -> (
        AgentHarness,
        tokio::sync::mpsc::Sender<(String, bool)>,
        crate::dag_state::ReducerHandle,
    ) {
        let inj = injector.unwrap_or_else(|| {
            Arc::new(BuiltinScopedContextInjector::default()) as Arc<dyn ContextInjector>
        });
        let built = self
            .harness(provider, tools, system_prompt)
            .envelope(envelope)
            .with_reducer_sink()
            .injector(inj)
            .build();
        (
            built.harness,
            built.approval_tx.expect("default approval"),
            built.reducer_handle.expect("sink requested"),
        )
    }

    // ── Emitter / replay seam (G17) ──────────────────────────────────────

    /// Default in-flight buffer for the emitter mpsc channel handed to the
    /// IDE. Sized to absorb a brief render stall (~1 second of text deltas
    /// at 256 events/s) without dropping live events; events that don't
    /// fit are still captured by the emitter's retention ring.
    pub const DEFAULT_EVENT_CHANNEL_BUFFER: usize = 256;

    /// Build a production harness wired with HITL approval AND an event
    /// emitter that the IDE can stream from (`event_rx`) and *replay* from
    /// (`replay_handle`) on UI reattach (gap G17).
    ///
    /// Returns `(harness, approval_tx, replay_handle, event_rx)`. The
    /// caller MUST consume `event_rx`.
    pub fn build_harness_with_emitter(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> (
        AgentHarness,
        tokio::sync::mpsc::Sender<(String, bool)>,
        ReplayHandle,
        tokio::sync::mpsc::Receiver<AgentEvent>,
    ) {
        let built = self
            .harness(provider, tools, system_prompt)
            .with_emitter()
            .build();
        (
            built.harness,
            built.approval_tx.expect("default approval"),
            built.replay_handle.expect("emitter requested"),
            built.event_rx.expect("emitter requested"),
        )
    }

    /// G1h / `native-loop-wire-v1` — the full Phase-G IDE surface.
    /// Combines the introspection sink (for the reducer / Agents-DAG
    /// projection) with an `AgentEventEmitter` channel (for the
    /// translator → ACP consumption) and HITL approval on the default
    /// tool set. This is the exact harness shape that
    /// [`Self::run_caduceus_loop_translated`] expects.
    ///
    /// Returns `(harness, approval_tx, reducer_handle, event_rx)`.
    pub fn build_harness_with_sink_and_emitter(
        &self,
        provider: Arc<dyn LlmAdapter>,
        tools: ToolRegistry,
        system_prompt: &str,
    ) -> (
        AgentHarness,
        tokio::sync::mpsc::Sender<(String, bool)>,
        crate::dag_state::ReducerHandle,
        tokio::sync::mpsc::Receiver<AgentEvent>,
    ) {
        let built = self
            .harness(provider, tools, system_prompt)
            .with_reducer_sink()
            .with_emitter()
            .build();
        (
            built.harness,
            built.approval_tx.expect("default approval"),
            built.reducer_handle.expect("sink requested"),
            built.event_rx.expect("emitter requested"),
        )
    }

    /// Snapshot the retention ring on a given replay handle. This is the
    /// IPC entry point the Zed agent panel calls when it (re)mounts so it
    /// can rebuild the timeline without losing events that were emitted
    /// while the UI was offline (gap G17).
    ///
    /// Cheap (one mutex acquire + clone of the bounded ring); safe to call
    /// from any task and across awaits.
    pub fn replay_session_events(handle: &ReplayHandle) -> Vec<AgentEvent> {
        handle.replay()
    }

    /// Get the project root.
    pub fn project_root(&self) -> &Path {
        &self.project_root
    }

    // ── Agent turn execution ─────────────────────────────────────────────

    /// ST-B4a / `native-loop-prep-v1` — is the engine-driven turn loop
    /// enabled? The IDE reads this on every turn dispatch to decide
    /// whether to call [`Self::run_caduceus_loop`] or fall back to its
    /// own `model.stream_completion` loop. Default: `false`.
    pub fn native_loop_enabled(&self) -> bool {
        self.native_loop_enabled
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Toggle the native-loop flag at runtime. Intended to be wired to
    /// a workspace setting (`caduceus.native_loop`) so operators can
    /// bake the new path on a per-project basis without recompiling.
    pub fn set_native_loop_enabled(&self, enabled: bool) {
        self.native_loop_enabled
            .store(enabled, std::sync::atomic::Ordering::Release);
    }

    /// Builder variant of [`Self::set_native_loop_enabled`] for
    /// fluent test / bootstrap construction.
    pub fn with_native_loop_enabled(self, enabled: bool) -> Self {
        self.set_native_loop_enabled(enabled);
        self
    }

    /// Cheap `Clone`-able handle to the shared native-loop flag. Lets a
    /// background worker (e.g. settings watcher) flip the bit without
    /// taking a reference to the whole `OrchestratorBridge`.
    pub fn native_loop_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.native_loop_enabled.clone()
    }

    /// ST-B4a / `native-loop-prep-v1` — engine-driven turn loop.
    ///
    /// Drives [`AgentHarness::run`] on behalf of the IDE and forwards
    /// every `AgentEvent` emitted during the turn into a session
    /// [`ReducerHandle`](crate::dag_state::ReducerHandle) so the IDE's
    /// feature / agent / session projections stay live without the IDE
    /// having to translate `AgentEvent`s to ACP shapes itself.
    ///
    /// **Scope (explicit):** this is the *engine side* of the native
    /// loop. The downstream IDE integration (ST-B4b) — making
    /// `thread.rs` delegate to this function under the
    /// `caduceus.native_loop` setting and translating the streamed
    /// events to the IDE's `ThreadEventStream` / ACP surface — is
    /// deferred to a follow-up PR because it requires a full
    /// `AgentEvent` → ACP translator plus tool-dispatch redirection
    /// away from Zed's `ToolRegistry`. Until then, `thread.rs` still
    /// owns the user-visible loop and this function is reachable only
    /// via tests and forthcoming IDE glue.
    ///
    /// Contract:
    /// * Requires the native-loop flag to be on — returns an error
    ///   otherwise. Callers must opt in explicitly so a mis-wired flag
    ///   cannot silently change behaviour.
    /// * `event_rx` MUST be the receiver half of the same
    ///   [`AgentEventEmitter`] that was installed on `harness` via
    ///   `AgentHarness::with_emitter`. If the two are mismatched the
    ///   reducer will simply see no events — the harness never writes
    ///   to a channel it does not own.
    /// * Events emitted *before* the forwarder task is spawned are
    ///   captured by the emitter's retention ring (P1.4 / G17) and can
    ///   still be replayed via [`Self::replay_session_events`], so a
    ///   slow task spawn does not lose state.
    /// * Events emitted *after* `harness.run` returns but before the
    ///   forwarder drains are still delivered — `run_caduceus_loop`
    ///   waits for the forwarder to observe the channel-close or drain
    ///   signal before returning.
    /// * Cancellation and error propagation match
    ///   [`AgentHarness::run`] verbatim — this function adds no new
    ///   exit conditions beyond the engine's own.
    pub async fn run_caduceus_loop(
        &self,
        harness: &AgentHarness,
        state: &mut SessionState,
        history: &mut ConversationHistory,
        user_input: &str,
        reducer: crate::dag_state::ReducerHandle,
        event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    ) -> Result<String, BridgeError> {
        if !self.native_loop_enabled() {
            return Err(BridgeError::NativeLoopDisabled);
        }

        // Forwarder pumps the emitter stream into the reducer until the
        // harness side closes the channel OR a drain signal is sent.
        // Using a oneshot drain signal (rather than just rx-close) lets
        // us return from `run_caduceus_loop` deterministically without
        // having to drop `harness` first — the harness is borrowed, so
        // the caller may want to keep it alive for the next turn.
        let reducer_fwd = reducer.clone();
        let (forwarder, drain_tx) = spawn_forwarder(event_rx, move |ev| {
            reducer_fwd.ingest_event(ev);
        });

        let run_result = harness
            .run(state, history, user_input)
            .await
            .map_err(BridgeError::from);

        // Signal drain; ignore send errors (forwarder already observed
        // channel close, which is the preferred path anyway).
        let _ = drain_tx.send(());
        await_forwarder_with_timeout(forwarder, "forwarder").await;

        run_result
    }

    /// G1b — variant of [`Self::run_caduceus_loop`] that additionally
    /// forwards every `AgentEvent` through [`event_translator::translate`]
    /// into `translated_tx` alongside the reducer pipeline.
    ///
    /// **Why a separate method:** thread.rs (and any IDE consumer) needs
    /// a stream of UI-shaped events (`TranslatedThreadEvent`) to drive
    /// the ThreadEventStream. Emitting them from the same forwarder that
    /// feeds the reducer keeps both consumers in lock-step — neither
    /// sees an event the other missed, and the two streams share
    /// ordering. `Swallow` outputs are forwarded too (consumers may log
    /// or drop them) so the stream length is always `translate(ev).len()`
    /// per engine event.
    ///
    /// Contract parity with `run_caduceus_loop`:
    /// * Native-loop flag must be on.
    /// * Forwarder drain is bounded (5s) and honours both channel-close
    ///   and an explicit drain signal.
    /// * The caller owns the `translated_tx` / `translated_rx` split;
    ///   dropping the receiver early causes `try_send` to fail. The
    ///   forwarder records the **first** such failure into
    ///   `translated_channel_closed` (an `Arc<AtomicBool>`) so the
    ///   caller can differentiate *the consumer cancelled / panicked*
    ///   from *no events were emitted*. Subsequent send failures are
    ///   silently dropped — translated events remain best-effort, the
    ///   reducer pipeline is authoritative, and the flag is purely
    ///   diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub async fn run_caduceus_loop_translated(
        &self,
        harness: &AgentHarness,
        state: &mut SessionState,
        history: &mut ConversationHistory,
        user_input: &str,
        reducer: crate::dag_state::ReducerHandle,
        event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
        translated_tx: tokio::sync::mpsc::UnboundedSender<
            crate::event_translator::TranslatedThreadEvent,
        >,
        translated_channel_closed: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<String, BridgeError> {
        if !self.native_loop_enabled() {
            return Err(BridgeError::NativeLoopDisabled);
        }

        let reducer_fwd = reducer.clone();
        let closed_flag = translated_channel_closed.clone();
        let (forwarder, drain_tx) = spawn_forwarder(event_rx, move |ev| {
            reducer_fwd.ingest_event(ev);
            for out in crate::event_translator::translate(ev) {
                if translated_tx.send(out).is_err() {
                    // ST7-prereq: record the first channel-closed event so
                    // the caller can differentiate *consumer dropped early*
                    // from *no progress was made*. Subsequent failures are
                    // best-effort and silently dropped.
                    closed_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });

        let run_result = harness
            .run(state, history, user_input)
            .await
            .map_err(BridgeError::from);

        let _ = drain_tx.send(());
        await_forwarder_with_timeout(forwarder, "translated forwarder").await;

        run_result
    }

    /// Run a single agent turn (non-streaming).
    pub async fn run_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, BridgeError> {
        harness
            .run_turn(state, user_input)
            .await
            .map_err(BridgeError::from)
    }

    /// Run a single agent turn with streaming.
    pub async fn stream_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, BridgeError> {
        harness
            .stream_turn(state, user_input)
            .await
            .map_err(BridgeError::from)
    }

    // ── PRD parsing ──────────────────────────────────────────────────────

    /// Extract `(heading, content)` pairs from markdown text.
    pub fn extract_sections(text: &str) -> Vec<(String, String)> {
        PrdParser::extract_sections(text)
    }

    /// Parse a markdown PRD document into a list of tasks.
    pub fn parse_prd(text: &str) -> Vec<PrdTask> {
        PrdParser::parse(text)
    }

    /// Infer dependency edges between tasks from keyword references.
    /// Returns `(dependent_id, dependency_id)` pairs.
    pub fn infer_dependencies(tasks: &[PrdTask]) -> Vec<(usize, usize)> {
        PrdParser::infer_dependencies(tasks)
    }

    // ── Task recommendation ──────────────────────────────────────────────

    /// Rank incomplete tasks by readiness, priority, and complexity.
    pub fn recommend_next(tasks: &[PrdTask], completed: &[usize]) -> Vec<TaskRecommendation> {
        TaskRecommender::recommend_next(tasks, completed)
    }

    // ── Progress inference ───────────────────────────────────────────────

    /// Estimate progress from git commit messages referencing a task title.
    pub fn infer_from_commits(task_title: &str, commit_messages: &[String]) -> InferredProgress {
        ProgressInferrer::infer_from_commits(task_title, commit_messages)
    }

    /// Progress from test suite pass rate (0–100).
    pub fn infer_from_tests(total: usize, passing: usize) -> f64 {
        ProgressInferrer::infer_from_tests(total, passing)
    }

    /// Progress from file creation ratio (0–100).
    pub fn infer_from_files(files_planned: usize, files_created: usize) -> f64 {
        ProgressInferrer::infer_from_files(files_planned, files_created)
    }

    /// Weighted average of commit/test/file progress (40/40/20).
    pub fn combined_progress(commits: f64, tests: f64, files: f64) -> f64 {
        ProgressInferrer::combined(commits, tests, files)
    }

    // ── Agent scaffolding ────────────────────────────────────────────────

    /// Suggest trigger phrases for an agent based on its description.
    pub fn suggest_triggers(description: &str) -> Vec<String> {
        AgentScaffolder::suggest_triggers(description)
    }

    /// List available tool set presets for agent scaffolding.
    pub fn available_tool_sets() -> Vec<(&'static str, &'static [&'static str])> {
        AgentScaffolder::available_tool_sets()
    }

    // ── Skill scaffolding ────────────────────────────────────────────────

    /// Extract a skill definition from a conversation history.
    pub fn from_conversation(messages: &[String]) -> String {
        SkillScaffolder::from_conversation(messages)
    }

    // ── Execution tree visualization ─────────────────────────────────────

    /// Render an execution tree as a Mermaid `graph TD` flowchart.
    pub fn to_mermaid(tree: &ExecutionTreeViz) -> String {
        tree.to_mermaid()
    }

    /// Render an execution tree as React Flow nodes + edges JSON.
    pub fn to_react_flow_json(tree: &ExecutionTreeViz) -> serde_json::Value {
        tree.to_react_flow_json()
    }

    /// Map a node status string to a CSS color.
    pub fn node_color(status: &str) -> &'static str {
        ExecutionTreeViz::node_color(status)
    }

    // ── Time tracking ────────────────────────────────────────────────────

    /// Create a new empty time tracker.
    pub fn new_time_tracker() -> TimeTracker {
        TimeTracker::new()
    }

    /// Start tracking a task with an estimated duration.
    pub fn start_task(tracker: &mut TimeTracker, task_id: usize, estimated: f64) {
        tracker.start_task(task_id, estimated);
    }

    /// Mark a task as completed with the actual hours spent.
    pub fn complete_task(tracker: &mut TimeTracker, task_id: usize, actual: f64) {
        tracker.complete_task(task_id, actual);
    }

    /// Velocity ratio (estimated / actual) for completed tasks.
    pub fn velocity(tracker: &TimeTracker) -> f64 {
        tracker.velocity()
    }

    /// Sum of estimated hours across all tracked tasks.
    pub fn total_estimated(tracker: &TimeTracker) -> f64 {
        tracker.total_estimated()
    }

    /// Sum of actual hours across all tracked tasks.
    pub fn total_actual(tracker: &TimeTracker) -> f64 {
        tracker.total_actual()
    }

    /// Task IDs that are still running and have exceeded their estimate.
    pub fn overdue_tasks(tracker: &TimeTracker) -> Vec<usize> {
        tracker.overdue_tasks()
    }

    // ── Hierarchical task tree ───────────────────────────────────────────

    /// Create a new empty task tree.
    pub fn new_task_tree() -> TaskTree {
        TaskTree::new()
    }

    /// Add a task to the tree, optionally under a parent. Returns the new task ID.
    pub fn add_task(tree: &mut TaskTree, title: &str, parent: Option<usize>) -> usize {
        tree.add_task(title, parent)
    }

    /// Get immediate children of a task.
    pub fn children(tree: &TaskTree, id: usize) -> Vec<&HierarchicalTask> {
        tree.children(id)
    }

    /// Get all descendants of a task (depth-first).
    pub fn subtree(tree: &TaskTree, id: usize) -> Vec<&HierarchicalTask> {
        tree.subtree(id)
    }

    /// Render the entire task tree as an indented string.
    pub fn to_tree_string(tree: &TaskTree) -> String {
        tree.to_tree_string()
    }
}

/// Phase-H H3 — shared forwarder primitive used by
/// [`OrchestratorBridge::run_caduceus_loop`] and
/// [`OrchestratorBridge::run_caduceus_loop_translated`].
///
/// Spawns a tokio task that pumps every [`AgentEvent`] from `event_rx`
/// into `on_event` until *either* the channel closes *or* the returned
/// drain signal is sent (at which point already-buffered events are
/// drained via `try_recv` before exit so tail events like `TurnComplete`
/// are not dropped).
///
/// Returns `(JoinHandle, drain_tx)`. The caller:
/// * awaits `harness.run()`,
/// * sends on `drain_tx` to request an orderly exit, then
/// * passes the `JoinHandle` to [`await_forwarder_with_timeout`].
fn spawn_forwarder<F>(
    mut event_rx: tokio::sync::mpsc::Receiver<AgentEvent>,
    mut on_event: F,
) -> (
    tokio::task::JoinHandle<()>,
    tokio::sync::oneshot::Sender<()>,
)
where
    F: FnMut(&AgentEvent) + Send + 'static,
{
    let (drain_tx, mut drain_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                maybe_ev = event_rx.recv() => {
                    match maybe_ev {
                        Some(ev) => on_event(&ev),
                        None => break,
                    }
                }
                _ = &mut drain_rx => {
                    while let Ok(ev) = event_rx.try_recv() {
                        on_event(&ev);
                    }
                    break;
                }
            }
        }
    });
    (handle, drain_tx)
}

/// Phase-H H3 — 5s-bounded wait on a forwarder task. Logs (never
/// panics) on join failure or timeout. `label` is used in the log
/// message to distinguish `forwarder` from `translated forwarder`.
async fn await_forwarder_with_timeout(forwarder: tokio::task::JoinHandle<()>, label: &'static str) {
    match tokio::time::timeout(std::time::Duration::from_secs(5), forwarder).await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            log::warn!("[caduceus_bridge] {label} task join failed: {e}");
        }
        Err(_) => {
            log::warn!("[caduceus_bridge] {label} task did not complete within 5s; abandoning");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ST-F2 — HarnessBuilder (contract `harness-builder-v2`)
// ─────────────────────────────────────────────────────────────────────────────
//
// Single fluent entry point for producing a configured `AgentHarness`.
// Replaces the 9 `build_harness_*` variants (now thin wrappers). Every
// knob is optional; defaults match `OrchestratorBridge::build_harness()`.

enum ApprovalChoice {
    Default,
    Custom(Vec<String>),
    None,
}

enum SinkChoice {
    None,
    AutoReducer,
    Custom(Arc<dyn caduceus_orchestrator::IntrospectionSink>),
}

enum EmitterChoice {
    None,
    AutoDefault,
    /// ST7 r3 #1: reuse a pre-existing emitter that callers built
    /// outside the harness builder. Used by zed's subagent path,
    /// where `Thread::new_subagent` seeds the emitter at construction
    /// time so `SubagentHandle::events()` can return a live receiver
    /// BEFORE `prompt()` runs the harness builder. Without this
    /// variant, `AutoDefault` would create a second emitter and the
    /// pump (subscribed to the seeded one) would observe nothing.
    ///
    /// Caller is responsible for draining the mpsc rx that pairs with
    /// the seeded emitter (otherwise emit() trips Closed).
    Reuse(caduceus_orchestrator::AgentEventEmitter),
}

enum InjectorChoice {
    None,
    Custom(Arc<dyn ContextInjector>),
}

/// Fluent builder for an [`AgentHarness`]. Obtain via
/// [`OrchestratorBridge::harness`]. Every knob is optional; the default
/// matches the pre-ST-F2 `build_harness()` shape.
pub struct HarnessBuilder<'a> {
    bridge: &'a OrchestratorBridge,
    provider: Arc<dyn LlmAdapter>,
    tools: ToolRegistry,
    system_prompt: String,
    approval: ApprovalChoice,
    envelope: Option<PermissionEnvelope>,
    sink: SinkChoice,
    emitter: EmitterChoice,
    injector: InjectorChoice,
    max_context_tokens: u32,
    tool_timeout: std::time::Duration,
    /// ST8-PR3 — opt-in to the orchestrator pause-before-deny-commit path.
    /// `false` by default to preserve existing behaviour.
    resume_on_grant: bool,
    /// ST8-PR3 — override for the orchestrator's grant-pause timeout.
    /// `None` falls back to the harness default (5s).
    grant_timeout: Option<std::time::Duration>,
}

/// Result of [`HarnessBuilder::build`].
#[non_exhaustive]
pub struct BuiltHarness {
    pub harness: AgentHarness,
    pub approval_tx: Option<tokio::sync::mpsc::Sender<(String, bool)>>,
    pub reducer_handle: Option<crate::dag_state::ReducerHandle>,
    pub event_rx: Option<tokio::sync::mpsc::Receiver<AgentEvent>>,
    pub replay_handle: Option<ReplayHandle>,
    /// ST-A2c: a clone of the underlying [`AgentEventEmitter`] when one
    /// was built. Long-lived (per-session) consumers hold this and call
    /// `.subscribe()` at the start of each turn to get a fresh broadcast
    /// receiver, bypassing the one-shot mpsc `event_rx`. Per-turn
    /// consumers that still take ownership of `event_rx` are unaffected.
    pub emitter: Option<AgentEventEmitter>,
}

impl<'a> HarnessBuilder<'a> {
    pub fn approval_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.approval = ApprovalChoice::Custom(tools.into_iter().map(Into::into).collect());
        self
    }

    pub fn no_approval(mut self) -> Self {
        self.approval = ApprovalChoice::None;
        self
    }

    pub fn envelope(mut self, envelope: PermissionEnvelope) -> Self {
        self.envelope = Some(envelope);
        self
    }

    pub fn with_reducer_sink(mut self) -> Self {
        self.sink = SinkChoice::AutoReducer;
        self
    }

    pub fn with_sink(mut self, sink: Arc<dyn caduceus_orchestrator::IntrospectionSink>) -> Self {
        self.sink = SinkChoice::Custom(sink);
        self
    }

    pub fn with_emitter(mut self) -> Self {
        self.emitter = EmitterChoice::AutoDefault;
        self
    }

    /// ST7 r3 #1: reuse a pre-existing AgentEventEmitter rather than
    /// having the builder mint a fresh one. Used by zed's subagent
    /// path, where the emitter must be live BEFORE the harness is
    /// built so spawn_agent_tool can attach its phase-tracking pump
    /// at `subagent.events(cx)` time (otherwise the pump task never
    /// starts, leaving last_phase pinned at ModelSelection through
    /// every timeout — a silent telemetry regression).
    ///
    /// The returned BuiltHarness carries this same emitter in
    /// `emitter` and leaves `event_rx` as None — caller must have
    /// already arranged a drain on the mpsc receiver paired with the
    /// seeded emitter at construction time.
    pub fn with_emitter_reuse(mut self, emitter: caduceus_orchestrator::AgentEventEmitter) -> Self {
        self.emitter = EmitterChoice::Reuse(emitter);
        self
    }

    pub fn injector(mut self, injector: Arc<dyn ContextInjector>) -> Self {
        self.injector = InjectorChoice::Custom(injector);
        self
    }

    pub fn max_context_tokens(mut self, n: u32) -> Self {
        self.max_context_tokens = n;
        self
    }

    pub fn tool_timeout(mut self, d: std::time::Duration) -> Self {
        self.tool_timeout = d;
        self
    }

    /// ST8-PR3 — enable the orchestrator's resume-on-grant pause path.
    /// When `true`, the harness will pause a would-be deny for up to
    /// [`Self::with_grant_timeout`] (5s default) and emit
    /// `AgentEvent::GrantPending`, waiting for an out-of-band call to
    /// [`AgentHarness::submit_grant`]. Default `false` — purely
    /// additive; existing callers see no behavioural change.
    pub fn with_resume_on_grant(mut self, enabled: bool) -> Self {
        self.resume_on_grant = enabled;
        self
    }

    /// ST8-PR3 — override the grant-pause timeout. Only meaningful when
    /// [`Self::with_resume_on_grant`] is `true`.
    pub fn with_grant_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.grant_timeout = Some(timeout);
        self
    }

    pub fn build(self) -> BuiltHarness {
        let HarnessBuilder {
            bridge,
            provider,
            tools,
            system_prompt,
            approval,
            envelope,
            sink,
            emitter,
            injector,
            max_context_tokens,
            tool_timeout,
            resume_on_grant,
            grant_timeout,
        } = self;

        let mut base = AgentHarness::new(provider, tools, max_context_tokens, &system_prompt)
            .with_tool_timeout(tool_timeout)
            .with_instructions(&bridge.project_root);

        if resume_on_grant {
            base = base.with_resume_on_grant(true);
        }
        if let Some(d) = grant_timeout {
            base = base.with_grant_timeout(d);
        }

        if let Some(env) = envelope {
            base = base.with_permission_envelope(env);
        }

        let reducer_handle = match sink {
            SinkChoice::None => None,
            SinkChoice::AutoReducer => {
                let handle = bridge.new_reducer_handle();
                let s: Arc<dyn caduceus_orchestrator::IntrospectionSink> = Arc::new(handle.clone());
                base = base.with_introspection_sink(s);
                Some(handle)
            }
            SinkChoice::Custom(s) => {
                base = base.with_introspection_sink(s);
                None
            }
        };

        let (event_rx, replay_handle, emitter_clone) = match emitter {
            EmitterChoice::None => (None, None, None),
            EmitterChoice::AutoDefault => {
                let (em, rx) =
                    AgentEventEmitter::channel(OrchestratorBridge::DEFAULT_EVENT_CHANNEL_BUFFER);
                let replay = ReplayHandle::new(em.clone());
                let em_for_subscribe = em.clone();
                base = base.with_emitter(em);
                (Some(rx), Some(replay), Some(em_for_subscribe))
            }
            EmitterChoice::Reuse(em) => {
                // ST7 r3 #1: caller already drains the mpsc rx that
                // pairs with this emitter — return None for event_rx
                // so we don't double-drain.
                let replay = ReplayHandle::new(em.clone());
                let em_for_subscribe = em.clone();
                base = base.with_emitter(em);
                (None, Some(replay), Some(em_for_subscribe))
            }
        };

        if let InjectorChoice::Custom(inj) = injector {
            base = base.with_context_injector(inj);
        }

        let base = bridge.attach_p12(base);

        let (harness, approval_tx) = match approval {
            ApprovalChoice::Default => {
                let (h, tx) = base.with_approval_flow(
                    OrchestratorBridge::DEFAULT_APPROVAL_TOOLS
                        .iter()
                        .map(|s| s.to_string()),
                );
                (h, Some(tx))
            }
            ApprovalChoice::Custom(set) => {
                let (h, tx) = base.with_approval_flow(set);
                (h, Some(tx))
            }
            ApprovalChoice::None => (base, None),
        };

        BuiltHarness {
            harness,
            approval_tx,
            reducer_handle,
            event_rx,
            replay_handle,
            emitter: emitter_clone,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_orchestrator::VizTreeNode;

    #[test]
    fn orchestrator_new_history() {
        let history = OrchestratorBridge::new_history();
        assert!(history.is_empty());
    }

    // ── F4 / G1c — check_tool_envelope ─────────────────────────────────
    #[test]
    fn check_tool_envelope_none_allows() {
        let out = check_tool_envelope(None, "bash", &serde_json::json!({"command": "ls"}));
        assert!(matches!(out, PreflightOutcome::Allow));
    }

    #[test]
    fn check_tool_envelope_plan_intercepts_writes() {
        let env = caduceus_permissions::envelope::PermissionEnvelope::plan_preset();
        let out = check_tool_envelope(
            Some(&env),
            "edit_file",
            &serde_json::json!({"path": "src/main.rs"}),
        );
        assert!(
            matches!(out, PreflightOutcome::Intercept(_)),
            "plan preset must intercept writes, got {out:?}"
        );
    }

    #[test]
    fn check_tool_envelope_plan_allows_reads() {
        let env = caduceus_permissions::envelope::PermissionEnvelope::plan_preset();
        let out = check_tool_envelope(
            Some(&env),
            "read_file",
            &serde_json::json!({"path": "src/main.rs"}),
        );
        assert!(matches!(out, PreflightOutcome::Allow));
    }

    #[test]
    fn check_tool_envelope_act_denies_exec_when_disabled() {
        // Act preset with empty write lists; default exec is disabled.
        let env = caduceus_permissions::envelope::PermissionEnvelope::act_preset(vec![], vec![]);
        let out = check_tool_envelope(
            Some(&env),
            "bash",
            &serde_json::json!({"command": "rm -rf /"}),
        );
        assert!(
            matches!(out, PreflightOutcome::Deny { .. }),
            "act preset with exec disabled must deny bash, got {out:?}"
        );
    }

    #[test]
    fn orchestrator_new_session() {
        let bridge = OrchestratorBridge::new(".");
        let state = bridge.new_session("anthropic", "claude-sonnet");
        assert_eq!(state.turn_count, 0);
    }

    #[test]
    fn orchestrator_load_instructions_empty() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let result = bridge.load_instructions();
        assert!(result.is_ok());
        let set = result.unwrap();
        assert!(set.system_prompt.is_empty());
    }

    #[test]
    fn orchestrator_load_instructions_with_caduceus_md() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("CADUCEUS.md"), "# Project\nUse Rust.").unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let set = bridge.load_instructions().unwrap();
        assert!(set.system_prompt.contains("Use Rust"));
    }

    #[test]
    fn orchestrator_extract_memories_preference() {
        let memories = OrchestratorBridge::extract_memories(
            "I prefer async/await over raw futures",
            "Got it, I'll use async/await.",
        );
        assert!(!memories.is_empty(), "Should extract preference");
    }

    #[test]
    fn orchestrator_extract_memories_no_signal() {
        let memories = OrchestratorBridge::extract_memories("What is 2+2?", "4.");
        assert!(memories.is_empty(), "No preference signal");
    }

    #[test]
    fn default_approval_tools_covers_destructive_set() {
        let set: std::collections::HashSet<_> = OrchestratorBridge::DEFAULT_APPROVAL_TOOLS
            .iter()
            .copied()
            .collect();
        // Regression guard: removing any of these silently disables HITL for
        // a tool that mutates user state. If you intentionally drop one,
        // delete the assertion together with the constant entry.
        for required in [
            "bash",
            "shell",
            "write_file",
            "edit_file",
            "delete_file",
            "apply_patch",
        ] {
            assert!(
                set.contains(required),
                "DEFAULT_APPROVAL_TOOLS missing {required}"
            );
        }
    }

    #[test]
    fn build_harness_wires_approval_channel() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let (_harness, approval_tx) = bridge.build_harness(provider, tools, "test");
        // The bridge MUST hand the sender back; if this regresses to returning
        // just `AgentHarness`, every PermissionRequest will hang for 300s.
        assert!(!approval_tx.is_closed(), "approval channel should be live");
    }

    // ── ST-F2 — HarnessBuilder ────────────────────────────────────────

    #[test]
    fn harness_builder_default_matches_legacy_build_harness() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let built = bridge.harness(provider, tools, "test").build();
        assert!(built.approval_tx.is_some(), "default = HITL on");
        assert!(
            !built.approval_tx.as_ref().unwrap().is_closed(),
            "channel live"
        );
        assert!(built.reducer_handle.is_none(), "no sink by default");
        assert!(built.event_rx.is_none(), "no emitter by default");
        assert!(built.replay_handle.is_none(), "no emitter by default");
    }

    #[test]
    fn harness_builder_no_approval_drops_channel() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let built = bridge
            .harness(provider, tools, "test")
            .no_approval()
            .build();
        assert!(
            built.approval_tx.is_none(),
            "no_approval must not yield a channel"
        );
    }

    #[test]
    fn harness_builder_full_phase_g_surface() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let built = bridge
            .harness(provider, tools, "test")
            .with_reducer_sink()
            .with_emitter()
            .build();
        assert!(built.approval_tx.is_some());
        assert!(built.reducer_handle.is_some(), "sink requested");
        assert!(built.event_rx.is_some(), "emitter requested");
        assert!(built.replay_handle.is_some(), "emitter requested");
    }

    #[test]
    fn harness_builder_max_context_tokens_threads_through() {
        // Regression guard for audit finding F-13 (max_context_tokens was
        // hardcoded to 200_000 even when a smaller-context model was used).
        // We can't introspect the AgentHarness budget directly from the
        // bridge, but constructing with a small budget must not panic.
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let _built = bridge
            .harness(provider, tools, "test")
            .max_context_tokens(8_192)
            .build();
    }

    #[test]
    fn build_harness_with_approval_empty_set_is_autopilot() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        // Empty slice = caller explicitly opts out (e.g. headless mode).
        // Channel still exists but no tool will ever block on it.
        let (_h, tx) = bridge.build_harness_with_approval::<&str>(provider, tools, "test", &[]);
        assert!(!tx.is_closed());
    }

    /// ST-B2 / `envelope-surface-v1` — preset-bytes round-trip.
    ///
    /// The bridge-level `PermissionEnvelope` is a straight
    /// re-export of `caduceus_permissions::PermissionEnvelope`, so
    /// serialising a preset built through either the bridge path or the
    /// engine path MUST produce byte-equal JSON. This is the invariant
    /// that lets both layers share the same envelope without a
    /// translation layer.
    #[test]
    fn envelope_surface_preset_bytes_round_trip_equal() {
        use caduceus_permissions::envelope::PermissionEnvelope;

        let presets: &[(&str, PermissionEnvelope, PermissionEnvelope)] = &[
            (
                "plan",
                PermissionEnvelope::plan_preset(),
                PermissionEnvelope::plan_preset(),
            ),
            (
                "research",
                PermissionEnvelope::research_preset(),
                PermissionEnvelope::research_preset(),
            ),
            (
                "act",
                PermissionEnvelope::act_preset(vec!["src/**".into()], vec![]),
                PermissionEnvelope::act_preset(vec!["src/**".into()], vec![]),
            ),
            (
                "autopilot",
                PermissionEnvelope::autopilot_preset(vec!["src/**".into()], vec![]),
                PermissionEnvelope::autopilot_preset(vec!["src/**".into()], vec![]),
            ),
        ];
        for (label, bridge, engine) in presets {
            let b = serde_json::to_vec(bridge).unwrap();
            let e = serde_json::to_vec(engine).unwrap();
            assert_eq!(
                b, e,
                "{label}: PermissionEnvelope bytes must equal engine bytes"
            );
            // And the round-trip must be semantically stable too.
            let back: PermissionEnvelope = serde_json::from_slice(&b).unwrap();
            assert_eq!(
                serde_json::to_vec(&back).unwrap(),
                b,
                "{label}: deserialize→reserialize must be byte-stable"
            );
        }
    }

    /// ST-B2 / `envelope-surface-v1` — `build_harness_with_envelope`
    /// must thread the caller's envelope through to
    /// `AgentHarness::permission_envelope()` unchanged.
    #[tokio::test]
    async fn build_harness_with_envelope_threads_envelope_to_harness() {
        use caduceus_permissions::envelope::{FanoutPolicy, PermissionEnvelope};
        use caduceus_providers::mock::MockLlmAdapter;

        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();

        // Pick a non-default envelope so the assertion actually proves threading.
        let env = {
            let mut e = PermissionEnvelope::research_preset();
            e.fanout_policy = FanoutPolicy::MultiPersona;
            e.skill_budget = 7;
            e
        };
        let expected_bytes = serde_json::to_vec(&env).unwrap();

        let (harness, _tx) =
            bridge.build_harness_with_envelope(provider, tools, "test", env.clone());

        let got = harness
            .permission_envelope()
            .expect("harness must carry the envelope threaded through by the bridge");
        let got_bytes = serde_json::to_vec(got).unwrap();
        assert_eq!(got_bytes, expected_bytes);
    }

    /// ST-B2 combined with ST-B1: `build_harness_with_envelope_and_sink`
    /// must thread BOTH the envelope AND the reducer-backed sink, and
    /// the returned reducer handle must be the one the sink feeds.
    #[tokio::test]
    async fn build_harness_with_envelope_and_sink_threads_both() {
        use caduceus_orchestrator::critique_fanout::spawn_critique_fanout_via_harness;
        use caduceus_orchestrator::modes::PersonaRegistry;
        use caduceus_permissions::envelope::{FanoutPolicy, PermissionEnvelope};
        use caduceus_providers::mock::MockLlmAdapter;

        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();

        let env = {
            let mut e = PermissionEnvelope::research_preset();
            e.fanout_policy = FanoutPolicy::MultiPersona;
            e
        };
        let (harness, _tx, handle) =
            bridge.build_harness_with_envelope_and_sink(provider, tools, "test", env.clone());

        // Envelope threaded.
        let got_env = harness.permission_envelope().expect("envelope");
        assert_eq!(
            serde_json::to_vec(got_env).unwrap(),
            serde_json::to_vec(&env).unwrap()
        );

        // Sink threaded — driving the harness-aware fan-out must populate
        // the reducer handle returned by the builder.
        let reg = PersonaRegistry::builtin_personas();
        use async_trait::async_trait;
        use caduceus_core::Critique;
        use caduceus_core::CritiqueSeverity;
        struct R;
        #[async_trait]
        impl caduceus_orchestrator::critique_fanout::CritiqueRunner for R {
            async fn critique(
                &self,
                persona: &str,
                _prefix: &str,
                _plan: &str,
                _env: &PermissionEnvelope,
            ) -> Result<Critique, anyhow::Error> {
                Ok(Critique {
                    persona: persona.to_string(),
                    severity: CritiqueSeverity::Info,
                    findings: vec![],
                    blocking: false,
                })
            }
            fn model_metadata(&self, _persona: &str) -> (String, String) {
                ("anthropic".into(), "opus".into())
            }
        }
        let got = spawn_critique_fanout_via_harness(&harness, "plan", &["cloud"], &reg, &R).await;
        assert_eq!(got.len(), 2, "rubber-duck + cloud");
        assert_eq!(handle.active_agents_dag(true).nodes.len(), 2);
    }

    /// ST-B3 / `context-injector-v1`: the `*_sink_and_injector` builder
    /// must thread an explicit `ContextInjector` through to the harness,
    /// AND installing `None` must fall back to
    /// `BuiltinScopedContextInjector::default()` (today's default
    /// scoping behaviour).
    #[tokio::test]
    async fn build_harness_with_envelope_sink_and_injector_threads_injector() {
        use caduceus_orchestrator::critique_fanout::spawn_critique_fanout_via_harness;
        use caduceus_orchestrator::modes::PersonaRegistry;
        use caduceus_orchestrator::{ContextInjector, ScopeRequest, ScopedContext};
        use caduceus_permissions::envelope::{FanoutPolicy, PermissionEnvelope};
        use caduceus_providers::mock::MockLlmAdapter;
        use std::sync::Mutex;
        use std::sync::atomic::AtomicUsize;

        struct CountingInjector {
            count: AtomicUsize,
            personas: Mutex<Vec<String>>,
        }
        impl ContextInjector for CountingInjector {
            fn scope_for(&self, req: ScopeRequest<'_>) -> ScopedContext {
                self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.personas.lock().unwrap().push(req.persona.to_string());
                caduceus_orchestrator::PassthroughContextInjector.scope_for(req)
            }
        }

        // Path 1: caller passes Some(custom injector) — must see each
        // runnable critic routed through it.
        let inj = Arc::new(CountingInjector {
            count: AtomicUsize::new(0),
            personas: Mutex::new(vec![]),
        });
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let env = {
            let mut e = PermissionEnvelope::research_preset();
            e.fanout_policy = FanoutPolicy::MultiPersona;
            e
        };
        let (harness, _tx, _h) = bridge.build_harness_with_envelope_sink_and_injector(
            provider,
            tools,
            "test",
            env.clone(),
            Some(inj.clone() as Arc<dyn ContextInjector>),
        );
        use async_trait::async_trait;
        use caduceus_core::Critique;
        use caduceus_core::CritiqueSeverity;
        struct R;
        #[async_trait]
        impl caduceus_orchestrator::critique_fanout::CritiqueRunner for R {
            async fn critique(
                &self,
                persona: &str,
                _prefix: &str,
                _plan: &str,
                _env: &PermissionEnvelope,
            ) -> Result<Critique, anyhow::Error> {
                Ok(Critique {
                    persona: persona.to_string(),
                    severity: CritiqueSeverity::Info,
                    findings: vec![],
                    blocking: false,
                })
            }
            fn model_metadata(&self, _persona: &str) -> (String, String) {
                ("anthropic".into(), "opus".into())
            }
        }
        let reg = PersonaRegistry::builtin_personas();
        let _got =
            spawn_critique_fanout_via_harness(&harness, "plan", &["cloud", "qa"], &reg, &R).await;
        assert_eq!(
            inj.count.load(std::sync::atomic::Ordering::SeqCst),
            3,
            "explicit injector must be invoked once per runnable critic"
        );

        // Path 2: caller passes None — builder must install the default
        // `BuiltinScopedContextInjector` so `harness.context_injector()` is Some.
        let dir2 = tempfile::tempdir().unwrap();
        let bridge2 = OrchestratorBridge::new(dir2.path());
        let provider2: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools2 = ToolRegistry::new();
        let (harness2, _tx2, _h2) = bridge2
            .build_harness_with_envelope_sink_and_injector(provider2, tools2, "test", env, None);
        assert!(
            harness2.context_injector().is_some(),
            "None must fall back to BuiltinScopedContextInjector::default()"
        );
    }

    /// ST-B3 / `context-injector-v1`: `envelope.with_skill_budget(n)`
    /// must set the bytes deterministically so IDE callers can tighten
    /// skill activation without reaching into the struct fields.
    #[test]
    fn envelope_with_skill_budget_fluently_overrides_field() {
        use caduceus_permissions::envelope::PermissionEnvelope;
        let env = PermissionEnvelope::research_preset().with_skill_budget(2);
        assert_eq!(env.skill_budget, 2);
        // Narrowing then widening must not silently re-widen past the
        // preset default (the builder is just a field write, but guard
        // against accidental clamps being added).
        let env2 = env.with_skill_budget(12);
        assert_eq!(env2.skill_budget, 12);
    }

    /// ST-B1 / `harness-sink-v1`: the harness built by `build_harness_with_sink`
    /// must carry an introspection sink that feeds the same reducer the
    /// returned handle reads from — so `spawn_critique_fanout_via_harness`
    /// populates the Agents-DAG without any separate wire-up.
    #[tokio::test]
    async fn build_harness_with_sink_routes_fanout_events_to_reducer() {
        use caduceus_orchestrator::critique_fanout::spawn_critique_fanout_via_harness;
        use caduceus_orchestrator::modes::PersonaRegistry;
        use caduceus_permissions::envelope::{FanoutPolicy, PermissionEnvelope};
        use caduceus_providers::mock::MockLlmAdapter;

        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let (harness, _tx, handle) = bridge.build_harness_with_sink(provider, tools, "test");

        // Install a MultiPersona envelope so the fan-out has critics to spawn.
        let env = {
            let mut e = PermissionEnvelope::research_preset();
            e.fanout_policy = FanoutPolicy::MultiPersona;
            e
        };
        let harness = harness.with_permission_envelope(env);

        let reg = PersonaRegistry::builtin_personas();
        // StubRunner is not public; use the real spawn helper — it calls
        // `self.critique(...)` through a trait we can define inline.
        use async_trait::async_trait;
        use caduceus_core::Critique;
        use caduceus_core::CritiqueSeverity;
        struct R;
        #[async_trait]
        impl caduceus_orchestrator::critique_fanout::CritiqueRunner for R {
            async fn critique(
                &self,
                persona: &str,
                _prefix: &str,
                _plan: &str,
                _env: &PermissionEnvelope,
            ) -> Result<Critique, anyhow::Error> {
                Ok(Critique {
                    persona: persona.to_string(),
                    severity: CritiqueSeverity::Info,
                    findings: vec![],
                    blocking: false,
                })
            }
            fn model_metadata(&self, _persona: &str) -> (String, String) {
                ("anthropic".into(), "opus".into())
            }
        }

        let got =
            spawn_critique_fanout_via_harness(&harness, "plan body", &["cloud", "qa"], &reg, &R)
                .await;
        assert_eq!(got.len(), 3, "rubber-duck + cloud + qa");

        // The reducer handle returned by `build_harness_with_sink` MUST be
        // the same reducer the harness sink fed — so the critic nodes show
        // up in its Agents-DAG without any additional wire-up.
        let agents = handle.active_agents_dag(true);
        assert_eq!(
            agents.nodes.len(),
            3,
            "3 critic nodes expected in the reducer the handle reads from"
        );
        assert!(
            handle.last_event_id() > 0,
            "reducer must have observed events through the harness sink"
        );
    }

    #[test]
    fn orchestrator_conversation_serialize_deserialize() {
        let history = OrchestratorBridge::new_history();
        let json = OrchestratorBridge::conversation_serialize(&history).unwrap();
        let restored = OrchestratorBridge::conversation_deserialize(&json).unwrap();
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn orchestrator_conversation_truncate() {
        let mut history = OrchestratorBridge::new_history();
        for _ in 0..10 {
            history.append(caduceus_providers::Message {
                role: "user".to_string(),
                content: String::new(),
                content_blocks: None,
                tool_calls: vec![],
                tool_result: None,
                cache_breakpoint: false,
            });
        }
        assert_eq!(history.len(), 10);
        OrchestratorBridge::conversation_truncate(&mut history, 5);
        assert!(history.len() <= 5);
    }

    #[test]
    fn orchestrator_conversation_clear() {
        let mut history = OrchestratorBridge::new_history();
        history.append(caduceus_providers::Message {
            role: "user".to_string(),
            content: String::new(),
            content_blocks: None,
            tool_calls: vec![],
            tool_result: None,
            cache_breakpoint: false,
        });
        assert!(!history.is_empty());
        OrchestratorBridge::conversation_clear(&mut history);
        assert!(history.is_empty());
    }

    #[test]
    fn orchestrator_compact_instructions_passthrough() {
        let short = "Use Rust.";
        let result = OrchestratorBridge::compact_instructions(short, 1000);
        assert_eq!(result, short);
    }

    #[test]
    fn orchestrator_compact_instructions_truncates() {
        let long = "# Rules\n- MUST use Rust\n- NEVER use C++\n\n```\nsome code block\n```\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(20);
        let result = OrchestratorBridge::compact_instructions(&long, 200);
        assert!(result.len() < long.len(), "Should be shorter than original");
    }

    #[test]
    fn orchestrator_conversation_deserialize_invalid() {
        let result = OrchestratorBridge::conversation_deserialize("not json");
        assert!(result.is_err());
    }

    // ── PRD parsing tests ────────────────────────────────────────────────

    #[test]
    fn orchestrator_extract_sections() {
        let md = "# Auth\nBuild login.\n## OAuth\nUse OAuth2.";
        let sections = OrchestratorBridge::extract_sections(md);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].0, "Auth");
        assert!(sections[0].1.contains("Build login"));
    }

    #[test]
    fn orchestrator_parse_prd() {
        let prd = "# Task A\nDo A.\n# Task B\nDo B.";
        let tasks = OrchestratorBridge::parse_prd(prd);
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].title, "Task A");
        assert_eq!(tasks[1].title, "Task B");
    }

    #[test]
    fn orchestrator_infer_dependencies() {
        let tasks =
            OrchestratorBridge::parse_prd("# Auth\nBuild auth.\n# API\nRequires Auth module.");
        let deps = OrchestratorBridge::infer_dependencies(&tasks);
        // "API" description mentions "Auth", so (1, 0) expected
        assert!(!deps.is_empty());
        assert!(deps.contains(&(1, 0)));
    }

    // ── Task recommendation tests ────────────────────────────────────────

    #[test]
    fn orchestrator_recommend_next() {
        let tasks =
            OrchestratorBridge::parse_prd("# Setup\nInit project.\n# Build\nBuild features.");
        let recs = OrchestratorBridge::recommend_next(&tasks, &[]);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].score >= recs[1].score);
    }

    #[test]
    fn orchestrator_recommend_next_with_completed() {
        let tasks = OrchestratorBridge::parse_prd("# A\nDo A.\n# B\nDo B.");
        let recs = OrchestratorBridge::recommend_next(&tasks, &[0]);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].task_id, 1);
    }

    // ── Progress inference tests ─────────────────────────────────────────

    #[test]
    fn orchestrator_infer_from_commits() {
        let commits = vec![
            "implement auth login".to_string(),
            "fix auth bug".to_string(),
        ];
        let progress = OrchestratorBridge::infer_from_commits("auth", &commits);
        assert!(progress.confidence > 0.0);
        assert!(!progress.evidence.is_empty());
    }

    #[test]
    fn orchestrator_infer_from_commits_empty() {
        let progress = OrchestratorBridge::infer_from_commits("auth", &[]);
        assert_eq!(progress.percentage, 0.0);
        assert_eq!(progress.confidence, 0.0);
    }

    #[test]
    fn orchestrator_infer_from_tests() {
        assert_eq!(OrchestratorBridge::infer_from_tests(10, 10), 100.0);
        assert_eq!(OrchestratorBridge::infer_from_tests(10, 5), 50.0);
        assert_eq!(OrchestratorBridge::infer_from_tests(0, 0), 0.0);
    }

    #[test]
    fn orchestrator_infer_from_files() {
        assert_eq!(OrchestratorBridge::infer_from_files(4, 2), 50.0);
        assert_eq!(OrchestratorBridge::infer_from_files(4, 4), 100.0);
        assert_eq!(OrchestratorBridge::infer_from_files(0, 5), 0.0);
    }

    #[test]
    fn orchestrator_combined_progress() {
        let result = OrchestratorBridge::combined_progress(100.0, 100.0, 100.0);
        assert_eq!(result, 100.0);
        let partial = OrchestratorBridge::combined_progress(50.0, 50.0, 50.0);
        assert!((partial - 50.0).abs() < 0.01);
    }

    // ── Agent scaffolding tests ──────────────────────────────────────────

    #[test]
    fn orchestrator_suggest_triggers() {
        let triggers = OrchestratorBridge::suggest_triggers("code review automation");
        assert!(!triggers.is_empty());
        assert!(triggers.iter().any(|t| t.contains("review")));
    }

    #[test]
    fn orchestrator_suggest_triggers_fallback() {
        let triggers = OrchestratorBridge::suggest_triggers("something unique");
        assert!(!triggers.is_empty());
    }

    #[test]
    fn orchestrator_available_tool_sets() {
        let sets = OrchestratorBridge::available_tool_sets();
        assert!(sets.len() >= 3);
        let names: Vec<&str> = sets.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"standard"));
        assert!(names.contains(&"read-only"));
    }

    // ── Skill scaffolding tests ──────────────────────────────────────────

    #[test]
    fn orchestrator_from_conversation() {
        let messages = vec![
            "First, run the linter.".to_string(),
            "Then fix the errors.".to_string(),
            "Finally, commit the changes.".to_string(),
        ];
        let skill = OrchestratorBridge::from_conversation(&messages);
        assert!(!skill.is_empty());
    }

    // ── Execution tree visualization tests ───────────────────────────────

    #[test]
    fn orchestrator_to_mermaid() {
        let mut tree = ExecutionTreeViz::new();
        tree.add_node(VizTreeNode {
            id: "root".to_string(),
            label: "Root Task".to_string(),
            status: "succeeded".to_string(),
            parent: None,
            error: None,
            depth: 0,
        });
        tree.add_node(VizTreeNode {
            id: "child".to_string(),
            label: "Child Task".to_string(),
            status: "active".to_string(),
            parent: Some("root".to_string()),
            error: None,
            depth: 1,
        });
        let mermaid = OrchestratorBridge::to_mermaid(&tree);
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("root"));
        assert!(mermaid.contains("child"));
    }

    #[test]
    fn orchestrator_to_react_flow_json() {
        let mut tree = ExecutionTreeViz::new();
        tree.add_node(VizTreeNode {
            id: "n1".to_string(),
            label: "Node 1".to_string(),
            status: "succeeded".to_string(),
            parent: None,
            error: None,
            depth: 0,
        });
        let json = OrchestratorBridge::to_react_flow_json(&tree);
        assert!(json.get("nodes").is_some());
        assert!(json.get("edges").is_some());
    }

    #[test]
    fn orchestrator_node_color() {
        assert_eq!(OrchestratorBridge::node_color("succeeded"), "#10b981");
        assert_eq!(OrchestratorBridge::node_color("failed"), "#ef4444");
        assert_eq!(OrchestratorBridge::node_color("active"), "#f59e0b");
        assert_eq!(OrchestratorBridge::node_color("pruned"), "#6b7280");
        assert_eq!(OrchestratorBridge::node_color("unknown"), "#6b7280");
    }

    // ── Time tracker tests ───────────────────────────────────────────────

    #[test]
    fn orchestrator_time_tracker_lifecycle() {
        let mut tracker = OrchestratorBridge::new_time_tracker();

        OrchestratorBridge::start_task(&mut tracker, 1, 2.0);
        OrchestratorBridge::start_task(&mut tracker, 2, 3.0);

        assert_eq!(OrchestratorBridge::total_estimated(&tracker), 5.0);
        assert_eq!(OrchestratorBridge::total_actual(&tracker), 0.0);

        OrchestratorBridge::complete_task(&mut tracker, 1, 1.5);
        assert!(OrchestratorBridge::total_actual(&tracker) > 0.0);

        let vel = OrchestratorBridge::velocity(&tracker);
        assert!(vel > 0.0);
    }

    #[test]
    fn orchestrator_time_tracker_velocity_default() {
        let tracker = OrchestratorBridge::new_time_tracker();
        assert_eq!(OrchestratorBridge::velocity(&tracker), 1.0);
    }

    #[test]
    fn orchestrator_overdue_tasks_empty() {
        let tracker = OrchestratorBridge::new_time_tracker();
        assert!(OrchestratorBridge::overdue_tasks(&tracker).is_empty());
    }

    // ── Hierarchical task tree tests ─────────────────────────────────────

    #[test]
    fn orchestrator_task_tree_add_and_children() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        let child1 = OrchestratorBridge::add_task(&mut tree, "Child 1", Some(root));
        let child2 = OrchestratorBridge::add_task(&mut tree, "Child 2", Some(root));

        let children = OrchestratorBridge::children(&tree, root);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].id, child1);
        assert_eq!(children[1].id, child2);
    }

    #[test]
    fn orchestrator_task_tree_subtree() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        let child = OrchestratorBridge::add_task(&mut tree, "Child", Some(root));
        let _grandchild = OrchestratorBridge::add_task(&mut tree, "Grandchild", Some(child));

        let sub = OrchestratorBridge::subtree(&tree, root);
        assert_eq!(sub.len(), 2); // child + grandchild
    }

    #[test]
    fn orchestrator_task_tree_to_string() {
        let mut tree = OrchestratorBridge::new_task_tree();
        let root = OrchestratorBridge::add_task(&mut tree, "Root", None);
        OrchestratorBridge::add_task(&mut tree, "Child", Some(root));

        let output = OrchestratorBridge::to_tree_string(&tree);
        assert!(output.contains("Root"));
        assert!(output.contains("Child"));
    }

    // ── G17: replay seam integration tests ──────────────────────────────

    #[test]
    fn build_harness_with_emitter_returns_replay_handle() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let (_h, approval_tx, replay_handle, event_rx) =
            bridge.build_harness_with_emitter(provider, tools, "test");
        // All four return slots must be live: handle and channel sender
        // must outlive the call, and the replay handle must report the
        // emitter's configured cap (default = 200, per orchestrator).
        assert!(!approval_tx.is_closed(), "approval channel should be live");
        assert!(
            replay_handle.retention_cap() > 0,
            "replay handle must wrap a retention-enabled emitter"
        );
        // Empty replay before any events have been emitted.
        let snap = OrchestratorBridge::replay_session_events(&replay_handle);
        assert!(snap.is_empty(), "replay must be empty before any emit");
        // Channel must still be the live receiver, not closed by the build call.
        drop(event_rx);
    }

    #[tokio::test]
    async fn replay_handle_observes_events_emitted_by_harness() {
        use caduceus_core::SessionState;
        use caduceus_providers::mock::MockLlmAdapter;
        // End-to-end: emitter wired into the harness, the bridge's
        // ReplayHandle observes events emitted during a real `run` call.
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        // MockLlmAdapter returns one chat response so the harness exits
        // after a single iteration.
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![
            caduceus_providers::ChatResponse {
                content: "ok".to_string(),
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                stop_reason: caduceus_providers::StopReason::EndTurn,
                tool_calls: vec![],
                logprobs: None,
                thinking: String::new(),
            },
        ]));
        let tools = ToolRegistry::new();
        let (harness, _approval_tx, replay_handle, mut event_rx) =
            bridge.build_harness_with_emitter(provider, tools, "test");
        let mut state = SessionState::new(
            dir.path(),
            caduceus_core::ProviderId::new("mock"),
            caduceus_core::ModelId::new("mock-1"),
        );
        let mut history = ConversationHistory::new();
        let _ = harness
            .run(&mut state, &mut history, "hi")
            .await
            .expect("agent run must succeed");
        // Drain live channel.
        let mut live = 0usize;
        while event_rx.try_recv().is_ok() {
            live += 1;
        }
        // Replay handle must observe at least one event (TurnComplete or text-delta).
        let snap = OrchestratorBridge::replay_session_events(&replay_handle);
        assert!(
            !snap.is_empty(),
            "replay handle must observe events emitted by the harness (live={live}, snap=0)"
        );
    }

    // ── P12 production wiring tests ──────────────────────────────────────

    #[test]
    fn bridge_attaches_speculative_cache_to_harness() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let cache = caduceus_tools::SpeculativeCache::new(std::time::Duration::from_secs(30));
        let bridge = OrchestratorBridge::new(dir.path()).with_speculative_cache(cache.clone());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let h = bridge.build_harness_no_approval(provider, ToolRegistry::new(), "sys");
        assert!(
            h.speculative_cache().is_some(),
            "harness must carry the speculative cache"
        );
        assert!(
            bridge.speculative_cache().is_some(),
            "bridge must expose the cache for prefetchers"
        );
    }

    #[test]
    fn bridge_attaches_reflexion_to_harness_and_records_outcome() {
        use caduceus_orchestrator::reflexion::{
            AttemptOutcome, HeuristicReflector, ReflexionMemory,
        };
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let mem = Arc::new(std::sync::Mutex::new(ReflexionMemory::new(8)));
        let bridge = OrchestratorBridge::new(dir.path()).with_reflexion(mem.clone());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let h = bridge.build_harness_no_approval(provider, ToolRegistry::new(), "sys");
        assert!(h.reflexion().is_some(), "harness must carry reflexion");

        let reflector = HeuristicReflector;
        let outcome = AttemptOutcome::Failure {
            error: "compile error: missing semicolon".into(),
            attempted_action: None,
        };
        let stored = bridge
            .record_attempt_outcome(&reflector, "task-1", &outcome)
            .expect("must record");
        assert_eq!(stored.task_tag, "task-1");
        let guard = mem.lock().unwrap();
        assert_eq!(guard.recent_for("task-1", 5).len(), 1);
    }

    #[test]
    fn bridge_attaches_tot_config_to_harness() {
        use caduceus_orchestrator::PlannerConfig;
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let cfg = PlannerConfig::default();
        let bridge = OrchestratorBridge::new(dir.path()).with_tot_config(cfg);
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let h = bridge.build_harness_no_approval(provider, ToolRegistry::new(), "sys");
        assert!(h.tot_config().is_some(), "harness must carry tot config");
        assert!(bridge.tot_config().is_some(), "bridge must expose tot cfg");
    }

    #[test]
    fn default_helpers_install_p12_primitives() {
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path())
            .with_default_speculative_cache()
            .with_default_reflexion()
            .with_default_tot_config();
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let h = bridge.build_harness_no_approval(provider, ToolRegistry::new(), "sys");
        assert!(h.speculative_cache().is_some());
        assert!(h.reflexion().is_some());
        assert!(h.tot_config().is_some());
    }

    #[tokio::test]
    async fn register_mcp_tools_returns_zero_for_empty_manager() {
        use caduceus_mcp::McpServerManager;
        let mgr = Arc::new(McpServerManager::new());
        let mut reg = ToolRegistry::new();
        let n = OrchestratorBridge::register_mcp_tools(mgr, &mut reg).await;
        assert_eq!(n, 0, "no servers => no tools registered");
    }

    // ── P13.7 — MCP hot‑reload diff ──────────────────────────────────

    #[tokio::test]
    async fn p13_7_diff_returns_zero_when_no_changes() {
        use caduceus_mcp::McpServerManager;
        use std::collections::HashSet;
        let mgr = Arc::new(McpServerManager::new());
        let mut reg = ToolRegistry::new();
        let prior: HashSet<String> = HashSet::new();
        let (added, removed, live) =
            OrchestratorBridge::register_mcp_tools_diff(mgr.clone(), &mut reg, &prior).await;
        assert_eq!((added, removed), (0, 0));
        assert!(live.is_empty());
        // Idempotent: calling again with the new live set yields zero.
        let (a2, r2, _) = OrchestratorBridge::register_mcp_tools_diff(mgr, &mut reg, &live).await;
        assert_eq!((a2, r2), (0, 0));
    }

    #[tokio::test]
    async fn p13_7_diff_removes_stale_tools() {
        // Simulate: a previous refresh registered "ghost_mcp_tool"
        // and recorded its name in `prior_mcp_names`. The manager
        // surface no longer contains it. Calling diff must remove it.
        use caduceus_mcp::McpServerManager;
        use caduceus_tools::Tool;
        use std::collections::HashSet;

        struct Stub(&'static str);
        #[async_trait::async_trait]
        impl Tool for Stub {
            fn spec(&self) -> caduceus_core::ToolSpec {
                caduceus_core::ToolSpec {
                    name: self.0.into(),
                    description: "stub".into(),
                    input_schema: serde_json::json!({"type":"object"}),
                    required_capability: None,
                }
            }
            async fn call(
                &self,
                _input: serde_json::Value,
            ) -> caduceus_core::Result<caduceus_core::ToolResult> {
                Ok(caduceus_core::ToolResult::success("stub"))
            }
        }

        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(Stub("ghost_mcp_tool")));
        assert!(reg.get("ghost_mcp_tool").is_some());

        let mgr = Arc::new(McpServerManager::new());
        let mut prior: HashSet<String> = HashSet::new();
        prior.insert("ghost_mcp_tool".into());

        let (added, removed, live) =
            OrchestratorBridge::register_mcp_tools_diff(mgr, &mut reg, &prior).await;
        assert_eq!(added, 0);
        assert_eq!(removed, 1, "stale tool must be unregistered");
        assert!(live.is_empty());
        assert!(
            reg.get("ghost_mcp_tool").is_none(),
            "tool must actually be gone from the registry"
        );
    }

    #[tokio::test]
    async fn p12_primitives_are_attached_via_emitter_path() {
        use caduceus_orchestrator::reflexion::ReflexionMemory;
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let cache = caduceus_tools::SpeculativeCache::new(std::time::Duration::from_secs(60));
        let mem = Arc::new(std::sync::Mutex::new(ReflexionMemory::new(4)));
        let bridge = OrchestratorBridge::new(dir.path())
            .with_speculative_cache(cache)
            .with_reflexion(mem)
            .with_default_tot_config();
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let (h, _tx, _replay, _rx) =
            bridge.build_harness_with_emitter(provider, ToolRegistry::new(), "sys");
        assert!(h.speculative_cache().is_some());
        assert!(h.reflexion().is_some());
        assert!(h.tot_config().is_some());
    }

    // ── P3.1 — amend_plan IPC bridge ────────────────────────────────────

    #[test]
    fn p3_1_apply_replace_amendment_via_bridge() {
        let mut plan = ActionPlan::new();
        plan.add("read_file", &serde_json::json!({"path": "/a"}));
        plan.add(
            "write_file",
            &serde_json::json!({"path": "/b", "content": "x"}),
        );
        let rev = plan.actions[0].revision;
        let amend = serde_json::json!({
            "kind": "replace",
            "step": 1,
            "args": {"path": "/c"},
            "description": "read /c",
            "expected_revision": rev,
        });
        let res = apply_plan_amendment(&mut plan, &amend.to_string()).unwrap();
        assert_eq!(res.step, 1);
        assert_eq!(plan.actions[0].args, serde_json::json!({"path": "/c"}));
    }

    #[test]
    fn p3_1_stale_revision_rejected() {
        let mut plan = ActionPlan::new();
        plan.add("read_file", &serde_json::json!({"path": "/a"}));
        let amend = serde_json::json!({
            "kind": "replace",
            "step": 1,
            "args": {"path": "/c"},
            "description": "x",
            "expected_revision": 9999,
        });
        let err = apply_plan_amendment(&mut plan, &amend.to_string()).unwrap_err();
        match err {
            AmendError::StaleRevision { .. } => {}
            other => panic!("expected StaleRevision, got {other:?}"),
        }
    }

    #[test]
    fn p3_1_malformed_json_returns_stale_not_panic() {
        let mut plan = ActionPlan::new();
        let err = apply_plan_amendment(&mut plan, "{not json").unwrap_err();
        assert!(matches!(err, AmendError::StaleRevision { .. }));
    }

    #[test]
    fn p3_1_insert_amendment_renumbers_subsequent_steps() {
        let mut plan = ActionPlan::new();
        plan.add("a", &serde_json::json!({}));
        plan.add("b", &serde_json::json!({}));
        let amend = serde_json::json!({
            "kind": "insert",
            "after_step": 1,
            "tool_name": "mid",
            "args": {},
            "description": "mid",
            "expected_plan_revision": plan.revision,
        });
        let res = apply_plan_amendment(&mut plan, &amend.to_string()).unwrap();
        assert_eq!(res.step, 2);
        assert_eq!(plan.actions.len(), 3);
        assert_eq!(plan.actions[1].tool_name, "mid");
        assert_eq!(plan.actions[2].tool_name, "b");
    }

    #[test]
    fn p3_1_remove_amendment_drops_step() {
        let mut plan = ActionPlan::new();
        plan.add("a", &serde_json::json!({}));
        plan.add("b", &serde_json::json!({}));
        let rev = plan.actions[0].revision;
        let amend = serde_json::json!({
            "kind": "remove",
            "step": 1,
            "expected_revision": rev,
        });
        apply_plan_amendment(&mut plan, &amend.to_string()).unwrap();
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(plan.actions[0].tool_name, "b");
    }

    #[test]
    fn p3_1_snapshot_plan_json_round_trips() {
        let mut plan = ActionPlan::new();
        plan.add("read_file", &serde_json::json!({"path": "/x"}));
        let json = snapshot_plan_json(&plan).unwrap();
        let restored: ActionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.actions.len(), 1);
        assert_eq!(restored.revision, plan.revision);
    }

    // ── P3.3 — checkpoint bridge ─────────────────────────────────────────

    #[test]
    fn p3_3_begin_record_commit_revert_round_trips() {
        let mut store = new_checkpoint_store();
        let id = begin_checkpoint(&mut store, 1, "edit_file", 100);
        record_checkpoint_edit(
            &mut store,
            id,
            PathBuf::from("/repo/src/lib.rs"),
            Some("old".into()),
        )
        .unwrap();
        record_checkpoint_edit(&mut store, id, PathBuf::from("/repo/new_file.rs"), None).unwrap();
        commit_checkpoint(&mut store, id).unwrap();

        let snaps = revert_checkpoint(&mut store, id).unwrap();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].before.as_deref(), Some("old"));
        assert!(snaps[1].before.is_none());
    }

    #[test]
    fn p3_3_record_after_commit_fails_closed() {
        let mut store = new_checkpoint_store();
        let id = begin_checkpoint(&mut store, 0, "noop", 0);
        commit_checkpoint(&mut store, id).unwrap();
        let err = record_checkpoint_edit(&mut store, id, PathBuf::from("/x"), Some("y".into()))
            .unwrap_err();
        assert!(matches!(err, CheckpointError::AlreadyCommitted(_)));
    }

    #[test]
    fn p3_3_revert_unknown_id_fails_closed() {
        let mut store = new_checkpoint_store();
        let err = revert_checkpoint(&mut store, CheckpointId(9999)).unwrap_err();
        assert!(matches!(err, CheckpointError::Unknown(_)));
    }

    #[test]
    fn p3_3_double_revert_fails_closed() {
        let mut store = new_checkpoint_store();
        let id = begin_checkpoint(&mut store, 0, "edit", 0);
        record_checkpoint_edit(&mut store, id, PathBuf::from("/a"), Some("x".into())).unwrap();
        commit_checkpoint(&mut store, id).unwrap();
        revert_checkpoint(&mut store, id).unwrap();
        let err = revert_checkpoint(&mut store, id).unwrap_err();
        assert!(matches!(err, CheckpointError::AlreadyReverted(_)));
    }

    #[test]
    fn p3_3_list_checkpoints_json_renders_timeline() {
        let mut store = new_checkpoint_store();
        let id1 = begin_checkpoint(&mut store, 1, "edit_file: a", 1);
        commit_checkpoint(&mut store, id1).unwrap();
        let id2 = begin_checkpoint(&mut store, 2, "edit_file: b", 2);
        commit_checkpoint(&mut store, id2).unwrap();

        let json = list_checkpoints_json(&store).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // CheckpointStore::list returns newest-first.
        assert_eq!(arr[0]["tool_summary"], "edit_file: b");
        assert_eq!(arr[1]["tool_summary"], "edit_file: a");
    }

    #[test]
    fn p3_3_commit_unknown_id_fails_closed() {
        let mut store = new_checkpoint_store();
        let err = commit_checkpoint(&mut store, CheckpointId(42)).unwrap_err();
        assert!(matches!(err, CheckpointError::Unknown(_)));
    }

    // ── P3.4 — notifications bridge ──────────────────────────────────────

    fn dummy_automation_result(success: bool) -> AutomationResult {
        use chrono::Utc;
        AutomationResult {
            automation_id: "nightly".into(),
            trigger_event: "cron".into(),
            started_at: Utc::now(),
            completed_at: Utc::now(),
            success,
            output: "done".into(),
            tokens_used: Default::default(),
            cost_usd: 0.0,
            commit_sha: None,
            pr_url: None,
        }
    }

    #[tokio::test]
    async fn p3_4_publish_notification_round_trips_through_bridge() {
        let bus = BroadcastBus::new();
        let mut rx = subscribe_notifications(&bus);
        let n = Notification::info("test", "title", "body");
        let delivered = publish_notification(&bus, n.clone()).unwrap();
        assert_eq!(delivered, 1);
        let msg = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.channel, NOTIFICATIONS_CHANNEL);
        let parsed = parse_notification(&msg).unwrap();
        assert_eq!(parsed, n);
    }

    #[tokio::test]
    async fn p3_4_automation_completion_published_via_bridge() {
        let bus = BroadcastBus::new();
        let mut rx = subscribe_notifications(&bus);
        publish_automation_completion(&bus, &dummy_automation_result(true)).unwrap();
        let msg = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        let n = parse_notification(&msg).unwrap();
        assert_eq!(n.severity, NotificationSeverity::Info);
        assert_eq!(n.source, "automation:nightly");
    }

    #[tokio::test]
    async fn p3_4_automation_failure_published_with_error_severity() {
        let bus = BroadcastBus::new();
        let mut rx = subscribe_notifications(&bus);
        publish_automation_completion(&bus, &dummy_automation_result(false)).unwrap();
        let msg = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .unwrap()
            .unwrap();
        let n = parse_notification(&msg).unwrap();
        assert_eq!(n.severity, NotificationSeverity::Error);
        assert!(n.title.contains("failed"));
    }

    #[test]
    fn p3_4_publish_without_subscribers_surfaces_no_subscribers() {
        let bus = BroadcastBus::new();
        let n = Notification::info("s", "t", "b");
        let err = publish_notification(&bus, n).unwrap_err();
        assert!(matches!(err, BusError::NoSubscribers(_)));
    }

    #[test]
    fn p3_4_parse_notification_rejects_malformed_payload() {
        let msg = BusMessage {
            from: "x".into(),
            content: "not json".into(),
            timestamp: 0,
            channel: NOTIFICATIONS_CHANNEL.into(),
        };
        assert!(parse_notification(&msg).is_err());
    }

    // ── P4.1 — MemoryBlocks bridge ───────────────────────────────────────

    #[test]
    fn p4_1_new_memory_blocks_uses_supplied_limits() {
        let limits = BlockLimits {
            persona_chars: 10,
            project_context_tokens: 100,
            working_history_tokens: 200,
            archival_summary_tokens: 50,
        };
        let m = new_memory_blocks(limits);
        assert_eq!(m.limits.persona_chars, 10);
        assert!(m.working_history.is_empty());
    }

    #[test]
    fn p4_1_set_persona_truncates_over_cap_and_reports_dropped() {
        let mut m = new_memory_blocks(BlockLimits {
            persona_chars: 5,
            ..BlockLimits::default()
        });
        let dropped = set_memory_persona(&mut m, "abcdefghij");
        assert_eq!(dropped, 5);
        assert_eq!(m.persona, "abcde");
    }

    #[test]
    fn p4_1_append_working_message_grows_history() {
        let mut m = new_memory_blocks(BlockLimits::default());
        append_working_message(
            &mut m,
            WorkingMessage {
                role: "user".into(),
                text: "hi".into(),
                tokens: 1,
                pair_id: None,
            },
        );
        assert_eq!(m.working_history.len(), 1);
    }

    #[test]
    fn p4_1_compact_returns_telemetry_report() {
        let mut m = new_memory_blocks(BlockLimits {
            working_history_tokens: 10,
            ..BlockLimits::default()
        });
        for i in 0..5 {
            append_working_message(
                &mut m,
                WorkingMessage {
                    role: "tool".into(),
                    text: format!("m{i}"),
                    tokens: 5,
                    pair_id: None,
                },
            );
        }
        let report = compact_memory_blocks(&mut m);
        // We don't assert exact eviction count (depends on internal
        // policy) but the report must serialise.
        let _json = serde_json::to_string(&report).unwrap();
    }

    #[test]
    fn p4_1_snapshot_memory_blocks_json_round_trips() {
        let mut m = new_memory_blocks(BlockLimits::default());
        set_memory_persona(&mut m, "agent persona");
        set_memory_project_context(&mut m, "open files: lib.rs");
        let json = snapshot_memory_blocks_json(&m).unwrap();
        let restored: MemoryBlocks = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.persona, "agent persona");
        assert_eq!(restored.project_context, "open files: lib.rs");
    }

    // ── P4.2 — context_fold bridge ───────────────────────────────────────

    #[test]
    fn p4_2_fold_then_expand_round_trips() {
        let mut store = new_transcript_store();
        let big = "x".repeat(8_000);
        let folded = fold_transcript(&mut store, "bash", "ok", big.clone());
        assert_eq!(folded.original_chars, 8_000);
        let recovered = expand_transcript(&store, folded.id).unwrap();
        assert_eq!(recovered.len(), 8_000);
        assert_eq!(folded_transcript_count(&store), 1);
    }

    #[test]
    fn p4_2_expand_unknown_id_fails_closed() {
        let store = new_transcript_store();
        let err = expand_transcript(&store, TranscriptId(999)).unwrap_err();
        assert!(matches!(err, ExpandError::Unknown(_)));
    }

    // ── P4.4 — per-model budget snapshot ────────────────────────────────

    #[test]
    fn p4_4_model_budget_spec_returns_known_caps() {
        let (ctx, reserved) = model_budget_spec("gpt-4o");
        assert!(ctx > 0);
        assert!(reserved > 0);
        assert!(ctx > reserved, "context_limit must exceed reserved_output");
    }

    #[test]
    fn p4_4_model_budget_spec_unknown_model_returns_defaults() {
        let (ctx, reserved) = model_budget_spec("totally-made-up-model-9999");
        // Unknown models still get safe defaults — never zero (would
        // break budget arithmetic).
        assert!(ctx > 0);
        assert!(reserved > 0);
    }

    #[test]
    fn p4_4_model_budget_spec_json_well_formed() {
        let json = model_budget_spec_json("gpt-4o");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["model"], "gpt-4o");
        assert!(v["context_limit"].as_u64().unwrap() > 0);
        assert!(v["reserved_output"].as_u64().unwrap() > 0);
    }

    #[test]
    fn p4_4_model_budget_spec_json_escapes_quotes_in_model_id() {
        let json = model_budget_spec_json("model\"with\"quotes");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["model"], "model\"with\"quotes");
    }

    // ── P5.1 — CompactionTelemetry bridge ────────────────────────────────

    fn dummy_event(strategy: &str, t_before: u32, t_after: u32, turn: u32) -> CompactionEvent {
        CompactionEvent {
            strategy: strategy.into(),
            tokens_before: t_before,
            tokens_after: t_after,
            messages_before: 10,
            messages_after: 5,
            turn_index: turn,
            at_secs: 0,
            downstream_re_ask: None,
        }
    }

    #[test]
    fn p5_1_record_and_jsonl_round_trip() {
        let mut tel = new_compaction_telemetry();
        record_compaction_event(&mut tel, dummy_event("summarize", 8000, 2000, 1));
        record_compaction_event(&mut tel, dummy_event("tool_collapse", 9000, 3000, 2));
        let jsonl = compaction_telemetry_jsonl(&tel);
        assert_eq!(jsonl.lines().count(), 2);
        assert!(jsonl.contains("summarize"));
        assert!(jsonl.contains("tool_collapse"));
    }

    #[test]
    fn p5_1_mark_re_ask_returns_true_for_existing_turn() {
        let mut tel = new_compaction_telemetry();
        record_compaction_event(&mut tel, dummy_event("summarize", 8000, 2000, 7));
        assert!(mark_compaction_re_ask(&mut tel, 7, true));
        assert!(!mark_compaction_re_ask(&mut tel, 999, false));
        let jsonl = compaction_telemetry_jsonl(&tel);
        assert!(jsonl.contains("\"downstream_re_ask\":true"));
    }

    #[test]
    fn p5_1_drain_clears_ring() {
        let mut tel = new_compaction_telemetry();
        record_compaction_event(&mut tel, dummy_event("summarize", 8000, 2000, 1));
        let jsonl = drain_compaction_telemetry_jsonl(&mut tel);
        assert!(jsonl.contains("summarize"));
        assert!(tel.is_empty());
    }

    #[test]
    fn p5_1_stats_json_aggregates_per_strategy() {
        let mut tel = new_compaction_telemetry();
        record_compaction_event(&mut tel, dummy_event("summarize", 8000, 2000, 1));
        record_compaction_event(&mut tel, dummy_event("summarize", 9000, 4000, 2));
        record_compaction_event(&mut tel, dummy_event("tool_collapse", 7000, 3000, 3));
        let json = compaction_telemetry_stats_json(&tel);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        // re_ask_rate is null when no events labelled.
        for row in arr {
            assert!(row["count"].as_u64().unwrap() >= 1);
            assert!(row["re_ask_rate"].is_null());
        }
    }

    // ── P5.2 — Bradley–Terry scorer bridge ───────────────────────────────

    fn labelled_event(strategy: &str, turn: u32, re_asked: bool) -> CompactionEvent {
        let mut ev = dummy_event(strategy, 8000, 2000, turn);
        ev.downstream_re_ask = Some(re_asked);
        ev
    }

    #[test]
    fn p5_2_fit_bradley_terry_assigns_higher_score_to_safer_strategy() {
        // summarize: 3 safe, 0 bad; tool_collapse: 0 safe, 3 bad.
        // Bucketed pairs make summarize the consistent winner.
        let events = vec![
            labelled_event("summarize", 1, false),
            labelled_event("tool_collapse", 2, true),
            labelled_event("summarize", 3, false),
            labelled_event("tool_collapse", 4, true),
            labelled_event("summarize", 5, false),
            labelled_event("tool_collapse", 6, true),
        ];
        let model = fit_bradley_terry(&events);
        let s_summ = model.scores.get("summarize").copied().unwrap_or(f64::NAN);
        let s_tool = model
            .scores
            .get("tool_collapse")
            .copied()
            .unwrap_or(f64::NAN);
        assert!(
            s_summ > s_tool,
            "summarize ({s_summ}) should beat tool_collapse ({s_tool})"
        );
    }

    #[test]
    fn p5_2_train_from_jsonl_round_trips_through_drain() {
        let mut tel = new_compaction_telemetry();
        for ev in [
            labelled_event("summarize", 1, false),
            labelled_event("tool_collapse", 2, true),
            labelled_event("summarize", 3, false),
            labelled_event("tool_collapse", 4, true),
        ] {
            record_compaction_event(&mut tel, ev);
        }
        let jsonl = drain_compaction_telemetry_jsonl(&mut tel);
        let model = train_bradley_terry_from_jsonl(&jsonl);
        assert!(model.scores.contains_key("summarize"));
        assert!(model.scores.contains_key("tool_collapse"));
    }

    #[test]
    fn p5_2_snapshot_and_load_round_trip() {
        let mut model = BradleyTerryModel::default();
        model.scores.insert("summarize".into(), 0.7);
        model.scores.insert("tool_collapse".into(), -0.4);
        model.iterations = 42;
        let json = snapshot_bradley_terry_json(&model).unwrap();
        let restored = load_bradley_terry_json(&json).unwrap();
        assert_eq!(restored.iterations, 42);
        assert_eq!(restored.scores.get("summarize"), Some(&0.7));
    }

    #[test]
    fn p5_2_load_malformed_json_fails_closed() {
        let err = load_bradley_terry_json("not json").unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    // ── P5.3 — Learned selector bridge ───────────────────────────────────

    fn trained_model() -> BradleyTerryModel {
        let mut m = BradleyTerryModel::default();
        m.scores.insert("summarize".into(), 0.9);
        m.scores.insert("tool_collapse".into(), -0.2);
        m.scores.insert("sliding_window".into(), 0.1);
        m.iterations = 10;
        m
    }

    #[test]
    fn p5_3_rank_orders_by_learned_score_in_learned_mode() {
        let sel = set_selector_mode(
            new_learned_selector(trained_model()),
            SelectionMode::Learned,
        );
        let candidates = ["tool_collapse", "summarize", "sliding_window"];
        let ranked = rank_candidates(&sel, &candidates);
        assert_eq!(ranked, vec!["summarize", "sliding_window", "tool_collapse"]);
    }

    #[test]
    fn p5_3_rank_preserves_input_order_in_heuristic_mode() {
        let sel = set_selector_mode(
            new_learned_selector(trained_model()),
            SelectionMode::Heuristic,
        );
        let candidates = ["tool_collapse", "summarize", "sliding_window"];
        let ranked = rank_candidates(&sel, &candidates);
        assert_eq!(ranked, vec!["tool_collapse", "summarize", "sliding_window"]);
    }

    #[test]
    fn p5_3_select_with_confidence_returns_margin() {
        let sel = set_selector_mode(
            new_learned_selector(trained_model()),
            SelectionMode::Learned,
        );
        let (top, margin) = select_with_confidence(&sel, &["tool_collapse", "summarize"]).unwrap();
        assert_eq!(top, "summarize");
        // 0.9 - (-0.2) = 1.1
        assert!((margin - 1.1).abs() < 1e-9, "margin = {margin}");
    }

    #[test]
    fn p5_3_select_with_confidence_single_candidate_yields_zero_margin() {
        let sel = new_learned_selector(trained_model());
        let (top, margin) = select_with_confidence(&sel, &["summarize"]).unwrap();
        assert_eq!(top, "summarize");
        assert_eq!(margin, 0.0);
    }

    #[test]
    fn p5_3_auto_mode_falls_back_to_heuristic_when_model_empty() {
        // Empty model ⇒ Auto must preserve input order.
        let sel = new_learned_selector(BradleyTerryModel::default());
        let candidates = ["c1", "c2", "c3"];
        let ranked = rank_candidates(&sel, &candidates);
        assert_eq!(ranked, vec!["c1", "c2", "c3"]);
    }

    #[test]
    fn p5_3_rank_empty_candidates_returns_empty() {
        let sel = new_learned_selector(trained_model());
        let ranked = rank_candidates(&sel, &[]);
        assert!(ranked.is_empty());
    }

    // ── P6 production-hardening bridge tests ────────────────────────────

    #[test]
    fn p6_7_mcp_error_kind_label_is_stable() {
        assert_eq!(mcp_error_kind_label(McpErrorKind::Transient), "transient");
        assert_eq!(mcp_error_kind_label(McpErrorKind::Auth), "auth");
        assert_eq!(mcp_error_kind_label(McpErrorKind::NotFound), "not_found");
        assert_eq!(mcp_error_kind_label(McpErrorKind::Permission), "permission");
        assert_eq!(mcp_error_kind_label(McpErrorKind::Permanent), "permanent");
        assert_eq!(mcp_error_kind_label(McpErrorKind::Config), "config");
    }

    #[test]
    fn p6_7_only_transient_is_retryable() {
        assert!(mcp_error_kind_is_retryable(McpErrorKind::Transient));
        for k in [
            McpErrorKind::Permanent,
            McpErrorKind::Auth,
            McpErrorKind::Config,
            McpErrorKind::NotFound,
            McpErrorKind::Permission,
        ] {
            assert!(
                !mcp_error_kind_is_retryable(k),
                "{:?} must NOT be retryable",
                k
            );
        }
    }

    #[tokio::test]
    async fn p6_8_shared_context_audit_records_writes() {
        let ctx = new_shared_context_with_audit();
        ctx.write("a", "1").await;
        ctx.write("b", "2").await;
        let json = recorded_writes_json(&ctx).await;
        assert!(
            json.contains("\"a\""),
            "audit json must include first key: {json}"
        );
        assert!(
            json.contains("\"b\""),
            "audit json must include second key: {json}"
        );
        assert!(!shared_context_had_collisions(&ctx).await);
    }

    #[tokio::test]
    async fn p6_8_collisions_isolated_from_clean_writes() {
        let ctx = new_shared_context_with_audit();
        ctx.write("k", "v1").await;
        ctx.write("k", "v2").await; // collision
        let collisions = recorded_collisions_json(&ctx).await;
        assert!(
            collisions.contains("\"overwrote\":true"),
            "collisions must record overwrite: {collisions}"
        );
        assert!(shared_context_had_collisions(&ctx).await);
    }

    #[tokio::test]
    async fn p6_8_audit_disabled_yields_empty_writes() {
        // Default `SharedContext::new()` does NOT enable audit.
        let ctx = SharedContext::new();
        ctx.write("a", "1").await;
        let json = recorded_writes_json(&ctx).await;
        assert_eq!(json, "[]", "audit must be off by default");
    }

    #[test]
    fn p6_9_wrap_round_trips_through_versioned_envelope() {
        let event = AgentEvent::TextDelta {
            text: "hi".to_string(),
        };
        let json = wrap_event_json(&event).expect("wrap");
        assert!(json.contains(&format!("\"v\":{}", BRIDGE_AGENT_EVENT_SCHEMA_VERSION)));
        let (decoded, from_newer) = parse_versioned_event_json(&json).expect("parse");
        assert!(!from_newer);
        match decoded {
            AgentEvent::TextDelta { text } => assert_eq!(text, "hi"),
            other => panic!("unexpected variant after round-trip: {:?}", other),
        }
    }

    #[test]
    fn p6_9_newer_producer_is_flagged() {
        // Hand-craft an envelope claiming a newer schema version.
        let future_v = BRIDGE_AGENT_EVENT_SCHEMA_VERSION + 5;
        let event = AgentEvent::TextDelta {
            text: "future".to_string(),
        };
        let body = serde_json::to_string(&event).unwrap();
        let json = format!("{{\"v\":{},\"event\":{}}}", future_v, body);
        let (_decoded, from_newer) = parse_versioned_event_json(&json).expect("parse");
        assert!(from_newer, "newer schema version must be flagged");
    }

    #[test]
    fn p6_9_malformed_envelope_returns_err() {
        let result = parse_versioned_event_json("not-json");
        assert!(result.is_err());
    }

    // P6.10 — defensive contracts: the new bridge-exported types must
    // remain `Send + Sync` (and `Clone` where the IDE relies on cheap
    // sharing) so the IPC layer can hand them across runtime threads
    // without lock-acquisition gymnastics.
    #[test]
    fn p6_10_defensive_contracts_send_sync_clone() {
        fn assert_send_sync<T: Send + Sync>() {}
        fn assert_clone<T: Clone>() {}
        assert_send_sync::<McpErrorKind>();
        assert_clone::<McpErrorKind>();
        assert_send_sync::<VersionedAgentEvent>();
        assert_clone::<VersionedAgentEvent>();
        assert_send_sync::<ContextWriteRecord>();
        // ContextWriteRecord must be cloneable so audit snapshots can
        // be returned by value across IPC.
        assert_clone::<ContextWriteRecord>();
        // ReplayHandle is Clone-shared per its doc comment.
        assert_send_sync::<ReplayHandle>();
        assert_clone::<ReplayHandle>();
    }

    // ── P7 / P8 bridge re-export contracts ──────────────────────────────

    #[test]
    fn p7_1_step_id_is_value_type() {
        // StepId is a Copy newtype around u64; the IDE uses it
        // as a key in event timelines.
        let s = StepId(42);
        let s2 = s; // Copy
        assert_eq!(s.0, s2.0);
    }

    #[test]
    fn p7_2_genai_exporter_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JsonlGenAiExporter>();
        assert_send_sync::<GenAiSpan>();
    }

    #[test]
    fn p7_3_trajectory_types_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TrajectoryRecorder>();
        assert_send_sync::<Trajectory>();
    }

    #[test]
    fn p8_1_step_verifier_object_safe() {
        // The trait must remain object-safe so callers can hold an
        // `Arc<dyn StepVerifier>` in IPC handlers.
        fn assert_object_safe(_: &dyn StepVerifier) {}
        let off = OffStepVerifier;
        assert_object_safe(&off);
    }

    #[test]
    fn p8_4_ensemble_combiner_variants_exposed() {
        // Sanity: every combiner variant is constructible from the bridge.
        let _ = EnsembleCombiner::Mean;
        let _ = EnsembleCombiner::Median;
        let _ = EnsembleCombiner::Threshold(0.5);
    }

    // ── ST-B4a / native-loop-prep-v1 ─────────────────────────────────────

    /// Flag surface: default off, set/get round-trips, cheap handle
    /// observes writes through the bridge.
    #[test]
    fn b4a_native_loop_flag_default_off_and_toggleable() {
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        assert!(
            !bridge.native_loop_enabled(),
            "native loop MUST be off by default — flipping it on is an explicit operator decision"
        );
        bridge.set_native_loop_enabled(true);
        assert!(bridge.native_loop_enabled());

        // Cloned handle sees the same shared atomic.
        let flag = bridge.native_loop_flag();
        assert!(flag.load(std::sync::atomic::Ordering::Acquire));
        bridge.set_native_loop_enabled(false);
        assert!(!flag.load(std::sync::atomic::Ordering::Acquire));

        // Fluent builder.
        let bridge2 = OrchestratorBridge::new(dir.path()).with_native_loop_enabled(true);
        assert!(bridge2.native_loop_enabled());
    }

    /// With the flag OFF, `run_caduceus_loop` MUST refuse to run.
    /// Surprise-activation is exactly the class of regression the flag
    /// exists to prevent.
    #[tokio::test]
    async fn b4a_native_loop_refuses_when_flag_off() {
        use caduceus_orchestrator::AgentEventEmitter;
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path());
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let tools = ToolRegistry::new();
        let (emitter, rx) = AgentEventEmitter::channel(16);
        let harness = AgentHarness::new(provider, tools, 4096, "test").with_emitter(emitter);
        let mut state = bridge.new_session("mock", "mock-model");
        let mut history = OrchestratorBridge::new_history();
        let handle = bridge.new_reducer_handle();

        let res = bridge
            .run_caduceus_loop(&harness, &mut state, &mut history, "hi", handle, rx)
            .await;
        assert!(res.is_err());
        let msg = res.unwrap_err();
        assert!(
            msg.to_string().contains("native loop is disabled"),
            "error MUST name the flag so operators can find it: got {msg:?}"
        );
    }

    /// PARITY: with the flag ON, `run_caduceus_loop` returns the same
    /// assistant text as calling `harness.run` directly AND every
    /// `AgentEvent` emitted during the turn flows through the reducer
    /// handle (non-empty `last_event_id`). This is the core B4a
    /// contract — "the bridge-driven loop is equivalent to the engine
    /// loop *and* exposes live events to the IDE".
    #[tokio::test]
    async fn b4a_native_loop_parity_with_direct_harness_run() {
        use caduceus_core::StopReason;
        use caduceus_orchestrator::AgentEventEmitter;
        use caduceus_providers::ChatResponse;
        use caduceus_providers::mock::MockLlmAdapter;

        fn reply(text: &str) -> ChatResponse {
            ChatResponse {
                content: text.into(),
                input_tokens: 1,
                output_tokens: 1,
                cache_read_tokens: 0,
                cache_creation_tokens: 0,
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                logprobs: None,
                thinking: String::new(),
            }
        }

        let dir = tempfile::tempdir().unwrap();

        // Path A — direct harness.run with a sink-only reducer. No
        // emitter here; reducer sees introspection events only (harness
        // emits none for a plain reply, so projections are empty).
        let bridge_a = OrchestratorBridge::new(dir.path());
        let provider_a: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![reply("pong")]));
        let harness_a = AgentHarness::new(provider_a, ToolRegistry::new(), 4096, "system prompt A");
        let mut state_a = bridge_a.new_session("mock", "mock-model");
        let mut history_a = OrchestratorBridge::new_history();
        let out_a = harness_a
            .run(&mut state_a, &mut history_a, "ping")
            .await
            .expect("direct run must succeed");

        // Path B — same inputs driven through the native loop with
        // the flag ON. Reducer is wired; event_rx is consumed by the
        // forwarder. The emitter is installed on harness_b so text
        // deltas / TurnComplete events flow into the reducer.
        let bridge_b = OrchestratorBridge::new(dir.path()).with_native_loop_enabled(true);
        let provider_b: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![reply("pong")]));
        let (emitter_b, rx_b) = AgentEventEmitter::channel(64);
        let harness_b = AgentHarness::new(provider_b, ToolRegistry::new(), 4096, "system prompt A")
            .with_emitter(emitter_b);
        let mut state_b = bridge_b.new_session("mock", "mock-model");
        let mut history_b = OrchestratorBridge::new_history();
        let reducer_b = bridge_b.new_reducer_handle();

        let out_b = bridge_b
            .run_caduceus_loop(
                &harness_b,
                &mut state_b,
                &mut history_b,
                "ping",
                reducer_b.clone(),
                rx_b,
            )
            .await
            .expect("native-loop run must succeed");

        // 1. The assistant text is byte-equal.
        assert_eq!(
            out_a, out_b,
            "native loop MUST produce the same output as direct harness.run"
        );

        // 2. The forwarder observed at least one event (harness.run
        //    emits at minimum a TurnComplete when it finishes, plus
        //    text deltas from the streaming path). If this is zero,
        //    the forwarder wiring is broken and the IDE would see a
        //    dead timeline.
        assert!(
            reducer_b.last_event_id() > 0,
            "reducer MUST observe events from the native loop — got last_event_id=0"
        );
    }

    /// Error propagation parity: when `harness.run` fails (here: a
    /// pre-cancelled token), `run_caduceus_loop` MUST surface that
    /// error verbatim (modulo the `Display` conversion) and still
    /// cleanly join the forwarder task. A hang here would be
    /// catastrophic — the IDE would wedge.
    #[tokio::test]
    async fn b4a_native_loop_propagates_cancellation_and_joins_forwarder() {
        use caduceus_core::CancellationToken;
        use caduceus_orchestrator::AgentEventEmitter;
        use caduceus_providers::mock::MockLlmAdapter;

        let token = CancellationToken::new();
        token.cancel();

        let dir = tempfile::tempdir().unwrap();
        let bridge = OrchestratorBridge::new(dir.path()).with_native_loop_enabled(true);
        let provider: Arc<dyn LlmAdapter> = Arc::new(MockLlmAdapter::new(vec![]));
        let (emitter, rx) = AgentEventEmitter::channel(8);
        let harness = AgentHarness::new(provider, ToolRegistry::new(), 4096, "sys")
            .with_emitter(emitter)
            .with_cancellation_token(token);
        let mut state = bridge.new_session("mock", "mock-model");
        let mut history = OrchestratorBridge::new_history();
        let reducer = bridge.new_reducer_handle();

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            bridge.run_caduceus_loop(&harness, &mut state, &mut history, "hi", reducer, rx),
        )
        .await
        .expect("run_caduceus_loop MUST NOT hang — forwarder join deadlock is a B4a regression");

        let err = result.expect_err("pre-cancelled token MUST surface as Err");
        assert!(
            err.to_string().contains("Cancelled"),
            "error MUST preserve the underlying cancellation reason: got {err:?}"
        );
    }
}

// ── P9: dynamic mode / lens / tool-gate queries ───────────────────────────────
//
// The IDE (Zed) should hold no mode catalog, no read-only tool ring, and no
// slash-help table. It queries this bridge instead, so adding a mode in the
// engine does not require recompiling the IDE.

/// Mode descriptor consumed by the IDE mode picker, slash-help generator,
/// and any other UI affordance. Kept `Serialize` so it can cross any future
/// process boundary; we don't actually cross one today.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModeDescriptor {
    /// Canonical lowercase name (`plan` | `act` | `research` | `autopilot`).
    pub name: String,
    /// Human-readable label (`"Plan"`, `"Act"`, ...).
    pub label: String,
    /// Short description — what the user gets in this mode.
    pub description: String,
    /// Lenses valid for this mode. Empty for modes with no lens.
    pub lenses: Vec<LensDescriptor>,
}

/// Lens descriptor — today lenses only apply to Act (`execute` / `debug` /
/// `review`). Empty `lenses` vec elsewhere.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LensDescriptor {
    pub name: String,
    pub label: String,
    pub description: String,
}

/// Coarse tool-allowance decision returned to the IDE.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum ToolDecision {
    /// Fire-and-forget allowed.
    Allow,
    /// Allowed but the orchestrator will synthesize a "would do" result
    /// instead of executing (e.g. writes under Plan).
    Intercept { reason: String },
    /// Hard-denied — IDE should surface the reason to the user, NOT retry.
    Deny { reason: String },
}

/// Enumerate the canonical modes plus their lenses. The IDE builds its mode
/// picker from this list; adding a mode in the engine appears in the UI
/// without IDE changes.
pub fn list_modes() -> Vec<ModeDescriptor> {
    use caduceus_orchestrator::modes::{ActLens, AgentMode};
    let lens_desc = |lens: ActLens| match lens {
        ActLens::Normal => LensDescriptor {
            name: "normal".into(),
            label: "Execute".into(),
            description: "Default execution — implement as requested.".into(),
        },
        ActLens::Debug => LensDescriptor {
            name: "debug".into(),
            label: "Debug".into(),
            description: "Trace bugs, inspect failures, propose fixes.".into(),
        },
        ActLens::Review => LensDescriptor {
            name: "review".into(),
            label: "Review".into(),
            description: "Critique diffs — findings list, minimal changes.".into(),
        },
    };
    AgentMode::all_modes()
        .iter()
        .map(|mode| {
            let lenses = match mode {
                AgentMode::Act => vec![
                    lens_desc(ActLens::Normal),
                    lens_desc(ActLens::Debug),
                    lens_desc(ActLens::Review),
                ],
                _ => Vec::new(),
            };
            ModeDescriptor {
                name: mode.name().to_string(),
                label: match mode {
                    // Caduceus: Plan now subsumes the former Research mode,
                    // so surface that combined identity in the picker.
                    AgentMode::Plan => "Plan & Research".to_string(),
                    _ => {
                        let mut chars = mode.name().chars();
                        match chars.next() {
                            Some(c) => c.to_uppercase().chain(chars).collect(),
                            None => String::new(),
                        }
                    }
                },
                description: mode.description().to_string(),
                lenses,
            }
        })
        .collect()
}

/// Return the lenses valid for a mode string. Empty vec for unknown modes or
/// modes without lenses.
pub fn list_lenses_for(mode: &str) -> Vec<LensDescriptor> {
    list_modes()
        .into_iter()
        .find(|m| m.name == mode)
        .map(|m| m.lenses)
        .unwrap_or_default()
}

/// Return the full layered mode prompt for a (mode, lens) pair — this is what
/// the engine's `effective_system_prompt` prepends. IDEs that build their own
/// system prompt MUST source the mode section from here so there's a single
/// source of truth.
///
/// Lens defaults to `normal` when None or unknown.
pub fn mode_prompt_for_profile(mode: &str, lens: Option<&str>) -> String {
    use caduceus_orchestrator::AgentHarness;
    use caduceus_orchestrator::modes::{ActLens, AgentMode};
    let agent_mode = AgentMode::from_str_loose(mode).unwrap_or(AgentMode::Plan);
    let lens = lens
        .and_then(ActLens::from_str_loose)
        .unwrap_or(ActLens::Normal);
    // F1: prepend the engine's behavior_rules preamble so the IDE's rendered
    // prompt matches what the engine's `effective_system_prompt` produces —
    // no hand-rolled mode text drifting from source of truth.
    let preamble = AgentHarness::behavior_rules_preamble();
    let mode_prefix = agent_mode.config_with_lens(lens).system_prompt_prefix;
    format!("{preamble}\n\n{mode_prefix}")
}

/// Coarse "does this mode allow writes without interception?" query. IDEs use
/// this as a last-mile gate when they don't carry a full envelope. Research
/// mode is reported as allowing writes because it allows markdown — the fine-
/// grained file-extension check lives in the engine's envelope preflight.
///
/// Callers SHOULD additionally route through `envelope.preflight(...)` for
/// the authoritative decision; this helper exists only to drive UI affordances.
pub fn mode_allows_writes(mode: &str) -> bool {
    use caduceus_orchestrator::modes::AgentMode;
    match AgentMode::from_str_loose(mode).unwrap_or(AgentMode::Plan) {
        // Plan (which now also covers former Research) allows markdown writes;
        // engine extension filter blocks code writes. Returning true here so
        // UI affordances permit the .md write attempt; envelope.preflight is
        // the authoritative gate.
        AgentMode::Plan => true,
        AgentMode::Act => true,
        AgentMode::Autopilot => true,
    }
}

/// Read-only tool allowlist (Plan + Research) — the **single source of truth**.
///
/// Contract `mode-policy-shim-v1` (decomposition §5): the Zed IDE MUST NOT
/// carry a local copy of this list. `thread.rs::is_tool_allowed_in_current_mode`
/// delegates to this function. When a new read-only tool ships, it is added
/// here and both the engine and the IDE pick it up without a second edit.
///
/// Entries split into two classes:
///   1. Zed built-in read tools (local only, no network side effects)
///   2. Caduceus read-only tools (strictly no state mutation)
///
/// Any tool not in this set is treated as write-ish and gated by mode.
fn is_read_only_tool(tool_name: &str) -> bool {
    const READ_ONLY_TOOLS: &[&str] = &[
        // ── Zed built-in read tools ────────────────────────────────────
        "read_file",
        "find_path",
        "grep",
        "list_directory",
        "diagnostics",
        "now",
        "open",
        // Thread introspection — pure reads of past sessions, no mutation.
        "read_thread",
        "compact_thread",
        "suggest_models",
        // ── Caduceus read-only tools ───────────────────────────────────
        "caduceus_semantic_search",
        "caduceus_index",
        "caduceus_code_graph",
        "caduceus_tree_sitter",
        "caduceus_git_read",
        "caduceus_memory_read",
        "caduceus_dependency_scan",
        "caduceus_security_scan",
        "caduceus_error_analysis",
        "caduceus_mcp_security",
        "caduceus_prd",
        "caduceus_progress",
        "caduceus_telemetry",
        "caduceus_conversation",
        "caduceus_marketplace",
        "caduceus_project",
        "caduceus_task_tree",
        "caduceus_time_tracking",
        "caduceus_policy",
        "caduceus_cross_search",
        "caduceus_api_registry",
        "caduceus_architect",
        "caduceus_product",
    ];
    READ_ONLY_TOOLS.contains(&tool_name)
}

/// Authoritative mode × tool gate — returns `true` when the mode allows the
/// tool to dispatch. This is the **single source of truth** the IDE calls
/// before dispatching every tool; the old hardcoded `thread.rs` allowlist is
/// retired.
///
/// Decision table:
///   - `plan`, `research` → only read-only tools dispatch; writes bounce
///     (Plan→intercept, Research→markdown-only via envelope preflight).
///   - `act`, `autopilot` → every tool dispatches; per-folder write gates
///     are enforced by the engine's `PermissionEnvelope::preflight`.
///
/// Tools not classified as read-only are permitted only in write-capable
/// modes. Callers MUST additionally route through `envelope.preflight(...)`
/// for path-level authority — this function only covers the mode-level gate.
pub fn mode_allows_tool(mode: &str, tool_name: &str) -> bool {
    // `spawn_agent` is the one tool exempt from mode gating — it transfers
    // work to a sub-agent whose own envelope decides what it can do.
    if tool_name == "spawn_agent" {
        return true;
    }
    if is_read_only_tool(tool_name) {
        return true;
    }
    // Any remaining tool is a write-capable tool. Gate on mode.
    mode_allows_writes(mode)
}

// ── P13c — catalog endpoints (personas, skills, models) ────────────────────
//
// The IDE renders the "who / what / how" of the agents-DAG and features-DAG
// by querying these at startup + on-change. No hardcoded lists in the IDE.

/// A persona the engine can spawn — e.g. `rubber-duck`, `cloud-architect`.
///
/// The Agents-DAG nodes (see `AssignmentSummaryV1`) carry a `persona_id`
/// that MUST exist in this catalog; the IDE uses this to render labels,
/// tooltips, and default-mode badges.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PersonaDescriptor {
    pub name: String,
    pub description: String,
    /// Default mode this persona ships with (e.g. `plan`, `act`). Opaque
    /// to the IDE — passed back to `mode_prompt_for_profile` if needed.
    pub default_mode: String,
    /// Tools this persona is pre-registered for. Advisory — the envelope
    /// is the source of truth on what's actually allowed at runtime.
    pub preferred_tools: Vec<String>,
    /// Bounded model knobs — useful for UI cost/latency estimates.
    pub temperature: f64,
    pub max_tokens: u32,
}

/// Enumerate the built-in personas the orchestrator can dispatch. Returns
/// a deterministic alphabetical order so the IDE mode-picker / sidebar is
/// stable across runs.
pub fn list_personas() -> Vec<PersonaDescriptor> {
    use caduceus_orchestrator::modes::PersonaRegistry;
    let reg = PersonaRegistry::builtin_personas();
    reg.list()
        .into_iter()
        .map(|p| PersonaDescriptor {
            name: p.name.clone(),
            description: p.description.clone(),
            default_mode: p.default_mode.clone(),
            preferred_tools: p.preferred_tools.clone(),
            temperature: p.temperature,
            max_tokens: p.max_tokens,
        })
        .collect()
}

/// A skill the engine can activate from the workspace's `.caduceus/skills/`.
///
/// The IDE can render the Features-DAG with skill badges and let the user
/// preview the full body on hover. `body_chars` is exposed separately from
/// `body` so the UI can render a char-count without paying for the full
/// prose until it's actually needed (all bundled skills already fit inside
/// `MAX_INSTRUCTION_FILE_CHARS`, but the split keeps future truncation
/// policies cheap).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SkillDescriptor {
    pub name: String,
    pub description: String,
    /// Full prose body. May have been truncated at load time (P3 loader);
    /// check `budget_hint_chars` for the per-skill cap.
    pub body: String,
    pub body_chars: usize,
    pub trigger_phrases: Vec<String>,
    pub budget_hint_chars: Option<usize>,
}

/// Enumerate skills loaded from the workspace. Empty Vec for workspaces
/// with no `.caduceus/skills/` dir. Result is alphabetized by name.
///
/// The IDE SHOULD call this on workspace open + when `.caduceus/skills/`
/// mutates on disk (file watcher). Not event-driven via `AgentEvent`
/// today — add a `SkillCatalogChanged` event if UIs need live updates
/// without polling.
pub fn list_bundled_skills(workspace_root: &Path) -> Result<Vec<SkillDescriptor>, BridgeError> {
    let set = InstructionLoader::new(workspace_root)
        .load()
        .map_err(|e| BridgeError::workspace(e.to_string()))?;
    let mut out: Vec<SkillDescriptor> = set
        .available_skills
        .into_iter()
        .map(|s| SkillDescriptor {
            name: s.name,
            description: s.description,
            body_chars: s.body.chars().count(),
            body: s.body,
            trigger_phrases: s.trigger_phrases,
            budget_hint_chars: s.budget_hint_chars,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Coarse vendor + tier catalog for models the orchestrator can route to.
///
/// Exposes ONLY vendor + tier + a human label + the safety/cost class —
/// never exact `model_id` strings. Exact ids are operator-configured and
/// considered sensitive; they appear in `AssignmentSummaryV1.model_id_exact`
/// only when `include_sensitive=true` is threaded from the bridge layer
/// (P13d). This keeps rogue UIs or log scrapers from lifting the full
/// routing config.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ModelDescriptor {
    pub vendor: String,
    pub tier: String,
    pub label: String,
    pub latency_class: String,
    pub cost_class: String,
}

/// Return the vendor/tier catalog. Deliberately small — just Anthropic
/// and OpenAI tiers that the orchestrator actively routes to. Adding a
/// new vendor/tier here does NOT enable it in routing; that's operator
/// config. This is purely the rendering vocabulary.
pub fn list_models() -> Vec<ModelDescriptor> {
    vec![
        ModelDescriptor {
            vendor: "anthropic".into(),
            tier: "opus".into(),
            label: "Anthropic · Opus tier".into(),
            latency_class: "slow".into(),
            cost_class: "high".into(),
        },
        ModelDescriptor {
            vendor: "anthropic".into(),
            tier: "sonnet".into(),
            label: "Anthropic · Sonnet tier".into(),
            latency_class: "medium".into(),
            cost_class: "medium".into(),
        },
        ModelDescriptor {
            vendor: "anthropic".into(),
            tier: "haiku".into(),
            label: "Anthropic · Haiku tier".into(),
            latency_class: "fast".into(),
            cost_class: "low".into(),
        },
        ModelDescriptor {
            vendor: "openai".into(),
            tier: "opus".into(),
            label: "OpenAI · top tier (GPT-5-class)".into(),
            latency_class: "slow".into(),
            cost_class: "high".into(),
        },
        ModelDescriptor {
            vendor: "openai".into(),
            tier: "mini".into(),
            label: "OpenAI · mini tier".into(),
            latency_class: "fast".into(),
            cost_class: "low".into(),
        },
    ]
}

#[cfg(test)]
mod p9_tests {
    use super::*;

    #[test]
    fn list_modes_returns_3_canonical_modes_with_act_having_3_lenses() {
        let modes = list_modes();
        assert_eq!(modes.len(), 3);
        let names: Vec<&str> = modes.iter().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"plan"));
        assert!(names.contains(&"act"));
        assert!(names.contains(&"autopilot"));
        // Research has been merged into Plan; it should NOT appear.
        assert!(!names.contains(&"research"));
        let act = modes.iter().find(|m| m.name == "act").unwrap();
        assert_eq!(act.lenses.len(), 3);
        assert_eq!(act.label, "Act");
        // Other modes have no lenses.
        for name in ["plan", "autopilot"] {
            let m = modes.iter().find(|m| m.name == name).unwrap();
            assert!(m.lenses.is_empty(), "{name} should have no lenses");
            assert!(!m.description.is_empty(), "{name} missing description");
        }
    }

    #[test]
    fn list_lenses_for_unknown_mode_is_empty() {
        assert!(list_lenses_for("banana").is_empty());
        assert!(list_lenses_for("plan").is_empty());
    }

    #[test]
    fn list_lenses_for_act_has_three() {
        let ls = list_lenses_for("act");
        let names: Vec<&str> = ls.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["normal", "debug", "review"]);
    }

    #[test]
    fn mode_prompt_for_profile_surfaces_mode_text() {
        // Plan prompt mentions PLAN and read-only semantics.
        let p = mode_prompt_for_profile("plan", None);
        assert!(p.contains("PLAN"));
        // Act prompt changes under the Debug lens.
        let act_exec = mode_prompt_for_profile("act", Some("normal"));
        let act_debug = mode_prompt_for_profile("act", Some("debug"));
        assert_ne!(act_exec, act_debug);
        assert!(act_debug.contains("Debug") || act_debug.contains("debug"));
        // Legacy lens string "dbg" is accepted.
        let act_dbg = mode_prompt_for_profile("act", Some("dbg"));
        assert_eq!(act_dbg, act_debug);
        // Unknown mode falls back to Plan.
        let unknown = mode_prompt_for_profile("banana", None);
        assert_eq!(unknown, p);
    }
}

#[cfg(test)]
mod p13c_catalog_tests {
    use super::*;

    #[test]
    fn list_personas_includes_domain_specialists_and_is_alphabetized() {
        let p = list_personas();
        assert!(!p.is_empty());
        // Must expose the 7 domain specialists used by the critique fan-out.
        for want in [
            "rubber-duck",
            "cloud-architect",
            "ml-architect",
            "data-engineer",
            "data-researcher",
            "data-scientist",
            "qa-strategist",
        ] {
            assert!(
                p.iter().any(|pp| pp.name == want),
                "missing persona {want} in catalog"
            );
        }
        // Alphabetical order — stable UI.
        let mut sorted: Vec<&str> = p.iter().map(|pp| pp.name.as_str()).collect();
        sorted.sort();
        let got: Vec<&str> = p.iter().map(|pp| pp.name.as_str()).collect();
        assert_eq!(got, sorted, "personas must be returned alphabetized");
    }

    #[test]
    fn list_personas_default_mode_is_valid_mode_name() {
        let modes: Vec<String> = list_modes().into_iter().map(|m| m.name).collect();
        for p in list_personas() {
            assert!(
                modes.contains(&p.default_mode),
                "persona {} has unknown default_mode {}",
                p.name,
                p.default_mode
            );
        }
    }

    #[test]
    fn list_bundled_skills_empty_workspace_is_ok_and_empty() {
        // Empty tempdir — no .caduceus/skills/ at all.
        let tmp = tempfile::tempdir().unwrap();
        let got = list_bundled_skills(tmp.path()).expect("load succeeds");
        assert!(got.is_empty(), "empty workspace → empty skills catalog");
    }

    #[test]
    fn list_bundled_skills_reads_dir_based_layout_and_sorts() {
        use std::fs;
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let skills_dir = root.join(".caduceus").join("skills");
        fs::create_dir_all(&skills_dir).unwrap();
        for name in ["zebra", "alpha", "mango"] {
            let d = skills_dir.join(name);
            fs::create_dir_all(&d).unwrap();
            fs::write(
                d.join("SKILL.md"),
                format!(
                    "---\nname: {name}\ndescription: {name} does a thing.\n---\n\nBody for {name}.\n"
                ),
            )
            .unwrap();
        }
        let got = list_bundled_skills(root).expect("load succeeds");
        assert_eq!(got.len(), 3);
        let names: Vec<&str> = got.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "mango", "zebra"], "alphabetized");
        for s in &got {
            assert!(!s.description.is_empty());
            assert!(s.body.contains("Body for"));
            assert_eq!(s.body.chars().count(), s.body_chars);
        }
    }

    #[test]
    fn list_models_exposes_vendor_tier_but_no_exact_ids() {
        let m = list_models();
        assert!(!m.is_empty());
        // At minimum, Anthropic opus + OpenAI opus tier must be present —
        // these are the primary orchestrator + cross-check vendors.
        assert!(
            m.iter()
                .any(|x| x.vendor == "anthropic" && x.tier == "opus")
        );
        assert!(m.iter().any(|x| x.vendor == "openai" && x.tier == "opus"));
        // Cost / latency classes are present on every entry.
        for entry in &m {
            assert!(!entry.latency_class.is_empty());
            assert!(!entry.cost_class.is_empty());
            assert!(!entry.label.is_empty());
        }
        // Security: descriptor must not leak exact model ids. Serialize
        // and assert no `model_id` field exists in the wire format.
        let wire = serde_json::to_string(&m[0]).unwrap();
        assert!(
            !wire.contains("model_id"),
            "list_models() wire format must NOT expose exact model_id — got {wire}"
        );
    }

    #[test]
    fn list_models_is_deterministic_across_calls() {
        assert_eq!(list_models(), list_models());
    }

    // ─────────────────────────────────────────────────────────────────────
    // GPUI × tokio crash regression tests.
    //
    // Both crashes from NW-4 dogfood root-caused to the same class of
    // bug: calling `tokio::spawn` / `tokio::time::*` on a thread that
    // does not have a Tokio runtime installed (e.g. GPUI's foreground
    // executor, which is smol-based). The default `#[tokio::test]`
    // macro installs a runtime on the test thread and hides this bug.
    //
    // These tests deliberately run as plain `#[test]` so the caller
    // thread has NO ambient runtime. They prove the contract:
    //
    //   * `spawn_forwarder` panics without a runtime (documents the
    //     requirement).
    //   * `spawn_forwarder` works when wrapped in
    //     `tokio_rt::bridge_runtime_handle().enter()` (the pattern
    //     thread.rs::try_run_turn_native uses).
    //
    // If someone deletes the `_bridge_rt_guard` line from thread.rs,
    // the production code crashes at runtime the same way NW-4 did —
    // but these tests still pass, because they test the bridge helpers
    // directly. The matching "thread.rs calls this with the guard"
    // contract is enforced by the doc comment on
    // `tokio_rt::bridge_runtime_handle`.
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn spawn_forwarder_panics_without_ambient_runtime() {
        use std::panic::AssertUnwindSafe;

        let (_tx, rx) = tokio::sync::mpsc::channel::<AgentEvent>(1);
        // No runtime on this thread; `tokio::spawn` inside
        // `spawn_forwarder` must `TryCurrentError`-panic. We use
        // catch_unwind so the test itself doesn't abort.
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = spawn_forwarder(rx, |_ev| {});
        }));
        assert!(
            result.is_err(),
            "spawn_forwarder must panic without an ambient Tokio runtime; \
             if this ever starts passing, the bridge either no longer \
             requires a runtime (re-audit thread.rs guard) OR silently \
             spawns into a hidden runtime (leak risk)."
        );
    }

    #[test]
    fn spawn_forwarder_works_under_bridge_runtime_enter_guard() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        // Mirror what thread.rs::try_run_turn_native does: install the
        // process-wide bridge runtime as the ambient runtime for this
        // thread, then call into bridge helpers.
        let handle = crate::tokio_rt::bridge_runtime_handle();
        let _guard = handle.enter();

        let seen = Arc::new(AtomicUsize::new(0));
        let seen_cb = Arc::clone(&seen);

        // The forwarder task needs a runtime to `.block_on`-style
        // drive it to completion. We reuse the same bridge runtime
        // that the guard points at.
        handle.block_on(async move {
            let (tx, rx) = tokio::sync::mpsc::channel::<AgentEvent>(4);
            let (jh, drain_tx) = spawn_forwarder(rx, move |_ev| {
                seen_cb.fetch_add(1, Ordering::SeqCst);
            });

            tx.send(AgentEvent::Error {
                message: "regression-test".into(),
            })
            .await
            .unwrap();

            drop(tx);
            let _ = drain_tx.send(());
            await_forwarder_with_timeout(jh, "regression_forwarder").await;
        });

        assert_eq!(
            seen.load(Ordering::SeqCst),
            1,
            "forwarder must drain exactly one event under the bridge runtime enter guard"
        );
    }
}
