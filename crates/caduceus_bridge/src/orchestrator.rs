//! Orchestrator bridge — agent harness, conversation history, instructions,
//! PRD parsing, task management, progress inference, scaffolding, and tree visualization.

use caduceus_orchestrator::{
    AgentHarness, ConversationHistory, execute_tool_calls,
    AgentEventEmitter,
    instructions::{self, InstructionLoader, InstructionSet},
    PrdParser, PrdTask, TaskRecommender, TaskRecommendation,
    ProgressInferrer, InferredProgress,
    AgentScaffolder, SkillScaffolder,
    ExecutionTreeViz,
    TimeTracker,
    TaskTree, HierarchicalTask,
};
use caduceus_core::AgentEvent;
use caduceus_core::{ModelId, ProviderId, SessionState};
use caduceus_mcp::{McpServerManager, mcp_tool_bridge::{McpToolBridge, McpInvoker}};
use caduceus_providers::LlmAdapter;
use caduceus_tools::ToolRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Re-export orchestrator types for consumers.
pub use caduceus_orchestrator::{
    PrdTask as BridgePrdTask,
    TaskRecommendation as BridgeTaskRecommendation,
    InferredProgress as BridgeInferredProgress,
    ExecutionTreeViz as BridgeExecutionTreeViz,
    VizTreeNode as BridgeVizTreeNode,
    TimeTracker as BridgeTimeTracker,
    TimeEntry as BridgeTimeEntry,
    TaskTree as BridgeTaskTree,
    HierarchicalTask as BridgeHierarchicalTask,
    kanban::{KanbanBoard, KanbanCard, KanbanColumn, CardStatus},
    automations::{Automation, AutomationTrigger, AutomationAgentConfig, AutomationRegistry},
    context::{self, ContextZone, ContextSource, ContextAssembler, AssembledContext, estimate_tokens, ContextManager, PinnedContext},
    compaction::{self, CompactMessage, CompactionPipeline, ContextStats, CompactionTrigger},
    workers::{TaskDAG, TaskDefinition, TaskStatus, SharedContext, TeamResult, AgentConfig},
    background::{BackgroundAgent, BackgroundStatus},
    headless::{HeadlessConfig, HeadlessResult, OutputFormat as HeadlessOutputFormat},
    mentions::MentionResolver,
};

// Re-export types needed by tools.
pub use caduceus_orchestrator::modes::AgentMode as BridgeAgentMode;
pub use caduceus_orchestrator::modes::{
    ActionPlan as BridgeActionPlan, AmendError as BridgeAmendError,
    AppliedAmendment as BridgeAppliedAmendment, PlanAmendment as BridgePlanAmendment,
    PlannedAction as BridgePlannedAction,
};
pub use caduceus_core::ModelId as BridgeModelId;

/// P3.1 — Apply a [`BridgePlanAmendment`] to a [`BridgeActionPlan`] from the
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
    plan: &mut BridgeActionPlan,
    amendment_json: &str,
) -> Result<BridgeAppliedAmendment, BridgeAmendError> {
    let amendment: BridgePlanAmendment = serde_json::from_str(amendment_json)
        .map_err(|_| BridgeAmendError::StaleRevision {
            expected: 0,
            actual: plan.revision,
        })?;
    plan.apply_amendment(amendment)
}

/// P3.1 — Render a snapshot of the current [`BridgeActionPlan`] as JSON for
/// the UI. The shape mirrors the per‑step revision so the React panel can
/// detect divergence on the next amendment attempt.
pub fn snapshot_plan_json(plan: &BridgeActionPlan) -> Result<String, String> {
    serde_json::to_string(plan).map_err(|e| e.to_string())
}

/// Exact token count using tiktoken (cl100k_base, used by GPT-4/Claude).
/// Falls back to the heuristic estimate if tokenizer initialization fails.
pub fn count_tokens_exact(text: &str) -> u32 {
    use std::sync::OnceLock;
    static BPE: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();

    let bpe = BPE.get_or_init(|| {
        tiktoken_rs::get_bpe_from_model("gpt-4o").ok()
    });

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
    reflexion:
        Option<Arc<std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>>>,
    /// Optional Tree-of-Thoughts planner config (P12.3) attached to
    /// every harness.
    tot_config: Option<caduceus_orchestrator::branching_planner::PlannerConfig>,
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
}

impl OrchestratorBridge {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            speculative_cache: None,
            reflexion: None,
            tot_config: None,
        }
    }

    // ── P12 primitive wiring (production side) ───────────────────────────

    /// Attach a [`SpeculativeCache`](caduceus_tools::SpeculativeCache) to
    /// every subsequent harness produced by `build_harness*`. Cheaply
    /// cloneable; the same `Arc`-backed map is shared by the bridge,
    /// harness, and any external prefetcher (UI worker) that calls
    /// [`OrchestratorBridge::speculative_cache`].
    pub fn with_speculative_cache(
        mut self,
        cache: caduceus_tools::SpeculativeCache,
    ) -> Self {
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
        memory: Arc<
            std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>,
        >,
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
    ) -> Option<&Arc<std::sync::Mutex<caduceus_orchestrator::reflexion::ReflexionMemory>>>
    {
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
    pub fn with_tot_config(
        mut self,
        cfg: caduceus_orchestrator::branching_planner::PlannerConfig,
    ) -> Self {
        self.tot_config = Some(cfg);
        self
    }

    /// Convenience: install the planner config defaults
    /// (`PlannerConfig::default`).
    pub fn with_default_tot_config(self) -> Self {
        self.with_tot_config(
            caduceus_orchestrator::branching_planner::PlannerConfig::default(),
        )
    }

    /// Borrow the bound ToT planner config (if any).
    pub fn tot_config(
        &self,
    ) -> Option<&caduceus_orchestrator::branching_planner::PlannerConfig> {
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
        if let Some(cfg) = self.tot_config.clone() {
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
                        Ok(value) => Ok(caduceus_core::ToolResult::success(
                            value.to_string(),
                        )),
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
        let live: std::collections::HashSet<String> =
            defs.iter().map(|d| d.name.clone()).collect();

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
    pub fn load_instructions(&self) -> Result<InstructionSet, String> {
        let loader = InstructionLoader::new(&self.project_root);
        loader.load().map_err(|e| e.to_string())
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
    pub fn conversation_serialize(history: &ConversationHistory) -> Result<String, String> {
        history.serialize().map_err(|e| e.to_string())
    }

    /// Deserialize conversation history from JSON.
    pub fn conversation_deserialize(json: &str) -> Result<ConversationHistory, String> {
        ConversationHistory::deserialize(json).map_err(|e| e.to_string())
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
        self.build_harness_with_approval(provider, tools, system_prompt, Self::DEFAULT_APPROVAL_TOOLS)
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
        let base = AgentHarness::new(provider, tools, 200_000, system_prompt)
            .with_tool_timeout(std::time::Duration::from_secs(120))
            .with_instructions(&self.project_root);
        let base = self.attach_p12(base);
        base.with_approval_flow(approval_tools.iter().map(|s| s.as_ref().to_string()))
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
        let base = AgentHarness::new(provider, tools, 200_000, system_prompt)
            .with_tool_timeout(std::time::Duration::from_secs(120))
            .with_instructions(&self.project_root);
        self.attach_p12(base)
    }

    // ── Emitter / replay seam (G17) ──────────────────────────────────────

    /// Default in-flight buffer for the emitter mpsc channel handed to the
    /// IDE. Sized to absorb a brief render stall (~1 second of text deltas
    /// at 256 events/s) without dropping live events; events that don't
    /// fit are still captured by the emitter's retention ring.
    pub const DEFAULT_EVENT_CHANNEL_BUFFER: usize = 256;

    /// Build a production harness wired with HITL approval AND an event
    /// emitter that the IDE can both stream from (`event_rx`) and *replay*
    /// from (`replay_handle`) on UI reattach.
    ///
    /// Closes gap G17: `AgentHarness` was already P1.4-retention-enabled,
    /// but no surface in the bridge actually exposed `replay()` to the IDE,
    /// making the retention ring a dead seam. The `replay_handle` is a
    /// cheap `Clone` of the same emitter held by the harness; events are
    /// observable through both.
    ///
    /// Returns `(harness, approval_tx, replay_handle, event_rx)`. The
    /// caller MUST consume `event_rx` (drop = dead channel = G27 territory)
    /// and stash `replay_handle` for the agent panel to call on mount.
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
        let (emitter, event_rx) =
            AgentEventEmitter::channel(Self::DEFAULT_EVENT_CHANNEL_BUFFER);
        let replay_handle = ReplayHandle::new(emitter.clone());
        let base = AgentHarness::new(provider, tools, 200_000, system_prompt)
            .with_tool_timeout(std::time::Duration::from_secs(120))
            .with_instructions(&self.project_root)
            .with_emitter(emitter);
        let base = self.attach_p12(base);
        let (harness, approval_tx) =
            base.with_approval_flow(Self::DEFAULT_APPROVAL_TOOLS.iter().map(|s| s.to_string()));
        (harness, approval_tx, replay_handle, event_rx)
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

    /// Run a single agent turn (non-streaming).
    pub async fn run_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, String> {
        harness.run_turn(state, user_input).await.map_err(|e| e.to_string())
    }

    /// Run a single agent turn with streaming.
    pub async fn stream_turn(
        harness: &AgentHarness,
        state: &mut SessionState,
        user_input: &str,
    ) -> Result<String, String> {
        harness.stream_turn(state, user_input).await.map_err(|e| e.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_orchestrator::VizTreeNode;

    #[test]
    fn orchestrator_new_history() {
        let history = OrchestratorBridge::new_history();
        assert!(history.is_empty());
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
        let memories = OrchestratorBridge::extract_memories(
            "What is 2+2?",
            "4.",
        );
        assert!(memories.is_empty(), "No preference signal");
    }

    #[test]
    fn default_approval_tools_covers_destructive_set() {
        let set: std::collections::HashSet<_> =
            OrchestratorBridge::DEFAULT_APPROVAL_TOOLS.iter().copied().collect();
        // Regression guard: removing any of these silently disables HITL for
        // a tool that mutates user state. If you intentionally drop one,
        // delete the assertion together with the constant entry.
        for required in ["bash", "shell", "write_file", "edit_file", "delete_file", "apply_patch"] {
            assert!(set.contains(required), "DEFAULT_APPROVAL_TOOLS missing {required}");
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
        let tasks = OrchestratorBridge::parse_prd(
            "# Auth\nBuild auth.\n# API\nRequires Auth module.",
        );
        let deps = OrchestratorBridge::infer_dependencies(&tasks);
        // "API" description mentions "Auth", so (1, 0) expected
        assert!(!deps.is_empty());
        assert!(deps.contains(&(1, 0)));
    }

    // ── Task recommendation tests ────────────────────────────────────────

    #[test]
    fn orchestrator_recommend_next() {
        let tasks = OrchestratorBridge::parse_prd(
            "# Setup\nInit project.\n# Build\nBuild features.",
        );
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
        assert!(replay_handle.retention_cap() > 0, "replay handle must wrap a retention-enabled emitter");
        // Empty replay before any events have been emitted.
        let snap = OrchestratorBridge::replay_session_events(&replay_handle);
        assert!(snap.is_empty(), "replay must be empty before any emit");
        // Channel must still be the live receiver, not closed by the build call.
        drop(event_rx);
    }

    #[tokio::test]
    async fn replay_handle_observes_events_emitted_by_harness() {
        use caduceus_providers::mock::MockLlmAdapter;
        use caduceus_core::SessionState;
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
        let cache = caduceus_tools::SpeculativeCache::new(
            std::time::Duration::from_secs(30),
        );
        let bridge = OrchestratorBridge::new(dir.path())
            .with_speculative_cache(cache.clone());
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

        let reflector = HeuristicReflector::default();
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
        use caduceus_orchestrator::branching_planner::PlannerConfig;
        use caduceus_providers::mock::MockLlmAdapter;
        let dir = tempfile::tempdir().unwrap();
        let cfg = PlannerConfig::default();
        let bridge = OrchestratorBridge::new(dir.path()).with_tot_config(cfg.clone());
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
            OrchestratorBridge::register_mcp_tools_diff(mgr.clone(), &mut reg, &prior)
                .await;
        assert_eq!((added, removed), (0, 0));
        assert!(live.is_empty());
        // Idempotent: calling again with the new live set yields zero.
        let (a2, r2, _) =
            OrchestratorBridge::register_mcp_tools_diff(mgr, &mut reg, &live).await;
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
        let cache = caduceus_tools::SpeculativeCache::new(
            std::time::Duration::from_secs(60),
        );
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
        let mut plan = BridgeActionPlan::new();
        plan.add("read_file", &serde_json::json!({"path": "/a"}));
        plan.add("write_file", &serde_json::json!({"path": "/b", "content": "x"}));
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
        let mut plan = BridgeActionPlan::new();
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
            BridgeAmendError::StaleRevision { .. } => {}
            other => panic!("expected StaleRevision, got {other:?}"),
        }
    }

    #[test]
    fn p3_1_malformed_json_returns_stale_not_panic() {
        let mut plan = BridgeActionPlan::new();
        let err = apply_plan_amendment(&mut plan, "{not json").unwrap_err();
        assert!(matches!(err, BridgeAmendError::StaleRevision { .. }));
    }

    #[test]
    fn p3_1_insert_amendment_renumbers_subsequent_steps() {
        let mut plan = BridgeActionPlan::new();
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
        let mut plan = BridgeActionPlan::new();
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
        let mut plan = BridgeActionPlan::new();
        plan.add("read_file", &serde_json::json!({"path": "/x"}));
        let json = snapshot_plan_json(&plan).unwrap();
        let restored: BridgeActionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.actions.len(), 1);
        assert_eq!(restored.revision, plan.revision);
    }
}
