//! Index-Access DAG — tracks which agents (sub-agent spawn IDs, slash
//! commands, tool calls) are currently reading or writing the engine's
//! semantic index, code graph, and other shared state. The point is to
//! make concurrent index access *visible* to the user — when two agents
//! query the same index, the dependency edge is recorded and surfaced
//! in the chat UI as an ASCII DAG.
//!
//! This is the data layer for the user request:
//!
//!   "when the same index is being read by agents the dependency
//!    should be clear and a DAG should be generated and also drawn
//!    to the chat"
//!
//! The DAG is a directed graph:
//!
//!   AgentNode (kind: subagent / slash / tool / user) ──reads──► IndexNode
//!   AgentNode ──writes──► IndexNode
//!
//! Two agents that both read the same index get an implicit edge
//! through that shared `IndexNode` — so the rendered DAG immediately
//! shows the contention point.
//!
//! Globally addressable so call sites in `engine::index_directory`,
//! `engine::semantic_search`, agent spawn paths, and the agent_panel
//! UI can all reach the same registry without threading it through
//! every API.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const MAX_EVENTS: usize = 256;
const MAX_SPAWN_EVENTS: usize = 128;
const EVENT_TTL: Duration = Duration::from_secs(60 * 5);

/// Monotonic version counter — bumped on every `record()` and
/// `record_spawn()`. UI panels poll this cheaply to know whether
/// the snapshot has changed since the last render. Saves a full
/// snapshot+rebuild on every paint.
static REGISTRY_VERSION: AtomicU64 = AtomicU64::new(0);

/// Public API: read the current registry version. UI code compares
/// this to the version it last rendered; if unchanged, skip re-render.
pub fn version() -> u64 {
    REGISTRY_VERSION.load(Ordering::Acquire)
}

fn bump_version() {
    REGISTRY_VERSION.fetch_add(1, Ordering::AcqRel);
}

/// Kind of index resource an agent is touching. Keep this enum small —
/// it shows up directly in the rendered DAG as the node label.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexResource {
    /// `engine.search_index` — the semantic chunk index.
    SemanticIndex,
    /// `engine.code_graph` — the code property graph.
    CodeGraph,
    /// `engine.federated_index` — the cross-project federated index.
    FederatedIndex,
    /// `engine.vector_space` — the visualization vector-space map.
    VectorSpace,
    /// `.caduceus/activeContext.md` — the living context file.
    ActiveContext,
    /// Catch-all for ad-hoc resources callers want to register.
    Other(String),
}

impl IndexResource {
    pub fn label(&self) -> &str {
        match self {
            IndexResource::SemanticIndex => "semantic_index",
            IndexResource::CodeGraph => "code_graph",
            IndexResource::FederatedIndex => "federated_index",
            IndexResource::VectorSpace => "vector_space",
            IndexResource::ActiveContext => "active_context",
            IndexResource::Other(s) => s.as_str(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

/// A single observed access. Per-event, not per-agent — the DAG view
/// is built by aggregating these.
#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub agent_id: String,
    pub agent_kind: AgentKind,
    pub resource: IndexResource,
    pub access: AccessKind,
    pub at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentKind {
    /// The interactive user prompt.
    User,
    /// A spawned subagent (`SpawnAgentTool`, multi-agent worker).
    Subagent,
    /// A `/slash` command handler.
    Slash,
    /// A tool invocation outside subagent context.
    Tool,
}

impl AgentKind {
    pub fn label(&self) -> &str {
        match self {
            AgentKind::User => "user",
            AgentKind::Subagent => "subagent",
            AgentKind::Slash => "slash",
            AgentKind::Tool => "tool",
        }
    }
}

fn registry() -> &'static Mutex<VecDeque<AccessEvent>> {
    static REG: OnceLock<Mutex<VecDeque<AccessEvent>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_EVENTS)))
}

/// A parent→child agent spawn. Recorded by `register_running_subagent`
/// in `agent::Thread` so the DAG view shows the full agent tree, not
/// just resource access edges.
#[derive(Debug, Clone)]
pub struct SpawnEvent {
    pub parent_id: String,
    pub parent_kind: AgentKind,
    pub child_id: String,
    pub child_kind: AgentKind,
    pub at: Instant,
}

fn spawn_registry() -> &'static Mutex<VecDeque<SpawnEvent>> {
    static REG: OnceLock<Mutex<VecDeque<SpawnEvent>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_SPAWN_EVENTS)))
}

/// Recover from a poisoned mutex by extracting the inner data and logging
/// a warning. F5: a panic inside ANY lock site (record, snapshot, render)
/// must not permanently silence the DAG — observability is too important
/// to lose to a single broken thread.
fn recover<'a, T>(
    lock_result: std::sync::LockResult<std::sync::MutexGuard<'a, T>>,
    site: &str,
) -> std::sync::MutexGuard<'a, T> {
    match lock_result {
        Ok(g) => g,
        Err(poisoned) => {
            log::warn!("[index_dag] mutex poisoned at {site}; recovering inner data");
            poisoned.into_inner()
        }
    }
}

/// Record one access. Cheap (two memory ops + a mutex) — safe to call
/// from hot paths like `semantic_search`.
pub fn record(
    agent_id: impl Into<String>,
    agent_kind: AgentKind,
    resource: IndexResource,
    access: AccessKind,
) {
    let event = AccessEvent {
        agent_id: agent_id.into(),
        agent_kind,
        resource,
        access,
        at: Instant::now(),
    };
    let mut q = recover(registry().lock(), "record");
    if q.len() == MAX_EVENTS {
        q.pop_front();
    }
    q.push_back(event);
    // F3 + F8: bump_version stays inside the critical section so the version
    // counter is consistent with the data view a snapshot() reader would
    // see if it acquired the lock immediately after us.
    bump_version();
}

/// Record a parent→child agent spawn. Surfaces sub-agent invocations
/// in the DAG so users see "user → spawned subagent bg-1 → reads
/// semantic_index" as a single coherent tree.
pub fn record_spawn(
    parent_id: impl Into<String>,
    parent_kind: AgentKind,
    child_id: impl Into<String>,
    child_kind: AgentKind,
) {
    let event = SpawnEvent {
        parent_id: parent_id.into(),
        parent_kind,
        child_id: child_id.into(),
        child_kind,
        at: Instant::now(),
    };
    let mut q = recover(spawn_registry().lock(), "record_spawn");
    if q.len() == MAX_SPAWN_EVENTS {
        q.pop_front();
    }
    q.push_back(event);
    bump_version();
}

/// Sonnet #7: remove all spawn edges for a given child id. Called when
/// a subagent terminates so the DAG doesn't show ghost edges for agents
/// that are no longer running.
pub fn remove_spawn(child_id: &str) {
    let mut q = recover(spawn_registry().lock(), "remove_spawn");
    let before = q.len();
    q.retain(|e| e.child_id != child_id);
    if q.len() != before {
        bump_version();
    }
}

/// Clear all recorded events — used by tests to start from a clean
/// state. Not exposed on the public API beyond test helpers.
#[cfg(test)]
pub(crate) fn clear() {
    {
        let mut q = recover(registry().lock(), "clear/access");
        q.clear();
    }
    {
        let mut q = recover(spawn_registry().lock(), "clear/spawn");
        q.clear();
    }
    bump_version();
}

/// Snapshot of currently relevant access events (within EVENT_TTL).
pub fn snapshot() -> Vec<AccessEvent> {
    let now = Instant::now();
    let q = recover(registry().lock(), "snapshot");
    q.iter()
        .filter(|e| now.saturating_duration_since(e.at) <= EVENT_TTL)
        .cloned()
        .collect()
}

/// Snapshot of currently relevant spawn events (within EVENT_TTL).
pub fn spawn_snapshot() -> Vec<SpawnEvent> {
    let now = Instant::now();
    let q = recover(spawn_registry().lock(), "spawn_snapshot");
    q.iter()
        .filter(|e| now.saturating_duration_since(e.at) <= EVENT_TTL)
        .cloned()
        .collect()
}

/// A node in the rendered DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagNode {
    Agent { id: String, kind: AgentKind },
    Resource(IndexResource),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagEdge {
    pub from: DagNode,
    pub to: DagNode,
    pub access: AccessKind,
    /// Number of times the edge was observed in the snapshot window.
    pub count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Dag {
    pub edges: Vec<DagEdge>,
    /// Parent → child agent spawn relationships. Rendered as a separate
    /// section in the ASCII view ("Agent Spawn Tree").
    pub spawns: Vec<SpawnEvent>,
}

impl Dag {
    /// Build a DAG from the current event snapshot. Edges are
    /// agent → resource for both Read and Write. Two agents that both
    /// touch the same resource are NOT linked directly — but the
    /// shared resource node makes the contention obvious in the
    /// rendered output.
    pub fn from_snapshot(events: &[AccessEvent]) -> Self {
        Self::from_snapshots(events, &[])
    }

    /// Build a DAG from BOTH access events and spawn events.
    pub fn from_snapshots(events: &[AccessEvent], spawns: &[SpawnEvent]) -> Self {
        let mut edges: Vec<DagEdge> = Vec::new();
        for ev in events {
            let from = DagNode::Agent {
                id: ev.agent_id.clone(),
                kind: ev.agent_kind,
            };
            let to = DagNode::Resource(ev.resource.clone());
            if let Some(existing) = edges
                .iter_mut()
                .find(|e| e.from == from && e.to == to && e.access == ev.access)
            {
                existing.count += 1;
            } else {
                edges.push(DagEdge {
                    from,
                    to,
                    access: ev.access,
                    count: 1,
                });
            }
        }
        Dag {
            edges,
            spawns: spawns.to_vec(),
        }
    }

    /// Returns the resources that have more than one distinct agent
    /// accessing them — these are the "contention points" worth
    /// highlighting in the UI.
    /// Resources where >1 distinct agent is touching the same resource AND
    /// at least one of those touches is a write. Sonnet #2: pure read-read
    /// concurrency is safe (no race) and must NOT light the warning chip,
    /// otherwise users learn to ignore the contention signal.
    pub fn contention_points(&self) -> Vec<&IndexResource> {
        #[derive(Default)]
        struct State<'a> {
            agents: std::collections::HashSet<&'a str>,
            has_write: bool,
        }
        let mut by_resource: std::collections::HashMap<&IndexResource, State<'_>> =
            std::collections::HashMap::new();
        for e in &self.edges {
            if let (DagNode::Agent { id, .. }, DagNode::Resource(r)) = (&e.from, &e.to) {
                let s = by_resource.entry(r).or_default();
                s.agents.insert(id.as_str());
                if matches!(e.access, AccessKind::Write) {
                    s.has_write = true;
                }
            }
        }
        by_resource
            .into_iter()
            .filter_map(|(r, s)| {
                if s.has_write && s.agents.len() > 1 {
                    Some(r)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Render the DAG as ASCII art for chat display. Format includes
    /// both the resource-access section AND the agent spawn tree.
    pub fn to_ascii(&self) -> String {
        // gpt-5.4 #3: sanitize all user-facing strings before rendering
        // them into the DAG. Raw IDs and IndexResource::Other(...) labels
        // come from arbitrary callers (subagent SessionIds, tool names,
        // file paths) and may contain control characters / newlines. An
        // unsanitized newline would split a DAG row and let a malicious
        // resource label spoof a "user spawned X" line in the chat output.
        fn sanitize_one_line(s: &str) -> String {
            s.chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect()
        }
        // Long IDs make the DAG unreadable. Truncate displayed IDs to ~16
        // chars (preserves enough of a SessionId UUID to disambiguate).
        fn display_id(s: &str) -> String {
            let clean = sanitize_one_line(s);
            const MAX: usize = 16;
            if clean.chars().count() > MAX {
                let mut t: String = clean.chars().take(MAX).collect();
                t.push('…');
                t
            } else {
                clean
            }
        }
        let mut out = String::new();
        // Section 1: Agent spawn tree (parent → child)
        if !self.spawns.is_empty() {
            out.push_str("Agent Spawn Tree (window: last 5 min)\n");
            out.push_str("─────────────────────────────────────\n");
            // Dedupe by (parent, child) so a re-spawn doesn't double-render.
            let mut seen: std::collections::HashSet<(String, String)> =
                std::collections::HashSet::new();
            let mut sorted = self.spawns.clone();
            sorted.sort_by(|a, b| {
                a.parent_id
                    .cmp(&b.parent_id)
                    .then_with(|| a.child_id.cmp(&b.child_id))
            });
            for s in &sorted {
                if !seen.insert((s.parent_id.clone(), s.child_id.clone())) {
                    continue;
                }
                out.push_str(&format!(
                    "  {}:{:<18} ──spawns──► {}:{}\n",
                    s.parent_kind.label(),
                    display_id(&s.parent_id),
                    s.child_kind.label(),
                    display_id(&s.child_id),
                ));
            }
            out.push('\n');
        }

        // Section 2: Index access edges
        if self.edges.is_empty() {
            if self.spawns.is_empty() {
                return "Index Access DAG\n────────────────\n  (no recent index access in the last 5 min)\n".to_string();
            }
            out.push_str("Index Access DAG\n");
            out.push_str("────────────────\n");
            out.push_str("  (no recent index access in the last 5 min)\n");
            return out;
        }
        let contention: std::collections::HashSet<&IndexResource> =
            self.contention_points().into_iter().collect();

        out.push_str("Index Access DAG (window: last 5 min)\n");
        out.push_str("─────────────────────────────────────\n");
        // Sort for stable rendering: by resource, then access, then agent.
        let mut sorted = self.edges.clone();
        sorted.sort_by(|a, b| {
            let ra = match &a.to {
                DagNode::Resource(r) => r.label(),
                _ => "",
            };
            let rb = match &b.to {
                DagNode::Resource(r) => r.label(),
                _ => "",
            };
            ra.cmp(rb)
                .then_with(|| (a.access as u8).cmp(&(b.access as u8)))
                .then_with(|| match (&a.from, &b.from) {
                    (DagNode::Agent { id: ai, .. }, DagNode::Agent { id: bi, .. }) => ai.cmp(bi),
                    _ => std::cmp::Ordering::Equal,
                })
        });

        for e in &sorted {
            let (agent_label, resource_label) = match (&e.from, &e.to) {
                (DagNode::Agent { id, kind }, DagNode::Resource(r)) => {
                    (format!("{}:{}", kind.label(), display_id(id)), r)
                }
                _ => continue,
            };
            let arrow = match e.access {
                AccessKind::Read => "──r──►",
                AccessKind::Write => "══w══►",
            };
            let warn = if contention.contains(resource_label) {
                "  ⚠ contention"
            } else {
                ""
            };
            out.push_str(&format!(
                "  {:<22} {} {:<20} (×{}){}\n",
                agent_label,
                arrow,
                sanitize_one_line(resource_label.label()),
                e.count,
                warn
            ));
        }
        // Trim trailing newline for callers that join with "\n".
        if out.ends_with('\n') {
            out.pop();
        }
        out
    }
}

/// Convenience: build the current DAG (access + spawns) and render it
/// as ASCII in one call. Used by the `/dag` slash command and the
/// agent_panel popover.
pub fn render_current_ascii() -> String {
    Dag::from_snapshots(&snapshot(), &spawn_snapshot()).to_ascii()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All DAG tests share a global registry, so they must run
    /// sequentially. The mutex serializes them; poison is fine to
    /// recover from since tests panic on assertion failure.
    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        match LOCK.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        }
    }

    #[test]
    fn empty_snapshot_renders_no_recent_access() {
        let _g = test_lock();
        clear();
        let s = render_current_ascii();
        assert!(s.contains("no recent index access"), "{s}");
    }

    #[test]
    fn single_access_appears_in_dag() {
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        let dag = Dag::from_snapshot(&snapshot());
        assert_eq!(dag.edges.len(), 1);
        assert_eq!(dag.edges[0].count, 1);
        assert_eq!(dag.edges[0].access, AccessKind::Read);
    }

    #[test]
    fn repeated_same_access_increments_count() {
        let _g = test_lock();
        clear();
        for _ in 0..3 {
            record(
                "u1",
                AgentKind::User,
                IndexResource::SemanticIndex,
                AccessKind::Read,
            );
        }
        let dag = Dag::from_snapshot(&snapshot());
        assert_eq!(dag.edges.len(), 1);
        assert_eq!(dag.edges[0].count, 3);
    }

    #[test]
    fn read_and_write_are_distinct_edges() {
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::CodeGraph,
            AccessKind::Read,
        );
        record(
            "u1",
            AgentKind::User,
            IndexResource::CodeGraph,
            AccessKind::Write,
        );
        let dag = Dag::from_snapshot(&snapshot());
        assert_eq!(dag.edges.len(), 2);
    }

    #[test]
    fn two_agents_writer_plus_reader_show_contention() {
        // Sonnet #2: contention requires ≥1 write. A reader + a writer on
        // the same resource is a real race risk.
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Write,
        );
        record(
            "bg-1",
            AgentKind::Subagent,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        let dag = Dag::from_snapshot(&snapshot());
        let contention = dag.contention_points();
        assert_eq!(contention.len(), 1);
        let ascii = dag.to_ascii();
        assert!(ascii.contains("⚠ contention"), "{ascii}");
    }

    #[test]
    fn two_readers_do_not_show_contention() {
        // Sonnet #2: pure read-read concurrency is safe — must NOT light
        // the warning chip.
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        record(
            "bg-1",
            AgentKind::Subagent,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        let dag = Dag::from_snapshot(&snapshot());
        assert!(
            dag.contention_points().is_empty(),
            "two readers must not be flagged as contention"
        );
    }

    #[test]
    fn two_writers_show_contention() {
        // Two distinct agents both writing the same resource is the
        // canonical race condition.
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Write,
        );
        record(
            "bg-1",
            AgentKind::Subagent,
            IndexResource::SemanticIndex,
            AccessKind::Write,
        );
        let dag = Dag::from_snapshot(&snapshot());
        assert_eq!(dag.contention_points().len(), 1);
    }

    #[test]
    fn single_agent_no_contention_warning() {
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        record(
            "u1",
            AgentKind::User,
            IndexResource::CodeGraph,
            AccessKind::Write,
        );
        let dag = Dag::from_snapshot(&snapshot());
        assert!(dag.contention_points().is_empty());
        let ascii = dag.to_ascii();
        assert!(!ascii.contains("⚠"), "{ascii}");
    }

    #[test]
    fn ring_buffer_caps_at_max() {
        let _g = test_lock();
        clear();
        for i in 0..(MAX_EVENTS + 50) {
            record(
                format!("u{i}"),
                AgentKind::Tool,
                IndexResource::SemanticIndex,
                AccessKind::Read,
            );
        }
        assert_eq!(snapshot().len(), MAX_EVENTS);
    }

    #[test]
    fn ascii_renders_three_agents_one_resource_with_contention() {
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        record(
            "bg-1",
            AgentKind::Subagent,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        record(
            "bg-2",
            AgentKind::Subagent,
            IndexResource::SemanticIndex,
            AccessKind::Write,
        );
        let ascii = render_current_ascii();
        assert!(ascii.contains("user:u1"));
        assert!(ascii.contains("subagent:bg-1"));
        assert!(ascii.contains("subagent:bg-2"));
        assert!(ascii.contains("semantic_index"));
        assert!(ascii.contains("⚠ contention"));
    }

    #[test]
    fn other_resource_label_is_passed_through() {
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::Other("custom_thing".into()),
            AccessKind::Read,
        );
        let ascii = render_current_ascii();
        assert!(ascii.contains("custom_thing"), "{ascii}");
    }

    #[test]
    fn record_bumps_version() {
        let _g = test_lock();
        clear();
        let v0 = version();
        record(
            "u1",
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        let v1 = version();
        assert!(
            v1 > v0,
            "version should advance after record() (was {v0}, now {v1})"
        );
        record(
            "u2",
            AgentKind::Tool,
            IndexResource::CodeGraph,
            AccessKind::Write,
        );
        let v2 = version();
        assert!(v2 > v1, "second record should also advance version");
    }

    #[test]
    fn spawn_event_recorded_and_rendered() {
        let _g = test_lock();
        clear();
        record_spawn("user-main", AgentKind::User, "bg-1", AgentKind::Subagent);
        let snaps = spawn_snapshot();
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0].parent_id, "user-main");
        assert_eq!(snaps[0].child_id, "bg-1");

        let ascii = render_current_ascii();
        assert!(ascii.contains("Agent Spawn Tree"), "{ascii}");
        assert!(ascii.contains("user:user-main"), "{ascii}");
        assert!(ascii.contains("subagent:bg-1"), "{ascii}");
        assert!(ascii.contains("spawns"), "{ascii}");
    }

    #[test]
    fn spawn_dedupes_repeated_pairs() {
        let _g = test_lock();
        clear();
        for _ in 0..5 {
            record_spawn("u-1", AgentKind::User, "c-1", AgentKind::Subagent);
        }
        let ascii = render_current_ascii();
        // Should only appear ONCE in rendered output even though recorded 5x.
        let occurrences = ascii.matches("c-1").count();
        assert_eq!(
            occurrences, 1,
            "duplicate spawns must dedupe in render. ascii=\n{ascii}"
        );
    }

    #[test]
    fn spawn_only_no_access_renders_both_sections() {
        let _g = test_lock();
        clear();
        record_spawn("u", AgentKind::User, "child", AgentKind::Subagent);
        let ascii = render_current_ascii();
        assert!(ascii.contains("Agent Spawn Tree"));
        assert!(ascii.contains("no recent index access"));
    }

    #[test]
    fn spawn_ring_buffer_caps_at_max() {
        let _g = test_lock();
        clear();
        for i in 0..(MAX_SPAWN_EVENTS + 30) {
            record_spawn(
                "parent",
                AgentKind::User,
                format!("child-{i}"),
                AgentKind::Subagent,
            );
        }
        assert_eq!(spawn_snapshot().len(), MAX_SPAWN_EVENTS);
    }

    #[test]
    fn version_advances_on_spawn_too() {
        let _g = test_lock();
        clear();
        let v0 = version();
        record_spawn("p", AgentKind::User, "c", AgentKind::Subagent);
        assert!(version() > v0);
    }

    #[test]
    fn remove_spawn_clears_ghost_edges_and_bumps_version() {
        // Sonnet #7: when a subagent terminates, its spawn edge must be
        // pruned so the DAG doesn't show ghosts of dead agents.
        let _g = test_lock();
        clear();
        record_spawn("parent-1", AgentKind::User, "child-A", AgentKind::Subagent);
        record_spawn("parent-1", AgentKind::User, "child-B", AgentKind::Subagent);
        assert_eq!(spawn_snapshot().len(), 2);

        let v_before = version();
        remove_spawn("child-A");
        let after = spawn_snapshot();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].child_id, "child-B");
        assert!(
            version() > v_before,
            "version must advance so the auto-refresh poller redraws"
        );
    }

    #[test]
    fn remove_spawn_unknown_child_does_not_bump_version() {
        // Removing a non-existent child must be a no-op — the version
        // counter is the auto-refresh trigger, and a no-op shouldn't
        // cause a redraw.
        let _g = test_lock();
        clear();
        record_spawn("parent-1", AgentKind::User, "child-A", AgentKind::Subagent);
        let v_before = version();
        remove_spawn("nonexistent");
        assert_eq!(version(), v_before);
    }

    #[test]
    fn ascii_sanitizes_control_chars_in_resource_label() {
        // gpt-5.4 #3 regression: a resource label containing newlines
        // must NOT split the rendered DAG row. Otherwise a malicious
        // label like "x\n  user:fake ──spawns──► subagent:victim" could
        // spoof an additional row in the chat output.
        let _g = test_lock();
        clear();
        record(
            "u1",
            AgentKind::User,
            IndexResource::Other("danger\nINJECTED".to_string()),
            AccessKind::Read,
        );
        let dag = Dag::from_snapshot(&snapshot());
        let ascii = dag.to_ascii();
        assert!(
            !ascii.contains("\nINJECTED"),
            "raw newline in resource label leaked into ASCII:\n{ascii}"
        );
        assert!(
            ascii.contains("danger INJECTED"),
            "expected newline replaced with space in label, got:\n{ascii}"
        );
    }

    #[test]
    fn ascii_truncates_long_agent_ids() {
        // gpt-5.4 #3 readability: SessionId UUIDs are 36+ chars long.
        // The DAG must truncate them so the chip + chat output stay
        // readable. Truncation marker '…' must be present.
        let _g = test_lock();
        clear();
        let long_id = "0123456789abcdef0123456789abcdef-extra";
        record(
            long_id,
            AgentKind::User,
            IndexResource::SemanticIndex,
            AccessKind::Read,
        );
        let dag = Dag::from_snapshot(&snapshot());
        let ascii = dag.to_ascii();
        assert!(
            !ascii.contains(long_id),
            "long agent id was not truncated:\n{ascii}"
        );
        assert!(
            ascii.contains("…"),
            "expected truncation marker '…' in:\n{ascii}"
        );
    }

    #[test]
    fn ascii_sanitizes_control_chars_in_spawn_ids() {
        // Same as above but for spawn-section IDs.
        let _g = test_lock();
        clear();
        record_spawn(
            "parent\nINJECTED",
            AgentKind::User,
            "child-1",
            AgentKind::Subagent,
        );
        let dag = Dag::from_snapshots(&snapshot(), &spawn_snapshot());
        let ascii = dag.to_ascii();
        assert!(
            !ascii.contains("\nINJECTED"),
            "raw newline in parent_id leaked into spawn-section ASCII:\n{ascii}"
        );
    }
}
