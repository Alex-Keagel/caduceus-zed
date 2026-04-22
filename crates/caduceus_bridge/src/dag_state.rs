//! P13d — bridge-layer live state reducer.
//!
//! Ingests the [`AgentEvent`] stream a session emits and materializes three
//! queryable views:
//!
//! * [`SessionStateReducer::active_features_dag`] — the plan (WHAT needs to
//!   happen): nodes are plan steps, edges are `depends_on` relationships.
//! * [`SessionStateReducer::active_agents_dag`] — the executors (HOW it
//!   happens): nodes are per-execution assignments (primary + every critic),
//!   edges are delegation / critique / handoff / retry / spawn.
//! * [`SessionStateReducer::active_session_snapshot`] — the single "right
//!   now" slice: current mode+lens, envelope summary, active fan-outs,
//!   pending approvals.
//!
//! ## Design
//!
//! **One reducer, three filtered views.** The rubber-duck critique
//! ([P13 design]) flagged that having three independent reducers would
//! lead to skew. Instead, all state lives in one [`SessionStateReducer`];
//! each `active_*` query is a read-only projection of the same state.
//!
//! **Monotonic `last_event_id`.** Every ingested event bumps a local u64
//! counter. Every query returns `last_event_id` so clients can bootstrap
//! a snapshot + subscribe from `last_event_id + 1` without gaps. This
//! counter is the reducer's OWN monotonic clock — it is NOT the same as
//! [`caduceus_core::EventId`] (which is minted at the persistence
//! boundary and may be absent on in-process events).
//!
//! **Security by default.** `include_sensitive=false` redacts
//! `model_id_exact`, `activated_skill_names`, `activated_agent_names`,
//! and `envelope.display_text`. The reducer always stores the full
//! records; redaction happens on the way out.

use caduceus_core::{
    AgentEdgeKind, AgentEvent, AssignmentSummaryV1, Critique, EnvelopeSummaryV1, ExecutionId,
    IntrospectionEventV1, ProvenanceEdgeKind, StepId,
};
use std::collections::{HashMap, VecDeque};

// ── Wire-shape records surfaced by the reducer ────────────────────────────

/// A node in the Features-DAG.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FeatureStepV1 {
    pub step_id: StepId,
    /// 1-indexed ordinal within the plan at the time the step was created.
    /// May shift under later amendments; use `step_id` for stable identity.
    pub ordinal: usize,
    pub tool_name: String,
    pub description: String,
    /// Stable step ids this step directly depends on.
    pub depends_on: Vec<StepId>,
    /// Optional parent step for sub-step decomposition.
    pub parent_step_id: Option<StepId>,
    pub plan_revision: u64,
    pub revision: u64,
    /// Coarse status derived from the event stream.
    pub status: FeatureStepStatus,
}

/// Derivable, coarse status. Today we only observe `PlanStepPending` so
/// everything starts `Pending`. Completion / failure tracking is future
/// work that slots in without wire changes.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureStepStatus {
    Pending,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct FeaturesDagV1 {
    pub steps: Vec<FeatureStepV1>,
    /// Reducer-local monotonic counter. Clients resume from this + 1.
    pub last_event_id: u64,
    /// Bumped whenever the DAG structure (not just per-step revision) changes.
    pub revision: u64,
}

/// A node in the Agents-DAG. Wraps [`AssignmentSummaryV1`] so we can redact
/// sensitive fields without mutating the stored record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AgentNodeV1 {
    pub assignment: AssignmentSummaryV1,
    pub attempt: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AgentEdgeV1 {
    pub kind: AgentEdgeKind,
    pub from: ExecutionId,
    pub to: ExecutionId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AgentsDagV1 {
    pub nodes: Vec<AgentNodeV1>,
    pub edges: Vec<AgentEdgeV1>,
    pub last_event_id: u64,
    /// Currently running fan-outs keyed by `parent_execution_id`.
    /// Useful for rendering "3 critics running..." without scanning events.
    pub active_fanouts: Vec<ActiveFanoutV1>,
    pub revision: u64,
    /// Phase-H H5 — count of `AgentEdgeV1`s dropped by the reducer's
    /// FIFO bound (see [`SessionStateReducer::AGENT_EDGE_CAP`]). Stays
    /// `0` on ordinary sessions; non-zero tells the UI to render a
    /// "graph truncated" marker. Defaults to `0` over the wire for
    /// back-compat with pre-H5 clients.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub edges_dropped: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ActiveFanoutV1 {
    pub step_id: StepId,
    pub parent_execution_id: ExecutionId,
    pub critic_count: u32,
    /// Personas currently critiquing (from `FanoutStarted.personas`).
    pub personas: Vec<String>,
}

/// Snapshot of "right now". Everything is a single coherent slice —
/// `last_event_id` tells clients the exact point-in-time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SessionSnapshotV1 {
    /// Opaque mode name (e.g. `"plan"`, `"act"`). Clients MUST route this
    /// through `list_modes()` rather than interpret it locally.
    pub mode: Option<String>,
    pub lens: Option<String>,
    pub envelope: Option<EnvelopeSummaryV1>,
    /// Pending critique block awaiting user accept/reject/amend.
    pub awaiting_approval: Option<AwaitingApprovalV1>,
    /// Pending scope-expansion request (see
    /// [`AgentEvent::ScopeExpansionRequested`]). None once resolved.
    pub pending_scope_expansion: Option<PendingScopeExpansionV1>,
    /// Cross-graph edges: which execution caused which plan mutation.
    pub provenance: Vec<ProvenanceEdgeV1>,
    pub last_event_id: u64,
    /// Phase-H H5 — count of `ProvenanceEdgeV1`s dropped by the reducer's
    /// FIFO bound (see [`SessionStateReducer::PROVENANCE_CAP`]). Same
    /// semantics as [`AgentsDagV1::edges_dropped`].
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub provenance_dropped: u64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct AwaitingApprovalV1 {
    pub plan_revision: String,
    pub critiques: Vec<Critique>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PendingScopeExpansionV1 {
    pub capability: String,
    pub resource: String,
    pub reason: String,
    pub tool: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ProvenanceEdgeV1 {
    pub kind: ProvenanceEdgeKind,
    pub execution_id: ExecutionId,
    pub target_step_id: Option<StepId>,
}

/// Phase-H H5 — skip-serializing helper for the `*_dropped` counters so
/// the common case (no truncation) costs zero wire bytes.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero_u64(n: &u64) -> bool {
    *n == 0
}

// ── The reducer ────────────────────────────────────────────────────────────

/// Per-session live state materializer.
///
/// Cheap to construct. Feed it every [`AgentEvent`] the session produces via
/// [`SessionStateReducer::ingest`]; it maintains the three projections
/// incrementally. Not `Sync` — wrap in `Mutex`/`RwLock` at the bridge layer
/// if concurrent access is needed.
#[derive(Debug, Default)]
pub struct SessionStateReducer {
    // Features-DAG — keyed by step_id for O(1) upserts on amend-style edits.
    // A secondary vec preserves insertion order.
    features: HashMap<StepId, FeatureStepV1>,
    feature_order: Vec<StepId>,
    features_revision: u64,

    // Agents-DAG
    agent_nodes: HashMap<ExecutionId, AssignmentSummaryV1>,
    agent_node_order: Vec<ExecutionId>,
    // ST-C3 / audit P-3: VecDeque so bounded-FIFO pop_front is O(1); previously
    // `Vec::remove(0)` was O(n) with AGENT_EDGE_CAP=50_000 (≈ pathological overflow).
    agent_edges: VecDeque<AgentEdgeV1>,
    /// Phase-H H5 — FIFO count of agent edges dropped once
    /// `agent_edges.len() == AGENT_EDGE_CAP`.
    agent_edges_dropped: u64,
    active_fanouts: HashMap<(StepId, ExecutionId), ActiveFanoutV1>,
    agents_revision: u64,

    // Session snapshot
    mode: Option<String>,
    lens: Option<String>,
    envelope: Option<EnvelopeSummaryV1>,
    awaiting_approval: Option<AwaitingApprovalV1>,
    pending_scope_expansion: Option<PendingScopeExpansionV1>,
    // ST-C3 / audit P-3: VecDeque for O(1) FIFO eviction (see agent_edges).
    provenance: VecDeque<ProvenanceEdgeV1>,
    /// Phase-H H5 — same semantics as `agent_edges_dropped`.
    provenance_dropped: u64,

    last_event_id: u64,
}

impl SessionStateReducer {
    /// Phase-H H5 — hard cap on the number of `AgentEdgeV1`s retained
    /// in the rolling window. Generous (50k edges ≈ hours of a
    /// multi-agent session) but bounded so a pathological runaway
    /// cannot OOM the IDE. When full, the oldest edge is evicted and
    /// `agent_edges_dropped` bumps by one.
    pub const AGENT_EDGE_CAP: usize = 50_000;

    /// Phase-H H5 — hard cap on provenance edges. Same semantics as
    /// [`Self::AGENT_EDGE_CAP`]. Provenance is denser than agent edges
    /// (every tool call can emit one) so a slightly larger cap.
    pub const PROVENANCE_CAP: usize = 100_000;

    pub fn new() -> Self {
        Self::default()
    }

    /// Current reducer-local event id (equals the number of events ingested).
    pub fn last_event_id(&self) -> u64 {
        self.last_event_id
    }

    /// Consume one agent event and fold it into state. Unknown events are
    /// ignored silently (forward-compat). Returns the reducer's updated
    /// `last_event_id`.
    pub fn ingest(&mut self, event: &AgentEvent) -> u64 {
        self.last_event_id += 1;
        match event {
            AgentEvent::PlanStepPending {
                step,
                step_id,
                revision,
                plan_revision,
                tool_name,
                description,
                depends_on,
                parent_step_id,
            } => {
                // Pre-P13 producers send step_id=0 for EVERY step. In that
                // legacy regime the positional `step` index IS the identity,
                // so synthesize a unique StepId from it. Post-P13 producers
                // send monotonic non-zero ids and we use them verbatim.
                let id = if step_id.0 == 0 {
                    StepId(*step as u64)
                } else {
                    *step_id
                };
                let existed = self.features.contains_key(&id);
                let rec = FeatureStepV1 {
                    step_id: id,
                    ordinal: *step,
                    tool_name: tool_name.clone(),
                    description: description.clone(),
                    depends_on: depends_on.clone(),
                    parent_step_id: *parent_step_id,
                    plan_revision: *plan_revision,
                    revision: *revision,
                    status: FeatureStepStatus::Pending,
                };
                self.features.insert(id, rec);
                if !existed {
                    self.feature_order.push(id);
                }
                self.features_revision += 1;
            }

            AgentEvent::ModeChanged {
                to_mode, to_lens, ..
            } => {
                self.mode = Some(to_mode.clone());
                self.lens = to_lens.clone();
            }

            AgentEvent::AwaitingApproval {
                plan_revision,
                critiques,
            } => {
                self.awaiting_approval = Some(AwaitingApprovalV1 {
                    plan_revision: plan_revision.clone(),
                    critiques: critiques.clone(),
                });
            }

            AgentEvent::ScopeExpansionRequested {
                capability,
                resource,
                reason,
                tool,
            } => {
                self.pending_scope_expansion = Some(PendingScopeExpansionV1 {
                    capability: capability.clone(),
                    resource: resource.clone(),
                    reason: reason.clone(),
                    tool: tool.clone(),
                });
            }

            AgentEvent::Introspection(ev) => self.ingest_introspection(ev),

            _ => { /* not state-bearing for this reducer */ }
        }
        self.last_event_id
    }

    fn ingest_introspection(&mut self, ev: &IntrospectionEventV1) {
        match ev {
            IntrospectionEventV1::EnvelopeApplied { summary } => {
                self.envelope = Some(summary.clone());
            }
            IntrospectionEventV1::StepAssigned { assignment } => {
                self.insert_agent_node(assignment.clone());
            }
            IntrospectionEventV1::SubAgentSpawned {
                parent_execution_id,
                assignment,
                ..
            } => {
                self.insert_agent_node(assignment.clone());
                self.push_agent_edge(AgentEdgeV1 {
                    kind: AgentEdgeKind::Spawn,
                    from: *parent_execution_id,
                    to: assignment.execution_id,
                });
                self.agents_revision += 1;
            }
            IntrospectionEventV1::AgentEdgeRecorded {
                edge,
                from_execution_id,
                to_execution_id,
            } => {
                self.push_agent_edge(AgentEdgeV1 {
                    kind: *edge,
                    from: *from_execution_id,
                    to: *to_execution_id,
                });
                self.agents_revision += 1;
            }
            IntrospectionEventV1::CritiqueEmitted { .. } => {
                // Pair event emitted alongside an AgentEdgeRecorded(Critique);
                // the edge carries the topology. Nothing extra to store.
            }
            IntrospectionEventV1::ProvenanceRecorded {
                edge,
                execution_id,
                target_step_id,
            } => {
                self.push_provenance(ProvenanceEdgeV1 {
                    kind: *edge,
                    execution_id: *execution_id,
                    target_step_id: *target_step_id,
                });
            }
            IntrospectionEventV1::FanoutStarted {
                step_id,
                parent_execution_id,
                critic_count,
                personas,
            } => {
                self.active_fanouts.insert(
                    (*step_id, *parent_execution_id),
                    ActiveFanoutV1 {
                        step_id: *step_id,
                        parent_execution_id: *parent_execution_id,
                        critic_count: *critic_count,
                        personas: personas.clone(),
                    },
                );
                self.agents_revision += 1;
            }
            IntrospectionEventV1::FanoutCompleted {
                step_id,
                parent_execution_id,
                ..
            } => {
                self.active_fanouts
                    .remove(&(*step_id, *parent_execution_id));
                self.agents_revision += 1;
            }
        }
    }

    fn insert_agent_node(&mut self, a: AssignmentSummaryV1) {
        let eid = a.execution_id;
        if !self.agent_nodes.contains_key(&eid) {
            self.agent_node_order.push(eid);
        }
        self.agent_nodes.insert(eid, a);
        self.agents_revision += 1;
    }

    /// Phase-H H5 — bounded push. Drops the oldest edge (FIFO) once the
    /// cap is hit and bumps `agent_edges_dropped`.
    fn push_agent_edge(&mut self, edge: AgentEdgeV1) {
        if self.agent_edges.len() >= Self::AGENT_EDGE_CAP {
            self.agent_edges.pop_front();
            self.agent_edges_dropped = self.agent_edges_dropped.saturating_add(1);
        }
        self.agent_edges.push_back(edge);
    }

    /// Phase-H H5 — bounded push. Same semantics as
    /// [`Self::push_agent_edge`] but for [`ProvenanceEdgeV1`].
    fn push_provenance(&mut self, edge: ProvenanceEdgeV1) {
        if self.provenance.len() >= Self::PROVENANCE_CAP {
            self.provenance.pop_front();
            self.provenance_dropped = self.provenance_dropped.saturating_add(1);
        }
        self.provenance.push_back(edge);
    }

    // ── Query projections ─────────────────────────────────────────────────

    /// Features-DAG view. `include_sensitive` is unused today but present
    /// for API symmetry — future Features-DAG fields (e.g. reasoning traces)
    /// may be redacted.
    pub fn active_features_dag(&self, _include_sensitive: bool) -> FeaturesDagV1 {
        let steps = self
            .feature_order
            .iter()
            .filter_map(|id| self.features.get(id).cloned())
            .collect();
        FeaturesDagV1 {
            steps,
            last_event_id: self.last_event_id,
            revision: self.features_revision,
        }
    }

    /// Agents-DAG view. Redacts exact model id + skill/agent names unless
    /// `include_sensitive`.
    pub fn active_agents_dag(&self, include_sensitive: bool) -> AgentsDagV1 {
        let nodes = self
            .agent_node_order
            .iter()
            .filter_map(|id| self.agent_nodes.get(id).cloned())
            .map(|mut a| {
                if !include_sensitive {
                    a.model_id_exact = None;
                    a.activated_skill_names = None;
                    a.activated_agent_names = None;
                }
                AgentNodeV1 {
                    attempt: a.attempt,
                    assignment: a,
                }
            })
            .collect();

        let mut active_fanouts: Vec<ActiveFanoutV1> =
            self.active_fanouts.values().cloned().collect();
        // Deterministic order so the wire shape is reproducible under test.
        active_fanouts.sort_by(|a, b| {
            a.step_id
                .0
                .cmp(&b.step_id.0)
                .then(a.parent_execution_id.0.cmp(&b.parent_execution_id.0))
        });

        AgentsDagV1 {
            nodes,
            edges: self.agent_edges.iter().cloned().collect(),
            last_event_id: self.last_event_id,
            active_fanouts,
            revision: self.agents_revision,
            edges_dropped: self.agent_edges_dropped,
        }
    }

    /// Session snapshot — mode, lens, envelope, pending gates.
    pub fn active_session_snapshot(&self, include_sensitive: bool) -> SessionSnapshotV1 {
        let envelope = self.envelope.clone().map(|mut e| {
            if !include_sensitive {
                e.display_text = None;
            }
            e
        });
        SessionSnapshotV1 {
            mode: self.mode.clone(),
            lens: self.lens.clone(),
            envelope,
            awaiting_approval: self.awaiting_approval.clone(),
            pending_scope_expansion: self.pending_scope_expansion.clone(),
            provenance: self.provenance.iter().cloned().collect(),
            last_event_id: self.last_event_id,
            provenance_dropped: self.provenance_dropped,
        }
    }
}

// ── ReducerHandle — auto-wire-ready, concurrent-safe wrapper ───────────────
//
// P13 follow-up: the bare [`SessionStateReducer`] is `Send` but not `Sync`
// (it mutates internal maps through `&mut self`). In production every
// session has one reducer fed by BOTH:
//   * the introspection pipeline (many parallel critic tasks calling
//     `IntrospectionSink::emit` on the same handle), AND
//   * the generic agent event stream (`AgentEvent::PlanStepPending`,
//     `ModeChanged`, `AwaitingApproval`, …).
//
// `ReducerHandle` is the shared, cloneable, sync-safe handle the bridge
// hands out to both sides. It wraps the reducer in `Arc<Mutex<…>>` and
// implements [`IntrospectionSink`], so passing `handle.as_sink()` into
// [`caduceus_orchestrator::critique_fanout::spawn_critique_fanout_with_introspection`]
// just works — no ad-hoc `Mutex<Reducer>` wrappers in caller code.
//
// The handle deliberately owns a single reducer per session (not per
// projection). Three reducers would skew; one reducer, three filtered
// views is the invariant from the P13 design.

/// Cheaply-cloneable, thread-safe handle to a per-session
/// [`SessionStateReducer`]. Use `OrchestratorBridge::new_reducer_handle`
/// to mint one per session; every collaborator that produces events —
/// the introspection fan-out driver, the IPC layer, the UI replay loop —
/// should share the same handle so the reducer sees every event exactly
/// once.
#[derive(Clone, Default)]
pub struct ReducerHandle {
    inner: std::sync::Arc<std::sync::Mutex<SessionStateReducer>>,
}

impl std::fmt::Debug for ReducerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Avoid taking the lock in Debug to prevent Debug-triggered deadlocks.
        f.debug_struct("ReducerHandle").finish_non_exhaustive()
    }
}

impl ReducerHandle {
    /// Mint a fresh handle with an empty reducer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire the reducer lock, recovering transparently from poison.
    ///
    /// **Phase-H F1**: previously we used `.expect("reducer mutex
    /// poisoned")` which turned any panic inside a reducer method into
    /// forwarder death + a 5 s timeout abandonment in the caller. We now
    /// emit a structured `log::error!` and unwrap the poison (the
    /// reducer's invariants tolerate partial state after a handled
    /// panic — worst case a DAG projection reflects truncated data for
    /// one tick). Call sites see the same `MutexGuard<SessionStateReducer>`
    /// regardless of poison state.
    fn lock_inner(&self, site: &'static str) -> std::sync::MutexGuard<'_, SessionStateReducer> {
        match self.inner.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                log::error!(
                    target: "caduceus_bridge.reducer",
                    "reducer mutex was poisoned at call-site {site}; recovering with potentially truncated state"
                );
                poisoned.into_inner()
            }
        }
    }

    /// Fold a single [`AgentEvent`] into state. The lock is held only for
    /// the duration of the `ingest` call — cheap map/vec pushes.
    pub fn ingest_event(&self, event: &AgentEvent) -> u64 {
        let mut g = self.lock_inner("ingest_event");
        g.ingest(event)
    }

    /// Bulk ingest — typical path for session replay on UI remount.
    /// Events are ingested in slice order under a single lock acquisition
    /// so no concurrent producer can interleave and split the replay.
    pub fn ingest_many(&self, events: &[AgentEvent]) -> u64 {
        let mut g = self.lock_inner("ingest_many");
        let mut last = g.last_event_id();
        for ev in events {
            last = g.ingest(ev);
        }
        last
    }

    /// Current reducer-local event id.
    pub fn last_event_id(&self) -> u64 {
        self.lock_inner("last_event_id").last_event_id()
    }

    /// Features-DAG view. `include_sensitive` is reserved for future
    /// redaction of per-step metadata; today the features projection has
    /// no sensitive fields but the signature is kept uniform with the
    /// other two projections.
    pub fn active_features_dag(&self, include_sensitive: bool) -> FeaturesDagV1 {
        self.lock_inner("active_features_dag")
            .active_features_dag(include_sensitive)
    }

    /// Agents-DAG view with redaction controlled by `include_sensitive`.
    pub fn active_agents_dag(&self, include_sensitive: bool) -> AgentsDagV1 {
        self.lock_inner("active_agents_dag")
            .active_agents_dag(include_sensitive)
    }

    /// Session snapshot with redaction controlled by `include_sensitive`.
    pub fn active_session_snapshot(&self, include_sensitive: bool) -> SessionSnapshotV1 {
        self.lock_inner("active_session_snapshot")
            .active_session_snapshot(include_sensitive)
    }

    /// View the handle as an [`IntrospectionSink`] trait object. Hands
    /// out a `&dyn IntrospectionSink` backed by this same handle so
    /// callers can feed it straight into the critique fan-out driver.
    pub fn as_sink(&self) -> &(dyn caduceus_orchestrator::IntrospectionSink + '_) {
        self
    }

    /// Strong ref-count of the underlying reducer — diagnostic helper for
    /// tests that want to assert "only the bridge holds the reducer".
    #[doc(hidden)]
    pub fn _strong_count(&self) -> usize {
        std::sync::Arc::strong_count(&self.inner)
    }
}

#[async_trait::async_trait]
impl caduceus_orchestrator::IntrospectionSink for ReducerHandle {
    async fn emit(&self, event: IntrospectionEventV1) {
        // Wrap in the outer AgentEvent envelope so the reducer's unified
        // `ingest` path handles it — introspection is just one flavor of
        // agent event from the reducer's perspective.
        let wrapped = AgentEvent::Introspection(event);
        self.ingest_event(&wrapped);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_core::CritiqueSeverity;

    fn plan_step(step: usize, id: u64, tool: &str, deps: Vec<u64>) -> AgentEvent {
        AgentEvent::PlanStepPending {
            step,
            step_id: StepId(id),
            revision: 0,
            plan_revision: 1,
            tool_name: tool.into(),
            description: format!("step {step}"),
            depends_on: deps.into_iter().map(StepId).collect(),
            parent_step_id: None,
        }
    }

    fn assignment(eid: u64, step: u64, persona: &str) -> AssignmentSummaryV1 {
        AssignmentSummaryV1 {
            execution_id: ExecutionId(eid),
            step_id: StepId(step),
            persona_id: persona.into(),
            model_vendor: "anthropic".into(),
            model_tier: "opus".into(),
            model_id_exact: Some("claude-opus-4.7".into()),
            activated_skills_count: 2,
            activated_agents_count: 1,
            activated_skill_names: Some(vec!["nontrivial-pipeline".into(), "qa-strategist".into()]),
            activated_agent_names: Some(vec!["rubber-duck".into()]),
            attempt: 1,
        }
    }

    #[test]
    fn reducer_starts_empty_with_zero_event_id() {
        let r = SessionStateReducer::new();
        let f = r.active_features_dag(true);
        let a = r.active_agents_dag(true);
        let s = r.active_session_snapshot(true);
        assert!(f.steps.is_empty());
        assert!(a.nodes.is_empty());
        assert!(a.edges.is_empty());
        assert_eq!(f.last_event_id, 0);
        assert_eq!(a.last_event_id, 0);
        assert_eq!(s.last_event_id, 0);
        assert!(s.mode.is_none());
        assert!(s.envelope.is_none());
    }

    #[test]
    fn features_dag_preserves_insertion_order_and_edges() {
        let mut r = SessionStateReducer::new();
        r.ingest(&plan_step(1, 10, "read_file", vec![]));
        r.ingest(&plan_step(2, 20, "edit_file", vec![10]));
        r.ingest(&plan_step(3, 30, "bash", vec![10, 20]));
        let f = r.active_features_dag(false);
        assert_eq!(f.steps.len(), 3);
        assert_eq!(f.steps[0].step_id, StepId(10));
        assert_eq!(f.steps[1].step_id, StepId(20));
        assert_eq!(f.steps[2].step_id, StepId(30));
        assert_eq!(f.steps[2].depends_on, vec![StepId(10), StepId(20)]);
        assert_eq!(f.last_event_id, 3);
        assert_eq!(f.revision, 3);
    }

    #[test]
    fn features_dag_legacy_zero_step_id_is_synthesized_from_ordinal() {
        // Pre-P13 producers send step_id=0 for everything. Our reducer must
        // still distinguish steps — use the ordinal.
        let mut r = SessionStateReducer::new();
        r.ingest(&plan_step(1, 0, "a", vec![]));
        r.ingest(&plan_step(2, 0, "b", vec![]));
        r.ingest(&plan_step(3, 0, "c", vec![]));
        let f = r.active_features_dag(false);
        assert_eq!(f.steps.len(), 3, "each legacy step stays distinct");
        let ids: Vec<u64> = f.steps.iter().map(|s| s.step_id.0).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn features_dag_amendment_updates_in_place() {
        let mut r = SessionStateReducer::new();
        r.ingest(&plan_step(1, 10, "read_file", vec![]));
        // Re-emit step_id=10 with a new description (amendment).
        let amended = AgentEvent::PlanStepPending {
            step: 1,
            step_id: StepId(10),
            revision: 1,
            plan_revision: 2,
            tool_name: "read_file".into(),
            description: "amended".into(),
            depends_on: vec![],
            parent_step_id: None,
        };
        r.ingest(&amended);
        let f = r.active_features_dag(false);
        assert_eq!(f.steps.len(), 1, "amendment must not duplicate");
        assert_eq!(f.steps[0].description, "amended");
        assert_eq!(f.steps[0].revision, 1);
    }

    #[test]
    fn agents_dag_fanout_lifecycle_tracked() {
        let mut r = SessionStateReducer::new();
        let fs = AgentEvent::Introspection(IntrospectionEventV1::FanoutStarted {
            step_id: StepId(5),
            parent_execution_id: ExecutionId(1),
            critic_count: 2,
            personas: vec!["rubber-duck".into(), "qa-strategist".into()],
        });
        r.ingest(&fs);
        let a = r.active_agents_dag(true);
        assert_eq!(a.active_fanouts.len(), 1);
        assert_eq!(a.active_fanouts[0].critic_count, 2);

        let sa_rd = AgentEvent::Introspection(IntrospectionEventV1::StepAssigned {
            assignment: assignment(100, 5, "rubber-duck"),
        });
        let sa_qa = AgentEvent::Introspection(IntrospectionEventV1::StepAssigned {
            assignment: assignment(101, 5, "qa-strategist"),
        });
        r.ingest(&sa_rd);
        r.ingest(&sa_qa);
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::AgentEdgeRecorded {
                edge: AgentEdgeKind::Critique,
                from_execution_id: ExecutionId(100),
                to_execution_id: ExecutionId(1),
            },
        ));
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::AgentEdgeRecorded {
                edge: AgentEdgeKind::Critique,
                from_execution_id: ExecutionId(101),
                to_execution_id: ExecutionId(1),
            },
        ));

        let a = r.active_agents_dag(true);
        assert_eq!(a.nodes.len(), 2);
        assert_eq!(a.edges.len(), 2);
        assert_eq!(
            a.active_fanouts.len(),
            1,
            "fanout still active until Completed"
        );

        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::FanoutCompleted {
                step_id: StepId(5),
                parent_execution_id: ExecutionId(1),
                critic_count: 2,
                blocking_count: 0,
            },
        ));
        let a = r.active_agents_dag(true);
        assert!(a.active_fanouts.is_empty(), "fanout cleared on completion");
    }

    #[test]
    fn agents_dag_spawn_event_adds_node_and_edge() {
        let mut r = SessionStateReducer::new();
        let spawn = AgentEvent::Introspection(IntrospectionEventV1::SubAgentSpawned {
            parent_execution_id: ExecutionId(1),
            child_session_id: "child-abc".into(),
            assignment: assignment(200, 5, "explorer"),
        });
        r.ingest(&spawn);
        let a = r.active_agents_dag(true);
        assert_eq!(a.nodes.len(), 1);
        assert_eq!(a.edges.len(), 1);
        assert_eq!(a.edges[0].kind, AgentEdgeKind::Spawn);
        assert_eq!(a.edges[0].from, ExecutionId(1));
        assert_eq!(a.edges[0].to, ExecutionId(200));
    }

    #[test]
    fn session_snapshot_tracks_mode_and_envelope() {
        let mut r = SessionStateReducer::new();
        r.ingest(&AgentEvent::ModeChanged {
            from_mode: "plan".into(),
            to_mode: "act".into(),
            from_lens: None,
            to_lens: Some("debug".into()),
        });
        let s = r.active_session_snapshot(true);
        assert_eq!(s.mode.as_deref(), Some("act"));
        assert_eq!(s.lens.as_deref(), Some("debug"));

        let envelope = EnvelopeSummaryV1 {
            read_scope_count: 3,
            write_scope_count: 1,
            write_deny_count: 2,
            network_enabled: true,
            exec_enabled: false,
            approval_cadence: "per_turn".into(),
            scope_source: "preset:act".into(),
            display_text: Some("Act mode envelope".into()),
        };
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::EnvelopeApplied {
                summary: envelope.clone(),
            },
        ));
        let s = r.active_session_snapshot(true);
        assert_eq!(s.envelope.as_ref().unwrap(), &envelope);
    }

    #[test]
    fn session_snapshot_tracks_awaiting_approval() {
        let mut r = SessionStateReducer::new();
        let critiques = vec![Critique {
            persona: "rubber-duck".into(),
            severity: CritiqueSeverity::Warn,
            findings: vec!["check edge case".into()],
            blocking: false,
        }];
        r.ingest(&AgentEvent::AwaitingApproval {
            plan_revision: "draft-1".into(),
            critiques: critiques.clone(),
        });
        let s = r.active_session_snapshot(true);
        let aa = s.awaiting_approval.unwrap();
        assert_eq!(aa.plan_revision, "draft-1");
        assert_eq!(aa.critiques, critiques);
    }

    #[test]
    fn session_snapshot_tracks_pending_scope_expansion() {
        let mut r = SessionStateReducer::new();
        r.ingest(&AgentEvent::ScopeExpansionRequested {
            capability: "write".into(),
            resource: "/etc/passwd".into(),
            reason: "NotInAllowList".into(),
            tool: "edit_file".into(),
        });
        let s = r.active_session_snapshot(true);
        let p = s.pending_scope_expansion.unwrap();
        assert_eq!(p.capability, "write");
        assert_eq!(p.resource, "/etc/passwd");
    }

    #[test]
    fn redaction_strips_sensitive_fields_when_not_trusted() {
        let mut r = SessionStateReducer::new();
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::StepAssigned {
                assignment: assignment(100, 5, "rubber-duck"),
            },
        ));
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::EnvelopeApplied {
                summary: EnvelopeSummaryV1 {
                    read_scope_count: 1,
                    write_scope_count: 0,
                    write_deny_count: 0,
                    network_enabled: true,
                    exec_enabled: false,
                    approval_cadence: "per_turn".into(),
                    scope_source: "preset:research".into(),
                    display_text: Some("secret-prompt-text".into()),
                },
            },
        ));

        // Untrusted consumer: sensitive fields stripped.
        let a = r.active_agents_dag(false);
        let node = &a.nodes[0].assignment;
        assert!(node.model_id_exact.is_none());
        assert!(node.activated_skill_names.is_none());
        assert!(node.activated_agent_names.is_none());
        // Coarse fields remain.
        assert_eq!(node.model_vendor, "anthropic");
        assert_eq!(node.model_tier, "opus");
        assert_eq!(node.activated_skills_count, 2);

        let s = r.active_session_snapshot(false);
        assert!(s.envelope.unwrap().display_text.is_none());

        // Trusted consumer: all fields present.
        let a = r.active_agents_dag(true);
        let node = &a.nodes[0].assignment;
        assert_eq!(node.model_id_exact.as_deref(), Some("claude-opus-4.7"));
        assert!(node.activated_skill_names.is_some());
        let s = r.active_session_snapshot(true);
        assert_eq!(
            s.envelope.unwrap().display_text.as_deref(),
            Some("secret-prompt-text")
        );
    }

    #[test]
    fn last_event_id_is_monotonic_and_consistent_across_views() {
        let mut r = SessionStateReducer::new();
        r.ingest(&plan_step(1, 10, "read", vec![]));
        r.ingest(&AgentEvent::ModeChanged {
            from_mode: "plan".into(),
            to_mode: "act".into(),
            from_lens: None,
            to_lens: Some("normal".into()),
        });
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::StepAssigned {
                assignment: assignment(100, 10, "builder"),
            },
        ));
        assert_eq!(r.last_event_id(), 3);
        assert_eq!(r.active_features_dag(false).last_event_id, 3);
        assert_eq!(r.active_agents_dag(false).last_event_id, 3);
        assert_eq!(r.active_session_snapshot(false).last_event_id, 3);
    }

    #[test]
    fn unknown_events_advance_event_id_but_do_not_mutate_state() {
        let mut r = SessionStateReducer::new();
        r.ingest(&AgentEvent::TextDelta {
            text: "hello".into(),
        });
        assert_eq!(r.last_event_id(), 1);
        assert!(r.active_features_dag(false).steps.is_empty());
        assert!(r.active_agents_dag(false).nodes.is_empty());
    }

    #[test]
    fn provenance_edges_accumulate_in_snapshot() {
        let mut r = SessionStateReducer::new();
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::ProvenanceRecorded {
                edge: ProvenanceEdgeKind::ExecutesStep,
                execution_id: ExecutionId(1),
                target_step_id: Some(StepId(10)),
            },
        ));
        r.ingest(&AgentEvent::Introspection(
            IntrospectionEventV1::ProvenanceRecorded {
                edge: ProvenanceEdgeKind::AmendsPlan,
                execution_id: ExecutionId(2),
                target_step_id: Some(StepId(20)),
            },
        ));
        let s = r.active_session_snapshot(true);
        assert_eq!(s.provenance.len(), 2);
        assert_eq!(s.provenance[0].kind, ProvenanceEdgeKind::ExecutesStep);
        assert_eq!(s.provenance[1].kind, ProvenanceEdgeKind::AmendsPlan);
    }

    // ── ReducerHandle tests ────────────────────────────────────────────────

    #[test]
    fn reducer_handle_ingest_matches_direct_reducer() {
        let direct = {
            let mut r = SessionStateReducer::new();
            r.ingest(&plan_step(1, 10, "bash", vec![]));
            r.ingest(&plan_step(2, 20, "edit", vec![10]));
            r.active_features_dag(false)
        };
        let h = ReducerHandle::new();
        h.ingest_event(&plan_step(1, 10, "bash", vec![]));
        h.ingest_event(&plan_step(2, 20, "edit", vec![10]));
        let via_handle = h.active_features_dag(false);
        assert_eq!(direct, via_handle);
    }

    #[test]
    fn reducer_handle_ingest_many_under_single_lock() {
        let h = ReducerHandle::new();
        let events = vec![
            plan_step(1, 10, "bash", vec![]),
            plan_step(2, 20, "edit", vec![10]),
            plan_step(3, 30, "test", vec![20]),
        ];
        let last = h.ingest_many(&events);
        assert_eq!(last, 3);
        assert_eq!(h.last_event_id(), 3);
        let dag = h.active_features_dag(false);
        assert_eq!(dag.steps.len(), 3);
        assert_eq!(dag.steps[2].depends_on, vec![StepId(20)]);
    }

    #[test]
    fn reducer_handle_clones_share_same_reducer() {
        let h1 = ReducerHandle::new();
        let h2 = h1.clone();
        h1.ingest_event(&plan_step(1, 7, "bash", vec![]));
        // h2 sees h1's ingestion — proof they share the same Arc<Mutex<…>>.
        assert_eq!(h2.last_event_id(), 1);
        assert_eq!(h2.active_features_dag(false).steps.len(), 1);
        // And the strong count reflects both handles.
        assert_eq!(h1._strong_count(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reducer_handle_is_introspection_sink_and_reduces_introspection_events() {
        use caduceus_orchestrator::IntrospectionSink;
        let h = ReducerHandle::new();
        // Feed an introspection event directly through the sink API.
        let ev = IntrospectionEventV1::EnvelopeApplied {
            summary: EnvelopeSummaryV1 {
                read_scope_count: 1,
                write_scope_count: 1,
                write_deny_count: 0,
                network_enabled: false,
                exec_enabled: false,
                approval_cadence: "never".into(),
                scope_source: "builtin".into(),
                display_text: Some("demo".into()),
            },
        };
        IntrospectionSink::emit(&h, ev).await;
        let snap = h.active_session_snapshot(true);
        assert!(snap.envelope.is_some());
        assert_eq!(
            snap.envelope.as_ref().unwrap().display_text.as_deref(),
            Some("demo")
        );
        // Redacted view strips display_text.
        let redacted = h.active_session_snapshot(false);
        assert!(redacted.envelope.as_ref().unwrap().display_text.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn reducer_handle_concurrent_emitters_do_not_lose_events() {
        use caduceus_orchestrator::IntrospectionSink;
        let h = ReducerHandle::new();
        let mut set = Vec::new();
        for i in 0..32u64 {
            let h = h.clone();
            set.push(tokio::spawn(async move {
                let ev = IntrospectionEventV1::ProvenanceRecorded {
                    edge: ProvenanceEdgeKind::ExecutesStep,
                    execution_id: ExecutionId(i),
                    target_step_id: Some(StepId(i)),
                };
                IntrospectionSink::emit(&h, ev).await;
            }));
        }
        for j in set {
            j.await.unwrap();
        }
        assert_eq!(h.last_event_id(), 32);
        assert_eq!(h.active_session_snapshot(false).provenance.len(), 32);
    }

    /// Phase-H F1: a poisoned reducer mutex must not bring down the
    /// forwarder. After another thread panics while holding the lock, the
    /// handle MUST keep serving reads/writes (possibly with the data the
    /// panic left behind) rather than propagating the poison upward.
    #[test]
    fn poisoned_reducer_recovers_and_keeps_serving() {
        let h = ReducerHandle::new();
        h.ingest_event(&plan_step(0, 100, "web_fetch", vec![]));
        assert_eq!(h.last_event_id(), 1);

        // Poison the mutex by panicking inside a scoped thread that held
        // the inner lock.
        let inner = h.inner.clone();
        let t = std::thread::spawn(move || {
            let _g = inner.lock().unwrap();
            panic!("deliberate panic to poison reducer mutex");
        });
        let join = t.join();
        assert!(join.is_err(), "poison thread must have panicked");

        // All query APIs must still work — they recover from poison
        // transparently and return the post-panic state (which is the
        // pre-panic state since the thread did nothing before panicking).
        assert_eq!(h.last_event_id(), 1, "last_event_id must survive poison");
        let _snap = h.active_session_snapshot(false);
        let _agents = h.active_agents_dag(false);
        let _features = h.active_features_dag(false);

        // Writes must also keep working.
        h.ingest_event(&plan_step(1, 101, "read_file", vec![StepId(100).0]));
        assert_eq!(
            h.last_event_id(),
            2,
            "writes after poison must advance event id"
        );
    }

    /// Phase-H H5: the reducer's agent-edge and provenance vecs are
    /// bounded. Once the cap is hit, the oldest element is evicted
    /// (FIFO) and the corresponding `*_dropped` counter bumps. The
    /// counters surface on the DAG / snapshot projections so the UI
    /// can render a "graph truncated" marker.
    #[test]
    fn bounded_agent_edges_evict_fifo_and_bump_drop_counter() {
        use caduceus_core::AgentEdgeKind;
        let mut r = SessionStateReducer::new();
        // Seed one node so edges reference something real; not required
        // for the bound, but keeps the shape realistic.
        r.insert_agent_node(assignment(1, 1, "p"));

        // Push CAP + 3 edges so we know the last 3 survive and the
        // drop counter reflects the overflow.
        let overflow = 3;
        for i in 0..(SessionStateReducer::AGENT_EDGE_CAP + overflow) {
            r.push_agent_edge(AgentEdgeV1 {
                kind: AgentEdgeKind::Spawn,
                from: ExecutionId(1),
                to: ExecutionId((i as u64) + 2),
            });
        }
        let dag = r.active_agents_dag(false);
        assert_eq!(dag.edges.len(), SessionStateReducer::AGENT_EDGE_CAP);
        assert_eq!(dag.edges_dropped, overflow as u64);
        // FIFO: the oldest `overflow` edges were dropped, so the first
        // retained edge's `to` is the (overflow+1)-th pushed value.
        assert_eq!(
            dag.edges.first().unwrap().to,
            ExecutionId(overflow as u64 + 2)
        );
    }

    #[test]
    fn bounded_provenance_evicts_fifo_and_bump_drop_counter() {
        use caduceus_core::ProvenanceEdgeKind;
        let mut r = SessionStateReducer::new();
        let overflow = 2;
        for i in 0..(SessionStateReducer::PROVENANCE_CAP + overflow) {
            r.push_provenance(ProvenanceEdgeV1 {
                kind: ProvenanceEdgeKind::ExecutesStep,
                execution_id: ExecutionId((i as u64) + 1),
                target_step_id: None,
            });
        }
        let snap = r.active_session_snapshot(false);
        assert_eq!(snap.provenance.len(), SessionStateReducer::PROVENANCE_CAP);
        assert_eq!(snap.provenance_dropped, overflow as u64);
    }
}
