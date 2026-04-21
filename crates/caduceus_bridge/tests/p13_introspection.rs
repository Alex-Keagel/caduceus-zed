//! P13e — integration tests for the two-DAG introspection surface.
//!
//! Drives the orchestrator's `critique_fanout` with a live
//! `IntrospectionSink` that forwards events into the bridge's
//! `SessionStateReducer`, then asserts the Features-DAG, Agents-DAG and
//! session snapshot reflect what actually happened.

use async_trait::async_trait;
use caduceus_bridge::dag_state::SessionStateReducer;
use caduceus_bridge::orchestrator::{list_bundled_skills, list_models, list_modes, list_personas};
use caduceus_core::{
    AgentEdgeKind, AgentEvent, Critique, CritiqueSeverity, EnvelopeSummaryV1, ExecutionId,
    IntrospectionEventV1, StepId,
};
use caduceus_orchestrator::critique_fanout::{
    CritiqueRunner, FanoutIntrospectionCtx, IntrospectionSink,
    spawn_critique_fanout_with_introspection,
};
use caduceus_orchestrator::modes::PersonaRegistry;
use caduceus_permissions::envelope::{FanoutPolicy, PermissionEnvelope};
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;

/// Bridge-layer sink that folds every introspection event into a reducer
/// (wrapped in a Mutex so the `&self` + `Send + Sync` constraints of
/// `IntrospectionSink` compose with the orchestrator's parallel driver).
struct ReducerSink {
    reducer: Mutex<SessionStateReducer>,
}
impl ReducerSink {
    fn new() -> Self {
        Self {
            reducer: Mutex::new(SessionStateReducer::new()),
        }
    }
    fn snapshot(
        &self,
    ) -> (
        caduceus_bridge::dag_state::FeaturesDagV1,
        caduceus_bridge::dag_state::AgentsDagV1,
        caduceus_bridge::dag_state::SessionSnapshotV1,
    ) {
        let r = self.reducer.lock().unwrap();
        (
            r.active_features_dag(true),
            r.active_agents_dag(true),
            r.active_session_snapshot(true),
        )
    }
}
#[async_trait]
impl IntrospectionSink for ReducerSink {
    async fn emit(&self, event: IntrospectionEventV1) {
        self.reducer
            .lock()
            .unwrap()
            .ingest(&AgentEvent::Introspection(event));
    }
}

struct OkRunner;
#[async_trait]
impl CritiqueRunner for OkRunner {
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

#[tokio::test]
async fn p13e_parallel_fanout_populates_agents_dag_with_nodes_and_edges() {
    let sink = ReducerSink::new();
    let env = {
        let mut e = PermissionEnvelope::research_preset();
        e.fanout_policy = FanoutPolicy::MultiPersona;
        e
    };
    let reg = PersonaRegistry::builtin_personas();
    let alloc = AtomicU64::new(100);
    let ctx = FanoutIntrospectionCtx {
        sink: &sink,
        primary_execution_id: ExecutionId(1),
        step_id: StepId(42),
        execution_id_allocator: &alloc,
    };

    // Seed the primary executor's assignment so edges have both endpoints
    // present in the reducer — matches how the orchestrator will wire
    // the driver once lifecycle emission lands for the primary.
    sink.emit(IntrospectionEventV1::StepAssigned {
        assignment: caduceus_core::AssignmentSummaryV1 {
            execution_id: ExecutionId(1),
            step_id: StepId(42),
            persona_id: "planner".into(),
            model_vendor: "anthropic".into(),
            model_tier: "opus".into(),
            model_id_exact: None,
            activated_skills_count: 0,
            activated_agents_count: 0,
            activated_skill_names: None,
            activated_agent_names: None,
            attempt: 1,
        },
    })
    .await;

    let got = spawn_critique_fanout_with_introspection(
        &env,
        "plan body",
        &["cloud", "qa"],
        &reg,
        &OkRunner,
        Some(&ctx),
    )
    .await;
    assert_eq!(got.len(), 3, "rubber-duck + cloud + qa");

    let (_features, agents, snapshot) = sink.snapshot();

    // Nodes: 1 primary + 3 critics = 4.
    assert_eq!(agents.nodes.len(), 4, "got nodes {:?}", agents.nodes);
    // Critics carry unique pre-minted execution ids 100..=102.
    let critic_ids: Vec<u64> = agents
        .nodes
        .iter()
        .filter(|n| n.assignment.execution_id != ExecutionId(1))
        .map(|n| n.assignment.execution_id.0)
        .collect();
    let mut sorted = critic_ids.clone();
    sorted.sort();
    assert_eq!(sorted, vec![100, 101, 102]);

    // Edges: one Critique edge from each critic → primary.
    assert_eq!(agents.edges.len(), 3);
    for e in &agents.edges {
        assert_eq!(e.kind, AgentEdgeKind::Critique);
        assert_eq!(e.to, ExecutionId(1));
        assert!(critic_ids.contains(&e.from.0));
    }

    // Fanout lifecycle must be balanced — no leftover active_fanouts.
    assert!(
        agents.active_fanouts.is_empty(),
        "FanoutCompleted must clear the active entry"
    );

    // Snapshot has no mode/envelope set (we didn't emit those events).
    assert!(snapshot.mode.is_none());
    assert!(snapshot.envelope.is_none());

    // last_event_id monotonic across all three projections.
    let eid = snapshot.last_event_id;
    assert!(eid > 0);
    assert_eq!(agents.last_event_id, eid);
}

#[tokio::test]
async fn p13e_fanout_failure_surfaces_blocking_critique_and_completed_count() {
    struct FailingRunner;
    #[async_trait]
    impl CritiqueRunner for FailingRunner {
        async fn critique(
            &self,
            _persona: &str,
            _prefix: &str,
            _plan: &str,
            _env: &PermissionEnvelope,
        ) -> Result<Critique, anyhow::Error> {
            Err(anyhow::anyhow!("synthetic"))
        }
    }

    let sink = ReducerSink::new();
    let env = PermissionEnvelope::research_preset();
    let reg = PersonaRegistry::builtin_personas();
    let alloc = AtomicU64::new(1);
    let ctx = FanoutIntrospectionCtx {
        sink: &sink,
        primary_execution_id: ExecutionId(0),
        step_id: StepId(1),
        execution_id_allocator: &alloc,
    };

    let got = spawn_critique_fanout_with_introspection(
        &env,
        "plan",
        &[],
        &reg,
        &FailingRunner,
        Some(&ctx),
    )
    .await;
    assert_eq!(got.len(), 1);
    assert!(got[0].blocking);

    let (_f, agents, _s) = sink.snapshot();
    assert_eq!(agents.nodes.len(), 1);
    assert_eq!(agents.edges.len(), 1);
    assert!(agents.active_fanouts.is_empty());
}

#[test]
fn p13e_catalogs_are_populated_and_aligned() {
    // Personas catalog exposes the domain specialists used by the fan-out.
    let personas = list_personas();
    for want in [
        "rubber-duck",
        "cloud-architect",
        "ml-architect",
        "qa-strategist",
    ] {
        assert!(
            personas.iter().any(|p| p.name == want),
            "persona catalog missing {want}"
        );
    }

    // Modes catalog has the 4 canonical modes.
    let modes = list_modes();
    let names: Vec<&str> = modes.iter().map(|m| m.name.as_str()).collect();
    for want in ["plan", "research", "act", "autopilot"] {
        assert!(names.contains(&want), "mode catalog missing {want}");
    }

    // Persona.default_mode MUST reference a mode that exists in the catalog.
    // Cross-catalog integrity check — catches drift between `modes.rs` and
    // the persona registry.
    for p in &personas {
        assert!(
            names.contains(&p.default_mode.as_str()),
            "persona {} default_mode={} missing from list_modes()",
            p.name,
            p.default_mode
        );
    }

    // Models catalog exposes vendor tiers without exact ids.
    let models = list_models();
    assert!(!models.is_empty());
    assert!(
        models
            .iter()
            .any(|m| m.vendor == "anthropic" && m.tier == "opus")
    );

    // list_bundled_skills is callable on any path (empty workspace → empty).
    let tmp = tempfile::tempdir().unwrap();
    let skills = list_bundled_skills(tmp.path()).unwrap();
    assert!(skills.is_empty());
}

#[test]
fn p13e_reducer_handles_full_mode_and_envelope_flow() {
    let mut r = SessionStateReducer::new();
    r.ingest(&AgentEvent::ModeChanged {
        from_mode: "plan".into(),
        to_mode: "act".into(),
        from_lens: None,
        to_lens: Some("debug".into()),
    });
    r.ingest(&AgentEvent::Introspection(
        IntrospectionEventV1::EnvelopeApplied {
            summary: EnvelopeSummaryV1 {
                read_scope_count: 5,
                write_scope_count: 2,
                write_deny_count: 1,
                network_enabled: true,
                exec_enabled: true,
                approval_cadence: "per_turn".into(),
                scope_source: "preset:act".into(),
                display_text: Some("internal prompt text".into()),
            },
        },
    ));

    let trusted = r.active_session_snapshot(true);
    assert_eq!(trusted.mode.as_deref(), Some("act"));
    assert_eq!(trusted.lens.as_deref(), Some("debug"));
    assert_eq!(
        trusted.envelope.as_ref().unwrap().display_text.as_deref(),
        Some("internal prompt text")
    );

    let untrusted = r.active_session_snapshot(false);
    assert!(untrusted.envelope.unwrap().display_text.is_none());
}
