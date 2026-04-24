//! Caduceus Bridge — direct Rust integration of the Caduceus AI engine into Zed.
//!
//! No MCP, no JSON-RPC, no serialization overhead. Native Rust types.
//!
//! This crate provides:
//! - `CaduceusEngine` — singleton holding all engine state
//! - Tool execution via the `ToolRegistry`
//! - Semantic search via `SemanticIndex`
//! - Code intelligence via `CodePropertyGraph`
//! - Git operations via `GitRepo`
//! - Security scanning via `SastScanner`
//! - Memory persistence via filesystem
//! - Session management via `SqliteStorage`

pub mod bridge_error;
pub mod context_events;
pub mod crdt;
pub mod dag_state;
pub mod engine;
pub mod event_translator;
pub mod git;
pub mod index_dag;
pub mod marketplace;
pub mod memory;
pub mod orchestrator;
pub mod runtime;
pub mod safety;
pub mod search;
pub mod security;
pub mod storage;
pub mod tokio_rt;
pub mod telemetry;
pub mod tools;
pub mod tree_sitter;

pub use engine::CaduceusEngine;

// ST-B2 / contract `envelope-surface-v1` — Bridge re-exports
// `PermissionEnvelope` as `PermissionEnvelope` so downstream IDE
// consumers (agent panel, settings UI, tool gates) don't have to reach
// across crates into `caduceus-permissions`. The type *is* the
// orchestrator's `caduceus_permissions::PermissionEnvelope` — no serde
// mirror, no boilerplate copy — so preset bytes round-trip byte-equal
// in both directions (golden-bytes test in `orchestrator::tests`).
pub use caduceus_permissions::envelope::{
    ApprovalCadence, EnvelopeScope, ExecPolicy, FanoutPolicy, NetworkPolicy, PathAllowlist,
    PermissionEnvelope,
};

// ST-B3 / contract `context-injector-v1` — bridge re-exports the
// scoped-context surface so IDE callers install injectors without
// importing `caduceus-orchestrator` directly. The default installed by
// `build_harness_with_injector` when the caller passes `None` is
// `BuiltinScopedContextInjector::default()` — preserves today's behaviour.
pub use caduceus_orchestrator::{
    BuiltinScopedContextInjector, ContextInjector, PassthroughContextInjector, ScopeRequest,
    ScopedContext,
};

// P13d — live two-DAG state reducer (features + agents + snapshot). IDE
// clients feed AgentEvents in and call `active_*` projections to render.
pub use dag_state::{
    ActiveFanoutV1, AgentEdgeV1, AgentNodeV1, AgentsDagV1, AwaitingApprovalV1, FeatureStepStatus,
    FeatureStepV1, FeaturesDagV1, PendingScopeExpansionV1, ProvenanceEdgeV1, ReducerHandle,
    SessionSnapshotV1, SessionStateReducer,
};
