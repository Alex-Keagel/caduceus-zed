//! G1a — per-Thread engine state held behind an async mutex so a turn
//! task can lock it for the entire duration of a
//! `run_caduceus_loop_translated` call without interleaving with
//! ingestion events or another turn start.
//!
//! `SessionState` + `ConversationHistory` live here instead of on the
//! harness itself because `AgentHarness::run` takes them by `&mut` —
//! sharing via `Arc` requires explicit synchronization. A tokio mutex
//! (not std) is used because the lock spans `.await` points.
//!
//! **Panic safety.** On panic inside the harness run, the mutex is
//! poisoned; `Thread::invalidate_caduceus_harness` nukes the state,
//! harness, and cancel token together and the next turn rebuilds from
//! defaults. Any in-flight history deltas that were in this struct but
//! not yet persisted are lost — accepted cost of a panic.
//!
//! **Generation fence.** The turn task stamps `turn_generation_at_lock`
//! when it acquires the mutex; after `harness.run` returns it re-reads
//! the Thread's current `turn_generation` and drops the writeback if
//! they differ (meaning a newer turn already owns the next mutation).

use caduceus_core::{ModelId, ProviderId, SessionState};
use caduceus_orchestrator::ConversationHistory;
use std::path::PathBuf;

/// Mutable engine state owned by one `Thread`. Behind
/// `Arc<tokio::sync::Mutex<_>>`.
#[allow(dead_code)] // fields consumed by G1b/G1d turn dispatcher
pub struct NativeLoopState {
    pub session: SessionState,
    pub history: ConversationHistory,
    /// The turn-generation the task recorded when it acquired the
    /// mutex. If Thread's current generation has advanced by the time
    /// the task is ready to write back, the writeback is skipped.
    pub turn_generation_at_lock: u64,
}

impl NativeLoopState {
    /// Build a fresh state bound to the given project root. Provider +
    /// model IDs are placeholders until G1d wires the real LLM binding
    /// from the Thread's selected model.
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            session: SessionState::new(
                project_root,
                ProviderId("zed-native-loop".into()),
                ModelId("pending".into()),
            ),
            history: ConversationHistory::new(),
            turn_generation_at_lock: 0,
        }
    }
}
