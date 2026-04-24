//! Process-wide Tokio runtime used by the native-loop path when it runs
//! inside GPUI's foreground executor (which is NOT a Tokio runtime).
//!
//! # Why this exists
//!
//! `caduceus_bridge::orchestrator::OrchestratorBridge::run_caduceus_loop_translated`
//! (and the `spawn_forwarder` helper it relies on) calls `tokio::spawn`
//! and `tokio::time::timeout`. Both require a Tokio runtime installed on
//! the calling thread (`Handle::current()` panics with `TryCurrentError`
//! otherwise).
//!
//! When the IDE drives the native loop through `agent::thread::Thread::
//! try_run_turn_native`, that future is polled by GPUI's foreground
//! executor — a `smol`-based executor with no Tokio runtime. The result
//! was an immediate `abort()` on the first turn (crash report
//! `35A11661-8255-4A8C-899F-339C495A7492`, 2026-04-24).
//!
//! # What this provides
//!
//! A single multi-thread Tokio runtime, created lazily on first access,
//! owned for the lifetime of the process. Callers obtain the handle via
//! [`bridge_runtime_handle`] and either:
//!
//! 1. call `handle.enter()` to install the runtime on the current
//!    thread (sufficient for async code that only calls `tokio::spawn`
//!    / `tokio::time::*`), or
//! 2. delegate an async block to `handle.spawn(...)` / `handle.block_on`
//!    when they want the work to run on the bridge runtime threads.
//!
//! # Why a dedicated runtime (not `Handle::try_current`)
//!
//! - Tests use `#[tokio::test]`, which installs its own runtime — those
//!   paths never touch this module.
//! - Production callers from GPUI have no runtime; spawning a fresh one
//!   per turn would be wasteful and leak threads. A process-wide
//!   [`LazyLock`]-owned runtime avoids both problems.

use std::sync::LazyLock;
use tokio::runtime::{Handle, Runtime};

static BRIDGE_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .thread_name("caduceus-bridge-rt")
        .enable_all()
        .build()
        .expect("failed to build caduceus_bridge Tokio runtime")
});

/// Returns a handle to the process-wide bridge runtime.
///
/// Cheap: first call initializes the runtime; subsequent calls hand back
/// the same handle.
pub fn bridge_runtime_handle() -> Handle {
    BRIDGE_RUNTIME.handle().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_runtime_spawns_tasks() {
        // Smoke: the runtime actually runs work. We use `block_on` here
        // because the test itself has no runtime (plain `#[test]`).
        let handle = bridge_runtime_handle();
        let out = handle.block_on(async {
            let jh = tokio::spawn(async { 42_u32 });
            jh.await.unwrap()
        });
        assert_eq!(out, 42);
    }

    #[test]
    fn enter_guard_makes_handle_current_resolve() {
        // Enter the runtime on the current thread and verify
        // `Handle::current()` no longer panics.
        let handle = bridge_runtime_handle();
        let _guard = handle.enter();
        let _current = Handle::current();
    }

    #[test]
    fn handle_is_stable_across_calls() {
        let a = bridge_runtime_handle();
        let b = bridge_runtime_handle();
        // Second handle drives the same runtime: spawning via `b` and
        // awaiting via `a.block_on` must work.
        let v = a.block_on(async move { b.spawn(async { 7_u32 }).await.unwrap() });
        assert_eq!(v, 7);
    }
}
