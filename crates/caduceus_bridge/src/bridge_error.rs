//! T6 — structured `BridgeError` taxonomy.
//!
//! Audit finding: 14 `Result<_, String>` return sites in
//! `caduceus_bridge::orchestrator` used `.map_err(|e| e.to_string())`
//! to erase error type information. That stringification hurt
//! debuggability (no source chain, no category) and prevented
//! structured handling at the IDE boundary. This module replaces
//! those `String` errors with a thin typed enum that preserves the
//! originating error variant while still implementing `Display` +
//! `std::error::Error` so existing `.unwrap()` / `to_string()` /
//! `?` call sites keep working.
//!
//! Scope is intentionally narrow: a thin wrapper around the three
//! concrete upstream error types that were being flattened
//! (`serde_json::Error`, `caduceus_core::CaduceusError`, and the
//! various bridge-internal `InstructionError` / `ConversationError`
//! kinds currently surfaced via `Display`). Consumers that need a
//! string still call `.to_string()`.

use thiserror::Error;

/// Typed error surface for `caduceus_bridge::orchestrator` APIs.
///
/// Each variant carries the originating error value so callers can
/// match (e.g. retry transient `Engine(CaduceusError::RateLimited)`)
/// or log with full context. The `Display` impl forwards to the
/// inner error, so the legacy stringification at the IDE boundary
/// is preserved byte-for-byte.
#[derive(Debug, Error)]
pub enum BridgeError {
    /// JSON (de)serialisation failure — snapshot_plan_json,
    /// list_checkpoints_json, parse_notification,
    /// snapshot_memory_blocks_json, snapshot_bradley_terry_json,
    /// load_bradley_terry_json, wrap_event_json,
    /// parse_versioned_event_json.
    #[error("{0}")]
    Serde(#[from] serde_json::Error),

    /// Engine turn failure — run_caduceus_loop*, run_turn, stream_turn.
    /// Preserves the full `CaduceusError` variant so retry/backoff
    /// logic can introspect without substring matching.
    #[error("{0}")]
    Engine(#[from] caduceus_core::CaduceusError),

    /// Flat string errors from upstream crates that do not yet
    /// expose a structured error type (InstructionLoader,
    /// ConversationHistory serialize/deserialize, list_bundled_skills).
    /// Variant exists so callers can still distinguish "engine
    /// turn failed" from "workspace instructions malformed"
    /// without regex.
    #[error("{0}")]
    Workspace(String),

    /// Native-loop feature flag is off. Caller should either
    /// enable `caduceus.native_loop` or fall back to the legacy
    /// path.
    #[error(
        "caduceus native loop is disabled; enable via \
         set_native_loop_enabled(true) (flag: caduceus.native_loop)"
    )]
    NativeLoopDisabled,
}

impl BridgeError {
    /// Convenience constructor for the `Workspace` variant so
    /// callsites don't need to type the full path.
    pub fn workspace(msg: impl Into<String>) -> Self {
        Self::Workspace(msg.into())
    }

    /// Convenience helper for tests that previously compared
    /// against a raw `String` error — `err.to_string()` still
    /// works (via `Display`), but callers can also use this.
    pub fn is_native_loop_disabled(&self) -> bool {
        matches!(self, Self::NativeLoopDisabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_loop_disabled_display_matches_legacy_string() {
        let err = BridgeError::NativeLoopDisabled;
        let s = err.to_string();
        assert!(s.contains("native loop is disabled"));
        assert!(s.contains("caduceus.native_loop"));
    }

    #[test]
    fn serde_error_propagates_from() {
        let json_err: serde_json::Error =
            serde_json::from_str::<serde_json::Value>("not-json").unwrap_err();
        let bridge: BridgeError = json_err.into();
        assert!(matches!(bridge, BridgeError::Serde(_)));
    }

    #[test]
    fn workspace_constructor() {
        let err = BridgeError::workspace("something bad");
        assert_eq!(err.to_string(), "something bad");
        assert!(matches!(err, BridgeError::Workspace(_)));
    }

    #[test]
    fn is_native_loop_disabled_discriminates() {
        assert!(BridgeError::NativeLoopDisabled.is_native_loop_disabled());
        assert!(!BridgeError::workspace("x").is_native_loop_disabled());
    }
}
