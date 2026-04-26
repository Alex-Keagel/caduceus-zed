//! T9 (plan v3.1 §6): Copilot's `Status::Disabled` must map to
//! `ProviderAuthState::DisabledByPolicy(SanitizedReason)` — NOT to
//! `NotAuthenticated { action: SignInImperative }`.
//!
//! T8 (auth_state_mapping.rs) already verifies that copilot_chat.rs routes
//! through `disabled_by_policy(...)`. T9 pins:
//!   1. The EXACT user-facing message string (so a reword can't slip past
//!      review without updating the test).
//!   2. The runtime contract that `disabled_by_policy(message)` yields a
//!      `DisabledByPolicy(SanitizedReason)` whose `as_str()` round-trips
//!      that message through the sanitizer.
//!   3. The dispatcher predicate (`is_configured` / `can_provide_models`)
//!      treats DisabledByPolicy as configured-but-terminal.
//!
//! Why integration test, not lib unit test: `cargo test -p language_models
//! --lib` does not compile on this branch (pre-existing `gpui::test` macro
//! issues in unrelated provider tests). Integration tests don't pull those
//! modules into their cfg(test) graph.

use language_model::ProviderAuthState;

/// The literal string copilot_chat.rs::auth_state passes to
/// `ProviderAuthState::disabled_by_policy(...)`. Keep in lock-step with that
/// site — if the message is reworded, update both.
const COPILOT_DISABLED_MESSAGE: &str =
    "Copilot is disabled. Enable Copilot in your organization settings to use Copilot Chat.";

#[test]
fn t9_copilot_chat_disabled_status_maps_to_disabled_by_policy() {
    // 1. Source-anchor: copilot_chat.rs still contains the canonical literal
    //    AND still routes Status::Disabled through `disabled_by_policy(...)`.
    let source = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/provider/copilot_chat.rs"
    ))
    .expect("read copilot_chat.rs");
    assert!(
        source.contains(COPILOT_DISABLED_MESSAGE),
        "T9: copilot_chat.rs no longer contains the canonical disabled \
         message. Update this test (and any UI strings table) if the \
         message changed intentionally."
    );
    assert!(
        source.contains("ProviderAuthState::disabled_by_policy("),
        "T9: copilot_chat.rs no longer routes Status::Disabled through \
         ProviderAuthState::disabled_by_policy"
    );

    // 2. Runtime contract: the constructor yields DisabledByPolicy and the
    //    sanitized reason round-trips the message bytes (no Bearer tokens or
    //    oversize length to redact).
    let state = ProviderAuthState::disabled_by_policy(COPILOT_DISABLED_MESSAGE);
    match state {
        ProviderAuthState::DisabledByPolicy(reason) => {
            assert_eq!(reason.as_str(), COPILOT_DISABLED_MESSAGE);
        }
        other => panic!(
            "T9: disabled_by_policy(...) produced unexpected variant: {:?}",
            other
        ),
    }

    // 3. Dispatcher contract: per `ProviderAuthState::is_configured` doc:
    //    "Disabled-by-policy is NOT configured (admin would need to flip a flag)".
    //    Cannot provide models either.
    let s = ProviderAuthState::disabled_by_policy(COPILOT_DISABLED_MESSAGE);
    assert!(
        !s.is_configured(),
        "DisabledByPolicy must NOT be is_configured() (per auth_state.rs doc — admin must intervene)"
    );
    assert!(
        !s.can_provide_models(),
        "DisabledByPolicy must NOT can_provide_models()"
    );
}
