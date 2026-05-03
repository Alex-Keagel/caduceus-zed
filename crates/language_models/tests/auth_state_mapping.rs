//! T8 (STOP-SHIP gate) — parametric provider-auth-state mapping test.
//!
//! The contract we lock in here is plan v3.1 §3 (AuthAction taxonomy):
//! every shipped LanguageModelProvider must, when its inner state reports
//! "not authenticated", return the `ProviderAuthState` variant + `AuthAction`
//! the dispatcher (ST1b) was built for.
//!
//! Why a SOURCE-anchor test instead of constructing every provider:
//! - 14 of the 16 providers require `Arc<dyn HttpClient>` + `Arc<dyn CredentialsProvider>`
//!   to construct, plus a tokio runtime (Bedrock additionally needs an AWS SDK
//!   client). Wiring all those fakes up just to call a 6-line `auth_state`
//!   would be more brittle than the contract it tests.
//! - Copilot's Disabled-by-policy path additionally requires a
//!   `GlobalCopilotAuth` resource and a fake `Status::Disabled` value.
//! - Bedrock (per fix-loop #5) explicitly defers its unit-level pinning to
//!   this T8 harness.
//!
//! Trade-off accepted: the test asserts on a hand-written EXPECTED table and
//! grep-matches the source. A reviewer can (and should) eyeball both. If a
//! future PR rewrites a provider's `auth_state` body without updating this
//! table, CI will fail with a clear diff.
//!
//! Plan ref: v3.1 §3 (taxonomy), §6 T8 (STOP-SHIP), §B4 risk #1 (Bedrock).

use std::path::PathBuf;

struct Expected {
    file: &'static str,
    /// One of: "EnterApiKeyInSettings", "SignInImperative", "None".
    expected_action: &'static str,
    /// Whether the provider may map to `DisabledByPolicy` (Copilot only).
    can_be_disabled_by_policy: bool,
}

const EXPECTED: &[Expected] = &[
    Expected {
        file: "anthropic.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "open_ai.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "google.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "ollama.rs",
        expected_action: "None",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "lmstudio.rs",
        expected_action: "None",
        can_be_disabled_by_policy: false,
    },
    // fix-loop #5: AWS IAM/STS — no in-app remediation.
    Expected {
        file: "bedrock.rs",
        expected_action: "None",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "mistral.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "deepseek.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "vercel.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "vercel_ai_gateway.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "open_router.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "opencode.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "x_ai.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    Expected {
        file: "open_ai_compatible.rs",
        expected_action: "EnterApiKeyInSettings",
        can_be_disabled_by_policy: false,
    },
    // Copilot has TWO unauth branches: Status::Disabled → DisabledByPolicy,
    // otherwise SignInImperative.
    Expected {
        file: "copilot_chat.rs",
        expected_action: "SignInImperative",
        can_be_disabled_by_policy: true,
    },
    Expected {
        file: "cloud.rs",
        expected_action: "SignInImperative",
        can_be_disabled_by_policy: false,
    },
];

fn provider_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("provider")
}

fn extract_auth_state_body(path: &std::path::Path) -> String {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    let start = source
        .find("fn auth_state(")
        .unwrap_or_else(|| panic!("no `fn auth_state(` in {}", path.display()));
    let body_start = source[start..]
        .find('{')
        .map(|o| start + o)
        .unwrap_or_else(|| panic!("no opening brace for auth_state in {}", path.display()));
    let mut depth = 0i32;
    for (i, b) in source.bytes().enumerate().skip(body_start) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return source[start..=i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unterminated auth_state body in {}", path.display());
}

#[test]
fn t8_every_provider_maps_unauth_to_expected_action() {
    let dir = provider_dir();

    // Lock the size of the table — plan v3.1 §6 lists 16 providers.
    assert_eq!(
        EXPECTED.len(),
        16,
        "T8: plan v3.1 §6 lists 16 providers; got {}",
        EXPECTED.len()
    );

    let mut failures = Vec::new();
    for ent in EXPECTED {
        let path = dir.join(ent.file);
        if !path.exists() {
            failures.push(format!(
                "{}: file not found at {}",
                ent.file,
                path.display()
            ));
            continue;
        }
        let body = extract_auth_state_body(&path);

        if !body.contains("ProviderAuthState::Authenticated") {
            failures.push(format!(
                "{}: auth_state body missing `ProviderAuthState::Authenticated` arm",
                ent.file
            ));
        }

        let needle = format!("AuthAction::{}", ent.expected_action);
        if !body.contains(&needle) {
            failures.push(format!(
                "{}: expected `{}` in auth_state body, but it was absent.\n--- body ---\n{}\n---",
                ent.file, needle, body
            ));
        }

        for forbidden in ["EnterApiKeyInSettings", "SignInImperative", "None"]
            .iter()
            .filter(|a| **a != ent.expected_action)
        {
            if ent.can_be_disabled_by_policy && *forbidden == "SignInImperative" {
                continue;
            }
            let forbidden_full = format!("AuthAction::{}", forbidden);
            if body.contains(&forbidden_full) {
                failures.push(format!(
                    "{}: contains forbidden `{}` (expected only `AuthAction::{}`)",
                    ent.file, forbidden_full, ent.expected_action
                ));
            }
        }

        let has_disabled = body.contains("DisabledByPolicy") || body.contains("disabled_by_policy");
        if has_disabled != ent.can_be_disabled_by_policy {
            failures.push(format!(
                "{}: DisabledByPolicy presence mismatch (got {}, expected {})",
                ent.file, has_disabled, ent.can_be_disabled_by_policy
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "T8 STOP-SHIP gate failed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn t8_table_covers_every_registered_provider_file() {
    let dir = provider_dir();
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.extension()? == "rs" {
                Some(p.file_name()?.to_str()?.to_owned())
            } else {
                None
            }
        })
        .collect();
    on_disk.sort();

    let in_table: std::collections::HashSet<&str> = EXPECTED.iter().map(|e| e.file).collect();
    let unknown: Vec<String> = on_disk
        .into_iter()
        .filter(|f| !in_table.contains(f.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "T8: provider files present but not in EXPECTED table: {:?}\n\
         (Add a row to crates/language_models/tests/auth_state_mapping.rs::EXPECTED.)",
        unknown
    );
}
