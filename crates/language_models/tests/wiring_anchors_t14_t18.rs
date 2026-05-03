//! T14–T18 (plan v3.1 §6) — UI-wiring source-anchor tests + CI grep gate.
//!
//! These tests live as **integration tests** of the `language_models` crate
//! because the `--lib` test cfg of language_models pre-existing-fails to
//! compile on this branch (unrelated `gpui::test` macro issues in some
//! provider modules; see PR notes). Integration tests don't pull those
//! modules into their cfg(test) graph, so they run cleanly.
//!
//! Why source-anchor: each of T14-T17 names a single line of UI code that
//! must call `auth_state(cx)` (or `is_configured()`/`can_provide_models()`)
//! instead of the deprecated `LanguageModelProvider::is_authenticated(cx)`.
//! Driving the actual UI (agent_panel, agent_configuration, ai_onboarding,
//! git_panel) in a unit test would require a Workspace + Window + theme +
//! settings store, which is many-hundred-line fixtures for a one-line check.
//! The source anchor IS the test: if a refactor moves the call to
//! `is_authenticated(cx)`, the test fails with a clear diff.
//!
//! T18 walks every Rust source file in `crates/` and fails on any unannotated
//! call-style that matches the deprecated trait shim.
//!
//! Plan ref: v3.1 §5 AC5 (annotation lock), §6 T14–T18.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    // language_models is at crates/language_models, so workspace = ../../
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e))
}

/// Helper: file `path` MUST contain the literal substring `needle`. Used to
/// pin a UI wiring site to a specific contract.
#[track_caller]
fn assert_contains(label: &str, path: &Path, needle: &str) {
    let body = read(path);
    assert!(
        body.contains(needle),
        "{}: {} no longer contains `{}`. Update the wiring or update this test.",
        label,
        path.display(),
        needle
    );
}

/// Helper: file `path` MUST NOT contain any of the FORBIDDEN substrings. Used
/// to keep deprecated `is_authenticated` patterns out of UI sites.
#[track_caller]
fn assert_not_contains(label: &str, path: &Path, forbidden: &[&str]) {
    let body = read(path);
    for f in forbidden {
        assert!(
            !body.contains(f),
            "{}: {} contains forbidden substring `{}` (use auth_state(cx) and match on variant)",
            label,
            path.display(),
            f
        );
    }
}

/// T14: agent_panel onboarding treats RateLimited as configured (no upsell).
#[test]
fn t14_agent_panel_onboarding_treats_rate_limited_as_configured() {
    let root = workspace_root();
    let path = root.join("crates/agent_ui/src/agent_panel.rs");
    // The wiring site (plan v3.1 §6 T14, was line 5341 pre-fix-loop, now ~5344):
    // `provider.auth_state(cx).is_configured()`.
    assert_contains("T14", &path, "provider.auth_state(cx).is_configured()");
    assert_not_contains(
        "T14",
        &path,
        &[
            "provider.is_authenticated(cx)",
            "provider.is_authenticated(&cx)",
        ],
    );
}

/// T15: agent_configuration shows the check icon ONLY for Authenticated.
#[test]
fn t15_agent_configuration_check_icon_only_for_authenticated() {
    let root = workspace_root();
    let path = root.join("crates/agent_ui/src/agent_configuration.rs");
    // Two anchor sites — both call `provider.auth_state(cx)`.
    assert_contains("T15", &path, "provider.auth_state(cx)");
    assert_contains("T15", &path, "provider.auth_state(cx).is_configured()");
    assert_not_contains("T15", &path, &["provider.is_authenticated(cx)"]);
}

/// T16: ai_onboarding hides when a configured (incl. RateLimited) provider exists.
#[test]
fn t16_ai_onboarding_hides_when_rate_limited_provider_present() {
    let root = workspace_root();
    for rel in [
        "crates/ai_onboarding/src/agent_panel_onboarding_content.rs",
        "crates/ai_onboarding/src/agent_api_keys_onboarding.rs",
    ] {
        let path = root.join(rel);
        assert_contains("T16", &path, "provider.auth_state(cx).is_configured()");
        assert_not_contains("T16", &path, &["provider.is_authenticated(cx)"]);
    }
}

/// T17: git_panel does not call `authenticate()` for RateLimited providers.
/// The wiring uses `match provider.auth_state(cx) { ... }` and only the
/// `NotAuthenticated` arm calls authenticate.
#[test]
fn t17_git_panel_does_not_call_authenticate_for_rate_limited() {
    let root = workspace_root();
    let path = root.join("crates/git_ui/src/git_panel.rs");
    assert_contains("T17", &path, "match provider.auth_state(cx)");
    assert_not_contains("T17", &path, &["provider.is_authenticated(cx)"]);
    // Belt-and-braces: file must use the dispatcher's enum to pick a
    // remediation path; RateLimited is explicitly handled (or maps to None).
    let body = read(&path);
    assert!(
        body.contains("ProviderAuthState::") || body.contains("AuthAction::"),
        "T17: git_panel.rs no longer references ProviderAuthState/AuthAction \
         in its match arms — wiring may have regressed."
    );
}

/// T18 (plan v3.1 §5 AC5): CI grep gate.
///
/// Every call to the deprecated trait shim
/// `LanguageModelProvider::is_authenticated(cx)` MUST carry a
/// `// auth-shim: <use>` annotation on the same line (or the immediately
/// preceding line). Calls without the annotation fail the test.
///
/// We detect calls by pattern `(<expr>).is_authenticated(cx)` — but exclude:
///  - `self.state.read(cx).is_authenticated()` and `_(cx).is_authenticated()`
///    (these are inner provider-State methods, not the trait shim).
///  - The trait DEFINITION itself (in language_model/src/language_model.rs).
///  - The auth_state.rs deprecation doc.
///  - This test file.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let p = ent.path();
        // Skip target/ and hidden dirs.
        if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if name == "target" || name.starts_with('.') {
                continue;
            }
        }
        if p.is_dir() {
            collect_rs_files(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

#[test]
fn t18_no_unannotated_is_authenticated_calls() {
    let root = workspace_root();
    let crates = root.join("crates");
    let mut files = Vec::new();
    collect_rs_files(&crates, &mut files);

    // Files we must NOT scan (definition / docs / this test).
    let exempt: &[&str] = &[
        "crates/language_model/src/language_model.rs",
        "crates/language_model/src/auth_state.rs",
        "crates/language_models/tests/wiring_anchors_t14_t18.rs",
    ];

    let mut failures = Vec::new();
    'file: for path in &files {
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .to_string();
        for ex in exempt {
            if rel == *ex {
                continue 'file;
            }
        }
        let body = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // Look ONLY for the trait-shim call shape `provider.is_authenticated(cx)` /
            // `p.is_authenticated(cx)` etc. — anything that takes the App context. Inner
            // State methods take no args (`is_authenticated()`) or take `cx` only when
            // they're shadowing inside provider modules; we accept both as exempt
            // unless on a `provider`/`p`/`prov`/`lm_provider` receiver.
            if !line.contains(".is_authenticated(") {
                continue;
            }
            // Filter out inner state methods (no `cx` arg, or receiver chain ends in
            // `.read(cx)` which is the inner-state idiom).
            let stripped = line.trim_start();
            // Skip comments / doc lines.
            if stripped.starts_with("//") || stripped.starts_with("///") {
                continue;
            }
            // Inner-state idiom: `.read(cx).is_authenticated()` or `.is_authenticated()`
            // with no cx — these are NOT the trait shim.
            let has_cx_arg =
                line.contains(".is_authenticated(cx)") || line.contains(".is_authenticated(&cx)");
            if !has_cx_arg {
                continue;
            }
            // Inner-state idiom even with cx: `.read(cx).is_authenticated(cx)`
            // (only Copilot's State does this — confirmed via grep). Exempt the
            // chained `.read(cx).is_authenticated(cx)` pattern.
            if line.contains(".read(cx).is_authenticated(cx)") {
                continue;
            }
            // Exempt trait-shim default-impl call site.
            if line.contains("self.auth_state(cx).can_provide_models()") {
                continue;
            }
            // Annotated? `// auth-shim: <use>` on the same line OR immediately above.
            let same_line_ok = line.contains("// auth-shim:");
            let prev_line_ok = i > 0 && lines[i - 1].trim_start().starts_with("// auth-shim:");
            if same_line_ok || prev_line_ok {
                continue;
            }
            failures.push(format!(
                "{}:{}: unannotated `is_authenticated(cx)` call.\n  \
                 line: {}\n  \
                 fix: replace with `auth_state(cx)` and match on variant, OR \
                 add `// auth-shim: <reason>` annotation per plan v3.1 §5 AC5.",
                rel,
                i + 1,
                line.trim()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "T18 CI grep gate failed (auth-shim annotation lock per plan v3.1 §5 AC5):\n{}",
        failures.join("\n\n")
    );
}
