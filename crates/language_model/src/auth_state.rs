//! `ProviderAuthState` and supporting types — ST1a (`caduceus-fix/st1-provider-auth-state`).
//!
//! Replaces the single boolean `LanguageModelProvider::is_authenticated()` with a 4-variant
//! enum that distinguishes (a) never signed in, (b) rate-limited, (c) policy-disabled, and
//! (d) authenticated. ST1a only delivers the data model + non-UI call-site migration; the
//! selector redesign + click→action dispatcher land in ST1b.
//!
//! Security caveats (per plan v3.1 §B4):
//!
//! * **S1**: `RateLimited::retry_after` is clamped to `MAX_RETRY_AFTER` (24h) at construction
//!   to prevent a malicious / buggy upstream from advertising a 1-year retry-after.
//! * **S2**: `AuthAction::OpenUrl` only accepts a [`SafeUrl`], which rejects any scheme that
//!   isn't `https://`. No `file://`, `javascript:`, or `data:` urls reach the dispatcher.
//! * **S3**: `sanitize_provider_reason` is provided for callers that embed an upstream error
//!   message into `DisabledByPolicy::reason`. It strips control characters, redacts common
//!   token patterns (Bearer, sk-, gho_, ya29.), and truncates to 256 chars.

use std::time::Duration;

/// Cap on `RateLimited::retry_after` (24 hours). A server-supplied retry-after greater than
/// this is silently clamped + logged; see `ProviderAuthState::rate_limited`.
pub const MAX_RETRY_AFTER: Duration = Duration::from_secs(24 * 3600);

/// Bounded fallback TTL applied when a 429 response carries no `Retry-After` header.
/// Without this, a headerless 429 would wedge the auth cache forever (no auto-expiry).
/// Picked to be short enough that genuine recoveries land quickly, but long enough
/// to absorb a typical token-bucket cooldown.
/// Plan v3.1 review item #8.
pub const HEADERLESS_RATE_LIMIT_TTL: Duration = Duration::from_secs(30);

/// Maximum length (in bytes) of a sanitized `DisabledByPolicy::reason` string.
pub const MAX_REASON_LEN: usize = 256;

/// The auth state of a [`LanguageModelProvider`](crate::LanguageModelProvider).
///
/// ST1a's central type. Replaces the single boolean `is_authenticated()` returned by every
/// provider today. Each non-`Authenticated` variant carries enough information for the UI
/// (in ST1b) to render an actionable affordance instead of silently hiding the provider.
#[derive(Clone, Debug)]
pub enum ProviderAuthState {
    /// Provider is signed in and can serve completion requests.
    Authenticated,
    /// Provider has no usable credentials. `action` is the remediation the dispatcher
    /// (ST1b) should take when the user clicks the row.
    NotAuthenticated { action: AuthAction },
    /// Provider returned HTTP 429 (or equivalent). `retry_after` is the server-supplied
    /// duration, clamped to [`MAX_RETRY_AFTER`].
    RateLimited {
        retry_after: Option<Duration>,
        action: AuthAction,
    },
    /// Provider is disabled by org/admin policy. Wrap the reason in [`SanitizedReason`]
    /// so it cannot be constructed without going through the S3 sanitizer.
    DisabledByPolicy(SanitizedReason),
}

/// Newtype around a sanitized reason string. The inner `String` is private — the only
/// way to construct one is [`SanitizedReason::new`] (or, transitively,
/// [`ProviderAuthState::disabled_by_policy`]), both of which run
/// [`sanitize_provider_reason`]. Pattern matches outside the crate must use
/// [`SanitizedReason::as_str`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SanitizedReason(String);

impl SanitizedReason {
    /// Construct from any string-ish input, applying [`sanitize_provider_reason`].
    pub fn new(reason: impl Into<String>) -> Self {
        Self(sanitize_provider_reason(&reason.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for SanitizedReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl ProviderAuthState {
    /// Construct a `RateLimited` state, applying the S1 cap (24h) to the `retry_after`.
    /// If the server-supplied value exceeds the cap, it's clamped and logged at WARN.
    pub fn rate_limited(retry_after: Option<Duration>, action: AuthAction) -> Self {
        let retry_after = retry_after.map(|d| {
            if d > MAX_RETRY_AFTER {
                log::warn!(
                    "provider sent retry_after={:?}, clamping to MAX_RETRY_AFTER={:?}",
                    d,
                    MAX_RETRY_AFTER
                );
                MAX_RETRY_AFTER
            } else {
                d
            }
        });
        Self::RateLimited {
            retry_after,
            action,
        }
    }

    /// Construct a `DisabledByPolicy` state, sanitizing `reason` per S3.
    pub fn disabled_by_policy(reason: impl Into<String>) -> Self {
        Self::DisabledByPolicy(SanitizedReason::new(reason))
    }

    /// True iff the provider can serve a completion request right now. This is the
    /// "usability" predicate the deprecated `is_authenticated()` shim defers to.
    pub fn can_provide_models(&self) -> bool {
        matches!(self, Self::Authenticated)
    }

    /// True iff the provider is configured (has credentials). Rate-limited providers ARE
    /// configured — they should not be shown an onboarding upsell. Disabled-by-policy is
    /// NOT configured (admin would need to flip a flag).
    pub fn is_configured(&self) -> bool {
        matches!(self, Self::Authenticated | Self::RateLimited { .. })
    }

    /// Stable string discriminant for telemetry. Exactly four values, no PII.
    pub fn telemetry_discriminant(&self) -> &'static str {
        match self {
            Self::Authenticated => "authenticated",
            Self::NotAuthenticated { .. } => "not_authenticated",
            Self::RateLimited { .. } => "rate_limited",
            Self::DisabledByPolicy(_) => "disabled_by_policy",
        }
    }
}

/// What the UI should do when the user clicks/activates a non-`Authenticated` row.
///
/// ST1a only stores the discriminant; the dispatcher mapping `AuthAction` → real work
/// (open browser, focus settings page, invoke `copilot_ui::initiate_sign_in`) lives in
/// ST1b. ST1a guarantees the variants are populated correctly by every provider.
#[derive(Clone, Debug)]
pub enum AuthAction {
    /// Provider has an in-process imperative sign-in flow that the dispatcher must
    /// look up by provider id (e.g. Copilot's device-flow via
    /// `copilot_ui::initiate_sign_in(Entity<Copilot>, &mut Window, &mut App)`).
    /// We can't store a closure here because the trait method takes `&App`, not
    /// `&mut App` + `&mut Window`. The dispatcher resolves the call by id.
    SignInImperative,
    /// Provider needs the user to enter an API key in Settings (anthropic / openai /
    /// google / mistral / etc.).
    EnterApiKeyInSettings,
    /// Open an https url (docs, status page). The constructor enforces `scheme == https`.
    OpenUrl(SafeUrl),
    /// Rate-limited or policy-disabled — no user action is meaningful right now.
    None,
}

/// New-type wrapper around a URL. `SafeUrl::https` is the only constructor and it
/// rejects any URL whose scheme is not `https` or whose authority/host is empty
/// (closes S2). String-prefix checks are insufficient because they accept things
/// like `https:///etc/passwd` (no host) or `https://?q=1` (empty host); we use
/// `url::Url::parse` so the full RFC-3986 grammar runs and we then enforce
/// `scheme() == "https"` AND `host_str().is_some_and(|h| !h.is_empty())`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeUrl(String);

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum UnsafeUrlError {
    #[error("url must use https:// scheme; got {0}")]
    NonHttpsScheme(String),
    #[error("url has no host")]
    NoHost,
    #[error("url is empty or unparseable")]
    Empty,
}

impl SafeUrl {
    /// Returns `Ok(SafeUrl)` iff `url` parses as a real `https://host[/...]` URL with
    /// a non-empty host. Closes S2: rejects `http://`, `file://`, `javascript:`,
    /// `data:`, `https:///etc/passwd` (no host), `https://?q=1` (empty host),
    /// `https://#frag` (empty host). The scheme check is naturally case-insensitive
    /// because `url::Url` lower-cases the scheme during parse.
    pub fn https(url: impl Into<String>) -> std::result::Result<Self, UnsafeUrlError> {
        let raw = url.into();
        if raw.is_empty() || raw.trim().len() != raw.len() {
            // Reject leading/trailing whitespace bypasses ("  https://...") rather than
            // silently trimming them.
            return Err(UnsafeUrlError::Empty);
        }
        // url::Url::parse is liberal about extra slashes — `https:///etc/passwd` is
        // accepted as `host=etc, path=/passwd`. Pre-screen the literal input so the
        // authority section starts with a non-slash character (the scheme check is
        // case-insensitive — url::Url lowercases on parse).
        let lower_prefix: String = raw.chars().take(8).collect::<String>().to_ascii_lowercase();
        if !lower_prefix.starts_with("https://") {
            return Err(UnsafeUrlError::NonHttpsScheme(
                raw.split(':').next().unwrap_or("").to_string(),
            ));
        }
        let after_scheme = &raw[8..];
        if after_scheme.starts_with('/') {
            // e.g. `https:///etc/passwd` — would be parsed as host=etc, path=/passwd.
            return Err(UnsafeUrlError::NoHost);
        }
        let parsed = url::Url::parse(&raw).map_err(|_| UnsafeUrlError::Empty)?;
        if parsed.scheme() != "https" {
            return Err(UnsafeUrlError::NonHttpsScheme(parsed.scheme().to_string()));
        }
        match parsed.host_str() {
            Some(host) if !host.is_empty() => Ok(Self(parsed.into())),
            _ => Err(UnsafeUrlError::NoHost),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Sanitize a free-form provider error message before storing it in
/// `DisabledByPolicy::reason` (S3). Three steps:
///
/// 1. Replace ASCII/Unicode control characters with a single space. We
///    REPLACE rather than STRIP so that control-char-separated token
///    patterns (e.g. `Bearer\nabc.def`, `word\rsk-SECRET`, `leak\x0bya29.X`)
///    still present a `\s` separator to the redact regex in step 2. (Round-3
///    fix: the previous strip-then-regex order let `Bearer\nabc` collapse to
///    `Bearerabc`, which the `\bbearer\s+\S+` pattern could not match.)
/// 2. Redact tokens matching, case-insensitively:
///      * `(?i)\bbearer\s+\S+`
///      * `\bsk-\S+`
///      * `\bgho_\S+`
///      * `\bya29\.\S+`
///    with `<redacted>`. The `\s` class catches double-spaces, tabs, etc.
/// 3. Truncate to [`MAX_REASON_LEN`] bytes (UTF-8 safe).
pub fn sanitize_provider_reason(input: &str) -> String {
    // Step 1: replace any control char (including \t, \n, \r, \x0b, ...) with
    // a single ASCII space. This ensures step-2 regex sees a `\s` boundary
    // between e.g. `Bearer` and the leaked token even when the original
    // separator was a non-space control char.
    let mut out: String = input
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();

    // Step 2: redact token-like substrings with the regex crate (avoid the literal
    // " " bug that missed `Bearer\tabc` and `Bearer  abc` in the previous scanner).
    out = TOKEN_REDACT_RE.replace_all(&out, "<redacted>").into_owned();

    // Step 3: byte-truncate to MAX_REASON_LEN, snapping to a UTF-8 char boundary.
    if out.len() > MAX_REASON_LEN {
        let mut end = MAX_REASON_LEN;
        while end > 0 && !out.is_char_boundary(end) {
            end -= 1;
        }
        out.truncate(end);
        out.push_str("…");
    }
    out
}

static TOKEN_REDACT_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    // Order matters: bearer first because it consumes a "Bearer<sep>" prefix that
    // could otherwise be partially matched as a sk-/gho_/ya29 token.
    regex::Regex::new(
        r"(?i)\bbearer\s+\S+|\bsk-\S+|\bgho_\S+|\bya29\.\S+",
    )
    .expect("static token-redact regex compiles")
});

#[cfg(test)]
mod tests {
    use super::*;

    // T1: AC2 — can_provide_models truth table.
    #[test]
    fn auth_state_can_provide_models_truth_table() {
        assert!(ProviderAuthState::Authenticated.can_provide_models());
        assert!(!ProviderAuthState::NotAuthenticated {
            action: AuthAction::None
        }
        .can_provide_models());
        assert!(
            !ProviderAuthState::rate_limited(Some(Duration::from_secs(10)), AuthAction::None)
                .can_provide_models()
        );
        assert!(
            !ProviderAuthState::disabled_by_policy("policy").can_provide_models()
        );
    }

    // T1b: is_configured — Authenticated and RateLimited are configured; others not.
    #[test]
    fn auth_state_is_configured_truth_table() {
        assert!(ProviderAuthState::Authenticated.is_configured());
        assert!(
            ProviderAuthState::rate_limited(None, AuthAction::None).is_configured()
        );
        assert!(!ProviderAuthState::NotAuthenticated {
            action: AuthAction::EnterApiKeyInSettings
        }
        .is_configured());
        assert!(!ProviderAuthState::disabled_by_policy("p").is_configured());
    }

    // T3: AC6 — telemetry discriminant strings are exactly the 4 documented values.
    #[test]
    fn auth_state_telemetry_discriminant_strings() {
        assert_eq!(
            ProviderAuthState::Authenticated.telemetry_discriminant(),
            "authenticated"
        );
        assert_eq!(
            ProviderAuthState::NotAuthenticated {
                action: AuthAction::None
            }
            .telemetry_discriminant(),
            "not_authenticated"
        );
        assert_eq!(
            ProviderAuthState::rate_limited(None, AuthAction::None).telemetry_discriminant(),
            "rate_limited"
        );
        assert_eq!(
            ProviderAuthState::disabled_by_policy("p").telemetry_discriminant(),
            "disabled_by_policy"
        );
    }

    // T4: AC-SEC1 — `retry_after` is clamped to `MAX_RETRY_AFTER`.
    #[test]
    fn rate_limited_clamps_to_max_retry_after() {
        let huge = Duration::from_secs(365 * 24 * 3600);
        let state = ProviderAuthState::rate_limited(Some(huge), AuthAction::None);
        match state {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(MAX_RETRY_AFTER));
            }
            other => panic!("expected RateLimited, got {:?}", other),
        }

        // Below the cap, the value passes through unmodified.
        let small = Duration::from_secs(42);
        let state = ProviderAuthState::rate_limited(Some(small), AuthAction::None);
        match state {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(small));
            }
            _ => unreachable!(),
        }

        // None passes through.
        let state = ProviderAuthState::rate_limited(None, AuthAction::None);
        match state {
            ProviderAuthState::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, None);
            }
            _ => unreachable!(),
        }
    }

    // T5: AC-SEC2 — `SafeUrl::https` rejects every non-https scheme AND every URL
    // that string-prefix checks would have accepted but is structurally bogus.
    // (fix-loop #3: pre-fix accepted `https:///etc/passwd`, `https://?q=1`,
    // `https://#frag`, `HTTPS://example.com` (case), and bypasses via stray
    // whitespace.)
    #[test]
    fn safe_url_rejects_non_https() {
        for bad in [
            "http://example.com",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>",
            "ftp://example.com",
            "",
            "https://", // no authority
            "  https://example.com",
            // Pre-fix-loop bugs the prefix check accepted:
            "https:///etc/passwd", // no host
            "https://?q=1",        // empty host
            "https://#frag",       // empty host
            "https:// space.com",  // space in authority
        ] {
            assert!(
                SafeUrl::https(bad).is_err(),
                "expected rejection for {bad:?}"
            );
        }
        assert!(SafeUrl::https("https://example.com").is_ok());
        assert!(SafeUrl::https("https://example.com/foo?bar=1").is_ok());
        // Case-insensitive scheme via url::Url::parse — both pass and the canonical
        // form has a lowercase scheme.
        let normalized = SafeUrl::https("HTTPS://example.com").expect("uppercase scheme ok");
        assert!(
            normalized.as_str().starts_with("https://"),
            "scheme must be normalized to lowercase, got {}",
            normalized.as_str()
        );
    }

    // T6: AC-SEC3 — sanitizer redacts tokens, strips control chars, caps length.
    // (fix-loop #4: pre-fix used a hand-rolled scanner that only matched the literal
    // `Bearer ` prefix, missing double-space, tab, and uppercase variants.)
    #[test]
    fn disabled_reason_sanitizer_redacts_tokens_and_caps_length() {
        // Bearer redaction (case-insensitive prefix).
        let s = sanitize_provider_reason("auth header was Bearer abc.def.ghi end");
        assert!(s.contains("<redacted>"), "got {s}");
        assert!(!s.contains("abc.def.ghi"));

        // Pre-fix bug: double-space between Bearer and token slipped through.
        let s = sanitize_provider_reason("Bearer  abc.def.ghi tail");
        assert!(s.contains("<redacted>"), "double-space leak: got {s}");
        assert!(!s.contains("abc.def.ghi"));

        // Pre-fix bug: tab + uppercase BEARER slipped through.
        let s = sanitize_provider_reason("BEARER\tabc.def.ghi tail");
        assert!(s.contains("<redacted>"), "BEARER\\tabc leak: got {s}");
        assert!(!s.contains("abc.def.ghi"));

        let s = sanitize_provider_reason("got key sk-1234567890ABC then fail");
        assert!(s.contains("<redacted>"), "got {s}");
        assert!(!s.contains("sk-1234567890ABC"));

        let s = sanitize_provider_reason("token gho_ABCDEF1234 was rejected");
        assert!(s.contains("<redacted>"));
        assert!(!s.contains("gho_ABCDEF1234"));

        let s = sanitize_provider_reason("creds ya29.A0ARr-foo_bar-baz failed");
        assert!(s.contains("<redacted>"));
        assert!(!s.contains("ya29.A0ARr-foo_bar-baz"));

        // Control chars are replaced with spaces; spaces preserved.
        // (Round-3 fix: we now replace rather than strip so that control-char
        // separators between `Bearer`/`sk-`/etc. and the leaked token still
        // present a `\s` boundary to the redact regex.)
        let s = sanitize_provider_reason("a\x00b\x07c\td e");
        assert_eq!(s, "a b c d e");

        // Round-3 regression (R2 must-fix #1+#2): control-char separators
        // (\n, \r, \x0b) between a token prefix and the secret previously
        // bypassed redaction because step-1 stripped the separator before
        // the regex saw it. Lock that in.
        for (input, leaked) in [
            ("Bearer\nabc.def.ghi tail", "abc.def.ghi"),
            ("prev\nBearer abc.def.ghi", "abc.def.ghi"),
            ("word\rsk-SECRETKEY end", "SECRETKEY"),
            ("leak\x0bya29.SECRET end", "ya29.SECRET"),
            ("line1\ngho_REALTOKEN line2", "gho_REALTOKEN"),
        ] {
            let s = sanitize_provider_reason(input);
            assert!(
                !s.contains(leaked),
                "control-separator leak: input={input:?} → output={s:?} still contains {leaked:?}",
            );
            assert!(s.contains("<redacted>"), "no redaction marker: {s:?}");
        }

        // Length cap with char-boundary safety.
        let long: String = "x".repeat(MAX_REASON_LEN + 100);
        let s = sanitize_provider_reason(&long);
        // After truncation we append "…"; total bytes <= MAX_REASON_LEN + len("…").
        assert!(s.ends_with("…"));
        assert!(s.len() <= MAX_REASON_LEN + "…".len());

        // Multibyte chars at the boundary do not panic.
        let s: String = std::iter::repeat("é").take(MAX_REASON_LEN).collect();
        let _ = sanitize_provider_reason(&s);
    }

    // Ensure that DisabledByPolicy::reason actually goes through the sanitizer.
    #[test]
    fn disabled_by_policy_constructor_sanitizes_reason() {
        let s = ProviderAuthState::disabled_by_policy("Bearer abcdefghijk denied");
        match s {
            ProviderAuthState::DisabledByPolicy(reason) => {
                assert!(reason.as_str().contains("<redacted>"), "got {}", reason);
                assert!(!reason.as_str().contains("abcdefghijk"));
            }
            _ => unreachable!(),
        }
    }

    // fix-loop #6: SanitizedReason has no public constructor that bypasses the
    // sanitizer. The only way in is `SanitizedReason::new` (or transitively
    // `ProviderAuthState::disabled_by_policy`), both of which sanitize.
    #[test]
    fn sanitized_reason_only_constructible_via_sanitizer() {
        let r = SanitizedReason::new("Bearer abcdefghijk leaked");
        assert!(r.as_str().contains("<redacted>"));
        assert!(!r.as_str().contains("abcdefghijk"));
        // Round-trip via Display + into_string preserves sanitized content.
        assert_eq!(r.to_string(), r.clone().into_string());
    }
}
