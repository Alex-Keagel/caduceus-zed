//! Run identity + addressing (ux01) — `RunRef` type + caduceus:// deeplink scheme.
//!
//! Per the implementation DAG and `spec-m-ui-run-identity-and-addressing.md`,
//! this module ships the addressing surface for caduceus runs as
//! consumed by the Zed UI:
//!
//! - **`RunRef`** — newtype wrapping a `(slug, run_id)` pair with a
//!   stable URL form for cross-process / cross-window addressing.
//! - **caduceus:// deeplink scheme** — `caduceus://run/<slug>/<run_id>`
//!   parses + serializes round-trip; used by zed's protocol handler
//!   to deep-link into the runs panel.
//!
//! This module is **self-contained** — no daemon or engine deps.  The
//! daemon-side snapshot integration is a separate todo (future work
//! adds caduceus-daemon as a workspace dep).
//!
//! Spec cross-references:
//!
//! - **`spec-m-ui-run-identity-and-addressing.md` §3 deeplink scheme**
//! - **`spec-multi-repo-workspace-model.md` §3.1 RepoSlug** — the slug
//!   rule used to construct RunRef matches the daemon's sanitize_repo_slug
//!   output regex `^[a-z0-9][a-z0-9_]{0,63}$`.
//! - **`spec-multi-repo-workspace-model.md` §3.2 sanitize_run_id** —
//!   the run_id rule matches the daemon's safe_run_id regex
//!   `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`.

use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// A reference to a Run, addressable across windows + processes via
/// the caduceus:// deeplink scheme.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunRef {
    pub slug: String,
    pub run_id: String,
}

impl RunRef {
    /// Construct a RunRef.  Validates both fields against the canonical
    /// daemon regexes.  Returns `Err` on a malformed input.
    pub fn new(slug: impl Into<String>, run_id: impl Into<String>) -> Result<Self, RunRefError> {
        let slug = slug.into();
        let run_id = run_id.into();
        validate_slug(&slug)?;
        validate_run_id(&run_id)?;
        Ok(Self { slug, run_id })
    }

    /// Construct without validation.  Internal use; tests + fixtures
    /// may bypass.
    pub fn from_parts_unchecked(slug: impl Into<String>, run_id: impl Into<String>) -> Self {
        Self {
            slug: slug.into(),
            run_id: run_id.into(),
        }
    }

    /// Format as a caduceus:// deeplink URL.  Round-trips with `from_str`.
    pub fn to_url(&self) -> String {
        format!("caduceus://run/{}/{}", self.slug, self.run_id)
    }
}

impl fmt::Display for RunRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_url())
    }
}

impl FromStr for RunRef {
    type Err = RunRefError;

    /// Parse a caduceus:// URL into a RunRef.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let rest = s
            .strip_prefix("caduceus://run/")
            .ok_or(RunRefError::InvalidScheme)?;
        let (slug, run_id) = rest.split_once('/').ok_or(RunRefError::MissingRunId)?;
        if slug.is_empty() {
            return Err(RunRefError::EmptySlug);
        }
        if run_id.is_empty() {
            return Err(RunRefError::EmptyRunId);
        }
        // Disallow trailing path segments (defence-in-depth).
        if run_id.contains('/') {
            return Err(RunRefError::TrailingPath);
        }
        Self::new(slug, run_id)
    }
}

/// Errors produced by `RunRef` parsing + construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RunRefError {
    #[error("URL must start with `caduceus://run/`")]
    InvalidScheme,
    #[error("missing run_id segment")]
    MissingRunId,
    #[error("slug is empty")]
    EmptySlug,
    #[error("run_id is empty")]
    EmptyRunId,
    #[error("URL has trailing path segments after run_id")]
    TrailingPath,
    #[error("invalid slug: must match `^[a-z0-9][a-z0-9_]{{0,63}}$`")]
    InvalidSlug,
    #[error("invalid run_id: must match `^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$`")]
    InvalidRunId,
}

fn validate_slug(s: &str) -> Result<(), RunRefError> {
    if s.is_empty() || s.len() > 64 {
        return Err(RunRefError::InvalidSlug);
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(RunRefError::InvalidSlug);
    }
    if chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
        Ok(())
    } else {
        Err(RunRefError::InvalidSlug)
    }
}

fn validate_run_id(s: &str) -> Result<(), RunRefError> {
    if s.is_empty() || s.len() > 128 {
        return Err(RunRefError::InvalidRunId);
    }
    if s == "." || s == ".." || s.contains("..") {
        return Err(RunRefError::InvalidRunId);
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphanumeric() {
        return Err(RunRefError::InvalidRunId);
    }
    if chars.all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-') {
        Ok(())
    } else {
        Err(RunRefError::InvalidRunId)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ref_construct_and_format() {
        let r = RunRef::new("github_com_o_r", "01H8XYZABC").unwrap();
        assert_eq!(r.slug, "github_com_o_r");
        assert_eq!(r.run_id, "01H8XYZABC");
        assert_eq!(r.to_url(), "caduceus://run/github_com_o_r/01H8XYZABC");
    }

    #[test]
    fn run_ref_from_url_round_trip() {
        let url = "caduceus://run/github_com_o_r/01H8XYZABC";
        let r: RunRef = url.parse().unwrap();
        assert_eq!(r.to_url(), url);
    }

    #[test]
    fn run_ref_display_matches_url() {
        let r = RunRef::new("g_x_y", "abc").unwrap();
        assert_eq!(format!("{r}"), r.to_url());
    }

    #[test]
    fn invalid_scheme_rejected() {
        let r: Result<RunRef, _> = "https://run/x/y".parse();
        assert_eq!(r, Err(RunRefError::InvalidScheme));
    }

    #[test]
    fn missing_run_id_rejected() {
        let r: Result<RunRef, _> = "caduceus://run/slug".parse();
        assert_eq!(r, Err(RunRefError::MissingRunId));
    }

    #[test]
    fn empty_slug_rejected() {
        let r: Result<RunRef, _> = "caduceus://run//runid".parse();
        assert_eq!(r, Err(RunRefError::EmptySlug));
    }

    #[test]
    fn empty_run_id_rejected() {
        let r: Result<RunRef, _> = "caduceus://run/slug/".parse();
        assert_eq!(r, Err(RunRefError::EmptyRunId));
    }

    #[test]
    fn trailing_path_rejected() {
        let r: Result<RunRef, _> = "caduceus://run/slug/runid/extra".parse();
        assert_eq!(r, Err(RunRefError::TrailingPath));
    }

    #[test]
    fn slug_uppercase_rejected() {
        let r = RunRef::new("Github_com_x_y", "abc");
        assert_eq!(r, Err(RunRefError::InvalidSlug));
    }

    #[test]
    fn slug_starting_with_underscore_rejected() {
        let r = RunRef::new("_github_com", "abc");
        assert_eq!(r, Err(RunRefError::InvalidSlug));
    }

    #[test]
    fn slug_too_long_rejected() {
        let s = "g".to_string() + &"_".repeat(64);
        assert!(RunRef::new(&s, "abc").is_err());
    }

    #[test]
    fn run_id_with_dotdot_rejected() {
        assert_eq!(RunRef::new("g_x", ".."), Err(RunRefError::InvalidRunId));
        assert_eq!(RunRef::new("g_x", "a..b"), Err(RunRefError::InvalidRunId));
    }

    #[test]
    fn run_id_starting_non_alnum_rejected() {
        assert_eq!(RunRef::new("g_x", "-abc"), Err(RunRefError::InvalidRunId));
        assert_eq!(RunRef::new("g_x", "_abc"), Err(RunRefError::InvalidRunId));
        assert_eq!(RunRef::new("g_x", ".abc"), Err(RunRefError::InvalidRunId));
    }

    #[test]
    fn run_id_with_slash_rejected() {
        assert_eq!(RunRef::new("g_x", "a/b"), Err(RunRefError::InvalidRunId));
    }

    #[test]
    fn slug_with_digits_at_start_accepted() {
        let r = RunRef::new("0_github_com", "x").unwrap();
        assert_eq!(r.slug, "0_github_com");
    }

    #[test]
    fn run_id_with_uppercase_accepted() {
        let r = RunRef::new("g_x", "01H8XYZ").unwrap();
        assert_eq!(r.run_id, "01H8XYZ");
    }
}
