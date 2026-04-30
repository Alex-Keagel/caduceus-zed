//! Multi-repo UX (ux03) — repo picker + slug display helpers.
//!
//! Per the implementation DAG and the multi-repo UX design from
//! `symphony-multirepo-ux.md` Part D.3, this module provides:
//!
//! - `RepoCard` — display-time summary of a repo (slug, host, owner,
//!   path, default branch).
//! - `parse_remote_url` — display-friendly parser that produces the
//!   pieces a UI surface needs (host + owner + path) WITHOUT
//!   committing to a slug (slug derivation is the daemon's job per
//!   spec #3 §3.1).
//! - `format_repo_card_short` — single-line display string for the
//!   repo picker dropdown.
//!
//! This module is intentionally **display-only**; it does NOT
//! sanitize or validate slugs against the daemon's regex.  Slugs
//! flow into the UI from the daemon's snapshot RPC pre-validated.

use serde::{Deserialize, Serialize};

/// Display-time summary of a repo.  The `slug` field MUST be the
/// daemon-canonical sticky slug per spec #3 I-4 (sticky once recorded).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoCard {
    pub slug: String,
    /// Lowercased host (e.g., `github.com`).
    pub host: String,
    /// Owner segment (e.g., `openai`).  Empty if the URL has no owner.
    pub owner: String,
    /// Repository name segment (e.g., `symphony`).
    pub repo: String,
    /// Default branch advisory (`Some("main")`, etc.).
    pub default_branch: Option<String>,
}

impl RepoCard {
    /// Construct from the canonical fields.  Slug MUST be daemon-derived;
    /// the constructor does not validate.
    pub fn new(
        slug: impl Into<String>,
        host: impl Into<String>,
        owner: impl Into<String>,
        repo: impl Into<String>,
        default_branch: Option<String>,
    ) -> Self {
        Self {
            slug: slug.into(),
            host: host.into(),
            owner: owner.into(),
            repo: repo.into(),
            default_branch,
        }
    }

    /// Single-line display string for a picker dropdown row.  Format:
    /// `<owner>/<repo>  ·  <host>  ·  @<branch>`
    pub fn picker_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.owner.is_empty() {
            parts.push(self.repo.clone());
        } else {
            parts.push(format!("{}/{}", self.owner, self.repo));
        }
        parts.push(self.host.clone());
        if let Some(b) = &self.default_branch {
            parts.push(format!("@{b}"));
        }
        parts.join("  ·  ")
    }

    /// Two-line tooltip: line 1 = picker_line(), line 2 = slug.
    pub fn tooltip(&self) -> String {
        format!("{}\nslug: {}", self.picker_line(), self.slug)
    }
}

/// Parsed pieces of a remote URL.  Display-only; slug derivation
/// belongs to the daemon (spec #3 §3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedRemote {
    pub host: String,
    pub path_segments: Vec<String>,
}

impl ParsedRemote {
    pub fn owner(&self) -> &str {
        self.path_segments
            .iter()
            .rev()
            .nth(1)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub fn repo(&self) -> &str {
        self.path_segments.last().map(String::as_str).unwrap_or("")
    }
}

/// Parse a remote URL into display-friendly pieces.  Accepts the same
/// forms as `caduceus_daemon::sanitize_repo_slug` (https / ssh / scp).
/// Returns None on a malformed input.
pub fn parse_remote_url(url: &str) -> Option<ParsedRemote> {
    let url = url.trim();
    if url.is_empty() {
        return None;
    }
    let (host, path) = if let Some(rest) = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .or_else(|| url.strip_prefix("ssh://"))
    {
        let rest = if let Some(idx) = rest.find('@') {
            let slash = rest.find('/').unwrap_or(rest.len());
            if idx < slash { &rest[idx + 1..] } else { rest }
        } else {
            rest
        };
        let (auth, p) = rest.split_once('/').unwrap_or((rest, ""));
        let host = auth.split_once(':').map(|(h, _)| h).unwrap_or(auth);
        (host.to_lowercase(), p.to_string())
    } else if let Some((before, after)) = url.split_once(':') {
        if let Some((_user, host)) = before.split_once('@') {
            (host.to_lowercase(), after.to_string())
        } else {
            return None;
        }
    } else {
        return None;
    };
    if host.is_empty() {
        return None;
    }
    let path = path.strip_suffix(".git").unwrap_or(&path);
    let path = path.trim_start_matches('/');
    let segments: Vec<String> = path
        .split('/')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Some(ParsedRemote {
        host,
        path_segments: segments,
    })
}

/// Build a `RepoCard` from a remote URL + daemon-derived slug.
pub fn repo_card_from_remote(
    url: &str,
    slug: impl Into<String>,
    default_branch: Option<String>,
) -> Option<RepoCard> {
    let parsed = parse_remote_url(url)?;
    let owner = parsed.owner().to_string();
    let repo = parsed.repo().to_string();
    Some(RepoCard::new(
        slug,
        parsed.host,
        owner,
        repo,
        default_branch,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_https_basic() {
        let p = parse_remote_url("https://github.com/openai/symphony").unwrap();
        assert_eq!(p.host, "github.com");
        assert_eq!(p.owner(), "openai");
        assert_eq!(p.repo(), "symphony");
    }

    #[test]
    fn parse_https_with_dot_git() {
        let p = parse_remote_url("https://github.com/openai/symphony.git").unwrap();
        assert_eq!(p.repo(), "symphony");
    }

    #[test]
    fn parse_ssh_with_user() {
        let p = parse_remote_url("ssh://git@github.com/openai/symphony.git").unwrap();
        assert_eq!(p.host, "github.com");
        assert_eq!(p.owner(), "openai");
        assert_eq!(p.repo(), "symphony");
    }

    #[test]
    fn parse_scp_form() {
        let p = parse_remote_url("git@github.com:openai/symphony.git").unwrap();
        assert_eq!(p.host, "github.com");
        assert_eq!(p.owner(), "openai");
        assert_eq!(p.repo(), "symphony");
    }

    #[test]
    fn parse_lowercases_host() {
        let p = parse_remote_url("https://GitHub.com/Openai/Symphony").unwrap();
        assert_eq!(p.host, "github.com");
        // Owner + repo are NOT lowercased here; UI may display case.
        assert_eq!(p.owner(), "Openai");
        assert_eq!(p.repo(), "Symphony");
    }

    #[test]
    fn parse_deeply_nested_path() {
        let p = parse_remote_url("https://example.com/a/very/deeply/nested/repo").unwrap();
        assert_eq!(p.host, "example.com");
        assert_eq!(p.owner(), "nested");
        assert_eq!(p.repo(), "repo");
        assert_eq!(p.path_segments.len(), 5);
    }

    #[test]
    fn parse_garbage_returns_none() {
        assert!(parse_remote_url("").is_none());
        assert!(parse_remote_url("not a url").is_none());
        assert!(parse_remote_url("https://").is_none());
    }

    #[test]
    fn picker_line_format() {
        let card = RepoCard::new(
            "github_com_openai_symphony",
            "github.com",
            "openai",
            "symphony",
            Some("main".into()),
        );
        assert_eq!(
            card.picker_line(),
            "openai/symphony  ·  github.com  ·  @main"
        );
    }

    #[test]
    fn picker_line_omits_branch_when_none() {
        let card = RepoCard::new("g_x", "github.com", "o", "r", None);
        assert_eq!(card.picker_line(), "o/r  ·  github.com");
    }

    #[test]
    fn tooltip_includes_slug() {
        let card = RepoCard::new("g_x_y", "github.com", "o", "r", Some("main".into()));
        let t = card.tooltip();
        assert!(t.contains("o/r"));
        assert!(t.contains("g_x_y"));
    }

    #[test]
    fn repo_card_from_remote_round_trip() {
        let card = repo_card_from_remote(
            "https://github.com/openai/symphony.git",
            "github_com_openai_symphony",
            Some("main".into()),
        )
        .unwrap();
        assert_eq!(card.host, "github.com");
        assert_eq!(card.owner, "openai");
        assert_eq!(card.repo, "symphony");
        assert_eq!(card.slug, "github_com_openai_symphony");
        assert_eq!(card.default_branch.as_deref(), Some("main"));
    }

    #[test]
    fn repo_card_serialize_round_trip() {
        let card = RepoCard::new("g", "github.com", "o", "r", Some("main".into()));
        let s = serde_json::to_string(&card).unwrap();
        let back: RepoCard = serde_json::from_str(&s).unwrap();
        assert_eq!(card, back);
    }

    #[test]
    fn picker_line_handles_missing_owner() {
        let card = RepoCard::new("g", "github.com", "", "r", None);
        assert_eq!(card.picker_line(), "r  ·  github.com");
    }
}
