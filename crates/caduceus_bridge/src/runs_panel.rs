//! Runs panel data model (ux02) — wire shapes the agent_ui runs panel
//! consumes, with helpers for grouping / sorting / pretty-printing.
//!
//! Per the implementation DAG and `spec-m-ui-runs-panel.md`, this
//! module ships the data model + projection helpers the panel needs
//! to render the snapshot served by the daemon.  The actual gpui UI
//! (panel layout + status indicators + keyboard navigation) lives in
//! `agent_ui::runs_panel` (added in a follow-up commit so this crate
//! stays gpui-free for testability).
//!
//! Data shapes mirror the daemon's `caduceus_daemon::snapshot_shapes`
//! but are re-derived locally so this crate has no daemon dependency.
//! When the engine ↔ daemon RPC client is added in a future revision,
//! this module will gain a conversion layer; until then, the shapes
//! are the canonical client-side projection.

use crate::run_ref::RunRef;
use serde::{Deserialize, Serialize};

/// Display-grouped buckets for the runs panel.  Spec
/// `spec-m-ui-runs-panel.md` §2 layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunsPanelView {
    pub running: Vec<RunsPanelRow>,
    pub retrying: Vec<RunsPanelRow>,
    pub disconnected: Vec<RunsPanelRow>,
    pub recent: Vec<RunsPanelRow>,
}

impl RunsPanelView {
    pub fn empty() -> Self {
        Self {
            running: Vec::new(),
            retrying: Vec::new(),
            disconnected: Vec::new(),
            recent: Vec::new(),
        }
    }

    /// Total runs visible.  Used by the panel header badge.
    pub fn total(&self) -> usize {
        self.running.len() + self.retrying.len() + self.disconnected.len() + self.recent.len()
    }

    /// Are there any actionable rows? (i.e., not just history).
    pub fn has_active(&self) -> bool {
        !self.running.is_empty() || !self.retrying.is_empty() || !self.disconnected.is_empty()
    }
}

/// A single row in the runs panel.  Composes a `RunRef` (ux01) with
/// presentation-time fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunsPanelRow {
    pub run_ref: SerRunRef,
    pub status: PanelStatus,
    pub attempt: u32,
    pub repo_display: String,
    pub tokens_input: u64,
    pub tokens_output: u64,
    pub last_event: Option<String>,
    /// `Some(...)` iff status is `Finished`.  Iter-28 #4-2 mirror.
    pub exit_summary: Option<String>,
}

impl RunsPanelRow {
    /// Keyboard / pointer activation: open the run detail view.
    pub fn url(&self) -> String {
        self.run_ref.to_url()
    }
}

/// Serializable RunRef wrapper.  RunRef itself doesn't derive Serde
/// (it's validated; we don't want to bypass via deserialize_unchecked);
/// this wrapper round-trips through the URL form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerRunRef(pub RunRef);

impl Serialize for SerRunRef {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0.to_url())
    }
}

impl<'de> Deserialize<'de> for SerRunRef {
    fn deserialize<D: serde::Deserializer<'de>>(deser: D) -> Result<Self, D::Error> {
        let s: String = Deserialize::deserialize(deser)?;
        s.parse::<RunRef>()
            .map(SerRunRef)
            .map_err(serde::de::Error::custom)
    }
}

impl SerRunRef {
    pub fn to_url(&self) -> String {
        self.0.to_url()
    }
}

/// Status as the panel renders it.  Matches the daemon's `RunStatus`
/// enum (iter-28 #4-2) plus the `Finished` terminal variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PanelStatus {
    Running,
    Retrying,
    Disconnected,
    Finished,
}

impl PanelStatus {
    /// Render as a single-character badge.  Used by terminal-style fallbacks.
    pub fn badge(&self) -> &'static str {
        match self {
            PanelStatus::Running => "▶",
            PanelStatus::Retrying => "↻",
            PanelStatus::Disconnected => "⌬",
            PanelStatus::Finished => "✓",
        }
    }

    /// Render a human-friendly status word.
    pub fn label(&self) -> &'static str {
        match self {
            PanelStatus::Running => "running",
            PanelStatus::Retrying => "retrying",
            PanelStatus::Disconnected => "disconnected",
            PanelStatus::Finished => "finished",
        }
    }

    /// Is this an actionable status (i.e., the user can reattach)?
    pub fn is_actionable(&self) -> bool {
        matches!(self, PanelStatus::Running | PanelStatus::Disconnected)
    }
}

/// Sort rows within a bucket: newest attempts first, then by run_id
/// (lexicographic), then by repo slug — fully deterministic order even
/// when two rows share the same `(attempt, run_id)` across distinct
/// repos.  Without the slug tiebreaker, refresh re-renders could shuffle
/// tied rows depending on snapshot input order.
pub fn sort_rows_by_recency(rows: &mut [RunsPanelRow]) {
    rows.sort_by(|a, b| {
        b.attempt
            .cmp(&a.attempt)
            .then_with(|| a.run_ref.0.run_id.cmp(&b.run_ref.0.run_id))
            .then_with(|| a.run_ref.0.slug.cmp(&b.run_ref.0.slug))
    });
}

/// Group rows into buckets per `PanelStatus`.  Used to convert a flat
/// snapshot row list into a `RunsPanelView`.
pub fn group_by_status(rows: Vec<RunsPanelRow>) -> RunsPanelView {
    let mut view = RunsPanelView::empty();
    for row in rows {
        match row.status {
            PanelStatus::Running => view.running.push(row),
            PanelStatus::Retrying => view.retrying.push(row),
            PanelStatus::Disconnected => view.disconnected.push(row),
            PanelStatus::Finished => view.recent.push(row),
        }
    }
    sort_rows_by_recency(&mut view.running);
    sort_rows_by_recency(&mut view.retrying);
    sort_rows_by_recency(&mut view.disconnected);
    sort_rows_by_recency(&mut view.recent);
    view
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(slug: &str, run_id: &str, status: PanelStatus, attempt: u32) -> RunsPanelRow {
        RunsPanelRow {
            run_ref: SerRunRef(RunRef::new(slug, run_id).unwrap()),
            status,
            attempt,
            repo_display: slug.to_string(),
            tokens_input: 100 * attempt as u64,
            tokens_output: 50 * attempt as u64,
            last_event: None,
            exit_summary: None,
        }
    }

    #[test]
    fn panel_view_empty() {
        let v = RunsPanelView::empty();
        assert_eq!(v.total(), 0);
        assert!(!v.has_active());
    }

    #[test]
    fn group_by_status_buckets_correctly() {
        let rows = vec![
            row("g_x_y", "01ABC", PanelStatus::Running, 1),
            row("g_x_y", "02DEF", PanelStatus::Retrying, 2),
            row("g_x_y", "03GHI", PanelStatus::Disconnected, 1),
            row("g_x_y", "04JKL", PanelStatus::Finished, 3),
        ];
        let v = group_by_status(rows);
        assert_eq!(v.running.len(), 1);
        assert_eq!(v.retrying.len(), 1);
        assert_eq!(v.disconnected.len(), 1);
        assert_eq!(v.recent.len(), 1);
        assert_eq!(v.total(), 4);
        assert!(v.has_active());
    }

    #[test]
    fn sort_rows_newest_attempt_first_then_run_id() {
        let mut rows = vec![
            row("g_x", "01", PanelStatus::Running, 1),
            row("g_x", "02", PanelStatus::Running, 3),
            row("g_x", "03", PanelStatus::Running, 2),
            row("g_x", "04", PanelStatus::Running, 3),
        ];
        sort_rows_by_recency(&mut rows);
        // Attempt 3 (run_id "02" < "04") -> 3/04 -> 2 -> 1.
        let order: Vec<&str> = rows.iter().map(|r| r.run_ref.0.run_id.as_str()).collect();
        assert_eq!(order, ["02", "04", "03", "01"]);
    }

    #[test]
    fn sort_rows_breaks_ties_on_repo_slug() {
        // Two rows with identical (attempt, run_id) across different
        // repos must sort deterministically by slug — input order MUST
        // NOT influence the result.
        let mut a = vec![
            row("g_a", "01", PanelStatus::Running, 1),
            row("g_b", "01", PanelStatus::Running, 1),
        ];
        let mut b = vec![
            row("g_b", "01", PanelStatus::Running, 1),
            row("g_a", "01", PanelStatus::Running, 1),
        ];
        sort_rows_by_recency(&mut a);
        sort_rows_by_recency(&mut b);
        let order_a: Vec<&str> = a.iter().map(|r| r.run_ref.0.slug.as_str()).collect();
        let order_b: Vec<&str> = b.iter().map(|r| r.run_ref.0.slug.as_str()).collect();
        assert_eq!(order_a, order_b);
        assert_eq!(order_a, ["g_a", "g_b"]);
    }

    #[test]
    fn finished_only_view_is_not_active() {
        // `has_active` MUST require at least one row in
        // running/retrying/disconnected; a recent-only view is inert.
        let v = group_by_status(vec![row("g_x", "01", PanelStatus::Finished, 1)]);
        assert_eq!(v.total(), 1);
        assert!(!v.has_active());
    }

    #[test]
    fn panel_status_badge_and_label() {
        assert_eq!(PanelStatus::Running.badge(), "▶");
        assert_eq!(PanelStatus::Retrying.label(), "retrying");
        assert!(PanelStatus::Running.is_actionable());
        assert!(PanelStatus::Disconnected.is_actionable());
        assert!(!PanelStatus::Finished.is_actionable());
        assert!(!PanelStatus::Retrying.is_actionable());
    }

    #[test]
    fn ser_run_ref_round_trip_via_url_string() {
        let r = SerRunRef(RunRef::new("g_x_y", "01ABC").unwrap());
        let s = serde_json::to_string(&r).unwrap();
        // Always serializes as a single string (the URL).
        assert_eq!(s, "\"caduceus://run/g_x_y/01ABC\"");
        let back: SerRunRef = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn ser_run_ref_rejects_malformed_url() {
        let r: Result<SerRunRef, _> = serde_json::from_str("\"https://run/x/y\"");
        assert!(r.is_err());
    }

    #[test]
    fn row_url_returns_run_ref_url() {
        let r = row("g_x", "01", PanelStatus::Running, 1);
        assert_eq!(r.url(), "caduceus://run/g_x/01");
    }

    #[test]
    fn panel_row_serialize_round_trip() {
        let r = row("g_x", "01", PanelStatus::Running, 1);
        let s = serde_json::to_string(&r).unwrap();
        let back: RunsPanelRow = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
