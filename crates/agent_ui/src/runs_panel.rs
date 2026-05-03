//! Caduceus runs panel — gpui View rendering the runs snapshot.
//!
//! Wired into the workspace via the `OpenCaduceusRunsPanel` action.
//! The action opens the panel as a workspace pane item (tab), mirroring
//! the AgentRegistryPage pattern.
//!
//! Data is sourced from `caduceus_bridge::runs_panel::RunsPanelView`;
//! v1 displays a static empty fixture because the daemon-IPC client
//! that streams real snapshots lands when caduceus-daemon becomes a
//! workspace dep.  Hooking real data is a one-line swap of the
//! fixture for a streamed view from the bridge's snapshot subscriber.

use caduceus_bridge::runs_panel::{
    RunsPanelRow, RunsPanelView, group_by_status, sort_rows_by_recency,
};
use gpui::{
    AnyElement, App, Context, Entity, EventEmitter, FocusHandle, Focusable, IntoElement,
    ParentElement, Render, SharedString, Styled, Window, div, prelude::FluentBuilder,
};
use ui::{Color, IconName, Label, LabelCommon, LabelSize, h_flex, prelude::*, v_flex};
use workspace::{
    Workspace,
    item::{Item, ItemEvent},
};

/// Workspace pane item rendering the runs snapshot.
///
/// `view` is the materialized projection consumed by render; replace
/// `set_view` with a streaming subscriber when daemon IPC lands.
pub struct RunsPanel {
    focus_handle: FocusHandle,
    view: RunsPanelView,
}

impl RunsPanel {
    pub fn new(
        _workspace: &Workspace,
        _window: &mut Window,
        cx: &mut Context<Workspace>,
    ) -> Entity<Self> {
        cx.new(|cx| RunsPanel {
            focus_handle: cx.focus_handle(),
            view: RunsPanelView::empty(),
        })
    }

    /// Replace the displayed snapshot.  Triggers a re-render.
    #[allow(dead_code)] // public API consumed by daemon-ipc client (follow-up)
    pub fn set_view(&mut self, view: RunsPanelView, cx: &mut Context<Self>) {
        // Sort each bucket for stable display ordering.
        let mut view = view;
        sort_rows_by_recency(&mut view.running);
        sort_rows_by_recency(&mut view.retrying);
        sort_rows_by_recency(&mut view.disconnected);
        sort_rows_by_recency(&mut view.recent);
        self.view = view;
        cx.notify();
    }

    /// Replace the displayed snapshot from a flat row list.  Convenience
    /// for callers that haven't bucketed yet.
    #[allow(dead_code)] // public API consumed by daemon-ipc client (follow-up)
    pub fn set_rows(&mut self, rows: Vec<RunsPanelRow>, cx: &mut Context<Self>) {
        self.set_view(group_by_status(rows), cx);
    }

    fn render_bucket(label: &'static str, rows: &[RunsPanelRow]) -> AnyElement {
        v_flex()
            .gap_1()
            .child(
                Label::new(format!("{label} ({})", rows.len()))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
            )
            .when(rows.is_empty(), |this| {
                this.child(
                    Label::new("(none)")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                )
            })
            .when(!rows.is_empty(), |this| {
                this.children(rows.iter().map(Self::render_row))
            })
            .into_any_element()
    }

    fn render_row(row: &RunsPanelRow) -> AnyElement {
        let badge: SharedString = row.status.badge().into();
        let status_label: SharedString = row.status.label().into();
        let run_id_short: SharedString = row.run_ref.0.run_id.clone().into();
        let repo: SharedString = row.repo_display.clone().into();
        let attempt = format!("attempt {}", row.attempt);
        let tokens = format!("{} in / {} out", row.tokens_input, row.tokens_output);
        h_flex()
            .gap_2()
            .child(Label::new(badge).size(LabelSize::Default))
            .child(Label::new(run_id_short).size(LabelSize::Default))
            .child(Label::new(repo).size(LabelSize::Small).color(Color::Muted))
            .child(
                Label::new(SharedString::from(attempt))
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
            .child(
                Label::new(SharedString::from(tokens))
                    .size(LabelSize::XSmall)
                    .color(Color::Muted),
            )
            .child(Label::new(status_label).size(LabelSize::XSmall).color(
                if row.status.is_actionable() {
                    Color::Accent
                } else {
                    Color::Muted
                },
            ))
            .into_any_element()
    }
}

impl Render for RunsPanel {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl gpui::IntoElement {
        let total = self.view.total();
        v_flex()
            .size_full()
            .p_4()
            .gap_4()
            .child(
                v_flex()
                    .gap_1()
                    .child(Label::new("Caduceus runs").size(LabelSize::Large))
                    .child(
                        Label::new(SharedString::from(format!("{total} total")))
                            .size(LabelSize::XSmall)
                            .color(Color::Muted),
                    ),
            )
            .child(div().border_t_1())
            .child(Self::render_bucket("Running", &self.view.running))
            .child(Self::render_bucket("Retrying", &self.view.retrying))
            .child(Self::render_bucket("Disconnected", &self.view.disconnected))
            .child(Self::render_bucket("Recent", &self.view.recent))
    }
}

impl Focusable for RunsPanel {
    fn focus_handle(&self, _cx: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

impl EventEmitter<ItemEvent> for RunsPanel {}

impl Item for RunsPanel {
    type Event = ItemEvent;

    fn tab_content_text(&self, _detail: usize, _cx: &App) -> SharedString {
        "Caduceus Runs".into()
    }

    fn tab_icon(&self, _window: &Window, _cx: &App) -> Option<ui::Icon> {
        Some(ui::Icon::new(IconName::Server))
    }

    fn telemetry_event_text(&self) -> Option<&'static str> {
        Some("Caduceus Runs Panel Opened")
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn to_item_events(event: &Self::Event, f: &mut dyn FnMut(ItemEvent)) {
        f(*event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use caduceus_bridge::run_ref::RunRef;
    use caduceus_bridge::runs_panel::{PanelStatus, RunsPanelRow, SerRunRef};

    fn row(id: &str, status: PanelStatus, attempt: u32) -> RunsPanelRow {
        RunsPanelRow {
            run_ref: SerRunRef(RunRef::new("g_o_r", id).unwrap()),
            status,
            attempt,
            repo_display: "g_o_r".into(),
            tokens_input: 100,
            tokens_output: 50,
            last_event: None,
            exit_summary: None,
        }
    }

    #[test]
    fn bucketing_helpers_call_through_to_bridge() {
        // Sanity: ensure the bridge helpers we re-use here produce
        // expected buckets.  NOTE: render/open-path gpui tests for
        // RunsPanel are still TODO — they belong with the daemon-IPC
        // streaming hookup landing in the follow-up PR.
        let rows = vec![
            row("01", PanelStatus::Running, 1),
            row("02", PanelStatus::Finished, 1),
        ];
        let view = group_by_status(rows);
        assert_eq!(view.running.len(), 1);
        assert_eq!(view.recent.len(), 1);
    }
}
