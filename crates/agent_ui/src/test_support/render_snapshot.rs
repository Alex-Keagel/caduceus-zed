//! Test-support utilities for `agent_ui` rendering tests.
//!
//! ## ST1b-prereq — render-snapshot harness
//!
//! Thin wrapper around [`gpui::VisualTestContext`] that gives selector-keyed
//! assertions over what an [`gpui::Element`] actually rendered. Used by ST1b
//! and any future UI work that needs to assert the *shape* of a render
//! (which rows are present, which badges are visible, what gets clicked).
//!
//! Intentionally minimal: GPUI's `debug_bounds(selector)` accessor takes
//! `&'static str` (the underlying `rendered_frame.debug_bounds` map is
//! `pub(crate)`), so we can't enumerate selectors generically. Tests pass the
//! selectors they care about up front and the harness records which ones
//! survived the render.
//!
//! Why not push for a public iterator API in GPUI? Two reasons:
//! 1. Selector-list-up-front maps cleanly to the spec-driven UI tests we
//!    actually want to write ("after rendering the unauth google row, expect
//!    selector `provider-row-google-signin` to be present").
//! 2. Keeps this PR self-contained — no GPUI surface-area changes that would
//!    need coordination across the larger zed codebase.

#![cfg(any(test, feature = "test-support"))]

use std::collections::BTreeMap;

use gpui::{
    App, AvailableSpace, Bounds, MouseButton, MouseDownEvent, MouseUpEvent, Pixels, Point, Size,
    VisualTestContext, Window,
};

/// Snapshot of an element's render result, keyed by selectors registered via
/// `Element::debug_selector(...)` during the render pass.
///
/// Construct via [`RenderSnapshot::capture`]. Query via [`RenderSnapshot::has`],
/// [`RenderSnapshot::assert_has`], [`RenderSnapshot::assert_missing`], and
/// [`RenderSnapshot::bounds_of`].
#[derive(Debug)]
pub struct RenderSnapshot {
    /// Selectors that survived the render (had nonzero presence in the painted
    /// frame).
    bounds: BTreeMap<&'static str, Bounds<Pixels>>,
    /// All selectors the test asked about, including ones that *didn't* survive
    /// the render. Used so error messages can list "you queried these N
    /// selectors, here's what was actually present" rather than just dumping
    /// the matched subset.
    queried: Vec<&'static str>,
}

impl RenderSnapshot {
    /// Draw `f` into `cx` at the given origin/size, then capture the bounds of
    /// every selector in `selectors` that was painted.
    ///
    /// Selectors not present in the painted frame are silently omitted from
    /// [`RenderSnapshot::bounds_of`] but remain in [`RenderSnapshot::queried`]
    /// so `assert_has` can produce useful error messages.
    pub fn capture<E: gpui::Element>(
        cx: &mut VisualTestContext,
        origin: Point<Pixels>,
        space: impl Into<Size<AvailableSpace>>,
        selectors: &[&'static str],
        f: impl FnOnce(&mut Window, &mut App) -> E,
    ) -> Self {
        cx.draw(origin, space, f);
        let mut bounds = BTreeMap::new();
        for &sel in selectors {
            if let Some(b) = cx.debug_bounds(sel) {
                bounds.insert(sel, b);
            }
        }
        Self {
            bounds,
            queried: selectors.to_vec(),
        }
    }

    /// True iff `selector` was rendered (its element appeared in the painted
    /// frame).
    ///
    /// Panics if `selector` was not in the queried set, since the answer would
    /// be a vacuous `false` — a footgun for refactor drift.
    #[track_caller]
    pub fn has(&self, selector: &str) -> bool {
        self.ensure_queried(selector, "has");
        self.bounds.contains_key(selector)
    }

    /// Panics with a list of which selectors were/weren't present if `selector`
    /// is missing.
    #[track_caller]
    pub fn assert_has(&self, selector: &str) {
        self.ensure_queried(selector, "assert_has");
        if !self.bounds.contains_key(selector) {
            let present: Vec<&&'static str> = self.bounds.keys().collect();
            let missing: Vec<&&'static str> = self
                .queried
                .iter()
                .filter(|s| !self.bounds.contains_key(*s))
                .collect();
            panic!(
                "expected selector `{selector}` to be rendered.\n  present: {present:?}\n  missing: {missing:?}"
            );
        }
    }

    /// Panics if `selector` *was* rendered, or if `selector` was never in the
    /// queried set (which would make this assertion vacuously true).
    #[track_caller]
    pub fn assert_missing(&self, selector: &str) {
        self.ensure_queried(selector, "assert_missing");
        if let Some(b) = self.bounds.get(selector) {
            panic!("selector `{selector}` should NOT be rendered, but was at {b:?}");
        }
    }

    /// The painted bounds of `selector`, or None if it wasn't rendered.
    ///
    /// Panics if `selector` was not in the queried set.
    #[track_caller]
    pub fn bounds_of(&self, selector: &str) -> Option<Bounds<Pixels>> {
        self.ensure_queried(selector, "bounds_of");
        self.bounds.get(selector).copied()
    }

    #[track_caller]
    fn ensure_queried(&self, selector: &str, op: &str) {
        if !self.queried.iter().any(|s| *s == selector) {
            panic!(
                "{op}: selector `{selector}` was not included in RenderSnapshot::capture selectors.\n  queried: {:?}\n  hint: add `{selector}` to the selectors list at capture time.",
                self.queried
            );
        }
    }

    /// Iterator over (selector, bounds) for every selector that was rendered.
    pub fn rendered(&self) -> impl Iterator<Item = (&'static str, Bounds<Pixels>)> + '_ {
        self.bounds.iter().map(|(s, b)| (*s, *b))
    }

    /// Selectors the test asked about, in original order.
    pub fn queried(&self) -> &[&'static str] {
        &self.queried
    }
}

/// Simulate a primary-button click at the centre of `selector`'s painted
/// bounds. Sends both `MouseDownEvent` and `MouseUpEvent` and runs the
/// foreground executor until parked, so any handlers fire before the call
/// returns.
///
/// Panics if the selector is not present in the most-recently rendered frame.
#[track_caller]
pub fn click_selector(cx: &mut VisualTestContext, selector: &'static str) {
    let bounds = cx
        .debug_bounds(selector)
        .unwrap_or_else(|| panic!("click_selector: selector `{selector}` not found in frame"));
    let position = bounds.center();
    cx.simulate_event(MouseDownEvent {
        button: MouseButton::Left,
        position,
        modifiers: Default::default(),
        click_count: 1,
        first_mouse: false,
    });
    cx.simulate_event(MouseUpEvent {
        button: MouseButton::Left,
        position,
        modifiers: Default::default(),
        click_count: 1,
    });
}

// ─── Tests for the harness itself ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        Hsla, InteractiveElement, IntoElement, ParentElement, Styled, TestAppContext, div, px,
    };

    fn full_window(cx: &mut TestAppContext) -> &mut VisualTestContext {
        cx.add_empty_window()
    }

    #[gpui::test]
    fn capture_records_selectors_that_render(cx: &mut TestAppContext) {
        let cx = full_window(cx);
        let snap = RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(400.0)),
                height: AvailableSpace::Definite(px(200.0)),
            },
            &["row-a", "row-b", "row-missing"],
            |_w, _cx| {
                div()
                    .size_full()
                    .child(
                        div()
                            .id("a")
                            .h(px(40.0))
                            .w_full()
                            .bg(Hsla::transparent_black())
                            .debug_selector(|| "row-a".to_string()),
                    )
                    .child(
                        div()
                            .id("b")
                            .h(px(40.0))
                            .w_full()
                            .bg(Hsla::transparent_black())
                            .debug_selector(|| "row-b".to_string()),
                    )
                    .into_any_element()
            },
        );
        assert!(snap.has("row-a"));
        assert!(snap.has("row-b"));
        assert!(!snap.has("row-missing"));
        snap.assert_has("row-a");
        snap.assert_missing("row-missing");
        assert_eq!(snap.queried().len(), 3);
        assert_eq!(snap.rendered().count(), 2);
    }

    #[gpui::test]
    #[should_panic(expected = "expected selector `row-x` to be rendered")]
    fn assert_has_panics_with_useful_message(cx: &mut TestAppContext) {
        let cx = full_window(cx);
        let snap = RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(100.0)),
                height: AvailableSpace::Definite(px(100.0)),
            },
            &["row-x"],
            |_w, _cx| div().into_any_element(),
        );
        snap.assert_has("row-x");
    }

    #[gpui::test]
    #[should_panic(expected = "selector `row-y` should NOT be rendered")]
    fn assert_missing_panics_when_present(cx: &mut TestAppContext) {
        let cx = full_window(cx);
        let snap = RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(100.0)),
                height: AvailableSpace::Definite(px(100.0)),
            },
            &["row-y"],
            |_w, _cx| {
                div()
                    .id("y")
                    .size_full()
                    .bg(Hsla::transparent_black())
                    .debug_selector(|| "row-y".to_string())
                    .into_any_element()
            },
        );
        snap.assert_missing("row-y");
    }

    #[gpui::test]
    fn bounds_of_returns_painted_bounds(cx: &mut TestAppContext) {
        let cx = full_window(cx);
        let snap = RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(400.0)),
                height: AvailableSpace::Definite(px(80.0)),
            },
            &["row-tagged"],
            |_w, _cx| {
                div()
                    .id("tagged")
                    .size_full()
                    .bg(Hsla::transparent_black())
                    .debug_selector(|| "row-tagged".to_string())
                    .into_any_element()
            },
        );
        let b = snap.bounds_of("row-tagged").expect("row-tagged painted");
        assert!(b.size.width > px(0.0));
        assert!(b.size.height > px(0.0));
    }

    #[gpui::test]
    #[should_panic(expected = "assert_missing: selector `row-unqueried` was not included")]
    fn assert_missing_panics_for_unqueried_selector(cx: &mut TestAppContext) {
        let cx = full_window(cx);
        let snap = RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(100.0)),
                height: AvailableSpace::Definite(px(100.0)),
            },
            &["row-a"],
            |_w, _cx| {
                div()
                    .id("x")
                    .size_full()
                    .bg(Hsla::transparent_black())
                    .debug_selector(|| "row-unqueried".to_string())
                    .into_any_element()
            },
        );
        // Even though `row-unqueried` IS rendered, this would silently pass
        // without the guard — so the guard must panic.
        snap.assert_missing("row-unqueried");
    }

    #[gpui::test]
    fn click_selector_dispatches_click(cx: &mut TestAppContext) {
        use gpui::InteractiveElement as _;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let cx = full_window(cx);
        let clicks = Arc::new(AtomicUsize::new(0));
        let clicks_for_handler = clicks.clone();

        RenderSnapshot::capture(
            cx,
            Point::default(),
            Size {
                width: AvailableSpace::Definite(px(120.0)),
                height: AvailableSpace::Definite(px(60.0)),
            },
            &["click-target"],
            move |_w, _cx| {
                div()
                    .id("click-target")
                    .size_full()
                    .bg(Hsla::transparent_black())
                    .debug_selector(|| "click-target".to_string())
                    .on_mouse_down(MouseButton::Left, move |_e, _w, _cx| {
                        clicks_for_handler.fetch_add(1, Ordering::SeqCst);
                    })
                    .into_any_element()
            },
        );

        click_selector(cx, "click-target");
        assert_eq!(clicks.load(Ordering::SeqCst), 1);
    }
}
