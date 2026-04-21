//! Global ring buffer of context-management events (auto-compaction,
//! eviction, skipped-compactions). The agent panel displays the most
//! recent N entries so the user can see — at a glance — that automatic
//! context maintenance is actually firing. **An empty list when the
//! conversation is long means something is broken.**
//!
//! Threads come and go; events outlive any single thread, which is why
//! this is global rather than per-Thread.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

/// Cap on retained events. Tuned for "show last few panels of activity"
/// without unbounded growth in long sessions.
const MAX_EVENTS: usize = 64;

#[derive(Debug, Clone)]
pub enum ContextEventKind {
    /// Auto-compaction ran and replaced N older messages with a summary.
    AutoCompacted {
        messages_compacted: usize,
        tokens_freed: usize,
        fill_pct: f64,
        zone: String,
        keep_recent: usize,
    },
    /// Compaction was triggered by the zone heuristic but blocked or no-op'd.
    /// `reason` is a short human-readable string ("cooldown", "degenerate
    /// summary", "zero budget", "below message threshold").
    CompactionSkipped {
        reason: String,
        fill_pct: f64,
        msg_count: usize,
    },
    /// Messages were dropped from the conversation tail (user resent an
    /// earlier message; turn was rolled back).
    MessageEvicted { count: usize, reason: String },
    /// Manual /compact slash-command outcome — recorded so users can see
    /// what their explicit command did.
    ManualCompactRequested { outcome: String },
}

#[derive(Debug, Clone)]
pub struct ContextEvent {
    pub at: SystemTime,
    pub kind: ContextEventKind,
}

fn buffer() -> &'static Mutex<VecDeque<ContextEvent>> {
    static BUF: OnceLock<Mutex<VecDeque<ContextEvent>>> = OnceLock::new();
    BUF.get_or_init(|| Mutex::new(VecDeque::with_capacity(MAX_EVENTS)))
}

pub fn record(kind: ContextEventKind) {
    let event = ContextEvent {
        at: SystemTime::now(),
        kind,
    };
    if let Ok(mut buf) = buffer().lock() {
        if buf.len() == MAX_EVENTS {
            buf.pop_front();
        }
        buf.push_back(event);
    }
}

/// Snapshot newest-first. Caller gets owned values so it can release
/// the lock immediately and render off-thread if it likes.
pub fn snapshot() -> Vec<ContextEvent> {
    buffer()
        .lock()
        .map(|b| b.iter().rev().cloned().collect())
        .unwrap_or_default()
}

/// Total events ever recorded since process start (including evicted ones).
/// Useful for the "if zero, something is off" sanity signal.
pub fn total_recorded() -> usize {
    counter().load(std::sync::atomic::Ordering::Relaxed)
}

fn bump_counter() {
    counter().fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Shared module-level counter. `total_recorded` and `bump_counter` previously
/// each declared their own function-scope `static OnceLock<AtomicUsize>`, which
/// made them two distinct statics — `bump_counter` incremented one, while
/// `total_recorded` read the other (always 0). The shared accessor here ensures
/// both observe the same atomic.
fn counter() -> &'static std::sync::atomic::AtomicUsize {
    static COUNTER: OnceLock<std::sync::atomic::AtomicUsize> = OnceLock::new();
    COUNTER.get_or_init(|| std::sync::atomic::AtomicUsize::new(0))
}

pub fn record_and_count(kind: ContextEventKind) {
    bump_counter();
    record(kind);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_caps_at_max() {
        // Use record (not record_and_count) so we don't pollute the global counter
        // shared across tests.
        for i in 0..(MAX_EVENTS * 2) {
            record(ContextEventKind::MessageEvicted {
                count: i,
                reason: "test".into(),
            });
        }
        let snap = snapshot();
        assert!(snap.len() <= MAX_EVENTS);
    }

    #[test]
    fn snapshot_is_newest_first() {
        record(ContextEventKind::MessageEvicted {
            count: 1,
            reason: "first".into(),
        });
        record(ContextEventKind::MessageEvicted {
            count: 2,
            reason: "second".into(),
        });
        let snap = snapshot();
        // The most recent matching event we just inserted should be ahead of
        // the older one in the snapshot.
        let idx_first = snap.iter().position(|e| {
            matches!(&e.kind,
            ContextEventKind::MessageEvicted { reason, .. } if reason == "first")
        });
        let idx_second = snap.iter().position(|e| {
            matches!(&e.kind,
            ContextEventKind::MessageEvicted { reason, .. } if reason == "second")
        });
        if let (Some(a), Some(b)) = (idx_first, idx_second) {
            assert!(b < a, "newest should appear before older in snapshot");
        }
    }

    #[test]
    fn record_and_count_increments_observable_total() {
        // Regression for: total_recorded() and bump_counter() previously
        // declared two distinct function-scope OnceLock<AtomicUsize> statics,
        // so total_recorded always returned 0. After unifying via counter()
        // they must share state.
        let before = total_recorded();
        for i in 0..3 {
            record_and_count(ContextEventKind::MessageEvicted {
                count: i,
                reason: "counter-regression".into(),
            });
        }
        let after = total_recorded();
        assert!(
            after >= before + 3,
            "total_recorded must reflect bump_counter writes (before={before}, after={after})"
        );
    }
}
