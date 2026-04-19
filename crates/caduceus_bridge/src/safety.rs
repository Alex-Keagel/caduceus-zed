//! Execution guardrails — loop detection, circuit breaker, compaction cooldown.
//!
//! These are Zed-side safety mechanisms extracted into testable structs.
//! They model the exact semantics from `agent/src/thread.rs`.

use std::time::{Duration, Instant};

// ── Loop Detector ──────────────────────────────────────────────────────────────

/// Detects when the same tool is called too many times consecutively
/// **with identical inputs**. Different inputs (e.g. parallel `edit_file`
/// on different files) are treated as legitimate concurrent work.
///
/// Semantics: the detector fires when the count reaches `threshold`.
/// The count starts at 1 on the first call and increments on each
/// consecutive call of the same `(tool_name, input_hash)`. Any change
/// to either name or input hash resets the count.
///
/// In `thread.rs`, the check happens BEFORE incrementing, so:
/// - threshold=3 means the 4th identical call is blocked
///   (count=3 when checked → blocked, then reset)
#[derive(Debug, Clone)]
pub struct LoopDetector {
    current_key: Option<(String, u64)>,
    consecutive_count: usize,
    threshold: usize,
}

/// Result of recording a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopCheckResult {
    /// Tool call is allowed.
    Ok,
    /// Loop detected — contains the tool name that looped.
    LoopDetected(String),
}

impl LoopDetector {
    pub fn new(threshold: usize) -> Self {
        assert!(threshold > 0, "LoopDetector threshold must be > 0");
        Self {
            current_key: None,
            consecutive_count: 0,
            threshold,
        }
    }

    /// Hash the tool input — used to distinguish parallel calls with
    /// different inputs from genuine retry loops.
    fn hash_input(input: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        input.hash(&mut h);
        h.finish()
    }

    /// Record a tool call by name only (legacy — assumes empty input).
    /// Prefer `record_call(name, input)` for accurate loop detection.
    pub fn record_tool(&mut self, tool_name: &str) -> LoopCheckResult {
        self.record_call(tool_name, "")
    }

    /// Record a tool call with its serialized input. Returns `LoopDetected`
    /// only if the same `(tool_name, input)` pair has been seen `threshold`
    /// times consecutively.
    pub fn record_call(&mut self, tool_name: &str, input: &str) -> LoopCheckResult {
        let key = (tool_name.to_string(), Self::hash_input(input));

        let is_loop = match &self.current_key {
            Some(prev) => *prev == key && self.consecutive_count >= self.threshold,
            None => false,
        };

        if is_loop {
            self.reset();
            return LoopCheckResult::LoopDetected(tool_name.to_string());
        }

        match &self.current_key {
            Some(prev) if *prev == key => {
                self.consecutive_count += 1;
            }
            _ => {
                self.current_key = Some(key);
                self.consecutive_count = 1;
            }
        }

        LoopCheckResult::Ok
    }

    /// Reset the detector (called internally after loop detection).
    fn reset(&mut self) {
        self.current_key = None;
        self.consecutive_count = 0;
    }

    /// Current consecutive count for the active (tool, input) pair.
    pub fn consecutive_count(&self) -> usize {
        self.consecutive_count
    }
}

// ── Circuit Breaker ────────────────────────────────────────────────────────────

/// Trips after N consecutive tool failures, stopping the agent from
/// endlessly retrying broken operations.
///
/// Semantics from thread.rs:
/// - `record_result(is_error)` increments on error, resets on success
/// - `is_tripped()` returns true when failures >= threshold
/// - `record_permission_denied()` resets the counter (permission denials
///   are not real failures — they indicate a mode mismatch)
/// - When tripped, caller should reset via `reset()`.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    consecutive_failures: u32,
    threshold: u32,
}

impl CircuitBreaker {
    pub fn new(threshold: u32) -> Self {
        assert!(threshold > 0, "CircuitBreaker threshold must be > 0");
        Self {
            consecutive_failures: 0,
            threshold,
        }
    }

    /// Record a tool result. Increments on error, resets on success.
    pub fn record_result(&mut self, is_error: bool) {
        if is_error {
            self.consecutive_failures += 1;
        } else {
            self.consecutive_failures = 0;
        }
    }

    /// Permission denials reset the failure counter (they are not real failures).
    pub fn record_permission_denied(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Check if the breaker is tripped (>= threshold consecutive failures).
    pub fn is_tripped(&self) -> bool {
        self.consecutive_failures >= self.threshold
    }

    /// Reset the breaker after it has tripped.
    pub fn reset(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Current consecutive failure count.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }
}

// ── Compaction Cooldown ────────────────────────────────────────────────────────

/// Prevents compaction from firing too frequently (e.g., double-fire
/// when multiple messages arrive in quick succession).
///
/// Uses injectable time for testability — `can_compact_at(now)` instead
/// of `can_compact()` with internal `Instant::now()`.
#[derive(Debug, Clone)]
pub struct CompactionCooldown {
    last_compacted: Option<Instant>,
    cooldown: Duration,
}

impl CompactionCooldown {
    pub fn new(cooldown: Duration) -> Self {
        Self {
            last_compacted: None,
            cooldown,
        }
    }

    /// Check if compaction is allowed at the given instant.
    pub fn can_compact_at(&self, now: Instant) -> bool {
        match self.last_compacted {
            None => true,
            Some(last) => now.duration_since(last) >= self.cooldown,
        }
    }

    /// Convenience: check using `Instant::now()`.
    pub fn can_compact(&self) -> bool {
        self.can_compact_at(Instant::now())
    }

    /// Record that compaction occurred at the given instant.
    pub fn record_compaction_at(&mut self, now: Instant) {
        self.last_compacted = Some(now);
    }

    /// Convenience: record using `Instant::now()`.
    pub fn record_compaction(&mut self) {
        self.record_compaction_at(Instant::now());
    }

    /// The last time compaction ran, if ever.
    pub fn last_compacted(&self) -> Option<Instant> {
        self.last_compacted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── LoopDetector ───────────────────────────────────────────────────────

    #[test]
    fn loop_detector_no_trigger_on_different_tools() {
        let mut ld = LoopDetector::new(3);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok);
        assert_eq!(ld.record_tool("read_file"), LoopCheckResult::Ok);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok);
        assert_eq!(ld.record_tool("read_file"), LoopCheckResult::Ok);
    }

    #[test]
    fn loop_detector_triggers_on_4th_consecutive_call() {
        // threshold=3: check happens BEFORE increment
        // Call 1: count=1 (set), Call 2: count=2, Call 3: count=3
        // Call 4: check count>=3 → LOOP
        let mut ld = LoopDetector::new(3);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok); // count=1
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok); // count=2
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok); // count=3
        assert_eq!(
            ld.record_tool("edit_file"),
            LoopCheckResult::LoopDetected("edit_file".to_string())
        ); // count=3 >= 3 → blocked
    }

    #[test]
    fn loop_detector_resets_after_detection() {
        let mut ld = LoopDetector::new(3);
        for _ in 0..3 {
            ld.record_tool("edit_file");
        }
        assert_eq!(
            ld.record_tool("edit_file"),
            LoopCheckResult::LoopDetected("edit_file".to_string())
        );
        // After detection, counter is reset — can start fresh
        assert_eq!(ld.consecutive_count(), 0);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok);
        assert_eq!(ld.consecutive_count(), 1);
    }

    #[test]
    fn loop_detector_different_tool_resets_count() {
        let mut ld = LoopDetector::new(3);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok);
        assert_eq!(ld.consecutive_count(), 2);
        assert_eq!(ld.record_tool("read_file"), LoopCheckResult::Ok);
        assert_eq!(ld.consecutive_count(), 1);
    }

    #[test]
    fn loop_detector_threshold_1_blocks_second_call() {
        let mut ld = LoopDetector::new(1);
        assert_eq!(ld.record_tool("edit_file"), LoopCheckResult::Ok); // count=1
        assert_eq!(
            ld.record_tool("edit_file"),
            LoopCheckResult::LoopDetected("edit_file".to_string())
        ); // count=1 >= 1 → blocked
    }

    #[test]
    #[should_panic(expected = "threshold must be > 0")]
    fn loop_detector_rejects_zero_threshold() {
        LoopDetector::new(0);
    }

    #[test]
    fn loop_detector_parallel_different_inputs_not_a_loop() {
        // Modern LLMs return parallel tool calls (e.g. edit_file on 5 different files).
        // These must NOT be flagged as a loop.
        let mut ld = LoopDetector::new(3);
        assert_eq!(
            ld.record_call("edit_file", r#"{"path":"a.rs","old":"x","new":"y"}"#),
            LoopCheckResult::Ok
        );
        assert_eq!(
            ld.record_call("edit_file", r#"{"path":"b.rs","old":"x","new":"y"}"#),
            LoopCheckResult::Ok
        );
        assert_eq!(
            ld.record_call("edit_file", r#"{"path":"c.rs","old":"x","new":"y"}"#),
            LoopCheckResult::Ok
        );
        assert_eq!(
            ld.record_call("edit_file", r#"{"path":"d.rs","old":"x","new":"y"}"#),
            LoopCheckResult::Ok
        );
        assert_eq!(
            ld.record_call("edit_file", r#"{"path":"e.rs","old":"x","new":"y"}"#),
            LoopCheckResult::Ok
        );
    }

    #[test]
    fn loop_detector_identical_input_is_a_loop() {
        // Same tool, identical input N times = real retry loop.
        let mut ld = LoopDetector::new(3);
        let input = r#"{"path":"a.rs","old":"x","new":"y"}"#;
        assert_eq!(ld.record_call("edit_file", input), LoopCheckResult::Ok); // 1
        assert_eq!(ld.record_call("edit_file", input), LoopCheckResult::Ok); // 2
        assert_eq!(ld.record_call("edit_file", input), LoopCheckResult::Ok); // 3
        assert_eq!(
            ld.record_call("edit_file", input),
            LoopCheckResult::LoopDetected("edit_file".to_string())
        ); // 4 → blocked
    }

    // ── CircuitBreaker ─────────────────────────────────────────────────────

    #[test]
    fn circuit_breaker_does_not_trip_below_threshold() {
        let mut cb = CircuitBreaker::new(5);
        for _ in 0..4 {
            cb.record_result(true);
            assert!(!cb.is_tripped());
        }
    }

    #[test]
    fn circuit_breaker_trips_at_threshold() {
        let mut cb = CircuitBreaker::new(5);
        for _ in 0..5 {
            cb.record_result(true);
        }
        assert!(cb.is_tripped());
        assert_eq!(cb.consecutive_failures(), 5);
    }

    #[test]
    fn circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(5);
        for _ in 0..4 {
            cb.record_result(true);
        }
        cb.record_result(false); // success resets
        assert_eq!(cb.consecutive_failures(), 0);
        assert!(!cb.is_tripped());
    }

    #[test]
    fn circuit_breaker_permission_denied_resets() {
        let mut cb = CircuitBreaker::new(5);
        for _ in 0..4 {
            cb.record_result(true);
        }
        assert_eq!(cb.consecutive_failures(), 4);
        cb.record_permission_denied();
        assert_eq!(cb.consecutive_failures(), 0);
        assert!(!cb.is_tripped());
    }

    #[test]
    fn circuit_breaker_reset_allows_reuse() {
        let mut cb = CircuitBreaker::new(5);
        for _ in 0..5 {
            cb.record_result(true);
        }
        assert!(cb.is_tripped());
        cb.reset();
        assert!(!cb.is_tripped());
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    #[should_panic(expected = "threshold must be > 0")]
    fn circuit_breaker_rejects_zero_threshold() {
        CircuitBreaker::new(0);
    }

    // ── CompactionCooldown ─────────────────────────────────────────────────

    #[test]
    fn compaction_cooldown_allows_first_compaction() {
        let cd = CompactionCooldown::new(Duration::from_secs(30));
        let now = Instant::now();
        assert!(cd.can_compact_at(now));
    }

    #[test]
    fn compaction_cooldown_blocks_immediate_retry() {
        let mut cd = CompactionCooldown::new(Duration::from_secs(30));
        let now = Instant::now();
        cd.record_compaction_at(now);
        assert!(!cd.can_compact_at(now));
        assert!(!cd.can_compact_at(now + Duration::from_secs(15)));
    }

    #[test]
    fn compaction_cooldown_allows_after_cooldown() {
        let mut cd = CompactionCooldown::new(Duration::from_secs(30));
        let now = Instant::now();
        cd.record_compaction_at(now);
        assert!(cd.can_compact_at(now + Duration::from_secs(30)));
        assert!(cd.can_compact_at(now + Duration::from_secs(60)));
    }

    #[test]
    fn compaction_cooldown_tracks_last_compaction() {
        let mut cd = CompactionCooldown::new(Duration::from_secs(30));
        assert!(cd.last_compacted().is_none());
        cd.record_compaction();
        assert!(cd.last_compacted().is_some());
    }

    #[test]
    fn compaction_cooldown_boundary_minus_one() {
        let mut cd = CompactionCooldown::new(Duration::from_secs(30));
        let now = Instant::now();
        cd.record_compaction_at(now);
        assert!(!cd.can_compact_at(now + Duration::from_millis(29_999)));
        assert!(cd.can_compact_at(now + Duration::from_millis(30_000)));
    }
}
