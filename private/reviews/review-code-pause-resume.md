# Hardcore Review: P0-3 Background Agent pause/resume

- **Date**: 2026-04-19 14:42
- **Content Type**: Code Changes (item #4 of 8 deferred items)
- **Iteration**: 1
- **Verdict**: SHIP IT

## Bug
`AgentHandle::pause_token` was a `CancellationToken`, which is one-shot. `pause()` cancelled it, `resume()` flipped status to `Running` but **could not un-cancel the token**, so the loop kept observing `pause_token.is_cancelled() == true` and slept forever. The existing `pause_and_resume` test passed because it only checked status, not actual loop progress. Comment in `resume()` openly admitted the bug ("we can't un-cancel a CancellationToken... a production version would use a watch channel").

Secondary issue: `pause()` and `cancel()` held `handles.read()` across `agents.write().await`, holding two locks across an await point.

## Fix
- Replace `pause_token: CancellationToken` with `pause_signal: Arc<AtomicBool>`.
- `pause()` → `store(true, Release)`. `resume()` → `store(false, Release)`. Loop polls with `Ordering::Acquire`.
- Atomic ordering pairs Release/Acquire so an observer that sees `status == Running` is guaranteed to see the loop unblocked on its next 50ms poll.
- Drop `handles` read guard before taking `agents` write guard in pause/cancel; in resume(), validate state under agents-write first, then briefly take handles-read to clear the signal — keeps lock-acquisition order consistent.
- Pause sleep tightened from 200ms → 50ms so cancel-during-pause unblocks faster.

## Final Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Correctness | 9/10 | Atomic Release/Acquire pairs status flip with signal observation; cancel takes priority over pause via early-loop check |
| 2 | Completeness | 9/10 | Idempotent pause; cancel-during-pause; concurrent stress; resume actually unblocks |
| 3 | Security | 8/10 | No new attack surface |
| 4 | Clarity | 9/10 | Doc comment explains why CancellationToken was wrong; loop comments label cooperative pause |
| 5 | Architecture | 8/10 | Same lock-order convention applied to cancel() too |
| 6 | Test Coverage | 9/10 | 5 new behavior tests including a deadlock-detector with timeout |
| 8 | SRP | 9/10 | One handle field per concern (cancel_token, pause_signal) |
| 9 | KISS / YAGNI / DRY | 9/10 | AtomicBool is the simplest fit; no watch channel/Notify needed |

**Average**: 8.75/10 — converged.

## Test Results
`cargo test -p caduceus-orchestrator --lib` → **362/362 pass** ✅

New tests:
- `resume_actually_unblocks_loop` — agent reaches Completed after pause+resume (would have hung on old code)
- `pause_actually_blocks_progress` — paused agent stays Paused for 500ms, doesn't sneak forward
- `pause_is_idempotent` — double pause OK, then resume works
- `pause_then_cancel_completes_with_cancelled` — cancel takes priority over pause
- `concurrent_pause_resume_no_deadlock` — 10 agents × 5 pause/resume cycles, 3-second deadlock timeout

## Recommended Actions
- ✅ All applied. Ready to commit.
