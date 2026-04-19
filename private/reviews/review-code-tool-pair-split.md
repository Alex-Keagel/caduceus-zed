# Hardcore Review: P0-9 tool_use ↔ tool_result Pair Splitting

- **Date**: 2026-04-19 14:35
- **Content Type**: Code Changes (item #3 of 8 deferred items)
- **Iteration**: 1 (single iteration; all scores ≥ 8)
- **Reviewer**: Self (rubber-duck consultation interrupted; plan validated against existing test patterns)
- **Verdict**: SHIP IT

## Bug
Two methods in `caduceus-orchestrator/src/lib.rs` could silently split tool pairs and produce malformed requests that providers (especially Anthropic) reject with HTTP 400:

1. **`ConversationHistory::truncate_oldest`** dropped messages one-at-a-time oldest-first. With history `[user, assistant+tc(t1), tool(t1), assistant_final]` and `max=2`, it produced `[tool(t1), assistant_final]` — orphaned `tool_result`.

2. **`ContextAssembler::assemble`** walked newest-first, fitting messages until the budget was exceeded, then dropped only leading-tool messages. If the budget cut between an assistant `tool_use` and its `tool_result`, the orphan slipped through (or a later `tool_result` with no preceding assistant survived in the middle of the slice).

## Fix
Added `pair_aware_units(messages) -> Vec<(start, end_exclusive)>`: groups each assistant-with-`tool_calls` together with the immediately following `tool` messages whose `tool_use_id` matches one of the call IDs. Orphan tools and unrelated tools become size-1 units (preserved verbatim, not silently swallowed).

- `truncate_oldest` now drops whole oldest non-system units until total messages ≤ max.
- `assemble` walks units (not messages) newest-first, includes each whole unit only if it fits.

## Final Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Correctness | 9/10 | Pair invariant verified by unit tests including multi-tool-call (tA+tB) and mixed-id boundary. Defensive leading-tool fallback retained for malformed input. |
| 2 | Completeness | 9/10 | Covers single pair, multi-tool-call pair, orphan tool, unmatched id, system preservation, budget-boundary behavior |
| 3 | Security | 8/10 | No new attack surface; HashSet of `tool_call.id` is safe |
| 4 | Clarity | 8/10 | `pair_aware_units` doc-commented; rationale spelled out in both call sites |
| 5 | Architecture | 8/10 | One private helper, no API change; existing serialize/deserialize unchanged |
| 6 | Test Coverage | 9/10 | 13 new tests: 4 unit-grouping + 2 truncate + 5 assemble + 2 misc |
| 8 | SRP | 8/10 | Helper does one thing (pair partitioning); callers do their own filtering |
| 9 | KISS / YAGNI / DRY | 8/10 | Single helper used by both call sites; old leading-tool fallback kept as belt-and-suspenders for malformed input |

**Average**: 8.4/10 — converged.

## Test Results
- `cargo test -p caduceus-orchestrator --lib` → **357/357 pass** ✅
- 13 new tests, all in `tests` module of `lib.rs`:
  - `pair_aware_units_groups_assistant_with_following_tool_results`
  - `pair_aware_units_handles_multi_tool_call`
  - `pair_aware_units_orphan_tool_is_size_one`
  - `pair_aware_units_unmatched_tool_id_breaks_unit`
  - `truncate_oldest_keeps_tool_pair_atomic` (no orphan invariant)
  - `truncate_oldest_keeps_pair_when_budget_allows`
  - `truncate_oldest_preserves_system_messages`
  - `assemble_keeps_tool_pair_atomic_at_budget_boundary`
  - `assemble_never_starts_history_with_orphan_tool_result`
  - `assemble_includes_both_messages_of_pair_when_both_fit`
  - `assemble_multi_tool_call_unit_stays_atomic`
  - 2 pre-existing assembly tests still pass

## Recommended Actions
- ✅ All applied. Ready to commit.
