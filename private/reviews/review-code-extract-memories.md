# Hardcore Review: gap-s2 extractMemories pipeline

- **Date**: 2026-04-19 14:50
- **Content Type**: Code Changes (item #5 of 8 deferred)
- **Iteration**: 1
- **Verdict**: SHIP IT

## Delivered
New `caduceus-orchestrator/src/memories.rs` (~520 lines, 15 tests):

- `MemoryDistiller` async trait — abstracts the LLM call so unit tests can inject deterministic responses.
- `FnDistiller<F>` — closure-based adapter so production callers can plug any provider/model in one line.
- `DEFAULT_EXTRACTION_PROMPT` — JSON-array-output prompt for OpenAI/Anthropic-compatible APIs.
- `parse_distiller_json` — tolerant parser (strips ```` ```json ```` fences, unknown categories → Other, garbage → empty Vec, never panics).
- `MemoryStore` — atomic file writer (`.tmp` + rename), creates parent dirs, FIFO trim at `max_entries`, dedupes by case+whitespace-normalized content, round-trips through hand-editable markdown.
- `MemoryExtractor` — orchestrates: skips below `min_messages`, truncates transcript at UTF-8 char boundary, distills, dedupes against existing store, returns only entries that were newly persisted.
- `MemoryCategory` — preference / fact / convention / skill / other; serializable; from-label tolerant of `pref` / `style` aliases.

The extracted entries land in `.caduceus/memory.md` which the existing `instructions.rs` (line 376) already loads into the system prompt — so the loop closes without any wiring change to existing modules.

## Final Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Correctness | 9/10 | Atomic write; UTF-8-safe truncation; pre-snapshot dedup avoids the "I just appended my own entry, it's now in the store" false-positive bug (caught & fixed during impl) |
| 2 | Completeness | 9/10 | Empty content, garbage JSON, multi-byte chars, FIFO trim, legacy markdown, distiller errors all covered |
| 3 | Security | 9/10 | No code execution path; markdown-only output; user-editable file is the boundary |
| 4 | Clarity | 9/10 | Module-level doc comment; each helper has a purpose comment |
| 5 | Architecture | 9/10 | Trait-based distiller decouples from `caduceus-providers`; FnDistiller for ergonomic wiring |
| 6 | Test Coverage | 9/10 | 15 tests: store round-trip, dedup variants, FIFO trim, atomic write, extractor min-messages skip, persistence, dedup vs store, char-boundary truncation, legacy parse, error propagation, JSON parser (4 cases), FnDistiller round-trip |
| 8 | SRP | 9/10 | Store handles persistence only; Extractor handles policy only; Distiller is the LLM seam |
| 9 | KISS / YAGNI / DRY | 8/10 | No premature provider-specific binding; left as opt-in via FnDistiller |

**Average**: 8.875/10 — converged on iteration 1.

## Test Results
`cargo test -p caduceus-orchestrator --lib` → **377/377 pass** ✅ (was 362; +15 memory tests)

## Notes
- One real bug caught by tests during impl: the post-append re-filter computed "what's in the store now" instead of "what wasn't in the store before", so duplicates were reported as new. Fixed by snapshotting `pre_existing` keys before append.
- Production wiring path: caller obtains a chat completion via existing `caduceus-providers`, wraps the call in `FnDistiller::new(|t| async move { ... parse_distiller_json(&resp) ... })`, and builds a `MemoryExtractor`. No new dependencies in the orchestrator crate.
