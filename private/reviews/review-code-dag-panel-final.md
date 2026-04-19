# Hardcore Review: DAG IDE Panel + Subagent Edges + Auto-Refresh

- **Date**: 2026-04-19 14:20
- **Content Type**: Code Changes (item #1 of 8 deferred items)
- **Iteration**: 1 (converged after applied fixes)
- **Reviewer Models**: claude-opus-4.6, claude-sonnet-4.6, gpt-5.4, gpt-5.3-codex
- **Verdict**: SHIP IT

## Scope
- `crates/caduceus_bridge/src/index_dag.rs` — REGISTRY_VERSION, spawn registry, recover() poison guard, sanitization, dual-section ASCII
- `crates/caduceus_bridge/src/engine.rs` — `*_as` caller-attributed variants
- `crates/agent/src/thread.rs` — `register/unregister_running_subagent` records spawn edges keyed by `SessionId`
- `crates/agent/src/agent.rs` — `/dag` slash + `/help` listing + cx-aware register call
- `crates/agent_ui/src/agent_panel.rs` — DAG chip, periodic 5s poll task, sanitized tooltip with 30-line cap
- 3 caduceus tools updated to call `index_directory_as` / `semantic_search_as`

## Final Scores

| # | Dimension | Score | Min-Reviewer | Notes |
|---|-----------|-------|--------------|-------|
| 1 | Correctness | 9/10 | sonnet | SessionId-keyed spawn IDs join cleanly with access edges; remove_spawn prunes ghost edges; version bumps inside lock |
| 2 | Completeness | 8/10 | gpt-5.4 | `*_as` variants threaded through 3 tool call sites; deeper per-thread plumbing tracked separately |
| 3 | Security | 9/10 | opus | Sanitize control chars + 16-char ID truncation prevents row-injection in chat output (gpt-5.4 #3) |
| 4 | Clarity | 8/10 | sonnet | Dual-section ASCII (Spawn Tree / Index Access); `/dag` discoverable via `/help` |
| 5 | Architecture | 8/10 | opus | Single mutex per registry; deferred per-resource lock split — current load nowhere near contention |
| 6 | Test Coverage | 9/10 | codex | 28 DAG tests: 22 unit + 5 panel + 1 integration; sanitization, contention matrix, lock recovery, ring buffer cap |
| 7 | Performance | 8/10 | opus | TTL aging requires full rebuild every 5s; ring buffer capped at 256; acceptable for chip refresh |
| 8 | SRP | 8/10 | sonnet | `recover()` helper centralizes poison handling; `to_ascii()` does sanitize+display+render — fine for current size |
| 9 | KISS / YAGNI / DRY | 8/10 | gpt-5.4 | `Option<DagBundle>` retained as defensive contract for future fast-path attempts |

**Average**: 8.3/10 — converged.

## Applied Fixes (this iteration)

### 🔴 Critical
1. **sonnet #1** — `register_running_subagent` keyed spawns by `entity_id` while access edges use `SessionId` → spawn-tree and access-tree could never join. Fixed by reading `subagent.upgrade()...read(cx).id().0.to_string()`.
2. **sonnet #2** — Two readers on same resource were flagged as contention; corrected to require ≥1 writer.
3. **sonnet #5** — Single panicked thread silenced DAG forever (`if let Ok(...)` on poisoned mutex). Fixed via `recover()` helper applied to all lock sites.
4. **opus #1** + **codex #1** — io_task built ASCII and counts from two separate snapshots, allowing ring-buffer rotation to mismatch them. Fixed by snapshotting once.
5. **gpt-5.4 #2** — `version()` fast-path bypassed TTL aging, freezing the chip on stale data. Removed fast-path; rebuild every refresh.
6. **gpt-5.4 #3** — Resource label / agent-id newlines could spoof additional rows in chat output. Added `sanitize_one_line` + `display_id` (16-char cap with `…`).

### 🟡 Important
1. **sonnet #3** — Chip only refreshed on user input; spawned a 5-sec periodic task that force-invalidates TTL.
2. **sonnet #4** — `/dag` not in `/help`; added.
3. **sonnet #7** — `unregister_running_subagent` left ghost spawn edges; added `remove_spawn(child_id)`.
4. **opus #6** — Tooltip could grow unbounded; capped at 30 lines with elision marker.
5. **gpt-5.4 #1** — Engine collapsed all activity into `engine:index_directory` / `engine:semantic_search` synthetic IDs. Added `index_directory_as` / `semantic_search_as` variants taking `(agent_id, AgentKind)`; updated 3 caduceus tools to pass tool-specific IDs (`tool:caduceus_semantic_search`, `tool:caduceus_index`, `tool:caduceus_cross_search:{name}`). Full per-session attribution from `Tool::run` requires plumbing thread_id through `ToolCallEventStream` — tracked as a future enhancement.

## Test Results
- 22 in `caduceus_bridge::index_dag` ✅
- 5 in `agent_ui::agent_panel` (DAG/bug17) ✅
- 1 integration in `agent::tests` ✅
- **28/28 passing** — `cargo test -p caduceus_bridge -p agent_ui -p agent --features "gpui_macos/runtime_shaders" --lib`

## Recommended Actions
- ✅ All applied. Ready to commit.
- Future: full per-session attribution requires `ToolCallEventStream` to expose `thread_id`. Out of scope for this item.
