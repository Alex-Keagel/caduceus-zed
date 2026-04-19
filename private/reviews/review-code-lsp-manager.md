# Hardcore Review: gap-s1 LSP Manager service

- **Date**: 2026-04-19 14:58
- **Content Type**: Code Changes (item #7 of 8 deferred)
- **Iteration**: 1
- **Verdict**: SHIP IT

## Delivered
New `caduceus-tools/src/lsp.rs` (~440 lines, 14 tests) — language-server façade for agent tools.

- `LspProvider` async trait — one adapter per language (definition, references, hover, document_symbols, diagnostics, shutdown).
- `LspManager` — thread-safe registry mapping file extension → language → provider. Cheap to clone (Arc<RwLock>).
- 27 default extension → language mappings seeded in `new()` (rs, ts, tsx, py, go, java, kt, rb, c/cpp, cs, swift, php, scala, ex/exs, erl, hs, ml, zig, dart, lua...).
- Uniform error model: `NoProvider` / `UnknownLanguage` / `Provider` / `Io` — surfaces actionable info to callers (e.g. "install rust-analyzer").
- Custom extension → language overrides via `register_extension`.
- `shutdown_all()` for clean teardown.

## Final Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Correctness | 9/10 | Resolve fails fast & explicit on missing extension/lang/provider; case-insensitive ext matching |
| 2 | Completeness | 9/10 | All 5 LSP ops, install/uninstall/list/shutdown, default + custom extensions, concurrent install stress |
| 3 | Security | 8/10 | Trait surface only; provider impls own their own sandboxing |
| 4 | Clarity | 9/10 | Module doc + wiring example; refactored confusing `Arc::get_mut` dead-code path away after first impl |
| 5 | Architecture | 9/10 | Clean trait/registry split; agent tools depend on LspManager not on individual providers |
| 6 | Test Coverage | 9/10 | 14 tests: defaults, unknown ext, no provider, custom override, case-insensitive, error propagation, list-sorted, uninstall, shutdown-all, clone-shares-state, concurrent-no-deadlock, serde |
| 8 | SRP | 9/10 | Manager handles routing; Provider handles I/O; types separate from policy |
| 9 | KISS / YAGNI / DRY | 8/10 | No premature LSP wire-protocol code; the trait is the seam concrete adapters fill |

**Average**: 8.75/10 — converged.

## Test Results
`cargo test -p caduceus-tools --lib` → **150/150 pass** ✅ (was 136; +14 LSP tests)

## Wiring Path (production)
1. Build a concrete `RustAnalyzerProvider: LspProvider` in a follow-up that spawns `rust-analyzer` over stdio JSON-RPC.
2. Inject `Arc<LspManager>` into agent tools (e.g. `caduceus_symbol_lookup_tool`) via constructor.
3. The manager survives across calls and pools provider connections.

## Notes
- One iteration cleanup: initial `seed_default_extensions` used a confusing `Arc::get_mut(&mut self.state.clone())` dead-code branch + `try_write` fallback. Refactored to populate `ManagerState` BEFORE wrapping it in `Arc<RwLock>`, eliminating the lock acquisition in `new()` entirely.
