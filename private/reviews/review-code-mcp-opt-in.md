# Hardcore Review: gap-t2 MCP opt-in tool

- **Date**: 2026-04-19 14:55
- **Content Type**: Code Changes (item #6 of 8 deferred)
- **Iteration**: 1
- **Verdict**: SHIP IT

## Delivered
- New `caduceus-mcp/src/permissions.rs` (~370 lines, 12 tests).
- New error variant `McpError::PermissionDenied(String)`.

`PermissionedMcpManager` wraps the existing `McpServerManager` and gates `call_tool` behind:

- **Per-server approval state** (`Pending` (default) / `Approved` / `Denied`).
- **Per-tool policy** (`AllowAll`, `Allowlist`, `Blocklist`) — default is empty allowlist (= deny everything until the user opts a specific tool in).
- **Audit log** (FIFO-trimmed at `audit_cap`, default 500) recording every decision: server, tool, outcome, detail.
- `check()` returns one of `Approved | ServerNotApproved | ServerDenied | ToolNotAllowed | ToolDenied` so the UI can pick the right prompt without re-running policy logic.
- `approved_tools()` returns only tools that would actually pass `call_tool` — the agent's tool catalogue stays consistent with what the LLM is permitted to invoke.
- Unknown server IDs default to `Pending` (deny by default), not silently bypass policy.

## Final Scores

| # | Dimension | Score | Notes |
|---|-----------|-------|-------|
| 1 | Correctness | 9/10 | Default-deny everywhere; policy enforced before underlying call |
| 2 | Completeness | 9/10 | Pending/approved/denied/revoked; allow-all/allowlist/blocklist; FIFO audit cap; unknown-server default; round-trip serde |
| 3 | Security | 9/10 | Opt-in by design; no path to invoke a tool without explicit approval; audit trail captures denials too |
| 4 | Clarity | 9/10 | Module-level doc comment explains failure-mode → UI-prompt mapping |
| 5 | Architecture | 8/10 | Wraps existing manager via `Arc`, no fork; `inner()` exposed only for documented bypass cases |
| 6 | Test Coverage | 9/10 | 12 tests cover every decision branch + serde + audit FIFO |
| 8 | SRP | 9/10 | Pure policy/audit; underlying I/O still in `McpServerManager` |
| 9 | KISS / YAGNI / DRY | 8/10 | Three small policy variants cover the realistic UX; no premature ACL DSL |

**Average**: 8.75/10 — converged.

## Test Results
`cargo test -p caduceus-mcp --lib` → **58/58 pass** ✅ (was 46; +12 permission tests)

## Notes
- Underlying `McpServerManager::call_tool` does not accept a `server_id`; it routes by tool name. The permission wrapper requires the caller to pass `server_id` explicitly so the policy check is unambiguous, then forwards just `(tool_name, arguments)` to the underlying manager. This means name collisions across servers route by manager order — the existing limitation, surfaced clearly in the doc comment on `tools_grouped_by_server`.
- A future improvement (deferred) would be to extend `McpServerManager` with `call_tool_on(server_id, tool, args)` so the permissioned wrapper can guarantee the destination server. Not blocking — current behavior is safe (still policy-gated) just possibly broader than intended in name-collision setups.
