# Caduceus IDE — Engine vs. UI Gap Audit

## Summary

- **37** caduceus tool files examined
- **16** bridge modules examined
- **8** UI indicators exist in `agent_panel.rs` (panel-level badges)
- **4** UI elements exist in `thread_view.rs` (checkpoint, mode safety, token usage, cost)
- **1** UI control in `mode_selector.rs` (7-mode selector)
- **≈27 tools + ≈10 bridge capabilities have NO frontend exposure**

---

## Full Capability Matrix

| # | Engine Capability | Has Tool? | Has UI Control? | Has UI Indicator? | Gap |
|---|---|---|---|---|---|
| 1 | **API Registry** — catalog OpenAPI/gRPC/GraphQL schemas | ✅ | ❌ | ✅ badge `🔌 N APIs` | No interactive API browser |
| 2 | **Architect** — Mermaid diagrams, health scoring | ✅ | ❌ | ✅ badge `❤️ N%` | No diagram viewer |
| 3 | **Automations** — cron/file-watch triggers | ✅ | ❌ | ✅ badge `⚡ N` | Count only — no create/edit/toggle UI |
| 4 | **Background Agent** — spawn/list/stop agents | ✅ | ❌ | ✅ badge `🤖 N` | Count only — no management panel |
| 5 | **Checkpoint** — create/restore/diff snapshots | ✅ | ✅ restore | ✅ count + status | **Adequate** |
| 6 | **Code Graph** — symbol neighbors, impact analysis | ✅ | ❌ | ❌ | **No UI at all** |
| 7 | **Conversation** — serialize/compact/extract | ✅ | ❌ | ❌ | **No UI at all** (internal) |
| 8 | **Cross Git** — multi-repo branches/commits | ✅ | ❌ | ❌ | **No UI at all** |
| 9 | **Cross Search** — federated search across repos | ✅ | ❌ | ❌ | **No UI at all** |
| 10 | **Dependency Scan** — CVE/vulnerability audit | ✅ | ❌ | ❌ | **No UI at all** |
| 11 | **Error Analysis** — classify errors, suggest fixes | ✅ | ❌ | ❌ | **No UI at all** |
| 12 | **File Lock** — advisory lock for `.caduceus/` | ✅ | ❌ | ❌ | Internal — acceptable |
| 13 | **Git Read** — status, diffs, freshness | ✅ | ❌ | ❌ | **No UI at all** |
| 14 | **Git Write** — stage, commit, branch | ✅ | ❌ | ❌ | **No UI at all** |
| 15 | **Index** — build semantic search index | ✅ | ❌ | ❌ | **No UI at all** |
| 16 | **Kanban** — multi-agent task board | ✅ | ❌ | ❌ | **No UI at all** |
| 17 | **Kill Switch** — emergency stop | ✅ | ✅ button | ❌ | **Adequate** |
| 18 | **Marketplace** — skill browse/evolve | ✅ | ❌ | ✅ badge `🧬 N` | Count only — no browse/install UI |
| 19 | **MCP Security** — scan MCP servers | ✅ | ❌ | ❌ | **No UI at all** |
| 20 | **Memory Read** — retrieve project memory | ✅ | ❌ | ❌ | **No UI at all** |
| 21 | **Memory Write** — store project memory | ✅ | ❌ | ❌ | **No UI at all** |
| 22 | **Mode Request** — switch 7 modes | ✅ | ✅ selector | ✅ safety indicator | **Adequate** |
| 23 | **Policy** — permissions, audit, compliance | ✅ | ❌ | ❌ | **No UI at all** |
| 24 | **PRD** — parse requirements into tasks | ✅ | ❌ | ❌ | **No UI at all** |
| 25 | **Product** — feature tracking, milestones | ✅ | ❌ | ❌ | **No UI at all** |
| 26 | **Progress** — infer progress from commits/tests | ✅ | ❌ | ❌ | **No UI at all** |
| 27 | **Project** — manage multi-repo project.json | ✅ | ❌ | ✅ badge `📦 name (N)` | Name/count only — no editor |
| 28 | **Project Wiki** — file-based wiki | ✅ | ❌ | ❌ | **No UI at all** |
| 29 | **Scaffold** — code generation templates | ✅ | ❌ | ❌ | **No UI at all** |
| 30 | **Security Scan** — OWASP/secrets scanning | ✅ | ❌ | ✅ badge `🛡️ N%` | Score only — no findings panel |
| 31 | **Semantic Search** — conceptual code search | ✅ | ❌ | ❌ | **No UI at all** |
| 32 | **Storage** — tasks, snapshots, audit | ✅ | ❌ | ❌ | **No UI at all** |
| 33 | **Task Tree** — hierarchical task management | ✅ | ❌ | ❌ | **No UI at all** |
| 34 | **Telemetry** — token costs, budget, drift, SLOs | ✅ | ❌ | ⚠️ partial (token % + beta cost) | Budget/SLO/drift NOT surfaced |
| 35 | **Time Tracking** — velocity, overdue detection | ✅ | ❌ | ❌ | **No UI at all** |
| 36 | **Tree Sitter** — AST symbol extraction | ✅ | ❌ | ❌ | **No UI at all** |
| 37 | **Wiki** — read/write wiki via storage bridge | ✅ | ❌ | ❌ | **No UI at all** |

### Bridge-Only Capabilities (no tool file, engine-internal)

| # | Bridge Capability | Source Module | Has UI? | Gap |
|---|---|---|---|---|
| 38 | **LoopDetector** — detects tool call loops | `safety.rs` | ❌ | No loop-stuck warning |
| 39 | **CircuitBreaker** — trips after N failures | `safety.rs` | ❌ | No tripped-state indicator |
| 40 | **CompactionCooldown** — prevents double-compact | `safety.rs` | ❌ | Internal — acceptable |
| 41 | **Context Zones** (Green→Critical) | `thread.rs` | ❌ | Zone name/color NOT shown; only token % |
| 42 | **Budget Limits** — set/check spending caps | `telemetry.rs` | ❌ | No budget bar or alert |
| 43 | **SLO Monitoring** — latency/quality targets | `telemetry.rs` | ❌ | No SLO dashboard |
| 44 | **Drift Detection** — behavioral drift scoring | `telemetry.rs` | ❌ | No drift warning |
| 45 | **Trust Scoring** — per-agent trust levels | `security.rs` | ❌ | No trust display |
| 46 | **E2b Snapshots/Volumes** — sandbox management | `runtime.rs` | ❌ | No snapshot UI |
| 47 | **CRDT Buffers** — collaborative editing | `crdt.rs` | ❌ | No multi-cursor / collab UI |
| 48 | **Provider Registry** — LLM provider management | `providers.rs` | ❌ | No provider switcher |

---

## TOP 10 GAPS (by severity)

### CRITICAL

| # | Gap | Description |
|---|---|---|
| **1** | **CircuitBreaker / LoopDetector invisible** | Safety systems can trip silently — user sees the agent "stop" with no explanation. Need a status bar indicator showing `🔴 Circuit breaker tripped (5 consecutive failures)` or `🔄 Loop detected: tool X called 8 times`. |
| **2** | **Context Zone indicator missing** | Engine has 5 zones (Green→Critical) with auto-compaction, but UI only shows token %. Need a colored zone badge (`🟢 Green 42%` → `🔴 Critical 97%`) so users know when compaction fired and why messages disappeared. |
| **3** | **Budget / Cost tracking not surfaced** | TelemetryBridge tracks cumulative costs, budget limits, and budget-remaining — none shown. A `$2.47 / $10.00` budget bar would prevent bill shock, especially in Autopilot mode. |

### HIGH

| # | Gap | Description |
|---|---|---|
| **4** | **Kanban board has no visual board** | Tool manages cards across 4 columns with dependency chains, but there's zero visual representation. Need a mini-board sidebar or `/kanban` panel showing Backlog→In Progress→Review→Done columns. |
| **5** | **Task Tree has no visualization** | Hierarchical tasks exist in engine but have no tree view. The tool generates Mermaid but there's no renderer. Need an expandable tree panel or inline Mermaid rendering. |
| **6** | **Memory has no browse/manage UI** | Project memory (persistent key-value pairs across sessions) is invisible. Users can't see what the agent "remembers." Need a Memory panel listing stored keys with view/edit/delete. |
| **7** | **Dependency vulnerabilities not surfaced** | Scan tool finds CVEs but results exist only in chat responses. Need a `⚠️ 3 vulnerabilities` badge in the panel indicators with a click-to-view findings list. |

### MEDIUM

| # | Gap | Description |
|---|---|---|
| **8** | **Progress / Time Tracking invisible** | Engine infers progress from commits/tests and tracks velocity, but nothing is shown. Need a progress bar on active tasks and a velocity indicator for estimation accuracy. |
| **9** | **Search Index status unknown** | Semantic search requires indexing but users can't tell if the index is populated, stale, or empty. Need an `📇 Indexed: 1,247 chunks` indicator or `⚠️ Index empty — run /index`. |
| **10** | **Scaffold template picker missing** | 29+ scaffold templates exist but users must know names. Need a `/scaffold` command with autocomplete or a template picker dropdown showing available templates by category. |

---

## What IS Working (UI-Engine Parity)

| Feature | UI Element | Location |
|---|---|---|
| Mode switching (7 modes) | Dropdown selector + safety badge | `mode_selector.rs`, `thread_view.rs` |
| Checkpoints | Count badge + restore button | `thread_view.rs` |
| Kill switch | Emergency stop button | `agent_panel.rs` |
| Token usage | Progress circle + used/max | `thread_view.rs` |
| Token limit warnings | Warning/Exceeded callouts | `thread_view.rs` |
| Security score | `🛡️ N%` badge (color-coded) | `agent_panel.rs` |
| Project info | `📦 name (N repos)` badge | `agent_panel.rs` |
| Automation count | `⚡ N` badge | `agent_panel.rs` |
| Background agent count | `🤖 N` badge | `agent_panel.rs` |
| Evolved skills count | `🧬 N` badge | `agent_panel.rs` |
| API count | `🔌 N APIs` badge | `agent_panel.rs` |
| Health score | `❤️ N%` badge (color-coded) | `agent_panel.rs` |
| Per-turn cost | `$X.XX` label (beta flag) | `thread_view.rs` |
| Slash commands | `/compact`, `/checkpoint`, `/mode`, `/context` | `agent.rs` |
