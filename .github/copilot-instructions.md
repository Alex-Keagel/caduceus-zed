# Caduceus Development Rules

## Engine ↔ IDE Parity (MANDATORY)

Every engine capability that is developed or changed MUST also be developed and wired in the IDE side:

1. **Engine change → Bridge wiring → AgentTool → UI exposure**. No engine feature ships without all 4 layers.
2. If you add a method to `caduceus_bridge`, create or update the corresponding `caduceus_*_tool.rs` in `crates/agent/src/tools/`.
3. If the feature has a user-visible aspect, wire it into `crates/agent_ui/src/` (toolbar, panel, indicator, etc.).
4. The bridge at `crates/caduceus_bridge/src/` is the single source of truth for capabilities.

## Pre-Commit Checklist (MANDATORY)

Before every commit, run the build and all tests in BOTH projects:

```bash
# Engine tests
cd ~/Dev/caduceus && cargo test --workspace

# Bridge + IDE tests
cd ~/Dev/zed && cargo test -p caduceus_bridge
cd ~/Dev/zed && cargo check -p agent

# Release build (verify it links)
cd ~/Dev/zed && cargo build --release -p zed --features gpui_platform/runtime_shaders
```

Do NOT commit if any of these fail.

## Testing Requirements

Every complex feature MUST include:
- **Unit tests** in the crate where the feature is implemented
- **Integration tests** in `crates/caduceus_bridge/tests/` validating the bridge layer
- **Hardcore review** via the `iterate-hardcore` skill before merging significant changes

For non-trivial features (multi-file, architectural, new tools):
1. Write tests FIRST (TDD when possible)
2. Run the `iterate-hardcore` skill for scored review
3. All dimensions must reach 8+ before shipping

## Tool Registration Pattern

When adding a new AgentTool:
1. Create `crates/agent/src/tools/caduceus_{name}_tool.rs`
2. Add `mod` + `pub use` in `crates/agent/src/tools.rs`
3. Add to the `tools!` macro in the same file
4. Add import in `crates/agent/src/thread.rs` `use crate::{...}` block
5. Register via `self.add_tool(...)` in `add_default_tools()`
6. Add integration test in `crates/caduceus_bridge/tests/`

## Mode & Safety Invariants

- Read-only modes (Plan, Research, Architect, Review) MUST block write tools via privilege rings
- Kill switch MUST be accessible from both UI button and AgentTool
- Checkpoints MUST be created before destructive operations
- Auto-compact MUST trigger before context explosion (threshold: 40 messages)

## Project-local `private/` files — read open, write requires grant + path

The repo's top-level `private/` directory holds audits, reviews, scratch notes, and reviewer/critique artifacts. The convention is **profile-agnostic** (plan, research, act, autopilot, custom modes alike) and is enforced engine-side as an envelope invariant — every preset in `caduceus-permissions` allows reading `private/**`.

**Read access — always permitted.** Any agent may read freely from `private/**`. No prompt, no grant flow.

**Write access — never silent.** Before writing, creating, or modifying any file under `private/`, the agent MUST do BOTH in a single `ask_user`:

1. **Ask permission to write** — state intent (artifact kind, producing workflow, why `private/` vs. session workspace).
2. **Ask where to write** — propose a target subpath (e.g. `private/audits/<slug>-<date>.md`) AND let the user override slug, subdirectory, or filename.

Applies to every write path — `create`, `edit`, `bash` redirects, `git add` of newly authored `private/**` files. **Exception:** when the user explicitly hands the agent a specific `private/` file to edit, no re-prompt for that file in that turn.

See `~/Dev/.github/copilot-instructions.md` for the cross-repo source of truth.

## Local CI gate — pre-commit + pre-push

Every commit and every push runs the same gates GitHub Actions runs, **before** the change goes out. The CI workflow at `.github/workflows/ci.yml` (`Caduceus CI`) takes ~42 minutes per run; catching a `clippy -D warnings`, `cargo fmt`, or test failure locally avoids that round-trip.

**Once per fresh clone:**

```bash
scripts/install-git-hooks.sh          # sets core.hooksPath = .githooks
```

After that, every `git commit` and `git push` runs the version-controlled hooks under `.githooks/`.

### `pre-commit` — fast gate, fires every commit

Runs (skipped automatically on docs-only diffs):

1. `cargo fmt --all --check`
2. `cargo check -p agent -p agent_ui -p caduceus_bridge`
   *(scope-limited; full-workspace check still runs at pre-push time.)*

### `pre-push` — full CI mirror, fires every push

Runs (skipped automatically on docs-only diffs vs `origin/main`):

1. Verify the sibling `../caduceus` checkout exists (path deps require it).
2. `cargo check -p agent -p agent_ui`
3. `cargo test -p caduceus_bridge --lib --tests`
4. `cargo test -p agent_ui --lib`
5. `cargo clippy --workspace --all-targets -- -D warnings`
6. `cargo check --release -p zed --features gpui_platform/runtime_shaders`

**On-demand invocation** (no commit/push needed):

```bash
scripts/ci-preflight.sh               # full gate
scripts/ci-preflight.sh --fast        # skip the slow release-build check
```

**Skip rules:** both hooks auto-skip the rust gate when the staged/pushed diff is **only** `*.md` / `docs/` / `private/` / `.github/` / `.githooks/` / `scripts/` files.

**Bypass for one commit/push:** `git commit --no-verify` / `git push --no-verify` — true emergencies only. CI will still catch a regression after the fact, but the round-trip cost is high.

**Why this rule exists:** the `.max(0)` clippy regression that broke run `25276602408` got merged because the previous local hook didn't run clippy and the user `--no-verify`'d a narrow-scope push. The new hooks prevent that exact failure mode by running fmt+check on every commit and the full clippy gate on every push.

## Architecture

- **Engine**: `~/Dev/caduceus` — 14 Rust crates, source of truth for capabilities
- **Bridge**: `~/Dev/zed/crates/caduceus_bridge/` — wires engine to IDE
- **AgentTools**: `~/Dev/zed/crates/agent/src/tools/caduceus_*.rs` — LLM-callable tools
- **UI**: `~/Dev/zed/crates/agent_ui/src/` — user-facing panels, buttons, indicators
- **Only provider**: GitHub Copilot Chat (all others disabled)
- **No MCP**: all tools use direct Rust calls, no JSON-RPC overhead
