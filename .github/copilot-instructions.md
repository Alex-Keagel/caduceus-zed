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

## Architecture

- **Engine**: `~/Dev/caduceus` — 14 Rust crates, source of truth for capabilities
- **Bridge**: `~/Dev/zed/crates/caduceus_bridge/` — wires engine to IDE
- **AgentTools**: `~/Dev/zed/crates/agent/src/tools/caduceus_*.rs` — LLM-callable tools
- **UI**: `~/Dev/zed/crates/agent_ui/src/` — user-facing panels, buttons, indicators
- **Only provider**: GitHub Copilot Chat (all others disabled)
- **No MCP**: all tools use direct Rust calls, no JSON-RPC overhead
