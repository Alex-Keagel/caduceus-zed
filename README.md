# 🐍 Caduceus

AI-powered code editor built on [Zed](https://github.com/zed-industries/zed) with a 14-crate Rust AI engine.

## What is this

Caduceus = Zed + a custom AI engine. Zed handles the editor (GPU rendering, Tree-sitter, LSP). Our engine provides 13 agent tools via MCP that any AI model can use.

## Quick Start

```bash
# Install the MCP server
cargo install --path ../caduceus-mcp-server

# Build Caduceus (needs Xcode on macOS)
./build-caduceus.sh

# Or just use stock Zed with our tools
brew install zed
# Tools auto-connect via ~/.config/zed/settings.json
```

## Architecture

```
┌─────────────────────────────────┐
│ Caduceus (Zed fork)             │
│  • GPUI — native GPU rendering  │
│  • Tree-sitter — syntax         │
│  • LSP — code intelligence      │
│  • Agent panel — AI chat        │
└────────────┬────────────────────┘
             │ MCP (stdio)
┌────────────▼────────────────────┐
│ caduceus-mcp-server (7.3MB)     │
│  13 tools from 14 Rust crates   │
│  bash, read/write/edit files,   │
│  grep, glob, tree, git, web     │
└─────────────────────────────────┘
```

## Tools

| Tool | What it does |
|------|-------------|
| `bash` | Execute shell commands |
| `read_file` | Read file contents |
| `write_file` | Create/overwrite files |
| `edit_file` | Surgical string replacement |
| `grep_search` | ripgrep content search |
| `glob_search` | Find files by pattern |
| `list_files` | List directory entries |
| `tree` | Directory tree view |
| `git_status` | Git working tree status |
| `git_diff` | Git diff output |
| `think` | Agent reasoning step |
| `web_search` | Search the web |
| `web_fetch` | Fetch URL content |

## Engine Crates

The AI engine lives in [caduceus](https://github.com/Alex-Keagel/caduceus) — 14 Rust crates, 1,121 tests:

`core` · `orchestrator` · `providers` · `tools` · `git` · `omniscience` · `telemetry` · `scanner` · `storage` · `permissions` · `crdt` · `runtime` · `mcp` · `marketplace`

## Building

Requires: Rust, cmake, Xcode (macOS)

```bash
git clone https://github.com/Alex-Keagel/caduceus-zed.git
cd caduceus-zed
./build-caduceus.sh
```

## License

GPL-3.0 (inherited from Zed). Engine crates are MIT.

## Credits

Built on [Zed](https://zed.dev) by Zed Industries.