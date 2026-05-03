#!/usr/bin/env bash
# scripts/ci-preflight.sh — run the same gates GitHub Actions runs, locally.
#
# Mirrors .github/workflows/ci.yml exactly so a green local run = a green CI run.
# Designed to be invoked by .githooks/pre-push, but is also fine to run on demand:
#
#     scripts/ci-preflight.sh           # full gate (matches CI)
#     scripts/ci-preflight.sh --fast    # skip the slow release check
#     scripts/ci-preflight.sh --skip-rust-when-docs-only
#         # auto-skip if only docs/specs/private/markdown changed since origin/main
#
# Bypass entirely with `git push --no-verify` only in true emergencies.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

FAST=0
SKIP_DOCS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --skip-rust-when-docs-only) SKIP_DOCS_ONLY=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ── Optional fast-path: skip the rust gate if every changed file matches
# a docs-only globbed path. Tracked-but-uncommitted changes don't count
# (the hook is invoked at push time, so we look at HEAD..origin/main).
if [ "$SKIP_DOCS_ONLY" = "1" ]; then
  base="$(git merge-base HEAD origin/main 2>/dev/null || echo '')"
  if [ -n "$base" ]; then
    nonmarkdown=$(git diff --name-only "$base"..HEAD | grep -Ev '\.md$|^docs/|^private/|^\.github/' || true)
    if [ -z "$nonmarkdown" ]; then
      echo "✓ ci-preflight: docs/markdown-only diff vs origin/main; skipping rust gate."
      exit 0
    fi
  fi
fi

# ── Sibling-repo invariant. caduceus-zed has 14 path deps on
# ../caduceus/crates/* and won't compile without it. The CI workflow
# clones it; locally we expect the user has the sibling checkout.
if [ ! -d "../caduceus" ]; then
  echo "✗ ci-preflight: expected sibling repo at $(cd .. && pwd)/caduceus" >&2
  echo "  Clone it:  git -C $(cd .. && pwd) clone https://github.com/Alex-Keagel/caduceus.git" >&2
  exit 1
fi

step() { echo; echo "▶ ci-preflight: $1"; }

step "cargo check -p agent -p agent_ui"
cargo check -p agent -p agent_ui

step "cargo test -p caduceus_bridge --lib --tests"
cargo test -p caduceus_bridge --lib --tests

step "cargo test -p agent_ui --lib"
cargo test -p agent_ui --lib

step "cargo clippy --workspace --all-targets -- -D warnings"
cargo clippy --workspace --all-targets -- -D warnings

if [ "$FAST" = "1" ]; then
  echo
  echo "⚠ ci-preflight: --fast set; skipping release-build check."
  echo "  CI will still run it; a failure there will block merge."
else
  step "cargo check --release -p zed --features gpui_platform/runtime_shaders"
  cargo check --release -p zed --features gpui_platform/runtime_shaders
fi

echo
echo "✓ ci-preflight: all CI gates green."
