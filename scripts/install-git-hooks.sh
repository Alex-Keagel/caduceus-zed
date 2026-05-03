#!/usr/bin/env bash
# scripts/install-git-hooks.sh — opt this clone into the version-controlled hooks.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git config core.hooksPath .githooks
chmod +x .githooks/* scripts/ci-preflight.sh 2>/dev/null || true
echo "✓ core.hooksPath = .githooks"
echo "  Pre-push gate will now run scripts/ci-preflight.sh on every push."
echo "  Bypass with: git push --no-verify (true emergencies only)."
