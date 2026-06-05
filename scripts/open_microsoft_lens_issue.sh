#!/usr/bin/env bash
# Open a show-and-tell issue on microsoft/Lens (requires: gh auth login).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BODY="${ROOT}/docs/OUTREACH-microsoft-lens-issue-body.md"
TITLE="Community: config-driven Lens-Base LoRA trainer (flow-match, Comfy export) — working on 24GB"

command -v gh >/dev/null 2>&1 || { echo "Install gh: https://cli.github.com/"; exit 1; }
[[ -f "${BODY}" ]] || { echo "Missing ${BODY}"; exit 1; }

gh auth status >/dev/null 2>&1 || {
  echo "Run: gh auth login"
  exit 1
}

gh issue create \
  --repo microsoft/Lens \
  --title "${TITLE}" \
  --body-file "${BODY}"
