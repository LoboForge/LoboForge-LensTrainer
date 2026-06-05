#!/usr/bin/env bash
# Publish the static docs Space to Hugging Face (trainer landing page).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_ID="${HF_SPACE_ID:-LoboForge/LoboForge-LensTrainer}"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

command -v hf >/dev/null 2>&1 || die "Install hf CLI: curl -LsSf https://hf.co/cli/install.sh | bash"

[[ -f "${ROOT}/huggingface/README.md" ]] || die "Missing ${ROOT}/huggingface/README.md"
[[ -f "${ROOT}/huggingface/index.html" ]] || die "Missing ${ROOT}/huggingface/index.html"

log "Space target: https://huggingface.co/spaces/${SPACE_ID}"
log "Account: $(hf auth whoami 2>/dev/null | sed -n 's/user:[[:space:]]*//p')"

hf repo create "${SPACE_ID}" --repo-type space --space_sdk static --exist-ok 2>/dev/null || true

log "Uploading README.md and index.html..."
hf upload "${SPACE_ID}" "${ROOT}/huggingface/README.md" README.md --repo-type space
hf upload "${SPACE_ID}" "${ROOT}/huggingface/index.html" index.html --repo-type space

log "Published: https://huggingface.co/spaces/${SPACE_ID}"
