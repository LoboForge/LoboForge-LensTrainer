#!/usr/bin/env bash
# RunPod / cloud GPU training — /workspace paths only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }

if [[ ! -f "${ROOT}/training.env" ]]; then
  die "Missing training.env — run: cp training.env.runpod.example training.env"
fi
if ! grep -qE '^(DATASET_PATH|MODEL_REPO)=/workspace' "${ROOT}/training.env" 2>/dev/null; then
  die "training.env is not a RunPod config. Use: cp training.env.runpod.example training.env"
fi

exec "${ROOT}/scripts/train.sh" "$@"
