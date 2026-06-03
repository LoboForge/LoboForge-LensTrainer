#!/usr/bin/env bash
# Local workstation training — never use RunPod /workspace paths here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }

if [[ ! -f "${ROOT}/training.env" ]]; then
  die "Missing training.env — run: cp training.env.local.example training.env"
fi
if grep -qE '^(DATASET_PATH|OUTPUT_DIR|MODEL_REPO)=/workspace' "${ROOT}/training.env" 2>/dev/null; then
  die "training.env has RunPod /workspace paths. Reset: cp training.env.local.example training.env"
fi
if grep -q 'train_runpod_gpu.yaml' "${ROOT}/training.env" 2>/dev/null; then
  die "training.env uses RunPod preset. Use TRAIN_PRESET=configs/train_lora_dual_character_24gb.yaml"
fi

exec "${ROOT}/scripts/train.sh" "$@"
