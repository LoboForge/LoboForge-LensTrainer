#!/usr/bin/env bash
# Publish Sebastian + Jessica v2 LoRA to CivitAI (civitai.com).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }

[[ -f "${ROOT}/training.env" ]] || [[ -n "${CIVITAI_API_TOKEN:-}" ]] || [[ -n "${CIVITAI_TOKEN:-}" ]] || \
  die "Set CIVITAI_API_TOKEN or add it to training.env (see training.env.local.example)"

if [[ -f "${ROOT}/training.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/training.env"
  set +a
fi

python3.12 "${ROOT}/scripts/publish_civitai_lora.py" "$@"
