#!/usr/bin/env bash
# Upload Sebastian + Jessica v2 LoRA to Hugging Face Model repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${HF_MODEL_ID:-LoboForge/lens-lora-sebastian-jessica-v2}"
OUT="${ROOT}/output/lens-lora-sebastian-jessica-v2"
CARD="${ROOT}/huggingface/lens-lora-sebastian-jessica-v2/README.md"
ASSETS="${ROOT}/docs/loras/assets/sebastian-jessica-v2"
LORA_STEP="${LORA_STEP:-5000}"
WEIGHTS="${LORA_WEIGHTS:-${OUT}/checkpoints/lora_step_$(printf '%06d' "${LORA_STEP}").safetensors}"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

command -v hf >/dev/null 2>&1 || die "Install hf CLI"
[[ -f "${WEIGHTS}" ]] || die "Missing weights: ${WEIGHTS}"
[[ -f "${CARD}" ]] || die "Missing ${CARD}"

log "Model target: https://huggingface.co/${MODEL_ID}"
log "Account: $(hf auth whoami 2>/dev/null | sed -n 's/user:[[:space:]]*//p')"

hf repo create "${MODEL_ID}" --repo-type model --exist-ok 2>/dev/null || true

log "Uploading weights (step ${LORA_STEP}) and config..."
hf upload "${MODEL_ID}" "${WEIGHTS}" lora_final.safetensors --repo-type model
hf upload "${MODEL_ID}" "${OUT}/config.resolved.json" config.resolved.json --repo-type model
hf upload "${MODEL_ID}" "${CARD}" README.md --repo-type model

log "Uploading sample images..."
for img in step_005000_lora_standing_forward_laughing.png step_005000_lora_beach_holding_hands.png; do
  src="${ASSETS}/${img}"
  [[ -f "${src}" ]] || src="${OUT}/samples/${img}"
  [[ -f "${src}" ]] || die "Missing sample ${img}"
  hf upload "${MODEL_ID}" "${src}" "samples/${img}" --repo-type model
done

log "Published: https://huggingface.co/${MODEL_ID}"
