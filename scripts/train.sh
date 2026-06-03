#!/usr/bin/env bash
# LensTrainer-LoboForge — run training (after bootstrap.sh).
#
# Configure in training.env (recommended) or export variables before running:
#   cp training.env.example training.env   # edit DATASET_PATH, LORA_NAME, STEPS, ...
#   bash scripts/train.sh
#
# Extra train.py overrides after --:
#   bash scripts/train.sh -- --set lora.rank=8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/runpod_env.sh"

if [[ -f "${ROOT}/training.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/training.env"
fi

# ---------------------------------------------------------------------------
# Defaults (training.env or env vars override these)
# ---------------------------------------------------------------------------
DATASET_PATH="${DATASET_PATH:-}"
LORA_NAME="${LORA_NAME:-my-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${JOB_OUTPUT_DIR:-./output/${LORA_NAME}}}"
TRAIN_PRESET="${TRAIN_PRESET:-${TRAIN_CONFIG:-configs/train_lora_lens_base_24gb.yaml}}"
STEPS="${STEPS:-8000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
SAMPLE_EVERY="${SAMPLE_EVERY:-400}"
TRIGGER_WORD="${TRIGGER_WORD:-${SAMPLE_TRIGGER:-mytrigger}}"
MODEL_REPO="${MODEL_REPO:-microsoft/Lens-Base}"
DISABLE_MXFP4="${DISABLE_MXFP4:-true}"
RESOLUTION="${RESOLUTION:-0}"
BASELINE_CONTROL="${BASELINE_CONTROL:-true}"
RESUME_FROM="${RESUME_FROM:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

usage() {
  cat <<EOF
Usage: bash scripts/train.sh [-- extra train.py args]

Edit ${ROOT}/training.env (see training.env.example):

  DATASET_PATH   folder of images + sidecar .txt captions (required)
  LORA_NAME      short name for this run (used in job.name)
  OUTPUT_DIR     where checkpoints and lora_final are written
  STEPS          training steps
  TRIGGER_WORD   token for [trigger] in sample prompts (empty if captions are full sentences)
  TRAIN_PRESET   yaml under configs/ (architecture + sample prompts)
  MODEL_REPO     microsoft/Lens-Base or path to local HF folder

Optional: RESUME_FROM=latest, BASELINE_CONTROL=false, LORA_RANK, LORA_ALPHA, RESOLUTION=0

Bootstrap first:  curl -fsSL .../scripts/bootstrap.sh | bash
EOF
}

EXTRA_ARGS=()
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

[[ -n "${DATASET_PATH}" ]] || die "Set DATASET_PATH in training.env (dataset folder with image + .txt pairs)"
[[ -d "${DATASET_PATH}" ]] || die "DATASET_PATH not found: ${DATASET_PATH}"
[[ -f "${ROOT}/${TRAIN_PRESET}" ]] || die "TRAIN_PRESET not found: ${TRAIN_PRESET}"

PRESET_PATH="${ROOT}/${TRAIN_PRESET}"
if [[ "${TRAIN_PRESET}" != /* ]]; then
  PRESET_PATH="${ROOT}/${TRAIN_PRESET}"
fi

log "Training: ${LORA_NAME}"
log "  dataset:  ${DATASET_PATH}"
log "  output:   ${OUTPUT_DIR}"
log "  steps:    ${STEPS}"
log "  preset:   ${TRAIN_PRESET}"
log "  model:    ${MODEL_REPO}"
[[ -n "${TRIGGER_WORD}" ]] && log "  trigger:  ${TRIGGER_WORD}"

ARGS=(
  "${ROOT}/train.py"
  "${PRESET_PATH}"
  --set "job.name=${LORA_NAME}"
  --set "job.output_dir=${OUTPUT_DIR}"
  --set "dataset.folder_path=${DATASET_PATH}"
  --set "train.steps=${STEPS}"
  --set "train.save_every=${SAVE_EVERY}"
  --set "train.sample_every=${SAMPLE_EVERY}"
  --set "model.repo_id=${MODEL_REPO}"
  --set "model.disable_mxfp4=${DISABLE_MXFP4}"
  --set "dataset.resolution=${RESOLUTION}"
  --set "sample.trigger_word=${TRIGGER_WORD}"
  --set "sample.baseline_control=${BASELINE_CONTROL}"
)

if [[ -n "${LORA_RANK}" ]]; then
  ARGS+=(--set "lora.rank=${LORA_RANK}")
fi
if [[ -n "${LORA_ALPHA}" ]]; then
  ARGS+=(--set "lora.alpha=${LORA_ALPHA}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  ARGS+=(--set "train.resume_from=${RESUME_FROM}")
  ARGS+=(--set "sample.baseline_control=false")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  ARGS+=("${EXTRA_ARGS[@]}")
fi

exec python "${ARGS[@]}"
