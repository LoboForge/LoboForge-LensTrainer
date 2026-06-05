#!/usr/bin/env bash
# Build an explicit "python train.py ..." command from training.env (all flags visible).
# Prefer passing flags directly to train.py; use --env-file only if you want a file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/_trainer_env.sh"
activate_trainer_env "${ROOT}" || exit 1

ENV_FILE="${TRAINING_ENV_FILE:-${ROOT}/training.env}"
[[ -f "${ENV_FILE}" ]] || {
  printf '==> [error] Missing %s\n' "${ENV_FILE}" >&2
  printf '    Local:  cp training.env.local.example training.env\n' >&2
  printf '    RunPod: cp training.env.runpod.example training.env\n' >&2
  exit 1
}
# shellcheck disable=SC1091
source "${ENV_FILE}"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

DATASET_PATH="${DATASET_PATH:-}"
LORA_NAME="${LORA_NAME:-my-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${LORA_NAME}}"
TRAIN_PRESET="${TRAIN_PRESET:-configs/train_lora_lens_base_24gb.yaml}"
STEPS="${STEPS:-8000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
SAMPLE_EVERY="${SAMPLE_EVERY:-400}"
SAMPLE_EVERY_EARLY="${SAMPLE_EVERY_EARLY:-}"
SAMPLE_STEPS="${SAMPLE_STEPS:-}"
TRIGGER_WORD="${TRIGGER_WORD:-}"
MODEL_REPO="${MODEL_REPO:-${ROOT}/models/Lens-Base}"
[[ "${MODEL_REPO}" != /* ]] && MODEL_REPO="${ROOT}/${MODEL_REPO#./}"
DISABLE_MXFP4="${DISABLE_MXFP4:-true}"
RESOLUTION="${RESOLUTION:-0}"
BASELINE_CONTROL="${BASELINE_CONTROL:-false}"
RESUME_FROM="${RESUME_FROM:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"

[[ -n "${DATASET_PATH}" ]] || die "Set DATASET_PATH in training.env"
[[ -d "${DATASET_PATH}" ]] || die "DATASET_PATH not found: ${DATASET_PATH}"

PRESET="${TRAIN_PRESET}"
[[ "${PRESET}" != /* ]] && PRESET="${ROOT}/${PRESET}"
[[ -f "${PRESET}" ]] || die "TRAIN_PRESET not found: ${PRESET}"

EXTRA=()
[[ "${1:-}" == "--" ]] && { shift; EXTRA=("$@"); }

ARGS=(
  "${ROOT}/train.py" "${PRESET}"
  --dataset-path "${DATASET_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --job-name "${LORA_NAME}"
  --model-repo "${MODEL_REPO}"
  --steps "${STEPS}"
  --save-every "${SAVE_EVERY}"
  --sample-every "${SAMPLE_EVERY}"
  --resolution "${RESOLUTION}"
)
[[ -n "${SAMPLE_EVERY_EARLY}" ]] && ARGS+=(--sample-every-early "${SAMPLE_EVERY_EARLY}")
[[ -n "${SAMPLE_STEPS}" ]] && ARGS+=(--sample-steps "${SAMPLE_STEPS}")
[[ -n "${TRIGGER_WORD}" ]] && ARGS+=(--trigger-word "${TRIGGER_WORD}")
[[ "${DISABLE_MXFP4}" == "true" ]] && ARGS+=(--disable-mxfp4) || ARGS+=(--no-disable-mxfp4)
[[ "${BASELINE_CONTROL}" == "true" ]] && ARGS+=(--baseline-control) || ARGS+=(--no-baseline-control)
[[ -n "${RESUME_FROM}" ]] && ARGS+=(--resume "${RESUME_FROM}")
[[ -n "${LORA_RANK}" ]] && ARGS+=(--lora-rank "${LORA_RANK}")
[[ -n "${LORA_ALPHA}" ]] && ARGS+=(--lora-alpha "${LORA_ALPHA}")
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=("${EXTRA[@]}")

export DISABLE_MXFP4
bash "${ROOT}/scripts/verify_gpu_ready.sh" 2>/dev/null || true

log "Command:"
printf '  '; printf '%q ' "${ARGS[@]}"; printf '\n'
exec "${ARGS[@]}"
