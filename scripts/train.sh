#!/usr/bin/env bash
# =============================================================================
# LensTrainer-LoboForge — ONE training script (run after bootstrap.sh).
#
#   nano training.env    # DATASET_PATH, LORA_NAME, STEPS, TRIGGER_WORD, ...
#   bash scripts/train.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/_trainer_env.sh"
activate_trainer_env "${ROOT}" || exit 1

if [[ ! -f "${ROOT}/training.env" ]]; then
  die "Missing training.env — local: cp training.env.example training.env | RunPod: cp training.env.runpod.example training.env"
fi
# shellcheck disable=SC1091
source "${ROOT}/training.env"

DATASET_PATH="${DATASET_PATH:-}"
LORA_NAME="${LORA_NAME:-my-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/${LORA_NAME}}"
TRAIN_PRESET="${TRAIN_PRESET:-configs/train_lora_lens_base_24gb.yaml}"
STEPS="${STEPS:-8000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
SAMPLE_EVERY="${SAMPLE_EVERY:-400}"
TRIGGER_WORD="${TRIGGER_WORD:-mytrigger}"
MODEL_REPO="${MODEL_REPO:-${ROOT}/models/Lens-Base}"
[[ "${MODEL_REPO}" != /* ]] && MODEL_REPO="${ROOT}/${MODEL_REPO#./}"
DISABLE_MXFP4="${DISABLE_MXFP4:-false}"
RESOLUTION="${RESOLUTION:-0}"
BASELINE_CONTROL="${BASELINE_CONTROL:-true}"
RESUME_FROM="${RESUME_FROM:-}"
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-}"

die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }
log() { printf '==> %s\n' "$*"; }

EXTRA_ARGS=()
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && {
  sed -n '2,8p' "$0"
  echo "Edit training.env — see training.env.example"
  echo "Extra overrides: bash scripts/train.sh --set model.disable_mxfp4=true   # 16GB / no MXFP4"
  exit 0
}
if [[ $# -gt 0 ]]; then
  [[ "${1}" == "--" ]] && shift
  EXTRA_ARGS=("$@")
fi

[[ -n "${DATASET_PATH}" ]] || die "Set DATASET_PATH in training.env"
[[ -d "${DATASET_PATH}" ]] || die "DATASET_PATH not found: ${DATASET_PATH} (fix training.env — local paths, not /workspace)"
[[ -f "${ROOT}/${TRAIN_PRESET}" ]] || die "TRAIN_PRESET not found: ${TRAIN_PRESET}"
if [[ "${MODEL_REPO}" == microsoft/* ]]; then
  :
elif python "${ROOT}/scripts/assemble_lens_repo.py" --output "${MODEL_REPO}" --check 2>/dev/null; then
  :
else
  die "MODEL_REPO incomplete (${MODEL_REPO}) — run: bash scripts/bootstrap.sh"
fi

PRESET="${ROOT}/${TRAIN_PRESET}"
[[ "${TRAIN_PRESET}" == /* ]] && PRESET="${TRAIN_PRESET}"

export DISABLE_MXFP4
bash "${ROOT}/scripts/verify_gpu_ready.sh"

log "Starting LoRA training: ${LORA_NAME}"
log "  dataset  ${DATASET_PATH}"
log "  output   ${OUTPUT_DIR}"
log "  steps    ${STEPS}"
log "  model    ${MODEL_REPO}"
log "  preset   ${TRAIN_PRESET}"
log "  mxfp4    disable=${DISABLE_MXFP4} (false=GPU MXFP4 cache; 16GB-only: true=CPU cache)"

ARGS=(
  "${ROOT}/train.py" "${PRESET}"
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
[[ -n "${LORA_RANK}" ]] && ARGS+=(--set "lora.rank=${LORA_RANK}")
[[ -n "${LORA_ALPHA}" ]] && ARGS+=(--set "lora.alpha=${LORA_ALPHA}")
[[ -n "${RESUME_FROM}" ]] && ARGS+=(--set "train.resume_from=${RESUME_FROM}" --set "sample.baseline_control=false")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && ARGS+=("${EXTRA_ARGS[@]}")

exec python "${ARGS[@]}"
