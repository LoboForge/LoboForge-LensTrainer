#!/usr/bin/env bash
# LensTrainer-LoboForge — one-shot bootstrap (RunPod, vast.ai, bare Linux GPU).
#
#   curl -fsSL https://raw.githubusercontent.com/LoboForge/LoboForge-LensTrainer/main/scripts/bootstrap.sh | bash
#
# From an existing clone:
#   bash scripts/bootstrap.sh
#
# Optional environment (all optional except dataset when auto-starting train):
#   LOBFORGE_TRAINER_DIR   install path (default: /workspace/... on RunPod, else ~/LoboForge-LensTrainer)
#   HF_TOKEN               Hugging Face token (hf auth + hub downloads)
#   SKIP_APT=1             skip apt packages
#   SKIP_SMOKE_TEST=1      skip GPU/import smoke test
#   START_TRAIN=1          run training when DATASET_PATH is set
#   DATASET_PATH           image folder with sidecar .txt captions
#   TRAIN_CONFIG           yaml preset (default: configs/train_lora_lens_base_24gb.yaml)
#   JOB_OUTPUT_DIR         output dir (default: ./output/lens-lora-run)
#   MODEL_REPO             HF id or path (default: microsoft/Lens-Base)
#   SAMPLE_TRIGGER         sample.trigger_word override
set -euo pipefail

REPO_URL="${LOBFORGE_TRAINER_REPO:-https://github.com/LoboForge/LoboForge-LensTrainer.git}"
PYTHON="${PYTHON:-python3}"
MIN_PYTHON=3.11
SKIP_APT="${SKIP_APT:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
START_TRAIN="${START_TRAIN:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_lora_lens_base_24gb.yaml}"
MODEL_REPO="${MODEL_REPO:-microsoft/Lens-Base}"
JOB_OUTPUT_DIR="${JOB_OUTPUT_DIR:-./output/lens-lora-run}"

default_install_dir() {
  if [[ -n "${LOBFORGE_TRAINER_DIR:-}" ]]; then
    printf '%s' "${LOBFORGE_TRAINER_DIR}"
    return 0
  fi
  if [[ -d /workspace ]] && [[ -w /workspace ]]; then
    printf '/workspace/LoboForge-LensTrainer'
    return 0
  fi
  printf '%s/LoboForge-LensTrainer' "${HOME}"
}

INSTALL_DIR="$(default_install_dir)"

log() { printf '==> %s\n' "$*"; }
warn() { printf '==> [warn] %s\n' "$*" >&2; }
die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }

run_apt() {
  if [[ "${SKIP_APT}" == "1" ]]; then
    warn "SKIP_APT=1 — not installing system packages"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get not found — assuming image already has git/python/venv"
    return 0
  fi
  if [[ "$(id -u)" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    warn "not root and no sudo — skipping apt"
    return 0
  fi
  local apt=(apt-get)
  if [[ "$(id -u)" -ne 0 ]]; then
    apt=(sudo apt-get)
  fi
  log "Installing system packages (git, python venv, build tools)"
  "${apt[@]}" update -qq
  DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git ca-certificates curl \
    "${PYTHON}" "${PYTHON}-venv" "${PYTHON}-pip" "${PYTHON}-dev" \
    build-essential \
    2>/dev/null || DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git ca-certificates curl python3 python3-venv python3-pip python3-dev build-essential
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "nvidia-smi not found — continuing (CPU-only smoke test may fail later)"
    return 0
  fi
  log "GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi -L
  local vram_mb
  vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  if [[ -n "${vram_mb}" && "${vram_mb}" -lt 15000 ]]; then
    warn "GPU <16GB VRAM (${vram_mb} MiB) — use 24GB+ if training OOMs"
  fi
}

check_python() {
  command -v "${PYTHON}" >/dev/null 2>&1 || die "${PYTHON} not found"
  "${PYTHON}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null \
    || die "Need Python ${MIN_PYTHON}+ (got $("${PYTHON}" --version))"
  log "Python: $("${PYTHON}" --version)"
}

clone_or_update_repo() {
  if [[ -f "${INSTALL_DIR}/train.py" && -f "${INSTALL_DIR}/requirements.txt" ]]; then
    log "Repo at ${INSTALL_DIR}"
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
      git -C "${INSTALL_DIR}" pull --ff-only || warn "git pull failed — using existing tree"
    fi
  else
    log "Cloning ${REPO_URL} → ${INSTALL_DIR}"
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    if [[ -d "${INSTALL_DIR}" ]]; then
      rm -rf "${INSTALL_DIR}"
    fi
    git clone "${REPO_URL}" "${INSTALL_DIR}"
  fi
  cd "${INSTALL_DIR}"
  chmod +x scripts/*.sh 2>/dev/null || true
  [[ -x train.sh ]] || chmod +x train.sh 2>/dev/null || true
}

install_venv() {
  if [[ ! -d .venv ]]; then
    log "Creating .venv"
    "${PYTHON}" -m venv .venv
  else
    log "Using existing .venv"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  pip install -U pip wheel setuptools
  log "Installing trainer dependencies"
  pip install -r requirements.txt
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/install_microsoft_lens.sh"
  install_microsoft_lens "${INSTALL_DIR}"
}

prepare_workspace_dirs() {
  local hf_home="${HF_HOME:-/workspace/.cache/huggingface}"
  if [[ -d /workspace ]]; then
    mkdir -p /workspace/output "${hf_home}"
    export HF_HOME="${hf_home}"
  fi
  mkdir -p "${INSTALL_DIR}/output"
}

hf_login_if_token() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/hf_auth.sh"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Hugging Face: applying HF_TOKEN"
    hf_apply_token_env
    hf_hub_login "${HF_TOKEN}" || warn "hf auth login failed — using token file/env only"
    return 0
  fi
  if hf_hub_logged_in; then
    log "Hugging Face: $(hf_hub_whoami)"
    return 0
  fi
  warn "No HF_TOKEN — set export HF_TOKEN=hf_... or run: hf auth login"
  warn "Accept https://huggingface.co/microsoft/Lens-Base before training"
}

smoke_test() {
  if [[ "${SKIP_SMOKE_TEST}" == "1" ]]; then
    return 0
  fi
  log "Smoke test"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/runpod_env.sh"
  python - <<'PY'
import yaml
import torch
import lens
print("ok: yaml, torch", torch.__version__, "cuda:", torch.cuda.is_available(), "lens:", lens.__file__)
PY
}

maybe_start_training() {
  if [[ "${START_TRAIN}" != "1" && -z "${DATASET_PATH:-}" ]]; then
    return 0
  fi
  if [[ -z "${DATASET_PATH:-}" ]]; then
    die "START_TRAIN=1 requires DATASET_PATH"
  fi
  if [[ ! -d "${DATASET_PATH}" ]]; then
    die "DATASET_PATH not found: ${DATASET_PATH}"
  fi
  log "Starting training (${TRAIN_CONFIG})"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/runpod_env.sh"
  local -a args=(
    "${INSTALL_DIR}/train.sh"
    "${TRAIN_CONFIG}"
    --set "model.repo_id=${MODEL_REPO}"
    --set "dataset.folder_path=${DATASET_PATH}"
    --set "job.output_dir=${JOB_OUTPUT_DIR}"
  )
  if [[ -n "${SAMPLE_TRIGGER:-}" ]]; then
    args+=(--set "sample.trigger_word=${SAMPLE_TRIGGER}")
  fi
  exec "${args[@]}"
}

print_done() {
  cat <<EOF

================================================================================
Bootstrap complete — LensTrainer-LoboForge

  cd ${INSTALL_DIR}
  source scripts/runpod_env.sh

Train (set your dataset path):

  ./train.sh configs/train_lora_dual_character_24gb.yaml \\
    --set dataset.folder_path=/workspace/YOUR_DATASET \\
    --set job.output_dir=/workspace/output/my-lora \\
    --set model.repo_id=${MODEL_REPO}

Re-bootstrap anytime:  bash scripts/bootstrap.sh

Auto-start train on next bootstrap:
  DATASET_PATH=/workspace/data START_TRAIN=1 HF_TOKEN=hf_... bash scripts/bootstrap.sh
================================================================================
EOF
}

main() {
  log "LensTrainer-LoboForge bootstrap"
  log "Install dir: ${INSTALL_DIR}"
  run_apt
  check_gpu
  check_python
  clone_or_update_repo
  prepare_workspace_dirs
  install_venv
  hf_login_if_token
  smoke_test
  print_done
  maybe_start_training
}

main "$@"
