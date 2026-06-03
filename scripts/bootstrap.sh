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
# After bootstrap, edit training.env and run:  bash scripts/train.sh
set -euo pipefail

REPO_URL="${LOBFORGE_TRAINER_REPO:-https://github.com/LoboForge/LoboForge-LensTrainer.git}"
PYTHON="${PYTHON:-python3}"
MIN_PYTHON=3.11
SKIP_APT="${SKIP_APT:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
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

ensure_training_env() {
  local example="${INSTALL_DIR}/training.env.example"
  local env_file="${INSTALL_DIR}/training.env"
  if [[ ! -f "${env_file}" && -f "${example}" ]]; then
    cp "${example}" "${env_file}"
    log "Created ${env_file} — edit DATASET_PATH, LORA_NAME, STEPS, then train"
  elif [[ -f "${env_file}" ]]; then
    log "Using existing ${env_file}"
  fi
}

print_done() {
  cat <<EOF

================================================================================
Bootstrap complete — two scripts total:

  1) bash scripts/bootstrap.sh     (once — installs everything)
  2) bash scripts/train.sh       (after editing training.env)

  cd ${INSTALL_DIR}
  nano training.env    # DATASET_PATH, LORA_NAME, STEPS, TRIGGER_WORD, TRAIN_PRESET, ...
  bash scripts/train.sh

Re-bootstrap:  bash scripts/bootstrap.sh
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
  ensure_training_env
  print_done
}

main "$@"
