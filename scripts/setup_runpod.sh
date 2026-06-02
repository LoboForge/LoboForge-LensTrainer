#!/usr/bin/env bash
# LensTrainer-LoboForge — full setup for a fresh RunPod GPU pod (PyTorch template).
# Run as root or a user with passwordless sudo. Idempotent: safe to re-run.
#
#   curl -fsSL https://raw.githubusercontent.com/LoboForge/LoboForge-LensTrainer/main/scripts/setup_runpod.sh | bash
#   # or from a cloned repo:
#   bash scripts/setup_runpod.sh
#
# Optional env:
#   LOBFORGE_TRAINER_DIR   install path (default: /workspace/LoboForge-LensTrainer)
#   HF_TOKEN               if set, runs `huggingface-cli login --token`
#   SKIP_APT=1             skip apt installs (air-gapped / pre-baked image)
#   SKIP_SMOKE_TEST=1      skip import/GPU smoke test after pip
set -euo pipefail

REPO_URL="${LOBFORGE_TRAINER_REPO:-https://github.com/LoboForge/LoboForge-LensTrainer.git}"
INSTALL_DIR="${LOBFORGE_TRAINER_DIR:-/workspace/LoboForge-LensTrainer}"
PYTHON="${PYTHON:-python3}"
MIN_PYTHON=3.11
SKIP_APT="${SKIP_APT:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"

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
    warn "not root and no sudo — skipping apt (install git/python3-venv manually if needed)"
    return 0
  fi
  local apt=(apt-get)
  if [[ "$(id -u)" -ne 0 ]]; then
    apt=(sudo apt-get)
  fi
  log "Installing system packages (git, venv, headers for pip wheels)"
  "${apt[@]}" update -qq
  DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git \
    ca-certificates \
    curl \
    "${PYTHON}" \
    "${PYTHON}-venv" \
    "${PYTHON}-pip" \
    "${PYTHON}-dev" \
    build-essential \
    2>/dev/null || DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git ca-certificates curl python3 python3-venv python3-pip python3-dev build-essential
}

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found. Use a RunPod **GPU** pod with the **PyTorch** template."
  fi
  log "GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi -L
  local vram_mb
  vram_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [[ -n "${vram_mb}" && "${vram_mb}" -lt 15000 ]]; then
    warn "GPU reports <16GB VRAM (${vram_mb} MiB). Lens-Base training may OOM — prefer 24GB+ (e.g. RTX 4090)."
  fi
}

check_python() {
  command -v "${PYTHON}" >/dev/null 2>&1 || die "${PYTHON} not found after apt install"
  local ver
  ver="$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  "${PYTHON}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null \
    || die "Python ${ver} is too old; need ${MIN_PYTHON}+"
  log "Python: $("${PYTHON}" --version)"
}

clone_or_update_repo() {
  if [[ -f "${INSTALL_DIR}/train.py" && -f "${INSTALL_DIR}/requirements.txt" ]]; then
    log "Repo already at ${INSTALL_DIR}"
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
      log "Pulling latest"
      git -C "${INSTALL_DIR}" pull --ff-only || warn "git pull failed — continuing with existing tree"
    fi
  else
    log "Cloning into ${INSTALL_DIR}"
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    if [[ -d "${INSTALL_DIR}/.git" ]]; then
      git -C "${INSTALL_DIR}" pull --ff-only
    else
      git clone "${REPO_URL}" "${INSTALL_DIR}"
    fi
  fi
  cd "${INSTALL_DIR}"
}

install_venv() {
  log "Creating venv at ${INSTALL_DIR}/.venv"
  "${PYTHON}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  pip install -U pip wheel setuptools
  log "Installing Python dependencies (Lens from GitHub — may take several minutes)"
  pip install -r requirements.txt
}

write_env_helper() {
  local env_file="${INSTALL_DIR}/scripts/runpod_env.sh"
  log "Writing ${env_file}"
  mkdir -p "${INSTALL_DIR}/scripts"
  cat >"${env_file}" <<'ENVEOF'
# Source before training:  source /workspace/LoboForge-LensTrainer/scripts/runpod_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"
export USE_HUB_KERNELS=NO
# Prefer persistent HF cache on RunPod network volume
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}"
cd "${SCRIPT_DIR}"
ENVEOF
  chmod +x "${env_file}"
}

hf_login_if_token() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN set — logging into Hugging Face"
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || die "huggingface-cli login failed"
    return 0
  fi
  if huggingface-cli whoami >/dev/null 2>&1; then
    log "Hugging Face: already logged in as $(huggingface-cli whoami 2>/dev/null | head -1)"
    return 0
  fi
  warn "Not logged into Hugging Face yet."
  warn "  export HF_TOKEN=hf_...   # or: huggingface-cli login"
  warn "  Accept https://huggingface.co/microsoft/Lens-Base before training."
}

smoke_test() {
  if [[ "${SKIP_SMOKE_TEST}" == "1" ]]; then
    warn "SKIP_SMOKE_TEST=1 — skipping import check"
    return 0
  fi
  log "Smoke test (imports + CUDA)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export USE_HUB_KERNELS=NO
  python - <<'PY'
import sys
print("python", sys.version.split()[0])

import yaml
print("pyyaml ok")

import torch
print("torch", torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available — check GPU pod / driver")
print("cuda", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

import lens  # noqa: F401 — installs microsoft/Lens from requirements.txt
print("lens ok", lens.__file__)

print("smoke test passed")
PY
}

print_next_steps() {
  cat <<EOF

================================================================================
LensTrainer-LoboForge is ready on this pod.

  cd ${INSTALL_DIR}
  source scripts/runpod_env.sh
  huggingface-cli login          # if not already (gated Lens-Base)
  # Upload dataset (SCP example — use your pod's Direct TCP SSH port):
  #   scp -P <PORT> -i ~/.ssh/id_ed25519 -r ./DualCharacterLoras root@<IP>:/workspace/

  python train.py configs/train_lora_dual_character_24gb.yaml \\
    --set model.repo_id=microsoft/Lens-Base \\
    --set dataset.folder_path=/workspace/DualCharacterLoras \\
    --set job.output_dir=/workspace/output/lens-lora-dual-character

Persistent paths (use a RunPod network volume mounted at /workspace):
  Repo:     ${INSTALL_DIR}
  HF cache: \${HF_HOME:-/workspace/.cache/huggingface}
  Outputs:  /workspace/output/...

Re-run this script anytime to refresh deps:  bash scripts/setup_runpod.sh
================================================================================
EOF
}

main() {
  log "LensTrainer-LoboForge RunPod setup"
  log "Install dir: ${INSTALL_DIR}"
  run_apt
  check_gpu
  check_python
  clone_or_update_repo
  install_venv
  write_env_helper
  hf_login_if_token
  smoke_test
  print_next_steps
}

main "$@"
