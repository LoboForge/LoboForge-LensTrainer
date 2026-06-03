#!/usr/bin/env bash
# =============================================================================
# LensTrainer-LoboForge — ONE bootstrap: environment + deps + Lens code + HF auth
# + microsoft/Lens-Base weights into models/Lens-Base.
#
#   export HF_TOKEN=hf_...    # required for gated Hub download (first time)
#   curl -fsSL https://raw.githubusercontent.com/LoboForge/LoboForge-LensTrainer/main/scripts/bootstrap.sh | bash
#
# Then edit training.env and run:
#   bash scripts/train.sh
# =============================================================================
set -euo pipefail

REPO_URL="${LOBFORGE_TRAINER_REPO:-https://github.com/LoboForge/LoboForge-LensTrainer.git}"
MODEL_REPO_ID="${MODEL_REPO_ID:-microsoft/Lens-Base}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-Lens-Base}"
PYTHON="${PYTHON:-python3}"
MIN_PYTHON=3.11
SKIP_APT="${SKIP_APT:-0}"
SKIP_SMOKE_TEST="${SKIP_SMOKE_TEST:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

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
MODEL_PATH="${INSTALL_DIR}/models/${MODEL_DIR_NAME}"

log() { printf '==> %s\n' "$*"; }
warn() { printf '==> [warn] %s\n' "$*" >&2; }
die() { printf '==> [error] %s\n' "$*" >&2; exit 1; }

run_apt() {
  [[ "${SKIP_APT}" == "1" ]] && return 0
  command -v apt-get >/dev/null 2>&1 || return 0
  local apt=(apt-get)
  [[ "$(id -u)" -ne 0 ]] && apt=(sudo apt-get)
  log "System packages (git, python3-venv, build-essential)"
  "${apt[@]}" update -qq
  DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git ca-certificates curl \
    "${PYTHON}" "${PYTHON}-venv" "${PYTHON}-pip" "${PYTHON}-dev" \
    build-essential \
    2>/dev/null || DEBIAN_FRONTEND=noninteractive "${apt[@]}" install -y -qq \
    git ca-certificates curl python3 python3-venv python3-pip python3-dev build-essential
}

check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { warn "no nvidia-smi"; return 0; }
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || nvidia-smi -L | head -1)"
}

check_python() {
  command -v "${PYTHON}" >/dev/null 2>&1 || die "${PYTHON} not found"
  "${PYTHON}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" \
    || die "Need Python ${MIN_PYTHON}+"
  log "Python: $("${PYTHON}" --version)"
}

clone_or_update_repo() {
  if [[ -f "${INSTALL_DIR}/train.py" ]]; then
    log "Trainer repo: ${INSTALL_DIR}"
    [[ -d "${INSTALL_DIR}/.git" ]] && git -C "${INSTALL_DIR}" pull --ff-only || true
  else
    log "Cloning trainer → ${INSTALL_DIR}"
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    rm -rf "${INSTALL_DIR}" 2>/dev/null || true
    git clone "${REPO_URL}" "${INSTALL_DIR}"
  fi
  cd "${INSTALL_DIR}"
  chmod +x scripts/*.sh train.sh 2>/dev/null || true
}

install_python_env() {
  if [[ ! -d .venv ]]; then
    log "Creating Python venv"
    "${PYTHON}" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export PATH="${INSTALL_DIR}/.venv/bin:${PATH}"
  pip install -q -U pip wheel setuptools
  log "Installing trainer packages (requirements.txt)"
  pip install -r requirements.txt
  log "Cloning microsoft/Lens → vendor/Lens"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/install_microsoft_lens.sh"
  install_microsoft_lens "${INSTALL_DIR}"
}

huggingface_auth() {
  export LENS_TRAINER_ROOT="${INSTALL_DIR}"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/hf_auth.sh"

  if [[ -z "${HF_TOKEN:-}" ]]; then
    if hf_hub_logged_in 2>/dev/null; then
      log "Hugging Face: $(hf_hub_whoami)"
      return 0
    fi
    die "Set HF_TOKEN before bootstrap (gated microsoft/Lens-Base). Example: export HF_TOKEN=hf_..."
  fi

  log "Hugging Face login (venv)"
  hf_apply_token_env
  hf_hub_login "${HF_TOKEN}" || hf_apply_token_env
  log "Hugging Face: $(hf_hub_whoami)"
}

download_lens_base_model() {
  [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]] && { warn "SKIP_MODEL_DOWNLOAD=1"; return 0; }

  mkdir -p "${INSTALL_DIR}/models"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/_trainer_env.sh"
  activate_trainer_env "${INSTALL_DIR}"

  if python "${INSTALL_DIR}/scripts/assemble_lens_repo.py" --output "${MODEL_PATH}" --check 2>/dev/null; then
    log "Lens-Base already present: ${MODEL_PATH}"
    return 0
  fi

  log "Downloading ${MODEL_REPO_ID} → ${MODEL_PATH} (large, one-time; needs HF_TOKEN + license)"
  python "${INSTALL_DIR}/scripts/assemble_lens_repo.py" \
    --output "${MODEL_PATH}" \
    --repo-id "${MODEL_REPO_ID}"
  log "Model ready: ${MODEL_PATH}"
}

write_training_env() {
  local env_file="${INSTALL_DIR}/training.env"
  local example="${INSTALL_DIR}/training.env.example"
  if [[ ! -f "${env_file}" ]]; then
    [[ -f "${example}" ]] && cp "${example}" "${env_file}" || die "missing training.env.example"
    log "Created training.env"
  fi
  # Point default model at local assembled folder
  if ! grep -q '^MODEL_REPO=' "${env_file}" 2>/dev/null; then
    echo "MODEL_REPO=${MODEL_PATH}" >>"${env_file}"
  fi
  sed -i "s|^MODEL_REPO=.*|MODEL_REPO=${MODEL_PATH}|" "${env_file}" 2>/dev/null \
    || sed -i '' "s|^MODEL_REPO=.*|MODEL_REPO=${MODEL_PATH}|" "${env_file}" 2>/dev/null || true
}

smoke_test() {
  [[ "${SKIP_SMOKE_TEST}" == "1" ]] && return 0
  log "Smoke test (CUDA + lens import)"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/_trainer_env.sh"
  activate_trainer_env "${INSTALL_DIR}"
  python - <<'PY'
import torch, lens
assert torch.cuda.is_available(), "CUDA required for training"
print("ok:", torch.cuda.get_device_name(0), "| lens:", lens.__file__)
PY
}

print_done() {
  cat <<EOF

================================================================================
BOOTSTRAP DONE

  cd ${INSTALL_DIR}
  nano training.env          # set DATASET_PATH, LORA_NAME, STEPS, TRIGGER_WORD, ...
  bash scripts/train.sh      # start training

  Model:    ${MODEL_PATH}
  HF cache: \${HF_HOME:-/workspace/.cache/huggingface}

Re-run bootstrap anytime (safe):  bash scripts/bootstrap.sh
================================================================================
EOF
}

main() {
  log "LensTrainer-LoboForge bootstrap (full environment + models)"
  log "Install: ${INSTALL_DIR}"
  run_apt
  check_gpu
  check_python
  clone_or_update_repo
  install_python_env
  huggingface_auth
  download_lens_base_model
  write_training_env
  smoke_test
  print_done
}

main "$@"
