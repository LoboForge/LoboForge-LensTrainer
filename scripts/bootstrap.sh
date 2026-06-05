#!/usr/bin/env bash
# =============================================================================
# LensTrainer-LoboForge — ONE bootstrap: environment + deps + Lens code + HF auth
# + microsoft/Lens-Base weights into models/Lens-Base.
#
# From an existing clone (recommended):
#   cd /path/to/LoboForge-LensTrainer
#   export HF_TOKEN=hf_...    # or: hf auth login when prompted
#   bash scripts/quickstart.sh
#
# Fresh machine (curl installer):
#   export HF_TOKEN=hf_...
#   curl -fsSL https://raw.githubusercontent.com/LoboForge/LoboForge-LensTrainer/main/scripts/bootstrap.sh | bash
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
FORCE_MODEL_REDOWNLOAD="${FORCE_MODEL_REDOWNLOAD:-0}"

default_install_dir() {
  if [[ -n "${LOBFORGE_TRAINER_DIR:-}" ]]; then
    printf '%s' "${LOBFORGE_TRAINER_DIR}"
    return 0
  fi
  if [[ -f "${PWD}/train.py" && -f "${PWD}/scripts/bootstrap.sh" ]]; then
    printf '%s' "${PWD}"
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

activate_project_venv() {
  if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VIRTUAL_ENV}/bin/activate"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
    return 0
  fi
  local name
  for name in .venv venv; do
    if [[ -f "${INSTALL_DIR}/${name}/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "${INSTALL_DIR}/${name}/bin/activate"
      export PATH="${INSTALL_DIR}/${name}/bin:${PATH}"
      return 0
    fi
  done
  return 1
}

install_python_env() {
  if ! activate_project_venv; then
    log "Creating Python venv (.venv)"
    "${PYTHON}" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export PATH="${INSTALL_DIR}/.venv/bin:${PATH}"
  else
    log "Using venv: ${VIRTUAL_ENV}"
  fi
  pip install -q -U pip wheel setuptools
  log "Installing trainer packages (requirements.txt)"
  pip install -r requirements.txt
  log "Cloning microsoft/Lens → vendor/Lens"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/install_microsoft_lens.sh"
  install_microsoft_lens "${INSTALL_DIR}"
  if [[ ! -f "${INSTALL_DIR}/vendor/Lens/lens/__init__.py" ]]; then
    die "vendor/Lens install failed — expected ${INSTALL_DIR}/vendor/Lens/lens/__init__.py"
  fi
  log "microsoft/Lens ready: vendor/Lens/lens/__init__.py"
}

huggingface_auth() {
  export LENS_TRAINER_ROOT="${INSTALL_DIR}"
  # shellcheck disable=SC1091
  source "${INSTALL_DIR}/scripts/hf_auth.sh"

  if hf_hub_logged_in 2>/dev/null; then
    log "Hugging Face: $(hf_hub_whoami)"
    return 0
  fi

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Hugging Face login (HF_TOKEN)"
    hf_apply_token_env
    hf_hub_login "${HF_TOKEN}" || hf_apply_token_env
    log "Hugging Face: $(hf_hub_whoami)"
    return 0
  fi

  if [[ -t 0 ]] && command -v hf >/dev/null 2>&1; then
    log "Hugging Face: not logged in — interactive login (accept Lens-Base license on huggingface.co first)"
    hf auth login || true
    if hf_hub_logged_in 2>/dev/null; then
      log "Hugging Face: $(hf_hub_whoami)"
      return 0
    fi
  fi

  die "Hugging Face login required for gated microsoft/Lens-Base.
  1) Accept license: https://huggingface.co/microsoft/Lens-Base
  2) Then either:
       export HF_TOKEN=hf_... && bash scripts/quickstart.sh
     or:
       hf auth login && bash scripts/quickstart.sh"
}

model_repo_ok() {
  python "${INSTALL_DIR}/scripts/assemble_lens_repo.py" --output "${MODEL_PATH}" --check >/dev/null 2>&1
}

download_lens_base_model() {
  [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]] && { warn "SKIP_MODEL_DOWNLOAD=1"; return 0; }

  mkdir -p "${INSTALL_DIR}/models"
  export PYTHONPATH="${INSTALL_DIR}/vendor/Lens:${PYTHONPATH:-}"

  if [[ "${FORCE_MODEL_REDOWNLOAD}" == "1" ]] && [[ -d "${MODEL_PATH}" ]]; then
    warn "FORCE_MODEL_REDOWNLOAD=1 — removing ${MODEL_PATH}"
    rm -rf "${MODEL_PATH}"
  elif [[ -d "${MODEL_PATH}" ]] && ! model_repo_ok; then
    warn "Broken or incomplete Lens-Base at ${MODEL_PATH} (common: git clone without git-lfs)"
    python "${INSTALL_DIR}/scripts/assemble_lens_repo.py" --output "${MODEL_PATH}" --check || true
    warn "Re-downloading via huggingface_hub → ${MODEL_PATH}"
    rm -rf "${MODEL_PATH}"
  elif model_repo_ok; then
    log "Lens-Base already present: ${MODEL_PATH}"
    return 0
  fi

  log "Downloading ${MODEL_REPO_ID} → ${MODEL_PATH} (large, one-time; needs HF login + license)"
  python "${INSTALL_DIR}/scripts/assemble_lens_repo.py" \
    --output "${MODEL_PATH}" \
    --repo-id "${MODEL_REPO_ID}"

  model_repo_ok || die "Lens-Base download finished but verification failed — run:
  python scripts/assemble_lens_repo.py --output ${MODEL_PATH} --check"
  log "Model ready: ${MODEL_PATH}"
}

write_training_env() {
  local env_file="${INSTALL_DIR}/training.env"
  local example="${INSTALL_DIR}/training.env.example"
  if [[ ! -f "${env_file}" ]]; then
    if [[ -d /workspace ]] && [[ -f "${INSTALL_DIR}/training.env.runpod.example" ]]; then
      cp "${INSTALL_DIR}/training.env.runpod.example" "${env_file}"
    elif [[ -f "${INSTALL_DIR}/training.env.local.example" ]]; then
      cp "${INSTALL_DIR}/training.env.local.example" "${env_file}"
    elif [[ -f "${example}" ]]; then
      cp "${example}" "${env_file}"
    else
      die "missing training.env.example"
    fi
    log "Created training.env"
  fi
  if ! grep -q '^MODEL_REPO=' "${env_file}" 2>/dev/null; then
    echo "MODEL_REPO=${MODEL_PATH}" >>"${env_file}"
  fi
  sed -i "s|^MODEL_REPO=.*|MODEL_REPO=${MODEL_PATH}|" "${env_file}" 2>/dev/null \
    || sed -i '' "s|^MODEL_REPO=.*|MODEL_REPO=${MODEL_PATH}|" "${env_file}" 2>/dev/null || true
}

smoke_test() {
  [[ "${SKIP_SMOKE_TEST}" == "1" ]] && return 0
  log "Smoke test (CUDA + lens import + Lens-Base weights)"
  export PYTHONPATH="${INSTALL_DIR}/vendor/Lens:${PYTHONPATH:-}"
  export MODEL_PATH
  activate_project_venv || true
  python - <<'PY'
import os
import torch
import lens
from pathlib import Path
from lens_trainer.hf_repo import is_complete_hf_repo

assert torch.cuda.is_available(), "CUDA required for training"
model_path = Path(os.environ["MODEL_PATH"])
assert is_complete_hf_repo(model_path), f"Lens-Base not loadable at {model_path}"
print("ok:", torch.cuda.get_device_name(0), "| lens:", lens.__file__)
print("ok: Lens-Base weights at", model_path)
try:
    import kernels  # noqa: F401
    print("ok: kernels (MXFP4 on GPU)")
except ImportError:
    print("note: kernels not installed — use --disable-mxfp4 on 16GB GPUs")
PY
  if python -c "import kernels" 2>/dev/null; then
    DISABLE_MXFP4=false bash "${INSTALL_DIR}/scripts/verify_gpu_ready.sh"
  fi
}

print_done() {
  local train_hint
  if [[ -d /workspace ]] && [[ -w /workspace ]]; then
    train_hint="  bash scripts/train_runpod.sh"
  else
    train_hint="  python train.py configs/train_lora_lens_base_24gb.yaml \\
    --dataset-path /path/to/images \\
    --output-dir ./output/my-lora \\
    --job-name my-lora \\
    --model-repo ${MODEL_PATH} \\
    --steps 2000 --disable-mxfp4"
  fi

  cat <<EOF

================================================================================
BOOTSTRAP DONE

  cd ${INSTALL_DIR}
  source .venv/bin/activate    # or: source venv/bin/activate

${train_hint}

  Model:    ${MODEL_PATH}
  Re-check: python scripts/assemble_lens_repo.py --output ${MODEL_PATH} --check
  Re-run:   bash scripts/quickstart.sh   (safe; fixes vendor/Lens + broken models)

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
