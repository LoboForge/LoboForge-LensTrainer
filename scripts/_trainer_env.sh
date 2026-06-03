#!/usr/bin/env bash
# Internal: activate venv + PYTHONPATH (sourced by bootstrap.sh and train.sh only).

activate_trainer_env() {
  local root="${1:?trainer root required}"
  if [[ ! -d "${root}/.venv" ]]; then
    echo "[error] Missing ${root}/.venv — run: bash scripts/bootstrap.sh" >&2
    return 1
  fi
  if [[ ! -d "${root}/vendor/Lens/lens" ]]; then
    echo "[error] Missing ${root}/vendor/Lens — run: bash scripts/bootstrap.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "${root}/.venv/bin/activate"
  export PATH="${root}/.venv/bin:${PATH}"
  export LENS_TRAINER_ROOT="${root}"
  export USE_HUB_KERNELS=NO
  export PYTHONPATH="${root}/vendor/Lens:${PYTHONPATH:-}"
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  mkdir -p "${HF_HOME}"
  if [[ -f "${root}/scripts/hf_auth.sh" ]]; then
    # shellcheck disable=SC1091
    source "${root}/scripts/hf_auth.sh"
    hf_apply_token_env
  fi
  cd "${root}"
}
