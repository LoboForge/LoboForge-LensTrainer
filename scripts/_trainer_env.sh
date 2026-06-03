#!/usr/bin/env bash
# Internal: activate venv + PYTHONPATH (sourced by bootstrap.sh and train.sh only).

_trainer_venv_dir() {
  local root="${1:?trainer root required}"
  if [[ -n "${VIRTUAL_ENV:-}" && -f "${VIRTUAL_ENV}/bin/activate" ]]; then
    printf '%s' "${VIRTUAL_ENV}"
    return 0
  fi
  local name
  for name in .venv venv; do
    if [[ -f "${root}/${name}/bin/activate" ]]; then
      printf '%s/%s' "${root}" "${name}"
      return 0
    fi
  done
  return 1
}

_trainer_hf_home_default() {
  if [[ -n "${HF_HOME:-}" ]]; then
    printf '%s' "${HF_HOME}"
    return 0
  fi
  if [[ -d /workspace ]] && [[ -w /workspace ]]; then
    printf '/workspace/.cache/huggingface'
    return 0
  fi
  printf '%s/.cache/huggingface' "${HOME}"
}

activate_trainer_env() {
  local root="${1:?trainer root required}"
  local venv_dir
  if ! venv_dir="$(_trainer_venv_dir "${root}")"; then
    echo "[error] No Python venv in ${root} (.venv or venv). Create one:" >&2
    echo "  cd ${root} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    return 1
  fi
  if [[ ! -d "${root}/vendor/Lens/lens" ]]; then
    echo "[error] Missing ${root}/vendor/Lens — run: bash scripts/install_microsoft_lens.sh" >&2
    return 1
  fi
  # shellcheck disable=SC1091
  source "${venv_dir}/bin/activate"
  export PATH="${venv_dir}/bin:${PATH}"
  export LENS_TRAINER_ROOT="${root}"
  export USE_HUB_KERNELS=NO
  export PYTHONPATH="${root}/vendor/Lens:${PYTHONPATH:-}"
  export HF_HOME="$(_trainer_hf_home_default)"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
  mkdir -p "${HF_HOME}"
  if [[ -f "${root}/scripts/hf_auth.sh" ]]; then
    # shellcheck disable=SC1091
    source "${root}/scripts/hf_auth.sh"
    hf_apply_token_env
  fi
  cd "${root}"
}
