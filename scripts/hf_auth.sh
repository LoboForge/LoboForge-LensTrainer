#!/usr/bin/env bash
# Hugging Face auth — always use the project venv (system `hf` on RunPod is often wrong).
# Used internally by scripts/bootstrap.sh (do not run standalone).

_hf_repo_root() {
  if [[ -n "${LENS_TRAINER_ROOT:-}" ]]; then
    printf '%s' "${LENS_TRAINER_ROOT}"
    return 0
  fi
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  printf '%s' "${here}"
}

_hf_venv_bin() {
  local root
  root="$(_hf_repo_root)"
  for name in .venv venv; do
    if [[ -d "${root}/${name}/bin" ]]; then
      printf '%s/%s/bin' "${root}" "${name}"
      return 0
    fi
  done
  if [[ -n "${VIRTUAL_ENV:-}" && -d "${VIRTUAL_ENV}/bin" ]]; then
    printf '%s/bin' "${VIRTUAL_ENV}"
    return 0
  fi
  return 1
}

_hf_python() {
  local venv_bin
  if venv_bin="$(_hf_venv_bin)"; then
    printf '%s/python' "${venv_bin}"
    return 0
  fi
  command -v python3 2>/dev/null || command -v python
}

_hf_cli() {
  local venv_bin
  if venv_bin="$(_hf_venv_bin)"; then
    if [[ -x "${venv_bin}/hf" ]]; then
      printf '%s/hf' "${venv_bin}"
      return 0
    fi
  fi
  return 1
}

hf_apply_token_env() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
    local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
    mkdir -p "${cache_dir}"
    printf '%s' "${HF_TOKEN}" >"${cache_dir}/token"
    chmod 600 "${cache_dir}/token" 2>/dev/null || true
  fi
}

hf_hub_login_python() {
  local token="${1:-${HF_TOKEN:-}}"
  [[ -n "${token}" ]] || return 1
  local py
  py="$(_hf_python)"
  HF_TOKEN="${token}" HUGGINGFACE_HUB_TOKEN="${token}" "${py}" - <<'PY'
import os
from huggingface_hub import login

token = os.environ["HF_TOKEN"]
login(token=token, add_to_git_credential=True)
print("huggingface_hub.login ok")
PY
}

hf_hub_login() {
  local token="${1:-${HF_TOKEN:-}}"
  if [[ -z "${token}" ]]; then
    return 1
  fi
  export HF_TOKEN="${token}"
  export HUGGINGFACE_HUB_TOKEN="${token}"
  hf_apply_token_env

  # Python login first — RunPod system `hf` / `huggingface-cli` are often broken.
  if hf_hub_login_python "${token}" 2>/dev/null; then
    return 0
  fi

  local hf_cmd
  if hf_cmd="$(_hf_cli)"; then
    if "${hf_cmd}" auth login --token "${token}" </dev/null 2>/dev/null; then
      return 0
    fi
  fi

  local venv_bin
  if venv_bin="$(_hf_venv_bin)" && [[ -x "${venv_bin}/huggingface-cli" ]]; then
    "${venv_bin}/huggingface-cli" login --token "${token}" --add-to-git-credential </dev/null 2>/dev/null && return 0
  fi

  hf_hub_login_python "${token}"
}

hf_hub_whoami() {
  hf_apply_token_env
  local hf_cmd
  if hf_cmd="$(_hf_cli)"; then
    "${hf_cmd}" auth whoami 2>/dev/null | head -1 && return 0
  fi
  local py
  py="$(_hf_python)"
  "${py}" -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null && return 0
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "(HF_TOKEN set in environment)"
    return 0
  fi
  return 1
}

hf_hub_logged_in() {
  hf_hub_whoami >/dev/null 2>&1
}
