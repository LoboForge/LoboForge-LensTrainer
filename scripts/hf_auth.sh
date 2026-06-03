#!/usr/bin/env bash
# Hugging Face auth helpers — supports new `hf` CLI and legacy `huggingface-cli`.
# Usage:  source scripts/hf_auth.sh

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

hf_hub_login() {
  local token="${1:-${HF_TOKEN:-}}"
  if [[ -z "${token}" ]]; then
    return 1
  fi
  export HF_TOKEN="${token}"
  export HUGGINGFACE_HUB_TOKEN="${token}"

  if command -v hf >/dev/null 2>&1; then
    # New Hugging Face CLI (huggingface_hub >= 1.0 on many cloud images).
    if hf auth login --token "${token}" </dev/null 2>/dev/null; then
      return 0
    fi
    # Some builds only accept interactive login; token file still works for downloads.
    hf_apply_token_env
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "${token}" --add-to-git-credential </dev/null
    return $?
  fi
  # Libraries read HUGGINGFACE_HUB_TOKEN / HF_TOKEN without a CLI.
  return 0
}

hf_hub_whoami() {
  hf_apply_token_env
  if command -v hf >/dev/null 2>&1; then
    hf auth whoami 2>/dev/null | head -1
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli whoami 2>/dev/null | head -1
    return 0
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "(HF_TOKEN set in environment)"
    return 0
  fi
  return 1
}

hf_hub_logged_in() {
  hf_hub_whoami >/dev/null 2>&1
}
