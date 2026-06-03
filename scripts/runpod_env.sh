#!/usr/bin/env bash
# Source before training on RunPod or any GPU box:
#   source /workspace/LoboForge-LensTrainer/scripts/runpod_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
  echo "[error] Missing ${SCRIPT_DIR}/.venv — run: bash scripts/setup_runpod.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"

export USE_HUB_KERNELS=NO
export PYTHONPATH="${SCRIPT_DIR}/vendor/Lens:${PYTHONPATH:-}"

if [[ -f "${SCRIPT_DIR}/scripts/hf_auth.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/scripts/hf_auth.sh"
  hf_apply_token_env
fi

if [[ ! -d "${SCRIPT_DIR}/vendor/Lens/lens" ]]; then
  echo "[error] Missing ${SCRIPT_DIR}/vendor/Lens — run: bash scripts/setup_runpod.sh" >&2
  return 1 2>/dev/null || exit 1
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}"

cd "${SCRIPT_DIR}"
