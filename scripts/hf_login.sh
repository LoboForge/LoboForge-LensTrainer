#!/usr/bin/env bash
# Log in to Hugging Face using the project venv (not the broken system `hf` on some pods).
#
#   export HF_TOKEN=hf_...
#   bash scripts/hf_login.sh
#
# Or interactive after bootstrap:
#   source scripts/runpod_env.sh && hf auth login
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LENS_TRAINER_ROOT="${ROOT}"

die_no_hf() {
  echo "[error] No hf in .venv — run: bash scripts/bootstrap.sh" >&2
  exit 1
}

# shellcheck disable=SC1091
source "${ROOT}/scripts/runpod_env.sh"
# shellcheck disable=SC1091
source "${ROOT}/scripts/hf_auth.sh"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "==> Logging in with HF_TOKEN (venv: $(_hf_python))"
  hf_hub_login "${HF_TOKEN}"
  echo "==> $(hf_hub_whoami)"
  exit 0
fi

HF_BIN="$(_hf_cli)" || die_no_hf
echo "==> Interactive login via ${HF_BIN}"
echo "    (Or: export HF_TOKEN=hf_... && bash scripts/hf_login.sh)"
exec "${HF_BIN}" auth login
