#!/usr/bin/env bash
# Download microsoft/Lens-Base into models/Lens-Base via Hugging Face Hub.
# Replaces broken git-clone / git-lfs pointer trees automatically.
#
#   export HF_TOKEN=hf_...   # or: hf auth login
#   bash scripts/download_lens_base.sh
#   bash scripts/download_lens_base.sh --force
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# shellcheck disable=SC1091
source "${ROOT}/scripts/_trainer_env.sh"
activate_trainer_env "${ROOT}" || {
  printf '==> [error] Run bash scripts/quickstart.sh first (venv + vendor/Lens)\n' >&2
  exit 1
}

# shellcheck disable=SC1091
source "${ROOT}/scripts/hf_auth.sh"
export LENS_TRAINER_ROOT="${ROOT}"

if ! hf_hub_logged_in 2>/dev/null; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    hf_apply_token_env
    hf_hub_login "${HF_TOKEN}" || true
  fi
fi
hf_hub_logged_in 2>/dev/null || {
  printf '==> [error] Hugging Face login required (HF_TOKEN or hf auth login)\n' >&2
  printf '    Accept license: https://huggingface.co/microsoft/Lens-Base\n' >&2
  exit 1
}

OUTPUT="${MODEL_PATH:-${ROOT}/models/Lens-Base}"
REPO_ID="${MODEL_REPO_ID:-microsoft/Lens-Base}"
EXTRA=()
if [[ "${FORCE_MODEL_REDOWNLOAD:-0}" == "1" ]]; then
  EXTRA+=(--force)
fi
for arg in "$@"; do
  EXTRA+=("${arg}")
done

printf '==> Downloading %s → %s\n' "${REPO_ID}" "${OUTPUT}"
python "${ROOT}/scripts/assemble_lens_repo.py" --output "${OUTPUT}" --repo-id "${REPO_ID}" "${EXTRA[@]}"
python "${ROOT}/scripts/assemble_lens_repo.py" --output "${OUTPUT}" --check"
