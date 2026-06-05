#!/usr/bin/env bash
# One-command setup from an existing clone (or fresh install dir):
#   vendor/Lens + Python venv + Hugging Face auth + models/Lens-Base
#
# From repo root:
#   export HF_TOKEN=hf_...   # or run `hf auth login` when prompted
#   bash scripts/quickstart.sh
#
# Already cloned elsewhere:
#   LOBFORGE_TRAINER_DIR=~/Desktop/LoboLens/LoboForge-LensTrainer bash scripts/quickstart.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LOBFORGE_TRAINER_DIR="${LOBFORGE_TRAINER_DIR:-${ROOT}}"
exec bash "${ROOT}/scripts/bootstrap.sh" "$@"
