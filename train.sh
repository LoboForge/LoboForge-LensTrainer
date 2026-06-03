#!/usr/bin/env bash
# Train wrapper — activates venv, PYTHONPATH, then runs train.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/runpod_env.sh"
exec python "${ROOT}/train.py" "$@"
