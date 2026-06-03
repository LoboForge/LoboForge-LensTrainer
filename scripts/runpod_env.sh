#!/usr/bin/env bash
# Internal alias — use scripts/bootstrap.sh then scripts/train.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_trainer_env.sh"
activate_trainer_env "${SCRIPT_DIR}"
