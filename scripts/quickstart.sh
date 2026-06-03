#!/usr/bin/env bash
# Deprecated alias — use scripts/bootstrap.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bootstrap.sh" "$@"
