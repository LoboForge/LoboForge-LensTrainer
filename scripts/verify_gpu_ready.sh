#!/usr/bin/env bash
# Fail fast before a multi-hour train if the GPU stack is not ready (RunPod / CUDA pods).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/_trainer_env.sh"
activate_trainer_env "${ROOT}"

DISABLE_MXFP4="${DISABLE_MXFP4:-false}"

export VERIFY_DISABLE_MXFP4="${DISABLE_MXFP4}"
python - <<'PY'
import os
import sys

import torch

if not torch.cuda.is_available():
    print("[error] No CUDA GPU visible. Use a GPU pod template.", file=sys.stderr)
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024**3)
print(f"==> GPU: {props.name} ({vram_gb:.1f} GB VRAM)")

disable = os.environ.get("VERIFY_DISABLE_MXFP4", "false").lower() in ("1", "true", "yes")
if disable:
    print("==> MXFP4 disabled (CPU/bf16 text path) — OK for 16GB-only setups")
    sys.exit(0)

try:
    import kernels  # noqa: F401
except ImportError:
    print(
        "[error] MXFP4 requires the kernels package on GPU pods.\n"
        "  pip install 'kernels>=0.12.0,<0.15' 'triton>=3.4.0'\n"
        "Without it, transformers loads bf16 (~40GB) and you get OOM / 12h CPU cache.",
        file=sys.stderr,
    )
    sys.exit(1)

print("==> kernels OK — MXFP4 text cache will use GPU (ComfyUI-class path)")
if vram_gb < 21.0:
    print(
        f"==> [warn] {vram_gb:.0f}GB VRAM: tight for Lens. If text cache OOMs, "
        "set DISABLE_MXFP4=true or use a 24GB+ GPU."
    )
PY
