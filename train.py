#!/usr/bin/env python3
"""Config-driven Microsoft Lens LoRA trainer."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings(
    "ignore",
    message="The pynvml package is deprecated.*",
    category=FutureWarning,
)

# transformers 5.9 + kernels>=0.15 crash at import (LayerRepository needs version=).
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import argparse
from pathlib import Path

import yaml

from lens_trainer.cuda_env import configure_cuda_libraries

configure_cuda_libraries()

from lens_trainer.cli import build_arg_parser, cli_to_overrides, format_run_summary
from lens_trainer.config import load_config
from lens_trainer.training_env import training_env_to_overrides
from lens_trainer.trainer import LensTrainer


def _parse_override(entry: str) -> dict:
    if "=" not in entry:
        raise ValueError(f"Invalid override '{entry}', expected key=value")
    key, raw = entry.split("=", 1)
    value = yaml.safe_load(raw)
    parts = key.split(".")
    nested: dict = {}
    cursor = nested
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return nested


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parent

    # Precedence (low → high): YAML preset < --env-file < --set < explicit CLI flags
    overrides: dict = {}

    env_path = Path(args.env_file) if args.env_file else None
    if env_path is not None:
        if not env_path.is_absolute():
            env_path = (root / env_path).resolve()
        from lens_trainer.training_env import _read_env_file

        overrides = _deep_merge(
            overrides,
            training_env_to_overrides(root, env=_read_env_file(env_path)),
        )

    for entry in args.set:
        overrides = _deep_merge(overrides, _parse_override(entry))

    overrides = _deep_merge(overrides, cli_to_overrides(args, root=root))

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (root / cfg_path).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path, overrides=overrides or None)
    print(format_run_summary(cfg), flush=True)
    trainer = LensTrainer(cfg, config_path=cfg_path)
    trainer.run()


if __name__ == "__main__":
    main()
