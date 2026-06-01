#!/usr/bin/env python3
"""Config-driven Microsoft Lens LoRA trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lens_trainer.config import load_config
from lens_trainer.trainer import LensTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA for Microsoft Lens")
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML training config (e.g. configs/train_lora_lens_base_24gb.yaml)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values, e.g. --set train.steps=500 --set dataset.folder_path=/data/subject",
    )
    return parser.parse_args()


def _parse_override(entry: str) -> tuple[str, object]:
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


def main() -> None:
    args = parse_args()
    overrides = {}
    for entry in args.set:
        patch = _parse_override(entry)
        overrides = _deep_merge(overrides, patch)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path, overrides=overrides or None)
    trainer = LensTrainer(cfg, config_path=cfg_path)
    trainer.run()


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


if __name__ == "__main__":
    main()
