#!/usr/bin/env python3
"""Assemble a Hugging Face–layout Lens repo, reusing local weight files when provided."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lens_trainer.hf_repo import assemble_lens_repo, diagnose_hf_repo, is_complete_hf_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standard Hugging Face Lens pipeline folder. "
            "Supply local transformer/VAE safetensors to skip re-downloading them; "
            "text encoder and tokenizer always come from the Hub."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML file with repo_id, output_dir, link, and optional sources.transformer/vae paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/Lens-Base",
        help="Output directory for the HF-layout repo (default: ./models/Lens-Base)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="microsoft/Lens-Base",
        help="Hugging Face repo id for missing components (default: microsoft/Lens-Base)",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default=None,
        help="Path to a local Lens DiT .safetensors (single-file checkpoint)",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="Path to a local FLUX.2 / Lens VAE .safetensors",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy local files instead of symlinking (default: symlink)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify whether --output is a complete HF repo; do not download",
    )
    return parser.parse_args()


def _load_yaml_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def main() -> None:
    args = parse_args()

    output = Path(args.output)
    repo_id = args.repo_id
    transformer = args.transformer
    vae = args.vae
    use_symlink = not args.copy

    if args.config:
        cfg = _load_yaml_config(args.config)
        output = Path(cfg.get("output_dir", output))
        repo_id = cfg.get("repo_id", repo_id)
        use_symlink = bool(cfg.get("link", use_symlink))
        sources = cfg.get("sources") or {}
        transformer = sources.get("transformer") or transformer
        vae = sources.get("vae") or vae

    if args.check:
        problems = diagnose_hf_repo(output)
        ok = not problems and is_complete_hf_repo(output)
        if ok:
            print(f"{output}: complete")
            raise SystemExit(0)
        print(f"{output}: incomplete")
        for line in problems:
            print(f"  - {line}")
        raise SystemExit(1)

    transformer_path = Path(transformer).expanduser() if transformer else None
    vae_path = Path(vae).expanduser() if vae else None

    assemble_lens_repo(
        output_dir=output,
        repo_id=repo_id,
        transformer_source=transformer_path,
        vae_source=vae_path,
        use_symlink=use_symlink,
    )


if __name__ == "__main__":
    main()
