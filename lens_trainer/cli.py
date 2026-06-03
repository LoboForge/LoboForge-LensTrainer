"""Explicit CLI → config overrides (visible startup parameters)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a LoRA for Microsoft Lens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example (local):\n"
            "  python train.py configs/train_lora_dual_character_24gb.yaml \\\n"
            "    --dataset-path /home/you/DualCharacterLoras \\\n"
            "    --output-dir ./output/my-lora \\\n"
            "    --model-repo ./models/Lens-Base \\\n"
            "    --steps 8000 --resume latest\n"
        ),
    )
    parser.add_argument(
        "config",
        type=str,
        help="YAML preset (e.g. configs/train_lora_dual_character_24gb.yaml)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional training.env file (CLI flags below override the file)",
    )

    parser.add_argument("--dataset-path", type=str, help="Folder of images + .txt captions")
    parser.add_argument("--output-dir", type=str, help="Checkpoints and lora_final.safetensors")
    parser.add_argument("--job-name", type=str, help="Run name (job.name)")
    parser.add_argument("--model-repo", type=str, help="Lens-Base folder or HF repo id")
    parser.add_argument("--steps", type=int, help="Training steps")
    parser.add_argument("--save-every", type=int, help="Checkpoint interval")
    parser.add_argument("--sample-every", type=int, help="Preview image interval")
    parser.add_argument("--resolution", type=int, help="0 = auto per-image native size")
    parser.add_argument("--trigger-word", type=str, default=argparse.SUPPRESS, help="Sample prompt token")
    parser.add_argument(
        "--resume",
        type=str,
        default=argparse.SUPPRESS,
        help='Resume checkpoint: "latest", path, or step file name',
    )
    parser.add_argument(
        "--disable-mxfp4",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="false = GPU MXFP4 text cache; true = CPU bf16 (16GB GPUs)",
    )
    parser.add_argument(
        "--baseline-control",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Step-0 control samples before training",
    )
    parser.add_argument("--lora-rank", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--lora-alpha", type=int, default=argparse.SUPPRESS)

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra YAML overrides: --set lora.rank=32",
    )
    return parser


def _set_nested(overrides: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor = overrides
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def cli_to_overrides(args: argparse.Namespace, *, root: Path) -> dict[str, Any]:
    """Map explicit CLI flags to nested config overrides."""
    overrides: dict[str, Any] = {}

    if getattr(args, "dataset_path", None):
        _set_nested(overrides, "dataset.folder_path", args.dataset_path)
    if getattr(args, "output_dir", None):
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = (root / out).resolve()
        _set_nested(overrides, "job.output_dir", str(out))
    if getattr(args, "job_name", None):
        _set_nested(overrides, "job.name", args.job_name)
    if getattr(args, "model_repo", None):
        repo = args.model_repo
        if not str(repo).startswith("microsoft/"):
            repo_path = Path(repo)
            if not repo_path.is_absolute():
                repo_path = (root / repo_path).resolve()
            repo = str(repo_path)
        _set_nested(overrides, "model.repo_id", repo)
    if getattr(args, "steps", None) is not None:
        _set_nested(overrides, "train.steps", args.steps)
    if getattr(args, "save_every", None) is not None:
        _set_nested(overrides, "train.save_every", args.save_every)
    if getattr(args, "sample_every", None) is not None:
        _set_nested(overrides, "train.sample_every", args.sample_every)
    if getattr(args, "resolution", None) is not None:
        _set_nested(overrides, "dataset.resolution", args.resolution)
    if hasattr(args, "trigger_word"):
        _set_nested(overrides, "sample.trigger_word", args.trigger_word)
    if hasattr(args, "resume"):
        _set_nested(overrides, "train.resume_from", args.resume)
        _set_nested(overrides, "sample.baseline_control", False)
    if hasattr(args, "disable_mxfp4"):
        _set_nested(overrides, "model.disable_mxfp4", args.disable_mxfp4)
    if hasattr(args, "baseline_control"):
        _set_nested(overrides, "sample.baseline_control", args.baseline_control)
    if hasattr(args, "lora_rank"):
        _set_nested(overrides, "lora.rank", args.lora_rank)
    if hasattr(args, "lora_alpha"):
        _set_nested(overrides, "lora.alpha", args.lora_alpha)

    return overrides


def format_run_summary(cfg) -> str:
    """One block of resolved run parameters (printed before training)."""
    lines = [
        "Run parameters:",
        f"  dataset      {cfg.dataset.folder_path}",
        f"  output       {cfg.job.output_dir}",
        f"  job          {cfg.job.name}",
        f"  model        {cfg.model.repo_id}",
        f"  steps        {cfg.train.steps}",
        f"  save_every   {cfg.train.save_every}",
        f"  sample_every {cfg.train.sample_every}",
        f"  resolution   {cfg.dataset.resolution}",
        f"  disable_mxfp4 {cfg.model.disable_mxfp4}",
        f"  resume_from  {cfg.train.resume_from or '(new run)'}",
        f"  baseline_ctl {cfg.sample.baseline_control}",
    ]
    return "\n".join(lines)
