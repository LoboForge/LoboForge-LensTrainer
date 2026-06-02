"""LoboForge startup branding (shown once per training run)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from lens_trainer.console import _style, println

if TYPE_CHECKING:
    from lens_trainer.config import TrainerConfig

PROJECT_NAME = "LensTrainer-LoboForge"
REPO_URL = "https://github.com/LoboForge/LoboForge-LensTrainer"
HOMEPAGE = "https://github.com/LoboForge"
VERSION = "0.1.0"

# LoboForge wordmark (provided art — do not re-flow or re-letter).
_ASCII_LOGO = r"""
░▒▓█▓▒░      ░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░
░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░
░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓█▓▒▒▓███▓▒░▒▓██████▓▒░
░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░     ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓████████▓▒░▒▓██████▓▒░░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░
""".strip(
    "\n"
)


def _ascii_logo() -> str:
    return _ASCII_LOGO


def print_startup_banner(cfg: "TrainerConfig", *, output_dir: str) -> None:
    """Print LoboForge branding and run metadata (always; colors when TTY)."""
    accent = "93"
    dim = "2"
    for line in _ascii_logo().splitlines():
        print(_style(line, accent, bold=True), flush=True)

    print(
        _style("  LensTrainer-LoboForge · Lens-Base LoRA training", accent, dim=True),
        flush=True,
    )

    println("info", f"{PROJECT_NAME} v{VERSION}", label="LoboForge")
    println(
        "info",
        "Official trainer for microsoft/Lens-Base — https://github.com/LoboForge",
        label="project",
    )
    println("info", REPO_URL, label="source")
    println(
        "info",
        f"job={cfg.job.name} · steps={cfg.train.steps} · output={output_dir}",
        label="run",
    )

    if not sys.stdout.isatty():
        return

    print(_style("─" * 62, dim), flush=True)
