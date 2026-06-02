"""ANSI terminal styling for long training runs (auto-off when not a TTY)."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from tqdm import tqdm

# SGR foreground codes
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_COLORS = {
    "train": "35",      # magenta — main training bar
    "cache": "36",      # cyan — precompute / cache
    "sample": "34",     # blue — mid-run image sampling
    "checkpoint": "32", # green — saves / resume success
    "dataset": "96",    # bright cyan — dataset scan
    "info": "37",       # white — neutral status
    "warn": "33",       # yellow
    "error": "31",      # red
    "accent": "93",     # bright yellow — job header / highlights
}


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR", "") != "":
        return False
    if os.environ.get("LENS_TRAINER_COLOR", "1").lower() in {"0", "false", "no", "off"}:
        return False
    return sys.stdout.isatty()


_ENABLED = _color_enabled()


def _style(text: str, code: str, *, bold: bool = False, dim: bool = False) -> str:
    if not _ENABLED:
        return text
    prefix = "\033["
    if bold:
        prefix += f"{_BOLD};"
    if dim:
        prefix += f"{_DIM};"
    prefix += f"{code}m"
    return f"{prefix}{text}{_RESET}"


def tag(label: str, kind: str = "info") -> str:
    code = _COLORS.get(kind, _COLORS["info"])
    return _style(f"[{label}]", code, bold=True)


def message(kind: str, text: str, *, label: Optional[str] = None) -> str:
    """Format a line with an optional category tag."""
    code = _COLORS.get(kind, _COLORS["info"])
    body = _style(text, code)
    if label:
        return f"{tag(label, kind)} {body}"
    return body


def println(kind: str, text: str, *, label: Optional[str] = None) -> None:
    print(message(kind, text, label=label), flush=True)


def train(text: str) -> None:
    println("train", text, label="train")


def cache(text: str) -> None:
    println("cache", text, label="cache")


def sample(text: str) -> None:
    println("sample", text, label="sample")


def checkpoint(text: str) -> None:
    println("checkpoint", text, label="save")


def dataset(text: str) -> None:
    println("dataset", text, label="data")


def resume(text: str) -> None:
    println("checkpoint", text, label="resume")


def info(text: str) -> None:
    println("info", text)


def warn(text: str) -> None:
    println("warn", text, label="warn")


def error(text: str) -> None:
    println("error", text, label="error")


def banner(text: str) -> None:
    print(_style(text, _COLORS["accent"], bold=True), flush=True)


_TQDM_COLOUR = {
    "train": "magenta",
    "cache_latent": "cyan",
    "cache_text": "blue",
}


def tqdm_bar(kind: str, iterable=None, **kwargs: Any) -> tqdm:
    """tqdm progress bar with a category-specific colour."""
    colour = _TQDM_COLOUR.get(kind)
    if colour and _ENABLED:
        kwargs.setdefault("colour", colour)
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)
