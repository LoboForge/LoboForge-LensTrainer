"""Parse project-root training.env into config overrides (same keys as scripts/train.sh)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    return yaml.safe_load(raw)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        out[key] = value.strip()
    return out


def training_env_to_overrides(
    root: str | Path | None = None,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build nested override dict from training.env (does not modify os.environ)."""
    if env is None:
        root = Path(root or Path.cwd())
        env_path = root / "training.env"
        env = _read_env_file(env_path)
        if not env and os.environ.get("LENS_TRAINER_ROOT"):
            env = _read_env_file(Path(os.environ["LENS_TRAINER_ROOT"]) / "training.env")

    if not env:
        return {}

    root_path = Path(root or Path.cwd()).resolve()
    overrides: dict[str, Any] = {}

    def set_key(dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cursor = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    if env.get("DATASET_PATH"):
        set_key("dataset.folder_path", _parse_value(env["DATASET_PATH"]))
    if env.get("LORA_NAME"):
        set_key("job.name", _parse_value(env["LORA_NAME"]))
    if env.get("OUTPUT_DIR"):
        out = _parse_value(env["OUTPUT_DIR"])
        if isinstance(out, str) and not out.startswith("/"):
            out = str((root_path / out).resolve())
        set_key("job.output_dir", out)
    if env.get("STEPS"):
        set_key("train.steps", _parse_value(env["STEPS"]))
    if env.get("SAVE_EVERY"):
        set_key("train.save_every", _parse_value(env["SAVE_EVERY"]))
    if env.get("SAMPLE_EVERY"):
        set_key("train.sample_every", _parse_value(env["SAMPLE_EVERY"]))
    if env.get("MODEL_REPO"):
        set_key("model.repo_id", _parse_value(env["MODEL_REPO"]))
    if env.get("DISABLE_MXFP4") is not None and env.get("DISABLE_MXFP4") != "":
        set_key("model.disable_mxfp4", _parse_value(env["DISABLE_MXFP4"]))
    if env.get("RESOLUTION") is not None and env.get("RESOLUTION") != "":
        set_key("dataset.resolution", _parse_value(env["RESOLUTION"]))
    if env.get("TRIGGER_WORD") is not None:
        set_key("sample.trigger_word", _parse_value(env.get("TRIGGER_WORD", "")))
    if env.get("BASELINE_CONTROL") is not None and env.get("BASELINE_CONTROL") != "":
        set_key("sample.baseline_control", _parse_value(env["BASELINE_CONTROL"]))
    if env.get("RESUME_FROM"):
        set_key("train.resume_from", _parse_value(env["RESUME_FROM"]))
        set_key("sample.baseline_control", False)
    if env.get("LORA_RANK"):
        set_key("lora.rank", _parse_value(env["LORA_RANK"]))
    if env.get("LORA_ALPHA"):
        set_key("lora.alpha", _parse_value(env["LORA_ALPHA"]))

    return overrides
