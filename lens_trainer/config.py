"""YAML configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JobConfig:
    name: str = "lens-lora"
    output_dir: str = "./output/lens-lora"


@dataclass
class ModelConfig:
    repo_id: str = "microsoft/Lens-Base"
    dtype: str = "bfloat16"
    disable_mxfp4: bool = True
    cpu_offload: bool = False
    cache_text_embeddings: bool = True


@dataclass
class DatasetConfig:
    folder_path: str = "./dataset"
    caption_ext: str = "txt"
    resolution: int = 1024
    cache_latents: bool = True
    max_sequence_length: int = 512


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "img_qkv",
            "txt_qkv",
            "to_out",
            "to_add_out",
            "img_mlp",
            "txt_mlp",
            "txt_in",
            "img_in",
            "proj_out",
        ]
    )


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "adamw8bit"
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    guidance_scale: float = 5.0
    timestep_type: str = "shift"
    save_every: int = 250
    sample_every: int = 100
    sample_every_early: int = 50
    sample_early_until: int = 500
    seed: int = 42
    max_grad_norm: float = 1.0
    resume_from: str = ""  # checkpoint path, or "latest" / "auto"


@dataclass
class SampleConfig:
    # Prompt list — sampled at step 0 as base-model control, then again during training with LoRA.
    prompts: list[Any] = field(
        default_factory=lambda: [
            {
                "name": "stage_full",
                "prompt": "a photo of [trigger] ballet dancer on stage, full body, soft spotlight, photorealistic",
            },
            {
                "name": "studio_portrait",
                "prompt": "portrait of [trigger], ballet studio, hair in bun, shallow depth of field",
            },
            {
                "name": "arabesque",
                "prompt": "[trigger] in arabesque on pointe, clean studio background, studio lighting",
            },
            {
                "name": "cafe_candid",
                "prompt": "candid photo of [trigger] in street clothes, natural lighting",
            },
            {
                "name": "sign_test",
                "prompt": "[trigger] holding a sign that says 'lens lora test', studio photo",
            },
        ]
    )
    trigger_word: str = "mytrigger"
    width: int = 1024
    height: int = 1024
    steps: int = 50
    cfg: float = 5.0
    seed: int = 42
    walk_seed: bool = True
    baseline_control: bool = True
    max_sequence_length: int = 512


@dataclass
class TrainerConfig:
    job: JobConfig = field(default_factory=JobConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _section(data: dict[str, Any], cls: type, key: str):
    section = data.get(key, {}) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in section.items() if k in allowed}
    return cls(**filtered)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainerConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _merge_dict(raw, overrides)
    return TrainerConfig(
        job=_section(raw, JobConfig, "job"),
        model=_section(raw, ModelConfig, "model"),
        dataset=_section(raw, DatasetConfig, "dataset"),
        lora=_section(raw, LoraConfig, "lora"),
        train=_section(raw, TrainConfig, "train"),
        sample=_section(raw, SampleConfig, "sample"),
    )


def resolve_output_dir(cfg: TrainerConfig) -> Path:
    out = Path(cfg.job.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
