"""PEFT LoRA helpers and ComfyUI-compatible export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file


def default_target_modules() -> list[str]:
    return [
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


def attach_lora(transformer, rank: int, alpha: int, dropout: float, target_modules: Iterable[str]):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    model = get_peft_model(transformer, config)
    model.print_trainable_parameters()
    return model


def _normalize_peft_key(key: str) -> str:
    if key.startswith("base_model.model."):
        key = key[len("base_model.model.") :]
    return key


def lora_state_dict_for_comfy(model) -> dict[str, torch.Tensor]:
    """Remap PEFT keys to ComfyUI Lens naming (``diffusion_model.*``)."""
    sd: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if "lora_" not in key:
            continue
        key = _normalize_peft_key(key)
        if key.startswith("transformer."):
            key = key.replace("transformer.", "diffusion_model.", 1)
        if not key.startswith("diffusion_model."):
            key = f"diffusion_model.{key}"
        sd[key] = value.detach().cpu()
    return sd


def save_lora(model, output_path: Path, metadata: dict | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sd = lora_state_dict_for_comfy(model)
    if not sd:
        raise RuntimeError("No LoRA weights found to save")
    meta = {k: str(v) for k, v in (metadata or {}).items()}
    save_file(sd, str(output_path), metadata=meta)
