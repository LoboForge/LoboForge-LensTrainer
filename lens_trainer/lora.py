"""PEFT LoRA helpers and ComfyUI-compatible export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file

# Lens uses nn.ModuleList([Linear, Identity]) for attn.to_out and GateMLP (w1/w2/w3)
# for img_mlp/txt_mlp. PEFT only supports leaf Linear modules.
_LENS_TARGET_ALIASES: dict[str, list[str]] = {
    "to_out": ["to_out.0"],
    "img_mlp": ["img_mlp.w1", "img_mlp.w2", "img_mlp.w3"],
    "txt_mlp": ["txt_mlp.w1", "txt_mlp.w2", "txt_mlp.w3"],
}


def default_target_modules() -> list[str]:
    return [
        "img_qkv",
        "txt_qkv",
        "to_out.0",
        "to_add_out",
        "img_mlp.w1",
        "img_mlp.w2",
        "img_mlp.w3",
        "txt_mlp.w1",
        "txt_mlp.w2",
        "txt_mlp.w3",
        "txt_in",
        "img_in",
        "proj_out",
    ]


def normalize_target_modules(target_modules: Iterable[str]) -> list[str]:
    """Expand Lens-specific module aliases to PEFT-compatible Linear leaf names."""
    expanded: list[str] = []
    for name in target_modules:
        expanded.extend(_LENS_TARGET_ALIASES.get(name, [name]))
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(expanded))


def attach_lora(transformer, rank: int, alpha: int, dropout: float, target_modules: Iterable[str]):
    targets = normalize_target_modules(target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
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
