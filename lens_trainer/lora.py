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


def _comfy_key_to_peft(key: str) -> str:
    if key.startswith("diffusion_model."):
        key = "base_model.model." + key[len("diffusion_model.") :]
    elif not key.startswith("base_model.model."):
        key = f"base_model.model.{key}"
    return key


def read_lora_checkpoint_metadata(checkpoint_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt") as handle:
        return dict(handle.metadata() or {})


def resolve_resume_checkpoint(
    resume_from: str,
    *,
    output_dir: Path,
    checkpoints_dir: Path,
) -> Path:
    token = resume_from.strip()
    if not token:
        raise ValueError("resume_from is empty")

    if token.lower() in {"latest", "auto"}:
        candidates = list(checkpoints_dir.glob("lora_step_*.safetensors"))
        final_path = output_dir / "lora_final.safetensors"
        if final_path.is_file():
            candidates.append(final_path)

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found under {checkpoints_dir} or {final_path}"
            )

        def _step(path: Path) -> int:
            if path.name == "lora_final.safetensors":
                meta = read_lora_checkpoint_metadata(path)
                return int(meta.get("step", 0))
            try:
                return int(path.stem.rsplit("_", 1)[-1])
            except ValueError:
                return 0

        return max(candidates, key=_step)

    path = Path(token)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def load_lora_weights(model, checkpoint_path: Path) -> dict[str, object]:
    """Load Comfy-export LoRA weights into a PEFT-wrapped model. Returns resume metadata."""
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)
    metadata = read_lora_checkpoint_metadata(checkpoint_path)
    raw = load_file(str(checkpoint_path))
    if not raw:
        raise RuntimeError(f"No tensors found in checkpoint: {checkpoint_path}")

    model_keys = set(model.state_dict().keys())
    mapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if "lora_" not in key:
            continue
        peft_key = _comfy_key_to_peft(key)
        if peft_key in model_keys:
            mapped[peft_key] = value
            continue
        # Some PEFT builds omit ".default" in module names — match by suffix.
        suffix = peft_key.split("base_model.model.", 1)[-1]
        alt_keys = [k for k in model_keys if k.endswith(suffix)]
        if len(alt_keys) == 1:
            mapped[alt_keys[0]] = value

    if not mapped:
        raise RuntimeError(
            f"Could not map any LoRA keys from {checkpoint_path.name} into the current model"
        )

    incompatible = model.load_state_dict(mapped, strict=False)
    missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
    if missing_lora:
        print(
            f"Warning: {len(missing_lora)} LoRA key(s) in model not present in checkpoint "
            f"(loaded {len(mapped)} tensors from {checkpoint_path.name})"
        )

    step = int(metadata.get("step", 0))
    return {
        "step": step,
        "rank": metadata.get("rank"),
        "alpha": metadata.get("alpha"),
        "path": str(checkpoint_path),
        "loaded_keys": len(mapped),
    }
