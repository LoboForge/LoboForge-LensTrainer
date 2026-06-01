"""Assemble a Hugging Face–layout Lens repo from local weights + Hub downloads."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

TRANSFORMER_MARKERS = ("img_in.weight", "proj_out.weight", "transformer_blocks.0.")
VAE_MARKERS = ("bn.running_mean", "encoder.conv_in.weight", "decoder.conv_out.weight")


def _keys(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt") as handle:
        return set(handle.keys())


def _has_markers(keys: Iterable[str], markers: tuple[str, ...]) -> bool:
    keyset = set(keys)
    return all(any(key.startswith(marker) or marker in key for key in keyset) for marker in markers)


def validate_transformer_weights(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Transformer weights not found: {path}")
    keys = _keys(path)
    if not _has_markers(keys, TRANSFORMER_MARKERS):
        raise ValueError(
            f"{path} does not look like a Lens transformer checkpoint "
            f"(expected keys like {TRANSFORMER_MARKERS})."
        )


def validate_vae_weights(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"VAE weights not found: {path}")
    keys = _keys(path)
    if not _has_markers(keys, VAE_MARKERS):
        raise ValueError(
            f"{path} does not look like a FLUX.2 / Lens VAE checkpoint "
            f"(expected keys like {VAE_MARKERS})."
        )


def _link_or_copy(src: Path, dest: Path, use_symlink: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if use_symlink:
        dest.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dest)


def is_complete_hf_repo(path: Path) -> bool:
    """Return True if path looks like a loadable Lens HF pipeline folder."""
    if not (path / "model_index.json").is_file():
        return False
    te_index = path / "text_encoder" / "model.safetensors.index.json"
    te_single = path / "text_encoder" / "model.safetensors"
    if not te_index.is_file() and not te_single.is_file():
        return False
    tr_index = path / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    tr_single = path / "transformer" / "diffusion_pytorch_model.safetensors"
    if not tr_index.is_file() and not tr_single.is_file():
        return False
    vae = path / "vae" / "diffusion_pytorch_model.safetensors"
    return vae.is_file()


def resolve_model_repo(repo_id: str) -> str:
    """If repo_id is a relative/absolute path to a complete HF folder, use it as-is."""
    candidate = Path(repo_id).expanduser()
    if candidate.is_dir() and is_complete_hf_repo(candidate):
        return str(candidate.resolve())
    return repo_id


def assemble_lens_repo(
    output_dir: Path,
    repo_id: str = "microsoft/Lens-Base",
    transformer_source: Path | None = None,
    vae_source: Path | None = None,
    use_symlink: bool = True,
) -> Path:
    """
    Build a Hugging Face–layout Lens folder at ``output_dir``.

    - Optional ``transformer_source`` / ``vae_source``: validated local safetensors
      linked or copied into the HF tree (avoids re-downloading large files).
    - Remaining components (text encoder, tokenizer, scheduler, configs) are fetched
      from the Hub via ``snapshot_download``.
    """
    from huggingface_hub import snapshot_download

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns: list[str] = []

    if transformer_source is not None:
        validate_transformer_weights(transformer_source)
        dest = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        _link_or_copy(transformer_source, dest, use_symlink)
        ignore_patterns.extend(
            [
                "transformer/diffusion_pytorch_model-*.safetensors",
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "transformer/diffusion_pytorch_model.safetensors",
            ]
        )
        print(f"Using local transformer weights -> {dest}")

    if vae_source is not None:
        validate_vae_weights(vae_source)
        dest = output_dir / "vae" / "diffusion_pytorch_model.safetensors"
        _link_or_copy(vae_source, dest, use_symlink)
        ignore_patterns.append("vae/diffusion_pytorch_model.safetensors")
        print(f"Using local VAE weights -> {dest}")

    print(f"Downloading remaining Lens components from {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        ignore_patterns=ignore_patterns or None,
    )

    if not is_complete_hf_repo(output_dir):
        raise RuntimeError(
            f"Assembled repo at {output_dir} is incomplete. "
            "Ensure Hugging Face login and Lens-Base license acceptance, then retry."
        )

    manifest = {
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "local_transformer": str(transformer_source) if transformer_source else None,
        "local_vae": str(vae_source) if vae_source else None,
        "symlink": use_symlink,
    }
    with (output_dir / ".lens_trainer_assembled.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ready: {output_dir}")
    print("Train with: python train.py configs/train_lora_lens_base_24gb.yaml \\")
    print(f"  --set model.repo_id={output_dir}")
    return output_dir
